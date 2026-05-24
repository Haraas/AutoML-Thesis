import sys, os, time, pickle, argparse, gc
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE       = os.path.dirname(os.path.abspath(__file__))
REF_PATH   = os.path.join(BASE, '..', 'references', 'qp_evaluation')
EVAL_PATH  = os.path.join(REF_PATH, 'evaluation')
DATA_PATH  = os.path.join(REF_PATH, 'data', 'imdb')
DRIVE_PATH = '/content/drive/MyDrive/datasets/qp_evaluation_data'
RESULTS    = os.path.join(REF_PATH, 'experiments', 'results', 'automl')
os.makedirs(RESULTS, exist_ok=True)

sys.path.insert(0, EVAL_PATH)
sys.path.insert(0, os.path.join(EVAL_PATH, 'algorithms'))

from dataset_utils import *
from trainer import Prediction, train

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    os.system("pip install -q optuna")
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_data():
    print("Φόρτωση δεδομένων...")
    df_list = [
        pd.read_csv(os.path.join(
            DATA_PATH, 'bao', 'plans', f'job_ext_arm{i}.csv'))
        for i in range(49)]
    with open(os.path.join(DATA_PATH, 'bao', 'plans', 'bao_dat.pkl'), 'rb') as f:
        dat = pickle.load(f)
    planss    = dat['planss']
    latencies = dat['latencies']
    rootss    = dat['rootss']
    del dat
    all_roots = sum(rootss, [])
    ds_info   = DatasetInfo({})
    ds_info.construct_from_plans(all_roots)
    minmax    = pd.read_csv(os.path.join(DATA_PATH, 'column_min_max_vals.csv'))
    ds_info.get_columns(get_col_min_max(minmax))
    ds_info.all_roots = all_roots
    print(f"Arms: {len(latencies)} | Queries/arm: {len(latencies[0])}")
    return df_list, planss, latencies, rootss, ds_info

def encode_node(node, ds_info, encoding_type):
    features = []
    op_types = ds_info.nodeTypes
    op_vec   = np.zeros(len(op_types))
    if node.nodeType in op_types:
        op_vec[op_types.index(node.nodeType)] = 1.0
    features.append(op_vec)
    if 'est' in encoding_type:
        cost = 0.0
        card = 0.0
        if node.cost_est is not None and ds_info.cost_est_norm is not None:
            cost = float(ds_info.cost_est_norm.normalize_label(node.cost_est))
        if node.card_est is not None and ds_info.card_norm is not None:
            card = float(ds_info.card_norm.normalize_label(node.card_est))
        features.append(np.array([cost, card]))
    if 'pred' in encoding_type:
        max_f    = max(ds_info.max_filters, 1)
        pred_vec = np.zeros(max_f * 3)
        if node.filters:
            for i, filt in enumerate(node.filters[:max_f]):
                col, op, val = filt[0], filt[1], filt[2]
                col_idx = ds_info.columns.index(col) if col in ds_info.columns else 0
                ops     = ['=', '!=', '<', '>', '<=', '>=']
                op_idx  = ops.index(op) if op in ops else 0
                try:
                    val_f = float(val)
                    if col in ds_info.column_min_max_vals:
                        lo, hi = ds_info.column_min_max_vals[col]
                        val_f  = (val_f - lo) / (hi - lo + 1e-6)
                except:
                    val_f = 0.0
                pred_vec[i*3]   = col_idx / max(len(ds_info.columns), 1)
                pred_vec[i*3+1] = op_idx  / 6.0
                pred_vec[i*3+2] = np.clip(val_f, 0, 1)
        features.append(pred_vec)
    if 'hist' in encoding_type:
        buf = 0.0
        if node.buffers is not None:
            try:
                buf = float(np.log(node.buffers + 1) / 20.0)
            except:
                buf = 0.0
        features.append(np.array([buf]))
    return np.concatenate(features).astype(np.float32)

def get_feature_dim(ds_info, encoding_type):
    dim = len(ds_info.nodeTypes)
    if 'est'  in encoding_type: dim += 2
    if 'pred' in encoding_type: dim += max(ds_info.max_filters, 1) * 3
    if 'hist' in encoding_type: dim += 1
    return dim

class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm       = nn.LSTM(input_dim, hidden_dim, batch_first=True)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return h.squeeze(0)

class SimpleTreeLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return h.squeeze(0)

class SimpleTreeCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.pool  = nn.AdaptiveMaxPool1d(1)
    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.pool(x).squeeze(-1)

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, nhead=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        nhead = min(nhead, hidden_dim)
        while hidden_dim % nhead != 0:
            nhead -= 1
        encoder_layer    = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead,
            dim_feedforward=hidden_dim*2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.pool        = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x.transpose(1, 2)
        return self.pool(x).squeeze(-1)

def build_tree_model(name, input_dim, hidden_dim):
    if name == 'lstm':          return SimpleLSTM(input_dim, hidden_dim)
    elif name == 'treelstm':    return SimpleTreeLSTM(input_dim, hidden_dim)
    elif name == 'treecnn':     return SimpleTreeCNN(input_dim, hidden_dim)
    elif name == 'transformer': return SimpleTransformer(input_dim, hidden_dim)
    else: raise ValueError(f"Unknown: {name}")

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, roots, costs, ds_info, encoding_type, max_nodes=50):
        self.ds_info       = ds_info
        self.encoding_type = encoding_type
        self.max_nodes     = max_nodes
        self.feat_dim      = get_feature_dim(ds_info, encoding_type)
        self.labels = torch.FloatTensor(
            ds_info.cost_norm.normalize_labels(costs)).reshape(-1, 1)
        self.features = self._encode_all(roots)

    def _encode_tree(self, root):
        nodes = []
        def dfs(node):
            nodes.append(encode_node(node, self.ds_info, self.encoding_type))
            for child in node.children:
                dfs(child)
        dfs(root)
        return nodes

    def _encode_all(self, roots):
        all_features = []
        for root in roots:
            nodes = self._encode_tree(root)
            if len(nodes) < self.max_nodes:
                pad   = [np.zeros(self.feat_dim, dtype=np.float32)
                         for _ in range(self.max_nodes - len(nodes))]
                nodes = nodes + pad
            else:
                nodes = nodes[:self.max_nodes]
            all_features.append(np.stack(nodes))
        return torch.FloatTensor(np.stack(all_features))

    def __len__(self):           return len(self.labels)
    def __getitem__(self, idx):  return self.features[idx], self.labels[idx]

def get_loader(roots, costs, ds_info, encoding_type, batch_size=32):
    ds = CustomDataset(roots, costs, ds_info, encoding_type)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)

class QueryRepModel(nn.Module):
    def __init__(self, tree_model, pred_hid):
        super().__init__()
        self.tree_model = tree_model
        self.pred       = Prediction(tree_model.hidden_dim, pred_hid)
    def forward(self, x):
        return self.pred(self.tree_model(x))

class BanditOptimizer():
    def __init__(self, planss, rootss, latencies,
                 look_back=800, N=400, freq=100):
        self.planss    = planss
        self.rootss    = rootss
        self.latencies = latencies
        self.look_back = look_back
        self.ptr       = 0
        self.tr, self.tm, self.tl, self.selections = [], [], [], []

    def sample_data(self):
        start = max(0, self.ptr - self.look_back)
        idx   = range(start, self.ptr + 1)
        return (sum([self.rootss[i]    for i in idx], []),
                sum([self.latencies[i] for i in idx], []))

    def train_time(self, t): self.tr.append(t)

    def select_plans(self, model, get_batch_fn):
        preds = []
        for k in range(len(self.planss)):
            t1    = time.time()
            batch = get_batch_fn(self.rootss[k])
            self.tl.append(time.time() - t1)
            t2    = time.time()
            with torch.no_grad():
                pred = model(batch).cpu().numpy()
            self.tm.append(time.time() - t2)
            preds.append(float(pred.mean()))
        self.selections.append(int(np.argmin(preds)))
        self.ptr += 1

def get_custom(latencies, res):
    total_lats, exe_lats = [], []
    for i, row in res.iterrows():
        sel = row['Selections']
        lat = latencies[sel][i] / 1000
        exe_lats.append(lat)
        total_lats.append(
            lat + row['Train Time'] + row['Inf Time'] + row['Preprocess Time'])
    return total_lats, exe_lats

def proxy_evaluation(config, ds_info, rootss, latencies, device, n_arms=5):
    try:
        encoding_type = config['encoding']
        feat_dim      = get_feature_dim(ds_info, encoding_type)
        tree_model    = build_tree_model(config['tree_model'], feat_dim, config['hidden_dim'])
        model         = QueryRepModel(tree_model, config['pred_hid']).to(device)

        train_roots  = sum(rootss[:n_arms], [])
        train_costs  = sum(latencies[:n_arms], [])
        val_roots    = sum(rootss[n_arms:n_arms+2], [])
        val_costs    = sum(latencies[n_arms:n_arms+2], [])
        train_loader = get_loader(train_roots, train_costs, ds_info, encoding_type)
        val_loader   = get_loader(val_roots,   val_costs,   ds_info, encoding_type)

        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, 0.9)
        crit      = nn.MSELoss()

        model.train()
        for epoch in range(config['epochs']):
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = crit(model(x), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
            scheduler.step()

        model.eval()
        preds_list = []
        with torch.no_grad():
            for x, y in val_loader:
                p = model(x.to(device)).squeeze().cpu().numpy()
                if p.ndim == 0: preds_list.append(float(p))
                else:           preds_list.extend(p.tolist())

        preds_un = ds_info.cost_norm.unnormalize_labels(np.array(preds_list))
        labels_un = np.array(val_costs[:len(preds_list)])
        q_errors  = [max(p/l, l/p) for p, l in zip(preds_un, labels_un) if p > 0 and l > 0]
        q_mean    = np.mean(q_errors) if q_errors else 9999.0

        del model
        torch.cuda.empty_cache()
        gc.collect()
        return q_mean
    except Exception as e:
        print(f"    Proxy error: {e}")
        torch.cuda.empty_cache()
        gc.collect()
        return 9999.0

def full_run(best_config, df_list, planss, latencies, rootss, ds_info, device):
    print(f"\n{'='*55}")
    print(f"  PHASE 2: Full Run | Config: {best_config}")
    print(f"{'='*55}")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    encoding_type = best_config['encoding']
    feat_dim      = get_feature_dim(ds_info, encoding_type)
    tree_model    = build_tree_model(best_config['tree_model'], feat_dim, best_config['hidden_dim'])
    model         = QueryRepModel(tree_model, best_config['pred_hid']).to(device)
    optimizer     = torch.optim.Adam(model.parameters(), lr=best_config['lr'])
    scheduler     = torch.optim.lr_scheduler.StepLR(optimizer, 20, 0.9)
    crit          = nn.MSELoss()

    def get_loader_fn(roots, costs):
        return get_loader(roots, costs, ds_info, encoding_type, batch_size=32)
    def get_batch_fn(roots):
        return next(iter(get_loader_fn(roots, [0]*len(roots))))[0].to(device)

    bo      = BanditOptimizer(planss, rootss, latencies)
    n_steps = len(latencies[0]) // 100
    t_start = time.time()

    for step in range(n_steps):
        t0  = time.time()
        dat = bo.sample_data()
        try:
            loader = get_loader_fn(*dat)
            model.train()
            for epoch in range(best_config['epochs']):
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    loss = crit(model(x), y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                    optimizer.step()
                scheduler.step()
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"  OOM στο step {step}")
                torch.cuda.empty_cache()
                return None
            raise e
        torch.cuda.empty_cache()
        gc.collect()
        bo.train_time(time.time() - t0)
        print(f"  {(step+1)*100:4d} | Train: {time.time()-t0:.1f}s | "
              f"Elapsed: {(time.time()-t_start)/60:.1f}m")
        bo.select_plans(model, get_batch_fn)

    res                    = df_list[0].copy()
    del res['json']
    res['Train Time']      = bo.tr
    res['Inf Time']        = bo.tm
    res['Preprocess Time'] = bo.tl
    res['Selections']      = bo.selections

    arms      = len(latencies)
    length    = len(latencies[0])
    best_lats = [min(latencies[k][i] for k in range(arms)) for i in range(length)]
    best      = np.cumsum(best_lats) / 1000 / 60
    post      = np.cumsum(latencies[0]) / 1000 / 60
    total_t, exe = get_custom(latencies, res)
    total_final  = np.cumsum(total_t) / 60
    exe_final    = np.cumsum(exe) / 60
    improvement  = (post[-1] - exe_final[-1]) / post[-1] * 100

    print(f"\n  Encoding      : {encoding_type}")
    print(f"  Tree Model    : {best_config['tree_model']}")
    print(f"  Best Possible : {best[-1]:.4f} min")
    print(f"  PostgreSQL    : {post[-1]:.4f} min")
    print(f"  Total Time    : {total_final[-1]:.4f} min")
    print(f"  Query Time    : {exe_final[-1]:.4f} min")
    print(f"  Improvement   : {improvement:.1f}%")
    print(f"  AVGDL baseline: 62.7722 min (49.2%)")
    diff = 62.7722 - exe_final[-1]
    print(f"  Διαφορά       : {abs(diff):.4f} min "
          f"({'Βελτίωση!' if diff > 0 else 'Χειρότερο'})")

    result = {
        'method': 'AutoML', 'encoding': encoding_type,
        'tree_model': best_config['tree_model'],
        'hidden_dim': best_config['hidden_dim'],
        'pred_hid': best_config['pred_hid'],
        'lr': best_config['lr'],
        'epochs': best_config['epochs'],
        'best_possible': round(best[-1], 4),
        'postgres': round(post[-1], 4),
        'total_time': round(total_final[-1], 4),
        'query_time': round(exe_final[-1], 4),
        'improvement_%': round(improvement, 1),
    }
    pd.DataFrame([result]).to_csv(os.path.join(RESULTS, 'automl_final.csv'), index=False)
    try:
        pd.DataFrame([result]).to_csv(
            os.path.join(DRIVE_PATH, 'automl_final.csv'), index=False)
        print("  Αποθηκεύτηκε στο Drive!")
    except: pass
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials',     type=int, default=10)
    parser.add_argument('--proxy-arms', type=int, default=5)
    args_cli = parser.parse_args()

    df_list, planss, latencies, rootss, ds_info = load_data()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    trial_results = []

    def objective(trial):
        config = {
            'encoding':   trial.suggest_categorical('encoding', [
                'est', 'pred', 'est_pred',
                'est_hist', 'pred_hist', 'est_pred_hist']),
            'tree_model': trial.suggest_categorical('tree_model', [
                'lstm', 'treecnn', 'treelstm', 'transformer']),
            'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128]),
            'pred_hid':   trial.suggest_categorical('pred_hid', [32, 64, 128, 256]),
            'lr':         trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            'epochs':     trial.suggest_categorical('epochs', [30, 50, 100]),
        }
        t0      = time.time()
        score   = proxy_evaluation(config, ds_info, rootss, latencies,
                                   device, n_arms=args_cli.proxy_arms)
        elapsed = (time.time() - t0) / 60
        print(f"  Trial {trial.number+1:2d}/{args_cli.trials} | "
              f"enc={config['encoding']:15s} | "
              f"tree={config['tree_model']:12s} | "
              f"hid={config['hidden_dim']:3d} | "
              f"lr={config['lr']:.5f} | "
              f"Q-Error={score:.3f} | "
              f"Time={elapsed:.1f}m")
        trial_results.append({**config, 'q_error': score,
                               'trial': trial.number+1,
                               'time_min': round(elapsed, 2)})
        pd.DataFrame(trial_results).to_csv(
            os.path.join(RESULTS, 'optuna_trials.csv'), index=False)
        try:
            pd.DataFrame(trial_results).to_csv(
                os.path.join(DRIVE_PATH, 'optuna_trials.csv'), index=False)
        except: pass
        return score

    print(f"\nPHASE 1: Optuna | Trials: {args_cli.trials}")
    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=args_cli.trials)

    best_config = study.best_params
    best_qerror = study.best_value
    print(f"\nΚαλύτερο Q-Error: {best_qerror:.4f}")
    print(f"Καλύτερο config : {best_config}")

    df_trials = pd.DataFrame(trial_results).sort_values('q_error')
    print(df_trials.to_string(index=False))
    df_trials.to_csv(os.path.join(RESULTS, 'optuna_trials_final.csv'), index=False)
    try:
        df_trials.to_csv(
            os.path.join(DRIVE_PATH, 'optuna_trials_final.csv'), index=False)
    except: pass

    full_run(best_config, df_list, planss, latencies, rootss, ds_info, device)
