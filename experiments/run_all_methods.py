"""
run_all_methods.py
Τρέχει κάθε representation method στο BAO framework (query optimization)
και αποθηκεύει τα αποτελέσματα σε CSV.

Τοποθεσία: AutoML-Thesis/experiments/run_all_methods.py

Χρήση:
    python run_all_methods.py              # τρέχει όλες
    python run_all_methods.py avgdl        # τρέχει μόνο avgdl
    python run_all_methods.py avgdl rtos   # τρέχει αυτές τις δύο
"""

import sys, os, time, pickle, argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# ── Paths ────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
REF_PATH  = os.path.join(BASE, '..', 'references', 'qp_evaluation')
EVAL_PATH = os.path.join(REF_PATH, 'evaluation')
DATA_PATH = os.path.join(REF_PATH, 'data', 'imdb')

sys.path.insert(0, EVAL_PATH)
sys.path.insert(0, os.path.join(EVAL_PATH, 'algorithms'))

from dataset_utils import *
from trainer import Prediction, train

# ═══════════════════════════════════════════════════════════════
# 1. ΦΟΡΤΩΣΗ ΔΕΔΟΜΕΝΩΝ
# ═══════════════════════════════════════════════════════════════
def load_data():
    print("Φόρτωση δεδομένων...")

    df_list = [
        pd.read_csv(os.path.join(
            DATA_PATH, 'bao', 'plans', f'job_ext_arm{i}.csv'))
        for i in range(49)
    ]

    pkl_path = os.path.join(DATA_PATH, 'bao', 'plans', 'bao_dat.pkl')
    with open(pkl_path, 'rb') as f:
        dat = pickle.load(f)
    planss    = dat['planss']
    latencies = dat['latencies']
    rootss    = dat['rootss']
    del dat

    all_roots = sum(rootss, [])
    ds_info   = DatasetInfo({})
    ds_info.construct_from_plans(all_roots)

    minmax_path = os.path.join(DATA_PATH, 'column_min_max_vals.csv')
    minmax      = pd.read_csv(minmax_path)
    col_min_max = get_col_min_max(minmax)
    ds_info.get_columns(col_min_max)

    # Αποθήκευση all_roots στο ds_info για χρήση στις μεθόδους
    ds_info.all_roots = all_roots

    print(f"✓ Arms: {len(latencies)} | Queries/arm: {len(latencies[0])}")
    return df_list, planss, latencies, rootss, ds_info

# ═══════════════════════════════════════════════════════════════
# 2. BANDIT OPTIMIZER
# ═══════════════════════════════════════════════════════════════
class BanditOptimizer():
    def __init__(self, planss, rootss, latencies,
                 look_back=800, N=100, freq=100):
        self.planss    = planss
        self.rootss    = rootss
        self.latencies = latencies
        self.look_back = look_back
        self.N         = N
        self.freq      = freq
        self.ptr       = 0
        self.tr, self.tm, self.tl, self.selections = [], [], [], []

    def sample_data(self):
        start = max(0, self.ptr - self.look_back)
        idx   = range(start, self.ptr + 1)
        roots = sum([self.rootss[i]    for i in idx], [])
        costs = sum([self.latencies[i] for i in idx], [])
        return roots, costs

    def train_time(self, t):
        self.tr.append(t)

    def select_plans(self, model, get_batch):
        preds = []
        for k in range(len(self.planss)):
            roots = self.rootss[k]
            t1 = time.time()
            batch = get_batch(roots, [0] * len(roots))
            self.tl.append(time.time() - t1)
            t2 = time.time()
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

# ═══════════════════════════════════════════════════════════════
# 3. ΟΡΙΣΜΟΣ ΚΑΘΕ ΜΕΘΟΔΟΥ
# ═══════════════════════════════════════════════════════════════

def make_avgdl(ds_info, device, hid=64):
    from algorithms.avgdl import AVGDL_Dataset, AVGDL, Encoding
    from algorithms.avgdl import DataLoader as Loader
    from algorithms.avgdl import collate
    enc   = Encoding()
    model = nn.Sequential(AVGDL(32, 64, 64),
                          Prediction(64, hid)).to(device)
    def get_loader(roots, costs):
        ds = AVGDL_Dataset(roots, enc, costs, ds_info)
        return Loader(ds, batch_size=len(roots),
                      collate_fn=collate, shuffle=False)
    return model, get_loader


def make_bao(ds_info, device, hid=64):
    from algorithms.bao.featurize import TreeFeaturizer
    from algorithms.bao.featurize import collate as bao_collate
    from algorithms.bao.model     import BaoTransformerWithPrediction
    from torch.utils.data         import DataLoader
    feat = TreeFeaturizer()
    feat.fit(ds_info.all_roots)
    model = nn.Sequential(
        BaoTransformerWithPrediction(feat),
        Prediction(64, hid)
    ).to(device)
    def get_loader(roots, costs):
        pairs = list(zip(roots, costs))
        return DataLoader(pairs, batch_size=len(roots),
                          collate_fn=lambda x: bao_collate(x, feat),
                          shuffle=False)
    return model, get_loader


def make_neo(ds_info, device, hid=64):
    from algorithms.neo.featurize import TreeFeaturizer
    from algorithms.neo.featurize import collate as neo_collate
    from algorithms.neo.model     import Neo
    from torch.utils.data         import DataLoader
    feat = TreeFeaturizer()
    feat.fit(ds_info.all_roots)
    model = nn.Sequential(
        Neo(feat),
        Prediction(64, hid)
    ).to(device)
    def get_loader(roots, costs):
        pairs = list(zip(roots, costs))
        return DataLoader(pairs, batch_size=len(roots),
                          collate_fn=lambda x: neo_collate(x, feat),
                          shuffle=False)
    return model, get_loader


def make_rtos(ds_info, device, hid=64):
    from algorithms.rtos import Encoding, get_rtos_dataset, TreeLSTM
    from torch.utils.data import DataLoader
    enc   = Encoding(ds_info)
    model = nn.Sequential(
        TreeLSTM(enc),
        Prediction(64, hid)
    ).to(device)
    def get_loader(roots, costs):
        ds = get_rtos_dataset(roots, costs, ds_info, enc)
        return DataLoader(ds, batch_size=len(roots),
                          collate_fn=lambda x: x, shuffle=False)
    return model, get_loader


def make_aimeetsai(ds_info, device, hid=64):
    from algorithms.aimeetsai import get_aimeetsai_ds
    from torch.utils.data     import DataLoader, TensorDataset
    dim = len(ds_info.nodeParallels)

    class AIMeetsAIModel(nn.Module):
        def __init__(self, in_dim, hid):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hid),
                nn.ReLU(),
                nn.Linear(hid, 1))
        def forward(self, x):
            return self.mlp(x)

    model = AIMeetsAIModel(dim * 5, hid).to(device)

    def get_loader(roots, costs):
        feats, labels = get_aimeetsai_ds(ds_info, roots, costs)
        ds = TensorDataset(
            torch.FloatTensor(feats),
            torch.FloatTensor(labels))
        return DataLoader(ds, batch_size=len(roots), shuffle=False)
    return model, get_loader


def make_rejoin(ds_info, device, hid=64):
    from algorithms.rejoin.featurize import StateVector
    from algorithms.rejoin.featurize import collate as rj_collate
    from torch.utils.data            import DataLoader
    sv   = StateVector(ds_info)
    in_d = sv.vector_size()

    class ReJOINModel(nn.Module):
        def __init__(self, in_d, hid):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(in_d, hid),
                nn.ReLU(),
                nn.Linear(hid, 1))
        def forward(self, x):
            return self.mlp(x)

    model = ReJOINModel(in_d, hid).to(device)

    def get_loader(roots, costs):
        pairs = [(sv.encode(r), c) for r, c in zip(roots, costs)]
        return DataLoader(pairs, batch_size=len(roots),
                          collate_fn=rj_collate, shuffle=False)
    return model, get_loader


# ═══════════════════════════════════════════════════════════════
# 4. ΚΕΝΤΡΙΚΗ ΣΥΝΑΡΤΗΣΗ ΕΚΤΕΛΕΣΗΣ
# ═══════════════════════════════════════════════════════════════
def run_method(name, make_fn, df_list, planss, latencies, rootss,
               ds_info, device='cuda:0',
               N=400, look_back=800, freq=100, hid=64, lr=1e-3):

    print(f"\n{'='*55}")
    print(f"  Μέθοδος: {name.upper()}")
    print(f"{'='*55}")

    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    try:
        model, get_loader = make_fn(ds_info, device, hid)
    except Exception as e:
        print(f"  ✗ Αποτυχία αρχικοποίησης: {e}")
        return None

    def get_batch(roots, costs):
        loader = get_loader(roots, costs)
        return next(iter(loader))[0].to(device)

    class Args:
        pass
    args           = Args()
    args.device    = device
    args.bs        = 128
    args.epochs    = 200
    args.lr        = lr
    args.hid       = hid
    args.save_path = os.path.join(
        REF_PATH, 'experiments', 'results', 'bao', name, '')
    os.makedirs(args.save_path, exist_ok=True)

    bo = BanditOptimizer(planss, rootss, latencies,
                         look_back=look_back, N=N, freq=freq)

    steps_total = len(latencies[0]) // freq
    t_start     = time.time()

    for step in range(steps_total):
        t0  = time.time()
        dat = bo.sample_data()
        try:
            loader = get_loader(*dat)
            train(model, loader, loader, dat[1], ds_info, args,
                  prints=False, record=False)
        except Exception as e:
            print(f"  ✗ Σφάλμα στο training (step {step}): {e}")
            return None
        bo.train_time(time.time() - t0)
        print(f"  {(step+1)*freq:4d} | "
              f"Train: {time.time()-t0:.1f}s | "
              f"Elapsed: {(time.time()-t_start)/60:.1f}m")
        bo.select_plans(model, get_batch)

    # ── Υπολογισμός αποτελεσμάτων ──
    res                    = df_list[0].copy()
    del res['json']
    res['Train Time']      = bo.tr
    res['Inf Time']        = bo.tm
    res['Preprocess Time'] = bo.tl
    res['Selections']      = bo.selections

    arms      = len(latencies)
    length    = len(latencies[0])
    best_lats = [min(latencies[k][i] for k in range(arms))
                 for i in range(length)]

    best        = np.cumsum(best_lats) / 1000 / 60
    post        = np.cumsum(latencies[0]) / 1000 / 60
    total_time, exe = get_custom(latencies, res)
    total_final = np.cumsum(total_time) / 60
    exe_final   = np.cumsum(exe) / 60
    improvement = (post[-1] - exe_final[-1]) / post[-1] * 100

    print(f"\n  ┌─ Αποτελέσματα {name.upper()} {'─'*28}")
    print(f"  │  Best Possible : {best[-1]:.4f} min")
    print(f"  │  PostgreSQL    : {post[-1]:.4f} min")
    print(f"  │  Total Time    : {total_final[-1]:.4f} min")
    print(f"  │  Query Time    : {exe_final[-1]:.4f} min")
    print(f"  │  Improvement   : {improvement:.1f}%")
    print(f"  └{'─'*45}")

    return {
        'method':        name,
        'best_possible': round(best[-1],        4),
        'postgres':      round(post[-1],         4),
        'total_time':    round(total_final[-1],  4),
        'query_time':    round(exe_final[-1],    4),
        'improvement_%': round(improvement,      1),
    }

# ═══════════════════════════════════════════════════════════════
# 5. MAIN
# ═══════════════════════════════════════════════════════════════
ALL_METHODS = {
    'avgdl':     make_avgdl,
    'bao':       make_bao,
    'neo':       make_neo,
    'rtos':      make_rtos,
    'aimeetsai': make_aimeetsai,
    'rejoin':    make_rejoin,
    # Προστίθενται μετά επιβεβαίωση API:
    # 'e2ecost':     make_e2ecost,
    # 'plancost':    make_plancost,
    # 'prestroid':   make_prestroid,
    # 'queryformer': make_queryformer,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('methods', nargs='*',
                        help='Μέθοδοι προς εκτέλεση (κενό = όλες)')
    args_cli = parser.parse_args()
    to_run   = args_cli.methods if args_cli.methods \
               else list(ALL_METHODS.keys())

    df_list, planss, latencies, rootss, ds_info = load_data()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"✓ Device: {device}")

    results  = []
    out_dir  = os.path.join(REF_PATH, 'experiments', 'results')
    os.makedirs(out_dir, exist_ok=True)
    partial_csv = os.path.join(out_dir, 'comparison_partial.csv')
    final_csv   = os.path.join(out_dir, 'comparison_final.csv')

    for name in to_run:
        if name not in ALL_METHODS:
            print(f"✗ Άγνωστη μέθοδος: {name} "
                  f"(διαθέσιμες: {list(ALL_METHODS.keys())})")
            continue
        result = run_method(
            name, ALL_METHODS[name],
            df_list, planss, latencies, rootss, ds_info, device)
        if result:
            results.append(result)
            pd.DataFrame(results).to_csv(partial_csv, index=False)
            print(f"  ✓ Ενδιάμεση αποθήκευση: {partial_csv}")

    if results:
        df_out = (pd.DataFrame(results)
                  .sort_values('query_time')
                  .reset_index(drop=True))
        print("\n" + "="*65)
        print("  ΤΕΛΙΚΑ ΑΠΟΤΕΛΕΣΜΑΤΑ")
        print("="*65)
        print(df_out.to_string(index=False))
        df_out.to_csv(final_csv, index=False)
        print(f"\n✓ Αποθηκεύτηκε: {final_csv}")
