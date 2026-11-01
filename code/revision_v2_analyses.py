"""
Revision v2 — single, seeded re-analysis of every number reported in the manuscript.

Input  : ../../02_特征与数据/de_features_all.mat  (read-only; 39680 x 56 DE features, 1280 trials x 31 windows)
Output : ../results/*.json, ../results/*.csv

Facts about the cache that this script relies on (verified, see 修改说明_v2.md):
  * Features are BAND-MAJOR: index f = band*14 + channel (MATLAB column-major flatten of a 14x4 matrix).
  * Each DEAP preprocessed trial has 8064 samples = 63 s (3 s pre-trial baseline + 60 s stimulus).
    floor(8064/256) = 31 non-overlapping 2-s windows cover 0-62 s, so windows 0 and 1 contain baseline.
  * Trials in data_preprocessed_matlab are stored in video (Experiment_id) order, not presentation order.

Usage:  python revision_v2_analyses.py rf      # all Random-Forest / label / asymmetry analyses (minutes)
        python revision_v2_analyses.py perm    # permutation test (~15 min)
        python revision_v2_analyses.py cnn     # CNN-LSTM / static-CNN analyses (longer)
"""
import json, sys, time
from pathlib import Path
import numpy as np, h5py
from scipy import stats
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score, recall_score)

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent.parent / '02_特征与数据' / 'de_features_all.mat'
OUT = HERE.parent / 'results'; OUT.mkdir(exist_ok=True)
SEED = 42
N_SUB, N_TR, N_WIN, N_CH, N_BAND = 32, 40, 31, 14, 4
CH = ['AF3', 'F3', 'F7', 'FC5', 'C3', 'P7', 'O1', 'AF4', 'F4', 'F8', 'FC6', 'T8', 'P8', 'O2']
BANDS = ['theta', 'alpha', 'beta', 'gamma']
STIM_WIN = slice(2, 31)  # windows fully after the 3-s pre-trial baseline (4-62 s)

with h5py.File(CACHE, 'r') as f:
    FEAT = np.array(f['all_features']).T            # (39680, 56)
    LAB = np.array(f['all_labels']).T               # (39680, 4)
XW = FEAT.reshape(N_SUB * N_TR, N_WIN, 56)          # (1280, 31, 56)
RAT = LAB.reshape(N_SUB * N_TR, N_WIN, 4)[:, 0, :]  # continuous SAM ratings per trial


def binarize(dim, rule='ge5'):
    r = RAT[:, dim]
    return (r >= 5).astype(int) if rule == 'ge5' else (r > 5).astype(int)


def metrics(y, pred, proba):
    out = dict(acc=accuracy_score(y, pred) * 100,
               bal_acc=balanced_accuracy_score(y, pred) * 100,
               f1_pos=f1_score(y, pred, zero_division=0) * 100,
               macro_f1=f1_score(y, pred, average='macro', zero_division=0) * 100,
               sens=recall_score(y, pred, pos_label=1, zero_division=0) * 100,
               spec=recall_score(y, pred, pos_label=0, zero_division=0) * 100)
    out['auroc'] = roc_auc_score(y, proba) * 100 if len(np.unique(y)) == 2 else np.nan
    return out


def summarize(per_sub):
    keys = per_sub[0].keys()
    return {k: dict(mean=float(np.nanmean([d[k] for d in per_sub])),
                    sd=float(np.nanstd([d[k] for d in per_sub], ddof=1)),
                    n=int(np.sum(~np.isnan([d[k] for d in per_sub]))))
            for k in keys}


def paired(a, b, label):
    """b - a, paired across subjects."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a; n = len(d)
    t, p = stats.ttest_rel(b, a)
    se = d.std(ddof=1) / np.sqrt(n); tc = stats.t.ppf(0.975, n - 1)
    try: w_p = float(stats.wilcoxon(b, a).pvalue)
    except ValueError: w_p = float('nan')
    return dict(comparison=label, mean_diff=float(d.mean()), ci95=[float(d.mean() - tc * se), float(d.mean() + tc * se)],
                t=float(t), df=n - 1, p=float(p), d_z=float(d.mean() / d.std(ddof=1)), wilcoxon_p=w_p)


def holm(results):
    ps = np.array([r['p'] for r in results]); order = np.argsort(ps); m = len(ps)
    adj = np.empty(m); run = 0
    for rank, i in enumerate(order):
        run = max(run, min(1, (m - rank) * ps[i])); adj[i] = run
    for r, a in zip(results, adj): r['p_holm'] = float(a)
    return results


def rf():
    return RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1)


def trial_folds(y_sub, stratified=False):
    kf = StratifiedKFold(5, shuffle=True, random_state=SEED) if stratified else KFold(5, shuffle=True, random_state=SEED)
    return list(kf.split(np.zeros(len(y_sub)), y_sub))


# ---------------------------------------------------------------- per-subject workers
def rf_trial_subject(s, y, feats, stratified=False):
    idx = slice(s * N_TR, (s + 1) * N_TR)
    X, ys = feats[idx], y[idx]
    pred = np.zeros(N_TR, int); proba = np.zeros(N_TR)
    for tr, te in trial_folds(ys, stratified):
        clf = rf().fit(X[tr], ys[tr])
        pred[te] = clf.predict(X[te])
        proba[te] = clf.predict_proba(X[te])[:, 1] if clf.n_classes_ == 2 else float(clf.classes_[0])
    return metrics(ys, pred, proba)


def rf_window_subject(s, y, grouped):
    """grouped=False: windows shuffled across folds (leaky). grouped=True: same trial folds as trial-level."""
    Xs = XW[s * N_TR:(s + 1) * N_TR]                     # (40, 31, 56)
    ys = y[s * N_TR:(s + 1) * N_TR]
    Xwin = Xs.reshape(-1, 56); ywin = np.repeat(ys, N_WIN); gid = np.repeat(np.arange(N_TR), N_WIN)
    pred = np.zeros(len(ywin), int); proba = np.zeros(len(ywin))
    if grouped:
        splits = [(np.isin(gid, tr), np.isin(gid, te)) for tr, te in trial_folds(ys)]
    else:
        splits = [(tr, te) for tr, te in KFold(5, shuffle=True, random_state=SEED).split(Xwin)]
    for tr, te in splits:
        clf = rf().fit(Xwin[tr], ywin[tr])
        pred[te] = clf.predict(Xwin[te])
        proba[te] = clf.predict_proba(Xwin[te])[:, 1] if clf.n_classes_ == 2 else float(clf.classes_[0])
    out = metrics(ywin, pred, proba)
    if grouped:  # trial-level decision by averaging window probabilities
        tp = proba.reshape(N_TR, N_WIN).mean(1)
        tm = metrics(ys, (tp > 0.5).astype(int), tp)
        out.update({f'trialvote_{k}': v for k, v in tm.items()})
    return out


def run_par(fn, *args):
    return Parallel(n_jobs=-1)(delayed(fn)(s, *args) for s in range(N_SUB))


# ---------------------------------------------------------------- analyses
def label_stats():
    rows = []
    for s in range(N_SUB):
        sl = slice(s * N_TR, (s + 1) * N_TR)
        row = dict(subject=s + 1)
        for dim, name in [(0, 'valence'), (1, 'arousal')]:
            r = RAT[sl, dim]
            for rule in ['ge5', 'gt5']:
                p = binarize(dim, rule)[sl].mean()
                row[f'{name}_{rule}_high_prop'] = float(p)
                row[f'{name}_{rule}_majority_acc'] = float(max(p, 1 - p) * 100)
            row[f'{name}_exactly5'] = int((r == 5).sum())
        rows.append(row)
    # trial-index order effect (Experiment_id order): high-valence proportion by trial position
    yv = binarize(0).reshape(N_SUB, N_TR); ya = binarize(1).reshape(N_SUB, N_TR)
    order = dict(valence_high_prop_by_trial=yv.mean(0).round(3).tolist(),
                 arousal_high_prop_by_trial=ya.mean(0).round(3).tolist(),
                 valence_high_first30=float(yv[:, :30].mean()), valence_high_last10=float(yv[:, 30:].mean()),
                 arousal_high_first30=float(ya[:, :30].mean()), arousal_high_last10=float(ya[:, 30:].mean()))
    return rows, order


def chronological(dim):
    y = binarize(dim); flat = XW.reshape(len(XW), -1); per = []; single_class = 0; shift = []
    for s in range(N_SUB):
        sl = slice(s * N_TR, (s + 1) * N_TR); X, ys = flat[sl], y[sl]
        clf = rf().fit(X[:30], ys[:30])
        pred = clf.predict(X[30:]); proba = clf.predict_proba(X[30:])[:, 1] if clf.n_classes_ == 2 else np.full(10, float(clf.classes_[0]))
        m = metrics(ys[30:], pred, proba)
        maj_train = int(ys[:30].mean() >= 0.5)
        m['train_majority_rule_acc'] = float((ys[30:] == maj_train).mean() * 100)
        per.append(m); single_class += len(np.unique(ys[30:])) == 1
        shift.append(ys[30:].mean() - ys[:30].mean())
    return dict(summary=summarize(per), n_subjects_single_class_test=int(single_class),
                mean_label_shift_last10_minus_first30=float(np.mean(shift)))


def loso(dim):
    y = binarize(dim); flat = XW.reshape(len(XW), -1); sub = np.repeat(np.arange(N_SUB), N_TR)

    def one(s):
        tr, te = sub != s, sub == s
        clf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1).fit(flat[tr], y[tr])
        m = metrics(y[te], clf.predict(flat[te]), clf.predict_proba(flat[te])[:, 1])
        p = y[te].mean(); m['majority_acc'] = max(p, 1 - p) * 100
        return m
    return summarize(Parallel(n_jobs=-1)(delayed(one)(s) for s in range(N_SUB)))


def frontal_asymmetry():
    pairs = [('AF4', 'AF3'), ('F4', 'F3'), ('F8', 'F7'), ('FC6', 'FC5')]
    yv = binarize(0); res = []
    for b in ['alpha', 'gamma']:
        bi = BANDS.index(b)
        de = XW[:, STIM_WIN, bi * 14:(bi + 1) * 14].mean(1)        # (1280, 14) trial-mean DE, stimulus windows only
        idxs = {f'{r}-{l}': de[:, CH.index(r)] - de[:, CH.index(l)] for r, l in pairs}  # = 0.5*ln(P_R/P_L)
        idxs['frontal_composite'] = np.mean(list(idxs.values()), axis=0)
        for name, ai in idxs.items():
            hi = [ai[s * N_TR:(s + 1) * N_TR][yv[s * N_TR:(s + 1) * N_TR] == 1].mean() for s in range(N_SUB)]
            lo = [ai[s * N_TR:(s + 1) * N_TR][yv[s * N_TR:(s + 1) * N_TR] == 0].mean() for s in range(N_SUB)]
            r = paired(lo, hi, f'{b} {name}: High - Low valence'); r['band'] = b; r['index'] = name
            res.append(r)
    pair_tests = holm([r for r in res if r['index'] != 'frontal_composite'])
    return pair_tests + [r for r in res if r['index'] == 'frontal_composite']


def part_rf():
    t0 = time.time(); R = {}
    rows, order = label_stats(); R['label_order'] = order
    import csv
    with open(OUT / 'per_subject_labels.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    R['majority'] = {d: {r: dict(mean=float(np.mean([x[f'{d}_{r}_majority_acc'] for x in rows])),
                                 sd=float(np.std([x[f'{d}_{r}_majority_acc'] for x in rows], ddof=1)))
                         for r in ['ge5', 'gt5']} for d in ['valence', 'arousal']}
    R['exactly5_trials'] = {d: int(sum(x[f'{d}_exactly5'] for x in rows)) for d in ['valence', 'arousal']}
    flat = XW.reshape(len(XW), -1); per_sub = {}; tests = []
    for dim, name in [(0, 'valence'), (1, 'arousal')]:
        y = binarize(dim)
        maj = [x[f'{name}_ge5_majority_acc'] for x in rows]
        trial = run_par(rf_trial_subject, y, flat)
        trial_strat = run_par(rf_trial_subject, y, flat, True)
        trial_gt5 = run_par(rf_trial_subject, binarize(dim, 'gt5'), flat)
        win_leak = run_par(rf_window_subject, y, False)
        win_group = run_par(rf_window_subject, y, True)
        R[f'rf_{name}'] = dict(trial=summarize(trial), trial_stratified=summarize(trial_strat),
                               trial_threshold_gt5=summarize(trial_gt5),
                               window_leaky=summarize(win_leak), window_trialsplit=summarize(win_group))
        a = lambda L, k='acc': [d[k] for d in L]
        tests += [paired(a(trial), a(win_leak), f'RF {name}: window-leaky - trial-level (acc)'),
                  paired(a(win_group), a(win_leak), f'RF {name}: window-leaky - window-trialsplit (acc, unit fixed)'),
                  paired(a(trial, 'bal_acc'), a(win_leak, 'bal_acc'), f'RF {name}: window-leaky - trial-level (bal acc)')]
        R[f'rf_{name}']['trial_vs_majority'] = paired(maj, a(trial), f'RF {name}: trial acc - majority baseline')
        R[f'rf_{name}']['trial_balacc_vs_50'] = dict(zip(['t', 'p'], map(float, stats.ttest_1samp(a(trial, 'bal_acc'), 50))))
        per_sub[name] = dict(trial_acc=a(trial), trial_bal=a(trial, 'bal_acc'), window_leaky_acc=a(win_leak),
                             window_trialsplit_acc=a(win_group), majority=maj)
        # band ablation (trial-level RF, correct band indexing)
        abl = {}
        for bi, b in enumerate(BANDS):
            fb = XW[:, :, bi * 14:(bi + 1) * 14].reshape(len(XW), -1)
            abl[b] = run_par(rf_trial_subject, y, fb)
        R[f'ablation_{name}'] = {b: summarize(v) for b, v in abl.items()}
        R[f'ablation_{name}_tests'] = holm([paired(a(abl[b], 'bal_acc'), a(abl['gamma'], 'bal_acc'),
                                                   f'{name} gamma - {b} (bal acc)') for b in BANDS[:3]])
        R[f'chronological_{name}'] = chronological(dim)
        R[f'loso_{name}'] = loso(dim)
    R['leakage_tests_rf'] = holm(tests)
    R['frontal_asymmetry'] = frontal_asymmetry()
    json.dump(per_sub, open(OUT / 'per_subject_rf.json', 'w'), indent=1)
    json.dump(R, open(OUT / 'results_rf.json', 'w'), indent=1)
    print(f'rf part done in {time.time() - t0:.0f}s')


def part_perm(n_perm=200):
    flat = XW.reshape(len(XW), -1); R = {}
    for dim, name in [(0, 'valence'), (1, 'arousal')]:
        y = binarize(dim)
        obs = run_par(rf_trial_subject, y, flat)
        obs_acc = np.mean([d['acc'] for d in obs]); obs_bal = np.mean([d['bal_acc'] for d in obs])
        rng = np.random.default_rng(SEED); null_acc, null_bal = [], []
        for i in range(n_perm):
            yp = y.copy()
            for s in range(N_SUB):
                sl = slice(s * N_TR, (s + 1) * N_TR); yp[sl] = rng.permutation(yp[sl])
            r = run_par(rf_trial_subject, yp, flat)
            null_acc.append(np.mean([d['acc'] for d in r])); null_bal.append(np.mean([d['bal_acc'] for d in r]))
        R[name] = dict(n_perm=n_perm, obs_acc=float(obs_acc), obs_bal=float(obs_bal),
                       null_acc_mean=float(np.mean(null_acc)), null_acc_95=float(np.percentile(null_acc, 95)),
                       null_bal_mean=float(np.mean(null_bal)), null_bal_95=float(np.percentile(null_bal, 95)),
                       p_acc=float((np.sum(np.array(null_acc) >= obs_acc) + 1) / (n_perm + 1)),
                       p_bal=float((np.sum(np.array(null_bal) >= obs_bal) + 1) / (n_perm + 1)))
        print(name, R[name], flush=True)
    json.dump(R, open(OUT / 'results_perm.json', 'w'), indent=1)


# ---------------------------------------------------------------- CNN part
def to_input(x, grid):
    """x (..., 56) band-major. 'original' = the published reshape (rows are NOT channels);
    'corrected' = true 14-channel x 4-band matrix."""
    if grid == 'original':
        return x
    lead = x.shape[:-1]
    return x.reshape(*lead, N_BAND, N_CH).swapaxes(-1, -2).reshape(*lead, 56)


def cnn_subject(s, dim, protocol, grid, seed):
    import torch, torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    torch.set_num_threads(1); torch.manual_seed(seed); np.random.seed(seed)

    class SpatioTemporalModel(nn.Module):  # identical to train_hybrid_emotion_model_remote.py
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, (3, 2), padding=1); self.relu = nn.ReLU(); self.pool = nn.MaxPool2d((2, 2))
            self.conv2 = nn.Conv2d(16, 32, (3, 2), padding=1); self.flatten = nn.Flatten()
            self.lstm = nn.LSTM(672, 64, num_layers=2, batch_first=True, dropout=0.3)
            self.dropout = nn.Dropout(0.5); self.fc = nn.Linear(64, 2)

        def forward(self, x):
            b, t, _ = x.size(); x = x.view(b * t, 1, 14, 4)
            x = self.flatten(self.relu(self.conv2(self.pool(self.relu(self.conv1(x))))))
            out, _ = self.lstm(x.view(b, t, -1))
            return self.fc(self.dropout(out[:, -1, :]))

    class StaticCNNModel(nn.Module):  # identical to train_window_level_leakage.py
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, (3, 2), padding=1); self.relu = nn.ReLU(); self.pool = nn.MaxPool2d((2, 2))
            self.conv2 = nn.Conv2d(16, 32, (3, 2), padding=1); self.flatten = nn.Flatten()
            self.dropout = nn.Dropout(0.5); self.fc1 = nn.Linear(672, 64); self.fc2 = nn.Linear(64, 2)

        def forward(self, x):
            x = x.view(-1, 1, 14, 4)
            x = self.dropout(self.flatten(self.relu(self.conv2(self.pool(self.relu(self.conv1(x)))))))
            return self.fc2(self.relu(self.fc1(x)))

    y = binarize(dim)[s * N_TR:(s + 1) * N_TR]
    Xs = to_input(XW[s * N_TR:(s + 1) * N_TR], grid).astype(np.float32)  # (40, 31, 56)

    def fit_predict(model, Xtr, ytr, Xte, epochs, bs, lr, wd, norm_dims):
        mu = Xtr.mean(norm_dims, keepdims=True); sd = Xtr.std(norm_dims, keepdims=True); sd[sd == 0] = 1e-6
        Xtr = torch.tensor(np.ascontiguousarray((Xtr - mu) / sd)); Xte = torch.tensor(np.ascontiguousarray((Xte - mu) / sd))
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd); lossf = nn.CrossEntropyLoss()
        g = torch.Generator().manual_seed(seed)
        ld = DataLoader(TensorDataset(Xtr, torch.tensor(ytr)), batch_size=bs, shuffle=True, generator=g)
        for _ in range(epochs):   # fixed epoch budget; no validation set, no early stopping, no LR schedule
            model.train()
            for bx, by in ld:
                opt.zero_grad(); lossf(model(bx), by).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            return torch.softmax(model(Xte), 1)[:, 1].numpy()

    if protocol == 'trial':        # CNN-LSTM, one sequence of 31 windows per trial
        proba = np.zeros(N_TR)
        for tr, te in trial_folds(y):
            proba[te] = fit_predict(SpatioTemporalModel(), Xs[tr], y[tr], Xs[te], 50, 8, 5e-4, 1e-4, (0, 1))
        return metrics(y, (proba > 0.5).astype(int), proba)
    Xwin = Xs.reshape(-1, 56); ywin = np.repeat(y, N_WIN).astype(np.int64); gid = np.repeat(np.arange(N_TR), N_WIN)
    if protocol == 'window_leaky':
        splits = list(KFold(5, shuffle=True, random_state=SEED).split(Xwin))
    else:                          # window_trialsplit: same trial folds as the trial-level protocol
        splits = [(np.where(np.isin(gid, tr))[0], np.where(np.isin(gid, te))[0]) for tr, te in trial_folds(y)]
    proba = np.zeros(len(ywin))
    for tr, te in splits:
        proba[te] = fit_predict(StaticCNNModel(), Xwin[tr], ywin[tr], Xwin[te], 50, 64, 1e-3, 0, (0,))
    out = metrics(ywin, (proba > 0.5).astype(int), proba)
    if protocol == 'window_trialsplit':
        tp = proba.reshape(N_TR, N_WIN).mean(1)
        out.update({f'trialvote_{k}': v for k, v in metrics(y, (tp > 0.5).astype(int), tp).items()})
    return out


def part_cnn(seeds=(42, 43, 44)):
    R = {}; per_sub = {}
    for dim, name in [(0, 'valence'), (1, 'arousal')]:
        for grid in ['original', 'corrected']:
            for protocol in ['trial', 'window_leaky', 'window_trialsplit']:
                t0 = time.time(); runs = []
                for sd in seeds:
                    runs.append(Parallel(n_jobs=-1)(delayed(cnn_subject)(s, dim, protocol, grid, sd) for s in range(N_SUB)))
                # per-subject value = mean over seeds
                keys = runs[0][0].keys()
                avg = [{k: float(np.nanmean([r[s][k] for r in runs])) for k in keys} for s in range(N_SUB)]
                key = f'cnn_{name}_{grid}_{protocol}'
                R[key] = dict(summary=summarize(avg), seed_means_acc=[float(np.mean([d['acc'] for d in r])) for r in runs])
                per_sub[key] = [d['acc'] for d in avg]
                print(key, f"{R[key]['summary']['acc']['mean']:.2f} ± {R[key]['summary']['acc']['sd']:.2f}",
                      f"seeds={R[key]['seed_means_acc']}", f'{time.time() - t0:.0f}s', flush=True)
                json.dump(R, open(OUT / 'results_cnn.json', 'w'), indent=1)
    tests = []
    for name in ['valence', 'arousal']:
        for grid in ['original', 'corrected']:
            k = lambda p: per_sub[f'cnn_{name}_{grid}_{p}']
            tests += [paired(k('trial'), k('window_leaky'), f'CNN({grid}) {name}: window-leaky - trial-level CNN-LSTM'),
                      paired(k('window_trialsplit'), k('window_leaky'), f'CNN({grid}) {name}: window-leaky - window-trialsplit (static CNN, unit fixed)')]
    R['leakage_tests_cnn'] = holm(tests)
    json.dump(R, open(OUT / 'results_cnn.json', 'w'), indent=1)
    json.dump(per_sub, open(OUT / 'per_subject_cnn.json', 'w'), indent=1)


if __name__ == '__main__':
    part = sys.argv[1] if len(sys.argv) > 1 else 'rf'
    {'rf': part_rf, 'perm': part_perm, 'cnn': part_cnn}[part]()
