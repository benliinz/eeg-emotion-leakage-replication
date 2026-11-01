"""Regenerate Figures 5, 6, 7, 9, 10 from ../results (same pixel size / style as the originals)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
import os
RES = Path(os.environ.get('V2_RES', HERE.parent / 'results')); OUT = Path(os.environ.get('V2_FIG', HERE.parent / 'figures')); OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.size': 11, 'axes.grid': True, 'grid.alpha': .4})
DPI = 300
BLUE, ORANGE, GREY = '#0072B2', '#D55E00', '#7f7f7f'

rf = json.load(open(RES / 'results_rf.json')); prf = json.load(open(RES / 'per_subject_rf.json'))
cnn = json.load(open(RES / 'results_cnn.json')); pcnn = json.load(open(RES / 'per_subject_cnn.json'))
sub = np.arange(1, 33)


def save(fig, name, px):
    fig.set_size_inches(px[0] / DPI, px[1] / DPI); fig.tight_layout()
    fig.savefig(OUT / name, dpi=DPI); plt.close(fig)


def per_subject(acc, maj, title, color, name):
    fig, ax = plt.subplots()
    ax.bar(sub, acc, color=color, edgecolor='black', alpha=.75, label='Accuracy (trial-level)')
    ax.scatter(sub, maj, marker='_', s=180, color='black', zorder=3, label='Majority-class baseline')
    ax.axhline(np.mean(acc), color='#d62728', ls='--', lw=2, label=f'Mean accuracy = {np.mean(acc):.2f}%')
    ax.axhline(np.mean(maj), color=GREY, ls=':', lw=2, label=f'Mean baseline = {np.mean(maj):.2f}%')
    ax.set(xlabel='Subject', ylabel='Accuracy (%)', ylim=(0, 100), xticks=sub, title=title)
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    save(fig, name, (2323, 1060))


maj_v = prf['valence']['majority']
per_subject(pcnn['cnn_valence_original_trial'], maj_v,
            'CNN-LSTM, Valence, subject-dependent trial-level 5-fold CV (mean of 3 seeds)', '#87CEEB', 'figure5_cnnlstm_valence.png')
per_subject(prf['valence']['trial_acc'], maj_v,
            'Random Forest, Valence, subject-dependent trial-level 5-fold CV', '#F4A261', 'figure6_rf_valence.png')

# Figure 7: three protocols x two models x two dimensions
fig, axes = plt.subplots(1, 2, sharey=True)
labels = ['Trial-level\n(trial unit)', 'Window unit,\ntrial-split', 'Window unit,\nwindow-split (leaky)']
for ax, dim in zip(axes, ['valence', 'arousal']):
    sets = {'Random Forest': [prf[dim]['trial_acc'], prf[dim]['window_trialsplit_acc'], prf[dim]['window_leaky_acc']],
            'CNN': [pcnn[f'cnn_{dim}_original_trial'], pcnn[f'cnn_{dim}_original_window_trialsplit'], pcnn[f'cnn_{dim}_original_window_leaky']]}
    x = np.arange(3); wdt = .38
    for k, (name, col) in enumerate([('Random Forest', BLUE), ('CNN', ORANGE)]):
        m = [np.mean(v) for v in sets[name]]; s = [np.std(v, ddof=1) for v in sets[name]]
        ax.bar(x + (k - .5) * wdt, m, wdt, yerr=s, capsize=3, color=col, alpha=.85, edgecolor='black',
               hatch=['', '', '//'], label=name if dim == 'valence' else None)
        for xi, mi, si in zip(x + (k - .5) * wdt, m, s): ax.text(xi, mi + si + 1, f'{mi:.1f}', ha='center', va='bottom', fontsize=7, rotation=90)
    ax.axhline(np.mean(prf[dim]['majority']), color=GREY, ls=':', lw=1.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7); ax.set_ylim(40, 100); ax.set_title(dim.capitalize())
axes[0].set_ylabel('Accuracy (%)'); axes[0].legend(fontsize=7, loc='upper left')
fig.suptitle('Effect of the partition on accuracy (n = 32; dotted line = majority-class baseline)', fontsize=9)
save(fig, 'figure7_leakage.png', (1858, 1393))

# Figure 9: generalization (Valence and Arousal, accuracy vs balanced accuracy)
fig, ax = plt.subplots()
prot = ['Trial-level\n(subject-dep.)', 'Last-10-videos\nsplit', 'Leave-one-\nsubject-out']
for k, (dim, col) in enumerate([('valence', BLUE), ('arousal', ORANGE)]):
    acc = [rf[f'rf_{dim}']['trial']['acc'], rf[f'chronological_{dim}']['summary']['acc'], rf[f'loso_{dim}']['acc']]
    bal = [rf[f'rf_{dim}']['trial']['bal_acc'], rf[f'chronological_{dim}']['summary']['bal_acc'], rf[f'loso_{dim}']['bal_acc']]
    x = np.arange(3) + (k - .5) * .38
    ax.bar(x, [a['mean'] for a in acc], .38, yerr=[a['sd'] for a in acc], capsize=3, color=col, alpha=.85, edgecolor='black', label=f'{dim.capitalize()} accuracy')
    ax.scatter(x, [b['mean'] for b in bal], marker='D', color='black', zorder=3, s=18, label='Balanced accuracy' if k == 0 else None)
ax.axhline(50, color=GREY, ls=':', lw=1.5)
ax.set_xticks(np.arange(3)); ax.set_xticklabels(prot, fontsize=8); ax.set_ylim(0, 100); ax.set_ylabel('Accuracy (%)')
ax.set_title('Random Forest under alternative partitions (n = 32)', fontsize=10); ax.legend(fontsize=7, loc='upper right')
save(fig, 'figure8_generalization.png', (1541, 1116))

# Figure 10: band ablation (balanced accuracy)
fig, ax = plt.subplots()
bands = ['theta', 'alpha', 'beta', 'gamma']
for k, (dim, col) in enumerate([('valence', BLUE), ('arousal', ORANGE)]):
    v = [rf[f'ablation_{dim}'][b]['bal_acc'] for b in bands] + [rf[f'rf_{dim}']['trial']['bal_acc']]
    x = np.arange(5) + (k - .5) * .38
    ax.bar(x, [a['mean'] for a in v], .38, yerr=[a['sd'] for a in v], capsize=3, color=col, alpha=.85, edgecolor='black', label=dim.capitalize())
ax.axhline(50, color=GREY, ls=':', lw=1.5)
ax.set_xticks(np.arange(5)); ax.set_xticklabels(['Theta', 'Alpha', 'Beta', 'Gamma', 'All bands'])
ax.set_ylim(30, 80); ax.set_ylabel('Balanced accuracy (%)'); ax.legend(fontsize=8)
ax.set_title('Single-band Random Forest, trial-level CV (n = 32)', fontsize=10)
save(fig, 'figure10_band_ablation.png', (1592, 1017))
print('figures written to', OUT)
