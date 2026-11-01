"""
Build the v2 manuscript from a COPY of 05_大修交付物/Manuscript_Cognitive_Neurodynamics_revised.docx.

  python build_manuscript_v2.py tracked   -> ../Manuscript_v2_tracked_changes.docx
  python build_manuscript_v2.py clean     -> ../Manuscript_v2_clean.docx

Every number written into the manuscript is read from ../results/*.json, so text and analysis cannot drift.
The original files in 04_/05_ are never opened for writing.
"""
import json, os, shutil, sys, zipfile, copy
from pathlib import Path
import numpy as np
from scipy import stats
from lxml import etree
from docx_redline import Redliner, w, NS

MODE = sys.argv[1] if len(sys.argv) > 1 else 'tracked'
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
SRC = ROOT.parent / '05_大修交付物' / 'Manuscript_Cognitive_Neurodynamics_revised.docx'
OUT = Path(os.environ.get('V2_OUTDIR', ROOT)) / ('Manuscript_v2_tracked_changes.docx' if MODE == 'tracked' else 'Manuscript_v2_clean.docx')
RES = Path(os.environ.get('V2_RES', ROOT / 'results')); FIG = Path(os.environ.get('V2_FIG', ROOT / 'figures'))
R = Redliner(track=(MODE == 'tracked'))

rf = json.load(open(RES / 'results_rf.json')); prf = json.load(open(RES / 'per_subject_rf.json'))
cnn = json.load(open(RES / 'results_cnn.json')); pcnn = json.load(open(RES / 'per_subject_cnn.json'))
perm = json.load(open(RES / 'results_perm.json')) if (RES / 'results_perm.json').exists() else None
ovp = json.load(open(RES / 'results_overlap_params.json'))


# ------------------------------------------------------------------ number helpers
def ms(d, k='acc'): d = d[k] if k in d else d; return f"{d['mean']:.2f} ± {d['sd']:.2f}"
def m(d, k='acc'): d = d[k] if k in d else d; return f"{d['mean']:.2f}"
def pfmt(p): return 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); d = b - a; n = len(d)
    t, p = stats.ttest_rel(b, a); se = d.std(ddof=1) / np.sqrt(n); tc = stats.t.ppf(.975, n - 1)
    return dict(diff=d.mean(), lo=d.mean() - tc * se, hi=d.mean() + tc * se, t=t, p=p, dz=d.mean() / d.std(ddof=1))


def holm(ps):
    ps = np.asarray(ps); o = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for r, i in enumerate(o): run = max(run, min(1, (len(ps) - r) * ps[i])); adj[i] = run
    return adj


RFk = lambda dim, c: rf[f'rf_{dim}'][c]
CN = lambda dim, proto, grid='original': cnn[f'cnn_{dim}_{grid}_{proto}']['summary']
CNs = lambda dim, proto, grid='original': cnn[f'cnn_{dim}_{grid}_{proto}']['seed_means_acc']

# leakage test family (8 accuracy comparisons) with Holm correction
fam = []
for dim in ['valence', 'arousal']:
    fam.append((f'RF, {dim}', 'window-split vs trial-level (unit changes)', paired(prf[dim]['trial_acc'], prf[dim]['window_leaky_acc'])))
    fam.append((f'RF, {dim}', 'window-split vs trial-split (window unit fixed)', paired(prf[dim]['window_trialsplit_acc'], prf[dim]['window_leaky_acc'])))
    fam.append((f'CNN, {dim}', 'static CNN window-split vs CNN-LSTM trial-level', paired(pcnn[f'cnn_{dim}_original_trial'], pcnn[f'cnn_{dim}_original_window_leaky'])))
    fam.append((f'CNN, {dim}', 'static CNN window-split vs static CNN trial-split', paired(pcnn[f'cnn_{dim}_original_window_trialsplit'], pcnn[f'cnn_{dim}_original_window_leaky'])))
for (a, b, r_), ph in zip(fam, holm([x[2]['p'] for x in fam])): r_['p_holm'] = ph
infl = [x[2]['diff'] for x in fam]; dzs = [x[2]['dz'] for x in fam]
infl_range = f'{min(infl):.1f}–{max(infl):.1f}'; dz_range = f'{min(dzs):.2f}–{max(dzs):.2f}'
max_ph = max(x[2]['p_holm'] for x in fam)
cnn_vs_rf = {dim: paired(prf[dim]['trial_acc'], pcnn[f'cnn_{dim}_original_trial']) for dim in ['valence', 'arousal']}
grid_fix = {dim: paired(pcnn[f'cnn_{dim}_original_trial'], pcnn[f'cnn_{dim}_corrected_trial']) for dim in ['valence', 'arousal']}
ov = ovp['overlap']; asym = rf['frontal_asymmetry']
abl_t = {dim: {t['comparison'].split(' - ')[1].split(' ')[0]: t for t in rf[f'ablation_{dim}_tests']} for dim in ['valence', 'arousal']}
lo = rf['label_order']

# ------------------------------------------------------------------ open the copy
work = ROOT / 'build' / f'work_{MODE}'
if work.exists(): shutil.rmtree(work)
work.mkdir(parents=True)
with zipfile.ZipFile(SRC) as z: z.extractall(work)
for f in work.rglob('*'):
    if f.is_symlink(): f.unlink()
# coalesce fragmented runs (inline replacement for the docx-skill merge_runs helper)
doc_path = work / 'word' / 'document.xml'
tree = etree.parse(str(doc_path)); body = tree.getroot().find(w('body'))


def _runs_equal(r1, r2):
    a = r1.find(w('rPr')); b = r2.find(w('rPr'))
    if (a is None) != (b is None): return False
    if a is None: return True
    return etree.tostring(a) == etree.tostring(b)


for p in body.iter(w('p')):
    runs = p.findall(w('r'))
    if len(runs) < 2: continue
    merged = [runs[0]]
    for r_ in runs[1:]:
        prev = merged[-1]
        if _runs_equal(prev, r_) and prev.find(w('t')) is not None and r_.find(w('t')) is not None:
            prev.find(w('t')).text = (prev.find(w('t')).text or '') + (r_.find(w('t')).text or '')
            prev.find(w('t')).set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
            p.remove(r_)
        else:
            merged.append(r_)
# drop Word's lastRenderedPageBreak hints
for r_ in body.iter(w('r')):
    for e in r_.findall(w('lastRenderedPageBreak')): r_.remove(e)
P = [c for c in body if c.tag == w('p')]; T = [c for c in body if c.tag == w('tbl')]
assert P[0].xpath('string()').startswith('Reevaluating') and len(T) == 4, 'unexpected source document'


def chk(i, start):
    assert R.text(P[i]).lstrip().startswith(start), f'P{i} does not start with {start!r}: {R.text(P[i])[:60]!r}'
    return P[i]


sect = body.find(w('sectPr')); pg = sect.find(w('pgSz')); mar = sect.find(w('pgMar'))
TEXT_W = int(pg.get(w('w'))) - int(mar.get(w('left'))) - int(mar.get(w('right')))
_mk = R.make_table
R.make_table = lambda like, rows, weights=None, **kw: _mk(like, rows, weights, total_width=TEXT_W, **kw)
body_style = chk(11, 'Despite')            # [Body Text]
first_style = chk(10, 'Emotion recognition')  # [First Paragraph]
h2_style = chk(103, '4.5 '); h3_style = chk(104, '4.5.1')
cap_style = chk(101, 'Table 1.')

# ================================================================== TITLE / AUTHOR ORDER
# Re-order the two co-first authors: Junwei Li first, Zhitong Li second (both retain the
# equal-contribution marker). Their affiliation letters (a/b) move with them.
R.replace_sub(P[1], 'Zhitong Li a,1, Junwei Li b,1,', 'Junwei Li a,1, Zhitong Li b,1,')
R.replace_sub(P[2], 'Faculty of Science, The University of Auckland', 'Faculty of Medical and Health Science, The University of Auckland')
R.replace_sub(P[3], 'Faculty of Medical and Health Science, The University of Auckland', 'Faculty of Science, The University of Auckland')

R.replace_sub(chk(0, 'Reevaluating'), 'A Topographic Differential Entropy Approach under Trial-Level Cross-Validation',
              'Quantifying Window-Level Leakage under Trial-Level Cross-Validation')

abstract_old = R.text(chk(7, 'Abstract'))[len('Abstract '):]
R.replace_sub(P[7], abstract_old,
    "Reported accuracies for EEG-based emotion recognition depend strongly on how data are partitioned for evaluation. "
    "When short windows from the same trial appear in both the training and the test set (window-level splitting), accuracy can be overestimated. "
    "We quantified this effect on the DEAP dataset (32 subjects; binary Valence and Arousal) using Differential Entropy (DE) features from 14 channels, "
    "a CNN-LSTM and a Random Forest (RF), and we report every result together with each subject's majority-class baseline and balanced accuracy. "
    f"Under subject-dependent trial-level 5-fold cross-validation, the CNN-LSTM reached {m(CN('valence','trial'))}% accuracy for Valence and {m(CN('arousal','trial'))}% for Arousal, "
    f"and the RF reached {m(RFk('valence','trial'))}% and {m(RFk('arousal','trial'))}% (balanced accuracy {m(RFk('valence','trial'),'bal_acc')}% and {m(RFk('arousal','trial'),'bal_acc')}%), "
    f"against majority-class baselines of {rf['majority']['valence']['ge5']['mean']:.2f}% and {rf['majority']['arousal']['ge5']['mean']:.2f}%. "
    f"Window-level splitting of the same non-overlapping 2-s windows raised accuracy by {infl_range} percentage points across models and dimensions "
    f"(paired t-tests, all Holm-corrected {'p < 0.001' if max_ph < 0.001 else f'p ≤ {max_ph:.3f}'}; d_z = {dz_range}), and the increase persisted when the classification unit was held fixed at the window level. "
    f"In an overlap scan with the RF, window-level accuracy for Valence increased monotonically from {ov['0']['mean']:.1f}% (0% overlap) to {ov['90']['mean']:.1f}% (90% overlap), "
    "showing that window-level splitting combined with high overlap can by itself yield accuracies above 90% on DEAP. "
    f"Leave-one-subject-out RF accuracy for Valence was {m(rf['loso_valence'])}% (balanced accuracy {m(rf['loso_valence'],'bal_acc')}%). "
    "Single-band analyses and a quantitative frontal-asymmetry test did not support a specific physiological interpretation of the Gamma band. "
    "These results provide a documented, reproducible trial-level baseline for this pipeline and show that the partitioning protocol must be reported and controlled before DEAP accuracies can be compared.")

# ================================================================== INTRODUCTION
R.replace_sub(chk(11, 'Despite'), '(Schlögl & Keinrath, 2007)', '(Schlögl et al., 2007)')
# keep in-text citation in step with the corrected reference year (Brain Sciences 16(1) = 2026)
R.replace_sub(chk(12, 'The paradigm'), 'Liu et al., 2025', 'Liu et al., 2026')
R.rewrite_para(chk(13, 'However, a critical'), [
    "However, accuracy estimates in this literature are sensitive to how the data are partitioned for evaluation (Moreno-Alcayde et al., 2023). "
    "In a typical EEG emotion recognition experiment, subjects watch emotional stimuli (e.g., one-minute music videos), producing long, continuous EEG trials. "
    "To enlarge the training set, these trials are commonly cut into short windows, often with substantial overlap (e.g., 2-s windows with 50% or 90% overlap). "
    "If the windows are then shuffled and split into folds without regard to the trial they came from (window-level splitting), windows from the same trial appear in both the training and the test set. "
    "Because adjacent windows of one trial share slowly varying physiological state, recording conditions and noise, and all windows of a trial carry the same label, "
    "a classifier can then score well by recognising the trial rather than the emotion. This is a form of data leakage, and the estimated accuracy no longer describes performance on unseen emotional episodes. "
    "How often this occurs in the published literature is difficult to establish, because many papers do not state the partition unit; "
    "for example, Zhang, Y. et al. (2020) report up to 94% on DEAP with 10-fold cross-validation without specifying whether folds were formed by trial or by segment. "
    "Appendix Table A1 summarises the evaluation protocols of the studies cited in this paper."])
R.rewrite_para(chk(14, 'To bridge'), [
    "In this paper we measure, rather than assume, the effect of window-level splitting on DEAP. "
    "We compare trial-level and window-level 5-fold cross-validation for two classifiers trained on the same Differential Entropy (DE) features (Duan et al., 2013): "
    "a CNN-LSTM that processes the sequence of 2-s windows of a trial, and a Random Forest. "
    "We hold the classification unit fixed in a control analysis, vary the window overlap, and report all accuracies against each subject's majority-class baseline together with balanced accuracy, macro-F1 and AUROC. "
    "We use the CNN-LSTM as a representative sequence model, not as a new architecture."])
R.rewrite_para(chk(15, 'This study makes'), [
    "This study makes three contributions. 1. Quantification of window-level leakage: using identical features and folds, we estimate how much window-level splitting inflates accuracy, "
    "separate the effect of the split from the effect of the classification unit, and show how the inflation grows with window overlap. "
    "2. A documented trial-level baseline: we report subject-dependent trial-level results for a CNN-LSTM and a Random Forest with complete hyperparameters, majority-class baselines, balanced metrics and code, "
    "as a reference point for this specific pipeline (DEAP, 14 channels, DE features); it is not an upper bound for EEG emotion recognition. "
    "3. A quantitative check of a topographic interpretation: we test the frontal-asymmetry pattern suggested by Gamma-band DE maps with a per-subject asymmetry index and find no significant valence effect, "
    "which illustrates why topographic visualisations need statistical testing."])

# ================================================================== LITERATURE REVIEW
p21 = chk(21, 'A significant breakthrough')
R.replace_sub(p21, 'Their seminal work on the SEED dataset proved that DE consistently outperformed PSD across various classical classifiers.',
              'Their work on the SEED dataset reported that DE outperformed PSD across several classical classifiers.')
R.replace_sub(p21, 'They were fundamentally incapable of modeling the deep hierarchical interactions and the sequential evolutionary nature of human emotion, establishing an empirical ceiling that rarely exceeded 70% in rigorous, trial-level subject-dependent evaluations.',
              'They also do not model the temporal evolution of the response within a trial.')
R.replace_sub(p21, 'In our proposed framework, we adopt the mathematically superior DE features identified by Zheng and Lu, but completely discard the shallow classical classifiers in favor of deep, non-linear sequence modeling to push past this historical empirical ceiling.',
              'In this study we use DE features and compare a deep sequence model with a classical Random Forest under identical partitions.')
p25 = chk(25, 'While CNNs excel')
R.replace_sub(p25, 'our work distinguishes itself by feeding mathematically rigorous Differential Entropy topologies into the hybrid network, rather than raw signal arrays, thereby marrying the best of advanced physiological feature engineering with deep spatio-temporal representation learning.',
              'our work uses DE features as the input to such a hybrid network and uses it as a test case for evaluation protocols rather than as a new architecture.')
R.replace_sub(chk(26, '2.3'), 'The “Data Leakage” Crisis in Evaluation Protocols', 'Data Leakage in Evaluation Protocols')
R.replace_para(chk(27, 'As deep learning'), [
    "As deep learning models grew more complex (e.g., attention mechanisms, Graph Convolutional Networks (Song et al., 2018) and Transformers (Wang, Z. et al., 2022)), "
    "reported subject-dependent accuracies on DEAP and SEED rose to 85–95% or higher (e.g., Tripathi et al., 2017; Song et al., 2018; Zhang, Y. et al., 2020). "
    "These figures come from different datasets, label definitions and evaluation protocols and are not directly comparable with each other or with the DEAP results reported here. "
    "In particular, they are not by themselves evidence of window-level leakage: Tripathi et al. (2017) held out one participant at a time, and Song et al. (2018) trained and tested on separate SEED trials. "
    "Methodological studies have nevertheless shown that the choice of data partition can change estimated performance substantially, both in affective EEG (Moreno-Alcayde et al., 2023) "
    "and in clinical EEG (Brookshire et al., 2024; Lee et al., 2023)."])
p28 = chk(28, 'The crux')
R.replace_sub(p28, 'However, a vast majority of the literature employs Window-Level Splitting: all windows from all trials are aggregated, randomized, and then split into training and testing sets (Hosseini et al., 2020).',
              'Under Window-Level Splitting, by contrast, all windows from all trials are pooled, shuffled and then split into training and test sets. '
              'More broadly, analysis choices that let information from the test data influence training or model selection are a known source of optimistic estimates in brain-data classification (Hosseini et al., 2020).')
p29 = chk(29, 'Because brain states')
R.replace_sub(p29, 'The high-capacity deep neural network simply acts as a Nearest-Neighbor lookup table, memorizing the idiosyncratic noise of the trial rather than learning the generalized neurophysiological patterns of the target emotion. This constitutes a severe form of data leakage.',
              'A classifier can then exploit trial-specific features of the signal rather than emotion-related patterns that generalise to new trials; this is a form of data leakage.')
R.replace_para(chk(30, 'Recent audits'), [
    "Brookshire et al. (2024) showed, for translational and clinical EEG, that segment-level partitioning, which lets segments from the same participant enter both training and test sets, "
    "substantially inflates performance relative to subject-level partitioning. That is subject-identity leakage; it is related to, but not the same as, the within-subject, within-trial leakage studied here, "
    "and their results do not establish how much DEAP accuracies change under trial-level evaluation. Lee et al. (2023) reported a similar overestimation for a deep-learning diagnostic system for psychiatric disorders. "
    "Moreno-Alcayde et al. (2023) are the closest prior work: they showed on affective EEG that the partitioning strategy can change classification performance substantially. "
    "The present study adds (i) a paired, per-subject estimate of the inflation for two model families with identical folds, (ii) a control in which the classification unit is held fixed, "
    "(iii) an overlap scan that relates the size of the inflation to window overlap, and (iv) reporting against majority-class baselines and balanced metrics."])
R.replace_para(chk(31, 'Unlike the overwhelming'), [
    "Our aim is therefore not a new state-of-the-art accuracy but a transparent comparison of evaluation protocols on one dataset and one feature pipeline, "
    "with all choices and code documented so that the measurements can be reproduced and extended."])

# ================================================================== METHODS
R.replace_sub(chk(33, 'To robustly'), 'we propose a mathematically rigorous evaluation pipeline', 'we use the evaluation pipeline described below')
p35 = chk(35, 'The empirical foundation')
R.replace_sub(p35, 'on a continuous discrete scale from 1 to 9', 'on a continuous scale from 1 to 9')
labs = [r for r in __import__('csv').DictReader(open(RES / 'per_subject_labels.csv'))]
hv = [float(x['valence_ge5_high_prop']) * 100 for x in labs]; ha = [float(x['arousal_ge5_high_prop']) * 100 for x in labs]
mv = [float(x['valence_ge5_majority_acc']) for x in labs]
R.insert_after(p35, body_style, [
    f"Ratings ≥ 5 were assigned to the High class and ratings < 5 to the Low class. DEAP's own analyses use ratings > 5 as High; "
    f"the two rules differ only for trials rated exactly 5, which occurred in {rf['exactly5_trials']['valence']} of 1,280 trials for Valence and {rf['exactly5_trials']['arousal']} for Arousal. "
    f"Across subjects, the proportion of High trials ranged from {min(hv):.1f}% to {max(hv):.1f}% for Valence (mean {np.mean(hv):.1f}%) and from {min(ha):.1f}% to {max(ha):.1f}% for Arousal (mean {np.mean(ha):.1f}%), "
    f"so the per-subject majority-class accuracy ranged from {min(mv):.1f}% to {max(mv):.1f}% for Valence (mean {rf['majority']['valence']['ge5']['mean']:.2f}%) "
    f"and averaged {rf['majority']['arousal']['ge5']['mean']:.2f}% for Arousal. Per-subject class proportions are given in the Supplementary file per_subject_labels.csv. "
    f"Results with the > 5 rule are reported as a sensitivity analysis (Section 4.2.2)."])

R.rewrite_para(chk(36, 'The raw EEG signals'), [
    "We used the preprocessed release of DEAP (data_preprocessed_matlab). According to the dataset documentation (Koelstra et al., 2012), these data were down-sampled to 128 Hz, "
    "EOG artifacts were removed with the method of Schlögl et al. (2007), a 4–45 Hz band-pass filter was applied and the data were re-referenced to the common average. "
    "We applied no further filtering; the band-pass described in earlier versions of this manuscript is the one already applied by the dataset providers, not an additional step. "
    "No ICA, channel rejection or muscle-artifact rejection was performed, which is a limitation for the interpretation of the Gamma band (Section 5.1). "
    "Fourteen channels were used: AF3, F3, F7, FC5, C3, P7 and O1 on the left and AF4, F4, F8, FC6, T8, P8 and O2 on the right (DEAP channel indices 2, 3, 4, 5, 7, 12, 14, 18, 20, 21, 22, 26, 30 and 32). "
    "Thirteen of these form homologous left–right pairs; C3 (left central) and T8 (right temporal) do not. The set corresponds to the Emotiv EPOC montage except that C3 was used where that montage has T7; "
    "we report this as found in the code used for all experiments and treat it as a limitation, and the frontal-asymmetry analysis below uses only the four homologous frontal pairs. "
    "Each preprocessed trial contains 8,064 samples (63 s), of which the first 3 s precede stimulus onset. Each trial was divided into non-overlapping 2-s windows (256 samples, step 256) starting at sample 1, "
    "giving floor(8064/256) = 31 windows that cover 0–62 s: the first window and half of the second contain the pre-stimulus baseline, and the last second of the trial is not used. "
    "No baseline correction was applied. This yields 31 windows per trial and 40 × 31 = 1,240 windows per subject for every protocol in this paper; "
    "the earlier statement of 30 windows per trial was incorrect. The DEAP preprocessed files store trials in video (Experiment_id) order rather than presentation order, which matters for the split in Section 4.5.1."])
R.replace_sub(chk(39, 'Figure 3:'),
              'and our pipeline output after bandpass filtering and baseline removal', 'and the band-limited signal used for feature extraction (no additional filtering or baseline removal was applied)')
p41 = chk(41, 'Following the digital')
R.replace_sub(p41, 'While historical literature relies heavily on Power Spectral Density (PSD), recent neuro-mathematical evaluations demonstrate that Differential Entropy (DE) offers superior sensitivity for distinguishing complex cognitive states (Zheng & Lu, 2015).',
              'Differential Entropy (DE) has been reported to perform well for EEG emotion classification (Duan et al., 2013; Zheng & Lu, 2015).')
R.replace_sub(chk(47, 'Equation (3)'), 'This logarithmic scaling acts as a powerful non-linear normalization mechanism, effectively compressing extreme outliers and amplifying subtle fluctuations in lower-energy high-frequency bands (such as Gamma), which are heavily implicated in affective cognitive processing.',
              'Because DE of a Gaussian band-limited signal is a monotonic (logarithmic) function of band power, it carries the same information as log band power; '
              'its practical benefit is the variance-stabilising effect of the logarithm, not additional information relative to PSD.')
R.rewrite_para(chk(48, 'For each 2-second'), [
    "For each 2-s window and each of the 14 channels we computed a 256-point FFT (rectangular window, 0.5 Hz resolution), summed the periodogram |X(f)|²/256 over the bins of four bands—Theta (4–8 Hz), "
    "Alpha (8–13 Hz), Beta (13–30 Hz) and Gamma (30–45 Hz), band edges inclusive—and applied Equation (3), DE = ½·ln(2πe·P_band). "
    "This gives 14 × 4 = 56 DE values per window and a sequence of T = 31 such vectors per trial."])
R.replace_sub(chk(49, '3.3'), 'Anatomical Spatial Topological Mapping', 'Input Representation for the CNN')
R.rewrite_para(chk(50, 'A critical flaw'), [
    "The CNN receives, for each window, the 56 DE values arranged as a 14 × 4 matrix that is convolved with 3 × 2 kernels. "
    "This matrix is not a scalp topography: no electrode coordinates were used, no 10-20 grid was constructed and no zero-padding to a spatial grid was applied, "
    "contrary to the description in the original submission."])
R.rewrite_para(chk(51, 'To construct'), [
    "During this revision we also found that the 56 values stored in the feature file are ordered band-major (all 14 channels of Theta, then Alpha, Beta and Gamma), "
    "whereas the network reshaped them row-major into 14 × 4. The rows of the matrix seen by the CNN therefore do not correspond to individual channels. "
    "All CNN results labelled 'as originally implemented' use this ordering. To check whether this affected the conclusions, we repeated every CNN analysis with the corrected ordering "
    "(rows = channels, columns = bands); both sets of results are reported (Table 1 and Section 4.2.1)."])
R.replace_sub(chk(57, 'where'), 'Following the final pooling layer, the multi-dimensional tensor is flattened into a 1D spatial embedding vector',
              'The output of the second convolution (32 × 7 × 3) is flattened into a 672-dimensional embedding vector')
R.replace_sub(chk(69, 'The terminal hidden'), 'optimized via the Adam algorithm with adaptive learning rates.',
              'optimized with Adam for a fixed budget of 50 epochs, without a validation set, early stopping or learning-rate schedule (Section 4.1, Table A2).')
p71 = chk(71, '1.  Input')
for old, new in [('BandpassFilter(D[s,i], [4,45] Hz)', 'D_pre[s,i]        // DEAP preprocessed data (already 4-45 Hz)'),
                 ('ComputeSTFT(W[i,t])', 'FFT256(W[i,t])'),
                 ('SpatialMap(DE[i,t])   // Map to 2D 10-20 topology', 'Reshape(DE[i,t], 14x4)   // channel x band matrix (Sec. 3.3)'),
                 ('while epoch < MaxEpochs do', 'for epoch = 1 to 50 do   // fixed budget, no validation set'),
                 ('end while', 'end for')]:
    try: R.replace_sub(p71, old, new)
    except ValueError as e: print('WARN algorithm:', e)
R.replace_sub(chk(72, 'Figure 4'), '(B) topological spatial encoding onto a 10-20 electrode grid', '(B) arrangement of the 56 DE values of each window as a 14 × 4 matrix (the scalp-grid illustration is schematic; see Section 3.3)')
R.replace_sub(chk(73, '3.5'), 'Rigorous Evaluation Protocol: Eradicating Data Leakage', 'Evaluation Protocols')
R.rewrite_para(chk(74, 'The most critical'), [
    "The central variable in this study is how the training and test data are partitioned. Windows from the same trial are strongly correlated and share one label. "
    "If some windows of a trial are used for training and others for testing, the classifier can exploit trial-specific information, and test accuracy no longer measures generalisation to new emotional episodes."])
R.replace_sub(chk(76, 'The Leaked Benchmark'), 'This method guarantees that windows from the same trial exist in both sets simultaneously.',
              'With 31 windows per trial, windows of essentially every trial appear in both sets.')
R.rewrite_para(chk(78, 'By evaluating'), [
    "Three protocols were evaluated for every subject separately (subject-dependent), all with 5-fold cross-validation and the same random seed (42): "
    "(a) trial-level: the unit of classification is the trial and the folds are formed from the 40 trials (32 for training, 8 for testing); "
    "(b) window unit with trial-level split: each window is classified separately, but the folds are the same trial folds as in (a), so no trial contributes to both sets; "
    "(c) window unit with window-level split: the 1,240 windows are shuffled and split into folds regardless of trial (the leaky protocol). "
    "The CNN-LSTM consumes a whole trial and was used in (a); because a sequence model cannot classify a single shuffled window, (b) and (c) used a static CNN with the same two convolutional layers "
    "followed by fully connected layers instead of the LSTM. The Random Forest used the 31 × 56 = 1,736 concatenated DE values of a trial in (a) and the 56 values of a window in (b) and (c). "
    "Comparing (c) with (b) therefore isolates the effect of the split with the classification unit held fixed, whereas (c) versus (a) reproduces the comparison in the original submission, "
    "in which the unit and, for the CNN, the architecture change together with the split. In (b) we also report a trial-level decision obtained by averaging the window probabilities of each test trial. "
    "Folds were not stratified by class (scikit-learn KFold with shuffling); stratified folds are reported as a sensitivity analysis. "
    "Model comparison is reported with threshold-free AUROC alongside balanced accuracy and macro-F1 at the default 0.5 threshold (cf. Karimi et al., 2025). "
    "For every subject, metrics were computed from the pooled out-of-fold predictions (each trial or window predicted once), and the tables report the mean and standard deviation across the 32 subjects (n − 1 denominator). "
    "With five equal folds of eight trials, the per-subject accuracy equals the mean of the five fold accuracies."])

# ================================================================== RESULTS 4.1
R.rewrite_para(chk(82, 'All experiments'), [
    "Implementation and hyperparameters are listed in Appendix Table A2. The CNN-LSTM (two convolutional layers with 16 and 32 kernels of size 3 × 2, one 2 × 2 max-pooling layer, "
    f"a two-layer LSTM with 64 hidden units and dropout 0.3 between layers, dropout 0.5 before the output layer; {ovp['params_cnnlstm']:,} parameters) was trained with Adam "
    "(learning rate 5 × 10⁻⁴, weight decay 10⁻⁴, batch size 8) for a fixed 50 epochs. There was no validation set, no early stopping, no learning-rate schedule and no batch normalisation; "
    "the earlier statements that the learning rate was reduced on a plateau of the validation loss and that batch normalisation was used were incorrect and have been removed. "
    "Consequently no test-fold information was used for model selection. Features were z-scored with the mean and standard deviation of the training fold only. "
    "The static CNN used for window-level protocols (Section 3.5) had the same convolutional layers followed by dropout 0.5 and fully connected layers 672 → 64 → 2 "
    f"({ovp['params_staticcnn']:,} parameters), trained with Adam (learning rate 10⁻³, batch size 64) for 50 epochs. "
    "Architectures and hyperparameters were fixed before this revision and were not tuned on DEAP within this study; they were, however, chosen by the authors with general knowledge of the dataset, "
    "so the trial-level results are best described as trial-level cross-validation rather than as leakage-free in every sense. "
    "All results in this revision were recomputed from the cached DE features on a CPU workstation (PyTorch 2.x, scikit-learn) with one script; "
    "CNN results are the mean of three random seeds (42, 43, 44)."])
R.replace_para(chk(83, 'For the traditional'), [
    "The Random Forest (scikit-learn RandomForestClassifier) used 100 trees, the Gini criterion and default settings otherwise (unlimited depth, min_samples_split = 2, √p features per split), with random_state = 42. "
    "The earlier statement of 500 trees was incorrect. Hyperparameters were not tuned."])
R.replace_para(chk(84, 'In strict adherence'), [
    "Classification was subject-dependent: each subject's 40 trials were split into five folds of eight trials, so every model was trained on 32 trials and tested on eight unseen trials of the same subject. "
    "Training and test trials come from the same participant; the cross-subject case is evaluated separately in Section 4.5.1. "
    "Besides accuracy we report balanced accuracy, macro-F1 and AUROC, each subject's majority-class accuracy, and paired statistics across subjects "
    "(paired t-test with Holm correction, mean difference with 95% confidence interval, and the paired effect size d_z) (Holm, 1979)."])

# ================================================================== 4.2
R.replace_sub(chk(85, '4.2'), 'The Rigorous Empirical Baseline: Trial-Level Evaluation', 'Trial-Level Evaluation')
R.replace_sub(chk(86, '4.2.1'), 'Deep Learning Performance (CNN-LSTM)', 'CNN-LSTM')
R.replace_para(chk(87, 'We first evaluated'), [
    f"Under trial-level cross-validation the CNN-LSTM reached {ms(CN('valence','trial'))}% accuracy for Valence (balanced accuracy {ms(CN('valence','trial'),'bal_acc')}%; AUROC {m(CN('valence','trial'),'auroc')}%) "
    f"and {ms(CN('arousal','trial'))}% for Arousal (balanced accuracy {ms(CN('arousal','trial'),'bal_acc')}%; AUROC {m(CN('arousal','trial'),'auroc')}%) (Figure 5, Table 1). "
    f"Seed-to-seed variation of the group mean was small (Valence {min(CNs('valence','trial')):.2f}–{max(CNs('valence','trial')):.2f}%, Arousal {min(CNs('arousal','trial')):.2f}–{max(CNs('arousal','trial')):.2f}%). "
    f"With the corrected channel × band ordering (Section 3.3) the results were {m(CN('valence','trial','corrected'))}% and {m(CN('arousal','trial','corrected'))}% "
    f"(paired difference to the original ordering: Valence {grid_fix['valence']['diff']:+.2f} points, {pfmt(grid_fix['valence']['p'])}; Arousal {grid_fix['arousal']['diff']:+.2f} points, {pfmt(grid_fix['arousal']['p'])}), "
    "so the ordering error did not change the conclusions. The originally submitted values (61.48% and 65.00%) came from single runs whose logs were only partly recoverable; "
    "they are replaced by the reproducible values above."])
R.replace_sub(chk(88, 'Figure 5'), R.text(P[88]).strip()[R.text(P[88]).strip().index('Figure 5'):],
    f"Figure 5: Per-subject accuracy of the CNN-LSTM for Valence under trial-level 5-fold cross-validation (mean of three seeds). "
    f"Black ticks mark each subject's majority-class accuracy; dashed and dotted lines show the mean accuracy ({m(CN('valence','trial'))}%) and the mean majority-class baseline ({rf['majority']['valence']['ge5']['mean']:.2f}%).")
R.replace_para(chk(89, 'The distribution'), [
    "Accuracy varied widely across subjects, and for a number of subjects it was below the majority-class accuracy, i.e. below what a classifier that always predicts the more frequent class would obtain. "
    "Mean accuracy must therefore be read against the mean majority-class baseline rather than against 50%; balanced accuracy and AUROC, which are not affected by class imbalance, are reported for this reason."])
R.replace_sub(chk(90, '4.2.2'), 'Traditional Machine Learning Verification (Random Forest)', 'Random Forest')
R.replace_para(chk(91, 'To definitively'), [
    "The Random Forest was trained on the same DE features. Because it does not model sequences, the 31 window vectors of a trial were concatenated into one 1,736-dimensional vector per trial."])
R.replace_sub(chk(92, 'Figure 6'), 'The global mean accuracy is 64.38%.',
              f"Black ticks mark each subject's majority-class accuracy. Mean accuracy {m(RFk('valence','trial'))}%; mean majority-class baseline {rf['majority']['valence']['ge5']['mean']:.2f}%.")
pv, pa = RFk('valence', 'trial'), RFk('arousal', 'trial'); tv, ta = rf['rf_valence']['trial_vs_majority'], rf['rf_arousal']['trial_vs_majority']
perm_txt = ''
if perm:
    perm_txt = (f" A label-permutation test (labels permuted within subject, {perm['valence']['n_perm']} permutations) gave p = {perm['valence']['p_bal']:.3f} for the Valence balanced accuracy "
                f"and p = {perm['arousal']['p_bal']:.3f} for Arousal (null means {perm['valence']['null_bal_mean']:.1f}% and {perm['arousal']['null_bal_mean']:.1f}%).")
R.rewrite_para(chk(93, 'As illustrated'), [
    f"The Random Forest reached {ms(pv)}% accuracy for Valence (balanced accuracy {ms(pv,'bal_acc')}%, macro-F1 {m(pv,'macro_f1')}%, AUROC {m(pv,'auroc')}%) "
    f"and {ms(pa)}% for Arousal (balanced accuracy {ms(pa,'bal_acc')}%, macro-F1 {m(pa,'macro_f1')}%, AUROC {m(pa,'auroc')}%) (Figure 6, Table 1). "
    "These values replace the 64.38% (Valence) and 63.83% (Arousal) reported in the original submission. Re-running the original script reproduced 61.33% for Valence exactly. "
    "The value 64.38% equals the mean majority-class accuracy for Arousal, and we believe it was transcribed in error. "
    f"Relative to the majority-class baseline, Valence accuracy was {tv['mean_diff']:+.2f} points higher (95% CI {tv['ci95'][0]:.2f} to {tv['ci95'][1]:.2f}; {pfmt(tv['p'])}) "
    f"and Arousal accuracy {ta['mean_diff']:+.2f} points different (95% CI {ta['ci95'][0]:.2f} to {ta['ci95'][1]:.2f}; {pfmt(ta['p'])}). "
    f"Balanced accuracy was above 50% for Valence (t(31) = {rf['rf_valence']['trial_balacc_vs_50']['t']:.2f}, {pfmt(rf['rf_valence']['trial_balacc_vs_50']['p'])}) "
    f"but not for Arousal (t(31) = {rf['rf_arousal']['trial_balacc_vs_50']['t']:.2f}, {pfmt(rf['rf_arousal']['trial_balacc_vs_50']['p'])}).{perm_txt} "
    "Thus, under trial-level evaluation the pipeline carries a small but consistent amount of Valence information. For Arousal the evidence is weak and inconsistent: accuracy did not exceed the majority-class baseline and balanced accuracy did not differ from 50% in the t-test, whereas the permutation test was marginal; the permutation null lies below 50% because small, imbalanced test folds bias balanced accuracy downward. "
    f"Sensitivity analyses gave similar results with stratified folds (Valence {m(RFk('valence','trial_stratified'))}%, balanced {m(RFk('valence','trial_stratified'),'bal_acc')}%; "
    f"Arousal {m(RFk('arousal','trial_stratified'))}%, balanced {m(RFk('arousal','trial_stratified'),'bal_acc')}%) and with the > 5 binarisation rule "
    f"(Valence {m(RFk('valence','trial_threshold_gt5'))}%, balanced {m(RFk('valence','trial_threshold_gt5'),'bal_acc')}%; Arousal {m(RFk('arousal','trial_threshold_gt5'))}%, balanced {m(RFk('arousal','trial_threshold_gt5'),'bal_acc')}%). "
    f"The CNN-LSTM and the Random Forest did not differ significantly (Valence {cnn_vs_rf['valence']['diff']:+.2f} points, {pfmt(cnn_vs_rf['valence']['p'])}; "
    f"Arousal {cnn_vs_rf['arousal']['diff']:+.2f} points, {pfmt(cnn_vs_rf['arousal']['p'])}). Because the two models receive different representations of the trial (an ordered sequence versus a concatenated vector), "
    "this similarity does not by itself show that data scarcity rather than architecture limits performance; it shows only that, for these features and this protocol, the added complexity of the CNN-LSTM did not improve accuracy."])

# ================================================================== 4.3
R.replace_sub(chk(94, '4.3'), 'Quantifying the “Data Leakage” Illusion: Window-Level Evaluation', 'Window-Level Evaluation and the Leakage Effect')
R.replace_para(chk(95, 'Having established'), [
    "We then evaluated protocols (b) and (c) of Section 3.5 on the same 1,240 non-overlapping windows per subject and with the same features. "
    "Under (c), 992 windows per fold were used for training and 248 for testing; under (b), the 248 test windows were the 31 windows of the eight held-out trials. "
    "For the CNN, the window-level protocols used the static CNN described in Section 3.5 because the CNN-LSTM requires a complete trial as input."])
R.replace_sub(chk(96, 'Figure 7'), R.text(P[96]).strip()[R.text(P[96]).strip().index('Figure 7'):],
    "Figure 7: Accuracy under the three partition protocols for the Random Forest and the CNN (CNN-LSTM in the trial-level protocol, static CNN in the window protocols), "
    "for Valence (left) and Arousal (right). Hatched bars: window-level split (leaky). Dotted line: mean majority-class baseline. Error bars: SD across 32 subjects.")
def fd(i): r_ = fam[i][2]; return f"{r_['diff']:+.2f} points (95% CI {r_['lo']:.2f} to {r_['hi']:.2f}; d_z = {r_['dz']:.2f})"
R.replace_para(chk(97, 'The results of this'), [
    f"For the Random Forest, the window-level split (c) gave {ms(RFk('valence','window_leaky'))}% for Valence and {ms(RFk('arousal','window_leaky'))}% for Arousal, "
    f"compared with {m(pv)}% and {m(pa)}% under trial-level evaluation (a): an increase of {fd(0)} and {fd(4)}, respectively. "
    f"With the classification unit held fixed (window unit, trial split (b)), accuracy was {ms(RFk('valence','window_trialsplit'))}% and {ms(RFk('arousal','window_trialsplit'))}%, "
    f"so the window-level split alone raised accuracy by {fam[1][2]['diff']:.2f} and {fam[5][2]['diff']:.2f} points. "
    f"For the CNN, the static CNN under the window-level split reached {ms(CN('valence','window_leaky'))}% (Valence) and {ms(CN('arousal','window_leaky'))}% (Arousal), "
    f"versus {ms(CN('valence','window_trialsplit'))}% and {ms(CN('arousal','window_trialsplit'))}% for the same static CNN under the trial split, "
    f"an increase of {fam[3][2]['diff']:.2f} and {fam[7][2]['diff']:.2f} points. "
    f"All eight comparisons in Table 2 were significant after Holm correction ({'all corrected p < 0.001' if max_ph < 0.001 else f'largest corrected p = {max_ph:.3f}'}). "
    "The across-subject standard deviation was smaller under the window-level split than under trial-level evaluation, partly because each subject's accuracy is then averaged over 1,240 test decisions instead of 40. "
    "These inflations are larger than the 3–7 points reported in the original submission, whose window-level values could not be reproduced from the archived code."])
R.replace_para(chk(98, 'This drastic'), [
    "The comparison between protocols (b) and (c) shows that the increase is caused by the split itself and not by the change of classification unit. "
    "It does not show what specifically the classifiers exploit under the window-level split; plausible candidates are slowly varying trial-specific signal properties and the fact that all windows of a trial share one label. "
    "The size of the effect depends on the overlap between windows (Section 4.5.3). "
    "We therefore recommend that studies on DEAP state the partition unit and the overlap explicitly, and that accuracies obtained with window-level splits are not compared with trial-level results."])

# ================================================================== 4.4 + Table 1
R.replace_para(chk(100, 'To facilitate'), [
    "Table 1 summarises all subject-dependent results: accuracy, balanced accuracy, macro-F1 and AUROC (mean ± SD across the 32 subjects), together with the mean majority-class accuracy."])
maj_v_s = f"{rf['majority']['valence']['ge5']['mean']:.2f}"; maj_a_s = f"{rf['majority']['arousal']['ge5']['mean']:.2f}"
rows = [['Model', 'Dim.', 'Protocol (unit / split)', 'Accuracy (%)', 'Balanced acc. (%)', 'Macro-F1 (%)', 'AUROC (%)', 'Majority (%)']]
for dim, dn, mj in [('valence', 'V', maj_v_s), ('arousal', 'A', maj_a_s)]:
    for name, d, proto in [('Random Forest', RFk(dim, 'trial'), 'trial / trial'), ('Random Forest', RFk(dim, 'window_trialsplit'), 'window / trial'),
                           ('Random Forest', RFk(dim, 'window_leaky'), 'window / window'),
                           ('CNN-LSTM', CN(dim, 'trial'), 'trial / trial'), ('CNN-LSTM (corrected order)', CN(dim, 'trial', 'corrected'), 'trial / trial'),
                           ('Static CNN', CN(dim, 'window_trialsplit'), 'window / trial'), ('Static CNN', CN(dim, 'window_leaky'), 'window / window')]:
        rows.append([name, dn, proto, ms(d), ms(d, 'bal_acc'), ms(d, 'macro_f1'), ms(d, 'auroc'), mj])
t1 = R.make_table(T[0], rows, weights=[2.2, .6, 1.6, 1.5, 1.5, 1.5, 1.5, 1.1]); T[0].addnext(t1); R.delete_table(T[0])
R.replace_para(chk(101, 'Table 1.'), [
    "Table 1. Subject-dependent 5-fold cross-validation on DEAP (n = 32 subjects; mean ± SD across subjects). Protocol: unit of classification / unit used to form the folds; "
    "'window / window' is the leaky window-level split. Majority: mean per-subject majority-class accuracy. CNN results are means over three seeds; "
    "'corrected order' uses the channel × band arrangement described in Section 3.3."])
R.replace_para(chk(102, 'As evidenced'), [
    f"Under trial-level evaluation, accuracies were {min(m(pv), m(CN('valence','trial')))}–{max(m(pv), m(CN('valence','trial')))}% for Valence and "
    f"{min(m(pa), m(CN('arousal','trial')))}–{max(m(pa), m(CN('arousal','trial')))}% for Arousal, close to the majority-class baselines of {maj_v_s}% and {maj_a_s}%; "
    f"balanced accuracies were modestly above 50% for Valence and close to 50% for Arousal ({m(pa,'bal_acc')}% for the Random Forest, {m(CN('arousal','trial'),'bal_acc')}% for the CNN-LSTM). " "Window-level splitting raised every metric, including balanced accuracy and AUROC. "
    "These numbers describe this pipeline (DEAP, 14 channels, DE features, two classifiers without tuning); they are a baseline, not an upper bound for EEG emotion recognition."])

# ================================================================== 4.5
R.replace_sub(chk(103, '4.5'), 'Supplementary Analyses: Generalization Boundary and Band Contribution', 'Supplementary Analyses')
R.replace_sub(chk(104, '4.5.1'), 'Temporal Non-Stationarity and Cross-Subject Generalization', 'Video-Order Split and Cross-Subject Evaluation')
ch_v, ch_a = rf['chronological_valence'], rf['chronological_arousal']
R.replace_para(chk(105, 'To go beyond'), [
    "Two further partitions were evaluated with the Random Forest for both dimensions. First, for every subject the model was trained on the first 30 trials and tested on the last 10 trials as stored in the file. "
    "This split was described in the original submission as chronological, but the DEAP preprocessed files store trials in video (Experiment_id) order, not in presentation order, "
    "so it is a split by stimulus rather than by time. Second, a leave-one-subject-out evaluation trained on 31 subjects and tested on the held-out subject (subject-independent). Results are shown in Figure 8 and Table 3."])
R.replace_sub(chk(106, 'Figure 9'), R.text(P[106]).strip()[R.text(P[106]).strip().index('Figure 9'):],
    "Figure 8: Random Forest accuracy (bars, mean ± SD) and balanced accuracy (diamonds) for subject-dependent trial-level cross-validation, the last-10-videos split, and leave-one-subject-out evaluation. Dotted line: 50%.")
R.replace_para(chk(107, 'The results are striking'), [
    f"In the last-10-videos split, Valence accuracy was {ms(ch_v['summary'])}% but balanced accuracy was {ms(ch_v['summary'],'bal_acc')}%. "
    f"The low accuracy reflects a label shift built into the stimulus order: {lo['valence_high_first30']*100:.0f}% of the first 30 videos were rated high in Valence but only {lo['valence_high_last10']*100:.0f}% of the last 10, "
    f"so a model that learns the training prior is systematically wrong on the test videos (a rule that always predicts the training majority class scored {m(ch_v['summary'],'train_majority_rule_acc')}%), "
    f"and {ch_v['n_subjects_single_class_test']} subjects had only one class among their last 10 trials. For Arousal the shift was in the opposite direction "
    f"({lo['arousal_high_first30']*100:.0f}% versus {lo['arousal_high_last10']*100:.0f}% high), with accuracy {ms(ch_a['summary'])}% and balanced accuracy {ms(ch_a['summary'],'bal_acc')}%. "
    "This analysis therefore cannot be interpreted as evidence of temporal non-stationarity of the EEG, and the earlier interpretation has been withdrawn; "
    "a chronological analysis would require reordering the trials by presentation order using the DEAP metadata. "
    f"Leave-one-subject-out accuracy was {ms(rf['loso_valence'])}% for Valence (balanced {ms(rf['loso_valence'],'bal_acc')}%; mean majority baseline of the held-out subject {m(rf['loso_valence'],'majority_acc')}%) "
    f"and {ms(rf['loso_arousal'])}% for Arousal (balanced {ms(rf['loso_arousal'],'bal_acc')}%; baseline {m(rf['loso_arousal'],'majority_acc')}%), "
    "i.e. close to chance for this feature set and classifier. These results are specific to the Random Forest on these features and do not establish a general limit for cross-subject emotion recognition. "
    "The values 48.12% and 53.28% in the original submission could not be reproduced and have been replaced."])
R.replace_para(chk(109, 'To determine'), [
    "For the band analysis, a Random Forest was trained on the DE values of a single band (14 channels × 31 windows = 434 features per trial) under trial-level cross-validation, for both dimensions. "
    "Bands were compared with paired t-tests on balanced accuracy (Gamma versus each other band, Holm correction)."])
R.replace_sub(chk(110, 'Figure 10'), R.text(P[110]).strip()[R.text(P[110]).strip().index('Figure 10'):],
    "Figure 10: Balanced accuracy of single-band Random Forests (trial-level cross-validation; mean ± SD across 32 subjects) for Valence and Arousal, with the all-band model for comparison. Dotted line: 50%.")
A = rf['ablation_valence']; Aa = rf['ablation_arousal']
R.replace_para(chk(111, 'The ablation reveals'), [
    f"For Valence, balanced accuracy was {m(A['theta'],'bal_acc')}% (Theta), {m(A['alpha'],'bal_acc')}% (Alpha), {m(A['beta'],'bal_acc')}% (Beta) and {m(A['gamma'],'bal_acc')}% (Gamma) "
    f"(accuracy {m(A['theta'])}, {m(A['alpha'])}, {m(A['beta'])} and {m(A['gamma'])}%). Gamma exceeded Theta (Holm-corrected p = {abl_t['valence']['theta']['p_holm']:.3f}) "
    f"but not Alpha (p = {abl_t['valence']['alpha']['p_holm']:.3f}) or Beta (p = {abl_t['valence']['beta']['p_holm']:.3f}). "
    f"For Arousal no band exceeded chance (balanced accuracy {min(Aa[b]['bal_acc']['mean'] for b in Aa):.1f}–{max(Aa[b]['bal_acc']['mean'] for b in Aa):.1f}%) and no band differences were significant. "
    "The ordering is therefore only partly supported statistically, and, because 30–45 Hz scalp activity is susceptible to muscle artifact (Goncharova et al., 2003) and no muscle-artifact rejection was performed, "
    "we do not interpret the Gamma result physiologically. The band values in the original submission (64.45%, 62.19%, 58.59% and 56.41%) could not be reproduced and have been replaced."])

# new 4.5.3 overlap scan (heading + text) inserted before 4.6
h453 = R.insert_after(P[111], h3_style, ['4.5.3 Effect of Window Overlap'], rPr=R.first_rPr(h3_style))
R.insert_after(h453, first_style, [
    "To relate the size of the inflation to window overlap, the 2-s windows were re-extracted from the preprocessed data with steps of 256, 128, 64 and 26 samples (0%, 50%, 75% and about 90% overlap), "
    "and a Random Forest was evaluated with the window-level split for Valence (same features, classifier settings and seed). "
    f"Mean accuracy rose monotonically: {ov['0']['mean']:.2f} ± {ov['0']['sd']:.2f}% (0%), {ov['50']['mean']:.2f} ± {ov['50']['sd']:.2f}% (50%), "
    f"{ov['75']['mean']:.2f} ± {ov['75']['sd']:.2f}% (75%) and {ov['90']['mean']:.2f} ± {ov['90']['sd']:.2f}% (90%) (Table 5). "
    f"The 0% value agrees with the window-level result obtained from the cached features ({m(RFk('valence','window_leaky'))}%), which were extracted independently. "
    f"Under trial-level evaluation the same classifier reached {m(pv)}%. Higher overlap also increases the number of training windows, so this scan does not separate the effect of near-duplicate windows from that of a larger training set; "
    "both, however, arise only because test windows have neighbours from the same trial in the training set. "
    "The scan shows that window-level splitting with high overlap can, by itself, produce accuracies above 90% on DEAP with a simple classifier. "
    "It does not show that any particular published study used such a protocol (Appendix Table A1)."])

# ================================================================== 4.6 + Tables 2-5
R.replace_para(chk(113, 'To rigorously'), [
    "Table 2 lists the paired comparisons across the 32 subjects for each model and dimension. Differences are window-level split minus the comparison protocol, so positive values indicate inflation. "
    "Holm correction was applied across the eight comparisons. Tables 3 and 4 report the supplementary partitions and the band analysis, and Table 5 the overlap scan."])
rows = [['Model, dimension', 'Comparison', 'Δ accuracy (pp)', '95% CI', 't(31)', 'p (Holm)', 'd_z']]
for a, b, r_ in fam:
    rows.append([a, b, f"{r_['diff']:+.2f}", f"{r_['lo']:.2f} to {r_['hi']:.2f}", f"{r_['t']:.2f}", ('< 0.001' if r_['p_holm'] < .001 else f"{r_['p_holm']:.3f}"), f"{r_['dz']:.2f}"])
t2 = R.make_table(T[1], rows, weights=[1.4, 3.2, 1.1, 1.4, .8, .9, .7]); T[1].addnext(t2); R.delete_table(T[1])
R.replace_para(chk(114, 'Table 2.'), ["Table 2. Paired comparisons of accuracy between partition protocols (n = 32, df = 31). Δ = window-level split minus comparison protocol; "
                                     "d_z = mean paired difference divided by the SD of the differences; p values Holm-corrected across the eight comparisons."])
rows = [['Partition', 'Train / test', 'Valence acc. (%)', 'Valence bal. acc. (%)', 'Arousal acc. (%)', 'Arousal bal. acc. (%)']]
for name, tt, kv, ka in [('Trial-level, subject-dependent', '32 / 8 trials, 5 folds', RFk('valence', 'trial'), RFk('arousal', 'trial')),
                         ('Last-10-videos split', 'first 30 / last 10 trials (video order)', ch_v['summary'], ch_a['summary']),
                         ('Leave-one-subject-out', '31 subjects / 1 subject', rf['loso_valence'], rf['loso_arousal'])]:
    rows.append([name, tt, ms(kv), ms(kv, 'bal_acc'), ms(ka), ms(ka, 'bal_acc')])
t3 = R.make_table(T[2], rows, weights=[2, 2.4, 1.3, 1.3, 1.3, 1.3]); T[2].addnext(t3); R.delete_table(T[2])
R.replace_para(chk(115, 'Table 3.'), ["Table 3. Random Forest under alternative partitions (mean ± SD across 32 subjects). The last-10-videos split is confounded by a label shift between the first 30 and last 10 videos (Section 4.5.1)."])
rows = [['Band', 'Valence acc. (%)', 'Valence bal. acc. (%)', 'Arousal acc. (%)', 'Arousal bal. acc. (%)']]
for b, lab in [('theta', 'Theta (4–8 Hz)'), ('alpha', 'Alpha (8–13 Hz)'), ('beta', 'Beta (13–30 Hz)'), ('gamma', 'Gamma (30–45 Hz)')]:
    rows.append([lab, ms(A[b]), ms(A[b], 'bal_acc'), ms(Aa[b]), ms(Aa[b], 'bal_acc')])
rows.append(['All bands', ms(pv), ms(pv, 'bal_acc'), ms(pa), ms(pa, 'bal_acc')])
t4 = R.make_table(T[3], rows, weights=[1.6, 1.3, 1.3, 1.3, 1.3]); T[3].addnext(t4); R.delete_table(T[3])
p116 = chk(116, 'Table 4.')
R.replace_para(p116, ["Table 4. Single-band Random Forest, trial-level cross-validation (mean ± SD across 32 subjects). "
                      f"Gamma versus Theta, Alpha and Beta (Valence, balanced accuracy, Holm-corrected): p = {abl_t['valence']['theta']['p_holm']:.3f}, {abl_t['valence']['alpha']['p_holm']:.3f} and {abl_t['valence']['beta']['p_holm']:.3f}."])
rows = [['Overlap', 'Step (samples)', 'Windows per trial', 'Window-level accuracy (%)', 'Range across subjects (%)']]
for k, step in [('0', 256), ('50', 128), ('75', 64), ('90', 26)]:
    rows.append([f'{k}%', str(step), str((8064 - 256) // step + 1), f"{ov[k]['mean']:.2f} ± {ov[k]['sd']:.2f}", f"{ov[k]['min']:.1f}–{ov[k]['max']:.1f}"])
t5 = R.make_table(T[3], rows, weights=[1, 1.2, 1.3, 1.8, 1.8]); p116.addnext(t5)
cap5 = R.insert_after(t5, cap_style, ["Table 5. Random Forest, Valence, window-level split (leaky) as a function of window overlap (2-s windows; n = 32). "
                                      f"For comparison, trial-level accuracy of the same classifier: {m(pv)}%. 90% overlap corresponds to a step of 26 samples (89.8%)."])

# ================================================================== 5. DISCUSSION
R.replace_para(chk(118, 'The empirical results'), [
    "Section 4 shows two things for this pipeline: under trial-level evaluation, accuracies on DEAP are close to the majority-class baselines, "
    "and window-level splitting inflates accuracy by roughly ten percentage points at zero overlap and by more at higher overlap. "
    "Below we discuss the topographic analysis, the mechanism of window-level leakage, and the limitations of subject-dependent models on small datasets."])
R.replace_sub(chk(119, '5.1'), 'Physiological Validation: The Frontal EEG Asymmetry Theory', 'Topographic Description and a Quantitative Test of Frontal Asymmetry')
R.replace_para(chk(120, 'A frequent'), [
    "Classifiers trained on EEG may exploit non-neural signal components such as eye movements, muscle activity or line noise that happen to correlate with the labels. "
    "Topographic maps of the input features can describe where differences between conditions occur, but they cannot by themselves show that a classifier relies on cortical activity. "
    "We therefore present the maps as descriptive and test the specific pattern they suggest."])
R.replace_sub(P[122], 'Figure 8:', 'Figure 9:')
R.replace_sub(P[122], 'The top row demonstrates pronounced left-frontal asymmetry during High Valence (positive) states. The bottom row illustrates dense, global parietal and temporal activation during High Arousal states.',
              'Top row: High versus Low Valence; bottom row: High versus Low Arousal. Group means, descriptive only; see Table 6 for the statistical test of frontal asymmetry.')
R.replace_sub(chk(121, 'To achieve this'), 'Figure 8 illustrates', 'Figure 9 illustrates')
p123 = chk(123, 'The topographic visualizations')
R.replace_sub(p123, '(Figure 8, top row)', '(Figure 9, top row)')
R.replace_sub(p123, 'The topographic visualizations yield striking physiological confirmations.', 'The maps are descriptive.')
R.replace_sub(p123, 'which is descriptively consistent with the “Frontal EEG Asymmetry” theory, a cornerstone of affective neuroscience originally pioneered by Davidson (1992).',
              'Frontal EEG asymmetry (Davidson, 1992) is, however, classically defined in the Alpha band, where relatively lower left-frontal Alpha power accompanies approach-related affect; a Gamma-band difference is a different observation and would not by itself corroborate that theory.')
R.replace_para(chk(124, 'Furthermore, in the Arousal'), [
    "In the Arousal dimension (Figure 9, bottom row), Gamma-band DE was higher during high-arousal trials, most visibly at the temporal site T8 and at P7 and P8. "
    "T8 lies over the temporalis muscle, and scalp activity at 30–45 Hz is strongly affected by muscle (EMG) activity, which is broadband and extends below 45 Hz (Goncharova et al., 2003). "
    "Because no muscle-artifact rejection was applied, this pattern is at least as consistent with arousal-related muscle tension as with cortical activity, and we do not interpret it physiologically."])
fa = {r_['comparison'].split(':')[0]: r_ for r_ in asym}
R.replace_para(chk(125, 'Although the topographic'), [
    "To test the frontal pattern quantitatively, we computed for each trial the asymmetry index ln(P_right) − ln(P_left), which for DE equals 2 × (DE_right − DE_left), for the four homologous frontal pairs "
    "(AF4–AF3, F4–F3, F8–F7, FC6–FC5) and their mean, in the Alpha and Gamma bands, using only windows after stimulus onset (windows 3–31). "
    "For each subject the index was averaged over High-Valence and over Low-Valence trials, and the two means were compared with paired t-tests across the 32 subjects (Holm correction across the eight pair-wise tests)."])
R.replace_sub(chk(126, 'To further substantiate'), 'To further substantiate the physiological validity of our features and to provide a comprehensive picture of the neural substrates underlying emotion, we present',
              'For completeness, we present')
R.replace_sub(chk(131, 'Figure 15'), 'illustrating consistent left-frontal asymmetry under High Valence despite inter-subject idiosyncrasy.', 'illustrating the variability of the patterns across subjects.')
comp_a, comp_g = fa['alpha frontal_composite'], fa['gamma frontal_composite']
pairs_min_p = min(r_['p_holm'] for r_ in asym if 'p_holm' in r_)
R.replace_para(chk(133, 'Several important'), [
    "Visually, the band-specific maps (Figures 11–14) suggest a left-frontal Gamma difference for Valence. The quantitative test did not support it (Table 6). "
    f"Neither the composite frontal index (Alpha: difference {comp_a['mean_diff']:+.4f}, 95% CI {comp_a['ci95'][0]:.4f} to {comp_a['ci95'][1]:.4f}, {pfmt(comp_a['p'])}, d_z = {comp_a['d_z']:.2f}; "
    f"Gamma: {comp_g['mean_diff']:+.4f}, 95% CI {comp_g['ci95'][0]:.4f} to {comp_g['ci95'][1]:.4f}, {pfmt(comp_g['p'])}, d_z = {comp_g['d_z']:.2f}) "
    f"nor any individual electrode pair (smallest Holm-corrected p = {pairs_min_p:.2f}) showed a significant difference between High- and Low-Valence trials. "
    "The visual impression from the group-mean maps therefore does not survive a per-subject test, and the topographic figures should be read as descriptive. "
    "This negative result replaces the claim in the original submission that the maps confirm the frontal-asymmetry theory. "
    "The analysis has limited power (32 subjects, about 15–25 trials per class per subject) and does not exclude small effects, and it does not address the EMG concern for the Gamma band, which would require dedicated artifact rejection (e.g., ICA) on the raw data."])
rows = [['Band', 'Index (right − left)', 'Δ High − Low', '95% CI', 't(31)', 'p', 'p (Holm)', 'd_z']]
for r_ in asym:
    b, idx = r_['band'], r_['index'].replace('frontal_composite', 'mean of 4 pairs')
    rows.append([b.capitalize(), idx, f"{r_['mean_diff']:+.4f}", f"{r_['ci95'][0]:.4f} to {r_['ci95'][1]:.4f}", f"{r_['t']:.2f}", f"{r_['p']:.3f}",
                 (f"{r_['p_holm']:.3f}" if 'p_holm' in r_ else '—'), f"{r_['d_z']:.2f}"])
t6 = R.make_table(t2, rows, weights=[.9, 1.5, 1.1, 1.6, .8, .8, .9, .7]); P[133].addnext(t6)
R.insert_after(t6, cap_style, ["Table 6. Frontal asymmetry index (DE_right − DE_left; Alpha and Gamma; stimulus windows only) for High- versus Low-Valence trials, paired across 32 subjects. "
                               "Holm correction across the eight pair-wise tests; the composite index was tested separately."])

R.replace_sub(chk(134, '5.2'), 'Deconstructing the Data Leakage Paradigm', 'Mechanism and Scope of Window-Level Leakage')
R.replace_para(chk(135, 'The most alarming'), [
    f"Changing only the partition from trial-level to window-level raised the accuracy of both classifiers (Table 2), from {m(pv)}% to {m(RFk('valence','window_leaky'))}% for the Random Forest on Valence, "
    f"for example, and the overlap scan shows that the same mechanism reaches {ov['90']['mean']:.0f}% at 90% overlap. "
    "This demonstrates that window-level splitting can produce accuracies comparable to those reported in the literature without any emotion-specific decoding. "
    "It does not show that published high accuracies were obtained this way; that can only be established study by study (Appendix Table A1)."])
R.rewrite_para(chk(137, 'When researchers utilize'), [
    "Under window-level splitting, most test windows have temporally adjacent windows of the same trial in the training set, and with overlap these neighbours share part of their samples. "
    "Because all windows of a trial carry the same label, a classifier that recognises the trial—through slowly varying spectral properties, electrode contact or other recording conditions—also predicts the label, "
    "without learning a representation of emotion that transfers to new trials. Our fixed-unit comparison (Table 2) supports this account: the inflation appears when, and only when, windows of a test trial are also present in training."])
R.rewrite_para(chk(138, 'This explains why'), [
    f"Under the window-level split, the across-subject standard deviation was also smaller (e.g., Random Forest, Valence: {RFk('valence','window_leaky')['acc']['sd']:.2f} versus {pv['acc']['sd']:.2f} points under trial-level evaluation). "
    "Part of this reduction is expected simply because each subject's accuracy is estimated from 1,240 rather than 40 test decisions; it should not be read as greater stability of the model. "
    "Models selected or evaluated with window-level splits may perform worse on new trials, sessions or participants (Brookshire et al., 2024; Lee et al., 2023). "
    "Trial-level (or stricter, e.g. session- or subject-level) partitioning is required when the goal is to estimate performance on new emotional episodes."])
p140 = chk(140, 'While our')
R.replace_sub(p140, 'While our rigorous baseline establishes a scientifically sound benchmark, it is crucial to engage in an honest, critical analysis of its limitations.', 'Several limitations apply.')
R.replace_sub(p140, 'A 61-65% global accuracy, while highly credible under strict non-leaked conditions, remains', 'Accuracies of about 60%, close to the majority-class baselines, remain')
R.replace_sub(chk(141, 'The primary failure'), 'hundreds of thousands of parameters', f"about {ovp['params_cnnlstm']/1000:.0f} thousand parameters")
R.replace_sub(P[141], '—and indeed, of any deep learning architecture operating under trial-level subject-dependent constraints—', '')
R.replace_para(chk(142, 'This is fundamentally'), [
    "This is consistent with the finding that the CNN-LSTM did not outperform the Random Forest. Random Forests are comparatively robust in high-dimensional, small-sample settings (Breiman, 2001). "
    "However, our comparison covers one feature type, 14 of 32 channels, 2-s windows and untuned models; other features, channels or properly tuned models may perform better under the same protocol, "
    "and trial-level DEAP results somewhat higher than ours have been reported. The present results should therefore be read as a baseline for this pipeline, not as evidence of a general 'data wall'."])
R.replace_para(chk(143, 'This limitation'), [
    "Further limitations are that only one dataset was analysed, that the 14-channel set includes the non-homologous pair C3/T8, that no artifact rejection beyond the dataset providers' preprocessing was applied, "
    "that the window-level CNN results use a static CNN rather than the CNN-LSTM, and that we did not re-implement published high-accuracy architectures under both protocols. "
    "Subject-independent models trained on larger cohorts, with domain adaptation to handle inter-subject differences (e.g., Li et al., 2021; Liu, K. et al., 2025), are a natural next step, "
    "but they too must be evaluated with partitions that keep test trials and test subjects unseen."])

# ================================================================== 6. CONCLUSION
R.replace_para(chk(145, 'This research'), [
    "This study re-evaluated subject-dependent EEG emotion recognition on DEAP with a CNN-LSTM and a Random Forest on Differential Entropy features, "
    "with the aim of measuring how much the evaluation protocol changes the reported accuracy."])
R.replace_para(chk(146, 'Our empirical'), [
    f"Under trial-level 5-fold cross-validation, the CNN-LSTM reached {m(CN('valence','trial'))}% (Valence) and {m(CN('arousal','trial'))}% (Arousal) and the Random Forest {m(pv)}% and {m(pa)}%, "
    f"against majority-class baselines of {maj_v_s}% and {maj_a_s}%; balanced accuracy was modestly above chance for Valence and near chance for Arousal. "
    f"Window-level splitting of the same non-overlapping windows inflated accuracy by {infl_range} percentage points, the inflation remained when the classification unit was held fixed, "
    f"and it grew with overlap to {ov['90']['mean']:.1f}% at 90% overlap. A split by the last ten stored trials is confounded by the video order of DEAP and does not measure temporal non-stationarity, "
    f"and leave-one-subject-out accuracy was near chance ({m(rf['loso_valence'])}% for Valence). These values are a baseline for this pipeline, not a ceiling for the field."])
R.replace_para(chk(147, 'Crucially'), [
    "Group-mean topographic maps suggested a left-frontal Gamma pattern for Valence, but a per-subject frontal-asymmetry test in the Alpha and Gamma bands found no significant effect. "
    "Topographic visualisations of input features should therefore be accompanied by quantitative tests, and Gamma-band results on scalp EEG require artifact control before physiological interpretation."])
R.replace_para(chk(148, 'Moving forward'), [
    "We recommend that studies report the partition unit, window length and overlap, the majority-class baseline and a balanced metric, and make code available. "
    "In future work we will evaluate subject-independent models with domain adaptation under strict subject-level partitions, and re-implement representative published architectures under trial-level and window-level protocols."])

# ================================================================== DECLARATIONS
p152 = chk(152, 'Data Availability')
code_p = R.insert_after(p152, p152, [("Code Availability:", dict(bold=True)),
    " The code for feature extraction, all models and protocols, the statistical analyses and the figures, the cached DE features, and the per-subject results are provided as supplementary material "
    "and will be deposited in a public repository upon acceptance [AUTHORS: insert repository URL / DOI]."], rPr=None)
R.replace_para(chk(153, 'Author Contributions'), [("Author Contributions:", dict(bold=True)), " ",
    ("Junwei Li: Conceptualization, Methodology, Software, Formal analysis, Investigation, Writing – original draft. "
     "Zhitong Li: Methodology, Software, Validation, Writing – review & editing. "
     "Wenjing Zhan: Data curation, Investigation. "
     "Jiemei Zeng: Data curation, Visualization. "
     "Bin Li: Conceptualization, Supervision, Project administration, Funding acquisition."),
    " All authors read and approved the final manuscript."], rPr=None)

# ================================================================== APPENDIX (before Statements and Declarations)
p149 = chk(149, 'Statements and Declarations')
anchor = P[148]
h_app = R.insert_after(anchor, h2_style, ['Appendix A. Supplementary tables'], rPr=R.first_rPr(h2_style))
rows = [['Study', 'Dataset', 'Partition / CV unit', 'Overlap', 'Reported accuracy', 'Same-trial windows in train and test?'],
        ['Koelstra et al. (2012)', 'DEAP', 'Leave-one-trial-out, within subject', 'n/a (trial features)', 'Valence 57.6%, Arousal 62.0% (EEG)', 'No'],
        ['Tripathi et al. (2017)', 'DEAP', 'One participant held out at a time', 'n/a', 'see Section 2.3 (leave-one-subject-out)', 'No'],
        ['Song et al. (2018)', 'SEED (also DREAMER)', 'Separate trials for training and test (SEED: first 9 / last 6 trials)', 'not applicable', '90.4% (SEED, 3 classes)', 'No'],
        ['Zhang, Y. et al. (2020)', 'DEAP', '10-fold CV; partition unit not stated', 'none stated (1-s segments)', 'up to 94.2% (CNN-LSTM, raw input)', 'Unclear'],
        ['Zhu et al. (2024)', 'SEED', 'Subject-dependent and subject-independent', 'not stated', '95.26% (subject-dep.), 94.22% (subject-indep.)', 'Unclear'],
        ['Moreno-Alcayde et al. (2023)', 'Affective EEG incl. DEAP', 'Compares partition strategies', 'varied', 'Partition choice changes accuracy substantially', 'Studied explicitly'],
        ['Brookshire et al. (2024)', 'Clinical / translational EEG', 'Segment-level vs subject-level hold-out', 'varied', 'Large inflation with segment-level hold-out', 'Subject-identity leakage'],
        ['This study', 'DEAP', 'Trial-level vs window-level, 5-fold, per subject', '0–90%', f"RF Valence {m(pv)}% (trial) vs {m(RFk('valence','window_leaky'))}% (window) and {ov['90']['mean']:.1f}% (90% overlap)", 'Measured']]
ta1 = R.make_table(t2, rows, weights=[1.6, 1.3, 2.2, 1.1, 2.2, 1.4]); h_app.addnext(ta1)
ca1 = R.insert_after(ta1, cap_style, ["Table A1. Evaluation protocols of the high-accuracy and methodological studies cited in this paper. Accuracies are not comparable across rows "
                                      "(different datasets, classes and protocols). 'Unclear' means the paper does not state whether windows of one trial can occur in both training and test folds."])
rows = [['Component', 'Setting'],
        ['Data', 'DEAP data_preprocessed_matlab, 32 subjects × 40 trials, 128 Hz, 4–45 Hz (provider), 14 channels'],
        ['Windows', '2 s (256 samples), step 256, 31 per trial (0–62 s incl. 3-s pre-stimulus), no baseline correction'],
        ['Features', 'DE = ½ ln(2πe P_band), FFT-256 periodogram, bands 4–8, 8–13, 13–30, 30–45 Hz; 56 per window'],
        ['Labels', 'Rating ≥ 5 → High; sensitivity: > 5'],
        ['CNN-LSTM', f"Conv 1→16 (3×2, pad 1), ReLU, MaxPool 2×2; Conv 16→32 (3×2, pad 1), ReLU; flatten 672; LSTM 2 layers × 64, dropout 0.3; dropout 0.5; linear 64→2; {ovp['params_cnnlstm']:,} parameters"],
        ['CNN-LSTM training', 'Adam, lr 5e-4, weight decay 1e-4, batch 8, 50 epochs, cross-entropy; no validation set, early stopping, LR schedule or batch norm; z-score from training fold'],
        ['Static CNN (window protocols)', f"Same conv layers; dropout 0.5; linear 672→64→2; Adam lr 1e-3, batch 64, 50 epochs; {ovp['params_staticcnn']:,} parameters"],
        ['Random Forest', '100 trees, Gini, max_depth None, min_samples_split 2, max_features √p, random_state 42'],
        ['Cross-validation', 'Per subject, KFold(5, shuffle, seed 42); trial folds 32/8; window folds 992/248; stratified folds as sensitivity'],
        ['Seeds', 'RF 42; CNN 42, 43, 44 (results averaged)'],
        ['Statistics', 'Paired t-test, Holm correction, 95% CI of mean difference, d_z; label-permutation test (RF, 200 permutations)'],
        ['Software', 'Python 3.12, PyTorch 2.x, scikit-learn, SciPy; script revision_v2_analyses.py']]
ta2 = R.make_table(T[3], rows, weights=[1.2, 4.0]); ca1.addnext(ta2)
R.insert_after(ta2, cap_style, ["Table A2. Implementation details and hyperparameters (all values as used for the results in this paper)."])

# ================================================================== REFERENCES
refs = {R.text(p)[:40]: p for p in P[156:]}
def ref_starting(s):
    hits = [p for k, p in refs.items() if k.startswith(s)]
    assert len(hits) == 1, s; return hits[0]
for s in ['Cao, L.', 'Chao, H.', 'Lin, Y.-P.', 'Nie, D.', 'Petrantonakis', 'Soleymani', 'Zhang, T.', 'Zheng, W. L., Liu, W.', 'Zheng, W. L., Zhu, J. Y.']:
    R.delete_para(ref_starting(s))
ref_like = ref_starting('Breiman')
R.insert_after(ref_starting('Duan, R. N.'), ref_like, ["Goncharova, I. I., McFarland, D. J., Vaughan, T. M., & Wolpaw, J. R. (2003). EMG contamination of EEG: Spectral and topographical characteristics. "
                                                       "Clinical Neurophysiology, 114(9), 1580-1593. https://doi.org/10.1016/S1388-2457(03)00093-2"])
holm_ref = R.insert_after(ref_starting('Hochreiter'), ref_like, ["Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics, 6(2), 65-70."])
R.insert_after(holm_ref, ref_like, ["Karimi, S., Nateghi, M., Cestero, G. I., Chitadze, L., Deepanshi, Yang, Y., Vyas, J. H., Chen, C., Bouzid, Z., Yaldiz, C. O., Harris, N., Bull, R., Stone, B. T., Lynn, S. K., Bracken, B. K., Inan, O. T., Bremner, J. D., & Sameni, R. (2025). Prescreening depression using wearable electrocardiogram and photoplethysmogram data from a psycholinguistic experiment. Physiological Measurement, 46(8), 085004. https://doi.org/10.1088/1361-6579/adf6fe"])
# Liu, Xue & Yang: Brain Sciences vol. 16 belongs to 2026 (reviewer's year mismatch note)
liu_ref = ref_starting('Liu, Y., Xue')
R.replace_sub(liu_ref, '(2025). Deep learning-based EEG emotion recognition', '(2026). Deep learning-based EEG emotion recognition')

# ------------------------------------------------------------------ figures (image blobs; not tracked by Word)
for img, fn in [('image5.png', 'figure5_cnnlstm_valence.png'), ('image6.png', 'figure6_rf_valence.png'), ('image7.png', 'figure7_leakage.png'),
                ('image8.png', 'figure8_generalization.png'), ('image9.png', 'figure10_band_ablation.png')]:
    shutil.copyfile(FIG / fn, work / 'word' / 'media' / img)
# topographic maps (regenerated with MATLAB; see scripts/regenerate_topoplots_v2.m)
for img, fn in [('image10.png', 'figure09_gamma_valence.png'), ('image11.png', 'figure11_bands_valence.png'),
                ('image12.png', 'figure12_bands_arousal.png'), ('image13.png', 'figure13_gamma_quadrants.png'),
                ('image14.png', 'figure14_dde_valence.png'), ('image15.png', 'figure15_single_subjects.png'),
                ('image16.png', 'figure16_grand_stats.png')]:
    topo = ROOT / 'figures_topo' / fn
    if topo.exists():
        shutil.copyfile(topo, work / 'word' / 'media' / img)
    else:
        print(f'WARNING: topographic figure missing, keeping original: {fn}')

tree.write(str(doc_path), xml_declaration=True, encoding='UTF-8', standalone=True)
if OUT.exists(): OUT.unlink()
with zipfile.ZipFile(SRC) as zsrc, zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as zout:
    for item in zsrc.infolist():
        zout.write(work / item.filename, item.filename)
print('wrote', OUT)

# ------------------------------------------------------------------ export numbers for the response letter / change log
NUM = dict(
    rf_v=ms(pv), rf_a=ms(pa), rf_v_bal=ms(pv, 'bal_acc'), rf_a_bal=ms(pa, 'bal_acc'), rf_v_auc=m(pv, 'auroc'), rf_a_auc=m(pa, 'auroc'),
    rf_v_f1=m(pv, 'macro_f1'), rf_a_f1=m(pa, 'macro_f1'), maj_v=maj_v_s, maj_a=maj_a_s,
    rf_v_strat=m(RFk('valence', 'trial_stratified')), rf_a_strat=m(RFk('arousal', 'trial_stratified')),
    rf_v_gt5=m(RFk('valence', 'trial_threshold_gt5')), rf_a_gt5=m(RFk('arousal', 'trial_threshold_gt5')),
    rf_v_vs_maj=f"{tv['mean_diff']:+.2f} (95% CI {tv['ci95'][0]:.2f} to {tv['ci95'][1]:.2f}; {pfmt(tv['p'])})",
    rf_a_vs_maj=f"{ta['mean_diff']:+.2f} (95% CI {ta['ci95'][0]:.2f} to {ta['ci95'][1]:.2f}; {pfmt(ta['p'])})",
    rf_v_bal50=pfmt(rf['rf_valence']['trial_balacc_vs_50']['p']), rf_a_bal50=pfmt(rf['rf_arousal']['trial_balacc_vs_50']['p']),
    perm=(f"p = {perm['valence']['p_bal']:.3f} (Valence), p = {perm['arousal']['p_bal']:.3f} (Arousal), {perm['valence']['n_perm']} permutations" if perm else 'not run'),
    cnn_v=ms(CN('valence', 'trial')), cnn_a=ms(CN('arousal', 'trial')), cnn_v_bal=ms(CN('valence', 'trial'), 'bal_acc'), cnn_a_bal=ms(CN('arousal', 'trial'), 'bal_acc'),
    cnn_v_seeds=f"{min(CNs('valence','trial')):.2f}–{max(CNs('valence','trial')):.2f}", cnn_a_seeds=f"{min(CNs('arousal','trial')):.2f}–{max(CNs('arousal','trial')):.2f}",
    cnnc_v=m(CN('valence', 'trial', 'corrected')), cnnc_a=m(CN('arousal', 'trial', 'corrected')),
    gridfix_v=f"{grid_fix['valence']['diff']:+.2f}, {pfmt(grid_fix['valence']['p'])}", gridfix_a=f"{grid_fix['arousal']['diff']:+.2f}, {pfmt(grid_fix['arousal']['p'])}",
    cnn_vs_rf_v=f"{cnn_vs_rf['valence']['diff']:+.2f}, {pfmt(cnn_vs_rf['valence']['p'])}", cnn_vs_rf_a=f"{cnn_vs_rf['arousal']['diff']:+.2f}, {pfmt(cnn_vs_rf['arousal']['p'])}",
    rf_v_leak=ms(RFk('valence', 'window_leaky')), rf_a_leak=ms(RFk('arousal', 'window_leaky')),
    rf_v_grp=ms(RFk('valence', 'window_trialsplit')), rf_a_grp=ms(RFk('arousal', 'window_trialsplit')),
    rf_v_vote=m(RFk('valence', 'window_trialsplit'), 'trialvote_acc'), rf_a_vote=m(RFk('arousal', 'window_trialsplit'), 'trialvote_acc'),
    cnn_v_leak=ms(CN('valence', 'window_leaky')), cnn_a_leak=ms(CN('arousal', 'window_leaky')),
    cnn_v_grp=ms(CN('valence', 'window_trialsplit')), cnn_a_grp=ms(CN('arousal', 'window_trialsplit')),
    infl_range=infl_range, dz_range=dz_range, max_p_holm=('< 0.001' if max_ph < 0.001 else f'{max_ph:.3f}'),
    table2_md='\n'.join(['| Model, dimension | Comparison | Δ (pp) | 95% CI | t(31) | p (Holm) | d_z |', '|---|---|---|---|---|---|---|'] +
                        [f"| {a} | {b} | {r_['diff']:+.2f} | {r_['lo']:.2f} to {r_['hi']:.2f} | {r_['t']:.2f} | {'< 0.001' if r_['p_holm'] < .001 else f'{r_['p_holm']:.3f}'} | {r_['dz']:.2f} |" for a, b, r_ in fam]),
    ov0=f"{ov['0']['mean']:.2f} ± {ov['0']['sd']:.2f}", ov50=f"{ov['50']['mean']:.2f} ± {ov['50']['sd']:.2f}",
    ov75=f"{ov['75']['mean']:.2f} ± {ov['75']['sd']:.2f}", ov90=f"{ov['90']['mean']:.2f} ± {ov['90']['sd']:.2f}",
    ch_v=ms(ch_v['summary']), ch_v_bal=ms(ch_v['summary'], 'bal_acc'), ch_a=ms(ch_a['summary']), ch_a_bal=ms(ch_a['summary'], 'bal_acc'),
    ch_maj_rule=m(ch_v['summary'], 'train_majority_rule_acc'), ch_single=str(ch_v['n_subjects_single_class_test']),
    v_first30=f"{lo['valence_high_first30']*100:.0f}", v_last10=f"{lo['valence_high_last10']*100:.0f}",
    a_first30=f"{lo['arousal_high_first30']*100:.0f}", a_last10=f"{lo['arousal_high_last10']*100:.0f}",
    loso_v=ms(rf['loso_valence']), loso_v_bal=ms(rf['loso_valence'], 'bal_acc'), loso_a=ms(rf['loso_arousal']), loso_a_bal=ms(rf['loso_arousal'], 'bal_acc'),
    abl_v=', '.join(f"{b.capitalize()} {m(A[b],'bal_acc')}" for b in ['theta', 'alpha', 'beta', 'gamma']),
    abl_v_acc=', '.join(f"{b.capitalize()} {m(A[b])}" for b in ['theta', 'alpha', 'beta', 'gamma']),
    abl_a=', '.join(f"{b.capitalize()} {m(Aa[b],'bal_acc')}" for b in ['theta', 'alpha', 'beta', 'gamma']),
    abl_p=f"Theta p = {abl_t['valence']['theta']['p_holm']:.3f}, Alpha p = {abl_t['valence']['alpha']['p_holm']:.3f}, Beta p = {abl_t['valence']['beta']['p_holm']:.3f}",
    asym_a=f"{comp_a['mean_diff']:+.4f} (95% CI {comp_a['ci95'][0]:.4f} to {comp_a['ci95'][1]:.4f}; {pfmt(comp_a['p'])}; d_z = {comp_a['d_z']:.2f})",
    asym_g=f"{comp_g['mean_diff']:+.4f} (95% CI {comp_g['ci95'][0]:.4f} to {comp_g['ci95'][1]:.4f}; {pfmt(comp_g['p'])}; d_z = {comp_g['d_z']:.2f})",
    asym_min_holm=f"{pairs_min_p:.2f}", exact5_v=str(rf['exactly5_trials']['valence']), exact5_a=str(rf['exactly5_trials']['arousal']),
    hv_range=f"{min(hv):.1f}–{max(hv):.1f}", ha_range=f"{min(ha):.1f}–{max(ha):.1f}", mv_range=f"{min(mv):.1f}–{max(mv):.1f}",
    params_cnnlstm=f"{ovp['params_cnnlstm']:,}", params_static=f"{ovp['params_staticcnn']:,}",
)
json.dump(NUM, open(RES / 'numbers_v2.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
