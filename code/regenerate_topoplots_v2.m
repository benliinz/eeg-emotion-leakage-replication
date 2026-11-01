% regenerate_topoplots_v2.m
% Regenerate the 7 topographic figures (Figure 9, 11-16) for the v2 revision.
% Data: ../02_特征与数据/de_features_all.mat  (all_features 56x39680, band-major:
%       cols 1-14 Theta, 15-28 Alpha, 29-42 Beta, 43-56 Gamma; channels
%       AF3 F3 F7 FC5 C3 P7 O1 AF4 F4 F8 FC6 T8 P8 O2)
% Output: ../figures_topo/*.png (300 dpi)
% Base MATLAB only (griddata/contourf); reproducible, no toolboxes.

function regenerate_topoplots_v2()
SRC = fullfile(fileparts(mfilename('fullpath')), '..', '02_特征与数据', 'de_features_all.mat');
OUT = fullfile(fileparts(mfilename('fullpath')), '..', 'figures_topo');
if ~exist(OUT, 'dir'); mkdir(OUT); end
DPI = 300;

S = load(SRC);
F = S.all_features;
if size(F, 1) == 56 && size(F, 2) > 56
    F = F';                          % ensure (N x 56)
end
L = S.all_labels;
if size(L, 1) == 4 && size(L, 2) > 4
    L = L';                          % ensure (N x 4)
end
N = size(F, 1);
fprintf('features: %d windows x %d\n', N, size(F, 2));

BANDS = {'Theta','Alpha','Beta','Gamma'};
bandCols = @(b) (b-1)*14 + (1:14);

X = [-0.3 -0.5 -0.8 -0.8 -0.6 -0.8 -0.3  0.3  0.5  0.8  0.8  0.9  0.8  0.3];
Y = [ 0.8  0.5  0.5  0.2  0.0 -0.6 -0.95 0.8  0.5  0.5  0.2  0.0 -0.6 -0.95];

TRIALS = N / 31;
assert(TRIALS == 1280, 'unexpected trial count');
trIdx = repelem((1:TRIALS)', 31);
trV = L(1:31:end, 1);  trA = L(1:31:end, 2);

hiV = trV(trIdx) >= 5;   loV = trV(trIdx) < 5;
hiA = trA(trIdx) >= 5;   loA = trA(trIdx) < 5;
qHV = hiV & hiA;  qHL = hiV & loA;  qLL = loV & loA;  qLH = loV & hiA;

[Xq, Yq] = meshgrid(linspace(-1.1, 1.1, 220));
mask = Xq.^2 + Yq.^2 <= 1;

set(0, 'DefaultAxesFontName', 'Times New Roman', 'DefaultTextFontName', 'Times New Roman');

%% Figure 9: Gamma DE, High vs Low Valence
hv = mean(F(hiV, bandCols(4)), 1);
lv = mean(F(loV, bandCols(4)), 1);
panel_fig(OUT, 'figure09_gamma_valence.png', {'High Valence','Low Valence'}, {hv, lv}, ...
    X, Y, Xq, Yq, mask, 2, [2200 760], '', false);

%% Figure 11: 4 bands x High/Low Valence
tv = cell(1,8); tt = cell(1,8); k = 0;
for g = 1:2
    if g == 1, mk = hiV; lab = 'High Valence'; else, mk = loV; lab = 'Low Valence'; end
    for b = 1:4
        k = k + 1;
        tv{k} = mean(F(mk, bandCols(b)), 1);
        tt{k} = sprintf('%s - %s', BANDS{b}, lab);
    end
end
panel_fig(OUT, 'figure11_bands_valence.png', tt, tv, X, Y, Xq, Yq, mask, 4, [3800 1900], ...
    'Band-specific DE topographies: High vs Low Valence (all 32 subjects)', true);

%% Figure 12: 4 bands x High/Low Arousal
ta = cell(1,8); tb = cell(1,8); k = 0;
for g = 1:2
    if g == 1, mk = hiA; lab = 'High Arousal'; else, mk = loA; lab = 'Low Arousal'; end
    for b = 1:4
        k = k + 1;
        ta{k} = mean(F(mk, bandCols(b)), 1);
        tb{k} = sprintf('%s - %s', BANDS{b}, lab);
    end
end
panel_fig(OUT, 'figure12_bands_arousal.png', tb, ta, X, Y, Xq, Yq, mask, 4, [3800 1900], ...
    'Band-specific DE topographies: High vs Low Arousal (all 32 subjects)', true);

%% Figure 13: Gamma DE, 4 Valence-Arousal quadrants
gHV = mean(F(qHV, bandCols(4)), 1);  gHL = mean(F(qHL, bandCols(4)), 1);
gLL = mean(F(qLL, bandCols(4)), 1);  gLH = mean(F(qLH, bandCols(4)), 1);
panel_fig(OUT, 'figure13_gamma_quadrants.png', ...
    {'High V / High A','High V / Low A','Low V / High A','Low V / Low A'}, ...
    {gHV, gHL, gLH, gLL}, X, Y, Xq, Yq, mask, 2, [2300 2300], ...
    'Gamma-band DE topography across the four Valence-Arousal quadrants', false);

%% Figure 14: dDE (High - Low Valence) by band
dv = cell(1,4); dt = cell(1,4);
for b = 1:4
    dv{b} = mean(F(hiV, bandCols(b)), 1) - mean(F(loV, bandCols(b)), 1);
    dt{b} = sprintf('%s \\DeltaDE (High - Low Valence)', BANDS{b});
end
panel_fig(OUT, 'figure14_dde_valence.png', dt, dv, X, Y, Xq, Yq, mask, 4, [3800 1100], ...
    'Valence-related DE difference topographies by band', true);

%% Figure 15: single-subject Gamma DE (S1, S10, S20; High/Low Valence)
subs = [1 10 20];
ss = cell(1,6); st = cell(1,6); k = 0;
for s = subs
    mSub = (trIdx == s);
    conds = {mSub & hiV, 'High Valence'; mSub & loV, 'Low Valence'};
    for c = 1:2
        k = k + 1;
        ss{k} = mean(F(conds{c,1}, bandCols(4)), 1);
        st{k} = sprintf('Subject %d - %s', s, conds{c,2});
    end
end
panel_fig(OUT, 'figure15_single_subjects.png', st, ss, X, Y, Xq, Yq, mask, 3, [2800 1900], ...
    'Gamma DE topographies - single-subject examples', false);

%% Figure 16: grand mean / inter-subject SD / frontal asymmetry (Gamma)
gm  = mean(F(:, bandCols(4)), 1);
sdm = std(F(:, bandCols(4)), 0, 1);
% frontal asymmetry index per homologous frontal electrode:
% AI = (mean DE_R over High - mean DE_R over Low) / (mean DE_R over High + mean DE_R over Low)
% plotted only at the 8 frontal channels; C3/P7/O1/T8/P8/O2 left as NaN (grey in jet via clim guard)
frontalL = [2 3 4 5];   % AF3 F3 F7 FC5 (indices into 14)
frontalR = [8 9 10 11]; % AF4 F4 F8 FC6
asym = nan(1, 14);
for p = 1:4
    lch = frontalL(p); rch = frontalR(p);
    vh = mean(F(hiV, bandCols(4)));   % not per-channel; compute per channel below
    ml_h = mean(F(hiV, bandCols(4)));
    ml_l = mean(F(loV, bandCols(4)));
    % per-channel High/Low means
    m_h = mean(F(hiV, bandCols(4)), 1);
    m_l = mean(F(loV, bandCols(4)), 1);
    % index of "more right-active for High than Low" per channel
    asym(rch) = (m_h(rch) - m_l(rch)) / (m_h(rch) + m_l(rch));
    asym(lch) = (m_h(lch) - m_l(lch)) / (m_h(lch) + m_l(lch));
end
panel_fig(OUT, 'figure16_grand_stats.png', ...
    {'Grand mean Gamma DE','Inter-subject SD','Frontal asymmetry index'}, ...
    {gm, sdm, asym}, X, Y, Xq, Yq, mask, 3, [2800 980], ...
    'Grand statistics of Gamma-band DE', false);

fprintf('All topographic figures regenerated -> %s\n', OUT);
end

%% ---------------------------------------------------------------- helpers
function draw_topo(ax, X, Y, vals, Xq, Yq, mask, ttl, clim)
    Vq = griddata(X, Y, vals, Xq, Yq, 'cubic');
    Vq(~mask) = NaN;
    axes(ax); hold(ax, 'on');
    contourf(ax, Xq, Yq, Vq, 60, 'LineColor', 'none');
    colormap(ax, jet);
    if ~isempty(clim); caxis(ax, clim); end
    th = linspace(0, 2*pi, 200);
    plot(ax, cos(th), sin(th), 'k-', 'LineWidth', 1.6);
    plot(ax, [-0.10 0 0.10], [0.98 1.10 0.98], 'k-', 'LineWidth', 1.6);
    plot(ax, [-1 -1.12 -1], [0.10 0 -0.10], 'k-', 'LineWidth', 1.6);
    plot(ax, [ 1  1.12  1], [0.10 0 -0.10], 'k-', 'LineWidth', 1.6);
    scatter(ax, X, Y, 8, 'k', 'filled');
    axis(ax, 'equal', 'off'); xlim(ax, [-1.15 1.15]); ylim(ax, [-1.15 1.15]);
    title(ax, ttl, 'FontSize', 9, 'FontWeight', 'normal', 'Interpreter', 'tex');
end

function panel_fig(OUT, name, titles, topoVals, X, Y, Xq, Yq, mask, ncols, px, suptitle, sharedClim)
    nFig = numel(titles);
    nrows = ceil(nFig / ncols);
    fig = figure('Visible','off','Color','w','Position',[50 50 round(px(1)/2.2) round(px(2)/2.2)]);
    clim = [];
    if sharedClim
        allv = cell2mat(topoVals);
        clim = [min(allv(:)) max(allv(:))];
    end
    axLast = [];
    for k = 1:nFig
        ax = subplot(nrows, ncols, k);
        draw_topo(ax, X, Y, topoVals{k}, Xq, Yq, mask, titles{k}, clim);
        axLast = ax;
    end
    if sharedClim
        cb = colorbar(axLast);
        cb.Label.String = 'DE (log units)';
    end
    if ~isempty(suptitle)
        sgtitle(suptitle, 'FontSize', 11, 'FontWeight', 'bold');
    end
    set(fig, 'PaperPositionMode', 'auto');
    exportgraphics(fig, fullfile(OUT, name), 'Resolution', 300);
    close(fig);
    fprintf('wrote %s\n', name);
end
