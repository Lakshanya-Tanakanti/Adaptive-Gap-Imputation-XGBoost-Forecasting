"""
Experiment : Comparison of gap-filling strategies for AirTC_18m, including
Linear, Spline, Diagonal Climatology, and LOCF baselines against a proposed
Adaptive Imputation method (cross-validated, rule-based method selection per
gap). Synthetic gaps are injected at several missing-data percentages, each
method's recovery accuracy is evaluated against ground truth, and downstream
XGBoost forecasting accuracy is compared across methods.
"""

import importlib
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_absolute_error, mean_squared_error

plt.rcParams.update({
    'text.color':           'black',
    'font.family':          'DejaVu Sans',
    'font.weight':          'bold',
    'font.size':            17,
    'axes.labelcolor':      'black',
    'axes.labelweight':     'bold',
    'axes.labelsize':       18,
    'axes.titlesize':       20,
    'axes.titleweight':     'bold',
    'axes.edgecolor':       'black',
    'axes.linewidth':       2.0,
    'xtick.color':          'black',
    'ytick.color':          'black',
    'xtick.labelsize':      15,
    'ytick.labelsize':      15,
    'xtick.major.width':    2.0,
    'ytick.major.width':    2.0,
    'xtick.major.size':     6,
    'ytick.major.size':     6,
    'legend.fontsize':      15,
    'legend.title_fontsize':15,
    'legend.edgecolor':     'black',
    'legend.framealpha':    1.0,
    'figure.titlesize':     22,
    'figure.titleweight':   'bold',
    'figure.dpi':           100,
    'savefig.dpi':          300,
    'savefig.facecolor':    'white',
    'savefig.edgecolor':    'black',
    'savefig.bbox':         'tight',
    'savefig.pad_inches':   0.1,
})

SUP_FS    = 22   # figure-level suptitles
TITLE_FS  = 20   # per-axes titles
LABEL_FS  = 18   # x/y axis labels
TICK_FS   = 16   # tick labels
LEGEND_FS = 16   # legends
ANNOT_FS  = 16   # in-plot text annotations (e.g. "n = ..." boxes, bar labels)

import xgboost_model
import interpolation_methods as im
importlib.reload(xgboost_model)
importlib.reload(im)

_ORIGINAL_SAVEFIG = plt.savefig
def _hq_savefig(*args, **kwargs):
    """Wrap plt.savefig to enforce >=300 DPI, tight bbox, and a white background."""
    kwargs['dpi'] = max(300, kwargs.get('dpi') or 0)
    kwargs.setdefault('bbox_inches', 'tight')
    kwargs.setdefault('facecolor', 'white')
    kwargs.setdefault('pad_inches', 0.08)
    return _ORIGINAL_SAVEFIG(*args, **kwargs)
plt.savefig = _hq_savefig
xgboost_model.plt.savefig = _hq_savefig

def _hq_plot_results(results, title='XGBoost Prediction', out_path=None):
    """Publication-quality replacement for xgboost_model.plot_results."""
    if out_path is None:
        out_path = xgboost_model.out(xgboost_model.FOLDER_RESULTS, "results.png")
    y_test  = results['y_test']; y_pred = results['y_pred']
    ts_test = results['ts_test']
    mae, rmse, r2 = results['mae'], results['rmse'], results['r2']
    step = 6; total = len(y_test); chunk = total // 4

    R_SUP, R_TITLE, R_LABEL, R_TICK, R_LEGEND, R_ANNOT = 30, 27, 25, 20, 24, 25

    fig = plt.figure(figsize=(28, 20))
    gs = fig.add_gridspec(3, 12, hspace=0.42, wspace=0.55,
                          left=0.045, right=0.985, top=0.90, bottom=0.045)
    part_axes = [
        fig.add_subplot(gs[0, 0:6]), fig.add_subplot(gs[0, 6:12]),
        fig.add_subplot(gs[1, 0:6]), fig.add_subplot(gs[1, 6:12]),
    ]
    ax_info = fig.add_subplot(gs[2, 1:5])
    ax_sc   = fig.add_subplot(gs[2, 6:10])

    date_fmt   = mdates.DateFormatter('%d %b')
    date_loc   = mdates.AutoDateLocator(minticks=4, maxticks=5)

    common_handles = []
    for i, ax in enumerate(part_axes):
        s = i * chunk; e = (i + 1) * chunk if i < 3 else total
        line_actual, = ax.plot(ts_test.iloc[s:e:step], y_test[s:e:step],
                color="#1565C0", linewidth=2.2, alpha=0.9, label="Actual")
        line_pred, = ax.plot(ts_test.iloc[s:e:step], y_pred[s:e:step],
                color="#E53935", linewidth=2.2, alpha=0.85,
                linestyle="--", label="Predicted")
        if i == 0:
            common_handles += [line_actual, line_pred]
        ax.set_title(f"Part {i+1}: {ts_test.iloc[s].strftime('%d %b')} to "
                     f"{ts_test.iloc[e-1].strftime('%d %b %Y')}",
                     fontsize=R_TITLE, fontweight="bold", color="black")
        ax.set_xlabel("Date", fontsize=R_LABEL, fontweight="bold", color="black",
                      labelpad=14)
        if i % 2 == 0:
            ax.set_ylabel("Air Temperature (°C)", fontsize=R_LABEL,
                          fontweight="bold", color="black", labelpad=14)
        ax.xaxis.set_major_locator(date_loc)
        ax.xaxis.set_major_formatter(date_fmt)
        ax.set_xmargin(0.04)
        ax.tick_params(axis="x", rotation=0, labelsize=R_TICK, colors="black",
                        pad=12)
        ax.tick_params(axis="y", labelsize=R_TICK, colors="black", pad=8)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight("bold")
        ax.grid(True, alpha=0.25)
        for spine in ax.spines.values():
            spine.set_edgecolor("black"); spine.set_linewidth(2.0)

    ax_info.axis('off')
    mae_str  = (f"{mae:.3f} +/- {results['mae_std']:.3f}"
                if 'mae_std' in results else f"{mae:.3f}")
    rmse_str = (f"{rmse:.3f} +/- {results['rmse_std']:.3f}"
                if 'rmse_std' in results else f"{rmse:.3f}")
    r2_str   = (f"{r2:.4f} +/- {results['r2_std']:.4f}"
                if 'r2_std' in results else f"{r2:.4f}")
    bias_str = (f"{results['bias']:.3f} +/- {results['bias_std']:.3f}"
                if 'bias_std' in results else f"{results['bias']:.3f}")
    val_line = (f"\nVal RMSE = {results['val_rmse']:.3f}  "
                f"Val R2 = {results['val_r2']:.4f}"
                if 'val_rmse' in results else "")
    ax_info.text(0.5, 0.5,
                 f"MAE  = {mae_str} °C\n"
                 f"RMSE = {rmse_str} °C\n"
                 f"R2   = {r2_str}\n"
                 f"Bias = {bias_str} °C{val_line}",
                 transform=ax_info.transAxes, fontsize=R_ANNOT,
                 va="center", ha="center", fontfamily="monospace",
                 fontweight="bold", color="black",
                 bbox=dict(boxstyle="round,pad=1.0", fc="white",
                           ec="black", lw=2.4, alpha=1.0))

    ax_sc.scatter(y_test, y_pred, alpha=0.30, s=70,
                  color="#6A1B9A", rasterized=True,
                  edgecolors="none")
    lims = [min(y_test.min(), y_pred.min()) - 1,
            max(y_test.max(), y_pred.max()) + 1]
    line_perfect, = ax_sc.plot(lims, lims, color="#F57C00", linewidth=2.9,
               linestyle="--", label="Perfect fit")
    common_handles.append(line_perfect)
    ax_sc.set_xlim(lims); ax_sc.set_ylim(lims)
    ax_sc.set_title(f"Actual vs Predicted  (R2 = {r2:.3f})",
                    fontsize=R_TITLE, fontweight="bold", color="black")
    ax_sc.set_xlabel("Actual (°C)", fontsize=R_LABEL, fontweight="bold",
                     color="black", labelpad=14)
    ax_sc.set_ylabel("Predicted (°C)", fontsize=R_LABEL, fontweight="bold",
                     color="black", labelpad=14)
    ax_sc.tick_params(labelsize=R_TICK, colors="black", pad=8)
    for lbl in ax_sc.get_xticklabels() + ax_sc.get_yticklabels():
        lbl.set_fontweight("bold")
    ax_sc.grid(True, alpha=0.25)
    for spine in ax_sc.spines.values():
        spine.set_edgecolor("black"); spine.set_linewidth(2.0)

    fig.legend(handles=common_handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.985), ncol=len(common_handles),
               framealpha=1.0, fontsize=R_LEGEND, edgecolor="black",
               prop={"weight": "bold", "size": R_LEGEND})

    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Plot -> {out_path}")

xgboost_model.plot_results = _hq_plot_results

def _hq_plot_boosting_curve(results, title_prefix, tag):
    """Publication-quality replacement for xgboost_model.plot_boosting_curve."""
    evals = results.get('evals_result', {})
    if not evals:
        print("  [Boosting curve] No evals_result, skipping."); return
    keys      = list(evals.keys())
    train_key = keys[0] if len(keys) > 0 else None
    val_key   = keys[1] if len(keys) > 1 else None
    metric    = 'rmse'
    B_TITLE, B_LABEL, B_TICK, B_LEGEND, B_ANNOT = 20, 18, 17, 17, 17

    fig, ax = plt.subplots(figsize=(16, 7.6))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.11)
    if train_key and metric in evals[train_key]:
        ax.plot(evals[train_key][metric], color='steelblue',
                lw=2.6, label='Train RMSE')
    if val_key and metric in evals[val_key]:
        val_scores = evals[val_key][metric]
        ax.plot(val_scores, color='#E53935', lw=2.6,
                label='Validation RMSE')
        best_iter  = int(np.argmin(val_scores))
        ax.axvline(best_iter, color='grey', lw=2.2, ls='--', alpha=0.8)
        y_pos = min(val_scores) + (max(val_scores) - min(val_scores)) * 0.6
        ax.text(best_iter + max(5, len(val_scores) * 0.02), y_pos,
                f'Best: tree {best_iter}\nRMSE = {val_scores[best_iter]:.3f}',
                fontsize=B_ANNOT, fontweight='bold', color='black', va='center',
                bbox=dict(boxstyle='round,pad=0.4', fc='white',
                          ec='black', lw=1.9, alpha=1.0))
    ax.set_xlabel('Number of Trees', fontsize=B_LABEL, fontweight='bold',
                  color='black')
    ax.set_ylabel('RMSE (°C)', fontsize=B_LABEL, fontweight='bold', color='black')
    ax.tick_params(labelsize=B_TICK, colors='black', width=2)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.99),
               ncol=2, fontsize=B_LEGEND, prop={'weight': 'bold', 'size': B_LEGEND},
               framealpha=1.0, edgecolor='black')
    ax.grid(True, alpha=0.25)
    for spine in ax.spines.values():
        spine.set_edgecolor('black'); spine.set_linewidth(2.0)
    path = xgboost_model.out(xgboost_model.FOLDER_EXTRA, f'boosting_curve_{tag}.png')
    plt.savefig(path, dpi=300, bbox_inches='tight'); plt.close()
    print(f"  Boosting curve -> {path}")

xgboost_model.plot_boosting_curve = _hq_plot_boosting_curve



def plot_combined_boosting_curves(evals_by_method, pct_label, pct_tag):
    """One figure, one subplot PER METHOD (Linear, Spline, Diagonal
    Climatology, LOCF, Adaptive Imputation), each showing its own
    train/validation RMSE-vs-trees curve with a best-iteration callout.
    Big and clear, no legend ever floats on top of a curve: there is
    exactly one shared legend (Train RMSE / Validation RMSE) placed in
    its own dedicated row above all the panels."""
    metric = 'rmse'
    CB_TITLE, CB_LABEL, CB_TICK, CB_LEGEND, CB_ANNOT = 22, 19, 17, 20, 15

    method_order = ['Linear', 'Spline', 'Diagonal Climatology',
                    'LOCF', 'Adaptive Imputation']
    methods_present = [m for m in method_order if evals_by_method.get(m)]
    n = len(methods_present)
    if n == 0:
        print("  [Combined boosting curve] No data, skipping."); return

    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(9 * ncols, 7.2 * nrows),
                              squeeze=False)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.90, bottom=0.07,
                         hspace=0.38, wspace=0.24)

    shared_handles = None
    for idx, method_name in enumerate(methods_present):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        evals = evals_by_method[method_name]
        keys      = list(evals.keys())
        train_key = keys[0] if len(keys) > 0 else None
        val_key   = keys[1] if len(keys) > 1 else None

        h_train = h_val = None
        if train_key and metric in evals[train_key]:
            h_train, = ax.plot(evals[train_key][metric], color='steelblue',
                                lw=2.8, label='Train RMSE')
        if val_key and metric in evals[val_key]:
            val_scores = evals[val_key][metric]
            h_val, = ax.plot(val_scores, color='#E53935', lw=2.8,
                              label='Validation RMSE')
            best_iter = int(np.argmin(val_scores))
            ax.axvline(best_iter, color='grey', lw=1.9, ls='--', alpha=0.8)
            ax.scatter([best_iter], [val_scores[best_iter]], color='#E53935',
                       s=90, edgecolor='black', linewidth=1.7, zorder=5)
            y_pos = min(val_scores) + (max(val_scores) - min(val_scores)) * 0.55
            x_pos = best_iter + max(5, len(val_scores) * 0.03)
            ha = 'left' if x_pos < 0.75 * len(val_scores) else 'right'
            if ha == 'right':
                x_pos = best_iter - max(5, len(val_scores) * 0.03)
            ax.text(x_pos, y_pos,
                    f'Best: tree {best_iter}\nRMSE = {val_scores[best_iter]:.3f}',
                    fontsize=CB_ANNOT, fontweight='bold', color='black',
                    va='center', ha=ha,
                    bbox=dict(boxstyle='round,pad=0.35', fc='white',
                              ec='black', lw=1.6, alpha=1.0))
        if shared_handles is None and h_train is not None and h_val is not None:
            shared_handles = [h_train, h_val]

        ax.set_title(method_name, fontsize=CB_TITLE, fontweight='bold',
                     color='black')
        ax.set_xlabel('Number of Trees', fontsize=CB_LABEL, fontweight='bold',
                      color='black')
        ax.set_ylabel('RMSE (°C)', fontsize=CB_LABEL, fontweight='bold',
                      color='black')
        ax.tick_params(labelsize=CB_TICK, colors='black', width=1.5)
        ax.grid(True, alpha=0.25)
        for spine in ax.spines.values():
            spine.set_edgecolor('black'); spine.set_linewidth(1.8)

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis('off')

    if shared_handles:
        fig.legend(handles=shared_handles, loc='upper center',
                   bbox_to_anchor=(0.5, 0.985), ncol=2,
                   framealpha=1.0, fontsize=CB_LEGEND, edgecolor='black',
                   prop={'weight': 'bold', 'size': CB_LEGEND})

    path = os.path.join(EXTRA, f'boosting_curve_ALL_METHODS_{pct_tag}.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Combined boosting curve ({pct_label}) -> {path}")

def _hq_plot_acf_pacf_lag_selection(df_clean, target='AirTC_18m'):
    """Publication-quality replacement for xgboost_model.plot_acf_pacf_lag_selection."""
    from scipy.stats import pearsonr

    acf  = xgboost_model.acf
    pacf = xgboost_model.pacf
    out  = xgboost_model.out
    FOLDER_EXTRA  = xgboost_model.FOLDER_EXTRA
    LAGS_AIRTEMP  = xgboost_model.LAGS_AIRTEMP
    LAGS_RH       = xgboost_model.LAGS_RH
    LAGS_WS       = xgboost_model.LAGS_WS
    LAGS_WINDDIR  = xgboost_model.LAGS_WINDDIR

    df_sorted = df_clean.sort_values('TIMESTAMP')
    temp      = df_sorted[target].values
    rh        = df_sorted['RH_18m'].values
    ws        = df_sorted['WS_ms_18m_Avg'].values
    wd        = df_sorted['WindDir_18m'].values

    n_lags    = 100
    acf_vals  = acf(temp, nlags=n_lags, fft=True)
    pacf_vals = pacf(temp, nlags=min(50, len(temp)//2 - 1))
    conf      = 1.96 / np.sqrt(len(temp))

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.05,
                                     hspace=0.03, wspace=0.03)

    def _annotate_no_overlap(ax, lags, values, base_offset=0.025):
        """Place value labels above each highlighted bar without letting
        adjacent labels collide. Lags that are close together on the
        x-axis (e.g. 1, 2, 3, 4) get a rotated, staggered label so text
        boxes don't overlap each other or the bar next to them."""
        sorted_lags = sorted(lags)
        for i, lag in enumerate(sorted_lags):
            if lag >= len(values):
                continue
            v = values[lag]
            close_to_prev = i > 0 and (lag - sorted_lags[i - 1]) <= 5
            close_to_next = (i < len(sorted_lags) - 1
                              and (sorted_lags[i + 1] - lag) <= 5)
            crowded = close_to_prev or close_to_next
            extra = base_offset * (1 + 1.6 * (i % 2)) if crowded else base_offset
            ax.text(lag, v + extra, f'{v:.3f}',
                     ha='center', va='bottom', fontsize=11, fontweight='bold',
                     color='#B71C1C', rotation=90 if crowded else 0)

    lags_acf = np.arange(len(acf_vals))
    axes[0].bar(lags_acf, acf_vals, color='#1565C0', alpha=0.5, width=0.8)
    axes[0].axhline( conf, color='red', ls='--', lw=1.9, alpha=0.8,
                     label=f'95% CI (+/-{conf:.3f})')
    axes[0].axhline(-conf, color='red', ls='--', lw=1.9, alpha=0.8)
    axes[0].axhline(0, color='black', lw=1.4)
    for lag in LAGS_AIRTEMP:
        if lag < len(acf_vals):
            axes[0].bar(lag, acf_vals[lag], color='#E53935', alpha=0.9, width=0.8)
    _annotate_no_overlap(axes[0], LAGS_AIRTEMP, acf_vals)
    axes[0].set_xlabel('Lag (steps, 1 step = 30 min)', fontsize=12)
    axes[0].set_ylabel('ACF', fontsize=12)
    axes[0].legend(fontsize=11, loc='upper right', framealpha=1.0, edgecolor='black')
    axes[0].set_xlim(-1, n_lags + 1)
    axes[0].set_ylim(min(acf_vals.min(), -conf) - 0.05, 1.22)
    axes[0].grid(True, alpha=0.25)

    lags_pacf = np.arange(len(pacf_vals))
    axes[1].bar(lags_pacf, pacf_vals, color='#2E7D32', alpha=0.5, width=0.8)
    axes[1].axhline( conf, color='red', ls='--', lw=1.9, alpha=0.8,
                     label=f'95% CI (+/-{conf:.3f})')
    axes[1].axhline(-conf, color='red', ls='--', lw=1.9, alpha=0.8)
    axes[1].axhline(0, color='black', lw=1.4)
    for lag in LAGS_AIRTEMP:
        if lag < len(pacf_vals):
            axes[1].bar(lag, pacf_vals[lag], color='#E53935', alpha=0.9,
                        width=0.8)
    _annotate_no_overlap(axes[1], [l for l in LAGS_AIRTEMP if l < len(pacf_vals)],
                         pacf_vals, base_offset=0.02)
    axes[1].set_xlabel('Lag (steps, 1 step = 30 min)', fontsize=12)
    axes[1].set_ylabel('PACF', fontsize=12)
    axes[1].legend(fontsize=11, loc='upper right', framealpha=1.0, edgecolor='black')
    axes[1].set_xlim(-1, min(51, len(pacf_vals) + 1))
    axes[1].set_ylim(min(pacf_vals.min(), -conf) - 0.05,
                      max(pacf_vals.max(), conf) + 0.35)
    axes[1].grid(True, alpha=0.25)

    path = out(FOLDER_EXTRA, 'acf_pacf_lag_justification.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', pad_inches=0.03)
    plt.close()
    print(f"  ACF/PACF plot -> {path}")

    check_lags = [1, 2, 3, 4, 6, 12, 24, 48]
    var_info = {
        'RH_18m':        (rh,  '#1565C0', LAGS_RH,      'Relative Humidity'),
        'WS_ms_18m_Avg': (ws,  '#2E7D32', LAGS_WS,      'Wind Speed'),
        'WindDir_18m':   (wd,  '#F57C00', LAGS_WINDDIR,  'Wind Direction'),
    }

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.05,
                                     hspace=0.03, wspace=0.08)

    for ax, (vname, (vseries, clr, sel_lags, vlabel)) in \
            zip(axes, var_info.items()):
        rs = []
        for lag in check_lags:
            r, _ = pearsonr(vseries[:-lag], temp[lag:])
            rs.append(r)

        bars = ax.bar(range(len(check_lags)), rs,
                      color=[('#E53935' if check_lags[i] in sel_lags
                               else clr)
                             for i in range(len(check_lags))],
                      alpha=0.8, edgecolor='black', lw=0.9)
        ax.axhline(0, color='black', lw=1.4)
        ax.axhline( 0.05, color='grey', ls='--', lw=1.4, alpha=0.6)
        ax.axhline(-0.05, color='grey', ls='--', lw=1.4, alpha=0.6)
        ax.set_xticks(range(len(check_lags)))
        ax.set_xticklabels([f'{l}\n({l*0.5:.1f}h)' for l in check_lags],
                            fontsize=11)
        ax.set_xlabel('Lag (steps)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Pearson r with AirTC_18m', fontsize=11,
                      fontweight='bold')
        ax.grid(True, alpha=0.25, axis='y')
        r_min, r_max = min(rs), max(rs)
        span = max(r_max - r_min, 0.05)
        ax.set_ylim(r_min - 0.22 * span, r_max + 0.30 * span)
        for i, (lag, r) in enumerate(zip(check_lags, rs)):
            offset = 0.05 * span if r >= 0 else -0.08 * span
            ax.text(i, r + offset,
                    f'{r:+.3f}', ha='center',
                    va='bottom' if r >= 0 else 'top',
                    fontsize=11, fontweight='bold',
                    color='#B71C1C' if lag in sel_lags else 'black')
        for spine in ax.spines.values():
            spine.set_edgecolor('black'); spine.set_linewidth(1.5)

    path2 = out(FOLDER_EXTRA, 'cross_correlation_lag_justification.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight', pad_inches=0.03)
    plt.close()
    print(f"  Cross-correlation plot -> {path2}")

    # Print summary (unchanged, informational only)
    print("\n  === Lag Selection Justification ===")
    print(f"\n  AirTC own lags {LAGS_AIRTEMP}:")
    reasons_temp = {
        1: 'ACF=0.991, strongest autocorrelation',
        2: 'ACF=0.972, strong 1h persistence',
        3: 'ACF=0.945, PACF significant',
        4: 'ACF=0.911, PACF significant, AR(4) confirmed',
        6: 'ACF=0.829, sub-diurnal transitions',
        12:'ACF=0.522, half-day cycle',
        48:'ACF=0.930, dominant diurnal cycle',
    }
    for lag in LAGS_AIRTEMP:
        print(f"    lag {lag:2d} ({lag*0.5:4.1f}h): {reasons_temp[lag]}")

    print(f"\n  RH lags {LAGS_RH}:")
    print("    Cross-correlation r=-0.68 to -0.65 at lags 1,2,4,6,12,48")
    print("    Strong negative relationship (when temp rises, RH drops)")
    print("    Diurnal cycle preserved at lag 48 (r=-0.645)")

    print(f"\n  Wind Speed lags {LAGS_WS}:")
    print("    Peak cross-correlation at lag 4-6 (r=+0.085)")
    print("    lag 48 retains diurnal signal (r=+0.082)")
    print("    lag 12 excluded, cross-correlation drops to r=0.015 (not significant)")

    print(f"\n  Wind Direction lags {LAGS_WINDDIR}:")
    print("    lag 12 most informative (r=-0.092, p=2e-32)")
    print("    lag 48 excluded, r=-0.007, p=0.35 (not significant)")

xgboost_model.plot_acf_pacf_lag_selection = _hq_plot_acf_pacf_lag_selection

GAP   = xgboost_model.FOLDER_GAP_PLOTS
RES   = xgboost_model.FOLDER_RESULTS
COMP  = xgboost_model.FOLDER_COMPARISON
EXTRA = xgboost_model.FOLDER_EXTRA
ROOT  = xgboost_model.OUTPUT_ROOT
TEST_START = xgboost_model.TEST_START

COND_DIR = os.path.join(ROOT, "interpolation_comparison")
os.makedirs(COND_DIR, exist_ok=True)

METHOD_COLORS = {
    'Linear':               '#E53935',
    'Spline':               '#FF9800',
    'Diagonal Climatology': '#00C853',
    'LOCF':                 '#AA00FF',
}
METHODS = ['Linear', 'Spline', 'Diagonal Climatology', 'LOCF']
# PCT_ORDER is set dynamically after the main loop

# LOAD DATA
df_clean = xgboost_model.load_and_clean('Proj data.csv')

# ACF/PACF + CROSS-CORRELATION
print("\n=== ACF/PACF + Cross-Correlation Lag Feature Justification ===")
xgboost_model.plot_acf_pacf_lag_selection(df_clean)

SEEDS = [42, 7, 13, 21, 99]

# HYPERPARAMETER TUNING — chronological 80/10/10
print("\n=== Hyperparameter Tuning (Optuna, Chronological 80/10/10) ===")
xgboost_model.tune_hyperparameters(df_clean, n_trials=25, test_start=TEST_START)

print("\n=== Base Model (clean data, chronological, lag features) ===")
_base_maes, _base_rmses, _base_r2s, _base_biases = [], [], [], []
base_results = None
for _seed in SEEDS:
    _r = xgboost_model.train_and_evaluate(df_clean, block_seed=_seed,
                                          test_start=TEST_START)
    _base_maes.append(_r["mae"]);  _base_rmses.append(_r["rmse"])
    _base_r2s.append(_r["r2"]);    _base_biases.append(_r["bias"])
    base_results = _r

_mean_mae,  _std_mae  = float(np.mean(_base_maes)),  float(np.std(_base_maes))
_mean_rmse, _std_rmse = float(np.mean(_base_rmses)), float(np.std(_base_rmses))
_mean_r2,   _std_r2   = float(np.mean(_base_r2s)),   float(np.std(_base_r2s))
_mean_bias, _std_bias = float(np.mean(_base_biases)),float(np.std(_base_biases))
base_results.update({
    "mae": _mean_mae, "rmse": _mean_rmse, "r2": _mean_r2, "bias": _mean_bias,
    "mae_std": _std_mae, "rmse_std": _std_rmse,
    "r2_std": _std_r2, "bias_std": _std_bias,
})
print(f"  Base Model: RMSE={_mean_rmse:.3f}+/-{_std_rmse:.3f} "
      f"over {len(SEEDS)} seeds")

xgboost_model.plot_results(base_results,
    title=f"Base Model — Chronological 80/10/10 (Lag Features, {len(SEEDS)} seeds)",
    out_path=os.path.join(RES, "base_model.png"))
xgboost_model.plot_residuals(base_results, "Base Model", "base")
xgboost_model.plot_feature_importance(base_results, "Base Model", "base")
xgboost_model.plot_error_by_hour(base_results, "Base Model", "base")
xgboost_model.plot_boosting_curve(base_results, "Base Model", "base")

# FULL DATASET SPLIT PLOT
df_indexed       = df_clean.copy()
df_indexed.index = pd.to_datetime(df_indexed["TIMESTAMP"])
df_indexed       = df_indexed.drop(columns=["TIMESTAMP"])
xgboost_model.plot_full_dataset_split(base_results, df_indexed, tag='base')

# FORECAST CONFIDENCE INTERVALS
print("\n=== Forecast Confidence Intervals ===")
xgboost_model.forecast_confidence_intervals(
    df_clean, n_bootstrap=30, show_steps=48, tag='base', test_start=TEST_START)

# INDEXED DATA
df_train_only = df_indexed[df_indexed.index <  TEST_START]
df_test_only  = df_indexed[df_indexed.index >= TEST_START]

# STORAGE
model_results     = []
recovery_records  = []
analysis_datasets = []
all_gap_row_sets  = {}
all_method_dfs    = {}

all_boosting_evals = {}

adaptive_model_results    = []
adaptive_recovery_records = []
adaptive_choice_logs      = {}
all_adaptive_dfs          = {}


# CONDITION THRESHOLDS
_TEMP_VERY_HIGH_THRESH = df_train_only['AirTC_18m'].quantile(0.90)
_TEMP_HIGH_THRESH      = df_train_only['AirTC_18m'].quantile(0.85)
_TEMP_LOW_THRESH       = df_train_only['AirTC_18m'].quantile(0.15)
_WIND_LOW_THRESH       = df_train_only['WS_ms_18m_Avg'].quantile(0.20)
_TEMP_STD              = df_train_only['AirTC_18m'].std()

print(f"\n  Condition thresholds:")
print(f"    Extreme high temp : AirTC >= {_TEMP_VERY_HIGH_THRESH:.1f} C")
print(f"    High temperature  : AirTC >= {_TEMP_HIGH_THRESH:.1f} C")
print(f"    Low temperature   : AirTC <= {_TEMP_LOW_THRESH:.1f} C")
print(f"    Stagnant wind     : WS <= {_WIND_LOW_THRESH:.2f} m/s")


def make_corrupted(count):
    """
    Synthetically remove AirTC_18m values from the training set at a target
    missing percentage, biased toward peaks, troughs, and steep transitions,
    with gap lengths drawn as short/medium/long blocks. Returns the corrupted
    dataframe, the actual missing percentage achieved, and the row indices of
    each gap block.
    """
    temp        = df_train_only["AirTC_18m"].ffill()
    peak_idx,   _ = find_peaks(temp.values, distance=6)
    trough_idx, _ = find_peaks(-temp.values, distance=6)
    slope         = np.abs(np.gradient(temp.values))
    trans_idx     = np.where(slope > np.percentile(slope, 85))[0]

    pool = {}
    for i in peak_idx:   pool[i] = 'peak'
    for i in trough_idx: pool[i] = 'trough'
    for i in trans_idx:
        if i not in pool: pool[i] = 'transition'

    pool_arr = np.array(list(pool.keys()))
    total    = len(df_train_only)
    target   = int(0.05 * count * total)

    np.random.seed(42)
    np.random.shuffle(pool_arr)

    df_train_corr = df_train_only.copy()
    used_rows     = set()
    gap_row_sets  = []

    for center in pool_arr:
        if len(used_rows) >= target: break
        gap_type = np.random.choice(['short', 'medium', 'long'],
                                     p=[0.5, 0.35, 0.15])
        if gap_type == 'short':
            w = np.random.randint(1, 7)
        elif gap_type == 'medium':
            w = np.random.randint(7, 25)
        else:
            w = np.random.randint(25, 49)

        block = []
        for i in range(max(0, center - w // 2),
                       min(total, center + w // 2 + 1)):
            if i not in used_rows:
                used_rows.add(i); block.append(i)
            if len(used_rows) >= target: break
        if block:
            gap_row_sets.append(sorted(block))

    for i in pool_arr:
        if len(used_rows) >= target: break
        if i not in used_rows:
            used_rows.add(i); gap_row_sets.append([i])

    all_rows     = sorted(used_rows)[:target]
    all_rows_set = set(all_rows)
    gap_row_sets = [
        [r for r in block if r in all_rows_set]
        for block in gap_row_sets
        if any(r in all_rows_set for r in block)
    ]

    df_train_corr.iloc[
        all_rows,
        df_train_corr.columns.get_loc('AirTC_18m')
    ] = np.nan

    actual_pct = df_train_corr['AirTC_18m'].isna().sum() / total * 100
    return df_train_corr, actual_pct, gap_row_sets


# CATEGORISATION HELPERS
def classify_condition(row_indices):
    """
    Oracle label for diagnostics only — not a decision input.

    Reads the true temperature/wind inside the gap from df_train_only,
    used solely to stratify recovery accuracy by ambient condition after
    the fact (e.g. condition heatmaps). Never passed to or used by
    classify_gap_adaptive / build_cv_rule_table, so it has no influence
    on which method fills any gap.
    """
    rows = df_train_only.iloc[row_indices]
    temp = rows['AirTC_18m'].mean()
    ws   = rows['WS_ms_18m_Avg'].mean()
    if temp >= _TEMP_VERY_HIGH_THRESH: return 'extreme_high_temp'
    if temp >= _TEMP_HIGH_THRESH:      return 'high_temp'
    if temp <= _TEMP_LOW_THRESH:       return 'low_temp'
    if ws   <= _WIND_LOW_THRESH:       return 'stagnant_wind'
    return 'normal'


def classify_boundary(row_indices, df_corr_local):
    """
    Classify the trend across a gap using only the nearest observed
    (non-NaN) points on either side, read from the corrupted series
    so no unobservable value is used.
    """
    total     = len(df_corr_local)
    first_row = min(row_indices); last_row = max(row_indices)
    airtc     = df_corr_local['AirTC_18m'].values

    pre_val = np.nan
    i = first_row - 1
    while i >= 0:
        if not np.isnan(airtc[i]):
            pre_val = airtc[i]; break
        i -= 1

    post_val = np.nan
    j = last_row + 1
    while j < total:
        if not np.isnan(airtc[j]):
            post_val = airtc[j]; break
        j += 1

    if np.isnan(pre_val) or np.isnan(post_val): return 'stable'
    pre_ext  = (pre_val  >= _TEMP_HIGH_THRESH) or (pre_val  <= _TEMP_LOW_THRESH)
    post_ext = (post_val >= _TEMP_HIGH_THRESH) or (post_val <= _TEMP_LOW_THRESH)
    if pre_ext != post_ext: return 'crossing'
    if abs(post_val - pre_val) < 0.5 * _TEMP_STD: return 'stable'
    return 'rising' if post_val > pre_val else 'falling'


def classify_length(row_indices):
    """Classify a gap as short (<=3h), medium (3-12h), or long (>12h)."""
    n = len(row_indices)
    if n <= 6:  return 'short'    # up to 3 hours
    if n <= 24: return 'medium'   # 3 to 12 hours
    return 'long'                  # over 12 hours


# RECOVERY EVALUATION
def evaluate_recovery(df_interpolated, gap_row_sets, method_name, pct_label,
                       df_corr_local):
    """Compute per-gap recovery error metrics (MAE, RMSE, R2, bias) for one
    imputation method, stratified by condition, boundary type, and length."""
    records = []
    for block in gap_row_sets:
        if not block: continue
        actual = df_train_only.iloc[block]['AirTC_18m'].values
        filled = df_interpolated.iloc[block]['AirTC_18m'].values
        valid  = ~(np.isnan(actual) | np.isnan(filled))
        if valid.sum() == 0: continue
        a, f = actual[valid], filled[valid]
        ss_res = np.sum((a - f) ** 2)
        ss_tot = np.sum((a - np.mean(a)) ** 2)
        r2_val = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
        records.append({
            'Method':    method_name,
            'Missing %': pct_label,
            'Condition': classify_condition(block),
            'Boundary':  classify_boundary(block, df_corr_local),
            'Length':    classify_length(block),
            'Gap_size':  len(block),
            'MAE':       round(mean_absolute_error(a, f), 3),
            'RMSE':      round(float(np.sqrt(mean_squared_error(a, f))), 3),
            'R2':        round(r2_val, 4) if not np.isnan(r2_val) else np.nan,
            'Bias':      round(float(np.mean(f - a)), 3),
        })
    return records


# RUN METHOD
def run_method(method_name, df_train_interp, pct_label, pct_tag):
    """Train/evaluate XGBoost across all seeds for one imputation method and
    missing-data level, then save its result plots and summary metrics."""
    df_full = pd.concat([df_train_interp, df_test_only])
    df_full = df_full.reset_index().rename(columns={"index": "TIMESTAMP"})
    tag     = f"{method_name.lower().replace(' ', '_')}_{pct_tag}"

    seed_maes, seed_rmses, seed_r2s, seed_biases = [], [], [], []
    last_r = None
    for seed in SEEDS:
        r = xgboost_model.train_and_evaluate(df_full, block_seed=seed, test_start=TEST_START)
        seed_maes.append(r["mae"]); seed_rmses.append(r["rmse"])
        seed_r2s.append(r["r2"]);   seed_biases.append(r["bias"])
        last_r = r

    mean_mae  = float(np.mean(seed_maes));  std_mae  = float(np.std(seed_maes))
    mean_rmse = float(np.mean(seed_rmses)); std_rmse = float(np.std(seed_rmses))
    mean_r2   = float(np.mean(seed_r2s));   std_r2   = float(np.std(seed_r2s))
    mean_bias = float(np.mean(seed_biases));std_bias = float(np.std(seed_biases))

    last_r.update({
        "mae": mean_mae, "rmse": mean_rmse, "r2": mean_r2, "bias": mean_bias,
        "mae_std": std_mae, "rmse_std": std_rmse,
        "r2_std": std_r2, "bias_std": std_bias,
    })

    xgboost_model.plot_results(last_r,
        title=f"{method_name} — {pct_label} missing ({len(SEEDS)} seeds)",
        out_path=os.path.join(RES, f"{tag}_results.png"))
    xgboost_model.plot_residuals(last_r, f"{method_name} {pct_label}", tag)
    xgboost_model.plot_boosting_curve(last_r, f"{method_name} {pct_label}", tag)

    all_boosting_evals.setdefault(pct_label, {})[method_name] = last_r.get('evals_result')

    model_results.append({
        "Missing %": pct_label, "Method": method_name,
        "MAE":      round(mean_mae,  3), "MAE_std":  round(std_mae,  3),
        "RMSE":     round(mean_rmse, 3), "RMSE_std": round(std_rmse, 3),
        "R2":       round(mean_r2,   4), "R2_std":   round(std_r2,   4),
        "Bias":     round(mean_bias, 3), "Bias_std": round(std_bias, 3),
    })
    print(f"    {method_name}: RMSE={mean_rmse:.3f}+/-{std_rmse:.3f} "
          f"over {len(SEEDS)} seeds")



ADAPTIVE_DIR   = os.path.join(ROOT, "adaptive_imputation")
os.makedirs(ADAPTIVE_DIR, exist_ok=True)

NEIGHBOR_WINDOW    = 12   # steps (~6h either side, 30-min sampling)
MIN_NEIGHBOR_COUNT = 4

CV_FOLDS            = 5
CV_MIN_CAT_SAMPLES  = 8   # min out-of-fold gaps in a category before
                          # trusting an empirical (rather than static) pick
CV_CANDIDATES       = ['Linear', 'Spline', 'Diagonal Climatology']


def _neighbor_density(df_corr_local, block):
    """Count real (non-NaN) AirTC_18m observations in a window around
    a gap block. Low density means curve-based methods are unreliable
    because there isn't enough signal nearby to fit a sensible curve
    or look up a climatological average."""
    total = len(df_corr_local)
    lo = max(0, min(block) - NEIGHBOR_WINDOW)
    hi = min(total, max(block) + NEIGHBOR_WINDOW + 1)
    window_vals = df_corr_local.iloc[lo:hi]['AirTC_18m']
    return int(window_vals.notna().sum())


def _static_fallback_rule(length_cls, boundary_cls):
    """The original, non-data-driven, physically-motivated rule
    (Spline for sharp transitions, Diagonal Climatology for long gaps
    since Linear/Spline flatten the diurnal cycle, Linear otherwise).
    Used only when a category doesn't have enough out-of-fold gaps to
    reliably estimate a per-category winner via cross-validation —
    never tuned on this dataset's own true values."""
    if boundary_cls == 'crossing':
        return 'Spline'
    if length_cls == 'long':
        return 'Diagonal Climatology'
    if length_cls == 'medium' and boundary_cls in ('rising', 'falling'):
        return 'Spline'
    return 'Linear'


def _assign_gap_folds(n_gaps, n_folds=CV_FOLDS, seed=42):
    """Deterministic, shuffled fold assignment. A gap's fold depends
    only on a fixed random split, never on its own recovery error."""
    rng   = np.random.RandomState(seed)
    order = np.arange(n_gaps)
    rng.shuffle(order)
    fold_of = np.empty(n_gaps, dtype=int)
    for rank, gi in enumerate(order):
        fold_of[gi] = rank % n_folds
    return fold_of


def build_cv_rule_table(gap_row_sets_local, df_train_corr_local,
                         method_dfs_local, n_folds=CV_FOLDS,
                         min_samples=CV_MIN_CAT_SAMPLES):
    """
    Cross-validated (length_class, boundary_class) -> method rule
    table. For every fold f, a category's chosen method is whichever
    of Linear / Spline / Diagonal Climatology has the lowest MEAN
    recovery RMSE computed ONLY over gaps in the *other* folds — i.e.
    gaps the rule will never actually be applied to. Gaps that would
    fall back to LOCF via the density rule are excluded from this
    entirely, since that rule always wins for them regardless of
    category/fold and isn't data-driven.

    Returns
    -------
    fold_of    : np.ndarray, gap index -> fold id (density-fallback
                 gaps still get a fold id but it's unused)
    rule_table : dict {fold_id: {(length_cls, boundary_cls): method}}
    """
    n_gaps  = len(gap_row_sets_local)
    fold_of = _assign_gap_folds(n_gaps, n_folds)

    gap_info = []
    for gi, block in enumerate(gap_row_sets_local):
        if not block:
            gap_info.append(None); continue
        density = _neighbor_density(df_train_corr_local, block)
        if density < MIN_NEIGHBOR_COUNT:
            gap_info.append(None)   # goes to LOCF regardless; not data-driven
            continue
        length_cls   = classify_length(block)
        boundary_cls = classify_boundary(block, df_train_corr_local)
        actual = df_train_only.iloc[block]['AirTC_18m'].values
        errs = {}
        for m in CV_CANDIDATES:
            filled = method_dfs_local[m].iloc[block]['AirTC_18m'].values
            valid  = ~(np.isnan(actual) | np.isnan(filled))
            errs[m] = (float(np.sqrt(np.mean((filled[valid] - actual[valid]) ** 2)))
                       if valid.sum() > 0 else np.nan)
        gap_info.append({'category': (length_cls, boundary_cls),
                          'fold': fold_of[gi], 'errs': errs})

    categories = sorted({g['category'] for g in gap_info if g is not None})
    rule_table = {f: {} for f in range(n_folds)}
    for f in range(n_folds):
        for cat in categories:
            train_errs = [g['errs'] for g in gap_info
                          if g is not None and g['category'] == cat
                          and g['fold'] != f]
            if len(train_errs) < min_samples:
                rule_table[f][cat] = _static_fallback_rule(*cat)
                continue
            mean_err = {m: np.nanmean([e[m] for e in train_errs])
                        for m in CV_CANDIDATES}
            mean_err_safe = {m: (v if not np.isnan(v) else np.inf)
                              for m, v in mean_err.items()}
            if all(np.isinf(v) for v in mean_err_safe.values()):
                rule_table[f][cat] = _static_fallback_rule(*cat)
            else:
                rule_table[f][cat] = min(mean_err_safe, key=mean_err_safe.get)

    return fold_of, rule_table


def classify_gap_adaptive(gi, block, df_corr_local, fold_of, rule_table):
    """
    Method selector for a single gap block. Density fallback (Rule 0)
    is fixed and leak-free. Otherwise the method comes from the
    cross-validated rule_table for this gap's fold, so the pick was
    always learned from OTHER gaps, never from this one's own error.

    Returns one of: 'Linear', 'Spline', 'Diagonal Climatology', 'LOCF'
    """
    density = _neighbor_density(df_corr_local, block)
    if density < MIN_NEIGHBOR_COUNT:
        return 'LOCF'

    length_cls   = classify_length(block)
    boundary_cls = classify_boundary(block, df_corr_local)
    cat          = (length_cls, boundary_cls)
    fold         = fold_of[gi]
    return rule_table[fold].get(cat, _static_fallback_rule(*cat))


def adaptive_imputation_interpolate(df_train_corr_local, gap_row_sets_local,
                                 method_dfs_local):
    """
    Build the adaptively-imputed series by stitching together, for each
    gap block, the column of whichever method (Linear / Spline /
    Diagonal Climatology / LOCF) was selected by classify_gap_adaptive(),
    using a cross-validated rule table so no gap's fill choice was
    informed by that gap's own true value.

    Parameters
    ----------
    df_train_corr_local : DataFrame with NaN gaps (AirTC_18m)
    gap_row_sets_local   : list of lists of row positions, one per gap
    method_dfs_local     : dict {method_name: fully-interpolated DataFrame}

    Returns
    -------
    df_adaptive   : DataFrame identical in shape to df_train_corr_local
                    but with AirTC_18m gaps filled adaptively
    choice_log_df : DataFrame logging the decision made per gap, for
                    transparency / plotting / auditing
    """
    fold_of, rule_table = build_cv_rule_table(
        gap_row_sets_local, df_train_corr_local, method_dfs_local)

    df_adaptive = df_train_corr_local.copy()
    col_loc     = df_adaptive.columns.get_loc('AirTC_18m')
    choice_log  = []

    for gi, block in enumerate(gap_row_sets_local):
        if not block:
            continue
        chosen_method = classify_gap_adaptive(
            gi, block, df_train_corr_local, fold_of, rule_table)
        fill_values   = method_dfs_local[chosen_method]\
            .iloc[block]['AirTC_18m'].values
        df_adaptive.iloc[block, col_loc] = fill_values

        choice_log.append({
            'Gap_num':        gi + 1,
            'Start_row':      min(block),
            'End_row':        max(block),
            'Gap_size':       len(block),
            'Length_class':   classify_length(block),
            'Boundary_class': classify_boundary(block, df_train_corr_local),
            'Condition':      classify_condition(block),
            'Neighbor_count': _neighbor_density(df_train_corr_local, block),
            'CV_fold':        int(fold_of[gi]),
            'Chosen_method':  chosen_method,
        })

    return df_adaptive, pd.DataFrame(choice_log)


def plot_adaptive_vs_original(df_orig, df_adaptive, choice_log_df,
                               pct_label, pct_tag):
    """Full-series plot: Original (clean) vs Adaptively-imputed series.
    Each imputed gap segment is coloured by the interpolation method the
    adaptive pipeline selected for it, so the reader can see WHICH
    method was used WHERE."""
    fi  = df_orig.index; tot = len(fi); cs = tot // 4
    fig, axes = plt.subplots(4, 1, figsize=(26, 25))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.94, bottom=0.075,
                        hspace=0.35)
    fig.suptitle(
        f'Adaptive Imputation vs Original — AirTC_18m '
        f'({pct_label} missing)',
        fontsize=SUP_FS, fontweight='bold', y=0.985)

    for i, ax in enumerate(axes):
        s = i * cs; e = (i + 1) * cs if i < 3 else tot
        st, en = fi[s], fi[e - 1]

        orig_seg = df_orig.loc[st:en, 'AirTC_18m']
        adap_seg = df_adaptive.loc[st:en, 'AirTC_18m']

        ax.plot(orig_seg.index, orig_seg.values, color='#1565C0',
                lw=1.7, ls='--', alpha=0.55,
                label='Original' if i == 0 else '_nolegend_', zorder=2)
        ax.plot(adap_seg.index, adap_seg.values, color='#212121',
                lw=1.5, alpha=0.8,
                label='Adaptive (unfilled)' if i == 0 else '_nolegend_',
                zorder=3)

        for _, row in choice_log_df.iterrows():
            s_row, e_row = int(row['Start_row']), int(row['End_row'])
            lo = max(0, s_row - 1); hi = min(tot - 1, e_row + 1)
            block_idx = df_orig.index[lo:hi + 1]
            seg_ts    = block_idx[(block_idx >= st) & (block_idx <= en)]
            if len(seg_ts) == 0:
                continue
            clr = METHOD_COLORS[row['Chosen_method']]
            ax.plot(seg_ts, df_adaptive.loc[seg_ts, 'AirTC_18m'].values,
                    color=clr, lw=2.7, alpha=0.95, zorder=4,
                    label='_nolegend_')

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        ax.tick_params(axis='x', rotation=0, labelsize=TICK_FS, colors='black')
        ax.tick_params(axis='y', labelsize=TICK_FS, colors='black')
        ax.set_ylabel('AirTC_18m (°C)', fontsize=LABEL_FS, fontweight='bold')
        ax.set_title(f'Part {i+1}: {st.strftime("%d %b")} to '
                     f'{en.strftime("%d %b %Y")}',
                     fontsize=TITLE_FS, fontweight='bold')
        ax.grid(True, alpha=0.2)
        for spine in ax.spines.values():
            spine.set_edgecolor('black'); spine.set_linewidth(1.6)

    legend_elements = [
        plt.Line2D([0], [0], color='#1565C0', lw=2.2, ls='--',
                   alpha=0.6, label='Original'),
        plt.Line2D([0], [0], color='#212121', lw=2.2, alpha=0.8,
                   label='Adaptive (unfilled)'),
    ] + [
        plt.Line2D([0], [0], color=METHOD_COLORS[m], lw=3.0,
                   label=f'Gap filled by: {m}')
        for m in METHODS
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               bbox_to_anchor=(0.5, 0.005), ncol=3, fontsize=LEGEND_FS,
               framealpha=1.0, edgecolor='black',
               prop={'weight': 'bold', 'size': LEGEND_FS})
    out_path = os.path.join(ADAPTIVE_DIR,
                             f'adaptive_vs_original_{pct_tag}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_adaptive_zoom(df_orig, df_adaptive, choice_log_df,
                        pct_label, pct_tag):
    """Zoomed-in view over a short window chosen automatically to
    contain the greatest variety of adaptively-selected methods, so
    each method's visual signature (colour) is clearly distinguishable
    side by side."""
    if choice_log_df.empty:
        print("  No gaps to zoom into — skipping adaptive zoom plot.")
        return

    tot    = len(df_orig)
    window = min(400, tot)
    step   = max(1, window // 4)
    best_start, best_variety = 0, -1
    for w_start in range(0, max(1, tot - window) + 1, step):
        w_end = w_start + window
        methods_here = choice_log_df[
            (choice_log_df['Start_row'] < w_end) &
            (choice_log_df['End_row']   >= w_start)
        ]['Chosen_method'].nunique()
        if methods_here > best_variety:
            best_variety, best_start = methods_here, w_start

    w_end = min(tot, best_start + window)
    st, en = df_orig.index[best_start], df_orig.index[w_end - 1]

    fig, ax = plt.subplots(figsize=(22, 12))
    fig.subplots_adjust(left=0.045, right=0.99, top=0.98, bottom=0.22)
    orig_seg = df_orig.loc[st:en, 'AirTC_18m']
    adap_seg = df_adaptive.loc[st:en, 'AirTC_18m']

    ax.plot(orig_seg.index, orig_seg.values, color='#1565C0', lw=2.6,
            ls='--', alpha=0.6, label='Original (clean)', zorder=2)
    ax.plot(adap_seg.index, adap_seg.values, color='#212121', lw=2.0,
            alpha=0.6, label='Adaptive (unfilled)', zorder=3)

    used_methods = set()
    for _, row in choice_log_df.iterrows():
        s_row, e_row = int(row['Start_row']), int(row['End_row'])
        if e_row < best_start or s_row >= w_end:
            continue
        lo = max(0, s_row - 1); hi = min(tot - 1, e_row + 1)
        block_idx = df_orig.index[lo:hi + 1]
        seg_ts    = block_idx[(block_idx >= st) & (block_idx <= en)]
        if len(seg_ts) == 0:
            continue
        clr = METHOD_COLORS[row['Chosen_method']]
        ax.plot(seg_ts, df_adaptive.loc[seg_ts, 'AirTC_18m'].values,
                color=clr, lw=3.6, alpha=0.95, zorder=4,
                marker='o', markersize=3)
        used_methods.add(row['Chosen_method'])
        ax.axvspan(seg_ts[0], seg_ts[-1], color=clr, alpha=0.08, zorder=0)

    ZOOM_LABEL_FS = 22
    ZOOM_TICK_FS  = 19   # slightly smaller than ZOOM_LABEL_FS, as requested
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %H:%M'))
    ax.tick_params(axis='x', rotation=0, labelsize=ZOOM_TICK_FS, colors='black')
    ax.tick_params(axis='y', labelsize=ZOOM_TICK_FS, colors='black')
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')
    ax.set_ylabel('AirTC_18m (°C)', fontsize=ZOOM_LABEL_FS, fontweight='bold',
                  color='black')
    ax.grid(True, alpha=0.25)
    for spine in ax.spines.values():
        spine.set_edgecolor('black'); spine.set_linewidth(1.8)

    legend_elements = [
        plt.Line2D([0], [0], color='#1565C0', lw=2.4, ls='--',
                   alpha=0.6, label='Original (clean)'),
        plt.Line2D([0], [0], color='#212121', lw=2.4, alpha=0.6,
                   label='Adaptive (unfilled)'),
    ] + [
        plt.Line2D([0], [0], color=METHOD_COLORS[m], lw=3.4,
                   marker='o', label=m)
        for m in METHODS if m in used_methods
    ]
    ZOOM_LEGEND_FS = 17
    ncol = min(3, len(legend_elements))
    fig.legend(handles=legend_elements, fontsize=ZOOM_LEGEND_FS,
               framealpha=1.0, edgecolor='black',
               loc='lower center', bbox_to_anchor=(0.5, 0.015), ncol=ncol,
               prop={'weight': 'bold', 'size': ZOOM_LEGEND_FS})
    out_path = os.path.join(ADAPTIVE_DIR, f'adaptive_zoom_{pct_tag}.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved: {out_path}")


def run_adaptive_method(df_adaptive, pct_label, pct_tag):
    """
    Train & evaluate the existing downstream XGBoost pipeline on the
    adaptively-imputed data, using the exact same harness as
    run_method() above (multi-seed averaging, chronological 80/10/10
    split, same plot outputs), and additionally reports MAPE.

    NOTE on MAPE: AirTC_18m (temperature, °C) can pass through values
    near zero, which makes standard MAPE numerically unstable (division
    by ~0). A small epsilon guard is used in the denominator; MAPE
    should therefore be read as indicative rather than authoritative
    for this particular variable.
    """
    df_full = pd.concat([df_adaptive, df_test_only])
    df_full = df_full.reset_index().rename(columns={"index": "TIMESTAMP"})
    tag     = f"adaptive_imputation_{pct_tag}"
    EPS     = 1e-3

    seed_maes, seed_rmses, seed_r2s, seed_biases, seed_mapes = \
        [], [], [], [], []
    last_r = None
    for seed in SEEDS:
        r = xgboost_model.train_and_evaluate(df_full, block_seed=seed, test_start=TEST_START)
        seed_maes.append(r["mae"]);  seed_rmses.append(r["rmse"])
        seed_r2s.append(r["r2"]);    seed_biases.append(r["bias"])
        mape = float(np.mean(
            np.abs(r['y_test'] - r['y_pred']) /
            np.maximum(np.abs(r['y_test']), EPS))) * 100
        seed_mapes.append(mape)
        last_r = r

    mean_mae  = float(np.mean(seed_maes));  std_mae  = float(np.std(seed_maes))
    mean_rmse = float(np.mean(seed_rmses)); std_rmse = float(np.std(seed_rmses))
    mean_r2   = float(np.mean(seed_r2s));   std_r2   = float(np.std(seed_r2s))
    mean_bias = float(np.mean(seed_biases));std_bias = float(np.std(seed_biases))
    mean_mape = float(np.mean(seed_mapes)); std_mape = float(np.std(seed_mapes))

    last_r.update({
        "mae": mean_mae, "rmse": mean_rmse, "r2": mean_r2, "bias": mean_bias,
        "mae_std": std_mae, "rmse_std": std_rmse,
        "r2_std": std_r2, "bias_std": std_bias,
    })

    xgboost_model.plot_results(last_r,
        title=f"Adaptive Imputation — {pct_label} missing ({len(SEEDS)} seeds)",
        out_path=os.path.join(ADAPTIVE_DIR, f"{tag}_results.png"))
    xgboost_model.plot_residuals(last_r, f"Adaptive Imputation {pct_label}", tag)
    xgboost_model.plot_boosting_curve(last_r, f"Adaptive Imputation {pct_label}", tag)

    all_boosting_evals.setdefault(pct_label, {})['Adaptive Imputation'] = last_r.get('evals_result')

    adaptive_model_results.append({
        "Missing %": pct_label, "Method": "Adaptive Imputation",
        "MAE":      round(mean_mae,  3), "MAE_std":  round(std_mae,  3),
        "RMSE":     round(mean_rmse, 3), "RMSE_std": round(std_rmse, 3),
        "R2":       round(mean_r2,   4), "R2_std":   round(std_r2,   4),
        "Bias":     round(mean_bias, 3), "Bias_std": round(std_bias, 3),
        "MAPE_%":   round(mean_mape, 3), "MAPE_std": round(std_mape, 3),
    })
    print(f"    Adaptive Imputation: RMSE={mean_rmse:.3f}+/-{std_rmse:.3f}  "
          f"MAPE={mean_mape:.2f}%+/-{std_mape:.2f}% over {len(SEEDS)} seeds")


# GAP LENGTH ANALYSIS
def plot_gap_length_analysis(all_gap_row_sets_local):
    """Produce the full set of gap-visualisation and method-comparison figures
    (missing-region timelines, error histograms, and summary tables)."""
    colors_pct = ['#1565C0', '#E53935', '#2E7D32', '#F57C00']
    pct_labels = list(all_gap_row_sets_local.keys())
    all_lengths = {p: [len(b) for b in all_gap_row_sets_local[p] if len(b) > 0]
                   for p in pct_labels}

    # Histogram per pct level
    fig, axes = plt.subplots(2, 2, figsize=(22, 15), constrained_layout=True)
    axes = axes.flatten()
    for ax, (pct, lengths) in zip(axes, all_lengths.items()):
        if not lengths: continue
        arr  = np.array(lengths)
        bins = np.arange(1, min(arr.max() + 2, 55))
        idx  = pct_labels.index(pct)
        ax.hist(arr, bins=bins, color=colors_pct[idx], alpha=0.75,
                edgecolor='white', lw=1.9)
        ax.axvline(np.mean(arr), color='black', lw=2.4, ls='--',
                   label=f'Mean={np.mean(arr):.1f}')
        ax.axvline(np.median(arr), color='red', lw=2.4, ls='-.',
                   label=f'Median={np.median(arr):.1f}')
        ax.set_xlabel('Gap Length (steps, 1 step = 30 min)', fontsize=LABEL_FS)
        ax.set_ylabel('Count', fontsize=LABEL_FS)
        ax.set_title(f'{pct} missing — {len(lengths)} gaps  |  '
                     f'Short={sum(1 for l in lengths if l<=6)}  '
                     f'Medium={sum(1 for l in lengths if 6<l<=24)}  '
                     f'Long={sum(1 for l in lengths if l>24)}',
                     fontsize=TITLE_FS, fontweight='bold')
        ax.tick_params(axis='both', labelsize=TICK_FS)
        ax.legend(fontsize=LEGEND_FS, loc='upper right', framealpha=1.0, edgecolor='black')
        ax.grid(True, alpha=0.25)
    plt.savefig(os.path.join(GAP, 'gap_length_histogram.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: gap_length_histogram.png")

    # Probability distribution (empirical PMF)
    fig, ax = plt.subplots(figsize=(18, 9), constrained_layout=True)
    for i, (pct, lengths) in enumerate(all_lengths.items()):
        if not lengths: continue
        arr          = np.array(lengths, dtype=float)
        unique, cnts = np.unique(arr, return_counts=True)
        ax.plot(unique, cnts / cnts.sum(), color=colors_pct[i], lw=2.4,
                marker='o', ms=4, alpha=0.8, label=f'{pct} missing')
    ax.set_xlabel('Gap Length (steps)', fontsize=LABEL_FS, fontweight='bold')
    ax.set_ylabel('Probability', fontsize=LABEL_FS, fontweight='bold', color='black')
    ax.tick_params(axis='both', labelsize=TICK_FS)
    ax.legend(fontsize=LEGEND_FS, loc='upper right', framealpha=1.0, edgecolor='black')
    ax.grid(True, alpha=0.25)
    for spine in ax.spines.values():
        spine.set_edgecolor('black'); spine.set_linewidth(2.0)
    plt.savefig(os.path.join(GAP, 'gap_length_probability.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: gap_length_probability.png")

    # CDF
    fig, ax = plt.subplots(figsize=(18, 9), constrained_layout=True)
    for i, (pct, lengths) in enumerate(all_lengths.items()):
        if not lengths: continue
        arr = np.sort(np.array(lengths, dtype=float))
        ax.plot(arr, np.arange(1, len(arr)+1) / len(arr),
                color=colors_pct[i], lw=2.9, label=f'{pct} missing')
    ax.axvline(6,  color='grey', lw=1.9, ls='--', alpha=0.7,
               label='Short/Medium boundary (6 steps = 3h)')
    ax.axvline(24, color='grey', lw=1.9, ls=':',  alpha=0.7,
               label='Medium/Long boundary (24 steps = 12h)')
    ax.set_xlabel('Gap Length (steps)', fontsize=LABEL_FS, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=LABEL_FS, fontweight='bold', color='black')
    ax.tick_params(axis='both', labelsize=TICK_FS)
    ax.legend(fontsize=LEGEND_FS, loc='lower right', framealpha=1.0, edgecolor='black'); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    for spine in ax.spines.values():
        spine.set_edgecolor('black'); spine.set_linewidth(2.0)
    plt.savefig(os.path.join(GAP, 'gap_length_cdf.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: gap_length_cdf.png")

    # Stacked bar
    shorts  = [sum(1 for l in all_lengths[p] if l<=6)    for p in pct_labels]
    mediums = [sum(1 for l in all_lengths[p] if 6<l<=24) for p in pct_labels]
    longs   = [sum(1 for l in all_lengths[p] if l>24)    for p in pct_labels]
    x = np.arange(len(pct_labels))
    fig, ax = plt.subplots(figsize=(16, 9), constrained_layout=True)
    ax.bar(x, shorts,  color='#2E7D32', alpha=0.85,
           label='Short (<=6 steps, <=3h)')
    ax.bar(x, mediums, bottom=shorts,
           color='#F57C00', alpha=0.85, label='Medium (7-24 steps, 3-12h)')
    ax.bar(x, longs,
           bottom=[s + m for s, m in zip(shorts, mediums)],
           color='#C62828', alpha=0.85, label='Long (>24 steps, >12h)')
    ax.set_xticks(x); ax.set_xticklabels(pct_labels, fontsize=TICK_FS, fontweight='bold', color='black')
    ax.set_xlabel('Missing Percentage Level', fontsize=LABEL_FS, fontweight='bold', color='black')
    ax.set_ylabel('Number of Gaps', fontsize=LABEL_FS, fontweight='bold', color='black')
    ax.legend(fontsize=LEGEND_FS, loc='upper center',
              bbox_to_anchor=(0.5, 1.18), ncol=3, framealpha=1.0,
              edgecolor='black')
    ax.grid(True, axis='y', alpha=0.25)
    ax.tick_params(axis='both', which='major', colors='black',
                   labelcolor='black', labelsize=TICK_FS, width=2.0)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color('black'); lbl.set_fontweight('bold')
    for spine in ax.spines.values():
        spine.set_edgecolor('black'); spine.set_linewidth(2.0)
    for i, (s, m, l) in enumerate(zip(shorts, mediums, longs)):
        if s > 0: ax.text(i, s/2,       str(s), ha='center', va='center',
                          fontsize=ANNOT_FS, fontweight='bold', color='white')
        if m > 0: ax.text(i, s + m/2,   str(m), ha='center', va='center',
                          fontsize=ANNOT_FS, fontweight='bold', color='white')
        if l > 0: ax.text(i, s + m + l/2, str(l), ha='center', va='center',
                          fontsize=ANNOT_FS, fontweight='bold', color='white')
    plt.savefig(os.path.join(GAP, 'gap_length_stacked_bar.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: gap_length_stacked_bar.png")

    # Boxplot
    fig, ax = plt.subplots(figsize=(16, 8), constrained_layout=True)
    bp = ax.boxplot([np.array(all_lengths[p]) for p in pct_labels],
                    patch_artist=True, notch=False,
                    medianprops=dict(color='black', lw=2.9))
    for patch, color in zip(bp['boxes'], colors_pct):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_xticklabels(pct_labels, fontsize=TICK_FS)
    ax.set_xlabel('Missing Percentage Level', fontsize=LABEL_FS, fontweight='bold', color='black')
    ax.set_ylabel('Gap Length (steps)', fontsize=LABEL_FS, fontweight='bold')
    ax.tick_params(axis='y', labelsize=TICK_FS)
    ax.grid(True, axis='y', alpha=0.25)
    for spine in ax.spines.values():
        spine.set_edgecolor('black'); spine.set_linewidth(2.0)
    plt.savefig(os.path.join(GAP, 'gap_length_boxplot.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: gap_length_boxplot.png")


# MAIN LOOP
for count in range(1, 5):
    df_train_corr, actual_pct, gap_row_sets = make_corrupted(count)
    pct_label = f"{round(actual_pct)}%"
    pct_tag   = pct_label.replace('%', 'pct')
    all_gap_row_sets[pct_label] = gap_row_sets
    print(f"\n=== {pct_label} missing ({len(gap_row_sets)} gap blocks) ===")

    lengths = [len(b) for b in gap_row_sets]
    if lengths:
        print(f"  Lengths: min={min(lengths)} max={max(lengths)} "
              f"mean={np.mean(lengths):.1f}  "
              f"short={sum(1 for l in lengths if l<=6)}  "
              f"medium={sum(1 for l in lengths if 6<l<=24)}  "
              f"long={sum(1 for l in lengths if l>24)}")

    # Gap visualisation
    fi = df_train_only.index; tot = len(fi); cs = tot // 4
    fig, axes = plt.subplots(4, 1, figsize=(24, 21))
    fig.subplots_adjust(left=0.055, right=0.99, top=0.95, bottom=0.035,
                        hspace=0.35)
    for i, ax in enumerate(axes):
        s = i * cs; e = (i + 1) * cs if i < 3 else tot
        st, en = fi[s], fi[e - 1]
        oc = df_train_only.loc[st:en, 'AirTC_18m']
        cc = df_train_corr.loc[st:en, 'AirTC_18m']
        ax.plot(oc.index, oc.values, color='steelblue', lw=1.9,
                alpha=0.4, ls='--', label='Original')
        ax.plot(cc.index, cc.values, color='steelblue', lw=1.9,
                label='With gaps')
        for ts in cc[cc.isna()].index:
            ax.axvspan(ts - pd.Timedelta(minutes=15),
                       ts + pd.Timedelta(minutes=15),
                       color='red', alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        ax.tick_params(axis='x', rotation=0, labelsize=TICK_FS)
        ax.tick_params(axis='y', labelsize=TICK_FS)
        ax.set_ylabel('AirTC_18m (C)', fontsize=LABEL_FS)
        ax.set_title(f'Part {i+1}: {st.strftime("%d %b")} to '
                     f'{en.strftime("%d %b %Y")}', fontsize=TITLE_FS)
    legend_elements = [
        plt.Line2D([0], [0], color='steelblue', lw=1.9, alpha=0.4,
                   ls='--', label='Original'),
        plt.Line2D([0], [0], color='steelblue', lw=1.9, label='With gaps'),
        Patch(facecolor='red', alpha=0.3, label='Missing region'),
    ]
    fig.legend(handles=legend_elements, fontsize=LEGEND_FS,
               loc='upper center', bbox_to_anchor=(0.5, 0.995),
               ncol=len(legend_elements), framealpha=1.0, edgecolor='black',
               prop={'weight': 'bold', 'size': LEGEND_FS})
    plt.savefig(os.path.join(GAP, f'missing_gaps_{pct_tag}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Interpolate — only AirTC_18m
    print("  Linear...")
    df_lin = df_train_corr.copy()
    df_lin['AirTC_18m'] = im.linear_inter(df_lin, 'AirTC_18m')['AirTC_18m']

    print("  Spline...")
    df_spl = df_train_corr.copy()
    df_spl['AirTC_18m'] = im.spline_inter(df_spl, 'AirTC_18m')['AirTC_18m']

    print("  Diagonal Climatology...")
    df_dc = df_train_corr.copy()
    temp_series = df_train_corr[['AirTC_18m']].copy()
    temp_series.index = df_train_corr.index
    df_dc['AirTC_18m'] = im.diag_climatology_practical(
        temp_series)['AirTC_18m']

    print("  LOCF...")
    df_locf = df_train_corr.copy()
    df_locf['AirTC_18m'] = im.locf_interpolation(
        df_locf, 'AirTC_18m')['AirTC_18m']

    all_method_dfs[('Linear',               pct_label)] = df_lin
    all_method_dfs[('Spline',               pct_label)] = df_spl
    all_method_dfs[('Diagonal Climatology', pct_label)] = df_dc
    all_method_dfs[('LOCF',                 pct_label)] = df_locf

    METHOD_DFS = {'Linear': df_lin, 'Spline': df_spl,
                  'Diagonal Climatology': df_dc, 'LOCF': df_locf}

    # Overlay plot
    METHOD_INTERPS = [
        ('Original',             df_train_only, '#1565C0', '-',  1.6, 0.9),
        ('With gaps',            df_train_corr, '#888888', '-',  1.0, 0.6),
        ('Linear',               df_lin,        METHOD_COLORS['Linear'],               '--', 1.4, 0.85),
        ('Spline',               df_spl,        METHOD_COLORS['Spline'],               '--', 1.4, 0.85),
        ('Diagonal Climatology', df_dc,         METHOD_COLORS['Diagonal Climatology'], '--', 1.4, 0.85),
        ('LOCF',                 df_locf,       METHOD_COLORS['LOCF'],                 '--', 1.4, 0.85),
    ]
    fi2 = df_train_only.index; tot2 = len(fi2); cs2 = tot2 // 4
    fig, axes = plt.subplots(4, 1, figsize=(26, 27))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.96, bottom=0.08,
                        hspace=0.35)
    for i, ax in enumerate(axes):
        s = i * cs2; e = (i + 1) * cs2 if i < 3 else tot2
        st, en = fi2[s], fi2[e - 1]
        gap_col = df_train_corr.loc[st:en, 'AirTC_18m']
        for ts in gap_col[gap_col.isna()].index:
            ax.axvspan(ts - pd.Timedelta(minutes=15),
                       ts + pd.Timedelta(minutes=15),
                       color='red', alpha=0.15, zorder=0)
        for label, df_src, clr, ls, lw, alpha in METHOD_INTERPS:
            series = df_src.loc[st:en, 'AirTC_18m']
            ax.plot(series.index, series.values, color=clr, ls=ls, lw=lw,
                    alpha=alpha,
                    label=label if i == 0 else '_nolegend_')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        ax.tick_params(axis='x', rotation=0, labelsize=TICK_FS, colors='black', width=2)
        ax.tick_params(axis='y', labelsize=TICK_FS, colors='black', width=2)
        ax.set_ylabel('AirTC_18m (°C)', fontsize=LABEL_FS, fontweight='bold', color='black')
        ax.set_title(f'Part {i+1}: {st.strftime("%d %b")} to '
                     f'{en.strftime("%d %b %Y")}',
                     fontsize=TITLE_FS, fontweight='bold', color='black')
        ax.grid(True, alpha=0.2)
        for spine in ax.spines.values():
            spine.set_edgecolor('black'); spine.set_linewidth(1.8)
    legend_elements = [
        plt.Line2D([0], [0], color='#1565C0', lw=1.9, ls='-',
                   label='Original'),
        plt.Line2D([0], [0], color='#888888', lw=1.9, ls='-',
                   label='With gaps'),
        plt.Line2D([0], [0], color=METHOD_COLORS['Linear'],
                   lw=1.9, ls='--', label='Linear'),
        plt.Line2D([0], [0], color=METHOD_COLORS['Spline'],
                   lw=1.9, ls='--', label='Spline'),
        plt.Line2D([0], [0], color=METHOD_COLORS['Diagonal Climatology'],
                   lw=1.9, ls='--', label='Diagonal Climatology'),
        plt.Line2D([0], [0], color=METHOD_COLORS['LOCF'],
                   lw=1.9, ls='--', label='LOCF'),
        Patch(facecolor='red', alpha=0.15, label='Missing region'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               bbox_to_anchor=(0.5, 0.005), ncol=4, fontsize=LEGEND_FS,
               framealpha=1.0, edgecolor='black',
               prop={'weight': 'bold', 'size': LEGEND_FS})
    plt.savefig(os.path.join(GAP, f'interpolation_overlay_{pct_tag}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Overlay plot saved for {pct_label}")

    # Gap-indexed plot
    gap_nums = []; actual_means = []
    method_means = {m: [] for m in METHOD_DFS}
    for gnum, block in enumerate(gap_row_sets, start=1):
        av = df_train_only.iloc[block]['AirTC_18m'].values
        if len(av) == 0: continue
        gap_nums.append(gnum)
        actual_means.append(float(np.mean(av)))
        for mname, mdf in METHOD_DFS.items():
            method_means[mname].append(
                float(np.nanmean(mdf.iloc[block]['AirTC_18m'].values)))

    fig, ax = plt.subplots(figsize=(max(20, len(gap_nums) * 0.12 + 4), 9),
                           constrained_layout=True)
    ax.plot(gap_nums, actual_means, color='black', lw=2.4, ls='-',
            marker='o', markersize=3, label='Actual', zorder=5)
    ls_cycle = ['--', '-.', ':', '--']
    for (mname, clr), ls in zip(METHOD_COLORS.items(), ls_cycle):
        ax.plot(gap_nums, method_means[mname], color=clr, lw=2.4, ls=ls,
                marker='s', markersize=2.5, alpha=0.9, label=mname)
    ax.set_xlabel('Gap Number', fontsize=LABEL_FS, fontweight='bold', color='black')
    ax.set_ylabel('Mean AirTC (C)', fontsize=LABEL_FS, fontweight='bold', color='black')
    tick_step = max(1, len(gap_nums) // 30)
    ax.set_xticks(gap_nums[::tick_step])
    ax.set_xticklabels([str(g) for g in gap_nums[::tick_step]],
                        fontsize=TICK_FS, fontweight='bold', rotation=45, color='black')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', which='major', colors='black',
                   labelcolor='black', labelsize=TICK_FS, width=2.0)
    for lbl in ax.get_yticklabels():
        lbl.set_color('black'); lbl.set_fontweight('bold')
    for spine in ax.spines.values():
        spine.set_edgecolor('black'); spine.set_linewidth(2.0)
    ax.legend(fontsize=LEGEND_FS, framealpha=1.0, edgecolor='black',
              loc='upper right', ncol=2,
              prop={'weight': 'bold', 'size': LEGEND_FS})
    plt.savefig(os.path.join(GAP, f'gap_labelled_{pct_tag}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Error histogram
    all_errors = {}
    for mname, mdf in METHOD_DFS.items():
        errors = []
        for block in gap_row_sets:
            av = df_train_only.iloc[block]['AirTC_18m'].values
            iv = mdf.iloc[block]['AirTC_18m'].values
            valid = ~(np.isnan(av) | np.isnan(iv))
            if valid.sum() > 0:
                errors.extend((iv[valid] - av[valid]).tolist())
        all_errors[mname] = errors

    all_vals = [e for errs in all_errors.values() for e in errs]
    bin_min  = np.percentile(all_vals, 1)
    bin_max  = np.percentile(all_vals, 99)
    bins     = np.linspace(bin_min, bin_max, 60)

    fig, ax = plt.subplots(figsize=(18, 10), constrained_layout=True)
    for mname, errors in all_errors.items():
        errors_arr = np.array(errors); clr = METHOD_COLORS[mname]
        kde     = gaussian_kde(errors_arr, bw_method=0.3)
        x_range = np.linspace(bin_min, bin_max, 300)
        ax.plot(x_range, kde(x_range), color=clr, lw=2.9,
                label=f'{mname} (n={len(errors)})')
        ax.axvline(np.mean(errors_arr), color=clr, lw=2.2, ls='--',
                   alpha=0.9)
    ax.axvline(0, color='black', lw=2.4, ls='-', label='Zero error')
    ax.set_xlabel('Interpolation Error (Filled - Actual) (C)',
                  fontsize=LABEL_FS, fontweight='bold')
    ax.set_ylabel('Density', fontsize=LABEL_FS, fontweight='bold', color='black')
    ax.tick_params(axis='both', labelsize=TICK_FS)
    ax.legend(fontsize=LEGEND_FS, framealpha=1.0, edgecolor='black',
              loc='upper right', prop={'weight': 'bold', 'size': LEGEND_FS})
    ax.grid(True, alpha=0.25, linestyle='--')
    for spine in ax.spines.values():
        spine.set_edgecolor('black'); spine.set_linewidth(2.0)
    plt.savefig(os.path.join(GAP, f'error_histogram_{pct_tag}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Variability + recovery evaluation
    for method_name, df_interp in METHOD_DFS.items():
        analysis_datasets.append({
            'Method':        method_name,
            'Missing %':     pct_label,
            'Temp_STD':      df_interp['AirTC_18m'].std(),
            'Temp_Diff_STD': df_interp['AirTC_18m'].diff().std(),
        })
        recs = evaluate_recovery(df_interp, gap_row_sets,
                                  method_name, pct_label, df_train_corr)
        recovery_records.extend(recs)
        print(f"    Recovery eval {method_name}: {len(recs)} gap blocks")

    # Model evaluation
    run_method('Linear',               df_lin,  pct_label, pct_tag)
    run_method('Spline',               df_spl,  pct_label, pct_tag)
    run_method('Diagonal Climatology', df_dc,   pct_label, pct_tag)
    run_method('LOCF',                 df_locf, pct_label, pct_tag)

    print("  Adaptive Imputation...")
    df_adaptive, choice_log_df = adaptive_imputation_interpolate(
        df_train_corr, gap_row_sets, METHOD_DFS)
    adaptive_choice_logs[pct_label] = choice_log_df
    all_adaptive_dfs[pct_label]     = df_adaptive
    analysis_datasets.append({
        'Method':        'Adaptive Imputation',
        'Missing %':     pct_label,
        'Temp_STD':      df_adaptive['AirTC_18m'].std(),
        'Temp_Diff_STD': df_adaptive['AirTC_18m'].diff().std(),
    })
    choice_log_df.to_csv(
        os.path.join(ADAPTIVE_DIR, f'method_choice_log_{pct_tag}.csv'),
        index=False)
    print("    Method usage:",
          choice_log_df['Chosen_method'].value_counts().to_dict())

    plot_adaptive_vs_original(df_train_only, df_adaptive, choice_log_df,
                               pct_label, pct_tag)
    plot_adaptive_zoom(df_train_only, df_adaptive, choice_log_df,
                        pct_label, pct_tag)

    adaptive_recs = evaluate_recovery(df_adaptive, gap_row_sets,
                                       'Adaptive Imputation', pct_label,
                                       df_train_corr)
    adaptive_recovery_records.extend(adaptive_recs)
    print(f"    Recovery eval Adaptive Imputation: {len(adaptive_recs)} gap blocks")

    run_adaptive_method(df_adaptive, pct_label, pct_tag)

    print("  Combined boosting curve (all methods)...")
    plot_combined_boosting_curves(all_boosting_evals[pct_label], pct_label, pct_tag)


# GAP LENGTH ANALYSIS
print("\n=== Gap Length Analysis ===")
plot_gap_length_analysis(all_gap_row_sets)


print("\n=== Adaptive Imputation — Summary ===")
adaptive_results_df = pd.DataFrame(adaptive_model_results)
adaptive_results_df.to_csv(
    os.path.join(ADAPTIVE_DIR, 'adaptive_imputation_metrics.csv'), index=False)
print(adaptive_results_df.to_string(index=False))

adaptive_pct_order = sorted(
    adaptive_results_df['Missing %'].unique(),
    key=lambda x: float(x.replace('%', '')))

adaptive_rec_df = pd.DataFrame(adaptive_recovery_records)
adaptive_rec_df.to_csv(
    os.path.join(ADAPTIVE_DIR, 'adaptive_recovery_records.csv'), index=False)

ALL_METHODS       = METHODS + ['Adaptive Imputation']
ALL_METHOD_COLORS = dict(METHOD_COLORS, **{'Adaptive Imputation': '#0D47A1'})
_ALL_METHOD_LABELS = ALL_METHODS
_ALL_METHOD_COLORS = ALL_METHOD_COLORS
_combined_results_df = pd.concat(
    [pd.DataFrame(model_results), adaptive_results_df],
    ignore_index=True, sort=False)

fig, axes = plt.subplots(2, 2, figsize=(22, 16))
fig.subplots_adjust(left=0.055, right=0.98, top=0.90, bottom=0.10,
                    hspace=0.35, wspace=0.22)
axes = axes.flatten()
handles_ah = []
for ax, metric in zip(axes, ['RMSE', 'MAE', 'R2', 'Bias']):
    for method in _ALL_METHOD_LABELS:
        mdf  = _combined_results_df[_combined_results_df['Method'] == method]\
                   .set_index('Missing %').reindex(adaptive_pct_order)
        vals = mdf[metric].values
        lw   = 3.6 if method == 'Adaptive Imputation' else 2.4
        ls   = '-' if method == 'Adaptive Imputation' else '--'
        line, = ax.plot(adaptive_pct_order, vals, marker='o', lw=lw,
                        ls=ls, markersize=8,
                        color=_ALL_METHOD_COLORS[method], label=method)
        if ax is axes[0]:
            handles_ah.append(line)
    ax.set_title(f'{metric} vs Missing % — Adaptive Imputation vs Baselines',
                 fontsize=TITLE_FS, fontweight='bold', color='black')
    ax.set_xlabel('Missing %', fontsize=LABEL_FS, fontweight='bold', color='black')
    ax.set_ylabel(metric, fontsize=LABEL_FS, fontweight='bold', color='black')
    ax.tick_params(labelsize=TICK_FS, colors='black')
    ax.grid(True, ls='--', alpha=0.4)
    for spine in ax.spines.values():
        spine.set_edgecolor('black'); spine.set_linewidth(1.5)
fig.suptitle('Adaptive Imputation vs Baseline Methods — Downstream Model Metrics',
             fontsize=SUP_FS, fontweight='bold', color='black')
fig.legend(handles=handles_ah, labels=_ALL_METHOD_LABELS,
           loc='lower center', bbox_to_anchor=(0.5, 0.01),
           ncol=len(_ALL_METHOD_LABELS),
           fontsize=LEGEND_FS, framealpha=1.0, edgecolor='black',
           prop={'weight': 'bold', 'size': LEGEND_FS})
_adaptive_cmp_path = os.path.join(
    ADAPTIVE_DIR, 'adaptive_vs_baseline_metrics.png')
plt.savefig(_adaptive_cmp_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {_adaptive_cmp_path}")

fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
mape_vals = adaptive_results_df.set_index('Missing %')\
    .reindex(adaptive_pct_order)['MAPE_%']
mape_stds = adaptive_results_df.set_index('Missing %')\
    .reindex(adaptive_pct_order)['MAPE_std']
ax.bar(adaptive_pct_order, mape_vals, yerr=mape_stds, capsize=6,
       color='#0D47A1', alpha=0.85, edgecolor='white')
ax.set_title('Adaptive Imputation — Downstream MAPE by Missing %\n'
             '(indicative only — AirTC_18m can approach 0°C)',
             fontsize=TITLE_FS, fontweight='bold', color='black')
ax.set_xlabel('Missing %', fontsize=LABEL_FS, fontweight='bold', color='black')
ax.set_ylabel('MAPE (%)', fontsize=LABEL_FS, fontweight='bold', color='black')
ax.tick_params(labelsize=TICK_FS, colors='black')
ax.grid(True, axis='y', alpha=0.3)
for spine in ax.spines.values():
    spine.set_edgecolor('black'); spine.set_linewidth(1.5)
_mape_path = os.path.join(ADAPTIVE_DIR, 'adaptive_mape_by_missing_pct.png')
plt.savefig(_mape_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {_mape_path}")


# OVERALL MODEL METRICS
results_df = pd.concat(
    [pd.DataFrame(model_results), adaptive_results_df],
    ignore_index=True, sort=False)

# deduplicate in case any pct labels collided
results_df = results_df.drop_duplicates(
    subset=['Method', 'Missing %'], keep='last')

pct_order = sorted(
    results_df['Missing %'].unique(),
    key=lambda x: float(x.replace('%', ''))
)
PCT_ORDER = pct_order
print("\nActual pct_order:", pct_order)
print(results_df[['Method', 'Missing %']].to_string())

pivot_df = results_df.pivot_table(
    index='Method', columns='Missing %',
    values=['MAE', 'RMSE', 'R2', 'Bias'])
pivot_df = pivot_df.swaplevel(axis=1).sort_index(axis=1)
pivot_df = pivot_df.reindex(pct_order, axis=1, level=0)  # dynamic
pivot_df.to_csv(os.path.join(COMP, 'interpolation_metrics.csv'))
results_df.to_csv(os.path.join(COMP, 'interpolation_metrics_with_std.csv'),
                  index=False)
print("\n=== Overall model metrics ===")
print(pivot_df.to_string())

COLORS  = ['steelblue', 'darkorange', 'green', 'red']
metrics = ['RMSE', 'MAE', 'R2', 'Bias']

# Original interpolation downstream metrics line chart (standalone)
fig, axes = plt.subplots(2, 2, figsize=(22, 16))
fig.subplots_adjust(left=0.055, right=0.98, top=0.93, bottom=0.09,
                    hspace=0.35, wspace=0.22)
axes = axes.flatten()
handles_legend = None
for ax, metric in zip(axes, metrics):
    h_list = []
    for method, color in zip(ALL_METHODS, [ALL_METHOD_COLORS[m] for m in ALL_METHODS]):
        mdf  = results_df[results_df['Method'] == method]\
               .set_index('Missing %').reindex(pct_order)
        vals = mdf[metric].values
        line, = ax.plot(pct_order, vals, marker='o', lw=2.9, markersize=8,
                        color=color, label=method)
        h_list.append(line)
    if handles_legend is None:
        handles_legend = h_list
    ax.set_title(f'{metric} vs Missing % (mean +/- std, {len(SEEDS)} seeds)',
                 fontsize=TITLE_FS, fontweight='bold', color='black')
    ax.set_xlabel('Missing %', fontsize=LABEL_FS, fontweight='bold', color='black')
    ax.set_ylabel(metric, fontsize=LABEL_FS, fontweight='bold', color='black')
    ax.tick_params(labelsize=TICK_FS, colors='black', labelcolor='black', width=2)
    ax.grid(True, ls='--', alpha=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor('black'); spine.set_linewidth(1.5)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color('black'); lbl.set_fontweight('bold')
fig.legend(handles=handles_legend, labels=ALL_METHODS,
           loc='lower center', bbox_to_anchor=(0.5, 0.01),
           ncol=len(ALL_METHODS),
           fontsize=LEGEND_FS, framealpha=1.0, edgecolor='black',
           prop={'weight': 'bold', 'size': LEGEND_FS})
plt.savefig(os.path.join(COMP, 'interpolation_downstream_metrics.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: interpolation_downstream_metrics.png")

# Grouped bar
fig, axes = plt.subplots(1, 2, figsize=(20, 9), constrained_layout=True)
x = np.arange(len(ALL_METHODS)); w = 0.16
bar_colors = ['#1565C0', '#E53935', '#2E7D32', '#F57C00']
for ax, metric in zip(axes, ['RMSE', 'MAE']):
    for j, pct in enumerate(pct_order):
        vals = []
        for m in ALL_METHODS:
            sub = results_df[(results_df['Method'] == m) &
                             (results_df['Missing %'] == pct)]
            vals.append(sub[metric].values[0] if len(sub) > 0 else 0.0)
        ax.bar(x + j * w, vals, w, label=pct,
               color=bar_colors[j % len(bar_colors)], alpha=0.85)
    ax.set_title(f'{metric} by Method and Missing %',
                 fontsize=TITLE_FS, fontweight='bold', color='black')
    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(ALL_METHODS, rotation=15, ha='right',
                       fontsize=TICK_FS, fontweight='bold')
    ax.set_ylabel(metric, fontsize=LABEL_FS, fontweight='bold', color='black')
    ax.tick_params(labelsize=TICK_FS, colors='black')
    ax.legend(title='Missing %', fontsize=LEGEND_FS, framealpha=1.0,
              edgecolor='black', prop={'weight': 'bold'},
              loc='upper center', bbox_to_anchor=(0.5, 1.18),
              ncol=len(pct_order))
    ax.grid(True, axis='y', alpha=0.3)
    for spine in ax.spines.values():
        spine.set_edgecolor('black'); spine.set_linewidth(1.5)
plt.savefig(os.path.join(COMP, 'grouped_bar_rmse_mae.png'),
            dpi=300, bbox_inches='tight')
plt.close()

# R2 heatmap
r2_mat = np.zeros((len(ALL_METHODS), len(pct_order)))
for i, m in enumerate(ALL_METHODS):
    for j, p in enumerate(pct_order):
        sub = results_df[(results_df['Method'] == m) &
                         (results_df['Missing %'] == p)]
        r2_mat[i, j] = sub['R2'].values[0] if len(sub) > 0 else 0.0

fig, ax = plt.subplots(figsize=(14, 9), constrained_layout=True)
im_h = ax.imshow(r2_mat, cmap='RdYlGn', aspect='auto',
                  vmin=r2_mat.min() - 0.01, vmax=1.0)
ax.set_xticks(range(len(pct_order)))
ax.set_xticklabels(pct_order, fontsize=TICK_FS, fontweight='bold', color='black')
ax.set_yticks(range(len(ALL_METHODS)))
ax.set_yticklabels(ALL_METHODS, fontsize=TICK_FS, fontweight='bold', color='black')
ax.tick_params(axis='both', which='both', length=0)
cb = plt.colorbar(im_h, ax=ax)
cb.set_label('R\u00b2', fontsize=LABEL_FS, fontweight='bold', color='black')
cb.ax.tick_params(labelsize=TICK_FS, colors='black')
for i in range(len(ALL_METHODS)):
    for j in range(len(pct_order)):
        ax.text(j, i, f"{r2_mat[i,j]:.4f}",
                ha='center', va='center', fontsize=ANNOT_FS,
                fontweight='bold', color='black')
plt.savefig(os.path.join(COMP, 'r2_heatmap.png'),
            dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(max(14, len(ALL_METHODS) * 2.4), 8),
                        constrained_layout=True)
xb = np.arange(len(ALL_METHODS)); wb = 0.8 / len(pct_order)
for j, p in enumerate(pct_order):
    ax.bar(xb + j * wb, r2_mat[:, j], wb, label=p,
           color=bar_colors[j % len(bar_colors)], alpha=0.85,
           edgecolor='black', linewidth=0.8)
    for i in range(len(ALL_METHODS)):
        ax.text(xb[i] + j * wb, r2_mat[i, j] + 0.002, f'{r2_mat[i, j]:.3f}',
                ha='center', va='bottom', fontsize=ANNOT_FS * 0.55,
                fontweight='bold', color='black', rotation=90)
ax.set_title('R\u00b2 by Method and Missing %', fontsize=TITLE_FS,
             fontweight='bold', color='black')
ax.set_xticks(xb + wb * (len(pct_order) - 1) / 2)
ax.set_xticklabels(ALL_METHODS, rotation=15, ha='right',
                   fontsize=TICK_FS, fontweight='bold')
ax.set_ylabel('R\u00b2', fontsize=LABEL_FS, fontweight='bold', color='black')
ax.set_ylim(max(0, r2_mat.min() - 0.03), 1.03)
ax.tick_params(labelsize=TICK_FS, colors='black')
ax.legend(title='Missing %', fontsize=LEGEND_FS, framealpha=1.0,
          edgecolor='black', prop={'weight': 'bold'},
          loc='upper center', bbox_to_anchor=(0.5, 1.16),
          ncol=len(pct_order))
ax.grid(True, axis='y', alpha=0.3)
for spine in ax.spines.values():
    spine.set_edgecolor('black'); spine.set_linewidth(1.5)
plt.savefig(os.path.join(COMP, 'r2_grouped_bar.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: r2_grouped_bar.png (bar-chart version of r2_heatmap.png)")


# RECOVERY ANALYSIS
rec_df = pd.concat(
    [pd.DataFrame(recovery_records), adaptive_rec_df],
    ignore_index=True, sort=False)

ylabels_map = {
    'RMSE': 'RMSE (°C)',
    'MAE':  'MAE (°C)',
    'R2':   'R²',
    'Bias': 'Bias (°C)',
}
metrics_order = ['RMSE', 'MAE', 'R2', 'Bias']

IC_TITLE_FS  = 30
IC_LABEL_FS  = 24
IC_TICK_FS   = 20
IC_LEGEND_FS = 20

fig, axes_left = plt.subplots(2, 2, figsize=(22, 17))
fig.subplots_adjust(left=0.07, right=0.93, top=0.94, bottom=0.14,
                    hspace=0.32, wspace=0.32)
axes_left = axes_left.flatten()

# Build legend handles once
legend_handles = []
legend_labels  = []
_leg_done = False

for ax_l, metric in zip(axes_left, metrics_order):
    ax_r = ax_l.twinx()   # right y-axis for recovery

    for method in ALL_METHODS:
        color = ALL_METHOD_COLORS[method]

        # Interpolation (dotted) on LEFT axis
        mdf_interp = results_df[results_df['Method'] == method] \
                         .set_index('Missing %').reindex(pct_order)
        vals_interp = mdf_interp[metric].values
        li, = ax_l.plot(pct_order, vals_interp,
                        marker='o', lw=3.9, markersize=13,
                        color=color, ls=':', alpha=0.85,
                        label=f'{method} (Interpolation)')

        # Recovery (solid) on RIGHT axis
        vals_rec = []
        for pct in pct_order:
            sub = rec_df[(rec_df['Method'] == method) &
                         (rec_df['Missing %'] == pct)][metric]
            vals_rec.append(sub.mean() if len(sub) > 0 else float('nan'))
        lr, = ax_r.plot(pct_order, vals_rec,
                        marker='o', lw=5.9, markersize=13,
                        color=color, ls='-',
                        label=f'{method} (Recovery)')

        if not _leg_done:
            legend_handles += [li, lr]
            legend_labels  += [f'{method} (Interpolation)',
                                f'{method} (Recovery)']

    _leg_done = True

    # Left axis styling (Interpolation)
    ax_l.set_title(f'{metric} vs Missing %',
                   fontsize=IC_TITLE_FS, fontweight='bold', color='black')
    ax_l.set_xlabel('Missing %', fontsize=IC_LABEL_FS, fontweight='bold', color='black')
    ax_l.set_ylabel(f'Interpolation  {ylabels_map[metric]}',
                    fontsize=IC_LABEL_FS, fontweight='bold', color='#444444')
    ax_l.tick_params(axis='both', labelsize=IC_TICK_FS, colors='black',
                     labelcolor='black', width=3.0)
    ax_l.grid(True, ls='--', alpha=0.35)
    for lbl in ax_l.get_xticklabels() + ax_l.get_yticklabels():
        lbl.set_color('black'); lbl.set_fontweight('bold')
    for spine in ax_l.spines.values():
        spine.set_edgecolor('black'); spine.set_linewidth(2.5)

    # Right axis styling (Recovery)
    ax_r.set_ylabel(f'Recovery  {ylabels_map[metric]}',
                    fontsize=IC_LABEL_FS, fontweight='bold', color='#444444')
    ax_r.tick_params(axis='y', labelsize=IC_TICK_FS, colors='black',
                     labelcolor='black', width=3.0)
    for lbl in ax_r.get_yticklabels():
        lbl.set_color('black'); lbl.set_fontweight('bold')
    # Right spine visible
    ax_r.spines['right'].set_edgecolor('black')
    ax_r.spines['right'].set_linewidth(2.5)

fig.legend(handles=legend_handles, labels=legend_labels,
           loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=5,
           fontsize=IC_LEGEND_FS, framealpha=1.0, edgecolor='black',
           prop={'weight': 'bold', 'size': IC_LEGEND_FS})
plt.savefig(os.path.join(COMP, 'interpolation_comparison.png'),
            dpi=300, bbox_inches='tight', pad_inches=0.15)
plt.close()
print("  Saved: interpolation_comparison.png (dual y-axis: interpolation left, recovery right)")



# GAP TEMPERATURE ANOMALY ANALYSIS
def gap_temp_analysis():
    """Summarise how gap recovery error varies with ambient temperature regime."""
    overall_mean = df_train_only['AirTC_18m'].mean()
    overall_std  = df_train_only['AirTC_18m'].std()
    print(f"\n=== Gap Temperature Anomaly Analysis ===")
    print(f"  Overall mean={overall_mean:.2f} C  std={overall_std:.2f} C")
    rows = []
    for pct_label, gap_row_sets in all_gap_row_sets.items():
        all_gap_rows  = [r for block in gap_row_sets for r in block]
        gap_temps     = df_train_only.iloc[all_gap_rows]['AirTC_18m']
        non_gap_mask  = ~df_train_only.index.isin(
            df_train_only.index[all_gap_rows])
        non_gap_temps = df_train_only[non_gap_mask]['AirTC_18m']
        gap_mean      = gap_temps.mean()
        non_gap_mean  = non_gap_temps.mean()
        z_score       = (gap_mean - overall_mean) / overall_std
        frac_ext      = ((gap_temps >= _TEMP_VERY_HIGH_THRESH) |
                         (gap_temps <= _TEMP_LOW_THRESH)).mean()
        frac_ext_all  = ((df_train_only['AirTC_18m'] >= _TEMP_VERY_HIGH_THRESH) |
                         (df_train_only['AirTC_18m'] <= _TEMP_LOW_THRESH)).mean()
        overrep       = frac_ext / frac_ext_all
        print(f"  {pct_label}: gap_mean={gap_mean:.2f}  "
              f"non_gap={non_gap_mean:.2f}  "
              f"diff={gap_mean-non_gap_mean:.2f}  "
              f"Z={z_score:.3f}  overrep={overrep:.2f}x")
        rows.append({
            'Missing_%':                 pct_label,
            'Overall_Mean':              round(overall_mean, 3),
            'Gap_Mean':                  round(gap_mean, 3),
            'NonGap_Mean':               round(non_gap_mean, 3),
            'Mean_Difference':           round(gap_mean - non_gap_mean, 3),
            'Z_Score':                   round(z_score, 3),
            'Frac_Extreme_In_Gaps':      round(frac_ext, 4),
            'Frac_Extreme_Overall':      round(frac_ext_all, 4),
            'Overrepresentation_Factor': round(overrep, 3),
        })
    pd.DataFrame(rows).to_csv(
        os.path.join(COND_DIR, 'gap_temperature_anomaly_analysis.csv'),
        index=False)
    pd.DataFrame(rows).to_excel(
        os.path.join(COND_DIR, 'gap_temperature_anomaly_analysis.xlsx'),
        index=False)
    print("  Saved: gap_temperature_anomaly_analysis.csv/.xlsx")

gap_temp_analysis()


# RECOVERY METRICS LINE GRAPHS
def plot_recovery_line_graphs():
    """Plot recovery RMSE/MAE trends across missing-data percentage per method."""
    if rec_df.empty: return
    metrics_rec = ['RMSE', 'MAE', 'R2', 'Bias']
    ylabels_rec = {
        'RMSE': 'RMSE (°C)',
        'MAE':  'MAE (°C)',
        'R2':   'R² (unitless)',
        'Bias': 'Bias (°C)',
    }

    # Overlaid: all methods per metric — 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(22, 16))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.09,
                        hspace=0.32, wspace=0.24)
    axes = axes.flatten()
    handles_rl = None
    for ax, metric in zip(axes, metrics_rec):
        h_list = []
        for method in ALL_METHODS:
            vals = []
            for pct in pct_order:
                sub = rec_df[(rec_df['Method'] == method) &
                             (rec_df['Missing %'] == pct)][metric]
                vals.append(sub.mean() if len(sub) > 0 else np.nan)
            line, = ax.plot(pct_order, vals, marker='o', lw=3.4, ms=9,
                    color=ALL_METHOD_COLORS[method], label=method)
            h_list.append(line)
        if handles_rl is None:
            handles_rl = h_list
        ax.set_xlabel('Missing %', fontsize=LABEL_FS, fontweight='bold', color='black')
        ax.set_ylabel(ylabels_rec[metric], fontsize=LABEL_FS, fontweight='bold',
                      color='black')
        ax.set_title(f'Recovery {metric} vs Missing %', fontweight='bold',
                     color='black', fontsize=TITLE_FS)
        ax.tick_params(axis='both', labelsize=TICK_FS, colors='black',
                       labelcolor='black', width=2)
        if metric == 'R2':
            ax.set_ylim(top=1.0)
        ax.grid(True, alpha=0.25)
        for spine in ax.spines.values():
            spine.set_edgecolor('black'); spine.set_linewidth(2.0)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_color('black'); lbl.set_fontweight('bold')
    fig.legend(handles=handles_rl, labels=ALL_METHODS,
               loc='lower center', bbox_to_anchor=(0.5, 0.01),
               ncol=len(ALL_METHODS),
               fontsize=LEGEND_FS, framealpha=1.0, edgecolor='black',
               prop={'weight': 'bold', 'size': LEGEND_FS})
    plt.savefig(os.path.join(COND_DIR, 'recovery_metrics_line_overlaid.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: recovery_metrics_line_overlaid.png")

    # Per method: all metrics on same plot
    n_methods = len(ALL_METHODS)
    n_cols    = 3
    n_rows    = -(-n_methods // n_cols)  # ceil division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8.2 * n_cols, 8 * n_rows))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.11,
                        hspace=0.35, wspace=0.28)
    axes = axes.flatten()
    metric_colors_local = {
        'RMSE': '#E53935', 'MAE': '#1565C0', 'Bias': '#2E7D32'}
    handles_pm = None
    for ax_idx, (ax, method) in enumerate(zip(axes, ALL_METHODS)):
        h_list = []
        for metric, mclr in metric_colors_local.items():
            vals = []
            for pct in pct_order:
                sub = rec_df[(rec_df['Method'] == method) &
                             (rec_df['Missing %'] == pct)][metric]
                vals.append(sub.mean() if len(sub) > 0 else np.nan)
            line, = ax.plot(pct_order, vals, marker='o', lw=2.9, ms=8,
                    color=mclr, label=metric)
            h_list.append(line)
        if handles_pm is None:
            handles_pm = h_list
        ax.axhline(0, color='black', lw=1.4, ls='--', alpha=0.5)
        ax.set_xlabel('Missing %', fontsize=LABEL_FS, fontweight='bold', color='black')
        ax.set_ylabel('Metric Value (°C)', fontsize=LABEL_FS, fontweight='bold',
                      color='black')
        ax.set_title(method, fontsize=TITLE_FS, fontweight='bold', color='black')
        ax.tick_params(axis='both', labelsize=TICK_FS, colors='black',
                       labelcolor='black', width=2)
        ax.grid(True, alpha=0.25)
        for spine in ax.spines.values():
            spine.set_edgecolor('black'); spine.set_linewidth(2.0)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_color('black'); lbl.set_fontweight('bold')
    for ax in axes[len(ALL_METHODS):]:
        ax.set_visible(False)
    fig.legend(handles=handles_pm, labels=list(metric_colors_local.keys()),
               loc='lower center', bbox_to_anchor=(0.5, 0.01),
               ncol=len(metric_colors_local),
               fontsize=LEGEND_FS, framealpha=1.0, edgecolor='black',
               prop={'weight': 'bold', 'size': LEGEND_FS})
    plt.savefig(os.path.join(COND_DIR, 'recovery_metrics_per_method.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: recovery_metrics_per_method.png")

    # RMSE by condition across missing %
    CONDITIONS_LOCAL = ['extreme_high_temp', 'high_temp', 'low_temp', 'normal']
    cond_colors = ['#C62828', '#FF9800', '#1565C0', '#6A1B9A']
    n_cols_c = 3
    n_rows_c = -(-len(ALL_METHODS) // n_cols_c)  # ceil division
    fig, axes = plt.subplots(n_rows_c, n_cols_c,
                              figsize=(8.2 * n_cols_c, 8 * n_rows_c))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.11,
                        hspace=0.35, wspace=0.28)
    axes = axes.flatten()
    handles_rc = []
    for ax_idx, (ax, method) in enumerate(zip(axes, ALL_METHODS)):
        sub_m = rec_df[rec_df['Method'] == method]
        for cond, clr in zip(CONDITIONS_LOCAL, cond_colors):
            vals = []
            for pct in pct_order:
                sub = sub_m[(sub_m['Condition'] == cond) &
                            (sub_m['Missing %'] == pct)]['RMSE']
                vals.append(sub.mean() if len(sub) > 0 else np.nan)
            if not all(np.isnan(v) for v in vals):
                line, = ax.plot(pct_order, vals, marker='o', lw=2.4, ms=6,
                        color=clr, label=cond, alpha=0.85)
                if ax_idx == 0:
                    handles_rc.append(line)
        ax.set_xlabel('Missing %', fontsize=LABEL_FS, fontweight='bold', color='black')
        ax.set_ylabel('Recovery RMSE (°C)', fontsize=LABEL_FS, fontweight='bold',
                      color='black')
        ax.set_title(method, fontsize=TITLE_FS, fontweight='bold', color='black')
        ax.tick_params(axis='both', labelsize=TICK_FS, colors='black',
                       labelcolor='black', width=2)
        ax.grid(True, alpha=0.25)
        for spine in ax.spines.values():
            spine.set_edgecolor('black'); spine.set_linewidth(2.0)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_color('black'); lbl.set_fontweight('bold')
    for ax in axes[len(ALL_METHODS):]:
        ax.set_visible(False)
    if handles_rc:
        fig.legend(handles=handles_rc,
                   labels=[h.get_label().replace('_', ' ') for h in handles_rc],
                   loc='lower center', bbox_to_anchor=(0.5, 0.01),
                   ncol=len(CONDITIONS_LOCAL),
                   fontsize=LEGEND_FS, framealpha=1.0, edgecolor='black',
                   prop={'weight': 'bold', 'size': LEGEND_FS})
    plt.savefig(os.path.join(COND_DIR, 'recovery_rmse_by_condition_line.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: recovery_rmse_by_condition_line.png")

plot_recovery_line_graphs()


# RECOVERY vs PREDICTION SCATTER
def plot_recovery_vs_prediction():
    """Compare gap-recovery accuracy against downstream XGBoost prediction accuracy."""
    recovery_summary = (
        rec_df.groupby(['Method', 'Missing %'])['RMSE']
              .mean().reset_index()
              .rename(columns={'RMSE': 'Recovery_RMSE'})
    )
    pred_summary = (
        results_df[['Method', 'Missing %', 'RMSE']]
        .rename(columns={'RMSE': 'Prediction_RMSE'})
    )
    scatter_df = recovery_summary.merge(
        pred_summary, on=['Method', 'Missing %'])

    fig, ax = plt.subplots(figsize=(16, 13), constrained_layout=True)
    all_points = []  # (recovery, prediction, missing_pct) across all methods, for label placement
    legend_handles = []
    for method in ALL_METHODS:
        sub = scatter_df[scatter_df['Method'] == method]
        h = ax.scatter(sub['Recovery_RMSE'], sub['Prediction_RMSE'],
                   label=method, color=ALL_METHOD_COLORS[method],
                   s=230, zorder=5, edgecolors='black', linewidths=1.3)
        legend_handles.append(h)
        for _, row in sub.iterrows():
            all_points.append((row['Recovery_RMSE'], row['Prediction_RMSE'],
                                row['Missing %']))

    ax.set_xlabel('Recovery RMSE at Gap Locations (C)',
                  fontsize=28, fontweight='bold', color='black')
    ax.set_ylabel('Model Prediction RMSE on Test Set (C)',
                  fontsize=28, fontweight='bold', color='black')
    ax.tick_params(axis='both', labelsize=26, colors='black',
                   labelcolor='black', width=2.5)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color('black'); lbl.set_fontweight('bold')
    fig.legend(handles=legend_handles, loc='outside lower center',
              ncol=len(legend_handles), fontsize=17, framealpha=1.0,
              edgecolor='black', prop={'weight': 'bold', 'size': 17},
              markerscale=0.9, handletextpad=0.5, columnspacing=1.4,
              borderpad=0.6)
    ax.grid(True, alpha=0.25)
    for spine in ax.spines.values():
        spine.set_edgecolor('black'); spine.set_linewidth(2.0)

    fig.canvas.draw()
    placed_disp = []
    OFFSETS = [(11, 8), (11, -20), (-15, 8), (-15, -20)]
    for rec, pred, pct in all_points:
        pt_disp = ax.transData.transform((rec, pred))
        used = {tuple(o) for o, prev in placed_disp
                if np.hypot(pt_disp[0]-prev[0], pt_disp[1]-prev[1]) < 45}
        offset = next((o for o in OFFSETS if o not in used), OFFSETS[0])
        ax.annotate(pct, (rec, pred), fontsize=19, fontweight='bold',
                    xytext=offset, textcoords='offset points')
        placed_disp.append((offset, pt_disp))

    plt.savefig(os.path.join(COND_DIR, 'recovery_vs_prediction_scatter.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: recovery_vs_prediction_scatter.png")

plot_recovery_vs_prediction()


# RECOVERY vs PREDICTION vs VARIABILITY TABLE
recovery_summary = (
    rec_df.groupby(['Method', 'Missing %'])['RMSE']
          .mean().reset_index()
          .rename(columns={'RMSE': 'Recovery_RMSE'})
)
pred_summary = (
    results_df[['Method', 'Missing %', 'RMSE']]
    .rename(columns={'RMSE': 'Prediction_RMSE'})
)
variability_summary = pd.DataFrame(analysis_datasets)
analysis_df = (
    recovery_summary
    .merge(pred_summary,        on=['Method', 'Missing %'])
    .merge(variability_summary, on=['Method', 'Missing %'])
)
analysis_df.to_excel(os.path.join(COND_DIR,
    'recovery_prediction_variability_analysis.xlsx'), index=False)
analysis_df.to_csv(os.path.join(COND_DIR,
    'recovery_prediction_variability_analysis.csv'), index=False)
print("\n=== Recovery vs Prediction Analysis ===")
print(analysis_df.round(3).to_string(index=False))

rec_df.to_csv(os.path.join(COND_DIR, 'recovery_records_raw.csv'), index=False)

METHOD_CLR = ALL_METHOD_COLORS
CONDITIONS = ['extreme_high_temp', 'high_temp', 'low_temp', 'normal']
BOUNDARIES = ['stable', 'rising', 'falling', 'crossing']
LENGTHS    = ['short', 'medium', 'long']


def pivot_for(df, group_col):
    """Pivot the model-results table to Method x Missing% for one metric."""
    return (df.groupby([group_col, 'Method'])['RMSE']
              .mean().unstack('Method').reindex(columns=ALL_METHODS))


def plot_grouped_bar(grouped_df, title, xlabel, savepath):
    """Plot a grouped bar chart comparing methods across missing-data levels."""
    categories = grouped_df.index.tolist()
    x = np.arange(len(categories)); w = 0.15
    n_methods = len(ALL_METHODS)
    fig, ax = plt.subplots(figsize=(max(11, len(categories) * 2.3), 8.0),
                           constrained_layout=True)
    for j, method in enumerate(ALL_METHODS):
        if method not in grouped_df.columns: continue
        ax.bar(x + j * w, grouped_df[method].values, w,
               label=method, color=METHOD_CLR[method], alpha=0.85)
    ax.set_title(title, fontsize=TITLE_FS, fontweight='bold', color='black')
    ax.set_xlabel(xlabel, fontsize=22, fontweight='bold', color='black')
    ax.set_ylabel('Mean RMSE (C)', fontsize=22, fontweight='bold', color='black')
    ax.set_xticks(x + w * (n_methods - 1) / 2)
    ax.set_xticklabels(categories, rotation=15, ha='right',
                        fontsize=19, fontweight='bold')
    ax.tick_params(axis='y', labelsize=19, width=2.0)
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight('bold')
    ax.set_xlim(x[0] - 0.35, x[-1] + n_methods * w + 0.20)
    ax.legend(fontsize=LEGEND_FS, loc='upper center',
              bbox_to_anchor=(0.5, 1.16), ncol=len(ALL_METHODS),
              framealpha=1.0, edgecolor='black')
    ax.grid(True, axis='y', alpha=0.3)
    for spine in ax.spines.values():
        spine.set_edgecolor('black'); spine.set_linewidth(2.0)
    plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {savepath}")


print("\n=== Recovery RMSE by meteorological condition ===")
cond_pivot = pivot_for(rec_df, 'Condition')
cond_pivot = cond_pivot.reindex(
    [c for c in CONDITIONS if c in cond_pivot.index])
print(cond_pivot.round(3).to_string())
cond_pivot.to_csv(os.path.join(COND_DIR, 'recovery_by_condition.csv'))
plot_grouped_bar(cond_pivot, 'Recovery RMSE by Meteorological Condition',
                 'Condition',
                 os.path.join(COND_DIR, 'recovery_by_condition.png'))

print("\n=== Recovery RMSE by boundary context ===")
bound_pivot = pivot_for(rec_df, 'Boundary')
bound_pivot = bound_pivot.reindex(
    [b for b in BOUNDARIES if b in bound_pivot.index])
print(bound_pivot.round(3).to_string())
bound_pivot.to_csv(os.path.join(COND_DIR, 'recovery_by_boundary.csv'))
plot_grouped_bar(bound_pivot, 'Recovery RMSE by Boundary Context',
                 'Boundary type',
                 os.path.join(COND_DIR, 'recovery_by_boundary.png'))

print("\n=== Recovery RMSE by gap length ===")
len_pivot = pivot_for(rec_df, 'Length')
len_pivot = len_pivot.reindex(
    [l for l in LENGTHS if l in len_pivot.index])
print(len_pivot.round(3).to_string())
len_pivot.to_csv(os.path.join(COND_DIR, 'recovery_by_length.csv'))
plot_grouped_bar(len_pivot, 'Recovery RMSE by Gap Length',
                 'Gap length',
                 os.path.join(COND_DIR, 'recovery_by_length.png'))

for heatmap_col, heatmap_vals, fname, fig_size in [
    ('Boundary', BOUNDARIES, 'heatmap_condition_x_boundary.png', (44, 48)),
    ('Length',   LENGTHS,    'heatmap_condition_x_length.png',   (38, 48)),
]:
    # Clean, concise single-line x-axis labels (no redundant sub-descriptions)
    label_map = {
        'short':    'Short',
        'medium':   'Medium',
        'long':     'Long',
        'stable':   'Stable',
        'rising':   'Rising',
        'falling':  'Falling',
        'crossing': 'Crossing',
    }
    display_labels = [label_map.get(v, v) for v in heatmap_vals]

    all_mats     = {}
    all_cnt_mats = {}
    for method in ALL_METHODS:
        sub     = rec_df[rec_df['Method'] == method]
        mat     = np.full((len(CONDITIONS), len(heatmap_vals)), np.nan)
        cnt_mat = np.zeros_like(mat, dtype=int)
        for ci, cond in enumerate(CONDITIONS):
            for vi, val in enumerate(heatmap_vals):
                cell = sub[(sub['Condition'] == cond) &
                           (sub[heatmap_col] == val)]['RMSE']
                if len(cell) > 0:
                    mat[ci, vi]     = cell.mean()
                    cnt_mat[ci, vi] = len(cell)
        all_mats[method]     = mat
        all_cnt_mats[method] = cnt_mat

    # Global colour scale
    all_valid = np.concatenate(
        [m[~np.isnan(m)] for m in all_mats.values()])
    g_vmin = float(all_valid.min()) if len(all_valid) else 0.0
    g_vmax = float(all_valid.max()) if len(all_valid) else 1.0

    n_panels   = len(ALL_METHODS)
    n_rows_hm  = -(-n_panels // 2)  # ceil division
    fig = plt.figure(figsize=fig_size)
    gs_bottom = 0.05 if heatmap_col == 'Length' else 0.02
    gs  = fig.add_gridspec(n_rows_hm, 3,
                           width_ratios=[1, 1, 0.025],
                           hspace=0.10, wspace=0.06,
                           top=0.99, bottom=gs_bottom,
                           left=0.13, right=0.97)
    subplot_positions = [(r, c) for r in range(n_rows_hm)
                         for c in range(2)][:n_panels]
    axes_list = [fig.add_subplot(gs[r, c]) for r, c in subplot_positions]
    cbar_ax   = fig.add_subplot(gs[:, 2])

    cond_label_map = {
        'extreme_high_temp': 'Ext. High Temp',
        'high_temp':         'High Temp',
        'low_temp':          'Low Temp',
        'stagnant_wind':     'Stagnant Wind',
        'normal':            'Normal',
    }
    y_labels = [cond_label_map.get(c, c) for c in CONDITIONS]

    im_last = None
    is_left_col = [c == 0 for r, c in subplot_positions]

    for idx, (ax, method) in enumerate(zip(axes_list, ALL_METHODS)):
        mat     = all_mats[method]
        cnt_mat = all_cnt_mats[method]
        im_h    = ax.imshow(mat, cmap='RdYlGn_r', aspect='auto',
                             vmin=g_vmin, vmax=g_vmax,
                             interpolation='nearest')
        im_last = im_h

        ax.set_xticks(range(len(heatmap_vals)))
        ax.set_xticklabels(display_labels, rotation=0,
                            fontsize=44, fontweight='bold', color='black')
        ax.tick_params(axis='x', which='both', length=0, colors='black')

        ax.set_yticks(range(len(CONDITIONS)))
        if is_left_col[idx]:
            ax.set_yticklabels(y_labels, fontsize=44, fontweight='bold',
                                color='black')
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis='y', which='both', length=0, colors='black')

        ax.grid(False)
        ax.set_xticks([], minor=True)
        ax.set_yticks([], minor=True)

        ax.set_title(method, fontweight='bold', fontsize=52, color='black',
                     pad=10)
        for spine in ax.spines.values():
            spine.set_edgecolor('black'); spine.set_linewidth(3.0)

        for ci in range(len(CONDITIONS)):
            for vi in range(len(heatmap_vals)):
                v = mat[ci, vi]; n = cnt_mat[ci, vi]
                if not np.isnan(v):
                    ax.text(vi, ci - 0.15, f'{v:.2f}',
                            ha='center', va='center',
                            fontsize=46, fontweight='bold', color='black')
                    ax.text(vi, ci + 0.22, f'n={n}',
                            ha='center', va='center',
                            fontsize=36, fontweight='bold', color='black')
                else:
                    ax.text(vi, ci, 'n/a',
                            ha='center', va='center',
                            fontsize=44, fontweight='bold', color='black')

    # ONE shared colorbar
    cbar = fig.colorbar(im_last, cax=cbar_ax)
    cbar.set_label('Mean Recovery RMSE (°C)',
                   fontsize=38, fontweight='bold', color='black', labelpad=10)
    cbar.ax.tick_params(labelsize=36, colors='black')
    cbar.ax.yaxis.label.set_color('black')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black',
             fontweight='bold')

    if heatmap_col == 'Length':
        key_text = ('Gap Length Key  —  '
                    'Short: ≤6 steps (≤3 hrs)     '
                    'Medium: 7–24 steps (3–12 hrs)     '
                    'Long: >24 steps (>12 hrs)')
        fig.text(0.52, 0.004, key_text,
                 ha='center', va='bottom',
                 fontsize=34, fontweight='bold', color='black',
                 bbox=dict(boxstyle='round,pad=0.4',
                           facecolor='white', edgecolor='black',
                           linewidth=2.4, alpha=0.92))

    plt.savefig(os.path.join(COND_DIR, fname), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")

    n_cond      = len(CONDITIONS)
    n_cols_bar  = 2
    n_rows_bar  = -(-n_cond // n_cols_bar)
    fig_bar, axes_bar = plt.subplots(
        n_rows_bar, n_cols_bar,
        figsize=(fig_size[0] * 0.55, fig_size[1] * 0.42),
        squeeze=False)
    fig_bar.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.07,
                            hspace=0.35, wspace=0.18)
    xb2 = np.arange(len(heatmap_vals)); wb2 = 0.8 / len(ALL_METHODS)

    method_handles = None
    for ci, cond in enumerate(CONDITIONS):
        r, c = divmod(ci, n_cols_bar)
        ax = axes_bar[r][c]

        n_per_val = [int(all_cnt_mats[ALL_METHODS[0]][ci, vi])
                     for vi in range(len(heatmap_vals))]
        for vi in range(len(heatmap_vals)):
            if n_per_val[vi] == 0:
                for m in ALL_METHODS[1:]:
                    n_per_val[vi] = max(n_per_val[vi],
                                        int(all_cnt_mats[m][ci, vi]))

        handles_here = []
        for mi, method in enumerate(ALL_METHODS):
            mat = all_mats[method]
            vals = mat[ci, :]
            plot_vals = np.nan_to_num(vals, nan=0.0)
            bar = ax.bar(xb2 + mi * wb2, plot_vals, wb2, label=method,
                        color=ALL_METHOD_COLORS[method], alpha=0.9,
                        edgecolor='black', linewidth=0.9)
            handles_here.append(bar)
            for vi in range(len(heatmap_vals)):
                if not np.isnan(vals[vi]):
                    ax.text(xb2[vi] + mi * wb2, vals[vi] + 0.02,
                            f'{vals[vi]:.2f}', ha='center', va='bottom',
                            fontsize=13, fontweight='bold', color='black',
                            rotation=90)
        if method_handles is None:
            method_handles = handles_here

        ax.set_title(cond_label_map.get(cond, cond), fontsize=24,
                     fontweight='bold', color='black')
        ax.set_xticks(xb2 + wb2 * (len(ALL_METHODS) - 1) / 2)
        ax.set_xticklabels(
            [f'{lbl}\n(n={n_per_val[vi] if n_per_val[vi] > 0 else "n/a"})'
             for vi, lbl in enumerate(display_labels)],
            fontsize=17, fontweight='bold', color='black')
        ax.set_ylabel('Mean Recovery RMSE (\u00b0C)', fontsize=17,
                      fontweight='bold', color='black')
        ax.tick_params(axis='y', labelsize=15, colors='black')
        ax.tick_params(axis='x', length=0)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(0, g_vmax * 1.45)
        for spine in ax.spines.values():
            spine.set_edgecolor('black'); spine.set_linewidth(1.5)

    for idx in range(n_cond, n_rows_bar * n_cols_bar):
        r, c = divmod(idx, n_cols_bar)
        axes_bar[r][c].axis('off')

    fig_bar.legend(handles=method_handles, labels=ALL_METHODS,
                   loc='upper center', bbox_to_anchor=(0.5, 0.985),
                   ncol=len(ALL_METHODS), fontsize=19, framealpha=1.0,
                   edgecolor='black', prop={'weight': 'bold', 'size': 19})

    if heatmap_col == 'Length':
        key_text = ('Gap Length Key  \u2014  '
                    'Short: \u22646 steps (\u22643 hrs)     '
                    'Medium: 7\u201324 steps (3\u201312 hrs)     '
                    'Long: >24 steps (>12 hrs)')
        fig_bar.text(0.52, 0.008, key_text, ha='center', va='bottom',
                    fontsize=17, fontweight='bold', color='black',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                              edgecolor='black', linewidth=1.5, alpha=0.92))

    bar_fname = fname.replace('heatmap_', 'bar_')
    plt.savefig(os.path.join(COND_DIR, bar_fname), dpi=300)
    plt.close()
    print(f"  Saved: {bar_fname} (bar-chart version of {fname})")

# Best method summaries
for group_col, group_vals, fname in [
    ('Condition', CONDITIONS, 'best_method_per_condition.csv'),
    ('Boundary',  BOUNDARIES, 'best_method_per_boundary.csv'),
    ('Length',    LENGTHS,    'best_method_per_length.csv'),
]:
    rows = []
    for val in group_vals:
        sub = rec_df[rec_df[group_col] == val]
        if sub.empty: continue
        method_rmse = sub.groupby('Method')['RMSE'].mean().reindex(ALL_METHODS)
        best  = method_rmse.idxmin(); worst = method_rmse.idxmax()
        row   = {group_col: val, 'Best': best,
                 'Best_RMSE': round(method_rmse[best], 3),
                 'Worst': worst,
                 'Worst_RMSE': round(method_rmse[worst], 3)}
        for m in ALL_METHODS:
            row[f'{m}_RMSE'] = round(method_rmse[m], 3)
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(COND_DIR, fname), index=False)


# FINAL SUMMARY
print("\n=== FINAL SUMMARY — Recovery RMSE vs Prediction RMSE ===")
rec_s  = rec_df.groupby(['Method', 'Missing %'])['RMSE']\
               .mean().reset_index()\
               .rename(columns={'RMSE': 'Recovery_RMSE'})
pred_s = results_df[['Method', 'Missing %', 'RMSE']]\
               .rename(columns={'RMSE': 'Prediction_RMSE'})
final  = rec_s.merge(pred_s, on=['Method', 'Missing %'])
print(final.sort_values(['Missing %', 'Recovery_RMSE']).to_string(index=False))


# OUTPUT STRUCTURE
print("\n=== Output structure ===")
for root_d, dirs, files in os.walk(ROOT):
    level  = root_d.replace(ROOT, '').count(os.sep)
    indent = '  ' * level
    print(f"{indent}{os.path.basename(root_d)}/")
    for f in sorted(files):
        print(f"{indent}  {f}")

print("\nDone — Experiment complete.")


print("\n=== Generating Paper Figures ===")

fi    = df_train_only.index
total = len(fi)
q_end = fi[total // 4]

fig, axes = plt.subplots(4, 1, figsize=(32, 42))

for ax, pct_label in zip(axes, pct_order):
    gap_row_sets_here = all_gap_row_sets[pct_label]
    all_rows_here     = sorted(
        set(r for block in gap_row_sets_here for r in block))
    actual_pct_here   = len(all_rows_here) / total * 100

    df_corr_here = df_train_only.copy()
    df_corr_here.iloc[
        all_rows_here,
        df_corr_here.columns.get_loc('AirTC_18m')
    ] = np.nan

    orig_q1 = df_train_only.loc[fi[0]:q_end, 'AirTC_18m']
    corr_q1 = df_corr_here.loc[fi[0]:q_end, 'AirTC_18m']

    # shade gap regions
    gap_ts = corr_q1[corr_q1.isna()].index
    if len(gap_ts) > 0:
        gap_arr    = gap_ts.sort_values()
        dt         = pd.Timedelta(minutes=15)
        in_span    = False
        span_start = None
        for i, ts in enumerate(gap_arr):
            if not in_span:
                span_start = ts; in_span = True
            if i == len(gap_arr) - 1 or \
               (gap_arr[i+1] - ts) > pd.Timedelta(minutes=35):
                ax.axvspan(span_start - dt, ts + dt,
                           color='#EF5350', alpha=0.4, zorder=0)
                in_span = False

    ax.plot(orig_q1.index, orig_q1.values,
            color='#1565C0', lw=2.4, alpha=0.5,
            ls='--', label='Original (clean)', zorder=2)
    ax.plot(corr_q1.index, corr_q1.values,
            color='#212121', lw=2.4, alpha=0.9,
            label='With gaps', zorder=3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.tick_params(axis='x', rotation=0, labelsize=42,
                   colors='black', width=3.0)
    ax.tick_params(axis='y', labelsize=42, colors='black', width=3.0)
    ax.set_ylabel('AirTC_18m (°C)', fontsize=46,
                  fontweight='bold', color='black')
    ax.set_title(
        f'{pct_label} missing  (actual: {actual_pct_here:.1f}%)  '
        f'— First Quarter (Jan–Mar 2017)',
        fontsize=46, fontweight='bold', color='black')
    ax.grid(True, alpha=0.25, color='grey')
    ax.set_xlim(fi[0], q_end)
    for spine in ax.spines.values():
        spine.set_edgecolor('black'); spine.set_linewidth(2.5)

legend_elements_gap = [
    plt.Line2D([0], [0], color='#1565C0', lw=4.4,
               ls='--', alpha=0.7, label='Original (clean)'),
    plt.Line2D([0], [0], color='#212121', lw=4.4,
               label='With gaps'),
    Patch(facecolor='#EF5350', alpha=0.4,
          label='Missing region'),
]
fig.legend(handles=legend_elements_gap,
           loc='lower center', ncol=3,
           fontsize=44, framealpha=1.0, edgecolor='black',
           prop={'weight': 'bold', 'size': 44},
           bbox_to_anchor=(0.5, 0.0))

plt.tight_layout(rect=[0, 0.035, 1, 1.0], pad=2.5)
path_fig1 = os.path.join(ROOT, 'missing_gaps_first_quarter.png')
plt.savefig(path_fig1, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {path_fig1}")

all_level_errors = {}
for pct_label in pct_order:
    gap_row_sets_here = all_gap_row_sets[pct_label]
    level_errors = {}
    for method in ALL_METHODS:
        if method == 'Adaptive Imputation':
            df_interp = all_adaptive_dfs.get(pct_label, None)
        else:
            key       = (method, pct_label)
            df_interp = all_method_dfs.get(key, None)
        if df_interp is None:
            level_errors[method] = []
            continue
        errors = []
        for block in gap_row_sets_here:
            actual = df_train_only.iloc[block]['AirTC_18m'].values
            filled = df_interp.iloc[block]['AirTC_18m'].values
            valid  = ~(np.isnan(actual) | np.isnan(filled))
            if valid.sum() > 0:
                errors.extend((filled[valid] - actual[valid]).tolist())
        level_errors[method] = errors
    all_level_errors[pct_label] = level_errors

HIST_TITLE_FS   = 24   # suptitle
HIST_PANEL_FS   = 21   # per-panel "X% missing" title
HIST_LABEL_FS   = 19   # axis labels
HIST_TICK_FS    = 17   # tick labels
HIST_ANNOT_FS   = 17   # "n = ..." box
HIST_LEGEND_FS  = 18   # shared bottom legend

fig, axes = plt.subplots(2, 2, figsize=(26, 19))
fig.subplots_adjust(left=0.055, right=0.98, top=0.94, bottom=0.13,
                    hspace=0.30, wspace=0.22)
axes = axes.flatten()

for ax, pct_label in zip(axes, pct_order):
    level_errors = all_level_errors[pct_label]
    all_vals = [e for errs in level_errors.values() for e in errs]
    if not all_vals:
        ax.set_visible(False); continue

    bin_min = np.percentile(all_vals, 1)
    bin_max = np.percentile(all_vals, 99)
    x_range = np.linspace(bin_min, bin_max, 400)

    for method in ALL_METHODS:
        errors = level_errors[method]
        if not errors: continue
        errors_arr = np.array(errors)
        clr        = ALL_METHOD_COLORS[method]
        kde        = gaussian_kde(errors_arr, bw_method=0.3)
        ax.plot(x_range, kde(x_range), color=clr, lw=3.4)
        ax.axvline(np.mean(errors_arr), color=clr,
                   lw=2.4, ls='--', alpha=0.85)

    ax.axvline(0, color='black', lw=2.9, ls='-', zorder=5)

    n_val = len(list(level_errors.values())[0])
    ax.text(0.02, 0.97,
            f'n = {n_val:,} errors per method',
            transform=ax.transAxes,
            fontsize=HIST_ANNOT_FS, fontweight='bold', color='black',
            va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.4', fc='white',
                      ec='black', alpha=1.0))

    ax.set_xlabel('Interpolation Error (Filled − Actual, °C)',
                  fontsize=HIST_LABEL_FS, fontweight='bold', color='black')
    ax.set_ylabel('Density', fontsize=HIST_LABEL_FS,
                  fontweight='bold', color='black')
    ax.set_title(
        f'{pct_label} missing',
        fontsize=HIST_PANEL_FS, fontweight='bold', color='black')
    ax.tick_params(axis='both', labelsize=HIST_TICK_FS,
                   colors='black', width=2)
    ax.locator_params(axis='x', nbins=6)
    ax.locator_params(axis='y', nbins=5)
    ax.grid(True, alpha=0.25, color='grey')
    for spine in ax.spines.values():
        spine.set_edgecolor('black'); spine.set_linewidth(2.0)

# shared legend below all panels
legend_elements_hist = [
    plt.Line2D([0], [0], color=ALL_METHOD_COLORS[m], lw=3.4, label=m)
    for m in ALL_METHODS
] + [
    plt.Line2D([0], [0], color='black', lw=2.9, ls='-',
               label='Zero error'),
    plt.Line2D([0], [0], color='grey', lw=2.4, ls='--',
               label='Method mean error'),
]
n_leg = len(legend_elements_hist)
leg_ncol = (n_leg + 1) // 2
fig.legend(handles=legend_elements_hist,
           loc='lower center',
           bbox_to_anchor=(0.5, 0.01),
           ncol=leg_ncol,
           fontsize=HIST_LEGEND_FS,
           framealpha=1.0,
           edgecolor='black',
           prop={'weight': 'bold', 'size': HIST_LEGEND_FS})

path_fig2 = os.path.join(ROOT, 'error_histograms_combined.png')
plt.savefig(path_fig2, dpi=300, bbox_inches='tight', pad_inches=0.08)
plt.close()
print(f"  Saved: {path_fig2}")