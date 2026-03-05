"""
compare_models.py
─────────────────────────────────────────────────────────────────────────────
Runs GA optimisation for ALL models (MLP, CNN, RNN, DNN, LSTM), collects
every result, ranks them, prints a rich console report, generates 7 charts
(RedBull dark-style) + 2 parameter tables and writes a full log file.

Usage (from main.py):
    from compare_models import compare_all_models
    compare_all_models(obj)
"""

import copy
import os
import time
import textwrap

import numpy as np

# ── optional pretty-table library (graceful fallback) ─────────────────────
try:
    from tabulate import tabulate
    _HAS_TABULATE = True
except ImportError:
    _HAS_TABULATE = False

# ── matplotlib ────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker

from GA import run_ga_optimization
from eval import evaluate_best_model
from print_resault import display_results

# ─────────────────────────────────────────────────────────────────────────────
ALL_MODELS = ['MLP', 'CNN', 'RNN', 'DNN', 'LSTM']
MEDALS     = {1: '🥇', 2: '🥈', 3: '🥉'}

# ── RedBull dark palette ──────────────────────────────────────────────────
RB_BG      = '#0D0D0D'     # near-black background
RB_PANEL   = '#1A1A1A'     # card background
RB_RED     = '#E8292A'     # RedBull red
RB_BLUE    = '#1B3A6B'     # RedBull navy
RB_GOLD    = '#FFD700'     # gold (1st place)
RB_SILVER  = '#C0C0C0'     # silver (2nd place)
RB_BRONZE  = '#CD7F32'     # bronze (3rd place)
RB_TEXT    = '#F0F0F0'     # primary text
RB_SUBTEXT = '#888888'     # secondary text
RB_GRID    = '#2A2A2A'     # grid lines

# Five model colours (vivid against dark bg)
MODEL_COLORS = {
    'MLP' : '#4FC3F7',   # light-blue
    'CNN' : '#E8292A',   # RedBull red
    'RNN' : '#69F0AE',   # green
    'DNN' : '#FFD54F',   # amber
    'LSTM': '#CE93D8',   # lavender
}

# ── Search space reference (used for the param-table charts) ──────────────
SEARCH_SPACES = {
    'MLP': [
        ('n_dense_layers',  '1 – 5'),
        ('dense_units',     '64 | 128 | 256 | 512'),
        ('dropout_rate',    '0.2 | 0.3 | 0.5'),
        ('learning_rate',   '0.001 | 0.01 | 0.1'),
        ('optimizer',       'Adam | RMSprop | SGD'),
        ('activation',      'relu | elu | selu | tanh'),
        ('batch_size',      '16 | 32 | 64 | 128'),
        ('n_epochs',        '50 | 100 | 150'),
    ],
    'CNN': [
        ('n_conv_layers',   '1 | 2 | 3'),
        ('conv_filters',    '32 | 64 | 128'),
        ('kernel_sizes',    '1 | 2 | 3'),
        ('pool_sizes',      '2 | 4 | 8'),
        ('n_dense_layers',  '1 – 5'),
        ('dense_units',     '64 | 128 | 256 | 512'),
        ('dropout_rate',    '0.0 – 0.5'),
        ('learning_rate',   '0.0001 – 0.05'),
        ('optimizer',       'Adam | RMSprop | SGD'),
        ('activation',      'relu | elu | selu | tanh'),
        ('batch_size',      '16 | 32 | 64 | 128'),
        ('n_epochs',        '50 | 100 | 150'),
    ],
    'RNN': [
        ('n_rnn_layers',    '1 | 2 | 3'),
        ('rnn_units',       '64 | 128 | 256'),
        ('n_dense_layers',  '1 – 5'),
        ('dense_units',     '64 | 128 | 256 | 512'),
        ('dropout_rate',    '0.0 – 0.5'),
        ('learning_rate',   '0.0001 – 0.05'),
        ('optimizer',       'Adam | RMSprop | SGD'),
        ('activation',      'relu | elu | selu | tanh'),
        ('batch_size',      '16 | 32 | 64 | 128'),
        ('n_epochs',        '50 | 100 | 150'),
    ],
    'DNN': [
        ('n_hidden_layers', '1 – 5'),
        ('hidden_units',    '32 | 64 | 128 | 256 | 512'),
        ('dropout_rate',    '0.0 – 0.5'),
        ('learning_rate',   '0.0001 – 0.05'),
        ('optimizer',       'Adam | RMSprop | SGD'),
        ('activation',      'relu | elu | selu | tanh'),
        ('batch_size',      '16 | 32 | 64'),
        ('n_epochs',        '50 | 100 | 150'),
    ],
    'LSTM': [
        ('n_lstm_layers',   '1 | 2 | 3'),
        ('lstm_units',      '32 | 64 | 128'),
        ('dropout_rate',    '0.0 – 0.5'),
        ('rec_dropout_rate','0.0 | 0.1 | 0.2'),
        ('n_dense_layers',  '1 | 2 | 3'),
        ('dense_units',     '64 | 128 | 256'),
        ('learning_rate',   '0.0001 – 0.05'),
        ('optimizer',       'Adam | RMSprop | SGD'),
        ('activation',      'relu | elu | selu | tanh'),
        ('batch_size',      '16 | 32 | 64'),
        ('n_epochs',        '50 | 100 | 150'),
    ],
}
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════
#  CORE RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def _reset_model_state(obj):
    obj.best_individual = None
    obj.best_fitness    = 0.0
    obj.best_metrics    = {}


def run_all_models(obj):
    X_train_2d = obj.X_train.copy()
    X_val_2d   = obj.X_val.copy()
    X_test_2d  = obj.X_test.copy()

    all_results = {}

    for model_type in ALL_MODELS:
        _banner(f"🚀  STARTING GA OPTIMISATION  —  {model_type}", char='═')
        obj.X_train = X_train_2d.copy()
        obj.X_val   = X_val_2d.copy()
        obj.X_test  = X_test_2d.copy()
        _reset_model_state(obj)

        try:
            exec_time = run_ga_optimization(obj, test=model_type)
            evaluate_best_model(obj, test=model_type)
            display_results(obj, exec_time, test=model_type)

            all_results[model_type] = {
                'best_individual': copy.deepcopy(obj.best_individual),
                'best_fitness'   : obj.best_fitness,
                'best_metrics'   : copy.deepcopy(obj.best_metrics),
                'execution_time' : exec_time,
                'error'          : None,
            }

        except Exception as exc:
            import traceback
            print(f"\n❌  {model_type} failed: {exc}")
            traceback.print_exc()
            all_results[model_type] = {
                'best_individual': None,
                'best_fitness'   : 0.0,
                'best_metrics'   : {'accuracy': 0.0, 'precision': 0.0,
                                    'recall': 0.0,   'f1_score': 0.0},
                'execution_time' : 0.0,
                'error'          : str(exc),
            }

    obj.X_train = X_train_2d
    obj.X_val   = X_val_2d
    obj.X_test  = X_test_2d
    return all_results


# ═══════════════════════════════════════════════════════════════════════════
#  RANKING
# ═══════════════════════════════════════════════════════════════════════════

def rank_models(all_results):
    rows = []
    for model, res in all_results.items():
        m = res['best_metrics']
        rows.append({
            'Model'    : model,
            'Accuracy' : round(m.get('accuracy',  0.0), 4),
            'Precision': round(m.get('precision', 0.0), 4),
            'Recall'   : round(m.get('recall',    0.0), 4),
            'F1-Score' : round(m.get('f1_score',  0.0), 4),
            'Time (s)' : round(res['execution_time'],   2),
            'Status'   : '✅' if res['error'] is None else '❌',
        })
    rows.sort(key=lambda r: r['F1-Score'], reverse=True)
    for i, row in enumerate(rows, start=1):
        row['Rank'] = i
    return rows


# ═══════════════════════════════════════════════════════════════════════════
#  CONSOLE REPORT
# ═══════════════════════════════════════════════════════════════════════════

def print_final_ranking(ranking_rows):
    _banner("🏆  FINAL MODEL COMPARISON & RANKING", char='═')

    for row in ranking_rows:
        rank  = row['Rank']
        medal = MEDALS.get(rank, f'#{rank}  ')
        print(f"\n  {medal}  Rank {rank} — {row['Model']}  {row['Status']}")
        print(f"     ┌─────────────┬──────────┐")
        print(f"     │ Accuracy    │  {row['Accuracy']:.4f}  │")
        print(f"     │ Precision   │  {row['Precision']:.4f}  │")
        print(f"     │ Recall      │  {row['Recall']:.4f}  │")
        print(f"     │ F1-Score    │  {row['F1-Score']:.4f}  │")
        print(f"     │ Time (s)    │  {row['Time (s)']:6.2f}  │")
        print(f"     └─────────────┴──────────┘")

    _banner("📊  SUMMARY TABLE", char='─')
    cols = ['Rank', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Time (s)', 'Status']
    if _HAS_TABULATE:
        print(tabulate(ranking_rows, headers='keys', tablefmt='fancy_grid', floatfmt='.4f'))
    else:
        header = "  ".join(f"{c:<12}" for c in cols)
        print(header)
        print("─" * len(header))
        for row in ranking_rows:
            print("  ".join(f"{str(row[c]):<12}" for c in cols))
    print()


# ═══════════════════════════════════════════════════════════════════════════
#  SHARED STYLE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _dark_fig(w, h):
    """Create a dark-background figure."""
    fig = plt.figure(figsize=(w, h), facecolor=RB_BG)
    return fig

def _style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(RB_PANEL)
    ax.tick_params(colors=RB_TEXT, labelsize=9)
    ax.xaxis.label.set_color(RB_TEXT)
    ax.yaxis.label.set_color(RB_TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
    ax.grid(color=RB_GRID, linestyle='--', linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=RB_TEXT, fontsize=11, fontweight='bold', pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=RB_SUBTEXT, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=RB_SUBTEXT, fontsize=9)

def _rb_header(fig, text, y=0.97):
    """RedBull-style top banner text."""
    fig.text(0.5, y, text, ha='center', va='top',
             color=RB_TEXT, fontsize=14, fontweight='bold',
             fontfamily='monospace')
    # red accent line
    fig.add_artist(plt.Line2D([0.05, 0.95], [y - 0.022, y - 0.022],
                               transform=fig.transFigure,
                               color=RB_RED, linewidth=1.5))

def _save_fig(fig, directory, filename):
    path = os.path.join(directory, filename)
    fig.savefig(path, dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"   ✔  {filename}")


# ═══════════════════════════════════════════════════════════════════════════
#  CHART 1 — 2×2 Per-Metric Dashboard
# ═══════════════════════════════════════════════════════════════════════════

def _chart_metric_dashboard(ranking_rows, output_dir):
    metrics     = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    models      = [r['Model'] for r in ranking_rows]
    model_clrs  = [MODEL_COLORS[m] for m in models]

    fig = _dark_fig(16, 10)
    _rb_header(fig, '◈  GA Hyperparameter Optimisation — Per-Metric Performance  ◈')
    gs  = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.30,
                   left=0.07, right=0.97, top=0.90, bottom=0.07)

    for idx, metric in enumerate(metrics):
        ax   = fig.add_subplot(gs[idx // 2, idx % 2])
        vals = [r[metric] for r in ranking_rows]
        bars = ax.bar(models, vals, color=model_clrs,
                      edgecolor='#000000', linewidth=0.7, alpha=0.92)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f'{v:.4f}', ha='center', va='bottom',
                    color=RB_TEXT, fontsize=8.5, fontweight='bold')
        # highlight best bar
        best_v = max(vals)
        for bar, v in zip(bars, vals):
            if v == best_v:
                bar.set_edgecolor(RB_GOLD)
                bar.set_linewidth(2.0)

        _style_ax(ax, title=metric, ylabel='Score')
        ax.set_ylim(0, min(1.20, max(vals) * 1.25 + 0.05))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, color=RB_TEXT, fontsize=9)

    _save_fig(fig, output_dir, 'chart_1_metric_dashboard.png')


# ═══════════════════════════════════════════════════════════════════════════
#  CHART 2 — Radar / Spider
# ═══════════════════════════════════════════════════════════════════════════

def _chart_radar(ranking_rows, output_dir):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    N       = len(metrics)
    angles  = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig = _dark_fig(9, 9)
    fig.patch.set_facecolor(RB_BG)
    _rb_header(fig, '◈  Radar — Model Performance Profile  ◈')
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor(RB_PANEL)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # style spokes & rings
    ax.set_thetagrids(np.degrees(angles[:-1]), metrics,
                      color=RB_TEXT, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'],
                        color=RB_SUBTEXT, fontsize=7.5)
    ax.grid(color=RB_GRID, linestyle='--', linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_color(RB_GRID)

    for row in ranking_rows:
        vals  = [row[m] for m in metrics] + [row[metrics[0]]]
        color = MODEL_COLORS[row['Model']]
        ax.plot(angles, vals, linewidth=2.2, color=color, label=row['Model'])
        ax.fill(angles, vals, color=color, alpha=0.12)
        # dot markers on vertices
        ax.scatter(angles, vals, s=45, color=color, zorder=5)

    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.42, 1.22),
                       fontsize=10, framealpha=0.15,
                       labelcolor=RB_TEXT, edgecolor=RB_GRID)
    legend.get_frame().set_facecolor(RB_PANEL)

    _save_fig(fig, output_dir, 'chart_2_radar.png')


# ═══════════════════════════════════════════════════════════════════════════
#  CHART 3 — Execution Time
# ═══════════════════════════════════════════════════════════════════════════

def _chart_exec_time(ranking_rows, output_dir):
    # sort by time ascending for cleaner look
    rows_by_time = sorted(ranking_rows, key=lambda r: r['Time (s)'])
    models = [r['Model'] for r in rows_by_time]
    times  = [r['Time (s)'] for r in rows_by_time]
    colors = [MODEL_COLORS[m] for m in models]
    max_t  = max(times) if max(times) > 0 else 1

    fig = _dark_fig(12, 5)
    _rb_header(fig, '◈  GA Execution Time per Model  ◈')
    ax = fig.add_axes([0.10, 0.12, 0.82, 0.72], facecolor=RB_PANEL)

    bars = ax.barh(models, times, color=colors,
                   edgecolor='#000', linewidth=0.6, alpha=0.90, height=0.55)
    for bar, t, m in zip(bars, times, models):
        ax.text(bar.get_width() + max_t * 0.012,
                bar.get_y() + bar.get_height() / 2,
                f'{t:.1f} s', va='center', color=RB_TEXT,
                fontsize=9.5, fontweight='bold')
    _style_ax(ax, xlabel='Execution Time (s)')
    ax.set_xlim(0, max_t * 1.18)
    ax.tick_params(axis='y', colors=RB_TEXT, labelsize=10)
    ax.axvline(x=np.mean(times), color=RB_RED, linewidth=1.4,
               linestyle='--', alpha=0.8, label=f'Avg {np.mean(times):.1f} s')
    ax.legend(facecolor=RB_PANEL, labelcolor=RB_TEXT,
              edgecolor=RB_GRID, fontsize=9, loc='lower right')

    _save_fig(fig, output_dir, 'chart_3_execution_time.png')


# ═══════════════════════════════════════════════════════════════════════════
#  CHART 4 — F1 Podium
# ═══════════════════════════════════════════════════════════════════════════

def _chart_f1_podium(ranking_rows, output_dir):
    models  = [r['Model']    for r in ranking_rows]
    f1_vals = [r['F1-Score'] for r in ranking_rows]
    ranks   = [r['Rank']     for r in ranking_rows]
    colors  = [MODEL_COLORS[m] for m in models]
    podium_colors = {1: RB_GOLD, 2: RB_SILVER, 3: RB_BRONZE}

    fig = _dark_fig(12, 6)
    _rb_header(fig, '◈  F1-Score Ranking — GA Optimised Models  ◈')
    ax = fig.add_axes([0.09, 0.12, 0.86, 0.72], facecolor=RB_PANEL)

    bars = ax.bar(models, f1_vals, color=colors,
                  edgecolor='#000', linewidth=0.8, alpha=0.92, width=0.55)
    for bar, v, rank, model in zip(bars, f1_vals, ranks, models):
        edge_c = podium_colors.get(rank, '#555555')
        bar.set_edgecolor(edge_c)
        bar.set_linewidth(2.2 if rank <= 3 else 0.8)
        medal  = MEDALS.get(rank, f'#{rank}')
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.010,
                f'{medal}\n{v:.4f}', ha='center', va='bottom',
                color=edge_c if rank <= 3 else RB_TEXT,
                fontsize=9, fontweight='bold', linespacing=1.4)

    _style_ax(ax, ylabel='F1-Score')
    ax.set_ylim(0, min(1.25, max(f1_vals) * 1.30 + 0.05))
    ax.set_xticklabels(models, color=RB_TEXT, fontsize=11, fontweight='bold')
    # horizontal reference line at best score
    ax.axhline(y=max(f1_vals), color=RB_GOLD, linewidth=1.0,
               linestyle=':', alpha=0.6)

    _save_fig(fig, output_dir, 'chart_4_f1_podium.png')


# ═══════════════════════════════════════════════════════════════════════════
#  CHART 5 — Heatmap
# ═══════════════════════════════════════════════════════════════════════════

def _chart_heatmap(ranking_rows, output_dir):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    models  = [r['Model'] for r in ranking_rows]
    data    = np.array([[r[m] for m in metrics] for r in ranking_rows])

    rb_cmap = LinearSegmentedColormap.from_list(
        'rb', ['#1A1A1A', '#1B3A6B', '#4FC3F7', '#69F0AE', '#FFD700'], N=256)

    fig = _dark_fig(10, 5)
    _rb_header(fig, '◈  Performance Heatmap — All Models × Metrics  ◈')
    ax = fig.add_axes([0.10, 0.14, 0.72, 0.70], facecolor=RB_PANEL)

    im = ax.imshow(data, cmap=rb_cmap, aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, color=RB_TEXT, fontsize=10, fontweight='bold')
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, color=RB_TEXT, fontsize=10, fontweight='bold')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    for i in range(len(models)):
        for j in range(len(metrics)):
            v = data[i, j]
            txt_c = '#000000' if v > 0.6 else RB_TEXT
            ax.text(j, i, f'{v:.4f}', ha='center', va='center',
                    color=txt_c, fontsize=9.5, fontweight='bold')

    cbar_ax = fig.add_axes([0.86, 0.14, 0.03, 0.70])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.ax.yaxis.set_tick_params(color=RB_TEXT, labelsize=8)
    cb.outline.set_edgecolor(RB_GRID)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=RB_TEXT)

    _save_fig(fig, output_dir, 'chart_5_heatmap.png')


# ═══════════════════════════════════════════════════════════════════════════
#  CHART 6 — Search Space Parameter Table
# ═══════════════════════════════════════════════════════════════════════════

def _chart_search_space_table(output_dir):
    """Render the GA search space for every model as a styled matplotlib table."""
    # Build a unified table: rows = unique param names, cols = models
    # collect all unique param names in insertion order
    all_params = []
    seen = set()
    for model in ALL_MODELS:
        for pname, _ in SEARCH_SPACES[model]:
            if pname not in seen:
                all_params.append(pname)
                seen.add(pname)

    col_labels = ['Parameter'] + ALL_MODELS
    cell_data  = []
    for pname in all_params:
        row = [pname]
        for model in ALL_MODELS:
            lookup = {k: v for k, v in SEARCH_SPACES[model]}
            row.append(lookup.get(pname, '—'))
        cell_data.append(row)

    n_rows = len(cell_data)
    n_cols = len(col_labels)
    fig_h  = max(5, n_rows * 0.45 + 1.5)
    fig    = _dark_fig(18, fig_h)
    _rb_header(fig, '◈  GA Search Space — Hyperparameter Intervals per Model  ◈')

    ax = fig.add_axes([0.02, 0.04, 0.96, 0.88])
    ax.axis('off')

    col_widths = [0.18] + [0.164] * len(ALL_MODELS)
    tbl = ax.table(
        cellText  = cell_data,
        colLabels = col_labels,
        loc       = 'center',
        cellLoc   = 'center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.55)

    # Style header row
    for j in range(n_cols):
        cell = tbl[0, j]
        cell.set_facecolor(RB_RED)
        cell.set_text_props(color='white', fontweight='bold', fontsize=9)
        cell.set_edgecolor('#000000')

    # Style model-name header cells (col 1-5 in row 0) with model colour
    for j, model in enumerate(ALL_MODELS, start=1):
        cell = tbl[0, j]
        cell.set_facecolor(MODEL_COLORS[model])
        cell.set_text_props(color='#000000', fontweight='bold')

    # Style data rows
    for i in range(1, n_rows + 1):
        row_bg = '#1E1E1E' if i % 2 == 0 else '#141414'
        for j in range(n_cols):
            cell = tbl[i, j]
            cell.set_facecolor(row_bg)
            cell.set_text_props(color=RB_TEXT)
            cell.set_edgecolor('#2A2A2A')
            if j == 0:  # param name col
                cell.set_text_props(color=RB_GOLD, fontweight='bold')
            if tbl[i, j].get_text().get_text() == '—':
                cell.set_text_props(color=RB_SUBTEXT)

    _save_fig(fig, output_dir, 'chart_6_search_space_table.png')


# ═══════════════════════════════════════════════════════════════════════════
#  CHART 7 — Best Found Hyperparameters Table
# ═══════════════════════════════════════════════════════════════════════════

def _chart_best_params_table(all_results, output_dir):
    """Show the best individual decoded for each model as a table."""
    from print_resault import (decode_individual, decode_cnn_individual,
                                decode_rnn_individual, decode_dnn_individual,
                                decode_lstm_individual)

    decoders = {
        'MLP' : decode_individual,
        'CNN' : decode_cnn_individual,
        'RNN' : decode_rnn_individual,
        'DNN' : decode_dnn_individual,
        'LSTM': decode_lstm_individual,
    }

    # Collect decoded params per model
    decoded = {}
    for model in ALL_MODELS:
        res = all_results[model]
        if res['error'] is None and res['best_individual'] is not None:
            try:
                decoded[model] = decoders[model](res['best_individual'])
            except Exception:
                decoded[model] = {}
        else:
            decoded[model] = {}

    # Gather all unique keys
    all_keys = []
    seen_k = set()
    for model in ALL_MODELS:
        for k in decoded.get(model, {}):
            if k not in seen_k:
                all_keys.append(k)
                seen_k.add(k)

    col_labels = ['Parameter'] + ALL_MODELS
    cell_data  = []
    for key in all_keys:
        row = [key]
        for model in ALL_MODELS:
            val = decoded.get(model, {}).get(key, '—')
            if isinstance(val, float):
                val = f'{val:.4g}'
            elif isinstance(val, (list, tuple)):
                val = str(list(val))
            row.append(str(val))
        cell_data.append(row)

    if not cell_data:
        return   # nothing to display

    n_rows = len(cell_data)
    n_cols = len(col_labels)
    fig_h  = max(5, n_rows * 0.50 + 1.5)
    fig    = _dark_fig(18, fig_h)
    _rb_header(fig, '◈  Best Found Hyperparameters per Model (GA Result)  ◈')

    ax = fig.add_axes([0.02, 0.04, 0.96, 0.88])
    ax.axis('off')

    tbl = ax.table(
        cellText  = cell_data,
        colLabels = col_labels,
        loc       = 'center',
        cellLoc   = 'center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.6)

    for j in range(n_cols):
        cell = tbl[0, j]
        cell.set_facecolor(RB_BLUE)
        cell.set_text_props(color='white', fontweight='bold', fontsize=9)
        cell.set_edgecolor('#000000')

    for j, model in enumerate(ALL_MODELS, start=1):
        cell = tbl[0, j]
        cell.set_facecolor(MODEL_COLORS[model])
        cell.set_text_props(color='#000000', fontweight='bold')

    for i in range(1, n_rows + 1):
        row_bg = '#1E1E1E' if i % 2 == 0 else '#141414'
        for j in range(n_cols):
            cell = tbl[i, j]
            cell.set_facecolor(row_bg)
            cell.set_text_props(color=RB_TEXT)
            cell.set_edgecolor('#2A2A2A')
            if j == 0:
                cell.set_text_props(color='#4FC3F7', fontweight='bold')
            if tbl[i, j].get_text().get_text() == '—':
                cell.set_text_props(color=RB_SUBTEXT)

    _save_fig(fig, output_dir, 'chart_7_best_params_table.png')


# ═══════════════════════════════════════════════════════════════════════════
#  GENERATE ALL CHARTS
# ═══════════════════════════════════════════════════════════════════════════

def generate_charts(all_results, ranking_rows, output_dir):
    print(f"\n📊  Generating charts  →  {output_dir}")
    _chart_metric_dashboard(ranking_rows, output_dir)
    _chart_radar(ranking_rows, output_dir)
    _chart_exec_time(ranking_rows, output_dir)
    _chart_f1_podium(ranking_rows, output_dir)
    _chart_heatmap(ranking_rows, output_dir)
    _chart_search_space_table(output_dir)
    _chart_best_params_table(all_results, output_dir)
    print(f"   ✔  All 7 charts saved.")


# ═══════════════════════════════════════════════════════════════════════════
#  LOG FILE
# ═══════════════════════════════════════════════════════════════════════════

def save_results_log(all_results, ranking_rows, output_dir):
    log_path = os.path.join(output_dir, 'ga_comparison_results.txt')

    with open(log_path, 'w', encoding='utf-8') as f:
        ts = time.strftime('%Y-%m-%d  %H:%M:%S')
        f.write("=" * 80 + "\n")
        f.write("  🏆  GA OPTIMISATION — ALL MODELS COMPARISON REPORT\n")
        f.write(f"  Generated : {ts}\n")
        f.write("=" * 80 + "\n\n")

        # ── ranking table ──────────────────────────────────────────────────
        f.write("📊  MODELS RANKING  (sorted by F1-Score)\n")
        f.write("─" * 80 + "\n")
        cols = ['Rank', 'Model', 'Accuracy', 'Precision',
                'Recall', 'F1-Score', 'Time (s)', 'Status']
        if _HAS_TABULATE:
            f.write(tabulate(ranking_rows, headers='keys',
                             tablefmt='grid', floatfmt='.4f'))
        else:
            header = "  ".join(f"{c:<12}" for c in cols)
            f.write(header + "\n")
            f.write("─" * len(header) + "\n")
            for row in ranking_rows:
                f.write("  ".join(f"{str(row[c]):<12}" for c in cols) + "\n")
        f.write("\n\n")

        # ── search space section ───────────────────────────────────────────
        f.write("🔧  GA SEARCH SPACE (hyperparameter intervals)\n")
        f.write("─" * 80 + "\n")
        for model in ALL_MODELS:
            f.write(f"\n  [{model}]\n")
            for pname, interval in SEARCH_SPACES[model]:
                f.write(f"    {pname:<22} :  {interval}\n")
        f.write("\n\n")

        # ── per-model detail ───────────────────────────────────────────────
        f.write("🔍  DETAILED RESULTS PER MODEL\n")
        f.write("─" * 80 + "\n")
        for model, res in all_results.items():
            m = res['best_metrics']
            f.write(f"\n  ┌─ {model} {'─'*(60-len(model))}┐\n")
            if res['error']:
                f.write(f"  │  ❌  ERROR : {res['error']}\n")
            else:
                f.write(f"  │  Accuracy   : {m.get('accuracy',  0):.4f}\n")
                f.write(f"  │  Precision  : {m.get('precision', 0):.4f}\n")
                f.write(f"  │  Recall     : {m.get('recall',    0):.4f}\n")
                f.write(f"  │  F1-Score   : {m.get('f1_score',  0):.4f}\n")
                f.write(f"  │  GA Fitness : {res['best_fitness']:.4f}\n")
                f.write(f"  │  Time (s)   : {res['execution_time']:.2f}\n")
            f.write(f"  └{'─'*62}┘\n")

        winner = ranking_rows[0]
        f.write("\n\n" + "=" * 80 + "\n")
        f.write(f"  🥇  WINNER : {winner['Model']}\n")
        f.write(f"       F1-Score  : {winner['F1-Score']:.4f}\n")
        f.write(f"       Accuracy  : {winner['Accuracy']:.4f}\n")
        f.write(f"       Precision : {winner['Precision']:.4f}\n")
        f.write(f"       Recall    : {winner['Recall']:.4f}\n")
        f.write("=" * 80 + "\n")

    print(f"📁  Log saved    →  {log_path}")
    return log_path


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def compare_all_models(obj, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(output_dir, exist_ok=True)

    total_start = time.time()

    all_results  = run_all_models(obj)
    ranking_rows = rank_models(all_results)
    print_final_ranking(ranking_rows)
    generate_charts(all_results, ranking_rows, output_dir)
    save_results_log(all_results, ranking_rows, output_dir)

    total_time = time.time() - total_start
    _banner(f"✅  All done in {total_time:.1f} s  |  Results → {output_dir}", char='═')

    return all_results, ranking_rows


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _banner(text, char='─', width=80):
    print("\n" + char * width)
    print(f"  {text}")
    print(char * width)


import copy
import os
import time

import numpy as np

# ── optional pretty-table library (graceful fallback) ─────────────────────
try:
    from tabulate import tabulate
    _HAS_TABULATE = True
except ImportError:
    _HAS_TABULATE = False

# ── matplotlib (non-interactive so it works on headless servers) ───────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from GA import run_ga_optimization
from eval import evaluate_best_model
from print_resault import display_results

# ─────────────────────────────────────────────────────────────────────────────
ALL_MODELS   = ['MLP', 'CNN', 'RNN', 'DNN', 'LSTM']
PALETTE      = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']
MEDALS       = {1: '🥇', 2: '🥈', 3: '🥉'}
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════
#  CORE RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def _reset_model_state(obj):
    """Zero-out best_* fields so each run starts clean."""
    obj.best_individual = None
    obj.best_fitness    = 0.0
    obj.best_metrics    = {}


def run_all_models(obj):
    """
    Iterate over every model type, run GA + final evaluation and collect
    results.  Data is restored between runs so RNN/LSTM reshaping does not
    pollute subsequent models.

    Returns
    -------
    dict  {model_name: {best_individual, best_fitness, best_metrics, execution_time}}
    """
    # ── save original (2-D) arrays once ────────────────────────────────────
    X_train_2d = obj.X_train.copy()
    X_val_2d   = obj.X_val.copy()
    X_test_2d  = obj.X_test.copy()

    all_results = {}

    for model_type in ALL_MODELS:
        _banner(f"🚀  STARTING GA OPTIMISATION  —  {model_type}", char='═')

        # ── always restore flat arrays before each run ────────────────────
        obj.X_train = X_train_2d.copy()
        obj.X_val   = X_val_2d.copy()
        obj.X_test  = X_test_2d.copy()
        _reset_model_state(obj)

        try:
            exec_time = run_ga_optimization(obj, test=model_type)
            evaluate_best_model(obj, test=model_type)
            display_results(obj, exec_time, test=model_type)

            all_results[model_type] = {
                'best_individual': copy.deepcopy(obj.best_individual),
                'best_fitness'   : obj.best_fitness,
                'best_metrics'   : copy.deepcopy(obj.best_metrics),
                'execution_time' : exec_time,
                'error'          : None,
            }

        except Exception as exc:
            print(f"\n❌  {model_type} failed: {exc}")
            all_results[model_type] = {
                'best_individual': None,
                'best_fitness'   : 0.0,
                'best_metrics'   : {'accuracy': 0.0, 'precision': 0.0,
                                    'recall': 0.0,   'f1_score': 0.0},
                'execution_time' : 0.0,
                'error'          : str(exc),
            }

    # ── restore 2-D data one last time ─────────────────────────────────────
    obj.X_train = X_train_2d
    obj.X_val   = X_val_2d
    obj.X_test  = X_test_2d

    return all_results


# ═══════════════════════════════════════════════════════════════════════════
#  RANKING TABLE
# ═══════════════════════════════════════════════════════════════════════════

def rank_models(all_results):
    """
    Build a list of dicts (one per model) sorted by F1-Score descending.
    Returns a plain list of dicts (no pandas dependency).
    """
    rows = []
    for model, res in all_results.items():
        m = res['best_metrics']
        rows.append({
            'Model'    : model,
            'Accuracy' : round(m.get('accuracy',  0.0), 4),
            'Precision': round(m.get('precision', 0.0), 4),
            'Recall'   : round(m.get('recall',    0.0), 4),
            'F1-Score' : round(m.get('f1_score',  0.0), 4),
            'Time (s)' : round(res['execution_time'],   2),
            'Status'   : '✅' if res['error'] is None else '❌',
        })

    rows.sort(key=lambda r: r['F1-Score'], reverse=True)
    for i, row in enumerate(rows, start=1):
        row['Rank'] = i

    return rows


# ═══════════════════════════════════════════════════════════════════════════
#  CONSOLE REPORT
# ═══════════════════════════════════════════════════════════════════════════

def print_final_ranking(ranking_rows):
    """Rich, colourful console summary."""
    _banner("🏆  FINAL MODEL COMPARISON & RANKING", char='═')

    for row in ranking_rows:
        rank   = row['Rank']
        medal  = MEDALS.get(rank, f'#{rank}  ')
        status = row['Status']
        print(f"\n  {medal}  Rank {rank} — {row['Model']}  {status}")
        print(f"     ┌─────────────┬────────┐")
        print(f"     │ Accuracy    │ {row['Accuracy']:.4f} │")
        print(f"     │ Precision   │ {row['Precision']:.4f} │")
        print(f"     │ Recall      │ {row['Recall']:.4f} │")
        print(f"     │ F1-Score    │ {row['F1-Score']:.4f} │")
        print(f"     │ Time (s)    │ {row['Time (s)']:6.2f} │")
        print(f"     └─────────────┴────────┘")

    # ── summary table ──────────────────────────────────────────────────────
    _banner("📊  SUMMARY TABLE", char='─')
    cols = ['Rank', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Time (s)', 'Status']
    if _HAS_TABULATE:
        print(tabulate(ranking_rows, headers='keys', tablefmt='fancy_grid',
                       floatfmt='.4f', numalign='center'))
    else:
        # manual fallback
        header = "  ".join(f"{c:<12}" for c in cols)
        print(header)
        print("─" * len(header))
        for row in ranking_rows:
            print("  ".join(f"{str(row[c]):<12}" for c in cols))
    print()


# ═══════════════════════════════════════════════════════════════════════════
#  CHARTS
# ═══════════════════════════════════════════════════════════════════════════

def generate_charts(all_results, ranking_rows, output_dir):
    """
    Produce 4 publication-quality charts and save them as PNG files.

    1. Grouped bar chart  — all metrics side-by-side per model
    2. Radar / spider web — per-model shape across the 4 metrics
    3. Horizontal time bar — GA execution time per model
    4. F1-Score ranking bar — sorted by F1
    """
    models  = [r['Model']    for r in ranking_rows]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    # ── 1. Grouped Bar Chart ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(15, 7))
    x     = np.arange(len(models))
    width = 0.18

    for i, metric in enumerate(metrics):
        vals = [r[metric] for r in ranking_rows]
        bars = ax.bar(x + i * width, vals, width,
                      label=metric, color=PALETTE[i], alpha=0.88,
                      edgecolor='white', linewidth=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2.,
                    bar.get_height() + 0.007,
                    f'{v:.3f}', ha='center', va='bottom',
                    fontsize=7.5, fontweight='bold')

    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('🏆  GA Optimisation — All-Model Metric Comparison',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11, loc='upper right')
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout()
    _save_fig(fig, output_dir, 'chart_1_comparison_bar.png')

    # ── 2. Radar Chart ────────────────────────────────────────────────────
    N      = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), metrics, fontsize=11)
    ax.set_ylim(0, 1)
    ax.yaxis.set_tick_params(labelsize=8)

    for i, row in enumerate(ranking_rows):
        values  = [row[m] for m in metrics] + [row[metrics[0]]]
        color   = PALETTE[i % len(PALETTE)]
        ax.plot(angles, values, linewidth=2, linestyle='solid',
                label=row['Model'], color=color)
        ax.fill(angles, values, color=color, alpha=0.10)

    ax.set_title('🕸️  Radar Chart — Model Performance',
                 y=1.10, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.38, 1.18), fontsize=10)
    plt.tight_layout()
    _save_fig(fig, output_dir, 'chart_2_radar.png')

    # ── 3. Execution Time Bar Chart ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    times = [r['Time (s)'] for r in ranking_rows]
    bars  = ax.barh(models, times,
                    color=[PALETTE[i % len(PALETTE)] for i in range(len(models))],
                    alpha=0.85, edgecolor='white', linewidth=0.6)
    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + max(times) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{t:.1f} s', va='center', fontsize=10, fontweight='bold')
    ax.set_xlabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('⏱️  GA Execution Time per Model',
                 fontsize=14, fontweight='bold', pad=12)
    ax.xaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout()
    _save_fig(fig, output_dir, 'chart_3_execution_time.png')

    # ── 4. F1-Score Ranking Bar ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    f1_vals = [r['F1-Score'] for r in ranking_rows]
    bar_colors = [PALETTE[i % len(PALETTE)] for i in range(len(models))]
    bars = ax.bar(models, f1_vals, color=bar_colors,
                  alpha=0.88, edgecolor='black', linewidth=0.7)
    for bar, v, row in zip(bars, f1_vals, ranking_rows):
        medal = MEDALS.get(row['Rank'], '')
        ax.text(bar.get_x() + bar.get_width() / 2.,
                bar.get_height() + 0.008,
                f'{medal} {v:.4f}', ha='center', va='bottom',
                fontsize=9.5, fontweight='bold')
    ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax.set_title('🥇  F1-Score Ranking (GA Optimised)',
                 fontsize=14, fontweight='bold', pad=12)
    ax.set_ylim(0, 1.15)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout()
    _save_fig(fig, output_dir, 'chart_4_f1_ranking.png')

    print(f"\n📊  Charts saved  →  {output_dir}")


# ═══════════════════════════════════════════════════════════════════════════
#  LOG FILE
# ═══════════════════════════════════════════════════════════════════════════

def save_results_log(all_results, ranking_rows, output_dir):
    """Write a human-readable .txt log with all results."""
    log_path = os.path.join(output_dir, 'ga_comparison_results.txt')

    with open(log_path, 'w', encoding='utf-8') as f:
        ts = time.strftime('%Y-%m-%d  %H:%M:%S')
        f.write("=" * 80 + "\n")
        f.write("  🏆  GA OPTIMISATION — ALL MODELS COMPARISON REPORT\n")
        f.write(f"  Generated : {ts}\n")
        f.write("=" * 80 + "\n\n")

        # ── ranking table ──────────────────────────────────────────────────
        f.write("📊  MODELS RANKING  (sorted by F1-Score)\n")
        f.write("─" * 80 + "\n")
        cols = ['Rank', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score',
                'Time (s)', 'Status']
        if _HAS_TABULATE:
            f.write(tabulate(ranking_rows, headers='keys',
                             tablefmt='grid', floatfmt='.4f'))
        else:
            header = "  ".join(f"{c:<12}" for c in cols)
            f.write(header + "\n")
            f.write("─" * len(header) + "\n")
            for row in ranking_rows:
                f.write("  ".join(f"{str(row[c]):<12}" for c in cols) + "\n")
        f.write("\n\n")

        # ── per-model detail ───────────────────────────────────────────────
        f.write("🔍  DETAILED RESULTS PER MODEL\n")
        f.write("─" * 80 + "\n")
        for model, res in all_results.items():
            m = res['best_metrics']
            f.write(f"\n  ┌─ {model} {'─'*(60-len(model))}┐\n")
            if res['error']:
                f.write(f"  │  ❌  ERROR : {res['error']}\n")
            else:
                f.write(f"  │  Accuracy   : {m.get('accuracy',  0):.4f}\n")
                f.write(f"  │  Precision  : {m.get('precision', 0):.4f}\n")
                f.write(f"  │  Recall     : {m.get('recall',    0):.4f}\n")
                f.write(f"  │  F1-Score   : {m.get('f1_score',  0):.4f}\n")
                f.write(f"  │  GA Fitness : {res['best_fitness']:.4f}\n")
                f.write(f"  │  Time (s)   : {res['execution_time']:.2f}\n")
            f.write(f"  └{'─'*62}┘\n")

        # ── winner summary ─────────────────────────────────────────────────
        winner = ranking_rows[0]
        f.write("\n\n" + "=" * 80 + "\n")
        f.write(f"  🥇  WINNER : {winner['Model']}\n")
        f.write(f"       F1-Score  : {winner['F1-Score']:.4f}\n")
        f.write(f"       Accuracy  : {winner['Accuracy']:.4f}\n")
        f.write(f"       Precision : {winner['Precision']:.4f}\n")
        f.write(f"       Recall    : {winner['Recall']:.4f}\n")
        f.write("=" * 80 + "\n")

    print(f"📁  Log saved    →  {log_path}")
    return log_path


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def compare_all_models(obj, output_dir=None):
    """
    Run GA for ALL model types, rank, chart, log and print.

    Parameters
    ----------
    obj        : MODEL instance (already has data loaded)
    output_dir : where to save charts & log (default: ./results/ next to this file)

    Returns
    -------
    (all_results dict, ranking_rows list)
    """
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(output_dir, exist_ok=True)

    total_start = time.time()

    # ── run ────────────────────────────────────────────────────────────────
    all_results  = run_all_models(obj)

    # ── rank ───────────────────────────────────────────────────────────────
    ranking_rows = rank_models(all_results)

    # ── console report ─────────────────────────────────────────────────────
    print_final_ranking(ranking_rows)

    # ── charts ─────────────────────────────────────────────────────────────
    generate_charts(all_results, ranking_rows, output_dir)

    # ── log file ───────────────────────────────────────────────────────────
    save_results_log(all_results, ranking_rows, output_dir)

    total_time = time.time() - total_start
    _banner(f"✅  All done in {total_time:.1f} s  |  Results → {output_dir}", char='═')

    return all_results, ranking_rows


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _banner(text, char='─', width=80):
    print("\n" + char * width)
    print(f"  {text}")
    print(char * width)


def _save_fig(fig, directory, filename):
    path = os.path.join(directory, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✔  {filename}")
