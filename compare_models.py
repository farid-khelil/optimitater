"""
compare_models.py
─────────────────────────────────────────────────────────────────────────────
Runs GA optimisation for ALL models (MLP, CNN, RNN, DNN, LSTM), collects
every result, ranks them, prints a rich console report, generates 4 charts
and writes a full log file.

Usage (from main.py):
    from compare_models import compare_all_models
    compare_all_models(obj)
"""

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
MEDALS       = {1: '🥇', 2: '🥈', 3: '🥉'}

# ── RedBull dark theme ────────────────────────────────────────────────────
RB_BG    = '#0D0D0D'
RB_PANEL = '#1A1A1A'
RB_RED   = '#E8292A'
RB_GOLD  = '#FFD700'
RB_WHITE = '#F0F0F0'
RB_GRAY  = '#888888'
MODEL_COLORS = {
    'MLP' : '#4FC3F7',
    'CNN' : '#E8292A',
    'RNN' : '#69F0AE',
    'DNN' : '#FFD54F',
    'LSTM': '#CE93D8',
}
METRIC_COLORS = ['#4FC3F7', '#69F0AE', '#FFD54F', '#E8292A']
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
                'logbook'        : copy.deepcopy(getattr(obj, 'logbook', None)),
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
                'logbook'        : None,
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
    Produce 4 RedBull dark-theme charts and save them as PNG files.

    1. Grouped bar chart  — all metrics side-by-side per model
    2. Radar / spider web — per-model shape across the 4 metrics
    3. Horizontal time bar — GA execution time per model
    4. F1-Score ranking bar — sorted by F1
    """
    models       = [r['Model'] for r in ranking_rows]
    metrics      = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    model_colors = [MODEL_COLORS.get(m, RB_RED) for m in models]

    # ── 1. Grouped Bar Chart ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(15, 7), facecolor=RB_BG)
    ax.set_facecolor(RB_PANEL)
    x     = np.arange(len(models))
    width = 0.18

    for i, metric in enumerate(metrics):
        vals = [r[metric] for r in ranking_rows]
        bars = ax.bar(x + i * width, vals, width,
                      label=metric, color=METRIC_COLORS[i], alpha=0.88,
                      edgecolor='#2a2a2a', linewidth=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2.,
                    bar.get_height() + 0.007,
                    f'{v:.3f}', ha='center', va='bottom',
                    fontsize=7.5, fontweight='bold', color=RB_WHITE)

    ax.set_xlabel('Model', fontsize=13, fontweight='bold', color=RB_WHITE)
    ax.set_ylabel('Score', fontsize=13, fontweight='bold', color=RB_WHITE)
    ax.set_title('🏆  GA Optimisation — All-Model Metric Comparison',
                 fontsize=15, fontweight='bold', pad=15, color=RB_GOLD)
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(models, fontsize=12, color=RB_WHITE)
    ax.set_ylim(0, 1.15)
    ax.tick_params(colors=RB_WHITE)
    ax.legend(fontsize=11, loc='upper right',
              facecolor=RB_PANEL, labelcolor=RB_WHITE, edgecolor='#333333')
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color=RB_GRAY)
    ax.set_axisbelow(True)
    ax.spines[:].set_color('#333333')
    plt.tight_layout()
    _save_fig(fig, output_dir, 'chart_1_comparison_bar.png')

    # ── 2. Radar Chart ────────────────────────────────────────────────────
    N      = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True),
                           facecolor=RB_BG)
    ax.set_facecolor(RB_PANEL)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), metrics,
                      fontsize=11, color=RB_WHITE)
    ax.set_ylim(0, 1)
    ax.yaxis.set_tick_params(labelsize=8, colors=RB_GRAY)
    ax.spines['polar'].set_color('#333333')
    ax.grid(color='#333333', linewidth=0.8)

    for i, row in enumerate(ranking_rows):
        vals  = [row[m] for m in metrics] + [row[metrics[0]]]
        color = model_colors[i]
        ax.plot(angles, vals, linewidth=2.5, linestyle='solid',
                label=row['Model'], color=color)
        ax.fill(angles, vals, color=color, alpha=0.12)

    ax.set_title('🕸️  Radar Chart — Model Performance',
                 y=1.10, fontsize=14, fontweight='bold', color=RB_GOLD)
    ax.legend(loc='upper right', bbox_to_anchor=(1.38, 1.18),
              fontsize=10, facecolor=RB_PANEL,
              labelcolor=RB_WHITE, edgecolor='#333333')
    plt.tight_layout()
    _save_fig(fig, output_dir, 'chart_2_radar.png')

    # ── 3. Execution Time Bar Chart ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=RB_BG)
    ax.set_facecolor(RB_PANEL)
    times = [r['Time (s)'] for r in ranking_rows]
    bars  = ax.barh(models, times, color=model_colors,
                    alpha=0.88, edgecolor='#2a2a2a', linewidth=0.6)
    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + max(times) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{t:.1f} s', va='center', fontsize=10,
                fontweight='bold', color=RB_WHITE)
    ax.set_xlabel('Execution Time (seconds)', fontsize=12,
                  fontweight='bold', color=RB_WHITE)
    ax.set_title('⏱️  GA Execution Time per Model',
                 fontsize=14, fontweight='bold', pad=12, color=RB_GOLD)
    ax.tick_params(colors=RB_WHITE)
    ax.spines[:].set_color('#333333')
    ax.xaxis.grid(True, linestyle='--', alpha=0.3, color=RB_GRAY)
    ax.set_axisbelow(True)
    plt.tight_layout()
    _save_fig(fig, output_dir, 'chart_3_execution_time.png')

    # ── 4. F1-Score Ranking Bar ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=RB_BG)
    ax.set_facecolor(RB_PANEL)
    f1_vals = [r['F1-Score'] for r in ranking_rows]
    bars = ax.bar(models, f1_vals, color=model_colors,
                  alpha=0.88, edgecolor='#2a2a2a', linewidth=0.7)
    for bar, v, row in zip(bars, f1_vals, ranking_rows):
        medal = MEDALS.get(row['Rank'], '')
        ax.text(bar.get_x() + bar.get_width() / 2.,
                bar.get_height() + 0.008,
                f'{medal} {v:.4f}', ha='center', va='bottom',
                fontsize=9.5, fontweight='bold', color=RB_WHITE)
    ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold', color=RB_WHITE)
    ax.set_title('🥇  F1-Score Ranking (GA Optimised)',
                 fontsize=14, fontweight='bold', pad=12, color=RB_GOLD)
    ax.set_ylim(0, 1.15)
    ax.tick_params(colors=RB_WHITE)
    ax.spines[:].set_color('#333333')
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color=RB_GRAY)
    ax.set_axisbelow(True)
    plt.tight_layout()
    _save_fig(fig, output_dir, 'chart_4_f1_ranking.png')

    _chart_generation_fitness(all_results, output_dir)
    _chart_recall_scores(ranking_rows, output_dir)
    _chart_per_model_generations(all_results, output_dir)
    _chart_best_params(all_results, output_dir)

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

# ═══════════════════════════════════════════════════════════════════════════
#  CHART 5 — GA FITNESS (RECALL) OVER GENERATIONS
# ═══════════════════════════════════════════════════════════════════════════

def _chart_generation_fitness(all_results, output_dir):
    """
    Line chart: best-fitness (recall) per generation for every model.
    Uses the DEAP logbook stored in all_results[model]['logbook'].
    """
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=RB_BG)
    ax.set_facecolor(RB_PANEL)

    plotted = False
    for model, res in all_results.items():
        logbook = res.get('logbook')
        if logbook is None or res['error']:
            continue
        gens     = logbook.select('gen')
        best_fit = logbook.select('max')   # max recall per generation
        avg_fit  = logbook.select('avg')
        color    = MODEL_COLORS.get(model, RB_RED)

        ax.plot(gens, best_fit, color=color, linewidth=2.5,
                label=f'{model} — best', marker='o', markersize=5)
        ax.plot(gens, avg_fit,  color=color, linewidth=1.2,
                linestyle='--', alpha=0.55, label=f'{model} — avg')
        plotted = True

    if not plotted:
        ax.text(0.5, 0.5, 'No generation data available',
                ha='center', va='center', color=RB_WHITE,
                fontsize=13, transform=ax.transAxes)

    ax.set_xlabel('Generation', fontsize=12, fontweight='bold', color=RB_WHITE)
    ax.set_ylabel('Fitness (Recall)', fontsize=12, fontweight='bold', color=RB_WHITE)
    ax.set_title('📈  GA Fitness (Recall) Over Generations — All Models',
                 fontsize=14, fontweight='bold', pad=14, color=RB_GOLD)
    ax.set_ylim(0, 1.05)
    ax.tick_params(colors=RB_WHITE)
    ax.spines[:].set_color('#333333')
    ax.yaxis.grid(True, linestyle='--', alpha=0.25, color=RB_GRAY)
    ax.xaxis.grid(True, linestyle='--', alpha=0.15, color=RB_GRAY)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc='lower right',
              facecolor=RB_PANEL, labelcolor=RB_WHITE, edgecolor='#333333',
              ncol=2)
    plt.tight_layout()
    _save_fig(fig, output_dir, 'chart_5_generation_fitness.png')


# ═══════════════════════════════════════════════════════════════════════════
#  CHART 6 — RECALL SCORE PER MODEL
# ═══════════════════════════════════════════════════════════════════════════

def _chart_recall_scores(ranking_rows, output_dir):
    """
    Horizontal bar chart of Recall per model, sorted by Recall descending.
    """
    rows_sorted = sorted(ranking_rows, key=lambda r: r['Recall'], reverse=True)
    models  = [r['Model']  for r in rows_sorted]
    recalls = [r['Recall'] for r in rows_sorted]
    colors  = [MODEL_COLORS.get(m, RB_RED) for m in models]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=RB_BG)
    ax.set_facecolor(RB_PANEL)

    bars = ax.barh(models, recalls, color=colors,
                   alpha=0.88, edgecolor='#2a2a2a', linewidth=0.6)

    max_val = max(recalls) if recalls else 1.0
    for bar, v in zip(bars, recalls):
        ax.text(bar.get_width() + max_val * 0.012,
                bar.get_y() + bar.get_height() / 2,
                f'{v:.4f}', va='center', fontsize=10,
                fontweight='bold', color=RB_WHITE)

    # Reference line at best recall
    if recalls:
        ax.axvline(x=recalls[0], color=RB_RED, linewidth=1.2,
                   linestyle='--', alpha=0.6, label=f'Best: {recalls[0]:.4f}')
        ax.legend(fontsize=9, facecolor=RB_PANEL,
                  labelcolor=RB_WHITE, edgecolor='#333333')

    ax.set_xlabel('Recall Score', fontsize=12, fontweight='bold', color=RB_WHITE)
    ax.set_title('🎯  Recall Score per Model (GA Optimised)',
                 fontsize=14, fontweight='bold', pad=12, color=RB_GOLD)
    ax.set_xlim(0, min(max_val * 1.18, 1.0))
    ax.tick_params(colors=RB_WHITE)
    ax.spines[:].set_color('#333333')
    ax.xaxis.grid(True, linestyle='--', alpha=0.3, color=RB_GRAY)
    ax.set_axisbelow(True)
    plt.tight_layout()
    _save_fig(fig, output_dir, 'chart_6_recall_scores.png')


# ═══════════════════════════════════════════════════════════════════════════
#  CHART 7 — PER-MODEL GENERATION SUBPLOTS
# ═══════════════════════════════════════════════════════════════════════════

def _chart_per_model_generations(all_results, output_dir):
    """
    One subplot per model showing max (best) fitness per generation.
    Solid line = best individual, dashed = population average.
    """
    valid = [(m, r) for m, r in all_results.items()
             if r.get('logbook') is not None and not r['error']]
    if not valid:
        return

    n     = len(valid)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 4.5 * nrows),
                             facecolor=RB_BG, squeeze=False)
    fig.suptitle('📈  GA Max Fitness per Generation — per Model',
                 fontsize=15, fontweight='bold', color=RB_GOLD, y=1.02)

    for idx, (model, res) in enumerate(valid):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        ax.set_facecolor(RB_PANEL)
        color = MODEL_COLORS.get(model, RB_RED)

        lb       = res['logbook']
        gens     = lb.select('gen')
        best_fit = lb.select('max')
        avg_fit  = lb.select('avg')

        ax.plot(gens, best_fit, color=color, linewidth=2.5,
                marker='o', markersize=5, label='Best', zorder=3)
        ax.plot(gens, avg_fit, color=color, linewidth=1.2,
                linestyle='--', alpha=0.55, label='Avg')
        ax.fill_between(gens, avg_fit, best_fit, color=color, alpha=0.08)

        # Annotate the final best value
        ax.annotate(f'{best_fit[-1]:.4f}',
                    xy=(gens[-1], best_fit[-1]),
                    xytext=(6, 6), textcoords='offset points',
                    fontsize=9, color=RB_WHITE, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

        ax.set_title(model, fontsize=13, fontweight='bold', color=color, pad=8)
        ax.set_xlabel('Generation', fontsize=9, color=RB_WHITE)
        ax.set_ylabel('Fitness (Recall)', fontsize=9, color=RB_WHITE)
        ax.set_ylim(0, 1.05)
        ax.tick_params(colors=RB_WHITE, labelsize=8)
        ax.spines[:].set_color('#333333')
        ax.yaxis.grid(True, linestyle='--', alpha=0.2, color=RB_GRAY)
        ax.xaxis.grid(True, linestyle='--', alpha=0.15, color=RB_GRAY)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, facecolor=RB_PANEL,
                  labelcolor=RB_WHITE, edgecolor='#333333')

    # Hide unused subplots
    for idx in range(len(valid), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    plt.tight_layout()
    _save_fig(fig, output_dir, 'chart_7_per_model_generations.png')


# ═══════════════════════════════════════════════════════════════════════════
#  CHART 8 — BEST HYPERPARAMETERS TABLE
# ═══════════════════════════════════════════════════════════════════════════

_OPTIMIZERS = {0: 'Adam', 1: 'SGD', 2: 'RMSprop'}


def _decode_best_params(model_type, individual):
    """
    Decode a DEAP Individual list into a readable {param: value} dict
    based on the gene layout in GA.py.
    """
    if individual is None:
        return {}
    ind = list(individual)
    p   = {}

    if model_type == 'MLP':
        # [0] n_dense_layers | [1-5] dense_units | [6] dropout | [7] lr
        # [8] optimizer | [9] activation | [10] batch_size | [11] n_epochs
        n = ind[0]
        p['n_dense_layers'] = n
        p['dense_units']    = str(ind[1 : 1 + n])
        p['dropout_rate']   = ind[6]
        p['learning_rate']  = ind[7]
        p['optimizer']      = _OPTIMIZERS.get(ind[8],  ind[8])
        p['activation']     = ind[9]
        p['batch_size']     = ind[10]
        p['n_epochs']       = ind[11]

    elif model_type == 'CNN':
        # [0] n_conv | [1-3] conv_filters | [4-6] kernel_sizes | [7-9] pool_sizes
        # [10] n_dense | [11-15] dense_units | [16] dropout | [17] lr
        # [18] optimizer | [19] activation | [20] batch_size | [21] n_epochs
        nc = ind[0]
        nd = ind[10]
        p['n_conv_layers']  = nc
        p['conv_filters']   = str(ind[1 : 1 + nc])
        p['kernel_sizes']   = str(ind[4 : 4 + nc])
        p['pool_sizes']     = str(ind[7 : 7 + nc])
        p['n_dense_layers'] = nd
        p['dense_units']    = str(ind[11 : 11 + nd])
        p['dropout_rate']   = ind[16]
        p['learning_rate']  = ind[17]
        p['optimizer']      = _OPTIMIZERS.get(ind[18], ind[18])
        p['activation']     = ind[19]
        p['batch_size']     = ind[20]
        p['n_epochs']       = ind[21]

    elif model_type == 'RNN':
        # [0] n_rnn | [1] rnn_units | [2] n_dense | [3-7] dense_units
        # [8] optimizer | [9] activation | [10] dropout | [11] lr
        # [12] batch_size | [13] n_epochs
        nd = ind[2]
        p['n_rnn_layers']   = ind[0]
        p['rnn_units']      = ind[1]
        p['n_dense_layers'] = nd
        p['dense_units']    = str(ind[3 : 3 + nd])
        p['dropout_rate']   = ind[10]
        p['learning_rate']  = ind[11]
        p['optimizer']      = _OPTIMIZERS.get(ind[8],  ind[8])
        p['activation']     = ind[9]
        p['batch_size']     = ind[12]
        p['n_epochs']       = ind[13]

    elif model_type == 'DNN':
        # [0] n_hidden | [1-5] hidden_units | [6] dropout | [7] lr
        # [8] optimizer_idx | [9] activation | [10] batch_size | [11] n_epochs
        n = ind[0]
        p['n_hidden_layers'] = n
        p['hidden_units']    = str(ind[1 : 1 + n])
        p['dropout_rate']    = ind[6]
        p['learning_rate']   = ind[7]
        p['optimizer']       = _OPTIMIZERS.get(ind[8],  ind[8])
        p['activation']      = ind[9]
        p['batch_size']      = ind[10]
        p['n_epochs']        = ind[11]

    elif model_type == 'LSTM':
        # [0] n_lstm | [1-3] lstm_units | [4] dropout | [5] rec_dropout
        # [6] n_dense | [7-9] dense_units | [10] lr | [11] optimizer_idx
        # [12] activation | [13] batch_size | [14] n_epochs
        nl = ind[0]
        nd = ind[6]
        p['n_lstm_layers']  = nl
        p['lstm_units']     = str(ind[1 : 1 + nl])
        p['dropout_rate']   = ind[4]
        p['rec_dropout']    = ind[5]
        p['n_dense_layers'] = nd
        p['dense_units']    = str(ind[7 : 7 + nd])
        p['learning_rate']  = ind[10]
        p['optimizer']      = _OPTIMIZERS.get(ind[11], ind[11])
        p['activation']     = ind[12]
        p['batch_size']     = ind[13]
        p['n_epochs']       = ind[14]

    return p


def _chart_best_params(all_results, output_dir):
    """
    Table chart: best hyperparameters for every model side-by-side.
    Rows = param names, columns = models.
    """
    model_params = {
        m: _decode_best_params(m, r['best_individual'])
        for m, r in all_results.items()
    }

    # Collect all param names in insertion order
    all_keys = []
    seen     = set()
    for params in model_params.values():
        for k in params:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    models = list(all_results.keys())
    if not all_keys or not models:
        return

    nrows = len(all_keys)
    ncols = len(models)

    fig_h = max(3.0, 0.6 * nrows + 2.0)
    fig_w = max(8.0, 2.8 * ncols + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=RB_BG)
    ax.set_facecolor(RB_BG)
    ax.axis('off')
    ax.set_title('🔧  Best Hyperparameters per Model (GA Optimised)',
                 fontsize=14, fontweight='bold', color=RB_GOLD,
                 pad=18, loc='center')

    # Build cell data
    cell_data = []
    for key in all_keys:
        cell_data.append([str(model_params[m].get(key, '—')) for m in models])

    # Alternating row backgrounds
    cell_colors = [
        (['#1E1E1E' if i % 2 == 0 else '#252525'] * ncols)
        for i in range(nrows)
    ]

    tbl = ax.table(
        cellText   = cell_data,
        rowLabels  = all_keys,
        colLabels  = models,
        cellLoc    = 'center',
        rowLoc     = 'right',
        loc        = 'center',
        cellColours= cell_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.7)

    # Style column headers (model names)
    for col_idx, model in enumerate(models):
        cell = tbl[0, col_idx]
        cell.set_facecolor(MODEL_COLORS.get(model, RB_RED))
        cell.set_text_props(color='#0D0D0D', fontweight='bold', fontsize=10)
        cell.set_edgecolor('#0D0D0D')

    # Style row label cells
    for row_idx in range(nrows):
        cell = tbl[row_idx + 1, -1]
        cell.set_facecolor('#111111')
        cell.set_text_props(color=RB_GOLD, fontsize=8.5)
        cell.set_edgecolor('#333333')

    # Style data cells
    for row_idx in range(nrows):
        bg = '#1E1E1E' if row_idx % 2 == 0 else '#252525'
        for col_idx in range(ncols):
            cell = tbl[row_idx + 1, col_idx]
            cell.set_facecolor(bg)
            cell.set_text_props(color=RB_WHITE, fontsize=8.5)
            cell.set_edgecolor('#333333')

    plt.tight_layout()
    _save_fig(fig, output_dir, 'chart_8_best_params.png')


def _banner(text, char='─', width=80):
    print("\n" + char * width)
    print(f"  {text}")
    print(char * width)


def _save_fig(fig, directory, filename):
    path = os.path.join(directory, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"   ✔  {filename}")
