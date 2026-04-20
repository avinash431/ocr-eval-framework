"""Regenerate all whitepaper charts from expanded GT metrics.

Produces 13 figures in docs/whitepaper/figures/ using data from
results/expanded_gt_metrics/.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# Style config
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'figure.figsize': (8, 5),
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2,
})

FIGURES_DIR = Path('docs/whitepaper/figures')
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Color palette
COLORS = {
    'tesseract': '#e74c3c',
    'mistral_ocr': '#3498db',
    'surya': '#2ecc71',
    'sarvam_ocr': '#9b59b6',
    'docling': '#f39c12',
    'paddleocr': '#1abc9c',
    'got_ocr': '#e67e22',
}

DISPLAY_NAMES = {
    'tesseract': 'Tesseract',
    'mistral_ocr': 'Mistral OCR',
    'surya': 'Surya',
    'sarvam_ocr': 'Sarvam OCR',
    'docling': 'Docling',
    'paddleocr': 'PaddleOCR',
    'got_ocr': 'GOT-OCR',
}


def load_data():
    """Load per-doc metrics and summaries."""
    rows = []
    with open('results/expanded_gt_metrics/per_doc_metrics.csv') as f:
        for r in csv.DictReader(f):
            rows.append(r)

    with open('results/expanded_gt_metrics/model_summaries.json') as f:
        summaries = json.load(f)

    return rows, summaries


def fig1_f1_comparison(rows, summaries):
    """Bar chart of avg F1 by model (expanded GT)."""
    models = ['tesseract', 'mistral_ocr', 'surya', 'sarvam_ocr', 'docling', 'got_ocr']
    # Use non-circular metrics for Mistral (forms only)
    f1_vals = []
    ns = []
    for m in models:
        if m == 'mistral_ocr':
            mr = [r for r in rows if r['model'] == m and r['category'] == '02_complex_tables/forms']
            f1_vals.append(np.mean([float(r['f1']) for r in mr]))
            ns.append(len(mr))
        elif m in summaries:
            f1_vals.append(summaries[m]['avg_f1'])
            ns.append(summaries[m]['n'])
        else:
            f1_vals.append(0)
            ns.append(0)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(models))
    bars = ax.bar(x, f1_vals, color=[COLORS[m] for m in models], edgecolor='white', linewidth=0.5)

    for i, (bar, val, n) in enumerate(zip(bars, f1_vals, ns)):
        label = f'{val:.3f}\n(n={n})'
        if models[i] == 'mistral_ocr':
            label += '*'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, label,
                ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Average F1 Score')
    ax.set_title('Token-Level F1 by Model (Expanded GT, 45 docs)')
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES[m] for m in models], rotation=15, ha='right')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.3, label='F1 = 0.7 baseline')
    ax.text(0.02, 0.02, '*Mistral: forms only (consensus GT circularity on other categories)',
            transform=ax.transAxes, fontsize=7, alpha=0.6)
    fig.savefig(FIGURES_DIR / 'fig1_f1_comparison.png')
    plt.close(fig)
    print('  fig1_f1_comparison.png')


def fig2_cer_comparison(rows, summaries):
    """Bar chart of avg CER by model."""
    models = ['tesseract', 'surya', 'sarvam_ocr', 'docling', 'paddleocr']
    # Exclude Mistral (circular) and GOT-OCR (off-scale CER=4.76)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(models))
    vals = [summaries[m]['avg_cer'] for m in models]
    ns = [summaries[m]['n'] for m in models]
    bars = ax.bar(x, vals, color=[COLORS[m] for m in models], edgecolor='white')

    for bar, val, n in zip(bars, vals, ns):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}\n(n={n})', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Average CER (lower is better)')
    ax.set_title('Character Error Rate by Model (Expanded GT)')
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES[m] for m in models], rotation=15, ha='right')
    ax.set_ylim(0, max(vals) * 1.3)
    ax.text(0.02, 0.95, 'Mistral OCR excluded (circular GT)\nGOT-OCR excluded (CER=4.76, off-scale)',
            transform=ax.transAxes, fontsize=7, alpha=0.6, va='top')
    fig.savefig(FIGURES_DIR / 'fig2_cer_comparison.png')
    plt.close(fig)
    print('  fig2_cer_comparison.png')


def fig3_category_f1_heatmap(rows):
    """Heatmap of F1 by model x category."""
    models = ['tesseract', 'surya', 'sarvam_ocr', 'docling', 'got_ocr']
    categories = [
        '02_complex_tables/financial',
        '02_complex_tables/forms',
        '02_complex_tables/multi_column',
        '06_mixed_content/equations_formulas',
        '06_mixed_content/receipts',
    ]
    cat_short = ['Financial', 'Forms', 'Multi-column', 'Equations', 'Receipts']

    # Build matrix
    matrix = np.full((len(models), len(categories)), np.nan)
    for i, m in enumerate(models):
        for j, c in enumerate(categories):
            mr = [float(r['f1']) for r in rows if r['model'] == m and r['category'] == c]
            if mr:
                matrix[i, j] = np.mean(mr)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels(cat_short, rotation=30, ha='right')
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels([DISPLAY_NAMES[m] for m in models])

    # Annotate cells
    for i in range(len(models)):
        for j in range(len(categories)):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, 'N/A', ha='center', va='center', fontsize=9, color='gray')
            else:
                color = 'white' if val < 0.4 or val > 0.85 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=9, color=color)

    plt.colorbar(im, ax=ax, label='F1 Score')
    ax.set_title('F1 Score by Model and Category (Expanded GT)')
    fig.savefig(FIGURES_DIR / 'fig3_category_heatmap.png')
    plt.close(fig)
    print('  fig3_category_heatmap.png')


def fig4_error_decomposition(summaries):
    """Stacked bar chart of S/I/D error profiles."""
    models = ['tesseract', 'mistral_ocr', 'surya', 'sarvam_ocr', 'docling', 'got_ocr']
    models = [m for m in models if m in summaries]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(models))
    width = 0.6

    subs = [summaries[m]['sub_pct'] for m in models]
    ins = [summaries[m]['ins_pct'] for m in models]
    dels = [summaries[m]['del_pct'] for m in models]

    ax.bar(x, subs, width, label='Substitutions', color='#e74c3c', edgecolor='white')
    ax.bar(x, ins, width, bottom=subs, label='Insertions', color='#f39c12', edgecolor='white')
    ax.bar(x, dels, width, bottom=[s+i for s, i in zip(subs, ins)], label='Deletions', color='#3498db', edgecolor='white')

    # Annotate
    for i, m in enumerate(models):
        total = summaries[m]['total_errors']
        ax.text(i, 102, f'n={summaries[m]["n"]}\n{total:,}', ha='center', va='bottom', fontsize=7)

    ax.set_ylabel('Error Composition (%)')
    ax.set_title('Error Decomposition: Substitution / Insertion / Deletion')
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES[m] for m in models], rotation=15, ha='right')
    ax.set_ylim(0, 120)
    ax.legend(loc='upper right')
    fig.savefig(FIGURES_DIR / 'fig4_error_decomposition.png')
    plt.close(fig)
    print('  fig4_error_decomposition.png')


def fig5_precision_recall(rows, summaries):
    """Scatter plot of precision vs recall."""
    models = ['tesseract', 'surya', 'sarvam_ocr', 'docling', 'paddleocr', 'got_ocr']
    models = [m for m in models if m in summaries]

    fig, ax = plt.subplots(figsize=(7, 6))
    for m in models:
        p = summaries[m]['avg_precision']
        r = summaries[m]['avg_recall']
        f1 = summaries[m]['avg_f1']
        ax.scatter(r, p, s=150, c=COLORS[m], edgecolors='black', linewidth=0.5, zorder=5)
        ax.annotate(f'{DISPLAY_NAMES[m]}\nF1={f1:.3f}', (r, p),
                    textcoords='offset points', xytext=(10, 5), fontsize=8)

    # F1 contour lines
    for f1_val in [0.5, 0.6, 0.7, 0.8]:
        r_range = np.linspace(0.01, 1, 100)
        p_range = (f1_val * r_range) / (2 * r_range - f1_val)
        valid = (p_range > 0) & (p_range <= 1)
        ax.plot(r_range[valid], p_range[valid], '--', color='gray', alpha=0.3, linewidth=0.8)
        # Label
        idx = np.argmin(np.abs(p_range - r_range))
        if valid[idx]:
            ax.text(r_range[idx], p_range[idx], f'F1={f1_val}', fontsize=7, alpha=0.4)

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision vs Recall (Expanded GT)')
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0.3, 0.95)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    fig.savefig(FIGURES_DIR / 'fig5_precision_recall.png')
    plt.close(fig)
    print('  fig5_precision_recall.png')


def fig6_success_rates(rows):
    """Bar chart of success rates from batch runs."""
    # Use known success rates from batch results
    success_data = {
        'tesseract': {'total': 60, 'of': 91},
        'mistral_ocr': {'total': 91, 'of': 91},
        'surya': {'total': 71, 'of': 91},
        'sarvam_ocr': {'total': 62, 'of': 91},
        'docling': {'total': 67, 'of': 91},
        'got_ocr': {'total': 29, 'of': 91},  # current progress
    }

    models = list(success_data.keys())
    rates = [success_data[m]['total'] / success_data[m]['of'] * 100 for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(models))
    bars = ax.bar(x, rates, color=[COLORS[m] for m in models], edgecolor='white')

    for bar, m in zip(bars, models):
        d = success_data[m]
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{d["total"]}/{d["of"]}', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Full-Dataset Success Rates')
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES[m] for m in models], rotation=15, ha='right')
    ax.set_ylim(0, 115)
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.3)
    fig.savefig(FIGURES_DIR / 'fig6_success_rates.png')
    plt.close(fig)
    print('  fig6_success_rates.png')


def fig7_category_success_heatmap():
    """Heatmap of success rates by model x category."""
    models = ['tesseract', 'mistral_ocr', 'surya', 'sarvam_ocr', 'docling']
    categories = ['financial', 'forms', 'multi_col', 'handwritten', 'hindi', 'equations', 'receipts']
    cat_full = ['Financial\n(15)', 'Forms\n(5)', 'Multi-col\n(15)', 'Handwritten\n(30)',
                'Hindi\n(1)', 'Equations\n(15)', 'Receipts\n(10)']

    # Known success rates
    data = {
        'tesseract':   [15, 5, 15,  0, 1, 14, 10],
        'mistral_ocr': [15, 5, 15, 30, 1, 15, 10],
        'surya':       [15, 5, 15, 10, 1, 15, 10],
        'sarvam_ocr':  [15, 5, 15, 24, 1,  2,  0],
        'docling':     [15, 5, 15,  7, 0, 15, 10],
    }
    totals = [15, 5, 15, 30, 1, 15, 10]

    matrix = np.array([[data[m][j]/totals[j]*100 for j in range(len(categories))] for m in models])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')

    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels(cat_full, fontsize=8)
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels([DISPLAY_NAMES[m] for m in models])

    for i in range(len(models)):
        for j in range(len(categories)):
            val = matrix[i, j]
            color = 'white' if val < 30 or val > 90 else 'black'
            ax.text(j, i, f'{data[models[i]][j]}/{totals[j]}', ha='center', va='center',
                    fontsize=8, color=color, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Success Rate (%)')
    ax.set_title('Category-Level Success Rates')
    fig.savefig(FIGURES_DIR / 'fig7_category_success_heatmap.png')
    plt.close(fig)
    print('  fig7_category_success_heatmap.png')


def fig8_forms_comparison(rows):
    """Grouped bar chart of forms metrics by model."""
    models = ['tesseract', 'mistral_ocr', 'surya', 'sarvam_ocr', 'docling', 'got_ocr']
    metrics_names = ['CER', 'WER', 'F1']

    forms_data = {}
    for m in models:
        mr = [r for r in rows if r['model'] == m and r['category'] == '02_complex_tables/forms']
        if mr:
            forms_data[m] = {
                'CER': np.mean([float(r['cer']) for r in mr]),
                'WER': np.mean([float(r['wer']) for r in mr]),
                'F1': np.mean([float(r['f1']) for r in mr]),
            }

    models = [m for m in models if m in forms_data]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    width = 0.25

    for i, metric in enumerate(metrics_names):
        vals = [forms_data[m][metric] for m in models]
        bars = ax.bar(x + i*width - width, vals, width, label=metric,
                      alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)

    ax.set_ylabel('Score')
    ax.set_title('Forms Subcategory Metrics (Independent FUNSD GT, n=5)')
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES[m] for m in models], rotation=15, ha='right')
    ax.set_ylim(0, 1.15)
    ax.legend()
    fig.savefig(FIGURES_DIR / 'fig8_forms_comparison.png')
    plt.close(fig)
    print('  fig8_forms_comparison.png')


def fig9_radar(summaries):
    """Radar chart comparing top models across dimensions."""
    models = ['surya', 'docling', 'sarvam_ocr', 'tesseract']
    dims = ['F1', '1-CER', 'Precision', 'Recall', 'Word Acc', '1-EditDist']
    _ = ['avg_f1', 'avg_cer', 'avg_precision', 'avg_recall', 'avg_word_accuracy', 'avg_edit_dist']

    angles = np.linspace(0, 2*np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for m in models:
        s = summaries[m]
        values = [
            s['avg_f1'],
            1 - s['avg_cer'],
            s['avg_precision'],
            s['avg_recall'],
            s['avg_word_accuracy'],
            1 - s['avg_edit_dist'],
        ]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=1.5, label=DISPLAY_NAMES[m], color=COLORS[m])
        ax.fill(angles, values, alpha=0.1, color=COLORS[m])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title('Model Comparison Radar (Expanded GT)', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    fig.savefig(FIGURES_DIR / 'fig9_radar.png')
    plt.close(fig)
    print('  fig9_radar.png')


def fig10_significance(rows):
    """Significance matrix heatmap."""
    from scipy.stats import wilcoxon
    models = ['tesseract', 'surya', 'sarvam_ocr', 'docling', 'got_ocr']

    model_docs = defaultdict(dict)
    for r in rows:
        model_docs[r['model']][r['doc']] = r

    n = len(models)
    p_matrix = np.ones((n, n))
    delta_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            a, b = models[i], models[j]
            shared = set(model_docs[a].keys()) & set(model_docs[b].keys())
            if len(shared) < 5:
                continue
            f1_a = [float(model_docs[a][d]['f1']) for d in sorted(shared)]
            f1_b = [float(model_docs[b][d]['f1']) for d in sorted(shared)]
            diffs = [x-y for x, y in zip(f1_a, f1_b)]
            if all(d == 0 for d in diffs):
                continue
            try:
                _, p = wilcoxon(f1_a, f1_b)
                p_matrix[i, j] = p
                p_matrix[j, i] = p
                delta_matrix[i, j] = np.mean(f1_a) - np.mean(f1_b)
                delta_matrix[j, i] = -delta_matrix[i, j]
            except (ValueError, TypeError):
                pass

    fig, ax = plt.subplots(figsize=(7, 6))
    # Use -log10(p) for visualization, cap at 5
    log_p = -np.log10(p_matrix + 1e-10)
    np.fill_diagonal(log_p, 0)
    log_p = np.clip(log_p, 0, 5)

    im = ax.imshow(log_p, cmap='YlOrRd', vmin=0, vmax=5, aspect='auto')
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels([DISPLAY_NAMES[m] for m in models], rotation=30, ha='right')
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([DISPLAY_NAMES[m] for m in models])

    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, '-', ha='center', va='center', fontsize=9)
            else:
                p = p_matrix[i, j]
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                delta = delta_matrix[i, j]
                sign = '+' if delta > 0 else '-'
                ax.text(j, i, f'{sig}\n{sign}{abs(delta):.3f}', ha='center', va='center',
                        fontsize=7, fontweight='bold' if p < 0.05 else 'normal')

    plt.colorbar(im, ax=ax, label='-log10(p-value)')
    ax.set_title('Pairwise F1 Significance (Wilcoxon signed-rank)')
    fig.savefig(FIGURES_DIR / 'fig10_significance_matrix.png')
    plt.close(fig)
    print('  fig10_significance_matrix.png')


def fig11_boxplot_f1(rows):
    """Box plot of per-doc F1 distributions."""
    models = ['tesseract', 'surya', 'sarvam_ocr', 'docling', 'got_ocr']

    fig, ax = plt.subplots(figsize=(9, 5))
    data = []
    labels = []
    colors_list = []
    for m in models:
        f1s = [float(r['f1']) for r in rows if r['model'] == m]
        if f1s:
            data.append(f1s)
            labels.append(f'{DISPLAY_NAMES[m]}\n(n={len(f1s)})')
            colors_list.append(COLORS[m])

    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=True,
                    flierprops=dict(marker='o', markersize=4, alpha=0.5))
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score Distribution by Model (Expanded GT)')
    ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.3)
    ax.grid(axis='y', alpha=0.2)
    fig.savefig(FIGURES_DIR / 'fig11_f1_boxplot.png')
    plt.close(fig)
    print('  fig11_f1_boxplot.png')


def fig12_category_cer(rows):
    """Grouped bar chart of CER by category."""
    models = ['tesseract', 'surya', 'sarvam_ocr', 'docling']
    categories = [
        ('02_complex_tables/financial', 'Financial'),
        ('02_complex_tables/forms', 'Forms'),
        ('02_complex_tables/multi_column', 'Multi-col'),
        ('06_mixed_content/equations_formulas', 'Equations'),
        ('06_mixed_content/receipts', 'Receipts'),
    ]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(categories))
    width = 0.2
    offsets = np.arange(len(models)) * width - width * (len(models)-1) / 2

    for i, m in enumerate(models):
        vals = []
        for cat, _ in categories:
            mr = [float(r['cer']) for r in rows if r['model'] == m and r['category'] == cat]
            vals.append(np.mean(mr) if mr else 0)
        ax.bar(x + offsets[i], vals, width, label=DISPLAY_NAMES[m],
               color=COLORS[m], alpha=0.85, edgecolor='white')

    ax.set_ylabel('Average CER (lower is better)')
    ax.set_title('CER by Category and Model (Expanded GT)')
    ax.set_xticks(x)
    ax.set_xticklabels([c[1] for c in categories])
    ax.legend()
    ax.set_ylim(0, 1.1)
    fig.savefig(FIGURES_DIR / 'fig12_category_cer.png')
    plt.close(fig)
    print('  fig12_category_cer.png')


def fig13_gt_coverage():
    """Pie chart of GT coverage."""
    categories = {
        'Forms (FUNSD)': 5,
        'Financial (consensus)': 9,
        'Multi-column (consensus)': 11,
        'Equations (consensus)': 10,
        'Receipts (Mistral-derived)': 10,
        'No GT': 46,
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#f39c12', '#e74c3c', '#bdc3c7']
    wedges, texts, autotexts = ax.pie(
        categories.values(), labels=categories.keys(), autopct='%1.0f%%',
        colors=colors, startangle=90, pctdistance=0.8,
        textprops={'fontsize': 9}
    )
    for t in autotexts:
        t.set_fontsize(8)

    ax.set_title('Ground Truth Coverage (45 / 91 docs = 49.5%)')
    fig.savefig(FIGURES_DIR / 'fig13_gt_coverage.png')
    plt.close(fig)
    print('  fig13_gt_coverage.png')


def main():
    rows, summaries = load_data()
    print('Generating charts...')

    fig1_f1_comparison(rows, summaries)
    fig2_cer_comparison(rows, summaries)
    fig3_category_f1_heatmap(rows)
    fig4_error_decomposition(summaries)
    fig5_precision_recall(rows, summaries)
    fig6_success_rates(rows)
    fig7_category_success_heatmap()
    fig8_forms_comparison(rows)
    fig9_radar(summaries)
    fig10_significance(rows)
    fig11_boxplot_f1(rows)
    fig12_category_cer(rows)
    fig13_gt_coverage()

    print(f'\nAll 13 charts saved to {FIGURES_DIR}/')


if __name__ == '__main__':
    main()
