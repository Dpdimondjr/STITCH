import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 3, figsize=(17, 6), constrained_layout=True)
blue, green, gray, orange = '#4472C4', '#2E8B57', '#8A8F98', '#D97A22'

def panel(ax, title, labels, values, xmax, note, best_index):
    y = list(range(len(labels)))
    colors = [green if i == best_index else blue for i in y]
    ax.barh(y, values, color=colors, height=.58)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, xmax)
    ax.set_xlabel('Median per-star scatter (%)')
    ax.set_title(title, loc='left', weight='bold')
    for i, v in enumerate(values):
        ax.text(v + xmax*.018, i, f'{v:.3f}%', va='center', fontsize=10,
                weight='bold' if i == best_index else 'normal')
    ax.text(0, -0.18, note, transform=ax.transAxes, fontsize=9, color='#555')
    ax.grid(axis='x', alpha=.3); ax.grid(axis='y', visible=False)

panel(axes[0], 'Main diagnostic split',
      ['Raw', 'Median-KNN', 'Residual HGB', 'Robust global'],
      [1.504, 1.217, 1.147, .980], 1.75,
      'Lower is better · 37,372 train stars · robust, untruncated rerun', 3)
panel(axes[1], 'Future-sector simulation',
      ['Raw', 'Median-KNN', 'Residual HGB', 'Robust global'],
      [1.478, 1.036, .970, .820], 1.75,
      'Train Sectors 1–59 · test Sectors 68–85 · 955 science stars', 3)
panel(axes[2], 'Feature / model checks',
      ['Spatial surface', 'FLFRCSAP*', 'Latent modes', 'Pairwise NSF'],
      [1.109, .877, .982, 1.045], 1.25,
      '*FLFRCSAP is a complete-case result; not directly comparable', 0)

fig.suptitle('STITCH experiment results — the robust global calibration model is best',
             fontsize=16, weight='bold')
fig.text(.5, .01,
         'Green = best within panel. Results are diagnostic until signal preservation is tested on a fresh holdout.',
         ha='center', fontsize=10, color='#555')
fig.savefig('eval/stitch_results_chart.png', dpi=220, bbox_inches='tight')
print('saved eval/stitch_results_chart.png')
