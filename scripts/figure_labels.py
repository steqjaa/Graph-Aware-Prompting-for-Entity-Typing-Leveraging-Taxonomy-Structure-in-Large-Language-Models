"""Consistent numeric labels; full precision remains in accompanying CSVs."""
import re
import numpy as np
from matplotlib.container import BarContainer
from matplotlib.collections import PathCollection

def label_values(fig):
    for ax in fig.axes:
        for text in list(ax.texts):
            if text.get_gid() != 'heatmap_value' and re.fullmatch(r'[+-]?\d+\.\d+', text.get_text()):
                text.remove()
        for container in ax.containers:
            if not isinstance(container, BarContainer):
                continue
            values = container.datavalues
            ax.bar_label(container, labels=[f'{v:.4f}' for v in values], padding=4, fontsize=9)
            if container.orientation == 'horizontal':
                lo, hi = ax.get_xlim()
                ax.set_xlim(lo, max(hi, float(np.nanmax(values)) + .09))
            else:
                lo, hi = ax.get_ylim()
                ax.set_ylim(lo, max(hi, float(np.nanmax(values)) + .09))
        for collection in ax.collections:
            if not isinstance(collection, PathCollection):
                continue
            for x, y in collection.get_offsets():
                if not np.isfinite([x,y]).all():
                    continue
                depth = 'target depth' in ax.get_xlabel().lower()
                value = y if depth else x
                ax.annotate(f'{value:+.4f}', (x,y), xytext=(6,8),
                    textcoords='offset points', fontsize=8,
                    rotation=55 if depth else 0)

def numeric_heatmap(ax, frame, title, xlabel):
    """Dense compositions use labelled cells rather than unreadable tiny bars."""
    values = frame.to_numpy(dtype=float)
    ax.imshow(values, aspect='auto', cmap='Blues', vmin=0, vmax=1)
    ax.set(xticks=range(len(frame.columns)), xticklabels=frame.columns,
           yticks=range(len(frame)), yticklabels=frame.index,
           title=title, xlabel=xlabel)
    ax.tick_params(axis='x', labelrotation=35)
    for (i,j), value in np.ndenumerate(values):
        ax.text(j,i,f'{value:.4f}',ha='center',va='center',fontsize=9,
                color='white' if value>.55 else '#122f40', gid='heatmap_value')
