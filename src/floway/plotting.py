"""Plotting functions for flow cytometry analysis."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

_POP_COLORS = {
    'debris':    'lightgray',
    'singlet':   'cornflowerblue',
    'doublet':   '#ffa44f',
    'saturated': '#c86bd8',
}

def plot_gating_strategy(df, fsc_col, ssc_col, labels):
    """
    Scatter plot of FSC vs SSC with each event colored by its GMM population label.

    Args:
        df (pd.DataFrame): Full event table (same rows as ``labels``).
        fsc_col (str): Forward-scatter column name.
        ssc_col (str): Side-scatter column name.
        labels (array-like of str): Per-event population label —
            ``'debris'``, ``'singlet'``, ``'doublet'`` or ``'saturated'``.
            Returned by :func:`floway.gate_populations`.
    """
    labels = np.asarray(labels)
    fig, ax = plt.subplots(figsize=(4, 4))
    for pop, color in _POP_COLORS.items():
        mask = labels == pop
        n = mask.sum()
        if n == 0:
            continue
        pct = 100.0 * n / len(labels) if len(labels) else 0
        ax.scatter(df.loc[mask, fsc_col], df.loc[mask, ssc_col],
                   s=1, c=color, alpha=0.6,
                   label=f'{pop}  {n:,} ({pct:.0f}%)')
    ax.set_xlim(0, 1e6)
    ax.set_ylim(0, 1e6)
    ax.set_xlabel(fsc_col, fontsize=8)
    ax.set_ylabel(ssc_col, fontsize=8)
    ax.set_title('GMM population gate\n(debris / singlet / doublet)', fontsize=10)
    ax.legend(markerscale=4, fontsize=7)
    plt.tight_layout()
    plt.show()



def plot_kde(df, unique_conds, remove_axes=True, count=False, export=False, ylim=None):
    """
    Plots KDE plots for each unique condition in a 3-column subplot layout.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        unique_conds (list): List of unique conditions to plot.
        remove_axes (bool): If True, hides redundant axes in the subplot grid.
        count (bool): If True, annotates each subplot with the event count.
        export (str | False): If a path is provided, saves the figure there instead of showing.
        ylim (tuple, optional): Optional y-axis limits applied to every subplot.
    """
    import matplotlib.pyplot as plt
    num_tads = len(unique_conds)
    num_cols = 3
    num_rows = (num_tads + num_cols - 1) // num_cols  # Calculate rows needed
    fig, axs = plt.subplots(num_rows, num_cols,
                            figsize=(2 * num_cols, max(2.0, 0.5 * num_rows)),
                            sharex=True, sharey=True)

    axs = np.array(axs).flatten()
    palette = ['#77dd77', '#f98071', '#84b6f4', '#81c29d', '#ffe389', '#5fa876', '#ffa2ba', '#ff6960',
               '#d4c7f1', '#ffd1dc', '#ffdb57', '#45cea1', '#86ceeb', '#ffb346', '#e7d4e7', '#dfddac',
               '#ffa44f', '#afdfe6', '#f6d6c8', '#ccccff', '#efe68b', '#40dfd0', '#f4a35f']

    for i, cond in enumerate(unique_conds):
        subset = df[df['sample'] == cond]
        total_count = len(subset)
        sns.kdeplot(data=subset, x="GFP_signal", fill=True, common_norm=True, log_scale=True, alpha=0.5, color=palette[i % len(palette)], ax=axs[i])
        axs[i].set_xscale('log')
        axs[i].set_ylabel('')
        axs[i].set_xlabel('GFP signal',fontsize=7)
        if count:
            axs[i].text(0.95, 0.85, f'n={total_count}', transform=axs[i].transAxes, fontsize=6, verticalalignment='top', horizontalalignment='right')
        axs[i].text(0.5, 0.85, cond, transform=axs[i].transAxes, fontsize=6, verticalalignment='top', horizontalalignment='center')
        axs[i].set_xlim(1e2, 4e6)
        axs[i].tick_params(axis='both', labelsize=6)
        if ylim:
            axs[i].set_ylim(ylim)
        sns.despine(offset=3, ax=axs[i])
        if remove_axes:
            yaxis_remove = [s for s in range(len(axs)) if s not in [i for i in range(0, len(axs), num_cols)]]
            xaxis_remove = [s for s in range(len(axs)) if s not in [i for i in range(len(axs)-3,len(axs))]]
            if i in yaxis_remove and i in xaxis_remove:
                axs[i].get_xaxis().set_visible(False)
                axs[i].get_yaxis().set_visible(False)
                axs[i].spines['bottom'].set_visible(False)
                axs[i].spines['left'].set_visible(False)
            elif i in yaxis_remove:
                axs[i].get_yaxis().set_visible(False)
                axs[i].spines['left'].set_visible(False)
            elif i in xaxis_remove:
                axs[i].get_xaxis().set_visible(False)
                axs[i].spines['bottom'].set_visible(False)
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    if export:
        plt.savefig(export)
    else:
        plt.show()


def plot_kde_grid(df, conditions, signal_col='signal', sample_col='sample',
                  neg_signal=None, threshold=None,
                  ncols=4, log_scale=True, signal_range=(1e2, 4e6),
                  title=None, annotate_pct=True, show_title_per_panel=True,
                  panel_height=1.2, panel_width=2.5, export=None):
    """
    Per-condition KDE grid with optional negative-control reference and
    positivity threshold.

    One panel per condition (in the order supplied). On top of each panel:
      * the negative-control KDE in gray dashed (reference baseline);
      * a vertical dotted line at ``threshold`` (positivity cutoff);
      * the % of cells above ``threshold`` annotated at top-right.

    Use this for activation-style readouts where you care about both
    the shift of the distribution *and* how many cells crossed a cutoff.

    Args:
        df (pd.DataFrame): Long-format data with one row per cell.
        conditions (Iterable[str]): Sample / condition names to plot,
            in the desired panel order.
        signal_col (str): Column holding the fluorescence intensity.
        sample_col (str): Column holding the per-cell condition label.
        neg_signal (array-like or list, optional): Either a single 1D array
            of negative-control signal values (broadcast as a gray dashed
            KDE on every panel), or a list/tuple the same length as
            ``conditions`` giving a different reference array per panel
            (e.g. each condition's own matched negative control).
        threshold (float or list, optional): Positivity threshold(s). Either
            a single value (broadcast to every panel) or a list/tuple the
            same length as ``conditions`` (one threshold per panel). Drawn
            as a vertical dotted line; drives the % positive annotation.
        ncols (int): Subplot columns.
        log_scale (bool): If True, x-axis is log.
        signal_range (tuple): (xmin, xmax) clip.
        title (str, optional): Overall figure title.
        annotate_pct (bool): Show "% positive" in the upper-right of each
            panel. Requires ``threshold``.
        show_title_per_panel (bool): Use the condition name as the panel
            title.
        panel_height, panel_width (float): Per-panel size in inches.
        export (str | False): If a path is provided, saves the figure
            there instead of showing.

    Returns:
        (fig, axes): The created Matplotlib figure and the flat array of
        axes (length nrows*ncols, with empty cells turned off).

    Example:
        >>> from floway import plot_kde_grid
        >>> plot_kde_grid(
        ...     df, conditions=summary['condition'].tolist(),
        ...     signal_col='PE_A', sample_col='condition',
        ...     neg_signal=neg_pe,
        ...     threshold=T_exp,
        ...     title='REP_1 (02-20-2026) - PE-A per condition',
        ... )

        Per-panel backgrounds (one negative control / threshold per
        condition, e.g. each reporter's own matched control)::

        >>> plot_kde_grid(
        ...     df, conditions=conditions,
        ...     signal_col='gfp', sample_col='condition',
        ...     neg_signal=[neg_by_condition[c] for c in conditions],
        ...     threshold=[t_by_condition[c] for c in conditions],
        ... )
    """
    conditions = list(conditions)
    n = len(conditions)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(panel_width * ncols,
                                       max(2.0, panel_height * nrows)),
                              sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)
    palette = sns.color_palette('Set2', n)

    def _per_panel(param, i):
        # A list/tuple matching len(conditions) is one value per panel;
        # anything else (a bare array, a scalar, None) is broadcast as-is.
        if isinstance(param, (list, tuple)) and len(param) == n:
            return param[i]
        return param

    last_drawn = -1
    for i, cond in enumerate(conditions):
        ax = axes[i]
        last_drawn = i
        sig = df.loc[df[sample_col] == cond, signal_col].values.astype(float)
        sig = sig[sig > 0] if log_scale else sig

        ng = _per_panel(neg_signal, i)
        if ng is not None:
            ng = np.asarray(ng, dtype=float)
            ng = ng[ng > 0] if log_scale else ng
            if len(ng) > 1:
                sns.kdeplot(x=ng, log_scale=log_scale, ax=ax, color='gray',
                            linestyle='--', linewidth=0.8, fill=False)
        if len(sig) > 1:
            sns.kdeplot(x=sig, log_scale=log_scale, ax=ax, color=palette[i],
                        fill=True, alpha=0.5, linewidth=0.8)
        thr = _per_panel(threshold, i)
        if thr is not None:
            ax.axvline(thr, color='black', ls=':', lw=0.6)
            if annotate_pct and len(sig):
                pct = 100.0 * (sig > thr).mean()
                ax.text(0.97, 0.85, f'{pct:.1f}%',
                        transform=ax.transAxes, ha='right',
                        fontsize=6, color='black')
        if show_title_per_panel:
            ax.set_title(str(cond), fontsize=6)
        ax.set_xlim(*signal_range)
        ax.set_xlabel(signal_col, fontsize=6)
        ax.set_ylabel('')
        ax.tick_params(labelsize=5)

    for j in range(last_drawn + 1, len(axes)):
        axes[j].axis('off')

    if title:
        fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    if export:
        plt.savefig(export)
    else:
        plt.show()
    return fig, axes
