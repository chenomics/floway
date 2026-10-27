"""Gating: from raw events to singlets.

The gate has three steps, in this order:

1. events at the detector ``ceiling`` in FSC-A or SSC-A are labelled
   ``'saturated'`` and set aside -- their pulse area is clipped, which would
   break step 3;
2. **cells vs debris** from a 2-component GMM on log10(FSC-A, SSC-A);
3. **singlets vs doublets** from ``log10(FSC-A / FSC-H)``, cut at the valley
   between the two modes. Doublets carry extra pulse area for their height, so
   this is where they separate -- FSC-A/SSC-A alone does not show them.

Two ways in:

* :class:`PlateGate` -- fit **once** on a whole run and apply the identical gate
  to every well. Preferred when all wells are the same cell type acquired in one
  sitting, because the fraction of events kept then reflects the sample rather
  than how a per-well fit happened to land.
* :func:`gate_populations` / :func:`analyze_fcs_data` -- fit one sample on its
  own. Use when the population genuinely shifts between samples.

Before v0.3 gating fitted a 3-component GMM per well and called the middle-FSC
cluster the singlets. That was removed: on an Attune plate it kept between 52%
and 93% of events depending on the well, and FSC-H showed 43% of the events in
its "doublet" component were in fact singlets.
"""

import warnings

import numpy as np
import pandas as pd

from .parser import FCSParser
from .plotting import plot_gating_strategy

__all__ = ["PlateGate", "gate_populations", "gate_singlets_mahalanobis",
           "analyze_fcs_data"]


CEILING = 1_048_575  # 2^20 - 1, the Attune detector maximum


def _as_frame(source, columns=None):
    """Accept a DataFrame or a path to an FCS file."""
    if isinstance(source, pd.DataFrame):
        return source
    return FCSParser(str(source)).get_data()


class PlateGate:
    """A cell + singlet gate fitted once on a whole plate.

    Attributes:
        gmm: The fitted 2-component :class:`sklearn.mixture.GaussianMixture`.
        cell_component (int): Index of the component holding the cells.
        ratio_cut (float): Cut on ``log10(FSC-A / FSC-H)``; below it is a singlet.
        modes (tuple): The singlet and doublet modes the cut sits between,
            in log10 units.
        n_pooled (int): Events used for the fit.
    """

    def __init__(self, gmm, cell_component, ratio_cut, modes, n_pooled,
                 fsc_col, ssc_col, fsch_col, cell_posterior, ceiling):
        self.gmm = gmm
        self.cell_component = cell_component
        self.ratio_cut = ratio_cut
        self.modes = modes
        self.n_pooled = n_pooled
        self.fsc_col, self.ssc_col, self.fsch_col = fsc_col, ssc_col, fsch_col
        self.cell_posterior = cell_posterior
        self.ceiling = ceiling

    # -- fitting ---------------------------------------------------------
    @classmethod
    def fit(cls, sources, fsc_col='FSC-A', ssc_col='SSC-A', fsch_col='FSC-H',
            events_per_source=5_000, cell_posterior=0.5, ceiling=CEILING,
            random_state=0, smoothing=4):
        """Fit the gate on the pooled events of a run.

        Args:
            sources (Iterable): FCS paths or DataFrames, one per well.
            fsc_col, ssc_col, fsch_col (str): Channel names.
            events_per_source (int): Events sampled per well for the fit.
            cell_posterior (float): Posterior cut for the cell component.
            ceiling (int): Detector maximum; events at it are excluded.
            random_state (int): Seed for the GMM and the subsampling.
            smoothing (float): Gaussian smoothing (in bins) of the ratio
                histogram before the valley search.

        Returns:
            PlateGate
        """
        from sklearn.mixture import GaussianMixture

        cols = [fsc_col, ssc_col, fsch_col]
        rng = np.random.default_rng(random_state)
        frames = []
        for src in sources:
            d = _as_frame(src)
            missing = [c for c in cols if c not in d.columns]
            if missing == [fsch_col]:
                warnings.warn(f'{fsch_col} not in the data: doublets cannot be separated, '
                              'the gate is cells-vs-debris only', RuntimeWarning)
                d = d[[fsc_col, ssc_col]].assign(**{fsch_col: np.nan})
            elif missing:
                raise ValueError(f'missing channels {missing} in {src}')
            else:
                d = d[cols]
            if len(d) > events_per_source:
                d = d.iloc[rng.choice(len(d), events_per_source, replace=False)]
            frames.append(d)
        if not frames:
            raise ValueError('no sources to fit on')
        pool = pd.concat(frames, ignore_index=True)

        gate = cls(None, 0, 0.0, (np.nan, np.nan), len(pool),
                   fsc_col, ssc_col, fsch_col, cell_posterior, ceiling)

        keep = gate._unsaturated(pool)
        if keep.sum() < 100:
            raise ValueError('too few unsaturated events to fit')
        gmm = GaussianMixture(n_components=2, covariance_type='full',
                              random_state=random_state, n_init=3)
        gmm.fit(gate._log_scatter(pool[keep]))
        gate.gmm = gmm
        gate.cell_component = int(np.argmax(gmm.means_[:, 0]))

        if pool[fsch_col].isna().all():
            gate.ratio_cut, gate.modes = float('inf'), (float('nan'), float('nan'))
            return gate
        ratio = gate._ratio(pool[gate.cells(pool)])
        try:
            gate.ratio_cut, gate.modes = _valley(ratio, smoothing)
        except ValueError:
            # No doublet mode to separate (can happen on a single well, or a clean sample).
            # Keep every cell rather than inventing a cut, and say so.
            warnings.warn('no doublet mode found in log10(FSC-A/FSC-H); '
                          'keeping all cells as singlets', RuntimeWarning)
            gate.ratio_cut, gate.modes = float('inf'), (float('nan'), float('nan'))
        return gate

    # -- internals -------------------------------------------------------
    def _log_scatter(self, d):
        return np.log10(np.maximum(d[[self.fsc_col, self.ssc_col]].to_numpy(float), 1.0))

    def _unsaturated(self, d):
        return ((d[self.fsc_col] < self.ceiling) &
                (d[self.ssc_col] < self.ceiling)).to_numpy()

    def _ratio(self, d):
        if self.fsch_col not in d.columns:
            # No height channel: doublets cannot be separated, so nothing is cut here.
            return np.full(len(d), -np.inf)
        return (np.log10(np.maximum(d[self.fsc_col].to_numpy(float), 1.0)) -
                np.log10(np.maximum(d[self.fsch_col].to_numpy(float), 1.0)))

    # -- applying --------------------------------------------------------
    def cells(self, source):
        """Boolean mask: unsaturated events in the cell (not debris) cluster."""
        d = _as_frame(source)
        post = self.gmm.predict_proba(self._log_scatter(d))[:, self.cell_component]
        return self._unsaturated(d) & (post > self.cell_posterior)

    def apply(self, source):
        """Boolean mask of singlets: cells whose FSC-A/FSC-H is below the cut."""
        d = _as_frame(source)
        return self.cells(d) & (self._ratio(d) < self.ratio_cut)

    def gated(self, source):
        """The singlet events of one well, as a DataFrame."""
        d = _as_frame(source)
        return d[self.apply(d)]

    def report(self, source):
        """Per-population counts for one well, as a dict."""
        d = _as_frame(source)
        sat = ~self._unsaturated(d)
        cells = self.cells(d)
        singlets = cells & (self._ratio(d) < self.ratio_cut)
        n = len(d)
        return {'events': n, 'saturated': int(sat.sum()),
                'debris': int((~sat & ~cells).sum()),
                'cells': int(cells.sum()),
                'doublets': int((cells & ~singlets).sum()),
                'singlets': int(singlets.sum()),
                'pct_saturated': 100 * sat.sum() / n,
                'pct_cells': 100 * cells.sum() / n,
                'pct_singlets_of_cells': 100 * singlets.sum() / max(cells.sum(), 1),
                'pct_singlets': 100 * singlets.sum() / n}

    def labels(self, source):
        """Per-event label: 'saturated', 'debris', 'doublet' or 'singlet'."""
        d = _as_frame(source)
        sat = ~self._unsaturated(d)
        cells = self.cells(d)
        singlet = cells & (self._ratio(d) < self.ratio_cut)
        out = np.where(singlet, 'singlet', np.where(cells, 'doublet', 'debris'))
        return np.where(sat, 'saturated', out)

    def plot(self, source, axes=None, title=None, bins=90,
             fsc_lim=(2e4, 1.1e6), ssc_lim=(5e3, 1.1e6), fsch_lim=(5e4, 6e5)):
        """Two-panel gating plot: the cell gate, then the singlet gate.

        Returns:
            (fig, axes)
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        d = _as_frame(source)
        if axes is None:
            fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.2))
        else:
            fig = np.ravel(axes)[0].figure
        ax0, ax1 = np.ravel(axes)[:2]

        def _density(ax, x, y, xlim, ylim, cmap):
            bx = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), bins)
            by = np.logspace(np.log10(ylim[0]), np.log10(ylim[1]), bins)
            h, _, _ = np.histogram2d(x, y, [bx, by])
            ax.pcolormesh(bx, by, np.ma.masked_equal(h.T, 0), cmap=cmap,
                          norm=LogNorm(), rasterized=True, linewidth=0)
            ax.set(xscale='log', yscale='log', xlim=xlim, ylim=ylim)

        _density(ax0, d[self.fsc_col].to_numpy(float), d[self.ssc_col].to_numpy(float),
                 fsc_lim, ssc_lim, 'Blues')
        gx, gy = np.meshgrid(np.linspace(*np.log10(fsc_lim), 300),
                             np.linspace(*np.log10(ssc_lim), 300))
        post = self.gmm.predict_proba(
            np.column_stack([gx.ravel(), gy.ravel()]))[:, self.cell_component]
        ax0.contour(10 ** gx, 10 ** gy, post.reshape(gx.shape),
                    levels=[self.cell_posterior], colors='k', linewidths=0.6)
        ax0.set(xlabel=self.fsc_col, ylabel=self.ssc_col)

        c = d[self.cells(d)]
        _density(ax1, c[self.fsc_col].to_numpy(float), c[self.fsch_col].to_numpy(float),
                 fsc_lim, fsch_lim, 'Purples')
        fa = np.logspace(*np.log10(fsc_lim), 50)
        ax1.plot(fa, fa / 10 ** self.ratio_cut, color='k', lw=0.6)
        ax1.set(xlabel=self.fsc_col, ylabel=self.fsch_col)

        r = self.report(d)
        ax0.set_title(f'cells {r["pct_cells"]:.0f}%  (sat. {r["pct_saturated"]:.0f}% excl.)',
                      fontsize=7)
        ax1.set_title(f'singlets {r["pct_singlets_of_cells"]:.0f}% of cells', fontsize=7)
        if title:
            fig.suptitle(title, fontsize=8)
        fig.tight_layout()
        return fig, (ax0, ax1)

    def summary(self, sources, labels=None):
        """:meth:`report` over many wells, as a DataFrame."""
        labels = labels if labels is not None else [str(s) for s in sources]
        return pd.DataFrame([{'source': l, **self.report(s)}
                             for l, s in zip(labels, sources)])

    def __repr__(self):
        m = self.gmm.means_[self.cell_component]
        return (f'PlateGate(fit on {self.n_pooled:,} events; '
                f'cells at log10 FSC/SSC {m.round(2).tolist()}, '
                f'posterior>{self.cell_posterior}; '
                f'FSC-A/FSC-H cut {10 ** self.ratio_cut:.2f} '
                f'between modes {10 ** self.modes[0]:.2f} and {10 ** self.modes[1]:.2f})')


def _valley(ratio, smoothing=4, lo=-0.1, hi=0.7, bins=400, split=0.3):
    """Cut between the singlet and doublet modes of log10(FSC-A/FSC-H)."""
    from scipy.ndimage import gaussian_filter1d

    h, edges = np.histogram(ratio, bins, range=(lo, hi))
    hs = gaussian_filter1d(h.astype(float), smoothing)
    centers = (edges[:-1] + edges[1:]) / 2
    m1 = centers[np.argmax(np.where(centers < split, hs, 0))]
    m2 = centers[np.argmax(np.where(centers > split + 0.05, hs, 0))]
    between = (centers > m1) & (centers < m2)
    if not between.any():
        raise ValueError('no valley found between the FSC-A/FSC-H modes')
    return float(centers[between][np.argmin(hs[between])]), (float(m1), float(m2))


def gate_populations(df, fsc_col='FSC-A', ssc_col='SSC-A', confidence_level=None,
                     random_state=0, min_events=100, fsch_col='FSC-H',
                     ceiling=1_048_575):
    """
    Classify events as debris, doublet, singlet (or saturated) for ONE sample.

    Runs the :class:`~floway.gating.PlateGate` algorithm:

    1. events at the detector ``ceiling`` in FSC-A or SSC-A are labelled
       ``'saturated'`` and excluded (their pulse area is clipped, which breaks
       step 3);
    2. cells vs debris from a **2-component** GMM on log10(FSC-A, SSC-A);
    3. singlets vs doublets from ``log10(FSC-A / FSC-H)``, cut at the valley
       between the two modes.

    Before v0.3 this fitted a 3-component GMM and called the middle-FSC cluster
    the singlets. That algorithm was removed in v0.3: FSC-A/SSC-A does not
    separate doublets, and on an Attune plate FSC-H showed 43% of the events in
    the highest-FSC component were singlets, so gating that component out
    discarded real cells.

    For a whole plate, prefer :class:`~floway.gating.PlateGate`, which fits
    **once** on all wells pooled and applies the identical gate to each. Fitting
    per well (what this function does) lets the gate drift between wells.

    Args:
        df (pd.DataFrame): Full (un-gated) event table.
        fsc_col, ssc_col, fsch_col (str): Channel names. Without ``fsch_col``
            the doublet step is skipped and a warning is issued.
        confidence_level: Deprecated and ignored -- it set the Mahalanobis
            ellipse of the removed 3-component gate, which has no counterpart
            here. Passing it warns.
        random_state (int): Seed for the GMM.
        min_events (int): Minimum event count required to fit.
        ceiling (int): Detector maximum; events at it are excluded.

    Returns:
        tuple:
            - labels (np.ndarray[str]): ``'saturated'``, ``'debris'``,
              ``'doublet'`` or ``'singlet'`` per event.
            - singlet_mask (pd.Series[bool]): ``True`` for singlets.
    """
    if confidence_level is not None:
        warnings.warn('confidence_level is ignored: the 3-component gate it belonged to was '
                      'removed in floway 0.3. Drop the argument.',
                      DeprecationWarning, stacklevel=2)
    if fsc_col not in df.columns or ssc_col not in df.columns:
        raise ValueError(
            f"Missing scatter columns. Need '{fsc_col}' and '{ssc_col}', "
            f"got: {list(df.columns)}"
        )
    if len(df) < min_events:
        raise ValueError(f"Too few events ({len(df)}) to fit a GMM reliably.")

    gate = PlateGate.fit([df], fsc_col=fsc_col, ssc_col=ssc_col, fsch_col=fsch_col,
                         events_per_source=len(df), ceiling=ceiling,
                         random_state=random_state)
    labels = gate.labels(df)
    return labels, pd.Series(labels == 'singlet', index=df.index)


def gate_singlets_mahalanobis(df, fsc_col='FSC-A', ssc_col='SSC-A',
                               confidence_level=0.95):
    """
    Gate singlets using a Mahalanobis ellipse fitted to the core of an
    already-cleaned (post-debris) population.

    Useful as a standalone step when you want to tighten a gate within an
    already-separated singlet cloud.  For full gating from raw FCS data,
    prefer :func:`gate_populations` which handles debris and doublets too.

    Args:
        df (pd.DataFrame): Event data (ideally post-debris-removal).
        fsc_col (str): Forward-scatter column name.
        ssc_col (str): Side-scatter column name.
        confidence_level (float): Chi-squared confidence level (0–1).

    Returns:
        tuple:
            - pd.Series[bool]: ``True`` for singlet events.
            - dict: Ellipse params (``center``, ``width``, ``height``,
              ``angle``) for plotting, or ``{}`` on failure.
    """
    if df.empty or fsc_col not in df.columns or ssc_col not in df.columns:
        return pd.Series([False] * len(df), index=df.index), {}

    X = df[[fsc_col, ssc_col]].values

    q1_fsc, q3_fsc = df[fsc_col].quantile(0.01), df[fsc_col].quantile(0.99)
    q1_ssc, q3_ssc = df[ssc_col].quantile(0.01), df[ssc_col].quantile(0.99)
    core = df[
        (df[fsc_col] > q1_fsc) & (df[fsc_col] < q3_fsc) &
        (df[ssc_col] > q1_ssc) & (df[ssc_col] < q3_ssc)
    ]
    if core.empty:
        return pd.Series([False] * len(df), index=df.index), {}

    mu  = core[[fsc_col, ssc_col]].mean().values
    cov = core[[fsc_col, ssc_col]].cov().values
    inv_cov = np.linalg.inv(cov)
    delta = X - mu
    maha_sq = np.sum(delta @ inv_cov * delta, axis=1)

    from scipy.stats import chi2
    threshold = chi2.ppf(confidence_level, df=2)
    mask = pd.Series(maha_sq < threshold, index=df.index)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * np.sqrt(threshold * eigenvalues)
    ellipse = {'center': mu, 'width': width, 'height': height, 'angle': angle}

    return mask, ellipse


def analyze_fcs_data(filepath, fsc_col='FSC-A', ssc_col='SSC-A',
                     confidence_level=None, visualize=True, report=False,
                     gate=None):
    """
    Parse an FCS file and return the singlet population.

    Uses :func:`gate_populations` to classify every event as saturated, debris,
    doublet or singlet, then returns only the singlets.

    Pass ``gate`` (a fitted :class:`~floway.gating.PlateGate`) to apply one
    plate-wide gate instead of fitting this file on its own -- the better choice
    when every well of a run is the same cell type.

    Args:
        filepath (str): Path to the ``.fcs`` file.
        fsc_col (str): Forward-scatter column name.
        ssc_col (str): Side-scatter column name.
        confidence_level: Deprecated and ignored (see
            :func:`~floway.gating.gate_populations`).
        visualize (bool): If ``True``, plots the gating scatter.
        gate (PlateGate, optional): A gate already fitted on the whole run.
        report (bool): If ``True``, prints per-population event counts.

    Returns:
        pd.DataFrame: Events classified as singlets.
    """
    parser = FCSParser(filepath)
    full_data = parser.get_data()

    if full_data is None or full_data.empty:
        print("No data found in the file.")
        return pd.DataFrame()

    available_cols = full_data.columns
    fsc_col = next((col for col in [fsc_col, 'FSC', 'FSCA'] if col in available_cols), None)
    ssc_col = next((col for col in [ssc_col, 'SSC', 'SSCA'] if col in available_cols), None)

    if not fsc_col or not ssc_col:
        print("Error: Could not find standard FSC/SSC columns in the data.")
        return full_data

    if gate is not None:
        labels = gate.labels(full_data)
        singlet_mask = pd.Series(labels == 'singlet', index=full_data.index)
    else:
        labels, singlet_mask = gate_populations(
            full_data, fsc_col=fsc_col, ssc_col=ssc_col,
            confidence_level=confidence_level,
        )
    gated_data = full_data[singlet_mask]

    if report:
        total = len(full_data)
        print(f"--- {filepath} ---")
        for pop in ('saturated', 'debris', 'doublet', 'singlet'):
            n = int((labels == pop).sum())
            print(f"  {pop:8s}: {n:6,}  ({100 * n / total:.1f}%)")

    if visualize:
        plot_gating_strategy(full_data, fsc_col, ssc_col, labels)

    return gated_data
