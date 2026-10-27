"""Automatic FSC/SSC gating via Gaussian Mixture Models."""

import numpy as np
import pandas as pd


def auto_thresholds(df, fsc_col='FSC-A', ssc_col='SSC-A',
                    method='gmm', quantile=0.05,
                    min_fsc=1.0, min_ssc=1.0, random_state=42):
    """
    Auto-pick FSC/SSC thresholds that separate the cell cloud from debris.

    Fits a 2-component Gaussian Mixture Model on ``log10(FSC, SSC)`` and
    treats the higher-mean cluster as the cell population. Returns
    thresholds at the ``quantile`` of that cluster along each axis.

    The result is intended for use with :func:`floway.analyze_fcs_data`
    (or its lower-level pieces) as a starting point — you can still tune
    by hand if a particular sample has unusual debris.

    Args:
        df (pd.DataFrame): Event data, e.g. from ``FCSParser.get_data()``.
            Should be the full (un-gated) event table.
        fsc_col (str): Forward-scatter column name.
        ssc_col (str): Side-scatter column name.
        method (str): Currently only ``"gmm"``. Kept as a parameter for
            forward compatibility.
        quantile (float): Quantile of the cell cluster used as the lower
            threshold along each axis. ``0.05`` keeps 95% of the cell
            cluster and excludes the dimmest 5%.
        min_fsc, min_ssc (float): Lower clamps for log transform. Values
            ``<= min_*`` are floored to ``min_*`` before the log so the
            GMM doesn't choke on zeros / negatives.
        random_state (int): Seed for the GMM init.

    Returns:
        tuple[float, float]: ``(fsc_threshold, ssc_threshold)``.

    Raises:
        ValueError: If sklearn is not installed, or if there are fewer
            than 100 events to fit.

    Example:
        >>> from floway import FCSParser, auto_thresholds, analyze_fcs_data
        >>> df = FCSParser('sample.fcs').get_data()
        >>> fsc_t, ssc_t = auto_thresholds(df)
        >>> gated = analyze_fcs_data('sample.fcs',
        ...                          fsc_threshold=fsc_t, ssc_threshold=ssc_t)
    """
    if method != 'gmm':
        raise ValueError(f"Unknown method '{method}'. Supported: 'gmm'.")

    try:
        from sklearn.mixture import GaussianMixture
    except ImportError as e:
        raise ValueError(
            "auto_thresholds requires scikit-learn. Install with: "
            "pip install scikit-learn"
        ) from e

    if fsc_col not in df.columns or ssc_col not in df.columns:
        raise ValueError(
            f"Missing scatter columns. Need '{fsc_col}' and '{ssc_col}', "
            f"got: {list(df.columns)}"
        )

    fsc = np.asarray(df[fsc_col], dtype=float)
    ssc = np.asarray(df[ssc_col], dtype=float)
    if len(fsc) < 100:
        raise ValueError(f"Too few events ({len(fsc)}) to fit a GMM reliably.")

    # Log transform with floor.
    log_fsc = np.log10(np.maximum(fsc, min_fsc))
    log_ssc = np.log10(np.maximum(ssc, min_ssc))
    X = np.column_stack([log_fsc, log_ssc])

    gmm = GaussianMixture(
        n_components=2, covariance_type='full',
        random_state=random_state, n_init=2, max_iter=200,
    )
    gmm.fit(X)

    # Cell cluster = higher-mean component (by FSC + SSC mean).
    means = gmm.means_.sum(axis=1)
    cell_idx = int(np.argmax(means))
    labels = gmm.predict(X)
    cell_mask = (labels == cell_idx)

    if cell_mask.sum() < 50:
        # Degenerate fit — fall back to a robust quantile of the whole cloud.
        cell_mask = np.ones_like(labels, dtype=bool)

    fsc_t = float(np.quantile(fsc[cell_mask], quantile))
    ssc_t = float(np.quantile(ssc[cell_mask], quantile))
    return fsc_t, ssc_t
