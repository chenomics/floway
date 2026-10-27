"""Diagnostics for FCS data: pick out which channels actually carry signal."""

import numpy as np
import pandas as pd


SCATTER_PREFIXES = ("FSC", "SSC")
# Bookkeeping columns recorded by some instruments (event index, image flags,
# acquisition time). Not physical fluorescence measurements -- never "active".
INDEX_LIKE = {"TIME", "EVENT", "EVENTNUMBERDP", "IMAGEFLAG"}


def channel_activity(df, parser=None, log_spread_threshold=1.5,
                     tail_threshold=1.0, eps=1.0):
    """
    Score each channel for whether it carries informative signal.

    Useful when an experimenter records detectors that weren't actually
    stained (e.g. a BFP channel left on by default) -- those channels sit
    in a narrow noise band and can be dropped from downstream plots.

    Heuristic
    ---------
    For each column, compute (after flooring values at `eps` so logs behave):

        log_spread = log10(p99) - log10(p10)        # overall dynamic range
        upper_tail = log10(p99) - log10(p50)        # bright-tail spread

    A fluorescence channel is flagged ``active=True`` if EITHER
    ``log_spread >= log_spread_threshold`` (default 1.5 decades) OR
    ``upper_tail >= tail_threshold`` (default 1.0 decade -- a clear bright
    subpopulation 10x above the median).

    Scatter channels (FSC*, SSC*) are always ``active=True`` since they
    drive gating; ``Time`` is always ``active=False``.

    Args:
        df (pd.DataFrame): Event data, e.g. from ``FCSParser.get_data()`` or
            ``analyze_fcs_data(...)``. Running on gated singlets gives the
            cleanest read because debris doesn't inflate spread.
        parser (FCSParser, optional): If supplied, ``$PnS`` stain labels are
            included in the output for context.
        log_spread_threshold (float): Decades of p10->p99 spread above which
            a fluorescence channel counts as active.
        tail_threshold (float): Decades of p50->p99 spread above which a
            fluorescence channel counts as active (catches bimodal cases
            where the median sits in the negative population).
        eps (float): Floor applied before log to handle zero / negative
            values from compensation.

    Returns:
        pd.DataFrame, one row per channel, columns:
            column, kind, stain, p10, p50, p99, log_spread, upper_tail, active

    Example:
        >>> from floway import FCSParser, channel_activity
        >>> parser = FCSParser("sample.fcs")
        >>> activity = channel_activity(parser.get_data(), parser=parser)
        >>> active_cols = activity.loc[activity['active'], 'column'].tolist()
    """
    stain_map = {}
    if parser is not None:
        chans = parser.list_channels()
        stain_map = dict(zip(chans['column'], chans['stain']))

    rows = []
    for col in df.columns:
        upper = str(col).upper()
        if upper.startswith(SCATTER_PREFIXES):
            kind = "scatter"
        elif upper in INDEX_LIKE:
            kind = "index"
        else:
            kind = "fluorescence"

        vals = pd.to_numeric(df[col], errors="coerce").dropna().values
        if len(vals) == 0:
            rows.append({
                'column': col, 'kind': kind, 'stain': stain_map.get(col, ''),
                'p10': np.nan, 'p50': np.nan, 'p99': np.nan,
                'log_spread': np.nan, 'upper_tail': np.nan, 'active': False,
            })
            continue

        p10, p50, p99 = np.percentile(vals, [10, 50, 99])
        log_spread = np.log10(max(p99, eps)) - np.log10(max(p10, eps))
        upper_tail = np.log10(max(p99, eps)) - np.log10(max(p50, eps))

        if kind == "scatter":
            active = True
        elif kind == "index":
            active = False
        else:
            active = (log_spread >= log_spread_threshold) or (upper_tail >= tail_threshold)

        rows.append({
            'column': col,
            'kind': kind,
            'stain': stain_map.get(col, ''),
            'p10': p10, 'p50': p50, 'p99': p99,
            'log_spread': round(float(log_spread), 2),
            'upper_tail': round(float(upper_tail), 2),
            'active': bool(active),
        })

    return pd.DataFrame(rows)
