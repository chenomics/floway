"""
Automatic channel / color identification across cytometer naming conventions.

Different machines name detectors differently:
    Attune NxT:   BL1-A, BL2-A, VL1-A, YL1-A, RL1-A, ...   (laser+slot)
    BD Fortessa:  FITC-A, PE-A, APC-A, BV421-A, ...        (fluorophore)
    Sony SH800:   FL1-A, FL2-A, ...                        (index)

This module wraps detection rules so a user-level call like
``find_channel(parser, "red")`` returns the right column regardless of
which machine produced the file.

Detection priority (most reliable first):
    1. ``$PnS`` stain field from the FCS metadata, if filled in.
    2. Name match against a fluorophore registry (FITC, PE, APC, GFP, ...).
    3. Laser-line nomenclature (BL1, VL2, YL3, RL1, FL4, ...) mapped to the
       typical filter slot for that machine family. These are *best-guess*
       and are flagged with ``confidence='low'``.
"""

import re
import numpy as np
import pandas as pd


SCATTER_PREFIXES = ("FSC", "SSC")
INDEX_LIKE = {"TIME", "EVENT", "EVENTNUMBERDP", "IMAGEFLAG"}


# Coarse color-family bins by emission wavelength (nm).
COLOR_BINS = [
    ('violet',     390, 450),
    ('blue',       450, 500),
    ('green',      500, 560),
    ('yellow',     560, 600),
    ('orange-red', 600, 650),
    ('red',        650, 720),
    ('far-red',    720, 850),
]


def _color_from_nm(nm):
    if nm is None:
        return None
    for name, lo, hi in COLOR_BINS:
        if lo <= nm < hi:
            return name
    return None


# Fluorophore name -> (typical excitation laser nm, typical emission nm).
# Values are representative — actual filter cubes may differ slightly.
FLUOROPHORE_TABLE = {
    # Green (488 ex)
    'FITC':         (488, 525),
    'GFP':          (488, 510),
    'eGFP':         (488, 510),
    'AF488':        (488, 525),
    'Alexa488':     (488, 525),
    'CFSE':         (488, 525),

    # Yellow-orange (488 or 561 ex)
    'PE':           (561, 585),

    # Orange-red (561 ex)
    'mCherry':      (561, 620),
    'PE-CF594':     (561, 620),
    'PE-Texas Red': (561, 620),
    'RFP':          (561, 605),
    'mScarlet':     (561, 600),
    'dsRed':        (561, 590),
    'mRFP':         (561, 610),

    # Red (561 long-Stokes tandem)
    'PE-Cy5':       (561, 670),
    'PerCP':        (488, 695),
    'PerCP-Cy5.5':  (488, 695),
    'PI':           (488, 617),

    # Far-red (638 ex)
    'APC':          (638, 660),
    'AF647':        (638, 670),
    'Alexa647':     (638, 670),
    'Cy5':          (638, 670),
    'AF700':        (638, 720),

    # Tandem far-red
    'APC-Cy7':      (638, 780),
    'APC-H7':       (638, 780),
    'PE-Cy7':       (561, 780),

    # Violet (405 ex)
    'BV421':        (405, 421),
    'Pacific Blue': (405, 455),
    'eFluor450':    (405, 450),
    'BV510':        (405, 510),
    'BV605':        (405, 605),
    'BV650':        (405, 650),
    'BV711':        (405, 711),
    'BV786':        (405, 786),
}


# Attune NxT — laser/filter conventional emissions.
#  BL = 488 nm laser; VL = 405 nm; YL = 561 nm; RL = 638 nm.
ATTUNE_LASER = {'BL': 488, 'VL': 405, 'YL': 561, 'RL': 638, 'UV': 355}
ATTUNE_SLOTS = {
    # (laser, slot) -> typical emission filter center nm
    ('BL', 1): 530, ('BL', 2): 590, ('BL', 3): 695,
    ('VL', 1): 440, ('VL', 2): 512, ('VL', 3): 603, ('VL', 4): 710,
    ('YL', 1): 585, ('YL', 2): 620, ('YL', 3): 695, ('YL', 4): 780,
    ('RL', 1): 670, ('RL', 2): 720, ('RL', 3): 780,
    ('UV', 1): 440, ('UV', 2): 530,
}
# Typical fluorophore that goes in each Attune slot (best-guess defaults).
ATTUNE_FLUOROPHORE = {
    ('BL', 1): 'GFP/FITC', ('BL', 2): 'PE', ('BL', 3): 'PerCP-Cy5.5',
    ('VL', 1): 'BV421/Pacific Blue', ('VL', 2): 'BV510',
    ('VL', 3): 'BV605', ('VL', 4): 'BV711',
    ('YL', 1): 'PE',  ('YL', 2): 'mCherry',
    ('YL', 3): 'PE-Cy5.5', ('YL', 4): 'PE-Cy7',
    ('RL', 1): 'APC', ('RL', 2): 'AF700', ('RL', 3): 'APC-Cy7',
    ('UV', 1): 'BUV395', ('UV', 2): 'BUV496',
}

# Sony SH800 / MA900 — default FL1..FL6 slots.
SONY_SLOTS = {
    1: (488, 525, 'GFP/FITC'),
    2: (561, 585, 'PE'),
    3: (561, 617, 'mCherry'),
    4: (638, 665, 'APC'),
    5: (638, 720, 'AF700'),
    6: (638, 780, 'APC-Cy7'),
}


def _classify_by_fluorophore_name(name):
    """Match name (or stain label) against known fluorophore strings."""
    if not name:
        return None
    s = str(name).strip()
    # Strip trailing -A / -H / -W and -Area / -Height
    s_clean = re.sub(r'[-_\s]?(Area|Height|Width|A|H|W)$', '', s, flags=re.IGNORECASE)
    s_clean = s_clean.strip(' -_')
    # Exact match first
    for fluor, (ex, em) in FLUOROPHORE_TABLE.items():
        if s_clean.lower() == fluor.lower():
            return dict(fluorophore_guess=fluor, laser_nm=ex, emission_nm=em,
                        color=_color_from_nm(em), confidence='high')
    # Substring match (Pacific Blue, BV421, etc.)
    s_low = s_clean.lower()
    for fluor, (ex, em) in FLUOROPHORE_TABLE.items():
        f_low = fluor.lower()
        if len(f_low) >= 3 and f_low in s_low:
            return dict(fluorophore_guess=fluor, laser_nm=ex, emission_nm=em,
                        color=_color_from_nm(em), confidence='high')
    return None


_ATTUNE_RE = re.compile(r'^(BL|VL|YL|RL|UV)(\d+)[-_]?[AHW]?$', re.IGNORECASE)
_SONY_RE   = re.compile(r'^FL[-_]?(\d+)[-_]?[AHW]?$', re.IGNORECASE)


def _classify_by_detector_name(name):
    if not name:
        return None
    s = str(name).strip()
    m = _ATTUNE_RE.match(s)
    if m:
        laser = m.group(1).upper()
        slot = int(m.group(2))
        ex = ATTUNE_LASER.get(laser)
        em = ATTUNE_SLOTS.get((laser, slot))
        fluor = ATTUNE_FLUOROPHORE.get((laser, slot))
        return dict(fluorophore_guess=fluor, laser_nm=ex, emission_nm=em,
                    color=_color_from_nm(em), confidence='low')
    m = _SONY_RE.match(s)
    if m:
        slot = int(m.group(1))
        if slot in SONY_SLOTS:
            ex, em, fluor = SONY_SLOTS[slot]
            return dict(fluorophore_guess=fluor, laser_nm=ex, emission_nm=em,
                        color=_color_from_nm(em), confidence='low')
    return None


def _kind_for(name):
    upper = str(name).upper()
    if upper.startswith(SCATTER_PREFIXES):
        return 'scatter'
    if upper in INDEX_LIKE:
        return 'index'
    # Strip trailing -A/-H/-W when checking for index columns
    if re.sub(r'[-_]?[AHW]$', '', upper) in INDEX_LIKE:
        return 'index'
    return 'fluorescence'


def detect_channels(source, scatter_only=False):
    """
    Identify the color / fluorophore for each channel in an FCS file.

    Args:
        source: One of:
            * ``floway.FCSParser`` — uses ``$PnS`` stain labels when present.
            * ``pd.DataFrame`` (from ``parser.get_data()``) — name-only.
            * list/tuple of column names — name-only.
        scatter_only (bool): If True, drop fluorescence rows from the
            result.

    Returns:
        pd.DataFrame with columns:
            ``column``           — detector name (also DataFrame column).
            ``kind``             — scatter / fluorescence / index.
            ``stain``            — operator-supplied $PnS label, or ''.
            ``color``            — coarse color family (e.g. 'green', 'red'),
                                   or None for non-fluorescence channels.
            ``fluorophore_guess``— most likely fluorophore (e.g. 'PE', 'GFP').
            ``laser_nm``         — typical excitation laser, if inferrable.
            ``emission_nm``      — typical emission peak, if inferrable.
            ``confidence``       — 'high' (matched fluorophore name or stain),
                                   'low' (laser-slot guess), or 'none'
                                   (couldn't classify).

    Example:
        >>> from floway import FCSParser, detect_channels
        >>> parser = FCSParser('sample.fcs')
        >>> detect_channels(parser)
            column  kind         stain  color       fluorophore_guess  laser_nm  emission_nm  confidence
            FSC-A   scatter      ''     None        None               None      None         none
            BL1-A   fluorescence ''     green       GFP/FITC           488       530          low
            YL2-A   fluorescence ''     orange-red  mCherry            561       620          low
    """
    # Resolve `source` to (columns, stain_lookup).
    stain_lookup = {}
    columns = None
    # Lazy import to avoid circular dep at module load.
    try:
        from .parser import FCSParser
    except ImportError:
        FCSParser = None

    if FCSParser is not None and isinstance(source, FCSParser):
        info = source.list_channels()
        columns = list(info['column'])
        stain_lookup = dict(zip(info['column'], info['stain']))
    elif isinstance(source, pd.DataFrame):
        columns = list(source.columns)
    elif isinstance(source, (list, tuple)):
        columns = list(source)
    else:
        raise TypeError(
            f"Unsupported source type {type(source).__name__}. "
            "Pass an FCSParser, DataFrame, or list of column names."
        )

    rows = []
    for col in columns:
        kind = _kind_for(col)
        stain = stain_lookup.get(col, '') or ''
        info = {'fluorophore_guess': None, 'laser_nm': None,
                'emission_nm': None, 'color': None, 'confidence': 'none'}

        if kind == 'fluorescence':
            # Priority 1: $PnS stain label (if it matches a known fluorophore).
            hit = _classify_by_fluorophore_name(stain) if stain else None
            # Priority 2: detector column name itself names a fluorophore.
            if hit is None:
                hit = _classify_by_fluorophore_name(col)
            # Priority 3: laser-slot inference (Attune / Sony).
            if hit is None:
                hit = _classify_by_detector_name(col)
            if hit:
                info.update(hit)

        rows.append({
            'column':            col,
            'kind':              kind,
            'stain':             stain,
            'color':             info['color'],
            'fluorophore_guess': info['fluorophore_guess'],
            'laser_nm':          info['laser_nm'],
            'emission_nm':       info['emission_nm'],
            'confidence':        info['confidence'],
        })

    out = pd.DataFrame(rows)
    if scatter_only:
        out = out[out['kind'] == 'scatter'].reset_index(drop=True)
    return out


def find_channel(source, query, prefer='A'):
    """
    Resolve a fluorophore or color query to a detector column name.

    Args:
        source: ``FCSParser``, ``DataFrame``, or list of column names.
        query (str): Fluorophore name (``"PE"``, ``"mCherry"``, ``"GFP"``)
            or color family (``"green"``, ``"red"``, ``"far-red"``).
            Case-insensitive.
        prefer (str): When multiple parameter variants exist for the same
            detector (``X-A`` / ``X-H`` / ``X-W``), prefer the one ending
            in this letter. Default ``"A"`` (area).

    Returns:
        str: The column name that best matches ``query``.

    Raises:
        KeyError: If no matching channel is found.

    Example:
        >>> find_channel(parser, "mCherry")   # Attune  -> "YL2-A"
        >>> find_channel(parser, "mCherry")   # Fortessa -> "PE-CF594-A"
        >>> find_channel(parser, "red")       # whichever red detector exists
    """
    info = detect_channels(source)
    info = info[info['kind'] == 'fluorescence'].copy()
    if info.empty:
        raise KeyError(f"No fluorescence channels found in source.")

    q = query.strip().lower()
    known_colors = {c[0] for c in COLOR_BINS}

    if q in known_colors:
        cand = info[info['color'].astype(str).str.lower() == q]
    else:
        # Match against fluorophore_guess (substring, case-insensitive).
        cand = info[info['fluorophore_guess']
                    .astype(str).str.lower().str.contains(q, na=False)]
        # Also try stain label
        if cand.empty:
            cand = info[info['stain'].astype(str).str.lower().str.contains(q, na=False)]
        # And the column name itself
        if cand.empty:
            cand = info[info['column'].astype(str).str.lower().str.contains(q, na=False)]

    if cand.empty:
        raise KeyError(
            f"No channel matches query '{query}'. "
            f"Available fluorescence channels: {list(info['column'])}"
        )

    # Prefer the "-A" variant (or whatever `prefer` says).
    pref_re = re.compile(rf'[-_]?{re.escape(prefer)}$', re.IGNORECASE)
    pref_hits = cand[cand['column'].astype(str).str.contains(pref_re, na=False)]
    if not pref_hits.empty:
        return str(pref_hits.iloc[0]['column'])
    # Otherwise prefer high-confidence over low.
    cand = cand.sort_values('confidence', key=lambda s: s.map({'high': 0, 'low': 1, 'none': 2}))
    return str(cand.iloc[0]['column'])
