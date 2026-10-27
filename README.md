# floway

A lightweight flow cytometry analysis toolkit — *flow* the *wei* (Wei Chen).

`floway` parses FCS 3.0/3.1 files from scratch (no external FCS library needed), performs plate-level gating (saturation drop, cells vs debris, FSC-A/FSC-H singlets), and ships small plotting helpers for QC and KDE summaries.

## Install

```bash
pip install git+https://github.com/chenomics/floway.git
```

Or from a clone, which is what you want if you plan to edit it:

```bash
git clone https://github.com/chenomics/floway.git
cd floway
pip install -e .
```

Python 3.12 or newer. `pip install -e ".[auto]"` adds scikit-learn for the
automatic threshold helpers.

## Quick start

End-to-end: parse an FCS file, remove debris, gate singlets, and plot.
Pass `"auto"` for either threshold to let floway pick it from the data.

```python
from floway import analyze_fcs_data

gated = analyze_fcs_data(
    "path/to/sample.fcs",
    fsc_threshold="auto",   # GMM-based threshold, or a number to override
    ssc_threshold="auto",
    visualize=True,
    report=True,
)
print(gated.head())
```

### Auto FSC/SSC thresholds

`auto_thresholds` fits a 2-component Gaussian mixture on `log(FSC, SSC)`,
keeps the cell cluster (higher-mean component), and returns the 5th
percentile of that cluster along each axis as the debris cutoff. Useful
when you load a new machine or a plate with unusual scatter.

```python
from floway import FCSParser, auto_thresholds

df = FCSParser("sample.fcs").get_data()
fsc_t, ssc_t = auto_thresholds(df)
print(fsc_t, ssc_t)
```

### Auto channel identification

Different machines name detectors differently — Attune uses laser-slot
names (`BL1-A`, `YL2-A`), BD names by fluorophore (`FITC-A`, `PE-A`),
Sony uses `FL1-A`...`FL6-A`. `detect_channels` returns the inferred
color/fluorophore for every detector. `find_channel` then resolves a
fluorophore or color name to the right column on whatever machine you
gave it.

```python
from floway import FCSParser, detect_channels, find_channel

parser = FCSParser("sample.fcs")
print(detect_channels(parser))
#   column   kind         stain  color       fluorophore_guess  laser_nm  emission_nm  confidence
#   FSC-A    scatter      ''     None        None               None      None         none
#   YL2-A    fluorescence ''     orange-red  mCherry            561       620          low
#   BL1-A    fluorescence ''     green       GFP/FITC           488       530          low

# Then ask for a fluorophore or a color family:
find_channel(parser, "mCherry")   # on Attune  -> 'YL2-A'
find_channel(parser, "mCherry")   # on Fortessa -> 'PE-CF594-A'
find_channel(parser, "red")       # whichever red detector exists
```

`confidence` is `"high"` when the channel name or `$PnS` stain field
matches a fluorophore string, `"low"` for laser-slot guesses (you'll
want to verify against the operator's panel design).

## Picking the right channels for your machine

Different cytometers label detectors differently. `floway` ships a small
registry of typical mappings plus an introspection helper so you can adapt
to any machine.

```python
from floway import FCSParser, PRESETS, resolve_channel

# Option A: use a preset (verify it matches your panel config first)
gfp_col     = resolve_channel("attune", "GFP")        # -> "BL1-A"
mcherry_col = resolve_channel("bd_fortessa", "mCherry")  # -> "PE-CF594-A"

# Option B: introspect any .fcs file directly
parser = FCSParser("path/to/sample.fcs")
print(parser.list_channels())
#    index   column   stain  range
# 0      1   FSC-A          1048576
# 1      2   SSC-A          1048576
# 2      3   BL1-A     GFP   1048576
# 3      4   YL2-A mCherry   1048576
# ...
```

Available presets: `attune`, `bd_fortessa`, `bd_aria`, `sony_sh800`, `sony_ma900`.
The presets cover the common detectors (FSC, SSC, GFP, FITC, PE, mCherry, APC,
BV421, etc.) and are starting points — always sanity-check against
`list_channels()` because the mapping ultimately depends on the operator's
panel design (laser line, filter cube).

### Filter out channels that weren't used

Acquisition templates often record detectors that nobody actually stained
(e.g. a leftover BFP channel). These columns sit in a narrow noise band and
just clutter downstream plots. `channel_activity()` flags them by signal
spread:

```python
from floway import channel_activity

activity = channel_activity(df, parser=parser)
print(activity)
#   column         kind  stain    p10    p50    p99  log_spread  upper_tail  active
#   FSC-A      scatter         3.7e5  5.0e5  7.3e5        0.29        0.17    True
#   BL1-A fluorescence    GFP   107    297  70354        2.82        2.37    True   <- bimodal: real signal
#   VL1-A fluorescence    BFP    87    151    269        0.49        0.25   False   <- unused: noise band
#   ...

active_cols = activity.loc[activity['active'], 'column'].tolist()
```

A fluorescence channel is flagged active when either its p10→p99 dynamic
range exceeds 1.5 decades or its p50→p99 upper tail exceeds 1 decade
(thresholds are tunable). Run on gated singlets for a cleaner read —
debris inflates spread artificially.

## End-to-end example

See [`examples/flow_nb_sharing.ipynb`](examples/flow_nb_sharing.ipynb) for a
two-color screen workflow: pick a gate visually with `analyze_fcs_data(...,
visualize=True)`, then loop over a folder of `.fcs` files, build a long-format
DataFrame keyed by sample, threshold on GFP/mCherry, and plot KDE panels +
barplots. Edit the `fcs_folder` and filename-parsing rule to match your data.

## Components

### Parse an FCS file

```python
from floway import FCSParser

parser = FCSParser("sample.fcs")
df = parser.get_data()           # pandas DataFrame, one row per event
meta = parser.get_metadata()     # dict of $-keywords + offsets
```

### One gate for a whole plate

The route to prefer when the wells have to be comparable. `PlateGate` fits the cell
and singlet cuts **once** on the pooled events of a run, then applies that one gate to
every well — a per-well fit drifts, and on a weak well it can mis-call the whole sample.

```python
from floway import PlateGate

gate = PlateGate.fit(fcs_paths)        # one fit, pooled over every well
singlets = gate.gated(fcs_paths[0])    # DataFrame of that well's singlets
mask = gate.apply(df)                  # or a boolean mask over events you already have
print(gate)                            # the cuts it settled on
```

`gate_populations()` below runs the same algorithm on a single sample; use it when one
file is all you have, and `PlateGate` whenever you are comparing wells.

### Singlet gating (Mahalanobis ellipse)

```python
from floway import gate_singlets_mahalanobis

mask, ellipse = gate_singlets_mahalanobis(df, fsc_col="FSC-A", ssc_col="SSC-A",
                                          confidence_level=0.95)
singlets = df[mask]
```

`ellipse` is a dict with `center`, `width`, `height`, `angle` — feed it to the plotter below.

### Plot the gating strategy

```python
from floway import plot_gating_strategy

plot_gating_strategy(df_original=df, df_gated=singlets,
                     fsc_col="FSC-A", ssc_col="SSC-A",
                     ellipse_params=ellipse,
                     fsc_threshold=300_000, ssc_threshold=60_000)
```

### KDE panel of fluorescence per condition

Expects a long-format DataFrame with columns `sample` and `GFP_signal`:

```python
from floway import plot_kde

plot_kde(df_long, unique_conds=df_long["sample"].unique(),
         count=True, ylim=(0, 1.2))
```

### Activation-style KDE grid (with negative-control reference)

For activation readouts where you want to see (a) the per-condition
distribution, (b) a negative-control reference baseline, and (c) what
fraction of cells crossed a positivity threshold — use
`plot_kde_grid`:

```python
from floway import plot_kde_grid
import numpy as np

# Per-experiment positivity threshold (99th percentile of neg-ctrl PE-A)
T_exp = np.percentile(neg_ctrl_df["PE_A"], 99)

plot_kde_grid(
    df_long,
    conditions=summary["condition"].tolist(),  # plot order
    signal_col="PE_A",
    sample_col="condition",
    neg_signal=neg_ctrl_df["PE_A"].values,     # gray dashed reference
    threshold=T_exp,                            # vertical dotted line + % positive
    title="REP_1 - PE-A per condition",
)
```

Each panel shows the condition's KDE filled in color, the negative
control as gray dashed, a vertical line at `threshold`, and the percent
of cells above the threshold annotated at top-right.

### Plate-map helper

```python
from floway import parse_plate_map

wells = parse_plate_map("plate_map.csv")
# {"A1": "DMSO", "A2": "Drug_X", ...}
```

The CSV is expected in 96-well layout: rows `A`–`H`, columns `1`–`12`, with sample names in cells.

## Dependencies

- numpy, pandas, scipy, matplotlib, seaborn

## License

MIT — see [LICENSE](LICENSE).
