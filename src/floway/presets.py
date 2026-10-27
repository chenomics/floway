"""
Machine-specific channel-name presets.

These are *typical* detector -> fluorophore mappings for common cytometers.
The exact mapping on YOUR machine depends on the operator's panel design
(which laser line, which optical filter is in front of which detector).
Always verify with `FCSParser(path).list_channels()` before trusting a preset.

Usage:
    from floway import PRESETS
    chans = PRESETS["attune"]
    gfp_col, mcherry_col = chans["GFP"], chans["mCherry"]
"""

# Detector short-name conventions:
#   FSC-A / SSC-A : forward / side scatter (area) -- universal across modern machines
#   <laser><n>-A  : Attune-style (BL1-A = 488 nm laser, detector 1 = ~530/30 GFP/FITC filter)
#   <fluorophore>-A : BD / Sony naming convention -- channel labelled by the dye it's set up for

PRESETS = {
    # Thermo Fisher / Invitrogen Attune NxT
    # Detector names are fixed by laser color; filter cubes determine fluorophore mapping.
    # Defaults below assume the standard 4-laser (V/B/Y/R) configuration with stock filters.
    "attune": {
        "FSC": "FSC-A",
        "SSC": "SSC-A",
        "GFP": "BL1-A",      # 488 nm, ~530/30
        "FITC": "BL1-A",
        "PE": "YL1-A",       # 561 nm, ~585/16   (also detectable on BL2 with 488 ex)
        "mCherry": "YL2-A",  # 561 nm, ~620/15
        "PE-Cy5": "YL3-A",
        "PE-Cy7": "YL4-A",
        "PerCP-Cy5.5": "BL3-A",
        "BV421": "VL1-A",    # 405 nm, ~440/50
        "Pacific Blue": "VL1-A",
        "APC": "RL1-A",      # 638 nm, ~670/14
        "AF647": "RL1-A",
        "APC-Cy7": "RL3-A",
    },

    # BD LSR / Fortessa / Symphony / FACSAria family
    # Detectors are commonly relabelled by fluorophore in the experiment template.
    "bd_fortessa": {
        "FSC": "FSC-A",
        "SSC": "SSC-A",
        "GFP": "FITC-A",
        "FITC": "FITC-A",
        "PE": "PE-A",
        "mCherry": "PE-CF594-A",  # also seen as "mCherry-A" or "PE-Texas Red-A"
        "PE-Cy5": "PE-Cy5-A",
        "PE-Cy7": "PE-Cy7-A",
        "PerCP-Cy5.5": "PerCP-Cy5-5-A",
        "BV421": "BV421-A",
        "Pacific Blue": "Pacific Blue-A",
        "APC": "APC-A",
        "AF647": "APC-A",
        "APC-Cy7": "APC-Cy7-A",
    },

    # BD FACSAria sorters use the same convention as Fortessa.
    "bd_aria": {
        "FSC": "FSC-A",
        "SSC": "SSC-A",
        "GFP": "FITC-A",
        "FITC": "FITC-A",
        "PE": "PE-A",
        "mCherry": "PE-CF594-A",
        "APC": "APC-A",
    },

    # Sony SH800 / MA900 cell sorters.
    # Default templates label detectors as FL1..FL6, but many users rename
    # to fluorophores in the panel editor. Both forms shown.
    "sony_sh800": {
        "FSC": "FSC-A",
        "SSC": "SSC-A",
        "GFP": "FL1-A",        # 488 nm, ~525/50
        "FITC": "FL1-A",
        "PE": "FL2-A",         # 488 or 561 nm, ~585/30
        "mCherry": "FL3-A",    # 561 nm, ~617/30
        "APC": "FL4-A",        # 638 nm, ~665/30
    },

    "sony_ma900": {
        "FSC": "FSC-A",
        "SSC": "SSC-A",
        "GFP": "FL1-A",
        "FITC": "FL1-A",
        "PE": "FL2-A",
        "mCherry": "FL3-A",
        "APC": "FL4-A",
    },
}


def resolve_channel(preset_name, fluorophore):
    """Look up the detector column for a fluorophore on a named machine.

    Args:
        preset_name (str): Key into PRESETS, e.g. "attune", "bd_fortessa".
        fluorophore (str): "GFP", "mCherry", "FSC", etc.

    Returns:
        str: The detector column name (e.g. "BL1-A").

    Raises:
        KeyError: If the preset or fluorophore is not registered.
    """
    if preset_name not in PRESETS:
        raise KeyError(
            f"Unknown machine preset '{preset_name}'. "
            f"Known: {sorted(PRESETS.keys())}"
        )
    chans = PRESETS[preset_name]
    if fluorophore not in chans:
        raise KeyError(
            f"Fluorophore '{fluorophore}' not in '{preset_name}' preset. "
            f"Known: {sorted(chans.keys())}"
        )
    return chans[fluorophore]
