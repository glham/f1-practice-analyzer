"""Bold L-axis chart styling for matplotlib/seaborn.

House style:
  * Left + bottom spines only, 1.8px, outward ticks
  * Hairline y-only grid, solid (never dashed), sitting behind the data
  * Colourblind-validated categorical palette in fixed slot order
  * Percent and currency axis formatters

Usage:
    import chartstyle as cs
    cs.use_style()                      # light (default)
    cs.use_style(dark=True)             # dark surface

    fig, ax = plt.subplots()
    ax.bar(names, values, color=cs.series(0))
    cs.percent_axis(ax)                 # 45%  (data already 0-100)
    cs.currency_axis(ax, axis='x')      # £5.5m
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter

# --------------------------------------------------------------------------
# Palette. Hex values taken verbatim from the validated reference palette.
# Slot ORDER is the colourblind-safety mechanism, not cosmetic -- assign in
# order, never cycle, never reorder.
#
# Validated (documented) results for this ordering, OKLab x100:
#   adjacent pairs  -- CVD dE 9.1 light / 8.4 dark   (>=8 target)
#                   -- normal-vision dE 19.6 / 19.3  (>=15 floor)
#
# IMPORTANT CAP: only the FIRST THREE slots clear the all-pairs gate. For any
# chart where every series can sit beside every other (scatter, bubble, small
# multiples), use at most 3 categorical colours -- past that, fold to "Other"
# or facet. Slot 4 puts yellow next to orange, which fails all-pairs.
# --------------------------------------------------------------------------

CATEGORICAL_LIGHT = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

CATEGORICAL_DARK = [
    "#3987e5", "#d95926", "#199e70", "#c98500",
    "#d55181", "#008300", "#9085e9", "#e66767",
]

# Single-hue blue ramp, light -> dark. Full range is for SEQUENTIAL encoding.
# For ORDINAL marks start no lighter than step 250 on a light surface.
SEQUENTIAL_BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]

DIVERGING = {"low": "#2a78d6", "mid_light": "#f0efec", "mid_dark": "#383835", "high": "#e34948"}

STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

CHROME = {
    "light": {
        "surface": "#fcfcfb",
        "ink_primary": "#0b0b0b",
        "ink_secondary": "#52514e",
        "ink_muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#0b0b0b",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink_primary": "#ffffff",
        "ink_secondary": "#c3c2b7",
        "ink_muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#ffffff",
    },
}

_MODE = "light"

# All-pairs safe cap -- see note above.
ALL_PAIRS_SAFE_SLOTS = 3


def use_style(dark: bool = False, base_size: int = 11) -> None:
    """Apply the house style globally. Call once per notebook/session."""
    global _MODE
    _MODE = "dark" if dark else "light"
    c = CHROME[_MODE]

    mpl.rcParams.update({
        # --- surface
        "figure.facecolor": c["surface"],
        "axes.facecolor": c["surface"],
        "savefig.facecolor": c["surface"],

        # --- bold L-axis: left + bottom only, 1.8px
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.linewidth": 1.8,
        "axes.edgecolor": c["axis"],

        # --- ticks point outward
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.color": c["ink_muted"],
        "ytick.color": c["ink_muted"],
        "xtick.labelcolor": c["ink_secondary"],
        "ytick.labelcolor": c["ink_secondary"],

        # --- hairline grid, y only, SOLID (dashed grids are noise), behind data
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": c["grid"],
        "grid.linewidth": 1.0,
        "grid.linestyle": "-",
        "grid.alpha": 1.0,

        # --- type
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
        "font.size": base_size,
        "axes.titlesize": base_size + 3,
        "axes.titleweight": "semibold",
        "axes.titlecolor": c["ink_primary"],
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelsize": base_size,
        "axes.labelcolor": c["ink_secondary"],
        "text.color": c["ink_primary"],

        # --- marks: 2px lines, >=8px markers
        "lines.linewidth": 2.0,
        "lines.markersize": 8.0,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "patch.linewidth": 0.0,

        # --- legend: no frame, it competes with the data
        "legend.frameon": False,
        "legend.fontsize": base_size - 1,
        "legend.labelcolor": c["ink_secondary"],

        # --- output
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",

        "axes.prop_cycle": mpl.cycler(color=palette()),
    })


def palette(dark: bool | None = None) -> list[str]:
    """The categorical palette, in validated slot order."""
    mode = _MODE if dark is None else ("dark" if dark else "light")
    return CATEGORICAL_DARK if mode == "dark" else CATEGORICAL_LIGHT


def series(i: int) -> str:
    """Colour for categorical slot i (0-based). Assign in order, never cycle."""
    p = palette()
    if i >= len(p):
        raise ValueError(
            f"slot {i} exceeds the {len(p)}-slot palette. A 9th series is never a "
            "generated hue -- fold into 'Other', facet, or use small multiples."
        )
    return p[i]


def ink(role: str = "primary") -> str:
    """Text/chrome colour: primary | secondary | muted | grid | surface | axis."""
    c = CHROME[_MODE]
    return c.get(f"ink_{role}", c.get(role, c["ink_primary"]))


def sequential(n: int) -> list[str]:
    """n evenly-spaced steps from the single-hue blue ramp (light -> dark)."""
    if n < 1:
        return []
    if n == 1:
        return [SEQUENTIAL_BLUE[6]]
    step = (len(SEQUENTIAL_BLUE) - 1) / (n - 1)
    return [SEQUENTIAL_BLUE[round(i * step)] for i in range(n)]


def ordinal(n: int, dark: bool | None = None) -> list[str]:
    """n DISCRETE ordered steps from the blue ramp, maximally separated.

    Use for a handful of ordered levels (price bands, tiers, quartiles) where a
    continuous colourbar would be unreadable -- with ~5 levels a smooth ramp
    gives near-identical neighbours. Pair with a legend, not a colourbar.

    Respects the ordinal contrast floor: the step nearest the surface must still
    clear 2:1, so light mode starts no lighter than ramp step 250 and dark mode
    goes no darker than step 600.
    """
    mode = _MODE if dark is None else ("dark" if dark else "light")
    lo, hi = (3, len(SEQUENTIAL_BLUE) - 1) if mode == "light" else (0, 10)
    if n < 1:
        return []
    if n == 1:
        return [SEQUENTIAL_BLUE[(lo + hi) // 2]]
    step = (hi - lo) / (n - 1)
    return [SEQUENTIAL_BLUE[round(lo + i * step)] for i in range(n)]


def shades(base: str, n: int = 2, lightest: float = 0.74, darkest: float = 0.34) -> list[str]:
    """n lightness steps of ONE hue, light -> dark.

    For pairing inside a colour family -- e.g. Home = two blues, Away = two reds --
    so each pair reads as one group while the steps stay tellable apart.

    Steps differ by LIGHTNESS, which survives every type of colour blindness;
    a hue pair may not. That makes this the safe way to subdivide a group.

    The intermediate hexes are derived from `base`, not drawn from the validated
    categorical slots, so treat them as house-derived rather than separately
    CVD-validated -- the lightness gap is what carries the distinction, and it
    holds regardless.
    """
    import colorsys

    base = base.lstrip("#")
    r, g, b = (int(base[i:i + 2], 16) / 255 for i in (0, 2, 4))
    h, _, s = colorsys.rgb_to_hls(r, g, b)
    if n < 1:
        return []
    if n == 1:
        return ["#" + base]
    out = []
    for i in range(n):
        light = lightest + (darkest - lightest) * i / (n - 1)
        rr, gg, bb = colorsys.hls_to_rgb(h, light, s)
        out.append("#%02x%02x%02x" % tuple(round(c * 255) for c in (rr, gg, bb)))
    return out


def signal(n: int = 3) -> list[str]:
    """High-contrast ORDERED progression -- the traffic-light idea, made CVD-safe.

    Use when an ordered split needs to shout and the single-hue `ordinal` ramp
    is too subtle. Runs cool -> warm (low -> high), which reads as a progression
    the way green/amber/red does.

    Deliberately NOT green->amber->red: red-vs-green is the one pair that
    collapses for the ~8% of men with deuteranopia/protanopia, and a chart whose
    whole point is telling three groups apart must not lean on that pair. Blue
    replaces green -- blue-vs-red survives every common CVD type. This is the
    standard accessible substitution, not a compromise.

    Hues are validated-palette slots (blue 1, yellow 4, red 8). Yellow sits
    below 3:1 on the light surface, so the relief rule applies: ship a legend
    or a table view alongside -- never hue alone.
    """
    ramp = [CATEGORICAL_LIGHT[0], CATEGORICAL_LIGHT[3], CATEGORICAL_LIGHT[7]]
    if _MODE == "dark":
        ramp = [CATEGORICAL_DARK[0], CATEGORICAL_DARK[3], CATEGORICAL_DARK[7]]
    if n == 2:
        return [ramp[0], ramp[2]]
    if n == 3:
        return ramp
    raise ValueError("signal() covers 2 or 3 ordered levels; past that use ordinal()")


def sequential_cmap(name: str = "chartstyle_blue"):
    """Continuous colormap from the single-hue blue ramp, for `c=`/`cmap=`.

    Use for CONTINUOUS magnitude only (price, minutes). Categorical identity
    uses series() slots -- never a colormap.
    """
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, SEQUENTIAL_BLUE)


# --------------------------------------------------------------------------
# Axis formatters
# --------------------------------------------------------------------------

def percent_axis(ax, axis: str = "y", decimals: int = 0, scale_100: bool = True):
    """Format an axis as percentages.

    scale_100=True  -> data is already 0-100 (e.g. 45.0 renders as "45%")
    scale_100=False -> data is a 0-1 fraction (e.g. 0.45 renders as "45%")
    """
    fmt = PercentFormatter(xmax=100 if scale_100 else 1, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)
    return ax


def currency_axis(ax, axis: str = "y", symbol: str = "£", suffix: str = "m",
                  decimals: int = 1):
    """Format an axis as currency, e.g. 5.5 -> "£5.5m"."""
    def _fmt(v, _pos):
        return f"{symbol}{v:,.{decimals}f}{suffix}"
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(FuncFormatter(_fmt))
    return ax


def thousands_axis(ax, axis: str = "y"):
    """Comma-separate large numbers, e.g. 3413 -> "3,413"."""
    fmt = FuncFormatter(lambda v, _p: f"{v:,.0f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)
    return ax


def title(ax, headline: str, subtitle: str | None = None):
    """Left-aligned headline with an optional muted subtitle beneath it.

    With a subtitle both lines are drawn as text above the axes so they stack
    cleanly -- ax.set_title() would sit at a fixed pad and collide with it.
    """
    base = mpl.rcParams["font.size"]
    if not subtitle:
        ax.set_title(headline, loc="left", color=ink("primary"))
        return ax

    # Offsets in POINTS, not axes fractions -- an axes-fraction offset scales
    # with figure height, so the gap balloons on tall charts.
    ax.set_title("")
    ax.annotate(headline, xy=(0.0, 1.0), xycoords="axes fraction",
                xytext=(0, 26), textcoords="offset points",
                fontsize=base + 3, fontweight="semibold",
                color=ink("primary"), va="bottom", ha="left",
                annotation_clip=False)
    ax.annotate(subtitle, xy=(0.0, 1.0), xycoords="axes fraction",
                xytext=(0, 9), textcoords="offset points",
                fontsize=base - 1, color=ink("secondary"),
                va="bottom", ha="left", annotation_clip=False)
    return ax
