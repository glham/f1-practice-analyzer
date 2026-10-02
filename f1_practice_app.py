import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full", app_title="F1 Practice Analyzer")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    import io
    import re

    import fastf1
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from fastf1 import plotting as f1plot

    import chartstyle as cs

    cs.use_style(base_size=13)

    _cache_dir = mo.notebook_dir() / "data" / "fastf1_cache"
    _cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(_cache_dir))
    return cs, f1plot, fastf1, io, np, pd, plt, re


@app.cell
def _(mo):
    mo.md(
        """
        # 🏎️ F1 Practice Analyzer

        Lap-by-lap practice pace for your F1 Fantasy picks: **qualifying sims**
        and **long runs**, coloured by constructor, plotted against tyre age.
        First load of a weekend downloads timing data (a minute or two);
        it's cached on disk after that.
        """
    )
    return


@app.cell
def _(mo):
    year_dd = mo.ui.dropdown(
        options={str(_y): _y for _y in range(2023, 2027)},
        value="2026",
        label="Season",
    )
    return (year_dd,)


@app.cell
def _(fastf1, mo, pd, year_dd):
    _sched = fastf1.get_event_schedule(year_dd.value, include_testing=False)
    _sched = _sched[_sched["RoundNumber"] > 0]
    _past = _sched[_sched["EventDate"] <= pd.Timestamp.now() + pd.DateOffset(days=2)]
    _default = _past.iloc[-1] if len(_past) else _sched.iloc[0]
    event_dd = mo.ui.dropdown(
        options={
            f"R{int(_r.RoundNumber):02d} · {_r.EventName}": int(_r.RoundNumber)
            for _r in _sched.itertuples()
        },
        value=f"R{int(_default.RoundNumber):02d} · {_default.EventName}",
        label="Race weekend",
    )
    return (event_dd,)


@app.cell
def _(mo):
    session_ms = mo.ui.multiselect(
        # SQ = Sprint Qualifying (sprint weekends only); it simply fails to
        # load on a normal weekend and is reported as such.
        options=["FP1", "FP2", "FP3", "SQ"],
        value=["FP1", "FP2", "FP3", "SQ"],
        label="Sessions",
    )
    load_btn = mo.ui.run_button(label="Load session data", kind="success")
    return load_btn, session_ms


@app.cell
def _(event_dd, load_btn, mo, session_ms, year_dd):
    mo.hstack(
        [year_dd, event_dd, session_ms, load_btn],
        justify="start",
        gap=1.5,
        wrap=True,
    )
    return


@app.cell
def _(event_dd, f1plot, fastf1, load_btn, mo, pd, session_ms, year_dd):
    mo.stop(
        not load_btn.value,
        mo.md("*Choose a weekend and press **Load session data**.*"),
    )

    _frames = []
    loaded_sessions = []
    failed_sessions = []
    team_colors = {}
    with mo.status.spinner(title="Loading practice sessions…") as _sp:
        for _name in session_ms.value:
            try:
                _sp.update(subtitle=_name)
                _ses = fastf1.get_session(year_dd.value, event_dd.value, _name)
                _ses.load(telemetry=False, weather=False, messages=False)
                _laps = _ses.laps.copy()
                _laps["Session"] = _name
                _frames.append(_laps)
                loaded_sessions.append(_name)
                for _t in _laps["Team"].dropna().unique():
                    if _t not in team_colors:
                        try:
                            team_colors[_t] = f1plot.get_team_color(_t, session=_ses)
                        except Exception:
                            pass
            except Exception:
                failed_sessions.append(_name)

    mo.stop(
        not _frames,
        mo.md(
            f"**No lap data for {' / '.join(session_ms.value)} at this event.** "
            "Has practice run yet? (Sprint weekends only have FP1.)"
        ),
    )
    all_laps = pd.concat(_frames, ignore_index=True)
    event_label = event_dd.selected_key
    # Panels for the lap charts: the three FP slots, with SQ taking the
    # place of a practice session that didn't run (sprint weekends).
    panel_sessions = [_s for _s in ["FP1", "FP2", "FP3"] if _s in loaded_sessions]
    if "SQ" in loaded_sessions and len(panel_sessions) < 3:
        panel_sessions.append("SQ")
    while len(panel_sessions) < 3:
        panel_sessions.append(
            next(_s for _s in ["FP1", "FP2", "FP3"] if _s not in panel_sessions)
        )
    return (
        all_laps,
        event_label,
        failed_sessions,
        loaded_sessions,
        panel_sessions,
        team_colors,
    )


@app.cell
def _(all_laps, pd, year_dd):
    def fmt_lap(sec):
        if pd.isna(sec):
            return "—"
        _m, _s = divmod(float(sec), 60)
        return f"{int(_m)}:{_s:06.3f}"

    laps_sec = all_laps[
        all_laps["LapTime"].notna() & all_laps["TyreLife"].notna()
    ].copy()
    laps_sec["LapSec"] = laps_sec["LapTime"].dt.total_seconds()

    # Some sessions arrive with empty Team fields (e.g. Barcelona 2026 FP1),
    # which would silently drop the whole session from every team-filtered
    # chart. Repair per driver from any session where the team IS known.
    laps_sec["Team"] = laps_sec["Team"].replace("", pd.NA)
    _team_of = (
        laps_sec.dropna(subset=["Team"])
        .drop_duplicates("Driver")
        .set_index("Driver")["Team"]
    )
    laps_sec["Team"] = laps_sec["Team"].fillna(laps_sec["Driver"].map(_team_of))

    # Drop test/rookie drivers. With several sessions loaded, anyone who only
    # ran FP1 is a rookie outing; with FP1 alone, fall back to the season's
    # championship standings to identify the regulars.
    excluded_drivers = []
    _sessions = set(laps_sec["Session"])
    if len(_sessions) > 1:
        _driver_sessions = laps_sec.groupby("Driver")["Session"].agg(set)
        excluded_drivers = sorted(
            _driver_sessions[_driver_sessions == {"FP1"}].index
        )
    else:
        try:
            from fastf1.ergast import Ergast

            _standings = Ergast().get_driver_standings(season=year_dd.value)
            _codes = set(_standings.content[0]["driverCode"].dropna())
            excluded_drivers = sorted(set(laps_sec["Driver"]) - _codes)
        except Exception:
            excluded_drivers = []
    laps_sec = laps_sec[~laps_sec["Driver"].isin(excluded_drivers)]
    return excluded_drivers, fmt_lap, laps_sec


@app.cell
def _(event_label, excluded_drivers, failed_sessions, loaded_sessions, mo):
    _msg = f"Loaded **{', '.join(loaded_sessions)}** for **{event_label}**."
    if failed_sessions:
        _msg += (
            f" Couldn't load {', '.join(failed_sessions)} — not run yet, "
            "or not part of this weekend's format."
        )
    if excluded_drivers:
        _msg += f" Excluded test/rookie drivers: {', '.join(excluded_drivers)}."
    mo.md(_msg)
    return


@app.cell
def _(laps_sec, mo):
    _pace = laps_sec.groupby("Team")["LapSec"].min().sort_values()
    team_sel = mo.ui.multiselect(
        options=sorted(_pace.index),
        value=list(_pace.index[:3]),
        label="Teams to compare",
    )
    team_sel
    return (team_sel,)


@app.cell
def _(cs, fmt_lap, io, mo, plt, re):
    def tyre_scatter(
        laps, team_colors, headline, subtitle, ylo, yhi, sessions=None
    ):
        from matplotlib.lines import Line2D
        from matplotlib.ticker import FuncFormatter, MaxNLocator

        comp_marker = {
            "SOFT": "o",
            "MEDIUM": "s",
            "HARD": "^",
            "INTERMEDIATE": "D",
            "WET": "X",
        }

        # ALWAYS a 2x2 grid: FP1 TL, FP2 TR, FP3 BL, legends BR — a session
        # with nothing to show keeps its panel with a note, so the layout
        # never jumps around between weekends.
        sessions = list(sessions or ["FP1", "FP2", "FP3"])
        plt.close("all")
        fig, _axes = plt.subplots(
            2, 2, figsize=(22.0, 9.5), sharey=True, sharex=True, squeeze=False
        )
        _flat = [ax for _row in _axes for ax in _row]
        axs = _flat[: len(sessions)]

        _fallback = cs.palette()
        teams = sorted(laps["Team"].dropna().unique())
        tcolor = {
            t: team_colors.get(t, _fallback[i % len(_fallback)])
            for i, t in enumerate(teams)
        }
        dorder = (
            laps[["Driver", "Team"]]
            .drop_duplicates()
            .sort_values(["Team", "Driver"])
            .reset_index(drop=True)
        )
        # Teammates share the constructor colour: first driver filled,
        # second hollow.
        hollow = {}
        for _t, _g in dorder.groupby("Team"):
            for _i, _r in enumerate(_g.itertuples()):
                hollow[_r.Driver] = _i % 2 == 1

        for ax, sname in zip(axs, sessions):
            sd = laps[laps["Session"] == sname]
            for (drv, team, comp), dd in sd.groupby(["Driver", "Team", "Compound"]):
                _c = tcolor.get(team)
                _m = comp_marker.get(comp, "P")
                if hollow.get(drv):
                    ax.scatter(
                        dd["TyreLife"], dd["LapSec"],
                        s=64, marker=_m, facecolors="none",
                        edgecolors=_c, linewidths=1.8, zorder=3,
                    )
                else:
                    ax.scatter(
                        dd["TyreLife"], dd["LapSec"],
                        s=64, marker=_m, color=_c,
                        edgecolors="white", linewidths=0.7, zorder=3,
                    )
            if ax is not axs[0]:
                ax.set_title(
                    sname, loc="left", fontsize=13,
                    color=cs.ink("secondary"), pad=8,
                )
            if sd.empty:
                ax.text(
                    0.5, 0.5, "no qualifying laps",
                    transform=ax.transAxes, ha="center", va="center",
                    color=cs.ink("muted"), fontsize=14, style="italic",
                )
            ax.set_ylim(ylo, yhi)
            ax.tick_params(labelleft=True, labelbottom=True, labelsize=11)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: fmt_lap(v)))
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlabel("Tyre age (laps)", fontsize=12)
            ax.set_ylabel("Lap time", fontsize=12)

        driver_handles = [
            Line2D(
                [], [], linestyle="", marker="o", markersize=10,
                markerfacecolor="none" if hollow[r.Driver] else tcolor[r.Team],
                markeredgecolor=tcolor[r.Team], markeredgewidth=2.0,
                label=f"{r.Driver} · {r.Team}",
            )
            for r in dorder.itertuples()
        ]
        comp_handles = [
            Line2D(
                [], [], linestyle="", marker=_m, markersize=10,
                color=cs.ink("secondary"), label=_c.title(),
            )
            for _c, _m in comp_marker.items()
            if _c in set(laps["Compound"].dropna())
        ]
        # spare bottom-right grid cell: no data, both legends live here
        _lax = _flat[3]
        _lax.axis("off")
        _leg1 = _lax.legend(
            handles=driver_handles, loc="upper left",
            bbox_to_anchor=(0.02, 0.98), ncol=2, fontsize=12,
            title="Driver", alignment="left",
        )
        _lax.add_artist(_leg1)
        if comp_handles:
            _lax.legend(
                handles=comp_handles, loc="lower left",
                bbox_to_anchor=(0.02, 0.02), ncol=len(comp_handles),
                fontsize=12, title="Tyre", alignment="left",
            )

        cs.title(axs[0], headline, f"{sessions[0]} · {subtitle}")
        plt.tight_layout()
        return fig

    def fig_svg(fig):
        """Crisp full-width vector render — not a scaled PNG screenshot."""
        _buf = io.StringIO()
        fig.savefig(_buf, format="svg", bbox_inches="tight")
        plt.close(fig)
        _svg = re.sub(r'width="[\d.]+pt"', 'width="100%"', _buf.getvalue(), count=1)
        _svg = re.sub(r'height="[\d.]+pt"', "", _svg, count=1)
        return mo.Html(_svg)

    return fig_svg, tyre_scatter


@app.cell
def _(
    event_label,
    fig_svg,
    laps_sec,
    mo,
    panel_sessions,
    team_colors,
    team_sel,
    tyre_scatter,
):
    mo.stop(not team_sel.value, mo.md("*Select at least one team above.*"))
    _best = laps_sec["LapSec"].min()
    # Filter (not just zoom) to the quali window so slow laps at high tyre
    # age don't stretch the x-axis.
    _sel = laps_sec[
        laps_sec["Team"].isin(team_sel.value)
        & (laps_sec["LapSec"] <= _best + 2.5)
    ]
    fig_svg(
        tyre_scatter(
            _sel,
            team_colors,
            "Qualifying sims",
            f"laps within ~2.5s of the weekend's best — {event_label}",
            _best - 0.4,
            _best + 2.5,
            sessions=panel_sessions,
        )
    )
    return


@app.cell
def _(laps_sec, np, pd):
    # A long run = 5+ consecutive clean laps in one stint, tolerating ONE
    # mid-run mistake: two clean streaks separated by exactly one bad lap
    # merge (the mistake lap itself is never plotted). A stint without such
    # a run is dropped WHOLE — alternating push/cool-down quali sims never
    # qualify, since only one bridge per run is allowed. Clean = flagged
    # accurate AND within 7% of the stint's own fastest lap (cool-down and
    # traffic laps run 10-30% over). Crawl stints (median >110% of the
    # fastest stint's median) are dropped too.
    _runs = laps_sec[
        laps_sec["IsAccurate"] & laps_sec["Session"].str.startswith("FP")
    ].copy()
    _fast = _runs.groupby(["Session", "Driver", "Stint"])["LapSec"].transform("min")
    _runs = _runs[_runs["LapSec"] <= _fast * 1.07]
    _keep = []
    for _key, _g in _runs.groupby(["Session", "Driver", "Stint"]):
        _g = _g.sort_values("LapNumber")
        _nums = _g["LapNumber"].to_numpy()
        _starts = np.flatnonzero(np.diff(_nums, prepend=_nums[0] - 9) != 1)
        _streaks = np.split(np.arange(len(_nums)), _starts[1:])
        _ok = set()
        for _i, _s in enumerate(_streaks):
            if len(_s) >= 5:
                _ok.update(_s)
            if _i + 1 < len(_streaks):
                _nxt = _streaks[_i + 1]
                if (
                    _nums[_nxt[0]] - _nums[_s[-1]] == 2
                    and len(_s) + len(_nxt) >= 5
                ):
                    _ok.update(_s)
                    _ok.update(_nxt)
        if _ok:
            _keep.append(_g.iloc[sorted(_ok)])
    if _keep:
        long_runs = pd.concat(_keep)
        _med = long_runs.groupby(["Session", "Driver", "Stint"])[
            "LapSec"
        ].transform("median")
        long_runs = long_runs[_med <= _med.min() * 1.10]
    else:
        long_runs = _runs.iloc[0:0]
    return (long_runs,)


@app.cell
def _(
    event_label,
    fig_svg,
    laps_sec,
    long_runs,
    mo,
    panel_sessions,
    team_colors,
    team_sel,
    tyre_scatter,
):
    mo.stop(not team_sel.value, mo.md(""))
    mo.stop(
        long_runs.empty,
        mo.md(
            "*No qualifying long runs (5+ consecutive clean laps) in the "
            "loaded sessions.*"
        ),
    )
    _sel = long_runs[long_runs["Team"].isin(team_sel.value)]
    # say who's missing and why, so an absent driver reads as data, not a bug
    _no_run = sorted(
        set(laps_sec.loc[laps_sec["Team"].isin(team_sel.value), "Driver"])
        - set(_sel["Driver"])
    )
    mo.stop(
        _sel.empty,
        mo.md(
            "*No qualifying long run (5+ consecutive clean laps) for any "
            f"selected-team driver: {', '.join(_no_run)}.*"
        ),
    )
    # Y-band from ALL teams' laps so the zoom stays stable while switching
    # team selections.
    _ylo = long_runs["LapSec"].quantile(0.01) - 0.4
    _yhi = long_runs["LapSec"].quantile(0.99) + 0.4
    _all = laps_sec[laps_sec["Team"].isin(team_sel.value)]
    _off = int((~_all["LapSec"].between(_ylo, _yhi)).sum())
    mo.vstack(
        [
            fig_svg(
                tyre_scatter(
                    _sel,
                    team_colors,
                    "Long runs",
                    "5+ consecutive clean laps, race-sim pace — "
                    f"{event_label}",
                    _ylo,
                    _yhi,
                    sessions=panel_sessions,
                )
            ),
            mo.md(
                "*No qualifying long run (needs 5+ consecutive clean laps): "
                f"{', '.join(_no_run)}*"
            )
            if _no_run
            else mo.md(""),
            # Same y-band, NO filtering — every lap the selected teams turned,
            # so the long-run rules above can be judged against the raw data.
            fig_svg(
                tyre_scatter(
                    _all,
                    team_colors,
                    "All laps (unfiltered)",
                    "every lap, same y-scale as the long-run chart — "
                    f"{_off} of {len(_all)} laps fall outside it — {event_label}",
                    _ylo,
                    _yhi,
                    sessions=panel_sessions,
                )
            ),
        ],
        gap=0,
    )
    return


@app.cell
def _(fmt_lap, long_runs, mo, pd, team_sel):
    mo.stop(not team_sel.value or long_runs.empty, mo.md(""))
    _d = long_runs[long_runs["Team"].isin(team_sel.value)]
    mo.stop(len(_d) == 0, mo.md(""))
    # one row per qualifying stint; ticked rows feed the box plot and the
    # degradation chart, so e.g. a flattering fresh-soft run can be dropped
    _stints = (
        _d.groupby(["Driver", "Team", "Session", "Stint"])
        .agg(
            Compound=("Compound", "first"),
            Laps=("LapSec", "size"),
            AgeFrom=("TyreLife", "min"),
            AgeTo=("TyreLife", "max"),
            MedianSec=("LapSec", "median"),
            BestSec=("LapSec", "min"),
        )
        .reset_index()
        .sort_values("MedianSec")
    )
    stint_table = mo.ui.table(
        pd.DataFrame(
            {
                "Driver": _stints["Driver"],
                "Team": _stints["Team"],
                "Session": _stints["Session"],
                "Stint": _stints["Stint"].astype(int),
                "Tyre": _stints["Compound"].str.title(),
                "Laps": _stints["Laps"],
                "Tyre age": [
                    f"{int(_a)}–{int(_b)}"
                    for _a, _b in zip(_stints["AgeFrom"], _stints["AgeTo"])
                ],
                "Median": _stints["MedianSec"].map(fmt_lap),
                "Best": _stints["BestSec"].map(fmt_lap),
            }
        ).reset_index(drop=True),
        selection="multi",
        initial_selection=list(range(len(_stints))),
        page_size=20,
    )
    return (stint_table,)


@app.cell
def _(long_runs, stint_table, team_sel):
    # laps from the ticked stints only — the box plot and degradation chart
    # both read from this
    _d = long_runs[long_runs["Team"].isin(team_sel.value)]
    _picked = stint_table.value
    _keys = (
        set(
            zip(
                _picked["Session"],
                _picked["Driver"],
                _picked["Stint"].astype(int),
            )
        )
        if len(_picked)
        else set()
    )
    active_runs = _d[
        [
            (_s, _dr, int(_st)) in _keys
            for _s, _dr, _st in zip(_d["Session"], _d["Driver"], _d["Stint"])
        ]
    ]
    return (active_runs,)


@app.cell
def _(
    active_runs,
    cs,
    event_label,
    fig_svg,
    fmt_lap,
    mo,
    np,
    plt,
    stint_table,
    team_colors,
):
    _d = active_runs
    _table_block = mo.vstack(
        [
            mo.md(
                "**The tyres behind each box** — one row per qualifying "
                "stint; untick rows to drop those stints from the box plot "
                "and the degradation chart:"
            ),
            stint_table,
        ],
        gap=0.5,
    )
    mo.stop(
        len(_d) == 0,
        mo.vstack([mo.md("*No stints ticked — tick at least one row.*"), _table_block]),
    )

    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter

    _order = (
        _d.groupby(["Driver", "Team"])["LapSec"].median().sort_values().reset_index()
    )
    _fallback = cs.palette()
    _rng = np.random.default_rng(0)
    plt.close("all")
    _fig, _ax = plt.subplots(figsize=(20.0, 6.0))
    for _xi, _r in enumerate(_order.itertuples()):
        _laps = _d.loc[_d["Driver"] == _r.Driver, "LapSec"].values
        _c = team_colors.get(_r.Team, _fallback[_xi % len(_fallback)])
        if len(_laps) >= 5:
            _ax.boxplot(
                [_laps], positions=[_xi], widths=0.5, patch_artist=True,
                showfliers=False,
                boxprops=dict(facecolor=_c, alpha=0.55,
                              edgecolor=cs.ink("primary"), linewidth=1.0),
                medianprops=dict(color=cs.ink("primary"), linewidth=1.8),
                whiskerprops=dict(color=cs.ink("primary"), linewidth=1.0),
                capprops=dict(color=cs.ink("primary"), linewidth=1.0),
            )
        _ax.scatter(
            _xi + (_rng.random(len(_laps)) - 0.5) * 0.3, _laps,
            s=26, color=_c, alpha=0.95,
            edgecolor=cs.ink("surface"), linewidths=0.7, zorder=3,
        )
        _ax.scatter(
            [_xi], [float(np.mean(_laps))], s=90, facecolor="white",
            edgecolor=cs.ink("primary"), linewidths=1.5, zorder=4,
        )
    _ax.set_xticks(range(len(_order)))
    _ax.set_xticklabels(
        [
            f"{_r.Driver}\n{int((_d['Driver'] == _r.Driver).sum())} laps"
            for _r in _order.itertuples()
        ],
        fontsize=12,
    )
    _ax.set_xlim(-0.6, len(_order) - 0.4)
    _ax.tick_params(axis="y", labelsize=11)
    _ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: fmt_lap(v)))
    _ax.grid(axis="x", visible=False)
    _ax.margins(y=0.08)
    _ax.legend(
        handles=[
            Patch(facecolor=team_colors.get(_t, "grey"), alpha=0.55,
                  edgecolor=cs.ink("primary"), label=_t)
            for _t in sorted(_d["Team"].unique())
        ],
        loc="upper left", fontsize=12,
    )
    cs.title(
        _ax,
        "Long-run pace, driver by driver",
        "clean long-run laps pooled across sessions, fastest median first; "
        f"black line = median, white dot = mean — {event_label}",
    )
    _ax.set_ylabel("Lap time", fontsize=12)
    plt.tight_layout()
    mo.vstack([fig_svg(_fig), _table_block], gap=0.5)
    return


@app.cell
def _(
    active_runs,
    cs,
    event_label,
    fig_svg,
    mo,
    np,
    plt,
    team_colors,
):
    _d = active_runs
    mo.stop(len(_d) == 0, mo.md(""))

    # Net degradation per stint: OLS slope of lap time vs tyre age across
    # the stint's clean laps. "Net" = fuel-burn gain is NOT corrected out,
    # matching what the stopwatch would show in the race.
    _rows = []
    for (_drv, _team, _sess, _stint), _g in _d.groupby(
        ["Driver", "Team", "Session", "Stint"]
    ):
        if _g["TyreLife"].nunique() < 4:
            continue
        _slope = float(np.polyfit(_g["TyreLife"], _g["LapSec"], 1)[0])
        _rows.append(
            dict(
                label=(
                    f"{_drv} · {_g['Compound'].iloc[0].title()} · {_sess} "
                    f"({len(_g)} laps)"
                ),
                team=_team,
                slope=_slope,
            )
        )
    mo.stop(not _rows, mo.md("*No stints long enough to estimate degradation.*"))
    _rows.sort(key=lambda r: r["slope"])

    _fallback = cs.palette()
    plt.close("all")
    _fig, _ax = plt.subplots(figsize=(20.0, 1.2 + 0.52 * len(_rows)))
    _y = np.arange(len(_rows))
    for _yi, _r in enumerate(_rows):
        _c = team_colors.get(_r["team"], _fallback[_yi % len(_fallback)])
        _ax.barh(
            _yi, _r["slope"], height=0.55, color=_c,
            edgecolor=cs.ink("surface"), linewidth=1.1,
        )
        _ax.annotate(
            f"{_r['slope']:+.3f}",
            (_r["slope"], _yi),
            xytext=(6 if _r["slope"] >= 0 else -6, 0),
            textcoords="offset points",
            va="center",
            ha="left" if _r["slope"] >= 0 else "right",
            fontsize=11,
            color=cs.ink("primary"),
        )
    _ax.set_yticks(_y)
    _ax.set_yticklabels([_r["label"] for _r in _rows], fontsize=12)
    _ax.invert_yaxis()
    _ax.axvline(0, color=cs.ink("primary"), linewidth=1.0)
    _ax.grid(axis="y", visible=False)
    _ax.grid(axis="x", visible=True)
    _ax.margins(x=0.12)
    _ax.tick_params(axis="x", labelsize=11)
    cs.title(
        _ax,
        "Tyre degradation on long runs",
        "seconds lost per lap of tyre age (per-stint regression, gentlest "
        f"first); negative = getting faster (fuel burn, track grip) — {event_label}",
    )
    _ax.set_xlabel("Lap time change per lap of tyre age (s)", fontsize=12)
    plt.tight_layout()
    fig_svg(_fig)
    return


@app.cell
def _(laps_sec, mo):
    # teams whose two cars run different engine specs get one bar PER
    # DRIVER in the engine-mode chart instead of best-of-team (e.g.
    # Monza 2026: Bearman on a new spec, Ocon on the old one)
    _teams = sorted(laps_sec["Team"].dropna().unique())
    split_ms = mo.ui.multiselect(
        options=_teams,
        value=[_t for _t in _teams if "Haas" in _t],
        label="Split into per-driver bars (cars on different engine "
              "specs)")
    return (split_ms,)


@app.cell
def _(
    cs,
    event_dd,
    event_label,
    fastf1,
    fig_svg,
    laps_sec,
    mo,
    np,
    pd,
    plt,
    split_ms,
    team_colors,
    year_dd,
):
    # Engine-mode sniff test (sandbag detector): straight-line speed vs the
    # field this weekend, baselined against the same team's gap at each of
    # the LAST THREE qualifyings — engines run wound up in quali. Negative
    # against all three = robust sandbag signal; a single negative bar can
    # just be that track's wing choice.
    _sched = fastf1.get_event_schedule(year_dd.value, include_testing=False)
    _prev = _sched[_sched["RoundNumber"].between(1, event_dd.value - 1)]
    mo.stop(
        len(_prev) == 0,
        mo.md(
            "*Engine-mode check needs an earlier round this season as a "
            "qualifying baseline — none available yet.*"
        ),
    )
    _rounds = sorted(_prev["RoundNumber"].astype(int))[-3:]

    _split = set(split_ms.value)

    def _team_trap(_laps):
        # per driver: median of their top-3 trap readings (smooths tow/DRS
        # one-offs); per team: best of its drivers — except split teams
        # (cars on different engine specs), which keep one row per driver
        _d = _laps.dropna(subset=["SpeedST"])
        _d = _d[_d["Team"].notna() & (_d["Team"] != "")]
        _drv = _d.groupby(["Team", "Driver"])["SpeedST"].apply(
            lambda s: s.nlargest(3).median()
        )
        _out = {}
        for (_tm, _dr), _v in _drv.items():
            _key = f"{_tm} — {_dr}" if _tm in _split else _tm
            if _key not in _out or _v > _out[_key]:
                _out[_key] = _v
        return pd.Series(_out, dtype=float)

    _baselines = []  # (label, field-relative deficit series), oldest first
    with mo.status.spinner(
        title="Loading recent qualifyings (engine-mode baselines)…"
    ) as _sp:
        for _rnd in _rounds:
            _name = _prev.loc[_prev["RoundNumber"] == _rnd, "EventName"].iloc[0]
            try:
                _sp.update(subtitle=f"R{_rnd:02d} {_name}")
                _q = fastf1.get_session(year_dd.value, _rnd, "Q")
                _q.load(telemetry=False, weather=False, messages=False)
                _t = _team_trap(_q.laps)
                _baselines.append((f"R{_rnd:02d} {_name}", _t - _t.median()))
            except Exception:
                pass
    mo.stop(
        not _baselines,
        mo.md(
            "*Couldn't load any recent qualifying for the engine-mode "
            "baseline.*"
        ),
    )

    _fp = _team_trap(laps_sec[laps_sec["Session"].str.startswith("FP")])
    _fp_def = _fp - _fp.median()
    _ind = pd.DataFrame(
        {_lab: _fp_def - _qdef for _lab, _qdef in _baselines}
    ).dropna(how="all")
    mo.stop(len(_ind) == 0, mo.md("*No overlapping speed-trap data to compare.*"))
    _ind = _ind.loc[_ind.mean(axis=1).sort_values().index]

    # Theory test: this weekend's ACTUAL qualifying, measured identically.
    # Down in practice but back to baseline in quali = was sandbagging;
    # down in both = just this track's config, not the engine.
    # Prefer the Grand Prix qualifying; on a sprint weekend before Saturday,
    # Sprint Qualifying is the first full-power session and stands in.
    _ind_q, _qnow_label = None, None
    for _qname, _qlabel in (("Q", "qualifying"), ("SQ", "Sprint Qualifying")):
        try:
            with mo.status.spinner(
                title=f"Loading this weekend's {_qlabel} (theory test)…"
            ):
                _qnow = fastf1.get_session(year_dd.value, event_dd.value, _qname)
                _qnow.load(telemetry=False, weather=False, messages=False)
            if _qnow.laps.empty:
                continue
            _qnow_t = _team_trap(_qnow.laps)
            _qnow_def = _qnow_t - _qnow_t.median()
            _ind_q = pd.DataFrame(
                {_lab: _qnow_def - _qdef for _lab, _qdef in _baselines}
            ).reindex(_ind.index)
            _qnow_label = _qlabel
            break
        except Exception:
            continue

    _n = len(_baselines)
    _fallback = cs.palette()
    plt.close("all")
    _fig, (_ax, _axq) = plt.subplots(
        1, 2, figsize=(22.0, 1.4 + 0.36 * _n * len(_ind)),
        sharey=True, sharex=True,
    )
    _h = 0.72 / _n

    def _grouped(_axx, _df):
        # split-team rows share the team colour: the second car of a
        # team draws hollow (chart-style teammate rule)
        _seen, _hollow = set(), set()
        for _lab0 in _df.index:
            _bt = _lab0.split(" — ")[0]
            if " — " in _lab0 and _bt in _seen:
                _hollow.add(_lab0)
            _seen.add(_bt)
        for _yi, (_team, _row) in enumerate(_df.iterrows()):
            _base = team_colors.get(
                _team.split(" — ")[0], _fallback[_yi % len(_fallback)])
            _steps = list(cs.shades(_base, _n)) if _n > 1 else [_base]
            for _bi, _lab in enumerate(_df.columns):
                if pd.isna(_row[_lab]):
                    continue
                if _team in _hollow:
                    _axx.barh(
                        _yi + (_bi - (_n - 1) / 2) * _h,
                        _row[_lab],
                        height=_h * 0.9,
                        facecolor="none",
                        edgecolor=_steps[_bi],
                        linewidth=1.6,
                    )
                else:
                    _axx.barh(
                        _yi + (_bi - (_n - 1) / 2) * _h,
                        _row[_lab],
                        height=_h * 0.9,
                        color=_steps[_bi],
                        edgecolor=cs.ink("surface"),
                        linewidth=0.8,
                    )
        _axx.axvline(0, color=cs.ink("primary"), linewidth=1.0)
        _axx.grid(axis="y", visible=False)
        _axx.margins(x=0.15, y=0.02)
        _axx.tick_params(axis="x", labelsize=11)
        _axx.set_xlabel(
            "Trap-speed deficit vs field, relative to own quali baseline (kph)",
            fontsize=12,
        )

    _grouped(_ax, _ind)
    _ax.set_yticks(range(len(_ind)))
    _ax.set_yticklabels(_ind.index, fontsize=12)
    _ax.invert_yaxis()

    _axq.set_title(
        f"…and in THIS weekend's actual {_qnow_label or 'qualifying'}",
        loc="left", fontsize=13, color=cs.ink("secondary"), pad=8,
    )
    if _ind_q is not None:
        _grouped(_axq, _ind_q)
    else:
        _axq.text(
            0.5, 0.5, "qualifying not run yet",
            transform=_axq.transAxes, ha="center", va="center",
            color=cs.ink("muted"), fontsize=14, style="italic",
        )

    from matplotlib.patches import Patch as _Patch

    _grey = (
        list(cs.shades(cs.ink("secondary"), _n)) if _n > 1
        else [cs.ink("secondary")]
    )
    _ax.legend(
        handles=[
            _Patch(facecolor=_grey[_bi], edgecolor=cs.ink("primary"), label=_lab)
            for _bi, _lab in enumerate(_ind.columns)
        ],
        loc="lower left",
        fontsize=11,
        title="Baseline quali (darker = more recent)",
    )
    cs.title(
        _ax,
        "Engine-mode sniff test — who's turned down?",
        "practice trap speed vs field, minus the same team's gap at each of "
        "the last three qualifyings; negative against ALL THREE = robust "
        f"sandbag signal — {event_label}",
    )
    plt.tight_layout()
    mo.vstack(
        [
            split_ms,
            fig_svg(_fig),
            mo.md(
                f"*Left: practice. Right: this weekend's {_qnow_label or 'qualifying'}, measured "
                "identically. Negative left but ~0 right = engine was turned "
                "down in practice (sandbagging confirmed). Negative in BOTH = "
                "that's just this track's config/wing, not the engine. Trust "
                "4+ kph, not 1–2.*"
            ),
        ],
        gap=0.5,
    )
    return


@app.cell
def _(cs, fig_svg, mo, pd, plt):
    # Race-pace model results (built offline by build_race_pace_model.py;
    # this cell only renders data/race_pace_results.json)
    import json as _json

    _rp_path = mo.notebook_dir() / "data" / "race_pace_results.json"
    mo.stop(
        not _rp_path.exists(),
        mo.md(
            "---\n## Race-pace model\n*No results yet — run "
            "`py build_race_pace_model.py` to build the dataset and "
            "backtest, then refresh.*"
        ),
    )
    _res = _json.loads(_rp_path.read_text())

    _sets = pd.DataFrame(_res["feature_sets"]).T
    _sets["spearman"] = _sets["spearman"].astype(float)
    _tbl = (
        pd.DataFrame({
            "feature set": _sets.index,
            "walk-forward Spearman": _sets["spearman"].round(3),
            "picked race-pace leader": _sets["leader_hits"].astype(str)
            + "/" + _sets["events"].astype(str),
        })
        .sort_values("walk-forward Spearman", ascending=False)
        .reset_index(drop=True)
    )

    _singles = pd.DataFrame(_res["single_feature_corr"]).T.reset_index()
    _singles.columns = ["feature", "Spearman vs race pace", "rows"]
    _singles = _singles.sort_values(
        "Spearman vs race pace", ascending=False
    ).reset_index(drop=True)

    _order = _sets.sort_values("spearman")
    _fig, _ax = plt.subplots(figsize=(9.5, 4.0))
    _ax.barh(_order.index, _order["spearman"], color=cs.series(0), height=0.55)
    cs.title(
        _ax,
        "What predicts race pace",
        "Mean per-event Spearman rank corr., chronological walk-forward "
        "backtest — higher is better",
    )
    _ax.set_xlim(0, max(0.75, float(_order["spearman"].max()) + 0.05))

    mo.vstack(
        [
            mo.md(
                "---\n## Race-pace model — which practice signals "
                "predict the race?"
            ),
            mo.md(
                f"*{_res['rows']} driver-events over {_res['events']} race "
                f"weekends, seasons {_res['seasons'][0]}–{_res['seasons'][-1]}, "
                f"generated {_res['generated']}. Each feature set trains only "
                "on earlier events and predicts the next; test-event counts "
                "differ because long-run and sniff coverage varies.*"
            ),
            fig_svg(_fig),
            mo.hstack(
                [
                    mo.vstack([
                        mo.md("**Feature sets, head-to-head**"),
                        mo.ui.table(_tbl, selection=None),
                    ]),
                    mo.vstack([
                        mo.md("**Single features, pooled correlation**"),
                        mo.ui.table(_singles, selection=None),
                    ]),
                ],
                gap=1.5,
            ),
            mo.md(
                "*Read: the FP3 best lap adjusted for tyre age/compound is "
                "the strongest simple predictor. Long-run median adds a "
                "little; deg slope and lap scatter add nothing out of "
                "sample. Permutation importances on the full fit are "
                "in-sample and overstate trap/sniff — trust the backtest "
                "column.*"
            ),
        ],
        gap=0.5,
    )
    return


if __name__ == "__main__":
    app.run()
