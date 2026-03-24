#!/usr/bin/env python
"""
Plot CDFs for 5 weac output variables.

For each variable, the three filtered data sources are pooled into one
combined dataset. The empirical CDF of that combined dataset is plotted
alongside 4 fitted distributions (Normal, Lognormal, Exponential, ExpNorm).

Outputs:
  - data/misc/cdf_<variable>.png  (5 figures)
  - data/misc/cdf_fit_params.txt  (all fitted parameters + KS statistics)
"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ANRISS_CSV = "../data/misc/weac_over_slf_anrissprofile.csv"
ECT_CSV = "../data/misc/weac_over_slf_ect.csv"
RB_CSV = "../data/misc/weac_over_slf_rb.csv"
OUTPUT_TXT = "../data/misc/cdf_fit_params.txt"
OUTPUT_FIG_TEMPLATE = "../data/misc/cdf_{var}.png"

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
df_anriss = pd.read_csv(ANRISS_CSV)
df_ect = pd.read_csv(ECT_CSV)
df_rb = pd.read_csv(RB_CSV)

# ---------------------------------------------------------------------------
# Filtered subsets
# ---------------------------------------------------------------------------

# ECT: num_taps <= 21  (max_stress / coupled_criterion)
df_ect["ECT_NumTaps"] = pd.to_numeric(df_ect["ECT_NumTaps"], errors="coerce")
df_ect_taps = df_ect[df_ect["ECT_NumTaps"] <= 21].copy()

# ECT: propagated == True  (Sxx_max_norm / slab_criterion / sserr)
df_ect_prop = df_ect[df_ect["ECT_Propagation"] == True].copy()

# RB: numeric score <= 4  (max_stress / coupled_criterion)
def _rb_score_num(val):
    m = re.search(r"\d+", str(val))
    return int(m.group()) if m else None

df_rb["_score_num"] = df_rb["RBlock_Score"].apply(_rb_score_num)
df_rb_score4 = df_rb[df_rb["_score_num"] <= 4].copy()

# RB: release type WB or MB  (Sxx_max_norm / slab_criterion / sserr)
df_rb_wb_mb = df_rb[df_rb["RBlock_Release_Type"].isin(["WB", "MB"])].copy()

# ---------------------------------------------------------------------------
# Distributions to fit
# ---------------------------------------------------------------------------
DISTS: dict[str, stats.rv_continuous] = {
    "Normal":      stats.norm,
    "Lognormal":   stats.lognorm,
    "Exponential": stats.expon,
    "ExpNorm":     stats.exponnorm,
}

DIST_COLORS = {
    "Normal":      "tab:blue",
    "Lognormal":   "tab:orange",
    "Exponential": "tab:green",
    "ExpNorm":     "tab:red",
}
DIST_STYLES = {
    "Normal":      ("-",  2.0),
    "Lognormal":   ("--", 2.0),
    "Exponential": (":",  2.5),
    "ExpNorm":     ("-.", 2.0),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
DIST_PARAM_LABELS = {
    "Normal":      ["loc", "scale"],
    "Lognormal":   ["s", "loc", "scale"],
    "Exponential": ["loc", "scale"],
    "ExpNorm":     ["K", "loc", "scale"],
}

def fmt_params(dist_name: str, params: tuple) -> str:
    labels = DIST_PARAM_LABELS[dist_name]
    return ", ".join(f"{l}={v:.6g}" for l, v in zip(labels, params))


# ---------------------------------------------------------------------------
# Plot config: 5 variables, each pooling 3 filtered sources
# ---------------------------------------------------------------------------
GROUP_A_SOURCES = [
    ("Anrissprofile (all)",  df_anriss),
    ("ECT (num_taps ≤ 21)", df_ect_taps),
    ("RB (score ≤ 4)",       df_rb_score4),
]
GROUP_B_SOURCES = [
    ("Anrissprofile (all)", df_anriss),
    ("ECT (propagated)",    df_ect_prop),
    ("RB (WB or MB)",       df_rb_wb_mb),
]

PLOTS = [
    {"var": "max_stress",            "title": "Max Stress",        "sources": GROUP_A_SOURCES},
    {"var": "coupled_criterion",     "title": "Coupled Criterion",  "sources": GROUP_A_SOURCES},
    {"var": "ss_max_Sxx_norm",       "title": "Sxx Max Norm",       "sources": GROUP_B_SOURCES},
    {"var": "slab_tensile_criterion","title": "Slab Criterion",     "sources": GROUP_B_SOURCES},
    {"var": "sserr_result",          "title": "SSERR Result",       "sources": GROUP_B_SOURCES},
]

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
txt_lines: list[str] = ["CDF Fit Parameters", "=" * 70, ""]

for cfg in PLOTS:
    var     = cfg["var"]
    title   = cfg["title"]
    sources = cfg["sources"]

    # Pool all sources into one combined array
    pieces = [df_src[var].dropna().values for _, df_src in sources]
    data   = np.concatenate(pieces)
    n      = len(data)

    source_desc = " + ".join(
        f"{name} (n={len(arr)})" for (name, _), arr in zip(sources, pieces)
    )

    txt_lines += [
        f"\n{'='*70}",
        f"Variable : {title}  ({var})",
        f"Combined : {source_desc}",
        f"Total n  : {n}",
        f"{'='*70}",
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Empirical CDF of the combined data
    sorted_data = np.sort(data)
    ecdf = np.arange(1, n + 1) / n
    ax.plot(sorted_data, ecdf, ".", color="black", alpha=0.3, markersize=2,
            label=f"Empirical CDF  (n={n})")

    # Fitted distributions
    for dist_name, dist in DISTS.items():
        ls, lw = DIST_STYLES[dist_name]
        color  = DIST_COLORS[dist_name]
        try:
            params    = dist.fit(data)
            cdf_fitted = dist.cdf(sorted_data, *params)
            ks_stat, ks_pval = stats.ks_1samp(
                data, dist.cdf, args=params, alternative="two-sided"
            )
            ax.plot(sorted_data, cdf_fitted,
                    linestyle=ls, linewidth=lw, color=color,
                    label=f"{dist_name}  (KS={ks_stat:.3f})")
            param_str = fmt_params(dist_name, params)
            txt_lines.append(
                f"  {dist_name:<14}  {param_str}"
                f"   |  KS={ks_stat:.4f}  p={ks_pval:.4f}"
            )
        except Exception as exc:
            txt_lines.append(f"  {dist_name:<14}  FAILED: {exc}")

    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlabel(var, fontsize=10)
    ax.set_ylabel("CDF", fontsize=10)
    ax.set_title(
        f"Cumulative Distribution — {title}\n"
        f"({'; '.join(name for name, _ in sources)})",
        fontsize=10,
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    out_path = OUTPUT_FIG_TEMPLATE.format(var=var)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")

# ---------------------------------------------------------------------------
# Write parameter file
# ---------------------------------------------------------------------------
with open(OUTPUT_TXT, "w") as fh:
    fh.write("\n".join(txt_lines))
print(f"Saved fit parameters to {OUTPUT_TXT}")
