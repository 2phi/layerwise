#!/usr/bin/env python
import os
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy import stats


from weac.components import (
    Segment,
    ScenarioConfig,
    CriteriaConfig,
    WEAK_LAYER,
)
from weac.analysis import CriteriaEvaluator
from weac.utils.snowpilot_parser import convert_to_mm, convert_to_deg
from layerwise.analysis.profile_utils import load_snowpilot_parsers, eval_avalanche_pit

## Settings
run_weac = True
dev = False
discard_pits_with_wl_above = 50  # mm
save_every = 50  # Flush to disk every 50 files

## Setup standard values
raw_data_dir = "../data/raw/slf-rb"
csv_file = "../data/misc/weac_over_slf_rb.csv"

# --- LOGGING SETUP ---
# This ensures you can track errors while you sleep
logging.basicConfig(
    filename="weac_batch_process_rb.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

## RUN WEAC
PHI = 35.0
weak_layer = WEAK_LAYER
segments = [
    Segment(length=10000, has_foundation=True, m=0.0),
    Segment(
        length=10000,
        has_foundation=True,
        m=0.0,
    ),
]
criteria_config = CriteriaConfig()
criteria_evaluator = CriteriaEvaluator(criteria_config)


def main():

    # 1. INITIAL SETUP & RESTART LOGIC
    processed_ids = set()
    if os.path.exists(csv_file):
        try:
            existing_df = pd.read_csv(csv_file)
            if "Pit_ID" in existing_df.columns:
                processed_ids = set(existing_df["Pit_ID"].astype(str).unique())
                print(
                    f"Found existing CSV. Skipping {len(processed_ids)} already processed pits."
                )
        except Exception as e:
            print(f"Could not read existing CSV, starting fresh: {e}")

    ## PARSE SNOWPYLOT
    paths, parsers = load_snowpilot_parsers(raw_data_dir)

    print(f"Overall number of files: {len(paths)}")

    paths_and_parsers = [(fp, pars) for fp, pars in zip(paths, parsers)]

    ## EXTRACT INFO
    pit_info_list = []
    unique_rb_entries = []
    for i, (fp, pit) in enumerate(paths_and_parsers):
        if i % 100 == 0:
            print(f"Processing pit {i} of {len(paths_and_parsers)}")
        for rb in pit.snowpit.stability_tests.RBlock:
            try:
                hs = pit.snowpit.snow_profile.hs
                if hs:
                    hs_mm = hs[0] * convert_to_mm[hs[1]]
                else:
                    hs_mm = None
                profile_depth = pit.snowpit.snow_profile.profile_depth
                if profile_depth:
                    profile_depth_mm = (
                        profile_depth[0] * convert_to_mm[profile_depth[1]]
                    )
                else:
                    profile_depth_mm = None
                depth_top_mm = rb.depth_top[0] * convert_to_mm[rb.depth_top[1]]
                test_score = rb.test_score
                fracture_character = rb.fracture_character
                release_type = rb.release_type
                slope_angle = pit.snowpit.core_info.location.slope_angle
                if slope_angle:
                    slope_angle_deg = slope_angle[0] * convert_to_deg[slope_angle[1]]
                else:
                    slope_angle_deg = PHI
                pit_info_dict = {
                    "Pit_ID": pit.snowpit.core_info.pit_id,
                    "Slope Angle": slope_angle_deg,
                    "HS": hs_mm,
                    "Profile Depth": profile_depth_mm,
                    "WL_Depth": depth_top_mm,
                    "RBlock_Score": test_score,
                    "RBlock_Fracture_Character": fracture_character,
                    "RBlock_Release_Type": release_type,
                }
                unique_rb_entries.append(pit)
                pit_info_list.append(pit_info_dict)
            except Exception as e:
                print(f"{e} Skipping pit {fp}")
    print(f"Number of unique RBlock entries: {len(unique_rb_entries)}")

    # PREFILTER PITS
    entries_to_run = [
        (p, d)
        for p, d in zip(unique_rb_entries, pit_info_list)
        if str(p.snowpit.core_info.pit_id) not in processed_ids
    ]

    if dev:
        entries_to_run = entries_to_run[:5]

    data_rows = []
    error_count = 0

    # Use os.cpu_count() or set a specific number like 4 or 8
    num_workers = os.cpu_count() - 4
    print(f"Launching with {num_workers} workers...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Map the work
        futures = {
            executor.submit(process_single_pit, entry): entry
            for entry in entries_to_run
        }

        for i, future in enumerate(
            tqdm(as_completed(futures), total=len(futures), desc="Parallel Processing")
        ):
            result, error = future.result()

            if result:
                data_rows.append(result)
            if error:
                error_count += 1
                logging.error("Error processing pit %s: %s", error[0], error[1])

            # PERIODIC SAVE (Every 50 completed tasks)
            if (i + 1) % save_every == 0 and data_rows:
                save_progress(data_rows, csv_file)
                data_rows = []

    # Final save for the last batch
    if data_rows:
        save_progress(data_rows, csv_file)

    print(f"Finished. Errors: {error_count}. Results in {csv_file}")
    print("Processing complete.")


def process_single_pit(entry):
    """
    Worker function: This runs on a separate CPU core.
    """
    pit, pit_info_dict = entry
    scenario_config = ScenarioConfig(
        phi=pit_info_dict["Slope Angle"],
        system_type="skier",
    )
    try:
        # We assume eval_avalanche_pit is CPU intensive
        updated_dict, _, _ = eval_avalanche_pit(
            pit,
            pit_info_dict,
            scenario_config,
            segments,
            weak_layer,
            criteria_evaluator,
        )
        return updated_dict, None
    except Exception as e:
        return None, (pit.snowpit.core_info.pit_id, str(e))


def save_progress(new_rows, filepath):
    """Appends new rows to CSV or creates it if it doesn't exist."""
    df_new = pd.DataFrame(new_rows)
    if not os.path.isfile(filepath):
        df_new.to_csv(filepath, index=False)
    else:
        # mode='a' is append, header=False prevents repeating column names
        df_new.to_csv(filepath, mode="a", index=False, header=False)


def analyze_data(df):
    sserr_median = df["sserr_result"].median()
    sserr_mean = df["sserr_result"].mean()
    sserr_std = df["sserr_result"].std()

    print(f"SSERR Median: {sserr_median}")
    print(f"SSERR Mean: {sserr_mean}")
    print(f"SSERR Std: {sserr_std}")

    cc_median = df["coupled_criterion"].median()
    cc_mean = df["coupled_criterion"].mean()
    cc_std = df["coupled_criterion"].std()

    print(f"CC Median: {cc_median}")
    print(f"CC Mean: {cc_mean}")
    print(f"CC Std: {cc_std}")

    max_stress_median = df["max_stress"].median()
    max_stress_mean = df["max_stress"].mean()
    max_stress_std = df["max_stress"].std()

    print(f"MAX STRESS Median: {max_stress_median}")
    print(f"MAX STRESS Mean: {max_stress_mean}")
    print(f"MAX STRESS Std: {max_stress_std}")

    ss_max_Sxx_norm_median = df["ss_max_Sxx_norm"].median()
    ss_max_Sxx_norm_mean = df["ss_max_Sxx_norm"].mean()
    ss_max_Sxx_norm_std = df["ss_max_Sxx_norm"].std()

    print(f"SS MAX SXX NORM Median: {ss_max_Sxx_norm_median}")
    print(f"SS MAX SXX NORM Mean: {ss_max_Sxx_norm_mean}")
    print(f"SS MAX SXX NORM Std: {ss_max_Sxx_norm_std}")

    slab_tensile_criterion_median = df["slab_tensile_criterion"].median()
    slab_tensile_criterion_mean = df["slab_tensile_criterion"].mean()
    slab_tensile_criterion_std = df["slab_tensile_criterion"].std()

    print(f"SLAB TENSILE CRITERION Median: {slab_tensile_criterion_median}")
    print(f"SLAB TENSILE CRITERION Mean: {slab_tensile_criterion_mean}")
    print(f"SLAB TENSILE CRITERION Std: {slab_tensile_criterion_std}")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["WL_Depth"],
            y=df["impact_criterion"],
            mode="markers",
            name="Impact Criterion",
            marker=dict(color="red"),
        )
    )
    fig.update_layout(xaxis_title="WL Depth (mm)", yaxis_title="Impact Criterion")
    fig.show()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["WL_Depth"],
            y=df["coupled_criterion"],
            mode="markers",
            name="Coupled Criterion",
            marker=dict(color="blue"),
        )
    )
    fig.update_layout(xaxis_title="WL Depth (mm)", yaxis_title="Coupled Criterion")
    fig.show()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["WL_Depth"],
            y=df["sserr_result"],
            mode="markers",
            name="SSERR",
            marker=dict(color="green"),
        )
    )
    fig.update_layout(xaxis_title="WL Depth (mm)", yaxis_title="SSERR")
    fig.show()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["WL_Depth"],
            y=df["touchdown_distance"],
            mode="markers",
            name="Touchdown Distance",
            marker=dict(color="yellow"),
        )
    )
    fig.update_layout(xaxis_title="WL Depth (mm)", yaxis_title="Touchdown Distance")
    fig.show()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["WL_Depth"],
            y=df["max_stress"],
            mode="markers",
            name="Max Stress",
            marker=dict(color="red"),
        )
    )
    fig.update_layout(xaxis_title="WL Depth (mm)", yaxis_title="Max Stress")
    fig.show()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["WL_Depth"],
            y=df["ss_max_Sxx_norm"],
            mode="markers",
            name="SS MAX SXX NORM",
            marker=dict(color="blue"),
        )
    )
    fig.update_layout(xaxis_title="WL Depth (mm)", yaxis_title="SS MAX SXX NORM")
    fig.show()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["WL_Depth"],
            y=df["slab_tensile_criterion"],
            mode="markers",
            name="SLAB TENSILE CRITERION",
            marker=dict(color="green"),
        )
    )
    fig.update_layout(xaxis_title="WL Depth (mm)", yaxis_title="SLAB TENSILE CRITERION")
    fig.show()

    # Bin wl depths according to 10 mm intervals
    wl_depths = df["WL_Depth"]
    max_wl_depth = max(wl_depths)
    min_wl_depth = min(wl_depths)

    # Create bins
    bin_width = 50
    bins = np.arange(min_wl_depth, max_wl_depth + bin_width, bin_width)

    # Use matplotlib's histogram which handles this automatically
    plt.hist(wl_depths, bins=bins, edgecolor="black", alpha=0.7)
    plt.xlabel("WL Depth (mm)")
    plt.ylabel("Number of Pits")
    plt.title("Number of Pits in Each WL Depth Bin")
    plt.show()

    wl_depths = df["WL_Depth"]
    df = df[df["WL_Depth"] > discard_pits_with_wl_above]

    # ### Coupled Criterion
    # Plot cumulative distribution of coupled criterion
    cc = df["coupled_criterion"][~np.isnan(df["coupled_criterion"])]
    sorted_cc = np.sort(cc)
    cdf = np.arange(1, len(sorted_cc) + 1) / len(sorted_cc)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sorted_cc,
            y=cdf,
            mode="markers",
            name="Coupled Criterion",
            marker=dict(color="red"),
        )
    )
    fig.update_layout(xaxis_title="Weight (kg)", yaxis_title="Cumulative Distribution")
    fig.show()

    # Fit a normal distribution to the data
    params_norm = stats.norm.fit(cc)
    cdf_values_norm = stats.norm.cdf(sorted_cc, *params_norm)

    # Fit a log-normal distribution to the data
    params_lognorm = stats.lognorm.fit(cc)
    cdf_values_lognorm = stats.lognorm.cdf(sorted_cc, *params_lognorm)

    # # Fit an exponential distribution to the data
    # params_expon = stats.expon.fit(cc)
    # cdf_values_expon = stats.expon.cdf(sorted_cc, *params_expon)

    # Fit an Exponential Normal distribution to the data
    params_exponnorm = stats.exponnorm.fit(cc)
    cdf_values_exponnorm = stats.exponnorm.cdf(sorted_cc, *params_exponnorm)
    print(params_exponnorm)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=sorted_cc, y=cdf_values_norm, mode="lines", name="Normal")
    )
    fig.add_trace(
        go.Scatter(x=sorted_cc, y=cdf_values_lognorm, mode="lines", name="Lognormal")
    )
    # fig.add_trace(go.Scatter(x=sorted_cc, y=cdf_values_expon, mode="lines", name="Exponential"))
    fig.add_trace(
        go.Scatter(
            x=sorted_cc, y=cdf_values_exponnorm, mode="lines", name="Exponential Normal"
        )
    )
    fig.add_trace(go.Scatter(x=sorted_cc, y=cdf, mode="markers", name="Data"))
    fig.show()

    # Plot cumulative distribution of coupled criterion
    sserr = df["sserr_result"][~np.isnan(df["sserr_result"])]
    sorted_sserr = np.sort(sserr)
    cdf = np.arange(1, len(sorted_sserr) + 1) / len(sorted_sserr)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sorted_sserr,
            y=cdf,
            mode="markers",
            name="SSERR",
            marker=dict(color="red"),
        )
    )
    fig.update_layout(xaxis_title="SSERR", yaxis_title="Cumulative Distribution")
    fig.show()

    # # Fit a normal distribution to the data
    # params_norm = stats.norm.fit(sserr)
    # cdf_values_norm = stats.norm.cdf(sorted_sserr, *params_norm)

    # Fit a log-normal distribution to the data
    params_lognorm = stats.lognorm.fit(sserr)
    cdf_values_lognorm = stats.lognorm.cdf(sorted_sserr, *params_lognorm)
    print(params_lognorm)

    # # Fit an exponential distribution to the data
    # params_expon = stats.expon.fit(sserr)
    # cdf_values_expon = stats.expon.cdf(sorted_sserr, *params_expon)

    # Fit an Exponential Normal distribution to the data
    params_exponnorm = stats.exponnorm.fit(sserr)
    cdf_values_exponnorm = stats.exponnorm.cdf(sorted_sserr, *params_exponnorm)

    fig = go.Figure()
    # fig.add_trace(go.Scatter(x=sorted_sserr, y=cdf_values_norm, mode="lines", name="Normal"))
    fig.add_trace(
        go.Scatter(x=sorted_sserr, y=cdf_values_lognorm, mode="lines", name="Lognormal")
    )
    # fig.add_trace(go.Scatter(x=sorted_sserr, y=cdf_values_expon, mode="lines", name="Exponential"))
    fig.add_trace(
        go.Scatter(
            x=sorted_sserr,
            y=cdf_values_exponnorm,
            mode="lines",
            name="Exponential Normal",
        )
    )
    fig.add_trace(go.Scatter(x=sorted_sserr, y=cdf, mode="markers", name="Data"))
    fig.show()

    # Plot cumulative distribution of coupled criterion
    max_stress = df["max_stress"][~np.isnan(df["max_stress"])]
    sorted_max_stress = np.sort(max_stress)
    cdf = np.arange(1, len(sorted_max_stress) + 1) / len(sorted_max_stress)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sorted_max_stress,
            y=cdf,
            mode="markers",
            name="max_stress",
            marker=dict(color="red"),
        )
    )
    fig.update_layout(xaxis_title="max_stress", yaxis_title="Cumulative Distribution")
    fig.show()

    # # Fit a normal distribution to the data
    params_norm = stats.norm.fit(max_stress)
    cdf_values_norm = stats.norm.cdf(sorted_max_stress, *params_norm)

    # Fit a log-normal distribution to the data
    params_lognorm = stats.lognorm.fit(max_stress)
    cdf_values_lognorm = stats.lognorm.cdf(sorted_max_stress, *params_lognorm)
    print(params_lognorm)

    # # Fit an exponential distribution to the data
    params_expon = stats.expon.fit(max_stress)
    cdf_values_expon = stats.expon.cdf(sorted_max_stress, *params_expon)

    # Fit an Exponential Normal distribution to the data
    params_exponnorm = stats.exponnorm.fit(max_stress)
    cdf_values_exponnorm = stats.exponnorm.cdf(sorted_max_stress, *params_exponnorm)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=sorted_max_stress, y=cdf_values_norm, mode="lines", name="Normal")
    )
    fig.add_trace(
        go.Scatter(
            x=sorted_max_stress, y=cdf_values_lognorm, mode="lines", name="Lognormal"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sorted_max_stress, y=cdf_values_expon, mode="lines", name="Exponential"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sorted_max_stress,
            y=cdf_values_exponnorm,
            mode="lines",
            name="Exponential Normal",
        )
    )
    fig.add_trace(go.Scatter(x=sorted_max_stress, y=cdf, mode="markers", name="Data"))
    fig.show()

    # Plot cumulative distribution of coupled criterion
    ss_max_Sxx_norm = df["ss_max_Sxx_norm"][~np.isnan(df["ss_max_Sxx_norm"])]
    sorted_ss_max_Sxx_norm = np.sort(ss_max_Sxx_norm)
    cdf = np.arange(1, len(sorted_ss_max_Sxx_norm) + 1) / len(sorted_ss_max_Sxx_norm)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sorted_ss_max_Sxx_norm,
            y=cdf,
            mode="markers",
            name="SS MAX SXX NORM",
            marker=dict(color="red"),
        )
    )
    fig.update_layout(
        xaxis_title="SS MAX SXX NORM", yaxis_title="Cumulative Distribution"
    )
    fig.show()

    # # Fit a normal distribution to the data
    params_norm = stats.norm.fit(ss_max_Sxx_norm)
    cdf_values_norm = stats.norm.cdf(sorted_ss_max_Sxx_norm, *params_norm)

    # Fit a log-normal distribution to the data
    params_lognorm = stats.lognorm.fit(ss_max_Sxx_norm)
    cdf_values_lognorm = stats.lognorm.cdf(sorted_ss_max_Sxx_norm, *params_lognorm)

    # # Fit an exponential distribution to the data
    params_expon = stats.expon.fit(ss_max_Sxx_norm)
    cdf_values_expon = stats.expon.cdf(sorted_ss_max_Sxx_norm, *params_expon)

    # Fit an Exponential Normal distribution to the data
    params_exponnorm = stats.exponnorm.fit(ss_max_Sxx_norm)
    cdf_values_exponnorm = stats.exponnorm.cdf(
        sorted_ss_max_Sxx_norm, *params_exponnorm
    )
    print(params_exponnorm)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sorted_ss_max_Sxx_norm, y=cdf_values_norm, mode="lines", name="Normal"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sorted_ss_max_Sxx_norm,
            y=cdf_values_lognorm,
            mode="lines",
            name="Lognormal",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sorted_ss_max_Sxx_norm,
            y=cdf_values_expon,
            mode="lines",
            name="Exponential",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sorted_ss_max_Sxx_norm,
            y=cdf_values_exponnorm,
            mode="lines",
            name="Exponential Normal",
        )
    )
    fig.add_trace(
        go.Scatter(x=sorted_ss_max_Sxx_norm, y=cdf, mode="markers", name="Data")
    )
    fig.show()

    # Plot cumulative distribution of coupled criterion
    slab_tensile_criterion = df["slab_tensile_criterion"][
        ~np.isnan(df["slab_tensile_criterion"])
    ]
    sorted_slab_tensile_criterion = np.sort(slab_tensile_criterion)
    cdf = np.arange(1, len(sorted_slab_tensile_criterion) + 1) / len(
        sorted_slab_tensile_criterion
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sorted_slab_tensile_criterion,
            y=cdf,
            mode="markers",
            name="SLAB TENSILE CRITERION",
            marker=dict(color="red"),
        )
    )
    fig.update_layout(
        xaxis_title="SLAB TENSILE CRITERION", yaxis_title="Cumulative Distribution"
    )
    fig.show()

    # # Fit a normal distribution to the data
    params_norm = stats.norm.fit(slab_tensile_criterion)
    cdf_values_norm = stats.norm.cdf(sorted_slab_tensile_criterion, *params_norm)

    # Fit a log-normal distribution to the data
    params_lognorm = stats.lognorm.fit(slab_tensile_criterion)
    cdf_values_lognorm = stats.lognorm.cdf(
        sorted_slab_tensile_criterion, *params_lognorm
    )
    print(params_lognorm)

    # # Fit an exponential distribution to the data
    params_expon = stats.expon.fit(slab_tensile_criterion)
    cdf_values_expon = stats.expon.cdf(sorted_slab_tensile_criterion, *params_expon)

    # Fit an Exponential Normal distribution to the data
    params_exponnorm = stats.exponnorm.fit(slab_tensile_criterion)
    cdf_values_exponnorm = stats.exponnorm.cdf(
        sorted_slab_tensile_criterion, *params_exponnorm
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sorted_slab_tensile_criterion,
            y=cdf_values_norm,
            mode="lines",
            name="Normal",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sorted_slab_tensile_criterion,
            y=cdf_values_lognorm,
            mode="lines",
            name="Lognormal",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sorted_slab_tensile_criterion,
            y=cdf_values_expon,
            mode="lines",
            name="Exponential",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sorted_slab_tensile_criterion,
            y=cdf_values_exponnorm,
            mode="lines",
            name="Exponential Normal",
        )
    )
    fig.add_trace(
        go.Scatter(x=sorted_slab_tensile_criterion, y=cdf, mode="markers", name="Data")
    )
    fig.show()


if __name__ == "__main__":
    if run_weac:
        main()
    else:
        df = pd.read_csv(csv_file)
        analyze_data(df)
