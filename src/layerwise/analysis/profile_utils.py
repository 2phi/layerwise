"""
profile_utils.py

This module provides utility functions for analyzing and manipulating snowpack profile data,
including loading, filtering, and parsing SnowPilot XML files and related structures for
layerwise snowpack analysis.

Dependencies:
    - os
    - copy
"""

import os
import copy
import numpy as np
from tqdm import tqdm

from weac.analysis.analyzer import Analyzer
from weac.analysis.criteria_evaluator import (
    CriteriaEvaluator,
    CoupledCriterionResult,
    SteadyStateResult,
    MaximalStressResult,
)
from weac.core.system_model import SystemModel
from weac.components import (
    ModelInput,
    Segment,
    ScenarioConfig,
    WeakLayer,
    Layer,
    Config,
)
from weac.utils.snowpilot_parser import SnowPilotParser


def load_snowpilot_parsers(
    data_dir: str,
    number_of_files: int | None = None,
    with_avalanche: bool | None = None,
    with_layer_of_concern: bool | None = None,
) -> tuple[list[str], list[SnowPilotParser]]:
    """
    Load SnowPilot XML files and return a list of parsers.
    """
    file_paths = []
    for directory in os.listdir(data_dir):
        dir_path = os.path.join(data_dir, directory)
        if not os.path.isdir(dir_path):
            continue
        for file in os.listdir(dir_path):
            if file.endswith(".xml"):
                file_paths.append(os.path.join(dir_path, file))

    # Build initial list of (path, parser) pairs
    paths_and_parsers = [
        (fp, SnowPilotParser(fp))
        for fp in (file_paths[:number_of_files] if number_of_files else file_paths)
    ]

    # Filter by avalanche proximity
    if with_avalanche is not None:
        paths_and_parsers = [
            (fp, p)
            for fp, p in paths_and_parsers
            if bool(p.snowpit.core_info.location.pit_near_avalanche) == with_avalanche
        ]

    # Filter by layer of concern
    if with_layer_of_concern is not None:
        paths_and_parsers = [
            (fp, p)
            for fp, p in paths_and_parsers
            if bool(p.snowpit.snow_profile.layer_of_concern) == with_layer_of_concern
        ]

    # Unzip back to separate lists
    if paths_and_parsers:
        temp_paths, temp_parsers = zip(*paths_and_parsers)
        return list(temp_paths), list(temp_parsers)
    return [], []


def calc_avg_density_profile(
    parsers: list[SnowPilotParser], spacing: int = 10, max_depth: int = 4000
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Calculate the average density profile from a list of SnowPilot parsers.
    """
    depth_profile = np.arange(0, max_depth, spacing)
    densities = np.zeros((len(parsers), len(depth_profile)))

    error_count = 0
    for i, parser in enumerate(parsers):
        try:
            layers, _ = parser.extract_layers()
        except Exception:
            densities[i, :] = np.nan
            error_count += 1
            continue

        heights = np.cumsum([layer.h for layer in layers])
        heights = np.concatenate([[0], heights])
        heights_idx = np.round(heights / spacing).astype(int)
        for j in range(1, len(heights_idx)):
            start_idx = int(heights_idx[j - 1])
            end_idx = int(heights_idx[j])
            if start_idx < len(depth_profile):
                densities[i, start_idx : min(end_idx, len(depth_profile))] = layers[
                    j - 1
                ].rho

    # Calculate the average density at each depth
    densities = np.where(densities == 0, np.nan, densities)
    average_density = np.nanmean(densities, axis=0)
    return depth_profile, average_density, error_count


def eval_weac_over_layers(
    layers: list[Layer],
    scenario_config: ScenarioConfig,
    segments: list[Segment],
    weak_layer: WeakLayer,
    criteria_evaluator: CriteriaEvaluator,
    spacing: float = 100,
) -> tuple[list[dict], list[Layer], WeakLayer]:
    """
    Evaluate WEAC criteria over different layer depths.
    """

    data_rows = []
    heights = np.cumsum([layer.h for layer in layers])
    # space evenly and append the last height
    wl_depths = np.arange(spacing, heights[-1], spacing).tolist()
    wl_depths.append(heights[-1])

    layers_copy = copy.deepcopy(layers)
    for wl_depth in tqdm(
        wl_depths,
        total=len(wl_depths),
        desc="Processing weak layers",
        leave=False,
    ):
        print("wl_depth: ", wl_depth)
        # only keep layers above the spacing
        mask = heights <= wl_depth
        new_layers = [layer for layer, keep in zip(layers_copy, mask) if keep]
        # Add truncated layer if needed
        depth = np.sum([layer.h for layer in new_layers]) if new_layers else 0.0
        if depth < wl_depth:
            additional_layer = copy.deepcopy(
                layers_copy[len(new_layers) if new_layers else 0]
            )
            additional_layer.h = wl_depth - depth
            new_layers.append(additional_layer)

        model_input = ModelInput(
            weak_layer=weak_layer,
            layers=new_layers,
            scenario_config=scenario_config,
            segments=segments,
        )
        system = SystemModel(model_input=model_input, config=Config(touchdown=True))
        analyzer = Analyzer(system)

        # Calculate stress envelope for impact resistance
        _, zs, _ = analyzer.rasterize_solution(mode="uncracked", num=4000)
        sigma_kPa = system.fq.sig(zs, unit="kPa")
        tau_kPa = system.fq.tau(zs, unit="kPa")

        stress_envelope = criteria_evaluator.stress_envelope(
            sigma=sigma_kPa,
            tau=tau_kPa,
            weak_layer=weak_layer,
        )
        max_stress = np.max(np.abs(stress_envelope))

        cc_result: CoupledCriterionResult = (
            criteria_evaluator.evaluate_coupled_criterion(
                system, print_call_stats=False
            )
        )

        ss_result: SteadyStateResult = criteria_evaluator.evaluate_SteadyState(
            system, vertical=False, print_call_stats=False
        )
        maximal_stress_result: MaximalStressResult = ss_result.maximal_stress_result

        data_rows.append(
            {
                "wl_depth": wl_depth,
                "impact_criterion": cc_result.initial_critical_skier_weight,
                "coupled_criterion": cc_result.critical_skier_weight,
                "sserr_result": ss_result.energy_release_rate,
                "touchdown_distance": ss_result.touchdown_distance,
                "ss_max_Sxx_norm": maximal_stress_result.max_Sxx_norm,
                "slab_tensile_criterion": maximal_stress_result.slab_tensile_criterion,
                "max_stress": max_stress,
            }
        )
    return data_rows, layers, weak_layer


def eval_weac_from_parser(
    parser: SnowPilotParser,
    scenario_config: ScenarioConfig,
    segments: list[Segment],
    weak_layer: WeakLayer,
    criteria_evaluator: CriteriaEvaluator | None = None,
    spacing: float = 100,
):
    """
    Wrapper for eval_weac_over_layers that takes a SnowPilotParser.
    """
    layers, _ = parser.extract_layers()
    data_rows, layers, weak_layer = eval_weac_over_layers(
        layers=layers,
        scenario_config=scenario_config,
        segments=segments,
        weak_layer=weak_layer,
        criteria_evaluator=criteria_evaluator,
        spacing=spacing,
    )
    return data_rows, layers, weak_layer


def eval_avalanche_pit(
    parser: SnowPilotParser,
    pit_info_dict: dict,
    scenario_config: ScenarioConfig,
    segments: list[Segment],
    weak_layer: WeakLayer,
    criteria_evaluator: CriteriaEvaluator,
):
    """
    Evaluates avalanche conditions for a given snowpit based on profile layers
    and specified weak layer depth.

    Parameters:
        parser (SnowPilotParser): Parser object for extracting snow profile layers.
        pit_info_dict (dict): Dictionary containing pit-specific information,
            including the weak layer depth ("WL_Depth").
        scenario_config (ScenarioConfig): Scenario configuration parameters.
        segments (list[Segment]): List of profile segments for analysis.
        weak_layer (WeakLayer): The weak layer object for evaluation.
        criteria_evaluator (CriteriaEvaluator): Object for evaluating instability criteria.

    Side Effects:
        Updates pit_info_dict with additional keys, e.g., "max_stress".

    Returns:
        None. Results are set in pit_info_dict as a side effect.
    """
    # Extract layers
    layers, _ = parser.extract_layers()
    heights = np.cumsum([layer.h for layer in layers])

    wl_depth = pit_info_dict["WL_Depth"]
    mask = heights <= wl_depth
    new_layers = [layer for layer, keep in zip(layers, mask) if keep]
    # Add truncated layer if needed
    depth = np.sum([layer.h for layer in new_layers]) if new_layers else 0.0
    if depth < wl_depth:
        additional_layer = copy.deepcopy(layers[len(new_layers) if new_layers else 0])
        additional_layer.h = wl_depth - depth
        new_layers.append(additional_layer)

    try:
        model_input = ModelInput(
            weak_layer=weak_layer,
            layers=new_layers,
            scenario_config=scenario_config,
            segments=segments,
        )
        system = SystemModel(model_input=model_input, config=Config(touchdown=True))
        analyzer = Analyzer(system)

        # Calculate stress envelope for impact resistance
        _, zs, _ = analyzer.rasterize_solution(mode="uncracked", num=4000)
        sigma_kPa = system.fq.sig(zs, unit="kPa")
        tau_kPa = system.fq.tau(zs, unit="kPa")

        stress_envelope = criteria_evaluator.stress_envelope(
            sigma=sigma_kPa,
            tau=tau_kPa,
            weak_layer=weak_layer,
        )
        max_stress = np.max(np.abs(stress_envelope))

        cc_result: CoupledCriterionResult = (
            criteria_evaluator.evaluate_coupled_criterion(
                system, print_call_stats=False
            )
        )
        ss_result: SteadyStateResult = criteria_evaluator.evaluate_SteadyState(
            system, vertical=False, print_call_stats=False
        )
        maximal_stress_result: MaximalStressResult = ss_result.maximal_stress_result

        pit_info_dict["max_stress"] = max_stress
        pit_info_dict["impact_criterion"] = cc_result.initial_critical_skier_weight
        pit_info_dict["coupled_criterion"] = cc_result.critical_skier_weight
        pit_info_dict["sserr_result"] = ss_result.energy_release_rate
        pit_info_dict["touchdown_distance"] = ss_result.touchdown_distance
        pit_info_dict["ss_max_Sxx_norm"] = maximal_stress_result.max_Sxx_norm
        pit_info_dict["slab_tensile_criterion"] = (
            maximal_stress_result.slab_tensile_criterion
        )

    except Exception as e:
        print(f"Error processing pit {parser.snowpit.core_info.pit_id}: {e}")
        raise e

    return pit_info_dict, layers, weak_layer
