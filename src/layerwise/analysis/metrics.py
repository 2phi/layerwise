"""
Defines the Metrices to calculate the Risk assosciated with
model outputs. (critical_weight & SSERR)

Data is taken from SnowPilot avalanche data points.
https://github.com/2phi/weac-data-hub
and SLF avalanche data points.
https://snowprofiler.slf.ch/
"""

from collections.abc import Callable
from typing import Literal
import numpy as np

from layerwise.analysis.distributions import (
    DATASET_TAGS,  # noqa: F401  re-exported for callers
    METRIC_DISTRIBUTIONS,
    DatasetTag,
    MetricName,
)

VARIANT_OPTIONS = Literal[
    "default",  # Combined CC and SSERR
    "all",  # All metrics
    "maxstress+cc_split_sserr+ssmaxsxx",  # Max stress + CC split + SSMAXSXX
    "maxstress+cc_split_sserr+slabtensilecriterion",  # Max stress + CC split + Slab tensile criterion
    "maxstress+cc_split_sserr+ssmaxsxx+slabtensilecriterion",  # Max stress + CC split + SSMAXSXX + Slab tensile criterion
]


def _cdf_percentile(
    value: float | np.ndarray,
    cdf_fn: Callable[..., float | np.ndarray],
    params: tuple[float | np.float64, ...],
    invert: bool = False,
) -> float | np.ndarray:
    percentile = cdf_fn(value, *params)
    return 1.0 - percentile if invert else percentile


def _get_distribution(
    dataset_tag: DatasetTag,
    metric_name: MetricName,
) -> tuple[Callable[..., float | np.ndarray], tuple[float | np.float64, ...]]:
    try:
        return METRIC_DISTRIBUTIONS[dataset_tag][metric_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown dataset_tag '{dataset_tag}' or metric '{metric_name}'. "
            f"Available tags: {tuple(METRIC_DISTRIBUTIONS.keys())}."
        ) from exc


def calc_MaxStress_CDF_percentile(
    max_stress: float | np.ndarray,
    dataset_tag: DatasetTag = "snowpilot",
) -> float | np.ndarray:
    """Utilizes the CDF of the lognormal distribution of avalanche data points to calculate the percentile of the max stress. The `max stress` is the percentage of overload on the weak layer due to the own weight of the slab. The distribution of avalanche data was used to compare the max stress of the slab with the max stress of the avalanche data."""
    cdf_fn, params = _get_distribution(dataset_tag, "max_stress")
    return _cdf_percentile(max_stress, cdf_fn, params)


def calc_CCWeight_CDF_percentile(
    cc_weight: float | np.ndarray,
    dataset_tag: DatasetTag = "snowpilot",
) -> float | np.ndarray:
    """Utilizes the CDF of the lognormal distribution of avalanche data points to calculate the percentile of the critical weight. The `critical weight` is the weight of the skier that causes the weak layer to be overloaded enough to form a crack (according to the coupled criterion). The distribution of avalanche data is used to compare any weight with available avalanche data."""
    cdf_fn, params = _get_distribution(dataset_tag, "cc_weight")
    return _cdf_percentile(cc_weight, cdf_fn, params, invert=True)


def calc_SSERR_CDF_percentile(
    sserr: float | np.ndarray,
    dataset_tag: DatasetTag = "snowpilot",
) -> float | np.ndarray:
    """Utilizes the CDF of the lognormal distribution of avalanche data points to calculate the percentile of the steady-state energy release rate. The `steady-state energy release rate` is the energy released by the slab in the steady state. The distribution of avalanche data is used to compare any energy release rate with available avalanche data."""
    cdf_fn, params = _get_distribution(dataset_tag, "sserr")
    return _cdf_percentile(sserr, cdf_fn, params)


def calc_SSMaxSxx_CDF_percentile(
    max_Sxx_stress_norm: float | np.ndarray,
    dataset_tag: DatasetTag = "snowpilot",
) -> float | np.ndarray:
    """Utilizes the CDF of the lognormal distribution of avalanche data points to calculate the percentile of the maximum axial normal stress. The `maximum normalized axial stress` is the maximum stress in the slab divided by the tensile strength of the layer in the steady state (the `steady state` describes the state of the crack propagation far enough from the initial crack). The distribution of avalanche data is used to compare any stress with available avalanche data."""
    cdf_fn, params = _get_distribution(dataset_tag, "ss_max_sxx")
    return _cdf_percentile(
        max_Sxx_stress_norm,
        cdf_fn,
        params,
        invert=True,
    )


def calc_SSSlabTensileCriterion_CDF_percentile(
    slab_tensile_criterion: float | np.ndarray,
    dataset_tag: DatasetTag = "snowpilot",
) -> float | np.ndarray:
    """Utilizes the CDF of the lognormal distribution of avalanche data points to calculate the percentile of the slab tensile criterion. The `slab tensile criterion` is the criterion for the slab tensile failure. The distribution of avalanche data is used to compare any criterion with available avalanche data."""
    cdf_fn, params = _get_distribution(dataset_tag, "ss_slab_tensile_criterion")
    return _cdf_percentile(
        slab_tensile_criterion,
        cdf_fn,
        params,
        invert=True,
    )


def combined_avalanche_criticality(
    CCWeight_CDF_percentile: float | np.ndarray,
    SSERR_CDF_percentile: float | np.ndarray,
    MaxStress_CDF_percentile: float | np.ndarray,
    SSMaxSxx_CDF_percentile: float | np.ndarray,
    SSlabTensileCriterion_CDF_percentile: float | np.ndarray,
    variant: VARIANT_OPTIONS = "default",
) -> float | np.ndarray:
    """Calculate combined avalanche criticality (0-1 scale)."""
    match variant:
        case "default":
            return CCWeight_CDF_percentile * SSERR_CDF_percentile
        case "all":
            return (
                MaxStress_CDF_percentile
                * CCWeight_CDF_percentile
                * SSERR_CDF_percentile
                * SSMaxSxx_CDF_percentile
                * SSlabTensileCriterion_CDF_percentile
            )
        case "maxstress+cc_split_sserr+ssmaxsxx":
            return (
                (MaxStress_CDF_percentile + CCWeight_CDF_percentile)
                / 2
                * (SSERR_CDF_percentile + SSMaxSxx_CDF_percentile)
                / 2
            )
        case "maxstress+cc_split_sserr+slabtensilecriterion":
            return (
                (MaxStress_CDF_percentile + CCWeight_CDF_percentile)
                / 2
                * (SSERR_CDF_percentile + SSlabTensileCriterion_CDF_percentile)
                / 2
            )
        case "maxstress+cc_split_sserr+ssmaxsxx+slabtensilecriterion":
            return (
                (MaxStress_CDF_percentile + CCWeight_CDF_percentile)
                / 2
                * (
                    SSERR_CDF_percentile
                    + SSMaxSxx_CDF_percentile
                    + SSlabTensileCriterion_CDF_percentile
                )
                / 3
            )


if __name__ == "__main__":
    print(calc_SSERR_CDF_percentile(30))
    print(calc_SSERR_CDF_percentile(300))
    print(calc_SSERR_CDF_percentile(3000))
    print(calc_SSERR_CDF_percentile(30000))

    print(combined_avalanche_criticality(0.5, 0.5, 0.5, 0.5, 0.5))
