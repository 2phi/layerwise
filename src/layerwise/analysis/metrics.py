"""
Defines the Metrices to calculate the Risk assosciated with
model outputs. (critical_weight & SSERR)

Data is taken from SnowPilot avalanche data points.
https://github.com/2phi/weac-data-hub
"""

from typing import Literal
import numpy as np
from scipy import stats


def calc_MaxStress_CDF_percentile(max_stress: float | np.ndarray) -> float | np.ndarray:
    """Utilizes the CDF of the lognormal distribution of avalanche data points to calculate the percentile of the max stress. The `max stress` is the percentage of overload on the weak layer due to the own weight of the slab. The distribution of avalanche data was used to compare the max stress of the slab with the max stress of the avalanche data."""
    lognorm_params = (
        np.float64(1.2979546316112562),
        -0.0005139743272564596,
        np.float64(0.024507949142599624),
    )
    return stats.lognorm.cdf(max_stress, *lognorm_params)


def calc_CCWeight_CDF_percentile(cc_weight: float | np.ndarray) -> float | np.ndarray:
    """Utilizes the CDF of the lognormal distribution of avalanche data points to calculate the percentile of the critical weight. The `critical weight` is the weight of the skier that causes the weak layer to be overloaded enough to form a crack (according to the coupled criterion). The distribution of avalanche data is used to compare any weight with available avalanche data."""
    lognorm_params = (
        np.float64(0.16659808869746978),
        np.float64(-393.01125799319584),
        np.float64(603.4546278244538),
    )
    return 1.0 - stats.lognorm.cdf(cc_weight, *lognorm_params)


def calc_SSERR_CDF_percentile(sserr: float | np.ndarray) -> float | np.ndarray:
    """Utilizes the CDF of the lognormal distribution of avalanche data points to calculate the percentile of the steady-state energy release rate. The `steady-state energy release rate` is the energy released by the slab in the steady state. The distribution of avalanche data is used to compare any energy release rate with available avalanche data."""
    lognorm_params = (
        np.float64(0.5709264240928602),
        -0.5320049310557289,
        np.float64(4.628583572796516),
    )
    return stats.lognorm.cdf(sserr, *lognorm_params)


def calc_SSMaxSxx_CDF_percentile(
    max_Sxx_stress_norm: float | np.ndarray,
) -> float | np.ndarray:
    """Utilizes the CDF of the lognormal distribution of avalanche data points to calculate the percentile of the maximum axial normal stress. The `maximum normalized axial stress` is the maximum stress in the slab divided by the tensile strength of the layer in the steady state (the `steady state` describes the state of the crack propagation far enough from the initial crack). The distribution of avalanche data is used to compare any stress with available avalanche data."""
    lognorm_params = (
        np.float64(0.5126855500517821),
        0.9105901961883057,
        np.float64(0.9805439259686874),
    )
    return 1.0 - stats.lognorm.cdf(max_Sxx_stress_norm, *lognorm_params)


def calc_SSSlabTensileCriterion_CDF_percentile(
    slab_tensile_criterion: float | np.ndarray,
) -> float | np.ndarray:
    """Utilizes the CDF of the lognormal distribution of avalanche data points to calculate the percentile of the slab tensile criterion. The `slab tensile criterion` is the criterion for the slab tensile failure. The distribution of avalanche data is used to compare any criterion with available avalanche data."""
    lognorm_params = (
        np.float64(6.11046745185916e-06),
        -16383.611971113376,
        np.float64(16384.387194107694),
    )
    return 1.0 - stats.lognorm.cdf(slab_tensile_criterion, *lognorm_params)


def combined_avalanche_criticality(
    CCWeight_CDF_percentile: float | np.ndarray,
    SSERR_CDF_percentile: float | np.ndarray,
    MaxStress_CDF_percentile: float | np.ndarray,
    SSMaxSxx_CDF_percentile: float | np.ndarray,
    SSlabTensileCriterion_CDF_percentile: float | np.ndarray,
    variant: Literal[
        "default",
        "all",
        "maxstress+cc_split_sserr+ssmaxsxx",
        "maxstress+cc_split_sserr+slabtensilecriterion",
        "maxstress+cc_split_sserr+ssmaxsxx+slabtensilecriterion",
    ] = "default",
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
