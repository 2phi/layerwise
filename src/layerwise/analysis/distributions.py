"""
Fitted CDF distributions for weac output metrics.

Each entry maps a DatasetTag → MetricName → (cdf_function, params).

Available datasets
------------------
snowpilot
    SnowPilot avalanche pits (https://github.com/2phi/weac-data-hub).
anrissprofile
    SLF Anrissprofile pits (https://snowprofiler.slf.ch/).
anrissprofile+ect+rb
    Combined SLF dataset: all Anrissprofile pits + ECT pits (num_taps ≤ 21
    for max_stress/cc_weight, propagated=True for the rest) + RBlock pits
    (score ≤ 4 for max_stress/cc_weight, release type WB or MB for the rest).
    Best-fit distributions chosen by KS statistic (see scripts/plot_cdf_comparison.py).
"""

from collections.abc import Callable
from typing import Literal

import numpy as np
from scipy import stats

MetricName = Literal[
    "max_stress",
    "cc_weight",
    "sserr",
    "ss_max_sxx",
    "ss_slab_tensile_criterion",
]
DatasetTag = Literal["snowpilot", "slf_anrissprofile", "slf_anrissprofile+ect+rb"]
DATASET_TAGS: list[DatasetTag] = [
    "snowpilot",
    "slf_anrissprofile",
    "slf_anrissprofile+ect+rb",
]

METRIC_DISTRIBUTIONS: dict[
    DatasetTag,
    dict[
        MetricName,
        tuple[Callable[..., float | np.ndarray], tuple[float | np.float64, ...]],
    ],
] = {
    "snowpilot": {
        "max_stress": (
            stats.lognorm.cdf,
            (
                np.float64(1.2979546316112565),
                -0.0005139743272564586,
                np.float64(0.024507949142599624),
            ),
        ),
        "cc_weight": (
            stats.lognorm.cdf,
            (
                np.float64(0.16659808869746978),
                np.float64(-393.01125799319584),
                np.float64(603.4546278244538),
            ),
        ),
        "sserr": (
            stats.lognorm.cdf,
            (
                np.float64(0.5745407848044918),
                -0.41043983297455267,
                np.float64(3.6609116754625632),
            ),
        ),
        "ss_max_sxx": (
            stats.lognorm.cdf,
            (
                np.float64(0.3392606872351497),
                0.5904020113186575,
                np.float64(1.1728490574959878),
            ),
        ),
        "ss_slab_tensile_criterion": (
            stats.lognorm.cdf,
            (
                np.float64(0.044075704298945494),
                -1.9568876135894047,
                np.float64(2.1731491313444753),
            ),
        ),
    },
    "slf_anrissprofile": {
        "cc_weight": (
            stats.exponnorm.cdf,
            (
                np.float64(2.1821100126666053),
                np.float64(117.72698049430437),
                np.float64(60.72077615464323),
            ),
        ),
        "sserr": (
            stats.lognorm.cdf,
            (
                np.float64(0.6657784272076132),
                -0.09376769618612446,
                np.float64(3.2223214289568745),
            ),
        ),
        "max_stress": (
            stats.lognorm.cdf,
            (
                np.float64(1.3975321083373058),
                -0.00016871790563175137,
                np.float64(0.023345346860194006),
            ),
        ),
        "ss_max_sxx": (
            stats.exponnorm.cdf,
            (
                np.float64(1.7956623148370587),
                np.float64(1.4523608465771498),
                np.float64(0.22084848666266566),
            ),
        ),
        "ss_slab_tensile_criterion": (
            stats.lognorm.cdf,
            (
                np.float64(2.9331117259122433e-05),
                -3209.874412916295,
                np.float64(3210.0885445618237),
            ),
        ),
    },
    # Best fits by KS statistic on n≈10 000 combined SLF pits.
    "slf_anrissprofile+ect+rb": {
        "max_stress": (
            stats.lognorm.cdf,
            (
                np.float64(1.43526),
                np.float64(-0.000178489),
                np.float64(0.00843273),
            ),
        ),
        "cc_weight": (
            stats.exponnorm.cdf,
            (
                np.float64(2.37918),
                np.float64(76.6142),
                np.float64(47.1068),
            ),
        ),
        "sserr": (
            stats.lognorm.cdf,
            (
                np.float64(0.518908),
                np.float64(-0.661442),
                np.float64(3.15899),
            ),
        ),
        "ss_max_sxx": (
            stats.exponnorm.cdf,
            (
                np.float64(1.80176),
                np.float64(1.54366),
                np.float64(0.243166),
            ),
        ),
        "ss_slab_tensile_criterion": (
            stats.norm.cdf,
            (
                np.float64(0.227036),
                np.float64(0.100183),
            ),
        ),
    },
}
