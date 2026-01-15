"""
plotly_snow_profile.py

This module provides functionality to generate and plot snow stratification profiles using Plotly.
It defines plotting utilities and color schemes to visualize the properties of snowpack layers, including
density, thickness, hardness, and grain type.

Dependencies:
    - plotly
    - pandas
    - numpy
    - scipy
    - PIL

Author: [Your Name or Organization]
"""

import os
import copy
from typing import List, Literal
from itertools import groupby
from io import BytesIO

from PIL import Image
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from weac.components import Layer
from layerwise.analysis.metrics import (
    calc_CCWeight_CDF_percentile,
    calc_SSERR_CDF_percentile,
    calc_MaxStress_CDF_percentile,
    calc_SSMaxSxx_CDF_percentile,
    calc_SSSlabTensileCriterion_CDF_percentile,
    combined_avalanche_criticality,
)


def snow_profile(layers: list[Layer]):
    """
    Generates a snow stratification profile plot using Plotly.

    Parameters:
    - weaklayer_thickness (float): Thickness of the weak layer in the snowpack.
    - layers (list of dicts): Each dict has keys density, thickness, hardness, and grain of a layer.

    Returns:
    - fig (go.Figure): A Plotly figure object representing the snow profile.
    """
    # Define colors
    COLORS = {
        "slab_fill": "#9ec1df",
        "slab_line": "rgba(4, 110, 124, 0.812)",
        "weak_layer_fill": "#E57373",
        "weak_layer_line": "#FFCDD2",
        "weak_layer_text": "#FFCDD2",
        "substratum_fill": "#607D8B",
        "substratum_line": "#ECEFF1",
        "substratum_text": "#ECEFF1",
        "background": "rgb(134, 148, 160)",
        "lines": "rgb(134, 148, 160)",
    }

    # reverse layers
    layers = copy.deepcopy(layers)

    # Compute total height and set y-axis maximum
    total_height = sum(layer.h for layer in layers)
    y_max = max(total_height, 450)  # Ensure y_max is at least 450

    # Compute x-axis maximum based on layer densities
    max_density = max((layer.rho for layer in layers), default=400)
    x_max = max(1.05 * max_density, 300)  # Ensure x_max is at least 300

    # Initialize the Plotly figure
    fig = go.Figure()

    # Initialize variables for plotting layers
    previous_density = 0  # Start from zero density

    # Define positions for annotations (table columns)
    col_width = 0.12
    col_width = min(col_width * x_max, 30)
    x_pos = {
        "col0_start": 0 * col_width,
        "col1_start": 1 * col_width,
        "col2_start": 2 * col_width,
        "col3_start": 3 * col_width,
        "col3_end": 4 * col_width,
    }

    # Compute midpoints for annotation placement
    first_column_mid = (x_pos["col0_start"] + x_pos["col1_start"]) / 2
    second_column_mid = (x_pos["col1_start"] + x_pos["col2_start"]) / 2
    third_column_mid = (x_pos["col2_start"] + x_pos["col3_start"]) / 2
    fourth_column_mid = (x_pos["col3_start"] + x_pos["col3_end"]) / 2

    # Calculate average height per table row
    num_layers = max(len(layers), 1)
    min_table_row_height = (y_max / 2) / num_layers
    max_table_row_height = 300
    avg_row_height = (y_max) / num_layers
    avg_row_height = min(avg_row_height, max_table_row_height)
    avg_row_height = max(avg_row_height, min_table_row_height)
    # Taken space for the table
    table_height = avg_row_height * num_layers
    table_offset = total_height - table_height

    # Initialize current table height
    current_height = 0
    current_table_y = table_offset

    # Loop through each layer and plot
    for layer in layers:
        density = layer.rho
        thickness = layer.h
        hand_hardness = layer.hand_hardness
        grain = layer.grain_type

        # Define layer boundaries
        layer_bottom = current_height
        layer_top = current_height + thickness

        # Plot the layer
        fig.add_shape(
            type="rect",
            x0=-density,
            x1=0,
            y0=layer_bottom,
            y1=layer_top,
            fillcolor=COLORS["slab_fill"],
            line=dict(width=0.4, color=COLORS["slab_fill"]),
            layer="above",
        )

        # Plot lines connecting previous and current densities
        fig.add_shape(
            type="line",
            x0=-previous_density,
            y0=layer_bottom,
            x1=-density,
            y1=layer_bottom,
            line=dict(color=COLORS["slab_line"], width=1.2),
        )
        fig.add_shape(
            type="line",
            x0=-density,
            y0=layer_bottom,
            x1=-density,
            y1=layer_top,
            line=dict(color=COLORS["slab_line"], width=1.2),
        )

        # Add heights on the right of layer changes
        fig.add_annotation(
            x=first_column_mid,
            y=layer_bottom,
            text=str(round(layer_bottom)),
            showarrow=False,
            font=dict(size=10),
            xanchor="center",
            yanchor="middle",
        )

        # Define table row boundaries
        table_bottom = current_table_y
        table_top = current_table_y + avg_row_height

        # Add table grid lines
        fig.add_shape(
            type="line",
            x0=x_pos["col1_start"],
            y0=table_bottom,
            x1=x_pos["col3_end"],
            y1=table_bottom,
            line=dict(color="lightgrey", width=0.5),
        )

        # Add annotations for density, grain form, and hand hardness
        fig.add_annotation(
            x=second_column_mid,
            y=(table_bottom + table_top) / 2,
            text=str(round(density)),
            showarrow=False,
            font=dict(size=10),
            xanchor="center",
            yanchor="middle",
        )
        fig.add_annotation(
            x=third_column_mid,
            y=(table_bottom + table_top) / 2,
            text=grain if grain else "-",
            showarrow=False,
            font=dict(size=10),
            xanchor="center",
            yanchor="middle",
        )
        fig.add_annotation(
            x=fourth_column_mid,
            y=(table_bottom + table_top) / 2,
            text=hand_hardness if hand_hardness else "-",
            showarrow=False,
            font=dict(size=10),
            xanchor="center",
            yanchor="middle",
        )

        # Lines from layer edges to table
        fig.add_shape(
            type="line",
            x0=0,
            y0=layer_top,
            x1=x_pos["col1_start"],
            y1=table_top,
            line=dict(color="lightgrey", width=0.5),
        )

        # Update variables for next iteration
        previous_density = density
        current_height = layer_top
        current_table_y = table_top

    # Additional cases which are not covered by the loop
    # Additional case: Add density line from last layer to x=0
    fig.add_shape(
        type="line",
        x0=-previous_density,
        y0=total_height,
        x1=0.0,
        y1=total_height,
        line=dict(width=1.2, color=COLORS["slab_line"]),
    )
    # Additional case: Add table grid of last layer
    fig.add_shape(
        type="line",
        x0=x_pos["col1_start"],
        y0=total_height,
        x1=x_pos["col3_end"],
        y1=total_height,
        line=dict(color="lightgrey", width=0.5),
    )
    # Additional case: Add layer edge line from first layer to table
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=x_pos["col1_start"],
        y1=table_offset,
        line=dict(width=0.5, color="lightgrey"),
    )

    fig.add_annotation(
        x=x_pos["col0_start"],
        y=total_height,
        text=str(total_height),
        showarrow=False,
        font=dict(size=10),
        xanchor="left",
        yanchor="middle",
    )

    # Vertical lines for table columns
    for x in [
        x_pos["col1_start"],
        x_pos["col2_start"],
        x_pos["col3_start"],
    ]:
        fig.add_shape(
            type="line",
            x0=x,
            y0=0,
            x1=x,
            y1=y_max,
            line=dict(color="lightgrey", width=0.5),
        )

    column_header_y = -200
    # Horizontal line at table header
    fig.add_shape(
        type="line",
        x0=0,
        y0=column_header_y,
        x1=x_pos["col3_end"],
        y1=column_header_y,
        line=dict(color="lightgrey", width=0.5),
    )

    # Annotations for table headers
    header_y_position = (column_header_y) / 2
    fig.add_annotation(
        x=first_column_mid,
        y=header_y_position,
        text="H",  # "H<br>cm",  # "H (cm)",
        showarrow=False,
        font=dict(size=10),
        xanchor="center",
        yanchor="middle",
    )
    fig.add_annotation(
        x=second_column_mid,
        y=header_y_position,
        text="D",  # 'D<br>kg/m³',  # "Density (kg/m³)",
        showarrow=False,
        font=dict(size=10),
        xanchor="center",
        yanchor="middle",
    )
    fig.add_annotation(
        x=third_column_mid,
        y=header_y_position,
        text="F",  # "GF",
        showarrow=False,
        font=dict(size=10),
        xanchor="center",
        yanchor="middle",
    )
    fig.add_annotation(
        x=fourth_column_mid,
        y=header_y_position,
        text="R",
        showarrow=False,
        font=dict(size=10),
        xanchor="center",
        yanchor="middle",
    )

    fig.add_annotation(
        x=0.0,
        y=-0.06,
        text="H: Height (cm)  D: Density (kg/m³)  F: Grain Form  R: Hand Hardness",
        showarrow=False,
        xref="paper",
        yref="paper",
        font=dict(size=10),
        align="left",
    )

    # Add horizontal grid lines at spacing of 100mm
    for y in np.arange(0, total_height, 100):
        fig.add_trace(
            go.Scatter(
                x=[0, -1.05 * x_max],
                y=[y, y],
                mode="lines",
                line=dict(color="lightgrey", width=1.0),
                showlegend=False,
            )
        )

    # Set axes properties
    fig.update_layout(
        xaxis=dict(
            range=[-1.05 * x_max, x_pos["col3_end"]],
            autorange=False,
            tickvals=[-400, -300, -200, -100, 0],
            ticktext=["400", "300", "200", "100", "0"],
        ),
        yaxis=dict(
            range=[total_height, -1 / 10 * total_height],
            domain=[0.0, 1.0],
            # showgrid=True,
            # gridcolor="lightgray",
            # gridwidth=1,
            zeroline=False,
            zerolinecolor="gray",
            zerolinewidth=1,
            showticklabels=False,
            # tickmode="linear",
            # tick0=0,
            # dtick=max(total_height * 0.2, 10),  # Tick every 50 units
            # tickcolor="black",
            # tickwidth=2,
            # ticklen=5,
        ),
        height=600,
        width=600,
        margin=dict(l=0, r=0, t=40, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig


def criticality_plots(dataframe: pd.DataFrame):
    """
    Generates a Plotly figure visualizing criticality parameters (such as sserr_result and coupled_criterion)
    as a function of weak layer depth from a given DataFrame. This includes normalization to known
    critical values and the plotting of these parameters over the snow profile depth.

    Parameters:
        dataframe (pd.DataFrame): DataFrame containing columns 'wl_depth', 'sserr_result', and 'coupled_criterion'.

    Returns:
        fig (go.Figure): A Plotly Figure object displaying the criticality curves against depth.
    """
    fig = go.Figure()

    # Extract cirtical values.
    critical_cc = 150.0
    critical_sserr = 5.5
    depth = max(dataframe["wl_depth"])

    # Extract highest values
    min_cc = min(dataframe["coupled_criterion"])

    # Append 0.0 depth to dataframe
    dataframe = pd.concat(
        [
            dataframe,
            pd.DataFrame(
                {
                    "wl_depth": [0.0],
                    "sserr_result": [0.0],
                    "coupled_criterion": [min_cc],
                }
            ),
        ]
    )
    dataframe = dataframe.sort_values(by="wl_depth")

    # Interpolate 1D densely: x10 resolution
    y_depths = np.linspace(0, depth, 10 * len(dataframe))
    x_sserr = np.interp(y_depths, dataframe["wl_depth"], dataframe["sserr_result"])
    x_cc = np.interp(y_depths, dataframe["wl_depth"], dataframe["coupled_criterion"])

    # Extract region where cc is self-collapsed
    cc_zero_mask = x_cc <= 1e-6

    # Robustify division
    epsilon = 1e-6
    x_cc = np.where(cc_zero_mask, epsilon, x_cc)

    x_sserr = x_sserr / critical_sserr
    x_cc = critical_cc / x_cc

    # Define colors for each axis
    AXIS_COLORS = {
        "sserr": "blue",
        "cc": "orange",
    }

    fig.add_trace(
        go.Scatter(
            x=x_sserr,
            y=y_depths,
            mode="lines",
            name="Energy Release Rate",
            line=dict(color=AXIS_COLORS["sserr"], width=3),
            marker=dict(size=6, color=AXIS_COLORS["sserr"]),
            xaxis="x1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_cc,
            y=y_depths,
            mode="lines",
            name="Critical Coupling",
            line=dict(color=AXIS_COLORS["cc"], width=3),
            marker=dict(size=6, color=AXIS_COLORS["cc"]),
            xaxis="x1",
        )
    )
    # fig.add_vline(x=1.0, line=dict(color="black", width=3))
    fig.add_trace(
        go.Scatter(
            x=[1.0, 1.0],
            y=[0.0, depth],
            mode="lines",
            name="Critical Point",
            line=dict(color="black", width=2),
            showlegend=False,  # optional
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[1.0],
            y=[0.0],
            mode="markers",
            name="Critical Point",
            marker=dict(size=10, color="black"),
            showlegend=False,  # optional
        )
    )

    # Create points for filled region between x_vals and x=1.0
    x_shading = np.concatenate(
        [
            x_sserr,
            np.full_like(x_sserr, 1.0)[::-1],
        ]
    )
    y_shading = np.concatenate([y_depths, y_depths[::-1]])
    above_mask = x_shading >= 1.0

    segments = []
    for is_above, group in groupby(enumerate(above_mask), lambda x: x[1]):
        if is_above:
            indices = [i for i, _ in group]
            segments.append(indices)

    for segment in segments:
        # only keep points where x_shading is >= 1.0
        plot_x = x_shading[segment]
        plot_y = y_shading[segment]

        fig.add_trace(
            go.Scatter(
                x=plot_x,
                y=plot_y,
                fill="toself",
                fillcolor="rgba(0, 0, 255, 0.2)",  # blue-ish transparent
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
                name="Shaded Criticality",
            )
        )

    # Create points for filled region between x_vals and x=1.0
    x_shading = x_cc[~cc_zero_mask]
    y_shading = y_depths[~cc_zero_mask]
    above_mask = x_shading >= 1.0

    segments = []
    for is_above, group in groupby(enumerate(above_mask), lambda x: x[1]):
        if is_above:
            indices = [i for i, _ in group]
            segments.append(indices)

    for segment in segments:
        # only keep points where x_shading is >= 1.0
        plot_x = np.concatenate(
            [
                x_shading[segment],
                np.full_like(x_shading[segment], 1.0)[::-1],
            ]
        )
        plot_y = np.concatenate([y_shading[segment], y_shading[segment][::-1]])

        fig.add_trace(
            go.Scatter(
                x=plot_x,
                y=plot_y,
                fill="toself",
                fillcolor="rgba(255, 165, 0, 0.2)",  # orange-ish transparent
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
                name="Shaded Criticality",
            )
        )

    # Create self-collapsed region
    x_shading = x_cc
    y_shading = y_depths
    segments = []
    for is_above, group in groupby(enumerate(cc_zero_mask), lambda x: x[1]):
        if is_above:
            indices = [i for i, _ in group]
            segments.append(indices)

    for segment in segments:
        # only keep points where x_shading is >= 1.0
        plot_x = np.concatenate(
            [
                x_shading[segment],
                np.full_like(x_shading[segment], 1.0)[::-1],
            ]
        )
        plot_y = np.concatenate([y_shading[segment], y_shading[segment][::-1]])

        fig.add_trace(
            go.Scatter(
                x=plot_x,
                y=plot_y,
                fill="toself",
                fillcolor="rgba(0, 0, 0, 0.1)",  # light-grey
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
                name="Self-Collapsed",
            )
        )

    # Configure multiple overlaying x-axes with enhanced colors and ticks
    fig.update_layout(
        # Main y-axis
        yaxis=dict(
            title="Depth [mm]",  # Remove built-in title, we'll use annotation
            range=[depth, -1 / 10 * depth],
            domain=[0.0, 1.0],
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=1,
            zeroline=True,
            zerolinecolor="gray",
            zerolinewidth=2,
            tickmode="linear",
            tick0=0,
            dtick=100,
            tickcolor="black",
            tickwidth=2,
            ticklen=5,
        ),
        # First x-axis (SSERR) - primary axis
        xaxis=dict(
            title="",  # Remove built-in title, we'll use annotation
            range=[0, 2.0],
            side="bottom",
            # autorange="reversed",
            showgrid=True,
            gridcolor="lightblue",
            gridwidth=1,
            tickmode="linear",
            tick0=0,
            dtick=2.0 * 0.1,  # 5 ticks across the range
            tickcolor="black",
            tickwidth=2,
            ticklen=8,
            tickfont=dict(color="black", size=10),
            linecolor="black",
            linewidth=2,
        ),
        showlegend=False,
        width=400,
        height=600,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=40, b=40),
    )

    # X-axis title annotations positioned above their respective axes
    fig.add_annotation(
        text="Criticality",
        x=0.5,  # Center of the plot
        y=0.0,  # Just above the bottom axis
        xref="paper",
        yref="paper",
        ax=0,
        ay=20,
        font=dict(size=12),
    )

    fig.add_annotation(
        text="Critical Point",
        x=0.5,
        y=1.0,
        xref="paper",
        yref="paper",
        ax=0,  # Shift text 40px right
        ay=-10,
        font=dict(color="black"),
    )
    return fig


def criticality_heatmap(
    dataframe: pd.DataFrame,
    variant: Literal[
        "default",
        "all",
        "maxstress+cc_split_sserr+ssmaxsxx",
        "maxstress+cc_split_sserr+slabtensilecriterion",
        "maxstress+cc_split_sserr+ssmaxsxx+slabtensilecriterion",
    ] = "default",
):
    """
    Generates a criticality heatmap from the provided DataFrame.

    The heatmap visualizes the criticality of weak layers as a function of depth, combining different
    criticality metrics, such as 'sserr_result' and 'coupled_criterion', through statistical transforms
    and combination rules, to highlight zones of concern in the profile.

    Parameters:
        dataframe (pd.DataFrame):
            DataFrame containing at least the columns 'wl_depth', 'sserr_result',
            and 'coupled_criterion'. Each row corresponds to a weak layer.

    Returns:
        fig (go.Figure):
            A Plotly figure object representing the criticality heatmap.
    """

    # Get max depth
    depth = max(dataframe["wl_depth"])

    dataframe = dataframe.sort_values(by="wl_depth")

    # # Extend dataframe with 0-depth row if not already present
    # if not (dataframe["wl_depth"] == 0.0).any():
    #     dataframe = pd.concat(
    #         [
    #             pd.DataFrame(
    #                 {
    #                     "wl_depth": [0.0],
    #                     "sserr_result": [0.0],
    #                     "coupled_criterion": [0.0],
    #                     "max_stress": [0.0],
    #                     "impact_criterion": [0.0],
    #                     "touchdown_distance": [0.0],
    #                     "ss_max_Sxx_norm": [0.0],
    #                     "slab_tensile_criterion": [0.0],
    #                 }
    #             ),
    #             dataframe,
    #         ]
    #     )

    wl_depth = dataframe["wl_depth"]
    cc = dataframe["coupled_criterion"]
    sserr = dataframe["sserr_result"]
    max_stress = dataframe["max_stress"]
    ss_max_Sxx_norm = dataframe["ss_max_Sxx_norm"]
    slab_tensile_criterion = dataframe["slab_tensile_criterion"]

    # Interpolate: y = depth in mm
    depth_grid = np.linspace(0, depth, 10 * len(dataframe))
    sserr_interp = np.interp(depth_grid, wl_depth, sserr)
    cc_interp = np.interp(depth_grid, wl_depth, cc)
    max_stress_interp = np.interp(depth_grid, wl_depth, max_stress)
    ss_max_Sxx_norm_interp = np.interp(depth_grid, wl_depth, ss_max_Sxx_norm)
    slab_tensile_criterion_interp = np.interp(
        depth_grid, wl_depth, slab_tensile_criterion
    )

    # Determine CDF percentiles acc. to avalanche data for each metric
    sserr_cdf = calc_SSERR_CDF_percentile(sserr_interp)
    cc_cdf = calc_CCWeight_CDF_percentile(cc_interp)
    max_stress_cdf = calc_MaxStress_CDF_percentile(max_stress_interp)
    ss_max_Sxx_norm_cdf = calc_SSMaxSxx_CDF_percentile(ss_max_Sxx_norm_interp)
    slab_tensile_criterion_cdf = calc_SSSlabTensileCriterion_CDF_percentile(
        slab_tensile_criterion_interp
    )

    # Calculate combined avalanche criticality
    combined_z = combined_avalanche_criticality(
        CCWeight_CDF_percentile=cc_cdf,
        SSERR_CDF_percentile=sserr_cdf,
        MaxStress_CDF_percentile=max_stress_cdf,
        SSMaxSxx_CDF_percentile=ss_max_Sxx_norm_cdf,
        SSlabTensileCriterion_CDF_percentile=slab_tensile_criterion_cdf,
        variant=variant,
    )

    # Extract region where cc is self-collapsed
    cc_self_collapsed_mask = cc_interp <= 1e-6
    combined_z = np.where(cc_self_collapsed_mask, 0.0, combined_z)

    # Create 2D z-values for heatmap (duplicate along x-axis)
    # Shape: (len(depth_grid), 2)
    cc_z = np.tile(cc_cdf.reshape(-1, 1), (1, 2))
    cc_x = [0.0, 0.5, 1.0]
    sserr_z = np.tile(sserr_cdf.reshape(-1, 1), (1, 2))
    sserr_x = [1.0, 1.5, 2.0]
    max_stress_z = np.tile(max_stress_cdf.reshape(-1, 1), (1, 2))
    max_stress_x = [2.0, 2.5, 3.0]
    ss_max_Sxx_norm_z = np.tile(ss_max_Sxx_norm_cdf.reshape(-1, 1), (1, 2))
    ss_max_Sxx_norm_x = [3.0, 3.5, 4.0]
    slab_tensile_criterion_z = np.tile(
        slab_tensile_criterion_cdf.reshape(-1, 1), (1, 2)
    )
    slab_tensile_criterion_x = [4.0, 4.5, 5.0]
    combined_z = np.tile(combined_z.reshape(-1, 1), (1, 2))
    combined_x = [5.0, 5.5, 6.0]

    value_pairs = [
        (cc_z, cc_x, "cc"),
        (sserr_z, sserr_x, "sserr"),
        (max_stress_z, max_stress_x, "max_stress"),
        (ss_max_Sxx_norm_z, ss_max_Sxx_norm_x, "ss_max_Sxx_norm"),
        (slab_tensile_criterion_z, slab_tensile_criterion_x, "slab_tensile_criterion"),
    ]

    # Create figure
    fig = go.Figure()

    for z, x, name in value_pairs:
        fig.add_trace(
            go.Heatmap(
                z=z,
                x=x,
                y=depth_grid,
                colorscale="Reds",
                showscale=False,
                reversescale=False,
                zmin=0.0,
                zmax=1.0,
                hoverinfo="skip",
                name=name,
            )
        )

    light_fade = [
        [0.00, "rgb(0,180,0)"],  # green
        [0.10, "rgb(80,200,0)"],  # lighter green
        [0.20, "rgb(170,220,0)"],  # yellow-green
        [0.33, "yellow"],  # yellow
        [0.45, "rgb(255,180,0)"],  # yellow-orange
        [0.55, "orange"],  # orange
        [0.70, "orangered"],  # deep orange
        [0.85, "red"],
        [1.00, "darkred"],
    ]

    fig.add_trace(
        go.Heatmap(
            z=combined_z,
            x=combined_x,
            y=depth_grid,
            colorscale=light_fade,
            showscale=True,
            colorbar=dict(title="Cum."),
            zmin=0.0,
            zmax=1.0,
            name="Combined Criticality",
        )
    )

    xs = [5.0, 5.3, 5.6, 5.9]
    for x in xs:
        fig.add_trace(
            go.Scatter(
                x=[x, x],
                y=[0, depth],
                mode="lines",
                line=dict(color="lightgrey", width=0.5),
                showlegend=False,
            )
        )

    # Manual horizontal grid lines (y-direction)
    y_step = 100  # or however you want to space the grid
    y_grid = np.arange(0, depth + y_step, y_step)

    for y in y_grid:
        fig.add_trace(
            go.Scatter(
                x=[0.0, 6.0],
                y=[y, y],
                mode="lines",
                line=dict(color="white", width=0.5),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    xs = combined_z.mean(axis=1) + 5.0
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=depth_grid,
            mode="lines",
            line=dict(color="black", width=2),
            showlegend=False,
        )
    )

    fig.update_layout(
        title=f"Criticality Heatmap ({variant})",
        font=dict(size=8),
        yaxis=dict(
            autorange=False,
            range=[depth, -1 / 10 * depth],
            domain=[0.0, 1.0],
            showticklabels=False,
        ),
        xaxis=dict(
            range=[0.0, 6.0],
            tickvals=[0.5, 1.5, 2.5, 3.5, 4.5, 5.0, 5.3, 5.6, 5.9],
            ticktext=[
                "Fracture",
                "Propagation",
                "MaxStress",
                "SSMaxSxxNorm",
                "SlabTensileCriterion",
                "0.0",
                "0.3",
                "0.6",
                "0.9",
            ],
        ),
        width=300,
        height=600,
        margin=dict(l=0, r=0, t=40, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig


def combine_plots(file_path: str, name: str, figures: List[go.Figure]):
    """
    Combine multiple Plotly figures into a single PNG image.
    """
    images = []
    for fig in figures:
        width = fig.layout.width * 2 if fig.layout.width else 1200
        height = fig.layout.height * 2 if fig.layout.height else 1200
        img_bytes = fig.to_image(format="png", width=width, height=height, scale=2)
        image = Image.open(BytesIO(img_bytes))
        images.append(image)

    total_width = sum(im.width for im in images)
    max_height = max(im.height for im in images)
    combined = Image.new("RGB", (total_width, max_height), color=(255, 255, 255))
    x_offset = 0
    for im in images:
        combined.paste(im, (x_offset, 0))
        x_offset += im.width

    os.makedirs(file_path, exist_ok=True)
    combined.save(os.path.join(file_path, f"{name}.png"))
