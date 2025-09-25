#imports
import glob
import os

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import image as mpimg

# ingest data
data = pd.read_csv("AutomobilePrice_Lab2.csv")


# preprocess data
# remove symboling column (irrelevant data)
# data.drop("symboling", axis=1, inplace=True)

# replace '?' with 'nan'
data.replace(to_replace='?', value=np.nan, inplace=True)
# remove any rows that contain 'nan'
data.dropna(axis=0, how='any', inplace=True)

#display resulting data
print('\n\n processed data.info():')
print(data.info())

# === Define features & labels ===

# Quantitative features (excluding price, which is a label)
quantitative_features = [
    'symboling',
    'wheel-base',
    'length',
    'width',
    'height',
    'curb-weight',
    'engine-size',
    'bore',
    'stroke',
    'compression-ratio',
    'horsepower',
    'peak-rpm',
    'city-mpg',
    'highway-mpg',
    'price',
]

# Categorical features
categorical_features = [
    'make',
    'fuel-type',
    'aspiration',
    'num-of-doors',
    'body-style',
    'drive-wheels',
    'engine-location',
    'engine-type',
    'num-of-cylinders',
    'fuel-system',
]

# Quantitative labels (price)
quantitative_labels = ['price']

categorical_labels = ['make', 'body-style']

# === Create arrays ===

# Features
X_quant = data[quantitative_features].to_numpy(dtype=float)
X_cat = data[categorical_features].to_numpy(dtype=object)
X_all = data[quantitative_features + categorical_features].to_numpy()

# Labels
y_quant = data[quantitative_labels].to_numpy(dtype=float)
y_cat = data[categorical_labels].to_numpy(dtype=object)
y_all = data[quantitative_labels + categorical_labels].to_numpy()



# subplot grid helper
def make_grid_manager(n_rows = 4, n_cols = 4, fig_size = (16, 12)):
    """
    Returns (next_axis, flush) closures for cycling through a nrows * ncols grid.
    Call next_axis() to get an Axes; when the grid fills, it auto-shows and
    starts a new figure. Call flush() once at the end to show the last, partial grid.

    :param n_rows:
    :param n_cols:
    :param fig_size:
    :return:
    """
    fig, axes = plt.subplots(n_rows, n_cols, figsize = fig_size)
    axes = axes.ravel()
    index = 0

    def next_axis():
        nonlocal fig, axes, index
        if index >= len(axes):
            plt.tight_layout()
            plt.show()
            fig, axes = plt.subplots(n_rows, n_cols, figsize = fig_size)
            axes = axes.ravel()
            index = 0
        axis = axes[index]
        index += 1
        return axis

    def flush():
        # hide any unused axes in the last (partial) grid
        for axis in axes[index:]:
            axis.axis("off")
        plt.tight_layout()
        plt.show()

    return next_axis, flush


# plot feature vs feature
def plot_exhaustive(x_list, y_list, kind="plot"):
    """
    Exhaustively plots feature2 (y) vs feature1 (x) with a chosen matplotlib method.
    kind can be 'plot', 'bar', 'scatter', etc. (anything that's a valid plt.<kind>)
    Assumes `data` is a pandas DataFrame already cleaned (no NaNs/bad samples).

    :param x_list: Feature list for x-axis
    :param y_list: Feature list for y-axis
    :param kind: Type of matplotlib plot to generate (plot, scatter, bar, etc.)
    """

    # workable aliases
    aliases = {
        "line": "plot",
        "bar": "bar",
        "scatter": "scatter",
    }
    method_name = aliases.get(kind, kind)

    next_axis, flush = make_grid_manager(n_rows=4, n_cols=4, fig_size=(16, 12))

    plot_fn = getattr(plt, method_name, None)
    if not callable(plot_fn):
        raise ValueError(f"matplotlib has no plotting function plt.{method_name}()")

    for feature1 in x_list:          # x-axis
        for feature2 in y_list:      # y-axis

            # do not plot same features against themselves
            if feature1 == feature2:
                continue

            sorted_data = data.sort_values(by=feature1)

            axis = next_axis()  # <<< get the next subplot
            # use the Axes' method instead of global plt.<method>
            plot_fn = getattr(axis, method_name, None)
            if not callable(plot_fn):
                raise ValueError(f"matplotlib has no Axes method ax.{method_name}()")

            plot_fn(sorted_data[feature1], sorted_data[feature2])
            axis.set_xlabel(feature1)
            axis.set_ylabel(feature2)
            axis.set_title(f"{method_name}: {feature1} vs {feature2}")

    flush()  # <<< show the last (possibly partial) 4×4



# ======== EXHAUSTIVE LINE GRAPH GENERATIONS ========
# line graphs | quantitative vs quantitative
# plot_exhaustive(quantitative_features, quantitative_features, kind="line")

# line graphs | categorical vs categorical
# plot_exhaustive(categorical_features, categorical_features, kind="line")

# line graphs | categorical vs quantitative
# plot_exhaustive(categorical_features, quantitative_features, kind="line")

# line graphs | quantitative vs categorical
# plot_exhaustive(quantitative_features, categorical_features, kind="line")
# ======== EXHAUSTIVE LINE GRAPH GENERATIONS ========




# ========= EXHAUSTIVE BAR GRAPH GENERATIONS =========
# bar graphs | quantitative vs quantitative
# plot_exhaustive(quantitative_features, quantitative_features, kind="bar")

# bar graphs | categorical vs categorical
# plot_exhaustive(categorical_features, categorical_features, kind="bar")

# bar graphs | categorical vs quantitative
# plot_exhaustive(categorical_features, quantitative_features, kind="bar")

# bar graphs | quantitative vs categorical
# plot_exhaustive(quantitative_features, categorical_features, kind="bar")
# ========= EXHAUSTIVE BAR GRAPH GENERATIONS =========




# ========= EXHAUSTIVE HISTOGRAM GENERATIONS =========
from pandas.core.dtypes.common import is_numeric_dtype as is_num_type

def binned_aggregate_bars(dataframe, feature_x, feature_y, *, bins=20, statistic="mean", axis=None):
    """
    Put feature_x on the x-axis via binning, and plot an aggregate of feature_y on the y-axis as bars.

    :param dataframe: pandas.DataFrame; cleaned dataset (no NaNs/bad samples).
    :param feature_x: str; feature to place on the x-axis (numeric or categorical).
    :param feature_y: str; numeric feature to summarize on the y-axis.
    :param bins: int | "auto"; number of bins (ignored if feature_x is categorical).
    :param statistic: str; one of {"mean","median","sum"} to summarize feature_y within each bin/category.
    :param axis: matplotlib.axes.Axes; subplot to draw on
    :return: None
    """
    if statistic not in {"mean", "median", "sum"}:
        raise ValueError("statistic must be one of {'mean','median','sum'}")
    if axis is None:
        axis = plt.gca()

    series_x = dataframe[feature_x]
    series_y = dataframe[feature_y].astype(float)

    if is_num_type(series_x):
        # numeric x -> bin, then aggregate y within bins
        values_x = series_x.to_numpy(dtype=float)
        bin_edges = np.histogram_bin_edges(values_x, bins=bins)
        bin_indices = np.digitize(values_x, bin_edges, right=False)

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        results = []
        for index in range(1, len(bin_edges)):  # bins are 1..len(edges)-1
            in_bin = (bin_indices == index)
            if not np.any(in_bin):
                results.append(np.nan)
                continue
            if statistic == "mean":
                results.append(series_y[in_bin].mean())
            elif statistic == "median":
                results.append(series_y[in_bin].median())
            else:
                results.append(series_y[in_bin].sum())

        bar_x = bin_centers
        bar_heights = np.array(results, dtype=float)

        axis.bar(bar_x, bar_heights, width=np.diff(bin_edges),
                 align="center", edgecolor="black", linewidth=0.4)
        axis.set_xlabel(f"{feature_x} (binned)")
        axis.set_ylabel(f"{statistic}({feature_y})")
        axis.set_title(f"Binned {feature_x} vs aggregated {feature_y} ({statistic})")

    else:
        # categorical x -> category bars of aggregated y
        if statistic == "mean":
            aggregated = series_y.groupby(series_x).mean()
        elif statistic == "median":
            aggregated = series_y.groupby(series_x).median()
        else:
            aggregated = series_y.groupby(series_x).sum()

        aggregated = aggregated.sort_index()

        axis.bar(aggregated.index.astype(str), aggregated.values,
                 edgecolor="black", linewidth=0.4)
        axis.set_xlabel(feature_x)
        axis.set_ylabel(f"{statistic}({feature_y})")
        axis.set_title(f"{feature_y} by {feature_x} ({statistic})")
        axis.tick_params(axis="x", labelrotation=90)


def hist_axes_experiment(dataframe, x_features, y_features, *, bins=20, statistic="mean", mode="binned"):
    """
    Loop over feature pairs and visualize with binned-aggregation bars in paged 4x4 subplots.

    :param dataframe: pandas.DataFrame; cleaned dataset.
    :param x_features: list[str]; features to place on the x-axis.
    :param y_features: list[str]; features to summarize/compare on the y-axis.
    :param bins: int | "auto"; passed to the underlying plotter.
    :param statistic: str; {'mean','median','sum'} for the binned aggregation.
    :param mode: str; kept for signature compatibility (unused here).
    :return: None
    """
    next_axis, flush = make_grid_manager(n_rows=4, n_cols=4, fig_size=(16, 12))

    for feature_x in x_features:
        for feature_y in y_features:
            if feature_x == feature_y:
                continue

            y_is_num = is_num_type(dataframe[feature_y])
            if not y_is_num:
                # cannot aggregate a categorical y into numeric bars
                continue

            axis = next_axis()
            binned_aggregate_bars(
                dataframe,
                feature_x,
                feature_y,
                bins=bins,
                statistic=statistic,
                axis=axis
            )

    flush()


# histograms | quantitative vs quantitative
# hist_axes_experiment(data, x_features=quantitative_features,
#                      y_features=quantitative_features, bins=20, statistic="mean", mode="binned")

# histograms | categorical vs categorical
# hist_axes_experiment(data, x_features=categorical_features,
#                      y_features=categorical_features, bins=20, statistic="mean", mode="binned")

# histograms | quantitative vs categorical
# hist_axes_experiment(data, x_features=quantitative_features,
#                      y_features=categorical_features, bins=20, statistic="mean", mode="binned")

# histograms | categorical vs quantitative
# hist_axes_experiment(data, x_features=categorical_features,
#                      y_features=quantitative_features, bins=20, statistic="mean", mode="binned")
# ========= EXHAUSTIVE HISTOGRAM GENERATIONS =========



# ========= EXHAUSTIVE SCATTER PLOT GENERATIONS =========
# scatter plot | quantitative vs quantitative
# plot_exhaustive(quantitative_features, quantitative_features, kind="scatter")

# scatter plot | categorical vs categorical
# plot_exhaustive(categorical_features, categorical_features, kind="scatter")

# scatter plot | categorical vs quantitative
# plot_exhaustive(categorical_features, quantitative_features, kind="scatter")

# scatter plot | quantitative vs categorical
# plot_exhaustive(quantitative_features, categorical_features, kind="scatter")
# ========= EXHAUSTIVE SCATTER PLOT GENERATIONS =========



# PRINTS GROUPS OF PROVIDED VISUALIZATIONS GIVEN IN SUBMITTED .ZIP FOLDER
def chunk(sequence, size):
    for start in range(0, len(sequence), size):
        yield sequence[start:start+size]

def grid_from_folder(folder_path, *, rows=4, cols=4, pattern="*.png", sort=True, title=True):
    """
    Render all images in folder as paged grids of rows×cols subplots.
    Starts a new figure every rows*cols images. Hides unused axes on last page.
    """
    # resolve folder relative to this file
    folder_path = os.path.join(os.path.dirname(__file__), folder_path)
    files = glob.glob(os.path.join(folder_path, pattern))

    if sort:
        # stable human-ish sort
        files.sort(key=lambda p: os.path.basename(p).lower())

    per_page = rows * cols
    if not files:
        print(f"[grid_from_folder] No images found in: {folder_path}")
        return

    for page in chunk(files, per_page):
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        axes = axes.ravel() if per_page > 1 else [axes]

        for ax, path in zip(axes, page):
            try:
                img = mpimg.imread(path)
            except FileNotFoundError:
                ax.text(0.5, 0.5, "Missing file", ha="center", va="center")
                ax.axis("off")
                continue

            ax.imshow(img)
            ax.axis("off")
            if title:
                ax.set_title(os.path.splitext(os.path.basename(path))[0], fontsize=9)

        # hide any unused axes on the last page
        for ax in axes[len(page):]:
            ax.axis("off")

        plt.tight_layout()
        plt.show()

# Selected Histograms provided in .zip folder
grid_from_folder("Histograms", rows=4, cols=4, pattern="*.png")

# Selected Scatter Plots Provided in .zip folder
grid_from_folder("Scatter Plots", rows=4, cols=4, pattern="*.png")