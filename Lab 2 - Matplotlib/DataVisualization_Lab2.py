#imports
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

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
    plot_fn = getattr(plt, method_name, None)
    if not callable(plot_fn):
        raise ValueError(f"matplotlib has no plotting function plt.{method_name}()")

    for feature1 in x_list:  # x-axis
        for feature2 in y_list:  # y-axis
            # do not plot the same features against themselves
            if feature1 == feature2:
                continue

            sorted_data = data.sort_values(by=feature1)

            # one figure per plot so each shows separately
            plt.figure()
            # for bar: x is categories / numeric, y is heights ("exhaustive" even if odd)
            plot_fn(sorted_data[feature1], sorted_data[feature2])

            plt.xlabel(feature1)
            plt.ylabel(feature2)
            plt.title(f"{method_name}: {feature1} vs {feature2}")
            plt.tight_layout()
            plt.show()


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

def hist_exhaustive(
    y_list,
    x_list,
    *,
    bins=0,
    density=False,
    overlay_mode="overlay",     # "overlay" | "stacked" for numeric-vs-numeric
    catcat_mode="grouped",      # "grouped" | "stacked" for categorical-vs-categorical
    max_legend_items=20
):
    """
    Draw classic bar-like histograms/exhaustive bar charts for every (feature1, feature2) pair.

    Behavior by dtype:
      • numeric vs numeric  -> two 1D histograms in the same axes (overlay or stacked), using shared edges
      • categorical vs categorical -> grouped (or stacked) bar chart from a crosstab of counts (all categories)
      • numeric vs categorical -> overlaid 1D histograms (one per category), shared edges

    Assumes `data` is a cleaned pandas DataFrame (no NaNs/bad samples).

    :param y_list: list[str]; features to iterate as the y-axis label (name only; plotting is 1D bars)
    :param x_list: list[str]; features to iterate as the x-axis label (name only; plotting is 1D bars)
    :param bins: "auto" | int | array-like; bin rule or explicit edges for numeric histograms
    :param density: bool; plot probability density instead of counts for numeric histograms
    :param overlay_mode: str; "overlay" or "stacked" for numeric vs numeric
    :param catcat_mode: str; "grouped" or "stacked" for categorical vs categorical
    :param max_legend_items: int; legend suppressed if more than this many entries
    :return: None
    """

    for feature1 in y_list:
        for feature2 in x_list:
            if feature1 == feature2:
                continue

            series1 = data[feature1]
            series2 = data[feature2]
            feature1_is_numeric = is_num_type(series1)
            feature2_is_numeric = is_num_type(series2)

            # ---------- numeric vs numeric: two classic 1D histograms ----------
            if feature1_is_numeric and feature2_is_numeric:
                values1 = series1.to_numpy(dtype=float)
                values2 = series2.to_numpy(dtype=float)

                # shared edges from both columns for fair comparison
                combined = np.concatenate([values1, values2])
                edges = np.histogram_bin_edges(combined, bins=bins)

                plt.figure()
                if overlay_mode == "stacked":
                    plt.hist(
                        [values1, values2],
                             bins=edges,
                             density=density,
                             stacked=True,
                             label=[feature1, feature2],
                             edgecolor="black",
                             linewidth=0.4
                             )
                else:
                    # overlay with transparency
                    plt.hist(values1, bins=edges, density=density, alpha=0.6, label=feature1, edgecolor="black", linewidth=0.4)
                    plt.hist(values2, bins=edges, density=density, alpha=0.6, label=feature2, edgecolor="black", linewidth=0.4)

                plt.xlabel("value")
                plt.ylabel("density" if density else "count")
                plt.title(f"Histogram: {feature1} vs {feature2} (1D, shared bins)")
                plt.legend()
                plt.tight_layout()
                plt.show()
                plt.close()

            # ---------- categorical vs categorical: grouped/stacked bars ----------
            elif (not feature1_is_numeric) and (not feature2_is_numeric):
                crosstab_counts = pd.crosstab(series1, series2)  # all categories

                plt.figure()
                ax = None
                if catcat_mode == "stacked":
                    ax = crosstab_counts.plot(kind="bar", stacked=True, edgecolor="black", linewidth=0.4)
                else:
                    ax = crosstab_counts.plot(kind="bar", stacked=False, edgecolor="black", linewidth=0.4)

                plt.xlabel(feature1)
                plt.ylabel("count")
                plt.title(f"Counts: {feature1} by {feature2}")
                plt.xticks(rotation=90)
                if crosstab_counts.shape[1] <= max_legend_items:
                    plt.legend(title=feature2, fontsize="x-small", ncol=2)
                else:
                    plt.legend([], [], frameon=False)  # suppress if too many
                plt.tight_layout()
                plt.show()
                plt.close()

            # ---------- numeric vs categorical: overlaid 1D histograms ----------
            else:
                categorical_feature, numeric_feature = (feature1, feature2) if not feature1_is_numeric else (feature2, feature1)
                series_cat = data[categorical_feature]
                series_num = data[numeric_feature].astype(float)

                # shared edges per numeric feature
                edges = np.histogram_bin_edges(series_num.to_numpy(), bins=bins)

                # stable category order: frequency, then name
                category_order = series_cat.value_counts().sort_index().sort_values(ascending=False).index

                plt.figure()
                for category_value in category_order:
                    values = series_num[series_cat == category_value].to_numpy()
                    if values.size == 0:
                        continue
                    plt.hist(values, bins=edges, density=density, alpha=0.35, label=str(category_value), edgecolor="black", linewidth=0.4)

                plt.xlabel(numeric_feature)
                plt.ylabel("density" if density else "count")
                plt.title(f"{numeric_feature} distribution by {categorical_feature} (all categories)")
                if len(category_order) <= max_legend_items:
                    plt.legend(title=categorical_feature, fontsize="x-small", ncol=2)
                plt.tight_layout()
                plt.show()
                plt.close()

# quantitative vs quantitative (2D hist)
# hist_exhaustive(quantitative_features, quantitative_features, bins=30)

# categorical vs categorical (heatmap)
# hist_exhaustive(categorical_features, categorical_features)

# numeric vs categorical (overlaid 1D hists)
# hist_exhaustive(quantitative_features, categorical_features, bins="auto")
# hist_exhaustive(categorical_features, quantitative_features, bins="auto")
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

# plot for mae


def pair_hist2d_numeric(dataframe, feature_x, feature_y, *, bins=20, density=False):
    """
    Plot a true 2D histogram where both axes are numeric features and the color encodes count or density.

    :param dataframe: pandas.DataFrame; cleaned dataset (no NaNs/bad samples).
    :param feature_x: str; name of the numeric feature on the x-axis.
    :param feature_y: str; name of the numeric feature on the y-axis.
    :param bins: int | (int, int) | "auto"; number of bins (or per-axis tuple) or rule.
    :param density: bool; normalize to show density instead of counts.
    :return: None
    """
    values_x = dataframe[feature_x].to_numpy(dtype=float)
    values_y = dataframe[feature_y].to_numpy(dtype=float)

    plt.figure()
    # 2D histogram: features on both axes; color = count (or density)
    plt.hist2d(values_x, values_y, bins=bins, density=density)
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.title(f"2D histogram: {feature_x} vs {feature_y}")
    plt.colorbar(label="density" if density else "count")
    plt.tight_layout()
    plt.show()


def binned_aggregate_bars(dataframe, feature_x, feature_y, *, bins=20, statistic="mean"):
    """
    Put feature_x on the x-axis via binning, and plot an aggregate of feature_y on the y-axis as bars.
    This is not a histogram in the strict sense (y != count), but it answers "what if the y-axis is a feature?"

    Rules:
      - If feature_x is numeric: bin it, then aggregate feature_y per bin -> bar heights.
      - If feature_x is categorical: categories on x, then aggregate feature_y per category -> bar heights.

    :param dataframe: pandas.DataFrame; cleaned dataset (no NaNs/bad samples).
    :param feature_x: str; feature to place on the x-axis (numeric or categorical).
    :param feature_y: str; numeric feature to summarize on the y-axis.
    :param bins: int | "auto"; number of bins (ignored if feature_x is categorical).
    :param statistic: str; one of {"mean","median","sum"} to summarize feature_y within each bin/category.
    :return: None
    """
    if statistic not in {"mean", "median", "sum"}:
        raise ValueError("statistic must be one of {'mean','median','sum'}")

    series_x = dataframe[feature_x]
    series_y = dataframe[feature_y].astype(float)

    if is_num_type(series_x):
        # numeric x -> bin, then aggregate y within bins
        bin_edges = np.histogram_bin_edges(series_x.to_numpy(dtype=float), bins=bins)
        bin_indices = np.digitize(series_x.to_numpy(dtype=float), bin_edges, right=False)

        # make readable bin labels from centers
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

        plt.figure()
        plt.bar(bar_x, bar_heights, width=np.diff(bin_edges), align="center", edgecolor="black", linewidth=0.4)
        plt.xlabel(f"{feature_x} (binned)")
        plt.ylabel(f"{statistic}({feature_y})")
        plt.title(f"Binned {feature_x} vs aggregated {feature_y} ({statistic})")
        plt.tight_layout()
        plt.show()

    else:
        # categorical x -> category bars of aggregated y
        if statistic == "mean":
            aggregated = series_y.groupby(series_x).mean()
        elif statistic == "median":
            aggregated = series_y.groupby(series_x).median()
        else:
            aggregated = series_y.groupby(series_x).sum()

        # stable order: by category name
        aggregated = aggregated.sort_index()

        plt.figure()
        plt.bar(aggregated.index.astype(str), aggregated.values, edgecolor="black", linewidth=0.4)
        plt.xlabel(feature_x)
        plt.ylabel(f"{statistic}({feature_y})")
        plt.title(f"{feature_y} by {feature_x} ({statistic})")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()


def hist_axes_experiment(dataframe, x_features, y_features, *, bins=20, statistic="mean", mode="binned"):
    """
    Loop over feature pairs and visualize with either a true 2D histogram (numeric-numeric)
    or the binned-aggregation bars (putting feature_y on the y-axis).

    :param dataframe: pandas.DataFrame; cleaned dataset.
    :param x_features: list[str]; features to place on the x-axis.
    :param y_features: list[str]; features to summarize/compare on the y-axis.
    :param bins: int | "auto"; passed to the underlying plotters.
    :param statistic: str; {'mean','median','sum'} for the binned aggregation.
    :param mode: str; 'binned' (default) uses binned_aggregate_bars; 'hist2d' uses true 2D histogram for numeric pairs.
    :return: None
    """
    for feature_x in x_features:
        for feature_y in y_features:
            if feature_x == feature_y:
                continue

            x_is_num = is_num_type(dataframe[feature_x])
            y_is_num = is_num_type(dataframe[feature_y])

            if mode == "hist2d" and x_is_num and y_is_num:
                pair_hist2d_numeric(dataframe, feature_x, feature_y, bins=bins, density=False)
            else:
                # binned aggregation requires y to be numeric (to summarize)
                if not y_is_num:
                    # skip politely if y is categorical (cannot compute mean/median/sum meaningfully as numeric)
                    # You could swap roles here if you want: aggregate x by y.
                    continue
                binned_aggregate_bars(dataframe, feature_x, feature_y, bins=bins, statistic=statistic)


# e.g., x = any features (numeric or categorical), y must be numeric
hist_axes_experiment(data, x_features=categorical_features,
                     y_features=quantitative_features, bins=20, statistic="mean", mode="binned")

