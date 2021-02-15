from itertools import combinations

import altair as alt
import numpy as np


# note about melting and altair data server
# enf: include some themes? or straight toaltari?
# enh: add chart width config for each
def nan(data):
    '''
    Only shows columns with NaNs
    '''
    cols_with_nans = data.columns[data.isna().any()]
    # Long form data
    data = (
        data[cols_with_nans]
        .isna()
        .reset_index()
        .melt(id_vars='index'))

    # Sorted counts of NaNs per column
    nan_counts = (
        data.query('value == True')
        .groupby('variable')
        .size()
        .reset_index()
        .rename(columns={0: 'count'})
        .sort_values('count'))
    sorted_nan_cols = nan_counts['variable'].to_list()
    max_nans = nan_counts.max()['count']

    # Bar chart of NaN counts per column
    zoom = alt.selection_interval(encodings=['x'])
    nan_bars = (
        alt.Chart(data.query('value == True'), title='NaN count').mark_bar(color='steelblue', height=17).encode(
        alt.X('count()', axis=None, scale=alt.Scale(domain=[0, max_nans])),
        alt.Y('variable', axis=alt.Axis(grid=False, title='', labels=False, ticks=False), sort=sorted_nan_cols))
        .properties(width=100))
    nan_bars_with_text = ((
        nan_bars
        + nan_bars.mark_text(align='left', dx=2)
        .encode(text='count()'))
        .transform_filter(zoom))

    # Heatmap of individual NaNs
    color_scale = alt.Scale(range=['#dde8f1', 'steelblue'][::1])

    nan_heatmap = alt.Chart(data, title='Individual NaNs').mark_rect(height=17).encode(
        alt.X('index:O', axis=None),
        alt.Y('variable', title=None, sort=sorted_nan_cols),
        alt.Color('value', scale=color_scale, sort=[False, True], legend=alt.Legend(orient='top', offset=-13), title=None),
        alt.Stroke('value', scale=color_scale, sort=[False, True], legend=None)).properties(width=data.shape[0] / 8).add_selection(zoom)

    # Bind bar chart update to zoom in individual chart and add hover to individual chart,
    # configurable column for tooltip, or index
    return (nan_heatmap | nan_bars_with_text).configure_view(strokeWidth=0).resolve_scale(y='shared')


def pair(data, color_col=None, tooltip=None, mark='point', width=150, height=150):
    # support categorical?
    col_dtype='number'
    # enh: zooming wihout panning
    # color_col = 'species:N'  # must be passed with a type, enh: autoetect 
    # tooltip = alt.Tooltip('species')
    cols = data.select_dtypes(col_dtype).columns

    # Setting a non-existing column with specified type passes through without effect
    # and eliminates the need to hvae a separate plotting section for colored bars below.
    if color_col is None:
        color_col = ':Q'
    if tooltip is None:
        tooltip = ':Q'

    # Infer color data type if not specified 
    if color_col[-2:] in [':Q', ':T', ':N', ':O']:
        color_alt = alt.Color(color_col, title=None, legend=alt.Legend(orient='left', offset=width * -1.6))
        # The selection fields parmeter does not work with the suffix
        legend_color = color_col.split(':')[0]
    else:
        color_alt = alt.Color(color_col, title=None, type=alt.utils.infer_vegalite_type(data[color_col]))
        legend_color = color_col

    # Set up interactions
    brush = alt.selection_interval()
    color = alt.condition(brush, color_alt, alt.value('lightgrey'))
    legend_click = alt.selection_multi(fields=[legend_color], bind='legend')
    opacity = alt.condition(legend_click, alt.value(0.8), alt.value(0.2))
    hidden_axis = alt.Axis(domain=False, title='', labels=False, ticks=False)

    # Create corner of pair-wise scatters
    i = 0
    # enh: Have options for different corner alignment of the charts.
    # histograms would look better on top of top corner than under botom corner
    col_combos = list(combinations(cols, 2))[::-1]
    subplot_row = []
    while i < len(cols) - 1:
        plot_column = []
        for num, (y, x) in enumerate(col_combos[:i+1]):
            if num == 0 and i == len(cols) - 2:
                subplot = alt.Chart(data, mark=mark).encode(x=x, y=y)
            elif num == 0:
                subplot = (
                    alt.Chart(data, mark=mark).encode(
                        alt.X(x, axis=hidden_axis), alt.Y(y)))
            elif i == len(cols) - 2:
                subplot = (
                    alt.Chart(data, mark=mark).encode(
                        alt.X(x), alt.Y(y, axis=hidden_axis)))
            else:
                subplot = (
                    alt.Chart(data, mark=mark).encode(
                        alt.X(x, axis=hidden_axis), alt.Y(y, axis=hidden_axis)))
            plot_column.append(
                subplot
                .encode(opacity=opacity, color=color, tooltip=tooltip)
                .properties(width=width, height=height))
        subplot_row.append(alt.hconcat(*plot_column))
        i += 1
        col_combos = col_combos[i:]

    return (
        alt.vconcat(*subplot_row)
        .add_selection(brush, legend_click))