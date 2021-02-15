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


