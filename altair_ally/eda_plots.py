from itertools import combinations, cycle

import altair as alt
import numpy as np


# TODO show examples of how to set chart width etc, might need to add a param
def corr(data, corr_types=['pearson', 'spearman'], mark='circle', select_on='mouseover'):
    """
    Plot the pairwise correlations between columns.

    Parameters
    ----------
    data : DataFrame
        pandas DataFrame with input data.
    corr_types: list of (str or function)
        Which correlations to calculate.
        Anything that is accepted by DataFrame.corr.
    mark: str
        Shape of the points. Passed to Chart.
        One of "circle", "square", "tick", or "point".
    select_on : str
        When to highlight points across plots.
        A string representing a vega event stream,
        e.g. 'click' or 'mouseover'.

    Returns
    -------
    ConcatChart
        Concatenated Chart of the correlation plots laid out in a single row.
    """
    # TODO support correlation of NA values, maybe via aly.corr(movies.isna())
    hover = alt.selection_multi(fields=['variable', 'index'], on=select_on, nearest=True, empty='all')

    subplot_row = []
    for num, corr_type in enumerate(corr_types):
        if num > 0:
            yaxis = alt.Axis(labels=False)
        else:
            yaxis = alt.Axis()
        corr_df = data.select_dtypes(['number', 'boolean']).corr(corr_type)
        mask = np.zeros_like(corr_df, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        corr_df[mask] = np.nan

        corr2 = corr_df.reset_index().melt(id_vars='index').dropna().sort_values('variable', ascending=False)
        var_sort = corr2['variable'].value_counts().index.tolist()
        ind_sort = corr2['index'].value_counts().index.tolist()
        # sorted_cols = (corr_df.abs().sum(axis=1) + corr_df.abs().sum(axis=0)).sort_values().index.to_list()

        subplot_row.append(
            alt.Chart(corr2, mark=mark, title=f'{corr_type.capitalize()} correlations')
            .transform_calculate(
                abs_value='abs(datum.value)')
            .encode(
               alt.X('index', sort=ind_sort, title=''),
               alt.Y('variable', sort=var_sort[::-1], title='', axis=yaxis),
               alt.Color('value', title='', scale=alt.Scale(domain=[-1, 1], scheme='blueorange')),
               alt.Size('abs_value:Q', scale=alt.Scale(domain=[0, 1]), legend=None),
               [alt.Tooltip('value', format='.2f').title('corr'), alt.Tooltip('index').title('x'), alt.Tooltip('variable').title('y')]
            )
        )

    return alt.concat(*subplot_row).resolve_axis(y='shared').configure_view(strokeWidth=0)


def dist(data, color=None, mark=None, dtype='number', columns=None, rug=True):
    """
    Plot the distribution of each dataframe column.

    Can visualize univariate distributions
    of either numerical or categorical variables
    depending on which **dtype** is used.
    Numercial distributions can be plotted as either density plots or histograms,
    depending on which **mark** is used.
    Since density plots can be misleadingly smooth with small datasets,
    a rug plot is included by default to indicate the number of observations in the data.

    Parameters
    ----------
    data : DataFrame
        pandas DataFrame with input data.
    color : str
        Column in **data** used for the color encoding.
    mark : str
        Wether to plot a density plot ('area'),
        or a histogram / barplot of counts ('bar').
        The default is to use an area for numerical variables,
        and a barplot for categorical variables.
    dtype : str or type
        Which column types to plot, passed to DataFrame.select_dtypes.
        If 'object', 'category', or 'bool',
        a barplot of counts for each categorical value will be plotted.
    columns : int
        The number of columns in the plot grid.
        The default is to create an as square grid as possible.
    rug : bool
        Wether to include a rug plot or not.

    Returns
    -------
    ConcatChart
        Concatenated Chart of the distribution plots laid out in a square grid.
    """
    # TODO add clickable legend
    bins = True
    # bins = alt.Bin(maxbins=30)
    # Layout out in a single row for up to 3 columns, after that switch to a squareish grid
    selected_data = data.select_dtypes(dtype)
    if columns is None:
        if selected_data.columns.size <= 3:
            columns = selected_data.columns.size
        else:
            # Ceil sqrt
            columns = int(-(-selected_data.columns.size ** (1/2) // 1))

    if mark is None:
        if dtype == 'number':  # TODO support floats etc
            mark = 'area'
        elif dtype in ['category', 'object', 'bool']:
            mark = 'bar'

    if mark not in ['area', 'line', 'bar', 'point']:
        print('not supported')
    # if mark == 'area':
        # mark = alt.MarkDef(mark, opacity=0.7)

    opacity = 0.7
    # Setting a non-existing column with specified type passes through without effect
    # and eliminates the need to hvae a separate plotting section for colored bars below.
    if color is None:
        color = ':Q'
        opacity = 0.9

    if dtype in ['category', 'object', 'bool']:
        # Counts of categorical distributions
        # TODO add count label on y-axis or write in docstring what it is use configure to add it
        if mark != 'bar':
            print("Only bar mark supported")
        else:
            charts = []
            for col in selected_data.nunique().sort_values().index:
                charts.append(
                    alt.vconcat(
                        alt.Chart(data.sample(data.shape[0]), width=120).mark_bar().encode(
                            x=alt.X('count()'),
                            y=alt.Y(color, title=None, axis=alt.Axis(domain=True, title='', labels=False, ticks=False)),
                            color=alt.Color(color, title=None),
                            row=alt.Row(col, title=None,
                                        header=alt.Header(labelAngle=0, labelAlign='left', labelPadding=5))),
                        title=alt.TitleParams(col, anchor='middle')))

            return (
                alt.concat(*charts, columns=columns)
                .configure_facet(spacing=0)
                .configure_view(stroke=None)
                .configure_scale(bandPaddingInner=0.06, bandPaddingOuter=0.4))

    else:  # TODO don't support dates...
        # Histograms
        # TODO add count label on y-axis or write in docstring what it is use configure to add it
        if mark == 'bar':
            return (
                alt.Chart(data, mark=alt.MarkDef(mark, opacity=opacity)).encode(
                    alt.X(alt.repeat(), type='quantitative', bin=bins),
                    alt.Y('count()', title='', stack=None),
                    alt.Color(color))
                .properties(width=185, height=120)
                .repeat(selected_data.columns.tolist()[::-1], columns=columns))

        # Density plots
        # TODO add density label on y-axis? Meaningless...
        elif mark in ['area', 'line']:
            subplot_row = []
            for col in selected_data.columns.tolist()[::-1]:
                subplot = (
                    alt.Chart(data, mark=alt.MarkDef(mark, opacity=0.1)).transform_density(
                        col, [col, 'density'], groupby=[color], minsteps=100)
                    .encode(
                        alt.X(col, axis=alt.Axis(grid=False)),
                        alt.Y('density:Q', title=None).stack(False),
                        alt.Color(color, title=None))
                    .properties(width=185, height=120)
                )
                subplot += subplot.mark_line()
                if rug:
                    rugplot = alt.Chart(data).mark_tick(color='black', opacity=0.3, yOffset=68 - 3, height=5).encode(
                        alt.X(col).axis(offset=8),
                        tooltip=alt.value('Individual observations')
                    )
                    subplot = subplot + rugplot

                subplot_row.append(subplot)
            return alt.concat(*subplot_row, columns=columns)


def heatmap(data, color=None, sort=None, rescale='min-max',
            cat_schemes=['tableau10', 'set2', 'accent'],
            num_scheme='yellowgreenblue'):
    """
    Plot the values of all columns and observations as a heatmap.

    This reshapes the dataframe to a longer format,
    so you might need to disable the max rows warning in Altair
    or use the 'data-server' backend:
    `aly.alt.data_transformers.disable_max_rows()` or
    `aly.alt.data_transformers('data_server')`

    Parameters
    ----------
    data : DataFrame
        pandas DataFrame with input data.
    color: str
        Which column(s) in **data** to use for the color encoding.
        Helpful to investigate if a categorical column is correlated
        with the value arrangement in the numerical columns.
    sort: str or list of str
        Which column(s) in **data** to use for sorting the observations.
        This can be helpful to see patterns in which columns look similar when sorted.
    rescale : str or fun
        How to rescale the values before plotting them.
        Ensures that one column does not dominate the plot.
        One of 'min-max', 'mean-sd', None, or a custom function.
        'min-max` rescales the data to lie in the range 0-1.
        'mean-sd' rescales the data to have mean 0 and sd 1.
        None uses the raw values.
    cat_schemes : list of str
        Color schemes to use for each of the categorical heatmaps.
        Cycles through when shorter than **color**,
        so set to a list with a single item
        if you want to use the same color scheme for all categorical heatmaps.
    num_scheme : str
        Color scheme to use for the numerical heatmap.

    Returns
    -------
    Chart or ConcatChart
        Single Chart with observed values if no color encoding is used,
        else concatenated Chart including categorical colors.
    """
    data = data.copy()
    num_cols = data.select_dtypes('number').columns.to_list()
    heatmap_width = data.shape[0]
    # TODO move this to a utils module since it is used in two places now
    if rescale == 'mean-sd':
        data[num_cols] = data[num_cols].apply(lambda x: (x - x.mean()) / x.std())
    elif rescale == 'min-max':
        data[num_cols] = data[num_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    elif callable(rescale):
        data[num_cols] = data[num_cols].apply(rescale)
    elif rescale is not None:
        print('not supported')

    # TODO autosort on color? then there is no way to not sort unless the
    # default is changed to 'auto', but there could be name collisions with the columns
    if sort is not None:
        data = data.sort_values(sort)

    scale = alt.Scale(scheme=num_scheme)
    num_heatmap = alt.Chart(data[num_cols]).transform_window(
            index='count()'
        ).transform_fold(
            num_cols
        ).mark_rect(height=16).encode(
            alt.Y('key:N', title=None),
            alt.X('index:O', title=None, axis=None),
            alt.Color('value:Q', scale=scale, title=None, legend=alt.Legend(orient='right', type='gradient')),
            alt.Stroke('value:Q', scale=scale),
            alt.Tooltip('value:Q')
    ).properties(width=heatmap_width)

    if color is None:
        return num_heatmap
    else:
        colors = color
        if isinstance(color, str):
            colors = [color]
        cat_heatmaps = []
        for color, scheme in zip(colors, cycle(cat_schemes)):
            color = [color]
            cat_heatmaps.append(alt.Chart(data[color]).transform_window(
                index='count()'
            ).transform_fold(
                color
            ).mark_rect(height=16).encode(
                alt.Y('key:N', title=None),
                alt.X('index:O', title=None, axis=None),
                alt.Color('value:N', title=None, scale=alt.Scale(scheme=scheme),
                          legend=alt.Legend(orient='bottom', offset=5)),
                alt.Stroke('value:N', scale=alt.Scale(scheme=scheme)),
                alt.Tooltip('value:N')
            ).properties(width=heatmap_width))
        return alt.vconcat(num_heatmap, *cat_heatmaps)


def nan(data):
    """
    Plot indiviudal missing values and overall counts for each column.

    There is a default interaction defined where selections in the heatmap
    will update the counts in the barplot.

    Parameters
    ----------
    data: DataFrame
        Pandas input dataframe.

    Returns
    -------
    ConcatChart
        Concatenated Altair chart with individual NaNs and overall counts.
    """
    cols_with_nans = data.columns[data.isna().any()]
    heatmap_width = data.shape[0]
    # TODO can transform_fold be used here too?
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

    nan_heatmap = (
        alt.Chart(data, title='Individual NaNs').mark_rect(height=17).encode(
            alt.X('index:O', axis=None),
            alt.Y('variable', title=None, sort=sorted_nan_cols),
            alt.Color('value', scale=color_scale, sort=[False, True],
                      legend=alt.Legend(orient='top', offset=-13), title=None),
            alt.Stroke('value', scale=color_scale, sort=[False, True], legend=None))
        .properties(width=heatmap_width).add_selection(zoom))

    # Bind bar chart update to zoom in individual chart and add hover to individual chart,
    # configurable column for tooltip, or index
    return (nan_heatmap | nan_bars_with_text).configure_view(strokeWidth=0).resolve_scale(y='shared')


def pair(data, color=None, tooltip=None, mark='point', width=150, height=150):
    """
    Create pairwise scatter plots of all column combinations.

    In contrast to many other pairplot tools,
    this function creates a single scatter plot per column pair,
    and no distribution plots along the diagonal.

    Parameters
    ----------
    data : DataFrame
        pandas DataFrame with input data.
    color : str
        Column in **data** used for the color encoding.
    tooltip: str
        Column in **data** used for the tooltip encoding.
    mark: str
        Shape of the points. Passed to Chart.
        One of "circle", "square", "tick", or "point".
    width: int or float
        Chart width.
    height: int or float
        Chart height.

    Returns
    -------
    ConcatChart
        Concatenated Chart of pairwise column scatter plots.
    """
    # TODO support categorical?
    col_dtype = 'number'
    # color = 'species:N'  # must be passed with a type, enh: autoetect
    # tooltip = alt.Tooltip('species')
    cols = data.select_dtypes(col_dtype).columns

    # Setting a non-existing column with specified type passes through without effect
    # and eliminates the need to hvae a separate plotting section for colored bars below.
    if color is None:
        color = ':Q'
    if tooltip is None:
        tooltip = ':Q'

    # Infer color data type if not specified
    if color[-2:] in [':Q', ':T', ':N', ':O']:
        color_alt = alt.Color(color).legend(orient='top')
        # The selection fields parmeter does not work with the suffix
        legend_color = color.split(':')[0]
    else:
        color_alt = alt.Color(color, type=alt.utils.infer_vegalite_type_for_pandas(data[color])).legend(orient='top')
        legend_color = color

    hidden_axis = alt.Axis(domain=False, title='', labels=False, ticks=False)

    if mark == 'rect':
        bins = {'maxbins': 30}
        color = 'count()'
        opacity = alt.value(1)
        tooltip = 'count()'
    else:
        bins = False
        # Set up interactions
        brush = alt.selection_interval()
        color = alt.condition(brush, color_alt, alt.value('lightgrey'))
        legend_click = alt.selection_multi(fields=[legend_color], bind='legend')
        opacity = alt.condition(legend_click, alt.value(0.6), alt.value(0.1))
        # plot_data = data.sample(400)
    # Create corner of pair-wise scatters
    i = 0
    exclude_zero = alt.Scale(zero=False)
    col_combos = list(combinations(cols, 2))[::-1]
    subplot_row = []
    while i < len(cols) - 1:
        plot_column = []
        for num, (y, x) in enumerate(col_combos[:i+1]):
            if num == 0 and i == len(cols) - 2:
                subplot = alt.Chart(data, mark=mark).encode(
                    alt.X(x, scale=exclude_zero).bin(bins),
                    alt.Y(y, scale=exclude_zero).bin(bins))
            elif num == 0:
                subplot = (
                    alt.Chart(data, mark=mark).encode(
                        alt.X(x, scale=exclude_zero, axis=hidden_axis).bin(bins),
                        alt.Y(y, scale=exclude_zero).bin(bins)))
            elif i == len(cols) - 2:
                subplot = (
                    alt.Chart(data, mark=mark).encode(
                        alt.X(x, scale=exclude_zero).bin(bins),
                        alt.Y(y, scale=exclude_zero, axis=hidden_axis).bin(bins)))
            else:
                subplot = (
                    alt.Chart(data, mark=mark).encode(
                        alt.X(x, scale=exclude_zero, axis=hidden_axis).bin(bins),
                        alt.Y(y, scale=exclude_zero, axis=hidden_axis).bin(bins)))
            plot_column.append(
                subplot
                .encode(opacity=opacity, color=color, tooltip=tooltip)
                .properties(width=width, height=height))
        subplot_row.append(alt.hconcat(*plot_column))
        i += 1
        col_combos = col_combos[i:]

    if mark == 'rect':
        return alt.vconcat(*subplot_row)
    else:
        return (
            alt.vconcat(*subplot_row)
            .add_selection(brush, legend_click)
        )


def parcoord(data, color=None, rescale='min-max'):
    """
    Plot the values of all columns and observations as a parallel coordinates plot.

    Parameters
    ----------
    data : DataFrame
        pandas DataFrame with input data.
    color: str
        Which column in **data** to use for the color encoding.
    rescale : str or fun
        How to rescale the values before plotting them.
        Ensures that one column does not dominate the plot.
        One of 'min-max', 'mean-sd', or a custom function.
        'min-max` rescales the data to lie in the range 0-1.
        'mean-sd' rescales the data to have mean 0 and sd 1.

    Returns
    -------
    Chart
        Chart with one x-value per column and one line per row in **data**.
    """
    data = data.copy()
    # Setting a non-existing column with specified type passes through without effect
    # and eliminates the need to hvae a separate plotting section for colored bars below.
    if color is None:
        color = ':Q'
    # rescale = None # give example of lamba fun and reccoment skleanr
    num_cols = data.select_dtypes('number').columns.to_list()

    if rescale == 'mean-sd':
        data[num_cols] = data[num_cols].apply(lambda x: (x - x.mean()) / x.std())
    elif rescale == 'min-max':
        data[num_cols] = data[num_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    elif callable(rescale):
        data[num_cols] = data[num_cols].apply(rescale)
    elif rescale is not None:
        print('not supported')

    legend_click = alt.selection_multi(fields=[color], bind='legend')

    return alt.Chart(data[num_cols + [color]]).transform_window(
        index='count()'
    ).transform_fold(
        num_cols
    ).mark_line().encode(
        alt.X('key:O', title=None, scale=alt.Scale(nice=False, padding=0.05)),
        alt.Y('value:Q', title=None),
        alt.Color(color, title=None),
        detail='index:N',
        opacity=alt.condition(legend_click, alt.value(0.6), alt.value(0.05))
    ).properties(width=len(num_cols) * 100).add_selection(legend_click)
