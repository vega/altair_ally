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
    heatmap_width = data.shape[0]
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
        alt.Stroke('value', scale=color_scale, sort=[False, True], legend=None)).properties(width=heatmap_width).add_selection(zoom)

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


def dist(data, color_col=None, mark=None, dtype='number', columns=None, rug=True):
    '''
    Densities can include a rug to indicate approximately how many obs there are
    since dns with samll obs not good
    '''
    # TODO column_wrap instead of columns?
    # TODO add clickable legend

    bins = True
    # bins = alt.Bin(maxbins=30)
    if columns == None:
        # Ceil sqrt
        columns = int(-(-data.select_dtypes(dtype).columns.size ** (1/2) // 1))

    if mark is None:
        if dtype == 'number':  # support floats etc
            mark = 'area'
        elif dtype in ['category', 'object', 'bool']:
            mark = 'bar'

    if not mark in ['area', 'line', 'bar', 'point']:
        print('not supported')
    # if mark == 'area':
        # mark = alt.MarkDef(mark, opacity=0.7)

    # Setting a non-existing column with specified type passes through without effect
    # and eliminates the need to hvae a separate plotting section for colored bars below.
    if color_col is None:
        color_col = ':Q'

    if dtype in ['category', 'object', 'bool']:
        # Counts of categorical distributions
        if mark != 'bar':
            print("Only bar mark supported")
        else:
            return (alt.Chart(data).mark_bar().encode(
                alt.X('count()', title='', stack=None),
                alt.Y(alt.repeat(), type='nominal', sort='x', title=None),
                alt.Color(color_col))
            .properties(width=120)
            .repeat(data.select_dtypes(dtype).columns.tolist()[::-1], columns=columns))
  
    else:  # TODO don't support dates...
        # Histograms
        if mark == 'bar':
            return (alt.Chart(data).mark_bar().encode(
                alt.X(alt.repeat(), type='quantitative', bin=bins),
                alt.Y('count()', title='', stack=None),
                alt.Color(color_col))
            .properties(width=185, height=120)
            .repeat(data.select_dtypes(dtype).columns.tolist()[::-1], columns=columns))

        # Density plots
        elif mark in ['area', 'line']: 
            subplot_row = []
            for col in data.select_dtypes(dtype).columns.tolist()[::-1]:
                subplot = (
                    alt.Chart(data, mark=mark).transform_density(
                    col, [col, 'density'], groupby=[color_col], minsteps=100)
                    .encode(
                    alt.X(col, axis=alt.Axis(grid=False)),
                    alt.Y('density:Q', title=None),
                    alt.Color(color_col))
        #              alt.Y('density:Q', title=None, axis=alt.Axis(labels=False, ticks=False, grid=False)))
        #              alt.Y('density:Q', title=None, axis=None)) #alt.Axis(labels=False, ticks=False, )))
                .properties(width=185, height=120))
                if rug:
                    rugplot = alt.Chart(data).mark_tick(color='black', opacity=0.3, yOffset=60 - 3, height=7).encode(
                        alt.X(col))
                    subplot = subplot + rugplot

                subplot_row.append(subplot)
            return alt.concat(*subplot_row, columns=columns)#.configure_view(strokeWidth=0)

def corr(data, corr_types=['pearson', 'spearman'], mark='circle', select_on='mouseover'):
    '''
    Correlation of numerical columns.
    '''
    # TODO support NA, maybe via aly.corr(movies.isna())
    hover = alt.selection_multi(fields=['variable', 'index'], on=select_on, nearest=True, empty='all')

    subplot_row = []
    for num, corr_type in enumerate(corr_types):
        if num > 0:
            yaxis = alt.Axis(labels=False)
        else:
            yaxis = alt.Axis()
        corr_df = data.select_dtypes('number').corr(corr_type)
        mask = np.zeros_like(corr_df, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        corr_df[mask] = np.nan

        corr2 = corr_df.reset_index().melt(id_vars='index').dropna().sort_values('variable', ascending=False)
        var_sort = corr2['variable'].value_counts().index.tolist()
        ind_sort = corr2['index'].value_counts().index.tolist()

        subplot_row.append(
            alt.Chart(corr2, mark=mark, title=f'{corr_type.capitalize()} correlations')
            .transform_calculate(
            abs_value='abs(datum.value)')
         .encode(
            alt.X('index', sort=ind_sort, title=''),
            alt.Y('variable', sort=var_sort[::-1], title='', axis=yaxis),
#     alt.Color('value', title=[corr_type.capitalize(), 'coefficient'], scale=alt.Scale(domain=[-1, 1], scheme='blueorange')),
            alt.Color('value', title='', scale=alt.Scale(domain=[-1, 1], scheme='blueorange')),
            alt.Size('abs_value:Q', scale=alt.Scale(domain=[0, 1]), legend=None),
            alt.Tooltip('value', format='.2f'),
            opacity = alt.condition(hover, alt.value(0.9), alt.value(0.2))).add_selection(hover))#.properties(width=300, height=300)
            
    return alt.concat(*subplot_row).resolve_axis(y='shared').configure_view(strokeWidth=0)


def parcoord(data, color_col=None, rescale=None):
    # Setting a non-existing column with specified type passes through without effect
    # and eliminates the need to hvae a separate plotting section for colored bars below.
    if color_col is None:
        color_col = ':Q'
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

    legend_click = alt.selection_multi(fields=[color_col], bind='legend')

    return alt.Chart(data[num_cols + [color_col]]).transform_window(
        index='count()'
    ).transform_fold(
        num_cols
    ).mark_line().encode(
        alt.X('key:O', title=None, scale=alt.Scale(nice=False, padding=0.05)),
        alt.Y('value:Q', title=None),
        alt.Color(color_col, title=None),
        detail='index:N',
        opacity=alt.condition(legend_click, alt.value(0.6), alt.value(0.05))
    ).properties(width=len(num_cols) * 100).add_selection(legend_click)


def heatmap(data, color_col=None, sort=None, rescale='min-max'):
    """
    Create a (normalized and sorted) heatmap of all columns
    """
    num_cols = data.select_dtypes('number').columns.to_list()
    heatmap_width = data.shape[0]
    # TODO rename color_col to color or color_by everywhere
    # TODO move this to a utils module since it is used in two places now
    if rescale == 'mean-sd':
        data[num_cols] = data[num_cols].apply(lambda x: (x - x.mean()) / x.std()) 
    elif rescale == 'min-max':
        data[num_cols] = data[num_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    elif callable(rescale):
        data[num_cols] = data[num_cols].apply(rescale)
    elif rescale is not None:
        print('not supported')
        
    # data = data.sort_values('MPAA Rating')
    if sort is not None:  # TODO autosort on color? then there is no way to not sort unless chnge paarm to 'auto'
        data = data.sort_values(sort)

    num_heatmap = alt.Chart(data[num_cols]).transform_window(
            index='count()'
        ).transform_fold(
            num_cols
        ).mark_rect(height=16).encode(
            alt.Y('key:N', title=None),
            alt.X('index:O', title=None, axis=None),
            alt.Color('value:Q', title=None, legend=alt.Legend(orient='right', type='gradient')),
            alt.Stroke('value:Q')
    ).properties(width=heatmap_width)

    if color_col is None:
        return num_heatmap
    else:
        cat_heatmap = alt.Chart(data[[color_col]]).transform_window(
            index='count()'
        ).transform_fold(
            [color_col]
        ).mark_rect(height=16).encode(
            alt.Y('key:N', title=None),
            alt.X('index:O', title=None, axis=None),
            alt.Color('value:N', title=None, legend=alt.Legend(orient='bottom')),
            alt.Stroke('value:N')
        ).properties(width=heatmap_width)
        return num_heatmap & cat_heatmap