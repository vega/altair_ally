from itertools import combinations, cycle

import altair as alt
import numpy as np
import pandas as pd


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
               alt.Color('value', title='', scale=alt.Scale(domain=[-1, 1], scheme='blueorange')),
               alt.Size('abs_value:Q', scale=alt.Scale(domain=[0, 1]), legend=None),
               alt.Tooltip('value', format='.2f'),
               opacity=alt.condition(hover, alt.value(0.9), alt.value(0.2))).add_params(hover))

    return alt.concat(*subplot_row).resolve_axis(y='shared').configure_view(strokeWidth=0)

def get_label_angle(
    labels,
    offset_groups,
    step_size=20,
    padding_between_offset=None,
    padding_between_x=None,
):
    # Defaults from https://vega.github.io/vega-lite/docs/scale.html#band
    if padding_between_offset is None:
        if offset_groups > 1:
            padding_between_offset = 0.1
        else:
            padding_between_offset = 0
    if padding_between_x is None:
        # padding_between_x = 0.2
        if offset_groups > 1:
            # Supposed to be 0.2 in the docs, but due to this bug https://github.com/vega/vega-lite/issues/8930 I am multiplying by the number of offset groups
            padding_between_x = 0.3 * offset_groups
        else:
            padding_between_x = 0.1

    # This dictionary was constructed based on a common font via the following snippet:
    # from string import ascii_letters
    # from PIL import ImageFont
    # font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    # letter_widths = {letter: font.getsize(letter * 10)[0] / 10 for letter in ascii_letters}
    letter_widths = {
        'a': 6.1,
        'b': 6.3,
        'c': 5.5,
        'd': 6.3,
        'e': 6.2,
        'f': 3.6,
        'g': 6.3,
        'h': 6.3,
        'i': 2.8,
        'j': 2.9,
        'k': 5.8,
        'l': 2.8,
        'm': 9.7,
        'n': 6.3,
        'o': 6.1,
        'p': 6.3,
        'q': 6.3,
        'r': 4.0,
        's': 5.2,
        't': 3.9,
        'u': 6.3,
        'v': 5.9,
        'w': 8.2,
        'x': 5.9,
        'y': 5.9,
        'z': 5.3,
        'A': 7.1,
        'B': 6.9,
        'C': 7.0,
        'D': 7.7,
        'E': 6.3,
        'F': 5.8,
        'G': 7.8,
        'H': 7.5,
        'I': 3.0,
        'J': 3.1,
        'K': 6.6,
        'L': 5.6,
        'M': 8.6,
        'N': 7.5,
        'O': 7.9,
        'P': 6.0,
        'Q': 7.9,
        'R': 7.0,
        'S': 6.3,
        'T': 6.1,
        'U': 7.3,
        'V': 6.9,
        'W': 9.9,
        'X': 6.9,
        'Y': 6.3,
        'Z': 6.9,
        '!': 4.0,
        '"': 4.6,
        '#': 8.4,
        '$': 6.4,
        '%': 9.5,
        '&': 7.8,
        "'": 2.8,
        '(': 3.9,
        ')': 3.9,
        '*': 5.0,
        '+': 8.4,
        ',': 3.2,
        '-': 3.6,
        '.': 3.2,
        '/': 3.4,
        ':': 3.4,
        ';': 3.4,
        '<': 8.4,
        '=': 8.4,
        '>': 8.4,
        '?': 5.3,
        '@': 10.0,
        '[': 3.9,
        '\\': 3.4,
        ']': 3.9,
        '^': 8.4,
        '_': 5.2,
        '`': 5.0,
        '{': 6.4,
        '|': 3.4,
        '}': 6.4,
        '~': 8.4,
        '0': 6.4,
        '1': 6.4,
        '2': 6.4,
        '3': 6.4,
        '4': 6.4,
        '5': 6.4,
        '6': 6.4,
        '7': 6.4,
        '8': 6.4,
        '9': 6.4,
        ' ': 3.2
    }
    mean_width = sum(letter_widths.values()) / len(letter_widths.values())
    label_widths = []
    for label in labels:
        label_widths.append(
            sum([
                letter_widths[letter]
                if letter in letter_widths
                # Default to mean width for unknowns
                else mean_width
                for letter in str(label)
            ])
        )
    # Rotate labels if they collide
    # print(max(label_widths))
    # print(offset_groups * step_size)
    # print(padding_between_offset * step_size * (offset_groups - 1))
    # print(padding_between_x * step_size)
    # print()
    # Compare the longest label width with the available space for each label
    if max(label_widths) > (
            offset_groups * step_size
            + padding_between_offset * step_size * (offset_groups - 1)
            + padding_between_x * step_size
    ):
        return -45
    else:
        return 0


# TODO It would be neat if any transform could be specified via .transform_* methods and then applied to all charts in the loop? That would just be a lot of things to keep track of, wouldn't it? But maybe they would all just work and aly is more about setting good default for multi charts in altair. Maybe this is more annoying for something like bin thought?
from typing import Union
def dist(
    data: pd.DataFrame,
    color: Union[str, alt.Color] = None, # Shortcut for color encoding
    dtype: str = 'numerical', # preface with ! for exluding  # TODO only allow cat and num?
    # Transform
    density: bool = None,
    bin: Union[bool, alt.Bin] = False,
    cumulative: bool = False,
    # Mark
    mark: Union[str, alt.MarkDef] = None,
    # Encodings
    encoding: Union[dict, alt.Encoding] = None,
    # TODO Should there also be a shortcut for `stack`?
    columns: int = None,
) -> alt.ConcatChart:
    """
    Plot the distribution of each dataframe column.

    Visualize univariate distributions
    of either numerical or categorical variables.
    Numercial distributions can be plotted as density plots, histograms, ECDFs, and rug plots.
    The default is to plot numerical distributions as density plots
    since these are easy to compare across multiple subgroups (colors).
    Since density plots can be misleadingly smooth with small datasets,
    a rug plot is included by default to indicate the number of observations in the data.
    Any encoding and mark option supported by Altair
    can be specified via their respective parameter,
    and the options are added to the default values.

    Parameters
    ----------
    data : DataFrame
        pandas DataFrame with input data.
    color : str, dict, or alt.Color
        Column in `data` used for the color encoding.
    dtype : str
        Which column types to plot, either '`numerical`' or `'categorical'`
    density : bool
        Whether to plot a kernel density estimate for the distribution.
    bin : bool
        Whether to plot a histogram for the distribution.
    cumulative :
        Whether to plot the cumulative version of the chart.
        When `density` and `bin` are both set to `False`,
        this creates an empiracal cumulative density function chart.
    mark : str or alt.MarkDef
        Mark options to be used in the chart.
    encoding : dict or alt.Encoding
        Encoding options to be added in addition to the defaults,
        e.g. to sort in another order.
    columns : int
        The number of columns in the plot grid.
        The default is to try to create a square grid.

    Returns
    -------
    ConcatChart
        Concatenated Chart containing one chart per data frame column
        laid out in a squarish grid.
    """
    # TODO is ValueError the correct one to raise here?
    if dtype == 'numerical':
        selected_data = data.select_dtypes(include='number').copy()
    elif dtype == 'categorical':
        selected_data = data.select_dtypes(exclude='number').copy()
        if bin:
            raise ValueError('Cannot bin categorical variables.')
        if density:
            raise ValueError('Cannot compute a density estimate for categorical variables.')
    else:
        raise ValueError("`dtype` needs to be either `'categorical'` or `'numerical'`'.")

    # TODO this could possible be refined when there are many categorical columns to create a certical column instead?
    if columns is None:
        if selected_data.columns.size <= 3:
            columns = selected_data.columns.size
        else:
            # Ceil sqrt to make grid square
            columns = int(-(-selected_data.columns.size ** (1/2) // 1))

    if color is None:
        xOffset = alt.XOffset()
        color = alt.Color()
    else:
        if isinstance(color, str):
            color = alt.utils.parse_shorthand(color)
        elif isinstance(color, alt.Color):
            print('here')
            color = color.to_dict(context=dict(data=data))
        else:
            raise ValueError('`color` needs to be a string or a `alt.Color` instance.')
        # This should happen after the columns grid computation above
        if dtype == 'numerical':
            selected_data[color['field']] = data[color['field']]
        # Make colors categorical unless otherwise specified
        if 'type' not in color:
            color['type'] = 'nominal'
        # TODO what if someone wants another order?
        # They could still override this via the encoding but it is maybe tedious?
        color_order = selected_data.value_counts(color['field']).index.tolist()
        xOffset = alt.XOffset(**color).scale(paddingInner=0.1).sort(color_order)
        color = alt.Color(**color).title('').sort(color_order)

    if density and bin:
        raise ValueError('Cannot compute a density estimate for binned variables.')
    if density:
        rug = False
    if density is None and not bin and dtype == 'numerical':
        density = True
        rug = True

    if encoding is None:
        encoding = {}
    elif isinstance(encoding, alt.Encoding):
        encoding = encoding.to_dict()

    if mark is None:
        mark = {}
    else:
        if isinstance(mark, str):
            mark = alt.MarkDef(mark).to_dict()
        elif isinstance(color, alt.MarkDef):
            mark = mark.to_dict()
        else:
            raise ValueError('`mark` needs to be a string or `alt.MarkDef` instance.')

    # Counts of categorical distributions
    if dtype == 'categorical':
        if 'type' not in mark:
            mark = dict(type='bar')
        charts = []
        # Sort based on the number of unique values to put smaller charts on top
        # so that they don't get lost
        chart_order = selected_data.nunique().sort_values().index
        for col in chart_order:
            # TODO alt.Encoding requires specification of types, is there a way around that?
            default_encoding = dict(
                y=alt.Y('count()').title('Count'),
                x=alt.X(col + ':N').sort('-y').axis(
                    labelAngle=get_label_angle(
                        selected_data[col].unique(),
                        selected_data[xOffset['field']].nunique(dropna=False)
                        if 'field' in xOffset.to_dict() else 1
                    )
                ),
                xOffset=xOffset,
                color=color
            )
            chart_encoding = encoding | {
                key: default_encoding[key]
                for key in default_encoding
                if key not in encoding
            }
            # If x channel options are set manually, but no field is given
            if 'x' in encoding and 'field' not in encoding['x']:
                chart_encoding['x']['field'] = col
            charts.append(
                alt.Chart(
                    selected_data,
                    height=120,
                    mark=mark,
                    encoding=chart_encoding,
                )
            )

        # return charts
        return (
            alt.concat(*charts, columns=columns)
            .configure_view(stroke=None)
            .configure_scale(bandPaddingInner=0.2, bandPaddingOuter=0.4)
        )

    # Histograms
    elif dtype == 'numerical':  # TODO don't support dates...
        if bin:
            if bin == True:
                bin = alt.Bin(maxbins=20)
            elif isinstance(bin, int):
                bin= alt.Bin(maxbins=bin)

            default_mark = dict(
                type = 'bar',
            )
            if color != alt.Color():
                default_mark['opacity'] = 0.7
            # Need to sort due to the VL bug of bars being in different z-order
            if 'field' in color.to_dict():
                selected_data=selected_data.sort_values(color['field'])

            mark.update({
                key: default_mark[key]
                for key in default_mark
                if key not in mark
            })

            charts = []
            for col in selected_data.select_dtypes('number'):
                default_encoding = dict(
                    x=alt.X(col).bin(bin, cumulative=cumulative).title(col),
                    y=alt.Y('count()').stack(False).title('Count'),
                    color=color
                )
                chart_encoding = encoding | {
                    key: default_encoding[key]
                    for key in default_encoding
                    if key not in encoding
                }
                # If x channel options are set manually, but no field is given
                if 'x' in encoding and 'field' not in encoding['x']:
                    chart_encoding['x']['field'] = col
                charts.append(
                    alt.Chart(
                        selected_data,
                        mark=mark,
                        encoding=chart_encoding
                    ).properties(
                        height=120,
                        width=200,
                    )
                )
            return alt.concat(*charts, columns=columns).configure_view(stroke=None)

        # Density plots
        elif density:
            charts = []
            for col in selected_data.select_dtypes('number'):
                default_encoding = dict(
                    x=alt.X('value:Q').title(col).axis(grid=False),
                    y=alt.Y('density:Q').title('Density'),
                    stroke=color.legend(None),
                    color=color,
                )
                chart_encoding = encoding | {
                    key: default_encoding[key]
                    for key in default_encoding
                    if key not in encoding
                }

                density_transform = alt.DensityTransform(
                    density=col,
                    groupby=[] if color == alt.Color() else [color['field']],
                    minsteps=100,
                    cumulative=cumulative,
                )
                # The default mark props don't really make sense for other mark types
                # so they are not propagated if a custom mark option is set
                # TODO will it be confusing that this is different than other marks? Should it work like this everywhere?
                if 'type' not in mark:
                    mark = dict(
                        type='line' if cumulative else 'area',
                        fill=None,
                        strokeWidth=2,
                        opacity=0.9,
                        stroke='#4c78a8',
                    )
                chart = (
                    alt.Chart(
                        data,
                        mark=mark,
                        encoding=chart_encoding,
                    ).transform_density(
                        **density_transform.to_dict()
                    ).properties(
                        width=200,
                        height=200 if cumulative else 120
                    )
                )
                # TODO return charts and move rug to the end
                if rug:
                    rugplot = alt.Chart(data).mark_tick(
                        opacity=0.4,
                        yOffset=-4,
                        height=7
                    ).encode(
                        alt.X(col),
                        y=alt.datum(0),
                        color=color
                    )
                    chart = chart + rugplot

                charts.append(chart)
            return alt.concat(*charts, columns=columns).configure_view(stroke=None)

        # If only cumulative is specified, draw an ecdf
        elif cumulative:
            default_mark = dict(
                type = 'line',
                interpolate="step-after",
                opacity=0.8,
            )

            mark.update({
                key: default_mark[key]
                for key in default_mark
                if key not in mark
            })

            charts = []
            for col in selected_data.select_dtypes('number'):
                default_encoding = dict(
                    x=alt.X(f"{col}:Q"),
                    y=alt.Y("ecdf:Q"),
                    color=color,
                )
                chart_encoding = encoding | {
                    key: default_encoding[key]
                    for key in default_encoding
                    if key not in encoding
                }
                # If x channel options are set manually, but no field is given
                if 'x' in encoding and 'field' not in encoding['x']:
                    chart_encoding['x']['field'] = col
                charts.append(
                    alt.Chart(
                        selected_data,
                        mark=mark,
                        encoding=chart_encoding
                    ).transform_window(
                        ecdf="cume_dist()",
                        sort=[{"field": col}],
                        groupby=[] if color == alt.Color() else [color['field']],
                    ).properties(
                        height=180,
                        width=180,
                    )
                )
            return alt.concat(*charts, columns=columns).configure_view(stroke=None)

        else:
            default_mark = dict(
                type = 'tick',
                opacity=0.4,
                # yOffset=-4,
                # height=7,
            )

            mark.update({
                key: default_mark[key]
                for key in default_mark
                if key not in mark
            })

            charts = []
            for col in selected_data.select_dtypes('number'):
                default_encoding = dict(
                    x=alt.X(f"{col}:Q"),
                    color=color,
                )
                chart_encoding = encoding | {
                    key: default_encoding[key]
                    for key in default_encoding
                    if key not in encoding
                }
                # If x channel options are set manually, but no field is given
                if 'x' in encoding and 'field' not in encoding['x']:
                    chart_encoding['x']['field'] = col
                charts.append(
                    alt.Chart(
                        data,
                        encoding=chart_encoding,
                        mark=mark,
                    ).properties(
                        width=180,
                    )
                )
            return alt.concat(*charts, columns=columns).configure_view(stroke=None)


    return selected_data


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
        .properties(width=heatmap_width).add_params(zoom))

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
        color_alt = alt.Color(color, title=None, legend=alt.Legend(orient='left', offset=width * -1.6))
        # The selection fields parmeter does not work with the suffix
        legend_color = color.split(':')[0]
    else:
        color_alt = alt.Color(color, title=None, type=alt.utils.infer_vegalite_type(data[color]))
        legend_color = color

    # Set up interactions
    brush = alt.selection_interval()
    color = alt.condition(brush, color_alt, alt.value('lightgrey'))
    legend_click = alt.selection_multi(fields=[legend_color], bind='legend')
    opacity = alt.condition(legend_click, alt.value(0.8), alt.value(0.2))
    hidden_axis = alt.Axis(domain=False, title='', labels=False, ticks=False)

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
                    alt.X(x, scale=exclude_zero),
                    alt.Y(y, scale=exclude_zero))
            elif num == 0:
                subplot = (
                    alt.Chart(data, mark=mark).encode(
                        alt.X(x, scale=exclude_zero, axis=hidden_axis),
                        alt.Y(y, scale=exclude_zero)))
            elif i == len(cols) - 2:
                subplot = (
                    alt.Chart(data, mark=mark).encode(
                        alt.X(x, scale=exclude_zero),
                        alt.Y(y, scale=exclude_zero, axis=hidden_axis)))
            else:
                subplot = (
                    alt.Chart(data, mark=mark).encode(
                        alt.X(x, scale=exclude_zero, axis=hidden_axis),
                        alt.Y(y, scale=exclude_zero, axis=hidden_axis)))
            plot_column.append(
                subplot
                .encode(opacity=opacity, color=color, tooltip=tooltip)
                .properties(width=width, height=height))
        subplot_row.append(alt.hconcat(*plot_column))
        i += 1
        col_combos = col_combos[i:]

    return (
        alt.vconcat(*subplot_row)
        .add_params(brush, legend_click))


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
    ).properties(width=len(num_cols) * 100).add_params(legend_click)

