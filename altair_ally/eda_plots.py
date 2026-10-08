from copy import deepcopy
from itertools import combinations, count, cycle
from math import ceil, sqrt
from typing import Union

import altair as alt
import numpy as np
import pandas as pd

# Separate helper calls remain independent when combined into a larger chart.
_chart_ids = count()


# TODO show examples of how to set chart width etc, might need to add a param
def corr(data, corr_types=('pearson', 'spearman'), mark='circle', select_on='mouseover'):
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
    numeric_data = data.select_dtypes(include=['number', 'bool'])
    if numeric_data.shape[1] < 2:
        raise ValueError('Correlation plots require at least two numerical or boolean columns.')
    if not corr_types:
        raise ValueError('Specify at least one correlation method.')
    prefix = f'corr_{next(_chart_ids)}'
    hover = alt.selection_point(
        name=f'{prefix}_hover', fields=['variable', 'index'],
        on=select_on, nearest=True, empty=True,
    )

    subplot_row = []
    for num, corr_type in enumerate(corr_types):
        if num > 0:
            yaxis = alt.Axis(labels=False)
        else:
            yaxis = alt.Axis()
        corr_df = numeric_data.corr(corr_type)
        mask = np.zeros_like(corr_df, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        corr_df[mask] = np.nan

        corr2 = (
            corr_df.rename_axis('index').reset_index()
            .melt(id_vars='index', var_name='variable')
            .dropna().sort_values('variable', ascending=False)
        )
        var_sort = corr2['variable'].value_counts().index.tolist()
        ind_sort = corr2['index'].value_counts().index.tolist()

        subplot_row.append(
            alt.Chart(
                corr2, mark=mark,
                name=f'{prefix}_view_{num}',
                title=f'{getattr(corr_type, "__name__", str(corr_type)).capitalize()} correlations',
            )
            .transform_calculate(
                abs_value='abs(datum.value)')
            .encode(
               alt.X('index:N').sort(ind_sort).title(''),
               alt.Y('variable:N').sort(var_sort[::-1]).title('').axis(yaxis),
               alt.Color('value:Q').title('').scale(domain=[-1, 1], scheme='blueorange'),
               alt.Size('abs_value:Q').scale(domain=[0, 1]).legend(None),
               tooltip=[
                   alt.Tooltip('value:Q').format('.2f').title('corr'),
                   alt.Tooltip('index:N').title('x'),
                   alt.Tooltip('variable:N').title('y'),
               ],
               opacity=alt.condition(hover, alt.value(0.9), alt.value(0.2))))

    return (
        alt.concat(
            *subplot_row,
            params=_selection_parameters([hover], [f'{prefix}_view_{i}' for i in range(len(subplot_row))]),
        )
        .resolve_axis(y='shared').configure_view(strokeWidth=0)
    )


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
    if not label_widths:
        return 0
    # Compare the longest label width with the available space for each label
    if max(label_widths) > (
            offset_groups * step_size
            + padding_between_offset * step_size * (offset_groups - 1)
            + padding_between_x * step_size
    ):
        return -45
    else:
        return 0


def _color_encoding(data, color, nominal=False):
    """Resolve a color field without depending on Altair's dtype helper names."""
    if color is None:
        return None
    if isinstance(color, str):
        options = alt.utils.parse_shorthand(color)
    elif isinstance(color, alt.Color):
        options = color.to_dict(context={'data': data})
    elif isinstance(color, dict):
        options = deepcopy(color)
    else:
        raise ValueError('`color` needs to be a string, dict, or `alt.Color` instance.')
    field = options.get('field')
    if not field or field not in data.columns:
        raise ValueError('`color` must reference a column in the input data.')
    if nominal and 'type' not in options:
        options['type'] = 'nominal'
    elif 'type' not in options:
        options['type'] = alt.utils.parse_shorthand(field, data=data)['type']
    options.setdefault('title', None)
    options.setdefault('legend', {'orient': 'top'})
    # Resolve bare fields while the original dataframe is available, before transforms.
    return alt.Color(**options).to_dict(context={'data': data})


def _selection_parameters(params, views):
    """Declare shared selections once, with explicit unit-view scopes."""
    return [
        alt.TopLevelSelectionParameter(**param.param.to_dict(), views=views)
        for param in params
    ]


def _merge_encodings(defaults, overrides, data):
    """Apply per-chart overrides without modifying the caller's encoding objects."""
    result = deepcopy(defaults)
    for channel, value in overrides.items():
        value = deepcopy(value)
        if hasattr(value, 'to_dict'):
            value = value.to_dict(context={'data': data})
        if isinstance(value, dict) and channel in defaults:
            # Options-only overrides inherit the default field and type.
            if not any(key in value for key in ('field', 'datum', 'value', 'aggregate', 'condition')):
                default = defaults[channel]
                if hasattr(default, 'to_dict'):
                    default = default.to_dict(context={'data': data})
                value = {**default, **value}
        result[channel] = value
    return result


def dist(
    data: pd.DataFrame,
    color: Union[str, dict, alt.Color] = None,
    mark: Union[str, dict, alt.MarkDef] = None,
    dtype: str = 'numerical',
    columns: int = None,
    rug: bool = None,
    *,
    density: bool = None,
    bin: Union[bool, int, alt.Bin] = False,
    cumulative: bool = False,
    encoding: Union[dict, alt.Encoding] = None,
) -> alt.ConcatChart:
    """
    Plot the distribution of each dataframe column.

    Visualize univariate distributions
    of either numerical or categorical variables.
    Numerical distributions can be plotted as density plots, histograms, ECDFs, and rug plots.
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
    mark : str, dict, or alt.MarkDef
        Mark options added to the defaults. For compatibility with version 0.1.1,
        ``mark='bar'`` selects a histogram for numerical data unless ``density``
        is explicitly enabled. Default densities use a faded area and an outline.
    dtype : str
        Either 'numerical' or 'categorical'. The legacy 'number' alias selects
        numerical columns; 'object' selects object and string columns, 'category'
        selects categorical columns, and 'bool' selects boolean columns.
    columns : int
        Number of columns in the plot grid. Defaults to a squarish grid.
    rug : bool or None
        Whether to add observations below density plots. By default, a rug is
        included for implicit densities and omitted when ``density=True``.
    density : bool or None
        Whether to compute a kernel density estimate. None defaults to a density
        for numerical data unless a histogram is requested.
    bin : bool, int, or alt.Bin
        Whether to plot a histogram. An integer sets the maximum number of bins.
    cumulative : bool
        Whether to plot the cumulative version of the chart.
        When `density` and `bin` are both set to `False`,
        this creates an empirical cumulative distribution function chart.
        With ``bin=True``, plot cumulative counts within each color group.
    encoding : dict or alt.Encoding
        Encoding options to be added in addition to the defaults,
        e.g. to sort in another order.

    Returns
    -------
    ConcatChart
        Concatenated Chart containing one chart per data frame column
        laid out in a squarish grid.
    """
    if data.empty:
        raise ValueError('Distribution plots require non-empty data.')
    if dtype in ('numerical', 'number'):
        selected_data = data.select_dtypes(include='number').copy()
        numerical = True
    elif dtype in ('categorical', 'object', 'category', 'bool') or dtype is object or dtype is bool:
        numerical = False
        if dtype == 'categorical':
            selected_data = data.select_dtypes(exclude='number').copy()
        elif dtype == 'object' or dtype is object:
            # pandas 3 infers strings as StringDtype; selecting 'object' warns.
            fields = [
                col for col in data
                if pd.api.types.is_object_dtype(data[col].dtype)
                or isinstance(data[col].dtype, pd.StringDtype)
            ]
            selected_data = data[fields].copy()
        else:
            selected_data = data.select_dtypes(include=dtype).copy()
    else:
        raise ValueError("`dtype` must be 'numerical', 'categorical', 'number', 'object', 'category', or 'bool'.")
    plot_columns = selected_data.columns.tolist()
    if not plot_columns:
        raise ValueError('No columns match the requested `dtype`.')
    if not numerical:
        if bin:
            raise ValueError('Cannot bin categorical variables.')
        if density:
            raise ValueError('Cannot compute a density estimate for categorical variables.')

    if columns is None:
        columns = len(plot_columns) if len(plot_columns) <= 3 else ceil(sqrt(len(plot_columns)))
    if not isinstance(columns, int) or isinstance(columns, bool) or columns < 1:
        raise ValueError('`columns` must be a positive integer.')

    if encoding is None:
        encoding = {}
    elif isinstance(encoding, alt.Encoding):
        encoding = encoding.to_dict(context={'data': data})
    elif not isinstance(encoding, dict):
        raise ValueError('`encoding` needs to be a dict or `alt.Encoding` instance.')
    encoding = deepcopy(encoding)
    # Field color overrides determine grouping; constant/conditional encodings
    # remain available through `encoding` without inventing a grouping field.
    override_color = encoding.get('color', color)
    if hasattr(override_color, 'to_dict'):
        override_color = override_color.to_dict(context={'data': data})
    field_color = not isinstance(override_color, dict) or 'field' in override_color
    color = _color_encoding(data, override_color, nominal=True) if field_color else None
    groups = [] if color is None else [color['field']]
    if color is not None:
        selected_data[color['field']] = data[color['field']]
        color_order = selected_data[color['field']].value_counts().index.tolist()
        color.setdefault('sort', color_order)
        xOffset = alt.XOffset(field=color['field'], type='nominal').scale(paddingInner=0.1).sort(color['sort'])
        # Reuse the normalized color rather than an unresolved shorthand override.
        encoding.pop('color', None)

    if mark is None:
        mark = {}
    else:
        if isinstance(mark, str):
            mark = alt.MarkDef(mark).to_dict()
        elif isinstance(mark, alt.MarkDef):
            mark = mark.to_dict()
        elif isinstance(mark, dict):
            mark = deepcopy(mark)
        else:
            raise ValueError('`mark` needs to be a string, dict, or `alt.MarkDef` instance.')
    outline = not mark or mark == {'type': 'area'}
    if numerical and mark.get('type') == 'bar' and density is not True and not bin:
        bin = True
    if density and bin:
        raise ValueError('Cannot compute a density estimate for binned variables.')
    implicit_density = numerical and density is None and not bin
    if rug is None:
        rug = implicit_density
    if density is None:
        density = implicit_density
    if not isinstance(bin, (bool, int, alt.Bin)) or (
        isinstance(bin, int) and not isinstance(bin, bool) and bin < 1
    ):
        raise ValueError('`bin` must be a bool, positive integer, or `alt.Bin` instance.')

    colors = {} if color is None else {'color': alt.Color(**color)}
    charts = []
    if not numerical:
        if cumulative:
            raise ValueError('Cumulative distributions require numerical columns.')
        chart_mark = {'type': 'bar', **mark}
        # Smaller categorical charts first, matching the released layout.
        chart_order = selected_data[plot_columns].nunique().sort_values(kind='stable').index
        for col in chart_order:
            sort = '-y'
            if isinstance(selected_data[col].dtype, pd.CategoricalDtype) and selected_data[col].cat.ordered:
                sort = selected_data[col].cat.categories.tolist()
            defaults = {
                'x': alt.X(field=col, type='nominal').sort(sort).axis(
                    labelAngle=get_label_angle(
                        selected_data[col].unique(),
                        selected_data[color['field']].nunique(dropna=False) if color else 1,
                    )
                ),
                'y': alt.Y('count()').title('Count'),
                **colors,
            }
            if color is not None:
                defaults['xOffset'] = xOffset
            charts.append(
                alt.Chart(selected_data, mark=chart_mark, height=120)
                .encode(**_merge_encodings(defaults, encoding, selected_data))
            )
    elif bin:
        if bin is True:
            bin = alt.Bin(maxbins=20)
        elif isinstance(bin, int) and bin > 0:
            bin = alt.Bin(maxbins=bin)
        elif not isinstance(bin, alt.Bin):
            raise ValueError('`bin` must be a bool, positive integer, or `alt.Bin` instance.')
        chart_mark = {'type': 'bar', 'opacity': 0.7 if color else 0.9, **mark}
        for col in plot_columns:
            chart = alt.Chart(selected_data, mark=chart_mark, height=120, width=200)
            defaults = {
                'x': alt.X(field=col, type='quantitative').bin(bin).title(col),
                'y': alt.Y('count()').stack(None).title('Count'),
                **colors,
            }
            if cumulative:
                # Bin, count, then sum the counts in bin order within each group.
                chart = (
                    chart.transform_filter(alt.expr.isValid(alt.datum[col])).transform_bin(
                        as_=['_ally_bin_start', '_ally_bin_end'], field=col, bin=bin,
                    ).transform_aggregate(
                        _ally_count='count()',
                        groupby=['_ally_bin_start', '_ally_bin_end', *groups],
                    ).transform_window(
                        _ally_cumulative_count='sum(_ally_count)',
                        sort=[alt.SortField('_ally_bin_start')],
                        groupby=groups, frame=[None, 0],
                    )
                )
                defaults.update(
                    x=alt.X('_ally_bin_start:Q').bin('binned').title(col),
                    x2=alt.X2('_ally_bin_end:Q'),
                    y=alt.Y('_ally_cumulative_count:Q').stack(None).title('Cumulative count'),
                )
            charts.append(chart.encode(**_merge_encodings(defaults, encoding, selected_data)))
    elif density:
        for col in plot_columns:
            defaults = {
                'x': alt.X('value:Q').title(col).axis(grid=False, offset=8 if rug else 0),
                'y': alt.Y('density:Q').stack(None).title('Cumulative density' if cumulative else 'Density'),
                **colors,
            }
            chart_mark = {
                'type': 'line' if cumulative else 'area',
                'opacity': 0.9 if cumulative else 0.1,
                **mark,
            }
            chart = (
                alt.Chart(selected_data, mark=chart_mark, width=200, height=200 if cumulative else 120)
                .transform_density(col, as_=['value', 'density'], groupby=groups, minsteps=100, cumulative=cumulative)
                .encode(**_merge_encodings(defaults, encoding, selected_data))
            )
            if outline and not cumulative:
                chart = chart + chart.mark_line(opacity=0.9, strokeWidth=2)
            if rug:
                rugplot = alt.Chart(selected_data).mark_tick(
                    opacity=0.3, yOffset=5, height=7,
                ).encode(
                    x=alt.X(field=col, type='quantitative').axis(grid=False, offset=8),
                    y=alt.datum(0),
                    tooltip=alt.value('Individual observations'),
                    **colors,
                )
                chart = chart + rugplot
            charts.append(chart)
    elif cumulative:
        chart_mark = {'type': 'line', 'interpolate': 'step-after', 'opacity': 0.8, **mark}
        for col in plot_columns:
            defaults = {
                'x': alt.X(field=col, type='quantitative'),
                'y': alt.Y('ecdf:Q').title('Cumulative probability'),
                **colors,
            }
            charts.append(
                alt.Chart(selected_data, mark=chart_mark, height=180, width=180)
                .transform_filter(alt.expr.isValid(alt.datum[col]))
                .transform_window(
                    window=[{'op': 'cume_dist', 'as': 'ecdf'}],
                    sort=[alt.SortField(col)], groupby=groups,
                )
                .encode(**_merge_encodings(defaults, encoding, selected_data))
            )
    else:
        chart_mark = {'type': 'tick', 'opacity': 0.4, **mark}
        for col in plot_columns:
            defaults = {'x': alt.X(field=col, type='quantitative'), **colors}
            charts.append(
                alt.Chart(selected_data, mark=chart_mark, width=180)
                .encode(**_merge_encodings(defaults, encoding, selected_data))
            )
    return alt.concat(*charts, columns=columns).configure_view(stroke=None)


def heatmap(data, color=None, sort=None, rescale='min-max',
            cat_schemes=['tableau10', 'set2', 'accent'],
            num_scheme='yellowgreenblue'):
    """
    Plot the values of all columns and observations as a heatmap.

    For large datasets, use Altair's VegaFusion data transformer or
    ``aly.alt.data_transformers.disable_max_rows()`` to relax the row limit.

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
        raise ValueError("`rescale` must be 'min-max', 'mean-sd', None, or a callable.")

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
    Plot individual missing values and overall counts for each column.

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
    if data.empty:
        raise ValueError('Missing-value plots require non-empty data.')
    cols_with_nans = data.columns[data.isna().any()]
    if cols_with_nans.empty:
        cols_with_nans = data.columns
    heatmap_width = data.shape[0]
    # TODO can transform_fold be used here too?
    # Long form data
    data = (
        data[cols_with_nans]
        .isna()
        .set_axis(pd.RangeIndex(len(data)))
        .rename_axis('index')
        .reset_index()
        .melt(id_vars='index', var_name='variable'))

    # Sorted counts of NaNs per column
    nan_counts = (
        data.query('value == True')
        .groupby('variable')
        .size()
        .reset_index()
        .rename(columns={0: 'count'})
        .sort_values('count'))
    sorted_nan_cols = nan_counts['variable'].to_list() or cols_with_nans.tolist()
    max_nans = int(nan_counts['count'].max()) if not nan_counts.empty else 1

    # Bar chart of NaN counts per column
    zoom = alt.selection_interval(name=f'nan_brush_{next(_chart_ids)}', encodings=['x'])
    nan_bars = (
        alt.Chart(data.query('value == True'), title='NaN count').mark_bar(color='steelblue', height=17).encode(
            alt.X('count()', axis=None, scale=alt.Scale(domain=[0, max_nans])),
            alt.Y('variable:N', axis=alt.Axis(grid=False, title='', labels=False, ticks=False), sort=sorted_nan_cols))
        .properties(width=100))
    nan_bars_with_text = ((
        nan_bars
        + nan_bars.mark_text(align='left', dx=2)
        .encode(text='count()'))
        .transform_filter(zoom))

    # Heatmap of individual NaNs
    color_scale = alt.Scale(domain=[False, True], range=['#dde8f1', 'steelblue'])

    nan_heatmap = (
        alt.Chart(data, title='Individual NaNs').mark_rect(height=17).encode(
            alt.X('index:O', axis=None),
            alt.Y('variable:N', title=None, sort=sorted_nan_cols),
            alt.Color('value:N', scale=color_scale, sort=[False, True],
                      legend=alt.Legend(orient='top', offset=-13), title=None),
            alt.Stroke('value:N', scale=color_scale, sort=[False, True], legend=None))
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
    color : str, dict, or alt.Color
        Color field, optionally with an Altair type suffix. Discrete colors
        have a clickable legend above the chart; numerical colors use a gradient.
    tooltip: str
        Column in **data** used for the tooltip encoding.
    mark: str
        Shape of the points. Passed to Chart.
        One of "circle", "square", "tick", or "point". Use "rect" for a
        two-dimensional histogram with color and tooltip showing counts.
    width: int or float
        Chart width.
    height: int or float
        Chart height.

    Returns
    -------
    ConcatChart
        Concatenated Chart of pairwise column scatter plots.
    """
    cols = data.select_dtypes('number').columns
    if data.empty or len(cols) < 2:
        raise ValueError('Pair plots require non-empty data with at least two numerical columns.')
    prefix = f'pair_{next(_chart_ids)}'
    params = []
    if mark == 'rect':
        bins = alt.Bin(maxbins=30)
        color = alt.Color('count()').title('Count').legend(orient='top')
        opacity = alt.value(1)
        tooltip = alt.Tooltip('count()').title('Count')
    else:
        bins = False
        color_options = _color_encoding(data, color)
        brush = alt.selection_interval(name=f'{prefix}_brush')
        params.append(brush)
        selected_color = alt.value('#4c78a8') if color_options is None else alt.Color(**color_options)
        color = alt.condition(brush, selected_color, alt.value('lightgrey'))
        opacity = alt.value(0.6)
        if color_options and color_options['type'] in ('nominal', 'ordinal') and color_options['legend'] is not None:
            legend_click = alt.selection_point(
                name=f'{prefix}_legend', fields=[color_options['field']], bind='legend',
            )
            params.append(legend_click)
            opacity = alt.condition(legend_click, alt.value(0.6), alt.value(0.1))
    hidden_axis = alt.Axis(domain=False, title='', labels=False, ticks=False)

    # Create corner of pair-wise scatters
    i = 0
    exclude_zero = alt.Scale(zero=False)
    col_combos = list(combinations(cols, 2))[::-1]
    subplot_row = []
    view_names = []
    while i < len(cols) - 1:
        plot_column = []
        for num, (y, x) in enumerate(col_combos[:i+1]):
            if num == 0 and i == len(cols) - 2:
                subplot = alt.Chart(data, mark=mark).encode(
                    alt.X(x).scale(exclude_zero).bin(bins),
                    alt.Y(y).scale(exclude_zero).bin(bins))
            elif num == 0:
                subplot = (
                    alt.Chart(data, mark=mark).encode(
                        alt.X(x).scale(exclude_zero).axis(hidden_axis).bin(bins),
                        alt.Y(y).scale(exclude_zero).bin(bins)))
            elif i == len(cols) - 2:
                subplot = (
                    alt.Chart(data, mark=mark).encode(
                        alt.X(x).scale(exclude_zero).bin(bins),
                        alt.Y(y).scale(exclude_zero).axis(hidden_axis).bin(bins)))
            else:
                subplot = (
                    alt.Chart(data, mark=mark).encode(
                        alt.X(x).scale(exclude_zero).axis(hidden_axis).bin(bins),
                        alt.Y(y).scale(exclude_zero).axis(hidden_axis).bin(bins)))
            if tooltip is not None:
                subplot = subplot.encode(tooltip=tooltip)
            view_name = f'{prefix}_view_{len(view_names)}'
            view_names.append(view_name)
            plot_column.append(
                subplot
                .encode(opacity=opacity, color=color)
                .properties(width=width, height=height, name=view_name))
        subplot_row.append(alt.hconcat(*plot_column))
        i += 1
        col_combos = col_combos[i:]

    return alt.vconcat(*subplot_row, params=_selection_parameters(params, view_names))


def parcoord(data, color=None, rescale='min-max'):
    """
    Plot the values of all columns and observations as a parallel coordinates plot.

    Parameters
    ----------
    data : DataFrame
        pandas DataFrame with input data.
    color : str, dict, or alt.Color
        Color field, optionally with a type suffix. Discrete colors have a
        clickable legend above the chart. None draws uncolored lines.
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
    color_options = _color_encoding(data, color)
    data = data.copy()
    num_cols = data.select_dtypes('number').columns.to_list()
    if data.empty or not num_cols:
        raise ValueError('Parallel coordinates require non-empty data with numerical columns.')

    if rescale == 'mean-sd':
        data[num_cols] = data[num_cols].apply(lambda x: (x - x.mean()) / x.std())
    elif rescale == 'min-max':
        data[num_cols] = data[num_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    elif callable(rescale):
        data[num_cols] = data[num_cols].apply(rescale)
    elif rescale is not None:
        raise ValueError("`rescale` must be 'min-max', 'mean-sd', None, or a callable.")

    fields = num_cols.copy()
    encodings = {'opacity': alt.value(0.6)}
    params = []
    if color_options is not None:
        if color_options['field'] not in fields:
            fields.append(color_options['field'])
        encodings['color'] = alt.Color(**color_options)
        if color_options['type'] in ('nominal', 'ordinal') and color_options['legend'] is not None:
            legend_click = alt.selection_point(
                name=f'parcoord_legend_{next(_chart_ids)}', fields=[color_options['field']], bind='legend',
            )
            params.append(legend_click)
            encodings['opacity'] = alt.condition(legend_click, alt.value(0.6), alt.value(0.05))

    return alt.Chart(data[fields]).transform_window(
        index='count()'
    ).transform_fold(
        num_cols
    ).mark_line().encode(
        alt.X('key:O').title(None).scale(nice=False, padding=0.05),
        alt.Y('value:Q').title(None),
        detail='index:N',
        **encodings,
    ).properties(width=len(num_cols) * 100).add_params(*params)
