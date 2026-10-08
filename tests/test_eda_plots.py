from copy import deepcopy
import re
from xml.etree import ElementTree

import altair as alt
import pandas as pd
import pytest
import vl_convert as vlc

import altair_ally.eda_plots as eda_plots


@pytest.fixture
def data():
    return pd.DataFrame({
        'x': [1., 2., 3., 4., 5., 6., 7., 8.],
        'y': [2., 1., 5., 3., 8., 6., 7., 4.],
        'z': [8., 7., 6., 5., 4., 3., 2., 1.],
        'group': ['a'] * 4 + ['b'] * 4,
        'flag': [True, False] * 4,
        'category': pd.Categorical(
            ['small', 'large'] * 4, categories=['large', 'small'], ordered=True,
        ),
    })


@pytest.fixture
def missing(data):
    data = data.copy()
    data.loc[1, 'x'] = float('nan')
    data.loc[2, 'y'] = float('nan')
    data.loc[3, 'group'] = None
    return data


def render(chart):
    """Validate and execute the specification with the matching Vega-Lite runtime."""
    spec = chart.to_dict()
    version = re.search(r'/v(\d+\.\d+)\.', spec['$schema']).group(1)
    svg = vlc.vegalite_to_svg(spec, vl_version=version)
    root = ElementTree.fromstring(svg)
    # A valid SVG envelope alone can hide an empty/broken chart.
    symbols = [node for node in root.iter() if node.get('role') == 'graphics-symbol']
    assert symbols, 'The chart rendered no data marks.'
    return spec, [node.get('aria-label', '') for node in symbols]


def nodes(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from nodes(child)


def test_get_label_angle():
    assert eda_plots.get_label_angle(list('abcdefg'), 2) == 0
    assert eda_plots.get_label_angle(list('abcde'), 1) == 0
    assert eda_plots.get_label_angle(['a very long category label'], 1) == -45
    assert eda_plots.get_label_angle([], 1) == 0


def test_dist():
    data = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
    chart = eda_plots.dist(data)
    assert isinstance(chart, alt.ConcatChart)
    render(chart)

    data = pd.DataFrame({'x': ['a', 'b', 'c', 'd', 'e'], 'y': [1, 2, 3, 4, 5]})
    chart = eda_plots.dist(data, color='y', dtype='categorical', mark='bar')
    assert isinstance(chart, alt.ConcatChart)
    render(chart)


@pytest.mark.parametrize('function, options', [
    ('corr', {}),
    ('corr', {'select_on': 'click', 'corr_types': ['pearson']}),
    ('heatmap', {}),
    ('heatmap', {'color': 'group', 'sort': 'x'}),
    ('heatmap', {'color': ['group', 'flag'], 'rescale': 'mean-sd'}),
    ('nan', {}),
    ('pair', {}),
    ('pair', {'color': 'group'}),
    ('pair', {'color': 'group:N', 'tooltip': 'group'}),
    ('pair', {'color': 'x:Q'}),
    ('pair', {'mark': 'rect'}),
    ('parcoord', {}),
    ('parcoord', {'color': 'group'}),
    ('parcoord', {'color': 'group:N'}),
    ('parcoord', {'color': 'x:Q', 'rescale': None}),
    ('dist', {}),
    ('dist', {'density': True}),
    ('dist', {'density': True, 'rug': True}),
    ('dist', {'rug': False}),
    ('dist', {'bin': True}),
    ('dist', {'bin': 6}),
    ('dist', {'bin': alt.Bin(step=2)}),
    ('dist', {'bin': True, 'cumulative': True}),
    ('dist', {'density': False, 'cumulative': True}),
    ('dist', {'cumulative': True}),
    ('dist', {'density': False}),
    ('dist', {'dtype': 'categorical'}),
    ('dist', {'dtype': 'categorical', 'color': 'group'}),
    ('dist', {'mark': alt.MarkDef(type='line', strokeWidth=4)}),
])
def test_plot_renders(function, options, data, missing):
    frame = missing if function == 'nan' else data
    original = frame.copy(deep=True)
    spec, _ = render(getattr(eda_plots, function)(frame, **options))
    pd.testing.assert_frame_equal(frame, original)
    assert all(node.get('field') != '' for node in nodes(spec))


@pytest.mark.parametrize('color', [
    'group', 'group:N', alt.Color('group:N'), {'field': 'group', 'type': 'nominal'},
])
@pytest.mark.parametrize('function', ['dist', 'pair', 'parcoord'])
def test_color_forms_preserve_real_fields_and_legends(function, color, data):
    spec, _ = render(getattr(eda_plots, function)(data, color=deepcopy(color)))
    color_nodes = [node for node in nodes(spec) if node.get('field') == 'group']
    assert color_nodes
    assert any(node.get('legend', {}).get('orient') == 'top' for node in color_nodes)
    for node in nodes(spec):
        assert 'group:N' not in node.get('groupby', [])
        if 'select' in node and node.get('bind') == 'legend':
            assert node['select']['fields'] == ['group']


@pytest.mark.parametrize('function', ['pair', 'parcoord'])
def test_continuous_colors_do_not_create_legend_selections(function, data):
    spec, _ = render(getattr(eda_plots, function)(data, color='x'))
    assert all(param.get('bind') != 'legend' for param in spec.get('params', []))


@pytest.mark.parametrize('function', ['pair', 'parcoord'])
def test_custom_legend_is_preserved(function, data):
    color = alt.Color('group:N').legend(orient='bottom', title='Groups')
    original = color.to_dict()
    spec, _ = render(getattr(eda_plots, function)(data, color=color))
    assert color.to_dict() == original
    assert any(node.get('legend') == original['legend'] for node in nodes(spec))


@pytest.mark.parametrize('function', ['corr', 'pair'])
def test_shared_selections_cover_every_subplot(function, data):
    options = {'color': 'group'} if function == 'pair' else {}
    spec, _ = render(getattr(eda_plots, function)(data, **options))
    unit_names = {node['name'] for node in nodes(spec) if 'mark' in node and 'name' in node}
    assert len(unit_names) > 1
    assert spec['params']
    for param in spec['params']:
        assert set(param['views']) == unit_names
        assert any(node.get('param') == param['name'] for node in nodes(spec))


def test_corr_includes_booleans_and_variable_tooltips(data):
    data.columns.name = 'measurements'
    spec, labels = render(eda_plots.corr(data))
    assert any('flag' in label for label in labels)
    tooltips = spec['concat'][0]['encoding']['tooltip']
    assert {item['title'] for item in tooltips} == {'corr', 'x', 'y'}


def test_pair_rect_uses_binned_counts(data):
    spec, labels = render(eda_plots.pair(data, mark='rect'))
    assert not spec.get('params')
    assert any('Count:' in label for label in labels)
    units = [
        node for node in nodes(spec)
        if node.get('mark') == 'rect' or isinstance(node.get('mark'), dict) and node['mark'].get('type') == 'rect'
    ]
    assert len(units) == 3
    for unit in units:
        assert unit['encoding']['x']['bin']['maxbins'] == 30
        assert unit['encoding']['y']['bin']['maxbins'] == 30
        assert unit['encoding']['color']['aggregate'] == 'count'


def test_dist_released_positional_arguments(data):
    spec, _ = render(eda_plots.dist(data, 'group', 'bar', 'number', 2, False))
    assert spec['columns'] == 2
    assert all(chart['encoding']['x'].get('bin') for chart in spec['concat'])


@pytest.mark.parametrize('dtype, expected', [
    ('number', {'x', 'y', 'z'}),
    ('object', {'group'}),
    (object, {'group'}),
    ('category', {'category'}),
    ('bool', {'flag'}),
    (bool, {'flag'}),
    ('categorical', {'group', 'flag', 'category'}),
])
def test_dist_dtype_aliases(dtype, expected, data):
    options = {'density': False} if dtype == 'number' else {}
    spec, _ = render(eda_plots.dist(data, dtype=dtype, **options))
    assert {chart['encoding']['x']['field'] for chart in spec['concat']} == expected


def test_ordered_categories_and_categorical_chart_order(data):
    spec, _ = render(eda_plots.dist(data, dtype='categorical'))
    categorical = next(chart for chart in spec['concat'] if chart['encoding']['x']['field'] == 'category')
    assert categorical['encoding']['x']['sort'] == ['large', 'small']
    widths = [data[chart['encoding']['x']['field']].nunique() for chart in spec['concat']]
    assert widths == sorted(widths)


def test_default_density_is_unstacked_with_outline_and_offset_rug(data):
    spec, _ = render(eda_plots.dist(data, color='group:N'))
    layers = spec['concat'][0]['layer']
    assert {layer['mark']['type'] for layer in layers} == {'area', 'line', 'tick'}
    for layer in layers:
        if layer['mark']['type'] in ('area', 'line'):
            assert layer['encoding']['y']['stack'] is None
            assert layer['transform'][0]['groupby'] == ['group']
        else:
            assert layer['mark']['yOffset'] > 0
            assert layer['encoding']['x']['axis']['offset'] > 0


@pytest.mark.parametrize('color', [None, 'group'])
@pytest.mark.parametrize('mark, expected_opacity', [
    ('line', 0.9),
    (alt.MarkDef(type='line', strokeWidth=4), 0.9),
    ({'type': 'line'}, 0.9),
    (alt.MarkDef(type='line', opacity=0.4), 0.4),
])
def test_density_lines_render_with_visible_or_custom_opacity(color, mark, expected_opacity, data):
    spec, _ = render(eda_plots.dist(data, color, mark=mark))
    version = re.search(r'/v(\d+\.\d+)\.', spec['$schema']).group(1)
    scenegraph = vlc.vegalite_to_scenegraph(spec, vl_version=version)
    lines = [
        node for node in nodes(scenegraph)
        if node.get('marktype') == 'line' and node.get('role') == 'mark'
    ]
    assert lines
    for line in lines:
        assert line['items']
        assert all(item['opacity'] == pytest.approx(expected_opacity) for item in line['items'])


@pytest.mark.parametrize('color, expected_total', [(None, 8), ('group:N', 4)])
def test_cumulative_histogram_counts_are_computed_per_group(color, expected_total, data):
    _, labels = render(eda_plots.dist(data[['x', 'group']], bin=4, cumulative=True, color=color))
    counts = [
        float(match.group(1)) for label in labels
        if (match := re.search(r'Cumulative count: ([\d.]+)', label))
    ]
    assert counts
    assert max(counts) == expected_total
    if color is not None:
        for group in ('a', 'b'):
            assert any(f'Cumulative count: {expected_total}' in label and f'group: {group}' in label for label in labels)


def test_ecdf_reaches_one_in_each_group(data):
    spec, _ = render(eda_plots.dist(data[['x', 'group']], density=False, cumulative=True, color='group:N'))
    version = re.search(r'/v(\d+\.\d+)\.', spec['$schema']).group(1)
    # SVG exposes one description per line; the scenegraph exposes every point.
    scenegraph = vlc.vegalite_to_scenegraph(spec, vl_version=version)
    labels = [node['description'] for node in nodes(scenegraph) if 'description' in node]
    for group in ('a', 'b'):
        assert any('Cumulative probability: 1' in label and f'group: {group}' in label for label in labels)


def test_cumulative_histograms_exclude_missing_observations(missing):
    _, labels = render(eda_plots.dist(missing[['x', 'group']], bin=4, cumulative=True, color='group:N'))
    for group, total in [('a', 2), ('b', 4)]:
        counts = [
            float(match.group(1)) for label in labels
            if f'group: {group}' in label and (match := re.search(r'Cumulative count: ([\d.]+)', label))
        ]
        assert max(counts) == total


def test_ecdf_excludes_missing_observations(missing):
    spec, _ = render(eda_plots.dist(missing[['x']], density=False, cumulative=True))
    version = re.search(r'/v(\d+\.\d+)\.', spec['$schema']).group(1)
    scenegraph = vlc.vegalite_to_scenegraph(spec, vl_version=version)
    probabilities = [
        float(match.group(1)) for node in nodes(scenegraph)
        if (match := re.search(r'Cumulative probability: ([\d.]+)', node.get('description', '')))
    ]
    assert len(probabilities) == missing['x'].notna().sum()
    assert min(probabilities) == pytest.approx(1 / missing['x'].notna().sum())
    assert max(probabilities) == 1


@pytest.mark.parametrize('function', ['pair', 'parcoord'])
def test_independent_plots_can_be_concatenated(function, data):
    first = getattr(eda_plots, function)(data, color='group')
    second = getattr(eda_plots, function)(data, color='group')
    spec, _ = render(alt.hconcat(first, second))
    names = [param['name'] for param in spec['params']]
    assert len(names) == len(set(names))


def test_bar_mark_with_density_disabled_still_selects_histograms(data):
    spec, _ = render(eda_plots.dist(data, mark='bar', density=False))
    assert all(chart['encoding']['x'].get('bin') for chart in spec['concat'])


@pytest.mark.parametrize('mode', [{'bin': True}, {'dtype': 'categorical'}, {'density': False}])
def test_encoding_and_mark_overrides_do_not_leak_between_columns(mode, data):
    encoding = {'x': {'axis': {'labelAngle': 15}}}
    mark = {'opacity': 0.55}
    original_encoding = deepcopy(encoding)
    original_mark = deepcopy(mark)
    spec, _ = render(eda_plots.dist(data, encoding=encoding, mark=mark, **mode))
    assert encoding == original_encoding
    assert mark == original_mark
    fields = [chart['encoding']['x']['field'] for chart in spec['concat']]
    assert len(fields) == len(set(fields))
    for chart in spec['concat']:
        assert chart['encoding']['x']['axis']['labelAngle'] == 15
        assert chart['mark']['opacity'] == 0.55


def test_altair_encoding_object_can_group_density(data):
    encoding = alt.Encoding(color=alt.Color('group:N'), x=alt.X('value:Q').axis(grid=True))
    original = encoding.to_dict()
    spec, _ = render(eda_plots.dist(data, density=True, encoding=encoding))
    assert encoding.to_dict() == original
    for layer in spec['concat'][0]['layer']:
        assert layer['transform'][0]['groupby'] == ['group']
        assert layer['encoding']['x']['axis']['grid'] is True


def test_constant_color_encoding_needs_no_grouping_field(data):
    spec, _ = render(eda_plots.dist(data, encoding={'color': alt.value('red')}, density=True))
    for layer in spec['concat'][0]['layer']:
        assert layer['transform'][0]['groupby'] == []
        assert layer['encoding']['color'] == {'value': 'red'}


@pytest.mark.parametrize('has_missing', [False, True])
def test_nan_handles_named_duplicate_indexes(has_missing, data, missing):
    frame = (missing if has_missing else data).copy()
    frame.index = pd.Index(['same'] * len(frame), name='observation')
    original = frame.copy(deep=True)
    spec, _ = render(eda_plots.nan(frame))
    assert spec['params'][0]['select']['encodings'] == ['x']
    records = next(records for records in spec['datasets'].values() if records and 'index' in records[0])
    assert {record['index'] for record in records} == set(range(len(frame)))
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize('options, message', [
    ({'density': True, 'bin': True}, 'binned'),
    ({'dtype': 'categorical', 'bin': True}, 'Cannot bin'),
    ({'dtype': 'categorical', 'density': True}, 'density estimate'),
    ({'dtype': 'categorical', 'cumulative': True}, 'numerical'),
    ({'dtype': 'invalid'}, 'dtype'),
    ({'columns': 0}, 'columns'),
    ({'bin': -1}, 'bin'),
    ({'bin': 0}, 'bin'),
    ({'bin': 'invalid'}, 'bin'),
    ({'mark': 42}, 'mark'),
    ({'encoding': 42}, 'encoding'),
    ({'color': 'missing'}, 'column'),
])
def test_dist_invalid_options_raise_clear_errors(options, message, data):
    with pytest.raises(ValueError, match=message):
        eda_plots.dist(data, **options)


@pytest.mark.parametrize('function, frame, options', [
    ('dist', pd.DataFrame({'group': ['a', 'b']}), {}),
    ('dist', pd.DataFrame({'x': [1, 2]}), {'dtype': 'categorical'}),
    ('pair', pd.DataFrame({'x': [1, 2]}), {}),
    ('corr', pd.DataFrame({'x': [1, 2]}), {}),
    ('parcoord', pd.DataFrame({'group': ['a', 'b']}), {}),
    ('nan', pd.DataFrame({'x': []}), {}),
])
def test_empty_or_insufficient_columns_raise_clear_errors(function, frame, options):
    with pytest.raises(ValueError):
        getattr(eda_plots, function)(frame, **options)


@pytest.mark.parametrize('function', ['heatmap', 'parcoord'])
def test_invalid_rescaling_raises_clear_errors(function, data):
    with pytest.raises(ValueError, match='rescale'):
        getattr(eda_plots, function)(data, rescale='invalid')
