# Compatibility and migration

## Development version

The unreleased **0.2.0.dev0** version combines the expanded distribution API
from `main` with the plotting improvements released in 0.1.1. Install it from
this checkout with `uv sync --locked`, then use `uv run` to run Python or notebooks
in the project environment.

Supported dependencies are Altair **6.x**, pandas **1.5.3 through 3.x**,
NumPy **1.23.5 or newer**, and Python **3.11 or newer**. CI checks the minimum
dependencies, Altair 6.0 and 6.3, and pandas 3.

Examples use Altair's built-in dataset loader, introduced in Altair 6.0:

```python
from altair.datasets import data
import altair_ally as aly

aly.pair(data.cars(), color='Origin')
```

All interactions use `selection_point()`, `selection_interval()`, and
`.add_params()`. Shared selections are declared once and scoped to all their
subplots. Customization with Altair's `condition()` remains supported.

## Distribution plots

The first six positional arguments retain the released order:

```python
dist(data, color=None, mark=None, dtype='numerical', columns=None, rug=None,
     *, density=None, bin=False, cumulative=False, encoding=None)
```

Use keyword arguments for plot options. The expanded, previously unreleased
`main` API had a different positional order; calls using its positional options
should be rewritten with keywords.

```python
import altair as alt
import altair_ally as aly

# Density, with a faded area, line outline, and observations below the axis.
aly.dist(df)
aly.dist(df, color='species')

# Histograms; integers and Altair bin objects configure the bins.
aly.dist(df, bin=True)
aly.dist(df, bin=15, color='species:N')
aly.dist(df, bin=alt.Bin(step=5))

# Cumulative bin counts are calculated separately for each color group.
aly.dist(df, bin=True, cumulative=True, color='species')

# Smooth cumulative density, empirical CDF, or observations only.
aly.dist(df, cumulative=True)
aly.dist(df, density=False, cumulative=True)
aly.dist(df, density=False)

# Categorical counts with dodged bars when a color field is supplied.
aly.dist(df, dtype='categorical', color='species')
```

### Compatibility with 0.1.1

- `mark='bar'` still requests a histogram for numerical columns. `bin=True` is
  the more explicit spelling. Enabling `density=True` explicitly allows a bar
  mark to be used for density estimates instead.
- `dtype='number'` is an alias for `'numerical'`.
- `dtype='object'` selects object and string columns, including pandas 3's
  inferred string dtype. `'category'` and `'bool'` select those specific dtypes.
  `'categorical'` selects all non-numerical columns.
- Categorical charts are ordered from fewer to more distinct values. Ordered
  pandas categories retain their category order on the axis.
- Default densities retain 0.1.1's faded area and line outline. Density areas
  are explicitly unstacked, so colored distributions overlap rather than sum.
- Explicit `mark='line'` density plots use opacity 0.9, matching the outlines
  rather than the faded area fill. Explicit custom opacity values take precedence.
- `rug=False` removes observations from density plots. Explicit `density=True`
  omits the rug by default; use `rug=True` to add it back.
- Uncolored charts omit color encodings and legend selections rather than use
  a synthetic empty field name.

### Custom marks and encodings

`color` accepts a column name, typed shorthand, `alt.Color`, or a field dictionary.
Bare color names in `dist()` are nominal by default. Legends appear above the
plots unless a custom legend is supplied.

`mark` accepts a mark name, `alt.MarkDef`, or a dictionary of mark options.
`encoding` accepts a dictionary or `alt.Encoding`. Options-only channel overrides
inherit the default field and type independently for each subplot. The supplied
dataframe, mark, and encoding objects are not modified.

```python
aly.dist(
    df,
    bin=True,
    mark=alt.MarkDef(type='bar', opacity=0.5),
    encoding={'x': {'axis': {'labelAngle': -30}}},
)
```

A field-based color override in `encoding` also controls grouping for densities,
ECDFs, and cumulative histograms. Constant color encodings create no grouping.
Missing observations are excluded before computing cumulative counts and ECDFs.
Cumulative plots require numerical columns. Conflicting density/bin options,
empty data, and dtype selections with no matching columns raise `ValueError`.

## Other plots

- `corr()` includes boolean columns and shows the coefficient and both variable
  names in tooltips. Hover highlighting works across the correlation methods;
  `select_on='click'` changes the triggering event.
- `pair()` works with or without color and tooltip fields. Bare and typed color
  names work. `mark='rect'` retains 0.1.1's two-dimensional histograms with
  binned axes and count-based colors/tooltips; its color and tooltip arguments
  are replaced by counts, and scatterplot selections are omitted.
- `pair()` and `parcoord()` put discrete color legends above the plot and make
  them clickable. Continuous color gradients do not create legend selections.
  Explicit custom legends, including `legend=None`, are honored.
  Independently created plots have independent selection names when concatenated.
- `parcoord()` works without a color field and accepts typed color names.
- `nan()` uses row positions for brushing, including when dataframe indexes are
  named or duplicated. With no missing values it shows the non-missing heatmap
  for all columns and an empty count panel.
  The brush has a dark outline; rows outside the selection fade to 30% opacity,
  while selected rows remain fully visible. Clearing the brush restores all rows.
- Invalid `rescale` values in `heatmap()` and `parcoord()` raise `ValueError`;
  supported values are `'min-max'`, `'mean-sd'`, `None`, or a callable.

## Running the checks

```sh
uv sync --locked --all-groups
uv run task test
uv run --group doc task doc-build
uv run task build
```

Tests treat warnings as errors and check both schema validation and actual
Vega-Lite rendering. They also verify cumulative counts, ECDF endpoints, shared
selection scopes, caller-owned objects, and legacy arguments.

The default `dev` dependency group includes the tests and task runner. The `doc`
group adds Jupyter Book and notebook dependencies. Development dependencies are
defined in `pyproject.toml` and resolved in `uv.lock`; contributor setup now uses
`uv sync` rather than `pip install -e '.[test,doc]'`.
