# Changelog

## 0.2.1

- Adjust opacity

## 0.2.0

### Requirements and migration

- Require Python 3.11 or newer and Altair 6.x. Support pandas 1.5.3 through 3.x
  and NumPy 1.23.5 or newer.
- Use Altair's built-in `altair.datasets` loader and remove the separate
  `vega_datasets` dependency.
- Retain the 0.1.1 positional argument order for `dist()` and its legacy dtype
  aliases. The expanded `density`, `bin`, `cumulative`, and `encoding` options
  are keyword-only. Calls using the previously unreleased `main` positional
  order should use keyword arguments.

### Plotting improvements

- Support density, histogram, cumulative histogram, ECDF, and rug-only
  distributions, including grouped distributions and custom Altair encodings.
- Compute cumulative counts and ECDFs within each color group, excluding missing
  observations before calculation.
- Normalize encodings for layered charts, accept `alt.MarkDef` overrides, and
  avoid modifying caller-owned data or leaking channel overrides across panels.
- Use unstacked density areas with opacity 0.2 and line outlines with opacity 0.9.
  Explicit line density plots also default to 0.9, and custom opacity is honored.
- Support pandas 3 string dtypes, ordered categories, and categorical chart
  ordering by the number of distinct values.
- Restore uncolored pair plots and parallel coordinates plots, and support bare
  and typed color names. Retain 2D histogram pair plots with `mark='rect'`.
- Include boolean variables in correlations and show both variable names and
  coefficients in tooltips.
- Use current selection APIs, share interactions across the intended views,
  and give independently created charts independent selection names.
- Improve missing-value brushing by fading unselected rows to 30% opacity,
  retaining Altair's default white outline and translucent gray brush fill.
  Preserve linked count updates and restore all rows when the selection is cleared.
- Handle named or duplicated indexes and datasets with no missing values in
  `nan()`.
- Raise clear errors for unsupported options and insufficient input data.

### Development and documentation

- Adopt uv, a committed lockfile, Hatchling builds, and Taskipy development
  commands. Keep runtime and distribution versions in one source file.
- Add schema-validation and rendering regression tests and a compatibility CI
  matrix covering Altair 6.0/6.3, pandas 1.5/2/3, and Python 3.11/3.13.
- Refresh the examples and migration guide and use the deployed documentation
  site at <https://vega.github.io/altair_ally/>.
