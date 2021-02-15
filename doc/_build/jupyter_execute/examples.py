# Examples

import altair_ally as aly
from vega_datasets import data

# aly.alt.data_transformers.enable('data_server')
aly.alt.data_transformers.disable_max_rows()

movies = (
    data
    .movies()
    .sample(400, random_state=234890)
    .query('`MPAA Rating` in ["G", "PG", "PG-13", "R"]')
    [['IMDB Votes', 'IMDB Rating', 'Rotten Tomatoes Rating',
      'Running Time min', 'MPAA Rating']])
# movies = data.cars()
movies.shape

## Missing values

Selecting an inerval in the heatmap will automatically update the bar plot of counts.

aly.nan(movies)

## Univariate distributions

Densities can be made as areas or lines,
and have a rug plot included to indicate the number of observations.
Histograms can be made with the `'bar'` mark
and a categorical value can be used to color the distributions.

aly.dist(movies)

aly.dist(movies, mark='bar')

aly.dist(movies, 'MPAA Rating')

## Pairwise variable relationships

Selecting in one plot highlights the same points across all subplots.
A categorical variable can be used to color the points.

aly.pair(movies)

aly.pair(movies, 'MPAA Rating')

## Pairwise variable correlation

Hovering over a point shows the exact coefficient
and highlights the point across all subplots.

aly.corr(movies)

## Parallel coordinates

aly.parcoord(movies, 'MPAA Rating', rescale='min-max')