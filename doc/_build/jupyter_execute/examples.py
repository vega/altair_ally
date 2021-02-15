# Examples

import altair_ally as aly
from vega_datasets import data

# aly.alt.data_transformers.enable('data_server')
aly.alt.data_transformers.disable_max_rows()

movies = data.movies().sample(400, random_state=234890)

## Missing values

aly.nan(movies)

## Univariate distributions

aly.dist(movies)

aly.dist(movies, mark='bar')

aly.dist(movies, 'MPAA Rating')

## Pairwise variable relationships

aly.pair(movies)

## Pairwise variable correlation

aly.corr(movies)

## Parallel coordinates

aly.parcoord(data.cars(), 'Origin', rescale='min-max')