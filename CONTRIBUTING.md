# Contributing

## Set up the project

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) 0.10.4 or
newer, then run these commands from the repository root:

```sh
uv sync --locked --all-groups
uv run task test
```

uv creates an editable installation in `.venv` and installs dependencies from
`uv.lock`. It can download the Python version selected by `.python-version`
(currently 3.13). Environment activation is optional when using `uv run`.
The package supports Python 3.11 or newer, and CI checks both 3.11 and 3.13.

## Development commands

```sh
uv run task test                     # Validate and render all test charts
uv run --group doc task doc-build    # Execute examples and build the book
uv run --group doc task doc-serve    # Serve the built book at localhost:8000
uv run task build                    # Build an sdist and wheel in dist/
uv run --locked task lock-check      # Check that uv.lock matches pyproject.toml
```

Tests treat warnings as errors. Documentation builds also fail on Sphinx
warnings. The `dev` group, enabled by default, includes the `test` group and
`taskipy`; the `doc` group contains documentation-only dependencies.
Use `uv sync --no-dev` for a runtime-only environment.

Metadata, dependency groups, pytest settings, and task definitions live in
`pyproject.toml`. Hatchling reads the package version from
`altair_ally/_version.py`, which is also used at runtime.

When migrating an older Setuptools checkout, remove the obsolete generated
`altair_ally.egg-info/` directory if it is still present, then sync again. Its
cached metadata can otherwise mask the installed package's version.

## Update dependencies

After editing dependency requirements, regenerate the lockfile:

```sh
uv lock
uv sync --locked --all-groups
uv run task test
```

To update all locked dependencies within the declared requirements:

```sh
uv lock --upgrade
```

Include `uv.lock` changes with dependency changes. CI uses `--locked` so an
out-of-date lockfile fails rather than being silently regenerated.

## Test supported dependency combinations

CI tests the locked development environment and separately overlays supported
Altair, pandas, and NumPy versions with `uv run --with`. For example:

```sh
uv run --locked --python 3.11 \
  --with 'altair==6.0.0' \
  --with 'pandas==1.5.3' \
  --with 'numpy==1.23.5' \
  task test
```

These overrides leave `pyproject.toml` and `uv.lock` unchanged. Use this workflow
to reproduce a compatibility-matrix failure locally.
The test task invokes `python -m pytest` so the Python interpreter selected by
uv, including temporary dependency overrides, is also used for the tests.

## Prepare and publish a release

1. Update `altair_ally/_version.py` to the final version and update `CHANGELOG.md`
   and version-specific documentation. The version file is also a uv build cache
   key, so syncing refreshes installed distribution metadata when it changes.
2. Run the checks and build the distributions into a version-specific directory:

   ```sh
   uv sync --locked --all-groups
   uv run --locked task test
   uv run --locked --group doc task doc-build --all
   uv build --out-dir dist/0.2.0
   uvx twine check --strict dist/0.2.0/*
   ```

3. Commit and push the release preparation, then confirm that both the tests
   and documentation workflows pass for that commit before tagging it.
4. Create and push the release tag, using the existing unprefixed tag convention:

   ```sh
   git tag -a 0.2.0 -m "Release 0.2.0"
   git push origin 0.2.0
   ```

5. Publish the exact version's wheel and source distribution to PyPI using your
   configured PyPI credentials:

   ```sh
   uv publish dist/0.2.0/altair_ally-0.2.0-py3-none-any.whl dist/0.2.0/altair_ally-0.2.0.tar.gz
   ```

6. Create a GitHub release from that existing tag and attach the same artifacts:

   ```sh
   gh release create 0.2.0 \
     dist/0.2.0/altair_ally-0.2.0-py3-none-any.whl \
     dist/0.2.0/altair_ally-0.2.0.tar.gz \
     --verify-tag --title "0.2.0" --notes-file CHANGELOG.md
   ```

Use the intended version in these commands for later releases. The repository
currently uses manual publication; creating a tag or GitHub release does not
automatically upload distributions to PyPI. A version-specific output directory
keeps older artifacts in `dist/` out of the upload.
