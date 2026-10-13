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
   Each changelog release starts with a `## <version>` heading; bracketed versions
   and a date after the version are also supported. Inspect the selected version
   and notes with:

   ```sh
   uv run --locked task release-version
   uv run --locked task release-notes
   ```

2. Run the release checks:

   ```sh
   uv sync --locked --all-groups
   uv run --locked task release-check
   ```

3. Commit and push the release preparation, then confirm that both the tests
   and documentation workflows pass for that commit before tagging it.
4. Create and push the annotated release tag:

   ```sh
   uv run --locked task release-tag
   ```

5. Build the artifacts from the tagged release checkout and upload them using your
   configured PyPI credentials:

   ```sh
   uv run --locked task release-build
   uv run --locked task release-publish
   ```

6. Create a GitHub release from that existing tag and attach the same artifacts:

   ```sh
   uv run --locked task release-github
   ```

All release tasks read the version from `_version.py` and use `dist/<version>/`.
For example, version `0.3.0` produces `dist/0.3.0/altair_ally-0.3.0-py3-none-any.whl`
and `dist/0.3.0/altair_ally-0.3.0.tar.gz`. Upload tasks pass these two exact files;
older distributions are not included. GitHub release notes include only the
matching changelog section, and release candidates are marked as prereleases.

| Task | What it runs |
| --- | --- |
| `release-version` | Show the version from `_version.py` |
| `release-notes` | Show that version's changelog section |
| `release-check` | Check the lockfile, run tests, execute/build docs, build artifacts, and run strict Twine checks |
| `release-build` | Build the versioned wheel/sdist and run strict Twine checks |
| `release-tag` | Create and push the annotated, unprefixed version tag |
| `release-publish` | Upload the tagged release's exact wheel/sdist to PyPI |
| `release-github` | Verify the remote tag, create the GitHub release, and attach artifacts |

Tagging requires a clean tracked working tree. Uploads also require HEAD to match
the local version tag and the artifacts to exist. These checks preserve the
release commit selected after CI passed. Version edits, committing the release
preparation, and waiting for CI are the manual steps between the tasks.

To preview commands without executing them:

```sh
uv run --locked task release-check --dry-run
uv run --locked task release-tag --dry-run
uv run --locked task release-publish --dry-run
uv run --locked task release-github --dry-run
```

Build and upload tasks also accept `--output-dir PATH` to override the default
artifact directory; use the same directory for subsequent upload tasks.
`release-check` runs local validation and builds artifacts. Publishing happens
when you explicitly run the tagging/upload tasks; creating a GitHub release
does not automatically upload anything to PyPI.
