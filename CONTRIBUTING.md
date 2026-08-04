# Contributing

Thanks for improving `ds-crawler`. Small, focused changes with tests are the
easiest to review.

## Set up a development environment

The project uses Python 3.9 or newer and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/d-rothen/ds-crawler.git
cd ds-crawler
uv sync --extra dev
```

## Before opening a pull request

Run the same core checks as CI:

```bash
uv run ruff check .
uv run pytest
uv build
uv run python scripts/verify_distribution.py dist/*
```

When changing modality metadata, regenerate the checked-in schema and run its
test:

```bash
uv run build-meta-schema
uv run pytest tests/test_meta_schema.py
```

Please include a regression test for behavior changes and update the README or
configuration reference when public behavior changes. Avoid committing dataset
archives, generated build output, local paths, credentials, or editor state.

## Reporting bugs

Open a GitHub issue with a minimal configuration, representative relative file
paths, the expected result, and the full error. Remove private dataset paths or
metadata before posting.

Security issues follow a separate private process described in
[SECURITY.md](SECURITY.md).
