# Contributing

Use Python 3.9 or newer, install the development dependencies, and run both checks before opening a
pull request:

```bash
pytest
ruff check src tests
```

Changes to ranking, candidate-pool construction, centering, or metrics must include a regression
test. Do not commit datasets, generated embeddings, access tokens, local absolute paths, or model
weights.
