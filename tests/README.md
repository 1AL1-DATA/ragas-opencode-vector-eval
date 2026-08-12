# Tests

Unit tests for the shortcut metrics and aggregate builders.

Run with:
```bash
python -m pytest tests/ -v
# or, without pytest:
python -m unittest discover tests/ -v
```

The tests exercise pure functions only and require no data files, so they run
offline with no model downloads.
