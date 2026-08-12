# Contributing

Thanks for considering a contribution. This is a small, deliberately
content-free research artifact — the guidelines reflect that.

## Ground rules

- **No conversation content.** Do not commit per-sample rows, user inputs,
  responses, references, or retrieved chunks. The dataset and RAGAS outputs
  live in the private eval workspace; this repository ships aggregates,
  figures, and code only.
- **Keep it reproducible.** Any claim you add must be backed by a number in
  `results/` and a script in `src/` that can regenerate it.
- **Tests stay offline.** New pure functions go into `src/` with unit tests
  in `tests/` that require no models and no data.

## Workflow

1. Fork the repo and create a branch.
2. Make your change; add or update tests.
3. Run the test suite (no network, no models):
   ```bash
   python -m unittest discover tests/ -v
   ```
4. If you changed aggregates, regenerate them and update `results/SUMMARY.md`:
   ```bash
   python -m src.build_aggregates
   ```
5. Open a pull request describing the change and how it was verified.

## License

By contributing you agree that your contributions are licensed under the
project's dual licence: MIT for code (`src/`, `tests/`, `data/download.sh`),
CC-BY-4.0 for prose and figures. Add yourself to `AUTHORS`.
