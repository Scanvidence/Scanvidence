## What this does
## Closes
Closes #

## Checklist
- [ ] `pytest -m "not slow"` passes locally
- [ ] `pre-commit run --all-files` passes
- [ ] If this touches data splitting, calibration, or evaluation code:
      `pytest tests/data/test_splitting.py -v` passes explicitly
- [ ] Docstrings added/updated for public functions
- [ ] No dataset files, checkpoints, or tokens in this diff (check `git status`)

## Notes for reviewer
Anything non-obvious, or any negative/null result this PR surfaces.
