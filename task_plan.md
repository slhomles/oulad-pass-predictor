# Task Plan: Diagnose Fast-Plateau Learning Curve

## Goal
Find data or pipeline reasons why the student-course completion model's learning curve plateaus too quickly.

## Phases
- [x] Phase 1: Initialize investigation notes.
- [x] Phase 2: Map repository structure, datasets, notebooks, and training scripts.
- [x] Phase 3: Inspect target distribution, row counts, missingness, duplicates, and split strategy.
- [x] Phase 4: Check for low feature signal, leakage, preprocessing issues, or train/test mismatch.
- [x] Phase 5: Summarize likely causes and concrete fixes.
- [x] Phase 6: Update preprocessing to use prediction-day cutoff 125 and remove `last_active_day`.
- [x] Phase 7: Regenerate processed feature files and verify schema.
- [x] Phase 8: Update notebook 03 text/assumptions for the new cutoff dataset.
- [x] Phase 9: Switch final prediction cutoff from day 125 to day 90 and use smaller learning-curve train sizes.

## Decisions
- Prefer read-only diagnostics unless a small helper/report file is useful.
- Do not modify training logic unless the cause is obvious and directly requested.
- Superseded decision: prediction was day 125.
- New final user decision: prediction is made after day 90. Only data available up to and including day 90 should be used. Remove `last_active_day` from features. Learning curve should start below 10% train size.

## Errors Encountered
| Error | Attempt | Resolution |
|---|---|---|
| Python notebook source scan hit Windows console `UnicodeEncodeError` | Attempted to print Vietnamese notebook text to default cp1252 stdout | Re-run with Python stdout forced to UTF-8 / replacement-safe output |
| Python notebook verification hit Windows console `UnicodeEncodeError` | Printed notebook cell text containing Vietnamese characters to default cp1252 stdout | Re-run verification with stdout forced to UTF-8 / replacement-safe output |
