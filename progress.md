# Progress Log

## 2026-05-12
- Created planning files for diagnosing why the learning curve plateaus early.
- Mapped top-level repo structure and identified ML code, notebooks, reports, and local data directory.
- Read backend training/preprocessing/pipeline code and located learning-curve logic in notebook 03 / `_build_notebook_03.py`.
- Confirmed current processed feature parquet files are missing from `data/processed/`; only comparison CSVs remain.
- Viewed saved learning-curve and target-distribution figures, then inspected notebook 01 feature-engineering cells.
- User added processed data files; confirmed `X.parquet`, `y.parquet`, metadata, samples, and best-param JSON files are now present.
- Ran processed-data diagnostics: target balance, missingness, duplicates, single-feature AUCs, feature-group CV, holdout split, student overlap, and RF grouped feature importances.
- Started dataset correction: prediction cutoff is day 125 and `last_active_day` must be removed.
- Updated notebook 01 preprocessing cells for cutoff day 125 and updated notebook 03 metadata/feature-count notes.
- Regenerated `data/processed/X.parquet`, `y.parquet`, `X_sample.csv`, and `feature_metadata.json` from raw CSVs using the day-125 cutoff.
- Verified new processed schema, backend preprocessor compatibility, notebook 03 builder syntax, and cleared stale notebook outputs from notebooks 01 and 03.
- Ran quick day-125 diagnostics: full-feature RF CV around F1 0.861; shuffled-stratified learning curve starts around CV F1 0.842 and ends around 0.857, so it no longer plateaus above 0.90 from the first point.
- Updated notebook 03 learning-curve CV to `StratifiedKFold(n_splits=3, shuffle=True, random_state=RNG)` for consistency with the main CV.
- Compared cutoff days 30, 60, 90, and 125 with RF learning curves using train sizes 2%-100%; smaller cutoffs lower the curve and make the upward slope more visible.
- Switched final preprocessing cutoff to day 90, regenerated processed parquet/metadata/sample files, verified backend preprocessor compatibility, and updated notebook 03 `LC_TRAIN_SIZES` to start at 2% and 5%.
