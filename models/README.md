# Local preprocessing artifacts

`preprocessing_pipeline.joblib` is created by `python run_preprocessing.py`. It contains transformations fitted only on the training split and is intentionally excluded from Git because it is data-derived. Rebuild it from the local cleaned dataset with the documented command.
