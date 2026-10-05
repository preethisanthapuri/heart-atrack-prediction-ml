# External validation data

`uci_heart_disease_hungarian.data` is a local-only copy of the processed Hungarian site file from the UCI Heart Disease collection. It is ignored by Git because it contains row-level health data. The UCI repository documents four distinct sites, the 14 commonly used variables, and the angiographic target (`num`: 0=no disease; 1–4=disease present). The historical source file is available at:

`https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data`

UCI dataset DOI: `10.24432/C52P4X`. Cite Janosi, A., Steinbrunn, W., Pfisterer, M., and Detrano, R. (1989), *Heart Disease*, UCI Machine Learning Repository. UCI lists a CC BY 4.0 license. A public mirror was used to retrieve the file in this environment; the SHA-256 of the local copy is recorded in `results/metrics/external_validation.json`.

Run `python run_external_validation.py` after obtaining the file. The fixed model and preprocessing artifacts are not refit. The evaluator applies explicit code mappings and the saved primary-training imputer.
