Copyright (2025) Databricks, Inc.

This Software includes software developed at Databricks (https://www.databricks.com/) and its use is subject to the included LICENSE file.
By using this repository and the notebooks within, you consent to Databricks collection and use of usage and tracking information in accordance with our privacy policy at www.databricks/privacypolicy.

## Where the data in this repository comes from

This repository contains no data files and no personally identifiable information (PII), in data
or code. Projects either generate synthetic data with their own code or read public datasets
that are de-identified, used under their license and attributed to their source, when their
notebooks run, and write the results to tables in the user's own Unity Catalog catalog. Each
project's section below says where its data comes from and, for a public dataset, its source and
license; today every project generates synthetic data.

`projects/ecg12-afib`:

- **ECG waveforms** come from `common/ecg_signal_gen.py`. Each 12-lead ECG is 10 seconds at
  250 Hz. Leads I, II and V1-V6 are synthesized from a sum of Gaussian P, Q, R, S and T waves
  (the wave shape of McSharry et al. 2003, "A dynamical model for generating synthetic ECG
  signals"); leads III, aVR, aVL and aVF are computed from I and II with the standard
  formulas. Three rhythm classes are produced: normal sinus rhythm, atrial fibrillation and
  sinus tachycardia. The neurokit2 package adds recording noise.
- **Patient records** come from `common/patient_info.py`. Age, sex, height, weight, blood
  pressure and acquisition time are random draws; heart rate is measured from the generated
  ECG. Patient IDs (`SYNPT-...`) and record numbers (`SYN-MRN-...`) are random and marked as
  synthetic. No names or addresses are generated.
