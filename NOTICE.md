Copyright (2026) Databricks, Inc.

This Software includes software developed at Databricks (https://www.databricks.com/) and its use is subject to the included LICENSE file.
No code in this repository collects usage or tracking information today. If a later version adds usage tracking, this notice will say so, and Databricks will collect and use that information in accordance with its Privacy Notice at https://www.databricks.com/legal/privacynotice.

## Where the data in this repository comes from

This repository contains no data files, no patient personally identifiable information (PII) and
no source patient records, in data or code. Projects either generate synthetic data with their own code or read public datasets
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
  ECG. Patient IDs are sequential numbers in a fixed synthetic population (`SYNPT-000000`,
  `SYNPT-000001`, ...), and each ECG in a batch is given a different one, drawn at random; record
  numbers (`SYN-MRN-...`) are random digits, fixed for each patient ID. Both are marked as
  synthetic. No names or addresses are generated.
- **Request logs.** The two serving endpoints the SETUP job creates log every request and response
  (the ECG signals sent for scoring and the predictions returned) to AI Gateway inference tables
  in the deployer's own catalog and schema, named `<model>_payload` (`bio_ecg12_arr_payload` and
  `bio_ecg12_afib_payload`). The same step turns on AI Gateway usage tracking, which records the
  endpoints' usage in the workspace's `system.serving.endpoint_usage` table. With this project's
  synthetic ECGs those tables hold synthetic data only. Anyone who adapts the project to real data
  should treat the inference tables as holding that data too, and control access to them the same
  way.
