# AIOps Evolution - Supplemental Materials
This repository contains the replication package for paper "When and How Should We Maintain AIOps Models Under Data Evolution? An Exploratory Study".

## Introduction
We organize the replication package as follow:
1. Preprocessing: code in this section is for extract data from raw datasets and hyperparameter tuning;
2. Experiment: code in this section is for our main experiment (e.g., evaluate concept drift detection methods and time-based ensemble approaches);
3. Analysis: code in this section is for analyze the dataset and experiment results.

Our experiment is carried on using the following packages and versions:
- Python: 3.8.3
- Numpy: 1.18.5
- Scipy: 1.5.0
- Pandas: 1.0.5
- Sklearn: 0.0
- Mkl: 2019.0
- Statsmodels: 0.11.1
- Xgboost: 1.2.1

We recommend use an `Anaconda` enviorment with Python version 3.8.3, then all packages except `xgboost` and `statsmodels` would be ready.

## Preprocessing
This part contains code and materials for preprocessing the dataset. All code could be found under the `preprocessing` folder.

### Prepare dataset
Since our datasets are extremely big, we only offer the code for extract samples from the Backblaze dataset.
To obtain the samples we used for the Backblaze data, please follow the following steps:
1. To obtain the raw dataset, please download from the [Backblaze website](https://www.backblaze.com/b2/hard-drive-test-data.html). 
   In our experiment, we use the disk trace data from 2015 to 2017 (including ZIP file for `2015 Data`, `2016 Data, Q1` to `2016 Data, Q4`, and `2017 Data, Q1` to `2017 Data, Q4`).
2. Unzip all the raw data files and place all the CSV files (should be in the format `YYYY-MM-DD.csv`) in one single folder.
3. Update the `folder` variable at the head of `preprocess_disk.py` file, then execute the same PY file to extract samples. 
   It could take a while and the output file would take several Gigabytes, please be prepared.

For the Google cluster trace dataset, you could download the prepared CSV file (`google_job_failure.csv`) from the release page of this project or build the same file following similar procedures to the Backblaze dataset using `preprocess_google.py` file. You can find the raw cluster data [here](https://github.com/google/cluster-data/blob/master/ClusterData2011_2.md).

### Hyperparemter tuning
We tune the hyperparameter on each sliding window and each single period in advance to save time when repeating our experiment 100 times.
The hyperparameter setting can be searched through executing the `tune_hyperparameter.py` file on each model and each datasets (see Section 3: Experiment for details about our command line arguments). 
For each hyperparameter tuning type, please merge the hyperparameter into `parameter_list_period.csv` and `parameter_list_window.csv` and place under the same folder of experiment code.
We also provide our version of tuned hyperparameters in the `preprocessing` folder.

## Experiments
This part contains code for our main experiment of evaluating various model updating approaches. All code could be found under the `experiment` folder.

For all of our experiment code, we have the following command line argument:
1. `-d` is a **required** parameter for choosing the dataset. Two choices are available: `g` for the Google dataset, `b` for the Backblaze dataset.
2. `-m` is a **required** parameter for choosing the model. Five choices are available: `lr, cart, rf, gbdt, nn`. Please note that the argument should be all *lowercase* letters.
3. `-n` is an optional parameter for the repetition time of the experiments. The default value is 100 runs, which is also the same iteration we used in our paper.

Please note that some experiments could take prolonged time to finish, so we recommend using `GNU Screen` or `nohup` to execute on a server.

We have the following experiment code available:
- `predict_concept_drift.py` contains code for testing approaches described in RQ2 (i.e., Static, Retrain, DDM, PERM, and Z-test). Note that code in this file relied on the `concept_drift_detection.py` file in the same folder.
- `predict_ensemble_methods.py` contains code for testing time-based ensemble approaches (i.e., AWE and SEA). Note that code in this file relies on the `ensemble_model.py` file in the same folder.
- `predict_oracle.py` contains code for the oracle approach used in RQ2.

## Analysis
This part contains code for the analysis of our datasets and experiment results. All code could be found under the `analysis` folder.
We have the following code available:
- `analyze_correlation.py` analyze the changes of dependent and independent variables in RQ1.
- `analyze_detection_judgement.py` analyze the performance of concept drift detection in RQ2.
- `scripy.R` contains code for plotting in the paper.
