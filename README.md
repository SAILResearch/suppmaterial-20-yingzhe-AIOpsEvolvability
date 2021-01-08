# AIOps Evolution - Supplemental Materials
This repository contains the replication package for the paper "When and How Should We Maintain AIOps Models Under Data Evolution? An Exploratory Study".

## Introduction
We organize the replication package into four file folders.
1. Data preprocessing: this folder contains code for extracting data from raw datasets and hyperparameter tuning;
2. Experiments: this folder contains code for our main experiment (i.e., evaluating concept drift detection methods and time-based ensemble approaches);
4. Results data: this folder contains the results for our paper, including the results for metrics other than the AUC metric used in our paper;
3. Results analysis: this folder contains code for analyzing the dataset and experiment results.

Our code is based the following packages and versions:

- Python: 3.8.3
- R: 3.6.3
- Numpy: 1.18.5
- Scipy: 1.5.0
- Pandas: 1.0.5
- Sklearn: 0.0
- Mkl: 2019.0
- Statsmodels: 0.11.1
- Xgboost: 1.2.1

We recommend using an [Anaconda](https://docs.anaconda.com/anaconda/install/) environment with Python version 3.8.3, then install the `xgboost` and `statsmodels` packages using `pip`, then every Python requirement should be met.

## Data preprocessing
This part contains code and materials for preprocessing the dataset. All code could be found under the `preprocessing` folder.

### Prepare dataset
We offer two approaches to prepare the data files: 1) download the data files provided by us on the [release page](https://github.com/SAILResearch/suppmaterial-20-yingzhe-AIOpsEvolvability/releases/); 2) or build the data files by yourself from the raw dataset and the preprocessing code provided by us.

#### Build the Backblaze data file
You could find the zipped CSV file (`disk_failure_v2.zip`) for the Backblaze disk trace dataset. Unzip and place under the same folder as the experiment code files would work.
Otherwise, you could build the raw data file following the following steps:
1. Download the raw dataset from the [Backblaze website](https://www.backblaze.com/b2/hard-drive-test-data.html). 
   In our experiment, we use the disk stats data from 2015 to 2017 (including ZIP file for `2015 Data`, `2016 Data, Q1` to `2016 Data, Q4`, and `2017 Data, Q1` to `2017 Data, Q4`).
2. Unzip all the raw data files and place all the CSV files (should be in the format `YYYY-MM-DD.csv`) in a single folder.
3. Update the `folder` variable at the head of the `preprocess_disk.py` file to where you store the CSV files, then execute the same Python file to extract samples. 
   Please be advised that it could take a while, and the output file would take several Gigabytes.

#### Build the Google data file
For the Google cluster trace dataset, you could also download the prepared CSV file (`google_job_failure.zip`) from this project's release page.
Otherwise, you could build the same file following similar procedures to the Backblaze dataset using the `preprocess_google.py` file. You can find the raw cluster data [here](https://github.com/google/cluster-data/blob/master/ClusterData2011_2.md).

### Hyperparemter tuning
We tune the hyperparameter on each sliding window and each single period in advance to save time when repeating our experiment 100 times.
The hyperparameter setting can be searched through executing the `tune_hyperparameter.py` file on each model and each datasets (see Section 3: Experiment for details about our command line arguments). 
For each hyperparameter tuning type, please merge the hyperparameter into `parameter_list_period.csv` and `parameter_list_window.csv` and place under the same folder of experiment code.
We also provide our version of tuned hyperparameters in the `preprocessing` folder.

## Experiments
This part contains code for our main experiment in evaluating various model updating approaches. All code could be found under the `experiment` folder.

The experiment code accepts the following command-line arguments to select model, dataset, and iteration rounds for maximum flexibility.
1. `-d` is a **required** parameter for choosing the dataset. Two choices are available: `g` for the Google dataset, `b` for the Backblaze dataset.
2. `-m` is a **required** parameter for choosing the model. Five choices are available: `lr`, `cart`, `rf`, `gbdt`, and `nn`. Please note that the argument should be all *lowercase* letters.
3. `-n` is an optional parameter for the repetition time of the experiments. The default value is 100 iterations, which is also the same iteration number we used in our paper.

As some experiments could take a prolonged time to finish, we recommend executing them on a server with tools like `GNU Screen` or `nohup`. An example of evaluating the ensemble approaches on the `Google` data set and `RF` model in `100` iteration with `nohup` in the `background` and dump the command line output to `output.out` would be: `nohup python -u evaluate_ensemble_approaches.py -d g -m rf -n 100 > output.out 2>&1 &`.

We have the following experiment code available:
- `evaluate_drift_detection_methods.py` contains code for testing approaches described in RQ2 (i.e., Static, Retrain, DDM, PERM, and Z-test). Note that the code in this file relied on the `concept_drift_detection.py` file in the same folder.
- `evaluate_ensemble_approaches.py` contains code for testing time-based ensemble approaches (i.e., AWE and SEA). Note that the code in this file relies on the `ensemble_model.py` file in the same folder.
- `evaluate_drift_detection_oracle.py` contains code for the oracle approach used in RQ2.

## Experiment Results Data
This part contains the output data from our main experiments. All output CSV files could be found under the `results` folder.
We organize the CSV files by which research question they belong. For example, the `rq1_results` subfolder contains CSV files used in the analysis of RQ1.

## Results Analysis
This part contains code for the analysis of our datasets and experiment results. All code could be found under the `analysis` folder.
We have the following code available:
- `analyze_correlation.py` analyze the changes in dependent and independent variables in RQ1.
- `analyze_detection_judgement.py` analyze the performance of concept drift detection in RQ2.
- `plot_figures.R` contains code for illustrating results figures in the paper.
