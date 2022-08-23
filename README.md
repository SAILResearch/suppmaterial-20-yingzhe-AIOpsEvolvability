# AIOps Evolution - Supplemental Materials
This repository contains the replication package for the paper "Assessing the Maturity of Model Maintenance Techniques for AIOps Solutions".

## Introduction
We organize the replication package into four file folders.
1. Data preprocessing: this folder contains code for extracting data from raw datasets and hyperparameter tuning;
2. Experiments: this folder contains code for our main experiment (i.e., evaluating concept drift detection methods and time-based ensemble approaches);
4. Results data: this folder contains the results for our paper, including the results for metrics other than the AUC metric used in our paper;
3. Results analysis: this folder contains code for analyzing the dataset and experiment results.

Our experiment code is using the following packages and versions:

- Python: 3.8.3
- Numpy: 1.18.5
- Scipy: 1.5.0
- Pandas: 1.0.5
- Sklearn: 0.0
- Mkl: 2019.0
- Statsmodels: 0.11.1
- Xgboost: 1.2.1
- Scikit-multiflow: 0.5.3

We recommend using an [Anaconda](https://docs.anaconda.com/anaconda/install/) environment with Python version 3.8.3, then install the `xgboost`, `statsmodels`, and `skmultiflow` packages using `pip`, then every Python requirement should be met.

## Data preprocessing
This part contains code and materials for preprocessing the dataset. All code could be found under the `preprocessing` folder.

### Prepare dataset
We offer two approaches to prepare the data files: 1) download the data files provided by us on the [release page](https://github.com/SAILResearch/suppmaterial-20-yingzhe-AIOpsEvolvability/releases/); 2) or build the data files by yourself from the raw dataset and the preprocessing code provided by us.

#### Build the Google data file
For the Google cluster trace dataset, you could also download the prepared CSV file (`google_job_failure.zip`) from this project's release page.
Otherwise, you could build the same file following similar procedures to the Backblaze dataset using the `preprocess_google.py` file. You can find the raw cluster data [here](https://github.com/google/cluster-data/blob/master/ClusterData2011_2.md).

#### Build the Backblaze data file
You could find the zipped CSV file (`disk_failure_v2.zip`) for the Backblaze disk trace dataset. Unzip and place under the same folder as the experiment code files would work.
Otherwise, you could build the raw data file following the following steps:
1. Download the raw dataset from the [Backblaze website](https://www.backblaze.com/b2/hard-drive-test-data.html). 
   In our experiment, we use the disk stats data from 2015 to 2017 (including ZIP file for `2015 Data`, `2016 Data, Q1` to `2016 Data, Q4`, and `2017 Data, Q1` to `2017 Data, Q4`).
2. Unzip all the raw data files and place all the CSV files (should be in the format `YYYY-MM-DD.csv`) in a single folder.
3. Update the `folder` variable at the head of the `preprocess_backblaze.py` file to where you store the CSV files, then execute the same Python file to extract samples. 
   Please be advised that it could take a while, and the output file would take several Gigabytes.

#### Build the Alibaba data file
You could also find the prepared CSV file for this dataset in the release page. 
Otherwise, you may build the same data file with our preprocessing script using the following steps:
1. Download the Alibaba GPU cluster following the instruction on the [Alibaba GPU trace data page](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2020).
2. Specify the `data` folder of the downloaded raw data in the `preprocess_alibaba.py` file then execute.

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
4. `-s` is an optional parameter for the starting round of the experiment. It would be useful if you would like to resume experiment from a specific round.

As some experiments could take a prolonged time to finish, we recommend executing them on a server with tools like `GNU Screen` or `nohup`. An example of evaluating the ensemble approaches on the `Google` data set and `RF` model in `100` iteration with `nohup` in the `background` and dump the command line output to `log.out` would be: `nohup python -u evaluate_ensemble_approaches.py -d g -m rf -n 100 > log.out 2>&1 &`.

We have the following experiment code available:
- `evaluate_retraining_approaches.py` contains code for testing approaches described in RQ2 (i.e., Static, Retrain, DDM, PERM, and Z-test). Note that the code in this file relied on the `drift_retrain_model.py` and `utilities.py` files in the same folder.
- `evaluate_ensemble_approaches.py` contains code for testing time-based ensemble approaches (i.e., AWE and SEA). Note that the code in this file relies on the `ensemble_model.py` and `utilities.py` files in the same folder.
- `evaluate_online_approaches.py` contains code for the online learning approaches. Note that the code in this file relies on the `online_models.py` and `utilities.py` files in the same folder. Also note that this part of code use multi-process to boost the execution. You can change the number of processes in the code file.
- `evaluate_retraining_oracle.py` contains code for the oracle approach used in the Discussion section.

The remained Python files contains implementation for models we experimented, including `drift_retrain_model.py`, `ensemble_model.py`, and `online_model.py`, also, `utilities.py` contains auxiliary functions for the experiment code. Please keep these files in the same folder.

## Experiment Results Data
This part contains the output data from our main experiments. All output CSV files could be found under the `results` folder.
We organize the CSV files into two folders. The `preliminary_results` folder contains files for the Preliminary Study section, while the `experiment_results` folder contains files for our main results, we only provide the CSV files after combining separate files together for simplicity.

## Results Analysis
This part contains code for the analysis of our datasets and experiment results. All code could be found under the `analysis` folder.
Before doing any other remained analysis, please first combine the separate CSV files together using `combine_result_files.py` first if you start the experiment on your own (you don't need this step if using the result files we provided).

We have the following code available:
- `combine_result_files.py` combine the separate CSV result files. You **don't** need to do this if using the result files we provided.
- `analyze_correlation.py` analyze the changes in dependent and independent variables in the Preliminary Study section.
- `analyze_detection_judgement.py` analyze the performance of concept drift detection in the Discussion section.
- `result_analysis.R` contains code for plotting results figures and tables in our main experiment results.
