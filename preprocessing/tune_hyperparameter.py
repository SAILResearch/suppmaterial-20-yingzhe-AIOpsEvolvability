import argparse
import timeit
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from utilities import obtain_param_dist, obtain_raw_model, obtain_period_data, downsampling
import warnings


DATASET = ''
MODEL_NAME = ''
N_ITER_SEARCH = 100


def experiment_driver(feature_list, label_list, out_file, period_type):
    out_columns = ['Dataset', 'Model', 'Period', 'AUC', 'Time', 'Hyper']
    out_ls = []

    num_periods = len(feature_list)
    window_size = num_periods//2

    if period_type == 'p':
        searching_range = range(num_periods)
    else:
        searching_range = range(num_periods//2, num_periods)

    for i in searching_range:
        if period_type == 'p':
            print('Searching on period', i + 1)
            features = feature_list[i]
            labels = label_list[i]
        else:
            print('Search on period', i-window_size+1, 'to', i)
            features = np.vstack(feature_list[i-window_size: i])
            labels = np.hstack(label_list[i-window_size: i])

        scaler = StandardScaler()
        model = obtain_raw_model(MODEL_NAME)
        random_search = RandomizedSearchCV(model, param_distributions=obtain_param_dist(MODEL_NAME), n_iter=N_ITER_SEARCH, scoring='roc_auc', cv=4)

        start_time = timeit.default_timer()
        features = scaler.fit_transform(features)
        features, labels = downsampling(features, labels)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            random_search.fit(features, labels)
        elapse_time = timeit.default_timer() - start_time

        print("Best parameters (AUC {0}): {1}".format(random_search.best_score_, random_search.best_params_))
        out_ls.append((DATASET, MODEL_NAME, i+1, random_search.best_score_, elapse_time, str(random_search.best_params_)))

        out_df = pd.DataFrame(out_ls[-1:], columns=out_columns)
        out_df.to_csv(out_file, mode='a', index=False, header=(not os.path.isfile(out_file)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Experiment on hyperparameter searching')
    parser.add_argument("-m", help="specify the model, random forest by default.", required=True, choices=['lr', 'cart', 'rf', 'gbdt', 'nn'])
    parser.add_argument("-d", help="specify the dataset, g=Google, b=Backblaze, and a=Alibaba.", required=True, choices=['g', 'b', 'a'])
    parser.add_argument("-n", help="specify the iteration rounds, 100 by default.", default=100)
    parser.add_argument("-t", help="specify the type, p for period and w for window.", required=True, choices=['p', 'w'])
    args = parser.parse_args()

    N_ITER_SEARCH = int(args.n)
    MODEL_NAME = args.m
    feature_list, label_list = obtain_period_data(args.d)

    if args.t == 'p':
        GOOGLE_OUTPUT_FILE = r'parameter_list_google_period_'
        BACKBLAZE_OUTPUT_FILE = r'parameter_list_backblaze_period_'
        ALIBABA_OUTPUT_FILE = r'parameter_list_alibaba_period_'
    elif args.t == 'w':
        GOOGLE_OUTPUT_FILE = r'parameter_list_google_window_'
        BACKBLAZE_OUTPUT_FILE = r'parameter_list_backblaze_window_'
        ALIBABA_OUTPUT_FILE = r'parameter_list_alibaba_window_'
    else:
        exit(-1)

    if args.d == 'g':
        print('Choose Google as dataset')
        OUTPUT_FILE = GOOGLE_OUTPUT_FILE + args.m + '.csv'
        DATASET = 'Google'
    elif args.d == 'b':
        print('Choose Backblaze as dataset')
        OUTPUT_FILE = BACKBLAZE_OUTPUT_FILE + args.m + '.csv'
        DATASET = 'Backblaze'
    elif args.d == 'a':
        print('Choose Alibaba as dataset')
        OUTPUT_FILE = ALIBABA_OUTPUT_FILE + args.m + '.csv'
        DATASET = 'Alibaba'
    else:
        exit(-1)

    if os.path.isfile(OUTPUT_FILE): 
        os.remove(OUTPUT_FILE)
    print('Output path:', OUTPUT_FILE)

    print(f'Test objective: {MODEL_NAME} model on dataset {DATASET}.')
    experiment_driver(feature_list, label_list, OUTPUT_FILE, args.t)

    print('Experiment completed!')
