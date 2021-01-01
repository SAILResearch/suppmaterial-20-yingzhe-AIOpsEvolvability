import os
import timeit
import argparse
import numpy as np
import pandas as pd
from concept_drift_detection import SlidingWindow, StaticModel, ZTestDetector, GamaDetector, HarelDetector
from utilities import obtain_data, obtain_intervals, obtain_period_data, obtain_metrics


GOOGLE_OUTPUT_FILE = r'concept_google_tuned_'
BACKBLAZE_OUTPUT_FILE = r'concept_disk_tuned_'
N_ROUNDS = 100
MODEL_NAME = ''
DATASET = ''


def experiment_driver(feature_list, label_list, out_file):
    out_columns = ['Scenario', 'Model', 'K', 'Retrain', 'Training Time', 'Testing Time', 'Test P', 'Test R', 'Test A', 'Test F', 'Test AUC', 'Test MCC', 'Test B']
    out_ls = []

    num_periods = len(feature_list)
    print('Total number of periods:', num_periods)
    print('Building models from first', num_periods//2, 'periods')

    models = [
        StaticModel(MODEL_NAME, num_periods//2, DATASET),
        SlidingWindow(MODEL_NAME, num_periods//2, DATASET),
        ZTestDetector(MODEL_NAME, num_periods//2, DATASET),
        GamaDetector(MODEL_NAME, num_periods//2, DATASET),
        HarelDetector(MODEL_NAME, num_periods//2, DATASET)
    ]
    model_preds = [[] for _ in range(len(models))]

    # obtain initial training features and labels
    X_list = feature_list[0: num_periods//2]
    y_list = label_list[0: num_periods//2]

    for idx, model in enumerate(models):
        start_time = timeit.default_timer()
        model.initial_fit(X_list, y_list)
        training_time = timeit.default_timer() - start_time

        start_time = timeit.default_timer()
        y_preds = model.predict_proba(feature_list[num_periods//2])
        testing_time = timeit.default_timer() - start_time
        model_preds[idx].append(y_preds)
        out_ls.append([model.get_name(), MODEL_NAME.upper(), num_periods//2 + 1, model.is_retrained(), training_time, testing_time] + obtain_metrics(label_list[num_periods//2], y_preds))
        
    out_df = pd.DataFrame(out_ls[-len(models):], columns=out_columns)
    out_df.to_csv(out_file, mode='a', index=False, header=(not os.path.isfile(out_file)))
    print()

    for i in range(num_periods//2, num_periods-1):
        print('Fitting models on period', i + 1)
        training_features = feature_list[i]
        training_labels = label_list[i]
        testing_features = feature_list[i + 1]
        testing_labels = label_list[i + 1]
        for idx, model in enumerate(models):
            start_time = timeit.default_timer()
            model.fit(training_features, training_labels)
            training_time = timeit.default_timer() - start_time

            start_time = timeit.default_timer()
            probas = model.predict_proba(testing_features)
            testing_time = timeit.default_timer() - start_time
            model_preds[idx].append(probas)
            out_ls.append([model.get_name(), MODEL_NAME.upper(), i + 2, model.is_retrained(), training_time, testing_time] + obtain_metrics(testing_labels, probas))
        
        out_df = pd.DataFrame(out_ls[-len(models):], columns=out_columns)
        out_df.to_csv(out_file, mode='a', index=False, header=(not os.path.isfile(out_file)))
    print()

    testing_labels = np.hstack(label_list[num_periods//2:])
    #print([[y.shape[0] for y in model_pred] for model_pred in model_preds])
    for idx, model in enumerate(models):
        print('Testing', model.get_name(), 'on whole data')
        out_ls.append([model.get_name(), MODEL_NAME.upper(), -1, False, 0, 0] + obtain_metrics(testing_labels, np.hstack(model_preds[idx])))
    print()

    out_df = pd.DataFrame(out_ls[-len(models):], columns=out_columns)
    out_df.to_csv(out_file, mode='a', index=False, header=(not os.path.isfile(out_file)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Experiment on concept drift detection approaches')
    parser.add_argument("-m", help="specify the model, random forest by default.", required=True, choices=['lr', 'cart', 'rf', 'gbdt', 'nn'])
    parser.add_argument("-d", help="specify the dataset, d for Googole and b for Backblaze.", required=True, choices=['g', 'b'])
    parser.add_argument("-n", help="specify the testing rounds, 100 by default.", default=100)
    args = parser.parse_args()

    N_ROUNDS = int(args.n)
    MODEL_NAME = args.m
    feature_list, label_list = obtain_period_data(args.d)

    if args.d == 'g':
        print('Choose Google as dataset')
        OUTPUT_FILE = GOOGLE_OUTPUT_FILE + args.m + '.csv'
        DATASET = 'Google'
    elif args.d == 'b':
        print('Choose Backblaze as dataset')
        OUTPUT_FILE = BACKBLAZE_OUTPUT_FILE + args.m + '.csv'
        DATASET = 'Backblaze'
    else:
        exit(-1)

    if os.path.isfile(OUTPUT_FILE): 
        os.remove(OUTPUT_FILE)
    print('Output path:', OUTPUT_FILE)

    for _ in range(N_ROUNDS):
        experiment_driver(feature_list, label_list, OUTPUT_FILE)
        
    print('Experiment completed!')
