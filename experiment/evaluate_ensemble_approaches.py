import os
import timeit
import argparse
import numpy as np
import pandas as pd
from utilities import obtain_period_data, obtain_metrics
from ensemble_model import SEAModel, AWEModel


GOOGLE_OUTPUT_FILE = r'ensemble_google_'
BACKBLAZE_OUTPUT_FILE = r'ensemble_backblaze_'
ALIBABA_OUTPUT_FILE = r'ensemble_alibaba_'
N_ROUNDS = 100
MODEL_NAME = ''
DATASET = ''
RANDOM_CONST = 114514


def experiment_driver(feature_list, label_list, out_file, n_round):
    out_columns = ['Scenario', 'Model', 'Round', 'Testing Period', 'Retrain', 'Training Time', 'Testing Time', 'Test P', 'Test R', 'Test A', 'Test F', 'Test AUC', 'Test MCC', 'Test B']
    out_ls = []
    
    num_periods = len(feature_list)
    print('Total number of periods:', num_periods)

    models = [
        SEAModel(MODEL_NAME, num_periods//2, DATASET),
        AWEModel(MODEL_NAME, num_periods//2, DATASET)
    ]
    model_probas_list = [[] for _ in range(len(models))]

    # iterate through each time period
    for i in range(num_periods - 1):
        print('Fitting models on period', i + 1)
        training_features = feature_list[i]
        training_labels = label_list[i]
        testing_features = feature_list[i + 1]
        testing_labels = label_list[i + 1]

        for idx, model in enumerate(models):
            np.random.seed(RANDOM_CONST+n_round*num_periods+i)
            start_time = timeit.default_timer()
            model.fit(training_features, training_labels, i+1)
            training_time = timeit.default_timer() - start_time

            # we don't predict on the first half of data
            if i < num_periods//2 - 1:
                continue

            start_time = timeit.default_timer()
            testing_probas = model.predict_proba(testing_features)[:, 1]
            testing_time = timeit.default_timer() - start_time
            model_probas_list[idx].append(testing_probas)
            out_ls.append([model.get_name(), MODEL_NAME.upper(), n_round, i + 2, model.is_added(), training_time, testing_time] + obtain_metrics(testing_labels, testing_probas))
    
        out_df = pd.DataFrame(out_ls[-len(models):], columns=out_columns)
        out_df.to_csv(out_file, mode='a', index=False, header=(not os.path.isfile(out_file)))

    testing_labels = np.hstack(label_list[num_periods//2:])
    for idx, model in enumerate(models):
        print('Testing', model.get_name(), 'on whole data')
        model_records = out_ls[idx:len(out_ls):len(models)]
        out_ls.append([model.get_name(), MODEL_NAME.upper(), n_round, -1, False, np.sum([row[5] for row in model_records]), np.sum([row[6] for row in model_records])] + 
                       obtain_metrics(testing_labels, np.hstack(model_probas_list[idx])))
    print()
    out_df = pd.DataFrame(out_ls[-len(models):], columns=out_columns)
    out_df.to_csv(out_file, mode='a', index=False, header=(not os.path.isfile(out_file)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Experiment on time-based ensemble models')
    parser.add_argument("-m", help="specify the model, random forest by default.", required=True, choices=['lr', 'cart', 'rf', 'gbdt', 'nn'])
    parser.add_argument("-d", help="specify the dataset, g=Google, b=Backblaze, and a=Alibaba.", required=True, choices=['g', 'b', 'a'])
    parser.add_argument("-n", help="specify the testing rounds, 100 by default.", default=100)
    parser.add_argument("-s", help="starting from this iteration.")
    args = parser.parse_args()

    N_ROUNDS = int(args.n)
    MODEL_NAME = args.m
    feature_list, label_list = obtain_period_data(args.d)
    start_round = 0
    if args.s != None:
        start_round = int(args.s)

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

    for i in range(start_round, start_round+N_ROUNDS):
        print('Round', i+1)
        experiment_driver(feature_list, label_list, OUTPUT_FILE, i)
        
    print('Experiment completed!')
