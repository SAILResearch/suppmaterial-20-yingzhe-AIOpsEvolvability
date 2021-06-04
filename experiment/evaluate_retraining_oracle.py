import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utilities import obtain_tuned_model, downsampling
from utilities import obtain_period_data, obtain_metrics


GOOGLE_OUTPUT_FILE = r'oracle_google_tuned_'
BACKBLAZE_OUTPUT_FILE = r'oracle_disk_tuned_'
N_ROUNDS = 100
MODEL_NAME = ''
DATASET = ''
RANDOM_CONST = 114514


def experiment_driver(feature_list, label_list, out_file, n_round):
    out_columns = ['Model', 'Training Period', 'Testing Period', 'Test P', 'Test R', 'Test A', 'Test F', 'Test AUC', 'Test MCC', 'Test B']
    out_ls = []

    num_periods = len(feature_list)
    window_size = num_periods // 2
    print('Total number of periods:', num_periods)
    
    for i in range(window_size, num_periods):
        print(f'Testing on periods: [{i-window_size}: {i}]')
        training_features = np.vstack(feature_list[i-window_size: i])
        training_labels = np.hstack(label_list[i-window_size: i])
        scaler = StandardScaler()
        training_features = scaler.fit_transform(training_features)
        np.random.seed(RANDOM_CONST+n_round*num_periods+i-1)
        training_features, training_labels = downsampling(training_features, training_labels)

        model = obtain_tuned_model(MODEL_NAME, DATASET, i+1, 'w')
        model.fit(training_features, training_labels)

        for j in range(i, num_periods):
            print('Testing on period', j+1)
            testing_features = feature_list[j]
            testing_labels = label_list[j]
            probas = model.predict_proba(scaler.transform(testing_features))[:, 1]
            out_ls.append([MODEL_NAME.upper(), i, j + 1] + obtain_metrics(testing_labels, probas))
    
        out_df = pd.DataFrame(out_ls[-(num_periods-i):], columns=out_columns)
        out_df.to_csv(out_file, mode='a', index=False, header=(not os.path.isfile(out_file)))
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Experiment on the oracle models for concept drift')
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

    for i in range(N_ROUNDS):
        experiment_driver(feature_list, label_list, OUTPUT_FILE, i)
        
    print('Experiment completed!')
