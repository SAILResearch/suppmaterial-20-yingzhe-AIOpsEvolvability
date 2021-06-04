import pandas as pd
import numpy as np


datasets = ['google', 'disk']
models = ['lr', 'cart', 'rf', 'nn', 'gbdt']
name_datasets = {
    'google': 'Google',
    'disk': 'Backblaze'
}
n_periods_dataset = {
    'google': 28,
    'disk': 36
}


def combine_retrain():
    df = None
    for dataset in datasets:
        for model in models:
            df_retrain = pd.read_csv('retrain_'+dataset+'_'+model+'.csv')
            df_retrain['Dataset'] = name_datasets[dataset]
            df_retrain = df_retrain[['Dataset', 'Scenario', 'Model', 'Round', 'Testing Period', 'Retrain', 'Training Time', 'Testing Time', 
                'Test P', 'Test R', 'Test A', 'Test F', 'Test AUC', 'Test MCC', 'Test B']]

            if df is None:
                df = df_retrain
            else:
                df = pd.concat([df, df_retrain])

    window_df = pd.read_csv('parameter_list_window.csv')
    period_df = pd.read_csv('parameter_list_period.csv')

    for dataset in datasets:
        for model in models:
            # add tuning time to all the models
            tune_time = window_df[np.logical_and.reduce((
                window_df['Dataset'] == name_datasets[dataset], 
                window_df['Model'] == model, 
                window_df['Period'] == n_periods_dataset[dataset]//2+1))]['Time']
            df.loc[np.logical_and.reduce((
                df['Dataset'] == name_datasets[dataset], 
                df['Model'] == model.upper(), 
                df['Testing Period'] == n_periods_dataset[dataset]//2+1)), 'Training Time'] += float(tune_time)

            for i in range(n_periods_dataset[dataset]//2+2, n_periods_dataset[dataset]+1):
                tune_time = window_df[np.logical_and.reduce((
                    window_df['Dataset'] == name_datasets[dataset], 
                    window_df['Model'] == model, 
                    window_df['Period'] == i))]['Time']

                # always add tuning time for retrain model
                df.loc[np.logical_and.reduce((
                    df['Dataset'] == name_datasets[dataset], 
                    df['Model'] == model.upper(), 
                    df['Scenario'] == 'Retrain',
                    df['Testing Period'] == i)), 'Training Time'] += float(tune_time)

                # add tuning time for concept drift detection models if they updated
                df.loc[np.logical_and.reduce((
                    df['Dataset'] == name_datasets[dataset], 
                    df['Model'] == model.upper(),
                    df['Scenario'] != 'Retrain',
                    df['Scenario'] != 'Stationary',
                    df['Retrain'],
                    df['Testing Period'] == i)), 'Training Time'] += float(tune_time)
                
                # add evaluation tuning for the PERM model
                tune_time = period_df[np.logical_and.reduce((
                    period_df['Dataset'] == name_datasets[dataset], 
                    period_df['Model'] == model, 
                    period_df['Period'] == i-2))]['Time']
                df.loc[np.logical_and.reduce((
                    df['Dataset'] == name_datasets[dataset], 
                    df['Model'] == model.upper(),
                    df['Scenario'] == 'PERM',
                    df['Testing Period'] == i)), 'Training Time'] += float(tune_time)

    df.to_csv('retrain_model_results.csv', index=False)


def combine_ensemble():
    df = None
    for dataset in datasets:
        for model in models:
            df_ensemble = pd.read_csv('ensemble_'+dataset+'_'+model+'.csv')
            df_ensemble['Dataset'] = name_datasets[dataset]
            df_ensemble['Retrain'] = True
            df_ensemble.loc[df_ensemble['Testing Period'] <= n_periods_dataset[dataset]//2+1,'Retrain'] = False
            df_ensemble = df_ensemble[['Dataset', 'Scenario', 'Model', 'Round', 'Testing Period', 'Retrain', 'Training Time', 'Testing Time', 
                'Test P', 'Test R', 'Test A', 'Test F', 'Test AUC', 'Test MCC', 'Test B']]

            if df is None:
                df = df_ensemble
            else:
                df = pd.concat([df, df_ensemble])
                
    period_df = pd.read_csv('parameter_list_period.csv')

    for dataset in datasets:
        for model in models:
            for i in range(1, n_periods_dataset[dataset]//2+1):
                tune_time = period_df[np.logical_and.reduce((
                    period_df['Dataset'] == name_datasets[dataset], 
                    period_df['Model'] == model, 
                    period_df['Period'] == i))]['Time']
                df.loc[np.logical_and.reduce((
                    df['Dataset'] == name_datasets[dataset], 
                    df['Model'] == model.upper(), 
                    df['Testing Period'] == n_periods_dataset[dataset]//2+1)), 'Training Time'] += float(tune_time)

            for i in range(n_periods_dataset[dataset]//2+2, n_periods_dataset[dataset]+1):
                tune_time = period_df[np.logical_and.reduce((
                    period_df['Dataset'] == name_datasets[dataset], 
                    period_df['Model'] == model, 
                    period_df['Period'] == i-1))]['Time']
                df.loc[np.logical_and.reduce((
                    df['Dataset'] == name_datasets[dataset], 
                    df['Model'] == model.upper(),
                    df['Testing Period'] == i)), 'Training Time'] += float(tune_time)

    df.to_csv('ensemble_model_results.csv', index=False)


def combine_online():
    df = None
    for dataset in datasets:
        df_online = pd.read_csv('online_'+dataset+'.csv')
        df_online['Dataset'] = name_datasets[dataset]
        df_online['Model'] = 'Online'
        df_online['Retrain'] = True
        df_online.loc[df_online['Testing Period'] <= n_periods_dataset[dataset]//2+1, 'Retrain'] = False
        df_online = df_online[['Dataset', 'Scenario', 'Model', 'Round', 'Testing Period', 'Retrain', 'Training Time', 'Testing Time', 
                'Test P', 'Test R', 'Test A', 'Test F', 'Test AUC', 'Test MCC', 'Test B']]

        if df is None:
            df = df_online
        else:
            df = pd.concat([df, df_online])

    df.to_csv('online_model_results.csv', index=False)
            

def combine_oracle():
    df = None
    for dataset in datasets:
        for model in models:
            df_oracle = pd.read_csv('oracle_'+dataset+'_'+model+'.csv')
            df_oracle['Dataset'] = name_datasets[dataset]
            df_oracle = df_oracle[['Dataset', 'Model', 'Training Period', 'Testing Period',
                'Test P', 'Test R', 'Test A', 'Test F', 'Test AUC', 'Test MCC', 'Test B']]

            if df is None:
                df = df_oracle
            else:
                df = pd.concat([df, df_oracle])

    df.to_csv('oracle_results.csv', index=False)


if __name__ == "__main__":
    combine_retrain()
    combine_ensemble()
    combine_online()
    combine_oracle()
