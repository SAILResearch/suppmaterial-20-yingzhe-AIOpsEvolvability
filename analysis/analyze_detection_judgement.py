import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu


models = ['lr', 'cart', 'rf', 'nn', 'gbdt']
dic_datasets = {
    'g': 'Google',
    'b': 'Backblaze'
}
dic_num_periods = {
    'g': 28,
    'b': 36
}


def analyze_detection_judgement(dataset):
    df_opt_all = pd.read_csv('experiment_results/oracle_results.csv')
    df_drift_all = pd.read_csv('experiment_results/retrain_model_results.csv')
    print('& &'+'& '.join(map(str, np.arange(dic_num_periods[dataset]//2+1, dic_num_periods[dataset])+1))+'\\\\')

    for model in models:
        result = []
        df_opt = df_opt_all[np.logical_and.reduce((
            df_opt_all['Dataset'] == dic_datasets[dataset],
            df_opt_all['Model'] == model.upper()
        ))]
        df_drift = df_drift_all[np.logical_and.reduce((
            df_drift_all['Dataset'] == dic_datasets[dataset],
            df_drift_all['Model'] == model.upper()
        ))]
        res_drift = df_drift[['Scenario', 'Testing Period', 'Retrain']][np.logical_and.reduce((
            df_drift['Testing Period'] > dic_num_periods[dataset]//2+1,
            df_drift['Scenario'] != 'Stationary'
        ))]
        res_drift = res_drift.groupby(['Scenario', 'Testing Period']).agg(lambda x:x.value_counts().index[0]).to_numpy().reshape((-1, dic_num_periods[dataset]//2-1))  
        # would be in order of DDM, PERM, Retrain, STEPD
        res_drift = res_drift[(2, 0, 1, 3), :]  # reorder to: Retrain, DDM, PERM, STEPD

        stationary_auc = np.mean(df_drift[(df_drift['Scenario'] == 'Stationary') & (df_drift['Testing Period'] == -1)]['Test AUC'])
        # calculate the optimal results that only update when it could improve performance
        opt_res = []
        opt_auc = []
        current_period = dic_num_periods[dataset]//2
        for i in np.arange(dic_num_periods[dataset]//2+2, dic_num_periods[dataset]+1):
            obs1 = df_opt[(df_opt['Training Period'] == current_period) & (df_opt['Testing Period'] == i)][['Test AUC']].to_numpy()
            obs2 = df_opt[(df_opt['Training Period'] == i-1) & (df_opt['Testing Period'] == i)][['Test AUC']].to_numpy()
            _, p = mannwhitneyu(obs1, obs2)

            if p < 0.05 and np.mean(obs1) < np.mean(obs2):
                opt_res.append(3)
                current_period = i - 1
                opt_auc.append(np.mean(obs2))
            else:
                opt_res.append(0)
                opt_auc.append(np.mean(obs1))

        # for each drift detection on each period
        # if not detected: nothing happens, record 0
        # if detected: compare with the last model, record 1 (less than), 2 (same), 3 (greater than)
        scenarios = ['Retrain', 'DDM', 'PERM', 'STEPD']
        for i in range(len(scenarios)):
            out = []
            same_trend = True
            current_period = dic_num_periods[dataset]//2
            for j in np.arange(dic_num_periods[dataset]//2+2, dic_num_periods[dataset]+1):
                if res_drift[i, j - dic_num_periods[dataset]//2 - 2] != (opt_res[j - dic_num_periods[dataset]//2 - 2] > 0):
                    same_trend = False

                old_current_period = current_period
                if res_drift[i, j - dic_num_periods[dataset]//2 - 2]:
                    obs1 = df_opt[(df_opt['Training Period'] == current_period) & (df_opt['Testing Period'] == j)][['Test AUC']].to_numpy()
                    obs2 = df_drift[(df_drift['Scenario'] == scenarios[i]) & (df_drift['Testing Period'] == j)][['Test AUC']].to_numpy()
                    _, p = mannwhitneyu(obs1, obs2)
                    if p < 0.05:
                        if np.mean(obs1) > np.mean(obs2):
                            out.append(1)
                        else:
                            out.append(3)
                    else:
                        out.append(2)
                    current_period = j - 1
                else:
                    out.append(0)

                if same_trend and out[-1] != opt_res[j - dic_num_periods[dataset]//2 - 2]:
                    print('================', scenarios[i], j)
                    #obs1 += df_opt[(df_opt['Training Period'] == old_current_period) & (df_opt['Testing Period'] == j)][['Test AUC']].to_numpy()
                    #obs2 += df_opt[(df_opt['Training Period'] == j-1) & (df_opt['Testing Period'] == j)][['Test AUC']].to_numpy()
                    out[-1] = opt_res[j - dic_num_periods[dataset]//2 - 2]
            
            #p_score = np.round(precision_score(np.greater(opt_res, 0), np.greater(out, 0)), decimals=2)
            #r_score = np.round(recall_score(np.greater(opt_res, 0), np.greater(out, 0)), decimals=2)
            n_retrain = np.count_nonzero(out)
            drift_auc = np.mean(df_drift[(df_drift['Scenario'] == scenarios[i]) & (df_drift['Testing Period'] == -1)][['Test AUC']].to_numpy())
            perf_imp = (drift_auc - stationary_auc) / stationary_auc
            acc_ratio = 0
            if n_retrain:
                acc_ratio = perf_imp / n_retrain * (dic_num_periods[dataset]//2 - 1)
            #result.append([scenarios2[i]]+out+[np.count_nonzero(out), p_score, r_score])
            result.append([scenarios[i]]+out+[n_retrain, '{:.1%}'.format(perf_imp).replace('%', '\%'), '{:.2f}'.format(acc_ratio)])

        n_retrain = np.count_nonzero(opt_res)
        opt_auc = np.mean(opt_auc)
        perf_imp = (opt_auc - stationary_auc) / stationary_auc
        acc_ratio = perf_imp / n_retrain * (dic_num_periods[dataset]//2 - 1)
        opt_res += [n_retrain, '{:.1%}'.format(perf_imp).replace('%', '\%'), '{:.2f}'.format(acc_ratio)]

        print('\\midrule')
        print('\\multirow{5}{*}{'+model.upper()+'}')

        print(' \\\\\n'.join(['& '+'& '.join(map(str, line[:-3])).replace('0', '').replace('1', '$\\downarrow$').replace('2', '$-$').replace('3', '$\\uparrow$') for line in result])+'\\\\')
        print('& Optimal& '+'& '.join(map(str, opt_res[:-3])).replace('0', '').replace('1', '$\\downarrow$').replace('2', '$-$').replace('3', '$\\uparrow$')+'\\\\')

        #print(' \\\\\n'.join(['& '+'& '.join(map(str, line[:-3])).replace('0', '').replace('1', '$\\downarrow$').replace('2', '$-$').replace('3', '$\\uparrow$') + '& ' + '& '.join(map(str, line[-3:])) for line in result])+'\\\\')
        #print('& Optimal& '+'& '.join(map(str, opt_res[:-3])).replace('0', '').replace('1', '$\\downarrow$').replace('2', '$-$').replace('3', '$\\uparrow$')+'& '+ '& '.join(map(str, opt_res[-3:]))+'\\\\')


if __name__ == "__main__":
    for dataset in ['g', 'b']:
        analyze_detection_judgement(dataset)
