import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import precision_score, recall_score


models = ['lr', 'cart', 'rf', 'nn', 'gbdt']
dic_datasets = {
    'g': 'google',
    'b': 'disk'
}
dic_num_periods = {
    'g': 28,
    'b': 36
}


def analyze_detection_judgement(dataset):
    result = None
    #df_corr = pd.read_csv('rq1_results/corr_dependent_'+dic_datasets[dataset]+'.csv')
    #df_corr = df_corr.to_numpy()
    for model in models:
        df = pd.read_csv('new_results/concept_'+dic_datasets[dataset]+'_tuned_'+model+'.csv')
        df = df[['Scenario', 'K', 'Retrain']][(df['K'] != -1) & (df['Scenario'] != 'Static Model') & (df['Scenario'] != 'Sliding Window')]
        res1 = df.groupby(['Scenario', 'K']).agg(lambda x:x.value_counts().index[0]).to_numpy().reshape((-1, dic_num_periods[dataset]//2))

        df = pd.read_csv('new_results/oracle_'+dic_datasets[dataset]+'_tuned_'+model+'.csv')
        df = df[['Training Period', 'Testing Period', 'Test AUC']]
        res2 = []
        current_period = dic_num_periods[dataset]//2
        for i in np.arange(dic_num_periods[dataset]//2+1, dic_num_periods[dataset]+1):
            obs1 = df[(df['Training Period'] == current_period) & (df['Testing Period'] == i)][['Test AUC']].to_numpy()
            obs2 = df[(df['Training Period'] == i-1) & (df['Testing Period'] == i)][['Test AUC']].to_numpy()
            _, p = mannwhitneyu(obs1, obs2)
            res2.append(p < 0.05)
            if p < 0.05:
                current_period = i-1
                
        p_scores = np.round([precision_score(res2, pred) for pred in res1] + [1.0], decimals=2)
        r_scores = np.round([recall_score(res2, pred) for pred in res1] + [1.0], decimals=2)
        res1 = np.append(np.array(['Gama', 'Harel', 'Z-test']).reshape((-1, 1)), res1, axis=1)
        res2 = np.array([['Oracle'] + res2])
        res = np.vstack((res1, res2))
        res = np.append(res, np.array(p_scores).reshape((-1, 1)), axis=1)
        res = np.append(res, np.array(r_scores).reshape((-1, 1)), axis=1)

        if type(result) != type(None):
            result = np.vstack((result, res))
        else:
            result = res
    print('& '.join(map(str, np.arange(dic_num_periods[dataset]//2, dic_num_periods[dataset])+1))+'\\\\')
    print(" \\\\\n".join(["& ".join(map(str,line)) for line in result]).replace('False', '').replace('True', '$\\circ$'))


if __name__ == "__main__":
    for dataset in ['g', 'b']:
        analyze_detection_judgement(dataset)
