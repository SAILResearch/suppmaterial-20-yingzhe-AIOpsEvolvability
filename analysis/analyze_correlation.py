import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from utilities import obtain_period_data
from statsmodels.stats.proportion import proportions_ztest

dic_datasets = {
    'g': 'google',
    'b': 'backblaze',
    'a': 'alibaba'
}


def runs(lst):
    """Iterator, chunks repeated values"""
    for j, two in enumerate(lst):
        if j == 0:
            one, i = two, 0
        if one != two:
            yield j - i, one
            i = j
        one = two
    yield j - i + 1, two


def cliffsDelta(lst1, lst2):
    dull = {'small': 0.147, 'medium': 0.33, 'large': 0.474} # effect sizes from (Hess and Kromrey, 2004)
    m, n = len(lst1), len(lst2)
    lst2 = sorted(lst2)
    j = more = less = 0
    for repeats, x in runs(sorted(lst1)):
        while j <= (n - 1) and lst2[j] < x:
            j += 1
        more += j*repeats
        while j <= (n - 1) and lst2[j] == x:
            j += 1
        less += (n - j)*repeats
    d = (more - less) / (m*n)
    return d


def cohen_d(lst1, lst2):
    n01 = np.count_nonzero(lst1.astype(int))
    n11 = np.count_nonzero(lst2.astype(int))
    n00 = lst1.shape[0] - n01
    n10 = lst2.shape[0] - n11
    LOR = np.log(n00) + np.log(n11) - np.log(n01) - np.log(n10)
    SE = np.sqrt(1/n00 + 1/n01 + 1/n10 + 1/n11)
    
    return LOR * np.sqrt(3) / np.pi, SE * 3 / np.pi / np.pi


def z_test(lst1, lst2):
    zstat, pval = proportions_ztest([np.count_nonzero(lst1.astype(int)), np.count_nonzero(lst2.astype(int))], [lst1.shape[0], lst2.shape[0]])
    return zstat, pval  # pval < 0.05: different population
    

def corr_dependent(label_list, dataset):
    num_periods = len(label_list)
    dic = [[0]*num_periods for _ in range(num_periods)]
    for i in range(num_periods):
        for j in range(i+1, num_periods):
            # p < 0.05: from different distribution
            #_, p = mannwhitneyu(label_list[i].astype(int), label_list[j].astype(int))
            _, p = z_test(label_list[i], label_list[j])
            dic[i][j] = dic[j][i] = p
    df = pd.DataFrame(dic, columns=range(1, num_periods+1), index=range(1, num_periods+1))
    df.to_csv('corr_dependent_'+dic_datasets[dataset]+'.csv', index=False)


def eff_dependent(label_list, dataset):
    num_periods = len(label_list)
    dic = [[0]*num_periods for _ in range(num_periods)]
    for i in range(num_periods):
        for j in range(i+1, num_periods):
            #d = cliffsDelta(label_list[i].astype(int), label_list[j].astype(int))
            d, _ = cohen_d(label_list[i], label_list[j])
            dic[i][j] = dic[j][i] = abs(d)
            print(i, j, d)
    df = pd.DataFrame(dic, columns=range(1, num_periods+1), index=range(1, num_periods+1))
    df.to_csv('eff_dependent_'+dic_datasets[dataset]+'.csv', index=False)
    

def corr_independent(feature_list, dataset):
    num_periods = len(feature_list)
    num_features = feature_list[0].shape[1]
    dic = [[0]*num_periods for _ in range(num_periods)]
    feature_counts = [0] * num_features
    for i in range(num_periods):
        for j in range(num_periods):
            count = 0 # how many features are from different distributions
            for k in range(num_features):
                _, p = mannwhitneyu(feature_list[i][k], feature_list[j][k], alternative='two-sided')
                if p <= 0.05:
                    count += 1        
                    feature_counts[k] += 1
            dic[i][j] = dic[j][i] = count
    df = pd.DataFrame(dic, columns=range(1, num_periods+1), index=range(1, num_periods+1))
    df.to_csv('corr_independent_'+dic_datasets[dataset]+'.csv', index=False)
    print(feature_counts)


def data_analysis(label_list, dataset):
    num_periods = len(label_list)
    failure_rates = []
    data_sizes = []
    for i in range(num_periods):
        failure_rates.append(np.count_nonzero(label_list[i])/len(label_list[i]))
        data_sizes.append(len(label_list[i]))
    df = pd.DataFrame({'x': range(1, num_periods+1), 'y': failure_rates})
    df.to_csv('failure_rate_'+dic_datasets[dataset]+'.csv', index=False)
    df = pd.DataFrame({'x': range(1, num_periods+1), 'y': data_sizes})
    df.to_csv('data_size_'+dic_datasets[dataset]+'.csv', index=False)


if __name__ == "__main__":
    for dataset in ['g', 'b', 'a']:
        feature_list, label_list = obtain_period_data(dataset)
        #corr_independent(feature_list, dataset)
        corr_dependent(label_list, dataset)
        eff_dependent(label_list, dataset)
        data_analysis(label_list, dataset)
