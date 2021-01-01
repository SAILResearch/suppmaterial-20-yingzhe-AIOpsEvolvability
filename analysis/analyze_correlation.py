import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from utilities import obtain_period_data
from statsmodels.stats.proportion import proportions_ztest

dic_datasets = {
    'g': 'google',
    'b': 'disk'
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
    dic = [[0]*len(label_list) for _ in range(len(label_list))]
    for i in range(len(label_list)):
        for j in range(i+1, len(label_list)):
            # p < 0.05: from different distribution
            #_, p = mannwhitneyu(label_list[i].astype(int), label_list[j].astype(int))
            _, p = z_test(label_list[i], label_list[j])
            dic[i][j] = dic[j][i] = p
    df = pd.DataFrame(dic, columns=range(1, len(label_list)+1), index=range(1, len(label_list)+1))
    df.to_csv('corr_dependent_'+dic_datasets[dataset]+'.csv', index=False)


def eff_dependent(label_list, dataset):
    dic = [[0]*len(label_list) for _ in range(len(label_list))]
    for i in range(len(label_list)):
        for j in range(i+1, len(label_list)):
            #d = cliffsDelta(label_list[i].astype(int), label_list[j].astype(int))
            d, _ = cohen_d(label_list[i], label_list[j])
            dic[i][j] = dic[j][i] = abs(d)
            print(i, j, d)
    df = pd.DataFrame(dic, columns=range(1, len(label_list)+1), index=range(1, len(label_list)+1))
    df.to_csv('eff_dependent_'+dic_datasets[dataset]+'.csv', index=False)
    

def corr_independent(feature_list, dataset):
    dic = [[0]*len(label_list) for _ in range(len(feature_list))]
    feature_counts = [0] * feature_list[0].shape[1]
    for i in range(feature_list[0].shape[1]):
        for j in range(len(feature_list)):
            count = 0 # how many features are from different distributions
            for k in range(feature_list[0].shape[1]):
                _, p = mannwhitneyu(feature_list[i][k], feature_list[j][k], alternative='two-sided')
                if p <= 0.05:
                    count += 1        
                    feature_counts[k] += 1
            dic[i][j] = dic[j][i] = count
    df = pd.DataFrame(dic, columns=range(1, len(label_list)+1), index=range(1, len(label_list)+1))
    df.to_csv('corr_independent_'+dic_datasets[dataset]+'.csv', index=False)
    print(feature_counts)


if __name__ == "__main__":
    for dataset in ['g', 'b']:
        feature_list, label_list = obtain_period_data(dataset)
        corr_independent(feature_list, dataset)
        corr_dependent(label_list, dataset)
        eff_dependent(label_list, dataset)
