from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
import os
import timeit

folder = './Backblaze'
start_date = date(2015, 1, 1)
end_date = date(2017, 12, 31)
out_file = 'disk_failure_v2.csv'

val_cols = ['smart_1_raw', 'smart_4_raw', 'smart_5_raw', 'smart_7_raw', 'smart_9_raw', 'smart_12_raw', 'smart_187_raw', 'smart_193_raw', 'smart_194_raw', 'smart_197_raw', 'smart_199_raw']
accu_cols = ['smart_4_raw', 'smart_5_raw', 'smart_9_raw', 'smart_12_raw', 'smart_187_raw', 'smart_193_raw', 'smart_197_raw', 'smart_199_raw']

type_dic = {
    'serial_number': object,
    'date': object,
    'model': object,
    'capacity_bytes': int,
    'smart_1_raw': float,
    'smart_4_raw': float,
    'smart_5_raw': float,
    'smart_7_raw': float,
    'smart_9_raw': float,
    'smart_12_raw': float,
    'smart_187_raw': float,
    'smart_193_raw': float,
    'smart_194_raw': float,
    'smart_197_raw': float,
    'smart_199_raw': float,
    'smart_4_raw_diff': float,
    'smart_5_raw_diff': float,
    'smart_9_raw_diff': float,
    'smart_12_raw_diff': float,
    'smart_187_raw_diff': float,
    'smart_193_raw_diff': float,
    'smart_197_raw_diff': float,
    'smart_199_raw_diff': float,
    'label': bool
}


def build_data(start_date):
    training_files = [datetime.strftime(start_date + timedelta(days=i), '%Y-%m-%d')+'.csv' for i in range(7)]
    testing_files = [datetime.strftime(start_date + timedelta(days=i+7), '%Y-%m-%d')+'.csv' for i in range(7)]

    print('Start building data from', training_files[0])

    training_dfs = [pd.read_csv(os.path.join(folder, file_name)) for file_name in training_files]

    # features that contains date and config level
    date_df = training_dfs[0][['serial_number', 'date', 'model', 'capacity_bytes']]
    date_df = date_df.set_index(date_df['serial_number'])
    
    df = pd.read_csv(os.path.join(folder, training_files[-1]))
    value_df = df[val_cols].set_index(df['serial_number'])
    
    df = pd.concat(training_dfs, ignore_index=True)
    df = df[['serial_number'] + accu_cols]
    df = df.groupby(['serial_number'])
    diff_df = (df.max() - df.min())
    diff_df.columns = [s+'_diff' for s in accu_cols]
    feature_df = pd.concat([value_df, diff_df], axis=1)
    feature_df = feature_df.dropna()

    # label: whether smart_5 changes in the testing period
    df = pd.concat([pd.read_csv(os.path.join(folder, testing_files[0])), pd.read_csv(os.path.join(folder, testing_files[-1]))], ignore_index=True)
    df = df[['serial_number', 'smart_5_raw']].groupby(['serial_number'])
    label_df = ((df.max() - df.min()) > 0)
    label_df.columns = ['label']

    final_df = pd.concat([date_df, feature_df, label_df], axis=1)
    final_df = final_df.dropna()
    final_df = final_df.astype(type_dic)
    print('Length:', len(final_df))
    return final_df


if __name__ == '__main__':
    start_time = timeit.default_timer()
    cur_date = start_date
    while cur_date + timedelta(days=13) <= end_date:
        out_df = build_data(cur_date)
        out_df.to_csv(out_file, mode='a', index=False, header=(not os.path.isfile(out_file)))
        cur_date = cur_date + timedelta(days=1)
    processing_time = timeit.default_timer() - start_time
    print('Job done, consuming time: ' + str(processing_time))
