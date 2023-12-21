import pandas as pd
import numpy as np
import glob
import os
import timeit

INPUT_FOLDER = './Google_cluster_data/'
INTERVAL = 300000000

    
def read_all(path):
    all_files = glob.glob(path+'/*.csv.gz')
    all_files.sort()
    print('File number:', len(all_files))

    dfs = []
    for file_name in all_files:
        dfs.append(pd.read_csv(file_name, header=None, index_col=None))

    return pd.concat(dfs, ignore_index=True, sort=False)


def google_job_level():
    print('Reading job_event files')
    job_df = read_all(INPUT_FOLDER+'job_events')
    print('Sorting entries by timestamp')
    job_df.sort_values(by=0,inplace=True)
    
    index = 0
    job_dic = {} # job_id ~ [status, start_t, end_t, uid, job_name, scheduling, set(task_id), priority, diff_machine, 3 requested, 3 usage]
    total_size = len(job_df.index)
    print('Total size:', total_size)

    print('Processing job events')
    # job_id, status, time, user_id, logical_job_name, scheduling_class
    for row in zip(job_df[2], job_df[3], job_df[0], job_df[4], job_df[7], job_df[5]):
        if index % 100000 == 0:
            print('Progress:', index, '/', total_size)
        index += 1
        job_id = row[0]
        if job_id not in job_dic:
            job_dic[job_id] = [row[1], row[2], -1, row[3], row[4], row[5], set(), -1, -1, [0.0, 0.0, 0.0], ([], [], [])]
            continue
    
        job_dic[job_id][0] = row[1] # update status
        job_dic[job_id][2] = row[2] # update end_time

        if row[1] > 6: # event update
            job_dic[job_id][3:6] = list(row[3:])

  
    print('Reading task_event files')
    task_df = read_all(INPUT_FOLDER+'task_events')

    index = 0
    total_size = len(task_df.index)
    print('Total size:', total_size)
    print('Processing task events')
    # job_id, task_index, priority, CPU_req, mem_req, disk_req, diff_machine
    for row in zip(task_df[2], task_df[3], task_df[8], task_df[9], task_df[10], task_df[11], task_df[12]):
        if index % 1000000 == 0:
            print('Progress:', index, '/', total_size)
        index += 1
        job_id = row[0]
        if job_id not in job_dic:
            continue
        job_dic[job_id][6].add(row[1]) # add task index

        job_dic[job_id][7] = row[2] # update priority
        if not np.isnan(row[6]):
            job_dic[job_id][8] = row[6] # update diff_machine
        if not np.isnan(row[3]):
            job_dic[job_id][9][0] += row[3]
        if not np.isnan(row[4]):
            job_dic[job_id][9][1] += row[4]
        if not np.isnan(row[5]):
            job_dic[job_id][9][2] += row[5]

    print('Processing task usage')
    all_files = glob.glob(INPUT_FOLDER+'task_usage/*.csv.gz')
    all_files.sort()
    print('File number:', len(all_files))

    for file_name in all_files:
        index = 0
        usage_df = pd.read_csv(file_name, header=None, index_col=None)
        total_size = len(usage_df.index)
        print('Reading task usage file:', file_name)
        print('Total size:', total_size)

        # end_time, job_id, cpu rate, memory usage, disk usage
        for row in zip(usage_df[1], usage_df[2], usage_df[5], usage_df[6], usage_df[12]):
            if index % 1000000 == 0:
                print('Progress:', index, '/', total_size)
            index += 1
            job_id = row[1]
            if job_id not in job_dic:
                continue

            if row[0] <= job_dic[job_id][1] + INTERVAL:
                job_dic[job_id][10][0].append(row[2])
                job_dic[job_id][10][1].append(row[3])
                job_dic[job_id][10][2].append(row[4])

    print('Calculating statistical properties')
    job_ls = []
    for k, v in job_dic.items():
        out = [k] + v[0: 6] + [len(v[6])] + v[7: 9] + v[9]
        if len(v[10][0]) >= 1:
            job_ls.append(out + [np.mean(v[10][0]), np.mean(v[10][1]), np.mean(v[10][2]), 
                                 np.std(v[10][0]), np.std(v[10][1]), np.std(v[10][2])])
        else:
            continue
    
    print('Saving CSV file')
    columns = ['Job ID', 'Status', 'Start Time', 'End Time', 'User ID', 'Job Name', 'Scheduling Class', 
               'Num Tasks', 'Priority', 'Diff Machine', 'CPU Requested', 'Mem Requested', 'Disk Requested',
               'Avg CPU', 'Avg Mem', 'Avg Disk', 'Std CPU', 'Std Mem', 'Std Disk']
    task_out_df = pd.DataFrame(job_ls, columns = columns)
    task_out_df.to_csv('./google_job_level.csv', index=False)


if __name__ == "__main__":
    start_time = timeit.default_timer()
    google_job_level()
    processing_time = timeit.default_timer() - start_time
    print('Job done, consuming time: ' + str(processing_time))
