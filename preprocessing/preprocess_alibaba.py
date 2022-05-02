
# Reference from Google's cluster features:
# configuration features: user id, logical job name, scheduling class, # tasks, priority, different machine, requested [CPU/memory/disk]
# temporal features: [avg/std] of [cpu/mem/disk] usage in the first five minutes

# pai_job_table:
# job_name, inst_id, user, status, start_time, end_time
# status: only Terminated tasked are successful

# pai_task_table: (# tasks, []plan_cpu, []plan_gpu, []plan_mem, []inst_num)
# job_name, task_name, inst_num, status, start_time, end_time, plan_cpu, plan_mem, plan_gpu, gpu_type

# pai_instance_table (job -- instance -- machine)
# job_name, task_name, inst_name, worker_name, inst_id, status, start_time, end_time, machine

# pai_sensor_table (job, first five minutes -- )
# Note of sensor: all the sensor metrics (CPU, GPU, Memory, I/O) in this table are collected for each instance (indexed by worker_name) but not task, taking the average of all data in the instance's lifetime (except for max_mem and max_gpu_wrk_mem being the maximum).
# job_name, task_name, worker_name, inst_id, machine, cpu_usage, gpu_wrk_util, avg_mem, max_mem, avg_gpu_wrk_mem,max_gpu_wrk_mem, read, write, read_count, write_count

# configuration features: [job_name, inst_id] user, success?, [start_time, end_time], # tasks, # inst, planed [cpu/gpu/mem]

import os
import numpy as np
import pandas as pd

DATA_FOLDER = r'~/clusterdata/cluster-trace-gpu-v2020/data/'

def get_df(file, header=None):
    df = pd.read_csv(file, header=None)
    # df.columns = DF_HEADER.get(key, df.columns)
    df.columns = pd.read_csv("{}.header".format(file.split('.csv')[0])).columns if header is None else header
    return df

# extract from job table
# collected info: job_name, inst_id, user, status, start_time, end_time
job_df = get_df(os.path.join(DATA_FOLDER, 'pai_job_table.csv'))
job_df = job_df[np.logical_and(job_df['status'] != 'Running', job_df['status'] != 'Waiting')]  # filter out unfinished jobs

#np.count_nonzero(np.logical_or((job_df['end_time'] - job_df['start_time']) > 300, np.isnan(job_df['end_time']))) / len(job_df)  
job_df = job_df[np.logical_or((job_df['end_time'] - job_df['start_time']) > 300, np.isnan(job_df['end_time']))]  # remove jobs last less than 5 min
job_df.sort_values(by='start_time', inplace=True)
#job_df['status'].value_counts()

# extract from task table 
# collected info: job_name, task_name, inst_num, plan_cpu, plan_mem, plan_gpu
# concatenate: job_df at job_name, one task can project onto multiple tasks
task_df = get_df(os.path.join(DATA_FOLDER, 'pai_task_table.csv'))
job_task_df = job_df.merge(task_df, on='job_name', how='left', suffixes=['', '_t'])
job_task_agg_df = job_task_df.groupby('job_name').agg({'task_name': 'nunique', 'inst_num': 'sum', 'plan_cpu': 'sum', 'plan_mem': 'sum', 'plan_gpu': 'sum'})

# extract from instance and sensor table
# collect the list of viable worker (in first five minutes)
inst_df = get_df(os.path.join(DATA_FOLDER, 'pai_instance_table.csv'))
job_inst_df = job_df.merge(inst_df[['job_name', 'worker_name', 'start_time', 'end_time', 'machine']], on='job_name', how='left', suffixes=['', '_i'])
job_inst_df['viable'] = np.logical_or(np.logical_not(job_inst_df['end_time_i']), job_inst_df['end_time_i'] <= job_inst_df['start_time'] + 300)
worker_list_df = job_inst_df[['job_name', 'worker_name', 'viable']]
#job_inst_df.groupby('job_name').agg({'viable': 'sum'})['viable'].value_counts()

#machine_df = get_df(os.path.join(DATA_FOLDER, 'pai_machine_metric.csv'))
#worker_machine_df = worker_list_df.merge(machine_df, on='worker_name', how='left')
#job_machine_agg_df = worker_machine_df.groupby('job_name').agg({'machine_cpu': 'mean', 'machine_gpu': 'mean'})
#job_machine_agg_df = job_machine_agg_df.fillna(0.0)

sensor_df = get_df(os.path.join(DATA_FOLDER, 'pai_sensor_table.csv'))
worker_sensor_df = worker_list_df.merge(sensor_df[['worker_name', 'cpu_usage', 'gpu_wrk_util', 'avg_mem', 'max_mem', 'avg_gpu_wrk_mem', 'max_gpu_wrk_mem']], on='worker_name', how='left')
worker_sensor_df = worker_sensor_df[worker_sensor_df['viable']].fillna(0.0)
job_sensor_agg_df = worker_sensor_df.groupby('job_name').agg({
    'cpu_usage': 'mean', 'gpu_wrk_util': 'mean', 
    'avg_mem': 'mean', 'max_mem': 'max', 
    'avg_gpu_wrk_mem': 'mean', 'max_gpu_wrk_mem': 'max'})
job_sensor_agg_df = job_sensor_agg_df.fillna(0.0)

final_df = job_df[['job_name', 'start_time', 'end_time', 'status', 'user']].merge(job_task_agg_df, on='job_name')
#final_df = final_df.merge(job_machine_agg_df, on='job_name')
final_df = final_df.merge(job_sensor_agg_df, on='job_name', how='left')
final_df[['cpu_usage', 'gpu_wrk_util', 'avg_mem', 'max_mem', 'avg_gpu_wrk_mem', 'max_gpu_wrk_mem']] = final_df[['cpu_usage', 'gpu_wrk_util', 'avg_mem', 'max_mem', 'avg_gpu_wrk_mem', 'max_gpu_wrk_mem']].fillna(0.0)

final_df.to_csv('alibaba_job_data.csv')
