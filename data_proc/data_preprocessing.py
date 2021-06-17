Author: Dilusha-Data61
import pandas as pd
import numpy as np
import os
import datetime

from model_build.feature import *
from config import *


def construct_pipe_failure_data(pdata_comb_file, pfail_file, csv_data_swc):

    df_pdata = pd.read_csv(pdata_comb_file)
    df_pfail = pd.read_csv(pfail_file)

    df_swc_columns = PIPE_ATTRIBUTES + FAILURE_ATTRIBUTES

    df_swc = pd.DataFrame(columns=df_swc_columns)

    working_index = 0
    for index, row in df_pdata.iterrows():
        sgan = row['sgan']
        row2write = []
        for pipe_attribute in PIPE_ATTRIBUTES:
            row2write.append(row[pipe_attribute])

        df_fail_sgan = df_pfail[df_pfail['sgan'].isin([sgan])]
        if df_fail_sgan.empty:
            row2write.append([])
        else:
            reportdates = list(df_fail_sgan['reportdate'].values)
            reportdates.sort()
            row2write.append(reportdates[0::2])

        #=======================================================================
        # if testing_para['using_new_feature']:
        #     df_new_sgan = df_new_feature[df_new_feature['sgan'].isin([sgan])]
        #     if df_new_sgan.empty:
        #         for new_attribute in new_attributes:
        #             row2write.append([])
        #     else:
        #         for new_attribute in new_attributes:
        #             row2write.append(df_new_sgan[new_attribute].values[0])
        #=======================================================================

        df_swc.loc[len(df_swc)] = row2write
        working_index += 1
        # if (working_index%1000 == 0):
        #     print(working_index)

    df_swc.to_csv(csv_data_swc, index=False)


def construct_training_testing_datas(csv_data_swc, csv_data_training, pfail_file, observation_years):
    print('Creating the training file..', csv_data_training)

    pipe_attributes = ['sgan', 'laid date', 'type', 'suburb', 'size', 'length', 'critical', 'environment',
                       'block number', 'trunk', 'source', 'laid_year', 'age', 'observation_year', '#_failures']


    target = ['Failed']
    df_fail = pd.read_csv(pfail_file)
    # ===========================================================================
    # df_fail = df_fail[df_fail['jpnum'].isin([testing_para['working_Zone']])]
    # ===========================================================================

    df_swc = pd.read_csv(csv_data_swc, low_memory=False)
    # ===========================================================================
    # df_swc = pd.read_csv(csv_data_swc)
    # ===========================================================================

    # if (Water_utility == 'water_more_features'):
    #     df_swc = df_swc[df_swc.sgan.str.contains("<Null>") == False]

    df_training_columns = target
    df_training_columns += pipe_attributes

    # df_training_columns += ['laid_year', 'age', 'observation_year', '#_failures']

    index_intervals = []
    j = 0
    step_size = 100
    while j < len(df_swc):
        index_range = range(j, j + step_size)
        index_a = index_range[0]
        index_b = index_range[-1]
        index_intervals.append((index_a, index_b))
        j += step_size

    i = 0
    while i < len(index_intervals):
        index_interval = index_intervals[i]
        df_training = pd.DataFrame(columns=df_training_columns)
        for index, row in df_swc[index_interval[0]:index_interval[1] + 1].iterrows():
            # ===================================================================
            # print index
            # ===================================================================
            sgan = row['sgan']
            laid_year = int(str(row['laid date'])[0:4])
            reportdates = row['reportdates']
            failed_years = []
            if len(reportdates) > 2:
                df_fail_sgan = df_fail[df_fail['sgan'].isin([int(float(str(sgan)))])]
                if not df_fail_sgan.empty:
                    failed_dates = list(df_fail_sgan['reportdate'].values)
                    for failed_date in failed_dates:
                        failed_year = int(str(failed_date)[0:4])
                        failed_years.append(failed_year)

            current_pipe_attributes = []
            for pipe_attribute in PIPE_ATTRIBUTES:
                current_pipe_attributes.append(row[pipe_attribute])

            current_new_attributes = []
            for new_attribute in NEW_ATTRIBUTES:
                current_new_attributes.append(row[new_attribute])

            for observation_year in observation_years:
                row2write = [0]
                row2write += current_pipe_attributes
                row2write += current_new_attributes
                age = observation_year - laid_year
                row2write += [laid_year, age, observation_year]
                row2write += [len([ii for ii in failed_years if ii < observation_year])]
                # ===============================================================
                # len([i for i in failed_years if i < observation_year])
                # failed_years.count(observation_year)
                # if np.sum(failed_years <observation_year)>0:
                #     print np.sum(failed_years >observation_year)
                # if failed_years.count(observation_year-1)>0:
                #     print failed_years, observation_year
                #     print 'haha'
                # ===============================================================

                # ===============================================================
                # row2write += [failed_years.count(observation_year-1)]
                # ===============================================================
                if failed_years.count(observation_year) > 0:
                    row2write[0] = 1
                row2write[1] = int(float(str(row2write[1])))
                row2write[2] = int(float((row2write[2])))
                df_training.loc[len(df_training)] = row2write
        # df_training.drop(['source'], 1, inplace=True)
        if i == 0:
            df_training.to_csv(csv_data_training, index=False)
        else:
            try:
                df_training.to_csv(csv_data_training, mode='a', header=False, index=False)
            except:
                print(
                    'Cant write to the file. Please close the file: ' + csv_data_training + ' if it is already opened and try again!!')
                exit(1)

        i += 1
    print('Training file created..')


def updating_2016_failure_records(csv_data_training_201601201602):
    df = pd.read_csv(csv_data_training_201601201602)

    df.loc[df['#_failures'] == 0, 'Failed'] = 0
    df.loc[df['#_failures'] > 0, 'Failed'] = 1

    df.to_csv(csv_data_training_201601201602, index=False)

def appending_2016_failure_records(csv_data_training, csv_data_training_201601201602, csv_data_training_with_20160102):
    df_swc = pd.read_csv(csv_data_training)
    df_swc_2016 = pd.read_csv(csv_data_training_201601201602)

    df_swc_new = df_swc.append(df_swc_2016, ignore_index=True)
    print('appending is done')

    df_swc_new.sort(['sgan', 'observation_year'], ascending=[0, 1], inplace=True)
    print('sorting is done')

    df_swc_new.loc[df_swc_new['#_failures'] == 0, 'Failed'] = 0
    df_swc_new.loc[df_swc_new['#_failures'] > 0, 'Failed'] = 1
    print('updating is done')

    df_swc_new.to_csv(csv_data_training_with_20160102, index=False)


def updating_training_testing_data(csv_data_training):
    df_swc = pd.read_csv(csv_data_training)

    df_swc.sort(['sgan', 'observation_year'], ascending=[0, 1], inplace=True)
    print('sorting is done')

    df_swc.loc[df_swc['#_failures'] == 0, 'Failed'] = 0
    df_swc.loc[df_swc['#_failures'] > 0, 'Failed'] = 1
    print('updating is done')

    df_swc.to_csv(csv_data_training, index=False)




# def unify_multi_observations_as_one():
#     df_swc = pd.read_csv('./data/training_swc_with_lags_with2016.csv')
#
#     df_swc_2015 = df_swc[df_swc['observation_year'] == 2014]
#
#     historical_failure_columns = construct_lag_cols()
#
#     working_index = 0
#     for index, row in df_swc_2015.iterrows():
#         working_index += 1
#         print(working_index, len(df_swc_2015))
#
#         count_failures = int(row['#_failures'])
#
#         for historical_failure_column in historical_failure_columns:
#             count_failures += int(row[historical_failure_column])
#
#         if count_failures != 0:
#             df_swc_2015.set_value(index, 'Failed', count_failures)
#
#     df_swc_2015.to_csv('./data/training_swc_2016.csv', index=False)
#
# def unify_multi_observations_as_one_for_each_year():
#     df_swc = pd.read_csv('./data/training_swc_with_lags_with2016.csv')
#
#     historical_failure_columns = construct_lag_cols()
#     working_index = 0
#     for index, row in df_swc.iterrows():
#         working_index += 1
#         print(working_index, len(df_swc))
#
#         count_failures = int(row['#_failures'])
#
#         for historical_failure_column in historical_failure_columns:
#             count_failures += int(row[historical_failure_column])
#
#         if count_failures != 0:
#             df_swc.set_value(index, 'Failed', count_failures)
#
#     df_swc.to_csv('./data/training_swc_all_years.csv', index=False)

def updating_failure_count(df_swc, lags=5):

    df_swc['Failed'] = df_swc['#_failures']
    lag_years = range(1,lags+1)
    for lag_year in lag_years:
        df_swc['Failed'] += df_swc['#_failures_year(-' + str(int(lag_year)) + ')']

    return df_swc


def get_failure_interval_features(csv_failure_records):
    date_range = ['2014-01-01', '2014-12-31']
    first_day, last_day = '1997-01-01', '2014-12-31'

    no_failure_days = datetime.datetime.strptime(last_day, '%Y-%m-%d') - datetime.datetime.strptime(first_day,'%Y-%m-%d')

    df = pd.read_csv(csv_failure_records)


    df_pfail_WR1A = df[df['jpnum'].isin(['WR1A'])]

    df_pfail_WR1A.is_copy = False

    df_pfail_WR1A['FIRST_DAY'] = first_day
    df_pfail_WR1A['LAST_DAY'] = last_day



    df_pfail_WR1A['reportdate'] = pd.to_datetime(df_pfail_WR1A['reportdate'], format='%Y%m%d')
    df_pfail_WR1A['FIRST_DAY'] = pd.to_datetime(df_pfail_WR1A['FIRST_DAY'], format='%Y-%m-%d')
    df_pfail_WR1A['LAST_DAY'] = pd.to_datetime(df_pfail_WR1A['LAST_DAY'], format='%Y-%m-%d')

    df_pfail_WR1A['FIRST_DAY_difference'] = (df_pfail_WR1A['reportdate'] - df_pfail_WR1A['FIRST_DAY']).dt.days
    df_pfail_WR1A['LAST_DAY_difference'] = (df_pfail_WR1A['LAST_DAY'] - df_pfail_WR1A['reportdate']).dt.days


    df_training = df_pfail_WR1A[df_pfail_WR1A['reportdate'] < date_range[0]]
    df_training.is_copy = False
    df_training.drop(['FIRST_DAY', 'LAST_DAY'], axis=1, inplace=True)

    sgans = df_training['sgan'].unique()
    df_training.sort(['LAST_DAY_difference'], ascending=[1], inplace=True)
    df_grouped = df_training.groupby('sgan')


    df_firstday_intervals_cols = ['sgan']
    df_lastday_intervals_cols = ['sgan']
    NUM_FAILURES_TO_COUNT = 5
    initial_row2wite = []
    initial_row2wite_firstday = []
    i = 1
    while i <= NUM_FAILURES_TO_COUNT:
        df_firstday_intervals_cols.append('-' + str(i) + "_failure")
        df_lastday_intervals_cols.append('-' + str(i) + "_failure")
        initial_row2wite.append(no_failure_days.days)
        initial_row2wite_firstday.append(-1)
        i += 1
    df_lastday_intervals = pd.DataFrame(columns=df_lastday_intervals_cols)
    df_firstday_intervals = pd.DataFrame(columns=df_firstday_intervals_cols)

    working_index = 0
    for sgan in sgans:
        working_index += 1
        print(working_index, len(sgans))
        #=======================================================================
        # if not np.isnan(sgan):
        #=======================================================================
        df_ingroup = df_grouped.get_group(sgan)

        row2write, row2write_firstday = [sgan], [sgan]
        row2write += initial_row2wite
        row2write_firstday += initial_row2wite_firstday

        failure_index = 0
        for index, row in df_ingroup.iterrows():
            failure_index += 1
            if failure_index > NUM_FAILURES_TO_COUNT:
                break
            row2write[failure_index] = row['LAST_DAY_difference']
            row2write_firstday[failure_index] = row['FIRST_DAY_difference']

        df_lastday_intervals.loc[len(df_lastday_intervals)] = row2write
        df_firstday_intervals.loc[len(df_firstday_intervals)] = row2write_firstday

    df_lastday_intervals.to_csv('./results/lastday_intervals.csv', index=False)
    df_firstday_intervals.to_csv('./results/firstday_intervals.csv', index=False)

def transfer_failure_intervals():
    csv2read = './results/training_swc_2016_withClusteingTemporalFailures_withLastDayInterval.csv'
    # csv2save = './results/training_swc_2016_withClusteingTemporalFailures_withAggInterval.csv'
    # csv2save = './results/training_swc_2016_withClusteingTemporalFailures_withFailureInterval.csv'
    csv2save = './results/training_swc_2016_withClusteingTemporalFailures_TransformFailureInterval.csv'

    df = pd.read_csv(csv2read)
    #
    # to_process_cols = ['-1_failure', '-2_failure', '-3_failure', '-4_failure', '-5_failure']
    # df['Agg_Interval'] = 0.0
    # for to_process_col in to_process_cols:
    #     df[to_process_col] = (max(df[to_process_col]) - df[to_process_col]) / max(df[to_process_col])
    #     df['Agg_Interval'] += df[to_process_col]
    #
    #
    #
    # print df.head(n=50)
    # df.to_csv(csv2save, index=False)

    to_transfer_cols = ['Interval(0-1)', 'Interval(1-2)', 'Interval(2-3)', 'Interval(3-4)', 'Interval(4-5)']
    df['Interval(0-1)'] = 1.0 - df['-1_failure'] / max(df['-1_failure'])
    df['Interval(1-2)'] = 1.0 - (df['-2_failure'] - df['-1_failure']) / max(df['-1_failure'])
    df['Interval(2-3)'] = 1.0 - (df['-3_failure'] - df['-2_failure']) / max(df['-1_failure'])
    df['Interval(3-4)'] = 1.0 - (df['-4_failure'] - df['-3_failure']) / max(df['-1_failure'])
    df['Interval(4-5)'] = 1.0 - (df['-5_failure'] - df['-4_failure']) / max(df['-1_failure'])

    print(df.head(n=50))
    df.to_csv(csv2save, index=False)

def getAllFailurePipes():

    date_range = ['2014-01-01', '2014-12-31']
    csv_fail_records = './data/raw_data/pfail.csv'
    df = pd.read_csv(csv_fail_records)
    df_pfail_WR1A = df[df['jpnum'].isin(['WR1A'])]
    df_pfail_WR1A['reportdate'] = pd.to_datetime(df_pfail_WR1A['reportdate'], format='%Y%m%d')
    df_pfail_WR1A.set_index('reportdate', inplace=True)
    df_2016 = df_pfail_WR1A[(df_pfail_WR1A.index >= date_range[0]) & (df_pfail_WR1A.index <= date_range[1])]
    sgans = df_2016['sgan'].unique()

    return sgans

def analyse_top_ranks(csv_fail_records):
    date_range = ['2016-03-01', '2016-07-31']

    df = pd.read_csv(csv_fail_records)
    df_pfail_WR1A = df[df['jpnum'].isin(['WR1A'])]
    df_pfail_WR1A['reportdate'] = pd.to_datetime(df_pfail_WR1A['reportdate'], format='%Y%m%d')
    df_pfail_WR1A.set_index('reportdate', inplace=True)
    df_2016 = df_pfail_WR1A[(df_pfail_WR1A.index >= date_range[0]) & (df_pfail_WR1A.index <= date_range[1])]
    sgans = df_2016['sgan'].unique()

    df_rank = pd.read_csv('./results/top5000.csv')
    df_rank['Detected'] = 'F'
    df_rank.loc[df_rank['sgan'].isin(sgans), './results/top5000.csv''Detected'] = 'T'
    df_rank.to_csv('./results/top5000.csv', index=False)

def DMA_sort(df_predicting_entity, DMA_file):
    unique_DMA = df_predicting_entity['source'].unique()
     
    DMA_data = pd.DataFrame(columns = [['DMA_name', 'total_score', 'pipe_number', 'pipe_length', '# Failed']])
    
    for DMA_i in unique_DMA:
        DMA_i_density = (df_predicting_entity.loc[df_predicting_entity['source'] == DMA_i]['y_pred'])
        DMA_i_length = (df_predicting_entity.loc[df_predicting_entity['source'] == DMA_i]['length'])
        DMA_i_failures = (df_predicting_entity.loc[df_predicting_entity['source'] == DMA_i]['Failed'])
        DMA_data.loc[len(DMA_data)] = [DMA_i, sum(DMA_i_density), len(DMA_i_density), sum(DMA_i_length), sum(DMA_i_failures)]
        

    # sort by predicted probability
    DMA_data.sort_values(by=['total_score'], ascending=[0], inplace=True)
    DMA_data.reset_index(drop=True, inplace=True)
    DMA_data['accu # Failed'] = np.cumsum(DMA_data['# Failed'])
    
    aa = DMA_data.loc[DMA_data['DMA_name']=='other']
    DMA_data = DMA_data[DMA_data.DMA_name.str.contains('other') == False]    
    DMA_data = DMA_data.append(aa)
    DMA_data.reset_index(drop=True, inplace=True)

    if not os.path.isfile(DMA_file):    
        DMA_data.to_csv(DMA_file, index=False)
        print('Zone based prediction file successfully created')
    else:
        print('Zone based prediction file already exists')


def clustering_analysis(df_training_entity, clustering_analysis_result_file):
    
    unique_clu = df_training_entity['clustering_till_1998'].unique()

    total_failed = sum(df_training_entity['Failed'])
    total_num = len(df_training_entity)
    total_length = sum(df_training_entity['length'])
     
    clu_data = pd.DataFrame(columns = [['clu_name', 'failed_number', 'pipe_number', 'pipe_length', 'failed_ratio', 'number_ratio', 'length_ratio']])
    
    for clu_i in unique_clu:
        clu_i_failure = (df_training_entity.loc[df_training_entity['clustering_till_1998'] == clu_i]['Failed'])
        clu_i_length = (df_training_entity.loc[df_training_entity['clustering_till_1998'] == clu_i]['length'])
        clu_data.loc[len(clu_data)] = [clu_i, sum(clu_i_failure), len(clu_i_failure), sum(clu_i_length), float(sum(clu_i_failure))/total_failed, float(len(clu_i_failure))/total_num, float(sum(clu_i_length))/total_length]

    clu_data.sort_values(by=['failed_number'], ascending=[0], inplace=True)
    
    
    if not os.path.isfile(clustering_analysis_result_file):    
        clu_data.to_csv(clustering_analysis_result_file, index=False)
#===============================================================================
#         df_swc = pd.DataFrame(columns=df_training_columns)
#         for index, row in df_swc_2015[index_interval[0]:index_interval[1] + 1].iterrows():
#             row_element = list(tuple(row))
#             sgan = row['sgan']
#             row_element[-2] = float(testing_year)
#             if sgan not in list_failed_sgans:
#                 row_element[-1] = 0.0
#             else:
#                 df_fail_sgan = df_fail_2016[df_fail_2016['sgan'].isin([sgan])]
#                 if not df_fail_sgan.empty:
#                     count_fails = len(df_fail_sgan)
#                     row_element[-1] = float(count_fails)
#                 else:
#                     row_element[-1] = 0.0
# 
#             df_swc.loc[len(df_swc)] = row_element
#===============================================================================
