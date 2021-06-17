"""

author: Dilusha Weeraddana-Data61
This script produces age vs failure rates for each material type

"""

import pandas as pd
import os

def get_age_vs_failureRate(utility_name, plot_case, raw_data_file, raw_failure_file, fail_start_year, fail_end_year):
    # 3 for burst and 4 for fitting in the raw data files
    utility_index=plot_case+2
    materials=['AC', 'PVC',  'Others']
    # materials=['AC', 'PVC', 'CICL', 'Others']
    raw_data = pd.read_csv(raw_data_file)
    # raw_data = raw_data.loc[raw_data.GEOMLENGTH > 50]
    raw_failure = pd.read_csv(raw_failure_file)
    for mat in materials:
        result_file = './training_data/' + utility_name + '/' + mat + '_long_term.csv'
        if not os.path.isfile(result_file):
            print(mat, 'does not exists')
            find_get_age_vs_failureRate(mat, raw_data, raw_failure, utility_index, fail_start_year, fail_end_year,
                                              result_file)

def find_get_age_vs_failureRate(mat, raw_data, raw_failure, utility_index, fail_start_year, fail_end_year, result_file):
    dic_print = {}
    dic_age_count = {}
    dic_age_length = {}
    for i in range(0, 200):
        dic_age_count[i] = 0
        dic_age_length[i] = 0

    mat_name = mat + ' pipes'
    if mat!='Others':
        mat_name = raw_data.loc[raw_data.PIPE_MATRL == mat]
    else:
        mat_name_1 = raw_data.loc[raw_data.PIPE_MATRL != 'AC']
        mat_name = mat_name_1.loc[mat_name_1.PIPE_MATRL != 'PVC']
        # mat_name = mat_name_2.loc[mat_name_2.PIPE_MATRL != 'CICL']
    pipe_id = mat_name['GID']
    if (utility_index==3 or utility_index==4):
       each_fail_type = raw_failure.loc[raw_failure.INCIDENT == utility_index]
    else:
       each_fail_type = raw_failure
    for fail_year in range(fail_start_year, fail_end_year + 1):
        for pipe in pipe_id:
            count = 0
            age = fail_year - list(mat_name.loc[mat_name.GID == pipe]['LaidYear'])[0]
            pipe_length = list(mat_name.loc[mat_name.GID == pipe]['Length(km)'])[0]
            if (age > 0):
                fail_year_data = each_fail_type.loc[each_fail_type.FinancialFailYear == fail_year]
                if pipe in list(fail_year_data['WS_GID']):
                    count += 1
                dic_age_count[age] = dic_age_count[age] + count
                # in 100 kms
                dic_age_length[age] = dic_age_length[age] + (pipe_length / 100)

    print(dic_age_count)

    dic_print['age'] = list(dic_age_count.keys())
    dic_print['failure count'] = list(dic_age_count.values())
    dic_print['length'] = list(dic_age_length.values())

    df = pd.DataFrame(dic_print)
    df['failure rate'] = df['failure count'] / df['length']

    df['failure rate'] = df['failure rate'].fillna(0)

    df.to_csv(result_file, index=False)