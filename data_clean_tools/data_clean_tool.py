# -*- coding: utf-8 -*-

"""
    Author:     Bin Liang
    Modified:   Dilusha Weeraddana
    Version:    1.0
    Date:       29/03/2018
    Data cleaning for Western Water

    usage:
    import date_clean_tool
    date_clean_tool.clean_data()
    
"""
import os
import pandas as pd

from data_clean_tools.dbf_utils import read_dbf

# raw data path
RAW_DATA_PATH = './processed_raw_data'
# RAW_PIPE_DATAFILE = os.path.join(RAW_DATA_PATH, 'SP_WATPIPE.dbf')
RAW_PIPE_DATAFILE = './data_clean_tools/raw_data/SP_WATPIPE.dbf'
# RAW_INCIDENT_DATAFILE = os.path.join(RAW_DATA_PATH, 'WS_Pipe_Incident_Match.xlsx')
RAW_INCIDENT_DATAFILE = './data_clean_tools/raw_data/WS_Pipe_Incident_Match.xlsx'

# temporary data path
TMP_DATA_PATH = './data_clean_tools/tmp_data'
PIPE_REGION_DATAFILE = os.path.join(TMP_DATA_PATH, 'pipe_region.csv')

# processed data path
PROC_DATA_PATH = './processed_raw_data'
# PROC_DATA_PATH = './data_clean_tools/proc_data'

if not os.path.exists(PROC_DATA_PATH):
    os.makedirs(PROC_DATA_PATH)
if not os.path.exists(TMP_DATA_PATH):
    os.makedirs(TMP_DATA_PATH)

CLN_PIPE_DATAFILE = os.path.join(PROC_DATA_PATH, 'cln_water_pipe.csv')
CLN_INCIDENT_DATAFILE = os.path.join(PROC_DATA_PATH, 'cln_incident.csv')

# PROC_CLN_PIPE_DATAFILE = os.path.join(PROC_DATA_PATH, 'proc_cln_water_pipe.csv')
PROC_CLN_PIPE_DATAFILE = os.path.join(PROC_DATA_PATH, 'all_water_pipe_data.csv')
# PROC_CLN_INCIDENT_DATAFILE = os.path.join(PROC_DATA_PATH, 'proc_cln_incide.csv')
PROC_CLN_INCIDENT_DATAFILE = os.path.join(PROC_DATA_PATH, 'incident_data.csv')


def clean_data():
    """
        clean data for later processing
    """

    print('Clean data ...')
    # Step 1. read dbf file, water pipe

    raw_pipe_df = read_dbf(RAW_PIPE_DATAFILE)
    cln_pipe_df = raw_pipe_df.copy()

    incident_df = pd.read_excel(RAW_INCIDENT_DATAFILE)
    cln_incident_df = incident_df.copy()
    # checking whether DATE_MADE, GID, EVENT_DATE, WS_GID exist
    if set(['DATE_MADE', 'GID']).issubset(cln_pipe_df.columns) and set(['EVENT_DATE', 'WS_GID']).issubset(cln_incident_df.columns):

       cln_pipe_df['DATE_MADE'] = pd.to_datetime(cln_pipe_df['DATE_MADE'], errors='coerce')
       current_date = pd.to_datetime('today')

    # invalid DATE_MADE values, e.g., 00000000, 1/01/2222, 1/01/5000, 1/10/9010.
       invalid_date_made_cond = (pd.isnull(cln_pipe_df['DATE_MADE'])) | (cln_pipe_df['DATE_MADE'] > current_date)
       cln_pipe_df = cln_pipe_df[~invalid_date_made_cond]

    # Step 2. read excel file, incident

       # incident_df = pd.read_excel(RAW_INCIDENT_DATAFILE)
       # cln_incident_df = incident_df.copy()

    # 'EVENT_DATE' column contains the number of days since 1900-01-01
       cln_incident_df['EVENT_DATE'] = pd.to_timedelta(cln_incident_df['EVENT_DATE'], unit='D')
       cln_incident_df['EVENT_DATE'] = cln_incident_df['EVENT_DATE'] + pd.to_datetime('1900-01-01') - pd.Timedelta(days=2)
       cln_incident_df.dropna(subset=['EVENT_DATE'], inplace=True)
    # keep records with 'EVENT_DATE' later than 2005-07-01
       cln_incident_df = cln_incident_df[cln_incident_df['EVENT_DATE'] > pd.to_datetime('2005-07-01')]

    # Step 3. merage two data, and remove invalid records, i.e., EVENT_DATE in incident > DATE_MADE in water pipe
       cln_pipe_df['GID'] = cln_pipe_df['GID'].astype('str')
       cln_incident_df['WS_GID'] = cln_incident_df['WS_GID'].astype('str')
       combined_df = cln_incident_df.merge(cln_pipe_df, how='inner', left_on='WS_GID', right_on='GID',
                                        suffixes=('_x', '_y'))
       invalid_incident_incident_ids = combined_df[combined_df['EVENT_DATE'] < combined_df['DATE_MADE']]['GID_x']
       cln_incident_df = cln_incident_df[~cln_incident_df['GID'].isin(invalid_incident_incident_ids)]

    # save results
       cln_pipe_df.to_csv(CLN_PIPE_DATAFILE, index=False)
       cln_incident_df.to_csv(CLN_INCIDENT_DATAFILE, index=False)

    else:
        print("Error: please insert all the colomns: 'DATE_MADE', 'GID', 'EVENT_DATE', 'WS_GID', in the raw data files")
        exit(2)


def process_data():
    """
        process data for model building
    """
    print('Process data ...')
    # Step 1. for water pipe
    # add LaidYear and Suburb to water pipe data
    cln_pipe_df = pd.read_csv(CLN_PIPE_DATAFILE, dtype={'GID': 'str'})
    cln_pipe_df['DATE_MADE'] = pd.to_datetime(cln_pipe_df['DATE_MADE'])
    cln_pipe_df['LaidYear'] = cln_pipe_df['DATE_MADE'].dt.year

    list_date_made = []
    for date in cln_pipe_df['DATE_MADE']:
        str_date = str(date)
        date_only = str_date[:11]
        date_only_nohash = date_only.replace('-', "")
        list_date_made.append(date_only_nohash)
    cln_pipe_df['DATE_MADE'] = list_date_made

    # read pipe region data file
    pipe_suburb_df = pd.read_csv(PIPE_REGION_DATAFILE, dtype={'GID': 'str'})
    proc_cln_pipe_df = cln_pipe_df.merge(pipe_suburb_df, how='left', on='GID')

    # added by Dilusha for length and material group
    proc_cln_pipe_df['Length(km)']=proc_cln_pipe_df['GEOMLENGTH']/1000
    material_list = ['AC', 'CI', 'CICL', 'DICL',
                     'MSCL', 'CU', 'PE', 'PVC', 'RC']

    copy_mat=proc_cln_pipe_df['PIPE_MATRL'].copy()
    df_mat=proc_cln_pipe_df['PIPE_MATRL']

    count=0

    for x in df_mat:
        if(x not in material_list):
            # df_mat.is_copy = False
            df_mat[count]='OTHER'
        count=count+1

    proc_cln_pipe_df['MaterialGroup']=df_mat
    proc_cln_pipe_df['PIPE_MATRL']=copy_mat

    proc_cln_pipe_df['Suburb'] = proc_cln_pipe_df['Suburb'].fillna('NA')
    proc_cln_pipe_df.to_csv(PROC_CLN_PIPE_DATAFILE, index=False)

    # Step 2. for incident
    # add FailYear to incident
    cln_incident_df = pd.read_csv(CLN_INCIDENT_DATAFILE)
    cln_incident_df['EVENT_DATE'] = pd.to_datetime(cln_incident_df['EVENT_DATE'])
    cln_incident_df['FailYear'] = cln_incident_df['EVENT_DATE'].dt.year

    # get financial year - Added by Dilusha
    cln_incident_df['month'] = cln_incident_df['EVENT_DATE'].dt.month

    # Added by Dilusha
    # Event date convert dd/mm/yyyy to yyyymmdd
    # Find financial faile year
    list_event_dates = []
    for date in cln_incident_df['EVENT_DATE']:
        str_date = str(date)
        date_only = str_date[:11]
        date_only_nohash = date_only.replace('-', "")
        list_event_dates.append(date_only_nohash)
    cln_incident_df['EVENT_DATE'] = list_event_dates

    cln_incident_df_small = cln_incident_df.loc[cln_incident_df.month <= 6]
    df_after_june = cln_incident_df['FailYear'].where(lambda x: x < 7, lambda x: x - 1)
    cln_incident_df_small.is_copy = False
    cln_incident_df_small['FinancialFailYear'] = df_after_june
    cln_incident_df_large = cln_incident_df.loc[cln_incident_df.month >6]
    cln_incident_df_large.is_copy = False
    cln_incident_df_large['FinancialFailYear']=cln_incident_df['FailYear']

    frames = [cln_incident_df_small, cln_incident_df_large]
    df_con = pd.concat(frames)
    df_con.drop(['month'], 1, inplace=True)


    cln_incident_df = pd.DataFrame(df_con)
    cln_incident_df.to_csv(PROC_CLN_INCIDENT_DATAFILE, index=False)

    print('Processed WATER PIPE data file is saved to {}'.format(PROC_CLN_PIPE_DATAFILE))
    print('Processed INCIDENT data file is saved to {}'.format(PROC_CLN_INCIDENT_DATAFILE))


def clean_data_handle():
    """
        data cleaning interface for being called
    """

    clean_data()

    process_data()




