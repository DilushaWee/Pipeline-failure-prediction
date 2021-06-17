"""

author:  Dilusha & Xuhui -Data61
This script transforms deals with entire data set for formatting

"""

from model_build.model import *
from model_build.feature import *

def pipe_data_transform(raw_data_file, pdata_comb_file, Water_utility):
    try:
        open(raw_data_file, 'r')
        training_data = pd.read_csv(raw_data_file)

    except FileNotFoundError:
        print("The file: "+raw_data_file+ " does not exist \nPlease place the file in correct folder and try again")
        exit(1)
    sample_data_format_file = './sample_format/pdata_comb.csv'

    try:
        open(sample_data_format_file, 'r')
        sample_data_format = pd.read_csv(sample_data_format_file)

    except FileNotFoundError:
        print("The file: "+sample_data_format_file+ " does not exist \nPlease place the file in correct folder and try again")
        exit(1)
    # 1, sgan, 2, type, 3, suburb, 4, size, 5, length, 6, critical, 7, environment, 8, laid date, 9, block number, 10, trunk, 11, source

    if set(['GID', 'MaterialGroup', 'Suburb', 'PIPE_DIA', 'Length(km)', 'CRITICALTY',
                                     'OP_STATUS', 'DATE_MADE']).issubset(training_data.columns):

            water_train = training_data[['GID', 'MaterialGroup', 'Suburb', 'PIPE_DIA', 'Length(km)', 'CRITICALTY',
                                     'OP_STATUS', 'DATE_MADE', 'OP_STATUS', 'OP_STATUS', 'Suburb']]
    else:
            print("Error: please insert all the required colomns: 'GID', 'MaterialGroup', 'Suburb', 'PIPE_DIA', 'Length(km)', 'CRITICALTY','OP_STATUS', 'DATE_MADE' in the all_water_pipe_data.csv file")
            exit(2)

    water_train.columns = sample_data_format.columns

    water_train = water_train[pd.notnull(water_train['sgan'])]
    if water_train['type'].isnull().values.any():
        water_train['type']=water_train['type'].fillna('other')
    if water_train['suburb'].isnull().values.any():
        water_train['suburb']=water_train['suburb'].fillna('other')
    if water_train['critical'].isnull().values.any():
        water_train['critical']=water_train['critical'].fillna(0)
    if water_train['length'].isnull().values.any():
        water_train['length']=water_train['length'].fillna(1)
    if water_train['size'].isnull().values.any():
        water_train['size']=water_train['size'].fillna(1)
    if water_train['environment'].isnull().values.any():
        water_train['environment']=water_train['environment'].fillna('other')
    if water_train['trunk'].isnull().values.any():
        if (Water_utility=='WesternWater_burst'):
            water_train['trunk']=water_train['trunk'].fillna(0)
        else:
            water_train['trunk']=water_train['trunk'].fillna('other')
    if water_train['laid date'].isnull().values.any():
        water_train['laid date']=water_train['laid date'].fillna(19500101)
    
    if water_train['block number'].isnull().values.any():    
        if (Water_utility=='water_more_features'):
            water_train['block number']=water_train['block number'].fillna('other')
        elif (Water_utility=='sewerwater'):
            water_train['block number']=water_train['block number'].fillna(0)
        elif (Water_utility=='WesternWater_burst'):
            water_train['block number']=water_train['block number'].fillna(0)
    
    
    
    if water_train['source'].isnull().values.any():
        if (Water_utility=='WesternWater_burst'):
            water_train['source']=water_train['source'].fillna(0)
        else:
            water_train['source']=water_train['source'].fillna('other')
    
    # print(pd.isnull(water_train).sum())
    
    water_train = water_train.reset_index(drop=True)
    
    try:
        water_train.to_csv(pdata_comb_file, header=True, index=False)
    except:
        print('Cant write to the file. Please close the file: '+ pdata_comb_file+' if it is already opened and try again!!')
        exit(1)

    
