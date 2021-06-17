"""

author: Dilusha & Xuhui  -Data61
This script formats failure data

"""


from data_proc.data_preprocessing import *
from model_build.model import *


def pipe_failure_transform(raw_failure_file, pfail_file, Water_utility):

    try:
        open(raw_failure_file, 'r')
        failure_data = pd.read_csv(raw_failure_file, encoding='cp1252')
    except FileNotFoundError:
        print("The file: "+raw_failure_file+ " does not exist \nPlease place the file in correct folder and try again")
        exit(1)


    sample_data_format_file = './sample_format/pfail.csv'
    try:
        open(sample_data_format_file, 'r')
        sample_data_format = pd.read_csv(sample_data_format_file)

    except FileNotFoundError:
        print(
            "The file: " + sample_data_format_file + " does not exist \nPlease place the file in correct folder and try again")
        exit(1)


    if 'INCIDENT' in failure_data.columns:
        if Water_utility=='WesternWater_burst':
            failure_data = failure_data[(failure_data.INCIDENT == 3)]
        elif Water_utility=='WesternWater_fitting':
            failure_data = failure_data[(failure_data.INCIDENT == 4)]
    else:
          print("Please insert the 'INCIDENT' coloumn in the incident.csv file")
          exit(2)
    if set(['WS_GID' ,'FinancialFailYear']).issubset(failure_data.columns):
      # if ('WS_GID' in failure_data.columns & 'FinancialFailYear' in failure_data.columns &'Sym_name' in failure_data.columns &'Sym_scale' in failure_data.columns ):
        water_failure = failure_data[['WS_GID', 'FinancialFailYear']]
        # water_failure = failure_data[['WS_GID', 'FinancialFailYear', 'Sym_name', 'Sym_scale', 'Sym_scale', 'Sym_scale']]
        water_failure.columns = sample_data_format.columns
    else:
          print("Error: please insert all the colomns: 'WS_GID' ,'FinancialFailYear'  in the incident.csv file")
          exit(2)
    try:
        water_failure.to_csv(pfail_file, header=True, index=False)
    except:
        print('Cant write to the file. Please close the file: '+ pfail_file+' if it is already opened and try again!!')
        exit(1)

