"""

authors: Dilusha Weeraddana & Bin Liang -Data61
This script predicts pipe failure prediction for next 20 years

"""
import pandas as pd



type_list = list(range(10))

material_list = ['AC', 'CI', 'CICL','OTHER', 'DICL',
                 'MSCL', 'OTHER','PE', 'PVC', 'RC']

material_dict = dict(zip(type_list, material_list))
start_year = 2017
future_years = 20

def type_to_material_group(type_val):

    material_val_group = material_dict.get(type_val)
    if material_val_group not in ['AC', 'PVC']:
    # if material_val not in ['AC', 'CICL', 'PVC']:
       material_val_group = 'Others'
    return material_val_group

def type_to_material(type_val):

    material_val = material_dict.get(type_val)

    return material_val

def get_20year_prediction_failureCount(water_2017_pred_filelpath, coe_type_dict, save_count_file_path):
    print(coe_type_dict)
    pred_cols = ['y_pred']
    water_pred_data = pd.read_csv(water_2017_pred_filelpath)
    water_pred_data['source'].replace('0', 'OTHERS', inplace=True)

    water_pred_data.rename(columns={'sgan': 'asset_num', 'source': 'DMA'}, inplace=True)

    water_pred_data['material_group'] = water_pred_data['type'].apply(type_to_material_group)
    water_pred_data['material'] = water_pred_data['type'].apply(type_to_material)
    water_pred_data['2017_pred'] = water_pred_data[pred_cols[0]]



    for i in range(1, future_years + 1):
        current_pred_year = start_year + i

        # initialization
        water_pred_data[str(current_pred_year) + '_pred'] = 0

        # iterate by group
        for material, params in coe_type_dict.items():
            group_r_idx_list = water_pred_data[water_pred_data['material_group'] == material].index.tolist()
            water_pred_data.loc[group_r_idx_list, str(current_pred_year) + '_pred'] = \
                (params[0] * (water_pred_data.loc[group_r_idx_list, 'length'] / 100)) * i + \
                 water_pred_data.loc[group_r_idx_list, str(start_year) + '_pred']
    # water_pred_data = water_pred_data.loc[water_pred_data.length > 0.05]

    water_pred_data.drop(['type'], 1, inplace=True)
    water_pred_data.drop(['y_pred'], 1, inplace=True)
    water_pred_data.drop(['Failed'], 1, inplace=True)


    try:
        water_pred_data.to_csv(save_count_file_path, index=False)
    except:
        print('Cant write to the file. Please close the file: '+ save_count_file_path+' if it is already opened and try again!!')
        exit(2)

def get_20year_prediction_failureRate(save_count_file_path, save_rate_file_path):
    water_pred_count = pd.read_csv(save_count_file_path)

    for i in range(0, future_years + 1):
         current_pred_year = start_year + i
         water_pred_count[str(current_pred_year) + '_pred_rate'] = 0
         water_pred_count[str(current_pred_year) + '_pred_rate'] = ((water_pred_count[str(current_pred_year) + '_pred'])/ water_pred_count['length'])
         water_pred_count.drop([str(current_pred_year) + '_pred'],1, inplace=True)
    #     filter out pipes of length > 50 m
    # water_pred_count=water_pred_count.loc[water_pred_count.length>0.05]

    try:
      water_pred_count.to_csv(save_rate_file_path, index=False)
    except:
        print('Cant write to the file. Please close the file: '+ save_rate_file_path+' if it is already opened and try again!!')
        exit(2)