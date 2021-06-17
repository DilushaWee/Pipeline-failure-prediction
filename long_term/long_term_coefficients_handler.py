"""

author: Dilusha Weeraddana-Data61
This script handles coefficient calculations

"""


from long_term.long_term_coefficients_calculator import coefficient_calculate
import pandas as pd

def get_mean_std(utility_name, plot_case, raw_data_file):

    coe_type_dict = {}

    if plot_case == 1:
        cates_seq = ['t3_AC', 't3_PVC', 't3_Others']
        # cates_seq = ['t3_AC', 't3_PVC', 't3_CICL', 't3_Others']
    elif plot_case == 2:
        cates_seq = ['t4_AC',  't4_PVC', 't4_Others']
        # cates_seq = ['t4_AC',  't4_PVC', 't4_CICL','t4_Others']
    elif plot_case == 3:
        cates_seq = ['t3_t4_AC', 't3_t4_PVC', 't3_t4_Others']
        # cates_seq = ['t3_t4_AC', 't3_t4_PVC',  't3_t4_CICL', 't3_t4_Others']

    data_features = pd.read_csv(raw_data_file)

    if (plot_case == 1)|(plot_case == 2)|(plot_case == 3):
        data_features = data_features[['PIPE_MATRL', 'LaidYear', 'Length(km)','PIPE_DIA']]
        data_features.columns = [['Material', 'LaidYear', 'MeasuredLength(km)','PIPE_DIA']]
    data_features = data_features[pd.notnull(data_features['LaidYear'])]
    

    for cates in cates_seq:
        list_coe=[]
        cates_exact, coe_star, coe_std=coefficient_calculate(plot_case, cates, utility_name,data_features)
        list_coe.append(coe_star)
        list_coe.append(coe_std)
        coe_type_dict[cates_exact]=list_coe

    return coe_type_dict