"""

author: Dilusha Weeraddana-Data61
This is the Main script for long term prediction generation

"""


from long_term.age_vs_failurerate import get_age_vs_failureRate
from long_term.long_term_20_year_prediction import *
from long_term.long_term_coefficients_handler import get_mean_std

def handle_long_term_prediction(utility, plot_case, starting_year, fail_start_year, fail_end_year, raw_data_file, raw_failure_file):

        water_2017_pred_filelpath = './resulting_data/' + utility + '/pipe_based_prediction_' + str(
            starting_year) + '.csv'
        save_count_file_path = './resulting_data/' + utility + '/20_years_pipe_failure_count.csv'
        save_rate_file_path = './resulting_data/' + utility + '/20_years_pipe_failure_rates.csv'

        get_age_vs_failureRate(utility, plot_case, raw_data_file, raw_failure_file, fail_start_year, fail_end_year)

        coe_type_dict=get_mean_std(utility, plot_case, raw_data_file)

        get_20year_prediction_failureCount(water_2017_pred_filelpath, coe_type_dict, save_count_file_path)
        # get_20year_prediction_failureRate(save_count_file_path, save_rate_file_path)
