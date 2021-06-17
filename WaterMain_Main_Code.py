"""

Author: Dilusha-Data61
This is the Main script which handles short and long term prediction

"""
#import hdbscan

from data_proc.Water_pipe_data_transform import *
from data_proc.Water_failure_data_transform import *
from model_build.model import pipeline_regression as regr
from model_build.feature import construct_files
from long_term.long_term_main import handle_long_term_prediction
# from data_clean_tools.data_clean_tool import clean_data_handle
import shutil



def train(features2use, df_training_entity, model2save, targets, model_name=None, clustering_label=None):
    y = df_training_entity[targets[0]]

    X = df_training_entity[features2use]

    ### setup the model to be used
    if model_name is None:
        md_regr = regr(LinearRegression())
    else:
        md_regr = regr(model_pipelines[model_name])

    ### training the regression model
    print('Start training ...')
    # print('Start training by', model_name, '...')
    md_regr.train(X, y)
    print('Training for ', model_name, 'is completed.')

    ### saving the regression model
    md_regr.save_model(model2save)
    print('Trained model is saved.')

def predict(features2use, df_testing_entity, model2read, model_name, clustering_label=None):
#===============================================================================
#     features2use = construct_features(testing_para)
# 
#     if len(categorical_features) > 0:
#         df_testing_entity = enco
# ding_categorical_features(df_testing_entity, categorical_features)
#     if testing_para['using_new_feature']:
#         df_testing_entity = converting_to_numeric_features(df_testing_entity, new_attributes)
#     if testing_para['using_temporal_feature']:
#         features2use += construct_lag_cols()
#===============================================================================
    X = df_testing_entity[features2use]

    md_regr = regr(model_pipelines[model_name])
    md_regr.load_model(model2read)

    feature_importance = md_regr.reg.feature_importances_
    f_im = pd.DataFrame(data = feature_importance)
    # feature_importance_file = './resulting_data/'+utility+'/feature_importance.csv'
    
    # f_im.to_csv(feature_importance_file, index=False)

    y_pred = md_regr.predict(X)

    df_testing_entity['y_pred'] = y_pred
    return df_testing_entity


# def clustering_with_temporal_patterns(testing_para, csv_data_training, training_csv2read, classical_features, categorical_features):
#     df_entity = pd.read_csv(csv_data_training)
#
#     features = []
#     features += classical_features
#     features += ['laid_year', 'age']
#     if testing_para['using_new_feature']:
#         features += NEW_ATTRIBUTES
#
#     years2include = []
#     historical_failure_columns = []
#     for lag_year in LAG_YEARS:
#         years2include.append(1998-lag_year)
#         if lag_year == 0:
#             historical_failure_columns.append('#_failures')
#         else:
#             historical_failure_columns.append('#_failures_year(-' + str(lag_year) + ')')
#
#     historical_failures = []
#     i = 0
#     while i < len(years2include):
#         clustering_label = 'clustering_till_' + str(years2include[i])
#         historical_failures.append(historical_failure_columns[i])
#         clustering_features = features + historical_failures
#         if len(categorical_features) > 0:
#             df_entity = encoding_categorical_features(df_entity, categorical_features)
#         if testing_para['using_new_feature']:
#             df_entity = converting_to_numeric_features(df_entity, NEW_ATTRIBUTES)
#
#         print('start clustering...')
#         X = df_entity[clustering_features]
#         clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
#                                     gen_min_span_tree=True, leaf_size=20, metric='euclidean', min_cluster_size=5,
#                                     min_samples=None, p=None)
#         cluster_labels = clusterer.fit_predict(X)
#         cluster_labels[cluster_labels>-1] = 0
#         cluster_labels[cluster_labels==-1] = 1
#         df_entity[clustering_label] = cluster_labels
#         #=======================================================================
#         # df_entity[clustering_label] += 2
#         #=======================================================================
#         i += 1
#     try:
#         df_entity.to_csv(training_csv2read, index=False)
#     except:
#         print(
#             'Cant write to the file. Please close the file: ' + training_csv2read + ' if it is already opened and try again!!')
#         exit(1)


def get_input():
    """
        Option interface
    """
    # Inputs added by Dilusha

    print(
        'Have you replaced the raw data files (yes/no)?')
    #
    user_input=inputYes_No('Yes/No: ')

    if (user_input):
     # clean_data_handle()
     try:
        if os.path.exists('./training_data'):
            shutil.rmtree('./training_data')

        if os.path.exists('./resulting_data'):
            shutil.rmtree('./resulting_data')

     except (OSError):
        print("Error: cannot delete the folders")
        exit(0)

    #
    # while(Water_utility_index>3 or Water_utility_index<1 ):
    #     Water_utility_index = inputNumber('Enter a valid selection number (1-3): ')
    #
    # Water_utility_seq = ['WesternWater_burst', 'WesternWater_fittingtofailure', 'Westernwarter_combined']
    #
    # Water_utility = Water_utility_seq[Water_utility_index - 1]
    #
    # testing_year = inputNumber('Enter a prediction year: ')
    #
    # while (testing_year > 2017 or testing_year < 2006):
    #     testing_year = inputNumber('Enter a valid prediction year (2006-2017): ')
    Water_utility = ['WesternWater_burst', 'WesternWater_fitting', 'WesternWater_combined']

    testing_year=2017

    fail_starting_year = 2005

    fail_end_year=2017

    observation_years = range(fail_starting_year, fail_end_year+1)

    classical_features = ['type', 'suburb', 'size', 'length', 'critical', 'trunk']

    categorical_features = ['type', 'suburb']

    return Water_utility, testing_year, categorical_features, classical_features, observation_years, fail_starting_year, fail_end_year


def inputNumber(message):
  while True:
    try:
       userInput = int(input(message))
    except ValueError:
       print("Oops not an integer! Please try again.")
       continue
    else:
       return userInput
       break

def inputYes_No(message):
    yes = {'yes', 'y', 'ye', ''}
    no = {'no', 'n'}
    while True:
       choice = input().lower()
       if choice in yes:
           return True
           break

       elif choice in no:
           return False
           break

       else:
           print("Please respond with 'yes' or 'no'")
           continue



def scaling_prediction(testing_year,starting_year,df,final_pipebased_result):

    value_1 = 0.01

    # df = pd.read_csv(full_ranking_file)
    year_devision = testing_year - starting_year
    # less than 10 m

    less_than_50 = df.loc[df.length < value_1]
    sum_pred = np.sum(less_than_50['y_pred'])
    fail_count_avg = (np.sum(less_than_50['#_failures'])) / year_devision
    scale = sum_pred / fail_count_avg
    scaled_list_50 = less_than_50['y_pred'] / scale
    less_than_50.is_copy = False
    less_than_50.iloc[:, 16] = scaled_list_50.values
    # less_than_50.iloc[:, 16] = scaled_list_50.values

    greater_than_50 = df.loc[df.length >= value_1]
    #
    # # concatenating all the data frames
    frames = [less_than_50, greater_than_50]
    df_con = pd.concat(frames)
    df_final = pd.DataFrame(df_con)
    df_final=df[['sgan',	'type',	'size',	'length',	'source',	'laid_year', 'Failed', 'y_pred']]


    # else:
    #     print('2017 prediction file already exists')
    return df_final

def create_dir(Water_utility):
    if not os.path.exists('./training_data/' + Water_utility):
        os.makedirs('./training_data/' + Water_utility)
    if not os.path.exists('./training_data/' + Water_utility+'/raw_data_formatted'):
        os.makedirs('./training_data/' + Water_utility+'/raw_data_formatted')
    if not os.path.exists('./resulting_data/' + Water_utility):
        os.makedirs('./resulting_data/' + Water_utility)

if __name__ == '__main__':
  print('Program started...')

  Water_utility, testing_year, categorical_features, classical_features, observation_years, fail_starting_year, fail_end_year = get_input()


  plot_case=0
  for utility in Water_utility:
    plot_case += 1
    result_files = construct_files(utility, testing_year)

    pdata_comb_file = result_files['pdata_comb_file']
    raw_data_file = result_files['raw_data_file']
    pfail_file = result_files['pfail_file']
    raw_failure_file = result_files['raw_failure_file']
    csv_data_swc = result_files['csv_data_swc']
    csv_data_training = result_files['csv_data_training']
    training_csv2read = result_files['training_csv2read']
    testing_para = result_files['testing_para']
    num_ranking_file = result_files['num_ranking_file']
    full_ranking_file = result_files['full_ranking_file']
    len_ranking_file = result_files['len_ranking_file']
    model2save = result_files['model2save']
    model_name = result_files['model_name']
    final_pipebased_result = result_files['final_pipebased_result']
    DMA_file = result_files['DMA_file']

    create_dir(utility)
    # This is for data pre-processing
    if not os.path.isfile(pdata_comb_file):
        pipe_data_transform(raw_data_file, pdata_comb_file, utility)

    if not os.path.isfile(pfail_file):
        pipe_failure_transform(raw_failure_file, pfail_file, utility)

    if not os.path.isfile(csv_data_swc):
        construct_pipe_failure_data(pdata_comb_file, pfail_file, csv_data_swc)

    if not os.path.isfile(csv_data_training):
        construct_training_testing_datas(csv_data_swc, csv_data_training, pfail_file, observation_years)

    # if not os.path.isfile(training_csv2read):
    #     # clustering_with_temporal_patterns(testing_para, csv_data_training, training_csv2read, classical_features, categorical_features)
    #     testing_para['using_cluserting_temporal_labels']=False
    #     features2use_full = construct_features(testing_para, classical_features, False)
    #     df_training_entity = pd.read_csv(csv_data_training)
    #     max_features=9

    df_training_entity = pd.read_csv(csv_data_training)
    max_features=9


    #===========================================================================
    # if not os.path.isfile(clustering_analysis_result_file):
    #     clustering_analysis(df_training_entity, clustering_analysis_result_file)
    #===========================================================================

 
    construct_model_pipelines(testing_para, len(df_training_entity), max_features)
    features2use_full = construct_features(testing_para, classical_features)

    df_testing_entity = df_training_entity[df_training_entity['observation_year'] == testing_year]
    df_training_entity = df_training_entity[df_training_entity['observation_year'] < testing_year]
    


    if len(categorical_features) > 0:
        df_training_entity = encoding_categorical_features(df_training_entity, categorical_features)
        df_testing_entity = encoding_categorical_features(df_testing_entity, categorical_features)
    if testing_para['using_new_feature']:
        df_training_entity = converting_to_numeric_features(df_training_entity, NEW_ATTRIBUTES)
        df_testing_entity = converting_to_numeric_features(df_testing_entity, NEW_ATTRIBUTES)
    if testing_para['using_temporal_feature']:
        features2use_full += construct_lag_cols()

    if not os.path.isfile(model2save):
        train(features2use_full, df_training_entity, model2save, TARGETS,model_name)

    df_predicting_entity = predict(features2use_full, df_testing_entity, model2save, model_name)

    df_predicting_entity.sort_values(by=['y_pred'], ascending=[0], inplace=True)

    fail_sort = np.cumsum(df_predicting_entity['Failed'])
    length_sort = np.cumsum(df_predicting_entity['length'])
    num_sort = np.arange(len(fail_sort))+1
    
    datai_spe = pd.DataFrame({'failed number' :fail_sort, 'pipe number': num_sort})

    datai_spe.reset_index(drop=True, inplace=True)

    df_predicting_entity.reset_index(drop=True, inplace=True)

    # if not os.path.isfile(full_ranking_file):
    #     try:
    #         df_predicting_entity.to_csv(full_ranking_file, index=False)
    #     except:
    #         print(
    #             'Cant write to the file. Please close the file: ' + full_ranking_file + ' if it is already opened and try again!!')
    #         exit(1)
    if not os.path.isfile(final_pipebased_result):

      df_final = scaling_prediction(testing_year, observation_years[0], df_predicting_entity, final_pipebased_result)
      df_final.to_csv(final_pipebased_result, index=False)
      print('Pipe based prediction file for year 2017 successfully created')

    if not (os.path.isfile('./resulting_data/' + utility + '/20_years_pipe_failure_count.csv') and
         os.path.isfile('./resulting_data/' + utility + '/20_years_pipe_failure_rates.csv')):

    # DMA_sort(df_final, DMA_file)
    # long term results generation
      failure_type=(utility.split('_'))[1]
      print('20 years forecasting started for '+failure_type+' failure...')
      handle_long_term_prediction(utility, plot_case, testing_year, fail_starting_year, fail_end_year, raw_data_file,
                                raw_failure_file)
      print(
          'Long term prediction files for '+failure_type+' have successfully been created. Please find the prediction for each pipe in ',
          './resulting_data/' + utility)
