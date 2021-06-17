"""

author: Xuhui & Dilusha -Data61
This script basically initialize the main parameters such as file names etc

"""

from data_proc.data_preprocessing import *
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def construct_files(Water_utility, testing_year):
    """
        construct files by water utility and testing year
    """
    pdata_comb_file = './training_data/' + Water_utility + '/raw_data_formatted/' + 'pdata_comb.csv'
    pfail_file = './training_data/' + Water_utility + '/raw_data_formatted/' + 'pfail.csv'

    csv_data_swc = './training_data/' + Water_utility + '/data_swc_'+ '.csv'
    csv_data_training = './training_data/' + Water_utility + '/training_swc_' + '.csv'

    training_csv2read = './training_data/' + Water_utility + '/training_swc' +'_withClusteingTemporalFailures.csv'

    num_ranking_file = './resulting_data/' + Water_utility + '/yearly_prediction_, ' + str(testing_year) + '_number.csv'
    len_ranking_file = './resulting_data/' + Water_utility + '/yearly_prediction_, ' + str(testing_year) + '_length.csv'
    full_ranking_file = './resulting_data/' + Water_utility + '/intermediate' + '/yearly_prediction_, ' + str(
        testing_year) + '_sort_full.csv'
    final_pipebased_result = './resulting_data/' + Water_utility + '/pipe_based_prediction_' + str(
        testing_year) + '.csv'
    DMA_file = './resulting_data/' + Water_utility + '/zone_based_prediction_' + str(testing_year) + '.csv'
    raw_data_file = './processed_raw_data/all_water_pipe_data.csv'
    raw_failure_file = './processed_raw_data/incident_data.csv'


    testing_para = {
        'using_temporal_feature': True,
        'using_new_feature': True,
        'using_feature_selection': False,
        'OneHotEncoder': False,
        'num_k_fold': 10,
        'training_is_done': False,
        'grouping_training': False,
        'clustering_training': False,
        'clustering_testing': True,
        'using_cluserting_temporal_labels': False,
        'using_interval_features': True,
        'filtering_testing': False,
        'combining_filtering_conditions': False,
        'reranking': False,
        'per_year_per_sample': False,
        'feature_testing': True
    }

    model_name = 'RandomForest'
    model2save = './training_data/' + Water_utility + '/swc_' + str(testing_year) + '_'
    model2save += model_name
    if testing_para['using_temporal_feature']:
        model2save += '_w_temporal'
    else:
        model2save += '_wo_temporal'
    if testing_para['using_new_feature']:
        model2save += '_w_newfeature'
    else:
        model2save += '_wo_newfeature'
    if testing_para['using_cluserting_temporal_labels']:
        model2save += '_w_clusteringTemp_'
    else:
        model2save += '_wo_clusteringTemp_'


        # new data added by Dilusha


    result_files = {}

    result_files['pdata_comb_file'] = pdata_comb_file
    result_files['raw_data_file'] = raw_data_file
    result_files['pfail_file'] = pfail_file
    result_files['raw_failure_file'] = raw_failure_file
    result_files['csv_data_swc'] = csv_data_swc
    result_files['csv_data_training'] = csv_data_training
    result_files['training_csv2read'] = training_csv2read
    result_files['testing_para'] = testing_para
    result_files['num_ranking_file'] = num_ranking_file
    result_files['full_ranking_file'] = full_ranking_file
    result_files['len_ranking_file'] = len_ranking_file
    result_files['model2save'] = model2save
    result_files['model_name'] = model_name
    result_files['final_pipebased_result'] = final_pipebased_result
    result_files['DMA_file'] = DMA_file

    return result_files

def construct_lag_cols():
    historical_failure_columns = []
    for lag_year in LAG_YEARS:
        if lag_year == 0:
            historical_failure_columns.append('#_failures')
        else:
            historical_failure_columns.append('#_failures_year(-' + str(lag_year) + ')')
    return historical_failure_columns

def construct_clustering_lag_cols():
    historical_failure_columns = []
    clustering_lag_years = range(0, 19)
    for lag_year in clustering_lag_years:
        if lag_year == 0:
            historical_failure_columns.append('#_failures')
        else:
            historical_failure_columns.append('#_failures_year(-' + str(lag_year) + ')')
    return historical_failure_columns

def construct_features(testing_para, classical_features):
    features = []
    features += classical_features
    features += ['laid_year', 'age']
    if testing_para['using_new_feature']:
        features += NEW_ATTRIBUTES
    if testing_para['clustering_training']:
        features.append(CLUSTERING_COL)
    if testing_para['using_cluserting_temporal_labels']:
        features.append(CLUSTERING_COL)

    return features

def encoding_categorical_features(df, categorical_features):
    for categorical_feature in categorical_features:
        df[categorical_feature] = df[categorical_feature].astype('category')
    cat_cols = df.select_dtypes(['category']).columns
    df[cat_cols] = df[cat_cols].apply(lambda x: x.cat.codes)
    return df

def converting_to_numeric_features(df, non_numeric_features):
    for non_numeric_feature in non_numeric_features:
        df[non_numeric_feature] = pd.to_numeric(df[non_numeric_feature], errors='coerce')
        df[non_numeric_feature] = df[non_numeric_feature].fillna(0.0)
        if non_numeric_feature == 'TreeCanopyCoverage':
            df['Norm_TreeCanopyCoverage'] = df[non_numeric_feature] / df['length']

    return df

def Encoder(df, toEncodeFeatures):
    encoder = OneHotEncoder()
    label_encoder = LabelEncoder()
    for toEncodeFeature in toEncodeFeatures:
        data_label_encoded = label_encoder.fit_transform(df[toEncodeFeature])
        df[toEncodeFeature] = data_label_encoded
    data_feature_one_hot_encoded = encoder.fit_transform(df[toEncodeFeatures].as_matrix())

    return df


def grouping_training_entities(df_entity):

    df = df_entity.copy()

    df.loc[df['Failed'] > 20, 'Failed'] = 0
    df.loc[(df['#_failures_year(-1)'] < 1) &
            (df['#_failures_year(-2)'] < 1) &
            (df['#_failures_year(-3)'] < 1) &
           (df['#_failures_year(-4)'] < 1) &
           (df['#_failures_year(-5)'] < 1) &
           (df['#_failures_year(-6)'] < 1) &
           (df['#_failures_year(-7)'] < 1) &
           (df['#_failures_year(-8)'] < 1) &
           (df['#_failures_year(-9)'] < 1) &
            (df['#_failures_year(-10)'] < 1), 'Failed'] = 0


    return df

def filtering_entities(df_entity):

    df = df_entity[df_entity['Failed'] <=11]

    df = df_entity[(df_entity['critical'] == 0) | (df_entity['critical'] == 1)]

    return df