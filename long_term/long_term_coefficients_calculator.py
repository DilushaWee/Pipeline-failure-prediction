"""

author: Dilusha Weeraddana-Data61
This script does actual coefficient calculations

"""

import numpy as np
import pandas as pd
from sklearn import linear_model

def coefficient_calculate(plot_case, cates, utility_name,data_features):

    if (plot_case == 3):
        cates = cates[6:]
    elif (plot_case == 1)|(plot_case == 2):
        cates = cates[3:]
    if (plot_case == 1)|(plot_case == 2)|(plot_case == 3):

        file_appendix = 'csv'
        age_vs_fr_file = './training_data/'+utility_name+'/'+cates+'_long_term.'+file_appendix
        df_ranking = pd.read_csv(age_vs_fr_file)
    df_ranking = df_ranking_filtering(df_ranking)
    regr = linear_model.LinearRegression()
    df_ranking.loc[len(df_ranking)] = [0, 0, 0, 0]
    regr.fit(pd.DataFrame(df_ranking['age']), pd.DataFrame(df_ranking['failure rate']))
    
    coef_seq = df_ranking['failure rate']/df_ranking['age']

    coef_std = np.std(coef_seq)

    valss = np.arange(np.min(coef_seq), np.max(coef_seq), 0.001)
    diss_err = []
    coef_star=0

    regr.intercept_[0] = 0
    for coe_i in valss:
        regr.coef_ = np.array([[coe_i]])
        diss_err.append(np.sum((regr.predict(pd.DataFrame(df_ranking['age']))-pd.DataFrame(df_ranking['failure rate']))**2))
        # diss_err.append(np.sum(((regr.predict(pd.DataFrame(df_ranking['age']))-pd.DataFrame(df_ranking['failure rate']))**2)*(pd.DataFrame(df_ranking['length']))**2))

    # for AC and PVC burst failures weighted linera regression is used

    if (plot_case == 1 and (cates=='AC' or cates=='PVC') ):
        coef_star = ((np.sum(
        ((((df_ranking['failure rate'])) * (df_ranking['age'])) * (
            (df_ranking['length'])) ** 2))) / (np.sum(
        ((((df_ranking['age'])) * (df_ranking['age'])) * (
            (df_ranking['length'])) ** 2))))
    else:
        if len(valss) != 0:
            coef_star = valss[np.argmin(diss_err)]

    return cates, coef_star, coef_std

def df_ranking_filtering(df_ranking):

    df_ranking = df_ranking[(df_ranking['age'] > 0)]
    df_ranking = df_ranking[(df_ranking['age'] < 60)]
    return df_ranking.reset_index(drop=True)
