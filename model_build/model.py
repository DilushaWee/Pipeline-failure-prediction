import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab
from mpl_toolkits.mplot3d import Axes3D

import pickle
#===============================================================================
# 
# import sklearn
# sklearn.__version__
#===============================================================================

from sklearn.externals import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model as sklm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import WhiteKernel


model_pipelines = {}

def getLinearRegressionError(df,col_x, col_y):
    x= df[col_x].values
    X = np.atleast_2d(x).T
    y = df[col_y].values

    linear_regression = LinearRegression()
    model = Pipeline([("linear_regression", linear_regression)])
    model.fit(X, y)
    y_pred_train = model.predict(X)
    train_mse = mean_squared_error(y, y_pred_train)

    rmse = np.sqrt(train_mse)

    return rmse


def construct_model_pipelines(parameters, num_features, max_features):
    '''
    constructing the pipelines based on the given parameters and the shape of data
    '''

    if not parameters['using_feature_selection']:
        n_features, n_dim_manifold, n_hidden = 1, 1, 1
    else:
        n_features, n_dim_manifold, n_hidden = 20, 3, 3

    model_pipelines['LinearRegression'] = LinearRegression()
    model_pipelines['RandomForest'] = RandomForestRegressor(n_estimators=114, min_samples_leaf=5,
                                                            random_state=None, max_features=max_features, n_jobs=-1)


class pipeline_regression:
    def __init__(self, model = LinearRegression()):
        self.model = model
        self.IsModelConstructed = False
        self.xs = np.linspace(0, np.pi, 1000)

    def train(self, x_train, y_train):
        self.reg = self.model.fit(x_train, y_train)
        self.IsModelConstructed = True

    def predict(self, x_test):
        if self.IsModelConstructed:
            return self.reg.predict(x_test)
        else:
            print('Model has not yet been constructed!')
            return None

    def save_model(self, name2save):
        if self.IsModelConstructed:
            joblib.dump(self.reg, name2save)
        else:
            print('Model has not yet been constructed!')
            return None

    def load_model(self, name2read):
        self.reg = joblib.load(name2read)
        self.IsModelConstructed = True

    def get_coef(self):
        if self.IsModelConstructed:
            return self.reg.coef_
        else:
            print('Model has not yet been constructed!')
            return None

    def get_intercept(self):
        if self.IsModelConstructed:
            return self.reg.intercept_
        else:
            print('Model has not yet been constructed!')
            return None

    def get_mse(self, x_true, y_true):
        if self.IsModelConstructed:
            return mean_squared_error(y_true, self.reg.predict(x_true))
        else:
            print('Model has not yet been constructed!')
            return None

    def get_r2_score(self, x_true, y_true):
        '''
        R^2 (coefficient of determination) regression score function
        Best possible score is 1.0.
        '''
        if self.IsModelConstructed:
            return r2_score(y_true, self.reg.predict(x_true))
        else:
            print('Model has not yet been constructed!')
            return None

    def k_fold_cross_val(self, folds, X, y):
        n = len(X)
        kf = KFold(n_splits=folds, shuffle=True, random_state=1)
        kf_dict_train = dict([("fold_%s" % i, []) for i in range(1, folds + 1)])
        kf_dict_test = dict([("fold_%s" % i, []) for i in range(1, folds + 1)])

        fold = 0
        for train_index, test_index in kf.split(X):
            fold += 1
            print("work on fold:",  fold)

            X_train, X_test = X.ix[train_index], X.ix[test_index]
            y_train, y_test = y.ix[train_index], y.ix[test_index]

            self.train(X_train, y_train)
            y_pred_train = self.predict(X_train)
            y_pred = self.predict(X_test)
            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred)

            kf_dict_train["fold_%s" % fold].append(train_mse)
            kf_dict_test["fold_%s" % fold].append(test_mse)

            # Convert these lists into numpy arrays to perform averaging
            kf_dict_train["fold_%s" % fold] = np.array(kf_dict_train["fold_%s" % fold])
            kf_dict_test["fold_%s" % fold] = np.array(kf_dict_test["fold_%s" % fold])

        # store summary rmse/mse for each fold
        training_folds_mse = np.hstack([v for k, v in kf_dict_train.items() if 'fold' in k])
        testing_folds_mse = np.hstack([v for k, v in kf_dict_test.items() if 'fold' in k])
        training_folds_rmse = np.hstack([np.sqrt(v) for k, v in kf_dict_train.items() if 'fold' in k])
        testing_folds_rmse = np.hstack([np.sqrt(v) for k, v in kf_dict_test.items() if 'fold' in k])

        feature_error_scores = {'training_folds_rmse': training_folds_rmse,
                                'testing_folds_rmse': testing_folds_rmse,
                                'training_folds_mse': training_folds_mse,
                                'testing_folds_mse': testing_folds_mse,
                                'training_avg_rmse': np.mean(training_folds_rmse),
                                'testing_avg_rmse': np.mean(testing_folds_rmse),
                                'training_avg_mse': np.mean(training_folds_mse),
                                'testing_avg_mse': np.mean(testing_folds_mse)}

        return feature_error_scores

    def visualization(self, x_test, y_test, y_lower=None, y_upper=None, fig2save=None):
        fig = plt.figure()
        ax = plt.subplot(1, 1, 1)
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(18)

        plt.plot(x_test, y_test, 'r.', markersize=10, label=u'Observations')
        # plt.scatter(x_test, self.predict(x_test), 'r.', label=u'Observations')
        # plt.plot(x_test, self.predict(x_test), 'b-', label=u'Prediction')
        if y_lower is not None and y_upper is not None:
            plt.fill(np.concatenate([x_test, x_test[::-1]]),
                     np.concatenate([y_lower, y_upper[::-1]]),
                     alpha=.5, fc='b', ec='None', label='95% confidence interval')
        plt.legend(loc='upper left')
        if fig2save is None:
            plt.show()
        else:
            plt.savefig(fig2save)