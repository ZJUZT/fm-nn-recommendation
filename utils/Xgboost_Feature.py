# coding: utf-8
from utils import *
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost.sklearn import XGBClassifier
import numpy as np
from scipy.sparse import hstack, vstack



class XgboostFeature():
    # 可以传入xgboost的参数
    # 常用传入特征的个数 即树的个数 默认30
    def __init__(self, n_estimators=30,
                 learning_rate=0.1,
                 max_depth=7,
                 min_child_weight=1,
                 gamma=0.3,
                 subsample=0.8,
                 colsample_bytree=0.8,
                 objective='binary:logistic',
                 nthread=2,
                 scale_pos_weight=1,
                 reg_alpha=1e-05,
                 reg_lambda=1,
                 seed=2018):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.objective = objective
        self.nthread = nthread
        self.scale_pos_weight = scale_pos_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.seed = seed

    ##切割训练
    def fit_model_split(self, train_data_path, test_data_path):
        # X_train_1用于生成模型  X_train_2用于和新特征组成新训练集合

        logging.info('split train data for gbdt model and feature transformation separately')
        logging.info('load data')
        X_train, y_train = load_svmlight_file(train_data_path)
        X_test, y_test = load_svmlight_file(test_data_path)

        X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(X_train, y_train, test_size=0.6, random_state=0)
        clf = XGBClassifier(
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            objective=self.objective,
            nthread=self.nthread,
            scale_pos_weight=self.scale_pos_weight,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            seed=self.seed)
        clf.fit(X_train_1, y_train_1,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                eval_metric='auc')
        # y_pre= clf.predict(X_train_2)
        # y_pro= clf.predict_proba(X_train_2)[:,1]
        # logging.info("pred_leaf=T AUC Score: {:.4f}".format(metrics.roc_auc_score(y_train_2, y_pro)))

        logging.info('transformed feature and one-hot encoding')
        new_feature = clf.apply(X_train_2)
        new_feature_test = clf.apply(X_test)
        # one-hot
        enc = OneHotEncoder()
        enc.fit(list(new_feature) + list(new_feature_test))
        new_feature = enc.transform(new_feature)
        X_train_new2 = hstack([X_train_2, new_feature])
        new_feature_test = enc.transform(new_feature_test)
        X_test_new = hstack([X_test, new_feature_test])

        logging.info('transformed training data dimension: {}'.format(X_train_new2.shape))
        logging.info('transformed test data dimension: {}'.format(X_test_new.shape))

        logging.info('save xgboost extended feature')
        dump_svmlight_file(X_train_new2, y_train_2, train_data_path+'_xgboost')
        dump_svmlight_file(X_test_new, y_test, test_data_path+'_xgboost')

        # return X_train_new2, y_train_2, X_test_new, y_test

    ##整体训练
    def fit_model(self, train_data_path, test_data_path, is_concat=True):
        clf = XGBClassifier(
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            objective=self.objective,
            nthread=self.nthread,
            scale_pos_weight=self.scale_pos_weight,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            seed=self.seed)

        logging.info('load data')
        x_train, y_train = load_svmlight_file(train_data_path)
        x_test, y_test = load_svmlight_file(test_data_path)

        clf.fit(x_train, y_train,
                eval_set=[(x_train, y_train), (x_test, y_test)],
                eval_metric='auc')
        y_pro = clf.predict_proba(x_test)[:, 1]
        logging.info("pred_leaf=T  AUC Score: {:.4f}".format(metrics.roc_auc_score(y_test, y_pro)))
        new_feature = clf.apply(x_train)
        new_feature_test = clf.apply(x_test)

        # one hot
        enc = OneHotEncoder()
        enc.fit(new_feature)
        new_feature = enc.transform(new_feature)
        new_feature_test = enc.transform(new_feature_test)

        if is_concat:
            x_train_new = hstack([x_train, new_feature])
            x_test_new = hstack([x_test, new_feature_test])
        else:
            x_train_new = new_feature
            x_test_new = new_feature_test

        logging.info('transformed training data dimension: {}'.format(x_train_new.shape))
        logging.info('transformed test data dimension: {}'.format(x_test_new.shape))

        logging.info('save xgboost extended feature')
        dump_svmlight_file(x_train_new, y_train, train_data_path + '_xgboost')
        dump_svmlight_file(x_test_new, y_test, test_data_path + '_xgboost')
