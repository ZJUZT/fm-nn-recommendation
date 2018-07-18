# coding: utf-8
import lightgbm as lgb
from .Base_Model import BaseModel
from utils import *


class LGBModel(BaseModel):
    def __init__(self):
        self.clf = None

    def fit(self, train_x, train_y, valid_x, valid_y):
        """
        :param train_x: train samples
        :param train_y: train labels
        :param valid_x: validation samples
        :param valid_y: validation labels
        :return:
        """

        logging.info('start LGB training')
        self.clf = lgb.LGBMClassifier(
            boosting_type=config['lgb']['boosting_type'],
            num_leaves=config['lgb']['num_leaves'],
            reg_alpha=config['lgb']['reg_alpha'],
            reg_lambda=config['lgb']['reg_lambda'],
            max_depth=config['lgb']['max_depth'],
            n_estimators=config['lgb']['n_estimators'],
            objective=config['lgb']['objective'],
            subsample=config['lgb']['subsample'],
            colsample_bytree=config['lgb']['colsample_bytree'],
            subsample_freq=config['lgb']['subsample_freq'],
            learning_rate=config['lgb']['learning_rate'],
            min_child_weight=config['lgb']['min_child_weight'],
            random_state=config['lgb']['random_state'],
            n_jobs=config['lgb']['n_jobs'],
            is_unbalance=config['lgb']['is_unbalance']
        )

        self.clf.fit(train_x, train_y,
                     eval_set=[(train_x, train_y), (valid_x, valid_y)],
                     eval_metric=config['lgb']['eval_metric'],
                     early_stopping_rounds=config['lgb']['early_stopping_rounds'])

    def predict(self, x):
        return self.clf.predict_proba(x)[:, 1]

    def dump(self, path):
        pass

    def load(self, path):
        pass
