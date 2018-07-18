# coding: utf-8
import xgboost as xgb
from .Base_Model import BaseModel
from utils import *


class XGBModel(BaseModel):
    def __init__(self):
        self.clf = None

    def fit(self, train_x, train_y, valid_x, valid_y):
        logging.info('start XGB training')
        self.clf = xgb.XGBClassifier(
            max_depth=config['xgb']['max_depth'],
            reg_lambda=config['xgb']['reg_lambda'],
            min_child_weight=config['xgb']['min_child_weight'],
            objective=config['xgb']['objective'],
            n_jobs=config['xgb']['n_jobs'],
            n_estimators=config['xgb']['n_estimators'],
            subsample=config['xgb']['subsample'],
            colsample_bytree=config['xgb']['colsample_bytree'],
        )

        self.clf.fit(train_x, train_y,
                     eval_set=[(train_x, train_y), (valid_x, valid_y)],
                     eval_metric=config['xgb']['eval_metric'],
                     early_stopping_rounds=config['xgb']['early_stopping_rounds'])

    def predict(self, x):
        return self.clf.predict_proba(x)[:, 1]

    def dump(self, path):
        pass

    def load(self, path):
        pass
