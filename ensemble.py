# coding: utf-8

from utils import *
from sklearn.metrics import roc_auc_score

if __name__ == '__main__':

    # average ensemble xgboost weight

    # ground truth
    _, y_test = load_svmlight_file('data/71_test.libsvm')

    # xgboost
    logging.info('load xgboost result')
    with open('res/xgboost_result', 'rb') as f:
        y_pred_xgboost = pickle.load(f)

    # deepfm
    logging.info('load deepfm result')
    with open('res/deepfm_result', 'rb') as f:
        y_pred_deepfm = pickle.load(f)

    assert len(y_pred_xgboost) == len(y_pred_deepfm)
    # average

    for weight_xgboost in range(0, 20):
        y_average = [weight_xgboost * y_pred_xgboost[i] / 20 + (1-weight_xgboost/20) * y_pred_deepfm[i]
                     for i in range(len(y_pred_xgboost))]

        # auc
        auc = roc_auc_score(y_test, y_average)
        logging.info('weight of xgboost:{}, auc after ensemble: {:.4f}'.format(weight_xgboost, auc))
