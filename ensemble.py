# coding: utf-8

from utils import *
from sklearn.metrics import roc_auc_score, recall_score, precision_score, log_loss, f1_score

if __name__ == '__main__':

    # average ensemble xgboost weight

    data = '116'
    # ground truth
    _, y_test = load_svmlight_file('data/0820_{}_test.libsvm'.format(data))

    # xgboost
    logging.info('load xgboost result')
    with open('res/xgboost.res', 'rb') as f:
        y_pred_xgboost = pickle.load(f)

    """
    y_label = [1 if i >= 0.5 else 0 for i in y_pred_xgboost]
    auc = roc_auc_score(y_test, y_pred_xgboost)
    recall = recall_score(y_test, y_label)
    precision = precision_score(y_test, y_label)
    loss = log_loss(y_test, y_pred_xgboost)
    f1 = f1_score(y_test, y_label)
    logging.info(
        'auc: {:.4f}, precision: {:.4f}, recall: {:.4f}, loss: {:.4f}, f1: {:.4f}'.format(
            auc, precision, recall, loss, f1))

    pass
    """

    # deepfm
    logging.info('load lldeepfm result')
    with open('res/lldeepfm.res', 'rb') as f:
        y_pred_deepfm = pickle.load(f)

    assert len(y_pred_xgboost) == len(y_pred_deepfm)
    # average

    for weight_xgboost in range(0, 21):
        y_average = [weight_xgboost * y_pred_xgboost[i] / 20 + (1 - weight_xgboost / 20) * y_pred_deepfm[i]
                     for i in range(len(y_pred_xgboost))]

        y_average_label = [1 if i >= 0.5 else 0 for i in y_average]
        # auc
        auc = roc_auc_score(y_test, y_average)
        recall = recall_score(y_test, y_average_label)
        precision = precision_score(y_test, y_average_label)
        loss = log_loss(y_test, y_average)
        f1 = f1_score(y_test, y_average_label)
        logging.info('weight of xgboost:{}, after ensemble, auc: {:.4f}, precision: {:.4f}, recall: {:.4f}, loss: {:.4f}, f1: {:.4f}'.format(
            weight_xgboost, auc, precision, recall, loss, f1))
