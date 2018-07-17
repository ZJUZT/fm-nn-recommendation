# coding: utf-8

# from model import *
from utils import *
from config import *
# from sklearn.metrics import roc_auc_score

if __name__ == '__main__':

    # get_df_from_raw()
    # logging.info('use libsvm feature only: {}'.format(config['libsvm_feature_only']))
    # train_x, train_y, test_x, test_y = get_df_from_raw()
    # # model = LGBModel()
    # model = XGBModel()
    # model.fit(train_x, train_y, test_x, test_y)
    # pred_y = model.predict(test_x)
    # auc = roc_auc_score(test_y, pred_y)
    # logging.info('auc: {}'.format(auc))

    in_file = 'data/130_train.libsvm'
    out_file = 'data/130_train.ffm'

    convert_to_ffm_format(in_file, out_file, config['field_info'])
