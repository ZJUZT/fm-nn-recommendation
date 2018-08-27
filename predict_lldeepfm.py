# coding: utf-8

from utils import *
from model import LLDeepFM
from model import XGBModel

if __name__ == '__main__':

    game = [116]
    train_data_list = ['data/0820_{}_train'.format(i) for i in game]
    test_data_list = ['data/0820_{}_test'.format(i) for i in game]

    for i in range(len(train_data_list)):
        logging.info('evaluation on: {}'.format(train_data_list[i]))

        metric_train_auc = []
        metric_test_auc = []
        metric_train_log_loss = []
        metric_test_log_loss = []

        model_list = ['FM', 'DeepFM']

        # get_df_from_raw(train_data_list[i], test_data_list[i])

        # fm
        xi_train, xv_train, y_train = get_deep_fm_data_format(train_data_list[i] + '.libsvm', config['field_info'])
        xi_test, xv_test, y_test = get_deep_fm_data_format(test_data_list[i] + '.libsvm', config['field_info'])

        x_train, _ = load_svmlight_file(train_data_list[i] + '.libsvm')
        x_test, _ = load_svmlight_file(test_data_list[i] + '.libsvm')

        # deep fm
        ll_deep_fm = LLDeepFM.LLDeepFM(config['field_size'], config['feature_size'], verbose=True, use_cuda=False,
                                weight_decay=0.0001, use_fm=True, use_ffm=False, use_deep=True)
        train_auc, train_loss, valid_auc, valid_loss = \
            ll_deep_fm.fit(xi_train, xv_train, y_train, x_train, xi_test, xv_test, y_test, x_test, adaptive_anchor=True,
                           early_stopping=True,
                           save_path='dump_model/lldeepfm.snapshot')

        logging.info('validating')
        y_pred_deepfm = ll_deep_fm.predict_proba(xi_test, xv_test, x_test)

        # dump deep_fm result
        with open('res/lldeepfm.res', 'wb') as f:
            pickle.dump(y_pred_deepfm, f)

