# coding: utf-8

from utils import *
from model import AFM
from model import XGBModel

if __name__ == '__main__':

    game = [71]
    train_data_list = ['data/{}_train'.format(i) for i in game]
    test_data_list = ['data/{}_test'.format(i) for i in game]

    for i in range(len(train_data_list)):
        logging.info('evaluation on: {}'.format(train_data_list[i]))

        metric_train_auc = []
        metric_test_auc = []
        metric_train_log_loss = []
        metric_test_log_loss = []

        model_list = ['FM', 'DeepFM']

        xi_train, xv_train, y_train = get_deep_fm_data_format(train_data_list[i] + '.libsvm', config['field_info'])
        xi_test, xv_test, y_test = get_deep_fm_data_format(test_data_list[i] + '.libsvm', config['field_info'])

        # afm
        afm = AFM.AFM(config['field_size'], config['feature_size'], verbose=True, use_cuda=False,
                                weight_decay=0.0001, use_fm=True, use_ffm=False)
        train_auc, train_loss, valid_auc, valid_loss = \
            afm.fit(xi_train, xv_train, y_train, xi_test, xv_test, y_test, early_stopping=True, refit=False)

        logging.info('validating')
        y_pred_afm = afm.predict_proba(xi_test, xv_test)

        # dump deep_fm result
        with open('afm_result', 'wb') as f:
            pickle.dump(y_pred_afm, f)
