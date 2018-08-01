# coding: utf-8

from utils import *
from model import DeepFM
from model import XGBModel

if __name__ == '__main__':

    game = [130]
    train_data_list = ['data/{}_train'.format(i) for i in game]
    test_data_list = ['data/{}_test'.format(i) for i in game]

    for i in range(len(train_data_list)):

        logging.info('evaluation on: {}'.format(train_data_list[i]))

        metric_train_auc = []
        metric_test_auc = []
        metric_train_log_loss = []
        metric_test_log_loss = []

        model_list = ['FM', 'DeepFM']

        # logging.info('prepare data for XgBoost')
        # dim_ori, dim_with_game = get_df_from_raw(train_data_list[i], test_data_list[i])

        # XgBoost baseline
        # xgb_model = XGBModel()
        # x_train, y_train = load_svmlight_file(train_data_list[i]+'.libsvm')
        # x_test, y_test = load_svmlight_file(test_data_list[i]+'.libsvm')
        # res = xgb_model.fit(x_train, y_train, x_test, y_test)
        # del x_train, y_train, x_test, y_test
        #
        # baseline_train = res['validation_0']['auc'][-1]
        # baseline_test = res['validation_1']['auc'][-1]
        #
        # logging.info('XgBoost AUC (train): {}'.format(baseline_train))
        # logging.info('XgBoost AUC (test): {}'.format(baseline_test))
        # fm
        xi_train, xv_train, y_train = get_deep_fm_data_format(train_data_list[i] + '.libsvm', config['field_info'])
        xi_test, xv_test, y_test = get_deep_fm_data_format(test_data_list[i] + '.libsvm', config['field_info'])

        # deep_fm = DeepFM.DeepFM(1, [dim_ori], verbose=True, use_cuda=True,
        #                         weight_decay=0.0001, use_fm=True, use_ffm=False, use_deep=False)
        # train_auc, train_loss, valid_auc, valid_loss = \
        #     deep_fm.fit(xi_train, xv_train, y_train, xi_test, xv_test, y_test, ealry_stopping=True, refit=False)
        # metric_train_auc.append(train_auc)
        # metric_train_log_loss.append(train_loss)
        # metric_test_auc.append(valid_auc)
        # metric_test_log_loss.append(valid_loss)

        # deep fm
        deep_fm = DeepFM.DeepFM(1, config['feature_size'], verbose=True, use_cuda=True,
                                weight_decay=0.0001, use_fm=True, use_ffm=False, use_deep=True)
        train_auc, train_loss, valid_auc, valid_loss = \
            deep_fm.fit(xi_train, xv_train, y_train, xi_test, xv_test, y_test, early_stopping=True, refit=False)
        metric_train_auc.append(train_auc)
        metric_train_log_loss.append(train_loss)
        metric_test_auc.append(valid_auc)
        metric_test_log_loss.append(valid_loss)

        # draw metrics
        # auc
        # draw_metrics(metric_train_auc,
        #              metric_test_auc,
        #              model_list,
        #              'auc',
        #              baseline_train,
        #              baseline_test,
        #              'XgBoost',
        #              'evaluation on game {}'.format(game[i]),
        #              'fig/{}_auc.pdf'.format(game[i]))

        del xi_train, xv_train, y_train, xi_test, xv_test, y_test