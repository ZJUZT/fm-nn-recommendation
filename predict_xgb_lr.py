# encoding: utf-8

import xlearn as xl

from utils import *

if __name__ == '__main__':
    # load data

    if not (os.path.exists('data/130_train_transformed.libsvm') and os.path.exists('data/130_test_transformed.libsvm')):
        logging.info('generating transformed data')
        logging.info('loading initial libsvm data')
        x_train, y_train = load_svmlight_file('data/130_train_game.libsvm')
        x_test, y_test = load_svmlight_file('data/130_test_game.libsvm')

        model = XgboostFeature()
        x_train_new, y_train_new, x_test_new, y_test_new = model.fit_model(x_train, y_train, x_test, y_test, is_concat=True)

        # save data as libsvm format
        logging.info('dump transformed data')
        dump_svmlight_file(x_train_new, y_train_new, 'data/130_train_transformed.libsvm')
        dump_svmlight_file(x_test_new, y_test_new, 'data/130_test_transformed.libsvm')

    # logging.info('load test transformed feature')
    # x_test, y_test = load_svmlight_file(f='data/116_test_transformed.libsvm')
    #
    # logging.info('load training transformed feature')
    # x_train, y_train = load_svmlight_file('data/116_train_transformed.libsvm', n_features=x_test.shape[1])
    # # in case train and test dimension mismatch
    #
    # lr = LogisticRegression()
    # logging.info('training transformed feature with lr')
    # lr.fit(x_train, y_train)
    # y_predict = lr.predict_proba(x_test)[:, 1]
    # auc = roc_auc_score(y_test, y_predict)
    # logging.info('auc with transformed feature with LogisticRegression: {:.4f}'.format(auc))

    logging.info('training transformed feature with fm')
    fm_model = xl.create_fm()
    fm_model.disableEarlyStop()

    fm_model.setTrain('data/130_train_transformed.libsvm')
    fm_model.setValidate('data/130_test_transformed.libsvm')

    param = {'task': 'binary',
             'lr': 0.02,
             'lambda': 0.001,
             'metric': 'auc',
             'k': 30,
             'stop_window': 10,
             'epoch': 200}

    fm_model.fit(param, '../model_dump')
