# coding: utf-8
import logging
import pandas as pd
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.feature_extraction.text import CountVectorizer

from config import *
import os.path
from scipy import sparse
import numpy as np
import pickle

from matplotlib import pyplot as plt


def init_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('wxg_game_rp.log')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s]%(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)


def get_game_mean_feature_by_attribute(df_game, field):
    """
    :param df_game: game
    :param field: login/pay.. list
    :return: list
    """
    reg_list = field.split(' ')
    reg_list = [int(i) for i in reg_list if i != '']

    game_feature_dim = 30

    if len(reg_list) == 0:
        res = [0 for _ in range(game_feature_dim)]
    else:
        data = df_game.loc[df_game.game_id.isin(reg_list), 'feature']
        mean_fea = []
        for d in data:
            mean_fea.append([float(i) for i in d.split(' ')])
        if len(mean_fea) == 0:
            res = [0 for _ in range(game_feature_dim)]
        else:
            res = np.mean(mean_fea, axis=0).tolist()

    return res


def get_game_mean_feature(df, df_game):
    """
    :param df_game: game vector df
    :param df: data frame train/test
    :return: mean game vector according to reg/login/pay/biz/center list
    """
    game_feature = []

    count = 0
    for row in df.itertuples():
        if count % 10000 == 0:
            logging.debug('precessed {} samples'.format(count))

        tmp = []
        # reg info
        tmp += get_game_mean_feature_by_attribute(df_game, row.reg)

        # login info
        tmp += get_game_mean_feature_by_attribute(df_game, row.login)

        # pay info
        tmp += get_game_mean_feature_by_attribute(df_game, row.pay)

        # biz info
        tmp += get_game_mean_feature_by_attribute(df_game, row.biz)

        # center info
        tmp += get_game_mean_feature_by_attribute(df_game, row.center)

        game_feature.append(tmp)
        count += 1

    return game_feature


def get_df_from_raw(train_data, test_data):
    """
    :return: training data frame & test data frame
    """
    logging.info('loading raw data')
    df_train = pd.read_csv(train_data, sep='\t', header=None,
                           names=config['all_field'])

    df_test = pd.read_csv(test_data, sep='\t', header=None,
                          names=config['all_field'])

    # save feature in libsvm format to file and reload it

    if not os.path.exists(train_data + '.libsvm'):
        df_libsvm_train = pd.DataFrame(df_train, columns=['tag', 'feature'])
        df_libsvm_train.to_csv(train_data + '.libsvm', index=False, header=False, sep='\t')

    logging.info('converting train libsvm data')
    x_libsvm_train, y_libsvm_train = load_svmlight_file(train_data + '.libsvm')

    if not os.path.exists(test_data + '.libsvm'):
        df_libsvm_test = pd.DataFrame(df_test, columns=['tag', 'feature'])
        df_libsvm_test.to_csv(test_data + '.libsvm', index=False, header=False, sep='\t')

    logging.info('converting test libsvm data')
    x_libsvm_test, y_libsvm_test = load_svmlight_file(test_data + '.libsvm')

    assert x_libsvm_train.shape[1] == x_libsvm_test.shape[1]
    dim_ori = x_libsvm_train.shape[1]
    logging.info('original train_x dimension: {}'.format(x_libsvm_train.shape))
    logging.info('original test_x dimension: {}'.format(x_libsvm_test.shape))

    # df_all = pd.concat([df_train, df_test])
    # logging.info('encode vector feature')
    # cv = CountVectorizer(min_df=1000)
    # fea = config['vector_feature'][0]
    # cv.fit(df_all[fea])
    # train_x = cv.transform(df_train[fea])
    # test_x = cv.transform(df_test[fea])
    #
    # for feature in config['vector_feature'][1:]:
    #     cv.fit(df_all[feature])
    #     train_a = cv.transform(df_train[feature])
    #     test_a = cv.transform(df_test[feature])
    #     train_x = sparse.hstack((train_x, train_a))
    #     test_x = sparse.hstack((test_x, test_a))
    #
    # logging.info('vector feature encoded')
    # logging.info('concat extra feature with original feature')
    #
    # train_x = sparse.hstack((train_x, x_libsvm_train))
    # test_x = sparse.hstack((test_x, x_libsvm_test))
    #
    # train_x = train_x.tocsr().astype(np.float64)
    # test_x = test_x.tocsr().astype(np.float64)
    #
    # logging.info('saving feature with log info')
    #
    # assert train_x.shape[1] == test_x.shape[1]
    # dim_with_log_info = train_x.shape[1]
    # logging.info('sparse train_x dimension: {}'.format(train_x.shape))
    # logging.info('sparse test_x dimension: {}'.format(test_x.shape))
    #
    # dump_svmlight_file(train_x, y_libsvm_train, train_data + '_sparse.libsvm')
    # dump_svmlight_file(test_x, y_libsvm_test, test_data + '_sparse.libsvm')

    # append game vector information
    logging.info('generating game feature')

    logging.info('loading game vector')

    if not os.path.exists(train_data + '_game.libsvm'):
        df_game = pd.read_csv(config['game_vector'], sep='\t', header=None, names=['game_id', 'feature'])

        if os.path.exists(train_data+'_game_vector'):
            train_game_df = pd.read_csv(train_data+'_game_vector', header=None)
        else:
            logging.info('generate game feature for training samples')
            train_game = get_game_mean_feature(df_train, df_game)

            logging.info('save game feature for training samples')
            train_game_df = pd.DataFrame(train_game)
            train_game_df.to_csv(train_data+'_game_vector', header=None, index=False)

        if os.path.exists(test_data+'_game_vector'):
            test_game_df = pd.read_csv(test_data +'_game_vector', header=None)
        else:
            logging.info('generate game feature for test samples')
            test_game = get_game_mean_feature(df_test, df_game)

            logging.info('save game feature for test samples')
            test_game_df = pd.DataFrame(test_game)
            test_game_df.to_csv(test_data+'_game_vector', header=None, index=False)

        train_x = sparse.hstack((x_libsvm_train, train_game_df))
        test_x = sparse.hstack((x_libsvm_test, test_game_df))

        logging.info('save extended feature')
        dump_svmlight_file(train_x, y_libsvm_train, train_data + '_game.libsvm')
        dump_svmlight_file(test_x, y_libsvm_test, test_data + '_game.libsvm')
    else:
        train_x, _ = load_svmlight_file(train_data + '_game.libsvm')
        test_x, _ = load_svmlight_file(test_data + '_game.libsvm')

    assert train_x.shape[1] == test_x.shape[1]
    dim_with_game = train_x.shape[1]
    logging.info('train_x_game dimension: {}'.format(train_x.shape))
    logging.info('test_x_game dimension: {}'.format(test_x.shape))

    return dim_ori, dim_with_game

    # return train_x, y_libsvm_train, test_x, y_libsvm_test


def convert_to_ffm_format(in_file, out_file, field_info):
    """
    convert data in libsvm format to ffm format
    label   field1:feat1:val1   field2:feat2:val2 ...
    :param field_info: dict indicating which field the feature belongs
    :param in_file: input data path
    :param out_file:  output data path
    :return:
    """

    logging.info('convert {} to ffm format in {}'.format(in_file, out_file))
    logging.info('load libsvm file')
    x, y = load_svmlight_file(in_file)
    m, n = x.shape
    data = []

    for i in range(m):
        if i % 10000 == 0:
            logging.debug('processed {} samples'.format(i))
        tmp = [y[i]]
        index, value = x[i].nonzero()[1], x[i].data
        for j in range(len(index)):
            tmp.append('{}:{}:{}'.format(field_info[index[j]], index[j], value[j]))
        data.append(' '.join([str(i) for i in tmp]))

    with open(out_file, 'w') as fp:
        for line in data:
            fp.write(line + '\n')

    logging.info('conversion done')


def get_deep_fm_data_format(in_file):
    """
    convert data in libsvm format into deep fm format
    :param in_file: file path in libsvm format
    :return: xi, xv, y
    """

    if os.path.exists(in_file + '_xi'):
        logging.info('load data from existing file')
        with open(in_file + '_xi', 'rb') as fp:
            xi = pickle.load(fp)

        with open(in_file + '_xv', 'rb') as fp:
            xv = pickle.load(fp)

        with open(in_file + '_y', 'rb') as fp:
            y = pickle.load(fp)
    else:
        logging.info('convert {} to deep fm format'.format(in_file))
        logging.info('load libsvm file')
        x, y = load_svmlight_file(in_file)
        m, n = x.shape
        xi = []
        xv = []

        for i in range(m):
            if i % 10000 == 0:
                logging.debug('processed {} samples'.format(i))
            index, value = x[i].nonzero()[1].tolist(), list(filter(lambda a: a != 0, x[i].data.tolist()))
            assert len(index) == len(value)
            xi.append(index)
            xv.append(value)

        logging.info('conversion done')
        logging.info('save data')

        with open(in_file + '_xi', 'wb') as fp:
            pickle.dump(xi, fp)

        with open(in_file + '_xv', 'wb') as fp:
            pickle.dump(xv, fp)

        with open(in_file + '_y', 'wb') as fp:
            pickle.dump(y, fp)

    return xi, xv, y


def draw_metrics(metric_train_list, metric_test_list, model_list, metric_name, baseline_train, baseline_test, baseline_model, fig_name, save_path):
    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 5))
    color = ['coral', 'green']
    for i in range(len(metric_train_list)):
        ax.plot(metric_train_list[i], linewidth=1.5, color=color[i], linestyle='--', marker='*', label='{}_train'.format(model_list[i], metric_name))
        ax.plot(metric_test_list[i], linewidth=1.5, color=color[i], marker='o', label='{}_test'.format(model_list[i], metric_name))

    # draw baseline
    ax.axhline(baseline_train, color='grey', linestyle='--', label=baseline_model+'_train')
    ax.axhline(baseline_test, color='firebrick', label=baseline_model+'_test')
    ax.set_title(fig_name)
    ax.set_xlabel('epoch')
    ax.set_ylabel(metric_name)
    ax.grid(True)
    ax.legend()
    fig.savefig(save_path)


if __name__ == '__main__':
    metric_train_list = [
        [0.7, 0.8, 0.9],
        [0.8, 0.9, 0.95]
    ]

    metric_test_list = [
        [0.6, 0.7, 0.8],
        [0.75, 0.85, 0.92]
    ]

    model_list = ['FM', 'DeepFM']
    metric_name = 'auc'

    baseline_train = 0.5
    baseline_test = 0.4

    baseline_model = 'XgBoost'
    fig_name = 'test'
    save_path = '../fig/test.pdf'

    draw_metrics(metric_train_list,
                 metric_test_list,
                 model_list,
                 metric_name,
                 baseline_train,
                 baseline_test,
                 baseline_model,
                 fig_name, save_path)
