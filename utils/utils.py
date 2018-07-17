# coding: utf-8
import logging
import pandas as pd
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from config import *
import os.path
from scipy import sparse
import numpy as np


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


def get_df_from_raw():
    """
    :return: training data frame & test data frame
    """
    logging.info('loading raw data')
    df_train = pd.read_csv(config['train_data'], sep='\t', header=None,
                           names=config['all_field'])

    df_test = pd.read_csv(config['test_data'], sep='\t', header=None,
                          names=config['all_field'])

    # save feature in libsvm format to file and reload it

    if not os.path.exists(config['train_data'] + '.libsvm'):
        df_libsvm_train = pd.DataFrame(df_train, columns=['tag', 'feature'])
        df_libsvm_train.to_csv(config['train_data'] + '.libsvm', index=False, header=False, sep='\t')

    logging.info('converting train libsvm data')
    x_libsvm_train, y_libsvm_train = load_svmlight_file(config['train_data'] + '.libsvm')

    if not os.path.exists(config['test_data'] + '.libsvm'):
        df_libsvm_test = pd.DataFrame(df_test, columns=['tag', 'feature'])
        df_libsvm_test.to_csv(config['test_data'] + '.libsvm', index=False, header=False, sep='\t')

    logging.info('converting test libsvm data')
    x_libsvm_test, y_libsvm_test = load_svmlight_file(config['test_data'] + '.libsvm')

    # if config['libsvm_feature_only']:
    #     return x_libsvm_train, y_libsvm_train, x_libsvm_test, y_libsvm_test
    # else:
    # append extra feature
    # df_all = pd.concat([df_train, df_test])
    #
    # logging.info('encode vector feature')
    #
    # cv = CountVectorizer()
    # fea = config['vector_feature'][0]
    # cv.fit(df_all[fea])
    # train_x = cv.transform(df_train[fea])
    # test_x = cv.transform(df_test[fea])

    # for feature in config['vector_feature'][1:]:
    #     cv.fit(df_all[feature])
    #     train_a = cv.transform(df_train[feature])
    #     test_a = cv.transform(df_test[feature])
    #     train_x = sparse.hstack((train_x, train_a))
    #     test_x = sparse.hstack((test_x, test_a))
    #
    # logging.info('vector feature encoded')
    # logging.info('concat extra feature with original feature')

    # train_x = sparse.hstack((train_x, x_libsvm_train))
    # test_x = sparse.hstack((test_x, x_libsvm_test))

    # train_x = train_x.tocsr().astype(np.float64)
    # test_x = test_x.tocsr().astype(np.float64)

    # append game vector information
    logging.info('generating game feature')

    logging.info('loading game vector')

    df_game = pd.read_csv(config['game_vector'], sep='\t', header=None, names=['game_id', 'feature'])

    if os.path.exists(config['train_game_feature']):
        train_game_df = pd.read_csv(config['train_game_feature'], header=None)
    else:
        logging.info('generate game feature for training samples')
        train_game = get_game_mean_feature(df_train, df_game)

        logging.info('save game feature for training samples')
        train_game_df = pd.DataFrame(train_game)
        train_game_df.to_csv(config['train_game_feature'], header=None, index=False)

    if os.path.exists(config['test_game_feature']):
        test_game_df = pd.read_csv(config['test_game_feature'], header=None)
    else:
        logging.info('generate game feature for test samples')
        test_game = get_game_mean_feature(df_test, df_game)

        logging.info('save game feature for test samples')
        test_game_df = pd.DataFrame(test_game)
        test_game_df.to_csv(config['test_game_feature'], header=None, index=False)

    train_x = sparse.hstack((x_libsvm_train, train_game_df))
    test_x = sparse.hstack((x_libsvm_test, test_game_df))

    logging.info('save extended feature')
    dump_svmlight_file(train_x, y_libsvm_train, config['train_data'] + '_game.libsvm')
    dump_svmlight_file(test_x, y_libsvm_test, config['test_data'] + '_game.libsvm')

    return train_x, y_libsvm_train, test_x, y_libsvm_test


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

    with open(out_file, 'wb') as fp:
        for line in enumerate(data):
            fp.write(line + '\n')

    logging.info('conversion done')
