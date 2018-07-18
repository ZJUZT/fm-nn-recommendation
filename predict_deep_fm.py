# coding: utf-8

from utils import *
from model import DeepFM

if __name__ == '__main__':
    # prepare data
    xi_train, xv_train, y_train = get_deep_fm_data_format('data/130_train.libsvm')
    xi_test, xv_test, y_test = get_deep_fm_data_format('data/130_test.libsvm')

    deep_fm = DeepFM.DeepFM(config['field_size'], config['feature_size'], verbose=True, use_cuda=True,
                            weight_decay=0.0001, use_fm=False, use_ffm=True, use_deep=True)
    deep_fm.fit(xi_train, xv_train, y_train, xi_test, xv_test, y_test, ealry_stopping=True, refit=True)
