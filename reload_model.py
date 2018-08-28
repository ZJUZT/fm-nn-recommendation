# coding: utf-8

from model import LLDeepFM
from utils import *

if __name__ == '__main__':
    game = [116]
    train_data_list = ['data/0820_{}_train'.format(i) for i in game]
    test_data_list = ['data/0820_{}_test'.format(i) for i in game]

    for i in range(len(train_data_list)):
        ll_deep_fm = LLDeepFM.LLDeepFM(config['field_size'], config['raw_feature_size'], config['feature_size'], verbose=True, use_cuda=False,
                                       weight_decay=0.0001, use_fm=True, use_ffm=False, use_deep=True)

        ll_deep_fm.load_model()
        xi_test, xv_test, y_test = get_deep_fm_data_format(test_data_list[i] + '.libsvm', config['field_info'])
        x_test, _ = load_svmlight_file(test_data_list[i] + '.libsvm')
        y_pred_deepfm = ll_deep_fm.predict_proba(xi_test, xv_test, x_test)
