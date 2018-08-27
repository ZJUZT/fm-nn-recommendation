# coding: utf-8

from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

if __name__ == '__main__':

    data = '116'

    # filter data
    file_path = 'data/0820_{}'.format(data)
    out_path = 'data/0820_{}.libsvm'.format(data)
    with open(file_path, 'r') as f, open(out_path, 'w') as f_out:
        for line in f.readlines():
            tokens = line.strip().split('\t')
            f_out.write(' '.join(tokens[1:]) + '\n')

    # split to train and test data set
    X, Y = load_svmlight_file('data/0820_{}.libsvm'.format(data))

    # do normalization
    X = normalize(X, axis=0)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

    dump_svmlight_file(X_train, Y_train, 'data/0820_{}_train.libsvm'.format(data))
    dump_svmlight_file(X_test, Y_test, 'data/0820_{}_test.libsvm'.format(data))