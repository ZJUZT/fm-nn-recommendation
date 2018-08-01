# coding: utf-8
import xlearn as xl

# create fm
fm_model = xl.create_fm()

# set training data and test data
fm_model.setTrain('data/130_train.libsvm')
fm_model.setValidate('data/130_test.libsvm')

# set hyper-parameters
param = {'task': 'binary',
         'lr': 0.1,
         'lambda': 0.001,
         'metric': 'auc',
         'k': 100,
         'epoch': 100}

# train model
fm_model.fit(param, './model_dump')

# predict
fm_model.setTest('data/130_test.libsvm')
fm_model.predict('./model_dump', 'output.txt')
