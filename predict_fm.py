# coding: utf-8
import xlearn as xl

# create fm
fm_model = xl.create_fm()

# set training data and test data
fm_model.setTrain('data/0820_71_train.libsvm')
fm_model.setValidate('data/0820_71_test.libsvm')

# set hyper-parameters
param = {'task': 'binary',
         'lr': 0.02,
         'lambda': 0,
         'metric': 'auc',
         'k': 10,
         'epoch': 200}

# train model
fm_model.fit(param, './model_dump')

# predict
# fm_model.setTest('data/130_test.libsvm')
# fm_model.predict('./model_dump', 'output.txt')
