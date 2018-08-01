# coding: utf-8
import xlearn as xl

# create fm
ffm_model = xl.create_ffm()

# set training data and test data
ffm_model.setTrain('data/130_train.ffm')
ffm_model.setValidate('data/130_test.ffm')


# set hyper-parameters

param = {'task': 'binary',
         'lr': 0.1,
         'lambda': 0.001,
         'metric': 'auc',
         'k': 10,
         'epoch': 100}

# train model
ffm_model.fit(param, './model_dump')

# predict
ffm_model.setTest('data/130_test.ffm')
ffm_model.predict('./model_dump', 'output.txt')
