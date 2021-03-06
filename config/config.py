# coding: utf-8
"""
configuration for:
1. data path
2. feature encoding (one-hot & continuous & word)
3. model params
"""

config = {}

# config['train_data'] = 'data/130_train'
# config['test_data'] = 'data/130_test'
config['game_vector'] = 'data/vec30.txt'

# config['train_game_feature'] = 'data/130_train_game'
# config['test_game_feature'] = 'data/130_test_game'

"""
appid -- 游戏id
uin -- 用户id
reg -- 注册游戏list
login -- 登录游戏list
pay -- 支付游戏list
biz -- 游戏公众号点击list
center -- 游戏中心点击list
feature -- 其他特征 (libsvm format)
"""
config['all_field'] = [
    'appid',
    'uid',
    'reg',
    'login',
    'pay',
    'biz',
    'center',
    'tag',
    'feature'
]

config['one_hot_feature'] = [
    'uid'
]

config['vector_feature'] = [
    'reg',
    'login',
    'pay',
    'biz',
    'center',
]

config['libsvm_feature'] = [
    'feature'
]

config['libsvm_feature_only'] = True

# parameters for lgb
lgb = {
    'boosting_type': 'gbdt',
    'num_leaves': 127,
    'reg_alpha': 0.0,
    'reg_lambda': 1,
    'max_depth': -1,
    'n_estimators': 50,
    'objective': 'binary',
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'subsample_freq': 1,
    'learning_rate': 0.05,
    'min_child_weight': 10,
    'random_state': 2018,
    'n_jobs': 6,
    'is_unbalance': False,
    'eval_metric': 'auc',
    'early_stopping_rounds': 100
}

config['lgb'] = lgb

xgb = {
    'max_depth': 15,
    'reg_lambda': 0.1,
    'min_child_weight': 50,
    'n_estimators': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eta': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'n_jobs': 2,
    'early_stopping_rounds': 100
}
0
config['xgb'] = xgb

# field info
# fields_index = [
#     # (0, 2192)
#     (0, 50),
#     # (28, 32),
#     # (32, 33),
#     # (33, 34),
#     # (34, 50),
#     (50, 1950),
#     # (200, 350),
#     # (350, 500),
#     # (500, 650),
#     # (650, 800),
#     # (800, 950),
#     # (950, 1100),
#     # (1100, 1250),
#     # (1250, 1300),
#     # (1300, 1600),
#     # (1600, 1635),
#     # (1635, 1730),
#     # (1800, 1950)
# ]

# field_dict = {k[0]: k[1] for k in zip(fields_index, range(len(fields_index)))}
#
# field_info = {}
# for k, v in field_dict.items():
#     for i in range(k[0], k[1]):
#         field_info[i] = v

# field_info = [29, 33, 34, 35, 51, 201, 351, 501, 651, 801, 951, 1101, 1251, 1301, 1601, 1636, 1731, 1801, 1950]
# field_info = [35, 1951]
field_info = [50, 60, 70, 80, 90, 100, 120]
# field_info = [i for i in range(120)]

feature_size = [field_info[i] - field_info[i-1] if i > 0 else field_info[i] + 1 for i in range(len(field_info))]


config['field_info'] = field_info

config['field_size'] = len(field_info)
config['feature_size'] = feature_size

config['raw_feature_size'] = 119
