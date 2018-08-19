import torch
from model import LLDeepFM
from config import *


def dump_model_to_file(model):
    """
    dump trained model in torch to file
    each deepfm model saved in a singe file/table
    whose format goes like:
    1. anchor point
    2. fm bias
    3. fm embedding (19 lines)
    4. input layer weight
    5. input layer bias
    6. hidden layer weight
    7. hidden layer bias
    :param model:
    :return:
    """
    anchor_num = 20
    fields_num = 19
    table_name = ['dump_model/deep_fm_' + str(i) for i in range(anchor_num)]

    for i in range(anchor_num):
        print('dumping model {}'.format(i))
        with open(table_name[i], 'w') as f:
            # anchor point
            value = model.get('anchor_points')[i]
            for v in value.tolist():
                f.write(str(v) + ' ')
            f.write('\n')

            # fm_bias
            value = model.get('bias').tolist()[i][0]
            f.write(str(value) + '\n')

            # fm_embedding
            for j in range(fields_num):
                value = model.get('fm_second_order_embeddings.{}.{}.weight'.format(i, j))
                row, col = value.shape
                value = value.tolist()
                for m in range(row):
                    for n in range(col):
                        f.write(str(value[m][n]) + ' ')
                f.write('\n')

            # input layer weight
            value = model.get('linear_1.{}.weight'.format(i))
            row, col = value.shape
            value = value.tolist()
            for m in range(row):
                for n in range(col):
                    f.write(str(value[m][n]) + ' ')

            f.write('\n')

            # input layer bias
            value = model.get('linear_1.{}.bias'.format(i))
            value = value.tolist()
            for j in range(len(value)):
                f.write(str(value[j]) + ' ')

            f.write('\n')

            # hidden layer weight
            value = model.get('linear_2.{}.weight'.format(i))
            row, col = value.shape
            value = value.tolist()
            for m in range(row):
                for n in range(col):
                    f.write(str(value[m][n]) + ' ')

            f.write('\n')

            # hidden layer bias
            value = model.get('linear_2.{}.bias'.format(i))
            value = value.tolist()
            for j in range(len(value)):
                f.write(str(value[j]) + ' ')


if __name__ == '__main__':
    res = torch.load('dump_model/lldeepfm.snapshot')
    dump_model_to_file(res)
