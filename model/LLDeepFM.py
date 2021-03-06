# -*- coding:utf-8 -*-

"""

A pytorch implementation of locally linear deepfm

Reference:
[1] DeepFM: A Factorization-Machine based Neural Network for CTR Prediction,
    Huifeng Guo, Ruiming Tang, Yunming Yey, Zhenguo Li, Xiuqiang He.
[2] Locally linear factorization machines
    Liu, Chenghao and Zhang, Teng and Zhao, Peilin and Zhou, Jun and Sun, Jianling

"""

from sklearn.metrics import roc_auc_score
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from sklearn.cluster import KMeans
import torch.backends.cudnn
from utils import *

"""
    网络结构部分
"""


class LLDeepFM(torch.nn.Module):
    """
    :parameter
    -------------
    field_size: size of the feature fields
    feature_sizes: a field_size-dim array, sizes of the feature dictionary
    embedding_size: size of the feature embedding
    is_shallow_dropout: bool, shallow part(fm or ffm part) uses dropout or not?
    dropout_shallow: an array of the size of 2, example:[0.5,0.5], the first element is for the-first order part and the second element is for the second-order part
    h_depth: deep network's hidden layers' depth
    deep_layers: a h_depth-dim array, each element is the size of corresponding hidden layers. example:[32,32] h_depth = 2
    is_deep_dropout: bool, deep part uses dropout or not?
    dropout_deep: an array of dropout factors,example:[0.5,0.5,0.5] h_depth=2
    deep_layers_activation: relu or sigmoid etc
    n_epochs: epochs
    batch_size: batch_size
    learning_rate: learning_rate
    optimizer_type: optimizer_type, 'adam', 'rmsp', 'sgd', 'adag'
    is_batch_norm：bool,  use batch_norm or not ?
    verbose: verbose
    weight_decay: weight decay (L2 penalty)
    random_seed: random_seed=950104 someone's birthday, my lukcy number
    use_fm: bool
    use_ffm: bool
    use_deep: bool
    loss_type: "logloss", only
    eval_metric: roc_auc_score
    use_cuda: bool use gpu or cpu?
    n_class: number of classes. is bounded to 1
    greater_is_better: bool. Is the greater eval better?


    Attention: only support logistics regression
    """

    def __init__(self, field_size, raw_feature_size, feature_sizes, embedding_size=4, anchor_num=100, nn_num=8, c=1e3,
                 is_shallow_dropout=True,
                 fm_first_order_used=False,
                 dropout_shallow=[0.5, 0.5],
                 h_depth=2, deep_layers=[32, 32], is_deep_dropout=True, dropout_deep=[0.5, 0.5, 0.5],
                 deep_layers_activation='relu', n_epochs=5, batch_size=256, learning_rate=0.02,
                 optimizer_type='adam', is_batch_norm=False, verbose=False, random_seed=950104, weight_decay=0.0,
                 use_fm=True, use_ffm=False, use_deep=True, loss_type='logloss', eval_metric=roc_auc_score,
                 use_cuda=True, n_class=1, greater_is_better=True
                 ):
        super(LLDeepFM, self).__init__()
        self.field_size = field_size
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.is_shallow_dropout = is_shallow_dropout
        self.dropout_shallow = dropout_shallow
        self.h_depth = h_depth
        self.deep_layers = deep_layers
        self.is_deep_dropout = is_deep_dropout
        self.dropout_deep = dropout_deep
        self.deep_layers_activation = deep_layers_activation
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.is_batch_norm = is_batch_norm
        self.verbose = verbose
        self.weight_decay = weight_decay
        self.random_seed = random_seed
        self.use_fm = use_fm
        self.use_ffm = use_ffm
        self.use_deep = use_deep
        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.use_cuda = use_cuda
        self.n_class = n_class
        self.greater_is_better = greater_is_better
        self.fm_first_order_used = fm_first_order_used
        self.raw_feature_size = raw_feature_size

        self.anchor_num = anchor_num
        self.nn_num = nn_num
        self.c = c

        torch.manual_seed(self.random_seed)
        self.anchor_points = nn.Parameter(torch.randn(self.anchor_num, self.raw_feature_size))

        """
            check cuda
        """
        # if self.use_cuda and not torch.cuda.is_available():
        self.use_cuda = False
        print("Cuda is not available, automatically changed into cpu model")

        """
            check use fm or ffm
        """
        if self.use_fm and self.use_ffm:
            print("only support one type only, please make sure to choose only fm or ffm part")
            exit(1)
        elif self.use_fm and self.use_deep:
            print("The model is deepfm(fm+deep layers)")
        elif self.use_ffm and self.use_deep:
            print("The model is deepffm(ffm+deep layers)")
        elif self.use_fm:
            print("The model is fm only")
        elif self.use_ffm:
            print("The model is ffm only")
        elif self.use_deep:
            print("The model is deep layers only")
        else:
            print("You have to choose more than one of (fm, ffm, deep) models to use")
            exit(1)

        """
            bias
        """
        if self.use_fm or self.use_ffm:
            self.bias = torch.nn.Parameter(torch.randn(self.anchor_num, 1))
        """
            fm part
        """
        if self.use_fm:
            print("Init fm part")
            if self.fm_first_order_used:
                self.fm_first_order_embeddings = nn.ModuleList(
                    [nn.ModuleList([nn.Embedding(feature_size, 1) for feature_size in self.feature_sizes])
                     for _ in range(self.anchor_num)])
                if self.dropout_shallow:
                    self.fm_first_order_dropout = nn.Dropout(self.dropout_shallow[0])
            self.fm_second_order_embeddings = nn.ModuleList([nn.ModuleList(
                [nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes])
                 for _ in range(self.anchor_num)])
            if self.dropout_shallow:
                self.fm_second_order_dropout = nn.Dropout(self.dropout_shallow[1])
            print("Init fm part succeed")

        """
            ffm part
        """
        if self.use_ffm:
            print("Init ffm part")
            self.ffm_first_order_embeddings = nn.ModuleList(
                [nn.Embedding(feature_size, 1) for feature_size in self.feature_sizes])
            if self.dropout_shallow:
                self.ffm_first_order_dropout = nn.Dropout(self.dropout_shallow[0])
            self.ffm_second_order_embeddings = nn.ModuleList(
                [nn.ModuleList([nn.Embedding(feature_size, self.embedding_size) for i in range(self.field_size)]) for
                 feature_size in self.feature_sizes])
            if self.dropout_shallow:
                self.ffm_second_order_dropout = nn.Dropout(self.dropout_shallow[1])
            print("Init ffm part succeed")

        """
            deep part
        """
        if self.use_deep:
            print("Init deep part")
            if not self.use_fm and not self.use_ffm:
                self.fm_second_order_embeddings = nn.ModuleList(
                    [nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes])

            if self.is_deep_dropout:
                self.linear_0_dropout = nn.Dropout(self.dropout_deep[0])

            self.linear_1 = nn.ModuleList([nn.Linear(self.field_size * self.embedding_size, deep_layers[0]) for _ in
                    range(anchor_num)])
            if self.is_batch_norm:
                self.batch_norm_1 = nn.BatchNorm1d(deep_layers[0])
            if self.is_deep_dropout:
                self.linear_1_dropout = nn.Dropout(self.dropout_deep[1])
            for i, h in enumerate(self.deep_layers[1:], 1):
                setattr(self, 'linear_' + str(i + 1), nn.ModuleList([nn.Linear(self.deep_layers[i - 1],
                    self.deep_layers[i]) for _ in range(anchor_num)]))
                if self.is_batch_norm:
                    setattr(self, 'batch_norm_' + str(i + 1), nn.BatchNorm1d(deep_layers[i]))
                if self.is_deep_dropout:
                    setattr(self, 'linear_' + str(i + 1) + '_dropout', nn.Dropout(self.dropout_deep[i + 1]))

            print("Init deep part succeed")

        print("Init succeed")

    def forward(self, Xi, Xv, X):
        """
        :param Xi_train: index input tensor, batch_size * k * 1
        :param Xv_train: value input tensor, batch_size * k * 1
        :return: the last output
        """
        """
            fm part
        """

        # local coding
        res = []

        if self.use_fm:

            for i in range(len(Xi)):
                # calculate distance
                nn_idx, nn_weight = self.nearest_neighbour(X[i])

                ll_res = []
                for k in range(len(nn_idx)):

                    """
                        fm part with first order
                    """
                    if self.fm_first_order_used:
                        fm_first_order_embedding = self.fm_first_order_embeddings[nn_idx[k]]
                        first_emb_list = [torch.mm(
                            torch.FloatTensor(Xv[i][j]).view(-1, 1).t(), emb(torch.LongTensor(Xi[i][j])))
                            if len(Xi[i][j]) > 0 else torch.FloatTensor([[0.0]])
                            for j, emb in enumerate(fm_first_order_embedding)
                            ]
                        fm_first_order = torch.sum(torch.cat(first_emb_list))

                    """
                        fm part with second order
                    """
                    fm_second_order_embeddings = self.fm_second_order_embeddings[nn_idx[k]]

                    second_emb_list = [torch.mm(
                        torch.FloatTensor(Xv[i][j]).view(-1, 1).t(), emb(torch.LongTensor(Xi[i][j]))
                    ) if len(Xi[i][j]) > 0 else torch.zeros([1, self.embedding_size])
                                       for j, emb in enumerate(fm_second_order_embeddings)
                                       ]

                    fm_deep_embedding = torch.cat(second_emb_list, 1)
                    tmp = torch.cat(second_emb_list, 0)
                    square_sum = torch.sum(tmp, 0)
                    square_sum = square_sum * square_sum
                    fm_sum_second_order_emb_square = square_sum

                    sum_square = torch.sum(tmp * tmp, 0)
                    fm_second_order_emb_square_sum = sum_square
                    fm_second_order = (fm_sum_second_order_emb_square - fm_second_order_emb_square_sum) * 0.5
                    if self.is_shallow_dropout:
                        fm_second_order = self.fm_second_order_dropout(fm_second_order)

                    """
                        deep part
                    """
                    if self.use_deep:
                        deep_emb = fm_deep_embedding
                        if self.deep_layers_activation == 'sigmoid':
                            activation = F.sigmoid
                        elif self.deep_layers_activation == 'tanh':
                            activation = F.tanh
                        else:
                            activation = F.relu
                        if self.is_deep_dropout:
                            deep_emb = self.linear_0_dropout(deep_emb)
                        x_deep = self.linear_1[nn_idx[k]](deep_emb)
                        if self.is_batch_norm:
                            x_deep = self.batch_norm_1(x_deep)
                        x_deep = activation(x_deep)
                        if self.is_deep_dropout:
                            x_deep = self.linear_1_dropout(x_deep)
                        for m in range(1, len(self.deep_layers)):
                            x_deep = getattr(self, 'linear_' + str(m + 1))[nn_idx[k]](x_deep)
                            if self.is_batch_norm:
                                x_deep = getattr(self, 'batch_norm_' + str(m + 1))(x_deep)
                            x_deep = activation(x_deep)
                            if self.is_deep_dropout:
                                x_deep = getattr(self, 'linear_' + str(m + 1) + '_dropout')(x_deep)

                    """
                        sum
                    """
                    if self.use_fm and self.use_deep:
                        bias = self.bias[nn_idx[k]]
                        sum_deep = torch.sum(x_deep)
                        total_sum = torch.sum(fm_second_order) + sum_deep + bias
                        if self.fm_first_order_used:
                            total_sum = total_sum + fm_first_order

                    elif self.use_fm:
                        total_sum = torch.sum(fm_second_order) + self.bias[nn_idx[k]]
                        if self.fm_first_order_used:
                            total_sum = total_sum + fm_first_order
                    else:
                        total_sum = torch.sum(x_deep, 1)

                    ll_res.append(total_sum)
                res.append(torch.sum(torch.cat(ll_res) * nn_weight))

        return torch.stack(res)

    def nearest_neighbour(self, x):
        """
        :param x: sample to be local coded
        :return:
        """
        x = torch.FloatTensor(x.todense())
        dis = torch.sum((self.anchor_points - x) ** 2, 1)
        sorted_dis, indice = torch.sort(dis)
        weight = sorted_dis[:self.nn_num]
        idx = indice[:self.nn_num]
        dis_scaled = torch.exp(-self.c * weight)
        weight = dis_scaled / (torch.sum(dis_scaled) + 1e-1)
        return idx, weight

    def fit(self, Xi_train, Xv_train, y_train, X_train, Xi_valid=None, Xv_valid=None,
            y_valid=None, X_valid=None, adaptive_anchor=False, adaptive_nn=False, early_stopping=False, save_path=None):
        """
        :param adaptive_anchor: whether use adaptive knn for anchor points selection
        :param Xi_train: [[ind1_1, ind1_2, ...], [ind2_1, ind2_2, ...], ..., [indi_1, indi_2, ..., indi_j, ...], ...]
                        indi_j is the list of feature index of feature field j of sample i in the training set
                        support multiple non-zero values in one field
        :param Xv_train: [[val1_1, val1_2, ...], [val2_1, val2_2, ...], ..., [vali_1, vali_2, ..., vali_j, ...], ...]
                        vali_j is the list of feature value of feature field j of sample i in the training set
                        vali_j can be either binary (1/0, for binary/categorical features) or float (e.g., 10.24, for numerical features)
        :param y_train: label of each sample in the training set
        :param Xi_valid: list of list of feature indices of each sample in the validation set
        :param Xv_valid: list of list of feature values of each sample in the validation set
        :param y_valid: label of each sample in the validation set
        :param early_stopping: perform early stopping or not
        :param save_path: the path to save the model
        :return:
        """
        """
        pre_process
        """

        """
            anchor points
        """
        logging.info('K-means to find {} anchor points'.format(self.anchor_num))
        kmeans = KMeans(n_clusters=self.anchor_num, n_init=1, max_iter=50, random_state=2018, verbose=1).fit(X_train)

        self.anchor_points = nn.Parameter(torch.from_numpy(kmeans.cluster_centers_).float(), requires_grad=adaptive_anchor)
        logging.info('K-means done')

        if self.verbose:
            print("pre_process data ing...")
        is_valid = False
        # Xi_train = np.array(Xi_train).reshape((-1, self.field_size, 1))
        Xi_train = np.array(Xi_train)
        Xv_train = np.array(Xv_train)
        # X_train = np.array(X_train)
        y_train = np.array(y_train)
        x_size = Xi_train.shape[0]
        if Xi_valid:
            # Xi_valid = np.array(Xi_valid).reshape((-1, self.field_size, 1))
            Xi_valid = np.array(Xi_valid)
            Xv_valid = np.array(Xv_valid)
            # X_valid = np.array(X_train)
            y_valid = np.array(y_valid)
            x_valid_size = Xi_valid.shape[0]
            is_valid = True
        if self.verbose:
            print("pre_process data finished")

        """
            train model
        """
        model = self.train()

        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate, weight_decay=self.weight_decay)
        if self.optimizer_type == 'adam':
            if self.fm_first_order_used:
                optimizer = torch.optim.Adam([
                    {'params': self.fm_first_order_embeddings.parameters()},
                    {'params': self.fm_second_order_embeddings.parameters()},
                    {'params': self.bias},
                    {'params': self.linear_1.parameters()},
                    {'params': self.linear_2.parameters()},
                    {'params': self.anchor_points, 'lr': self.learning_rate / 100}],
                    lr=self.learning_rate, weight_decay=self.weight_decay)
            else:
                optimizer = torch.optim.Adam([
                    {'params': self.fm_second_order_embeddings.parameters()},
                    {'params': self.bias},
                    {'params': self.linear_1.parameters()},
                    {'params': self.linear_2.parameters()},
                    {'params': self.anchor_points, 'lr': self.learning_rate / 100}],
                    lr=self.learning_rate, weight_decay=self.weight_decay)

        elif self.optimizer_type == 'rmsp':
            optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'adag':
            optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate, weight_decay=self.weight_decay)

        criterion = F.binary_cross_entropy_with_logits

        train_result = []
        valid_result = []
        train_loss_record = []
        valid_loss_record = []
        for epoch in range(self.n_epochs):
            total_loss = 0.0
            batch_iter = x_size // self.batch_size
            epoch_begin_time = time()
            batch_begin_time = time()
            for i in range(batch_iter + 1):
                offset = i * self.batch_size
                end = min(x_size, offset + self.batch_size)
                if offset == end:
                    break

                # convert to sparse tensor
                # batch_xi = Variable(torch.LongTensor(Xi_train[offset:end]))
                # batch_xv = Variable(torch.FloatTensor(Xv_train[offset:end]))

                batch_xi = Xi_train[offset: end]
                batch_xv = Xv_train[offset: end]
                batch_x = X_train[offset:end]

                batch_y = Variable(torch.FloatTensor(y_train[offset:end]))
                if self.use_cuda:
                    batch_xi, batch_xv, batch_y = batch_xi.cuda(), batch_xv.cuda(), batch_y.cuda()
                optimizer.zero_grad()
                outputs = model(batch_xi, batch_xv, batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.data.item()
                if self.verbose:
                    if i % 100 == 99:  # print every 100 mini-batches
                        eval = self.evaluate(batch_xi, batch_xv, batch_x, batch_y)
                        logging.info('[%d, %5d] loss: %.6f metric: %.6f time: %.1f s' %
                              (epoch + 1, i + 1, total_loss / 100.0, eval, time() - batch_begin_time))
                        total_loss = 0.0
                        batch_begin_time = time()

            train_loss, train_eval = self.eval_by_batch(Xi_train, Xv_train, X_train, y_train, x_size)
            train_result.append(train_eval)
            train_loss_record.append(train_loss)
            print('*' * 50)
            logging.info('[%d] loss: %.6f metric: %.6f time: %.1f s' %
                         (epoch + 1, train_loss, train_eval, time() - epoch_begin_time))
            print('*' * 50)

            if is_valid:
                valid_loss, valid_eval = self.eval_by_batch(Xi_valid, Xv_valid, X_valid, y_valid, x_valid_size)
                valid_result.append(valid_eval)
                valid_loss_record.append(valid_loss)
                print('*' * 50)
                logging.info('[%d] loss: %.6f metric: %.6f time: %.1f s' %
                             (epoch + 1, valid_loss, valid_eval, time() - epoch_begin_time))
                print('*' * 50)
            if save_path:
                torch.save(self.state_dict(), save_path)
            if is_valid and early_stopping and self.training_termination(valid_result):
                print("early stop at [%d] epoch!" % (epoch + 1))
                break

        return train_result, train_loss_record, valid_result, valid_loss_record

    def eval_by_batch(self, Xi, Xv, X, y, x_size):
        total_loss = 0.0
        y_pred = []
        if self.use_ffm:
            batch_size = 16384 * 2
        else:
            batch_size = 16384
        batch_iter = x_size // batch_size
        criterion = F.binary_cross_entropy_with_logits
        model = self.eval()
        for i in range(batch_iter + 1):
            offset = i * batch_size
            end = min(x_size, offset + batch_size)
            if offset == end:
                break
            # batch_xi = Variable(torch.LongTensor(Xi[offset:end]))
            # batch_xv = Variable(torch.FloatTensor(Xv[offset:end]))
            batch_xi = Xi[offset:end]
            batch_xv = Xv[offset:end]
            batch_x = X[offset:end]
            batch_y = Variable(torch.FloatTensor(y[offset:end]))
            if self.use_cuda:
                batch_xi, batch_xv, batch_y = batch_xi.cuda(), batch_xv.cuda(), batch_y.cuda()
            outputs = model(batch_xi, batch_xv, batch_x)
            pred = F.sigmoid(outputs).cpu()
            y_pred.extend(pred.data.numpy())
            loss = criterion(outputs, batch_y)
            total_loss += loss.data.item() * (end - offset)
        total_metric = self.eval_metric(y, y_pred)
        return total_loss / x_size, total_metric

    # shuffle three lists simutaneously
    def shuffle_in_unison_scary(self, a, b, c):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)

    def training_termination(self, valid_result):
        if len(valid_result) > 4:
            if self.greater_is_better:
                if valid_result[-1] < valid_result[-2] and \
                        valid_result[-2] < valid_result[-3] and \
                        valid_result[-3] < valid_result[-4]:
                    return True
            else:
                if valid_result[-1] > valid_result[-2] and \
                        valid_result[-2] > valid_result[-3] and \
                        valid_result[-3] > valid_result[-4]:
                    return True
        return False

    def predict(self, Xi, Xv):
        """
        :param Xi: the same as fit function
        :param Xv: the same as fit function
        :return: output, ont-dim array
        """
        Xi = np.array(Xi).reshape((-1, self.field_size, 1))
        Xi = Variable(torch.LongTensor(Xi))
        Xv = Variable(torch.FloatTensor(Xv))
        if self.use_cuda and torch.cuda.is_available():
            Xi, Xv = Xi.cuda(), Xv.cuda()

        model = self.eval()
        pred = F.sigmoid(model(Xi, Xv)).cpu()
        return (pred.data.numpy() > 0.5)

    def predict_proba(self, Xi, Xv, X):
        # Xi = np.array(Xi).reshape((-1, self.field_size, 1))
        # Xi = Variable(torch.LongTensor(Xi))
        # Xv = Variable(torch.FloatTensor(Xv))
        # if self.use_cuda and torch.cuda.is_available():
        #     Xi, Xv = Xi.cuda(), Xv.cuda()

        model = self.eval()
        pred = F.sigmoid(model(Xi, Xv, X)).cpu()
        return pred.data.numpy().tolist()

    def inner_predict(self, Xi, Xv):
        """
        :param Xi: tensor of feature index
        :param Xv: tensor of feature value
        :return: output, numpy
        """
        model = self.eval()
        pred = F.sigmoid(model(Xi, Xv)).cpu()
        return (pred.data.numpy() > 0.5)

    def inner_predict_proba(self, Xi, Xv, X):
        """
        :param Xi: tensor of feature index
        :param Xv: tensor of feature value
        :return: output, numpy
        """
        model = self.eval()
        pred = F.sigmoid(model(Xi, Xv, X)).cpu()
        return pred.data.numpy()

    def evaluate(self, Xi, Xv, X, y):
        """
        :param Xi: tensor of feature index
        :param Xv: tensor of feature value
        :param y: tensor of labels
        :return: metric of the evaluation
        """
        y_pred = self.inner_predict_proba(Xi, Xv, X)
        return self.eval_metric(y.cpu().data.numpy(), y_pred)

    def dump_model(self):
        """
        dump model for later prediction
        :return:
        """

    def load_model(self):
        """
        load pre-trained model
        :return:
        """
        dic = torch.load('dump_model/lldeepfm.snapshot')
        self.load_state_dict(dic)
