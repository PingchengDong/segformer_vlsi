import tensorly as tl
import numpy as np
import torch
from tensorly.decomposition import tucker
from tensorly.decomposition import matrix_product_state
from mmseg.t3nsor.layers import TTLinear
from mmseg.tensorized_rnn.lstm import LSTMCell, LSTM
from mmseg.tensorized_rnn.tt_linearset import TTLinearSet
from mmseg.tensorized_rnn.rnn_utils import tt_shape

n_gate = 3
hidden_size = 64
n_cores = 4
tt_rank = 8
te = torch.rand(3,64).cuda()
shape = tt_shape(64, 64,
                 4, 3, new_core=None)
print(shape)
TTqkv_linear = TTLinear(out_features=n_gate*hidden_size, shape=shape,
                            bias=False, auto_shapes=False, d=n_cores,
                            tt_rank=tt_rank).cuda()
#print(vars(TTqkv_linear))
print("linearset")
TTqkv = TTLinearSet(in_features=64,
                            out_features=64, n_gates=3,
                            bias=False, auto_shapes=True,
                            d=4, tt_rank=8).cuda()#4 cores, 3 gates, 8rank
#print(vars(TTqkv))
out = TTqkv_linear(te)
print(out.shape)

