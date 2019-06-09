from deepautoencoder import StackedAutoEncoder
import numpy as np
import numpy
import scipy.io
import scipy.optimize
import scipy.io

mat = scipy.io.loadmat('dca_res.mat')
matrix = mat['trainZ1']
feat = matrix.T

mat1 = scipy.io.loadmat('lablels.mat')
matrix1 = mat1['train']

labl = numpy.array(matrix1).reshape(1269,8)
labl = labl

data, target = feat,labl
# train / test  split
idx = np.random.rand(data.shape[0]) < 0.8
train_X, train_Y = data[idx], target[idx]
test_X, test_Y = data[~idx], target[~idx]

model = StackedAutoEncoder(dims=[500, 300], activations=['tanh', 'tanh'], epoch=[
                           5, 5], loss='rmse', lr=0.0001, batch_size=100, print_step=1)
model.fit(train_X)
test_X_ = model.transform(test_X)   

