%[xTrainImages,tTrain] = digitTrainCellArrayData;
xTrainImages = csml;
tTrain = one_hot;
rng('default')
hiddenSize1 = 500;
autoenc1 = trainAutoencoder(xTrainImages,hiddenSize1, ...
    'MaxEpochs',350, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);
feat1 = encode(autoenc1,xTrainImages);
hiddenSize2 = 300;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',350, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);
feat2 = encode(autoenc2,feat1);
softnet = trainSoftmaxLayer(feat2,tTrain,'MaxEpochs',350);
stackednet = stack(autoenc1,autoenc2,softnet);
