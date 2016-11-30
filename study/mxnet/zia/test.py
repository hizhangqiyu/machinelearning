import mxnet as mx
train = mx.io.MNISTIter(
    image      = "mnist/train-images-idx3-ubyte",
    label      = "mnist/train-labels-idx1-ubyte",
    batch_size = 128,
    data_shape = (784, ))

#val   = mx.io.MNISTIter(...)
val =	mx.io.MNISTIter(
    image      = "mnist/t10k-images-idx3-ubyte",
    label      = "mnist/t10k-labels-idx1-ubyte",
    batch_size = 128,
    data_shape = (784, ))

data = mx.symbol.Variable('data')
fc1  = mx.symbol.FullyConnected(data = data, num_hidden=128)
act1 = mx.symbol.Activation(data = fc1, act_type="relu")
fc2  = mx.symbol.FullyConnected(data = act1, num_hidden = 64)
act2 = mx.symbol.Activation(data = fc2, act_type="relu")
fc3  = mx.symbol.FullyConnected(data = act2, num_hidden=10)
mlp  = mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')

model = mx.model.FeedForward(
    symbol = mlp,
    num_epoch = 20,
    learning_rate = .1)

model.fit(X = train, eval_data = val)


#test = mx.io.MNISTIter(...)
model.predict(X = val)