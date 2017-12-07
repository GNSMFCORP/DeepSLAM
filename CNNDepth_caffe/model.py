import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop

weight_param = {"lr_mult":1, "decay_mult":1}
bias_param = {"lr_mult":2, "decay_mult":0}
param = [weight_param, bias_param]

def conv_relu(bottom, nout, ks=3, stride=1, pad=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
        num_output=nout, pad=pad,
        param=param)
    return conv, L.ReLU(conv, in_place=True)

def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def cnndepth(data, label=None, train=False):
    n = caffe.NetSpec()
    n.data = data

    # the base vggnet
    n.conv1_1, n.relu1_1 = conv_relu(n.data, 64, pad=100)
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 64)
    n.pool1 = max_pool(n.relu1_2)

    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 128)
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 128)
    n.pool2 = max_pool(n.relu2_2)

    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 256)
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 256)
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 256)
    n.pool3 = max_pool(n.relu3_3)

    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512)
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 512)
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512)
    n.pool4 = max_pool(n.relu4_3)

    n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 512)
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 512)
    n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 512)
    n.pool5 = max_pool(n.relu5_3)

    n.fc6, n.relu6 = conv_relu(n.pool5, 4096, ks=7, pad=0)
    if train:
      n.drop6 = fc7input = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)
    else:
      fc7input = n.relu6
    n.fc7, n.relu7 = conv_relu(fc7input, 4096, ks=1, pad=0)
    if train:
      n.drop7 = frinput = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)
    else:
      frinput = n.relu7

    # fully conv
    n.score_fr = L.Convolution(frinput, num_output=21, kernel_size=1, pad=0, param=param)
    n.upscore2 = L.Deconvolution(n.score_fr,
        convolution_param={"num_output":21, "kernel_size":4, "stride":2, "bias_term":False},
        param=[{"lr_mult":0}])

    n.score_pool4 = L.Convolution(n.pool4, num_output=21, kernel_size=1, pad=0, param=param)
    n.score_pool4c = crop(n.score_pool4, n.upscore2)
    n.fuse_pool4 = L.Eltwise(n.upscore2, n.score_pool4c, operation=P.Eltwise.SUM)
    n.upscore16 = L.Deconvolution(n.fuse_pool4,
        convolution_param={"num_output":21, "kernel_size":32, "stride":16, "bias_term":False},
        param=[{"lr_mult":0}])

    n.score = crop(n.upscore16, n.data)
    if label is not None:
      n.label = label
      n.loss = L.SoftmaxWithLoss(n.score, n.label, loss_param={"normalize":False, "ignore_label":255})

    return n.to_proto()