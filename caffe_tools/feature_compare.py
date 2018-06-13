
#coding=utf-8
'''
对所有文件修改名称，同时修改XML中保存的名称
'''
import sys
# reload(sys)
# sys.setdefaultencoding('utf8')

sys.path.insert(0, '/media/ly/14023ab3-051e-4b1d-bf78-4ec9f4ca01c1/project2/caffe/python')

import caffe
import numpy as np

def showResult(data_1, data_2, i):
    data_diff = 100000 * np.abs(data_1 - data_2)
    x1 = np.nonzero(data_diff)
    if (x1[0].size > 0):
        print ("i = ", i,end="\t")
        print (pr, "difference")
    else:
        print("i = ", i, end="\t")
        print (pr, "same")


caffe.set_mode_cpu()

model_def = './deploy.prototxt'
model_weights = './caffe_DenseNet_iter_64000.caffemodel'
# model_weights = './check_weights/caffe_DenseNet_iter_90000.caffemodel'
model_surgery_weights = './check_weights/caffe_DenseNet_iter_100000_fc_all.caffemodel'
net = caffe.Net(model_def,  # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)  # use test mode (e.g., don't perform dropout)

net_surgery = caffe.Net(model_def,  # defines the structure of the model
                        model_surgery_weights,  # contains the trained weights
                        caffe.TEST)  # use test mode (e.g., don't perform dropout)

print ('***************Network modifying*****************z************************')
for pr in net.params:
    lidx = list(net._layer_names).index(pr)
    layer = net.layers[lidx]

    if len(net.params[pr]) > 0:
        for i in range(len(net.params[pr])):
            data_1 = net.params[pr][i].data[:]
            data_2 = net_surgery.params[pr][i].data[:]
            showResult(data_1, data_2, i)
    """
    if layer.type == 'Convolution':
        # print pr + "(conv)"
        # data_1 = net.params[pr][0].data[:]
        # data_2 = net_surgery.params[pr][0].data[:]
        # showResult(data_1, data_2, 0)
        if len(net.params[pr]) > 0:
            for i in range(len(net.params[pr])):
                data_1 = net.params[pr][i].data[:]
                data_2 = net_surgery.params[pr][i].data[:]
                showResult(data_1, data_2, i)

   
    elif layer.type == 'BatchNorm':
        # print pr + "(batchnorm)"
        if len(net.params[pr]) > 0:
            for i in range(len(net.params[pr])):
                data_1 = net.params[pr][i].data[:]
                data_2 = net_surgery.params[pr][i].data[:]
                showResult(data_1, data_2, i)
    elif layer.type == 'Scale':
        # print pr + "(scale)"
        if len(net.params[pr]) > 0:
            for i in range(len(net.params[pr])):
                data_1 = net.params[pr][i].data[:]
                data_2 = net_surgery.params[pr][i].data[:]
                showResult(data_1, data_2, i)
    else:
        print "WARNING: unsupported layer, " + pr
    """








