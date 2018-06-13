import sys
caffe_root = '/media/ly/14023ab3-051e-4b1d-bf78-4ec9f4ca01c1/project2/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
from lib.net import Net
import numpy as np
import copy
from collections import OrderedDict
import csv
import os

###################### 加载模型 ######################
deploy = "./deploy.prototxt"
model = "./caffe_DenseNet_iter_64000.caffemodel"
# 加载网络
net = Net(deploy, model=model, phase=caffe.TEST) # test类型
###################### 加载模型 ######################

###################### 测试模型 ######################
def predict(net):
    # 图片预处理设置
    transformer = caffe.io.Transformer({'data': net.caffenet.blobs['data'].data.shape})  # 设定图片的shape格式(1,3,28,28)
    transformer.set_transpose('data', (2, 0, 1))  # 改变维度的顺序，由原始图片(28,28,3)变为(3,28,28)
    transformer.set_mean('data', np.array([118.319, 114.977, 118.625]))  # 减去均值，前面训练模型时没有减均值，这儿就不用
    transformer.set_raw_scale('data', 255)  # 缩放到【0，255】之间
    transformer.set_channel_swap('data', (2, 1, 0))  # 交换通道，将图片由RGB变为BGR

    data = "/media/ly/14023ab3-051e-4b1d-bf78-4ec9f4ca01c1/project2/caffe/data/DenseNet/Val.txt"
    with open(data, 'r') as fr:
        filelists = fr.readlines()

    data = [line.strip().split(' ', maxsplit=1) for line in filelists]
    prefix = "/media/ly/14023ab3-051e-4b1d-bf78-4ec9f4ca01c1/project2/caffe/data/DenseNet/"
    data_names = [os.path.join(prefix, item[0]) for item in data]
    labels = [item[1].split() for item in data]
    print(data_names)
    labels = np.array(labels, dtype=np.int)
    print(labels)
    print(len(data_names), labels.shape)

    predicts = np.zeros(labels.shape, dtype=np.int)
    for i, img in enumerate(data_names):
        im = caffe.io.load_image(img)  # 加载图片
        net.caffenet.blobs['data'].data[...] = transformer.preprocess('data', im)  # 执行上面设置的图片预处理操作，并将图片载入到blob中

        # 执行测试
        out = net.forward()

        prob01 = out["prob01"].reshape(-1).argmax()
        prob02 = out["prob02"].reshape(-1).argmax()
        prob03 = out["prob03"].reshape(-1).argmax()
        prob04 = out["prob04"].reshape(-1).argmax()
        prob05 = out["prob05"].reshape(-1).argmax()
        prob06 = out["prob06"].reshape(-1).argmax()
        prob07 = out["prob07"].reshape(-1).argmax()
        prob08 = out["prob08"].reshape(-1).argmax()
        prob09 = out["prob09"].reshape(-1).argmax()
        prob10 = out["prob10"].reshape(-1).argmax()
        predicts[i] = np.array([prob01, prob02, prob03, prob04, prob05, prob06, prob07, prob08, prob09, prob10],
                               dtype=int)
        print(i, img)

    print(predicts)
    accuracy = predicts == labels
    accuracy = np.sum(accuracy, axis=0)
    print(accuracy)
    return accuracy

###################### 测试模型 ######################




def sensitivities_to_csv(sensitivities, fname):
    """Create a CSV file listing from the sensitivities dictionary.

    The 'sensitivities' argument is expected to have the dict-of-dict structure
    described in the documentation of perform_sensitivity_test.
    """
    with open(fname, 'w') as csv_file:
        writer = csv.writer(csv_file)
        # write the header
        writer.writerow(['parameter', 'sparsity', 'precision', 'recall', 'fscore'])

        for param_name, sensitivity in sensitivities.items():
            for sparsity, values in sensitivity.items():
                writer.writerow([param_name] + [sparsity] + list(values))

def sensitivity_analysis(net):
    sensitivities = OrderedDict()

    # 获取所有的卷积层名
    Convs_all = net.get_layernames(['Convolution'])
    print(Convs_all)
    Convs_all = ['conv4_22/x2', 'conv4_23/x1', 'conv4_23/x2', 'conv4_24/x1', 'conv4_24/x2', 'conv4_blk', 'conv5_1/x1', 'conv5_1/x2', 'conv5_2/x1', 'conv5_2/x2', 'conv5_3/x1', 'conv5_3/x2', 'conv5_4/x1', 'conv5_4/x2', 'conv5_5/x1', 'conv5_5/x2', 'conv5_6/x1', 'conv5_6/x2', 'conv5_7/x1', 'conv5_7/x2', 'conv5_8/x1', 'conv5_8/x2', 'conv5_9/x1', 'conv5_9/x2', 'conv5_10/x1', 'conv5_10/x2', 'conv5_11/x1', 'conv5_11/x2', 'conv5_12/x1', 'conv5_12/x2', 'conv5_13/x1', 'conv5_13/x2', 'conv5_14/x1', 'conv5_14/x2', 'conv5_15/x1', 'conv5_15/x2', 'conv5_16/x1', 'conv5_16/x2', 'fc01', 'fc02', 'fc03', 'fc04', 'fc05', 'fc06', 'fc07', 'fc08', 'fc09', 'fc10']
    for conv in Convs_all:
        sensitivity = OrderedDict()

        Filters = net.get_layerparam([conv]) #　获取该层的weight
        filter = Filters[(conv, 0)]
        ori_filter = copy.deepcopy(filter)
        view_filter = filter.reshape(filter.shape[0], -1)
        filter_mags = np.mean(abs(view_filter), axis=1)

        for fraction_to_prune in np.arange(0.0, 0.9, 0.1):
            if fraction_to_prune != 0:
                topk_filters = int(fraction_to_prune * filter_mags.shape[0])

                # 获取filter的mask
                binary_map = copy.deepcopy(filter_mags)
                filter_mags.sort() # 升序
                threshold = filter_mags[topk_filters-1]
                binary_map[binary_map > threshold] = 1
                binary_map[binary_map <= threshold] = 0
                binary_map = np.expand_dims(binary_map,axis=1)
                mask = np.tile(binary_map, (1, filter.shape[1]*filter.shape[2]*filter.shape[3]))
                mask = mask.reshape((filter.shape[0], filter.shape[1], filter.shape[2], filter.shape[3]))

                # 得到Sparse filter,并修改caffemodel
                final_filter = mask*filter
                WPQ = {}
                WPQ[(conv, 0)] = final_filter
                net.set_caffemodel(WPQ)

            # test sensitivity
            precision = predict(net)
            sensitivity[fraction_to_prune] = (precision)
            sensitivities[conv] = sensitivity
            if fraction_to_prune != 0:
                WPQ = {}
                WPQ[(conv, 0)] = ori_filter
                net.set_caffemodel(WPQ)

            # save the sensitivity
            fname = 'sensitivity_analysis3.csv'
            sensitivities_to_csv(sensitivities, fname)

# predict(net)
sensitivity_analysis(net)