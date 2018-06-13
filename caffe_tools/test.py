# coding=utf-8
import sys
caffe_root = '/media/ly/14023ab3-051e-4b1d-bf78-4ec9f4ca01c1/project2/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np
import os

deploy = "./deploy_half.prototxt"
caffe_model = "./Result/caffe_DenseNet_iter_200000.caffemodel"
net = caffe.Net(deploy, caffe_model, caffe.TEST)  # 加载model和network

# 图片预处理设置
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})  # 设定图片的shape格式(1,3,28,28)
transformer.set_transpose('data', (2, 0, 1))  # 改变维度的顺序，由原始图片(28,28,3)变为(3,28,28)
transformer.set_mean('data', np.array([118.319, 114.977, 118.625]))  # 减去均值，前面训练模型时没有减均值，这儿就不用
transformer.set_raw_scale('data', 255)  # 缩放到【0，255】之间
transformer.set_channel_swap('data', (2, 1, 0))  # 交换通道，将图片由RGB变为BGR


data = "/media/ly/14023ab3-051e-4b1d-bf78-4ec9f4ca01c1/project2/caffe/data/DenseNet/Val.txt"
with open(data,'r') as fr:
    filelists = fr.readlines()

data = [line.strip().split(' ', maxsplit=1) for line in filelists]
prefix = "/media/ly/14023ab3-051e-4b1d-bf78-4ec9f4ca01c1/project2/caffe/data/DenseNet/"
data_names = [os.path.join(prefix, item[0])for item in data]
labels = [item[1].split() for item in data]
print(data_names)
labels = np.array(labels,dtype=np.int)
print(labels)
print(len(data_names),labels.shape)

predicts = np.zeros(labels.shape,dtype=np.int)
for i,img in enumerate(data_names):

    im = caffe.io.load_image(img)  # 加载图片
    net.blobs['data'].data[...] = transformer.preprocess('data', im)  # 执行上面设置的图片预处理操作，并将图片载入到blob中

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
    predicts[i] = np.array([prob01,prob02,prob03,prob04,prob05,prob06,prob07,prob08,prob09,prob10], dtype=int)
    print(i, img)

print(predicts)
accuracy = predicts==labels
print(np.sum(accuracy,axis=0))

