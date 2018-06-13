import sys
import os.path as osp
sys.path.insert(0, osp.dirname(__file__)+'/lib')
from lib.net import Net#, underline
import caffe
from .builder import Net as NetBuilder
import numpy as np


class filler:
    msra = "msra"
    xavier = "xavier"
    orthogonal = "orthogonal"
    constant = "constant"

class conv_param():
    def __init__(self, name=None, num_output=0, new_name=None, pad_h=None, pad_w=None, kernel_h=None, kernel_w=None, stride=None, bias=None, group=None):
        self.name = name
        self.num_output = num_output
        self.new_name = new_name
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.stride = stride
        self.bias = bias
        self.group = group

class insert_conv():
    def __init__(self, bottom=[], name=None, num_output=None, kernel_size=3, pad=1,  stride=1,  decay = True,  bias = False,  freeze = False,
                 filler = filler.xavier, phase=None, group=1, engine=2, bringforward=True, change=True, update_nodes=None, bringto=None,):
        self.name =name


        self.engine = engine

class insert_relu():
    def __init__(self, bottom, name=None, bringforward=True, change=True, update_nodes=None, bringto=None):
        self.bottom = bottom


        self.bringto = bringto

class insert_bn_scale():
    def __init__(self, bottom, name=None, bringforward=True, change=True, update_nodes=None, bringto=None):
        self.bottom = bottom

        self.bringto = bringto

class insert_pool():
    def __init__(self, bottom, name=None, bringforward=True, change=True, update_nodes=None, bringto=None, pool='MAX',
                 global_pooling=False, pad=0, stride=1, kernel_size=3):
        self.bottom = bottom

        self.bringto = bringto

class insert_EvalDetection():
    def __init__(self, bottom, name=None, layer_type="EvalDetection", bringforward=True, bringto=None, top=None,
                    propagate_down=True, side_w=12, side_h=12, num_class=1, num_object=5, threshold = 0.0, nms = 0.4, max_objnum = 10):
        self.bottom = bottom

        self.bringto =bringto

class insert_EuclideanLoss():
    def __init__(self, bottom, name=None, layer_type="EvalDetection", bringforward=True, bringto=None,loss_weight=-1, propagate_down=True):
        self.bottom = bottom

        self.bringto = bringto

class insert_RegionLoss():
    def __init__(self, bottom, name=None, layer_type="RegionLoss", bringforward=True, bringto=None, name,
                    top=None,
                    loss_weight=-1,
                    phase=caffe.TRAIN,
                    side_w=12,
                    side_h=12,
                    num_class=1,
                    coords=4,
                    num=2,
                    softmax=1,
                    jitter=0.2,
                    rescore=1,
                    object_scale=5.0,
                    noobject_scale=1.0,
                    class_scale=1.0,
                    coord_scale=1.0,
                    absolute=1,
                    thresh=0.5,
                    random=0 ):
        self.bottom = bottom

        self.bringto = bringto


##phase=caffe.TRAIN

def get_layerparam(self, layernames=[]):
    """
    返回指type类型层的权值和偏置
    :param pt:    prototxt
    :param model:  caffemodel
    :param type[list]:   只有conv ,bn ，scale层有权值或偏置 (对type会有判断，如果不是这三个类型，会给出提示；为空则默认返回这三个类型的所有权值)
    :param undo_layer[list]:    不需要获得权值的层(可为空)
    :return:  W_B=dict(){是有序还是无序},  W_B[('conv1','w')]:表示conv1的权值； W_B[('conv1','b')]:表示conv1的偏置 (需要判断是否有偏置，如无则不会返回偏置)
    """


def get_featuremap(self, layernames=[]):
    """
    返回指定type类型层的featuremap
    :param pt:   prototxt
    :param model:  caffemodel
    :param type[list]: 需要求取的层类型
    :return: feat=dict(),       feat['conv1']:表示conv1的featuremap
    """

def set_conv(self, conv_params=[]):
    """
    设置给定层的参数(只包括：num_output, new_name, pad_h, pad_w, kernel_h, kernel_w, stride, bias, group)
    :param pt:    prototxt
    :param conv_param:    待修改的conv层参数
    :return:状态:    修改后的prototxt
    """

def set_param(pt, layers=[], lr_mult=0, decay_mult=0, new_pt_name=None):
    """
    设置layers中每个层的param中的lr_mulit和decay_mult参数
    :param pt:    prototxt
    :param layers:    需要修改param的层list
    :param lr_mult:   将layers中所有层的lr_mult修改为设定值
    :param decay_mult:  将layers中所有层的decay_mult修改为设定值
    :param new_pt_name: 保存的prototxt名
    :return:new_pt:     修改后的prototxt
    """

def set_propagate_down(pt, layers=[], propagate_down=False, new_pt_name=None):
    """
    设置layers中的每一层的propagate_down参数
    :param pt:      prototxt
    :param layers:  需要修改propagate_down的层list
    :param propagate_down:  将layers中所有层的propagate_down修改为设定值
    :param new_pt_name:     保存的prototxt名
    :return:new_pt:     修改后的prototxt
    """

def set_use_global_stats(pt, bn_layers=[], stats=True, new_pt_name=None):
    """
    设置bn_layers中的每一层的use_global_stats
    :param pt:      prototxt
    :param bn_layers:   需要修改use_global_stats的层list
    :param stats:       将layers中所有层的use_global_stats修改为设定值
    :param new_pt_name: 保存的prototxt名
    :return:new_pt:     修改后的prototxt
    """

def insert_conv(pt, insert_convs=[], new_pt_name=None):
    """
    插入insert_convs中所有层
    :param pt:      prototxt
    :param insert_convs: 待插入的conv层list
    :param new_pt_name:  保存的prototxt名
    :return:new_pt:     修改后的prototxt
    """
def inset_relu(pt, insert_relus=[], new_pt_name=None):
    """
    插入insert_relus中所有层
    :param pt:      prototxt
    :param insert_relu:     待插入的relu层list
    :param new_pt_name:     保存的prototxt名
    :return:new_pt:     修改后的prototxt
    """

def inset_bn_scale(pt, insert_bn_scales=[], new_pt_name=None):
    """
    插入insert_bn_scales中所有层
    :param pt:      prototxt
    :param insert_bn_scale:     待插入的bn_scales层list
    :param new_pt_name:     保存的prototxt名
    :return:new_pt:     修改后的prototxt
    """

def inset_pool(pt, insert_pools=[], new_pt_name=None):
    """
    插入insert_pools中所有层
    :param pt:      prototxt
    :param insert_pool:     待插入的pool层list
    :param new_pt_name:     保存的prototxt名
    :return:new_pt:     修改后的prototxt
    """
def insert_EvalDetection(pt ,insert_evaldetections=[], new_pt_name=None):
    """
    插入insert_evaldetections中所有层
    :param pt:     prototxt
    :param insert_evaldetections:    待插入的evaldetections层list
    :param new_pt_name:     保存的prototxt名
    :return:new_pt:     修改后的prototxt
    """

def insert_EuclideanLoss(pt, insert_euclideanlosss=[], new_pt_name=None):
    """
    插入insert_euclideanlosss中所有层
    :param pt:     prototxt
    :param insert_euclideanlosss:      待插入的euclideanlosss层list
    :param new_pt_name:         保存的prototxt名
    :return:new_pt:     修改后的prototxt
    """

def insert_RegionLoss(pt, insert_RegionLoss=[], new_pt_name=None):
    """
    插入insert_RegionLoss中所有层
    :param pt:     prototxt
    :param insert_RegionLoss:      待插入的RegionLoss层list
    :param new_pt_name:         保存的prototxt名
    :return:new_pt:     修改后的prototxt
    """

def insert_input(pt, dim=[1, 3, 256, 512], new_pt_name=None):
    """
    插入input和input_shape(是否需要先删除data层)
    :param pt:     prototxt
    :param dim=[1, 3, 256, 512]:      输入图片的尺寸[1,c,h,w]
    :param new_pt_name:         保存的prototxt名
    :return:new_pt:     修改后的prototxt
    """

# def merge_pt(main_pt, added_pts=[], new_pt_name=None):
#     """
#     将added_pts中的层合并到main_pt中(只负责合并，不负责修改参数)
#     :param main_pt:     prototxt
#     :param added_pts=[]:      需要加入的prototxt(data层必须为input)
#     :param new_pt_name:         保存的prototxt名
#     :return:new_pt:     修改后的prototxt
#     """

def remove_layers(self, rm_layers=[]):
    """
    将rm_layers中的所有层删除
    :param pt:     prototxt
    :param rm_layers=[]:      需要删除的layer层list
    :param new_pt_name:         保存的prototxt名
    :return:new_pt:     修改后的prototxt
    """

def get_layernames(self ,type=[]):
    """
    返回pt中指定type的层（是pt中的层，还是caffemodel中的层）
    :param pt:     prototxt
    :param type=[]:      需要获取的层类型 ['Convolution', 'BatchNorm', 'Scale', 'ReLU', 'Pooling', 'RegionLoss']
    :return:layers:     得到的层名list
    """

def save_pt(self, new_name='save_test'):
    """

    :param pt:
    :param new_name:
    :return:
    """
def set_caffemodel(self,WPQ):
    """

    :param self:
    :param WPQ:
    :return:
    """

def save_caffemodel(self, WPQ, new_pt_name, new_caffemodel_name):
    """
    保存caffemodel(如果只修改权值或偏置，则不需要对pt进行修改；如果对层名及其他参数进行了修改，则输入的pt需要作相应修改)
    :param pt:
    :param caffemodel:
    :param WPQ: WPQ[('conv1',0)]:表示conv1的权值； WPQ[('conv1',1)]:表示conv1的偏置
    :param new_pt_name:
    :param new_caffemodel_name:
    :return:
    """

if __name__ == '__main__':

    ### 导入prototxt和caffemodel
    pt = "/home/zhaona/caffe_net/temp/darknet-16_train_V1_2.prototxt"
    model = "/home/zhaona/caffe_net/temp/darknet_16_2anchor_iter_120000_2.caffemodel"

    ### 使用pt和model初始化net
    net = Net(pt, model=model, phase=caffe.TRAIN)  #phase指明是在训练阶段还是测试阶段

    ### (1)获取层权重
    # for conv in net.convs:
        weights = net.param_data(conv)      ###获取conv层的权重
    #     bias = net.param_b_data(conv)     ###获取conv层的偏置
    #     print(conv, weights.shape)
    # bn2 = net.param_data('bn2')
    # scale = net.param_data('scale_conv2')
    # print('bn2',bn2.shape)
    # print('scale2',scale.shape)

    ### (2)获取层feature map
    # net.forward()
    # for conv in net.convs:
    #     feat = net.blobs_data(conv)           ###获取conv层的feature map
    #     print(conv, feat.shape)

    ### (3)修改层属性(层名，num_ouput，kernel_size，pad，stride,param,propagate,loss_weight等)

    # ### (3.1)设置conv层的num_ouput,name,pad,kernel,stride,bias,group参数，可以修改任意一个或几个
    # num_output = net.param_shape('conv1')[0]
    # net.set_conv('conv1', num_output=num_output/2, new_name='conv1_s', pad_h=4, kernel_h=4, kernel_w=1, stride=[4,1], bias=True, group=num_output)
    # new_pt = net.save_pt(new_name='test_set_conv')

    ### (3.2)将类型为'Convolution', 'BatchNorm', 'Scale', 'ReLU', 'Pooling', 'RegionLoss'的层的param中的lr_mult和decay_mult改为0
    # for layer in net.net_param.type2names(['Convolution', 'BatchNorm', 'Scale', 'ReLU', 'Pooling', 'RegionLoss']):
    #     net.set_param(layer, lr_mult=0, decay_mult=0)
    #     ##或者
    #     # net.set_param(layer, 0, 0)
    # ##在conv1层中添加param参数
    # net.net_param.add_param('conv1', lr_mult=10, decay_mult=0)
    # ##或者
    # # net.net_param.add_param('conv1', 10, 0)
    # new_pt =net.save_pt(new_name='param')

    ### (3.3)设置 propagate_down(会自动判断该层的bottom,并添加同样数目的propagate_down参数)
    # for layer in ['conv16', 'bn16', 'scale_conv16', 'relu16', 'conv_reg1']:
    #     net.net_param.set_propagate_down(layer, stats=False)
    # new_pt = net.save_pt(new_name='propagate_down')

    ### (3.4)设置 bn层中use_global_stats参数
    # for bn_name in net.net_param.type2names("BatchNorm"):
    #     net.net_param.set_use_global_stats(bn_name, stats=True)
    # new_pt = net.save_pt(new_name='use_global_stats')

    ### (3.5) eg:生成student网络(1.除data层外,其他层的层名都加'_s'; 2.将conv16,conv_reg1除外的卷积层的num_output减半)
    for conv in net.net_param.type2names():
        if conv not in ['conv16', 'conv_reg1']:
            newnum_output = net.param_shape(conv)[0]/2
        else:
            newnum_output = net.param_shape(conv)[0]
        net.set_conv(conv, new_name=underline(conv, 's'), num_output=newnum_output)
    for layer in net.net_param.type2names(['BatchNorm', 'Scale', 'ReLU', 'Pooling', 'RegionLoss']): #
        net.net_param.ch_name(layer, underline(layer, 's'))      #underline('conv1', 's')返回‘conv1_s’
    # net.remove('data')
    # net.net_param.insert_input(dim=[1, 3, 256, 512])
    new_pt = net.save_pt(new_name='my_student')   #保存的pt名为new_name,保存路径为原pt的路径
    # new_pt = net.save_pt(prefix='s')              #不设置new_name时,保存的pt名称为原pt前加's_'(两种方式可以根据需求设置)

    ### (4)增加/删除层(包括conv, bn/scale, relu, pool, eval_detection, l2_loss, region_loss, inout)

    ### (4.1)增加conv+bn/scale+relu层(如果bn/scale和relu要同时添加，relu要在bn前加入)
    ### net.insert可以插入conv层，bn/scale层，pooling层，通过layer_type指定，默认为conv层
    # for conv in net.convs:
    #     oldnum_output = net.param_shape(conv)[0]
    #     net.insert(bottom=[conv], name=underline(conv, 'add'), num_output=int(oldnum_output/2))#,pad=0, kernel_size=1, bias=True, stride=1，group=oldnum_output
    #     net.insert(bottom=[underline(conv, 'add')], layer_type='ReLU')
    #     net.insert(bottom=[underline(conv, 'add')], layer_type='BatchNorm')
    # new_pt = net.save_pt(new_name='insert_conv_bn_relu')

    ### (4.2)设置insert中设置change=False则只添加层,不会改变其他结构;默认change=True,即不设置该参数时,会该层会插入到网络结构并修改网络结构
    # net.insert(['data'], 'conv1_add', change=False)
    # net.insert(['conv3'], 'pool3', layer_type="Pooling", change=True, bringto='relu3', pool='MAX')
    # new_pt = net.save_pt(new_name='insert_pool')

    ### (4.3)增加EvalDetection层
    # net.insert_EvalDetection(bottom=['conv_reg1'], name='eval_detection', bringto='det_loss', propagate_down=False)
    # new_pt = net.save_pt(new_name='eval_detection')

    ### (4.4)增加l2_loss层
    # net.insert_EvalDetection(bottom=['conv_reg1'], name='eval_detection', bringto='det_loss', propagate_down=False)
    # net.insert_EuclideanLoss(bottom=['conv16', 'eval_detection'], name='det_loss_l2', bringto='eval_detection')
    # new_pt = net.save_pt(new_name='l2_loss')

    ### (4.5)增加det_loss层
    # net.remove('det_loss')
    # net.insert_RegionLoss(bottom=['conv_reg1', 'label'], name='det_loss', loss_weight=1, side_w=16, side_h=8)
    # new_pt = net.save_pt(new_name='det_loss')

    ### (4.6)增加input,input_shape
    ### eg:生成deploy.prototxt
    # net.remove('data')
    # net.remove('det_loss')
    net.net_param.insert_input(dim=[1, 3, 256, 512])
    # for bn_name in net.net_param.type2names("BatchNorm"):
    #     net.net_param.set_use_global_stats(bn_name, stats=True)
    # new_pt = net.save_pt(new_name='deploy.prototxt')

    ### (4.7)合并多个prototxt
    ### 从teacher和student.prototxt生成 teacher_student.prototxt
    # for conv in net.net_param.type2names():
    #     net.set_param(conv, 0, 0)
    # for layer in net.net_param.type2names(['BatchNorm', 'Scale', 'ReLU', 'Pooling', 'RegionLoss']):
    #     net.set_param(layer, 0, 0)
    # for bn in net.net_param.type2names('BatchNorm'):
    #     net.net_param.set_use_global_stats(bn, True)
    #     net.net_param.add_param(bn, 0, 0)
    #     net.net_param.add_param(bn, 0, 0)
    #     net.net_param.add_param(bn, 0, 0)
    # for scale in net.net_param.type2names('Scale'):
    #     net.net_param.add_param(scale, 0, 0)
    #     net.net_param.add_param(scale, 0, 0)
    # ### 设置propagate_down=false
    # for layer in ['conv16', 'bn16', 'scale_conv16', 'relu16', 'conv_reg1']:
    #     net.net_param.set_propagate_down(layer, stats=False)
    # new_pt = net.save_pt(new_name='propagate_down')
    # net.remove('det_loss')
    # student_pt = "/home/zhaona/caffe_net/temp/my_student.prototxt"
    # student_net = NetBuilder(student_pt)
    # # student_net.rm_layer('data')
    # net.net_param.merge_net(new_nets=[student_net])   #student_net.net_param
    # ### 增加EvalDetection层
    # net.insert_EvalDetection(bottom=['conv_reg1'], name='eval_detection', bringto='det_loss_s', propagate_down=False)
    # ### 增加l2_loss层
    # net.insert_EuclideanLoss(bottom=['conv16', 'conv16_s', 'eval_detection'], name='det_loss_l2', bringto='eval_detection')
    # new_pt = net.save_pt(new_name='my_teacher_student')


    ### (4.8)删除层
    # net.remove('conv2')
    # net.remove('bn2')
    # net.remove('scale_conv2')
    # net.remove('relu2')
    # net.remove('pool2')
    # net.remove('data')
    # net.remove('det_loss')
    # new_pt = net.save_pt(prefix='remove')

    ### (5)获取指定type的所有层名
    ### (5.1)获取caffemodel中的各类层名
    # convs = net.type2names()
    # relus = net.type2names('ReLU')
    # bns = net.type2names('BatchNorm')
    # scales = net.type2names('Scale')
    # pools = net.type2names('Pooling')
    # eltwise = net.type2names('Eltwise')
    # innerproduct = net.type2names('InnerProduct')
    # ### (5.2)获取prototxt中的各类层名
    # convs = net.net_param.type2names()
    # relus = net.net_param.type2names('ReLU')
    # bns = net.net_param.type2names('BatchNorm')
    # scales = net.net_param.type2names('Scale')
    # pools = net.net_param.type2names('Pooling')
    # eltwise = net.net_param.type2names('Eltwise')
    # innerproduct = net.net_param.type2names('InnerProduct')
    # all_layer = net.net_param.type2names(['Convolution', 'BatchNorm', 'Scale', 'ReLU', 'Pooling', 'RegionLoss'])

    ### (6)保存修改的pt和caffemodel
    ### (6.1)保存pt
    # new_pt = net.save_pt(new_name='save_test')
    # ### (6.2)保存caffemodel(首先将pt中的参数更改，比如num_ouput,将待更新的参数放在WPQ{0为权值，1为偏置}中，再使用finalmodel更新caffemodel,最后用save保存)
    # WPQ = dict()
    # conv2_w = np.zeros((32,32,3,3))
    # ### conv2_b = np.zeros(32)
    # conv3_w = np.zeros((128,32,3,3))
    # WPQ[('conv2',0)] = conv2_w        #conv2的权值
    # ### WPQ[('conv2',1)] = conv2_b      #conv2的偏置
    # WPQ[('conv3',0)] = conv3_w        #
    # net.set_conv('conv2', num_output=32)
    # net.remove('bn2')                 ###为了方便测试，删除了conv2的bn和scale层
    # net.remove('scale_conv2')
    # net.WPQ = WPQ
    # net.finalmodel(save=False, prefix='conv2')
    # new_pt, new_model = net.save(prefix='conv2_32')
    ###通过对保存的pt和caffemodel读取权值，得到conv2的权值变成了(32,32,3,3),conv3的权值变成了(128,32,3,3)

    print("end")