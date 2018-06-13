from __future__ import print_function
import sys
sys.path.insert(0, "/media/ly/14023ab3-051e-4b1d-bf78-4ec9f4ca01c1/project2/caffe/python")
import caffe
import numpy as np
import os.path as osp
import os
from warnings import warn
from .builder import Net as NetBuilder


# class layertypes:
#     BatchNorm="BatchNorm"
#     Scale="Scale"
#     ReLU = 'ReLU'
#     Pooling = 'Pooling'
#     Eltwise = 'Eltwise'
#     innerproduct= 'InnerProduct'

class Net():
    def __init__(self, pt, model=None, phase=caffe.TEST, gpu=True):
        self.caffe_device()
        if gpu:     #默认使用gpu
            caffe.set_mode_gpu()
            caffe.set_device(0)
            print("using GPU caffe")
        else:
            caffe.set_mode_cpu()
            print("using CPU caffe")
        self.caffenet = caffe.Net(pt, phase)#
        self.pt_dir = pt
        self.phase = phase
        if model is not None:
            self.caffenet.copy_from(model)                               #########################
            self.caffemodel_dir = model                               #########################
        self.net_param = NetBuilder(pt, phase) # instantiate the NetBuilder -by Mario
        self.batchsize = None  # batch size of th validation data batch size -by Mario
        self._layers = dict()
        self._bottom_names = None
        self._top_names = None
        self.WPQ={} # stores pruned values, which will be saved to caffemodel later (since Net couldn't be dynamically changed) -by Mario
        self.convs= self.type2names()  # convs contains a list of strings -by Mario
        self.relus = self.type2names(layer_type='ReLU')
        self.bns = self.type2names(layer_type='BatchNorm')
        self.scales = self.type2names(layer_type='Scale')
        self.pools = self.type2names(layer_type='Pooling')
        self.eltwise = self.type2names('Eltwise')
        self.innerproduct = self.type2names('InnerProduct')

    def caffe_device(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    @property
    def top_names(self):
        if self._top_names is None:
            self._top_names = self.caffenet.top_names
        return self._top_names

    @property
    def bottom_names(self):
        if self._bottom_names is None:
            self._bottom_names = self.caffenet.bottom_names
        return self._bottom_names

    def layer_bottom(self, name):
        return self.net_param.layer_bottom(name)

    def _save(self, new_name, orig_name, prefix='acc'):
        path, name = osp.split(orig_name)
        name_s, orig_ext = osp.splitext(name)
        if new_name is None:
            # avoid overwrite
            new_name = osp.join(path, underline(prefix, name))
        else:
            # print("overwriting", new_name)
            new_name_s, ext = osp.splitext(new_name)
            if ext == '':
                ext = orig_ext
                new_name = new_name_s + ext
            new_name = osp.join(path, new_name)
        return new_name


    def save_pt(self, new_name=None, **kwargs):
        new_name = self._save(new_name, self.pt_dir, **kwargs)
        self.net_param.write(new_name)
        return new_name

    def save_caffemodel(self, new_name=None, **kwargs):
        new_name = self._save(new_name, self.caffemodel_dir, **kwargs)
        self.caffenet.save(new_name)
        return new_name

    def set_caffemodel(self, WPQ=None):
        self.WPQ = WPQ
        self.finalmodel(WPQ)

    def save(self, new_pt=None, new_caffemodel=None, **kwargs):
        return self.save_pt(new_pt, **kwargs), self.save_caffemodel(new_caffemodel, **kwargs)

    def param(self, name):
        if name in self.caffenet.params:
            return self.caffenet.params[name]
        else:
            raise Exception("no this layer")

    def blobs(self, name):
        return self.caffenet.blobs[name]

    def forward(self):
        ret = self.caffenet.forward()
        return ret

    def param_w(self, name):
        return self.param(name)[0]

    def param_b(self, name):
        return self.param(name)[1]

    def param_data(self, name):
        return self.param_w(name).data

    def param_b_data(self, name):
        return self.param_b(name).data

    def set_param_data(self, name, data):
        if isinstance(name, tuple):
            self.param(name[0])[name[1]].data[...] = data.copy()
        else:
            self.param_w(name).data[...] = data.copy()

    def set_param_b(self, name, data):
        self.param_b(name).data[...] = data.copy()

    def ch_param_data(self, name, data):
        if isinstance(name, tuple):
            if name[1] == 0:
                self.ch_param_data(name[0], data)
            elif name[1] == 1:
                self.ch_param_b(name[0], data)
            else:
                NotImplementedError
        else:
            self.param_reshape(name, data.shape)
            self.param_w(name).data[...] = data.copy()

    def ch_param_b(self, name, data):
        self.param_b_reshape(name, data.shape)
        self.param_b(name).data[...] = data.copy()

    def param_shape(self, name):
        return self.param_data(name).shape

    def param_b_shape(self, name):
        return self.param_b_data(name).shape

    def param_reshape(self, name, shape):
        self.param_w(name).reshape(*shape)

    def param_b_reshape(self, name, shape):
        self.param_b(name).reshape(*shape)

    def data(self, name='data', **kwargs):
        return self.blobs_data(name, **kwargs)

    def label(self, name='label', **kwargs):
        return self.blobs_data(name, **kwargs)


    def blobs_data(self, name, **kwargs):
        return self.blobs(name, **kwargs).data

    def blobs_type(self, name):
        return self.blobs_data(name).dtype

    def blobs_shape(self, name):
        return self.blobs_data(name).shape

    def blobs_reshape(self, name, shape):
        return self.blobs(name).reshape(*shape)

    def blobs_num(self, name):
        if self.batchsize is None:
            self.batchsize = self.blobs(name).num
        return self.batchsize

    def blobs_count(self, name):
        return self.blobs(name).count

    def blobs_height(self, name):
        return self.blobs(name).height
    def blobs_channels(self, name):
        return self.blobs(name).channels

    def blobs_width(self, name):
        return self.blobs(name).width

    def blobs_CHW(self, name):
        return self.blobs_count(name) / self.blobs_num(name)


    def get_layerparam(self, layernames=[]):
        """获取layernames中所有层(只能获取conv,bn,scale层)的权值或偏置"""
        W_B = {}
        for layername in layernames:
            if layername in self.caffenet.params:
                weight = self.param_data(layername)
                W_B[(layername, 0)] = weight
                if len(self.param(layername)) > 1:
                    bias = self.param_b_data(layername)
                    W_B[(layername, 1)] = bias
            else:
                return False
        return W_B

    def get_featuremap(self, layernames=[]):
        """获取layernames中所有层(只能获取data,label,conv,pool层)的feature map"""
        feat = {}
        for layername in layernames:
            if layername in self.caffenet.blobs:
                feature_map = self.blobs_data(layername)
                feat[layername] = feature_map
            else:
                return False
        return feat


    # =============== protobuf ===============
    def get_layer(self, conv):
        """return self.net_param.layer[conv][0]"""
        return self.net_param.layer[conv][0]

    def conv_param_stride(self, conv):
        stride = self.conv_param(conv).stride
        if len(stride) == 0:
            return 1
        else:
            assert len(stride) == 1
            return stride[0]

    def conv_param_pad(self, conv):
        pad = self.conv_param(conv).pad
        # assert len(pad) == 1
        # my adding
        if len(pad) == 0:
            pad = 0
            return pad
        # my adding
        return pad[0]

    def conv_param_kernel_size(self, conv):
        kernel_size = self.conv_param(conv).kernel_size
        assert len(kernel_size) == 1
        return kernel_size[0]

    def conv_param_num_output(self, conv):
        return self.conv_param(conv).num_output

    def net_param_layer(self, conv):
        """return self.net_param.layer[conv]"""
        return self.net_param.layer[conv]

    def conv_param(self, conv):
        return self.get_layer(conv).convolution_param

    def set_conv(self, name, num_output=None, new_name=None, pad_h=None, pad_w=None, kernel_h=None, kernel_w=None, stride=None, bias=None, group=None):
        conv_param = self.conv_param(name)
        if num_output is not None:
            conv_param.num_output = type(conv_param.num_output)(num_output)
        if pad_h is not None:
            while len(conv_param.pad):
                conv_param.pad.remove(conv_param.pad[0])
            conv_param.pad.append(pad_h)
            if pad_w is not None and pad_w != pad_h:
                conv_param.pad.append(pad_w)
        if kernel_h is not None:
            while len(conv_param.kernel_size):
                conv_param.kernel_size.remove(conv_param.kernel_size[0])
            conv_param.kernel_size.append(kernel_h)
            if kernel_w is not None and kernel_w != kernel_h:
                conv_param.kernel_size.append(kernel_w)

        if stride is not None:
            while len(conv_param.stride):
                conv_param.stride.remove(conv_param.stride[0])
            for i in stride:
                conv_param.stride.append(i)

        if bias is not None:
            conv_param.bias_term = bias

        if group is not None:
            conv_param.group = group

        if new_name is not None:
            self.net_param.ch_name(name, new_name)

    def set_param(self, name, lr_mult, decay_mult):
        self.net_param.this = self.net_param.layer[name]
        lens = self.net_param.rm_param(name)
        for i in range(lens):
            self.net_param.add_param(name, lr_mult, decay_mult)

    def add_param(self, name, num=1):
        self.net_param.this = self.net_param.layer[name]
        for i in range(num):
            self.net_param.add_param(name, 0, 0)

    def type2names(self, layer_type='Convolution'):
        if layer_type not in self._layers:
            self._layers[layer_type] = self.net_param.type2names(layer_type)
        return self._layers[layer_type]

    def set_layerparam(self, layers_param=[]):
        for layer_param in layers_param:
            if layer_param.type == 'Convolution':
                if layer_param.lr_mult is not None and layer_param.decay_mult is not None:
                    self.set_param(layer_param.name, layer_param.lr_mult, layer_param.decay_mult)
                if layer_param.propagate_down is not None:
                    self.net_param.set_propagate_down(layer_param.name, layer_param.propagate_down)
                param = {}
                for member, value in vars(layer_param).items():
                    if member not in ['lr_mult', 'decay_mult', 'propagate_down', 'type'] and value is not None:
                        param[member] = value
                self.set_conv(**param)
            elif layer_param.type == 'BatchNorm':
                if layer_param.lr_mult is not None and layer_param.decay_mult is not None:
                    self.set_param(layer_param.name, layer_param.lr_mult, layer_param.decay_mult)
                if layer_param.propagate_down is not None:
                    self.net_param.set_propagate_down(layer_param.name, layer_param.propagate_down)
                if layer_param.use_global_stats is not None:
                    self.net_param.set_use_global_stats(layer_param.name, layer_param.use_global_stats)
            elif layer_param.type == 'Scale':
                if layer_param.lr_mult is not None and layer_param.decay_mult is not None:
                    self.set_param(layer_param.name, layer_param.lr_mult, layer_param.decay_mult)
                if layer_param.propagate_down is not None:
                    self.net_param.set_propagate_down(layer_param.name, layer_param.propagate_down)
            elif layer_param.type == 'ReLU':
                if layer_param.propagate_down is not None:
                    self.net_param.set_propagate_down(layer_param.name, layer_param.propagate_down)
            elif layer_param.type == 'EvalDetection':
                if layer_param.propagate_down is not None:
                    self.net_param.set_propagate_down(layer_param.name, layer_param.propagate_down)
            elif layer_param.type == 'RegionLoss':
                if layer_param.propagate_down is not None:
                    self.net_param.set_propagate_down(layer_param.name, layer_param.propagate_down)
            else:
                NotImplementedError

    def insert_layers(self, insert_layers=[]):
        for insert_layer in insert_layers:
            if hasattr(insert_layer, 'type'):
                self.net_param.insert_input(dim=insert_layer.dim)
            else:
                bringforward = insert_layer.bringforward
                change = insert_layer.change
                update_nodes = insert_layer.update_nodes
                bringto = insert_layer.bringto
                layer_param = insert_layer.layerparam
                if layer_param.type == 'Convolution':
                    param = {}
                    for member, value in vars(layer_param).items():
                        if member not in ['propagate_down'] and value is not None:
                            param[member] = value
                    self.insert(**param,bringforward=bringforward, change=change,update_nodes=update_nodes,bringto=bringto)
                elif layer_param.type == 'BatchNorm':
                    param = {}
                    for member, value in vars(layer_param).items():
                        if member not in ['propagate_down'] and value is not None:
                            param[member] = value
                    self.insert(**param, bringforward=bringforward, change=change, update_nodes=update_nodes,bringto=bringto)
                elif layer_param.type == 'Scale':
                    param = {}
                    for member, value in vars(layer_param).items():
                        if member not in ['propagate_down'] and value is not None:
                            param[member] = value
                    self.insert(**param, bringforward=bringforward, change=change, update_nodes=update_nodes,bringto=bringto)
                elif layer_param.type == 'ReLU':
                    param = {}
                    for member, value in vars(layer_param).items():
                        if member not in ['propagate_down'] and value is not None:
                            param[member] = value
                    self.insert(**param, bringforward=bringforward, change=change, update_nodes=update_nodes, bringto=bringto)
                elif layer_param.type == 'Pooling':
                    param = {}
                    for member, value in vars(layer_param).items():
                        if member not in ['propagate_down'] and value is not None:
                            param[member] = value
                    self.insert(**param, bringforward=bringforward, change=change, update_nodes=update_nodes, bringto=bringto)
                elif layer_param.type == 'EvalDetection':
                    param = {}
                    for member, value in vars(layer_param).items():
                        if value is not None:
                            param[member] = value
                    self.insert_EvalDetection(**param, bringforward=bringforward, change=change, update_nodes=update_nodes, bringto=bringto)
                elif layer_param.type == 'RegionLoss':
                    param = {}
                    for member, value in vars(layer_param).items():
                        if value is not None:
                            param[member] = value
                    self.insert_RegionLoss(**param, bringforward=bringforward, change=change, update_nodes=update_nodes, bringto=bringto)
                elif layer_param.type == 'EuclideanLoss':
                    param = {}
                    for member, value in vars(layer_param).items():
                        if value is not None:
                            param[member] = value
                    self.insert_EuclideanLoss(**param, bringforward=bringforward, change=change, update_nodes=update_nodes, bringto=bringto)
                else:
                    NotImplementedError

    def remove_layers(self, rm_layernames=[]):
        for layer in rm_layernames:
            if layer in self.net_param.layer:
                self.remove(layer)
            else:
                return False
        return True

    def get_layernames(self, layer_type=[]):
        layer_names = self.net_param.type2names(layer_type)
        return layer_names



    def insert(self, bottom, name=None, type="Convolution", bringforward=True, change=True, update_nodes=None, bringto=None, **kwargs):
        self.net_param.set_cur(bottom[0])
        if type == "Convolution":
            self.net_param.Convolution(name, bottom=bottom, **kwargs)
            # clone previous layer
            if "stride" not in kwargs:
                new_conv_param = self.conv_param(name)
                while len(new_conv_param.stride):
                    new_conv_param.stride.remove(new_conv_param.stride[0])
                for i in self.conv_param(bottom[0]).stride:
                    new_conv_param.stride.append(i)

            # update input nodes for others
            if change:
                if update_nodes is None:
                    update_nodes = self.net_param.layer
                for i in update_nodes:
                    if i == name:
                        continue
                    if self.net_param.layer[i][0].type not in ["BatchNorm", "Scale", "ReLU"]:
                        for btm in bottom:
                            self.net_param.ch_bottom(i, name, btm)
            if bringforward:
                bottom = bottom[0]  #########################33
                if bringto is not None:
                    bottom = bringto
                self.net_param.bringforward(bottom)

        elif type == "BatchNorm":
            bottom = bottom[0]  #########################33
            bnname = self.net_param.BatchNorm(name, **kwargs)
            self.net_param.bringforward(bottom)
            return bnname

        elif type == "Scale":
            sname = self.net_param.Scale(name, **kwargs)
            bottom = bottom[0]  #########################33
            self.net_param.bringforward(bottom)
            return sname

        elif type == "ReLU":
            reluname = self.net_param.ReLU(name, **kwargs)
            bottom = bottom[0]  #########################33
            self.net_param.bringforward(bottom)
            return reluname

        elif type == "Pooling":
            poolname = self.net_param.Pooling(name,**kwargs)

            if change:
                if update_nodes is None:
                    update_nodes = self.net_param.layer
                for i in update_nodes:
                    if i == name:
                        continue
                    ## my adding
                    if self.net_param.layer[i][0].type not in ["BatchNorm", "Scale", "ReLU"]:
                        for btm in bottom:
                            self.net_param.ch_bottom(i, name, btm)
            if bringforward:
                bottom = bottom[0]  #########################33
                if bringto is not None:
                    bottom = bringto
                self.net_param.bringforward(bottom)
            return poolname

    def insert_EvalDetection(self, bottom, name=None, type="EvalDetection", bringforward=True, change=True, update_nodes=None, bringto=None, **kwargs):
        self.net_param.set_cur(bottom[0])
        assert type == "EvalDetection"
        self.net_param.EvalDetection(name, bottom=bottom, **kwargs)
        if change:
            if update_nodes is None:
                update_nodes = self.net_param.layer
            for i in update_nodes:
                if i == name:
                    continue
                ## my adding
                if self.net_param.layer[i][0].type not in ["BatchNorm", "Scale", "ReLU"]:
                    for btm in bottom:
                        self.net_param.ch_bottom(i, name, btm)
        if bringforward:
            bottom = bottom[0]               #########################33
            if bringto is not None:
                bottom = bringto
            self.net_param.bringforward(bottom)

    def insert_EuclideanLoss(self, bottom, name=None, type="EuclideanLoss", bringforward=True, change=True, update_nodes=None, bringto=None, **kwargs):
        self.net_param.set_cur(bottom[0])
        assert type == "EuclideanLoss"
        self.net_param.EuclideanLoss(name, bottom=bottom, **kwargs)
        if change:
            if update_nodes is None:
                update_nodes = self.net_param.layer
            for i in update_nodes:
                if i == name:
                    continue
                ## my adding
                if self.net_param.layer[i][0].type not in ["BatchNorm", "Scale", "ReLU"]:
                    for btm in bottom:
                        self.net_param.ch_bottom(i, name, btm)
        if bringforward:
            bottom = bottom[0]
            if bringto is not None:
                bottom = bringto
            self.net_param.bringforward(bottom)

    def insert_RegionLoss(self, bottom, name=None, type="RegionLoss", bringforward=True, change=True, update_nodes=None, bringto=None, **kwargs):
        self.net_param.set_cur(bottom[0])
        assert type == "RegionLoss"
        self.net_param.RegionLoss(name, bottom=bottom, **kwargs)
        if change:
            if update_nodes is None:
                update_nodes = self.net_param.layer
            for i in update_nodes:
                if i == name:
                    continue
                ## my adding
                if self.net_param.layer[i][0].type not in ["BatchNorm", "Scale", "ReLU"]:
                    for btm in bottom:
                        self.net_param.ch_bottom(i, name, btm)
        if bringforward:
            bottom = bottom[0]  #########################33
            if bringto is not None:
                bottom = bringto
            self.net_param.bringforward(bottom)

    def remove(self, name, inplace=False):
        self.net_param.rm_layer(name, inplace)

    def finalmodel(self, WPQ=None, **kwargs): # the prefix for the name of the saved model is added by self.linear() -by Mario
        """ load weights into caffemodel"""
        if WPQ is None:
            WPQ = self.WPQ
        return self.linear(WPQ, **kwargs)

    def infer_pad_kernel(self, W, origin_name):
        num_output, _, kernel_h, kernel_w = W.shape
        assert kernel_h in [3,1]
        assert kernel_w in [3,1]
        pad_h = 1 if kernel_h == 3 else 0
        pad_w = 1 if kernel_w == 3 else 0
        stride = self.conv_param(origin_name).stride
        if len(stride) == 1:
            pass
        elif len(stride) == 0:
            stride = [1]
        else:
            NotImplementedError
        if stride[0] == 1:
            pass
        elif stride[0] == 2:
            stride = [stride[0] if pad_h else 1, stride[0] if pad_w else 1]
            # stride = [1 if pad_h else stride[0], 1 if pad_w else stride[0]]
            warn("stride 2 decompose dangerous")
        else:
            NotImplementedError
        return {"pad_h":pad_h, "pad_w":pad_w, "kernel_h":kernel_h, "kernel_w":kernel_w, "num_output":num_output, "stride":stride}
    # =========algorithms=========

    def linear(self, WPQ, prefix='VH', save=False, DEBUG=0):   #原默认save=True
        for i, j in WPQ.items():
            if save:
                self.set_param_data(i, j)
            else:
                self.ch_param_data(i, j)
        if save:
            return self.save_caffemodel(prefix=prefix)


    def add_bn(self, ids=True):
        forbid = ['_id', '_proj']
        # loop over all samples
        bs = self.blobs_shape('data')[0]
        # self.usexyz(False)
        # self.dp.dataset_size
        iters = 50000 // bs
        means ={}
        variances = {}
        for i in range(iters):
            print(i,iters)
            self.forward()
            for conv in self.nonsconvs:
                if conv not in means:
                    means[conv]=[]
                    variances[conv] = []
                else:
                    means[conv].append(self.blobs_data(conv).mean((0,2,3)))
                    variances[conv].append(self.blobs_data(conv).var((0,2,3)))
        for r in self.relus:
            if self.top_names[r][0] == r:
                conv = self.bottom_names[r][0]
                if conv not in self.convs:
                    continue
                self.net_param.ch_top(r, conv, r)
                for i in self.net_param.layer:
                    if i != r:
                        self.net_param.ch_bottom(i, conv, r)

        for conv in self.nonsconvs:
            skip=False
            if forbid[0] in conv:
                # prune ids
                self.remove(conv)
                print("remove", conv)
                skip = True
            if forbid[1] in conv:
                skip = True
            if skip:
                continue
            bn, scal = self.insert(bottom=conv, layer_type=layertypes.BatchNorm)
            self.WPQ[(scal,0)] = np.array(variances[conv]).mean(0)**.5
            self.WPQ[(scal,1)] = np.array(means[conv]).mean(0)

        pt = self.save_pt(prefix='s')
        print("ready to train", pt)
        return pt, self.WPQ

    def layercomputation(self, conv, channels=1., outputs=1.):
        bottom = self.bottom_names[conv]
        assert len(bottom) == 1
        bottom = bottom[0]
        s = self.blobs_shape(bottom)
        p = self.param_shape(conv)
        if conv in self.convs:
            if conv in self.spation_convs:
                channels = 1
            else:
                assert s[1]==p[1]
                channels *= p[1]
            outputs *= p[0]
            c = s[2]*s[3]*outputs*channels*p[2]*p[3] / self.conv_param_stride(conv)**2
        elif conv in self.innerproduct:
            c = p[0]*p[1]
        else:
            pass
        return int(c)

    def computation(self, params=False):
        comp=0
        if params:
            NotImplementedError
        else:
            l = []
            for conv in self.convs:
                l.append(self.layercomputation(conv))
        comp = sum(l)
        print("flops", comp)
        for conv,i in zip(self.convs, l):
            print(conv, i, int(i*1000/comp))
        return comp

    def getBNaff(self, bn, affine, scale=1.):
        eps = 1e-9
        mean = scale * self.param_data(bn)
        variance = (scale * self.param_b_data(bn) + eps)**.5
        k =  self.param_data(affine)
        b =  self.param_b_data(affine)
        return mean, variance, k, b

    def merge_bn(self, DEBUG=0):                                 #######################################################################
        """
        Return:
            merged Weights
        """
        print("begin merge_bn")
        nobias=False
        def scale2tensor(s):
            return s.reshape([len(s), 1, 1, 1])

        BNs = self.type2names("BatchNorm")
        Affines = self.type2names("Scale")
        ReLUs = self.type2names("ReLU")
        Convs = self.type2names()
        assert len(BNs) == len(Affines)

        WPQ = dict()
        for affine in Affines:
            if self.bottom_names[affine][0] in BNs:
                # non inplace BN
                noninplace = True
                bn = self.bottom_names[affine][0]
                conv = self.bottom_names[bn][0]
                assert conv in Convs

            else:
                noninplace = False
                conv = self.bottom_names[affine][0]
                for bn in BNs:
                    if self.bottom_names[bn][0] == conv:
                        break

            triplet = (conv, bn, affine)
            print("Merging", triplet)

            if not DEBUG:
                scale = 1.

                mva = self.param(bn)[2].data[0]
                if mva != scale:
                    #raise Exception("Using moving average "+str(mva)+" NotImplemented")
                    scale /= mva

                mean, variance, k, b = self.getBNaff(bn, affine, scale)
                # y = wx + b
                # (y - mean) / var * k + b
                weights = self.param_data(conv)
                weights = weights / scale2tensor(variance) * scale2tensor(k)

                if len(self.param(conv)) == 1:
                    bias = np.zeros(weights.shape[0])
                    self.set_conv(conv, bias=True)
                    self.param(conv).append(self.param_b(bn))
                    nobias=True
                else:
                    bias = self.param_b_data(conv)
                bias -= mean
                bias = bias / variance * k + b

                WPQ[(conv, 0)] = weights
                WPQ[(conv, 1)] = bias

            self.remove(affine)
            self.remove(bn)
            if not noninplace:
                have_relu=False
                for r in ReLUs:
                    if self.bottom_names[r][0] == conv:
                        have_relu=True
                        break
                if have_relu:
                    self.net_param.ch_top(r, r, conv)
                    for i in self.net_param.layer:
                        if i != r:
                            self.net_param.ch_bottom(i, r, conv)

        if True:
            if not nobias:
                new_pt = self.save_pt(prefix = 'bn')
                return WPQ, new_pt
            else:
                new_pt, new_model = self.save(prefix='bn')
                return WPQ, new_pt, new_model

        new_pt, new_model = self.save(prefix='bn')
        print("end merge_bn")
        return WPQ, new_pt, new_model

    def invBN(self, arr, Y_name):
        if isinstance(arr, int) or len(self.bns) == 0 or len(self.scales) == 0:
            return arr
        interstellar = Y_name.split('_')[0]
        for i in self.bottom_names[interstellar]:
            if i in self.bns and 'branch2c' in i:
                bn = i
                break
        for i in self.scales:
            if self.layer_bottom(i) == bn:
                affine = i
                break

        if 1: print('inverted bn', bn, affine, Y_name)
        mean, std, k, b = self.getBNaff(bn, affine)
        # (y - mean) / std * k + b
        #return (arr - b) * std / k + mean
        return arr * std / k
        #embed()


    def save_no_bn(self, WPQ, prefix='bn'):
        self.forward()
        for i, j in WPQ.items():
            self.set_param_data(i, j)

        return self.save_caffemodel(prefix=prefix)

    def get_layers_bytop(self, top):
        layers = []
        for layer in self.net_param.layer:
            layer_top = self.net_param.layer_top(layer)
            if layer_top == top:
                layers.append(layer)
        return layers

    def set_data_source(self, source=None):
        data_layer = self.net_param.layer['data']
        if len(data_layer) == 2:
            assert len(source) == 2
            train_data_param = data_layer[0].data_param
            train_data_param.source = source[0]

            test_data_param = data_layer[1].data_param
            test_data_param.source = source[1]
        else:
            assert len(source) == 1
            data_param = data_layer.data_param
            data_param.source = source[0]
        return True

    def get_data_source(self):
        data_layer = self.net_param.layer['data']
        source = []
        if len(data_layer) == 2:
            train_data_param = data_layer[0].data_param
            source.append(train_data_param.source)

            test_data_param = data_layer[1].data_param
            source.append(test_data_param.source)
        else:
            data_param = data_layer.data_param
            source.append(data_param.source)
        return source

def arr2strarr(*kwargs):
    mylist = []
    for i in kwargs:
        if isinstance(i, str):
            mylist.append(i)
            continue
        mylist.append(str(i))
    return mylist



def underline(*kwargs):
    strarr = arr2strarr(*kwargs)
    return '_'.join(strarr)

def generate_solver(savepath, train_net_prototxt, max_iter, snapshot_prefix):
    solver_str = '''
net: "%s"
test_iter: 558
test_interval: 99999999
test_initialization: false
display: 20
average_loss: 500
lr_policy: "multifixed"
stagelr:   0.00001
stageiter: 10000
max_iter: %d
momentum: 0.99
weight_decay: 0.0005
snapshot: 5000
snapshot_prefix: "%s"
solver_mode: GPU

''' % (train_net_prototxt, max_iter, snapshot_prefix)
    fp = open(savepath, 'w')
    fp.write(solver_str)
    fp.close()
