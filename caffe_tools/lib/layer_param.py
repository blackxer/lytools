class Conv_param():
    def __init__(self, name, bottom=None, top=None, num_output=None, new_name=None, pad_h=None, pad_w=None, kernel_h=None, kernel_w=None,
                 stride=None, bias=None, group=None, lr_mult=None, decay_mult=None, propagate_down=None):
        self.name = name
        self.type = 'Convolution'
        self.bottom = bottom
        self.top = top
        self.num_output = num_output
        self.new_name = new_name
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.stride = stride
        self.bias = bias
        self.group = group
        self.lr_mult = lr_mult
        self.decay_mult = decay_mult
        self.propagate_down = propagate_down


class Bn_param():
    def __init__(self, name=None, bottom=None, top=None, use_global_stats=None, lr_mult=None, decay_mult=None, propagate_down=None):
        self.name = name
        self.type = 'BatchNorm'
        self.bottom = bottom
        self.top = top
        self.use_global_stats = use_global_stats
        self.lr_mult = lr_mult           ##如果prototxt中没有该参数的话，则设置不了
        self.decay_mult = decay_mult
        self.propagate_down = propagate_down

class Scale_param():
    def __init__(self, name=None, bottom=None, top=None, lr_mult=None, decay_mult=None, propagate_down=None, bias=None):
        self.name = name
        self.type = 'Scale'
        self.bottom = bottom
        self.top = top
        self.lr_mult = lr_mult           ##如果prototxt中没有该参数的话，则设置不了
        self.decay_mult = decay_mult
        self.propagate_down = propagate_down
        self.bias = bias

class Relu_param():
    def __init__(self, name=None, bottom=None, top=None, propagate_down=False):
        self.name = name
        self.type = 'ReLU'
        self.bottom = bottom
        self.top = top
        self.propagate_down = propagate_down

class Pool_param():
    def __init__(self, name=None, bottom=None, top=None, propagate_down=False, pool=None, kernel_size=None, stride=None, global_pooling=None):
        self.name = name
        self.type = 'Pooling'
        self.bottom = bottom
        self.top = top
        self.propagate_down = propagate_down
        self.pool = pool
        self.kernel_size = kernel_size
        self.stride = stride
        self.global_pooling = global_pooling

class Evaldetection_param():
    def __init__(self, name=None, bottom=None, top=None, propagate_down=None, side_w=None, side_h=None, num_class=None,
                 num_object=5, threshold=None, nms=None, max_objnum=None):
        self.name = name
        self.type = 'EvalDetection'
        self.bottom = bottom
        self.top = top
        self.propagate_down = propagate_down
        self.side_w = side_w
        self.side_h = side_h
        self.num_class = num_class
        self.num_object = num_object
        self.threshold = threshold
        self.nms = nms
        self.max_objnum =max_objnum

class RegionLoss_param():
    def __init__(self, name=None, bottom=None, top=None, propagate_down=None, loss_weight=None, phase=None, side_w=None, side_h=None, num_class=None,
                 coords=None, num=None, softmax=None, jitter=None, rescore=None, object_scale=None, noobject_scale=None,
                 class_scale=None, coord_scale=None, absolute=None, thresh=None, random=None):
        self.name = name
        self.type = 'RegionLoss'
        self.bottom = bottom
        self.top = top
        self.propagate_down = propagate_down
        self.loss_weight = loss_weight
        self.phase = phase
        self.side_w = side_w
        self.side_h = side_h
        self.num_class = num_class
        self.coords = coords
        self.num = num
        self.softmax =softmax
        self.jitter = jitter
        self.rescore = rescore
        self.object_scale = object_scale
        self.noobject_scale = noobject_scale
        self.class_scale = class_scale
        self.coord_scale = coord_scale
        self.absolute = absolute
        self.thresh = thresh
        self.random = random

class Euclideanloss_param():
    def __init__(self, name=None, bottom=None, top=None):
        self.name = name
        self.type = 'EuclideanLoss'
        self.bottom = bottom
        self.top = top

class Input_param():
    def __init__(self, dim=[1,3,256,512]):
        self.dim = dim
        self.type = 'deploy'

class Insert_layer():
    def __init__(self, layerparam=None, bringforward=True, change=True, update_nodes=None, bringto=None):
        self.layerparam = layerparam
        self.bringforward = bringforward
        self.change = change
        self.update_nodes = update_nodes
        self.bringto = bringto


