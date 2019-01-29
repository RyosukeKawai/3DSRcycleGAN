# coding:utf-8
import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np

class EdgeEnhanceFilter(chainer.Chain):
    def __init__(self, in_channels, out_channels, direction='x'):
        super(EdgeEnhanceFilter, self).__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels
        filter_weight, self.pad_width = self.check_direction(direction)
        with self.init_scope():
            self.filter = L.ConvolutionND(ndim=3,
                                   in_channels = in_channels,
                                   out_channels = out_channels,
                                   ksize=filter_weight.shape[2:], pad=0, nobias=True)
            self.filter.W.array = filter_weight

    def check_direction(self, direction):
        xp = self.xp
        in_ch = self.in_ch
        out_ch = self.out_ch

        def get_filter_weight(shape=[]):
            """
            Assume:
            shape = [z, y, x]
            """
            if not len(shape)==3:
                raise NotImplementedError('Length Error, shape size {}'.format(len(shape)))

            filter_weight = xp.asarray([-1, 0, 1], dtype=xp.float32)

            filter_weight = filter_weight.reshape(1, shape[0], shape[1], shape[2]) # (inch, z, y, x)
            filter_weight = xp.repeat(filter_weight, in_ch, axis=0)
            filter_weight = filter_weight.reshape(1, -1,  shape[0], shape[1], shape[2]) # (chout, chin, z, y, x)
            filter_weight = xp.repeat(filter_weight, out_ch, axis=0)

            return filter_weight

        if direction.lower() == 'x':
            filter_weight = get_filter_weight([1, 1, 3])
            pad_width = ((0,0), (0,0), (0,0), (0,0), (1,1))
            return filter_weight, pad_width

        elif direction.lower() == 'y':
            filter_weight = get_filter_weight([1, 3, 1])
            pad_width = ((0,0), (0,0), (0,0), (1,1), (0,0))
            return filter_weight, pad_width

        elif direction.lower() == 'z':
            filter_weight = get_filter_weight([3, 1, 1])
            pad_width = ((0,0), (0,0), (1,1), (0,0), (0,0))
            return filter_weight, pad_width

        else:
            raise NotImplementedError('Direction [{}] is not found'.format(direction))

    def forward(self, x):
        return self.filter(F.pad(x, pad_width=self.pad_width, mode='edge'))

class GradientDifferenceLoss(chainer.Chain):
    def __init__(self):
        """
        Reference:
        https://github.com/ginobilinie/medSynthesis/blob/master/loss_functions.py#L36
        """
        super(GradientDifferenceLoss, self).__init__()
        with self.init_scope():
            self.filter_x = EdgeEnhanceFilter(in_channels=1, out_channels=1, direction='x')
            self.filter_y = EdgeEnhanceFilter(in_channels=1, out_channels=1, direction='y')
            self.filter_z = EdgeEnhanceFilter(in_channels=1, out_channels=1, direction='z')

    def __call__(self, x, gt):
        batchsize = len(x.array)

        gen_dx = F.absolute(self.filter_x(x))
        gen_dy = F.absolute(self.filter_y(x))
        gen_dz = F.absolute(self.filter_z(x))

        gt_dx = F.absolute(self.filter_x(gt))
        gt_dy = F.absolute(self.filter_y(gt))
        gt_dz = F.absolute(self.filter_z(gt))

        loss = F.squared_error(gt_dx, gen_dx) + F.squared_error(gt_dy, gen_dy) + F.squared_error(gt_dz, gen_dz)
        loss = F.sum(loss) / batchsize

        return loss

class GradientDifferenceLossV2(chainer.Chain):
    def __init__(self):
        """
        Reference:
        This function dont use L2 norm
        """
        super(GradientDifferenceLoss, self).__init__()
        with self.init_scope():
            self.filter_x = EdgeEnhanceFilter(in_channels=1, out_channels=1, direction='x')
            self.filter_y = EdgeEnhanceFilter(in_channels=1, out_channels=1, direction='y')
            self.filter_z = EdgeEnhanceFilter(in_channels=1, out_channels=1, direction='z')

    def __call__(self, x, gt):
        gen_dx = F.absolute(self.filter_x(x))
        gen_dy = F.absolute(self.filter_y(x))
        gen_dz = F.absolute(self.filter_z(x))

        gt_dx = F.absolute(self.filter_x(gt))
        gt_dy = F.absolute(self.filter_y(gt))
        gt_dz = F.absolute(self.filter_z(gt))

        loss = F.mean_squared_error(gt_dx, gen_dx) + F.mean_squared_error(gt_dy, gen_dy) + F.mean_squared_error(gt_dz, gen_dz)

        return loss
