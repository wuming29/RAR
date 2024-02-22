# coding: utf-8
# Copyright @tongshiwei

from .DKTModule import Parameters, DKTModule


def load_net(load_epoch=Parameters.end_epoch, **kwargs):
    params = Parameters(
        **kwargs
    )

    mod = DKTModule(params)

    net = mod.sym_gen()
    net = mod.load(net, load_epoch, mod.params.ctx)
    return net


# todo 重命名eval_DKT函数到需要的模块名
def eval_DKT(load_epoch):
    net = load_net()
    pass


# todo 重命名use_DKT函数到需要的模块名
class DKT(object):
    def __init__(self, load_epoch):
        self.net = load_net()

    def __call__(self, *args, **kwargs):
        pass
