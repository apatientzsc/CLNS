import torch.nn.functional as F
from code.tools.cross_modal_match.match import *
from torch import nn
from code.tools.loss.MMDLoss import MMD_loss

from .record_memory import RecordMemory


class CrossMemory(nn.Module):
    def __init__(self, gnn_k1_cross, gnn_k2_cross, temp, smooth_weight, loss_type, bg_knn, use_id_loss, use_weight, stage, balance, use_soft, otpm, ncmu, reg):
        super(CrossMemory, self).__init__()
        self.smooth_weight, self.temp = smooth_weight, temp
        self.k1, self.k2 = gnn_k1_cross, gnn_k2_cross

        self.r2i_proxy, self.i2r_proxy = {}, {}
        self.r2i_label, self.i2r_label = {}, {}
        self.use_id_loss = use_id_loss
        self.stage = stage
        self.balance = balance
        self.use_soft = use_soft
        self.otpm = otpm
        self.ncmu = ncmu
        self.reg = reg


        #初始化记忆库
        self.rgb_memory = RecordMemory(self.smooth_weight, self.temp, loss_type, bg_knn, use_id_loss, use_weight, stage, balance, ncmu)
        self.ir_memory = RecordMemory(self.smooth_weight, self.temp, loss_type, bg_knn, use_id_loss, use_weight, stage, balance, ncmu)
    def refresh(self):
        #重置
        self.r2i_proxy, self.i2r_proxy = {}, {}
        self.r2i_label, self.i2r_label = {}, {}
        self.rgb_memory.refresh()
        self.ir_memory.refresh()

    def update(self, rgb_info, ir_info):  #更新两模态记忆库
        self.rgb_memory.update(rgb_info)
        self.ir_memory.update(ir_info)

    def creat_cross(self):                   #生成跨膜态伪标签
        if self.otpm:
            self.r2i_proxy, self.i2r_proxy = otpm(self.k1, self.k2, self.rgb_memory.proxy_centers,self.ir_memory.proxy_centers, self.reg)
            self.r2i_label, self.i2r_label = otpm(self.k1, self.k2, self.rgb_memory.class_centers,self.ir_memory.class_centers, self.reg)

        else:
            self.r2i_proxy, self.i2r_proxy = cross_modal_matching(self.k1, self.k2, self.rgb_memory.proxy_centers,
                                                                  self.ir_memory.proxy_centers, None, None)
            self.r2i_label, self.i2r_label = cross_modal_matching(self.k1, self.k2, self.rgb_memory.class_centers,
                                                                  self.ir_memory.class_centers, None, None)


    def get_cross_proxies(self):
        # 返回 r2i_proxy 和 i2r_proxy
        return self.r2i_proxy, self.i2r_proxy

    def forward(self,
                inputs_rgb, proxy_rgb, label_rgb, correlation_rgb, weight_rgb, correlation_rgb_cross, weight_rgb_cross, cross_proxy_rgb, cross_label_rgb, cross_class_rgb,
                inputs_ir, proxy_ir, label_ir, correlation_ir, weight_ir, correlation_ir_cross, weight_ir_cross, cross_proxy_ir, cross_label_ir, cross_class_ir
                ):

        inputs_rgb, inputs_ir = map(lambda x: F.normalize(x, dim=1), (inputs_rgb, inputs_ir))

        if self.use_soft:
            loss_proxy_rgb = self.rgb_memory.compute_soft_loss(inputs_rgb, proxy_rgb, label_rgb, label_rgb, correlation_rgb, weight_rgb)
            loss_proxy_ir = self.ir_memory.compute_soft_loss(inputs_ir, proxy_ir, label_ir, label_ir, correlation_ir, weight_ir)

            # Lcross
            loss_cross_rgb = self.ir_memory.compute_soft_loss(inputs_rgb, cross_proxy_rgb, cross_label_rgb,
                                                              cross_class_rgb, correlation_rgb_cross, weight_rgb_cross)
            loss_cross_ir = self.rgb_memory.compute_soft_loss(inputs_ir, cross_proxy_ir, cross_label_ir, cross_class_ir,
                                                              correlation_ir_cross, weight_ir_cross)

        else:
            loss_proxy_rgb = self.rgb_memory.compute_loss(inputs_rgb, proxy_rgb, label_rgb, label_rgb)
            loss_proxy_ir = self.ir_memory.compute_loss(inputs_ir, proxy_ir, label_ir, label_ir)
            # Lcross
            loss_cross_rgb = self.ir_memory.compute_soft_loss(inputs_rgb, cross_proxy_rgb, cross_label_rgb,
                                                              cross_class_rgb, None, None)
            loss_cross_ir = self.rgb_memory.compute_soft_loss(inputs_ir, cross_proxy_ir, cross_label_ir, cross_class_ir,
                                                              None, None)


        return loss_proxy_rgb, loss_proxy_ir, loss_cross_rgb, loss_cross_ir

def creat_Memory(args):    #创建cross_memory实例
    model = CrossMemory(
        gnn_k1_cross=args.gnn_k1_cross,
        gnn_k2_cross=args.gnn_k2_cross,
        temp = args.temp,
        loss_type = args.loss_type,
        bg_knn = args.bg_knn,
        use_id_loss = args.use_id_loss,
        smooth_weight=args.smooth_weight,
        use_weight=args.use_weight,
        stage=args.stage,
        balance = args.balance,
        use_soft = args.use_soft,
        otpm = args.otpm,
        ncmu = args.ncmu,
        reg = args.reg
    )
    return model
