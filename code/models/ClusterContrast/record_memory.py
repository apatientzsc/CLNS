import torch
import torch.nn.functional as F
from code.tools import loss
from code.tools.output.log_utils import AverageMeter
from code.tools.utils.train_utils import generate_center_features, creat_dic
from torch import nn

from .cm import *

class RecordMemory(nn.Module):
    def __init__(self, smooth_weight, temp, loss_type, bg_knn, use_id_loss, use_weight, stage, balance, ncmu):
        super(RecordMemory, self).__init__()

        self.unique_cameras = None
        self.label2proxy,self.cam2proxy,self.proxy2label,self.proxy2cam = None, None, None, None
        self.label2instance, self.cam2instance, self.proxy2instance, self.instance2proxy, self.instance2cam, self.instance2label = None, None, None, None, None, None
        self.proxy_centers = None

        self.get_proxy_associate_loss = loss.create(loss_type)
        self.get_soft_proxy_associate_loss = loss.create(loss_type)
        self.temp = temp
        self.smooth_weight = smooth_weight
        self.use_weight = use_weight
        self.use_id_loss = use_id_loss
        self.bg_knn = bg_knn
        self.stage = stage
        self.balance = balance
        self.ncmu = ncmu


        if self.stage == 2:
            self.class_centers = None

    def refresh(self):
        #重置所有映射
        self.unique_cameras = None
        self.label2proxy,self.cam2proxy,self.proxy2label,self.proxy2cam = None, None, None, None
        self.label2instance, self.cam2instance, self.proxy2instance, self.instance2proxy, self.instance2cam, self.instance2label = None, None, None, None, None, None
        self.proxy_centers = None

        if self.stage == 2:
            self.class_centers = None

    def update(self, info):
        [features, pseudo_labels, proxy_labels, cams, images, n_class, n_instance, real_labels, correlations_all, weights_all] = info

        unique_cameras = torch.unique(cams).cpu()
        self.unique_cameras= unique_cameras
        self.label2proxy, self.cam2proxy, self.proxy2label, self.proxy2cam,\
        self.label2instance, self.proxy2instance, self.cam2instance, self.instance2label, self.instance2proxy, self.instance2cam = \
        creat_dic(pseudo_labels,proxy_labels,unique_cameras,cams)

        #生成类别簇中心和相机代理簇中心
        self.class_centers = generate_center_features(features.cuda(),pseudo_labels.cuda())
        self.proxy_centers = generate_center_features(features.cuda(),proxy_labels.cuda())

    def compute_loss(self, inputs, proxies, labels, classes=None):
        loss_proxy, loss_class = 0, 0

        # ALL
        if proxies is not None and labels is not None:
            outputs_proxy = CM.apply(inputs, proxies, self.proxy_centers, self.smooth_weight, self.use_weight)
            scores_proxy = outputs_proxy / self.temp

            s_proxies = self.label2proxy[labels]
            loss_proxy = self.get_soft_proxy_associate_loss(self.cam2proxy, s_proxies, scores_proxy, bg_knn=self.bg_knn)

        if self.use_id_loss and classes is not None:
            outputs_label = CM.apply(inputs, classes, self.class_centers, self.smooth_weight, self.use_weight)
            scores_label = outputs_label / self.temp

            loss_class = F.cross_entropy(scores_label, classes.to(torch.long))

        return loss_proxy + loss_class

    def compute_soft_loss(self, inputs, proxies, labels, classes=None, correlation=None, weight=None):
        loss_proxy, loss_class = 0, 0

        # ALL
        if proxies is not None and labels is not None:
            # proxy_center 代理簇中心[num_proxy,d]    outputs_proxy [b,num_proxy] 样本与所有代理中心的相似度
            #outputs_proxy = CM.apply(inputs, proxies, self.proxy_centers, self.smooth_weight, self.use_weight)
            if self.ncmu:
                outputs_proxy = NoiseAwareCM.apply(inputs, proxies, self.proxy_centers, self.smooth_weight, self.use_weight)
                #outputs_proxy = HM.apply(inputs, proxies, self.proxy_centers, self.use_weight)
            else:
                outputs_proxy = CM.apply(inputs, proxies, self.proxy_centers, self.use_weight)

            #outputs_proxy = DualMemoryCM.apply(inputs, proxies, self.proxy_centers, self.smooth_weight, self.use_weight)
            scores_proxy = outputs_proxy / self.temp

            s_proxies = self.label2proxy[labels]   #每个样本在每个相机下对应正代理
            loss_proxy = self.get_soft_proxy_associate_loss(
                self.cam2proxy, s_proxies, scores_proxy, bg_knn=self.bg_knn,
                correlation=correlation, weight=weight, balance=self.balance
            )

        if self.use_id_loss and classes is not None:
            # input 输入特征[b,d]    classes 伪标签[b]   class 类别簇中心[n_class,d]
            # smooth_weight 平滑值 outputs_label 每个样本与所有标签相似度[b,n_classes]
            #outputs_label = CM.apply(inputs, classes, self.class_centers, self.smooth_weight, self.use_weight)
            if self.ncmu:
                outputs_label = NoiseAwareCM.apply(inputs, classes, self.class_centers, self.smooth_weight, self.use_weight)
                #outputs_label = HM.apply(inputs, classes, self.class_centers,self.use_weight)
            else:
                outputs_label = CM.apply(inputs, classes, self.class_centers, self.smooth_weight, self.use_weight)

            #outputs_label = DualMemoryCM.apply(inputs, classes, self.class_centers, self.smooth_weight, self.use_weight)
            scores_label = outputs_label / self.temp  # 缩放后相似度
            # InfoNCE损失，单个样本与其对应伪标签的聚类中心做交叉熵损失，使其更接近与该中心
            loss_class = F.cross_entropy(scores_label, classes.to(torch.long))

        return loss_class + loss_proxy