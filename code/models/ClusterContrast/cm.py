import torch
import torch.nn.functional as F
from torch import autograd



import torch
import torch.nn.functional as F
from torch import autograd


class NoiseAwareCM(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, clusters, features, smooth_weight, use_weight=True):
        alpha = 1
        ctx.features = features

        outputs = inputs.mm(ctx.features.t())  # [b, n]

        unique_clusters = torch.unique(clusters)
        select_features = torch.zeros_like(inputs[:len(unique_clusters)])

        for i, unique_cluster in enumerate(unique_clusters):
            index = torch.where(clusters == unique_cluster)[0]
            cluster_inputs = inputs[index]  # [k, d]
            cluster_center = features[unique_cluster]  # [d]

            # 1. 计算距离 (与原 CM 一致)
            cluster_outputs = cluster_inputs.mm(cluster_center.unsqueeze(1)).squeeze(1)
            d = 1.0 - cluster_outputs  # [k]

            # 2. 归一化为噪声概率
            p = d / (d.max() + 1e-6)

            # 3. 权重 (softmax of exp(-αp))
            weights = torch.softmax(-alpha * p, dim=0)

            # 4. 加权更新簇原型
            select_features[i] = (cluster_inputs * weights.unsqueeze(1)).sum(dim=0)

        ctx.save_for_backward(select_features, unique_clusters)
        ctx.alpha = alpha
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        select_features, unique_clusters = ctx.saved_tensors
        grad_inputs = None

        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features.half())

        # 更新 prototype
        for x, y in zip(select_features, unique_clusters):
            ctx.features[y] = x / (x.norm() + 1e-6)

        return grad_inputs, None, None, None, None



class CM(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, clusters, features, smooth_weight, use_weight = False):
        # 保存了特征数据、动量和缩放因子等信息到 ctx 上下文中，以便在反向传播中使用。
        ctx.features = features

        outputs = inputs.mm(ctx.features.t()) # b,f * f,n = b,n

        unique_clusters = torch.unique(clusters)
        select_features = torch.zeros_like(inputs[:len(unique_clusters)])

        # 处理每个簇
        for i, unique_cluster in enumerate(unique_clusters):
            index = torch.where(clusters == unique_cluster)[0]  # 获取当前簇的索引
            cluster_inputs = inputs[index]  # 获取当前簇的输入特征
            cluster_outputs = outputs[index, unique_cluster]  # 获取当前簇的输出

            if use_weight:
                d = 1.0 - cluster_outputs  # 计算距离度量
                above_mean = d >= d.mean()  # 找到大于平均值的部分
                weights = F.softmax(d[above_mean], dim=0) * smooth_weight  # 计算权重并做软最大化处理
                weights[torch.argmax(d[above_mean])] += (1-smooth_weight)  # 给最大值增加额外权重
                select_features[i] = (cluster_inputs[above_mean] * weights.unsqueeze(1)).sum(dim=0)  # 计算加权特征
            else:
                select_features[i] = cluster_inputs[torch.randint(len(index), (1,))]  # 随机选择特征

        ctx.save_for_backward(select_features, unique_clusters)
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        select_features, unique_clusters = ctx.saved_tensors

        grad_inputs  = None
        # 如果需要计算梯度，则使用输出梯度与特征数据进行矩阵乘法运算。
        if ctx.needs_input_grad[0]:
            grad_inputs  = grad_outputs.mm(ctx.features.half())

        for x, y in zip(select_features, unique_clusters):
             ctx.features[y] = x / x.norm()    #直接替换

        # 返回计算得到的输入数据的梯度，以及其他不需要计算梯度的 None 值。
        return grad_inputs, None, None, None, None


class HM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features.half())

        # momentum update
        for x, y in zip(inputs, indexes):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def hm(inputs, indexes, features, momentum=0.5):
    return HM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))
