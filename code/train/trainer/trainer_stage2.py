from statistics import correlation

import torch.nn.functional as F
import torch
from code.train.trainer.base_trainer import Base_trainer


class Trainer_stage2(Base_trainer):
    def __init__(self,model, args, kind, writer = None, optimizer = None, scheduler = None, scaler = None):
        super().__init__(model, optimizer, scheduler, scaler, writer, args, kind)
        self.train_metric_names = ['Loss_All', 'Loss_RGB', 'Loss_IR', 'Loss_Cross_RGB', 'Loss_Cross_IR']

    @staticmethod
    def _parser_data(item):
        if len(item) == 12:
            img = torch.cat((item[0], item[1]), 0)
            proxy = torch.cat((item[2], item[2]), 0)
            label = torch.cat((item[3], item[3]), 0)
            cam = torch.cat((item[4], item[4]), 0)
            correlations = torch.cat((item[5], item[5]), 0)
            weights = torch.cat((item[6], item[6]), 0)
            correlations_cross = torch.cat((item[7], item[7]), 0)
            weights_cross = torch.cat((item[8], item[8]), 0)
            cross_proxy = torch.cat((item[9], item[9]), 0)
            cross_label = torch.cat((item[10], item[10]), 0)
            cross_class = torch.cat((item[11], item[11]), 0)
        else:
            img, proxy, label, cam, correlations, weights, correlations_cross, weights_cross, cross_proxy, cross_label, cross_class = item

        img, proxy, label, cross_proxy, cross_label, cross_class = [x.cuda() for x in [img, proxy, label, cross_proxy, cross_label, cross_class]]
        return img, proxy, label, cam, correlations, weights, correlations_cross, weights_cross,cross_proxy, cross_label, cross_class



    def run(self, rgb_trainloader, ir_trainloader, magnification = 1):
        self._pre_for_train(rgb_trainloader, ir_trainloader)
        metrics = self._init_metric(self.train_metric_names)
        for i in range(self.train_iters):
            self.optimizer.zero_grad()

            img_rgb, proxy_rgb, label_rgb, cam_rgb, correlations_rgb, weights_rgb, correlations_rgb_cross, weights_rgb_cross,cross_proxy_rgb, cross_label_rgb, cross_class_rgb = self._parser_data(rgb_trainloader.next())
            img_ir, proxy_ir, label_ir, cam_ir, correlations_ir, weights_ir, correlations_ir_cross, weights_ir_cross, cross_proxy_ir, cross_label_ir, cross_class_ir = self._parser_data(ir_trainloader.next())

            batch_size = self.batch_size * magnification
            with self.amp_autocast():
                feats = self.model(x1=img_rgb, x2=img_ir, modal=0)
                feats_rgb, feats_ir = feats[: -batch_size], feats[-batch_size:]





                loss_proxy_rgb, loss_proxy_ir, loss_cross_rgb, loss_cross_ir = self.memory(
                    feats_rgb, proxy_rgb, label_rgb, correlations_rgb, weights_rgb, correlations_rgb_cross, weights_rgb_cross, cross_proxy_rgb, cross_label_rgb, cross_class_rgb,
                    feats_ir, proxy_ir, label_ir, correlations_ir, weights_ir, correlations_ir_cross, weights_ir_cross, cross_proxy_ir, cross_label_ir, cross_class_ir
                )
                loss = loss_proxy_rgb + loss_proxy_ir + loss_cross_rgb + loss_cross_ir

            self._optimize(loss)
            self._update_train(
                i=i,
                metrics = metrics, magnification = magnification,
                Loss_All=loss, Loss_RGB=loss_proxy_rgb, Loss_IR=loss_proxy_ir, Loss_Cross_RGB=loss_cross_rgb, Loss_Cross_IR=loss_cross_ir
            )
        self.scheduler.step()
        self.pbar.close()
        self._record_metrics(metrics, 'train')