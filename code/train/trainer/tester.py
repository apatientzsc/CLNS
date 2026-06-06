import copy
import logging

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from code.tools import rerank
from code.tools.eavl.get_test_data import creat_test_data
from code.train.trainer.base_trainer import Base_trainer
from code.tools import eavl as e
import time

class Tester(Base_trainer):
    def __init__(self, model, args, kind=None, writer=None, optimizer=None, scheduler=None, scaler=None):
        super().__init__(model, optimizer, scheduler, scaler, writer, args, kind)
        self.rerank = rerank.creat(args.reranking_type)
        self.rerank_improved = rerank.creat(args.reranking_type_imp)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    @staticmethod
    def _update_test(trial, all_metrics, current_metrics):
        if trial == 1:
            return current_metrics
        else:
            return [all_val + curr_val for all_val, curr_val in zip(all_metrics, current_metrics)]

    def test(self, args, test_mode, mode):
        num_iter = int(args.eval_iter)
        logging.info(f'Test mode: {test_mode} | mode: {mode}')

        if args.dataset != 'RegDB':
            query_loader = creat_test_data(args, mode=mode, kind='query')
            query_feat, query_label, query_cam = self.extract_features_test(query_loader, test_mode[0], boost=args.test_boost)

        all_metrics = [None] * 9  # all_cmc_1, all_cmc_2, all_mAP_1, all_mAP_2, all_mINP_1, all_mINP_2

        all_time = 0
        all_time_rerank_improved = 0
        for trial in range(1, num_iter + 1):
            gall_loader = creat_test_data(args, trial, mode=mode, kind='gallery')
            gall_feat, gall_label, gall_cam = self.extract_features_test(gall_loader, test_mode[1], boost=args.test_boost)

            if args.dataset == 'RegDB':
                query_loader = creat_test_data(args, trial=trial, mode=mode, kind='query')
                query_feat, query_label, query_cam = self.extract_features_test(query_loader, test_mode[0], boost=args.test_boost)

            dist = -torch.matmul(query_feat, gall_feat.T).cpu().numpy()
            cmc_1, mAP_1, mINP_1 = e.use(args.dataset, dist, query_label, gall_label, query_cam, gall_cam)

            start_time = time.time()
            dist = -self.rerank(query_feat, gall_feat, query_cam, gall_cam, k1=args.gnn_k1, k2=args.gnn_k2, la=args.la)
            elapsed_time = time.time() - start_time
            all_time += elapsed_time
            cmc_2, mAP_2, mINP_2 = e.use(args.dataset, dist, query_label, gall_label, query_cam, gall_cam)


            start_time = time.time()
            dist_rerank_improved = -self.rerank_improved(query_feat, gall_feat, query_cam, gall_cam)
            elapsed_time_rerank_improved = time.time() - start_time
            all_time_rerank_improved += elapsed_time_rerank_improved
            cmc_3, mAP_3, mINP_3 = e.use(args.dataset, dist_rerank_improved, query_label, gall_label, query_cam,
                                         gall_cam)



            logging.info(f'Test Trial: {trial}, Elapsed time: {elapsed_time:.3}s')
            logging.info(f"Performance: Rank-1: {cmc_1[0]:.2%} | Rank-5: {cmc_1[4]:.2%} | Rank-10: {cmc_1[9]:.2%}| Rank-20: {cmc_1[19]:.2%}| mAP: {mAP_1:.2%}| mINP: {mINP_1:.2%}")
            logging.info(f"R Performance: Rank-1: {cmc_2[0]:.2%} | Rank-5: {cmc_2[4]:.2%} | Rank-10: {cmc_2[9]:.2%}| Rank-20: {cmc_2[19]:.2%}| mAP: {mAP_2:.2%}| mINP: {mINP_2:.2%}")
            logging.info(f"R_improved Performance: Rank-1: {cmc_3[0]:.2%} | Rank-5: {cmc_3[4]:.2%} | Rank-10: {cmc_3[9]:.2%}| Rank-20: {cmc_3[19]:.2%}| mAP: {mAP_3:.2%}| mINP: {mINP_3:.2%}")
            logging.info("-----------------------Next Trial--------------------")

            current_metrics = [cmc_1, cmc_2, cmc_3, mAP_1, mAP_2, mAP_3, mINP_1, mINP_2, mINP_3]
            all_metrics = self._update_test(trial, all_metrics, current_metrics)

        """"# 3閿斿繆鍎?閻㈢喐鍨?top-10 gallery 鐞涖劍鐗?
                gallery_paths = gall_loader.dataset.img_paths
                query_paths = query_loader.dataset.img_paths
                topk = 10
                rows = []
                for i in range(len(query_feat)):
                    sorted_idx = dist[i].argsort()[:topk]
                    topk_gallery_paths = [os.path.abspath(gallery_paths[j]) for j in sorted_idx]
                    rows.append([os.path.abspath(query_paths[i])] + topk_gallery_paths)

                columns = ['query'] + [f'top{i + 1}' for i in range(topk)]
                df = pd.DataFrame(rows, columns=columns)
                save_path = f'/home/xrs/zsc/CLNS_ee/top10.csv'
                df.to_csv(save_path, index=False)
                logging.info(f'Top-10 gallery paths saved to {save_path}')"""


        all_metrics = [metric / num_iter for metric in all_metrics]

        logging.info("---------------All Performance---------------")
        logging.info(f'Re Ranking time: {float(all_time)*1000:.4f}ms')
        logging.info(f'Improved Re Ranking time: {float(all_time_rerank_improved) * 1000 :.4f}ms')
        logging.info(f'All Average:')
        logging.info(f'Performance: Rank-1: {all_metrics[0][0]:.2%} | Rank-5: {all_metrics[0][4]:.2%} | Rank-10: {all_metrics[0][9]:.2%}| Rank-20: {all_metrics[0][19]:.2%}| mAP: {all_metrics[3]:.2%}| mINP: {all_metrics[6]:.2%}')
        logging.info(f'R Performance: Rank-1: {all_metrics[1][0]:.2%} | Rank-5: {all_metrics[1][4]:.2%} | Rank-10: {all_metrics[1][9]:.2%}| Rank-20: {all_metrics[1][19]:.2%}| mAP: {all_metrics[4]:.2%}| mINP: {all_metrics[7]:.2%}')
        logging.info(f'R_improved Performance: Rank-1: {all_metrics[2][0]:.2%} | Rank-5: {all_metrics[2][4]:.2%} | Rank-10: {all_metrics[2][9]:.2%}| Rank-20: {all_metrics[2][19]:.2%}| mAP: {all_metrics[5]:.2%}| mINP: {all_metrics[8]:.2%}')
        logging.info('End Test')
        logging.info('---------------------------------------------')
