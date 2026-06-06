# coding=utf-8
import logging
import time

from code.tools.utils.pre_utils import load_model, do_pre, creat_config, setup_dbscan
from code.train.trainer.trainer_stage1 import Trainer_stage1
from code.train.trainer.trainer_stage2 import Trainer_stage2

from code.train.train import do_train
from code.tool import *


def main_worker_1(args, writer):
    start_time = time.monotonic()

    model, optimizer, scheduler, scaler, start_epoch = load_model(args, f'{args.base_dir}{args.resume_path}', args.resume)

    cluster = setup_dbscan(args)

    logging.info("==> Start train stage1")
    trainer_stage1 = Trainer_stage1(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        writer=writer,
        args=args,
        kind='train',
    )
    do_train(
        start_epoch = start_epoch,
        args = args,
        trainer = trainer_stage1,
        cluster = cluster,
        stage=1
    )

    logging.info(f'stage1 running time: {time.monotonic() - start_time}')
    writer.close()




def main_worker_2(args, writer):
    start_time = time.monotonic()

    model, optimizer, scheduler, scaler, start_epoch = load_model(args, f'{args.base_dir}{args.resume_path}', True)

    cluster = setup_dbscan(args)

    logging.info("==> Start train stage2")
    trainer_stage2 = Trainer_stage2(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        writer=writer,
        args=args,
        kind='train',
    )

    do_train(
        start_epoch = 51,
        args = args,
        trainer = trainer_stage2,
        cluster = cluster,
        stage= 2
    )

    logging.info(f'stage2 running time: {time.monotonic() - start_time}')
    writer.close()


if __name__ == '__main__':
    kind = 'sysu'

    txt_path = 'code/txt'

    clear_all_txt_files(txt_path)

    #config_1 = creat_config(kind, 'stage1')
    #writer = do_pre(config_1, 'train')
    #main_worker_1(config_1, writer)

    config_2 = creat_config(kind, 'stage2')
    _writer = do_pre(config_2, 'train')
    main_worker_2(config_2, _writer)