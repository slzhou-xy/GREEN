import os
import logging
import argparse
import warnings

from config.config import Config
from Exp.multi_cls_trainer import Exp as Exp_multi
from Exp.binary_cls_trainer import Exp as Exp_binary


def parse_args():
    # dont set default value here! -- it will incorrectly overwrite the values in config.py.
    # config.py is the correct place for default values.
    parser = argparse.ArgumentParser(description='Train Classification')
    parser.add_argument('--dataset', type=str, help='')

    args = parser.parse_args()
    return dict(filter(lambda kv: kv[1] is not None, vars(args).items()))

def get_log_path(exp_id):
    if Config.dataset != 'chengdu':
        log_path = f'./log/{Config.dataset}/{exp_id}/multi_cls'
    else:
        log_path = f'./log/{Config.dataset}/{exp_id}/binary_cls'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    return log_path

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    Config.update(parse_args())

    exp_id =  f'exp_bs128_road{Config.road_trm_layer}_grid{Config.grid_trm_layer}_inter{Config.inter_trm_layer}_epoch30_2e-4_cls_{Config.mask_length}_{Config.mask_ratio}ratio'
    log_path = get_log_path(exp_id)
    pretrain_path = f'./log/{Config.dataset}/{exp_id}/pretrain/best_pretrain_model.pth'

    logging.basicConfig(level=logging.INFO,
                        format="[%(filename)s:%(lineno)s %(funcName)s()] -> %(message)s",
                        handlers=[logging.FileHandler(log_path + '/train_log.log', mode='w'),
                                  logging.StreamHandler()]
                        )
    Config.training_lr = 1e-4
    print("Args in experiment:")
    logging.info('=================================')
    logging.info(Config.to_str())
    logging.info('=================================')

    if Config.dataset == 'chengdu':
        exp = Exp_binary(log_path, pretrain_path)
    else:
        exp = Exp_multi(log_path, pretrain_path)
    exp.train()
