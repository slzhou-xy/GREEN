import os
import logging
import argparse
import warnings

from config.config import Config
from Exp.green_trainer import Exp


def parse_args():
    # dont set default value here! -- it will incorrectly overwrite the values in config.py.
    # config.py is the correct place for default values.
    parser = argparse.ArgumentParser(description='Train Green')
    parser.add_argument('--dumpfile_uniqueid', type=str, help='see config.py')
    parser.add_argument('--dataset', type=str, help='')

    args = parser.parse_args()
    return dict(filter(lambda kv: kv[1] is not None, vars(args).items()))


def get_log_path(exp_id):
    log_name = f'./log/{Config.dataset}/{exp_id}/pretrain'
    if not os.path.exists(log_name):
        os.makedirs(log_name)
    return log_name


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    Config.update(parse_args())

    exp_id = f'exp_bs128_road{Config.road_trm_layer}_grid{Config.grid_trm_layer}_inter{Config.inter_trm_layer}_epoch30_2e-4_cls_{Config.mask_length}_{Config.mask_ratio}ratio'
    log_path = get_log_path(exp_id)

    logging.basicConfig(level=logging.INFO,
                        format="[%(filename)s:%(lineno)s %(funcName)s()] -> %(message)s",
                        handlers=[logging.FileHandler(log_path + '/train_log.log', mode='w'),
                                  logging.StreamHandler()]
                        )

    print("Args in experiment:")
    logging.info('=================================')
    logging.info(Config.to_str())
    logging.info('=================================')

    exp = Exp(log_path)
    exp.train()
