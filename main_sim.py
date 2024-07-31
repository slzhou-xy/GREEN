import os
import logging
import argparse
import warnings

from config.config import Config
from Exp.sim_trainer import Exp


def parse_args():
    # dont set default value here! -- it will incorrectly overwrite the values in config.py.
    # config.py is the correct place for default values.
    parser = argparse.ArgumentParser(description='Exp Most Similar Trajectory Search')
    parser.add_argument('--dataset', type=str, help='')

    args = parser.parse_args()
    return dict(filter(lambda kv: kv[1] is not None, vars(args).items()))

def get_log_path(exp_id):
    log_path = f'./log/{Config.dataset}/{exp_id}/similar'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    return log_path

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    Config.update(parse_args())

    exp_id = f'exp_bs128_road{Config.road_trm_layer}_grid{Config.grid_trm_layer}_inter{Config.inter_trm_layer}_epoch30_2e-4_cls_{Config.mask_length}_{Config.mask_ratio}ratio'
    log_path = get_log_path(exp_id)
    pretrain_path = f'./log/{Config.dataset}/{exp_id}/pretrain/best_pretrain_model.pth'

    logging.basicConfig(level=logging.INFO,
                        format="[%(filename)s:%(lineno)s %(funcName)s()] -> %(message)s",
                        handlers=[logging.FileHandler(log_path + '/sim_result.log', mode='w'),
                                  logging.StreamHandler()]
                        )
    
    print("Args in experiment:")
    logging.info('=================================')
    logging.info(Config.to_str())
    logging.info('=================================')

    exp = Exp(log_path, pretrain_path)
    exp.test()
