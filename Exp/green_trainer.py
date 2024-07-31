import os
import os.path as osp
import logging

from timm.scheduler import CosineLRScheduler

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from tqdm import tqdm
from dataset.preprocess import Preprocess
from dataset.green_loader import TrajDataLoader
from model.green import GREEN
from config.config import Config


class Exp:
    def __init__(self, log_path):

        self.device = Config.device
        self.epochs = Config.training_epochs
        self.start_epoch = 0
        self.checkpoint_dir = log_path
        self.bad_patience = Config.bad_patience

        self.preprocess = Preprocess()
        self.preprocess_data = self.preprocess.run()
        self.road_graph = self.preprocess_data['road_graph']
        self.grid_image = self.preprocess_data['grid_image']
        self.road_feature = self.preprocess_data['road_feature']
        Config.g_fea_size = self.road_feature.shape[1]
        Config.grid_num = self.preprocess.gs.grid_num
        Config.road_num = self.preprocess.edge.shape[0]
        Config.road_type = self.preprocess.edge.highway_type.nunique()

        self.dataloader = TrajDataLoader()
        self.train_loader = self.dataloader.get_data_loader(self.preprocess_data['train_traj'], is_shuffle=True)
        self.eval_loader = self.dataloader.get_data_loader(self.preprocess_data['eval_traj'])
        self.test_loader = self.dataloader.get_data_loader(self.preprocess_data['test_traj'])

        self.model = self._build_model()
        print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in self.model.parameters())))
        self.optimizer = self._build_optimize()
        self.lr_scheduler = self._build_schduler()

        self.writer = SummaryWriter(self.checkpoint_dir)

        self.train_mlm_loss_list = []
        self.train_cl_loss_list = []
        self.train_loss_list = []
        self.train_mlm_pred_list = []

        self.eval_loss_list = []
        self.eval_mlm_loss_list = []
        self.eval_cl_loss_list = []
        self.eval_mlm_pred_list = []

        self.best_eval_loss = 1e9

        self.model = self.model.to(self.device)
        self.best_save_config = None

    def _build_model(self):
        return GREEN()

    def _build_optimize(self):
        return optim.Adam(params=self.model.parameters(),
                          lr=Config.training_lr
                          )

    def _build_schduler(self):
        return CosineLRScheduler(optimizer=self.optimizer,
                                 t_initial=self.epochs,
                                 warmup_t=Config.warm_up_epoch,
                                 warmup_lr_init=Config.warmup_lr_init,
                                 lr_min=Config.lr_min
                                 )

    def save_model_with_epoch(self, epoch):
        save_config = dict()
        save_config['model'] = self.model.cpu()
        save_config['optimizer'] = self.optimizer.state_dict()
        save_config['lr_scheduler'] = self.lr_scheduler.state_dict()
        save_config['epoch'] = epoch
        save_config['best_eval_loss'] = self.best_eval_loss,
        save_config['train_loss_list'] = self.train_loss_list,
        save_config['train_cl_loss_list'] = self.train_cl_loss_list,
        save_config['train_mlm_loss_list'] = self.train_mlm_loss_list,
        save_config['eval_loss_list'] = self.eval_loss_list,
        save_config['eval_cl_loss_list'] = self.eval_cl_loss_list,
        save_config['eval_mlm_loss_list'] = self.eval_mlm_loss_list,
        save_config['train_mlm_pred_list'] = self.train_mlm_pred_list
        save_config['eval_mlm_pred_list'] = self.eval_mlm_pred_list
        cache_name = osp.join(self.checkpoint_dir, f'pretrain_model_{epoch}.pth')
        torch.save(save_config, cache_name)
        self.model.to(self.device)
        logging.info(f"Saved [{epoch} epoch] model at {cache_name}")
        return save_config

    def load_model_with_epoch(self, epoch):
        cache_name = osp.join(self.checkpoint_dir, f'pretrain_{epoch}.pth')
        assert os.path.exists(cache_name), f'Weights at {cache_name} not found'
        checkpoint = torch.load(cache_name, map_location='cpu')
        self.model = checkpoint['model'].to(self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_eval_loss = checkpoint['best_eval_loss']
        self.train_loss_list = list(checkpoint['train_loss_list'])
        self.train_mlm_loss_list = list(checkpoint['train_mlm_loss_list'])
        self.train_cl_loss_list = list(checkpoint['train_cl_loss_list'])
        self.eval_loss_list = list(checkpoint['eval_loss_list'])
        self.eval_mlm_loss_list = list(checkpoint['eval_mlm_loss_list'])
        self.eval_cl_loss_list = list(checkpoint['eval_cl_loss_list'])
        self.train_mlm_pred_list = list(checkpoint['train_mlm_pred_list'])
        self.eval_mlm_pred_list = list(checkpoint['eval_mlm_pred_list'])
        logging.info(f"Loaded model at {cache_name}")

    def train(self):
        bad_epoch = 0
        best_epoch = 0
        for epoch in range(self.start_epoch, self.epochs):
            train_bar = tqdm(self.train_loader)
            train_loss = []
            train_mlm_loss = []
            train_cl_loss = []
            preds = []
            self.model.train()

            for i, batch_data in enumerate(train_bar):
                road_data, grid_data = batch_data
                for k, v in road_data.items():
                    if k == 'mask_road_index':
                        continue
                    road_data[k] = v.to(self.device)
                for k, v in grid_data.items():
                    grid_data[k] = v.to(self.device)
                road_data['mask_road_index'] = (road_data['mask_road_index'][0].to(self.device), road_data['mask_road_index'][1].to(self.device))
                road_data['g_input_feature'] = torch.FloatTensor(self.road_feature).to(self.device)
                road_data['g_edge_index'] = torch.LongTensor(self.road_graph).to(self.device)
                grid_data['grid_image'] = torch.FloatTensor(self.grid_image).to(self.device)

                cl_loss, mlm_loss, mlm_prediction = self.model(grid_data, road_data)
                loss = cl_loss + mlm_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())
                train_cl_loss.append(cl_loss.item())
                train_mlm_loss.append(mlm_loss.item())
                with torch.no_grad():
                    mlm_prediction = F.log_softmax(mlm_prediction, dim=-1)
                    mlm_label = road_data['road_traj'][road_data['mask_road_index']]
                    mlm_label_prediction = mlm_prediction.argmax(dim=-1)
                    correct_l = mlm_label.eq(mlm_label_prediction).sum().item() / mlm_label.shape[0]
                    preds.append(correct_l)
                train_bar.set_description(f'[GREEN Train Epoch {epoch}/{self.epochs}: loss: {loss:.10f}]')

            with torch.no_grad():
                eval_loss, eval_cl_loss, eval_mlm_loss, eval_preds = self._eval(epoch)

                average_epoch_train_loss = np.array(train_loss).mean()
                average_epoch_train_cl_loss = np.array(train_cl_loss).mean()
                average_epoch_train_mlm_loss = np.array(train_mlm_loss).mean()

                self.train_loss_list.append(average_epoch_train_loss)
                self.train_cl_loss_list.append(average_epoch_train_cl_loss)
                self.train_mlm_loss_list.append(average_epoch_train_mlm_loss)

                average_epoch_eval_loss = np.array(eval_loss).mean()
                average_epoch_eval_cl_loss = np.array(eval_cl_loss).mean()
                average_epoch_eval_mlm_loss = np.array(eval_mlm_loss).mean()

                self.eval_loss_list.append(average_epoch_eval_loss)
                self.eval_cl_loss_list.append(average_epoch_eval_cl_loss)
                self.eval_mlm_loss_list.append(average_epoch_eval_mlm_loss)

                self.writer.add_scalar('loss/train_loss', average_epoch_train_loss, global_step=epoch)
                self.writer.add_scalar('loss/train_mlm_loss', average_epoch_train_mlm_loss, global_step=epoch)
                self.writer.add_scalar('loss/train_cl_loss', average_epoch_train_cl_loss, global_step=epoch)

                self.writer.add_scalar('loss/eval_loss', average_epoch_eval_loss, global_step=epoch)
                self.writer.add_scalar('loss/eval_mlm_loss', average_epoch_eval_mlm_loss, global_step=epoch)
                self.writer.add_scalar('loss/eval_cl_loss', average_epoch_eval_cl_loss, global_step=epoch)

                train_mlm_pred = np.array(preds).mean()
                eval_mlm_pred = np.array(eval_preds).mean()
                self.train_mlm_pred_list.append(train_mlm_pred)
                self.eval_mlm_pred_list.append(eval_mlm_pred)

                self.writer.flush()

                logging.info(
                    f'[Epoch {epoch}/{self.epochs}] [Train loss: {average_epoch_train_loss:.10f}] [Eval loss: {average_epoch_eval_loss:.10f}]'
                )
                logging.info(
                    f'[Epoch {epoch}/{self.epochs}] [Train cl  loss: {average_epoch_train_cl_loss:.10f}] [Eval cl  loss: {average_epoch_eval_cl_loss:.10f}]'
                )
                logging.info(
                    f'[Epoch {epoch}/{self.epochs}] [Train mlm loss: {average_epoch_train_mlm_loss:.10f}] [Eval mlm loss: {average_epoch_eval_mlm_loss:.10f}]'
                )
                logging.info(
                    f'[Epoch {epoch}/{self.epochs}] [Train mlm pred: {train_mlm_pred:.10f}] [Eval mlm pred: {eval_mlm_pred:.10f}]'
                )

                self.lr_scheduler.step(epoch + 1)
                if average_epoch_eval_loss >= self.best_eval_loss:
                    bad_epoch += 1

                if bad_epoch == self.bad_patience:
                    break

                tmp_config = self.save_model_with_epoch(epoch)
                if average_epoch_eval_loss < self.best_eval_loss:
                    bad_epoch = 0
                    self.best_eval_loss = average_epoch_eval_loss
                    self.best_save_config = tmp_config
                    best_epoch = epoch

        logging.info(f'Best model at [epoch {best_epoch}]!')
        cache_name = osp.join(self.checkpoint_dir, 'best_pretrain_model.pth')
        torch.save(self.best_save_config, cache_name)
        logging.info(f"Saved model at {cache_name}")

    def _eval(self, epoch):
        self.model.eval()
        eval_bar = tqdm(self.eval_loader)
        eval_loss = []
        eval_mlm_loss = []
        eval_cl_loss = []
        preds = []
        for batch_data in eval_bar:
            road_data, grid_data = batch_data
            for k, v in road_data.items():
                if k == 'mask_road_index':
                    continue
                road_data[k] = v.to(self.device)
            for k, v in grid_data.items():
                grid_data[k] = v.to(self.device)
            road_data['mask_road_index'] = (road_data['mask_road_index'][0].to(self.device), road_data['mask_road_index'][1].to(self.device))
            road_data['g_input_feature'] = torch.FloatTensor(self.road_feature).to(self.device)
            road_data['g_edge_index'] = torch.LongTensor(self.road_graph).to(self.device)
            grid_data['grid_image'] = torch.FloatTensor(self.grid_image).to(self.device)

            cl_loss, mlm_loss, mlm_prediction = self.model(grid_data, road_data)
            loss = cl_loss + mlm_loss
            eval_loss.append(loss.item())
            eval_cl_loss.append(cl_loss.item())
            eval_mlm_loss.append(mlm_loss.item())
            eval_bar.set_description(f'[GREEN Eval  Epoch {epoch}/{self.epochs}: loss: {loss:.10f}]')

            mlm_prediction = F.log_softmax(mlm_prediction, dim=-1)
            mlm_label = road_data['road_traj'][road_data['mask_road_index']]
            mlm_label_prediction = mlm_prediction.argmax(dim=-1)
            correct_l = mlm_label.eq(mlm_label_prediction).sum().item() / mlm_label.shape[0]

            preds.append(correct_l)

        return eval_loss, eval_cl_loss, eval_mlm_loss, preds
