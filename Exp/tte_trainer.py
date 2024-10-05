import os
import os.path as osp
import logging

from timm.scheduler import CosineLRScheduler

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

from tqdm import tqdm
from dataset.preprocess import Preprocess
from dataset.tte_loader import TrajDataLoader
from model.green import GREEN
from model.tte import TTE
from config.config import Config


class Exp:
    def __init__(self, log_path, pretrain_path):

        self.device = Config.device
        self.epochs = Config.training_epochs
        self.start_epoch = 0
        self.checkpoint_dir = log_path
        self.pretrain_path = pretrain_path

        self.preprocess = Preprocess()
        self.preprocess_data = self.preprocess.run()
        self.road_graph = self.preprocess_data['road_graph']
        self.grid_image = self.preprocess_data['grid_image']
        self.road_feature = self.preprocess_data['road_feature']
        Config.g_fea_size = self.road_feature.shape[1]
        Config.grid_num = self.preprocess.gs.grid_num
        Config.road_num = self.preprocess.edge.shape[0]

        self.dataloader = TrajDataLoader()
        self.train_loader = self.dataloader.get_data_loader(self.preprocess_data['train_traj'], is_shuffle=True)
        self.eval_loader = self.dataloader.get_data_loader(self.preprocess_data['eval_traj'])
        self.test_loader = self.dataloader.get_data_loader(self.preprocess_data['test_traj'])

        self.model = self._build_model()
        self.model = self.model.to(self.device)

        logging.info(f"Total number of paramerters in networks is {sum(x.numel() for x in self.model.parameters())}")
        self.optimizer = self._build_optimize()
        self.lr_scheduler = self._build_schduler()

        self.writer = SummaryWriter(self.checkpoint_dir)

        self.train_loss_list = []
        self.eval_loss_list = []
        self.best_eval_loss = 1e9
        self.bad_patience = Config.bad_patience

    def _build_model(self):
        pretraining_model = GREEN()
        cache_name = self.pretrain_path
        assert os.path.exists(cache_name), f'Weights at {cache_name} not found'
        checkpoint = torch.load(cache_name, map_location='cpu')
        pretraining_model = checkpoint['model']
        tte_model = TTE(pretraining_model)
        return tte_model

    def _build_optimize(self):
        return optim.Adam(params=self.model.parameters(),
                          lr=Config.training_lr,
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
        save_config['eval_loss_list'] = self.eval_loss_list,
        cache_name = osp.join(self.checkpoint_dir, 'tte_best.pth')
        torch.save(save_config, cache_name)
        self.model.to(self.device)
        logging.info(f"Best epoch is {epoch}, save model at {cache_name}")

    def load_model_with_epoch(self):
        cache_name = osp.join(self.checkpoint_dir, 'tte_best.pth')
        assert os.path.exists(cache_name), f'Weights at {cache_name} not found'
        checkpoint = torch.load(cache_name, map_location='cpu')
        self.model = checkpoint['model'].to(self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_eval_loss = checkpoint['best_eval_loss']
        self.train_loss_list = list(checkpoint['train_loss_list'])
        self.eval_loss_list = list(checkpoint['eval_loss_list'])
        logging.info(f"Loaded model at {cache_name}")

    def _evaluation(self, preds, labels):
        error_index = preds < 0
        preds[error_index] = 0
        mae = mean_absolute_error(labels, preds)
        mape = mean_absolute_percentage_error(labels, preds)
        rmse = mean_squared_error(labels, preds) ** 0.5
        return mae, mape, rmse

    def train(self):
        bad_epoch = 0
        best_epoch = 0
        for epoch in range(self.start_epoch, self.epochs):
            train_bar = tqdm(self.train_loader)
            train_loss = []
            pred_list = []
            tte_label_list = []

            self.model.pretraining_model.train()
            self.model.train()
            for i, batch_data in enumerate(train_bar):
                road_data, grid_data, travel_time = batch_data
                for k, v in road_data.items():
                    road_data[k] = v.to(self.device)
                for k, v in grid_data.items():
                    grid_data[k] = v.to(self.device)
                road_data['g_input_feature'] = torch.FloatTensor(self.road_feature).to(self.device)
                road_data['g_edge_index'] = torch.LongTensor(self.road_graph).to(self.device)
                grid_data['grid_image'] = torch.FloatTensor(self.grid_image).to(self.device)
                travel_time = travel_time.to(self.device)

                prediction = self.model(grid_data, road_data)
                loss = F.mse_loss(prediction, travel_time)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())
                pred_list.append(prediction.cpu().detach().numpy())
                tte_label_list.append(travel_time.cpu().numpy())
                train_bar.set_description(f'[Travel Time Estimation Train Epoch {epoch}/{self.epochs}: loss: {loss:.10f}]')

            with torch.no_grad():
                average_epoch_train_loss = np.array(train_loss).mean()
                self.train_loss_list.append(average_epoch_train_loss)
                preds = np.vstack(pred_list)
                tte_labels = np.vstack(tte_label_list)
                train_mae, train_mape, train_rmse = self._evaluation(preds, tte_labels)

                eval_loss, eval_mae, eval_mape, eval_rmse = self._eval(epoch)
                average_epoch_eval_loss = np.array(eval_loss).mean()
                self.eval_loss_list.append(average_epoch_eval_loss)

                self.writer.add_scalar('loss/train_loss', average_epoch_train_loss, global_step=epoch)
                self.writer.add_scalar('loss/eval_loss', average_epoch_eval_loss, global_step=epoch)

                self.writer.flush()

                logging.info(
                    f'[Epoch {epoch}/{self.epochs}] [Train loss: {average_epoch_train_loss:.10f}] [MAE: {train_mae:.10f}] [RMSE: {train_rmse:.10f}] [MAPE: {train_mape:.10f}] '
                )

                logging.info(
                    f'[Epoch {epoch}/{self.epochs}] [Eval  loss: {average_epoch_eval_loss:.10f}] [MAE: {eval_mae:.10f}] [RMSE: {eval_rmse:.10f}] [MAPE: {eval_mape:.10f}]'
                )

                self.lr_scheduler.step(epoch + 1)
                if average_epoch_eval_loss >= self.best_eval_loss:
                    bad_epoch += 1
                if bad_epoch == self.bad_patience:
                    break

                if average_epoch_eval_loss < self.best_eval_loss:
                    bad_epoch = 0
                    self.best_eval_loss = average_epoch_eval_loss
                    self.save_model_with_epoch(epoch)
                    best_epoch = epoch

        logging.info(f'Best model at [epoch {best_epoch}]!')
        self.test()

    def _eval(self, epoch):
        self.model.pretraining_model.eval()
        self.model.eval()
        eval_bar = tqdm(self.eval_loader)
        eval_loss = []
        pred_list = []
        tte_label_list = []
        for batch_data in eval_bar:
            road_data, grid_data, travel_time = batch_data
            for k, v in road_data.items():
                road_data[k] = v.to(self.device)
            for k, v in grid_data.items():
                grid_data[k] = v.to(self.device)
            road_data['g_input_feature'] = torch.FloatTensor(self.road_feature).to(self.device)
            road_data['g_edge_index'] = torch.LongTensor(self.road_graph).to(self.device)
            grid_data['grid_image'] = torch.FloatTensor(self.grid_image).to(self.device)
            travel_time = travel_time.to(self.device)

            prediction = self.model(grid_data, road_data)
            loss = F.mse_loss(prediction, travel_time)
            eval_loss.append(loss.item())
            eval_bar.set_description(f'[Travel Time Estimation Eval  Epoch {epoch}/{self.epochs}: loss: {loss:.10f}]')

            pred_list.append(prediction.cpu().detach().numpy())
            tte_label_list.append(travel_time.cpu().detach().numpy())

        preds = np.vstack(pred_list)
        tte_labels = np.vstack(tte_label_list)
        mae, mape, rmse = self._evaluation(preds, tte_labels)
        return eval_loss, mae, mape, rmse

    @torch.no_grad()
    def test(self):
        self.load_model_with_epoch()
        self.model.pretraining_model.eval()
        self.model.eval()
        test_bar = tqdm(self.test_loader, desc='Travel Time Estimation Test')
        # test_loss = []
        pred_list = []
        tte_label_list = []
        for batch_data in test_bar:
            road_data, grid_data, travel_time = batch_data
            for k, v in road_data.items():
                road_data[k] = v.to(self.device)
            for k, v in grid_data.items():
                grid_data[k] = v.to(self.device)
            road_data['g_input_feature'] = torch.FloatTensor(self.road_feature).to(self.device)
            road_data['g_edge_index'] = torch.LongTensor(self.road_graph).to(self.device)
            grid_data['grid_image'] = torch.FloatTensor(self.grid_image).to(self.device)
            travel_time = travel_time.to(self.device)
            prediction = self.model(grid_data, road_data)

            pred_list.append(prediction.cpu().detach().numpy())
            tte_label_list.append(travel_time.cpu().detach().numpy())

        preds = np.vstack(pred_list)
        tte_labels = np.vstack(tte_label_list)
        mae, mape, rmse = self._evaluation(preds, tte_labels)
        logging.info(
            f'[Travel Time Estimation Test] [MAE: {mae:.10f}] [RMSE: {rmse:.10f}] [MAPE: {mape:.10f}]'
        )
