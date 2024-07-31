import os
import logging
import torch
import pandas as pd
import numpy as np

from tqdm import tqdm
from dataset.preprocess import Preprocess
from dataset.green_loader import TrajDataLoader
from model.green import GREEN
from config.config import Config


class Exp:
    def __init__(self, log_path, pretrain_path):

        self.device = Config.device
        self.epochs = Config.training_epochs
        self.start_epoch = 0
        self.pretrain_path = pretrain_path
        self.checkpoint_dir = log_path

        self.preprocess = Preprocess()
        self.preprocess_data = self.preprocess.run(load_traj=False)

        self.road_graph = self.preprocess_data['road_graph']
        self.grid_image = self.preprocess_data['grid_image']
        self.road_feature = self.preprocess_data['road_feature']
        Config.g_fea_size = self.road_feature.shape[1]
        Config.grid_num = self.preprocess.gs.grid_num
        Config.road_num = self.preprocess.edge.shape[0]
        Config.road_type = self.preprocess.edge.highway_type.nunique()

        dataloader = TrajDataLoader()
        
        if not os.path.exists(f'data/{Config.dataset}/database_traj_od.pkl') or \
        not os.path.exists(f'data/{Config.dataset}/query_traj_od.pkl') or \
            not os.path.exists(f'data/{Config.dataset}/query_sim_traj_od.pkl') :
            logging.info('Preprocessing ratio 0.5 similarity database...')
            database = pd.read_csv(f'data/{Config.dataset}/database_traj_od.csv')
            query = pd.read_csv(f'data/{Config.dataset}/query_traj_od.csv')
            query_sim = pd.read_csv(f'data/{Config.dataset}/query_sim_traj_od.csv')
            query = self.preprocess._to_pkl(query)
            query_sim = self.preprocess._to_pkl(query_sim)
            database = self.preprocess._to_pkl(database)
            
            query.to_pickle(f'data/{Config.dataset}/query_traj_od.pkl')
            query_sim.to_pickle(f'data/{Config.dataset}/query_sim_traj_od.pkl')
            database.to_pickle(f'data/{Config.dataset}/database_traj_od.pkl')
        else:
            logging.info('Loading ratio 0.5 similarity database...')
            query = pd.read_pickle(f'data/{Config.dataset}/query_traj_od.pkl')
            query_sim = pd.read_pickle(f'data/{Config.dataset}/query_sim_traj_od.pkl')
            database = pd.read_pickle(f'data/{Config.dataset}/database_traj_od.pkl')

        self.database_loader = dataloader.get_data_loader(database)
        self.query_loader = dataloader.get_data_loader(query)
        self.query_sim_loader = dataloader.get_data_loader(query_sim)

        self.model = self._build_model()
        self.model = self.model.to(self.device)


    def _build_model(self):
        return GREEN()

    def load_checkpoint(self):
        cache_name = self.pretrain_path
        assert os.path.exists(cache_name), f'Weights at {cache_name} not found'
        checkpoint = torch.load(cache_name, map_location='cpu')
        self.model = checkpoint['model'].to(self.device)
        logging.info(f"Loaded model at {cache_name}")
    
    def hit_ratio(self, truth, pred, Ks):
        hit_K = {}
        for K in Ks:
            top_K_pred = pred[:, :K]
            hit = 0
            for i, pred_i in enumerate(top_K_pred):
                if truth[i] in pred_i:
                    hit += 1
            hit_K[K] = hit / pred.shape[0]
        return hit_K

    @torch.no_grad()
    def test(self):
        self.load_checkpoint()
        self.model.eval()
        
        query_fusion_traj_emb_list = []
        for batch_data in tqdm(self.query_loader, desc='Query     Inference'):
            road_data, grid_data = batch_data
            for k, v in road_data.items():
                if k == 'mask_road_index':
                    continue
                road_data[k] = v.to(self.device)
            for k, v in grid_data.items():
                grid_data[k] = v.to(self.device)
            road_data['g_input_feature'] = torch.FloatTensor(self.road_feature).to(self.device)
            road_data['g_edge_index'] = torch.LongTensor(self.road_graph).to(self.device)
            grid_data['grid_image'] = torch.FloatTensor(self.grid_image).to(self.device)

            road_traj_emb, grid_traj_emb, fusion_traj_emb = self.model.test(grid_data, road_data)
            query_fusion_traj_emb_list.append(fusion_traj_emb.cpu().numpy())

        query_fusion_traj_emb = np.vstack(query_fusion_traj_emb_list)
        
        query_fusion_sim_traj_emb_list = []
        for batch_data in tqdm(self.query_sim_loader, desc='Query Sim Inference'):
            road_data, grid_data = batch_data
            for k, v in road_data.items():
                if k == 'mask_road_index':
                    continue
                road_data[k] = v.to(self.device)
            for k, v in grid_data.items():
                grid_data[k] = v.to(self.device)
            road_data['g_input_feature'] = torch.FloatTensor(self.road_feature).to(self.device)
            road_data['g_edge_index'] = torch.LongTensor(self.road_graph).to(self.device)
            grid_data['grid_image'] = torch.FloatTensor(self.grid_image).to(self.device)

            road_traj_emb, grid_traj_emb, fusion_traj_emb = self.model.test(grid_data, road_data)
            query_fusion_sim_traj_emb_list.append(fusion_traj_emb.cpu().numpy())

        query_fusion_sim_traj_emb = np.vstack(query_fusion_sim_traj_emb_list)

        database_fusion_traj_emb_list = []
        for batch_data in tqdm(self.database_loader, desc='Database  Inference'):
            road_data, grid_data = batch_data
            for k, v in road_data.items():
                if k == 'mask_road_index':
                    continue
                road_data[k] = v.to(self.device)
            for k, v in grid_data.items(): 
                grid_data[k] = v.to(self.device)
            road_data['g_input_feature'] = torch.FloatTensor(self.road_feature).to(self.device)
            road_data['g_edge_index'] = torch.LongTensor(self.road_graph).to(self.device)
            grid_data['grid_image'] = torch.FloatTensor(self.grid_image).to(self.device)

            road_traj_emb, grid_traj_emb, fusion_traj_emb = self.model.test(grid_data, road_data)
            database_fusion_traj_emb_list.append(fusion_traj_emb.cpu().numpy())
        database_fusion_traj_emb = np.vstack(database_fusion_traj_emb_list)
        
        for end in [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]:
            q = query_fusion_traj_emb[:end]
            q_sim = query_fusion_sim_traj_emb[:end]
            db = np.concatenate([q_sim, database_fusion_traj_emb])
            dists = q @ db.T
            targets = np.diag(dists)  # [1000]
            result = np.sum(np.greater_equal(dists.T, targets)) / dists.shape[0]
            logging.info(f'{end} fusion {result}')
            
            scores = torch.tensor(dists)
            scores = torch.argsort(scores, descending=True)[:, :10]
            scores = scores.cpu().numpy()
            truth = list(range(end))
            result = self.hit_ratio(truth, scores, [1, 5, 10])
            logging.info(result)