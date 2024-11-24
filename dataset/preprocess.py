import os.path as osp

from tqdm import tqdm
from datetime import datetime

import shapely
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from math import atan2, degrees

from config.config import Config
from utils.tools import pload
from dataset.grid_space import GridSpace
from dataset.Date2Vec import Date2vec


def generate_spatial_features(src, cs):
    tgt = []

    for i in range(1, len(src)):
        coord1 = src[i - 1]
        coord2 = src[i]
        coord1[0], coord1[1] = coord1[1], coord1[0]
        coord2[0], coord2[1] = coord2[1], coord2[0]

        distance = geodesic(coord1, coord2).kilometers

        coord1[0], coord1[1] = coord1[1], coord1[0]
        coord2[0], coord2[1] = coord2[1], coord2[0]

        bearing = atan2(coord2[1] - coord1[1], coord2[0] - coord1[0])
        bearing = (degrees(bearing) + 360) % 360 / 360

        x = (coord2[0] - cs.x_min) / (cs.x_max - cs.x_min)
        y = (coord2[1] - cs.y_min) / (cs.y_max - cs.y_min)
        tgt.append([x, y, distance, bearing])

    x = (src[0][0] - cs.x_min) / (cs.x_max - cs.x_min)
    y = (src[0][1] - cs.y_min) / (cs.y_max - cs.y_min)
    tgt.insert(0, [x, y, 0.0, 0.0])
    return tgt


class Preprocess:
    def __init__(self):
        self.path = 'data/{}'.format(Config.dataset)

        self.edge_rn_path = osp.join(self.path, 'rn/edge_rn.csv')
        self.edge_path = osp.join(self.path, 'rn/edge.csv')

        self.edge_rn = pd.read_csv(self.edge_rn_path)
        self.edge = pd.read_csv(self.edge_path)

        x_min, y_min = Config.min_lon, Config.min_lat
        x_max, y_max = Config.max_lon, Config.max_lat

        self.gs = GridSpace(Config.grid_size, Config.grid_size, x_min, y_min, x_max, y_max)

        Config.grid_num = self.gs.grid_num
        self.d2v = Date2vec(Config.hidden_emb_dim, model_path=f'./dataset/d2v_{Config.hidden_emb_dim}d.pt')

    def _data_coverter(self, path):
        traj = pd.read_csv(path)
        traj_id = list(traj.id)
        user = list(traj.user_id)
        road_list = list(traj.cpath)
        geo_list = [shapely.from_wkt(geo).xy for geo in list(traj.geometry)]
        gps_traj_list = [[[x, y] for x, y in zip(X, Y)] for X, Y in geo_list]

        road_len = [len(eval(road)) for road in road_list]
        gps_len = [len(gps_traj) for gps_traj in gps_traj_list]

        if Config.dataset == 'porto':
            m = {'A': 0, 'B': 1, 'C': 2}
            call_type = [m[type] for type in list(traj.call_type)]
            df = pd.DataFrame(
                {
                    'traj_id': traj_id,
                    'user_id': user,
                    'call_type': call_type,
                    'road_len': road_len,
                    'gps_len': gps_len,
                    'road': road_list,
                    'gps': gps_traj_list,
                    'time': list(traj.time),
                    'ptime': list(traj.enhanced_time)
                }
            )
        elif Config.dataset == 'rome':
            df = pd.DataFrame(
                {
                    'traj_id': traj_id,
                    'user_id': user,
                    'road_len': road_len,
                    'gps_len': gps_len,
                    'road': road_list,
                    'gps': gps_traj_list,
                    'time': list(traj.time),
                    'ptime': list(traj.enhanced_time)
                }
            )
        elif Config.dataset == 'chengdu':
            df = pd.DataFrame(
                {
                    'traj_id': traj_id,
                    'user_id': user,
                    'flag': traj.flag,
                    'road_len': road_len,
                    'gps_len': gps_len,
                    'road': road_list,
                    'gps': gps_traj_list,
                    'time': list(traj.time),
                    'ptime': list(traj.enhanced_time)
                }
            )
        else:
            raise NotImplementedError

        return df

    def _generate_road_feature(self):

        def normalization(data):
            return (data - np.min(data)) / (np.max(data) - np.min(data))

        speed = self.edge['speed_kph'].fillna(0)
        speed = normalization(speed.to_numpy())
        
        travel_time = self.edge['travel_time'].fillna(0)
        travel_time = normalization(travel_time.to_numpy())
        
        bearing = self.edge['bearing'].fillna(0)
        bearing = normalization(bearing.to_numpy())
        
        length = self.edge['length'].fillna(0)
        length = normalization(length.to_numpy())
        
        out_degree = self.edge['out_degree'].fillna(0)
        out_degree = normalization(out_degree.to_numpy())
        
        in_degree = self.edge['in_degree'].fillna(0)
        in_degree = normalization(in_degree.to_numpy())
        highway_type = pd.get_dummies(self.edge['highway_type']).to_numpy()
        # add other?
        feature = np.concatenate([length[:, np.newaxis],
                                  speed[:, np.newaxis],
                                  travel_time[:, np.newaxis],
                                  bearing[:, np.newaxis],
                                  out_degree[:, np.newaxis],
                                  in_degree[:, np.newaxis],
                                  highway_type], axis=1)
        np.save(f'data/{Config.dataset}/road_feature.npy', feature)
        return feature

    def _construct_grid_image(self, trajs):
        gps_traj = list(trajs['gps_list'])
        traffic_image = np.zeros((self.gs.x_size, self.gs.y_size))
        for traj in gps_traj:
            for x, y in traj:
                p_x, p_y = self.gs.get_xyidx_by_point(x, y)
                traffic_image[p_x, p_y] += 1
        traffic_image = (traffic_image - np.min(traffic_image)) / (np.max(traffic_image) - np.min(traffic_image))

        x_image = np.zeros((self.gs.x_size, self.gs.y_size))
        y_image = np.zeros((self.gs.x_size, self.gs.y_size))
        for i_x in range(self.gs.x_size):
            for i_y in range(self.gs.y_size):
                c_x, c_y = self.gs.get_center_point(i_x, i_y)
                x_image[i_x, i_y] = c_x
                y_image[i_x, i_y] = c_y
        x_max = self.gs.x_size * self.gs.x_unit + self.gs.x_min
        y_max = self.gs.y_size * self.gs.y_unit + self.gs.y_min
        x_image = (x_image - self.gs.x_min) / (x_max - self.gs.x_min)
        y_image = (y_image - self.gs.y_min) / (y_max - self.gs.y_min)
        image = np.concatenate([x_image[:, :, np.newaxis], y_image[:, :, np.newaxis], traffic_image[:, :, np.newaxis]], axis=-1)
        np.save(f'data/{Config.dataset}/grid_image.npy', image)
        return image

    def _split_traj(self):
        train_traj_path = osp.join(self.path, f'traj_train.pkl')
        eval_traj_path = osp.join(self.path, f'traj_eval.pkl')
        test_traj_path = osp.join(self.path, f'traj_test.pkl')
        if osp.exists(eval_traj_path):
            print('Loading dataset...')
            train_traj = pload(train_traj_path)
            eval_traj = pload(eval_traj_path)
            test_traj = pload(test_traj_path)
            print('Loading finish...')
            return train_traj, eval_traj, test_traj
        print('Trajectory dataset split...')
        traj = pd.read_csv(osp.join(self.path, 'traj.csv'))

        if not osp.exists(osp.join(self.path, 'train_index.npy')):
            index = np.arange(traj.shape[0])
            index = np.random.permutation(index)
            if Config.dataset == 'rome':
                train_index = index[:int(0.8 * index.shape[0])]
                eval_index = index[int(0.8 * index.shape[0]): int(0.9 * index.shape[0])]
                test_index = index[int(0.9 * index.shape[0]):]
            else:
                train_index = index[:int(0.6 * index.shape[0])]
                eval_index = index[int(0.6 * index.shape[0]): int(0.8 * index.shape[0])]
                test_index = index[int(0.8 * index.shape[0]):]
            np.save(osp.join(self.path, 'train_index.npy'), train_index)
            np.save(osp.join(self.path, 'eval_index.npy'), eval_index)
            np.save(osp.join(self.path, 'test_index.npy'), test_index)

        train_index = np.load(osp.join(self.path, 'train_index.npy'))
        eval_index = np.load(osp.join(self.path, 'eval_index.npy'))
        test_index = np.load(osp.join(self.path, 'test_index.npy'))

        train_traj = traj.iloc[train_index]
        train_traj.reset_index(drop=True, inplace=True)

        eval_traj = traj.iloc[eval_index]
        eval_traj.reset_index(drop=True, inplace=True)
        test_traj = traj.iloc[test_index]
        test_traj.reset_index(drop=True, inplace=True)

        train_traj = self._to_pkl(train_traj)
        eval_traj = self._to_pkl(eval_traj)
        test_traj = self._to_pkl(test_traj)

        train_traj.to_pickle(train_traj_path)
        eval_traj.to_pickle(eval_traj_path)
        test_traj.to_pickle(test_traj_path)
        return train_traj, eval_traj, test_traj

    def _to_pkl(self, data):
        user_id = []
        road_list = []
        road_type_list = []
        grid_traj_list = []
        grid_feature_list = []
        gps_list = []
        temporal_list = []
        travel_time = []
        ptemporal_list = []
        temporal_emb_list = []
        flag_list = []

        for i in tqdm(range(data.shape[0])):
            row_data = data.iloc[i]
            road = eval(row_data['road'])
            # cell
            gps = eval(row_data['gps'])
            time_list = eval(row_data['time'])

            grid_traj = [(self.gs.get_gridid_by_point(x, y), [x, y]) for x, y in gps]
            grid_traj = [[v, t] for i, (v, t) in enumerate(zip(grid_traj, time_list)) if i == 0 or v[0] != grid_traj[i-1][0]]
            time_list = [t for _, t in grid_traj]
            grid_traj = [v for v, _ in grid_traj]
            grid_traj, coordinate = zip(*grid_traj)
            grid_feature = generate_spatial_features(coordinate, self.gs)

            road_type_list.append([self.edge.iloc[r]['highway_type'] for r in road])
            grid_feature_list.append(grid_feature)
            user_id.append(row_data['user_id'])
            grid_traj_list.append(list(grid_traj))
            travel_time.append(time_list[-1] - time_list[0])
            road_list.append(road)
            gps_list.append(list(coordinate))
            ptemporal_list.append(eval(row_data['ptime']))
            temporal_list.append(time_list)

            grid_date = [datetime.fromtimestamp(t) for t in time_list]
            temporal_emb_list.append(self.d2v(grid_date))
            if Config.dataset == 'porto':
                flag_list.append(row_data['call_type'])
            elif Config.dataset == 'chengdu':
                flag_list.append(row_data['flag'])
            else:
                raise NotImplementedError

        data_dict = {
            'user_id': user_id,
            'class_type': flag_list,
            'travel_time': travel_time,
            'time': temporal_list,
            'time_emb': temporal_emb_list,
            'ptime': ptemporal_list,
            'road_traj': road_list,
            'road_type': road_type_list,
            'grid_traj': grid_traj_list,
            'grid_feature': grid_feature_list,
            'gps_list': gps_list,
        }

        return pd.DataFrame(data_dict)

    def run(self, load_traj=True):
        road_graph = np.array([self.edge_rn.from_edge_id, self.edge_rn.to_edge_id])
        if not osp.exists(osp.join(self.path, 'road_feature.npy')):
            road_feature = self._generate_road_feature()
        else:
            road_feature = np.load(osp.join(self.path, 'road_feature.npy'))
            
        if load_traj is False:
            train_traj = None
            eval_traj = None
            test_traj = None
        else:
            train_traj, eval_traj, test_traj = self._split_traj()

        if not osp.exists(osp.join(self.path, 'grid_image.npy')):
            grid_image = self._construct_grid_image(train_traj)
        else:
            grid_image = np.load(osp.join(self.path, 'grid_image.npy'))

        return {
            'train_traj': train_traj,
            'eval_traj': eval_traj,
            'test_traj': test_traj,
            'grid_image': grid_image,
            'road_graph': road_graph,
            'road_feature': road_feature
        }
