import random
import numpy
import torch


def set_seed(seed=-1):
    if seed == -1:
        return
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    is_resume = False
    is_cls = True
    seed = 42
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    # dataset
    dataset = 'porto'

    grid_special_tokens = {
        'padding_token': 0,
        'cls_token': 1,
    }

    road_special_tokens = {
        'padding_token': 0,
        'cls_token': 1,
        'mask_token': 2,
    }

    mask_length = 2
    mask_ratio = 0.2

    grid_size = 100
    grid_num = 0

    road_num = 0
    road_type = 0

    batch_size = 128
    training_epochs = 30
    training_lr = 2e-4
    bad_patience = 5

    # cos scheduler
    warm_up_epoch = 10
    warmup_lr_init = 1e-6
    lr_min = 1e-6
    weight_decay = 0.02

    # ===========Common Setting=========
    hidden_emb_dim = 256
    pe_dropout = 0.1
    out_emb_dim = 128

    # ===========Grid Encoder==============
    grid_in_channel = 3
    grid_out_channel =64

    grid_trm_head = 4
    grid_trm_dropout = 0.1
    grid_trm_layer = 2
    grid_ffn_dim = hidden_emb_dim * 4

    # ===========Road Encoder=============
    g_fea_size = 0
    g_heads_per_layer = [4, 4, 4]
    g_dim_per_layer = [hidden_emb_dim, hidden_emb_dim, hidden_emb_dim]
    g_num_layers = 3
    g_dropout = 0.1

    road_trm_head = 4
    road_trm_dropout = 0.1
    road_trm_layer = 4
    road_ffn_dim = hidden_emb_dim * 4

    # ===========Interactor=============
    inter_trm_head = 2
    inter_trm_dropout = 0.1
    inter_trm_layer = 2
    inter_ffn_dim = out_emb_dim * 4

    # downstream tasks
    tuning_all = True
    prediction_length = 5

    @classmethod
    def update(cls, dic: dict):
        for k, v in dic.items():
            if k in cls.__dict__:
                assert type(getattr(Config, k)) == type(v)
            setattr(Config, k, v)
        cls.post_value_updates()

    @classmethod
    def post_value_updates(cls):
        if 'porto' == cls.dataset:
            cls.min_lon = -8.68942
            cls.min_lat = 41.13934
            cls.max_lon = -8.55434
            cls.max_lat = 41.18586
            cls.cls_num = 3
            cls.weight_decay = 0.0001

        elif 'rome' == cls.dataset:
            cls.min_lon = 12.37351
            cls.max_lon = 12.61587
            cls.min_lat = 41.79417
            cls.max_lat = 41.99106
            cls.cls_num = 315
        elif 'chengdu' == cls.dataset:
            cls.min_lon = 103.92994
            cls.max_lon = 104.20535
            cls.min_lat = 30.56799
            cls.max_lat = 30.78813
            cls.cls_num = 2
            cls.weight_decay = 0.0001
        else:
            raise NotImplementedError

        set_seed(cls.seed)

    @classmethod
    def to_str(cls):
        dic = cls.__dict__.copy()
        lst = list(filter(
            lambda p: (not p[0].startswith('__')) and type(p[1]) != classmethod,
            dic.items()
        ))
        return '\n'.join([str(k) + ' = ' + str(v) for k, v in lst])
