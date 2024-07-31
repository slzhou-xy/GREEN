import torch.nn as nn
from config.config import Config


class CLS(nn.Module):
    def __init__(self, pretraining_model) -> None:
        super().__init__()
        self.pretraining_model = pretraining_model
        self.linear = nn.Linear(Config.out_emb_dim, Config.cls_num)

        if Config.tuning_all is False:
            for name, param in self.pretraining_model.named_parameters():
                param.requires_grad = False

    def forward(self, grid_data, road_data):
        fusion_traj_emb = self.pretraining_model.cls_test(grid_data, road_data)
        pred = self.linear(fusion_traj_emb)
        return pred
