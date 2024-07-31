import torch.nn as nn
from config.config import Config


class TTE(nn.Module):
    def __init__(self, pretraining_model) -> None:
        super().__init__()
        self.pretraining_model = pretraining_model
        self.proj = nn.Sequential(
            nn.Linear(Config.out_emb_dim, Config.out_emb_dim // 2),
            nn.ReLU(),
            nn.Linear(Config.out_emb_dim // 2, 1)
        )

        if Config.tuning_all is False:
            for name, param in self.pretraining_model.named_parameters():
                param.requires_grad = False

    def forward(self, grid_data, road_data):
        fusion_traj_emb = self.pretraining_model.tte_test(grid_data, road_data)
        pred = self.proj(fusion_traj_emb)
        return pred
