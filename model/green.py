import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from config.config import Config
from model.GridTrm import GridTrm
from model.RoadTrm import RoadTrm
from model.RoadGNN import RoadGNN
from model.GridConv import GridConv
from model.InterTrm import InterTrm


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 300):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(torch.arange(0, emb_size, 2) * (-math.log(10000)) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:, :token_embedding.size(1)])


class GREEN(nn.Module):
    def __init__(self):
        super().__init__()
        self.pe = PositionalEncoding(Config.hidden_emb_dim, Config.pe_dropout)

        # Grid encoder
        self.grid_cls_token = nn.Parameter(torch.randn(Config.hidden_emb_dim))
        self.grid_padding_token = nn.Parameter(torch.zeros(Config.hidden_emb_dim), requires_grad=False)
        
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.grid_conv = GridConv(Config.grid_in_channel, Config.grid_out_channel)
        self.grid_enc = GridTrm(
            Config.hidden_emb_dim,
            Config.grid_ffn_dim,
            Config.grid_trm_head,
            Config.grid_trm_layer,
            Config.grid_trm_dropout,
        )
        self.fusion_linear = nn.Linear(Config.hidden_emb_dim + 4, Config.hidden_emb_dim)

        # road encoder
        self.week_emb_layer = nn.Embedding(7 + 1, Config.hidden_emb_dim, padding_idx=0)
        self.minute_emb_layer = nn.Embedding(1440 + 1, Config.hidden_emb_dim, padding_idx=0)
        
        self.road_cls_token = nn.Parameter(torch.randn(Config.hidden_emb_dim))
        self.road_padding_token = nn.Parameter(torch.zeros(Config.hidden_emb_dim), requires_grad=False)
        self.road_mask_token = nn.Parameter(torch.randn(Config.hidden_emb_dim))
        
        self.road_emb_layer = RoadGNN(
            Config.g_fea_size,
            Config.g_dim_per_layer,
            Config.g_heads_per_layer,
            Config.g_num_layers,
            Config.g_dropout
        )
        self.type_emb_layer = nn.Embedding(Config.road_type + 1, Config.hidden_emb_dim, padding_idx=0)
        self.road_enc = RoadTrm(
            Config.hidden_emb_dim,
            Config.road_ffn_dim,
            Config.road_trm_head,
            Config.road_trm_layer,
            Config.road_trm_dropout,
        )

        # InterLayer
        self.inter_layer = InterTrm(
            Config.out_emb_dim,
            Config.inter_ffn_dim,
            Config.inter_trm_head,
            Config.inter_trm_layer,
            Config.inter_trm_dropout,
        )

        # cl_linear
        self.grid_cl_linear = nn.Linear(
            Config.hidden_emb_dim,
            Config.out_emb_dim
        )

        self.road_cl_linear = nn.Linear(
            Config.hidden_emb_dim,
            Config.out_emb_dim
        )
        
        self.temp = nn.Parameter(torch.ones([]) * 0.07)
        
        # mlm_linear
        self.road_mlm_linear = nn.Linear(
            Config.out_emb_dim,
            Config.road_num + len(Config.road_special_tokens)
        )

    def forward(self, grid_data, road_data):
        # road input
        g_input_feature, g_edge_index = road_data['g_input_feature'], road_data['g_edge_index']
        road_traj, mask_road_index = road_data['road_traj'], road_data['mask_road_index']
        road_weeks, road_minutes = road_data['road_weeks'], road_data['road_minutes']
        road_type = road_data['road_type']

        # grid input
        grid_image = grid_data['grid_image']
        grid_traj = grid_data['grid_traj']
        grid_time_emb = grid_data['grid_time_emb']
        grid_feature = grid_data['grid_feature']

        road_weeks_emb = self.week_emb_layer(road_weeks)
        road_minutes_emb = self.minute_emb_layer(road_minutes)

        # road encoder
        g_emb = self.road_emb_layer(g_input_feature, g_edge_index)
        g_emb = torch.vstack([self.road_padding_token, self.road_cls_token, self.road_mask_token, g_emb])
        road_seq_emb = g_emb[road_traj] + road_weeks_emb + road_minutes_emb
        road_seq_emb = self.pe(road_seq_emb)
        road_type_emb = self.pe(self.type_emb_layer(road_type))

        road_padding_mask = road_traj > 0

        road_seq_emb = self.road_enc(
            src=road_seq_emb,
            type_src=road_type_emb,
            mask=road_padding_mask
        )
        
        road_seq_emb = self.road_cl_linear(road_seq_emb)
        road_traj_emb = road_seq_emb[:, 0]

        # grid encoder
        grid_image = self.transform(grid_image.permute(2, 0, 1))
        grid_image_emb = self.grid_conv(grid_image.unsqueeze(0))
        grid_image_emb = torch.vstack([self.grid_padding_token, self.grid_cls_token, grid_image_emb])
        grid_seq_emb = grid_image_emb[grid_traj]
        grid_seq_emb = torch.cat([grid_seq_emb, grid_feature], dim=-1)
        grid_seq_emb = self.fusion_linear(grid_seq_emb) + grid_time_emb 
        grid_seq_emb = self.pe(grid_seq_emb)

        grid_padding_mask = grid_traj > 0
        grid_seq_emb = self.grid_enc(
            src=grid_seq_emb,
            mask=grid_padding_mask,
        )


        grid_seq_emb = self.grid_cl_linear(grid_seq_emb)
        grid_traj_emb = grid_seq_emb[:, 0]

        road_e = F.normalize(road_traj_emb, dim=-1)
        grid_e = F.normalize(grid_traj_emb, dim=-1)
        logits = torch.matmul(grid_e, road_e.T) / self.temp

        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        cl_loss = (loss_i + loss_t)/2

        mask_road_traj = road_traj.clone()
        mask_road_traj[mask_road_index] = Config.road_special_tokens['mask_token']
        mask_road_seq_emb = g_emb[mask_road_traj] + road_weeks_emb + road_minutes_emb
        mask_road_seq_emb = self.pe(mask_road_seq_emb)

        mask_road_seq_emb = self.road_enc(
            src=mask_road_seq_emb,
            type_src=road_type_emb,
            mask=road_padding_mask,
        )

        mask_road_seq_emb = self.road_cl_linear(mask_road_seq_emb)

        fusion_seq_emb = self.inter_layer(
            src=mask_road_seq_emb,
            src_mask=road_padding_mask,
            memory=grid_seq_emb,
            memory_mask=grid_padding_mask
        )

        mlm_prediction = fusion_seq_emb[mask_road_index]
        road_label = road_traj[mask_road_index]
        mlm_prediction = self.road_mlm_linear(mlm_prediction)
        mlm_loss = F.cross_entropy(mlm_prediction, road_label)

        return cl_loss, mlm_loss, mlm_prediction

    def test(self, grid_data, road_data):
        g_input_feature, g_edge_index = road_data['g_input_feature'], road_data['g_edge_index']
        road_traj = road_data['road_traj']
        road_weeks, road_minutes = road_data['road_weeks'], road_data['road_minutes']
        road_type = road_data['road_type']

        # grid input
        grid_image = grid_data['grid_image']
        grid_traj = grid_data['grid_traj']
        grid_time_emb = grid_data['grid_time_emb']
        grid_feature = grid_data['grid_feature']

        road_weeks_emb = self.week_emb_layer(road_weeks)
        road_minutes_emb = self.minute_emb_layer(road_minutes)

        # road encoder
        g_emb = self.road_emb_layer(g_input_feature, g_edge_index)
        g_emb = torch.vstack([self.road_padding_token, self.road_cls_token, self.road_mask_token, g_emb])
        road_seq_emb = g_emb[road_traj] + road_weeks_emb + road_minutes_emb
        road_seq_emb = self.pe(road_seq_emb)
        road_type_emb = self.pe(self.type_emb_layer(road_type))

        road_padding_mask = road_traj > 0
        road_seq_emb = self.road_enc(
            src=road_seq_emb,
            type_src=road_type_emb,
            mask=road_padding_mask
        )

        road_seq_emb = self.road_cl_linear(road_seq_emb)
        road_traj_emb = road_seq_emb[:, 0]

        road_traj_emb = F.normalize(road_traj_emb, dim=-1)

        # grid encoder
        grid_image = self.transform(grid_image.permute(2, 0, 1))
        grid_image_emb = self.grid_conv(grid_image.unsqueeze(0))
        grid_image_emb = torch.vstack([self.grid_padding_token, self.grid_cls_token, grid_image_emb])
        grid_seq_emb = grid_image_emb[grid_traj]
        grid_seq_emb = torch.cat([grid_seq_emb, grid_feature], dim=-1)
        grid_seq_emb = self.fusion_linear(grid_seq_emb) + grid_time_emb
        grid_seq_emb = self.pe(grid_seq_emb)

        grid_padding_mask = grid_traj > 0
        grid_seq_emb = self.grid_enc(
            src=grid_seq_emb,
            mask=grid_padding_mask
        )

        grid_seq_emb = self.grid_cl_linear(grid_seq_emb)
        grid_traj_emb = grid_seq_emb[:, 0]

        grid_traj_emb = F.normalize(grid_traj_emb, dim=-1)

        fusion_seq_emb = self.inter_layer(
            src=road_seq_emb,
            src_mask=road_padding_mask,
            memory=grid_seq_emb,
            memory_mask=grid_padding_mask
        )
        fusion_traj_emb = fusion_seq_emb[:, 0]

        return road_traj_emb, grid_traj_emb, fusion_traj_emb


    def tte_test(self, grid_data, road_data):
        g_input_feature, g_edge_index = road_data['g_input_feature'], road_data['g_edge_index']
        road_traj = road_data['road_traj']
        road_type = road_data['road_type']
        road_weeks, road_minutes = road_data['road_weeks'], road_data['road_minutes']

        # grid input
        grid_image = grid_data['grid_image']
        grid_traj = grid_data['grid_traj']
        grid_time_emb = grid_data['grid_time_emb']
        grid_feature = grid_data['grid_feature']

        road_weeks_emb = self.week_emb_layer(road_weeks)
        road_minutes_emb = self.minute_emb_layer(road_minutes)

        # road encoder
        g_emb = self.road_emb_layer(g_input_feature, g_edge_index)
        g_emb = torch.vstack([self.road_padding_token, self.road_cls_token, self.road_mask_token, g_emb])
        road_seq_emb = g_emb[road_traj] + road_minutes_emb + road_weeks_emb
        road_seq_emb = self.pe(road_seq_emb)
        road_type_emb = self.pe(self.type_emb_layer(road_type))

        road_padding_mask = road_traj > 0
        road_seq_emb = self.road_enc(
            src=road_seq_emb,
            type_src=road_type_emb,
            mask=road_padding_mask
        )

        road_seq_emb = self.road_cl_linear(road_seq_emb)

        # grid encoder
        grid_image = self.transform(grid_image.permute(2, 0, 1))
        grid_image_emb = self.grid_conv(grid_image.unsqueeze(0))
        grid_image_emb = torch.vstack([self.grid_padding_token, self.grid_cls_token, grid_image_emb])
        grid_seq_emb = grid_image_emb[grid_traj]
        grid_seq_emb = torch.cat([grid_seq_emb, grid_feature], dim=-1)
        grid_seq_emb = self.fusion_linear(grid_seq_emb) + grid_time_emb
        grid_seq_emb = self.pe(grid_seq_emb)

        grid_padding_mask = grid_traj > 0
        grid_seq_emb = self.grid_enc(
            src=grid_seq_emb,
            mask=grid_padding_mask
        )

        grid_seq_emb = self.grid_cl_linear(grid_seq_emb)

        fusion_seq_emb = self.inter_layer(
            src=road_seq_emb,
            src_mask=road_padding_mask,
            memory=grid_seq_emb,
            memory_mask=grid_padding_mask
        )
        fusion_traj_emb = fusion_seq_emb[:, 0]

        return fusion_traj_emb

    def cls_test(self, grid_data, road_data):
        g_input_feature, g_edge_index = road_data['g_input_feature'], road_data['g_edge_index']
        road_traj = road_data['road_traj']
        road_type = road_data['road_type']
        road_weeks, road_minutes = road_data['road_weeks'], road_data['road_minutes']

        # grid input
        grid_image = grid_data['grid_image']
        grid_traj = grid_data['grid_traj']
        grid_time_emb = grid_data['grid_time_emb']
        grid_feature = grid_data['grid_feature']

        road_weeks_emb = self.week_emb_layer(road_weeks)
        road_minutes_emb = self.minute_emb_layer(road_minutes)

        # road encoder
        g_emb = self.road_emb_layer(g_input_feature, g_edge_index)
        g_emb = torch.vstack([self.road_padding_token, self.road_cls_token, self.road_mask_token, g_emb])
        road_seq_emb = g_emb[road_traj] + road_minutes_emb + road_weeks_emb
        road_seq_emb = self.pe(road_seq_emb)
        road_type_emb = self.pe(self.type_emb_layer(road_type))

        road_padding_mask = road_traj > 0
        road_seq_emb = self.road_enc(
            src=road_seq_emb,
            type_src=road_type_emb,
            mask=road_padding_mask
        )

        road_seq_emb = self.road_cl_linear(road_seq_emb)

        # grid encoder
        grid_image = self.transform(grid_image.permute(2, 0, 1))
        grid_image_emb = self.grid_conv(grid_image.unsqueeze(0))
        grid_image_emb = torch.vstack([self.grid_padding_token, self.grid_cls_token, grid_image_emb])
        grid_seq_emb = grid_image_emb[grid_traj]
        grid_seq_emb = torch.cat([grid_seq_emb, grid_feature], dim=-1)
        grid_seq_emb = self.fusion_linear(grid_seq_emb) + grid_time_emb
        grid_seq_emb = self.pe(grid_seq_emb)

        grid_padding_mask = grid_traj > 0
        grid_seq_emb = self.grid_enc(
            src=grid_seq_emb,
            mask=grid_padding_mask
        )

        grid_seq_emb = self.grid_cl_linear(grid_seq_emb)

        fusion_seq_emb = self.inter_layer(
            src=road_seq_emb,
            src_mask=road_padding_mask,
            memory=grid_seq_emb,
            memory_mask=grid_padding_mask
        )

        fusion_traj_emb = fusion_seq_emb[:, 0]

        return fusion_traj_emb