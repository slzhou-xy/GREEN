import torch.nn as nn
from dataset.Model import Date2VecConvert
import torch


class Date2vec(nn.Module):
    def __init__(self, dim, model_path):
        super(Date2vec, self).__init__()
        self.d2v = Date2VecConvert(dim, model_path)

    def forward(self, time_seq):
        one_list = []
        for timestamp in time_seq:
            t = [timestamp.hour, timestamp.minute, timestamp.second, timestamp.year, timestamp.month, timestamp.day]
            x = torch.Tensor(t).float()
            embed = self.d2v(x)
            one_list.append(embed)

        one_list = torch.vstack(one_list).numpy()

        return one_list
