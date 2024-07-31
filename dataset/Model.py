import torch
from torch import nn


class Date2VecConvert:
    def __init__(self, dim, model_path):
        self.model = Date2Vec(k=dim)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model = self.model.eval()

    def __call__(self, x):
        with torch.no_grad():
            return self.model.encode(torch.Tensor(x).unsqueeze(0)).squeeze(0)# .cpu()


class Date2Vec(nn.Module):
    def __init__(self, k=32, act="sin"):
        super(Date2Vec, self).__init__()

        if k % 2 == 0:
            k1 = k // 2
            k2 = k // 2
        else:
            k1 = k // 2
            k2 = k // 2 + 1

        self.fc1 = nn.Linear(6, k1)

        self.fc2 = nn.Linear(6, k2)
        self.d2 = nn.Dropout(0.3)

        if act == 'sin':
            self.activation = torch.sin
        else:
            self.activation = torch.cos

        self.fc3 = nn.Linear(k, k // 2)
        self.d3 = nn.Dropout(0.3)

        self.fc4 = nn.Linear(k // 2, 6)

        self.fc5 = torch.nn.Linear(6, 6)

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.d2(self.activation(self.fc2(x)))
        out = torch.cat([out1, out2], 1)
        out = self.d3(self.fc3(out))
        out = self.fc4(out)
        out = self.fc5(out)
        return out

    def encode(self, x):
        out1 = self.fc1(x)
        out2 = self.activation(self.fc2(x))
        out = torch.cat([out1, out2], 1)
        return out
