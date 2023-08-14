import torch.nn as nn


# 768 -> hidden per visual
# 512 -> hidden per text
class BackboneAdapter(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.down_proj = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.up_proj = nn.Linear(hidden_dim, input_dim)

        # initialize down proj with Kaiming Normal, up proj with zeros
        nn.init.kaiming_normal_(self.down_proj.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.up_proj.weight)

    def forward(self, x):
        x = self.down_proj(x)
        x = self.relu(x)
        x = self.up_proj(x)
        return x


class PreFusionAdapter(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class PostFusionAdapter(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x