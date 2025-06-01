import torch
import torch.nn as nn


class MaskAgent(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super(MaskAgent, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, out_dim)
        )
        self.softmax = nn.Softmax(dim=-1)

    def sample_matrix(self, probs):
        dist = torch.distributions.Categorical(probs)
        idx = dist.sample().item()
        rates = [0.1*i for i in range(10)]
        return rates[idx] # mask_rate

    def forward(self, X):
        probs = self.softmax(self.mlp(X))
        mask_rate = self.sample_matrix(probs)
        return mask_rate, probs
