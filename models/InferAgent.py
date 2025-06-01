import sys
import math
import torch
import torch.nn as nn
import torch.nn.init as init


class mlp(nn.Module):
    def __init__(self, in_dim, pred_len):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim*2),
            nn.ReLU(),
            nn.Linear(in_dim*2, pred_len),
        )
    
    def forward(self, hidden_state):
        return self.mlp(hidden_state)

class moe(nn.Module):
    def __init__(self, in_dim, n_expert, pred_len):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_dim, in_dim*2),
            nn.ReLU(),
            nn.Linear(in_dim*2, n_expert),
            nn.Softmax(dim=-1)
        )
        experts = []
        for _ in range(n_expert):
            experts.append(
                nn.Sequential(
                    nn.Linear(in_dim, in_dim*2),
                    nn.ReLU(),
                    nn.Linear(in_dim*2, pred_len)
                )
            )
        self.experts = nn.ModuleList(experts)

    def forward(self, hidden_state):
        """
        :input hidden_state: shape(B,N,D)
        :output pred_y: shape(B,T,N)
        """
        att_experts = self.gate(hidden_state) # B,N,n_expert
        # 步骤 1: 获取 Top2 的值和索引（基于原始 logits）
        top2_values, top2_indices = torch.topk(att_experts, k=2, dim=-1)
        
        # 步骤 2: 对 Top2 的值单独应用 Softmax
        top2_softmax = torch.softmax(top2_values, dim=-1)
        
        # 步骤 3: 创建全零张量，并将 Softmax 后的值填充到对应位置
        mask = torch.zeros_like(att_experts)
        mask.scatter_(dim=-1, index=top2_indices, src=top2_softmax)
        att_experts = mask.unsqueeze(-1)
        
        expert_preds = None
        for expert in self.experts:
            pred = expert(hidden_state).unsqueeze(2) # B,N,1,Ts
            if expert_preds is None:
                expert_preds = pred
            else:
                expert_preds = torch.cat([expert_preds, pred], dim=-2)
        
        final_pred = torch.sum(att_experts * expert_preds, dim=-2).transpose(-2,-1).unsqueeze(-1) # B,T,N,1
        return final_pred, att_experts


class InferAgent(nn.Module):
    def __init__(self, in_dim, out_dim, n_expert, pred_len):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        #z_t
        self.U_z = nn.Parameter(torch.Tensor(self.in_dim, out_dim))
        init.xavier_uniform_(self.U_z)
        self.V_z = nn.Parameter(torch.Tensor(out_dim, out_dim))
        init.xavier_uniform_(self.V_z)
        self.b_z = nn.Parameter(torch.Tensor(out_dim))
        init.zeros_(self.b_z)
        
        #r_t
        self.U_r = nn.Parameter(torch.Tensor(self.in_dim, out_dim))
        init.xavier_uniform_(self.U_r)
        self.V_r = nn.Parameter(torch.Tensor(out_dim, out_dim))
        init.xavier_uniform_(self.V_r)
        self.b_r = nn.Parameter(torch.Tensor(out_dim))
        init.zeros_(self.b_r)
        
        #h_t
        self.U_h = nn.Parameter(torch.Tensor(self.in_dim, out_dim))
        init.xavier_uniform_(self.U_h)
        self.V_h = nn.Parameter(torch.Tensor(out_dim, out_dim))
        init.xavier_uniform_(self.V_h)
        self.b_h = nn.Parameter(torch.Tensor(out_dim))
        init.zeros_(self.b_h)
        
        self.decoder = moe(out_dim, n_expert, pred_len)

    
    def forward(self, X, H):
        """
        :input X: shape(B,T,N)
        :input hidden_states: shape(B,N,D)
        :input pre_pred: shape(B,N)
        :input H: shape(B,N,D)
        :output acts: shape(B,N)
        """
        B,N,D = X.size()
        bs = B*N
        x_t = X.reshape(bs, D)
        
        if H is None:
            h_t = torch.zeros(bs, self.out_dim).to(x_t.device)
        else:
            h_t = H.reshape(bs, self.out_dim)

        Z = torch.sigmoid(x_t @ self.U_z + h_t @ self.V_z + self.b_z) # bs,out_dim
        R = torch.sigmoid(x_t @ self.U_r + h_t @ self.V_r + self.b_r) # bs,out_dim
        H_tilda = torch.tanh(x_t @ self.U_h + (R * h_t) @ self.V_h + self.b_h) # bs,out_dim
        h_t = Z * h_t + (1.-Z) * H_tilda # bs,out_dim

        h_t = h_t.reshape(B,N,self.out_dim) # B,N,D
        pred, att_experts = self.decoder(h_t)
        
        return pred, h_t, att_experts