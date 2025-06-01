import os
import copy
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import sys
import warnings
from data.GenerateDataset import loadnpz
import datetime
import codecs
import time


# Import models
from models.AGCRN.AGCRN import AGCRN
from models.BiaTCGNet.BiaTCGNet import Model as BiaTCGNet
from models.DSFormer.model import DSFormer
from models.InferAgent import InferAgent

"""
执行命令
MTSF models: AGCRN BitGraph DSFormer
Datasets: ETTh1 Weather Elec
`python inferagent_offline.py --dataset ETTh1 --model_name BitGraph --cudaidx 0 --agent_lr 0.0001 --n_expert 5 --k_enhance 3 --model_tag offline00_best --steps 300`
"""


# torch.autograd.set_detect_anomaly(True)
torch.multiprocessing.set_sharing_strategy("file_system")
node_number = 207
parser = argparse.ArgumentParser()
parser.add_argument("--model_tag", type=str)  # model存储名
parser.add_argument("--cudaidx", type=int, default=-1)  # gpu选择
parser.add_argument("--model_name", type=str)  # model选择
parser.add_argument("--modeidx", type=int, default=0)  # agent选择
parser.add_argument("--dataset", default="Elec")  # dataset选择

parser.add_argument("--k_enhance", default=5, type=int)  # 数据增强
parser.add_argument("--steps", default=1000, type=int)  # 训练长度
parser.add_argument("--n_expert", default=5, type=int)  # moe 专家数

parser.add_argument("--seq_len", default=24, type=int)  # 训练序列长度
parser.add_argument("--pred_len", default=24, type=int)  # 预测序列长度
parser.add_argument("--patch_len", default=1, type=int)  # Patch 长度
parser.add_argument("--mask_ratio", type=float, default=0.0)  # 遮盖长度
parser.add_argument("--softloss_ratio", type=float, default=0.5)  # softloss

parser.add_argument("--lr", type=float, default=0.0001)  # 0.001
parser.add_argument("--agent_lr", type=float, default=0.0001)  # 0.001
parser.add_argument("--gamme", type=float, default=0.0)
parser.add_argument("--milestone", type=str, default="")
parser.add_argument("--max_norm", type=float, default=0.0)
parser.add_argument("--epochs", type=int)
parser.add_argument("--batch_size", type=int, default=64)

parser.add_argument("--task", default="prediction", type=str)
parser.add_argument("--kernel_set", type=list, default=[2, 3, 6, 7], help="kernel set")  # 2,3,6,7


args = parser.parse_args()

# gpu 选择
device = "cpu" if args.cudaidx == -1 else "cuda:" + str(args.cudaidx)
print(f"We are using {device}!!!!")

### 数据集配置
if args.dataset == "ETTh1":
    node_number = 7
    args.num_nodes = 7
    args.enc_in = 7
    args.dec_in = 7
    args.c_out = 7
elif args.dataset == "Elec":
    node_number = 321
    args.num_nodes = 321
    args.enc_in = 321
    args.dec_in = 321
    args.c_out = 321
elif args.dataset == "Weather":
    node_number = 21
    args.num_nodes = 21
    args.enc_in = 21
    args.dec_in = 21
    args.c_out = 21
else:
    warnings.warn(f"Wrong dataset: {args.dataset}!!!!!!!")
    sys.exit(1)


### 模型配置
model = None
models = []
if args.model_name == "AGCRN":
    args.epochs = 100
    args.batch_size = 64
    args.lr = 0.003
    args.gamme = 0.3
    args.milestone = "5,20,40,70"
    args.seq_len = 12
    args.pred_len = 12

    model = AGCRN(num_nodes=node_number, input_dim=1, rnn_units=64, output_dim=1, 
                horizon=12, num_layers=2, default_graph=True,
                embed_dim=10, cheb_k=2).to(device)
    
    state_path = f"./model_pth/pretrained_AGCRN_{args.dataset}/sft00_best.pth"
    model.load_state_dict(torch.load(state_path, map_location=device))

    hidden_dim = 64
    print("Model init success!!!!!")
elif args.model_name == "BitGraph":
    args.epochs = 80

    args.seq_len = 24
    args.pred_len = 24

    model = BiaTCGNet(
        True, True, 2, node_number, args.kernel_set, device, predefined_A=None,
        dropout=0.3, subgraph_size=5, node_dim=3, dilation_exponential=1,
        conv_channels=16, residual_channels=16, skip_channels=32, end_channels=64,
        seq_length=args.seq_len, in_dim=1, out_len=args.pred_len,
        out_dim=1, layers=3, propalpha=0.05,
        tanhalpha=3, layer_norm_affline=True).to(device)

    state_path = f"./model_pth/pretrained_BitGraph_{args.dataset}/sft00_best.pth"
    model.load_state_dict(torch.load(state_path, map_location=device))

    hidden_dim = 32
elif args.model_name == "DSFormer":
    args.epochs = 80

    args.seq_len = 96
    args.pred_len = 96

    model = DSFormer(
        Input_len=args.seq_len, out_len=args.pred_len,
        num_id=node_number, num_layer=2, dropout=0.15,
        muti_head=2, num_samp=2, IF_node=True).to(device)
    
    state_path = f"./model_pth/pretrained_DSFormer_{args.dataset}/sft00_best.pth"
    model.load_state_dict(torch.load(state_path, map_location=device))

    hidden_dim = 96
else:
    warnings.warn(f"Wrong model_name: {args.model_name}!!!!!!!")
    sys.exit(1)


inferAgent = InferAgent(
                in_dim=hidden_dim+args.seq_len, out_dim=128, n_expert=args.n_expert,
                pred_len=args.pred_len).to(device)


### train lr
gamme, milestone = args.gamme, []
if gamme > 0:
    milestone = [int(x) for x in args.milestone.split(",")]


### train loss
criteron1 = nn.L1Loss().to(device)
criteron2 = nn.MSELoss().to(device)
criteron3 = nn.SmoothL1Loss().to(device)

def mask_loss(Logits, Target, mode):
    mask = (Target != 0.0).float()
    Logits = Logits * mask
    if mode == 1:
        return criteron1(Logits, Target)
    elif mode == 2:
        return criteron2(Logits, Target)
    elif mode == 3:
        return criteron3(Logits, Target)


### optim
agent_optimizer = optim.Adam(inferAgent.parameters(), lr=args.agent_lr)


### Log dir
if args.seed < 0:
    args.seed = np.random.randint(1e9)
torch.set_num_threads(1)
exp_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
exp_name = f"{args.mask_ratio}_{exp_name}_{args.seed}"
logdir = os.path.join("./logs", args.dataset, args.model_name, exp_name)
# save config for logging
os.makedirs(logdir, exist_ok=True)


### model path
os.makedirs(f"./model_pth/pretrained_{args.model_name}_{args.dataset}", exist_ok=True)
state_path = f"./model_pth/pretrained_{args.model_name}_{args.dataset}/{args.model_tag}.pth"
print("Model store in ", state_path)



### Mask Generation
def generate_mask(X, mask_ratio):
    if mask_ratio > 1 or mask_ratio < 0:
        print("Wrong mask_ratio, ", mask_ratio)
        sys.exit(1)

    shape = X.shape
    num_elements = X.numel()
    num_zeros = int(num_elements * mask_ratio)
    num_ones = num_elements - num_zeros

    mask = torch.cat([torch.zeros(num_zeros), torch.ones(num_ones)], dim=0)
    mask = mask[torch.randperm(mask.numel())].reshape(shape).to(X.device)
    mask = (X != 0.0).float() * mask

    return mask

def gen_mask(pre_mask: torch.Tensor) -> torch.Tensor:
    B, N, _ = pre_mask.shape
    new_mask = pre_mask.detach()
    for i in range(B):
        new_mask[i] = pre_mask[i][torch.randperm(N)]
    return new_mask


### Training
def train(model, train_dataloader, val_dataloader, scaler):
    
    with codecs.open(os.path.join(logdir, "train.log"), "a") as fout:
        fout.write(f"Training !!!!!")

    model.eval()

    best_loss = 9999999.99
    for epoch in range(args.epochs):
        inferAgent.train()

        trn_losses = []
        sum_loss = torch.tensor([0.]).to(device)

        x_arr, y_arr = [], []

        X_input, y_label = None, None
        masks = None

        states = None
        start = time.perf_counter()
        for batch_i, (x, y) in enumerate(train_dataloader):

            x, y = x.to(device), y.to(device)
            if X_input is None:
                x_arr.append(x)
                y_arr.append(y)
                if len(x_arr) < args.seq_len:
                    continue
                X_input = torch.cat(x_arr, dim=0).repeat(10 * args.k_enhance, 1, 1, 1)  # 50,T,N,1
                y_label = torch.cat(y_arr, dim=0).repeat(10 * args.k_enhance, 1, 1, 1)  # 50,T,N,1
            else:
                x = x.repeat(10 * args.k_enhance, 1, 1, 1)  # 10,1,N,1
                y = y.repeat(10 * args.k_enhance, 1, 1, 1)  # 10,1,N,1
                X_input = torch.cat([X_input[:, 1:, :, :], x], dim=1)  # 10,T,N,1
                y_label = torch.cat([y_label[:, 1:, :, :], y], dim=1)  # 10,T,N,1

            ### mask ###
            if masks is not None:
                mask_ts = gen_mask(masks[:, 0, :, :])  # 10,N,1
                masks = torch.cat([masks[:, 1:, :, :], mask_ts.unsqueeze(1)], dim=1)  # 10,T,N,1
            else:
                enhance_masks = []
                for _ in range(args.k_enhance):
                    mask_tmp = []
                    for i in range(10):
                        mask_tmp += [generate_mask(X_input[i], i * 0.1)]
                    masks = torch.stack(mask_tmp, dim=0)  # 10,T,N,1
                    enhance_masks.append(masks)
                masks = torch.cat(enhance_masks, dim=0) # 10*k,T,N,1

            X = X_input * masks  # 10,T,N,1
            _, T, N, _ = X.shape
            with torch.no_grad():
                if args.model_name == "AGCRN":
                    y_pred_base, hidden_state = model(X, None)
                elif args.model_name == "BitGraph":
                    y_pred_base, hidden_state = model(X, mask=masks, k=10)
                elif args.model_name == "GinAR":
                    y_pred_base, hidden_state = model(X)
                elif args.model_name == "DSFormer":
                    y_pred_base, hidden_state = model(X.squeeze(-1))
                elif args.model_name == "TimesNet":
                    y_pred_base, hidden_state = model(X.squeeze(-1), None, None, None).unsqueeze(-1)

            infer_input = torch.cat([X.squeeze(-1).transpose(-2,-1), hidden_state], dim=-1)
            # inferAgent
            y_pred_infer, states, att_experts = inferAgent(infer_input, states)

            y_pred_infer += y_pred_base.detach()
            pred_loss = mask_loss(y_pred_infer, y_label, 1)

            att_experts = att_experts.reshape(-1, 10, N, args.n_expert)
            mrate_att = att_experts.mean(dim=(0,1))
            nodes_att = att_experts.mean(dim=(0,2))
            mean_att = att_experts.mean(dim=(0,1,2))
            moe_loss = 5. - (torch.abs(mrate_att - mean_att).mean() + torch.abs(nodes_att - mean_att).mean())
            trn_loss = pred_loss / pred_loss.detach() + moe_loss / moe_loss.detach() * 0.1


            trn_losses.append(pred_loss.detach())
            sum_loss += trn_loss
            
            if batch_i % args.steps == 0 or batch_i + 1 == len(train_dataloader):
                agent_optimizer.zero_grad()
                sum_loss.backward()
                if args.max_norm > 0:
                    nn.utils.clip_grad_norm_(
                        inferAgent.parameters(), max_norm=args.max_norm
                    )
                agent_optimizer.step()
                
                states = states.detach()
                sum_loss = torch.tensor([0.]).to(device)

        end = time.perf_counter()
        print("Epoch train time:", end - start)

        if gamme > 0:
            if (epoch + 1) in milestone:
                for params in agent_optimizer.param_groups:
                    params["lr"] *= gamme
                    params["weight_decay"] *= gamme

        
        val_loss = evaluate(model, val_dataloader, scaler)
        val_mean_loss = torch.mean(val_loss)
        

        loss_str = f"Epoch {epoch}, train-loss = {sum(trn_losses)/args.steps:.4f}, valid-loss = {val_mean_loss:.4f}\n"
        print(loss_str)
        with codecs.open(os.path.join(logdir, "train.log"), "a") as fout:
            fout.write(loss_str)

        if val_mean_loss < best_loss:
            best_loss = val_mean_loss
            best_model = copy.deepcopy(inferAgent.state_dict())
            torch.save(best_model, state_path)
            with codecs.open(os.path.join(logdir, "train.log"), "a") as fout:
                fout.write(f"Save better model, epoch {epoch}, val_loss is {val_mean_loss:.6f}\n")



### Evaluate
def evaluate(model, val_iter, scaler):
    model.eval()
    inferAgent.eval()

    val_losses = []
    with torch.no_grad():
        x_arr, y_arr = [], []
        masks = None
        states = None
        X_input, y_label = None, None
        for batch_i, (x, y) in enumerate(val_iter):
            x, y = x.to(device), y.to(device)
            if X_input is None:
                x_arr.append(x)
                y_arr.append(y)
                if len(x_arr) < args.seq_len:
                    continue
                X_input = torch.cat(x_arr, dim=0).unsqueeze(0).repeat(10, 1, 1, 1)  # 10,T,N,1
                y_label = torch.cat(y_arr, dim=0).unsqueeze(0).repeat(10, 1, 1, 1)  # 10,T,N,1
            else:
                x = x.unsqueeze(0).repeat(10, 1, 1, 1)  # 10,1,N,1
                y = y.unsqueeze(0).repeat(10, 1, 1, 1)  # 10,1,N,1
                X_input = torch.cat([X_input[:, 1:, :, :], x], dim=1)  # 10,T,N,1
                y_label = torch.cat([y_label[:, 1:, :, :], y], dim=1)  # 10,T,N,1

            ### mask ###
            if masks is not None:
                mask_ts = gen_mask(real_masks[:, 0, :, :])  # 10,N,1
                masks = torch.stack(mask_tmp, dim=0)  # 10,T,N,1
                real_masks = masks.detach()
            else:
                mask_tmp = []
                for i in range(10):
                    mask_tmp.append(generate_mask(X_input[i], i * 0.1))
                masks = torch.stack(mask_tmp, dim=0)  # 10,T,N,1
                real_masks = masks.detach()

            X = X_input * masks  # 10,T,N,1
            _, T, N, _ = X.shape
            with torch.no_grad():
                if args.model_name == "AGCRN":
                    y_pred_base, hidden_state = model(X, None)
                elif args.model_name == "BitGraph":
                    y_pred_base, hidden_state = model(X, mask=masks, k=10)
                elif args.model_name == "GinAR":
                    y_pred_base, hidden_state = model(X)
                elif args.model_name == "DSFormer":
                    y_pred_base, hidden_state = model(X.squeeze(-1))
                elif args.model_name == "TimesNet":
                    y_pred_base, hidden_state = model(X.squeeze(-1), None, None, None).unsqueeze(-1)

            infer_input = torch.cat([X.squeeze(-1).transpose(-2,-1), hidden_state], dim=-1)
            y_pred_infer, states, att_experts = inferAgent(infer_input, states)
            y_pred_infer += y_pred_base
            
            # y_pred_infer[torch.isnan(y_pred_infer)] = 0.0
            y1_pred = scaler.inverse_transform(y_pred_infer)
            y1_label = scaler.inverse_transform(y_label)
            val_losses.append(torch.abs(y1_pred - y1_label).mean(dim=(1, 2, 3)))  # 10

    val_loss = torch.stack(val_losses, dim=1).mean(dim=1)
    return torch.cat([val_loss, val_loss.mean(dim=0, keepdim=True)])  # 11


def run():

    print(f"Dataset = {args.dataset}, Seq_len = {args.seq_len}, Pred_len = {args.pred_len}")
    trn_dataloader, val_dataloader, _, scaler = loadnpz(dataset=args.dataset, batch_size=1, pred_len=args.pred_len, mode="online")
    print("###################### Train Data load success ##########################")
    
    
    ### train
    train(model, trn_dataloader, val_dataloader, scaler)


if __name__ == "__main__":
    run()
