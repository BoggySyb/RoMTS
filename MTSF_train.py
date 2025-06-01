import os
import copy
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import sys
import time
import warnings
from data.GenerateDataset import loadnpz
import datetime
import codecs
import random

# import models
from models.AGCRN.AGCRN import AGCRN
from models.BiaTCGNet.BiaTCGNet import Model as BiaTCGNet
from models.DSFormer.model import DSFormer
from models.RNNImputer.rnnimputers import BiRNNImputer as RNNImputer

"""
执行命令
models: AGCRN BitGraph DSFormer RNNImputer
datasets: ETTh1 Weather Elec
`python MTSF_train.py --dataset ETTh1 --model_name BitGraph --cudaidx 3 --mask_ratio 0.8  --model_tag 0.8_best`
`python MTSF_train.py --dataset Elec --model_name RNNImputer --cudaidx 0 --mask_ratio 0.5 --model_tag 96 --task imputation`
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
models, patchModels = [], []
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
    
    print("Model init success!!!!!")
elif args.model_name == "BitGraph":
    args.epochs = 60
    args.batch_size = 32
    args.lr = 0.001
    args.seq_len = 24
    args.pred_len = 24

    model = BiaTCGNet(
        True,
        True,
        2,
        node_number,
        args.kernel_set,
        device,
        predefined_A=None,
        dropout=0.3,
        subgraph_size=5,
        node_dim=3,
        dilation_exponential=1,
        conv_channels=16,
        residual_channels=16,
        skip_channels=32,
        end_channels=64,
        seq_length=args.seq_len,
        in_dim=1,
        out_len=args.pred_len,
        out_dim=1,
        layers=3,
        propalpha=0.05,
        tanhalpha=3,
        layer_norm_affline=True,
    ).to(device)
elif args.model_name == "DSFormer":
    args.epochs = 100
    args.batch_size = 16
    args.lr = 0.0001
    args.gamme = 0.5
    args.milestone = "25,50,75"
    args.seq_len = 96
    args.pred_len = 96

    model = DSFormer(
        Input_len=args.seq_len,
        out_len=args.pred_len,
        num_id=node_number,
        num_layer=2,
        dropout=0.15,
        muti_head=2,
        num_samp=2,
        IF_node=True,
    ).to(device)
elif args.model_name == "RNNImputer":
    args.epochs = 60
    args.batch_size = 64
    args.lr = 0.001
    args.seq_len = 96
    args.pred_len = 96 

    model = RNNImputer(d_in=node_number, d_model=64).to(device)
else:
    warnings.warn(f"Wrong model_name: {args.model_name}!!!!!!!")
    sys.exit(1)


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


# optim
optimizer = optim.Adam(model.parameters(), lr=args.lr)

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
os.makedirs("./model_pth/pretrained_" + args.model_name + "_" + args.dataset, exist_ok=True)
state_path = (f"./model_pth/pretrained_{args.model_name}_{args.dataset}/{args.model_tag}.pth")
print("Model store in ", state_path)


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


def train(model, train_dataloader, val_dataloader, scaler):
    
    best_loss = 9999999.99
    for epoch in range(args.epochs):
        start = time.perf_counter()
        trn_losses = []
        model.train()

        for i, (x, y) in enumerate(train_dataloader):
            ### x shape: B,L1,N,D  ### y shape: B,L2,N,D
            x, y = x.to(device), y.to(device)

            if args.mask_ratio == 1.0:
                mask = generate_mask(x, random.random())
            else:
                mask = generate_mask(x, args.mask_ratio)
            x_input = x * mask

            if args.model_name == "AGCRN":
                x_hat, hidden_states = model(x_input, None)
            elif args.model_name == "BitGraph":
                x_hat, hidden_states = model(x_input, mask=mask, k=10)
            elif args.model_name == "DSFormer":
                x_hat, hidden_states = model(x_input.squeeze(-1))
            elif args.model_name == "RNNImputer":
                if args.task == "imputation":
                    mask = mask == 1.0
                    x_hat, _, _ = model(x_input.squeeze(-1), mask.squeeze(-1))
                    x_hat = x_hat.unsqueeze(-1)

            if args.task == "prediction":
                trn_loss = 0.35 * mask_loss(x_hat, y, 1) + 0.65 * mask_loss(x_hat, y, 2)
            elif args.task == "imputation":
                trn_loss = mask_loss(x, x_hat, 2)

            # update student model
            trn_losses.append(trn_loss)
            optimizer.zero_grad()
            trn_loss.backward()
            if args.max_norm > 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.max_norm, error_if_nonfinite=True
                )
            optimizer.step()

            if i % 100 == 0:
                with codecs.open(os.path.join(logdir, "train.log"), "a") as fout:
                    fout.write(f"Epoch {epoch} batch {i}: trn-loss = {trn_loss:.3f}\n")

            del trn_loss

        end = time.perf_counter()
        print(f"train time cost {end - start}")
        ### Evaluate model
        val_mean_loss = evaluate(model, val_dataloader, scaler)


        loss_str = f"Epoch {epoch}, train-loss = {sum(trn_losses)/len(train_dataloader):.6f}, valid-loss = {val_mean_loss:.6f}\n"
        print(loss_str)
        with codecs.open(os.path.join(logdir, "train.log"), "a") as fout:
            fout.write(loss_str)

        if val_mean_loss < best_loss:
            best_loss = val_mean_loss
            best_model = copy.deepcopy(model.state_dict())
            torch.save(best_model, state_path)
            with codecs.open(os.path.join(logdir, "train.log"), "a") as fout:
                fout.write(
                    f"Save better model, epoch {epoch}, val_loss is {val_mean_loss:.6f}\n"
                )

        if gamme > 0:
            if (epoch + 1) in milestone:
                for params in optimizer.param_groups:
                    params["lr"] *= gamme
                    params["weight_decay"] *= gamme



def evaluate(model, val_iter, scaler):
    loss = []
    with torch.no_grad():
        for i, (x, y) in enumerate(val_iter):
            x, y = x.to(device), y.to(device)

            if args.mask_ratio == 1.0:
                mask = generate_mask(x, random.random())
            else:
                mask = generate_mask(x, args.mask_ratio)
            x_input = x * mask

            if args.model_name == "AGCRN":
                x_hat, _ = model(x_input, y)
            elif args.model_name == "BitGraph":
                x_hat, _ = model(x_input, mask=mask, k=10)
            elif args.model_name == "DSFormer":
                x_hat, _ = model(x_input.squeeze(-1))
            elif args.model_name == "RNNImputer":
                if args.task == "imputation":
                    mask = mask == 1.0
                    x_hat, _, _ = model(x_input.squeeze(-1), mask.squeeze(-1))
                    x_hat = x_hat.unsqueeze(-1)

            x_hat = x_hat * (y != 0.0).float()
            x_hat = scaler.inverse_transform(x_hat)
            y = scaler.inverse_transform(y)
            losses = mask_loss(x_hat, y, 1)

            loss.append(losses)

    val_loss = sum(loss) / len(loss)
    return val_loss.detach()


def run():

    print(f"Dataset = {args.dataset}, Seq_len = {args.seq_len}, Pred_len = {args.pred_len}")
    trn_dataloader, val_dataloader, _, scaler = loadnpz(dataset=args.dataset, batch_size=args.batch_size, pred_len=args.pred_len, mode="offline")

    # print("############### Data Load Success ###############")
    #### train ####
    train(model, trn_dataloader, val_dataloader, scaler)


if __name__ == "__main__":
    run()
