import os
import configparser
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

from models.BiaTCGNet.BiaTCGNet import Model as BiaTCGNet
from models.DSFormer.model import DSFormer
from models.AGCRN.AGCRN import AGCRN
from models.MaskAgent import MaskAgent as MaskAgent

"""
执行命令
models: AGCRN BitGraph DSFormer
datasets: ETTh1 Weather Elec
`python maskagent_train.py --dataset ETTh1 --model_name AGCRN --mask_ratio 1.0 --cudaidx 3 --modelreward_ratio 1.0 --model_tag sft10_best`
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
parser.add_argument("--modelreward_ratio", type=float, default=0.5) 

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
if(args.dataset=='ETTh1'):
    node_number= 7
    args.num_nodes= 7
    args.enc_in = 7
    args.dec_in = 7
    args.c_out = 7
    baseline_losses = torch.tensor([1.5638,1.5807,1.6006,1.6325,1.6724,1.7318,1.8177,1.9512,2.1886,2.6352,1.8375]) # AGCRN
    # baseline_losses = torch.tensor([1.437,1.452,1.466,1.477,1.494,1.525,1.569,1.637,1.827,2.165,1.605]) # BitGraph
    # baseline_losses = torch.tensor([1.9625,1.9796,1.9982,2.0228,2.0513,2.0894,2.1381,2.2071,2.3175,2.5095,2.1276]) # DSFormer
elif(args.dataset=='Elec'):
    node_number= 321
    args.num_nodes= 321
    args.enc_in = 321
    args.dec_in = 321
    args.c_out = 321

    baseline_losses = torch.tensor([317.1962,323.7838,332.3411,343.0705,355.9985,373.3489,396.9244,431.7692,488.9304,588.4797,395.1843]) # AGCRN
    # baseline_losses = torch.tensor([374.7239,375.6831,377.9878,382.4309,388.8169,397.1690,407.1245,420.3522,442.5085,489.5441,405.6341]) # DSFormer
    # baseline_losses = torch.tensor([249.9904,251.2011,252.6127,253.7013,255.1357,256.9283,259.5644,264.2984,272.8969,291.8684,260.8198]) # BitGraph
elif args.dataset == "Weather":
    node_number = 21
    args.num_nodes = 21
    args.enc_in = 21
    args.dec_in = 21
    args.c_out = 21

    baseline_losses = torch.tensor([21.7624,22.1388,22.6222,23.2411,24.0233,24.9868,26.3994,28.2753,31.4015,37.3430,26.2194]) # AGCRN
    # baseline_losses = torch.tensor([27.0305,27.1737,27.4102,27.7032,28.0670,28.4199,28.8664,29.4249,30.2952,32.3109,28.6702]) # DSFormer
    # baseline_losses = torch.tensor([21.1062,21.1823,21.2673,21.4456,21.6096,21.8709,22.2580,22.7710,23.8654,27.1146,22.4491]) # BitGraph
else:
    warnings.warn(f"Wrong dataset: {args.dataset}!!!!!!!")
    sys.exit(1)


### 模型配置
model = None
teacher_agent = None
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
    if args.task == "distillation":
        args.epochs = 200
        args.lr = 0.001
        args.gamme = 0.5
        args.milestone = "40,70,90"
    elif args.task == "prediction":
        args.epochs = 100
        args.lr = 0.001
    args.batch_size = 32
    args.seq_len = 24
    args.pred_len = 24
    
    model = BiaTCGNet(True, True, 2, node_number, args.kernel_set,
        device, predefined_A=None,
        dropout=0.3, subgraph_size=5,
        node_dim=3,
        dilation_exponential=1,
        conv_channels=16, residual_channels=16,
        skip_channels=32, end_channels=64,
        seq_length=args.seq_len // args.patch_len, in_dim=1, out_len=args.pred_len // args.patch_len, out_dim=1,
        layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True).to(device)

elif args.model_name == "DSFormer":
    args.epochs = 100
    args.batch_size = 16
    args.lr = 0.0001
    args.gamme = 0.5
    args.milestone = "25,50,75"
    args.seq_len = 96
    args.pred_len = 96

    model = DSFormer(Input_len=args.seq_len, out_len=args.pred_len, num_id=node_number,
                        num_layer=2, dropout=0.15, muti_head=2, num_samp=2, IF_node=True).to(device)

else:
    warnings.warn(f"Wrong model_name: {args.model_name}!!!!!!!")
    sys.exit(1)


# 缺失Mask 定义
maskAgent = MaskAgent(in_dim=42, mid_dim=128, out_dim=10).to(device)


### train lr
gamme, milestone = args.gamme, []
if gamme > 0:
    milestone = [int(x) for x in args.milestone.split(',')]


### train loss
criteron1 = nn.L1Loss().to(device)
criteron2 = nn.MSELoss().to(device)
criteron3 = nn.SmoothL1Loss().to(device)
def mask_loss(Logits, Target, mode):
    mask = (Target != 0.).float()
    Logits = Logits * mask
    if mode == 1: return criteron1(Logits, Target)
    elif mode == 2: return criteron2(Logits, Target)
    elif mode == 3: return criteron3(Logits, Target)


# optim
optimizer = optim.Adam(model.parameters(), lr=args.lr)
agent_optimizer = optim.Adam(maskAgent.parameters(), lr=0.0003)


### Log dir
if args.seed < 0:
    args.seed = np.random.randint(1e9)
torch.set_num_threads(1)
exp_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
exp_name = f"{args.mask_ratio}_{exp_name}_{args.seed}"
logdir = os.path.join('./logs', args.dataset, args.model_name, exp_name)
# save config for logging
os.makedirs(logdir, exist_ok=True)


### model path
os.makedirs('./model_pth/pretrained_'+args.model_name+'_'+args.dataset, exist_ok=True)
state_path = f'./model_pth/pretrained_{args.model_name}_{args.dataset}/{args.model_tag}.pth'
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
    mask = (X != 0.).float() * mask

    return mask


def train(model, train_dataloader, val_dataloader, scaler):

    best_loss = 9999999.99
    pre_modelL = evaluate(model, val_dataloader, scaler)
    maskR_use = torch.tensor([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
    
    for epoch in range(args.epochs):
        start = time.perf_counter()
        trn_losses = []
        model.train()

        # reinforcelearn
        states, actions, rewards = [], [], []
        
        # 不同 maskrate 和平均的loss [11]
        val_losses = batch_evaluate(model, val_dataloader, scaler) # [11]
        rewards.append(val_losses.detach())
        if epoch == 0:
            # 上一次所使用的maskrate one-hot [10]
            pre_maskR = torch.tensor([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]).to(device)
            # 上一个 step 的 val_losses [11]
            pre_valL = val_losses
        for i, (x, y) in enumerate(train_dataloader):
            if i == 300:
                break
            
            ### x shape: B,L1,N,D  ### y shape: B,L2,N,D
            x, y = x.to(device), y.to(device)
            
            ### mask generate
            # 不同maskrate使用频率 [10]
            preUse = (maskR_use / (maskR_use.sum() + 1e-6)).to(device)
            # maskAgent state [42]
            state = torch.cat([preUse, pre_maskR, val_losses / val_losses.max(), val_losses - pre_valL])
            states.append(state)
            # mask_agent 行动
            with torch.no_grad():
                mask_rate, _ = maskAgent(state)
            action = int(mask_rate * 10)
            actions.append(action)
            # 更新 state
            maskR_use[action] += 1.
            pre_maskR = torch.tensor([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]).to(device)
            pre_maskR[action] = 1.
            pre_valL = val_losses
            if i % 10 == 0:
                with codecs.open(os.path.join(logdir, 'train.log'), 'a') as fout:
                    fout.write(f"Epoch {epoch} batch {i}: mask_rate = {mask_rate:.3f}\n")
            mask = generate_mask(x, mask_rate)

            x_input = x * mask


            if args.model_name == "AGCRN":
                x_hat, hidden_state = model(x_input, None)
            elif args.model_name == "BitGraph":
                x_hat, hidden_state = model(x_input, mask=mask, k=10)
            elif args.model_name == "DSFormer":
                x_hat, hidden_state = model(x_input.squeeze(-1))


            # multi loss
            pred_loss = 0.35 * mask_loss(x_hat, y, 1) + 0.65 * mask_loss(x_hat, y, 2)
            trn_loss = pred_loss
            
            # update student model
            trn_losses.append(pred_loss)
            optimizer.zero_grad()
            trn_loss.backward()
            if args.max_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm, error_if_nonfinite=True)
            optimizer.step()
            end = time.perf_counter()
            epoch_time = end - start

            if i % 100 == 0:
                with codecs.open(os.path.join(logdir, 'train.log'), 'a') as fout:
                    fout.write(f"Epoch {epoch} batch {i}: trn-loss = {trn_loss:.3f}\n")
            
            del trn_loss

            # 记录这一step 的 val_losses
            val_losses = batch_evaluate(model, val_dataloader, scaler)
            rewards.append(val_losses.detach())

        ### Evaluate model
        val_loss = evaluate(model, val_dataloader, scaler)

        modelRewardLoss = torch.mean(pre_modelL / val_loss - 1.).detach()

        val_mean_loss = torch.mean(val_loss - baseline_losses)

        pre_modelL = val_loss

        loss_str = f"Epoch {epoch}, train-loss = {sum(trn_losses)/len(train_dataloader):.6f}, valid-loss = {val_mean_loss:.6f}\n"
        print(loss_str)
        with codecs.open(os.path.join(logdir, 'train.log'), 'a') as fout:
            fout.write(loss_str)
            fout.write("Val_loss is " + ','.join(map(lambda x: str(x.item()), val_loss)) + "\n")

        if val_mean_loss < best_loss:
            best_loss = val_mean_loss
            best_model = copy.deepcopy(model.state_dict())
            torch.save(best_model, state_path)
            with codecs.open(os.path.join(logdir, 'train.log'), 'a') as fout:
                fout.write(f"Save better model, epoch {epoch}, val_loss is {val_mean_loss:.6f}\n")

        ######  update maskAgent model  ######
        if args.task == "distillation" and args.mask_ratio == 1.0:
            start = time.perf_counter()
            maskAgent.train()
            meanR = []
            ### 计算reward ###
            # R = 进行这一步的变化和整一个epoch的变化
            for r in rewards[:-1]:
                tmpR = torch.mean(r / rewards[-1] - 1.).detach()
                tmpR += args.modelreward_ratio * modelRewardLoss
                meanR.append(tmpR)
            if epoch % 20 == 0:
                run.log({f"reward_e{epoch}": meanR})
            ### 更新 maskAgent ###
            agentLosses = []
            for pre_state, pre_action, pre_reward in zip(states, actions, meanR):
                _, probs = maskAgent(pre_state)
                agentLosses.append(-torch.log(probs[pre_action]) * pre_reward)
            
            # print mask rate probs
            with codecs.open(os.path.join(logdir, 'train.log'), 'a') as fout:
                fout.write("Mask rate Probs is " + ','.join(map(lambda x: str(x.item()), probs)) + "\n")
            
            agent_optimizer.zero_grad()
            agentLoss = sum(agentLosses)
            agentLoss.backward()
            agent_optimizer.step()
            end = time.perf_counter()
            epoch_time += end - start
            print(epoch_time)

            del agentLoss
            maskAgent.eval()


        if gamme > 0:
            if (epoch + 1) in milestone:
                for params in optimizer.param_groups:
                    params['lr'] *= gamme
                    params["weight_decay"] *= gamme
    
    print(maskR_use)


def batch_evaluate(model, val_iter, scaler):
    
    val_losses = [[] for _ in range(10)]
    
    model.eval()
    with torch.no_grad():
        for i, (x,y) in enumerate(val_iter):
            if i >= 1:
                break
            x, y = x.to(device), y.to(device) # B,T,N,1
            B,T,N,_ = x.shape

            masks = []
            for j in range(10):
                masks.append(generate_mask(x, j*0.1))
            mask = torch.stack(masks, dim=0).reshape(-1,T,N,1) # 10*B,T,N,1
            x = x.unsqueeze(0).repeat(1,10,1,1,1).reshape(-1,T,N,1) # 10*B,T,N,1

            x_input = x * mask

            if args.model_name == "AGCRN":
                x_hat, _ = model(x_input, None)
            elif args.model_name == "BitGraph":
                x_hat, _ = model(x_input, mask=mask, k=10)
            elif args.model_name == "DSFormer":
                x_hat, _ = model(x_input.squeeze(-1))


            x_hat = x_hat.reshape(10,B,T,N,1)
            x_hat = x_hat * (y != 0.).unsqueeze(0).float()
            x_hat = scaler.inverse_transform(x_hat)
            y = scaler.inverse_transform(y)
            
            for j in range(10):
                val_losses[j].append(mask_loss(x_hat[j], y, 1))
            

    mean_losses = [sum(x) / len(x) for x in val_losses] # 10
    abs_losses = [x / mean_losses[0] for x in mean_losses]
    return torch.stack(abs_losses + [sum(abs_losses) / len(abs_losses)], dim=-1).detach().to(device) # 11


def evaluate(model, val_iter, scaler):
    
    val_losses = [[] for _ in range(10)]

    model.eval()
    with torch.no_grad():
        for i, (x,y) in enumerate(val_iter):
            x, y = x.to(device), y.to(device) # B,T,N,1
            B,T,N,_ = x.shape
            
            masks = []
            for j in range(10):
                masks.append(generate_mask(x, j*0.1))
            mask = torch.stack(masks, dim=0).reshape(-1,T,N,1) # 10*B,T,N,1
            x = x.unsqueeze(0).repeat(1,10,1,1,1).reshape(-1,T,N,1) # 10*B,T,N,1

            x_input = x * mask
            
            if args.model_name == "AGCRN":
                x_hat, _ = model(x_input, None)
            elif args.model_name == "BitGraph":
                x_hat, _ = model(x_input, mask=mask, k=10)
            elif args.model_name == "DSFormer":
                x_hat, _ = model(x_input.squeeze(-1))

            x_hat = x_hat.reshape(10,B,T,N,1)
            x_hat = x_hat * (y != 0.).unsqueeze(0).float()
            x_hat = scaler.inverse_transform(x_hat)
            y = scaler.inverse_transform(y)
            
            
            for j in range(10):
                val_losses[j].append(mask_loss(x_hat[j], y, 1))


    mean_losses = [sum(arr) / len(arr) for arr in val_losses]
    val_loss = torch.tensor(mean_losses + [sum(mean_losses) / len(mean_losses)]) # 11
    return val_loss.detach()



def run():

    print(f"Dataset = {args.dataset}, Seq_len = {args.seq_len}, Pred_len = {args.pred_len}")
    trn_dataloader, val_dataloader, _, scaler = loadnpz(dataset=args.dataset, batch_size=args.batch_size, pred_len=args.pred_len, mode="offline")

    ### train
    train(model, trn_dataloader, val_dataloader, scaler)


if __name__ == '__main__':
    run()
