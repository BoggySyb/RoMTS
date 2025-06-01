import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import sys
import warnings
from data.GenerateDataset import loaddataset, loadonlinedf, loadnpz
import codecs
import sys
import time


from models.InferAgent import InferAgent
from models.AGCRN.AGCRN import AGCRN
from models.BiaTCGNet.BiaTCGNet import Model as BiaTCGNet
from models.DSFormer.model import DSFormer
from models.RNNImputer.rnnimputers import BiRNNImputer as RNNImputer

"""
执行命令
models: AGCRN BitGraph DSFormer RNNImputer
datasets: ETTh1 Weather Elec
`python online_test.py --dataset ETTh1 --model_name BitGraph --cudaidx 3 --train_tag 0.4 --task_mode online --use_agent True --online_tag offline11`
`python online_test.py --dataset Elec --model_name AGCRN --cudaidx 2 --train_tag sft10 --task_mode online_train --use_agent True --online_tag offline11`
"""

print("import success!!!")

# torch.autograd.set_detect_anomaly(True)
torch.multiprocessing.set_sharing_strategy("file_system")
node_number = 207
parser = argparse.ArgumentParser()
parser.add_argument("--task_mode", type=str)  # task_mode
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
print(f"Testing on Dataset {args.dataset}......")


### 模型配置
model, patchModels = None, []
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

    hidden_dim = 64 + args.seq_len
elif args.model_name == "BitGraph":
    args.epochs = 100
    args.gamme = 0.5
    args.milestone = "50,75"

    args.lr = 0.00001
    args.seq_len = 24
    args.pred_len = 24

    model = BiaTCGNet(
        True, True, 2, node_number, args.kernel_set, device, predefined_A=None,
        dropout=0.3, subgraph_size=5, node_dim=3, dilation_exponential=1,
        conv_channels=16, residual_channels=16, skip_channels=32, end_channels=64,
        seq_length=args.seq_len, in_dim=1, out_len=args.pred_len,
        out_dim=1, layers=3, propalpha=0.05,
        tanhalpha=3, layer_norm_affline=True).to(device)

    hidden_dim = 32 + args.seq_len
elif args.model_name == "DSFormer":
    args.epochs = 100
    args.batch_size = 16
    args.lr = 0.00001
    args.gamme = 0.5
    args.milestone = "25,50,75"
    args.seq_len = 96
    args.pred_len = 96

    model = DSFormer(
        Input_len=args.seq_len, out_len=args.pred_len,
        num_id=node_number, num_layer=2, dropout=0.15,
        muti_head=2, num_samp=2, IF_node=True).to(device)

    hidden_dim = 96 + args.seq_len
else:
    warnings.warn(f"Wrong model_name: {args.model_name}!!!!!!!")
    sys.exit(1)


###### Loading Model ######
if args.train_tag != "test":
    state_path = (f"./model_pth/pretrained_{args.model_name}_{args.dataset}/{args.train_tag}_best.pth")
    print("We are loading model ", state_path)
    model.load_state_dict(torch.load(state_path, map_location=device))
    print(f"Model {args.model_name} load success!!!")


if args.task == "imputation":
    impute_model = RNNImputer(d_in=node_number, d_model=64).to(device)
    state_path = (f"./model_pth/pretrained_RNNImputer_{args.dataset}/{args.seq_len}.pth")
    print("We are loading model ", state_path)
    impute_model.load_state_dict(torch.load(state_path, map_location=device))
    print(f"Model {args.model_name} load success!!!")


##### Loading inferAgent #####
inferAgent = None
if args.use_agent:
    inferAgent = InferAgent(
            in_dim=hidden_dim, out_dim=128, 
            n_expert=5, pred_len=args.pred_len).to(device)
    
    state_path = f"./model_pth/pretrained_{args.model_name}_{args.dataset}/{args.online_tag}_best.pth"
    print("We are loading model ", state_path)
    inferAgent.load_state_dict(torch.load(state_path, map_location=device))
    print(f"Model {args.model_name} load success!!!")


### Log dir
torch.set_num_threads(1)
logdir = os.path.join("./logs", args.dataset, args.model_name + "_test")
# save config for logging
os.makedirs(logdir, exist_ok=True)

####### 评测指标 #########
criteron = nn.L1Loss().to(device)


def MAE_np(pred, true, mask_value=0.0):
    if mask_value != None:
        mask = np.where(np.abs(true) > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    return np.mean(np.absolute(true - pred))


def MAPE_np(pred, true, mask_value=0.0):
    if mask_value != None:
        mask = np.where(np.abs(true) > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    return np.mean(np.absolute(np.divide((true - pred), (true)))) * 100


def RMSE_np(pred, true, mask_value=0.0):
    if mask_value != None:
        mask = np.where(np.abs(true) > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    RMSE = np.sqrt(np.mean(np.square(pred - true)))
    return RMSE


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

def generate_binary_tensor(a: int, b: int, X: torch.Tensor) -> torch.Tensor:
    tensor = torch.zeros(X.shape, dtype=torch.float32)  # 创建全零张量
    indices = torch.randperm(b)[:a]               # 生成随机排列并取前a个索引
    tensor[:, indices, :] = 1.0                      # 填充1
    return tensor

def gen_mask(pre_mask: torch.Tensor) -> torch.Tensor:
    # (torch.rand(size=X.shape, device=X.device) > rate).float()
    # random missing rate
    # num = random.randint(0, node_number)
    # return generate_binary_tensor(num, node_number, pre_mask).to(pre_mask.device)
    
    # fixed missing rate
    B, N, D = pre_mask.shape
    new_mask = pre_mask.detach()
    for i in range(B):
        new_mask[i] = pre_mask[i][torch.randperm(N)]
    return new_mask


def online_test(model, test_dataloader, scaler):
    
    MAE_loss = [[] for _ in range(10)]
    RMSE_loss = [[] for _ in range(10)]
    MAPE_loss = [[] for _ in range(10)]



    def collect_loss(labels, preds):
        for i in range(10):
            MAE = MAE_np(preds[i].squeeze(), labels[i].squeeze())
            RMSE = RMSE_np(preds[i].squeeze(), labels[i].squeeze())
            MAPE = MAPE_np(preds[i].squeeze(), labels[i].squeeze())
            MAE_loss[i].append(MAE)
            RMSE_loss[i].append(RMSE)
            MAPE_loss[i].append(MAPE)


    model.eval()
    with torch.no_grad():
        x_arr, y_arr = [], []

        states, masks = None, None

        X_input, y_label = None, None

        time_list = []
        for batch_i, (x, y) in enumerate(test_dataloader):
            x, y = x.to(device), y.to(device)  # 1,N,1
            if X_input is None:
                x_arr.append(x)
                y_arr.append(y)
                if len(x_arr) < args.seq_len:
                    continue
                X_input = torch.cat(x_arr, dim=0).unsqueeze(0).repeat(10, 1, 1, 1)  # 10, T, N, 1
                y_label = torch.cat(y_arr, dim=0).unsqueeze(0).repeat(10, 1, 1, 1)  # 10, T, N, 1
            else:
                x = x.unsqueeze(0).repeat(10, 1, 1, 1)  # 10,1,N,1
                y = y.unsqueeze(0).repeat(10, 1, 1, 1)  # 10,1,N,1
                X_input = torch.cat([X_input[:, 1:, :, :], x], dim=1)  # 10,T,N,1
                y_label = torch.cat([y_label[:, 1:, :, :], y], dim=1)  # 10,T,N,1


            ### mask ###
            if masks is not None:
                mask_ts = gen_mask(masks[:, 0, :, :])  # 10,N,1
                masks = torch.cat([masks[:, 1:, :, :], mask_ts.unsqueeze(1)], dim=1)  # 10,T,N,1
            else:
                mask_tmp = []
                for i in range(10):
                    mask_tmp.append(generate_mask(X_input[i], i*0.1))
                masks = torch.stack(mask_tmp, dim=0)  # 10,T,N,1


            X_input = X_input * masks  # 10,T,N,1
            _, T, N, _ = X_input.shape

            start = time.perf_counter()

            if args.task == "imputation":
                tmp_masks = masks == 1.0
                # x_hat = impute_model(x_input.squeeze(-1), tmp_masks.squeeze(-1)).unsqueeze(-1)
                x_hat, _, _ = impute_model(X_input.squeeze(-1), tmp_masks.squeeze(-1))
                x_hat = x_hat.unsqueeze(-1)
                X_input = (torch.ones(masks.shape).to(device) - masks) * x_hat + X_input
                zero_masks = generate_mask(X_input, 0.0)


            if args.model_name == "AGCRN":
                y_pred, hidden_state = model(X_input, None)
            elif args.model_name == "BitGraph":
                y_pred, hidden_state = model(X_input, mask=masks, k=10)
            elif args.model_name == "DSFormer":
                y_pred, hidden_state = model(X_input.squeeze(-1))

            if args.use_agent:
                infer_input = torch.cat([X_input.squeeze(-1).transpose(-2,-1), hidden_state], dim=-1)
                y_pred_diff, states, att_experts = inferAgent(infer_input, states)
                y_pred += y_pred_diff


            y1_pred = scaler.inverse_transform(y_pred)
            y1_label = scaler.inverse_transform(y_label)

            end = time.perf_counter()
            time_list.append(end-start)

            pred = y1_pred.squeeze().cpu().numpy()
            label = y1_label.squeeze().cpu().numpy()

            collect_loss(label, pred)
    
    print(f"evaluate time cost = {sum(time_list) / len(time_list)}")

    for i in range(10):
        loss_str = "MAE,RMSE,MAPE: %.4f & %.4f & %.4f" % (np.mean(MAE_loss[i]),np.mean(RMSE_loss[i]),np.mean(MAPE_loss[i]))
        print(loss_str)
        with codecs.open(os.path.join(logdir, f"{args.train_tag}_test.log"), "a") as fout:
            fout.write(f"mask_ratio = {i*0.1:.1f}\n" + loss_str + "\n")


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



def online_train(model, test_dataloader, scaler):
    MAE_loss = [[] for _ in range(10)]
    RMSE_loss = [[] for _ in range(10)]
    MAPE_loss = [[] for _ in range(10)]

    model.eval()
    inferAgent.train()

    for i in range(10):
        agent_optimizer = optim.Adam(inferAgent.parameters(), lr=0.00003)
    
        state_path = f"./model_pth/pretrained_{args.model_name}_{args.dataset}/{args.online_tag}_best.pth"
        print("We are loading model ", state_path)
        inferAgent.load_state_dict(torch.load(state_path, map_location=device))
        print(f"Model {args.model_name} load success!!!")

        x_arr, y_arr, pre_inputs = [], [], []

        states, masks = None, None

        X_input, y_label = None, None

        time_list = []

        for batch_i, (x, y) in enumerate(test_dataloader):
            x, y = x.to(device), y.to(device)  # 1,N,1
            if X_input is None:
                x_arr.append(x)
                y_arr.append(y)
                if len(x_arr) < args.seq_len:
                    continue
                X_input = torch.cat(x_arr, dim=0).unsqueeze(0)  # 1, T, N, 1
                y_label = torch.cat(y_arr, dim=0).unsqueeze(0)  # 1, T, N, 1
            else:
                x = x.unsqueeze(0)  # 1,1,N,1
                y = y.unsqueeze(0)  # 1,1,N,1
                X_input = torch.cat([X_input[:, 1:, :, :], x], dim=1)  # 1,T,N,1
                y_label = torch.cat([y_label[:, 1:, :, :], y], dim=1)  # 1,T,N,1


            ### mask ###
            if masks is not None:
                mask_ts = gen_mask(masks[:, 0, :, :])  # 10,N,1
                masks = torch.cat([masks[:, 1:, :, :], mask_ts.unsqueeze(1)], dim=1)  # 10,T,N,1
            else:
                masks = generate_mask(X_input, i * 0.1)


            X_input *= masks  # 10,T,N,1
            _, T, N, _ = X_input.shape

            start = time.perf_counter()

            ### update immidiately ###
            if len(pre_inputs) == args.pred_len:
                pre_input, pre_states, pre_pred = pre_inputs[0]
                
                y_pred_diff, _, _ = inferAgent(pre_input, pre_states)
                online_loss = mask_loss(y_pred_diff+pre_pred, X_input, 1)
                
                agent_optimizer.zero_grad()
                online_loss.backward()
                if args.max_norm > 0:
                    nn.utils.clip_grad_norm_(
                        inferAgent.parameters(), max_norm=args.max_norm
                    )
                agent_optimizer.step()

                pre_inputs = pre_inputs[1:]
            

            with torch.no_grad():
                if args.model_name == "AGCRN":
                    y_pred, hidden_state = model(X_input, None)
                elif args.model_name == "BitGraph":
                    y_pred, hidden_state = model(X_input, mask=masks, k=10)
                elif args.model_name == "DSFormer":
                    y_pred, hidden_state = model(X_input.squeeze(-1))

                infer_input = torch.cat([X_input.squeeze(-1).transpose(-2,-1), hidden_state], dim=-1)
                pre_inputs.append((infer_input, states, y_pred))
                
                y_pred_diff, states, _ = inferAgent(infer_input, states)
                y_pred += y_pred_diff
            end = time.perf_counter()
            time_list.append(end-start)


            y1_pred = scaler.inverse_transform(y_pred)
            y1_label = scaler.inverse_transform(y_label)
            pred = y1_pred.squeeze().cpu().numpy()
            label = y1_label.squeeze().cpu().numpy()

            MAE = MAE_np(pred.squeeze(), label.squeeze())
            RMSE = RMSE_np(pred.squeeze(), label.squeeze())
            MAPE = MAPE_np(pred.squeeze(), label.squeeze())
            MAE_loss[i].append(MAE)
            RMSE_loss[i].append(RMSE)
            MAPE_loss[i].append(MAPE)

        with codecs.open(os.path.join(logdir, f"{args.online_tag}_test.log"), "a") as fout:
            fout.write(f"time cost = {sum(time_list) / len(time_list)}\n")

    for i in range(10):
        loss_str = "MAE,RMSE,MAPE: %.4f & %.4f & %.4f" % (np.mean(MAE_loss[i]),np.mean(RMSE_loss[i]),np.mean(MAPE_loss[i]))
        print(loss_str)
        with codecs.open(os.path.join(logdir, f"{args.online_tag}_test.log"), "a") as fout:
            fout.write(f"mask_ratio = {i*0.1:.1f}\n" + loss_str + "\n")


def run():

    print(f"Config:\nDataset = {args.dataset}, Seq_len = {args.seq_len}, Pred_len = {args.pred_len}")
    # loaddataset(history_len=args.seq_len, pred_len=args.pred_len, dataset=args.dataset, batch_size=args.batch_size)
    # loadonlinedf(history_len=args.seq_len, pred_len=args.pred_len, dataset=args.dataset, val_ratio=0.2, tst_ratio=0.2)
    # sys.exit()
    
    _, _, tst_dataloader, scaler = loadnpz(dataset=args.dataset, batch_size=args.batch_size, pred_len=args.pred_len, mode=args.task_mode)
    #### 测试离线模式 ####
    if args.task_mode == "online_train":
        online_train(model, tst_dataloader, scaler)  ## 并行测试 0.0 - 0.9
    elif args.task_mode == "online":
        online_test(model, tst_dataloader, scaler)  ## 并行测试 0.0 - 0.9
    else:
        print(f"Wrong mode - {args.task_mode} !!!!!!!!!!!!!!!!!!!!!!!!")


if __name__ == "__main__":
    run()
