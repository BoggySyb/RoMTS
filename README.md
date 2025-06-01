# RobustMTSF: A Robust Learning Framework for Multivariate Time‑Series Forecasting under Online–Offline Data Skew

Paper url: Coming soon...

### How to Start

------

##### 1.Install requirements.

```shell
git clone https://github.com/BoggySyb/RobustMTSF.git
pip install -r requirements.txt
```

##### 2.Downloads dataset.

Downloads ETTh1, Weather, Electricity from https://github.com/ChengqingYu/MTS_dataset.

Put them in "/RobustMTSF/data/".

##### 3.Clone MTSF model codes.

Clone from [AGCRN](https://github.com/LeiBAI/AGCRN), [BitGraph](https://github.com/chenxiaodanhit/BiTGraph?tab=readme-ov-file), [DSFormer](https://github.com/GestaltCogTeam/DSformer).

Put model codes in "/RobustMTSF/models/".

You need to change forward function, since our InferAgent need the hiddenstates from MTSF models.

Take BitGraph as an example:

```python
def forward(self,):
  ......
  ......
  hidden_states = F.relu(self.skipE(x) + skip)
  x = F.relu(self.end_conv_1(hidden_states))
  x = self.end_conv_2(x)
  B,T,N,D=x.shape
  x=x.reshape(B,-1,self.output_dim,N)
  x=x.permute(0,1,3,2)

  hidden_states = F.normalize(hidden_states.squeeze(-1).transpose(1,2), dim=-1)
  return x, hidden_states
```

##### 4.Train MTSF model

```shell
python MTSF_train.py --dataset ETTh1 --model_name BitGraph --cudaidx 0 --mask_ratio 0.0  --model_tag 0.0_best
```

##### 5.Stage1 - Train with MaskAgent

```shell
python maskagent_train.py --dataset ETTh1 --model_name BitGraph --mask_ratio 1.0 --cudaidx 0 --modelreward_ratio 1.0 --model_tag sft00_best
```

##### 6.Stage2 - Train InferAgent

```shell
python inferagent_offline.py --dataset ETTh1 --model_name BitGraph --cudaidx 0 --agent_lr 0.0001 --n_expert 5 --k_enhance 3 --model_tag offline00_best --steps 300
```

##### 7.Streaming Test Model Performance

```shell
python online_test.py --dataset ETTh1 --model_name BitGraph --cudaidx 0 --train_tag 0.0 --task_mode online --use_agent True --online_tag offline00
```

If you want to use online fine-tuning method, change task_model from "online" to "online_train".

### Citation

------

```
Citation will be available after acceptance
```

