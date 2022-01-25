# Performance of BCQ and BEAR on Mujoco
[BCQ](http://proceedings.mlr.press/v97/fujimoto19a.html) and [BEAR](https://arxiv.org/abs/1906.00949) are two off-policy deep reinforcement learning algorithms.

Method is tested on [MuJoCo](http://www.mujoco.org/) continuous control tasks in [OpenAI gym](https://github.com/openai/gym). 
Networks are trained using [PyTorch 1.4](https://github.com/pytorch/pytorch) and Python 3.6. 

### Overview

If you want to use the dataset, you have to install the [d4rl](https://github.com/rail-berkeley/d4rl) package:
```
git clone https://github.com/rail-berkeley/d4rl.git
cd d4rl
pip install -e .
```

数据集格式（dic，键值）
> 'actions', 'infos/action_log_probs', 'infos/qpos', 'infos/qvel', 'metadata/algorithm', 'metadata/iteration', 'metadata/policy/fc0/bias', 'metadata/policy/fc0/weight', 'metadata/policy/fc1/bias', 'metadata/policy/fc1/weight', 'metadata/policy/last_fc/bias', 'metadata/policy/last_fc/weight', 'metadata/policy/last_fc_log_std/bias', 'metadata/policy/last_fc_log_std/weight', 'metadata/policy/nonlinearity', 'metadata/policy/output_distribution', 'next_observations', 'observations', 'rewards', 'terminals', 'timeouts'

### run
1. 直接运行（算法 BCQ，环境 hopper-medium-v2，seed 0，运行次数 1e6）:
```
python algos_main.py
```
2. Settings can be adjusted with different arguments to algos_main.py.
```
python algos_main.py --algos "BEAR" --env "halfcheetah-medium-v2" --seed 7 --max_timesteps 1000000
```
3. if you want to run the program in the background:
```
nohup python -u algos_main.py --algos "BEAR" --env "halfcheetah-medium-v2" --seed 7 --max_timesteps 1000000 > ./files/BEAR_halfcheetah_medium_7.file 2>&1 &
```
### test
1. 运行后的测试(newtest.py)：
```
python newtest.py
```
ps:可以通过更改参数绘制不同环境/数据集的reward曲线。

2. 原测试文件（test.py),对应于main.py，是通过自己训练的数据集（DDPG）进行训练时的一些测试文件