import numpy as np
import matplotlib.pyplot as plt
import os


def finder(pattern, root='.'):
    mats = []
    for root, dirs, files in os.walk(root):
        # print(dirs)
        for dir in dirs :
            if pattern in dir:
                nd = os.path.join(root, dir)
                mats.append(nd)
    return mats

def plot_reward(algo, env_name):
    match = f'{algo}_{env_name}_'
    dirs = finder(match, root=r'./results')
    fig, ax = plt.subplots()
    for a in dirs:
        seed = a.split('_')
        # print(f"{a}\\reward_tran.npy",x[2])
        x = np.linspace(0, 1, 200)  # 创建x的取值范围
        y = np.load(f"{a}/reward_tran.npy")
        ax.plot(x, y, label=f'seed={seed[2]}')  # 作y1 = x 图，并标记此线名为linear
    plt.title(f"{algo}_{env_name}")
    if len(a)>1:
        ax.legend()
    # a = np.load(f"./results/{algo}_{env_name}_{aseed}/reward_tran.npy")
    # plt.plot(a)



if __name__ == '__main__':
    # envs = "hopper-medium-v2"
    # envs = "hopper-expert-v2"
    # envs = "halfcheetah-medium-v2"
    envs = "halfcheetah-expert-v2"
    # envs = "walker2d-medium-v2"
    # envs = "walker2d-expert-v2"
    # plot_reward("BCQ", envs)
    plot_reward("BEAR", envs)
    plt.show()