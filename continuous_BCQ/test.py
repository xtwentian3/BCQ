import random

import gym
import torch
import BCQ
import numpy as np
import utils
import argparse
import BEAR
import DDPG
import matplotlib.pyplot as plt


# Trains BCQ offline
def train_BCQ(state_dim, action_dim, max_action, device, args):
    # For saving files
    setting = 'Hopper-v1_0'
    buffer_name = 'Robust_Hopper-v1_0'

    # Initialize policy
    policy = BCQ.BCQ(state_dim, action_dim, max_action, device)

    # Load buffer
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
    replay_buffer.load(f"./buffers/{buffer_name}")

    evaluations = []
    episode_num = 0
    done = True
    training_iters = 0

    while training_iters < 10000:
        pol_vals = policy.train(replay_buffer, iterations=int(args.eval_freq), batch_size=args.batch_size)

        evaluations.append(eval_policy(policy, args.env, args.seed))
        np.save(f"./results/BCQ_{setting}", evaluations)

        training_iters += args.eval_freq
        print(f"Training iterations: {training_iters}")


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


def run_BCQ2(training_iters, isImit=True, aseed=0):
    parser = argparse.ArgumentParser()
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--lmbda", default=0.75)  # Weighting for clipped double Q-learning in BCQ
    parser.add_argument("--phi", default=0.05)  # Max perturbation hyper-parameter for BCQ
    args = parser.parse_args()
    if isImit:
        stype = 'Imitat'
    else:
        stype = 'Robust'
    env = gym.make('Hopper-v2')
    # env = gym.make('PongNoFrameskip-v0')
    seed = random.randint(1,99)
    env.seed(seed)
    batch_size = 100
    # env.action_space.seed(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    eval_freq = 500
    # env.action_space.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = BCQ.BCQ(state_dim, action_dim, max_action, device, args.discount, args.tau, args.lmbda, args.phi)
    policy.load(f"./results/{stype}_BCQ_Hopper-v2_{aseed}/num_{training_iters}.0")

    a = np.load(f"./results/{stype}_BCQ_Hopper-v2_{aseed}/reward_tran.npy")
    plt.plot(a)
    plt.show()

    for _ in range(10):
        obs = env.reset()
        for _ in range(10000):
            env.render()
            # action = env.action_space.sample()
            action = policy.select_action(obs)
            obs, reward, done, info = env.step(action)
            print(obs)
            print(action)
            if done:
                break
    env.close()


def run_BCQ():
    env = gym.make('Hopper-v2')
    seed = 0
    # env = gym.make('PongNoFrameskip-v0')
    env.seed(seed)
    batch_size = 100
    # env.action_space.seed(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    eval_freq = 500
    env.seed(0)
    # env.action_space.seed(args.seed)
    torch.manual_seed(0)
    np.random.seed(0)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = torch.load('./results/BCQ1_Hopper-v2_0.pt')
    for _ in range(30):
        obs = env.reset()
        for _ in range(100):
            env.render()
            # action = env.action_space.sample()
            action = policy.select_action(obs)
            obs, reward, done, info = env.step(action)
            print(obs)
            print(action)
            if done:
                break
    env.close()


def run_BEAR2(training_iters, p=0.3, aseed=0):
    parser = argparse.ArgumentParser()
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--lmbda", default=0.75)  # Weighting for clipped double Q-learning in BCQ
    parser.add_argument("--phi", default=0.05)  # Max perturbation hyper-parameter for BCQ
    args = parser.parse_args()

    env = gym.make('Hopper-v2')
    seed = 0

    env.seed(seed)
    batch_size = 100
    # env.action_space.seed(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    eval_freq = 500
    env.seed(0)
    # env.action_space.seed(args.seed)
    torch.manual_seed(0)
    np.random.seed(0)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = BEAR.BEAR(state_dim, action_dim, max_action, device, lambda_=0.0, num_samples_match=5, use_ensemble=False)
    policy.load(f"./results/BEAR_Hopper-v2_{p}_{aseed}/BEAR_Hopper-v2_{aseed}_{training_iters}.0")

    a = np.load(f"./results/BEAR_Hopper-v2_{p}_{aseed}/BEAR1_Hopper-v2_{aseed}.npy")
    plt.plot(a)
    plt.show()

    for _ in range(30):
        obs = env.reset()
        for _ in range(10000):
            env.render()
            # action = env.action_space.sample()
            action = policy.select_action(obs)
            obs, reward, done, info = env.step(action)
            print(obs)
            print(action)
            if done:
                break
    env.close()


def run_BEAR():
    env = gym.make('Hopper-v1')
    seed = 0
    # env = gym.make('PongNoFrameskip-v0')

    env.seed(seed)
    batch_size = 100
    # env.action_space.seed(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    eval_freq = 500
    env.seed(0)
    # env.action_space.seed(args.seed)
    torch.manual_seed(0)
    np.random.seed(0)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = torch.load('./results/BEAR_Hopper-v1_0.pt')
    for _ in range(3000):
        obs = env.reset()
        env.render()
        # action = env.action_space.sample()
        action = policy.select_action(obs)
        obs, reward, done, info = env.step(action)
        print(obs)
        print(action)
        # if done:
        #     break
    env.close()


def test_env():
    env = gym.make('HalfCheetah-v3')  # 'Walker2d-v2'  'HalfCheetah-v2'
    print(env.action_space)
    print(env.observation_space)
    _ = env.reset()
    for _ in range(100000):
        env.render()
        obs, reward, done, info = env.step(env.action_space.sample())
        # if done:
        #     break
    env.close()


def plot_reward(addr_a,addr_b):
    a = np.load(f"./results/Robust_BCQ_Hopper-v2_16/reward_tran.npy")
    plt.plot(a)
    plt.show()
    # a = np.load(addr_a)
    # # b = np.load('./results/buffer_performance_Hopper-v2_0.npy')
    # # a = np.load('./results/BCQ1_Hopper-v1_0.npy')
    # b = np.load(addr_b)
    # import matplotlib.pyplot as plt
    # plt.plot(a)
    # plt.plot(b)
    # plt.show()
    # # print(a)
    # # print(b)


def sparse_try():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_behavioral", action="store_true")  # If true, train behavioral (DDPG)
    parser.add_argument("--generate_buffer", action="store_true")  # If true, generate buffer
    args = parser.parse_args()
    if args.train_behavioral or args.generate_buffer:
        print("I am in.")
    else:
        print("I am out.")


# Handles interactions with the environment, i.e. train behavioral or generate buffer
def genBuffer(buffer_name="Robust", seed=0, rand_action_p=0.3, gaussian_std=0.3):
    max_timesteps = 1e5
    env = "Hopper-v2"
    # For saving files
    setting1 = f"{env}_0"
    setting = f"{env}_{seed}"
    buffer_name = f"{buffer_name}_{setting}_{rand_action_p}_{gaussian_std}"

    env1 = gym.make(env)
    env1.seed(seed)
    # env.action_space.seed(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    state_dim = env1.observation_space.shape[0]
    action_dim = env1.action_space.shape[0]
    max_action = float(env1.action_space.high[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize and load policy
    policy = DDPG.DDPG(state_dim, action_dim, max_action, device)  # , args.discount, args.tau)
    policy.load(f"./models/behavioral_{setting1}")

    # Initialize buffer
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)

    evaluations = []

    state, done = env1.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    # Interact with the environment for max_timesteps
    for t in range(int(max_timesteps)):

        episode_timesteps += 1

        # Select action with noise
        if np.random.uniform(0, 1) < rand_action_p:
            action = env1.action_space.sample()
        else:
            action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, max_action * gaussian_std, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        env1.render()
        next_state, reward, done, _ = env1.step(action)
        done_bool = float(done) if episode_timesteps < env1._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env1.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

    # Save final buffer and performance
    replay_buffer.save(f"./buffers/{buffer_name}")
    evaluations.append(eval_policy(policy, env, seed))
    np.save(f"./results/buffer_performance_{setting}", evaluations)


def datatry():
    import gym
    import d4rl  # Import required to register environments

    # Create the environment
    env = gym.make('hopper-expert-v2')

    # d4rl abides by the OpenAI gym interface
    env.reset()
    env.step(env.action_space.sample())

    # Each task is associated with a dataset
    # dataset contains observations, actions, rewards, terminals, and infos
    dataset = env.get_dataset()
    print(dataset.keys())  # An N x dim_observation Numpy array of observations

    # Alternatively, use d4rl.qlearning_dataset which
    # also adds next_observations.
    # dataset = d4rl.qlearning_dataset(env)
    print(~dataset['terminals'][0:5])


if __name__ == '__main__':
    datatry()
    # genBuffer(buffer_name="Robust1", seed=19, rand_action_p=0.0, gaussian_std=0.0)
    # sparse_try()
    # test_env()

    # #1   isImit=False,
    # run_BCQ2(470000, aseed=0)  # [0.3,11],  300000，400000还凑活
    # # [0.1,15],200000最好

    # 3
    # run_BEAR2(30000, p=0.0, aseed=19)
    #
    # run_BEAR2(5000)

    #2
    # addr_a = './buffers/Robust1_Hopper-v2_19_0.0_reward.npy'
    # addr_b = './buffers/Robust_Hopper-v2_0_reward.npy'
    # a = np.load(addr_a)
    # b = np.load(addr_b)
    # print(a.size)
    # print(b.size)
    # plot_reward(addr_a, addr_b)
