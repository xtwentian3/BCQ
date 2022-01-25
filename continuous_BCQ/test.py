import gym
import torch
import BCQ
import numpy as np
import utils


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


def run_BCQ():
    env = gym.make('Hopper-v1')
    seed = 0
    import gym.spaces.box as Box

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
    policy = torch.load('./results/BCQ_Hopper-v1_0.pt')
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
    env = gym.make('Walker2d-v1')
    print(env.action_space)
    print(env.observation_space)
    _ = env.reset()
    for _ in range(1000):
        env.render()
        obs, reward, done, info = env.step(env.action_space.sample())
        # if done:
        #     break
    env.close()


def plot_reward():
    a = np.load('./results/BCQ_Hopper-v1_0.npy')
    import matplotlib.pyplot as plt
    plt.plot(a)
    plt.show()


def train_BEAR(state_dim, action_dim, max_action, device, args):
    # For saving files
    setting = f"{args.env}_{args.seed}"
    buffer_name = f"{args.buffer_name}_{setting}"
    print(setting)
    print(buffer_name)
    # Initialize policy
    policy = BCQ.BCQ(state_dim, action_dim, max_action, device, args.discount, args.tau, args.lmbda, args.phi)

    # Load buffer
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
    replay_buffer.load(f"./buffers/{buffer_name}")

    evaluations = []
    episode_num = 0
    done = True
    training_iters = 0

    while training_iters < args.max_timesteps:
        pol_vals = policy.train(replay_buffer, iterations=int(args.eval_freq), batch_size=args.batch_size)

        evaluations.append(eval_policy(policy, args.env, args.seed))
        np.save(f"./results/BCQ_{setting}", evaluations)

        training_iters += args.eval_freq
        print(f"Training iterations: {training_iters}")
    # Save final policy
    torch.save(policy, f"./results/BCQ_{setting}.pt")


if __name__ == '__main__':
    # test_env()
    run_BCQ()
