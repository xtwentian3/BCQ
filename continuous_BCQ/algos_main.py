import argparse
import gym
import numpy as np
import os
import torch
import d4rl
import BCQ
import BEAR


def train_2B(state_dim, action_dim, max_action, device, dataset, args):
    print(dataset.keys())
    # For saving files
    setting = f"{args.env}_{args.seed}"
    # buffer_name1 = f"{args.buffer_name}_{args.env}_0"
    print(setting)
    # Initialize policy
    if args.algos == "BCQ":
        policy = BCQ.BCQ(state_dim, action_dim, max_action, device, args.discount, args.tau, args.lmbda, args.phi)
        file_name = f"./results/BCQ_{setting}"
    elif args.algos == "BEAR":
        policy = BEAR.BEAR(state_dim, action_dim, max_action, device, lambda_=0.0, num_samples_match=5, use_ensemble=False)
        file_name = f"./results/BEAR_{setting}"
    training_iters = 0
    evaluations = []
    if not os.path.exists(file_name):
        os.makedirs(file_name)
    # Load buffer
    if args.load_model:
        policy.load(f"{file_name}/final_policy")
        training_iters = 500000
        evaluations = np.load(f"{file_name}/reward_tran.npy")

    # if training_iters < args.max_timesteps * 0.85:  # 只是为了测试
    #     policy.save(f"{file_name}/num_{training_iters}")
    while training_iters < args.max_timesteps:
        pol_vals = policy.train(dataset, iterations=int(args.eval_freq), batch_size=args.batch_size)

        evaluations = np.append(evaluations, eval_policy(policy, args.env, args.seed))
        np.save(f"{file_name}/reward_tran", evaluations)

        training_iters += args.eval_freq
        print(f"Training iterations: {training_iters}")

        if (training_iters > args.max_timesteps * 0.85 and training_iters % 20000 == 0)\
                or (training_iters % 200000 == 0):
            policy.save(f"{file_name}/num_{training_iters}")
        # # Save final policy
        policy.save(f"{file_name}/final_policy")
        np.save(f"{file_name}/training_iters", training_iters)


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="hopper-medium-v2")  # OpenAI gym environment name
    parser.add_argument("--algos", default="BCQ")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=5e3, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6,
                        type=int)  # Max time steps to run environment or train for (this defines buffer size)
    # parser.add_argument("--start_timesteps", default=25e3,
    #                     type=int)  # Time steps initial random policy is used before training behavioral
    parser.add_argument("--rand_action_p", default=0.3,
                        type=float)  # Probability of selecting random action during batch generation
    parser.add_argument("--gaussian_std", default=0.3,
                        type=float)  # Std of Gaussian exploration noise (Set to 0.1 if DDPG trains poorly)
    parser.add_argument("--batch_size", default=100, type=int)  # Mini batch size for networks
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--lmbda", default=0.75)  # Weighting for clipped double Q-learning in BCQ
    parser.add_argument("--phi", default=0.05)  # Max perturbation hyper-parameter for BCQ
    parser.add_argument("--load_model", action="store_true")  # If true, generate buffer
    args = parser.parse_args()

    print("---------------------------------------")
    print(f"Setting: Training {args.algos}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    env = gym.make(args.env)

    env.seed(args.seed)
    # env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = env.get_dataset()
    train_2B(state_dim, action_dim, max_action, device, dataset, args)
