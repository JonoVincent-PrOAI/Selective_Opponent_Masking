import os
import torch
import gymnasium as gym
import argparse
import sys
import wandb
import ray
import functools

if os.path.abspath("../") not in sys.path:
    sys.path.append(os.path.abspath("../"))
if os.path.abspath("./") not in sys.path:
    sys.path.append(os.path.abspath("./"))

from pettingzoo.atari import surround_v2
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.algorithm import Algorithm


from utils.surround_v2_wrapper import Surround_v2_Wrapper
from utils.self_play_callback import SelfPlayCallback


parser = argparse.ArgumentParser(description="Pretraining for model in the surroun_v2 env. Trains a model in wrapped surround_v5")
parser.add_argument("-ldir", "--loadDirectory", help="Model checkpoint directory.")
parser.add_argument("-sdir", "--saveDirectory", help="Directory checkpoints are saved to.")
parser.add_argument("-chkpt", "--checkpoint", help="After how many episodes should a checkpoint be saved.")
parser.add_argument("-sz", "--batchSize", help="Size of training batches.")
parser.add_argument("-rl", "--rolloutLength", help="Lenght of rollout fragments.")
parser.add_argument("-ner", "--numEnvRunners", help="Number of env ruuners.")
parser.add_argument("-ngpu", "--numGPU", help="Number of GPUs available for training.")
parser.add_argument("-ncpu", "--numCPU", help="Number of CPUs available for training.")
parser.add_argument("-ni", "--numIter", help="Number of Training Iterations.")
parser.add_argument("-v", "--verbose", help="True/Falser whether outputs should be given.")
parser.add_argument("-wnb", "--WandBKey", help="API key W and B logger.")
args = parser.parse_args()

if args.loadDirectory:
    load_dir = args.loadDirectory
else:
    print('Error Load directory not provided!')
if args.saveDirectory:
    save_dir = args.saveDirectory
else:
    save_dir = "./ray_results/PPO_surround_v2/"
if args.checkpoint:
    checkpoint = args.checkpoint
else:
    checkpoint = 10
if args.batchSize:
    batch_size = int(args.batchSize)
else:
    batch_size = 1536
if args.rolloutLength:
    rollout_fragment_length = int(args.rolloutLength)
else:
    rollout_fragment_length = 512
if args.numEnvRunners:
    num_runners = int(args.numEnvRunners)
else:
    num_runners = 1
if args.numGPU:
    num_gpus = int(args.numGPU)
else:
    num_gpus = 1
if args.numCPU:
    num_cpus = int(args.numCPU)
else:
    num_cpus = 0
if args.numIter:
    num_iterations = int(args.numIter)
else:
    num_iterations = 10
if args.verbose:
    verbose =bool(args.verbose)
else:
    verbose = True
if args.WandBKey:

    wandb_key = args.WandBKey
    wandb.login(key = wandb_key)

    run = wandb.init(
        project = 'Selective_Masking_Training',
        config={
            "learning rate" : 2.5e-4,
            "epochs" : num_iterations,
            "batch size" : batch_size,
        },
    )
else:
    wandb_key = None



# --- Environment creator ---
def env_creator(config):
    env = Surround_v2_Wrapper(
        surround_v2.parallel_env(
        obs_type="rgb_image",
        full_action_space=False,
        max_cycles=1000,
    )
    )

    # IMPORTANT for RLlib env checks
    env.reset()
    return env


# --- Register env with RLlib ---
ENV_NAME = "surround_v2"

register_env(
    ENV_NAME,
    lambda config: ParallelPettingZooEnv(env_creator(config)),
)

config = (
    PPOConfig()
    .environment(env=ENV_NAME)
    .callbacks(
        functools.partial(
            (
                SelfPlayCallback
            ),
            win_rate_threshold=0.7,
            max_league_size = 15
        )
    )
    .framework("torch")
    .rl_module(
        model_config=DefaultModelConfig(
            conv_filters=[
                [16, 4, 2],
                [32, 4, 2],
                [64, 4, 2],
                [128, 4, 2],
            ],
            fcnet_activation="relu",
        )
    )
    .training(
        train_batch_size=batch_size,
        gamma=0.99,
        lr=2.5e-4,
        clip_param=0.2,
        vf_loss_coeff=0.5,
        entropy_coeff=0.01,
        minibatch_size=512,
        num_epochs=1,
    )
    .resources(num_gpus=num_gpus)
    .multi_agent(
    policies={"main"},
    policies_to_train=["main"],
    policy_mapping_fn=lambda agent_id, *args, **kwargs: "main",
    )
    .env_runners(
        num_env_runners = num_runners,
        #num_cpus_per_env_runner=1,
        #num_envs_per_env_runner = 3,
        rollout_fragment_length=rollout_fragment_length,
        batch_mode="truncate_episodes",

    )
    .learners(
    num_learners=1,
    num_gpus_per_learner=1,
    )
)

ray.init(
     num_cpus=int(num_cpus),
     num_gpus=int(num_gpus),
)

algo = config.build_algo()
algo.restore(load_dir)
policy_loss = {}
env_reward = []
for i in range(num_iterations):
    print(str(i + 1) + '/' + str(num_iterations))
    metrics = (algo.train())
    env_reward.append(metrics["env_runners"].get("episode_return_mean"))
    win_rate = (
            metrics["env_runners"]["win_rate"]
        )
    reward = metrics["env_runners"]["module_episode_returns_mean"]["main"]
    if wandb_key != None:
        print('logged to wandb')
        wandb.log({'Main Policy Winrate': win_rate})
        wandb.log({'Main Policy Mean Reward': reward})
    if i % int(checkpoint) == 0:
        dir = os.path.abspath(save_dir + "/ep-" + str(i))
        algo.save(dir)
if wandb_key != None:
    wandb.finish()
