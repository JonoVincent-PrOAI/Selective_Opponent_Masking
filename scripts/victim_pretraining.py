import os
import torch
import gymnasium as gym
import argparse
import sys
import wandb
import ray

if os.path.abspath("../") not in sys.path:
    sys.path.append(os.path.abspath("../"))
if os.path.abspath("./") not in sys.path:
    sys.path.append(os.path.abspath("./"))

from utils.surround_v5_wrapper import Surround_v5_Wrapper
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig

# --- Defines and parses arguments ---
parser = argparse.ArgumentParser(description="Pretraining for model in the surroun_v2 env. Trains a model in wrapped surround_v5")
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

if args.saveDirectory:
    save_dir = args.saveDiectory
else:
    save_dir = "./ray_results/PPO_surround_v5/"
if args.checkpoint:
    checkpoint = args.checkpoint
else:
    checkpoint = 10
if args.batchSize:
    batch_size = int(args.batchSize)
else:
    batch_size = 1024
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
if args.numGPU:
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
        project = 'Selective_Masking_Pretraining',
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
    env = Surround_v5_Wrapper()
    # IMPORTANT for RLlib env checks
    env.reset()
    return env


# --- Register env with RLlib ---
ENV_NAME = "surround_v5"

register_env(
    ENV_NAME,
    lambda config: env_creator(config),
)

config = (
    PPOConfig()
    .environment(env=ENV_NAME)
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
    )
    .resources(num_gpus=num_gpus)
    .env_runners(
        num_env_runners = num_runners,
        num_cpus_per_env_runner=num_cpus/num_runners,   
        rollout_fragment_length=rollout_fragment_length,
        batch_mode="complete_episodes",
    )
)

ray.init(num_cpus=num_cpus, num_gpus=num_gpus)
algo = config.build()
policy_loss = {}
env_reward = []
for i in range(num_iterations):
    print(str(i+1) + '/' + str(num_iterations))
    metrics = (algo.train())
    ep_reward = metrics["env_runners"].get("episode_return_mean")
    print("Episode reward mean:", ep_reward)
    if wandb_key != None:
        print('logged to wandb')
        wandb.log({'Episode Reward Mean': ep_reward})


    env_reward.append(metrics["env_runners"].get("episode_return_mean"))

    if i % int(checkpoint) == 0:
        dir = os.path.abspath(save_dir + "/ep-" + str(i))
        algo.save(dir)

if wandb_key != None:
    wandb.finish()
