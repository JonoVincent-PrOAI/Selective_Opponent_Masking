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
from utils.PFSP_callback import PFSPCallback

parser = argparse.ArgumentParser(description="Pretraining for model in the surroun_v2 env. Trains a model in wrapped surround_v5")
parser.add_argument("-ldir", "--loadDirectory", help="Model checkpoint directory.")
parser.add_argument("-sdir", "--saveDirectory", help="Directory checkpoints are saved to.", default= "./ray_results/PPO_surround_v2/")
parser.add_argument("-chkpt", "--checkpoint", help="After how many episodes should a checkpoint be saved.", default=10)

parser.add_argument("-sz", "--batchSize", help="Size of training batches.", default=1536)
parser.add_argument("-rl", "--rolloutLength", help="Lenght of rollout fragments.", default=512)
parser.add_argument('-sgd', "--sgdIterations", help="Number of Sgd updates per batch.", default=20)
parser.add_argument("-mbsz", "--minibatchSize", help='Size of sgd minibatches', default = 512)

parser.add_argument("-ner", "--numEnvRunners", help="Number of env ruuners.", default=1)
parser.add_argument("-ngpu", "--numGPU", help="Number of GPUs available for training.", default=1)
parser.add_argument("-ncpu", "--numCPU", help="Number of CPUs available for training.", default=0)
parser.add_argument("-cpurun", "--numCPUperRun", help="Number of CPUs per env runner instance.", default=1)
parser.add_argument("-envrun", "--numEnvPerRun", help="Number of env instances per env runner.", default=3)

parser.add_argument("-nl", "--numLearners", help="Number of learner instances.", default=1)
parser.add_argument("-gpul", "--numGPUperLearn", help="Number of GPUs per learner instance.", default=1)
parser.add_argument("-cpul", "--numCPUperLearn", help="Number of CPUs per learner instance.", default=1)


parser.add_argument("-ni", "--numIter", help="Number of Training Iterations.", default=10)
parser.add_argument("-nop", "--numOpp", help="Max Number of Opponents in League.", default=15)
parser.add_argument("-nmod", "--numMod", help ="the Number of RLLIB modules that will be created.", default=15)
parser.add_argument("-win", "--winThreshold", help="Wunrate threshold where a new opponent is added to the league.", default=0.7)

parser.add_argument("-v", "--verbose", help="True/False whether outputs should be given.", default=True)
parser.add_argument("-wnb", "--WandBKey", help="API key W and B logger.")
args = parser.parse_args()

if args.loadDirectory:
    load_dir = args.loadDirectory
else:
    raise ValueError("Load directory must be provided")
save_dir = args.saveDirectory
checkpoint = args.checkpoint

batch_size = int(args.batchSize)
rollout_fragment_length = int(args.rolloutLength)
num_sgd_iter = int(args.sgdIterations)
minibatch_size = int(args.minibatchSize)

num_runners = int(args.numEnvRunners)
num_gpus = int(args.numGPU)
num_cpus = int(args.numCPU)
num_cpu_per_env_runner = int(args.numCPUperRun)

num_env_per_env_runner = int(args.numEnvPerRun)
num_learners = int(args.numLearners)
num_GPUs_per_learner = int(args.numGPUperLearn)
num_CPUs_per_learner = int(args.numCPUperLearn)

num_iterations = int(args.numIter)
max_league_size = int(args.numOpp)
num_mod = int(args.numMod)
win_threshold = float(args.winThreshold)
verbose =bool(args.verbose)
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

def betas_tensor_to_float(learner):
    for param_grp_key in learner._optimizer_parameters.keys():
        param_grp = param_grp_key.param_groups[0]
        param_grp["betas"] = tuple(beta.item() for beta in param_grp["betas"])

# --- Environment creator ---
def env_creator(config):
    env = Surround_v2_Wrapper(
        surround_v2.parallel_env(
        obs_type="rgb_image",
        full_action_space=False,
        max_cycles=15000,
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
                PFSPCallback
            ),
            win_rate_threshold=win_threshold,
            max_league_size = max_league_size,
            num_modules = num_mod,
        )
    )
    .framework("torch")
    .rl_module(
        model_config=DefaultModelConfig(
            conv_filters=[
                [32, 4, 2],
                [64, 4, 2],
                [128, 4, 2],
                [256, 4, 2],
            ],
            conv_activation="silu",
            head_fcnet_hiddens=[256],
        )
    )
    .training(
        train_batch_size=batch_size,
        gamma=0.99,
        lr=2.5e-4,
        clip_param=0.2,
        vf_loss_coeff=0.5,
        entropy_coeff=0.01,
        minibatch_size=minibatch_size,
        num_epochs=num_sgd_iter,
    )
    .resources(num_gpus=num_gpus)
    .multi_agent(
    policies={"main"},
    policies_to_train=["main"],
    policy_mapping_fn=lambda agent_id, *args, **kwargs: "main",
    )
    .env_runners(
        num_env_runners = num_runners,
        num_cpus_per_env_runner = num_cpu_per_env_runner,
        num_envs_per_env_runner = num_env_per_env_runner,
        rollout_fragment_length=rollout_fragment_length,
        batch_mode="truncate_episodes",

    )
    .learners(
        num_learners = num_learners,
        num_gpus_per_learner = num_GPUs_per_learner,
        num_cpus_per_learner = num_CPUs_per_learner, 
    )
)

ray.init(
     num_cpus=int(num_cpus),
     num_gpus=int(num_gpus),
)

algo = config.build_algo()
algo.restore(os.path.abspath(load_dir))
algo.learner_group.foreach_learner(betas_tensor_to_float)
policy_loss = {}
env_reward = []
if wandb_key != None:
    wandb.init()

for i in range(num_iterations):
    print(str(i + 1) + '/' + str(num_iterations))
    metrics = (algo.train())
    env_reward.append(metrics["env_runners"].get("episode_return_mean"))

    if wandb_key != None:
        if 'main' in metrics['env_runners']['win_rate'].keys():
            win_loss = (
                    metrics["env_runners"]["win_rate"]["main"]
                )
            win_rate = (
                metrics["env_runners"]["win_rate"]["main"]['wins']/(metrics["env_runners"]["win_rate"]["main"]['wins'] + metrics["env_runners"]["win_rate"]["main"]['losses'])
            )
        wandb.log({'Main Policy Win/loss': win_loss})
        wandb.log({'Main Policy win rate': win_rate})
        print('logged to wandb')
        if 'main' in metrics["env_runners"]["module_episode_returns_mean"].keys():
            reward = metrics["env_runners"]["module_episode_returns_mean"]["main"]
            wandb.log({'Episode Mean Reward': reward})
    if i % int(checkpoint) == 0:
        dir = os.path.abspath(save_dir + "/ep-" + str(i))
        algo.save(dir)
if wandb_key != None:
    wandb.finish()
