from collections import defaultdict

import numpy as np

from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS

'''
Self_play_callback adapted from:
https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent/utils/self_play_callback.py

Further adapted to implement priotitised fictious self-play.
'''
class SelfPlayCallback(RLlibCallback):
    def __init__(self, win_rate_threshold, max_league_size):
        super().__init__()
        # 0=RandomPolicy, 1=1st main policy snapshot, -> no random so indexing form 0
        # 2=2nd main policy snapshot, etc..
        self.current_opponent = 0 #tracks ooponent version number
        self.win_rate_threshold = win_rate_threshold #threshold foradding new opponent to league
        self.max_league_size = max_league_size
        self.non_opponent_modules = ['__all_modules__', 'default_policy', 'main']


    def on_episode_end(
        self,
        *,
        episode,
        env_runner,
        metrics_logger,
        env,
        env_index,
        rl_module,
        **kwargs,
    ) -> None:
        # Compute the win rate for this episode and log it with a window of 100.
        if episode.module_for("first_0") == "main":
            main_agent = "first_0"
        else:
            main_agent = "second_0"
        
        rewards = episode.get_rewards()
        if main_agent in rewards:
            main_won = sum(rewards[main_agent]) > 0.0 #changed for surround env
            metrics_logger.log_value(
                key = "win_rate",
                value = float(main_won),
                reduce="mean",
            )
        
        modules = []
        for agent_id in ['first_0', 'second_0']:
            modules.append(episode.module_for(agent_id))

        metrics_logger.log_value(
            key = ("matchups", (str(modules[0]) + ", " + str(modules[1]))),
            value = 1,
            reduce="sum"
        )

    # Re-define the mapping function, such that "main" is forced
    # to play against any of the previously played modules
    # (excluding "random"). -> no need to exclude random
    def agent_to_module_mapping_fn(self, agent_id, episode, **kwargs):
        # agent_id = [0|1] -> policy depends on episode ID
        # This way, we make sure that both modules sometimes play
        # (start player) and sometimes agent1 (player to move 2nd).

        modules = episode.env_runner.module.keys()
        league_opponents = [
        m for m in modules
        if m not in self.non_opponent_modules
        ]

        hash_id = hash(episode.id_)

        if len(league_opponents) == 0:
            opponent = "main"
        else:
            idx = hash((hash_id, "opponent")) % len(league_opponents)
            opponent = league_opponents[idx]

        side = hash((hash_id, "side")) % 2

        if side == 0:
            return "main" if agent_id == "first_0" else opponent
        else:
            return "main" if agent_id == "second_0" else opponent
    


    def add_new_module(self, algorithm):

        new_module_id = f"main_v{self.current_opponent}"
        print(f"adding new opponent to the mix ({new_module_id}).")

        main_module = algorithm.get_module("main")
        main_state = main_module.get_state()

        algorithm.add_module(
            module_id=new_module_id,
            module_spec=RLModuleSpec.from_module(main_module),
            config_overrides = {"policies_to_train":["main"]},
            new_agent_to_module_mapping_fn=self.agent_to_module_mapping_fn,
            new_should_module_be_updated = ['main'],
        )

        algorithm.get_module(new_module_id).set_state(main_state)
        self.current_opponent += 1


    def on_train_result(self, *, algorithm, metrics_logger=None, result, **kwargs):
        
        if self.current_opponent == 0:
            self.add_new_module(algorithm)
        win_rate = result.get("env_runners", {}).get("win_rate")

        if win_rate is None:
            print(f"Iter={algorithm.iteration} no win_rate yet.")
        elif win_rate > self.win_rate_threshold:
            print('Win Rate: ' + str(win_rate))
            self.add_new_module(algorithm)
        else:
            print('Win Rate: ' + str(win_rate))
            print("not good enough; will keep learning ...")

        # +2 = main + random -> +1 no random
        result["league_size"] = self.current_opponent + 1
        print(
            "Modules:",
            list(
                algorithm.env_runner_group
                        .local_env_runner
                        .module.keys()
            )
        )

        modules = algorithm.local_env_runner.module.keys()
        league_opponents = sorted(
            m for m in modules if m not in self.non_opponent_modules
        )

        if len(league_opponents) > self.max_league_size:
            oldest_opponent = league_opponents.pop(0)
            algorithm.remove_module(oldest_opponent)
            print('removed opponent: ' + str(oldest_opponent))

        print("Matchups: ")
        matchups = result['env_runners']['matchups']
        for key in matchups.keys():
            print(key +': '+ str(matchups[key]))

        for k,v in result["learners"].items():
            if "num_module_steps_trained" in v:
                print(k, v["num_module_steps_trained"])
