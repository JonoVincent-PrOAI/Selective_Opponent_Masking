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
class PFSPCallback(RLlibCallback):
    def __init__(self, win_rate_threshold, max_league_size, self_play_prob):
        super().__init__()
        # 0=RandomPolicy, 1=1st main policy snapshot, -> no random so indexing form 0
        # 2=2nd main policy snapshot, etc..
        self.current_opponent = 0 #tracks ooponent version number
        self.league = [] #list of opponent included in the legue
        self.win_rate_threshold = win_rate_threshold #threshold foradding new opponent to league
        self.max_league_size = max_league_size
        self.self_play_prob = self_play_prob

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
        rewards = episode.get_rewards()
        for agent in rewards:
            module = episode.module_for(agent)
            module_won = sum(rewards[agent]) > 0.0 #changed for surround env
            metrics_logger.log_value(
                key = ("win_rate", module),
                value = float(module_won),
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

    def add_new_module(self, algorithm, result):
        
        win_rates = result['env_runners']['win_rate']
        win_rates.pop('main')

        new_module_id = f"main_v{self.current_opponent}"
        print(f"adding new opponent to the mix ({new_module_id}).")
        #updating league
        self.current_opponent += 1
        self.league.append(new_module_id)

        if len(self.league) > self.max_league_size:
            oldest_opponent = self.league[0]
            self.league.pop(0)
            algorithm.remove_module(oldest_opponent)
            win_rates.pop(oldest_opponent)
        #Using snapshot of league so that it is consistent across all env_runner instances
        league_snapshot = list(self.league)

        for key in win_rates:
            if np.isnan(win_rates[key]):
                win_rates[key] = 0
        if len(league_snapshot) ==1:
            match_up_probs = [1]
        else:
            match_up_probs = [
                self.self_play_prob if opp == new_module_id
                else
                ((win_rates[opp] / sum(win_rates.values())) * (1-self.self_play_prob))
                for opp in league_snapshot
            ]
        # Re-define the mapping function, such that "main" is forced
        # to play against any of the previously played modules
        def agent_to_module_mapping_fn(agent_id, episode, **kwargs):

            opponent = (np.random.choice(a = league_snapshot, p = match_up_probs))

            module = 'error'
            if hash(episode.id_) % 2 == 0:
                if agent_id == 'first_0':
                    module = "main"
                else:
                    module = opponent
            else:
                if agent_id == 'second_0':
                    module =  "main"
                else:
                    module = opponent
            return(module)
        
        main_module = algorithm.get_module("main")

        algorithm.add_module(
            module_id=new_module_id,
            module_spec=RLModuleSpec.from_module(main_module),
            config_overrides = {"policies_to_train":["main"]},
            new_agent_to_module_mapping_fn=agent_to_module_mapping_fn,
            new_should_module_be_updated = ['main'],
        )
        main_state = main_module.get_state()
        algorithm.get_module(new_module_id).set_state(main_state)

    def on_train_result(self, *, algorithm, metrics_logger=None, result, **kwargs):
        
        win_rate = (
            result["env_runners"]["win_rate"]["main"]
        )
        print('Win Rate: ' + str(win_rate))

        print("Matchups: ")
        matchups = result['env_runners']['matchups']
        for key in matchups.keys():
            if not np.isnan(matchups[key]):
                print('    ' + key +': '+ str(matchups[key]))

        print('Winrates: ')  
        win_rates = result['env_runners']['win_rate']
        for key in win_rates:
            if not np.isnan(win_rates[key]):
                print('    ' + key + ": " + str(win_rates[key]))


        if self.current_opponent == 0:
            self.add_new_module(algorithm, result)
        elif win_rate is None:
            print(f"Iter={algorithm.iteration} no win_rate yet.")
            return
        elif win_rate > self.win_rate_threshold:
            self.add_new_module(algorithm, result)
        else:
            print("not good enough; will keep learning ...")

        # +2 = main + random -> +1 no random
        result["league_size"] = self.current_opponent + 1