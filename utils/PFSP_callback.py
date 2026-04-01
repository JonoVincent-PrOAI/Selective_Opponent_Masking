from collections import defaultdict

import numpy as np
import math
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS

'''
Prioritised Fictitious self-play sans meta strategies.
This is a callback function, with the primary goal of sampling previous version of the main policy
based on their win-rate against the main policy. The higher the win-rate the higher lieklihood of being
sampled.
'''
class PFSPCallback(RLlibCallback):

    def __init__(self, win_rate_threshold, max_league_size, num_modules):
        super().__init__()
        # 0=RandomPolicy, 1=1st main policy snapshot, -> no random so indexing form 0
        # 2=2nd main policy snapshot, etc..
        self.current_opponent = 0 #tracks ooponent version number

        self.opponent_weights = {}#Dictionary of opponent names and corresponding weights
        self.module_to_opponent_mapping = {}#Dictionary of opponent name and their corresponding module

        self.win_rate_threshold = win_rate_threshold #threshold foradding new opponent to league

        self.max_league_size = max_league_size
        self.num_modules = num_modules
    '''
    Creates adds opponent modules to algrithm. Assigining different model weights to these modules
    is used to influence the modles likelihood of being chosen.
    During initial creation all mdoules are passed the weigths of the current main policy.
    '''
    def intitialise_league(self, algorithm):
        print('Initialising league...')
        #Gets the weigths of the main policy
        main_module = algorithm.get_module('main')
        main_state = main_module.get_state(inference_only = True)
        #adds opponent to module to opponent mapping
        opponent = f"opponent_{self.current_opponent}"
        self.current_opponent +=1
        self.module_to_opponent_mapping[opponent] = []
        #Creates modules
        for i in range(self.num_modules):
            module_id = f"opp_mod_{i}"
            algorithm.add_module(
            module_id=module_id,
            module_spec=RLModuleSpec.from_module(main_module),
            config_overrides = {"policies_to_train":["main"]},
            new_agent_to_module_mapping_fn=self.agent_to_module_mapping_fn,
            new_should_module_be_updated = ['main'],
            )
            print("...Added module " + module_id + " with " + opponent + " weights.")
            #assigns weights to modules and adds them to opponent/module mapping
            algorithm.get_module(module_id).set_state(main_state)
            self.module_to_opponent_mapping[opponent].append(module_id)
        #saves the weights used for the opponent
        self.opponent_weights[opponent] = main_state
    '''
    Mapping function to map agents in the environment to modules in algorithm
    '''
    def agent_to_module_mapping_fn(self, agent_id, episode, **kwargs):
        #Uniformly samples a module. 
        #Each module an opponents weights are assigned to increases the models likelihood of being sampled by:
        # 1/num_ modules
        module_index = (np.random.choice(a = range(self.num_modules)))

        opponent = f"opp_mod_{module_index}"
        #Assigns opponent and main to an agent depending on episode.id_
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
    '''
    Adds a new opponent to the pool of opponents.
    The added opponent the current version of main frozen.
    '''
    def add_new_opponent(self, algorithm, win_rates):
        
        new_opponent = f"opponent_{self.current_opponent}"
        print(f"adding new opponent to the mix ({new_opponent}).")
        self.current_opponent += 1
        #Gets current weights of main policy
        main_module = algorithm.get_module('main')
        main_state = main_module.get_state(inference_only = True)
        #Stores weights for later
        self.opponent_weights[new_opponent] = main_state
        #Adds opponent to opponent/module mapping
        self.module_to_opponent_mapping[new_opponent] = []

        win_rates[new_opponent] = 0.5 #Assigns the opponent a win_rate of 0.5 as it is the current model
        #removes the oldest opponent if the league is too large
        if len(self.opponent_weights.keys()) > self.max_league_size:
            oldest_opponent = list(self.opponent_weights.keys())[0]
            win_rates[oldest_opponent] = 0#setting win-rate to 0 means it will be culled during module update

        self.update_module_weights(self, win_rates, algorithm)#updates weights of modules
    '''
    Assigns new weights to modules based on the win-rate of opponents
    '''
    def update_module_weights(self, win_rates, algorithm):
        #Calculates the number of modules for each opponent
        mod_per_opp = {}
        max = 0
        max_key = None
        #normalises the win-rates so they roughly sum to the number of modules
        for opp in win_rates.keys():
            num_mod = math.floor((win_rates[opp] / sum(win_rates.values())) * (self.num_modules))
            mod_per_opp[opp] = num_mod
            if num_mod > max:
                max = mod_per_opp[opp]
                max_key = opp
        #Use floor to round so sum(mod_per_opp) > num_modules
        #Leftover modules as a result of floor operation are assigned to the best performing opponent
        diff = self.num_modules - sum(mod_per_opp.values())
        mod_per_opp[max_key] += diff

        #Calaculates the write action which need to be performed to assign modules their opponet weights
        write_actions = []
        overwrite_actions = []
        for opp in mod_per_opp:
            if opp in list(self.module_to_opponent_mapping.keys()):
                diff = mod_per_opp[opp] - len(self.module_to_opponent_mapping[opp])
            else:
                diff = mod_per_opp[opp]

            if diff > 0:
                write_actions += diff * [opp]
            elif diff < 0:
                overwrite_actions += -diff * [opp]
        #Perfroms write actions
        for write, overwrite in zip(write_actions, overwrite_actions):
            #Changes weights
            module_id = self.module_to_opponent_mapping[overwrite][0]
            module = algorithm.get_module(module_id)
            module.set_state(self.opponent_weights[write])
            #Updates opponent/module mapping
            self.module_to_opponent_mapping[overwrite].pop(0)
            self.module_to_opponent_mapping[write].append(module_id)
        #Culls opponents which weren't assigned a module
        opponents = list(self.module_to_opponent_mapping.keys())
        for opp in opponents:
            if len(self.module_to_opponent_mapping[opp]) == 0:
                print('Removing ' + opp + " from league.")
                self.module_to_opponent_mapping.pop(opp)
                self.opponent_weights.pop(opp)
    '''
    Tracks custom metrics like win-rate 
    '''  
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
            if module_won:
                metrics_logger.log_value(
                    key = ("win_rate", module, 'wins'),
                    value = 1,
                    reduce="sum",
                )
                metrics_logger.log_value(
                    key = ("win_rate", module, 'losses'),
                    value = 0,
                    reduce="sum",
                )
            else:
                metrics_logger.log_value(
                    key = ("win_rate", module, 'losses'),
                    value = 1,
                    reduce="sum",
                )
                metrics_logger.log_value(
                    key = ("win_rate", module, 'wins'),
                    value = 0,
                    reduce="sum",
                )
        
        modules = []
        for agent_id in ['first_0', 'second_0']:
            modules.append(episode.module_for(agent_id))

        metrics_logger.log_value(
            key = ("matchups", (str(modules[0]) + ", " + str(modules[1]))),
            value = 1,
            reduce="sum"
        )

    def on_train_result(self, *, algorithm, metrics_logger=None, result, **kwargs):
        main_results = result["env_runners"]["win_rate"]["main"]
        win_rate = main_results["wins"]/(main_results["wins"] + main_results["losses"])

        print('Win Rate: ' + str(win_rate))

        print("Matchups: ")
        matchups = result['env_runners']['matchups']
        for key in matchups.keys():
            if not np.isnan(matchups[key]):
                print('    ' + key +': '+ str(matchups[key]))
        
        mod_win_rates = result['env_runners']['win_rate']
        win_rates = {}
        for opp in self.module_to_opponent_mapping.keys():
                total_wins = 0
                total_losses = 0
                for mod in self.module_to_opponent_mapping[opp]:
                    if mod in mod_win_rates.keys():
                        if not np.isnan(mod_win_rates[mod]['wins']):
                            total_wins += mod_win_rates[mod]['wins']
                            total_losses += mod_win_rates[mod]['losses']
                if total_losses == 0:
                    rate = 1
                else:
                    rate = total_wins / (total_wins + total_losses)
                win_rates[opp] = rate

        print('Winrates: ')  

        for opp in win_rates.keys():
            print('   ' + opp + ': ' + str(win_rates[opp]))

        if self.current_opponent == 0:
            self.intitialise_league(algorithm)
        elif win_rate is None:
            print(f"Iter={algorithm.iteration} no win_rate yet.")
            return
        elif win_rate > self.win_rate_threshold:
            self.add_new_opponent(self, algorithm, win_rates)
        else:
            self.update_module_weights(self, algorithm, win_rates)

        print(self.module_to_opponent_mapping)

        algorithm.env_runner_group.sync_weights()
