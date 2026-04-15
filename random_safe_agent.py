from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.columns import Columns
from ray.rllib.utils.annotations import override

import numpy as np

class random_safe_surround(RLModule):
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.player = 1
        self.safe = 0
        self.action_transforms = {1: [-1,0], 2:[0,1], 3:[0,-1], 4:[1,0]}

    def get_safe_actions(self, obs):
        safe_actions = []
        player_pos = self.get_player_pos(obs)
        if player_pos != None:
            safe_actions = []
            for action, transform in zip(self.action_transforms.keys(), self.action_transforms.values()):
                pos = [(player_pos[0] + transform[0]), (player_pos[1] + transform[1])]
                if pos[0] < len(obs) and pos[1] < len(obs[0]):
                    if obs[pos[0]][pos[1]] == self.safe:
                        safe_actions.append(action)
        if safe_actions == []:
            safe_actions = [0]
        return(safe_actions)


    def get_player_pos(self, obs):
        player_pos = None
        pos = np.argwhere(obs == 2).tolist()
        if len(pos[0]) > 0:
            player_pos = [pos[0][0], pos[1][0]]
        return(player_pos)

    @override(RLModule)
    def _forward_inference(self, batch, **kwargs):
        ret = []

        for i, obs in enumerate(batch[Columns.OBS]):
            ret.append(np.random.choice(self.get_safe_actions(obs)))

        return {Columns.ACTIONS: np.array(ret)}

    @override(RLModule)
    def _forward_exploration(self, batch, **kwargs):
        return self._forward_inference(batch, **kwargs)

    @override(RLModule)
    def _forward_train(self, batch, **kwargs):
        raise NotImplementedError(
            "safe_random is not trainable! Make sure you do NOT include it "
            "in your `config.multi_agent(policies_to_train={...})` set."
        )
