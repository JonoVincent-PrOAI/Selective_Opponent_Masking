from typing import Dict, Optional, Tuple, Union

import gymnasium as gym

from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType
import torch
import numpy as np


    
class ActionMaskingPPO(PPOTorchRLModule):
    @override(RLModule)
    def __init__(
        self,
        *,
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
        inference_only: Optional[bool] = None,
        learner_only: bool = False,
        model_config: Optional[Union[dict, DefaultModelConfig]] = None,
        catalog_class=None,
        **kwargs,
    ):
        
        self.empty_space = 0
        self.opp_space = 1
        self.pla_space = 2
        self.actions = np.array([1,2,3,4])
        self.transforms = np.array([[-1,0],[0,1],[0,-1],[1,0]])

        self.transforms = torch.tensor(
            self.transforms,
            dtype=torch.long
        )

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            inference_only=inference_only,
            learner_only=learner_only,
            model_config=model_config,
            catalog_class=catalog_class,
            **kwargs,
        )

    def get_safe_actions(self, batch):

        obs_batch = batch[Columns.OBS]
        if obs_batch.ndim == 4:
            obs_batch = obs_batch[..., 0]
        
        device = obs_batch.device

        B,H,W = obs_batch.shape
        A = len(self.actions)

        player_pos_batch, has_player = self.get_player_pos(obs_batch)
        next_pos = player_pos_batch[:, None, :] + self.transforms[None, :, :].to(device)
        x = next_pos[:, :, 0]
        y = next_pos[:, :, 1]

        in_bounds = (x >= 0) & (x < H) & (y >= 0) & (y < W)
        if isinstance(in_bounds, np.ndarray):
            in_bounds = torch.from_numpy(in_bounds)

        x_clipped = torch.clamp(x, 0, H - 1)
        y_clipped = torch.clamp(y, 0, W - 1)

        batch_idx = torch.arange(B, device=device).unsqueeze(1)


        is_empty = obs_batch[batch_idx, x_clipped, y_clipped] == self.empty_space        
        
        safe_mask = in_bounds & is_empty
        safe_mask[~has_player] = False
        

        return safe_mask

    def get_player_pos(self, obs_batch):
    

        B, H, W = obs_batch.shape
        device = obs_batch.device

        player_mask = (obs_batch == self.pla_space)
        flat_player = player_mask.view(B, -1)

        player_idx = torch.argmax(flat_player.int(), dim=1)
        has_player = flat_player[torch.arange(B, device=device), player_idx]

        opp_mask = (obs_batch == self.opp_space)
        flat_opp = opp_mask.view(B, -1)

        opp_idx = torch.argmax(flat_opp.int(), dim=1)
        has_opp = flat_opp[torch.arange(B, device=device), opp_idx]

        final_idx = torch.where(has_player, player_idx, opp_idx)

        rows = final_idx // W
        cols = final_idx % W
        positions = torch.stack([rows, cols], dim=1)

        has_player = has_player | has_opp

        positions[~has_player] = -1

        return positions, has_player      


    @override(PPOTorchRLModule)
    def _forward_inference(self, batch: Dict[str, TensorType], **kwargs) -> Dict[str, TensorType]:
        outs = super()._forward_inference(batch, **kwargs)[Columns.ACTION_DIST_INPUTS]
       
        safe_mask = self.get_safe_actions(batch)
        padding_mask = safe_mask.sum(dim=1, keepdim=True) == 0
        safe_mask = torch.cat([padding_mask, safe_mask], dim=1)

        masked_outs = outs + ((~safe_mask) * (-1e9))

        print(masked_outs)

        return {
        Columns.ACTION_DIST_INPUTS: masked_outs
        }
    
    @override(PPOTorchRLModule)
    def _forward_exploration(self, batch: Dict[str, TensorType], **kwargs) -> Dict[str, TensorType]:
        outs = super()._forward_inference(batch, **kwargs)[Columns.ACTION_DIST_INPUTS]

        safe_mask = self.get_safe_actions(batch)
        padding_mask = safe_mask.sum(dim=1, keepdim=True) == 0
        safe_mask = torch.cat([safe_mask, padding_mask], dim=1)

        masked_outs = outs + ((~safe_mask) * (-1e9))

        return {
        Columns.ACTION_DIST_INPUTS: masked_outs
        }
    
    @override(PPOTorchRLModule)
    def _forward_train(self, batch: Dict[str, TensorType], **kwargs) -> Dict[str, TensorType]:
        outs = super()._forward_inference(batch, **kwargs)[Columns.ACTION_DIST_INPUTS]
        
        safe_mask = self.get_safe_actions(batch)
        padding_mask = safe_mask.sum(dim=1, keepdim=True) == 0
        safe_mask = torch.cat([safe_mask, padding_mask], dim=1)

        masked_outs = outs + ((~safe_mask) * (-1e9))

        return {
        Columns.ACTION_DIST_INPUTS: masked_outs
        }