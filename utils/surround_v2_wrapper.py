import gymnasium as gym
import pettingzoo
from pettingzoo.atari import surround_v2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
import supersuit

"""
This is a wrapper for the mutli agent pettingzoo.atari.surround_v2 environment.
The wrapper converts the observation sapce from a a 210 x 160 rgb image to a 38x16 grid world.
The wrapper also skips frames so the environment updates when the game updates.

Observation Space: (4,(38x16))
Agents: (first_0, second_0)
Actions: (0:NOOP, 1:FIRE, 2:UP, 3:RIGHT, 4:LEFT, 5:DOWN) - fire is a dummy action that does nothing
"""

class Surround_v2_Wrapper():

    def __init__(self, surround_env = None, frame_skip = None):
        
        self.BOARD_BOUNDARY = [27,207,0,160]#the edges of the board in the rgb image (y,y,x,x)
        self.BOARD_CELL_SIZE = {'height': 20, 'width': 40}#the dimensions of the baord in number of cells
        self.CELL_DIMENSION = {'height' : 9, 'width' : 4}#the pixel dimensions of cells in the rgb image
        self.action_transforms = {1: [-1,0], 2:[0,1], 3:[0,-1], 4:[1,0]}

        self.AGENT_INFO = {
            'first_0': #agents name
            {
                'colour': np.array([92,186,92]),#rgb colour of the agent
                #agents have unique colour mapping so the agent controlled by the policy is the same colour
                'colour mapping': [[184,50,50], [92,186,92], [45,50,184], [227,151,89]],
                'start pixel': [116,126,40,44]#region of pixels the agent starts in (y,y,x,x)
            }, 
            'second_0': 
            {
                'colour':  np.array([45,50,184,]),
                'colour mapping': [[184,50,50], [45,50,184], [92,186,92], [227,151,89]],
                'start pixel': [116,126,120,124]
            }
            }
        
        if surround_env == None:
            self.env = surround_v2.parallel_env(
            obs_type="rgb_image",
            full_action_space=False,
            max_cycles=27000,
            )
        else:
            self.env = surround_env


        if frame_skip == None:
            self.frame_skip = 15
        else:
            self.frame_skip = frame_skip


        self.env.reset(seed=42)
        self.possible_agents = self.env.possible_agents
        self.agents = self.env.agents

        self.observation_spaces = {
            agent: gym.spaces.Box(
                low=0,
                high=4,
                shape=(20, 40, 1),
                dtype=np.float32,
            )
            for agent in self.env.possible_agents
        }

        self.action_spaces_dict = self.env.action_spaces

        self.last_env_obs = None

    '''
    Resets the environemnt, required by gym.
    '''
    def reset(self, *, seed=None, options=None):
        board_img, info = self.env.reset(seed = seed, options = options)
        self.agents = self.env.agents
        self.last_env_obs = board_img
        #converts reward into down-sampled obs space
        obs = {}
        for agent in self.agents:
            obs[agent] = self.update_board(board_img[agent], agent)

        return obs, info
    
    '''
    Takes a step in the environment. Converts obs into a compressed space used by the wrapper.
    Also skips frames to avoid states where no change has occurred.
    '''
    def step(self, action_dict):

        total_reward = {agent: 0 for agent in action_dict}
        for _ in range(self.frame_skip):
            img_obs, reward, termination, truncation, info = self.env.step(action_dict)
            self.last_env_obs = img_obs
            for agent in reward:
                total_reward[agent] += reward[agent]

            if any(termination.values()) or any(truncation.values()):
                break

        #converts obs to down-sampled obs
        obs = {}
        for agent in reward.keys():
            obs[agent] = self.update_board(img_obs[agent], agent)

        #adds reward for not dying on each timestep
        #for agent in reward.keys():
            # if action_dict[agent] in self.get_safe_actions(obs[agent]):
            #     total_reward[agent] += 0.1

        return obs, total_reward, termination, truncation, info
    
    def get_safe_actions(self, obs):
        safe_actions = []
        player_pos = self.get_player_pos(obs)
        if player_pos != None:
            safe_actions = []
            for action, transform in zip(self.action_transforms.keys(), self.action_transforms.values()):
                pos = [(player_pos[0] + transform[0]), (player_pos[1] + transform[1])]
                if pos[0] < len(obs) and pos[1] < len(obs[0]):
                    if obs[pos[0]][pos[1]] == 0:
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
    
    def close(self):
        self.env.close()
        self.agents = self.env.agents
    
    def observation_space(self, agent_id):
        return(self.observation_spaces[agent_id])
    
    def action_space(self, agent_id):
        return(self.action_spaces_dict[agent_id])

    def _get_obs(self):
        obs = {}
        for agent in self.agents:
            obs[agent] = self.update_board(self.last_env_obs[agent], agent)
        
        return obs

    def _get_info(self):

        return self.env._get_info()

    def get_unwrapped_obs(self):
        return(self.last_env_obs)

    '''
    Converts rgb image to 38x18 grid world array.
    '''
    def update_board(self,board_img, agent_id):

        #gets rgb colours and corresponding integer value
        colours = np.array(self.AGENT_INFO[agent_id]['colour mapping'])

        #crops the image to only contain the board
        px_board = board_img[self.BOARD_BOUNDARY[0] : self.BOARD_BOUNDARY[1], self.BOARD_BOUNDARY[2] : self.BOARD_BOUNDARY[3]]
        
        #down samples the board by the dimensions of the cells so each cell is one pixel
        rgb_board = px_board[::self.CELL_DIMENSION['height'], ::self.CELL_DIMENSION['width']]
        board = np.empty((rgb_board.shape[0],rgb_board.shape[1]),dtype=int)
        board[:] = -1
        #replaces each pixel (rgb colour) with coresponding integer from colour mapping
        R,C = np.where(cdist(rgb_board.reshape(-1,3),colours)==0) 
        board.ravel()[R] = C

        return(board[..., np.newaxis].astype(np.float32))
