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
        
        self.BOARD_BOUNDARY = [36,198,4,156]#the edges of the board in the rgb image (y,y,x,x)
        self.BOARD_CELL_SIZE = {'height': 18, 'width': 38}#the dimensions of the baord in number of cells
        self.CELL_DIMENSION = {'height' : 9, 'width' : 4}#the pixel dimensions of cells in the rgb image

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
                shape=(18, 38, 1),
                dtype=np.float32,
            )
            for agent in self.env.possible_agents
        }

        self.action_spaces_dict = self.env.action_spaces

    '''
    Resets the environemnt, required by gym.
    '''
    def reset(self, *, seed=None, options=None):
        board_img, info = self.env.reset(seed = seed, options = options)
        self.agents = self.env.agents

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
        #sums rewards across skipped frames to ensure no reward signal is missed
        total_reward = {agent: 0 for agent in action_dict}
        for _ in range(self.frame_skip):
            img_obs, reward, termination, truncation, info = self.env.step(action_dict)

            for agent in reward:
                total_reward[agent] += reward[agent]

            if any(termination.values()) or any(truncation.values()):
                break

        #converts obs to down-sampled obs
        obs = {}
        for agent in reward.keys():
            obs[agent] = self.update_board(img_obs[agent], agent)

        return obs, total_reward, termination, truncation, info
    
    def close(self):
        self.env.close()
        self.agents = self.env.agents
    
    def observation_space(self, agent_id):
        return(self.observation_spaces[agent_id])
    
    def action_space(self, agent_id):
        return(self.action_spaces_dict[agent_id])

    def _get_obs(self):
        img_obs, reward, termination, truncation, info = self.env._get_obs()
        obs = {}
        for agent in self.agents:
            obs[agent] = self.update_board(img_obs[agent], agent)
        
        return obs

    def _get_info(self):

        return self.env._get_info()

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
