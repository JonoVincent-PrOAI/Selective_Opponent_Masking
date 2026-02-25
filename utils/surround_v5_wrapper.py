import gymnasium as gym
import ale_py
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
import supersuit
import random

"""
This is a wrapper for the single agent ale_py.surround_v5 environment.This wrapper is designed to bridge the gap 
between surround_v5 and the surround_v2_wrapper.
The wrapper:

1. converts the observation sapce from a a 210 x 160 rgb image to a 38x16 grid world.
2. skips frames so the environment updates when the game updates.
3. projections v5's action space (has no fire action) into v2's ation space (1 = fire)
4. horizontally flips obs and left and right actions, so the policy can learn to control both the left and right 
   agents.

Observation Space: (4,(38x16))
Actions: (0:NOOP, 1:UP, 2:RIGHT, 3:LEFT, 4:DOWN) - left and right are sometimes swapped
"""
class Surround_v5_Wrapper(gym.Env):

    def __init__(self, surround_env = None, frame_skip = None):
        
        self.BOARD_BOUNDARY = [36,198,4,156]#the edges of the board in the rgb image (y,y,x,x)
        self.BOARD_CELL_SIZE = {'height': 18, 'width': 38}#the dimensions of the baord in number of cells
        self.CELL_DIMENSION = {'height' : 9, 'width' : 4}#the pixel dimensions of cells in the rgb image

        self.AGENT_INFO = {
            'first_0': #agents name
            {
                'colour': np.array([200,72,72]),#rgb colour of the agent
                #agents have unique colour mapping so the agent controlled by the policy is the same colour
                'colour mapping': [[84,92,214], [183,194,95], [200,72,72], [212,108,195]],
                'start pixel': [116,126,40,44],#region of pixels the agent starts in (y,y,x,x)
                'action mapping': [0,0,1,3,2,4],#flips left and right actions
                'flip': True
            }, 
            'second_0': 
            {
                'colour':  np.array([183,194,95]),
                'colour mapping': [[84,92,214], [200,72,72], [183,194,95], [212,108,195]],
                'start pixel': [116,126,120,124],
                'action mapping': [0,0,1,2,3,4],
                'flip': False
            }
            }
        
        #in v5 the policy always controls the right (second_0) agent. We flip the environment so the policy can 
        #learn to control both the left (first_0) and right (second_0) agents.
        self.agent = random.choice(list(self.AGENT_INFO.keys()))

        if frame_skip == None:
            self.frame_skip = 15
        else:
            self.frame_skip = frame_skip 
        
        if surround_env == None:
            self.env = gym.make("ALE/Surround-v5", frameskip = 15, obs_type = 'rgb', full_action_space=False).env
        else:
            self.env = surround_env

        self.env.reset(seed=42)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=4,
            shape=(18, 38, 1),
            dtype=np.float32,
        )

        self.action_space = gym.spaces.Discrete(5)

    '''
    Resets the environemnt, required by gym.
    '''
    def reset(self, *, seed=None, options=None):
        board_img, info = self.env.reset(seed = seed, options = options)

        #converts obs into down-sampled obs space
        obs = self.update_board(board_img, self.agent)

        return obs, info
    
    '''
    Takes a step in the environment. Converts obs into a compressed space used by the wrapper.
    Also skips frames to avoid states where no change has occurred.
    '''
    def step(self, action):
        action = self.AGENT_INFO[self.agent]['action mapping'][action]
        img_obs, reward, termination, truncation, info = self.env.step(action)

        #converts obs to down-sampled obs
        obs = self.update_board(img_obs, self.agent)

        return obs, reward, termination, truncation, info
    
    def close(self):
        self.env.close()
    
    def observation_space(self, agent_id):
        return(self.observation_spaces[agent_id])
    
    def action_space(self, agent_id):
        return(self.action_spaces[agent_id])

    def _get_obs(self):
        img_obs, reward, termination, truncation, info = self.env._get_obs()
        obs = self.update_board(img_obs, self.agent)
        return obs

    def _get_info(self):
        return self.env._get_info()

    '''
    Converts rgb image to 38x18 grid world array.
    '''
    def update_board(self,board_img, agent_id):
        info = self.AGENT_INFO[agent_id]
        #gets rgb colours and corresponding integer value
        colours = np.array(info['colour mapping'])

        #crops the image to only contain the board
        px_board = board_img[self.BOARD_BOUNDARY[0] : self.BOARD_BOUNDARY[1], self.BOARD_BOUNDARY[2] : self.BOARD_BOUNDARY[3]]
        
        #down samples the board by the dimensions of the cells so each cell is one pixel
        rgb_board = px_board[::self.CELL_DIMENSION['height'], ::self.CELL_DIMENSION['width']]
        board = np.empty((rgb_board.shape[0],rgb_board.shape[1]),dtype=int)
        board[:] = -1
        #replaces each pixel (rgb colour) with coresponding integer from colour mapping
        for idx, colour in enumerate(colours):
            mask = np.all(rgb_board == colour, axis=-1)
            board[mask] = idx

        #flips board horizontally so policy can learn to control the left agent
        if info['flip']:
            np.flip(board, axis = 1)

        return(board[..., np.newaxis].astype(np.float32))