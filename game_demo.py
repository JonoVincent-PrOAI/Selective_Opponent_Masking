import torch
import pygame
import pygame_menu
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from ray.rllib.core.rl_module.rl_module import RLModule
import os
import sys
from pettingzoo.atari import surround_v2
import gymnasium as gym
import numpy as np
import math
import csv
if os.path.abspath("../") not in sys.path:
    sys.path.append(os.path.abspath("../"))
if os.path.abspath("./") not in sys.path:
    sys.path.append(os.path.abspath("./"))

from utils.surround_v2_wrapper import Surround_v2_Wrapper

class GameDemo():

    def __init__(self, scale, model_dir = None, log_file = None):
        pygame.mixer.init()
        pygame.init()

        if model_dir == None: 
            self.model_dir = os.path.abspath("./ray_results/PPO_surround_v2/run-5")
        else:
            self.model_dir = os.path.abspath(model_dir)

        if log_file == None:
            self.log_file = 'test.csv'
        else:
            self.log_file = log_file

        self.settings = {'FPS':3,
                         'turn_based':False, 
                         'music_volume':0.5, 
                         'fx_volume':0.25, 
                         'scale':scale,
                         'agent_human': {'first_0': True, 'second_0':False},
                         'p1_id': 'anon',
                         'p2_id': 'anon',
                         }

        self.move_fx = pygame.mixer.Sound('demo_files/sfx/move.wav')
        self.move_fx.set_volume(0.25)

        self.img_obs_width = 160
        self.img_obs_height = 210

        self.font = pygame_menu.font.FONT_MUNRO
        self.main_menu_theme = pygame_menu.Theme(
            title_font=self.font, title_background_color = (184,50,50), 
            title_bar_style = pygame_menu.widgets.MENUBAR_STYLE_SIMPLE,
            title_font_size = 100, title_offset = (50,0),
            widget_font = self.font, background_color=(227,151,89),
            )

        self.settings_toggle = False
        self.screen = pygame.display.set_mode(((160 * scale[0]), (210 * scale[1])), pygame.RESIZABLE)
        self.main_menu()
        self.menu.mainloop(self.screen)

    def main_menu(self):
        self.settings_toggle = False
        pygame.mixer.music.load('demo_files/sfx/music_loop.mp3')
        pygame.mixer.music.play(-1,0.0)
        pygame.mixer.music.set_volume(self.settings['music_volume'])

        self.settings['agent_human']['first_0'] = True
        self.settings['agent_human']['second_0'] = False

        w, h = pygame.display.get_surface().get_size()
        menu = pygame_menu.Menu('Surround_AI', w, h, theme= self.main_menu_theme)
        menu.add.button('Play', self.go_to_log)
        menu.add.selector('Player 1:', [('Human', 1), ('AI', 2)], onchange=self.set_p1_mapping)
        menu.add.selector('Player 2:', [('AI', 1), ('Human', 2)], onchange=self.set_p2_mapping)
        menu.add.button('Quit', pygame_menu.events.EXIT)
        menu.add.button('Settings', self.add_settings)
        
        self.menu = menu

    def log_in_menu(self):
        p1_id_input = None
        p2_id_input = None

        def log_ids():
            if self.settings['agent_human']['first_0']:
                self.settings['p1_id'] = p1_id_input.get_value()
            else:
                self.settings['p1_id'] = 'ai'
            if self.settings['agent_human']['second_0']:
                self.settings['p2_id'] = p2_id_input.get_value()
            else:
                self.settings['p2_id'] = 'ai'
            self.start_game()

        w, h = pygame.display.get_surface().get_size()
        menu = pygame_menu.Menu('Enter Player IDs', w, h, theme = self.main_menu_theme)
        if self.settings['agent_human']['first_0']:
            p1_id_input = menu.add.text_input(title="Player-1 ID: ", textinput_id="p1_id")
        if self.settings['agent_human']['second_0']:
            p2_id_input = menu.add.text_input(title="Player-2 ID: ", textinput_id="p2_id")
        menu.add.button('Start', log_ids)
        menu.add.button('Back', self.go_to_main)
        self.menu = menu

    def pause_menu(self):
        self.settings_toggle = False
        w, h = pygame.display.get_surface().get_size()
        menu = pygame_menu.Menu('Pause', w, h, theme=self.main_menu_theme)
        menu.add.button('Resume', menu.disable)
        menu.add.button('Restart', self.start_game)
        menu.add.button('Quit to Menu', self.go_to_main)
        menu.add.button('Quit', pygame_menu.events.EXIT)
        menu.add.button('Settings', self.add_settings)

        self.menu = menu


    def update_music_volume(self, value):
        self.settings['music_volume'] = value/100
        pygame.mixer.music.set_volume(self.settings['music_volume'])

    def update_fx_volume(self, value):
        self.settings['fx_volume'] = value/100
        self.move_fx.set_volume(self.settings['fx_volume'])

    def update_speed(self, value):
        self.settings['FPS'] = value

    def update_turn_based(self, value):
        self.settings['turn_based'] = value

    def go_to_main(self):
        pygame_menu.events.EXIT
        self.main_menu()
        self.menu.mainloop(self.screen)

    def go_to_log(self):
        pygame_menu.events.EXIT
        self.log_in_menu()
        self.menu.mainloop(self.screen)

    def add_settings(self):
        if not self.settings_toggle:
            self.menu.add.range_slider(title="Music", default=(self.settings['music_volume'] *100), 
                                       range_values=(0, 100), increment=1, value_format=lambda x: str(int(x)), 
                                       rangeslider_id="music", onchange = self.update_music_volume,)
            self.menu.add.range_slider(title="Sound Effects", default=(self.settings['fx_volume'] * 100), 
                                       range_values=(0, 100), increment=1, value_format=lambda x: str(int(x)), 
                                       rangeslider_id="fx", onchange = self.update_fx_volume) 
            self.menu.add.range_slider(title="Game Speed", default=self.settings['FPS'], range_values=(1, 10), 
                                       increment=1, value_format=lambda x: str(int(x)), 
                                       rangeslider_id="spd", onchange = self.update_speed)
            self.menu.add.toggle_switch(title="Trun Based", default=self.settings['turn_based'], 
                                       onchange= self.update_turn_based)
        else:
            self.menu.remove_widget('music')
            self.menu.remove_widget('fx')
            self.menu.remove_widget('spd')
        self.settings_toggle = not(self.settings_toggle)

    def set_p1_mapping(self, value, select):
        if select == 2:
            self.settings['agent_human']['first_0'] = False
        else:
            self.settings['agent_human']['first_0'] = True

    def set_p2_mapping(self, value, select):
        if select == 2:
            self.settings['agent_human']['second_0'] = True
        else:
            self.settings['agent_human']['second_0'] = False


    def start_game(self):
        
        clock = pygame.time.Clock()
        running = True

        env = self.instance_model_env()
        obs, _ = env.reset()

        main_model = self.instance_model()
        img_obs = env.get_unwrapped_obs()['first_0']
        img_obs = self.convert_obs(img_obs)

        game_results = {'first_0' : 0, 'second_0': 0}

        while running:

            clock.tick(self.settings['FPS'])
            actions = {}
            human_actions = {'first_0':0, 'second_0':0}
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    self.screen.blit(pygame.transform.scale(self.screen, event.dict['size']), (0, 0))
                    w, h = pygame.display.get_surface().get_size()
                    self.scale = [w/self.img_obs_width, h/self.img_obs_height]
                elif event.type == pygame.KEYDOWN:#or event.type == pygame.KEYUP:

                    if event.key == pygame.K_UP:
                        human_actions['first_0'] = 1
                    if event.key == pygame.K_RIGHT:
                        human_actions['first_0'] = 2
                    if event.key == pygame.K_LEFT:
                        human_actions['first_0'] = 3
                    if event.key == pygame.K_DOWN:
                        human_actions['first_0'] = 4

                    if event.key == pygame.K_w:
                        human_actions['second_0'] = 1
                    if event.key == pygame.K_d:
                        human_actions['second_0']  = 2
                    if event.key == pygame.K_a:
                        human_actions['second_0'] = 3
                    if event.key == pygame.K_s:
                        human_actions['second_0']  = 4

                    if event.key == pygame.K_p:
                        self.pause_menu()
                        self.menu.mainloop(self.screen)

            for agent_id, agent_obs in obs.items():
                
                if not(self.settings['agent_human'][agent_id]):
                    obs_tensor = torch.from_numpy(agent_obs).unsqueeze(0)

                    with torch.no_grad():
                        output = main_model.forward_inference(
                            {"obs": obs_tensor}
                        )
                    
                    logits = output["action_dist_inputs"]
                    action = torch.argmax(logits, dim=-1).item()
                    actions[agent_id] = action
                else:
                    actions[agent_id] = human_actions[agent_id]

            human_acted = True
            if self.settings['turn_based']:
                for agent_id, agent_obs in obs.items():
                    if self.settings['agent_human'][agent_id]:
                        if actions[agent_id] == 0:
                            human_acted = False
            
            if (human_acted):
                obs, rewards, terminations, truncations, infos = env.step(actions)

                for agent_id, agent_obs in obs.items():
                    if rewards[agent_id] == 1:
                            game_results[agent_id] += 1
                img_obs = env.get_unwrapped_obs()['first_0']
                img_obs = self.convert_obs(img_obs)


                running = (not(all(terminations.values()) or all(truncations.values()))) and running
                
                pygame.surfarray.blit_array(self.screen, img_obs)
                pygame.display.update()
                self.move_fx.play()

        self.log_game(game_results)
        self.main_menu()
        self.menu.mainloop(self.screen)

    def log_game(self, game_results):
        game_results['draw'] = 10 - (max(game_results.values())) 
        log = self.settings
        log['result'] = game_results
        with open(self.log_file, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow(log.values())

        


    def convert_obs(self, img_obs):
        w, h = pygame.display.get_surface().get_size()
        self.scale = [w/self.img_obs_width, h/self.img_obs_height]
        img_obs = img_obs.swapaxes(0,1)
        img_obs = img_obs.repeat(self.scale[0],axis=0).repeat(self.scale[1],axis=1)
        width_diff = w - img_obs.shape[0]
        height_diff = h -  img_obs.shape[1]
        padding = ((math.floor(width_diff/2), math.ceil(width_diff/2)), (math.floor(height_diff/2), math.ceil(height_diff/2)), (0,0))
        img_obs = np.pad(img_obs, padding, 'edge')
        return(img_obs)

    def instance_model_env(self):

        def env_creator(config):
            env = Surround_v2_Wrapper(
                surround_v2.parallel_env(
                obs_type="rgb_image",
                full_action_space=False,
                max_cycles=15000,
                frame_skip = 0,
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

        env = Surround_v2_Wrapper(
            surround_v2.parallel_env(
            obs_type="rgb_image",
            full_action_space=False,
            max_cycles=15000,
        )
        )
        env.reset()
        return(env)

    def instance_model(self):
        main_module_path = os.path.join(
            self.model_dir,
            "learner_group",
            "learner",
            "rl_module",
            "main"
        )

        main_model = RLModule.from_checkpoint(main_module_path)
        return(main_model)
    

