import numpy as np
from ple.games.catcher import Catcher
from ple import PLE
import os

class Catcher_Env:
    def __init__(self,
                 random_seed=0,
                 init_lives=3, 
                 normalise=True, 
                 display=False):
        
        self._random_seed = random_seed
        self._game = Catcher(init_lives=init_lives)
        self._normalise = normalise
        self._display = display

        if self._display == False: 
            os.putenv('SDL_VIDEODRIVER', 'fbcon')
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        if self._normalise:
            self._env = PLE(self._game, fps=30, state_preprocessor=self._normalise_ob, display_screen=display)
        else:
            self._env = PLE(self._game, fps=30, state_preprocessor=self._ob, display_screen=display)

        self._env.init()
        self._actions = self._env.getActionSet()
        self._env.rng.seed(random_seed)

        # Tracker
        self._cum_reward = 0

    def _ob(self, state):
        return np.array([state['player_x'], state['player_vel'], state['fruit_x'], state['fruit_y']])

    def _normalise_ob(self, state):
        state = np.array([state['player_x'], state['player_vel'], state['fruit_x'], state['fruit_y']])
        state[0] = (state[0] - 26) / 26 # makes range -1 1
        state[1] = (state[1]) / 8 # makes range -1 1
        state[2] = (state[2] - 26) / 26 # makes range -1 1
        state[3] = (state[3] - 20) / 45 # makes range -1 1

        return state

    def reset(self):
        self._cum_reward = 0
        self._env.reset_game()
        
        return self._env.getGameState()

    def action_set(self):
        return self._actions

    def num_actions(self):
        return len(self._actions)

    def episode_return(self):
        return self._cum_reward

    def act(self, a):
        reward = self._env.act(self._actions[a])
        if reward == -6:
            reward = -1

        self._cum_reward += reward

        next_obs = self._env.getGameState()
        terminal = self._env.game_over()
        if self._cum_reward >= 200:
            self._cum_reward = 200
            terminal = True
        return  reward, next_obs, terminal