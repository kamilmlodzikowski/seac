import sys

sys.path.append("/home/kami/Documents/RLMAPF")

from rlmapf import RLMAPF
import gymnasium as gym
import numpy as np




class GymRLMAPF(gym.Env):
    def __init__(self, env_config):
        self.env = RLMAPF(env_config)
        self.n_agents = len(self.env.get_agent_ids())
        self.action_space = gym.spaces.Tuple([self.env.action_space for _ in range(self.n_agents)])
        self.observation_space = self.get_observation_space()
        self._seed = self.env.get_seed()
        
    
    def get_observation_space(self):
        obs_space = gym.spaces.Tuple([gym.spaces.flatten_space(self.env.observation_space) for _ in range(self.n_agents)])
        return obs_space

    def seed(self, seed=None):
        self.env.set_seed(seed)
        return self._seed
    
    def flatten_observation(self, obs):
        obs = [gym.spaces.flatten(self.env.observation_space, o) for o in obs.values()]
        print(obs)
        return obs

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self.seed(seed)
        obs, info = self.env.reset()
        # flatten the observation space dictionary
        obs = self.flatten_observation(obs)
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self.flatten_observation(obs)
        return obs, reward, terminated, truncated, info
    
    def render(self):
        self.env.render()

    def get_agent_ids(self):
        return self.env.get_agent_ids()


# Register the environment in gymnasium
gym.register(
    id="RLMAPF",
    entry_point="rlmapf_wrapper:GymRLMAPF",
    kwargs={"env_config":{
        "agents_num": 2,
        "render_mode": "human",
        "render_delay": 1,
        "max_steps": 1000,
        "observation_type": "position",
        "map_path": "/home/kami/Documents/RLMAPF/maps/",
        "maps_names_with_variants": {
            "empty_1-4a-5x4": None,
        }
    }},
)

if __name__ == "__main__":
    env_config = {
        "agents_num": 2,
        "render_mode": "human",
        "render_delay": 1,
        "max_steps": 1000,
        "observation_type": "position",
        "map_path": "/home/kami/Documents/RLMAPF/maps/",
        "maps_names_with_variants": {
            "empty_1-4a-5x4": None,
        }
    }

    env = GymRLMAPF(env_config=env_config)

    for ep in range(5):
        _ = env.reset()
        actions = {agent: env.action_space.sample() for agent in env.get_agent_ids()}

        _ = env.step(actions)
        env.render()
        print("--- Episode Finished ---")

