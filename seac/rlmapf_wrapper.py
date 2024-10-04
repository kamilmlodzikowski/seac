import sys

sys.path.append("/home/kami/Documents/RLMAPF")

from rlmapf import RLMAPF
import gymnasium as gym
import numpy as np

# From Ray multi-agent env to gymnasium env

class GymRLMAPF(gym.Env):
    def __init__(self, env_config):
        self.env = RLMAPF(env_config)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        return self.env.step(action)
    
    def render(self):
        self.env.render()

    def get_agent_ids(self):
        return self.env.get_agent_ids()


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