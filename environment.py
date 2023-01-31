# Third party code
#
# The following code are copied or modified from:
# https://github.com/gjzheng93/tlc-baseline and https://github.com/zhc134/tlc-baselines

import gym
import numpy as np
import cityflow
from parl.utils import logger


class CityFlowEnv(gym.Env):
    """
    Environment for Traffic Signal Control task.

    Parameters
    ----------
    world: World object
    obs_reward_generator(object): generator of the obs and rewards
    """

    def __init__(self, world, obs_reward_generator):
        self.world = world
        self.n_agents = len(self.world.intersection_ids)
        self.n = self.n_agents
        # agents action space dim, each roadnet file may have different action dims
        self.action_dims = []
        self.intersection_to_index = {}
        self.road_id_to_intersection_index = {}
        for index, i in enumerate(self.world.intersections):
            self.action_dims.append(len(i.phases))
            self.intersection_to_index[i.id] = index

        self.action_space = gym.spaces.MultiDiscrete(self.action_dims)
        self.obs_reward_generator = obs_reward_generator
        self.all_roads = [road['id'] for road in self.world.roadnet['roads']]
        for road in self.world.roadnet["roads"]:
            road_id, start_intersection, end_intersection = road["id"], road["startIntersection"], road["endIntersection"] 
            self.road_id_to_intersection_index[road_id]=[]
            if start_intersection in self.intersection_to_index:
                self.road_id_to_intersection_index[road_id].append(self.intersection_to_index[start_intersection])
            if end_intersection in self.intersection_to_index:
                self.road_id_to_intersection_index[road_id].append(self.intersection_to_index[end_intersection])
            

    def step(self, actions):
        """
        actions: list
        """
        self.world.step(actions)
        next_obs = self.obs_reward_generator.generate_obs()
        rewards = self.obs_reward_generator.generate_reward()
        dones = [False] * self.n_agents
        infos = {}
        return next_obs, rewards, dones, infos

    def reset(self, seed=False):
        self.world.reset(seed)
        obs = self.obs_reward_generator.generate_obs()
        return obs



if __name__ == "__main__":

    from world import World
    from presslight_obs_reward import PressureLightGenerator
    world = World("./examples/config_n.json", thread_num=1)
    PressureLightGenerator = PressureLightGenerator(world, ["lane_count"],
                                                    ["pressure"], False, None)
    env = CityFlowEnv(world, PressureLightGenerator)
    actions = [0 for _ in range(env.n_agents)]
    for _ in range(200):
        obs, rewards, dones, infos = env.step(actions)
        print(obs, rewards)
        print(env.action_space)
        __import__('ipdb').set_trace()
