#-*- coding: utf-8 -*-
#File: utils.py
#from obs_reward.presslight_obs_reward import PressureLightGenerator
from operator import truediv
import parl
from obs_reward.colight_obs_reward import ColightGenerator
from environment import CityFlowEnv
from world import World
import numpy as np
import json
import torch
from wrapper import CongestionEvent, FrameStack, ActionRepeat, Timeout, EvalMetrics
import env_utils

def create_env(config, scenario, hist_len, skip_len=1, seed=None):
    world = World(
        scenario,
        thread_num=config['thread_num'],
        yellow_phase_time=config['yellow_phase_time'])
    PLGenerator = ColightGenerator(world, config['obs_fns'],
                                         config['reward_fns'],
                                         config['in_only'], config['average'])
    obs_dims = PLGenerator.obs_dims
    number_of_cars = env_utils.number_of_cars(scenario)
    env = CityFlowEnv(world, PLGenerator)
    act_dims = env.action_dims
    n_agents = env.n_agents
    env = Timeout(env, max_timestep=3600)
    env = EvalMetrics(env, number_of_cars)
    env = CongestionEvent(env, difficulty=1, seed=seed)
    env = ActionRepeat(env, repeat_times=skip_len)
    env = FrameStack(env, hist_len)
    env_info = dict()
    #env_info['obs_dims'] = obs_dims
    env_info['act_dims'] = act_dims
    env_info['n_agents'] = n_agents
    env_info['edge_index'] = get_edge_index(config, scenario)
    obs = env.reset()
    env_info['obs_dim'] = obs.shape[-1]
    return env, env_info

class MultiEnv(object):
  def __init__(self, config, data_path, hist_len, skip_len):
    self.env_num = len(data_path)
    self.envs = []
    self.obs_dims = None
    for path in data_path:
      env, obs_dims = create_env(config, path, hist_len, skip_len)
      self.envs.append(env)
      self.obs_dims = obs_dims
      self.env = None
      self.action_dims = env.action_dims
      self.n_agents = env.n_agents

  def get_average_travel_time(self):
    return self.env.world.eng.get_average_travel_time()

  def step(self, action):
    return self.env.step(action)

  def reset(self):
    idx = np.random.randint(self.env_num)
    self.env = self.envs[idx]
    return self.env.reset()

def create_multi_env(config, data_path, hist_len, skip_len=1):
    env = MultiEnv(config, data_path, hist_len, skip_len)
    return env, env.obs_dims

def get_edge_index(config, scneario):
    #read intersections' adjacency information from road net file

    with open(scneario, "r") as f:
        config_dict = json.load(f)
        roadfile = config_dict["dir"] + config_dict["roadnetFile"]
    
    with open(roadfile, "r") as f:
        roadnet_dict = json.load(f)

    invalid_roads = []
    net_node_dict_inter2id = {}
    node_adjacent_node_matrix = []

    cur_num = 0
    for node_dict in roadnet_dict["intersections"]:
        if node_dict["virtual"]:
            for i in node_dict["roads"]:
                invalid_roads.append(i)
            continue
        node_id = node_dict["id"]
        net_node_dict_inter2id[node_id] = cur_num
        cur_num +=1


    for node_dict in roadnet_dict["intersections"]:
        if node_dict["virtual"]:
            continue        
        node_id = node_dict["id"]
        road_links = node_dict['roads']
        input_nodes = [] #should be node_degree
        for road_link_id in road_links:
            for item in roadnet_dict['roads']:
                if item['id'] == road_link_id:
                    road_link_dict = item
                    break
            if road_link_dict['endIntersection'] == node_id:
                start_node = road_link_dict['startIntersection']
                if start_node in net_node_dict_inter2id.keys():
                    start_node_id = net_node_dict_inter2id[start_node]
                    input_nodes.append(start_node_id)

        node_adjacent_node_matrix.append(input_nodes)

    edge_index = []
    for road_id,  adjacent_road_ids in enumerate(node_adjacent_node_matrix):
        edge_index.append([road_id, road_id])
        for adjacent_road_id in adjacent_road_ids:
            edge_index.append([adjacent_road_id, road_id])
            
    edge_index = torch.LongTensor(edge_index).transpose(0, 1)

    print("edge_index_shape", edge_index.shape)
    print("edge_index", edge_index)
    return edge_index

if __name__ == '__main__':
    from config import colight_config as config
    import parl
    env, env_info = create_env(config, scenario='./examples/hz.json', hist_len=10, skip_len=10)
    
    obs = env.reset()
    # __import__("pdb").set_trace()
    import tqdm
    bar = tqdm.tqdm()
    while True:
        bar.update()
        actions = np.random.randint(0, 8, size=16)
        obs, reward, done, info = env.step(actions)
        if "event_info" in info:
            print(info)
            __import__("pdb").set_trace()
        if info["timeout"] == True:
            print(info)
            break
