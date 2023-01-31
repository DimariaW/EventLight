#-*- coding: utf-8 -*-
#File: wrapper.py
import gym
import numpy as np
from collections import deque


class Timeout(gym.Wrapper):
  def __init__(self, env, max_timestep=3600):
    gym.Wrapper.__init__(self, env)
    self.max_timestep = max_timestep
    self.timestep = 0
    self.raw_env = env

  def step(self, action):
    self.timestep += 1
    obs, reward, done, info = self.env.step(action)
    info['timeout'] = False
    if self.timestep >= self.max_timestep:
      info['timeout'] = True
    return obs, reward, done, info

  def reset(self):
    self.timestep = 0
    return self.env.reset()

class EvalMetrics(gym.Wrapper):
  def __init__(self, env, number_of_cars):
    gym.Wrapper.__init__(self, env)
    self.raw_env = self.env.raw_env
    self.number_of_cars = number_of_cars

  def get_average_travel_time(self):
    return self.raw_env.world.eng.get_average_travel_time()

  def get_throughput(self):
    rest_cars = self.env.raw_env.world.eng.get_vehicles()
    throughput = self.number_of_cars - len(rest_cars)
    return throughput

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    if info['timeout'] == True:
      info['average_travel_time'] = self.get_average_travel_time()
      info['throughput'] = self.get_throughput()
    return obs, reward, done, info

  def reset(self):
    return self.env.reset()

class ActionRepeat(gym.Wrapper):
  def __init__(self, env, repeat_times=10):
    gym.Wrapper.__init__(self, env)
    self.raw_env = self.env.raw_env
    self.repeat_times = repeat_times

  def step(self, action):
    reward_list = []
    event_info = None
    for _ in range(self.repeat_times):
      obs, reward, done, info = self.env.step(action)
      if "event_info" in info:
        event_info = info["event_info"]
      reward_list.append(reward)
    reward = np.mean(reward_list, axis=0)
    if event_info is not None and "event_info" not in info:
      info["event_info"] = event_info
    return obs, reward, done, info

  def reset(self):
    return self.env.reset()

class CongestionEvent(gym.Wrapper):
  def __init__(self, env, max_congested_seconds=1800, difficulty=1, seed=None):
    gym.Wrapper.__init__(self, env)
    self.raw_env = self.env.raw_env
    self.max_congested_seconds = max_congested_seconds
    self.congested_road = None
    self.congested_seconds = 0
    self.difficulty = difficulty
    self.seed = seed
    self.congested = False
    self.congested_flow = set()

  def reset_event(self):
    if self.congested_road is not None:
      for road in self.congested_road:
        self.raw_env.world.eng.set_road_max_speed(road, 11.111)
    self.congested_road = None
    self.congested_seconds = 0

  def event(self):
    if self.congested_road is not None:
      for road in self.congested_road:
        self.raw_env.world.eng.set_road_max_speed(road, 11.111)

    event_info = {"intersections":[]}
    self.congested_seconds = 0
    congested_road_id = np.random.randint(len(self.all_roads), size=self.difficulty)
    self.congested_road = [self.all_roads[road_id] for road_id in congested_road_id]
    self.congested = False
    for road in self.congested_road:
      if np.random.random() < 0.5:
        self.raw_env.world.eng.set_road_max_speed(road, 2.0)
        self.congested = True
        event_info["intersections"].extend(self.road_id_to_intersection_index[road])
    return event_info


  def route_replanning(self):
        # 20% vehicles change their routes due to the unexpected events
        assert self.difficulty == 1 # note this function only supports difficulty==1
        vehicle_ids = self.raw_env.world.eng.get_vehicles()
        for vehicle in vehicle_ids:
            if vehicle in self.congested_flow: 
              continue
            vehicle_info = self.raw_env.world.eng.get_vehicle_info(vehicle)
            if vehicle_info["running"] == False:
              continue
            route = vehicle_info["route"].split(' ')
            if self.congested_road[0] in route[1: ]:
                self.congested_flow.add(vehicle)
                new_route = route[1:] # from str to list
                new_route.remove(self.congested_road[0])
                if np.random.random() < 0.2:
                    self.raw_env.world.eng.set_vehicle_route(vehicle, new_route, self.congested_road[0])

  def step(self, action):
    self.congested_seconds += 1
    info = {}
    if (self.congested_seconds >= self.max_congested_seconds) or (self.congested_road is None):
      event_info = self.event()
      info["event_info"] = event_info
    if self.congested:
        self.route_replanning()
    info["affected_vehs"] = len(self.congested_flow)
    obs, reward, done, _info = self.env.step(action)
    info.update(_info)
    return obs, reward, done, info

  def reset(self):
    if self.seed is not None:
      np.random.seed(self.seed)
    self.reset_event()
    obs = self.env.reset()
    self.congested = False
    self.congested_flow = set()
    return obs

class FrameStack(gym.Wrapper):
  def __init__(self, env, hist_len):
    gym.Wrapper.__init__(self, env)
    self.raw_env = self.env.raw_env
    self.hist_len = hist_len
    self.obs_stack = deque(maxlen=hist_len-1)
    self.act_stack = deque(maxlen=hist_len-1)
    self.n_agents = self.raw_env.n_agents

  def _cat_obs(self, obs):
    ret_obs = []
    for state, action in zip(self.obs_stack, self.act_stack):
      ret_obs.extend([state, action])
    ret_obs.append(obs)
    ret_obs.append(np.zeros((self.n_agents, 8)))
    ret_obs = np.concatenate(ret_obs, -1)
    return ret_obs

  def step(self, action):
    onehot_actions = np.zeros((self.n_agents, 8))
    onehot_actions[np.arange(self.n_agents), action] = 1
    self.act_stack.append(onehot_actions)
    next_obs, reward, done, info = self.env.step(action)
    ret_obs = self._cat_obs(next_obs)
    self.obs_stack.append(next_obs)
    return ret_obs, reward, done, info

  def reset(self):
    obs = self.env.reset()
    self.obs_stack.extend([np.zeros_like(obs) for _ in range(self.hist_len)])
    self.act_stack.extend([np.zeros((self.n_agents, 8)) for _ in range(self.hist_len)])
    ret_obs = self._cat_obs(obs)
    self.obs_stack.append(obs)
    return ret_obs