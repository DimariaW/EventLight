#-*- coding: utf-8 -*-
#File: eval_actor.py
import parl
from utils import create_env
from model.colight_model import ColightModel as CModel
from model.colight_model_vanilla_vae import ColightModel as CModelVAE
from model.colight_model_event_vae import ColightModel as CModelEventVAE
from ddqn import DDQN
from agent.agent import Agent
import numpy as np
import env_utils

@parl.remote_class(wait=False)
class EvalActor(object):
  def __init__(self, config, scenario, graph_layers=1, hist_len=1, skip_len=1, seed=None):
    env, env_info = create_env(config, scenario, hist_len, skip_len, seed)
    act_dims = env_info['act_dims']
    n_agents = env_info['n_agents']
    obs_dim = env_info['obs_dim']
    self.config = config
    self.env = env
    self.n_agents = n_agents
    edge_index = env_info['edge_index']
    model_type_map = {
        "CModel": CModel,
        "CModelVAE": CModelVAE,
        "CModelEvent": CModelEventVAE,
    }
    model = model_type_map[config["model_type"]](obs_dim, act_dims[0], edge_index, graph_layers)
   
    model.eval()
    algorithm = DDQN(model, config)
    self.agent = Agent(algorithm, config)

  def set_weights(self, weights):
      self.agent.set_weights(weights)

  def reset(self):
    return self.env.reset()
  
  def step(self, action, repeat_times=1):
    for i in range(repeat_times):
      next_obs, rewards, dones, info = self.env.step(action)
    return next_obs, rewards, dones, info

  def get_average_travel_time(self):
    return self.env.world.eng.get_average_travel_time()

  def run(self, eval_times=1):
    mean_reward = []
    mean_travel_time = []
    mean_throughput = []
    for eval_count in range(eval_times):
      episodes_rewards = np.zeros(self.n_agents)
      obs = self.env.reset()
      info = {'timeout': False}
      event_infos=[]
      kl_divergences = []
      event_probs = []
      while info['timeout'] == False:
          actions, show_infos = self.agent.predict(obs)
          kl_divergences.append(show_infos["kl_divergence"])
          if "event_prob" in show_infos:
            event_probs.append(show_infos["event_prob"])
          rewards_list = []
          next_obs, rewards, dones, info = self.env.step(actions)
          if "event_info" in info:
            info["event_info"]["step"] = len(kl_divergences)
            event_infos.append(info["event_info"])
          # calc the episodes_rewards and will add it to the tensorboard
          assert len(episodes_rewards) == len(rewards)
          episodes_rewards += rewards
          obs = next_obs
      avg_travel_time = info['average_travel_time']
      throughput = info['throughput']
      affected_vehs = info["affected_vehs"] # if "affected_vehs" in info else 0
      mean_reward.append(episodes_rewards[0]) # compute the evarage reward of the first agent
      mean_travel_time.append(avg_travel_time)
      mean_throughput.append(throughput)
      
    return np.mean(mean_reward), np.mean(mean_travel_time), np.mean(mean_throughput), affected_vehs, {"event_infos": event_infos, "kl_divergences": kl_divergences, "event_probs":event_probs}
