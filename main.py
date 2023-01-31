#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import torch
import parl
from parl.utils import logger

import numpy as np
from tqdm import tqdm
from parl.utils import summary
from replay_buffer import ReplayMemory
from world import World
from obs_reward.colight_obs_reward import ColightGenerator
from config import colight_config as config
from environment import CityFlowEnv
from model.colight_model import ColightModel as CModel
from model.colight_model_vanilla_vae import ColightModel as CModelVAE
from model.colight_model_event_vae import ColightModel as CModelEventVAE
from ddqn import DDQN
from agent.agent import Agent
from eval_actor import EvalActor
from utils import create_env

GRAPH_LAYERS = 1
HISTORY_LENGTH = 10
SKIP_LENGTH = 10


def log_metrics(summary, learn_infos, step_forward):
    # logger.info(metric)
    for key, value in learn_infos.items():
        if key == 'train_count':
            continue
        if isinstance(value, np.ndarray):
            if step_forward >= 360*30:
                summary.add_histogram(key, value, step_forward)
        else:
            summary.add_scalar(key, value, step_forward)

def create_eval_actors(test_scenerios):
    eval_actors = []
    seeds = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    for i, path in enumerate(test_scenerios):
      eval_actors.append(EvalActor(config, path, GRAPH_LAYERS, HISTORY_LENGTH, SKIP_LENGTH, seed=seeds[i]))
    logger.info("{} actors created".format(len(eval_actors)))
    return eval_actors

def evaluate(eval_actors, agent):
  weights = agent.get_weights()
  for actor in eval_actors:
    actor.set_weights(weights).get()
  future_objs = [actor.run() for actor in eval_actors]
  return future_objs

def main(logdir):
    """
    all intersections share one model.
    """

    ####################

    logger.info('building the env...')

    train_path = [config["data_path"]] * 10
    env, env_info = create_env(config, train_path[0], HISTORY_LENGTH, SKIP_LENGTH)
    act_dims = env_info['act_dims']
    obs_dim = env_info['obs_dim']
    env = EvalActor(config, train_path[0], GRAPH_LAYERS, HISTORY_LENGTH, SKIP_LENGTH)
    eval_actors = create_eval_actors(train_path)
    obs = env.reset().get()
    print("OOOOOOOOOOOOOO:{}".format(obs_dim))
    print(f'obs shape: {obs_dim}')
    episode_count = 0
    step_forward = 0
    ####################
    print(f'action dim: {act_dims[0]}')
    n_agents = env_info['n_agents']
    replay_buffer = ReplayMemory(config['memory_size'], obs_dim, 0, n_agents)
    
    ###################
    edge_index = env_info['edge_index']
    graph_layers = 1

    model_type_map = {
        "CModel": CModel,
        "CModelVAE": CModelVAE,
        "CModelEvent": CModelEventVAE,
    }
    model = model_type_map[config["model_type"]](obs_dim, act_dims[0], edge_index, GRAPH_LAYERS)
    model.train()
    algorithm = DDQN(model, config)
    agent = Agent(algorithm, config)
    logger.info('successfully creating the agent...')
    ###################
    # train the model
    ###################
    future_objs = None
    episodes_rewards = np.zeros(n_agents)
    with tqdm(total=config['episodes'], desc='[Training Model]') as pbar:
        while episode_count <= config['episodes']:
            info = {'timeout': False}
            while info['timeout'] == False:
                actions = agent.sample(obs)
                next_obs, rewards, dones, info = env.step(actions).get()
                # calc the episodes_rewards and will add it to the tensorboard
                assert len(episodes_rewards) == len(rewards)
                episodes_rewards += rewards
                replay_buffer.append(obs, actions, rewards, next_obs, dones)
                step_forward += 1
                obs = next_obs
                if len(replay_buffer) >= config[
                        'begin_train_mmeory_size'] and step_forward % config[
                            'learn_freq'] == 0:
                    sample_data = replay_buffer.sample_batch(
                        config['sample_batch_size'])
                    train_obs, train_actions, train_rewards, train_next_obs, train_terminals = sample_data
                    train_actions = train_actions.reshape(-1)
                    train_rewards = train_rewards.reshape(-1)
                    train_terminals = train_terminals.reshape(-1)

                    learn_infos: dict = \
                        agent.learn(train_obs, train_actions, train_terminals, train_rewards, train_next_obs)
                   
                    # tensorboard
                    if learn_infos["train_count"] % config['train_count_log'] == 0:
                        log_metrics(summary, learn_infos, step_forward)
            
            avg_travel_time = info['average_travel_time']
            # just calc the first agent's rewards for show.
            summary.add_scalar('episodes_reward', episodes_rewards[0],
                               episode_count)
            # the avg travel time is same for all agents.
            summary.add_scalar('average_travel_time', avg_travel_time,
                               episode_count)
            logger.info('episode_count: {}, average_travel_time: {}.'.format(
                episode_count, avg_travel_time))
            # reset to zeros
            
            # save the last 10 model
            if episode_count >= config['episodes'] - 10:
                save_path = "{}/episode_count{}.ckpt".format(logger.get_dir(), episode_count)
                agent.save(save_path)

            if episode_count % config['test_freq'] == 0:
              if future_objs is not None:
                data = [future_obj.get() for future_obj in future_objs]
                eval_reward, eval_travel_time, eval_throughput, affected_vehs, event_detects= zip(*data)
                for test_id in range(len(eval_actors)):
                  summary.add_scalar('eval/reward_{}'.format(test_id), eval_reward[test_id], episode_count)
                  summary.add_scalar('eval/travel_time_{}'.format(test_id), eval_travel_time[test_id], episode_count)
                  summary.add_scalar('eval/throughput_{}'.format(test_id), eval_throughput[test_id], episode_count)
                  summary.add_scalar('eval/affected_vehs_{}'.format(test_id), affected_vehs[test_id], episode_count)
                summary.add_scalar('eval_reward', np.mean(eval_reward), episode_count)
                summary.add_scalar('eval_travel_time', np.mean(eval_travel_time), episode_count)
                summary.add_scalar('eval_throughput', np.mean(eval_throughput), episode_count)
                summary.add_scalar("affected_vehs", np.mean(affected_vehs), episode_count)
                logger.info('eval_travel_time: {}'.format(np.mean(eval_travel_time)))
                with open(f"{logdir}/kl_eval_episode_count_{episode_count}.json","w") as f:
                    json.dump(event_detects, f)
              future_objs = evaluate(eval_actors, agent)

            obs = env.reset().get()
            episodes_rewards = np.zeros(n_agents)
            episode_count += 1
            #replay_buffer.save('./rpm.npz')

            pbar.update(1)


if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cluster_ip', default='10.90.243.39', type=str, help="IP adress of the CPU cluster")
    parser.add_argument(
        '--aux_coef', default=0.0, type=float, help="coefficient for auxiliary loss")
    parser.add_argument(
        '--logdir', default="train_log", type=str, help="log directory")
    parser.add_argument(
        "--model_type", default="CModel", type=str, help="CModel, CModelVAE, CModelEventVAE"
    )
    parser.add_argument(
        "--data_path", default="./examples/config.json", type=str, help="data path"
    )
    args = parser.parse_args()
    config['aux_coef'] = args.aux_coef
    config["model_type"] = args.model_type
    config["data_path"] = args.data_path
    logger.set_dir('./train_log/{}/'.format(args.logdir))
    parl.connect("{}:8018".format(args.cluster_ip), \
        distributed_files=['./model/*.py', './agent/*.py', './obs_reward/*.py', './examples/*.json', './data'])
    os.mkdir('./train_log/{}/kl_eval'.format(args.logdir))
    main('./train_log/{}/kl_eval'.format(args.logdir))
