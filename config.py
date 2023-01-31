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


colight_config = {

    #==========  env config ==========
    'thread_num': 1,
    'obs_fns': ['lane_count'],
    'reward_fns': ['pressure'],
    'in_only': False,
    'average': None,
    'action_interval': 10,
    'metric_period': 3600,  #3600
    'yellow_phase_time': 5,
    'yellow_phase_id': -1, # add yellow phase to action

    #==========  learner config ==========
    'gamma': 0.8,  # also can be set to 0.95
    'epsilon': 0.9,
    'epsilon_min': 0.1,
    'epsilon_decay': 0.9995,
    'start_lr': 0.00025,
    'episodes': 500,
    'algo': 'DQN',  # DQN
    'max_train_steps': int(1e6),
    'lr_decay_interval': 100,
    'epsilon_decay_interval': 1,
    'sample_batch_size': 128,  # also can be set to 32, which doesn't matter much.
    'learn_freq': 2,  # update parameters every 2 or 5 steps
    'test_freq': 2,  # test model every 5 episodes
    'decay': 0.995,  # soft update of double DQN
    'reward_normal_factor': 4,  # rescale the rewards, also can be set to 20,
    'train_count_log': 5,  # add to the tensorboard
    'is_show_log': False,  # print in the screen
    'step_count_log': 1000,

    # save checkpoint frequent episode
    'save_rate': 50,
    'save_dir': './save_model/presslight',
    'train_log_dir': './train_log/presslight',

    # memory config
    'memory_size': 10000,
    'begin_train_mmeory_size': int(1000)
}
