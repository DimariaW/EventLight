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

import torch
import parl
import numpy as np


class Agent(parl.Agent):
    def __init__(self, algorithm, config):
        super(Agent, self).__init__(algorithm)
        self.device = torch.device("cuda" if torch.cuda.
                                   is_available() else "cpu")
        self.config = config
        self.epsilon = self.config['epsilon']

    def sample(self, obs):
        # The epsilon-greedy action selector.
        def sample_random(act_dim):
            # Random samples
            return np.random.randint(0, act_dim)

        obs = torch.FloatTensor(obs).to(self.device)
        logits = self.alg.sample(obs)
        act_dim = logits.shape[-1]
        act_values = logits.cpu().detach().numpy()
        actions = np.argmax(act_values, axis=-1)
        for i in range(obs.shape[0]):
            if np.random.rand() <= self.epsilon:
                actions[i] = sample_random(act_dim)
        return actions

    def predict(self, obs):

        obs = torch.FloatTensor(obs).to(self.device)
        predict_actions, show_infos = self.alg.predict(obs)
        return predict_actions, show_infos

    def learn(self, obs, actions, dones, rewards, next_obs):

        obs = torch.FloatTensor(obs).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)

        learn_infos = self.alg.learn(obs, actions, dones, rewards, next_obs)

        self.alg.sync_target(decay=self.config['decay'])
        self.epsilon = learn_infos["epsilon"]

        return learn_infos
