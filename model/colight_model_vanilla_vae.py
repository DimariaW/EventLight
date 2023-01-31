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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import parl
from parl.utils import logger
import math
from einops import reduce, rearrange

def generate_square_subsequent_mask(sz):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

def sinusoidal_init_(tensor):
    """
        tensor: (max_len, d_model)
    """
    max_len, d_model = tensor.shape
    position = rearrange(torch.arange(0.0, max_len), 's -> s 1')
    div_term = torch.exp(-math.log(10000.0) * torch.arange(0.0, d_model, 2.0) / d_model)
    tensor[:, 0::2] = torch.sin(position * div_term)
    tensor[:, 1::2] = torch.cos(position * div_term)
    return tensor

# Adapted from https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=False, initializer=None):
        super().__init__()
        self.batch_first = batch_first
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.empty(max_len, d_model)
        if initializer is None:
            sinusoidal_init_(pe)
            pe = rearrange(pe, 's d -> 1 s d' if self.batch_first else 's d -> s 1 d')
            self.register_buffer('pe', pe)
        else:
            hydra.utils.call(initializer, pe)
            pe = rearrange(pe, 's d -> 1 s d' if self.batch_first else 's d -> s 1 d')
            self.pe = nn.Parameter(pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim] if not batch_first else [B, S, D]
            output: [sequence length, batch size, embed dim] if not batch_first else [B, S, D]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x + (self.pe[:, :x.size(1)] if self.batch_first else self.pe[:x.size(0)])
        return self.dropout(x)

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)

        return x

class TransformerEncoderLayer(nn.Module):
  def __init__(self, embed_dim, num_heads=2,batch_first=True, dropout=0.1):
    super(TransformerEncoderLayer, self).__init__()
    self.slf_attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=batch_first, dropout=dropout)
    self.pos_ffn = PositionwiseFeedForward(embed_dim, d_hid=256, dropout=dropout)
    self.drop1 = nn.Dropout(dropout)
    self.drop2 = nn.Dropout(dropout)
    self.ln1 = nn.LayerNorm(embed_dim, eps=1e-6)
    self.ln2 = nn.LayerNorm(embed_dim, eps=1e-6)

  def forward(self, x, src_mask):
    att_out, attn_weights = self.slf_attn(x, x, x, attn_mask=src_mask)
    att_out = self.drop1(att_out)
    out_1 = self.ln1(x + att_out)
    ffn_out = self.pos_ffn(out_1)
    ffn_out = self.drop2(ffn_out)
    out = self.ln2(out_1 + ffn_out)
    return out

class ColightModel(parl.Model):
    def __init__(self, stacked_obs_act_dim, act_dim, edge_index=None, graph_layers=None):
        super(ColightModel, self).__init__()
        
        hidden_dim = 128
        num_heads = 4

        self.act_dim = act_dim
        self.obs_dim = 32
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # casual mask
        hist_len = stacked_obs_act_dim // (self.obs_dim + self.act_dim)
        self.src_mask = generate_square_subsequent_mask(hist_len).to(self.device)
        
        # encoder 
        self.pos_encoder = PositionalEncoding(self.obs_dim + self.act_dim, dropout=0.1, batch_first=True)
        self.transformers = nn.ModuleList([TransformerEncoderLayer(self.obs_dim + self.act_dim) for _ in range(3)])
        self.mu_head = nn.Sequential(
            nn.Linear(self.obs_dim + self.act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.obs_dim + self.act_dim)
        )
        self.log_std_head = nn.Sequential(
            nn.Linear(self.obs_dim + self.act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.obs_dim + self.act_dim)
        )
        # decoder
        self.dynamics = nn.Sequential(
            nn.Linear(2*(self.obs_dim + self.act_dim), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.obs_dim-self.act_dim)
        )

        # policy net
        self.obs_embedding = nn.Sequential(
            nn.Linear(self.obs_dim + self.act_dim + self.obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # self.graphs = nn.ModuleList([GATv2Conv(hidden_dim, hidden_dim // num_heads, num_heads) for _ in range(graph_layers)])
        # logger.info("graph layer:{}".format(len(self.graphs)))
        self.q_net = nn.Sequential(
            nn.Linear(hidden_dim, act_dim),
        )
        # self.edge_index = nn.parameter.Parameter(edge_index, requires_grad=False)
        self.reset_params()

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def attention(self, obs, aux_loss=False, add_noise=False, show_kl=True):
        # change obs shape from (b, n, hist_len*(obs_dim + act_dim)) to (b*n, hist_len*(obs_dim + act_dim))
        if obs.dim() == 3:
            batch_size, num_agents, _ = obs.size()
            obs = obs.view(batch_size*num_agents, -1)
        assert obs.shape[-1] % (self.act_dim + self.obs_dim) == 0
        hist_len = int(obs.shape[-1] / (self.act_dim + self.obs_dim))
        dim = self.obs_dim + self.act_dim
        batch = obs.shape[0]
        # encode
        transitions = obs.view(batch, hist_len, dim)
        hidden = transitions
        hidden = self.pos_encoder(hidden)
        for transformer in self.transformers:
          hidden = transformer(hidden, self.src_mask)
        mu = self.mu_head(hidden)
        log_std = self.log_std_head(hidden)
        std = torch.log(1+torch.exp(log_std))
        log_std = torch.log(std+1e-10)
        if add_noise:
            z = mu + torch.randn_like(mu)*torch.exp(log_std)
        else:
            z = mu
        
        # policy input
        # ret = z.reshape(batch, -1)
        ret = z[:, -1, :]
        current_obs = transitions[:, -1, :-self.act_dim]
        ret = torch.cat((ret, current_obs), -1)

        if not aux_loss:
            if show_kl:
                show_infos = {}
                current_mu, current_log_std, prev_mu, prev_log_std = mu[:, -1], log_std[:, -1], mu[:, -2], log_std[:, -2]
                current_var, prev_var = torch.exp(2*current_log_std), torch.exp(2*prev_log_std)
                kl_loss = 0.5*((current_var + torch.square(current_mu-prev_mu))/(prev_var+1e-10)) - current_log_std + prev_log_std
                show_infos["kl_divergence"] = kl_loss.sum(dim=-1).detach().cpu().numpy().tolist()           
                return ret, None, show_infos
            else:
                return ret, None, None
        
        aux_loss_infos = {}
        show_infos = {}
        # reconstruction loss
        if hist_len == 1:
            pred_loss = torch.Tensor(0).to(self.device)
            mape = torch.Tensor(0).to(self.device)
        else:
            # z_cat_x_a = torch.cat([z[:, 1:], transitions[:, :-1]], dim=-1)
            # pred_flow = self.dynamics(z_cat_x_a)
            # target_flow = transitions[:, 1:, :self.obs_dim-self.act_dim]
            # predict loss
            z_cat_x_a = torch.cat([torch.zeros_like(z[:, :-1]), transitions[:, :-1]], dim=-1)
            pred_flow = self.dynamics(z_cat_x_a)
            target_flow = transitions[:, 1:, :self.obs_dim-self.act_dim]
            pred_loss = 0.5 * torch.square(target_flow - pred_flow).sum(dim=-1).mean()
            mape = (torch.abs(target_flow - pred_flow) / torch.abs(target_flow + 1)).mean()
        aux_loss_infos["pred_loss"] = pred_loss
        show_infos["pred_loss"] = pred_loss.item()
        show_infos["mape"] = mape.item()
        # KL regularization
        prior_mu = torch.cat((torch.zeros_like(mu[:, :1]), mu[:, :-1]), dim=-2)
        prior_log_std = torch.cat((torch.zeros_like(log_std[:, :1]), log_std[:, :-1]), dim=-2)
        prior_var = torch.exp(2*prior_log_std)
        var = torch.exp(2*log_std)
        kl_loss = 0.5*((var + torch.square(mu-prior_mu))/(prior_var+1e-10)) - log_std + prior_log_std
        kl_loss = kl_loss.sum(dim=-1)
        show_infos["kl_loss_dist"] = kl_loss.detach().cpu().numpy()
        kl_loss = kl_loss.mean()*1
        aux_loss_infos["kl_loss"] = kl_loss
        show_infos["kl_loss"] = kl_loss.item()
        return ret, aux_loss_infos, show_infos

    def forward(self, obs, aux_loss=False, add_noise=False, show_kl=True):
        hidden, aux_loss_infos, show_infos = self.attention(obs, aux_loss, add_noise, show_kl)
        hidden = self.obs_embedding(hidden)
        q_val = self.q_net(hidden)
        if aux_loss:
          return q_val, aux_loss_infos, show_infos
        else:
          return q_val, show_infos

if __name__ == "__main__":
    model = ColightModel(400, 8).cuda()
    x = torch.randn(64, 16, 400).cuda()
    hidden = model(x)
    hidden, pred_loss, mape = model(x, aux_loss=True, add_noise=True)
    __import__("pdb").set_trace()
