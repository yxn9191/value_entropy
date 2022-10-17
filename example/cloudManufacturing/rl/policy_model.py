import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from gym.spaces import Box, Dict
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import override


_MASK_NAME = "action_mask"
_OTHER_NAME = "others"


def apply_logit_mask(logits, mask):
    """Mask values of 1 are valid actions."
    " Add huge negative values to logits with 0 mask values."""
    logit_mask = torch.ones_like(logits) * -10000000
    logit_mask = logit_mask * (1 - mask)

    return logits + logit_mask


def attention(self_input, other_inputs):
    alpha = [torch.dot(self_input, other_inputs[i]) / math.sqrt(self_input.shape.size) for i in
             range(other_inputs.shape[1])]
    bate = [torch.exp(alpha[i]) / sum([torch.exp(alpha)]) for i in range(other_inputs.shape[0])]
    c_i = sum([torch.dot(bate[i], other_inputs[i]) for i in range(other_inputs.shape[0])])
    return c_i




class Attn(nn.Module):
    def __init__(self, query_size, key_size, value_size1, value_size2, output_size):
        """初始化函数中的参数有5个，query_size代表query的最后一维大小
           key_size代表key的最后一维大小，value_size1代表value的倒数第二维大小，
           value = (1, value_size1, value_size2)
           value_size2代表value的倒数第一维大小, output_size输出的最后一维大小"""
        super(Attn, self).__init__()
        # 将参数传入类中
        self.query_size = query_size
        self.key_size = key_size
        self.value_size1 = value_size1
        self.value_size2 = value_size2
        self.output_size = output_size

        # 初始化注意力机制实现第一步中需要的线性层
        self.attn = nn.Linear(self.query_size + self.key_size, value_size1)

        # 初始化注意力机制实现第三步中需要的线性层
        self.attn_combine = nn.Linear(self.query_size + value_size2, output_size)

    def forward(self, Q, K, V):
        """forward函数的输入参数有三个，分别是Q， K， V，输入给Attention机制的张量一般情况都是三维张量，
        因此假设Q,K，V都是三维张量"""
        # 第一步，按照计算规则进行计算
        # 将Q，k进行纵轴拼接，做一次线性变化，最后用softmax处理获得结果

        attn_weights = F.softmax(
            self.attn(torch.cat((Q[0], K[0]), 1)), dim=1)

        # 然后进行第一步的后半部分, 将得到的权重矩阵与V做矩阵乘法计算,
        # 当二者都是三维张量且第一维代表为batch条数时, 则做bmm运算
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), V)

        # 之后进行第二步, 通过取[0]是用来降维, 根据第一步采用的计算方法,
        # 需要将Q与第一步的计算结果再进行拼接
        output = torch.cat((Q[0], attn_applied[0]), 1)

        # 最后是第三步, 使用线性层作用在第三步的结果上做一个线性变换并扩展维度，得到输出
        # 因为要保证输出也是三维张量, 因此使用unsqueeze(0)扩展维度
        output = self.attn_combine(output).unsqueeze(0)
        return output, attn_weights


class AgentPolicy(TorchModelV2, nn.Module):
    """"
    """
    custom_name = "AgentPolicy"

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        if hasattr(obs_space, "original_space"):
            obs_space = obs_space.original_space

        if not isinstance(obs_space, Dict):
            if isinstance(obs_space, Box):
                raise TypeError(
                    "({}) Observation space should be a gym Dict."
                    " Is a Box of shape {}".format(name, obs_space.shape)
                )
            raise TypeError(
                "({}) Observation space should be a gym Dict."
                " Is {} instead.".format(name, type(obs_space))
            )
        self.fc_input_shape = 0
        self.fc_keys = []
        self._input_keys = []
        self.atten_shape = 0
        for k, v in obs_space.spaces.items():
            self._input_keys.append(k)
            if k == _MASK_NAME :
                pass
            elif k == _OTHER_NAME:
                self.atten_shape=v.shape[1]
            else:
                self.fc_input_shape += v.shape[0]
                self.fc_keys.append(k)

        self.fc1_dim = self.model_config["custom_model_config"]["fc1_dim"]
        self.fc2_dim = self.model_config["custom_model_config"]["fc2_dim"]
        self.fc1 = SlimFC(self.fc_input_shape, self.fc1_dim)
        self.attention = Attn(self.fc1_dim, self.fc1_dim, self.atten_shape, self.fc1_dim, self.fc1_dim)
        self.fc2 = SlimFC(self.fc1_dim, self.fc2_dim)
        self.fc3 = SlimFC(self.fc1_dim, self.fc2_dim)

        self.softmax = nn.Softmax(dim=None)

        self.fc4 = SlimFC(self.fc1_dim + 1, 1)

    @override(ModelV2)
    def forward(self, input_dict,
                state,
                seq_lens):

        x = torch.cat([input_dict["obs"][k] for k in self.fc_keys], -1)
        x = self.fc1(x)
        y = self.fc1(input_dict["obs"][_OTHER_NAME])
        # raise TypeError(x.shape, y.shape)
        c = self.attention(x, y, y)
        out = torch.cat([x, c], -1)
        out1 = self.fc2(out)
        out2 = self.fc3(out)

        out1 = self.softmax(out1)
        out2 = self.fc4(out2)

        logits = apply_logit_mask(out1, input_dict["obs"][_MASK_NAME])
        self._value_out = out2

        return torch.reshape(logits, (-1, self.num_outputs))

    def value_function(self):
        return torch.reshape(self._value_out, [-1])


ModelCatalog.register_custom_model(AgentPolicy.custom_name, AgentPolicy)
