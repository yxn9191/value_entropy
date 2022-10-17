import math

import torch
import torch.nn as nn
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
             range(other_inputs.shape[0])]
    bate = [torch.exp(alpha[i]) / sum([torch.exp(alpha)]) for i in range(other_inputs.shape[0])]
    c_i = sum([torch.dot(bate[i], other_inputs[i]) for i in range(other_inputs.shape[0])])
    return c_i


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
        for k, v in obs_space.spaces.items():
            self._input_keys.append(k)
            if k == _MASK_NAME or k == _OTHER_NAME:
                pass
            else:
                self.fc_input_shape += v.shape[0]
                self.fc_keys.append(k)

        self.fc1_dim = self.model_config["custom_model_config"]["fc1_dim"]
        self.fc2_dim = self.model_config["custom_model_config"]["fc2_dim"]
        self.fc1 = SlimFC(self.fc_input_shape, self.fc1_dim)
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
        c = attention(x, y)
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
