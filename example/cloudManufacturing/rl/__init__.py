<<<<<<< HEAD
from example.cloudManufacturing.env import CloudManufacturing

from ray.tune.registry import register_env
from algorithm.rl.env_warpper import RLlibEnvWrapper

def env_creator(env_config):  # 此处的 env_config对应 我们在建立trainer时传入的dict env_config
    return RLlibEnvWrapper(env_config, CloudManufacturing)


register_env(CloudManufacturing.name, env_creator)
=======
>>>>>>> 0ea7186f5ff1f557087e095c3e6bfb476e8ce558



