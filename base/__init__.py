from base.environment import env_registry as envs


def make_env_instance(env_name, **kwargs):
    env_class = envs.get(env_name)
    return env_class(**kwargs)