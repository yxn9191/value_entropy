import argparse
import os
import sys

import ray
import yaml
import policy_model  # 必须引入，不然模型没有注册
from example.cloudManufacturing_network.env import CloudManufacturing_network

current_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(current_path)

from utils.saving_and_loading import *
# 必须后面引入不然会报错
from algorithm.rl.env_warpper import RLlibEnvWrapper
from algorithm.rl.callback import MyCallbacks
from ray.rllib.algorithms.a2c import A2C
from ray.tune.logger import pretty_print

ray.init(log_to_driver=False)

logging.basicConfig(stream=sys.stdout, format="%(asctime)s %(message)s")
logger = logging.getLogger("main")
logger.setLevel(logging.DEBUG)


def process_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-dir", type=str, default="phase", help="Path to the directory for this run."
    )

    args = parser.parse_args()
    run_dir = os.path.join(current_path, args.run_dir)
    config_path = os.path.join(run_dir, "config.yaml")

    with open(config_path, "r") as f:
        run_configuration = yaml.safe_load(f)
    return run_dir, run_configuration


def build_Trainer(run_configuration):
    trainer_config = run_configuration.get("trainer")
    # env_config = run_configuration.get("env")["env_config"]
    env_config = {
        "env_config_dict": run_configuration.get("env")["env_config"],
        "num_envs_per_worker": trainer_config.get("num_envs_per_worker"),
    }
    trainer_config["callbacks"] = MyCallbacks
    # === Multiagent Policies ===

    dummy_env = RLlibEnvWrapper(env_config, CloudManufacturing_network)
    # Policy tuples for agent/planner policy types
    agent_policy_tuple = (
        None,
        dummy_env.observation_space,
        dummy_env.action_space,
        run_configuration.get("agent_policy"),
    )

    policies = {"a": agent_policy_tuple}

    def policy_mapping_fun(i):
        return "a"

    trainer_config.update({
        "env_config": env_config,
        'framework': 'torch',
        "multiagent": {
            "policies": policies,
            "policies_to_train": ["a"],
            "policy_mapping_fn": policy_mapping_fun,
        },
        "metrics_num_episodes_for_smoothing": trainer_config.get("num_workers")
                                              * trainer_config.get("num_envs_per_worker"),
        "num_workers": trainer_config.get("num_workers")
    })

    trainer = A2C(
        # env = RLlibEnvWrapper,
        env=run_configuration.get("env")["env_name"],
        config=trainer_config
    )

    return trainer


if __name__ == "__main__":

    # 获取参数
    run_dir, run_config = process_args()
    # 创建训练器

    trainer = build_Trainer(run_config)

    (
        dense_log_dir,
        ckpt_dir,
        restore_from_crashed_run,
        step_last_ckpt,
        num_parallel_episodes_done,
    ) = set_up_dirs_and_maybe_restore(run_dir, run_config, trainer)

    # 开始训练
    ckpt_frequency = run_config["general"].get("ckpt_frequency_steps", 0)
    # 当前训练轮数
    num_episodes_done = 0

    while num_episodes_done < run_config["general"]["episodes"]:
        # Training
        result = trainer.train()

        # === Counters++ ===
        num_episodes_done = result["episodes_total"]
        global_step = result["timesteps_total"]
        curr_iter = result["training_iteration"]

        logger.info(
            "Iter %d: steps this-iter %d total %d -> %d/%d episodes done",
            curr_iter,
            result["time_this_iter_s"],
            global_step,
            num_episodes_done,
            run_config["general"]["episodes"],
        )

        if curr_iter == 1 or result["episodes_this_iter"] > 0:
            logger.info(pretty_print(result))

        # 保存训练器参数
        step_last_ckpt = save_ckpt(trainer, result, ckpt_frequency, run_dir)

    # Finish up
    logger.info("Complete training!")
    path = trainer.save()
    print(path)
    # save_snapshot(trainer, run_dir)
    # save_torch_model_weights(trainer, run_dir, global_step)
    logger.info("Final ckpt saved! All done.")

    ray.shutdown()  # shutdown Ray after use
