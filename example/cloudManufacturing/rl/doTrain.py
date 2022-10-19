import argparse
import logging
import os
import sys

import ray
import yaml
from ray.rllib.agents.a3c.a3c import A3CTrainer
from ray.tune.logger import pretty_print

from algorithm.rl.env_warpper import RLlibEnvWrapper
import policy_model
from example.cloudManufacturing.env import CloudManufacturing

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

    config_path = os.path.join(args.run_dir, "config.yaml")

    with open(config_path, "r") as f:
        run_configuration = yaml.safe_load(f)
    return args.run_dir, run_configuration


def build_Trainer(run_configuration):
    trainer_config = run_configuration.get("trainer")
    env_config = run_configuration.get("env")["env_config"]

    # === Multiagent Policies ===
    dummy_env = RLlibEnvWrapper(env_config, CloudManufacturing)

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
        "num_workers": trainer_config.get("num_workers")
    })

    trainer = A3CTrainer(
        env= run_configuration.get("env")["env_name"],
        config=trainer_config
    )
    return trainer


def save_ckpt(trainer, result, ckpt_frequency, run_dir):
    ckpt_dir = os.path.join(run_dir, "ckpts")

    for sub_dir in [ckpt_dir]:
        os.makedirs(sub_dir, exist_ok=True)

    global_step = result["timesteps_total"]
    if global_step % ckpt_frequency == 0:
        path = trainer.save(ckpt_dir)
        print("checkpoint saved at", path)
        return path
    else:
        pass


if __name__ == "__main__":

    # 获取参数
    run_dir, run_config = process_args()
    # 创建训练器
    trainer = build_Trainer(run_config)

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
    logger.info("Final ckpt saved! All done.")

    ray.shutdown()  # shutdown Ray after use
