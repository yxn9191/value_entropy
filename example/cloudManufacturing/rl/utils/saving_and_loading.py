import logging
import os

import sys
import pickle
import shutil
import torch
import numpy as np


logging.basicConfig(stream=sys.stdout, format="%(asctime)s %(message)s")
logger = logging.getLogger("main")
logger.setLevel(logging.DEBUG)


def save_torch_model_weights(trainer, ckpt_dir, global_step):
    w = trainer.get_weights(["a"])
    pol = trainer.get_policy("a")
    model_w_array = pol.model.state_dict()

    fn = os.path.join(
        ckpt_dir, "torch.weights.global-step-{}".format(global_step)
    )
    with open(fn, "wb") as f:
        pickle.dump(w, f)

    fn = os.path.join(
        ckpt_dir,
        "policy-model-weight-array.global-step-{}.pth".format(global_step),
    )
    torch.save(model_w_array, fn)
    logger.info("Saved torch weights @ %s", fn)


def save_ckpt(trainer, result, ckpt_frequency, run_dir):
    ckpt_dir = os.path.join(run_dir, "ckpts")

    for sub_dir in [ckpt_dir]:
        os.makedirs(sub_dir, exist_ok=True)

    global_step = result["timesteps_total"]
    if global_step % ckpt_frequency == 0:
        save_torch_model_weights(trainer, ckpt_dir, global_step)
        path = trainer.save(ckpt_dir)
        #remote_env_fun(trainer, lambda env_wrapper: env_wrapper.save_game_object(ckpt_dir))
        print("checkpoint saved at", path)
        return path
    else:
        pass


def load_torch_model_weights(trainer, ckpt):
    print(ckpt)
    # assert os.path.isfile(ckpt)
    with open(ckpt, "rb") as f:
        weights = pickle.load(f)
        trainer.set_weights(weights)
    logger.info("loaded torch model weights:\n\t%s\n", ckpt)



def remote_env_fun(trainer, env_function):
    """
    Create a dictionary with the following mapping:
        result[env_wrapper.env_id] = env_function(env)
    where each entry in the dictionary comes from one of the active envs in the trainer.
    env_function must be a function that takes an environment as its single argument
    """

    def func(w):
        if w.async_env:
            return [(env.env_id, env_function(env)) for env in w.async_env.envs]


    nested_env_ids_and_results = trainer.workers.foreach_worker( func)
    nested_env_ids_and_results = nested_env_ids_and_results[
        1:
    ]  # Ignore the local worker

    # Store them first this way in case they don't come out sorted
    # (gets sorted by env_id before being returned)
    result = {}

    for worker_stuff in nested_env_ids_and_results:
        for env_id, output in worker_stuff:
            result[env_id] = output
    return result


def get_trainer_envs(trainer):
    return remote_env_fun(trainer, lambda env: env)


def collect_stored_rollouts(trainer):
    aggregate_rollouts = {}

    rollout_dict = remote_env_fun(trainer, lambda e: e.rollout)
    n_envs = len(rollout_dict)

    for env_id, env_rollout in rollout_dict.items():
        for k, v in env_rollout.items():
            if k not in aggregate_rollouts:
                sz = v.shape
                sz = [sz[0], n_envs] + sz[1:]
                aggregate_rollouts[k] = np.zeros(sz)
            aggregate_rollouts[k][:, env_id] = v

    return aggregate_rollouts


# def accumulate_and_broadcast_saez_buffers(trainer):
#     component_name = "PeriodicBracketTax"
#
#     def extract_local_saez_buffers(env_wrapper):
#         return env_wrapper.env.get_component(component_name).get_local_saez_buffer()
#
#     replica_buffers = remote_env_fun(trainer, extract_local_saez_buffers)
#
#     global_buffer = []
#     for local_buffer in replica_buffers.values():
#         global_buffer += local_buffer
#
#     def set_global_buffer(env_wrapper):
#         env_wrapper.env.get_component(component_name).set_global_saez_buffer(
#             global_buffer
#         )
#
#     _ = remote_env_fun(trainer, set_global_buffer)

# def save_snapshot(trainer, ckpt_dir, suffix=""):
#     # Create a new trainer snapshot
#     filepath = trainer.save(ckpt_dir)
#     filepath_metadata = filepath + ".tune_metadata"

#     # Copy this to a standardized name (to only keep the latest)
#     latest_filepath = os.path.join(
#         ckpt_dir, "latest_checkpoint{}.pkl".format("." + suffix if suffix != "" else "")
#     )
#     latest_filepath_metadata = latest_filepath + ".tune_metadata"
#     shutil.copy(filepath, latest_filepath)
#     shutil.copy(filepath_metadata, latest_filepath_metadata)
#     # Get rid of the timestamped copy to prevent accumulating too many large files
#     os.remove(filepath)
#     os.remove(filepath_metadata)

#     # Also take snapshots of each environment object
#     remote_env_fun(trainer, lambda env_wrapper: env_wrapper.save_game_object(ckpt_dir))

#     logger.info("Saved Trainer snapshot + Env object @ %s", latest_filepath)


def load_snapshot(trainer, run_dir, ckpt=None, suffix="", load_latest=False):

    assert ckpt or load_latest

    loaded_ckpt_success = False

    if not ckpt:
        if load_latest:
            # Restore from the latest checkpoint (pointing to it by path)
            ckpt_fp = os.path.join(
                run_dir,
                "ckpts",
                "latest_checkpoint{}.pkl".format("." + suffix if suffix != "" else ""),
            )
            if os.path.isfile(ckpt_fp):
                trainer.restore(ckpt_fp)
                loaded_ckpt_success = True
                logger.info(
                    "load_snapshot -> loading %s SUCCESS for %s %s",
                    ckpt_fp,
                    suffix,
                    trainer,
                )
            else:
                logger.info(
                    "load_snapshot -> loading %s FAILED,"
                    " skipping restoring cpkt for %s %s",
                    ckpt_fp,
                    suffix,
                    trainer,
                )
        else:
            raise NotImplementedError
    elif ckpt:

        trainer.restore(ckpt)
        loaded_ckpt_success = True
        logger.info(
                "load_snapshot -> loading %s SUCCESS for %s %s", ckpt, suffix, trainer
            )
        # else:
        #     logger.info(
        #         "load_snapshot -> loading %s FAILED,"
        #         " skipping restoring cpkt for %s %s",
        #         ckpt,
        #         suffix,
        #         trainer,
        #     )
    else:
        raise AssertionError

    # Also load snapshots of each environment object
    # remote_env_fun(
    #     trainer,
    #     lambda env_wrapper: env_wrapper.load_game_object(
    #         os.path.join(run_dir, "ckpts")
    #     ),
    # )

    return loaded_ckpt_success

def fill_out_run_dir(run_dir):
    dense_log_dir = os.path.join(run_dir, "dense_logs")
    ckpt_dir = os.path.join(run_dir, "ckpts")

    for sub_dir in [dense_log_dir, ckpt_dir]:
        os.makedirs(sub_dir, exist_ok=True)

    latest_filepath = os.path.join(ckpt_dir, "latest_checkpoint.pkl")
    restore = bool(os.path.isfile(latest_filepath))

    return dense_log_dir, ckpt_dir, restore

def set_up_dirs_and_maybe_restore(run_directory, run_configuration, trainer_obj):
    # === Set up Logging & Saving, or Restore ===
    # All model parameters are always specified in the settings YAML.
    # We do NOT overwrite / reload settings from the previous checkpoint dir.
    # 1. For new runs, the only object that will be loaded from the checkpoint dir
    #    are model weights.
    # 2. For crashed and restarted runs, load_snapshot will reload the full state of
    #    the Trainer(s), including metadata, optimizer, and models.
    (
        dense_log_directory,
        ckpt_directory,
        restore_from_crashed_run,
    ) = fill_out_run_dir(run_directory)

    # If this is a starting from a crashed run, restore the last trainer snapshot
    if restore_from_crashed_run:
        logger.info(
            "ckpt_dir already exists! Planning to restore using latest snapshot from "
            "earlier (crashed) run with the same ckpt_dir %s",
            ckpt_directory,
        )

        at_loads_a_ok = load_snapshot(
            trainer_obj, run_directory, load_latest=True
        )

        # at this point, we need at least one good ckpt restored
        if not at_loads_a_ok:
            logger.fatal(
                "restore_from_crashed_run -> restore_run_dir %s, but no good ckpts "
                "found/loaded!",
                run_directory,
            )
            sys.exit()

        # === Trainer-specific counters ===
        training_step_last_ckpt = (
            int(trainer_obj._timesteps_total) if trainer_obj._timesteps_total else 0
        )
        epis_last_ckpt = (
            int(trainer_obj._episodes_total) if trainer_obj._episodes_total else 0
        )
    elif run_configuration["general"].get("ckpt_path", ""):
        at_loads_a_ok = load_snapshot(
           trainer = trainer_obj, run_dir=run_directory, ckpt=run_configuration["general"].get("ckpt_path", "") 
        )

        # at this point, we need at least one good ckpt restored
        if not at_loads_a_ok:
            logger.fatal(
                "restore_from_crashed_run -> restore_run_dir %s, but no good ckpts "
                "found/loaded!",
                run_directory,
            )
            sys.exit()

        # === Trainer-specific counters ===
        training_step_last_ckpt = (
            int(trainer_obj._timesteps_total) if trainer_obj._timesteps_total else 0
        )
        epis_last_ckpt = (
            int(trainer_obj._episodes_total) if trainer_obj._episodes_total else 0
        )
    else:
        logger.info("Not restoring trainer...")
        # === Trainer-specific counters ===
        training_step_last_ckpt = 0
        epis_last_ckpt = 0

        # For new runs, load only torch checkpoint weights
        starting_weights_path_agents = run_configuration["general"].get(
            "restore_torch_weights_agents", ""
        )
        if starting_weights_path_agents:
            logger.info("Restoring agents Torch weights...")
            load_torch_model_weights(trainer_obj, starting_weights_path_agents)
        else:
            logger.info("Starting with fresh agent Torch weights.")


    return (
        dense_log_directory,
        ckpt_directory,
        restore_from_crashed_run,
        training_step_last_ckpt,
        epis_last_ckpt,
    )
