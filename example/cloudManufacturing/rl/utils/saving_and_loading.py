import logging
import os

import sys
import pickle
import shutil
import torch

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

    nested_env_ids_and_results = trainer.workers.foreach_worker(
        lambda w: [(env.env_id, env_function(env)) for env in w.async_env.envs]
    )
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

def save_snapshot(trainer, ckpt_dir, suffix=""):
    # Create a new trainer snapshot
    filepath = trainer.save(ckpt_dir)
    filepath_metadata = filepath + ".tune_metadata"

    # Copy this to a standardized name (to only keep the latest)
    latest_filepath = os.path.join(
        ckpt_dir, "latest_checkpoint{}.pkl".format("." + suffix if suffix != "" else "")
    )
    latest_filepath_metadata = latest_filepath + ".tune_metadata"
    shutil.copy(filepath, latest_filepath)
    shutil.copy(filepath_metadata, latest_filepath_metadata)
    # Get rid of the timestamped copy to prevent accumulating too many large files
    os.remove(filepath)
    os.remove(filepath_metadata)

    # Also take snapshots of each environment object
    remote_env_fun(trainer, lambda env_wrapper: env_wrapper.save_game_object(ckpt_dir))

    logger.info("Saved Trainer snapshot + Env object @ %s", latest_filepath)


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
        if os.path.isfile(ckpt):
            trainer.restore(ckpt)
            loaded_ckpt_success = True
            logger.info(
                "load_snapshot -> loading %s SUCCESS for %s %s", ckpt, suffix, trainer
            )
        else:
            logger.info(
                "load_snapshot -> loading %s FAILED,"
                " skipping restoring cpkt for %s %s",
                ckpt,
                suffix,
                trainer,
            )
    else:
        raise AssertionError

    # Also load snapshots of each environment object
    remote_env_fun(
        trainer,
        lambda env_wrapper: env_wrapper.load_game_object(
            os.path.join(run_dir, "ckpts")
        ),
    )

    return loaded_ckpt_success
