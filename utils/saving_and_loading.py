import logging
import os
import pickle
import sys

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
    assert os.path.isfile(ckpt)
    with open(ckpt, "rb") as f:
        weights = pickle.load(f)
        trainer.set_weights(weights)
    logger.info("loaded torch model weights:\n\t%s\n", ckpt)
