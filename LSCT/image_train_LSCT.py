"""
Train a diffusion model on images.
"""

import argparse
import os
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append("..")

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from LSCT.script_util_LSCT import (
    model_defaults,
    create_model,
    create_trianing_loss,
    args_to_dict,
    add_dict_to_argparser,
)
from LSCT.train_util_LSCT import TrainLoop


def main():
    args = create_argparser().parse_args()
    # if args.device == "cuda0":
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # elif args.device == "cuda1":
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    dist_util.setup_dist()
    logger.configure(dir="../../LSCT_Logger")

    logger.log("creating CLASSIFIER model...")
    model = create_model(
        **args_to_dict(args, model_defaults().keys())
    )
    model.to(dist_util.dev())

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=args.deterministic,
        dose=args.dose,
    )
    loss_fn = create_trianing_loss(type=args.loss_type)

    logger.log("training...")
    TrainLoop(
        model=model,
        iteration=args.iteration,
        loss_fn=loss_fn,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        lr=1e-4,
        iteration=100001,
        # loss_type="crossEntropyAndAcc",
        loss_type="crossEntropy",
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        deterministic=False,
        device="cuda0",
        dose="ALL",
        class_cond=True,
    )
    defaults.update(model_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
