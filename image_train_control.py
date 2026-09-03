"""
Train a diffusion model on images.
"""

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    NoneLSCC_and_diffusion_defaults,
    create_LSCC_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util_control import TrainLoop


def main():
    args = create_argparser().parse_args()
    # if args.device == "cuda0":
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # elif args.device == "cuda1":
    #     

    dist_util.setup_dist()
    logger.configure(dir="/home/lqg/Desktop/HR/LDPET/NoneLSCC_Logger")

    logger.log("creating model and diffusion...")
    model, diffusion = create_LSCC_and_diffusion(
        **args_to_dict(args, NoneLSCC_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable_params: {trainable_params}")

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=args.deterministic,
        dose=args.dose,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        iteration=args.iteration,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        iteration=100001,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=30000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        deterministic=False,
        device="cuda0",
        dose="D100",
        class_cond=False,
    )
    defaults.update(NoneLSCC_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
