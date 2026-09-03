"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import time
import numpy as np
import torch as th
import torch.distributed as dist

from improved_diffusion.image_datasets import load_data
from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    NoneLSCC_and_diffusion_defaults,
    create_LSCC_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from skimage.measure import compare_ssim, compare_psnr, compare_mse

def main():
    args = create_argparser().parse_args()
    # if args.device == "cuda0":
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # elif args.device == "cuda1":
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    dist_util.setup_dist()
    # logger.configure(dir="/home/lqg/Desktop/HR/LDPET/ControlNet_sample")
    logger.configure(dir="/media/lqg/Elements SE/Logger-Sample/controlNet_sample")

    logger.log("creating model and diffusion...")
    model, diffusion = create_LSCC_and_diffusion(
        **args_to_dict(args, NoneLSCC_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("sampling...")

    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=args.deterministic,
        dose=args.dose,
    )

    all_images = []
    all_labels = []
    all_inputs = []
    psnr_total = 0.
    for step, batchNcond in enumerate(data):
        startTime = time.time()
        if step >= args.num_samples:
            break
        batch = batchNcond[0]
        cond = batchNcond[1]
        batch = batch.to(dist_util.dev())
        cond = cond['lq'].to(dist_util.dev())
        model_kwargs = {"hint": cond}

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 1, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        pad_width = [(0, 0), (0, 0), (52, 52), (52, 52)]
        try:
            psnr = compare_psnr(
                np.pad(sample.detach().cpu().numpy(), pad_width, mode='constant', constant_values=0),
                np.pad(batch.detach().cpu().numpy(), pad_width, mode='constant', constant_values=0)
            )
        except ValueError:
            psnr = 0.
            print("value out of range, skipped psnr")
        psnr_total += psnr
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        batch = batch.permute(0, 2, 3, 1)
        batch = batch.contiguous()
        cond = cond.permute(0, 2, 3, 1)
        cond = cond.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        gathered_labels = [th.zeros_like(batch) for _ in range(dist.get_world_size())]
        gathered_conds = [th.zeros_like(cond) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        dist.all_gather(gathered_labels, batch)
        dist.all_gather(gathered_conds, cond)
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        all_labels.extend([batch.cpu().numpy() for batch in gathered_labels])
        all_inputs.extend([cond.cpu().numpy() for cond in gathered_conds])
        logger.log(f"created {len(all_images) * args.batch_size} samples, PSNR: {psnr}" "iteration time {:.2f}".format(startTime - time.time()))

    logger.log(f"avg PSNR: {psnr_total / args.num_samples}")

    arr = np.concatenate(all_images, axis=0)
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.save(out_path, arr)
    arr = np.concatenate(all_labels, axis=0)
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"labels_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.save(out_path, arr)
    arr = np.concatenate(all_inputs, axis=0)
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"inputs_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.save(out_path, arr)
    
    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        data_dir="",
        clip_denoised=False,
        batch_size=1,
        num_samples=8,
        use_ddim=False,
        model_path="",
        deterministic=True,
        device="cuda1",
        dose="D100",
        class_cond=False,
    )
    defaults.update(NoneLSCC_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
