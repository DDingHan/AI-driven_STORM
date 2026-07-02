#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2023-03-11 17:17:41

import os, sys
import argparse
from pathlib import Path

from omegaconf import OmegaConf
from sampler import ResShiftSampler

from utils.util_opts import str2bool
from basicsr.utils.download_util import load_file_from_url


_LINK = {
    'vqgan': 'https://github.com/DDingHan/AI-driven_STORM/releases/download/v1.0/autoencoder_vq_f4.pth',
    'mito': 'https://github.com/DDingHan/AI-driven_STORM/releases/download/v1.0/mito_encoder.pth',
    'tubule': 'https://github.com/DDingHan/AI-driven_STORM/releases/download/v1.0/tubule_encoder.pth',
     }

def get_parser(**parser_kwargs):
    # python demo.py -i sample/tubule -o result/tubule --type tubule
    # CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nproc_per_node=1 --nnodes=1 inference_resshift_semiconductor.py
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-i", "--in_path", type=str, default="", help="Input path.")
    parser.add_argument("-o", "--out_path", type=str, default="./results", help="Output path.")
    parser.add_argument("--mask_path", type=str, default="", help="Mask path for inpainting.")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint path.")
    parser.add_argument("--scale", type=int, default=1, help="Scale factor for SR.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument("--bs", type=int, default=1, help="Batch size.")
    parser.add_argument(
            "--type",
            type=str,
            default="tubule",
            choices=["tubule", "mito"],
            help="Choose the type of the input image",
            )
    parser.add_argument(
            "--chop_size",
            type=int,
            default=512,
            choices=[512, 256, 64],
            help="Chopping forward.",
            )
    parser.add_argument(
            "--chop_stride",
            type=int,
            default=-1,
            help="Chopping stride.",
            )
    args = parser.parse_args()

    return args

def get_configs(args):
    ckpt_dir = Path('./weights')
    if not ckpt_dir.exists():
        ckpt_dir.mkdir()

    if args.type == 'tubule' or args.type == 'mito':
        configs = OmegaConf.load('config.yaml')

        vqgan_url = _LINK['vqgan']
        vqgan_path = ckpt_dir / 'autoencoder_vq_f4.pth'

        if args.ckpt is None:
            ckpt_url = _LINK[args.type]
            ckpt_path = ckpt_dir / f'{args.type}_encoder.pth'

            # prepare the checkpoint
            if not ckpt_path.exists():
                load_file_from_url(
                    url=ckpt_url,
                    model_dir=ckpt_dir,
                    progress=True,
                    file_name=ckpt_path.name,
                )
        else:
            ckpt_path = Path(args.ckpt)

            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    else:
        raise TypeError(f"Unexpected task type: {args.type}!")


    if not vqgan_path.exists():
        load_file_from_url(
            url=vqgan_url,
            model_dir=ckpt_dir,
            progress=True,
            file_name=vqgan_path.name,
        )

    configs.model.ckpt_path = str(ckpt_path)
    configs.autoencoder.ckpt_path = str(vqgan_path)

    # save folder
    if not Path(args.out_path).exists():
        Path(args.out_path).mkdir(parents=True)

    if args.chop_stride < 0:
        if args.chop_size == 512:
            chop_stride = (512 - 64) * (4 // args.scale)
        elif args.chop_size == 256:
            chop_stride = (256 - 32) * (4 // args.scale)
        elif args.chop_size == 64:
            chop_stride = (64 - 16) * (4 // args.scale)
        else:
            raise ValueError("Chop size must be in [512, 256]")
    else:
        chop_stride = args.chop_stride * (4 // args.scale)
    args.chop_size *= (4 // args.scale)
    print(f"Chopping size/stride: {args.chop_size}/{chop_stride}")

    return configs, chop_stride

def main():
    args = get_parser()

    configs, chop_stride = get_configs(args)

    resshift_sampler = ResShiftSampler(
            configs,
            sf=args.scale,
            chop_size=args.chop_size,
            chop_stride=chop_stride,
            chop_bs=1,
            use_amp=True,
            seed=args.seed,
            padding_offset=configs.model.params.get('lq_size', 64),
            )


    resshift_sampler.inference(
            args.in_path,
            args.out_path,
            mask_path=None,
            bs=args.bs,
            noise_repeat=False
            )

if __name__ == '__main__':
    main()
