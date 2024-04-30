"""
Utilities for training, testing and caching results
for HICO-DET and V-COCO evaluations

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Microsoft Research Asia
"""

import os
import pickle
import time
import sys
import torch
import random
import warnings
import argparse
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler

from pvic.pvic import build_detector
from hoidet.core.engine import HOIEngine
from hoidet.dataset import custom_collate, DataFactory, HICO_DET_INFO, VCOCO_INFO
from hoidet.detector.detr_detector import detr_detector_args

warnings.filterwarnings("ignore")


def main(rank, args):
    # 确保0号进程先启动
    if rank != 0:
        time.sleep(1)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=rank
    )

    # Fix seed
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.set_device(rank)

    # 数据集
    dataset_info = {
        HICO_DET_INFO.get_name(): HICO_DET_INFO,
        VCOCO_INFO.get_name(): VCOCO_INFO
    }[args.dataset]

    trainset = DataFactory(
        partition=dataset_info.get_training_partition(),
        dataset_info=dataset_info
    )
    testset = DataFactory(
        partition=dataset_info.get_testing_partition(),
        dataset_info=dataset_info
    )

    train_loader = DataLoader(
        dataset=trainset,
        collate_fn=custom_collate, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True,
        sampler=DistributedSampler(
            trainset, num_replicas=args.world_size,
            rank=rank, drop_last=True, shuffle=True)
    )
    test_loader = DataLoader(
        dataset=testset,
        collate_fn=custom_collate, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True,
        sampler=DistributedSampler(
            testset, num_replicas=args.world_size,
            rank=rank, drop_last=False, shuffle=False)
    )

    if args.dataset == 'hicodet':
        object_to_target = train_loader.dataset.dataset.object_to_verb
        args.num_verbs = 117
    elif args.dataset == 'vcoco':
        object_to_target = list(train_loader.dataset.dataset.object_to_verb.values())
        args.num_verbs = 24
    
    model = build_detector(args, object_to_target)

    if os.path.exists(args.resume):
        print(f"=> Rank {rank}: PViC loaded from saved checkpoint {args.resume}.")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"=> Rank {rank}: PViC randomly initialised.")

    engine = HOIEngine(model, train_loader, test_loader, args)

    # engine.cache_hico()
    # engine.cache_results()
    # ap = engine.test_hico()
    # if rank == 0:
    #     print(f"The mAP is {ap.mean():.4f}")
    #     # Fetch indices for rare and non-rare classes
    #     num_anno = torch.as_tensor(trainset.dataset.hoi_instance_num)
    #     rare = torch.nonzero(num_anno < 10).squeeze(1)
    #     non_rare = torch.nonzero(num_anno >= 10).squeeze(1)
    #     print(
    #         f"The mAP is {ap.mean():.4f},"
    #         f" rare: {ap[rare].mean():.4f},"
    #         f" none-rare: {ap[non_rare].mean():.4f}"
    #     )
    #     with open("ap.pkl", "wb") as f:
    #         pickle.dump(ap, f)
    #     print("OK")
    # return

    model.freeze_detector()
    param_dicts = [{"params": [p for p in model.parameters() if p.requires_grad]}]
    optim = torch.optim.AdamW(param_dicts, lr=args.lr_head, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, args.lr_drop, gamma=args.lr_drop_factor)
    # Override optimiser and learning rate scheduler
    engine.update_state_key(optimizer=optim, lr_scheduler=lr_scheduler)

    engine(args.epochs)


@torch.no_grad()
def sanity_check(args):
    dataset = DataFactory(name='hicodet', partition=args.partitions[0], data_root=args.data_root)
    args.num_verbs = 117
    args.num_triplets = 600
    object_to_target = dataset.dataset.object_to_verb
    model = build_detector(args, object_to_target)
    if args.eval:
        model.eval()
    if os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location='cpu')
        print(f"Loading checkpoints from {args.resume}.")
        model.load_state_dict(ckpt['model_state_dict'])

    image, target = dataset[998]
    outputs = model([image], targets=[target])

if __name__ == '__main__':

    parser = argparse.ArgumentParser(parents=[detr_detector_args(), ])
    parser.add_argument('--detector', default='base', type=str)
    parser.add_argument('--raw-lambda', default=2.8, type=float)

    parser.add_argument('--kv-src', default='C5', type=str, choices=['C5', 'C4', 'C3'])
    parser.add_argument('--repr-dim', default=384, type=int)
    parser.add_argument('--triplet-enc-layers', default=1, type=int)
    parser.add_argument('--triplet-dec-layers', default=2, type=int)

    parser.add_argument('--alpha', default=.5, type=float)
    parser.add_argument('--gamma', default=.1, type=float)

    parser.add_argument('--resume', default='', help='Resume from a model')
    parser.add_argument('--use-wandb', default=False, action='store_true')

    parser.add_argument('--port', default='1234', type=str)
    parser.add_argument('--seed', default=140, type=int)
    parser.add_argument('--world-size', default=8, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--sanity', action='store_true')

    parser.add_argument('--cache-root', default="./predict", type=str)

    args = parser.parse_args()
    print(args)

    if args.sanity:
        sanity_check(args)
        sys.exit()
    if not args.use_wandb:
        os.environ["WANDB_MODE"] = "disabled"

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.port

    mp.spawn(main, nprocs=args.world_size, args=(args,))
