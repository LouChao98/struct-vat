# -*- coding: utf-8 -*-

import copy
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from supar.utils import Config
from supar.utils.logging import init_logger, logger
from supar.utils.parallel import get_free_port


def init(parser):
    parser.add_argument('--path', '-p', help='path to model file')
    parser.add_argument('--conf', '-c', default='', help='path to config file')
    parser.add_argument('--device', '-d', default='-1', help='ID of GPU to use')
    parser.add_argument('--seed', '-s', default=1, type=int, help='seed for generating random numbers')
    parser.add_argument('--threads', '-t', default=16, type=int, help='max num of threads')

    parser.add_argument('--no-ckpt', action='store_true', help='do not save ckpt when training')

    args, unknown = parser.parse_known_args()
    args, unknown = parser.parse_known_args(unknown, args)
    args = Config.load(**vars(args), unknown=unknown)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device_count = torch.cuda.device_count()
    if device_count > 1:
        os.environ['MASTER_ADDR'] = 'tcp://localhost'
        os.environ['MASTER_PORT'] = get_free_port()
        mp.spawn(parse, args=(args,), nprocs=device_count)
    else:
        parse(0 if torch.cuda.is_available() else -1, args)


def parse(local_rank, args):
    Parser = args.pop('Parser')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    init_logger(logger, f"{args.path}.{args.mode}.log", 'a' if args.get('checkpoint') else 'w')
    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend='nccl',
                                init_method=f"{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
                                world_size=torch.cuda.device_count(),
                                rank=local_rank)
    torch.cuda.set_device(local_rank)
    logger.info('\n' + str(args))

    args.local_rank = local_rank
    if args.mode == 'train':
        parser = Parser.load(**args) if args.checkpoint or args.get('load_from_saved') else Parser.build(**args)
        parser.train(**args)
    elif args.mode == 'evaluate':
        parser = Parser.load(**args)
        parser.evaluate(**args)
    elif args.mode == 'predict':
        parser = Parser.load(**args)
        parser.predict(**args)
