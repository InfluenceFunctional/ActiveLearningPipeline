"""
Computes evaluation metrics from a pre-trained GFlowNet model.
"""
from comet_ml import Experiment
from argparse import ArgumentParser
import copy
import gzip
import heapq
import itertools
import os
import pickle
from collections import defaultdict
from itertools import count, product
from pathlib import Path
import yaml
import time

import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from oracles import linearToy, toyHamiltonian, PottsEnergy, seqfoldScore, nupackScore
from utils import get_config, namespace2dict, numpy2python
from gflownet import AptamerSeq, make_mlp, sample

# Float and Long tensors
_dev = [torch.device("cpu")]
tf = lambda x: torch.FloatTensor(x).to(_dev[0])
tl = lambda x: torch.LongTensor(x).to(_dev[0])


def add_args(parser):
    """
    Adds command-line arguments to parser

    Returns:
        argparse.Namespace: the parsed arguments
    """
    args2config = {}
    parser.add_argument(
        "-y",
        "--yaml_config",
        default=None,
        type=str,
        help="Configuration file of the experiment",
    )
    args2config.update({"yaml_config": ["yaml_config"]})
    parser.add_argument("--device", default="cpu", type=str)
    args2config.update({"device": ["device"]})
    parser.add_argument(
        "--model_ckpt",
        default=None,
        type=str,
        help="Checkpoint of the model to evaluate",
    )
    args2config.update({"model_ckpt": ["model_ckpt"]})
    parser.add_argument(
        "--test_data",
        default=None,
        type=str,
        help="Path to CSV file containing test data",
    )
    args2config.update({"test_data": ["test_data"]})
    parser.add_argument(
        "--n_samples",
        default=None,
        type=int,
        help="Number of sequences to sample",
    )
    args2config.update({"n_samples": ["n_samples"]})
    parser.add_argument(
        "--k",
        default=None,
        nargs="*",
        type=int,
        help="List of K, for Top-K",
    )
    args2config.update({"k": ["k"]})
    parser.add_argument("--rand_model", action="store_true", default=False)
    args2config.update({"rand_model": ["rand_model"]})
    parser.add_argument("--do_logq", action="store_true", default=False)
    args2config.update({"do_logq": ["do_logq"]})
    parser.add_argument("--do_sample", action="store_true", default=False)
    args2config.update({"do_sample": ["do_sample"]})
    return parser, args2config


def set_device(dev):
    _dev[0] = dev


def indstr2seq(indstr):
    return [int(el) - 1 for el in str(indstr)]


def logq(traj, actions, model, env):
    traj = traj[::-1]
    actions = actions[::-1]
    traj_obs = np.asarray([env.seq2obs(seq) for seq in traj])
    with torch.no_grad():
        logits_traj = model(tf(traj_obs))
    logsoftmax = torch.nn.LogSoftmax(dim=1)
    logprobs_traj = logsoftmax(logits_traj)
    log_q = torch.tensor(0.0)
    for s, a, logprobs in zip(*[traj, actions, logprobs_traj]):
        log_q = log_q + logprobs[a]
    return log_q.item()


def main(args):
    device_torch = torch.device(args.device)
    device = device_torch
    set_device(device_torch)
    workdir = Path(args.config_file).parent
    # Environment
    env = AptamerSeq(
        args.gflownet.max_seq_length,
        args.gflownet.min_seq_length,
        args.gflownet.nalphabet,
        args.gflownet.min_word_len,
        args.gflownet.max_word_len,
        func=args.gflownet.func,
    )
    # Model
    model = make_mlp(
        [args.gflownet.max_seq_length * args.gflownet.nalphabet]
        + [args.gflownet.n_hid] * args.gflownet.n_layers
        + [len(env.action_space) + 1]
    )
    model.to(device_torch)
    if not args.rand_model:
        model_alias = "gfn"
        if args.model_ckpt:
            model_ckpt = args.model_ckpt
        else:
            model_ckpt = workdir / "model_final.pt"
        model.load_state_dict(torch.load(model_ckpt, map_location=device_torch))
    else:
        model_alias = "rand"
        print("No trained model will be loaded - using random weights")
    # Data set
    if args.n_samples:
        n_samples = args.n_samples
    if args.test_data:
        df_test = pd.read_csv(args.test_data, index_col=0)
        n_samples = len(df_test)
        print("\nTest data")
        print(f"\tAverage score: {df_test.scores.mean()}")
        print(f"\tStd score: {df_test.scores.std()}")
        print(f"\tMin score: {df_test.scores.min()}")
        print(f"\tMax score: {df_test.scores.max()}")

    # Sample data
    if args.do_sample:
        print("\nSampling from GFlowNet model")
        samples_dict, times = sample(
            model,
            n_samples,
            args.gflownet.max_seq_length,
            args.gflownet.min_seq_length,
            args.gflownet.nalphabet,
            args.gflownet.min_word_len,
            args.gflownet.max_word_len,
            args.gflownet.func,
        )
        samples_mat = samples_dict["samples"]
        seq_ints = ["".join([str(el) for el in seq if el > 0]) for seq in samples_mat]
        seq_letters = [
            "".join(
                env.seq2letters(seq[seq > 0], alphabet={1: "A", 2: "T", 3: "C", 4: "G"})
            )
            for seq in samples_mat
        ]
        df_samples = pd.DataFrame(
            {
                "letters": seq_letters,
                "indices": seq_ints,
                "scores": samples_dict["scores"],
            }
        )
        print("Sampled data")
        print(f"\tAverage score: {df_samples.scores.mean()}")
        print(f"\tStd score: {df_samples.scores.std()}")
        print(f"\tMin score: {df_samples.scores.min()}")
        print(f"\tMax score: {df_samples.scores.max()}")
        output_samples = workdir / "{}_samples_n{}.csv".format(model_alias, n_samples)
        df_samples.to_csv(output_samples)
        if any([s in env.func for s in ["pins", "pairs"]]):
            scores_sorted = np.sort(df_samples["scores"].values)[::-1]
        else:
            scores_sorted = np.sort(df_samples["scores"].values)
        for k in args.k:
            mean_topk = np.mean(scores_sorted[:k])
            print(f"\tAverage score top-{k}: {mean_topk}")

    # log q(x)
    if args.do_logq:
        print("\nComputing log q(x)")
        data_logq = []
        for seqint, score in tqdm(zip(df_test.indices, df_test.scores)):
            traj, actions = env.trajectories(
                indstr2seq(seqint), [indstr2seq(seqint)], [env.eos]
            )
            data_logq.append(logq(traj, actions, model, env))
        corr = np.corrcoef(data_logq, df_test.scores)
        df_test["logq"] = data_logq
        print(f"Correlation between e(x) and q(x): {corr[0, 1]}")
        print(f"Data log-likelihood: {df_test.logq.sum()}")
        output_test_logq = workdir / Path(args.test_data).name
        df_test.to_csv(output_test_logq)


if __name__ == "__main__":
    parser = ArgumentParser()
    _, override_args = parser.parse_known_args()
    parser, args2config = add_args(parser)
    args = parser.parse_args()
    config = get_config(args, override_args, args2config)
    config.config_file = args.yaml_config
    torch.set_num_threads(1)
    main(config)
