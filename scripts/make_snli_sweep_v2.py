# Create a script to run a random hyperparameter search.

import copy
import getpass
import os
import random
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--use_encode", action="store_true", default=False, dest="use_encode")
parser.add_argument("--use_reinforce", action="store_true", default=False, dest="use_reinforce")
parser.add_argument("--rl_baseline", type=str, default="ema")
parser.add_argument("--runs", type=int, default=4)
parser.add_argument("--datapath", type=str, default="/scratch/apd283")
parser.add_argument("--logpath", type=str, default="../logs")
parser.add_argument("--venv", type=str, default="~/spinn/.venv-hpc/bin/activate")
args = parser.parse_args()

NYU_NON_PBS = True

LIN = "LIN"
EXP = "EXP"
SS_BASE = "SS_BASE"

# Instructions: Configure the variables in this block, then run
# the following on a machine with qsub access:
# python make_sweep.py > my_sweep.sh
# bash my_sweep.sh

# - #

# Non-tunable flags that must be passed in.

FIXED_PARAMETERS = {
    "data_type":     "snli",
    "model_type":      "SPINN",
    "training_data_path":    "{}/snli_1.0/snli_1.0_train.jsonl".format(args.datapath),
    "eval_data_path":    "{}/snli_1.0/snli_1.0_dev.jsonl".format(args.datapath),
    "embedding_data_path": "{}/glove/glove.840B.300d.txt".format(args.datapath),
    "log_path": "{}".format(args.logpath),
    "word_embedding_dim":	"300",
    "model_dim":   "600",
    "seq_length":	"50",
    "eval_seq_length":	"50",
    "eval_interval_steps": "500",
    "statistics_interval_steps": "500",
    "use_internal_parser": "",
    "batch_size":  "32",
    "ckpt_path":  "{}".format(args.logpath)
}

if args.use_encode:
    FIXED_PARAMETERS["use_encode"] = ""
if args.use_reinforce:
    FIXED_PARAMETERS["use_reinforce"] = ""
    FIXED_PARAMETERS["rl_baseline"] = "{}".format(args.rl_baseline)

# Tunable parameters.
SWEEP_PARAMETERS = {
    "learning_rate":      ("lr", EXP, 0.0002, 0.002),  # RNN likes higher, but below 009.
    "l2_lambda":   		  ("l2", EXP, 8e-7, 2e-5),
    "semantic_classifier_keep_rate": ("skr", LIN, 0.7, 0.95),  # NB: Keep rates may depend considerably on dims.
    "embedding_keep_rate": ("ekr", LIN, 0.7, 0.95),
    "learning_rate_decay_per_10k_steps": ("dec", EXP, 0.5, 1.0),
    "tracking_lstm_hidden_dim": ("tdim", EXP, 24, 128),
    "transition_weight":  ("trwt", EXP, 0.5, 4.0),
    "num_mlp_layers": ("mlp", LIN, 1, 3)
}

sweep_name = "sweep_" + \
    FIXED_PARAMETERS["data_type"] + "_" + FIXED_PARAMETERS["model_type"]
if "use_encode" in FIXED_PARAMETERS:
    sweep_name += "_enc"
if "use_reinforce" in FIXED_PARAMETERS:
    sweep_name += "_rl"
    if "rl_baseline" in FIXED_PARAMETERS:
        sweep_name += "_{}".format(FIXED_PARAMETERS["rl_baseline"])
sweep_runs = args.runs
queue = "jag"

# - #
print "# NAME: " + sweep_name
print "# NUM RUNS: " + str(sweep_runs)
print "# SWEEP PARAMETERS: " + str(SWEEP_PARAMETERS)
print "# FIXED_PARAMETERS: " + str(FIXED_PARAMETERS)
print

for run_id in range(sweep_runs):
    params = {}
    name = sweep_name + "_" + str(run_id)

    params.update(FIXED_PARAMETERS)
    for param in SWEEP_PARAMETERS:
        config = SWEEP_PARAMETERS[param]
        t = config[1]
        mn = config[2]
        mx = config[3]

        r = random.uniform(0, 1)
        if t == EXP:
            lmn = np.log(mn)
            lmx = np.log(mx)
            sample = np.exp(lmn + (lmx - lmn) * r)
        elif t==SS_BASE:
            lmn = np.log(mn)
            lmx = np.log(mx)
            sample = 1 - np.exp(lmn + (lmx - lmn) * r)
        else:
            sample = mn + (mx - mn) * r

        if isinstance(mn, int):
            sample = int(round(sample, 0))
            val_disp = str(sample)
        else: 
            val_disp = "%.2g" % sample

        params[param] = sample
        name += "-" + config[0] + val_disp

    flags = ""
    for param in params:
        value = params[param]
        val_str = ""
        flags += " \\\n--" + param + " " + str(value)

    flags += " \\\n--experiment_name " + name
    if NYU_NON_PBS:
        run_type = "spinn"
        if "use_encode" in FIXED_PARAMETERS:
            run_type += "-enc"
        if "use_reinforce" in FIXED_PARAMETERS:
            run_type += "-rl"
            if "rl_baseline" in FIXED_PARAMETERS:
                run_type += "-{}".format(FIXED_PARAMETERS["rl_baseline"])
        filename = "{}-{}-{}.sh".format(FIXED_PARAMETERS["data_type"], run_type, run_id)
        output = """#!/bin/bash

module load cuda/7.5.18
module load cudnn/7.0v4.0
module load numpy/intel/1.10.1
module load pandas/intel/0.17.1
module load scikit-learn/intel/0.18

. {} # Source Virtual Environment

export PYTHONPATH=$PYTHONPATH:./python:../python:spinn/python
cd ~/spinn/checkpoints

python -m spinn.models.fat_classifier {}
""".format(args.venv, flags)
        with open(filename, "w") as f:
            f.write(output)

    else:
        print "export SPINN_FLAGS=\"" + flags + "\"; export DEVICE=gpuX; qsub -v SPINN_FLAGS,DEVICE ../scripts/train_spinn_classifier.sh -q " + queue + " -l host=jagupardX"
    print
