# Create a script to run a random hyperparameter search.

import copy
import getpass
import os
import random
import numpy as np

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
}

# Tunable parameters.
SWEEP_PARAMETERS = {
    "embedding_keep_rate": (LIN, 0.7, 0.95),
    "enc_embedding_dim":   (LIN, 100, 300),
    "l2_lambda":          (EXP, 8e-7, 2e-5),
    "learning_rate":      (EXP, 0.0002, 0.01),  # RNN likes higher end of range, but below 009.
    "num_sentence_pair_combination_layers": (LIN, 1, 3),
    "scheduled_sampling_exponent_base": (SS_BASE, 1e-5, 8e-5),
    "semantic_classifier_keep_rate": (LIN, 0.90, 0.98),  # NB: Keep rates may depend considerably on dims.
    "tracking_lstm_hidden_dim": (EXP, 40, 128),
    "transition_cost_scale": (LIN, 0.5, 4.0),
}

sweep_runs = 4

# - #
print "# SWEEP PARAMETERS: " + str(SWEEP_PARAMETERS)
print "# FIXED_PARAMETERS: " + str(FIXED_PARAMETERS)
print

for run_id in range(sweep_runs):
    params = {}
    params.update(FIXED_PARAMETERS)
    for param in SWEEP_PARAMETERS:
        config = SWEEP_PARAMETERS[param]
        t = config[0]
        mn = config[1]
        mx = config[2]

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

        params[param] = sample

    flags = " \\\n"
    for param in params:
        value = params[param]
        val_str = ""
        flags += " --" + param + " " + str(value)
        flags += " \\\n"
        if param not in FIXED_PARAMETERS:
            if isinstance(value, int):
                val_disp = str(value)
            else:
                val_disp = "%.2g" % value
    print "export SPINN_FLAGS=\"" + flags + "\";\n"
    print
