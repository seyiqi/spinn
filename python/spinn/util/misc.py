import numpy as np
from collections import deque
import os

from spinn.util.sparks import sparks


class GenericClass(object):
    def __init__(self, **kwargs):
        super(GenericClass, self).__init__()
        for k, v in kwargs.iteritems():
            setattr(self, k, v)

    def __repr__(self):
        s = "{}"
        return s.format(self.__dict__)


class Args(GenericClass): pass


class EncodeArgs(GenericClass): pass


class Vocab(GenericClass): pass


class Example(GenericClass): pass


def time_per_token(num_tokens, total_time):
    return sum(total_time) / float(sum(num_tokens))


class Accumulator(object):
    """Accumulator. Makes it easy to keep a trailing list of statistics."""

    def __init__(self, maxlen=None):
        self.maxlen = maxlen
        self.cache = dict()

    def add(self, key, val):
        self.cache.setdefault(key, deque(maxlen=self.maxlen)).append(val)

    def get(self, key, clear=True):
        ret = self.cache.get(key, [])
        if clear:
            try:
                del self.cache[key]
            except:
                pass
        return ret

    def get_avg(self, key, clear=True):
        return np.array(self.get(key, clear)).mean()


class MetricsLogger(object):
    """MetricsLogger."""

    def __init__(self, metrics_path):
        self.metrics_path = metrics_path

    def Log(self, key, val, step):
        log_path = os.path.join(self.metrics_path, key) + ".metrics"
        with open(log_path, 'a') as f:
            f.write("{} {}\n".format(step, val))


class EvalReporter(object):
    def __init__(self):
        self.batches = []

    def save_batch(self, reporter_args):

        preds = reporter_args["preds"]
        target = reporter_args["target"]
        example_ids = reporter_args["example_ids"]

        reporter_args.setdefault("batch_size", target.size(0))
        reporter_args.setdefault("sent1_transitions", [None] * len(example_ids))
        reporter_args.setdefault("sent2_transitions", [None] * len(example_ids))
        reporter_args.setdefault("sent1_strength", [None] * len(example_ids))
        reporter_args.setdefault("sent2_strength", [None] * len(example_ids))
        reporter_args.setdefault("sent1_transitions_given", [None] * len(example_ids))
        reporter_args.setdefault("sent2_transitions_given", [None] * len(example_ids))

        self.batches.append(reporter_args)

    def write_report(self, filename):
        with open(filename, 'w') as f:
            for b in self.batches:

                batch_size = b["batch_size"]
                preds = b["preds"]
                target = b["target"]
                example_ids = b["example_ids"]
                _output = b["output"]
                _dist = b["dist"]
                _sent1_transitions = b["sent1_transitions"]
                _sent2_transitions = b["sent2_transitions"]
                _sent1_strength = b["sent1_strength"]
                _sent2_strength = b["sent2_strength"]
                _sent1_transitions_given = b["sent1_transitions_given"]
                _sent2_transitions_given = b["sent2_transitions_given"]

                for i in range(batch_size):
                    filter_skip = lambda t: t != 2

                    pred = preds[i]
                    truth = target[i]
                    eid = example_ids[i]
                    output = _output[i]
                    dist = _dist[i]
                    sent1_transitions = filter(filter_skip, _sent1_transitions[i])
                    sent2_transitions = filter(filter_skip, _sent2_transitions[i])
                    sent1_transitions_given = filter(filter_skip, _sent1_transitions_given[i])
                    sent2_transitions_given = filter(filter_skip, _sent2_transitions_given[i])
                    sent1_strength = _sent1_strength[i][-len(sent1_transitions_given):]
                    sent2_strength = _sent2_strength[i][-len(sent2_transitions_given):]

                    report_dict = {
                        "eid": eid,
                        "correct": truth == pred,
                        "truth": truth,
                        "pred": pred,
                        "output": " ".join([str(o) for o in output]),
                        "dist": " ".join([str(o) for o in dist]),
                        "sent1_transitions": '{}'.format("".join(str(t) for t in sent1_transitions)) if sent1_transitions is not None else None,
                        "sent2_transitions": '{}'.format("".join(str(t) for t in sent2_transitions)) if sent2_transitions is not None else None,
                        "sent1_strength": sparks([1] + sent1_strength.tolist())[1:].encode('utf-8') if sent1_strength is not None else None,
                        "sent2_strength": sparks([1] + sent2_strength.tolist())[1:].encode('utf-8') if sent2_strength is not None else None,
                        "sent1_transitions_given": '{}'.format("".join(str(t) for t in sent1_transitions_given)) if sent1_transitions_given is not None else None,
                        "sent2_transitions_given": '{}'.format("".join(str(t) for t in sent2_transitions_given)) if sent2_transitions_given is not None else None,
                    }

                    report_str = "{eid} {correct} {truth} {pred} ({output}) ({dist})"

                    report_str_len = len(report_str.format(**report_dict))

                    if sent1_strength is not None:
                        report_str += "\n{sent1_strength}"
                    if sent2_strength is not None:
                        report_str += " {sent2_strength}"

                    if sent1_transitions is not None:
                        report_str += "\n{sent1_transitions}"
                    if sent2_transitions is not None:
                        report_str += " {sent2_transitions}"
                        
                    if sent1_transitions_given is not None:
                        report_str += "\n{sent1_transitions_given}"
                    if sent2_transitions_given is not None:
                        report_str += " {sent2_transitions_given}"

                    report_str += "\n"

                    f.write(report_str.format(**report_dict))


def recursively_set_device(inp, gpu):
    if hasattr(inp, 'keys'):
        for k in inp.keys():
            inp[k] = recursively_set_device(inp[k], gpu)
    elif isinstance(inp, list):
        return [recursively_set_device(ii, gpu) for ii in inp]
    elif isinstance(inp, tuple):
        return (recursively_set_device(ii, gpu) for ii in inp)
    elif hasattr(inp, 'cpu'):
        if gpu >= 0:
            inp = inp.cuda()
        else:
            inp = inp.cpu()
    return inp
