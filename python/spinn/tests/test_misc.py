import unittest
import tempfile
import math

from nose.plugins.attrib import attr
import numpy as np

import pytest

from spinn import util
from spinn.fat_stack import SPINN

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from spinn.util.misc import Accumulator
from spinn.util.misc import complete_tree


class MiscTestCase(unittest.TestCase):

    def test_accumulator(self):
        A = Accumulator()

        A.add('key', 0)
        A.add('key', 0)

        assert len(A.get('key')) == 2
        assert len(A.get('key')) == 0

    def test_complete_tree(self):
        padto = 60
        for n in range(1, 31):
            ts = complete_tree(n, padto)
            most = 0
            assert len(ts) == padto
            for i in range(1, len(ts)):
                if ts[i-1] == 0 and ts[i] == 0:
                    most += 1
                else:
                    most = 0
                assert most < 2


if __name__ == '__main__':
    unittest.main()
