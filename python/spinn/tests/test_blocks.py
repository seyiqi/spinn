import unittest
import argparse

from nose.plugins.attrib import attr
import numpy as np

import pytest

from spinn import util
from spinn.fat_stack import SPINN

# Chainer imports
import chainer
from chainer import reporter, initializers
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
from chainer.functions.connection import embed_id
from chainer.functions.normalization.batch_normalization import batch_normalization
from chainer.functions.evaluation import accuracy
import chainer.links as L
from chainer.training import extensions

import spinn.util.chainer_blocks as blocks


class BlocksTestCase(unittest.TestCase):

    def test_expand_along(self):
        mock_rewards = np.array([1.0, 0.0, 2.0])
        mock_mask = np.array([[True, True], [False, True], [False, False]])
        ret = blocks.expand_along(mock_rewards, mock_mask)
        expected = [1., 1., 0.]
        assert len(ret) == len(expected)
        assert all(r == e for r, e in zip(ret, expected))

    def test_model_save(self):
        c = Chain()
        c.add_persistent('baseline', 2)
        c2 = c.copy()

        s = serializers.DictionarySerializer()
        s.save(c)

        d = serializers.NpzDeserializer(s.target)
        d.load(c2)

        assert c.baseline == c2.baseline


if __name__ == '__main__':
    unittest.main()
