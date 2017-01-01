import unittest

import numpy as np

import os
from spinn import util
from spinn.data.snli import load_snli_data
from collections import Counter


data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_snli.jsonl")
embedding_data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_embedding_matrix.5d.txt")
word_embedding_dim = 5

class MockLogger(object):
    def Log(self, *args, **kwargs):
        pass
        

class SNLITestCase(unittest.TestCase):

    def test_load(self):
        data_manager = load_snli_data
        raw_data, _ = data_manager.load_data(data_path)
        assert len(raw_data) == 20

        hyp_seq_lengths = Counter([len(x['hypothesis_transitions'])
                        for x in raw_data])
        assert hyp_seq_lengths == {13: 4, 15: 4, 11: 2, 17: 2, 23: 2, 5: 1, 39: 1, 9: 1, 19: 1, 7: 1, 29: 1}

        prem_seq_lengths = Counter([len(x['premise_transitions'])
                        for x in raw_data])
        assert prem_seq_lengths == {35: 8, 67: 3, 19: 3, 53: 3, 33: 3}

        min_seq_lengths = Counter([min(len(x['hypothesis_transitions']), len(x['premise_transitions']))
                        for x in raw_data])
        assert min_seq_lengths == {5: 1, 7: 1, 9: 1, 11: 2, 13: 4, 15: 4, 17: 2, 19: 2, 23: 2, 35: 1}

    def test_vocab(self):
        data_manager = load_snli_data
        raw_data, _ = data_manager.load_data(data_path)
        data_sets = [(data_path, raw_data)]
        vocabulary = util.BuildVocabulary(
            raw_data, data_sets, embedding_data_path, logger=MockLogger(),
            sentence_pair_data=data_manager.SENTENCE_PAIR_DATA)
        assert len(vocabulary) == 10

    def test_load_embed(self):
        data_manager = load_snli_data
        raw_data, _ = data_manager.load_data(data_path)
        data_sets = [(data_path, raw_data)]
        vocabulary = util.BuildVocabulary(
            raw_data, data_sets, embedding_data_path, logger=MockLogger(),
            sentence_pair_data=data_manager.SENTENCE_PAIR_DATA)
        initial_embeddings = util.LoadEmbeddingsFromASCII(
            vocabulary, word_embedding_dim, embedding_data_path)
        assert initial_embeddings.shape == (10, 5)

    def test_preprocess(self):
        seq_length = 25
        for_rnn = False
        use_left_padding = True

        data_manager = load_snli_data
        raw_data, _ = data_manager.load_data(data_path)
        data_sets = [(data_path, raw_data)]
        vocabulary = util.BuildVocabulary(
            raw_data, data_sets, embedding_data_path, logger=MockLogger(),
            sentence_pair_data=data_manager.SENTENCE_PAIR_DATA)
        initial_embeddings = util.LoadEmbeddingsFromASCII(
            vocabulary, word_embedding_dim, embedding_data_path)

        data = util.PreprocessDataset(
            raw_data, vocabulary, seq_length, data_manager, eval_mode=False, logger=MockLogger(),
            sentence_pair_data=data_manager.SENTENCE_PAIR_DATA,
            for_rnn=for_rnn, use_left_padding=use_left_padding)

        # Filter pairs that don't have both hyp and prem transition length <= seq_length
        assert len(data) == 4

if __name__ == '__main__':
    unittest.main()
