import itertools

import numpy as np
from spinn import util

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from spinn.util.blocks import LSTMState, Embed, MLP, Linear, LSTM
from spinn.util.blocks import reverse_tensor
from spinn.util.blocks import bundle, unbundle, to_cpu, to_gpu, treelstm, lstm
from spinn.util.blocks import get_h, get_c
from spinn.util.misc import Args, Vocab, Example
from spinn.util.blocks import HeKaimingInitializer

from spinn.data import T_SHIFT, T_REDUCE, T_SKIP, T_STRUCT


def build_model(data_manager, initial_embeddings, vocab_size, num_classes, FLAGS):
    model_cls = BaseModel
    use_sentence_pair = data_manager.SENTENCE_PAIR_DATA

    return model_cls(model_dim=FLAGS.model_dim,
         word_embedding_dim=FLAGS.word_embedding_dim,
         vocab_size=vocab_size,
         initial_embeddings=initial_embeddings,
         num_classes=num_classes,
         embedding_keep_rate=FLAGS.embedding_keep_rate,
         encode_style=FLAGS.encode_style,
         encode_reverse=FLAGS.encode_reverse,
         encode_bidirectional=FLAGS.encode_bidirectional,
         encode_num_layers=FLAGS.encode_num_layers,
         use_sentence_pair=use_sentence_pair,
         use_lengths=FLAGS.use_lengths,
         use_difference_feature=FLAGS.use_difference_feature,
         use_product_feature=FLAGS.use_product_feature,
         classifier_keep_rate=FLAGS.semantic_classifier_keep_rate,
         mlp_dim=FLAGS.mlp_dim,
         num_mlp_layers=FLAGS.num_mlp_layers,
         mlp_bn=FLAGS.mlp_bn,
        )


class Reduce(nn.Module):
    def __init__(self, size):
        super(Reduce, self).__init__()
        self.left = Linear(initializer=HeKaimingInitializer)(size, 5 * size)
        self.right = Linear(initializer=HeKaimingInitializer)(size, 5 * size, bias=False)

    def forward(self, left_in, right_in):
        left, right = bundle(left_in), bundle(right_in)
        lstm_in = self.left(left.h)
        lstm_in += self.right(right.h)
        out = unbundle(treelstm(left.c, right.c, lstm_in, training=self.training))
        return out


class SPINN(nn.Module):

    def __init__(self, args):
        super(SPINN, self).__init__()

        # Optional debug mode.
        self.debug = False

        # Reduce function for semantic composition.
        self.reduce = Reduce(args.size)

    def reset_state(self):
        self.memories = []

    def forward(self, example, use_internal_parser=False, validate_transitions=True):
        self.buffers_n = (example.tokens.data != 0).long().sum(1).view(-1).tolist()

        if self.debug:
            seq_length = example.tokens.size(1)
            assert all(buf_n <= (seq_length + 1) // 2 for buf_n in self.buffers_n), \
                "All sentences (including cropped) must be the appropriate length."

        self.bufs = example.bufs

        # Trim unused tokens.
        self.bufs = [b[-b_n:] for b, b_n in zip(self.bufs, self.buffers_n)]

        self.stacks = [[] for buf in self.bufs]

        if not hasattr(example, 'transitions'):
            # TODO: Support no transitions. In the meantime, must at least pass dummy transitions.
            raise ValueError('Transitions must be included.')
        return self.run(example.transitions,
                        run_internal_parser=True,
                        use_internal_parser=use_internal_parser,
                        validate_transitions=validate_transitions)

    def t_shift(self, buf, stack, buf_tops):
        """SHIFT: Should dequeue buffer and item to stack."""
        buf_tops.append(buf.pop())

    def t_reduce(self, buf, stack, lefts, rights):
        """REDUCE: Should compose top two items of the stack into new item."""

        # The right-most input will be popped first.
        for reduce_inp in [rights, lefts]:
            reduce_inp.append(stack.pop())

    def t_skip(self):
        """SKIP: Acts as padding and is a noop."""
        pass

    def shift_phase(self, tops, stacks, idxs):
        """SHIFT: Should dequeue buffer and item to stack."""
        if len(stacks) > 0:
            shift_candidates = iter(tops)
            for stack in stacks:
                new_stack_item = next(shift_candidates)
                stack.append(new_stack_item)

    def reduce_phase(self, lefts, rights, stacks):
        if len(stacks) > 0:
            reduced = iter(self.reduce(lefts, rights))
            for stack in stacks:
                new_stack_item = next(reduced)
                stack.append(new_stack_item)

    def reduce_phase_hook(self, lefts, rights, reduce_stacks):
        pass

    def loss_phase_hook(self):
        pass

    def run(self, inp_transitions, run_internal_parser=False, use_internal_parser=False, validate_transitions=True):
        num_transitions = inp_transitions.shape[1]
        batch_size = inp_transitions.shape[0]

        # Transition Loop
        # ===============

        for t_step in range(num_transitions):
            transitions = inp_transitions[:, t_step]
            transition_arr = list(transitions)

            # Memories
            # ========
            # Keep track of key values to determine accuracy and loss.
            self.memory = {}

            # Pre-Action Phase
            # ================

            # For SHIFT
            s_stacks, s_tops, s_idxs = [], [], []

            # For REDUCE
            r_stacks, r_lefts, r_rights = [], [], []

            batch = zip(transition_arr, self.bufs, self.stacks)

            for batch_idx, (transition, buf, stack) in enumerate(batch):
                if transition == T_SHIFT: # shift
                    self.t_shift(buf, stack, s_tops)
                    s_idxs.append(batch_idx)
                    s_stacks.append(stack)
                elif transition == T_REDUCE: # reduce
                    self.t_reduce(buf, stack, r_lefts, r_rights)
                    r_stacks.append(stack)
                elif transition == T_SKIP: # skip
                    self.t_skip()

            # Action Phase
            # ============

            self.shift_phase(s_tops, s_stacks, s_idxs)
            self.reduce_phase(r_lefts, r_rights, r_stacks)
            self.reduce_phase_hook(r_lefts, r_rights, r_stacks)

            # Memory Phase
            # ============

            # APPEND ALL MEMORIES. MASK LATER.

            self.memories.append(self.memory)

        self.loss_phase_hook()

        if self.debug:
            assert all(len(stack) == 3 for stack in self.stacks), \
                "Stacks should be fully reduced and have 3 elements: " \
                "two zeros and the sentence encoding."
            assert all(len(buf) == 1 for buf in self.bufs), \
                "Stacks should be fully shifted and have 1 zero."

        return [stack[-1] for stack in self.stacks]


class BaseModel(nn.Module):

    def __init__(self, model_dim=None,
                 word_embedding_dim=None,
                 vocab_size=None,
                 initial_embeddings=None,
                 num_classes=None,
                 embedding_keep_rate=None,
                 encode_style=None,
                 encode_reverse=None,
                 encode_bidirectional=None,
                 encode_num_layers=None,
                 use_sentence_pair=False,
                 use_difference_feature=False,
                 use_product_feature=False,
                 mlp_dim=None,
                 num_mlp_layers=None,
                 mlp_bn=None,
                 classifier_keep_rate=None,
                 use_projection=None,
                 **kwargs
                ):
        super(BaseModel, self).__init__()

        self.use_sentence_pair = use_sentence_pair
        self.use_difference_feature = use_difference_feature
        self.use_product_feature = use_product_feature
        self.hidden_dim = hidden_dim = model_dim / 2

        args = Args()
        args.size = model_dim/2

        self.initial_embeddings = initial_embeddings
        self.word_embedding_dim = word_embedding_dim
        self.model_dim = model_dim
        classifier_dropout_rate = 1. - classifier_keep_rate

        vocab = Vocab()
        vocab.size = initial_embeddings.shape[0] if initial_embeddings is not None else vocab_size
        vocab.vectors = initial_embeddings

        # Build parsing component.
        self.spinn = self.build_spinn(args)

        # Build classiifer.
        features_dim = self.get_features_dim()
        self.mlp = MLP(features_dim, mlp_dim, num_classes,
            num_mlp_layers, mlp_bn, classifier_dropout_rate)

        self.embedding_dropout_rate = 1. - embedding_keep_rate

        # Projection will effectively be done by the encoding network.
        use_projection = True if encode_style is None else False
        input_dim = model_dim if use_projection else word_embedding_dim

        # Create dynamic embedding layer.
        self.embed = Embed(input_dim, vocab.size, vectors=vocab.vectors, use_projection=use_projection)

        # Optionally build input encoder.
        if encode_style is not None:
            self.encode = self.build_input_encoder(encode_style=encode_style,
                word_embedding_dim=word_embedding_dim, model_dim=model_dim,
                num_layers=encode_num_layers, bidirectional=encode_bidirectional, reverse=encode_reverse,
                dropout=self.embedding_dropout_rate)

    def get_features_dim(self):
        features_dim = self.hidden_dim * 2 if self.use_sentence_pair else self.hidden_dim
        if self.use_sentence_pair:
            if self.use_difference_feature:
                features_dim += self.hidden_dim
            if self.use_product_feature:
                features_dim += self.hidden_dim
        return features_dim

    def build_features(self, h):
        if self.use_sentence_pair:
            h_prem, h_hyp = h
            features = [h_prem, h_hyp]
            if self.use_difference_feature:
                features.append(h_prem - h_hyp)
            if self.use_product_feature:
                features.append(h_prem * h_hyp)
            features = torch.cat(features, 1)
        else:
            features = h[0]
        return features

    def build_input_encoder(self, encode_style="LSTM", word_embedding_dim=None, model_dim=None,
                            num_layers=None, bidirectional=None, reverse=None, dropout=None):
        if encode_style == "LSTM":
            encoding_net = LSTM(word_embedding_dim, model_dim,
                num_layers=num_layers, bidirectional=bidirectional, reverse=reverse,
                dropout=dropout)
        else:
            raise NotImplementedError
        return encoding_net

    def build_spinn(self, args):
        return SPINN(args)

    def run_spinn(self, example, use_internal_parser, validate_transitions=True):
        self.spinn.reset_state()
        h_list = self.spinn(example,
                               use_internal_parser=use_internal_parser,
                               validate_transitions=validate_transitions)
        h = self.wrap(h_list)
        return h

    def output_hook(self, output, sentences, transitions, y_batch=None, embeds=None):
        pass

    def forward(self, sentences, transitions, y_batch=None,
                 use_internal_parser=False, validate_transitions=True):
        example = self.unwrap(sentences, transitions)

        b, l = example.tokens.size()[:2]

        embeds = self.embed(example.tokens)
        embeds = F.dropout(embeds, self.embedding_dropout_rate, training=self.training)
        embeds = torch.chunk(to_cpu(embeds), b, 0)

        if hasattr(self, 'encode'):
            to_encode = torch.cat([e.unsqueeze(0) for e in embeds], 0)
            encoded = self.encode(to_encode)
            embeds = [x.squeeze() for x in torch.chunk(encoded, b, 0)]

        # Make Buffers
        embeds = [torch.chunk(x, l, 0) for x in embeds]
        buffers = [list(reversed(x)) for x in embeds]

        example.bufs = buffers

        h = self.run_spinn(example, use_internal_parser, validate_transitions)

        self.spinn_outp = h

        # Build features
        features = self.build_features(h)

        output = self.mlp(features)

        self.output_hook(output, sentences, transitions, y_batch, embeds)

        return output

    # --- Sentence Style Switches ---

    def unwrap(self, sentences, transitions):
        if self.use_sentence_pair:
            return self.unwrap_sentence_pair(sentences, transitions)
        return self.unwrap_sentence(sentences, transitions)

    def wrap(self, h_list):
        if self.use_sentence_pair:
            return self.wrap_sentence_pair(h_list)
        return self.wrap_sentence(h_list)

    # --- Sentence Model Specific ---

    def unwrap_sentence(self, sentences, transitions):
        batch_size = sentences.shape[0]

        # Build Tokens
        x = sentences

        # Build Transitions
        t = transitions

        example = Example()
        example.tokens = to_gpu(Variable(torch.from_numpy(x), volatile=not self.training))
        example.transitions = t

        return example

    def wrap_sentence(self, h_list):
        batch_size = len(h_list) / 2
        h = get_h(torch.cat(h_list, 0), self.hidden_dim)
        return [h]

    # --- Sentence Pair Model Specific ---

    def unwrap_sentence_pair(self, sentences, transitions):
        batch_size = sentences.shape[0]

        # Build Tokens
        x_prem = sentences[:,:,0]
        x_hyp = sentences[:,:,1]
        x = np.concatenate([x_prem, x_hyp], axis=0)

        # Build Transitions
        t_prem = transitions[:,:,0]
        t_hyp = transitions[:,:,1]
        t = np.concatenate([t_prem, t_hyp], axis=0)

        example = Example()
        example.tokens = to_gpu(Variable(torch.from_numpy(x), volatile=not self.training))
        example.transitions = t

        return example

    def wrap_sentence_pair(self, h_list):
        batch_size = len(h_list) / 2
        h_premise = get_h(torch.cat(h_list[:batch_size], 0), self.hidden_dim)
        h_hypothesis = get_h(torch.cat(h_list[batch_size:], 0), self.hidden_dim)
        return [h_premise, h_hypothesis]
