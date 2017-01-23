from functools import partial
import os
import pprint
import sys
import time
from collections import deque

# Chainer Type Check is a major bottleneck for small networks,
# sometimes giving more than 4x speedup by turning it off.
# We turn type check off by default, but can set an env var to
# turn it on if debugging.
if os.environ.get('FORCE_CHAINER_TYPE_CHECK', '0') == '0':
    os.environ['CHAINER_TYPE_CHECK'] = '0'

import gflags
import numpy as np

from spinn import afs_safe_logger
from spinn import util
from spinn.data.arithmetic import load_simple_data
from spinn.data.boolean import load_boolean_data
from spinn.data.sst import load_sst_data
from spinn.data.snli import load_snli_data
from spinn.util.data import SimpleProgressBar
from spinn.util.chainer_blocks import gradient_check, l2_cost, flatten

import spinn.fat_stack
import spinn.plain_rnn
import spinn.cbow
import spinn.nti

# Try to avoid chainer imports as much as possible.
from chainer import optimizers
import chainer.functions as F

from spinn.util.data import print_tree
from sklearn import metrics


FLAGS = gflags.FLAGS


def build_sentence_pair_model(model_cls, trainer_cls, vocab_size, model_dim, word_embedding_dim,
                              seq_length, num_classes, initial_embeddings, use_sentence_pair,
                              gpu, mlp_dim):
    model = model_cls(model_dim, word_embedding_dim, vocab_size,
             seq_length, initial_embeddings, num_classes, mlp_dim=mlp_dim,
             input_keep_rate=FLAGS.embedding_keep_rate,
             classifier_keep_rate=FLAGS.semantic_classifier_keep_rate,
             use_input_norm=FLAGS.use_input_norm,
             tracker_keep_rate=FLAGS.tracker_keep_rate,
             tracking_lstm_hidden_dim=FLAGS.tracking_lstm_hidden_dim,
             transition_weight=FLAGS.transition_weight,
             use_tracking_lstm=FLAGS.use_tracking_lstm,
             use_sentence_pair=use_sentence_pair,
             num_mlp_layers=FLAGS.num_mlp_layers,
             mlp_bn=FLAGS.mlp_bn,
             gpu=gpu,
             use_skips=FLAGS.use_skips,
             use_encode=FLAGS.use_encode,
             projection_dim=FLAGS.projection_dim,
             use_difference_feature=FLAGS.use_difference_feature,
             use_product_feature=FLAGS.use_product_feature,
            )

    classifier_trainer = trainer_cls(model, gpu=gpu)

    return classifier_trainer

def hamming_distance(s1, s2):
    """ source: https://en.wikipedia.org/wiki/Hamming_distance
        Return the Hamming distance between equal-length sequences
    """
    if len(s1) != len(s2):
        raise ValueError("Undefined for sequences of unequal length")
    return sum(el1 != el2 for el1, el2 in zip(s1, s2))


def evaluate(classifier_trainer, eval_set, logger, step, eval_data_limit=-1,
             use_internal_parser=False, vocabulary=None):
    # Evaluate
    acc_accum = 0.0
    action_acc_accum = 0.0
    eval_batches = 0.0
    total_batches = len(eval_set[1])
    progress_bar = SimpleProgressBar(msg="Run Eval", bar_length=60, enabled=FLAGS.show_progress_bar)
    progress_bar.step(0, total=total_batches)

    accum_preds = deque()
    accum_truth = deque()
    model = classifier_trainer.optimizer.target
    evalb_parses = []
    parses = []
    for i, (eval_X_batch, eval_transitions_batch, eval_y_batch, eval_num_transitions_batch) in enumerate(eval_set[1]):
        # Calculate Local Accuracies
        if eval_data_limit == -1 or i < eval_data_limit:
            ret = classifier_trainer.forward({
                "sentences": eval_X_batch,
                "transitions": eval_transitions_batch,
                }, eval_y_batch, train=False, predict=False,
                use_internal_parser=use_internal_parser,
                use_reinforce=False,
                validate_transitions=FLAGS.validate_transitions,
                use_random=FLAGS.use_random)
            y, loss, class_loss, transition_acc, transition_loss = ret
            acc_value = float(classifier_trainer.model.accuracy.data)

            if transition_loss is not None:
                preds = [m["preds_cm"] for m in model.spinn.memories]
                truth = [m["truth_cm"] for m in model.spinn.memories]
                accum_preds.append(preds)
                accum_truth.append(truth)
        else:
            break

        # Update Aggregate Accuracies
        acc_accum += acc_value
        eval_batches += 1.0

        if FLAGS.print_tree:
            memories = classifier_trainer.model.spinn.memories
            all_preds = [el['preds'] for el in memories]
            all_preds = zip(*all_preds)

            if vocabulary is not None:
                inv_vocab = {v: k for k, v in vocabulary.iteritems()}
            
            for ii in range(len(eval_X_batch)):
                sentence = eval_X_batch[ii]
                ground_truth = eval_transitions_batch[ii]
                predicted = all_preds[ii]
                evalb_parses.append(print_tree(sentence, ground_truth, predicted, inv_vocab, evalb=True))
                parses.append(print_tree(sentence, ground_truth, predicted, inv_vocab, evalb=False))

        # Print Progress
        progress_bar.step(i+1, total=total_batches)
    progress_bar.finish()

    # Print Trees to file for use in Error Analysis
    if FLAGS.print_tree:
        gld_filename = "{}.trees.gld".format(FLAGS.experiment_name)
        tst_filename = "{}.trees.tst".format(FLAGS.experiment_name)
        both_filename = "{}.trees.txt".format(FLAGS.experiment_name)
        with open(gld_filename, "w") as f:
            for ground_truth_tree, predicted_tree in evalb_parses:
                f.write("(S {})\n".format(ground_truth_tree))
        with open(tst_filename, "w") as f:
            for ground_truth_tree, predicted_tree in evalb_parses:
                f.write("(S {})\n".format(predicted_tree))
        with open(both_filename, "w") as f:
            for ground_truth_tree, predicted_tree in parses:
                f.write("{}\t{}\n".format(ground_truth_tree, predicted_tree))

    # Accumulate Action Accuracy this way because of the UseSkips/NoUseSkips toggle.
    all_preds = flatten(accum_preds)
    all_truth = flatten(accum_truth)
    if transition_loss is not None:
        trans_acc = metrics.accuracy_score(all_preds, all_truth) if len(all_preds) > 0 else 0.0
    else:
        trans_acc = 0.0

    logger.Log("Step: %i\tEval acc: %f\t %f\t%s" %
              (step, acc_accum / eval_batches, trans_acc, eval_set[0]))
    return acc_accum / eval_batches


def run(only_forward=False):
    logger = afs_safe_logger.Logger(os.path.join(FLAGS.log_path, FLAGS.experiment_name) + ".log")

    if FLAGS.data_type == "bl":
        data_manager = load_boolean_data
    elif FLAGS.data_type == "sst":
        data_manager = load_sst_data
    elif FLAGS.data_type == "snli":
        data_manager = load_snli_data
    elif FLAGS.data_type == "arithmetic":
        data_manager = load_simple_data
    else:
        logger.Log("Bad data type.")
        return

    pp = pprint.PrettyPrinter(indent=4)
    logger.Log("Flag values:\n" + pp.pformat(FLAGS.FlagValuesDict()))

    # Load the data.
    raw_training_data, vocabulary = data_manager.load_data(
        FLAGS.training_data_path)

    # Load the eval data.
    raw_eval_sets = []
    if FLAGS.eval_data_path:
        for eval_filename in FLAGS.eval_data_path.split(":"):
            eval_data, _ = data_manager.load_data(eval_filename)
            raw_eval_sets.append((eval_filename, eval_data))

    # Prepare the vocabulary.
    if not vocabulary:
        logger.Log("In open vocabulary mode. Using loaded embeddings without fine-tuning.")
        train_embeddings = False
        vocabulary = util.BuildVocabulary(
            raw_training_data, raw_eval_sets, FLAGS.embedding_data_path, logger=logger,
            sentence_pair_data=data_manager.SENTENCE_PAIR_DATA)
    else:
        logger.Log("In fixed vocabulary mode. Training embeddings.")
        train_embeddings = True

    # Load pretrained embeddings.
    if FLAGS.embedding_data_path:
        logger.Log("Loading vocabulary with " + str(len(vocabulary))
                   + " words from " + FLAGS.embedding_data_path)
        initial_embeddings = util.LoadEmbeddingsFromASCII(
            vocabulary, FLAGS.word_embedding_dim, FLAGS.embedding_data_path)
    else:
        initial_embeddings = None

    # Trim dataset, convert token sequences to integer sequences, crop, and
    # pad.
    logger.Log("Preprocessing training data.")
    training_data = util.PreprocessDataset(
        raw_training_data, vocabulary, FLAGS.seq_length, data_manager, eval_mode=False, logger=logger,
        sentence_pair_data=data_manager.SENTENCE_PAIR_DATA,
        for_rnn=FLAGS.model_type == "RNN" or FLAGS.model_type == "CBOW",
        use_left_padding=FLAGS.use_left_padding)
    training_data_iter = util.MakeTrainingIterator(
        training_data, FLAGS.batch_size, FLAGS.smart_batching, FLAGS.use_peano)

    eval_iterators = []
    for filename, raw_eval_set in raw_eval_sets:
        logger.Log("Preprocessing eval data: " + filename)
        e_X, e_transitions, e_y, e_num_transitions = util.PreprocessDataset(
            raw_eval_set, vocabulary, FLAGS.seq_length, data_manager, eval_mode=True, logger=logger,
            sentence_pair_data=data_manager.SENTENCE_PAIR_DATA,
            for_rnn=FLAGS.model_type == "RNN" or FLAGS.model_type == "CBOW",
            use_left_padding=FLAGS.use_left_padding)
        eval_it = util.MakeEvalIterator((e_X, e_transitions, e_y, e_num_transitions),
            FLAGS.batch_size, shuffle=FLAGS.shuffle_eval, rseed=FLAGS.shuffle_eval_seed)
        eval_iterators.append((filename, eval_it))

    # Set up the placeholders.

    logger.Log("Building model.")

    if FLAGS.model_type == "CBOW":
        model_module = spinn.cbow
    elif FLAGS.model_type == "RNN":
        model_module = spinn.plain_rnn
    elif FLAGS.model_type == "NTI":
        model_module = spinn.nti
    elif FLAGS.model_type == "SPINN":
        model_module = spinn.fat_stack
    else:
        raise Exception("Requested unimplemented model type %s" % FLAGS.model_type)


    if data_manager.SENTENCE_PAIR_DATA:
        if hasattr(model_module, 'SentencePairTrainer') and hasattr(model_module, 'SentencePairModel'):
            trainer_cls = model_module.SentencePairTrainer
            model_cls = model_module.SentencePairModel
        else:
            raise Exception("Unimplemented for model type %s" % FLAGS.model_type)

        num_classes = len(data_manager.LABEL_MAP)
        use_sentence_pair = True
        classifier_trainer = build_sentence_pair_model(model_cls, trainer_cls,
                              len(vocabulary), FLAGS.model_dim, FLAGS.word_embedding_dim,
                              FLAGS.seq_length, num_classes, initial_embeddings,
                              use_sentence_pair,
                              FLAGS.gpu,
                              FLAGS.mlp_dim)
    else:
        if hasattr(model_module, 'SentenceTrainer') and hasattr(model_module, 'SentenceModel'):
            trainer_cls = model_module.SentenceTrainer
            model_cls = model_module.SentenceModel
        else:
            raise Exception("Unimplemented for model type %s" % FLAGS.model_type)

        num_classes = len(data_manager.LABEL_MAP)
        use_sentence_pair = False
        classifier_trainer = build_sentence_pair_model(model_cls, trainer_cls,
                              len(vocabulary), FLAGS.model_dim, FLAGS.word_embedding_dim,
                              FLAGS.seq_length, num_classes, initial_embeddings,
                              use_sentence_pair,
                              FLAGS.gpu,
                              FLAGS.mlp_dim)

    if ".ckpt" in FLAGS.ckpt_path:
        checkpoint_path = FLAGS.ckpt_path
    else:
        checkpoint_path = os.path.join(FLAGS.ckpt_path, FLAGS.experiment_name + ".ckpt")

    if os.path.isfile(checkpoint_path):
        # TODO: Check that resuming works fine with tf summaries.
        logger.Log("Found checkpoint, restoring.")
        step, best_dev_error = classifier_trainer.load(checkpoint_path)
        logger.Log("Resuming at step: {} with best dev accuracy: {}".format(step, 1. - best_dev_error))
    else:
        assert not only_forward, "Can't run an eval-only run without a checkpoint. Supply a checkpoint."
        step = 0
        best_dev_error = 1.0

    if FLAGS.write_summaries:
        from spinn.tf_logger import TFLogger
        train_summary_logger = TFLogger(summary_dir=os.path.join(FLAGS.summary_dir, FLAGS.experiment_name, 'train'))
        dev_summary_logger = TFLogger(summary_dir=os.path.join(FLAGS.summary_dir, FLAGS.experiment_name, 'dev'))

    # Setup Trainer
    classifier_trainer.init_optimizer(
        lr=FLAGS.learning_rate,
        clip=FLAGS.clipping_max_value,
        opt=FLAGS.opt,
        )

    model = classifier_trainer.optimizer.target

    # Do an evaluation-only run.
    if only_forward:
        for index, eval_set in enumerate(eval_iterators):
            acc = evaluate(classifier_trainer, eval_set, logger, step,
                use_internal_parser=FLAGS.use_internal_parser,
                vocabulary=vocabulary,
                eval_data_limit=FLAGS.eval_data_limit)
    else:
         # Train
        logger.Log("Training.")

        # New Training Loop
        progress_bar = SimpleProgressBar(msg="Training", bar_length=60, enabled=FLAGS.show_progress_bar)
        accum_class_preds = deque(maxlen=FLAGS.deq_length)
        accum_class_truth = deque(maxlen=FLAGS.deq_length)
        accum_class_acc = deque(maxlen=FLAGS.deq_length)
        accum_preds = deque(maxlen=FLAGS.deq_length)
        accum_truth = deque(maxlen=FLAGS.deq_length)
        printed_total_weights = False
        for step in range(step, FLAGS.training_steps):
            X_batch, transitions_batch, y_batch, _ = training_data_iter.next()

            # Reset cached gradients.
            classifier_trainer.optimizer.zero_grads()

            # Calculate loss and update parameters.
            ret = classifier_trainer.forward({
                "sentences": X_batch,
                "transitions": transitions_batch,
                }, y_batch, train=True, predict=False,
                    validate_transitions=FLAGS.validate_transitions,
                    use_internal_parser=FLAGS.use_internal_parser,
                    use_reinforce=FLAGS.use_reinforce,
                    rl_style=FLAGS.rl_style,
                    use_random=FLAGS.use_random)
            y, xent_loss, class_acc, transition_acc, transition_loss = ret

            xent_loss *= FLAGS.y_lambda

            accum_class_preds.append(y.data.argmax(axis=1))
            accum_class_truth.append(y_batch)

            if not printed_total_weights:
                printed_total_weights = True
                def prod(l):
                    return reduce(lambda x, y: x * y, l, 1.0)
                total_weights = sum([prod(w.shape) for w in model.params()])
                logger.Log("Total Weights: {}".format(total_weights))

            if transition_loss is not None:
                preds = [m["preds_cm"] for m in model.spinn.memories]
                truth = [m["truth_cm"] for m in model.spinn.memories]
                accum_preds.append(preds)
                accum_truth.append(truth)

            # Boilerplate for calculating loss.
            transition_cost_val = transition_loss.data if transition_loss is not None else 0.0
            accum_class_acc.append(class_acc)

            # Extract L2 Cost
            l2_loss = l2_cost(model, FLAGS.l2_lambda)

            # Accumulate Total Loss Data
            total_cost_val = 0.0
            total_cost_val += xent_loss.data
            total_cost_val += l2_loss.data
            total_cost_val += transition_cost_val

            # Accumulate Total Loss Variable
            total_loss = 0.0
            total_loss += xent_loss
            total_loss += l2_loss
            if hasattr(transition_loss, 'backward'):
                total_loss += transition_loss

            # Get gradients
            total_loss.backward()

            # Apply gradients
            classifier_trainer.update()

            if FLAGS.use_lr_decay:
                try:
                    # Update Learning Rate
                    learning_rate = FLAGS.learning_rate * (FLAGS.learning_rate_decay_per_10k_steps ** (step / 10000.0))
                    classifier_trainer.optimizer.lr = learning_rate
                except AttributeError:
                    # Some optimizers (like Adam) do not allow you to set learning rate this way.
                    # Fortunately, they tend to have some sort of built-in decay.
                    pass


            # Accumulate accuracy for current interval.
            acc_val = float(classifier_trainer.model.accuracy.data)

            if FLAGS.write_summaries:
                train_summary_logger.log(step=step, loss=total_cost_val, accuracy=acc_val)

            progress_bar.step(
                i=max(0, step-1) % FLAGS.statistics_interval_steps + 1,
                total=FLAGS.statistics_interval_steps)

            if step % FLAGS.statistics_interval_steps == 0:
                progress_bar.finish()
                avg_class_acc = np.array(accum_class_acc).mean()
                all_preds = flatten(accum_preds)
                all_truth = flatten(accum_truth)
                if transition_loss is not None:
                    avg_trans_acc = metrics.accuracy_score(all_preds, all_truth) if len(all_preds) > 0 else 0.0
                else:
                    avg_trans_acc = 0.0
                logger.Log(
                    "Step: %i\tAcc: %f\t%f\tCost: %5f %5f %5f %5f"
                    % (step, avg_class_acc, avg_trans_acc, total_cost_val, xent_loss.data, transition_cost_val, l2_loss.data))
                if FLAGS.transitions_confusion_matrix:
                    cm = metrics.confusion_matrix(
                        np.array(all_preds),
                        np.array(all_truth),
                        )
                    logger.Log("Transitions Confusion Matrix\n{}".format(cm))
                if FLAGS.class_confusion_matrix:
                    np.set_printoptions(threshold=np.nan)
                    np.set_printoptions(linewidth=np.nan)
                    all_class_preds = flatten(accum_class_preds)
                    all_class_truth = flatten(accum_class_truth)
                    cm = metrics.confusion_matrix(
                        np.array(all_class_preds),
                        np.array(all_class_truth),
                        )
                    logger.Log("Class Confusion Matrix\n{}".format(cm[:]))
                accum_class_preds.clear()
                accum_class_truth.clear()
                accum_class_acc.clear()
                accum_preds.clear()
                accum_truth.clear()

            if step > 0 and (step % FLAGS.ckpt_interval_steps == 0 or step % FLAGS.eval_interval_steps == 0):
                if step % FLAGS.ckpt_interval_steps == 0:
                    dev_error_threshold = best_dev_error
                else:
                    dev_error_threshold = 0.99 * best_dev_error

                for index, eval_set in enumerate(eval_iterators):
                    acc = evaluate(classifier_trainer, eval_set, logger, step, vocabulary=vocabulary if FLAGS.print_tree else None,
                        eval_data_limit=FLAGS.eval_data_limit, use_internal_parser=FLAGS.use_internal_parser)
                    if FLAGS.ckpt_on_best_dev_error and index == 0 and (1 - acc) < dev_error_threshold and step > FLAGS.ckpt_step:
                        best_dev_error = 1 - acc
                        logger.Log("Checkpointing with new best dev accuracy of %f" % acc)
                        classifier_trainer.save(checkpoint_path, step, best_dev_error)
                    if FLAGS.write_summaries:
                        dev_summary_logger.log(step=step, loss=0.0, accuracy=acc)
                progress_bar.reset()


            if FLAGS.profile and step >= FLAGS.profile_steps:
                break


if __name__ == '__main__':
    # Debug settings.
    gflags.DEFINE_bool("debug", False, "Set to True to disable debug_mode and type_checking.")
    gflags.DEFINE_bool("transitions_confusion_matrix", False, "Periodically print CM on transitions.")
    gflags.DEFINE_bool("class_confusion_matrix", False, "Periodically print CM on classes.")
    gflags.DEFINE_bool("gradient_check", False, "Randomly check that gradients match estimates.")
    gflags.DEFINE_bool("profile", False, "Set to True to quit after a few batches.")
    gflags.DEFINE_bool("write_summaries", False, "Toggle which controls whether summaries are written.")
    gflags.DEFINE_bool("show_progress_bar", True, "Turn this off when running experiments on HPC.")
    gflags.DEFINE_bool("show_intermediate_stats", False, "Print stats more frequently than regular interval."
                                                         "Mostly to retain timing with progress bar")
    gflags.DEFINE_integer("profile_steps", 3, "Specify how many steps to profile.")
    gflags.DEFINE_string("branch_name", "", "")
    gflags.DEFINE_string("sha", "", "")
    gflags.DEFINE_boolean("print_tree", False, "Print trees to file.")

    # Experiment naming.
    gflags.DEFINE_string("experiment_name", "", "")

    # Data types.
    gflags.DEFINE_enum("data_type", "snli", ["bl", "sst", "snli", "arithmetic"],
        "Which data handler and classifier to use.")

    # Where to store checkpoints
    gflags.DEFINE_string("ckpt_path", ".", "Where to save/load checkpoints. Can be either "
        "a filename or a directory. In the latter case, the experiment name serves as the "
        "base for the filename.")
    gflags.DEFINE_string("log_path", ".", "A directory in which to write logs.")
    gflags.DEFINE_string("summary_dir", ".", "A directory in which to write summaries.")

    # Data settings.
    gflags.DEFINE_string("training_data_path", None, "")
    gflags.DEFINE_string("eval_data_path", None, "Can contain multiple file paths, separated "
        "using ':' tokens. The first file should be the dev set, and is used for determining "
        "when to save the early stopping 'best' checkpoints.")
    gflags.DEFINE_integer("ckpt_step", 1000, "Steps to run before considering saving checkpoint.")
    gflags.DEFINE_integer("deq_length", 10, "Max trailing examples to use for statistics.")
    gflags.DEFINE_integer("seq_length", 30, "")
    gflags.DEFINE_integer("eval_seq_length", 30, "")
    gflags.DEFINE_boolean("use_internal_parser", False, "Use predicted parse rather than ground truth.")
    gflags.DEFINE_boolean("smart_batching", True, "Organize batches using sequence length.")
    gflags.DEFINE_boolean("use_peano", True, "A mind-blowing sorting key.")
    gflags.DEFINE_integer("eval_data_limit", -1, "Truncate evaluation set. -1 indicates no truncation.")
    gflags.DEFINE_boolean("shuffle_eval", False, "Shuffle evaluation data.")
    gflags.DEFINE_integer("shuffle_eval_seed", 123, "Seed shuffling of eval data.")
    gflags.DEFINE_string("embedding_data_path", None,
        "If set, load GloVe-formatted embeddings from here.")

    # Model architecture settings.
    gflags.DEFINE_enum("model_type", "CBOW",
                       ["CBOW", "RNN", "SPINN", "NTI"],
                       "")
    gflags.DEFINE_boolean("allow_gt_transitions_in_eval", False,
        "Whether to use ground truth transitions in evaluation when appropriate "
        "(i.e., in Model 1 and Model 2S.)")
    gflags.DEFINE_integer("gpu", -1, "")
    gflags.DEFINE_integer("model_dim", 8, "")
    gflags.DEFINE_integer("mlp_dim", 1024, "")
    gflags.DEFINE_integer("word_embedding_dim", 8, "")
    gflags.DEFINE_float("y_lambda", 1.0, "Linear scale for classification loss.")
    gflags.DEFINE_float("transition_weight", None, "")
    gflags.DEFINE_integer("tracking_lstm_hidden_dim", 4, "")
    gflags.DEFINE_boolean("use_reinforce", False, "Use RL to provide tracking lstm gradients")
    gflags.DEFINE_enum("rl_style", "zero-one", ["zero-one", "xent"], "Specify REINFORCE configuration.")
    gflags.DEFINE_boolean("use_encode", False, "Encode output of projection layer using bidirectional RNN")
    gflags.DEFINE_integer("projection_dim", -1, "Dimension for projection network.")
    gflags.DEFINE_boolean("use_skips", False, "Pad transitions with SKIP actions.")
    gflags.DEFINE_boolean("use_left_padding", True, "Pad transitions only on the RHS.")
    gflags.DEFINE_boolean("validate_transitions", True, "Constrain predicted transitions to ones"
        "that give a valid parse tree.")
    gflags.DEFINE_boolean("use_tracking_lstm", True,
        "Whether to use LSTM in the tracking unit")
    gflags.DEFINE_float("semantic_classifier_keep_rate", 0.9,
        "Used for dropout in the semantic task classifier.")
    gflags.DEFINE_float("embedding_keep_rate", 1.0,
        "Used for dropout on transformed embeddings.")
    gflags.DEFINE_boolean("use_random", False, "When predicting parse, rather than logits,"
        "use a uniform distribution over actions.")
    gflags.DEFINE_boolean("use_input_norm", False, "Apply batch normalization to transformed embeddings.")
    gflags.DEFINE_float("tracker_keep_rate", 1.0, "Keep rate for tracker input dropout.")
    gflags.DEFINE_integer("num_mlp_layers", 2, "")
    gflags.DEFINE_boolean("mlp_bn", True, "Use batch normalization within semantic classifier.")
    gflags.DEFINE_boolean("use_difference_feature", True,
        "Supply the sentence pair classifier with sentence difference features.")
    gflags.DEFINE_boolean("use_product_feature", True,
        "Supply the sentence pair classifier with sentence product features.")

    # Optimization settings.
    gflags.DEFINE_integer("training_steps", 500000, "Stop training after this point.")
    gflags.DEFINE_integer("batch_size", 32, "SGD minibatch size.")
    gflags.DEFINE_enum("opt", "RMSProp", ["RMSProp", "Adam"], "Specify optimization method.")
    gflags.DEFINE_boolean("use_lr_decay", True, "Used in RMSProp.")
    gflags.DEFINE_float("learning_rate", 0.001, "Used in RMSProp.")
    gflags.DEFINE_float("learning_rate_decay_per_10k_steps", 0.75, "Used in RMSProp.")
    gflags.DEFINE_float("clipping_max_value", 5.0, "")
    gflags.DEFINE_float("l2_lambda", 1e-5, "")
    gflags.DEFINE_float("init_range", 0.005, "Mainly used for softmax parameters. Range for uniform random init.")

    # Display settings.
    gflags.DEFINE_integer("statistics_interval_steps", 100, "Print training set results at this interval.")
    gflags.DEFINE_integer("eval_interval_steps", 100, "Evaluate at this interval.")

    gflags.DEFINE_integer("ckpt_interval_steps", 5000, "Update the checkpoint on disk at this interval.")
    gflags.DEFINE_boolean("ckpt_on_best_dev_error", True, "If error on the first eval set (the dev set) is "
        "at most 0.99 of error at the previous checkpoint, save a special 'best' checkpoint.")

    # Evaluation settings
    gflags.DEFINE_boolean("expanded_eval_only_mode", False,
        "If set, a checkpoint is loaded and a forward pass is done to get the predicted "
        "transitions. The inferred parses are written to the supplied file(s) along with example-"
        "by-example accuracy information. Requirements: Must specify checkpoint path.")
    gflags.DEFINE_string("eval_output_paths", None,
        "Used when expanded_eval_only_mode is set. The number of supplied paths should be same"
        "as the number of eval sets.")
    gflags.DEFINE_boolean("write_predicted_label", False,
        "Write the predicted labels in a <eval_output_name>.lbl file.")

    # Parse command line flags.
    FLAGS(sys.argv)

    if not FLAGS.experiment_name:
        timestamp = str(int(time.time()))
        FLAGS.experiment_name = "{}-{}-{}".format(
            FLAGS.data_type,
            FLAGS.model_type,
            timestamp,
            )

    if FLAGS.debug:
        chainer.set_debug(True)

    if not FLAGS.branch_name:
        FLAGS.branch_name = os.popen('git rev-parse --abbrev-ref HEAD').read().strip()

    if not FLAGS.sha:
        FLAGS.sha = os.popen('git rev-parse HEAD').read().strip()

    run(only_forward=FLAGS.expanded_eval_only_mode)
