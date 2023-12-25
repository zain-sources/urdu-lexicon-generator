import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import tensorflow as tf
import data_utils
import seq2seq_model
from six.moves import xrange

class G2PModel(object):
  """Grapheme-to-Phoneme translation model class.

  Constructor parameters (for training mode only):
    train_lines: Train dictionary;
    valid_lines: Development dictionary;
    test_lines: Test dictionary.

  Attributes:
    gr_vocab: Grapheme vocabulary;
    ph_vocab: Phoneme vocabulary;
    train_set: Training buckets: words and sounds are mapped to ids;
    valid_set: Validation buckets: words and sounds are mapped to ids;
    session: Tensorflow session;
    model: Tensorflow Seq2Seq model for G2PModel object.
    train: Train method.
    interactive: Interactive decode method;
    evaluate: Word-Error-Rate counting method;
    decode: Decode file method.
  """
  # We use a number of buckets and pad to the closest one for efficiency.
  # See seq2seq_model.Seq2SeqModel for details of how they work.
  _BUCKETS = [(5, 10), (10, 15), (40, 50)]

  def __init__(self, model_dir:str="models"):
    """Initialize model directory."""
    self.model_dir = model_dir

  def load_decode_model(self):
    """Load G2P model and initialize or load parameters in session."""
    if not os.path.exists(os.path.join(self.model_dir, 'checkpoint')):
      raise RuntimeError("Model not found in %s" % self.model_dir)
    
    self.batch_size = 1 # We decode one word at a time.
    #Load model parameters.
    num_layers, size = data_utils.load_params(self.model_dir)
    # Load vocabularies
    print("Loading vocabularies from %s" % self.model_dir)
    self.gr_vocab = data_utils.load_vocabulary(os.path.join(self.model_dir,
                                                            "vocab.grapheme"))
    self.ph_vocab = data_utils.load_vocabulary(os.path.join(self.model_dir,
                                                            "vocab.phoneme"))

    self.rev_ph_vocab =\
      data_utils.load_vocabulary(os.path.join(self.model_dir, "vocab.phoneme"),
                                 reverse=True)

    self.session = tf.Session()

    # Restore model.
    print("Creating %d layers of %d units." % (num_layers, size))
    self.model = seq2seq_model.Seq2SeqModel(len(self.gr_vocab),
                                            len(self.ph_vocab), self._BUCKETS,
                                            size, num_layers, 0,
                                            self.batch_size, 0, 0,
                                            forward_only=True)
    self.model.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
    # Check for saved models and restore them.
    print("Reading model parameters from %s" % self.model_dir)
    self.model.saver.restore(self.session, os.path.join(self.model_dir,
                                                        "model"))


  def decode_word(self, word):
    """Decode input word to sequence of phonemes.

    Args:
      word: input word;

    Returns:
      phonemes: decoded phoneme sequence for input word;
    """
    # Check if all graphemes attended in vocabulary
    gr_absent = [gr for gr in word if gr not in self.gr_vocab]
    if gr_absent:
      print("Symbols '%s' are not in vocabulary" % "','".join(gr_absent).encode('utf-8'))
      return ""

    # Get token-ids for the input word.
    token_ids = [self.gr_vocab.get(s, data_utils.UNK_ID) for s in word]
    # Which bucket does it belong to?
    bucket_id = min([b for b in xrange(len(self._BUCKETS))
                     if self._BUCKETS[b][0] > len(token_ids)])
    # Get a 1-element batch to feed the word to the model.
    encoder_inputs, decoder_inputs, target_weights = self.model.get_batch(
        {bucket_id: [(token_ids, [])]}, bucket_id)
    # Get output logits for the word.
    _, _, output_logits = self.model.step(self.session, encoder_inputs,
                                          decoder_inputs, target_weights,
                                          bucket_id, True)
    # This is a greedy decoder - outputs are just argmaxes of output_logits.
    outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
    # If there is an EOS symbol in outputs, cut them at that point.
    if data_utils.EOS_ID in outputs:
      outputs = outputs[:outputs.index(data_utils.EOS_ID)]
    # Phoneme sequence corresponding to outputs.
    return " ".join([self.rev_ph_vocab[output] for output in outputs])


  def decode(self, decode_lines, output_file=None):
    """Decode words from file.

    Returns:
      if [--output output_file] pointed out, write decoded word sequences in
      this file. Otherwise, print decoded words in standard output.
    """
    phoneme_lines = []

    # Decode from input file.
    if output_file:
      for word in decode_lines:
        word = word.strip()
        phonemes = self.decode_word(word)
        output_file.write(word)
        output_file.write(' ')
        output_file.write(phonemes)
        output_file.write('\n')
        phoneme_lines.append(phonemes)
      output_file.close()
    else:
      for word in decode_lines:
        word = word.strip()
        phonemes = self.decode_word(word)
        phoneme_lines.append(phonemes)
    return phoneme_lines
