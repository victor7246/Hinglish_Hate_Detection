import tensorflow as tf
import tensorflow.keras.backend as K

from sklearn.metrics import f1_score
from sklearn_crfsuite.metrics import flat_classification_report, flat_f1_score

import numpy as np

def f1_keras(y_true, y_pred):
    y_pred = K.round(y_pred)

    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)

    return K.mean(f1)


def categorical_focal_loss(alpha, gamma=2.):
    """
    Softmax version of focal loss.
    When there is a skew between different categories/labels in your data set, you can try to apply this function as a
    loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
      categories/labels, the size of the array needs to be consistent with the number of classes.
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    alpha = np.array(alpha, dtype=np.float32)

    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return K.mean(K.sum(loss, axis=-1))

    return categorical_focal_loss_fixed
    
class F1Callback(tf.keras.callbacks.Callback):
  def __init__(self, model, inputs, targets, filename, patience):
    self.model = model
    self.inputs = inputs
    self.targets = targets
    self.filename = filename
    self.patience = patience

    self.best_score = 0
    self.bad_epoch = 0

  def on_epoch_end(self, epoch, logs):
    pred = self.model.predict(self.inputs)
    score = f1_score(self.targets.argmax(-1), pred.argmax(-1), average='macro')

    if score > self.best_score:
      self.best_score = score
      self.model.save_weights(self.filename)
      print ("\nScore {}. Model saved in {}.".format(score, self.filename))
      self.bad_epoch = 0
    else:
      print ("\nScore {}. Model not saved.".format(score))
      self.bad_epoch += 1

    if self.bad_epoch >= self.patience:
      print ("\nEpoch {}: early stopping.".format(epoch))
      self.model.stop_training = True

class SeqF1Callback(tf.keras.callbacks.Callback):
  def __init__(self, model, inputs, targets, filename, patience):
    self.model = model
    self.inputs = inputs
    self.targets = targets
    self.filename = filename
    self.patience = patience

    self.best_score = 0
    self.bad_epoch = 0

  def on_epoch_end(self, epoch, logs):
    pred = self.model.predict(self.inputs)
    score = flat_f1_score(self.targets.argmax(-1), pred.argmax(-1), average='macro')

    if score > self.best_score:
      self.best_score = score
      self.model.save_weights(self.filename)
      print ("\nScore {}. Model saved in {}.".format(score, self.filename))
      self.bad_epoch = 0
    else:
      print ("\nScore {}. Model not saved.".format(score))
      self.bad_epoch += 1

    if self.bad_epoch >= self.patience:
      print ("\nEpoch {}: early stopping.".format(epoch))
      self.model.stop_training = True

class Snapshot(tf.keras.callbacks.Callback):
  def __init__(self, model, val_inputs, val_targets, test_inputs=None):
    self.model = model
    self.val_inputs = val_inputs
    self.val_targets = val_targets
    self.test_inputs = test_inputs
    
    self.best_score = 0

    self.all_val_snapshots = []
    self.best_scoring_val_snapshots = []

    self.all_test_snapshots = []
    self.best_scoring_test_snapshots = []
    
  def on_epoch_end(self, epoch, logs):
    val_pred = self.model.predict(self.val_inputs)
    
    if self.test_inputs is None:
        test_pred = self.model.predict(self.test_inputs)
    else:
        test_pred = []
        
    score = f1_score(self.val_targets.argmax(-1), val_pred.argmax(-1), average='macro')

    self.all_val_snapshots.append(val_pred)
    self.all_test_snapshots.append(test_pred)
    
    if score > self.best_score:
      self.best_score = score
      self.best_scoring_val_snapshots.append(val_pred)
      self.best_scoring_test_snapshots.append(test_pred)
    
        