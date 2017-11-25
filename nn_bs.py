#!/usr/bin/env python

import os
import numpy as np
from scipy import ndimage
import mnist

def AffinePropagate(W, b, in_value):
  out_value = np.dot(in_value, W.T) + b
  return out_value

def AffineBackprop(W, b, in_value, out_deriv):
  num_samples = in_value.shape[0]
  assert num_samples == out_deriv.shape[0]
  in_deriv = np.dot(out_deriv, W)
  W_deriv = np.dot(out_deriv.T, in_value) / num_samples
  b_deriv = np.sum(out_deriv, 0) / num_samples
  return (in_deriv, W_deriv, b_deriv)

def LogSoftMaxPropagate(in_value):
  col_max = np.amax(in_value, 1)
  in_value = (in_value.T - col_max).T
  exp_in_value = np.exp(in_value)
  col_log_sum_exp = np.log(np.sum(exp_in_value, 1))
  out_value = (in_value.T - col_log_sum_exp).T
  return out_value

def LogSoftMaxBackprop(out_value, out_deriv):
  exp_out_value = np.exp(out_value)
  col_sum_out_deriv = np.sum(out_deriv, 1)
  in_deriv = out_deriv - (exp_out_value.T * col_sum_out_deriv).T
  return in_deriv

def TanhPropagate(in_value):
  out_value = np.tanh(in_value)
  return out_value

def TanhBackprop(out_value, out_deriv):
  in_deriv = out_deriv * (1.0 - out_value * out_value)
  return in_deriv
  
def ReLUPropagate(in_value):
  out_value = np.maximum(in_value, 0)
  return out_value

def ReLUBackprop(out_value, out_deriv):
  in_deriv = out_deriv * ((out_value > 0.0) * 1.0 + (out_value <= 0.0) * 0.0)
  return in_deriv

def SigmoidPropagate(in_value):
  out_value = 1.0 / (1.0 + np.exp(-in_value))
  return out_value

def SigmoidBackprop(out_value, out_deriv):
  in_deriv = out_deriv * out_value * (1.0 - out_value)
  return in_deriv
  
class NN(object):
  def __init__(self, num_layers, input_dim, hidden_dim, num_classes, batch_size,
               test_examples=None, nonlin='Tanh', update='simple', alpha=0.0):
    self.num_layers = num_layers
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.num_classes = num_classes
    self.batch_size = batch_size
    self.test_examples = test_examples
    self.nonlin = nonlin
    self.update = update
    self.alpha = alpha

    self.lr_init = 1e-3
    self.eps = 1e-6
    self.decay = 1e-4

    # trainable parameters
    self.W = [np.random.randn(self.hidden_dim,
        self.input_dim if i == 0
        else self.hidden_dim) for i in range(num_layers)]
    self.b = [np.random.randn(self.hidden_dim) for i in range(self.num_layers)]
    self.W.append(np.random.randn(self.num_classes, self.hidden_dim))
    self.b.append(np.random.randn(self.num_classes))

    # first and second order moments for Adam
    self.sum_dW = [np.zeros([self.hidden_dim,
        self.input_dim if i == 0
        else self.hidden_dim]) for i in range(num_layers)]
    self.sum_db = [np.zeros(self.hidden_dim) for i in range(self.num_layers)]
    self.sum_dW.append(np.zeros([self.num_classes, self.hidden_dim]))
    self.sum_db.append(np.zeros(self.num_classes))

    self.sum_dW2 = [np.zeros([self.hidden_dim,
        self.input_dim if i == 0
        else self.hidden_dim]) for i in range(num_layers)]
    self.sum_db2 = [np.zeros(self.hidden_dim) for i in range(self.num_layers)]
    self.sum_dW2.append(np.zeros([self.num_classes, self.hidden_dim]))
    self.sum_db2.append(np.zeros(self.num_classes))

    if self.update == 'natural':
      # moving estimates of inverse Fisher matrices
      self.inv_F = [np.identity(self.hidden_dim * (self.input_dim if i == 0
          else self.hidden_dim) + self.hidden_dim)
          for i in range(self.num_layers)]
      self.inv_F.append(np.identity(self.num_classes * self.hidden_dim +
          self.num_classes))

    # input to each hidden layer, i.e., input to affine
    self.hidden_inputs =[np.zeros([self.batch_size, self.input_dim if i == 0
        else self.hidden_dim]) for i in range(self.num_layers)]
    # input to final affine
    self.hidden_inputs.append(np.zeros([self.batch_size, self.hidden_dim]))

  def NonlinPropagate(self, in_value):
    if self.nonlin == 'Tanh':
      return TanhPropagate(in_value)
    elif self.nonlin == 'Sigmoid':
      return SigmoidPropagate(in_value)
    elif self.nonlin == 'ReLU':
      return ReLUPropagate(in_value)

  def NonlinBackprop(self, out_value, out_deriv):
    if self.nonlin == 'Tanh':
      return TanhBackprop(out_value, out_deriv)
    elif self.nonlin == 'Sigmoid':
      return SigmoidBackprop(out_value, out_deriv)
    elif self.nonlin == 'ReLU':
      return ReLUBackprop(out_value, out_deriv)


  def Propagate(self, X, labels=None, test_mode=False):
    cur_batch_size = X.shape[0]
    in_value = X
    for i in range(self.num_layers):
      if not test_mode:
        self.hidden_inputs[i][:cur_batch_size, :] = in_value
      out_value = AffinePropagate(self.W[i], self.b[i], in_value)
      in_value = out_value
      out_value = self.NonlinPropagate(in_value)
      in_value = out_value
    if not test_mode:
      self.hidden_inputs[-1][:cur_batch_size, :] = in_value
    out_value = AffinePropagate(self.W[-1], self.b[-1], in_value)
    in_value = out_value
    out_value = LogSoftMaxPropagate(in_value)
    if test_mode:
      assert not labels is None and labels.shape[0] == cur_batch_size
      predicted_labels = np.argmax(out_value, axis=1)
      accuracy = float(sum(predicted_labels == labels)) / cur_batch_size
      return out_value, accuracy
    return out_value

  def Backprop(self, out_value, out_deriv, learning_rate, iter):
    cur_batch_size = out_value.shape[0]
    assert cur_batch_size == out_deriv.shape[0]
    in_deriv = LogSoftMaxBackprop(out_value, out_deriv)
    in_value = self.hidden_inputs[-1][:cur_batch_size, :]
    out_deriv = in_deriv
    in_deriv, W_deriv, b_deriv = AffineBackprop(self.W[-1], self.b[-1],
                                                in_value, out_deriv)
    self.UpdateParams(self.num_layers, W_deriv, b_deriv, learning_rate, iter)
    for i in range(self.num_layers)[::-1]:
      out_value = in_value
      out_deriv = in_deriv
      in_deriv = self.NonlinBackprop(out_value, out_deriv)
      in_value = self.hidden_inputs[i][:cur_batch_size, :]
      out_deriv = in_deriv
      in_deriv, W_deriv, b_deriv = AffineBackprop(self.W[i], self.b[i],
                                                  in_value, out_deriv)
      self.UpdateParams(i, W_deriv, b_deriv, learning_rate, iter)
    return in_deriv

  def UpdateParamsSimple(self, i, W_deriv, b_deriv, learning_rate, iter):
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    self.sum_dW[i] = beta1 * self.sum_dW[i] + (1.0 - beta1) * W_deriv
    self.sum_db[i] = beta1 * self.sum_db[i] + (1.0 - beta1) * b_deriv
    self.sum_dW2[i] = beta2 * self.sum_dW2[i] + (1.0 - beta2) * W_deriv * W_deriv
    self.sum_db2[i] = beta2 * self.sum_db2[i] + (1.0 - beta2) * b_deriv * b_deriv

    sum_dW_hat = self.sum_dW[i] / (1.0 - np.power(beta1, iter + 1))
    sum_db_hat = self.sum_db[i] / (1.0 - np.power(beta1, iter + 1))
    sum_dW2_hat = self.sum_dW2[i] / (1.0 - np.power(beta2, iter + 1))
    sum_db2_hat = self.sum_db2[i] / (1.0 - np.power(beta2, iter + 1))
    self.W[i] -= learning_rate * self.sum_dW[i] / (np.sqrt(sum_dW2_hat[i]) + epsilon)
    self.b[i] -= learning_rate * self.sum_db[i] / (np.sqrt(sum_db2_hat[i]) + epsilon)
    '''
    self.W[i] -= learning_rate * (1.0 + self.alpha) * W_deriv
    self.b[i] -= learning_rate * (1.0 + self.alpha) * b_deriv
    '''

  def UpdateParamsSimpleBackstitch(self, i, W_deriv, b_deriv, learning_rate):
    self.W[i] += learning_rate * self.alpha * W_deriv
    self.b[i] += learning_rate * self.alpha * b_deriv

  # first update the parameters with inverse fisher estimated from previous
  # examples, then update the inverse fisher.
  # F_{t+1}=(1-gamma)*F_t+gamma*S_t, where S_t= model_deriv * model_deriv^T.
  # Update F_{t+1}^{-1} by making use of Woodbury formula
  # (A+uu^T)^{-1}=A^{-1}-A^{-1}uu^TA^{-1}/(1+u^TA^{-1}u)
  def UpdateParamsNatural(self, i, W_deriv, b_deriv, learning_rate):
    #assert np.all(self.inv_F[i] == self.inv_F[i].T) ###
    deriv_concat = np.concatenate((np.reshape(W_deriv, (-1)), b_deriv))
    natural_deriv = np.dot(self.inv_F[i], deriv_concat)
    natural_deriv *= (np.linalg.norm(deriv_concat) /
                      np.linalg.norm(natural_deriv))
    self.W[i] -= learning_rate * np.reshape(natural_deriv[:W_deriv.size],
                                            W_deriv.shape)
    self.b[i] -= learning_rate * natural_deriv[W_deriv.size:]
    gamma = 0.001
    #print(np.mean(abs(self.inv_F[i])), np.mean(abs(deriv_concat))) ##
    self.inv_F[i] *= 1.0 / (1.0 - gamma)
    u = np.sqrt(gamma) * deriv_concat
    inv_F_u = np.dot(self.inv_F[i], u)
    inv_F_u /= np.sqrt(1.0 + np.dot(u, inv_F_u))
    #num = np.random.randint(100000) ##
    #if num % 10 == 0 and i == 1:
    #  temp = np.linalg.inv(np.linalg.inv(self.inv_F[i]) + np.outer(u, u)) ##
    self.inv_F[i] -= np.outer(inv_F_u, inv_F_u)
    #if num % 10 == 0 and i == 1:
    #  print('diff' + str(np.mean(abs(temp - self.inv_F[i]))))##
    # smooth F with beta * aa^T where a is a random one-hot vector
    sqrt_beta = 1e-3
    k = np.random.randint(0, self.inv_F[i].shape[0])
    inv_F_a = sqrt_beta * self.inv_F[i][k]
    inv_F_a /= np.sqrt(1.0 + sqrt_beta * inv_F_a[k])
    self.inv_F[i] -= np.outer(inv_F_a, inv_F_a)
    
  def UpdateParamsNaturalBackstitch(self, i, W_deriv, b_deriv, learning_rate):
    deriv_concat = np.concatenate((np.reshape(W_deriv, (-1)), b_deriv))
    natural_deriv = np.dot(self.inv_F[i], deriv_concat)
    natural_deriv *= (np.linalg.norm(deriv_concat) /
                      np.linalg.norm(natural_deriv))
    self.W[i] += learning_rate * self.alpha * np.reshape(
        natural_deriv[:W_deriv.size], W_deriv.shape)
    self.b[i] += learning_rate * self.alpha * natural_deriv[W_deriv.size:]
 
  def UpdateParams(self, i, W_deriv, b_deriv, learning_rate, iter, backstitch_step1=False):
    if self.update == 'simple':
      if not backstitch_step1:
        return self.UpdateParamsSimple(i, W_deriv, b_deriv, learning_rate, iter)
      return self.UpdateParamsSimpleBackstitch(i, W_deriv, b_deriv, learning_rate, iter)
    elif self.update == 'natural':
      if not backstitch_step1:
        return self.UpdateParamsNatural(i, W_deriv, b_deriv, learning_rate)
      return self.UpdateParamsNaturalBackstitch(i, W_deriv, b_deriv, learning_rate)
 
  # examples is a 2-tuple (images, labels)
  def Train(self, examples):
    num_examples = examples[0].shape[0]
    assert num_examples == examples[1].shape[0]
    assert examples[0].shape[1] == self.input_dim
    lr_init = 1e-1
    eps = 1e-6
    decay = 1e-3
    iter = 0
    train_loss = float('Inf')
    num_iters_per_epoch = int(np.ceil(float(num_examples) / self.batch_size))
    lr = lr_init #########
    while (iter == 0 or iter % num_iters_per_epoch != 0 or
        iter // num_iters_per_epoch == 1 or
        abs((train_loss_prev - train_loss) / train_loss_prev) >= eps):
      np.random.seed(iter)
      if iter % num_iters_per_epoch == 0:
        idx_shuffled = np.random.permutation(num_examples)
      epoch = iter // num_iters_per_epoch
      #lr = lr_init * (1. / (1. + decay * epoch))
      cur_batch_size = (num_examples - (num_iters_per_epoch - 1) *
          self.batch_size) if (iter % num_iters_per_epoch == num_iters_per_epoch
          - 1) else self.batch_size
      idx = idx_shuffled[(iter % num_iters_per_epoch) * self.batch_size :
          (iter % num_iters_per_epoch) * self.batch_size + cur_batch_size]
      X = examples[0][idx, :]
      Y = np.zeros([cur_batch_size, self.num_classes])
      if self.alpha > 0.0:
        out_value = self.Propagate(X)
        self.Backprop(out_value, Y, lr, iter, backstitch_step1=True)
      out_value = self.Propagate(X)
      self.Backprop(out_value, Y, lr, iter, backstitch_step1=False)
      # -1 since we are minimizing the loss
      Y[range(cur_batch_size), examples[1][idx]] = -1
      out_value = self.Propagate(X)
      self.Backprop(out_value, Y, lr, iter)
      iter += 1
      temp1, temp2 = self.Propagate(examples[0], examples[1], test_mode=True) ###
      temp3 = -np.sum(temp1[range(num_examples), examples[1]]) / num_examples ##
      print("epoch " + str(epoch) + ####
            ", train_loss=" + str(temp3) + ####
            ", train_accuracy=" + str(temp2)) ###
      if iter % num_iters_per_epoch == 0:
        train_loss_prev = train_loss
        out_value, train_accuracy = self.Propagate(examples[0], examples[1],
                                                   test_mode=True)
        train_loss = -np.sum(out_value[range(num_examples),
            examples[1]]) / num_examples
        num_test_examples = self.test_examples[0].shape[0]
        out_value, test_accuracy = self.Propagate(self.test_examples[0],
                                                  self.test_examples[1],
                                                  test_mode=True)
        test_loss = -np.sum(out_value[range(num_test_examples),
            self.test_examples[1]]) / num_test_examples
        print("epoch " + str(epoch) + ": learning_rate=" + str(lr) +
              ", train_loss=" + str(train_loss) +
              ", train_accuracy=" + str(train_accuracy) +
              ", test_loss=" + str(test_loss) +
              ", test_accuracy=" + str(test_accuracy))
        if epoch > 0 and (train_loss - train_loss_prev) / abs(train_loss_prev) > -1e-4: ####
          lr /= 2.0  ###


def main():
  hidden_dim = 200
  batch_size = 100
  print("hidden_dim: " + str(hidden_dim) + ", batch_size: " + str(batch_size))
  training_set = np.array(list(mnist.read(dataset="training",
                                          path="/home/ywang/mnist")))
  print("size of the training set: " + str(len(training_set)))
  testing_set = np.array(list(mnist.read(dataset="testing",
                                         path="/home/ywang/mnist")))
  print("size of the testing set: " + str(len(testing_set)))
  np.random.seed(0)
  p = 1.0
  backstitch_alpha = 0.3
  print("backstitch alpha: " + str(backstitch_alpha))
  ratio = 0.5
  training_subset = training_set[np.random.binomial(1, p,
                                                    len(training_set)) == 1]
  print("number of the actual training examples: " + str(len(training_subset)))
  # resize the images to make the number of input features 4 times smaller
  #training_images = np.stack([training_subset[i][0] for i in range(len(training_subset))])
  training_images = np.stack([ndimage.zoom(training_subset[i][0], ratio)
                             for i in range(len(training_subset))])
  print("image size is: " + str(training_images.shape[1]) + " by " + str(training_images.shape[2]))
  training_images = np.reshape(training_images, [training_images.shape[0], -1])
  training_labels = np.stack([training_subset[i][1]
                             for i in range(len(training_subset))])
  training_examples = (training_images, training_labels)
  
  #testing_images = np.stack([testing_set[i][0] for i in range(len(testing_set))])
  testing_images = np.stack([ndimage.zoom(testing_set[i][0], ratio)
                            for i in range(len(testing_set))])
  testing_images = np.reshape(testing_images, [testing_images.shape[0], -1])
  testing_labels = np.stack([testing_set[i][1]
                            for i in range(len(testing_set))])
  testing_examples = (testing_images, testing_labels)

  nnet = NN(num_layers=1, input_dim=training_images.shape[1],
            hidden_dim=hidden_dim, num_classes=10, batch_size=batch_size,
            test_examples=testing_examples, nonlin='Tanh', update='natural',
            alpha=backstitch_alpha)
  nnet.Train(training_examples)

if __name__ == "__main__":
  main()

