#!/usr/bin/env python

import numpy as np
import math

def PredLR(x, params):
  y_hat = 1.0 / (1.0 + np.exp(-(params[0] + params[1] * x)))
  return y_hat
def ObjfLR(x, y, params):
  y_hat = PredLR(x, params)
  objf = np.zeros(np.shape(y_hat))
  objf[y == 1.0] = -np.log(y_hat[y == 1.0])
  objf[y != 1.0] = -np.log(1.0 - y_hat[y != 1.0])
  return objf
def GradientLR(x, y, params):
  y_hat = PredLR(x, params)
  g = np.stack([np.ones(np.shape(x), dtype=np.float64), x], -1) * np.expand_dims(y_hat - y, 1)
  return g
def HessianLR(x, y, params):
  y_hat = PredLR(x, params)
  if len(np.shape(x)) > 0:
    h = np.array(map(lambda x, y_hat: np.array([[1.0,  x], [x, x * x]]) * y_hat * (1.0 - y_hat), x, y_hat))
  else:
    h = np.array([[1.0,  x], [x, x * x]]) * y_hat * (1.0 - y_hat)
  return h

# optimum params
b0 = 0.0
b1 = 3.0
optimum_params = np.array([b0, b1])

num_tries = 20000
num_samples = 100

np.random.seed(0)
x = np.random.uniform(1, 2, 20000)
h_bar_inv = np.linalg.inv(np.sum(HessianLR(x, None, optimum_params), 0) / len(x))
tot_opt = np.zeros(2)
tot_bias = np.zeros(2)
for t in range(num_tries):
  np.random.seed(t)
  x = np.random.uniform(1, 2, num_samples)
  y = np.array(map(lambda x: np.random.binomial(1, PredLR(x, optimum_params)), x))
  gs = GradientLR(x, y, optimum_params)
  hs = HessianLR(x, None, optimum_params)
  tot_g = np.sum(gs, 0)
  tot_h = np.sum(hs, 0)
  tot_h_h_bar_inv_g = np.sum(np.array(map(lambda g, h: np.dot(h, np.dot(h_bar_inv, g)), gs, hs)), 0)
  this_bias = 1.0 / num_samples * np.dot(h_bar_inv, tot_h_h_bar_inv_g / num_samples)
  tot_bias += this_bias
  this_opt = optimum_params - np.dot(np.linalg.inv(tot_h), tot_g)
  tot_opt += this_opt

avg_bias = tot_bias * (1.0 / num_tries)
print "avg_bias is ", avg_bias
avg_opt = tot_opt * (1.0 / num_tries)
print "avg_opt is ", avg_opt
print "true_opt is ", optimum_params


num_tries = 1000
num_samples = 100
batch_size = 1
init_params = np.array([1.0, 4.0])
lr_init = 1e-3
decay = 1e-4
eps=1e-7
'''
# SGD
print "SGD..."
tot_opt = np.zeros(2)
tot_objf = 0.0
for t in range(num_tries):
  np.random.seed(t)
  params = np.copy(init_params)
  f_x = float('Inf')
  x = np.random.uniform(1, 2, num_samples)
  y = np.array(map(lambda x: np.random.binomial(1, PredLR(x, optimum_params)), x))
  data = np.stack([x, y], 1)
  #print  'objf at opt', np.sum(ObjfLR(data[:, 0], data[:, 1], optimum_params)) / num_samples
  iter = 0
  while iter == 0 or (batch_size * iter) % num_samples != 0 or np.abs(f_x_prev - f_x) >= eps:
    epoch = (batch_size * iter) / num_samples
    lr = lr_init * (1. / (1. + decay * epoch))
    np.random.seed(iter)
    idx = np.random.choice(num_samples, batch_size, replace=False)
    batch = data[idx, :]
    params -= lr * np.sum(GradientLR(batch[:, 0], batch[:, 1], params), 0) / batch_size
    iter += 1
    if (batch_size * iter) % num_samples == 0:
      f_x_prev = f_x
      f_x = np.sum(ObjfLR(data[:, 0], data[:, 1], params)) / num_samples
    #if iter % 10000 == 0:
    #  print 'objf=' + str(f_x), params
  # \int sigmoid(x) dx = log(1 + exp(x))
  print 'trial ' + str(t) + ': objf=' + str(-1.0 / (1.0 * params[1]) * (np.log(1.0 + np.exp(params[0] + 2.0 * params[1])) - np.log(1.0 + np.exp(params[0] + params[1])))), params
  tot_opt += params
  tot_objf += f_x

avg_opt = tot_opt * (1.0 / num_tries)
print "avg_opt is ", avg_opt
print "avg_objf_is ", tot_objf * (1.0 / num_tries)
print "bias = ", avg_opt - optimum_params

'''
# Backstitch SGD
print "Backstitch..."
tot_opt = np.zeros(2)
tot_objf = 0.0
for t in range(num_tries):
  np.random.seed(t)
  params = np.copy(init_params)
  f_x = float('Inf')
  x = np.random.uniform(1, 2, num_samples)
  y = map(lambda x: np.random.binomial(1, PredLR(x, optimum_params)), x)
  data = np.stack([x, y], 1)
  print  'objf at opt', np.sum(ObjfLR(data[:, 0], data[:, 1], optimum_params)) / num_samples
  iter = 0
  while iter == 0 or (batch_size * iter) % num_samples != 0 or np.abs(f_x_prev - f_x) >= eps:
    epoch = (batch_size * iter) / num_samples
    lr = lr_init * (1. / (1. + decay * epoch))
    np.random.seed(iter)
    idx = np.random.choice(num_samples, batch_size, replace=False)
    batch = data[idx, :]
    alpha = 0.5 * (np.sqrt(1.0 + 4.0 / (lr * num_samples / batch_size)) - 1.0)
    params += alpha * lr * np.sum(GradientLR(batch[:, 0], batch[:, 1], params), 0) / batch_size
    params -= (1 + alpha) * lr * np.sum(GradientLR(batch[:, 0], batch[:, 1], params), 0) / batch_size
    iter += 1
    if (batch_size * iter) % num_samples == 0:
      f_x_prev = f_x
      f_x = np.sum(ObjfLR(data[:, 0], data[:, 1], params)) / num_samples
    #if iter % 10000 == 0:
    #  print 'objf=' + str(f_x), params
  print 'trial ' + str(t) + ': objf=' + str(-1.0 / (1.0 * params[1]) * (np.log(1.0 + np.exp(params[0] + 2.0 * params[1])) - np.log(1.0 + np.exp(params[0] + params[1])))), params
  tot_opt += params
  tot_objf += f_x

avg_opt = tot_opt * (1.0 / num_tries)
print "avg_opt is ", avg_opt
print "avg_objf_is ", tot_objf * (1.0 / num_tries)
print "bias = ", avg_opt - optimum_params

