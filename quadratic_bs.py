#!/usr/bin/env python

import numpy as np
import math

# minimize quadratic function f(b) = 0.5 * b^T * C * b + y^T * b
def ObjfQuad(C, y, params):
  objf = 0.5 * np.dot(np.dot(params, C), params) + np.dot(y, params)
  return objf
def GradientQuad(C, y, params):
  g = np.dot(C, params) + y
  return g
def HessianQuad(C, y, params):
  h = C
  return h

# - inv(E[C]) * E[y]
optimum_params = -1.0 * np.dot(np.linalg.inv(2.0 / 3 * np.eye(2, dtype=np.float64)), np.array([0.5, 0.5]))

num_tries = 50000
num_samples = 100

h_bar_inv = np.linalg.inv(HessianQuad(2.0 / 3 * np.eye(2, dtype=np.float64), None, optimum_params))
tot_opt = np.zeros(2)
tot_bias = np.zeros(2)
for t in range(num_tries):
  np.random.seed(t)
  x = np.random.uniform(-1, 1, [num_samples, 2, 2])
  C = np.array(map(lambda x: np.dot(x, np.transpose(x)), x))
  y = np.random.uniform(0, 1, [num_samples, 2])
  gs = GradientQuad(C, y, optimum_params)
  hs = HessianQuad(C, y, optimum_params)
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
init_params = np.array([5.0, 10.0])
lr_init = 1e-3
decay = 1e-4
eps=1e-6


# SGD
print "SGD..."
tot_opt = np.zeros(2)
tot_close_opt=np.zeros(2)
tot_objf = 0.0
for t in range(num_tries):
  np.random.seed(t)
  params = np.copy(init_params)
  f_x = float('Inf')
  x = np.random.uniform(-1, 1, [num_samples, 2, 2])
  C = np.array(map(lambda x: np.dot(x, np.transpose(x)), x))
  y = np.random.uniform(0, 1, [num_samples, 2])
  # print  'objf at opt', np.sum(ObjfQuad(C, y, optimum_params)) / num_samples
  iter = 0
  while iter == 0 or (batch_size * iter) % num_samples != 0 or np.abs(f_x_prev - f_x) >= eps:
    epoch = (batch_size * iter) / num_samples
    lr = lr_init * (1. / (1. + decay * epoch))
    np.random.seed(iter)
    idx = np.random.choice(num_samples, batch_size, replace=False)
    params -= lr * np.sum(GradientQuad(C[idx, :, :], y[idx, :], params), 0) / batch_size
    iter += 1
    if (batch_size * iter) % num_samples == 0:
      f_x_prev = f_x
      f_x = np.sum(ObjfQuad(C, y, params)) / num_samples
    #if iter % 10000 == 0:
    #  print 'objf=' + str(f_x), params
  print 'trial ' + str(t) + ': objf=' + str(f_x), params
  print 'closed form params ', np.sum(ObjfQuad(C, y, -np.dot(np.linalg.inv(np.sum(C, 0)), np.sum(y, 0)))) / num_samples, -np.dot(np.linalg.inv(np.sum(C, 0)), np.sum(y, 0))
  # print 'trial ' + str(t) + ': objf=' + str(ObjfQuad(2.0 / 3 * np.eye(2, dtype=np.float64), np.array([0.5, 0.5]), params)), params
  tot_opt += params
  tot_objf += f_x
  tot_close_opt += -np.dot(np.linalg.inv(np.sum(C, 0)), np.sum(y, 0))

avg_opt = tot_opt * (1.0 / num_tries)
print "avg_opt is ", avg_opt
print "avg_close_opt is", tot_close_opt * (1.0 / num_tries)
print "avg_objf_is ", tot_objf * (1.0 / num_tries)
print "bias = ", avg_opt - optimum_params


# Backstitch SGD
print "Backstitch..."
tot_opt = np.zeros(2)
tot_objf = 0.0
for t in range(num_tries):
  np.random.seed(t)
  params = np.copy(init_params)
  f_x = float('Inf')
  x = np.random.uniform(-1, 1, [num_samples, 2, 2])
  C = np.array(map(lambda x: np.dot(x, np.transpose(x)), x))
  y = np.random.uniform(0, 1, [num_samples, 2])
  # print  'objf at opt', np.sum(ObjfQuad(C, y, optimum_params)) / num_samples
  iter = 0
  while iter == 0 or (batch_size * iter) % num_samples != 0 or np.abs(f_x_prev - f_x) >= eps:
    epoch = (batch_size * iter) / num_samples
    lr = lr_init * (1. / (1. + decay * epoch))
    np.random.seed(iter)
    idx = np.random.choice(num_samples, batch_size, replace=False)
    alpha = 0.5 * (np.sqrt(1.0 + 1.5 * 4.0 / (lr * num_samples / batch_size)) - 1.0)
    params += alpha * lr * np.sum(GradientQuad(C[idx, :, :], y[idx, :], params), 0) / batch_size
    params -= (1 + alpha) * lr * np.sum(GradientQuad(C[idx, :, :], y[idx, :], params), 0) / batch_size
    iter += 1
    if (batch_size * iter) % num_samples == 0:
      f_x_prev = f_x
      f_x = np.sum(ObjfQuad(C, y, params)) / num_samples
    #if iter % 10000 == 0:
    #  print 'objf=' + str(f_x), param
  #print 'trial ' + str(t) + ': objf=' + str(f_x), params
  print 'trial ' + str(t) + ': objf=' + str(ObjfQuad(2.0 / 3 * np.eye(2, dtype=np.float64), np.array([0.5, 0.5]), params)), params
  tot_opt += params
  tot_objf += f_x

avg_opt = tot_opt * (1.0 / num_tries)
print "avg_opt is ", avg_opt
print "avg_objf_is ", tot_objf * (1.0 / num_tries)
print "bias = ", avg_opt - optimum_params

