#!/usr/bin/env python

import numpy as np
import math

# minimize f = -0.5 * log(precision) + 0.5 * precision * (x - mu)^2
def ObjfGaussian(x, params):
  objf =  -0.5 * np.log(params[1]) + 0.5 * params[1] * (x - params[0]) * (x - params[0])
  return objf
def GradientGaussian(x, params):
  g = np.stack([params[1] * (params[0] - x), - 0.5 / params[1] + 0.5 * (x - params[0]) * (x - params[0])], -1)
  return g
def HessianGaussian(x, params):
  if len(np.shape(x)) > 0:
    h = np.array(map(lambda x: np.array([[params[1], params[0] - x], [params[0] - x, 0.5 / (params[1] * params[1])]]), x))
  else:  
    h = np.array([[params[1], params[0] - x], [params[0] - x, 0.5 / (params[1] * params[1])]])
  return h

# optimum mean, precision if the data comes from
# underlying distribution with mean, precision.
mu = 0.0
precision = 0.8
optimum_params = np.array([mu, precision])

num_tries = 20000
num_samples = 100

h_bar_inv = np.linalg.inv(HessianGaussian(mu, optimum_params))
tot_opt = np.zeros(2)
tot_bias = np.zeros(2)
for t in range(num_tries):
  np.random.seed(t)
  tot_g = np.zeros(2)
  tot_h = np.zeros((2, 2))
  tot_h_h_bar_inv_g = np.zeros(2)

  for s in range(num_samples):
    x = np.random.normal() * math.sqrt(1.0 / precision)
    this_g = GradientGaussian(x, optimum_params)
    this_h = HessianGaussian(x, optimum_params)
    tot_g += this_g
    tot_h += this_h
    tot_h_h_bar_inv_g += np.dot(this_h, np.dot(h_bar_inv, this_g))

  this_bias = 1.0 / num_samples * np.dot(h_bar_inv, tot_h_h_bar_inv_g / num_samples)
  tot_bias += this_bias
  this_opt = optimum_params - np.dot(np.linalg.inv(tot_h), tot_g)
  tot_opt += this_opt

avg_bias = tot_bias * (1.0 / num_tries)
print "avg_bias is ", avg_bias
avg_opt = tot_opt * (1.0 / num_tries)
print "avg_opt is ", avg_opt

'''
# do actual ML and work out the variance and precision; use these
# to compute the bias.
tot_var = 0.0
tot_stddev = 0.0
tot_log_var = 0.0
tot_precision = 0.0
for t in range(num_tries):
  np.random.seed(t)
  tot_x = 0.0
  tot_x2 = 0.0
  for s in range(num_samples):
    x = np.random.normal() * math.sqrt(1.0 / precision)
    tot_x += x
    tot_x2 += x * x
  avg_x = tot_x / num_samples
  avg_x2 = tot_x2 / num_samples
  var = avg_x2 - avg_x * avg_x
  tot_var += var
  tot_stddev += math.sqrt(var)
  tot_log_var += math.log(var)
  tot_precision += 1.0 / var

print "variance bias = ", ((tot_var / num_tries) - 1.0 / precision)
print "stddev bias = ", ((tot_stddev / num_tries) - math.sqrt(1.0 / precision))
print "log-var bias = ", ((tot_log_var / num_tries) - math.log(1.0 / precision))
print "precision bias = ", ((tot_precision / num_tries) - precision)
'''

num_tries = 1000
num_samples = 100
batch_size = 1
init_params = np.array([0.0, 2.0])
lr_init = 1e-3
decay = 1e-4
eps=1e-8


# SGD
print "SGD..."
tot_opt = np.zeros(2)
tot_objf = 0.0
tot_mean = 0.0
tot_precision = 0.0
for t in range(num_tries):
  np.random.seed(t)
  params = np.copy(init_params)
  f_x = float('Inf')
  data = np.array([np.random.normal() * math.sqrt(1.0 / precision) for i in range(num_samples)])
  #print  'objf at opt', np.sum(ObjfGaussian(data, optimum_params)) / num_samples
  iter = 0
  while iter == 0 or (batch_size * iter) % num_samples != 0 or np.abs(f_x_prev - f_x) >= eps:
    epoch = (batch_size * iter) / num_samples
    lr = lr_init * (1. / (1. + decay * epoch))
    np.random.seed(iter)
    batch = np.random.choice(data, batch_size, replace=False)
    params -= lr * np.sum(GradientGaussian(batch, params), 0) / batch_size
    iter += 1
    if (batch_size * iter) % num_samples == 0:
      f_x_prev = f_x
      f_x = np.sum(ObjfGaussian(data, params)) / num_samples
    #if iter % 1000 == 0:
    #  print 'objf=' + str(f_x), params
  avg_x = np.sum(data) / num_samples
  avg_x2 = np.sum(data * data) / num_samples
  tot_mean += avg_x
  var = avg_x2 - avg_x * avg_x
  tot_precision += 1.0 / var
  print 'closed form params: ', [avg_x, 1.0 / var]
  print 'trial ' + str(t) + ': objf=' + str(np.sum(ObjfGaussian(0.0, params))), params
  tot_opt += params
  tot_objf += f_x

print 'avg closed form params: ', [tot_mean * (1.0 / num_tries), tot_precision * (1.0 / num_tries)]
avg_opt = tot_opt * (1.0 / num_tries)
print "avg_opt is ", avg_opt
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
  data = np.array([np.random.normal() * math.sqrt(1.0 / precision) for i in range(num_samples)])
  #print  'objf at opt', np.sum(ObjfGaussian(data, optimum_params)) / num_samples
  iter = 0
  while iter == 0 or (batch_size * iter) % num_samples != 0 or np.abs(f_x_prev - f_x) >= eps:
    epoch = (batch_size * iter) / num_samples
    lr = lr_init * (1. / (1. + decay * epoch))
    np.random.seed(iter)
    batch = np.random.choice(data, batch_size, replace=False)
    alpha = 0.5 * (np.sqrt(1.0 + 0.8 * 4.0 / (lr * num_samples)) - 1.0)
    params += alpha * lr * np.sum(GradientGaussian(batch, params), 0) / batch_size
    params -= (1 + alpha) * lr * np.sum(GradientGaussian(batch, params), 0) / batch_size
    iter += 1
    if (batch_size * iter) % num_samples == 0:
      f_x_prev = f_x
      f_x = np.sum(ObjfGaussian(data, params)) / num_samples
    #if iter % 10000 == 0:
    #  print 'objf=' + str(f_x), params
  print 'trial ' + str(t) + ': objf=' +  str(np.sum(ObjfGaussian(0.0, params))), params
  tot_opt += params
  tot_objf += f_x
  if math.isnan(f_x):
    print np.isnan(params)

avg_opt = tot_opt * (1.0 / num_tries)
print "avg_opt is ", avg_opt
print "avg_objf_is ", tot_objf * (1.0 / num_tries)
print "bias = ", avg_opt - optimum_params

