#!/usr/bin/env python

import numpy
import scipy
import scipy.special
import scipy.optimize
import mpmath
mpmath.mp.prec = 64

def square_frobenius_error_norm(matrix):
  return numpy.sum(matrix * matrix);

def relative_difference(a, b):
  return 2.0 * numpy.abs(a - b) / (numpy.abs(a) + numpy.abs(b))

def goalFunction(weights, abscissas, f, fprime):
  assert len(weights) == len(abscissas)
  
  N = len(weights)
  
  functionValues = numpy.array([[f(i, xi) for xi in abscissas] for i in xrange(N)])
  derivativeValues = numpy.array([[fprime(i, xi) for xi in abscissas] for i in xrange(N)])

  functionValues = numpy.reshape(functionValues, (N, N))
  derivativeValues = numpy.reshape(derivativeValues, (N, N))
  
  error_matrix = numpy.einsum('p,ip,jp->ij', weights, functionValues, functionValues) - numpy.eye(N)
  
  weight_derivatives = 2.0 * numpy.einsum('ij,ip,jp->p', error_matrix, functionValues, functionValues)
  abscissa_derivatives = 4.0 * numpy.einsum('ij,p,ip,jp->p', error_matrix, weights, functionValues, derivativeValues)
  
  error_norm = square_frobenius_error_norm(error_matrix)
  
  return error_norm, weight_derivatives, abscissa_derivatives

def bessel_neumann_S_guess(N):
  return 0.5*numpy.sum(numpy.double([mpmath.besseljzero(0, i) for i in xrange(N, N+2)]))

def bessel_dirichlet_S_guess(N):
  return numpy.double(mpmath.besseljzero(0, N+1))

def bessel_initial_guess(N, S_guess):
  bessel_j_zeros = [mpmath.besseljzero(0, i) for i in xrange(1, N+1)]
  abscissas = [zero / S_guess for zero in bessel_j_zeros]
  weights = [2.0 / (S_guess * mpmath.besselj(1, bessel_j_zeros[i]))**2 for i in xrange(N)]

  weights = numpy.reshape(numpy.double(weights), (N,))
  abscissas = numpy.reshape(numpy.double(abscissas), (N,))
  
  return weights, abscissas

def optimise_weights_abscissas(weights, abscissas, f, fprime):
  assert len(weights) == len(abscissas)
  
  N = len(weights)
  initial_guess = numpy.concatenate((weights, abscissas))
  
  def optimise_function(x):
    weights, abscissas = numpy.split(x, 2)
    error_norm, derror_dweight, derror_dabscissa = goalFunction(weights, abscissas, f, fprime)
    return error_norm, numpy.concatenate((derror_dweight, derror_dabscissa))
  
  result = scipy.optimize.minimize(optimise_function, initial_guess, jac = True, options = dict(disp=True))
  return numpy.split(result.x, 2)

def simple_bessel_optimise(N, f, fprime, S_guess):
  
  def optimise_function(S):
    weights, abscissas = bessel_initial_guess(N, S)
    weights = numpy.reshape(weights, (N,))
    abscissas = numpy.reshape(abscissas, (N,))
    error_norm, derror_dweight, derror_dabscissa = goalFunction(weights, abscissas, f, fprime)
    
    derror_ds = -2.0 / S * numpy.sum(derror_dweight * weights) - 1.0/S * numpy.sum(derror_dabscissa * abscissas)
    
    return error_norm, derror_ds
    
  result = scipy.optimize.minimize(optimise_function, S_guess, jac = True, options = dict(disp=True))
  return bessel_initial_guess(N, result.x)

N = 100

def test_bessel_neumann(N):
  S_guess = bessel_neumann_S_guess(N)
  weights, abscissas = bessel_initial_guess(N, S_guess)
  bessel_j_zeros = numpy.double([mpmath.besseljzero(0, i, derivative=True) for i in xrange(1, N+1)])
  bessel_j_values = numpy.sqrt(2.0) / numpy.abs(scipy.special.jn(0, bessel_j_zeros))

  f = lambda i, x: bessel_j_values[i] * scipy.special.jn(0, bessel_j_zeros[i] * x)
  fprime = lambda i, x: -bessel_j_values[i] * bessel_j_zeros[i] * scipy.special.jn(1, bessel_j_zeros[i] * x)

  print 'bessel', goalFunction(weights, abscissas, f, fprime)[0]

  simple_optimised_weights, simple_optimised_abscissas = simple_bessel_optimise(N, f, fprime, S_guess)
  print 'simple bessel optimised', goalFunction(simple_optimised_weights, simple_optimised_abscissas, f, fprime)[0]

  optimised_weights, optimised_abscissas = optimise_weights_abscissas(simple_optimised_weights, simple_optimised_abscissas, f, fprime)
  print 'bessel optimised', goalFunction(optimised_weights, optimised_abscissas, f, fprime)[0]
  

def test_bessel_dirichlet(N):
  S_guess = bessel_dirichlet_S_guess(N)
  weights, abscissas = bessel_initial_guess(N, S_guess)
  bessel_j_zeros = numpy.double([mpmath.besseljzero(0, i) for i in xrange(1, N+1)])
  bessel_j_values = numpy.sqrt(2.0) / numpy.abs(scipy.special.jn(1, bessel_j_zeros))

  f = lambda i, x: bessel_j_values[i] * scipy.special.jn(0, bessel_j_zeros[i] * x)
  fprime = lambda i, x: -bessel_j_values[i] * bessel_j_zeros[i] * scipy.special.jn(1, bessel_j_zeros[i] * x)

  print 'bessel', goalFunction(weights, abscissas, f, fprime)[0]

  simple_optimised_weights, simple_optimised_abscissas = simple_bessel_optimise(N, f, fprime, S_guess)
  print 'simple bessel optimised', goalFunction(simple_optimised_weights, simple_optimised_abscissas, f, fprime)[0]

  optimised_weights, optimised_abscissas = optimise_weights_abscissas(weights, abscissas, f, fprime)
  print 'bessel optimised', goalFunction(optimised_weights, optimised_abscissas, f, fprime)[0]

test_bessel_neumann(N)
# test_bessel_dirichlet(N)