#!/usr/bin/env python

import numpy
import scipy
import scipy.special
import scipy.optimize
import mpmath
mpmath.mp.prec = 64

def relative_difference(a, b):
  return numpy.abs(a - b) / (numpy.abs(a) + numpy.abs(b))

def quadrature_goal_function(weights, abscissas, f, fprime, N_basis_functions = None):
  """
  This function computes the error of a given quadrature (i.e. specification of weights and abscissas) for
  a given set of basis functions.  The derivatives of the basis functions are also needed to compute the derivative
  of the error with respect to the weights and abscissas.
  
  Quadrature integration approximates the definite integral \int_a^b w(x) f(x) dx as \sum_i w_i f(x_i) where
  w(x) is a weight function, the w_i are the quadrature weights and the x_i are the abscissas.  The weight function
  and the weights themselves must all be non-negative, and the abscissas x_i must be in the interval [a, b].
  Examples of weight functions include:
    * w(x) = 1          (normal integration over an interval)
    * w(x) = x          (integration over a disk)
    * w(x) = x^2        (integration over a sphere)
    * w(x) = e^(-x^2)   (integration over an infinite domain)
  
  In spectral method codes, we want to decompose the solution in terms of N orthonormal basis functions f_i(x).  However, we often need
  the solution at the grid points either because we need to compute terms like V(x) u(x) or u(x)^2 where u(x) is the solution quantity.
  We can always interpolate the solution onto a given set of coordinate points x_i by using the expansion in terms of the basis functions:
    u(x) = \sum_j u_j f_j(x)
  where u_j is the amplitude of the f_j basis function in the solution.  So, in principle, we can choose the x_i by any process and
  interpolate to compute u(x_i). We can then compute the expression V(x_i) u(x_i) straight-forwardly.  To get the basis decomposition of
  V(x) u(x) we need a way to invert our interpolation.  Now if V(x) u(x) can be represented perfectly in terms of the finite set of basis
  functions f_i(x), then we can just invert our interpolation matrix to determine the basis function decomposition of V(x) u(x). i.e.
  interpolation can be considered to be the application of the matrix T_ij to the vector u_j:
    u(x_i) = T_ij u_j; where T_ij = f_j(x_i)
  In which case, this matrix can be simply inverted to find u_j:
    u_j = T^{-1}_ji u(x_i)
  However, in general, V(x) u(x) will be outside the span of the *finite* set of basis functions.  In this case, applying the above procedure
  can lead to disastrous results.  For example, if we assume that V(x) u(x) is a (normalised) basis function *outside* our basis set, we
  can find that the total amplitude of the basis functions after inversion is very much larger than one.  i.e. the sum of u_i^2 can be large.
  
  An alternative procedure would be to compute the basis coefficients of an arbitrary function by using the orthonormality of the basis functions,
  i.e. u_j = \int_a^b w(x) g(x) f_j(x) dx
  where g(x) is our unknown function and f_j(x) are our basis functions.  This will work well, but typically these integrals will be hard to
  do, or g(x) itself may not be known, it may only be the g(x_i) that are known.  In this case, we need to approximate the integral, which 
  requires the specification of a quadrature formula.  i.e., weights w_i and abscissas x_i.
  
  How should we choose the w_i and x_i in general?  Ideally we'd like our quadrature formula to preserve the orthonormality of our basis functions
  i.e. \sum_p w_p f_i(x_p) f_j(x_p) = \delta_ij
  where \delta_ij is the Kronecker delta.  However, if we have N basis functions and (perhaps arbitrarily) decide to limit ourselves to
  N abscissas x_i, we only have 2N degrees of freedom: N weights w_i and N abscissas x_i.  But we have 1/2 (N^2 + N) orthonormality constraints
  to enforce (N^2, but interchanging i and j above leaves the expression symmetric).
  
  In some situations, f_i(x) f_j(x) can be expressed in terms of basis functions no higher than 2N, in which case, choosing our 2N degrees
  of freedom to exactly integrate the first 2N basis functions ensures that the quadrature integrals of every pair of basis functions
  f_i(x) f_j(x) are equal to their exact integrals.  This situation occurs when the f_i are polynomials with highest degree i-1 (any set of
  orthogonal polynomials satisfies this).  This also occurs for trigonometric series because exp(i n x) exp(i m x) = exp[i (n + m) x].
  
  Unfortunately, this is not always the case, a good example are the Bessel functions.  In this case our basis functions are J_m(k_i x), but
  products of basis functions cannot be expressed as a *finite* sum of other basis functions, i.e. J_m(k_i x) J_m(k_j x) cannot be expressed
  as a finite sum of J_m(k_i x).  In this case, we just need to do the best we can with our 2N degrees of freedom (or increase the number of
  spatial degrees of freedom used.  What we want is for the orthonormality matrix to be as close to the identity as possible.  To quantify this
  we need an error norm.
  
  Here we choose the Frobenius norm.  We choose it because it is simple to calculate it and its derivatives, but also because it can be related
  to meaningful quantities that we want to minimise.  Intuitively, we want all the eigenvalues of the orthonormality matrix to be as close to 1
  as possible to ensure long-term stability of the solution after repeated application of interpolation and quadrature integration operations. 
  If we define the error matrix \Delta_ij = \sum_p w_p f_i(x_p) f_j(x_p) - \delta_ij, then we want the eigenvalues of this matrix to be small.
  We want to minimise something like:
      E_1({w_p}, {x_p}) = \sum_i |\lambda_i|^2
  It turns out that this is the Schatten p-norm of \Delta_ij with p=1.  This is not particularly convenient to calculate because we need to find
  the eigenvalues \lambda_i, and more importantly, to know the derivative of E_1 with respect to the w_p and x_p we need to know how the
  eigenvalues depend on the entries of the error matrix \Delta_ij.  A numerically more convenient norm is the p=2 Schatten p-norm, which is also
  the Frobenius norm:
      E_2({w_p}, {x_p}) = \sqrt{ \sum_i |\lambda_i|^4 } = \sqrt{ \sum_ij |\Delta_ij|^2 }
  The latter expression is simple to evaluate *and* simple to find an expression for the derivative of the error.  This is the error norm we choose.
  
  Actually, we use the square of this because it particularly makes the expressions for the derivatives to be much simpler.
  
  This function computes the square of the Frobenius norm of the error matrix, and returns the derivatives of the error with respect to the 
  weights w_i and abscissas x_i.
  
  Note that all of the above as assumed real basis functions f_i(x).  If they are complex, the relevant orthonormality integral is of course
  \int_a^b w(x) f^*_i(x) f_j(x) dx = \delta_ij
  """
  assert len(weights) == len(abscissas)
  
  N_coordinates = len(weights)
  N_basis_functions = N_basis_functions or N_coordinates
  
  # Evaluate f_i(x_j)
  functionValues = numpy.array([[f(i, xi) for xi in abscissas] for i in xrange(N_basis_functions)])
  # Evaluate f_i'(x_j)
  derivativeValues = numpy.array([[fprime(i, xi) for xi in abscissas] for i in xrange(N_basis_functions)])

  functionValues = numpy.reshape(functionValues, (N_basis_functions, N_coordinates))
  derivativeValues = numpy.reshape(derivativeValues, (N_basis_functions, N_coordinates))
  
  # The error matrix is \Delta_ij = w_p f_i(x_p) f_j(x_p) - \delta_ij
  # Note that numpy.einsum implements Einstein summation.  It's a pretty handy function.
  error_matrix = numpy.einsum('p,ip,jp->ij', weights, functionValues, functionValues) - numpy.eye(N_basis_functions)
  
  # The derivative of the error norm with respect to the weights is:
  #    (dE/dw)_p = 2 \Delta_ij f_i(x_p) f_j(x_p)
  weight_derivatives = 2.0 * numpy.einsum('ij,ip,jp->p', error_matrix, functionValues, functionValues)
  # The derivative of the error norm with respect to the abscissas is:
  #    (dE/dx)_p = 4 \Delta_ij w_p f_i(x_p) f_j'(x_p)
  abscissa_derivatives = 4.0 * numpy.einsum('ij,p,ip,jp->p', error_matrix, weights, functionValues, derivativeValues)
  
  # Finally compute the error quantity 
  error_norm = numpy.einsum('ij,ij', error_matrix, error_matrix)
  
  return error_norm, weight_derivatives, abscissa_derivatives

def bessel_neumann_S_guess(m, N):
  return numpy.double(mpmath.besseljzero(m, N+1, derivative=True))

def bessel_dirichlet_S_guess(m, N):
  return numpy.double(mpmath.besseljzero(m, N+1))

def bessel_initial_guess(m, N, S_guess):
  bessel_j_zeros = [mpmath.besseljzero(m, i) for i in xrange(1, N+1)]
  abscissas = [zero / S_guess for zero in bessel_j_zeros]
  weights = [2.0 / (S_guess * mpmath.besselj(m + 1, bessel_j_zeros[i]))**2 for i in xrange(N)]

  weights = numpy.reshape(numpy.double(weights), (N,))
  abscissas = numpy.reshape(numpy.double(abscissas), (N,))
  
  return weights, abscissas

def bessel_neumann_kspace_initial_guess(m, N, S_guess):
  bessel_j_zeros = [mpmath.besseljzero(m, i, derivative=True) for i in xrange(1, N+1)]
  abscissas = [zero / S_guess for zero in bessel_j_zeros]
  additional_factor = (lambda i: 1.0) if m == 0 else (lambda i: 1.0 - (m/bessel_j_zeros[i])**2)
  
  weights = [2.0 / (S_guess * additional_factor(i) * mpmath.besselj(m, bessel_j_zeros[i]))**2 for i in xrange(N)]
  
  weights = numpy.reshape(numpy.double(weights), (N,))
  abscissas = numpy.reshape(numpy.double(abscissas), (N,))
  
  return weights, abscissas

def optimise_weights_abscissas(weights, abscissas, f, fprime, N_basis_functions = None):
  assert len(weights) == len(abscissas)
  
  N_coordinates = len(weights)
  N_basis_functions = N_basis_functions or N_coordinates
  initial_guess = numpy.concatenate((weights, abscissas))
  
  def optimise_function(x):
    weights, abscissas = numpy.split(x, 2)
    error_norm, derror_dweight, derror_dabscissa = quadrature_goal_function(weights, abscissas, f, fprime, N_basis_functions)
    return error_norm, numpy.concatenate((derror_dweight, derror_dabscissa))
  
  result = scipy.optimize.minimize(optimise_function, initial_guess, jac = True, options = dict(disp=True))
  return numpy.split(result.x, 2)

def simple_bessel_optimise(m, N_coordinates, f, fprime, S_guess, N_basis_functions = None):
  N_basis_functions = N_basis_functions or N_coordinates
  
  def optimise_function(S):
    weights, abscissas = bessel_initial_guess(m, N_coordinates, S)
    weights = numpy.reshape(weights, (N_coordinates,))
    abscissas = numpy.reshape(abscissas, (N_coordinates,))
    error_norm, derror_dweight, derror_dabscissa = quadrature_goal_function(weights, abscissas, f, fprime, N_basis_functions)
    
    derror_ds = -2.0 / S * numpy.sum(derror_dweight * weights) - 1.0/S * numpy.sum(derror_dabscissa * abscissas)
    
    return error_norm, derror_ds
    
  result = scipy.optimize.minimize(optimise_function, S_guess, jac = True, options = dict(disp=True))
  return bessel_initial_guess(m, N, result.x)

def simple_bessel_neumann_optimise(m, N_coordinates, f, fprime, S_guess, N_basis_functions = None):
  N_basis_functions = N_basis_functions or N_coordinates
  
  def optimise_function(S):
    weights, abscissas = bessel_neumann_kspace_initial_guess(m, N_coordinates, S)
    weights = numpy.reshape(weights, (N_coordinates,))
    abscissas = numpy.reshape(abscissas, (N_coordinates,))
    error_norm, derror_dweight, derror_dabscissa = quadrature_goal_function(weights, abscissas, f, fprime, N_basis_functions)
    
    derror_ds = -2.0 / S * numpy.sum(derror_dweight * weights) - 1.0/S * numpy.sum(derror_dabscissa * abscissas)
    
    return error_norm, derror_ds
    
  result = scipy.optimize.minimize(optimise_function, S_guess, jac = True, options = dict(disp=True))
  return bessel_neumann_kspace_initial_guess(m, N, result.x)


N = 40

def test_bessel(m, S_guess, N_coordinates, bessel_j_zeros, bessel_j_values):
  N_coordinates = N_coordinates or len(bessel_j_zeros)
  N_basis_functions = len(bessel_j_zeros)
  
  f = lambda i, x: bessel_j_values[i] * scipy.special.jn(m, bessel_j_zeros[i] * x)
  fprime = lambda i, x: bessel_j_values[i] * bessel_j_zeros[i] * scipy.special.jvp(m, bessel_j_zeros[i] * x)

  weights, abscissas = bessel_initial_guess(m, N_coordinates, S_guess)
  print 'bessel', quadrature_goal_function(weights, abscissas, f, fprime, N_basis_functions)[0]

  simple_optimised_weights, simple_optimised_abscissas = simple_bessel_optimise(m, N_coordinates, f, fprime, S_guess, N_basis_functions)
  print 'simple bessel optimised', quadrature_goal_function(simple_optimised_weights, simple_optimised_abscissas, f, fprime, N_basis_functions)[0]

  optimised_weights, optimised_abscissas = optimise_weights_abscissas(simple_optimised_weights, simple_optimised_abscissas, f, fprime, N_basis_functions)
  print 'bessel optimised', quadrature_goal_function(optimised_weights, optimised_abscissas, f, fprime, N_basis_functions)[0]
  return optimised_weights, optimised_abscissas


def test_bessel_neumann(m, N_coordinates, N_basis_functions = None):
  N_basis_functions = N_basis_functions or N_coordinates
  
  S_guess = bessel_neumann_S_guess(m, N_coordinates)
  bessel_j_zeros = numpy.double([mpmath.besseljzero(m, i, derivative=True) for i in xrange(1, N_basis_functions+1)])
  bessel_j_values = numpy.sqrt(2.0) / numpy.abs(scipy.special.jn(m, bessel_j_zeros))
  
  if m > 0:
    bessel_j_values /= numpy.sqrt(1.0 - (m / bessel_j_zeros)**2)

  return test_bessel(m, S_guess, N_coordinates, bessel_j_zeros, bessel_j_values)

def test_bessel_dirichlet(m, N_coordinates, N_basis_functions = None):
  N_basis_functions = N_basis_functions or N_coordinates
  
  S_guess = bessel_dirichlet_S_guess(m, N_coordinates)
  bessel_j_zeros = numpy.double([mpmath.besseljzero(m, i) for i in xrange(1, N+1)])
  bessel_j_values = numpy.sqrt(2.0) / numpy.abs(scipy.special.jn(m + 1, bessel_j_zeros))
  
  return test_bessel(m, S_guess, N_coordinates, bessel_j_zeros, bessel_j_values)

def test_bessel_neumann_neumann(m, N_coordinates, N_basis_functions = None):
  N_basis_functions = N_basis_functions or N_coordinates
  
  S_guess = bessel_neumann_S_guess(m, N_coordinates)
  bessel_j_zeros = numpy.double([mpmath.besseljzero(m, i, derivative=True) for i in xrange(1, N_basis_functions+1)])
  bessel_j_values = numpy.sqrt(2.0) / numpy.abs(scipy.special.jn(m, bessel_j_zeros))
  
  if m > 0:
    bessel_j_values /= numpy.sqrt(1.0 - (m / bessel_j_zeros)**2)
  
  f = lambda i, x: bessel_j_values[i] * scipy.special.jn(m, bessel_j_zeros[i] * x)
  fprime = lambda i, x: bessel_j_values[i] * bessel_j_zeros[i] * scipy.special.jvp(m, bessel_j_zeros[i] * x)
  
  weights, abscissas = bessel_neumann_kspace_initial_guess(m, N_coordinates, S_guess)
  print 'bessel', quadrature_goal_function(weights, abscissas, f, fprime, N_basis_functions)[0]
  
  simple_optimised_weights, simple_optimised_abscissas = simple_bessel_neumann_optimise(m, N_coordinates, f, fprime, S_guess, N_basis_functions)
  print 'simple bessel optimised', quadrature_goal_function(simple_optimised_weights, simple_optimised_abscissas, f, fprime, N_basis_functions)[0]
  
  optimised_weights, optimised_abscissas = optimise_weights_abscissas(simple_optimised_weights, simple_optimised_abscissas, f, fprime, N_basis_functions)
  print 'bessel optimised', quadrature_goal_function(optimised_weights, optimised_abscissas, f, fprime, N_basis_functions)[0]
  return optimised_weights, optimised_abscissas

test_bessel_neumann(1, N)
# w2, x2 = test_bessel_neumann_neumann(0, N)

# print relative_difference(w1, w2)
# print relative_difference(x1, x2)
# print x1
# print x2

test_bessel_dirichlet(1, N)