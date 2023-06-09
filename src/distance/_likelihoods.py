from numpy import exp, Inf
# from numpy import tanh, arctan

# Basic convex models
def identity(x): return x
def inverse(x): 1/x if x != 0 else Inf
def negexp(x): return exp(-x)


# Sigmoid models
def logistic(x): return 1 / (1 + negexp(x))
# def tanh(x): return tanh(x)
# def arctan(x): return arctan(x)
