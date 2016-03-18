"""
Author: Farbod Faghihi

This code computes the entropy of a markov trajectory conditioned on visiting some intermediate states on the way
 before reaching the destination state.

The entropy of the whole markov trajectories were done before based on the following paper:

  The entropy of Markov Trajectories
  http://www-isl.stanford.edu/~cover/papers/paper101.pdf

This code implements the algorithm as described by the following paper for computing the entropy of conditional
markov trajectories:

  The entropy of conditional Markov trajectories
  http://arxiv.org/abs/1212.2831

"""

