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

import numpy as np
import scipy.linalg
import scipy.stats


def local_entropy(p):
    """
    :param p: p is the input probability transition matrix of a Markov Chain
    :return: H(P_k.) the local entropy of each state in the chain
    """
    l = np.copy(p)
    l[p > 0] = np.log2(p[p > 0])
    k = np.dot(p, np.transpose(l))
    entropy_out = -1*np.diagonal(k)
    return entropy_out.reshape((p.shape[0], 1))


def get_segment_entropy(probability_transition_matrix, start, end):
    """
    :param probability_transition_matrix: the modified probability transition matrix that the end and destination
    states are absorbing
    :param start: is the starting state of the markov trajectory segment
    :param end: is the ending state of the markov trajectory segment
    :return: returns the entropy of the specified segment of our trajectory where the end state and the destination
    in the probability matrix are absorbing
    """
    p_matrix = probability_transition_matrix.copy()
    subset = np.array([x for x in range(p_matrix.shape[0]) if x != end])
    qd = p_matrix[subset]
    qd = qd[:, subset]
    indicies = range(p_matrix.shape[0])
    indicies = [x for x in indicies if x != end]
    temp_mat = np.identity(qd.shape[0]) - qd
    temp_mat = np.linalg.pinv(temp_mat)
    l_ent = local_entropy(p_matrix)
    H_sd = 0
    for i in range(len(indicies)):
        H_sd += (temp_mat[start, i]*l_ent[indicies[i]])[0]
    return H_sd


def get_ad(matrix, dst):
    """
    :param matrix: the probability transition matrix of a markov chain that one intermediate state and destination
     states are absorbing
    :param dst: destination state
    :return: the vector ad, each element 'i' in ad corresponds to the probability from state 'i' the path get trapped
     in the destination state
    """
    ad = np.real(scipy.linalg.eig(matrix)[1][:, destination])
    ad = ad * 1/ad[dst]
    return ad


def absorbing_p(probability_transition_matrix, absorbing_states):
    """
    creates a new transition matrix in which the states contained in absorbing_states will be absorbing, in order to
    be able to compute the probability that a trajectory will at least visit one specific intermediate state before
    reaching the destination state
    :param probability_transition_matrix: probability transition matrix to be changed
    :param absorbing_states: the states that should become absorbing
    :return: the modified matrix that contains the absorbing states
    """
    new_p = probability_transition_matrix.copy()
    for state in absorbing_states:
        new_p[state, ] = 0
        new_p[state, state] = 1
    return new_p


def compute_p_prime(probability_transition_matrix, intermediate_state, destination):
    """
    :param intermediate_state: the sequence of intermediate states visited before reaching the destination
    :param destination: destination state
    :return: a new probability transition matrix in which the destination and the intermediate state are absorbing and
    also some values are specifically modified for the queried segment.
    """
    p_prime = probability_transition_matrix.copy()
    p_bar = absorbing_p(probability_transition_matrix, [intermediate_state, destination])
    ad = get_ad(p_bar, destination)
    n = p_prime.shape[0]
    for i in range(n):
        for j in range(n):
            if i in [intermediate_state, destination] and i != j:
                p_prime[i, j] = 0.0
            elif i in [intermediate_state, destination] and i == j:
                p_prime[i, j] = 1.0
            elif i not in [intermediate_state, destination] and ad[i] < 1:
                p_prime[i, j] = probability_transition_matrix[i, j] * ((1-ad[j])/(1-ad[i]))
    return p_prime


def conditional_trajectory_entropy(probability_transition_matrix, source, destination, intermediate_states):
    """
    Here is the main algorithm as described in the "The entropy of conditional markov trajectories" paper.
    :param probability_transition_matrix: The original probability transition matrix of our MC
    :param source: the starting state
    :param destination: the destination state
    :param intermediate_states: the sequence of visited states before reaching the destination, our entropy is
    conditioned on visiting these states
    :return: the entropy of a conditional markov trajectories
    """
    states = [source]
    states.extend(intermediate_states)
    sum_of_entropy_values = 0
    for index in range((len(states)-1)):
        p_prime = compute_p_prime(probability_transition_matrix, states[(index + 1)], destination)
        segment_entropy = get_segment_entropy(p_prime, states[index], states[(index + 1)])
        sum_of_entropy_values += segment_entropy
    segment_entropy = get_segment_entropy(probability_transition_matrix, states[(len(states)-1)], destination)
    sum_of_entropy_values += segment_entropy
    return sum_of_entropy_values


if __name__ == "__main__":
    P = np.zeros((5, 5))
    P[0, 1], P[0, 2] = 0.25, 0.75
    P[1, 4] = 1
    P[2, 1], P[2, 3] = 0.5, 0.5
    P[3, 4] = 1
    P[4, 0], P[4, 3] = 0.5, 0.5
    np.set_printoptions(precision=4, suppress=True)
    print "Transition probability matrix \n {}".format(P)
    source = int(input("Enter the source state index number: "))
    destination = int(input("Enter the destination state index number: "))
    print("Enter the intermediate states index numbers, "
          "(please press enter after each entry, when done enter use the number -1)")
    entry = ""
    intermediate_states = []
    while True:
        entry = int(input("your entry: "))
        if entry == -1:
            break
        else:
            intermediate_states.append(entry)
    value = conditional_trajectory_entropy(P, source, destination, intermediate_states)
    value = value if value > 0.0000000001 else 0
    print(value)