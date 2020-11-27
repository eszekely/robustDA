import numpy as np
import random


def partition(list_in, n):
    random.shuffle(list_in)
    return [list(list_in[i::n]) for i in range(n)]


def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier


def nonlinear_anchors(y, fct):
    if fct == "square":
        nonlinear_y = y ** 2
    elif fct == "abs":
        nonlinear_y = np.abs(y)

    return nonlinear_y


def display_nonlinear_anchors(fct):
    if fct == "square":
        str_fct = "$A^2$"
    elif fct == "abs":
        str_fct = "$|A|$"

    return str_fct


def compute_distance(ideal, pf_X, pf_Y):
    d = np.zeros([len(pf_X), 1])
    for i in range(len(pf_X)):
        d[i] = np.sqrt(0.5*(ideal[0] - pf_X[i])**2 + 0.5*(ideal[1] - pf_Y[i])**2)
    return d
