#!/usr/bin/env python3

import numpy as np
from scipy.interpolate import interp1d


def euler_critical_values(simp_comp, direction):
    """
    Returns an ordered list of Euler critical values for an embedded simplicial complex filtered in given direction
    :param simp_comp: Embedded simplicial complex stored as list of simplices. Each simplex is a list of vertices,
                        given as tuples.
    :param direction: Direction (i.e. unit vector) to filter in. Given as tuple.
    :return: Ordered list of Euler critical values, filtration as list of elements (simplex, filtration_value)
    """
    direction = np.array(direction)
    critical_vals = []
    filtration = []

    for simplex in simp_comp:
        fv = np.max([np.array(v).dot(direction) for v in simplex])
        critical_vals.append(fv)
        filtration.append((simplex, fv))

    critical_vals = list(set(critical_vals))
    critical_vals.sort()

    return critical_vals, filtration


def euler_char(filtration, threshold, max_dim: int = 3):
    char = 0
    for i in range(len(filtration)):
        simplex, fv = filtration[i]
        if fv <= threshold:
            char += (-1) ** simplex.shape[0]

    return -char


def euler_curve(simp_comp, direction, interval=(-1, 1), points: int = 100):
    """
    Returns Euler curve as evenly spaced evaluations on given interval
    :param simp_comp: Embedded simplicial complex
    :param direction: Direction (i.e. unit vector) for filtration as tuple
    :param interval: Interval on which to evaluate the Euler curve on
    :param points: Number of evenly spaced points at which to evaluate the Euler curve
    :return: 1-D array representing evenly spaced evaluations of Euler curve on interval
    """
    critical_values, filtration = euler_critical_values(simp_comp, direction)
    value, c = 0, 0
    step_size = (interval[1] - interval[0]) / (points - 1)

    chi = []

    # Includes first point of interval but not last! x spaced as (interval[1] - interval[0]) / points
    critical_values.insert(0, interval[0])
    for x in np.linspace(interval[0], interval[1], points):
        if x - step_size < critical_values[c] <= x:
            value = euler_char(filtration, critical_values[c])
            while critical_values[c] <= x + step_size and c < len(critical_values) - 1:
                c += 1

        chi.append(value)

    return np.array(chi, dtype=np.float)


def euler_curve_callable(simp_comp, direction, interval=(-1, 1)):
    """
    Returns Euler curve as evenly spaced evaluations on given interval
    :param simp_comp: Embedded simplicial complex
    :param direction: Direction (i.e. unit vector) for filtration as tuple
    :param interval: Interval on which to evaluate the Euler curve on
    :return: 1-D array representing evenly spaced evaluations of Euler curve on interval
    """
    critical_values, filtration = euler_critical_values(simp_comp, direction)
    # Includes first point of interval but not last! x spaced as (interval[1] - interval[0]) / points

    critical_values.insert(0, interval[0])

    def ret_func(alpha):
        if alpha >= critical_values[-1]:
            return euler_char(filtration, critical_values[-1])
        else:
            i = 0
            while alpha >= critical_values[i] and i < len(critical_values):
                i += 1
            return euler_char(filtration, critical_values[i-1])

    return ret_func, np.array(critical_values)


def cumulative_euler_curve(simp_comp, direction=None, interval=(-1, 1), points: int = 100, factor: int = 3):
    """
    Evaluations of cumulative Euler curve (i.e. integral of EC - mean on given interval)
    :param simp_comp: Embedded simplicial complex
    :param direction: Direction (i.e. unit vector) for filtration as tuple
    :param interval: Interval on which to evaluate the Euler curve on
    :param points: Number of evenly spaced points at which to evaluate the Euler curve
    :param factor: how many points to use for integration
    :return: 1-D array representing evenly spaced evaluations of cumulative Euler curve on interval
    """
    ec = euler_curve(simp_comp, direction, interval, (points - 1) * factor)
    step_size = (interval[1] - interval[0]) / (points * factor - 1)

    ec -= np.mean(ec)
    cec = [0]
    for i in range(points - 1):
        val = sum([ec[i * factor + j] for j in range(factor)])
        cec.append(cec[-1] + val * step_size)

    return np.array(cec)


def integrate_ec_transform(simp_comp, k=20, interval=(-1, 1), points: int = 100, mode='mean'):
    """
    Currently only for shapes embedded into 2D
    Integrates over cumulative Euler curves over all directions in S¹
    :param simp_comp: Embedded simplicial complex
    :param k: number of evenly spaced points on S¹ to use for averaging
    :param interval: interval over which to construct cumulative Euler curves and Euler curves
    :param points: number of evaluations of the above curve
    :param mode: to return mean or std over points
    :return: 1D array, integrating the above curves over all directions in S¹
    """
    cecs = []
    for theta in np.linspace(0, 2 * np.pi, k, endpoint=False):
        direction = (np.sin(theta), np.cos(theta))
        cecs.append(cumulative_euler_curve(simp_comp, direction, interval=interval, points=points))

    if mode == 'mean':
        return np.mean(np.stack(cecs), axis=0)
    elif mode == 'std':
        return np.std(np.stack(cecs), axis=0)
    elif mode == 'dist':
        return np.stack(cecs)
