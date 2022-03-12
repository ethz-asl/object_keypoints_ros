#!/usr/bin/env python3

import pytest
import numpy as np
np.seterr(all='raise')
import time
from cylinder_fitting import fit
from cylinder_fitting import show_fit
from cylinder_fitting import show_G_distribution
from cylinder_fitting import geometry
from cylinder_fitting import fitting_rmsd


def make_points_on_a_cylinder(theta, phi, C, r, N):
    '''Make N points on a cylinder defined by the center C, direction defined theta and phi and radius r.
    Also return the direction of the cylinder'''
    
    M = np.dot(geometry.rotation_matrix_from_axis_and_angle(np.array([0, 0, 1]), phi),
               geometry.rotation_matrix_from_axis_and_angle(np.array([0, 1, 0]), theta))

    x = np.dot(M, np.array([1, 0, 0]))
    y = np.dot(M, np.array([0, 1, 0]))
    z = np.dot(M, np.array([0, 0, 1]))

    delta = np.radians(10)
    t = 0.1

    return [C + r * (np.cos(i * delta) * x + np.sin(i * delta) * y + i * t * z) for i in range(N)], z

def test_fit():
    print("test fit.")

    C = np.array([0, 0, 0])
    r = 10
    data, w = make_points_on_a_cylinder(1, -0.3, C, r, 6)

    tic = time.perf_counter()
    w_fit, C_fit, r_fit, fit_err = fit(data)
    toc = time.perf_counter()
    #show_G_distribution(data)

    print("Fitting RMSD =", fitting_rmsd(w_fit, C_fit, r_fit, data))
    
    
    show_fit(w_fit, C_fit, r_fit, data)
    
    
    print(f"Fitting done in {toc - tic:0.4f} seconds")
    
    assert(np.absolute(1 - np.absolute(np.dot(w, w_fit))) < 1e-4)

    assert(np.absolute(np.dot(w, C - C_fit)) / np.linalg.norm(w) / np.linalg.norm(C - C_fit) > 0.999)

    assert(np.absolute(r - r_fit) < r_fit * 1e-4)
    
    
                         
if __name__ == "__main__":
    test_fit()
