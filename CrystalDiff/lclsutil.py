import numpy as np

from CrystalDiff import crystal

"""
This module contains the most high-level functions. None of the
other modules can depend on this module.
"""


def get_crystal_list(num,
                     hlen=None,
                     eta=None,
                     theta=None,
                     rho=None,
                     tau=None,
                     surface_points=None,
                     chi0=None,
                     chih_sigma=None,
                     chih_pi=None,
                     chihbar_sigma=None,
                     chihbar_pi=None):
    """
    Get a list of crystals with the specified properties.

    :param num:
    :param hlen:
    :param eta:
    :param theta:
    :param rho:
    :param tau:
    :param surface_points:
    :param chi0:
    :param chih_sigma:
    :param chih_pi:
    :param chihbar_sigma:
    :param chihbar_pi:
    :return:
    """
    # First get the list
    crystal_list = [crystal.CrystalBlock3D() for _ in range(num)]

    # ----------------------------------------------
    # Set h vectors, surface normal and surface position
    # For simplicity, we always assumes that the h vector is in the x-y plane of the device frame
    # ----------------------------------------------
    for idx in range(num):
        my_crystal = crystal_list[idx]
        my_crystal.set_thickness(1e6)

        # Update the reciprocal lattice
        if hlen is not None:
            h = np.array([np.sin(eta[idx]),
                          np.cos(eta[idx]) * np.sin(theta[idx]),
                          np.cos(eta[idx]) * np.cos(theta[idx])])
            my_crystal.set_h(h * hlen[idx])

        if rho is not None:
            normal = np.array([np.sin(tau[idx]),
                               np.cos(tau[idx]) * np.sin(rho[idx]),
                               np.cos(tau[idx]) * np.cos(rho[idx])])
            my_crystal.set_surface_normal(normal)

        # Update the surface points
        if surface_points is not None:
            my_crystal.set_surface_point(surface_points[idx])

        # ----------------------------------------------
        # Set chi values
        # ----------------------------------------------
        if chi0 is not None:
            my_crystal.set_chi0(chi0)
        if chih_sigma is not None:
            my_crystal.set_chih_sigma(chih_sigma)
        if chihbar_sigma is not None:
            my_crystal.set_chihbar_sigma(chihbar_sigma)
        if chih_pi is not None:
            my_crystal.set_chih_pi(chih_pi)
        if chihbar_pi is not None:
            my_crystal.set_chihbar_pi(chihbar_pi)

    return crystal_list


def update_crystal_list(crystal_list,
                        hlen=None,
                        eta=None,
                        theta=None,
                        rho=None,
                        tau=None,
                        surface_points=None,
                        chi0=None,
                        chih_sigma=None,
                        chih_pi=None,
                        chihbar_sigma=None,
                        chihbar_pi=None):
    """
    Get a list of crystals with the specified properties.

    :param crystal_list:
    :param hlen:
    :param eta:
    :param theta:
    :param rho:
    :param tau:
    :param surface_points:
    :param chi0:
    :param chih_sigma:
    :param chih_pi:
    :param chihbar_sigma:
    :param chihbar_pi:
    :return:
    """
    # First get the number of crystals in this list
    num = len(crystal_list)

    # ----------------------------------------------
    # Set h vectors, surface normal and surface position
    # For simplicity, we always assumes that the h vector is in the x-y plane of the device frame
    # ----------------------------------------------
    for idx in range(num):
        my_crystal = crystal_list[idx]
        my_crystal.set_thickness(1e6)

        # Update the reciprocal lattice
        if hlen is not None:
            h = np.array([np.sin(eta[idx]),
                          np.cos(eta[idx]) * np.sin(theta[idx]),
                          np.cos(eta[idx]) * np.cos(theta[idx])])
            my_crystal.set_h(h * hlen[idx])

        if rho is not None:
            normal = np.array([np.sin(tau[idx]),
                               np.cos(tau[idx]) * np.sin(rho[idx]),
                               np.cos(tau[idx]) * np.cos(rho[idx])])
            my_crystal.set_surface_normal(normal)

        # Update the surface points
        if surface_points is not None:
            my_crystal.set_surface_point(surface_points[idx])

        # ----------------------------------------------
        # Set chi values
        # ----------------------------------------------
        if chi0 is not None:
            my_crystal.set_chi0(chi0)
        if chih_sigma is not None:
            my_crystal.set_chih_sigma(chih_sigma)
        if chihbar_sigma is not None:
            my_crystal.set_chihbar_sigma(chihbar_sigma)
        if chih_pi is not None:
            my_crystal.set_chih_pi(chih_pi)
        if chihbar_pi is not None:
            my_crystal.set_chihbar_pi(chihbar_pi)

    return crystal_list
