import numpy as np

from CrystalDiff import crystal

"""
This module contains the most high-level functions. None of the
other modules can depend on this module.
"""


#######################################################################################
#                  Get configurations for the regular split delay
#######################################################################################
def get_delay_line_angles(angle_offset,
                          theta,
                          rho,
                          inclined_angle=0.,
                          asymmetric_angle=0.):
    """
    I need to include deviation parameters with respect to the ideal position.

    :param angle_offset: The is the angle adjusted from the grating diffraction
    :param theta: The geometric bragg angle.
    :param rho: The angle associated with the normal direction
    :param inclined_angle: The angle associated for the inclination. This change the normal direction of
                            the reflection surface angle
    :param asymmetric_angle: The angle associated with the asymmetric reflection. This changes the
                                normal direction of the reflection surface angle.
    :return:
    """
    theta_vals = np.array([theta,
                           np.pi + theta,
                           - theta,
                           np.pi - theta]) + angle_offset

    rho_vals = np.array([rho,
                         np.pi + rho - asymmetric_angle,
                         - rho,
                         np.pi - rho + asymmetric_angle]) + angle_offset

    tau_vals = np.array([0, inclined_angle, inclined_angle, 0])

    eta_vals = np.zeros(4)
    return theta_vals, eta_vals, rho_vals, tau_vals


#######################################################################################
#                  Get bragg crystal lists
#######################################################################################
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


#########################################################################################
#                       Jump between frames
#########################################################################################
def get_output_frame(displacement, observe, pulse, crystal_lists):
    # ------------------------------
    # Shift the position
    # ------------------------------
    pulse.x0 += displacement
    # Shift the observation position
    observe += displacement

    # Shift the grating
    for crystal_list in crystal_lists:
        for x in crystal_list:
            x.shift(displacement=displacement)

    return pulse, observe, crystal_lists


def get_intermediate_frame(displacement, rotation, observe, pulse, crystal_lists):
    """
    Shift the crystals and the rotate the frame to calculate the intermediate field
    
    :param displacement: 
    :param rotation: 
    :param observe: 
    :param pulse: 
    :param crystal_lists: 
    :return: 
    """
    # ------------------------------
    # Shift the position
    # ------------------------------
    pulse.x0 += displacement
    # Shift the observation position
    observe += displacement

    # Shift the grating
    for crystal_list in crystal_lists:
        for x in crystal_list:
            x.shift(displacement=displacement)

    # ------------------------------
    # Rotate the crystals, observation and pulse origin
    # ------------------------------
    pulse.x0 = np.dot(rotation, pulse.x0)
    pulse.k0 = np.dot(rotation, pulse.k0)
    pulse.polar = np.dot(rotation, pulse.polar)

    pulse.sigma_mat = np.dot(np.dot(rotation, pulse.sigma_mat), rotation.T)

    # Shift the observation position
    observe = np.dot(rotation, observe)

    # Shift the grating
    for crystal_list in crystal_lists:
        for x in crystal_list:
            x.rotate(rot_mat=rotation)
            x.set_surface_point(surface_point=rotation.dot(x.surface_point))

    return pulse, observe, crystal_lists


