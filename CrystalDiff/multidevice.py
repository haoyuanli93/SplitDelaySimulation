import numpy as np

from CrystalDiff import util

two_pi = 2. * np.pi


####################################################
#       Single incident wave vector
####################################################
def get_kout(device_list, kin):
    """
    Get the output momentum vectors from each device.

    :param device_list:
    :param kin:
    :return:
    """

    # Create a variable for the kout list.
    # The reason to use is numpy array is that it's easy to determine the
    # total number of kouts generates and with numpy array, it might be more
    # efficient.
    kout_list = np.zeros((len(device_list) + 1, 3), dtype=np.float64)
    kout_list[0] = kin[:]

    for idx in range(len(device_list)):
        # Get the device
        device = device_list[idx]

        # Get the output wave vector
        kout_list[idx + 1] = util.get_kout(device=device,
                                           kin=kout_list[idx])

    return kout_list


def get_lightpath(device_list, kin, initial_point, final_plane_point, final_plane_normal):
    """
    This function is used to generate the light path of the incident wave vector in the series of
    devices.

    This function correctly handles the light path through the telescopes.

    :param device_list:
    :param kin:
    :param initial_point:
    :param final_plane_normal:
    :param final_plane_point:
    :return:
    """

    # Create a holder for kout vectors
    kout_list = [np.copy(kin), ]

    # Create a list for the intersection points
    intersection_list = [np.copy(initial_point)]

    # Path length
    path_length = 0.

    # Loop through all the devices.
    for idx in range(len(device_list)):

        ###############################################################
        # Step 1: Get the device
        device = device_list[idx]

        ###############################################################
        # Step 2: Find the intersection and kout
        if device.type == "Crystal: Bragg Reflection":
            intersection_list.append(util.get_intersection(s=intersection_list[-1],
                                                           k=kout_list[-1],
                                                           n=device.normal,
                                                           x0=device.surface_point))
            # Find the path length
            displacement = intersection_list[-1] - intersection_list[-2]
            path_length += np.dot(displacement, kout_list[-1]) / util.l2_norm(kout_list[-1])

            # Find the output k vector
            kout_list.append(util.get_bragg_kout(kin=kout_list[-1],
                                                 h=device.h,
                                                 normal=device.normal))

        if device.type == "Transmissive Grating":
            intersection_list.append(util.get_intersection(s=intersection_list[-1],
                                                           k=kout_list[-1],
                                                           n=device.normal,
                                                           x0=device.surface_point))
            # Find the path length
            displacement = intersection_list[-1] - intersection_list[-2]
            path_length += np.dot(displacement, kout_list[-1]) / util.l2_norm(kout_list[-1])

            # Find the wave vecotr
            kout_list.append(kout_list[-1] + device.momentum_transfer)

    ################################################################
    # Step 3: Find the output position on the observation plane
    intersection_list.append(util.get_intersection(s=intersection_list[-1],
                                                   k=kout_list[-1],
                                                   x0=final_plane_point,
                                                   n=final_plane_normal))
    # Update the path length
    displacement = intersection_list[-1] - intersection_list[-2]
    path_length += np.dot(displacement, kout_list[-1]) / util.l2_norm(kout_list[-1])

    return intersection_list, kout_list, path_length


def get_trajectory(device_list, kin, initial_point, final_plane_point, final_plane_normal):
    """
    This function is used to generate the light path of the incident wave vector in the series of
    devices.

    This function correctly handles the light path through the telescopes.

    :param device_list:
    :param kin:
    :param initial_point:
    :param final_plane_normal:
    :param final_plane_point:
    :return:
    """

    # Create a holder for kout vectors
    kout_list = [np.copy(kin), ]

    # Create a list for the intersection points
    intersection_list = [np.copy(initial_point)]

    # Path length
    path_length = 0.

    # Loop through all the devices.
    for idx in range(len(device_list)):

        ###############################################################
        # Step 1: Get the device
        device = device_list[idx]

        ###############################################################
        # Step 2: Find the intersection and kout
        if device.type == "Crystal: Bragg Reflection":
            intersection_list.append(util.get_intersection(s=intersection_list[-1],
                                                           k=kout_list[-1],
                                                           n=device.normal,
                                                           x0=device.surface_point))
            # Find the path length
            displacement = intersection_list[-1] - intersection_list[-2]
            path_length += np.dot(displacement, kout_list[-1]) / util.l2_norm(kout_list[-1])

            # Find the output k vector
            kout_list.append(util.get_bragg_kout(kin=kout_list[-1],
                                                 h=device.h,
                                                 normal=device.normal))

        if device.type == "Transmissive Grating":
            intersection_list.append(util.get_intersection(s=intersection_list[-1],
                                                           k=kout_list[-1],
                                                           n=device.normal,
                                                           x0=device.surface_point))
            # Find the path length
            displacement = intersection_list[-1] - intersection_list[-2]
            path_length += np.dot(displacement, kout_list[-1]) / util.l2_norm(kout_list[-1])

            # Find the wave vecotr
            kout_list.append(kout_list[-1] + device.momentum_transfer)

    ################################################################
    # Step 3: Find the output position on the observation plane
    intersection_list.append(util.get_intersection(s=intersection_list[-1],
                                                   k=kout_list[-1],
                                                   x0=final_plane_point,
                                                   n=final_plane_normal))
    # Update the path length
    displacement = intersection_list[-1] - intersection_list[-2]
    path_length += np.dot(displacement, kout_list[-1]) / util.l2_norm(kout_list[-1])

    return intersection_list, kout_list


def get_intensity_efficiency_sigma_polarization(device_list, kin):
    """
    Get the reflectivity of this kin.
    Notice that this function is not particularly useful.
    It just aims to make the function lists complete.

    :param device_list:
    :param kin:
    :return:
    """
    efficiency_list = np.zeros(len(device_list), dtype=np.float64)

    # Variable for the kout
    kout_list = np.zeros((len(device_list) + 1, 3), dtype=np.float64)
    kout_list[0] = kin[:]

    # Loop through all the devices
    for idx in range(len(device_list)):
        # Get the device
        device = device_list[idx]

        # Get the efficiency
        efficiency_list[idx] = util.get_intensity_efficiency_sigma_polarization(device=device,
                                                                                kin=kout_list[idx])
        # Get the output wave vector
        kout_list[idx + 1] = util.get_kout(device=device, kin=kout_list[idx])

    # Get the overall efficiency
    total_efficiency = np.prod(efficiency_list)

    return total_efficiency, efficiency_list, kout_list


####################################################
#       Multiple incident wave vector
####################################################
def get_output_efficiency_curve(device_list, kin_list):
    """
    Get the reflectivity for each kin.

    :param kin_list:
    :param device_list:
    :return:
    """
    d_num = len(device_list)  # number of devices
    k_num = kin_list.shape[0]  # number of kin vectors

    efficiency_holder = np.zeros((k_num, d_num))
    kout_holder = np.zeros((k_num, d_num + 1, 3))
    total_efficiency_holder = np.zeros(k_num)

    # Loop through all the kin
    for idx in range(k_num):
        (total_efficiency,
         efficiency_tmp,
         kout_tmp) = get_intensity_efficiency_sigma_polarization(device_list=device_list,
                                                                 kin=kin_list[idx])

        efficiency_holder[idx, :] = efficiency_tmp[:]
        kout_holder[idx, :, :] = kout_tmp[:, :]
        total_efficiency_holder[idx] = total_efficiency

    return total_efficiency_holder, efficiency_holder, kout_holder
