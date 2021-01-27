import numpy as np
from scipy.spatial.transform import Rotation
import copy

from CrystalDiff import lclsutil, crystal, lightpath


def get_crystals_in_delayed_branch(asymmetric_angles=np.array([0, -5, 5, 0, 0, 5, -5, 0]),
                                   lattice_distance=1.9201 * 1e-4,
                                   chi_dict=None):
    ###########################################################################
    #              Define  physical properties
    ###########################################################################
    # Some meta data for the delay line.
    h_length = 2. * np.pi / lattice_distance

    # Some crystal properties
    chi0 = complex(-0.97631E-05, 0.14871E-06)
    chih_sigma = complex(0.59310E-05, -0.14320E-06)
    chihbar_sigma = complex(0.59310E-05, -0.14320E-06)
    chih_pi = complex(0.46945E-05, -0.11201E-06)
    chihbar_pi = complex(0.46945E-05, -0.11201E-06)
    if chi_dict is not None:
        chi0 = chi_dict['chi0']
        chih_sigma = chi_dict['chih_sigma']
        chihbar_sigma = chi_dict['chihbar_sigma']
        chih_pi = chi_dict['chih_pi']
        chihbar_pi = chi_dict['chihbar_pi']

    ###########################################################################
    #              Define crystal arrays
    ###########################################################################
    thetas = np.array([np.pi / 2,
                       - np.pi / 2,
                       - np.pi / 2,
                       np.pi / 2,
                       np.pi / 2,
                       - np.pi / 2,
                       - np.pi / 2,
                       np.pi / 2])

    rhos = thetas + np.pi
    for x in range(8):
        rhos[x] += asymmetric_angles[x]

    crystal_list = lclsutil.get_crystal_list(num=8,
                                             hlen=np.array([h_length, ] * 8),
                                             theta=thetas,
                                             eta=np.zeros(8),
                                             rho=rhos,
                                             tau=np.zeros(8),
                                             chi0=chi0, chih_sigma=chih_sigma, chihbar_sigma=chihbar_sigma,
                                             chih_pi=chih_pi, chihbar_pi=chihbar_pi)
    # ----------------------------------------------------------------------
    #          Crystal 1
    # ----------------------------------------------------------------------
    boundary = np.array([[0, 10, 39.052],
                         [50, 10, 39.052],
                         [50, 0, 39.052],
                         [0, 0, 39.052]]) * 1000

    crystal_list[0].boundary = boundary[:, [2, 1, 0]]
    crystal_list[0].set_surface_point(crystal_list[0].boundary[0])
    # ----------------------------------------------------------------------
    #          Crystal 2
    # ----------------------------------------------------------------------
    boundary = np.array([[25.070, 26.440, 39.052],
                         [90.070, 20.753, 39.052],
                         [90.070, 36.440, 39.052],
                         [25.070, 36.440, 39.052]]) * 1000

    crystal_list[1].boundary = boundary[:, [2, 1, 0]]
    crystal_list[1].set_surface_point(crystal_list[1].boundary[0])
    # ----------------------------------------------------------------------
    #          Crystal 3
    # ----------------------------------------------------------------------
    boundary = np.copy(crystal_list[1].boundary)
    boundary[:, 2] *= -1

    crystal_list[2].boundary = np.copy(boundary)
    crystal_list[2].set_surface_point(crystal_list[2].boundary[0])
    # ----------------------------------------------------------------------
    #          Crystal 4
    # ----------------------------------------------------------------------
    boundary = np.copy(crystal_list[0].boundary)
    boundary[:, 2] *= -1

    crystal_list[3].boundary = np.copy(boundary)
    crystal_list[3].set_surface_point(crystal_list[3].boundary[0])
    #######################################################################################################
    #######################################################################################################
    # ----------------------------------------------------------------------
    #          Crystal 5
    # ----------------------------------------------------------------------
    crystal_list[4].boundary = np.copy(crystal_list[3].boundary)
    crystal_list[4].set_surface_point(crystal_list[4].boundary[0])
    # ----------------------------------------------------------------------
    #          Crystal 7
    # ----------------------------------------------------------------------
    boundary = np.array([[25.070, 26.440, 39.052],
                         [90.070, 20.753, 39.052],
                         [90.070, 30.753, 39.052],
                         [25.070, 36.440, 39.052]]) * 1000

    crystal_list[6].boundary = boundary[:, [2, 1, 0]]
    crystal_list[6].set_surface_point(crystal_list[6].boundary[0])
    # ----------------------------------------------------------------------
    #          Crystal 6
    # ----------------------------------------------------------------------
    boundary = np.copy(crystal_list[6].boundary)
    boundary[:, 2] *= -1

    crystal_list[5].boundary = np.copy(boundary)
    crystal_list[5].set_surface_point(crystal_list[5].boundary[0])
    # ----------------------------------------------------------------------
    #          Crystal 8
    # ----------------------------------------------------------------------
    crystal_list[7].boundary = np.copy(crystal_list[0].boundary)
    crystal_list[7].set_surface_point(crystal_list[7].boundary[0])

    return crystal_list


def get_crystal_in_delay_fixed_branch(lattice_distance=1.9201 * 1e-4, chi_dict=None, miscut=None):
    ###########################################################################
    #              Define  physical properties
    ###########################################################################
    # Some meta data for the delay line.
    h_length = 2. * np.pi / lattice_distance

    # Some crystal properties
    chi0 = complex(-0.97631E-05, 0.14871E-06)
    chih_sigma = complex(0.59310E-05, -0.14320E-06)
    chihbar_sigma = complex(0.59310E-05, -0.14320E-06)
    chih_pi = complex(0.46945E-05, -0.11201E-06)
    chihbar_pi = complex(0.46945E-05, -0.11201E-06)
    if chi_dict is not None:
        chi0 = chi_dict['chi0']
        chih_sigma = chi_dict['chih_sigma']
        chihbar_sigma = chi_dict['chihbar_sigma']
        chih_pi = chi_dict['chih_pi']
        chihbar_pi = chi_dict['chihbar_pi']

    ###########################################################################
    #              Define crystal arrays
    ###########################################################################
    thetas = np.array([-np.pi / 2,
                       np.pi / 2,
                       np.pi / 2,
                       -np.pi / 2])
    rhos = thetas + np.pi
    if miscut is not None:
        for x in range(4):
            rhos[x] += miscut[x]

    crystals_list = lclsutil.get_crystal_list(num=4,
                                              hlen=np.array([h_length, ] * 4),
                                              theta=thetas,
                                              eta=np.zeros(4),
                                              rho=rhos,
                                              tau=np.zeros(4),
                                              chi0=chi0, chih_sigma=chih_sigma, chihbar_sigma=chihbar_sigma,
                                              chih_pi=chih_pi, chihbar_pi=chihbar_pi)
    # ----------------------------------------------------------------------
    #          Crystal 1
    # ----------------------------------------------------------------------
    boundary = np.array([[-60, 12.57, 40],
                         [-20, 12.57, 40],
                         [-20, 22.57, 40],
                         [-22.68, 22.57, 40]])
    boundary -= np.array([-60, -22.57, 0])[np.newaxis, :]
    boundary *= 1000

    crystals_list[0].boundary = boundary[:, [2, 1, 0]]
    crystals_list[0].set_surface_point(crystals_list[0].boundary[0])
    # ----------------------------------------------------------------------
    #          Crystal 2
    # ----------------------------------------------------------------------
    boundary = np.array([[-60, -12.58, 40],
                         [60, -12.58, 40],
                         [60, -22.57, 40],
                         [-60, -22.57, 40]])
    boundary -= np.array([-60, -22.57, 0])[np.newaxis, :]
    boundary *= 1000

    crystals_list[1].boundary = boundary[:, [2, 1, 0]]
    crystals_list[1].set_surface_point(crystals_list[1].boundary[0])
    # ----------------------------------------------------------------------
    #          Crystal 3
    # ----------------------------------------------------------------------
    boundary = np.array([[-65, -12.90, 40],
                         [65, -12.90, 40],
                         [65, -22.90, 40],
                         [-65, -22.90, 40]])
    boundary -= np.array([-65, -22.90, 0])[np.newaxis, :]
    boundary *= 1000

    crystals_list[2].boundary = boundary[:, [2, 1, 0]]
    crystals_list[2].set_surface_point(crystals_list[2].boundary[0])
    # ----------------------------------------------------------------------
    #          Crystal 4
    # ----------------------------------------------------------------------
    boundary = np.array([[45, 12.90, 40],
                         [65, 12.90, 40],
                         [65, 22.90, 40],
                         [45, 22.90, 40]])
    boundary -= np.array([-65, -22.90, 0])[np.newaxis, :]
    boundary *= 1000

    crystals_list[3].boundary = boundary[:, [2, 1, 0]]
    crystals_list[3].set_surface_point(crystals_list[3].boundary[0])
    return crystals_list


def align_delayed_branch(delayed_branch_crystals, angle_offset, bragg, tweak):
    d1 = tweak[0]
    d2 = tweak[1]
    d3 = tweak[2]
    d4 = tweak[3]

    # ----------------------------------------------------------------------
    #          CC 1
    # ----------------------------------------------------------------------
    displacement = np.array([322.27, 638.69, 253.74]) * 1000
    rot_mat = Rotation.from_euler('x', -bragg + angle_offset + d1)

    delayed_branch_crystals[0].shift(displacement=displacement[[2, 1, 0]], include_boundary=True)
    delayed_branch_crystals[0].rotate_wrt_point(rot_mat.as_dcm(), ref_point=displacement[[2, 1, 0]],
                                                include_boundary=True)

    delayed_branch_crystals[1].shift(displacement=displacement[[2, 1, 0]], include_boundary=True)
    delayed_branch_crystals[1].rotate_wrt_point(rot_mat.as_dcm(), ref_point=displacement[[2, 1, 0]],
                                                include_boundary=True)

    # ----------------------------------------------------------------------
    #          CC 2
    # ----------------------------------------------------------------------
    displacement = np.array([551.90, 642.47, 252.67]) * 1000
    rot_mat = Rotation.from_euler('x', bragg + angle_offset + d2)

    delayed_branch_crystals[2].shift(displacement=displacement[[2, 1, 0]], include_boundary=True)
    delayed_branch_crystals[2].rotate_wrt_point(rot_mat.as_dcm(), ref_point=displacement[[2, 1, 0]],
                                                include_boundary=True)

    delayed_branch_crystals[3].shift(displacement=displacement[[2, 1, 0]], include_boundary=True)
    delayed_branch_crystals[3].rotate_wrt_point(rot_mat.as_dcm(), ref_point=displacement[[2, 1, 0]],
                                                include_boundary=True)

    # ----------------------------------------------------------------------
    #          CC 3
    # ----------------------------------------------------------------------
    displacement = np.array([671.47, 677.99, 252.66]) * 1000
    rot_mat = Rotation.from_euler('x', bragg + angle_offset + d3 + np.pi)

    delayed_branch_crystals[4].shift(displacement=displacement[[2, 1, 0]], include_boundary=True)
    delayed_branch_crystals[4].rotate_wrt_point(rot_mat.as_dcm(), ref_point=displacement[[2, 1, 0]],
                                                include_boundary=True)

    delayed_branch_crystals[5].shift(displacement=displacement[[2, 1, 0]], include_boundary=True)
    delayed_branch_crystals[5].rotate_wrt_point(rot_mat.as_dcm(), ref_point=displacement[[2, 1, 0]],
                                                include_boundary=True)
    # ----------------------------------------------------------------------
    #          Crystal 4
    # ----------------------------------------------------------------------
    displacement = np.array([886.70, 677.98, 252.65]) * 1000
    rot_mat = Rotation.from_euler('x', -bragg + angle_offset + d4 + np.pi)

    delayed_branch_crystals[6].shift(displacement=displacement[[2, 1, 0]], include_boundary=True)
    delayed_branch_crystals[6].rotate_wrt_point(rot_mat.as_dcm(), ref_point=displacement[[2, 1, 0]],
                                                include_boundary=True)

    delayed_branch_crystals[7].shift(displacement=displacement[[2, 1, 0]], include_boundary=True)
    delayed_branch_crystals[7].rotate_wrt_point(rot_mat.as_dcm(), ref_point=displacement[[2, 1, 0]],
                                                include_boundary=True)

    return delayed_branch_crystals


def align_delay_fixed_branch(delay_fixed_branch_crystals, angle_offset, bragg, tweak):
    # ----------------------------------------------------------------------
    #          CC 1
    # ----------------------------------------------------------------------
    displacement = np.array([128.10, 627.61, 252.70]) * 1000
    rot_mat = Rotation.from_euler('x', bragg + angle_offset)

    delay_fixed_branch_crystals[0].shift(displacement=displacement[[2, 1, 0]], include_boundary=True)
    delay_fixed_branch_crystals[0].rotate_wrt_point(rot_mat.as_dcm(), ref_point=displacement[[2, 1, 0]],
                                                    include_boundary=True)

    delay_fixed_branch_crystals[1].shift(displacement=displacement[[2, 1, 0]], include_boundary=True)
    delay_fixed_branch_crystals[1].rotate_wrt_point(rot_mat.as_dcm(), ref_point=displacement[[2, 1, 0]],
                                                    include_boundary=True)

    # ----------------------------------------------------------------------
    #          CC 2
    # ----------------------------------------------------------------------
    displacement = np.array([973.84, 588.51, 252.64]) * 1000
    rot_mat = Rotation.from_euler('x', -bragg + angle_offset + tweak)

    delay_fixed_branch_crystals[2].shift(displacement=displacement[[2, 1, 0]], include_boundary=True)
    delay_fixed_branch_crystals[2].rotate_wrt_point(rot_mat.as_dcm(), ref_point=displacement[[2, 1, 0]],
                                                    include_boundary=True)

    delay_fixed_branch_crystals[3].shift(displacement=displacement[[2, 1, 0]], include_boundary=True)
    delay_fixed_branch_crystals[3].rotate_wrt_point(rot_mat.as_dcm(), ref_point=displacement[[2, 1, 0]],
                                                    include_boundary=True)
    return delay_fixed_branch_crystals


def get_light_path(ref_crystal_up,
                   ref_crystal_low,
                   ref_grating, ref_pulse, shift_upper, shift_lower, pathlength=None):
    # ------------------------------------------------------
    #   Copy the reference config
    # ------------------------------------------------------
    local_grating = [crystal.RectangleGrating(), crystal.RectangleGrating()]
    for x in range(2):
        local_grating[x].set_a(ref_grating[x].a)
        local_grating[x].set_b(ref_grating[x].b)
        local_grating[x].set_surface_point(ref_grating[x].surface_point)

    local_crystal_upper = [copy.deepcopy(ref_crystal_up[x]) for x in range(8)]
    local_crystal_lower = [copy.deepcopy(ref_crystal_low[x]) for x in range(4)]

    # ------------------------------------------------------
    #   shift the crystal
    # ------------------------------------------------------
    for x in range(8):
        if shift_upper[x] != 0:
            local_crystal_upper[x].shift(displacement=np.array([0, shift_upper[x], 0]),
                                         include_boundary=True)

    for x in range(4):
        if shift_lower[x] != 0:
            local_crystal_lower[x].shift(displacement=np.array([0, shift_lower[x], 0]),
                                         include_boundary=True)

    # ------------------------------------------------------
    #   get the light path
    # ------------------------------------------------------
    if pathlength is None:
        pathlength=20e6
        
    local_crystal_list = [local_grating[0]]
    for x in range(8):
        local_crystal_list.append(local_crystal_upper[x])
    local_crystal_list.append(local_grating[1])

    local_light_path_upper = lightpath.get_light_trajectory_with_total_path(kin_vec=ref_pulse.k0,
                                                                            init_point=ref_pulse.x0,
                                                                            total_path=pathlength,
                                                                            crystal_list=local_crystal_list,
                                                                            g_orders=[1, -1])

    local_crystal_list = [local_grating[0]]
    for x in range(4):
        local_crystal_list.append(local_crystal_lower[x])
    local_crystal_list.append(local_grating[1])

    local_light_path_lower = lightpath.get_light_trajectory_with_total_path(kin_vec=ref_pulse.k0,
                                                                            init_point=ref_pulse.x0,
                                                                            total_path=pathlength,
                                                                            crystal_list=local_crystal_list,
                                                                            g_orders=[-1, 1])

    return local_light_path_upper, local_light_path_lower, local_crystal_upper, local_crystal_lower
