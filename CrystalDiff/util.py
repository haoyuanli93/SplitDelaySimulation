import datetime
import time

import h5py as h5
import numpy as np

"""
This module is the lowest-level module. It does not depend on another modules.
"""
pi = np.pi
two_pi = 2. * np.pi

hbar = 0.0006582119514  # This is the reduced planck constant in keV/fs

c = 299792458. * 1e-9  # The speed of light in um / fs


# --------------------------------------------------------------
#               Simple functions
# --------------------------------------------------------------
def exp_stable(x):
    """
    This function calculate the exponential of a complex variable in a stable way.
    :param x:
    :return:
    """
    re = x.real
    im = x.imag

    im = np.mod(im, 2 * pi)
    phase = np.cos(im) + 1.j * np.sin(im)

    # Build a mask to find too small values
    # Assume that when re is less than -100, define the value to be 0
    magnitude = np.zeros_like(re, dtype=np.complex128)
    magnitude[re >= -100] = np.exp(re[re >= -100]) + 0.j
    return np.multiply(magnitude, phase)


def l2_norm(x):
    return np.sqrt(np.sum(np.square(x)))


def l2_square(x):
    return np.sum(np.square(x))


def l2_norm_batch(x):
    return np.sqrt(np.sum(np.square(x), axis=-1))


def l2_square_batch(x):
    return np.sum(np.square(x), axis=-1)


# --------------------------------------------------------------
#               Unit conversion
# --------------------------------------------------------------
def kev_to_petahertz_frequency(energy):
    return energy / hbar * 2 * pi


def kev_to_petahertz_angular_frequency(energy):
    return energy / hbar


def kev_to_wave_number(energy):
    return energy / hbar / c


def petahertz_frequency_to_kev(frequency):
    return hbar * 2 * pi * frequency


def petahertz_angular_frequency_to_kev(angular_frequency):
    return hbar * angular_frequency


def petahertz_angular_frequency_to_wave_number(angular_frequency):
    return angular_frequency / c


def wave_number_to_kev(wavevec):
    return wavevec * hbar * c


def sigma_to_fwhm(sigma):
    return 2. * np.sqrt(2 * np.log(2)) * sigma


def fwhm_to_sigma(fwhm):
    return fwhm / (2. * np.sqrt(2 * np.log(2)))


def intensity_fwhm_to_field_sigma(fwhm):
    return fwhm / (2. * np.sqrt(2 * np.log(2))) * np.sqrt(2)


def field_sigma_to_intensity_fwhm(sigma):
    return sigma * (2. * np.sqrt(2 * np.log(2))) / np.sqrt(2)


def bandwidth_sigma_kev_to_duration_sigma_fs(bandwidth_kev):
    return hbar / 2. / bandwidth_kev


def get_intensity_fwhm_duration_from_intensity_bandwidth(bandwidth_kev):
    # Convert intensity bandwidth to field bandwidth
    field_bandwidth = bandwidth_kev * np.sqrt(2)
    field_bandwidth_sigma = fwhm_to_sigma(field_bandwidth)

    # Calcualte the pulse duration
    field_duration_sigma = bandwidth_sigma_kev_to_duration_sigma_fs(field_bandwidth_sigma)
    field_duration_fwhm = sigma_to_fwhm(field_duration_sigma)

    # Convert the field duration fwhm to intensity duration fwhm
    intensity_duration_fwhm = field_duration_fwhm / np.sqrt(2)
    return intensity_duration_fwhm


# --------------------------------------------------------------
#          Get output wave vectors
# --------------------------------------------------------------
def get_bragg_kout(kin, h, normal, compare_length=False):
    """
    This function produce the output wave vector from a Bragg reflection.

    :param kin: (3,) numpy array. The incident wave vector
    :param h: The reciprocal lattice of the crystal
    :param normal: The normal direction of the reflection surface.
                    For a bragg reflection, n is pointing to the inside of the crystal.
    :param compare_length: Whether compare the length of the incident wave vector and the output wave vector

    :return: kout: (3,) numpy array. The diffraction wave vector.
            ratio: When compare_length=True, the second output is the ratio between the incident wave number
                                        and the output wave number.
    """

    # kout holder
    kout = kin + h

    # Incident wave number
    klen = np.sqrt(np.dot(kin, kin))

    # Get gamma and alpha
    gammah = np.dot(kin + h, normal) / klen
    alpha = (2 * np.dot(kin, h) + np.dot(h, h)) / np.square(klen)

    if np.abs(-gammah - np.sqrt(gammah ** 2 - alpha)) > np.abs(-gammah + np.sqrt(gammah ** 2 - alpha)):
        momentum = klen * (-gammah + np.sqrt(gammah ** 2 - alpha))
    else:
        momentum = klen * (-gammah - np.sqrt(gammah ** 2 - alpha))

    # Add momentum transfer
    kout += normal * momentum

    if compare_length:
        return kout, klen / l2_norm(kout)
    else:
        return kout


# --------------------------------------------------------------
#          Geometry functions
# --------------------------------------------------------------
def get_intersection(s, k, n, x0):
    """
    Assume that a line starts from point s along the direction k. It will intersect with
    the plane that passes through point x0 and has normal direction n. The function find the
    resulted intersection point.

    This function assumes that the arguments are arrays of points.

    :param s: array of shape [3], starting points for each array
    :param k: array of shape [3], the direction for each array
    :param n: array of shape [3], the normal direction of the surface
    :param x0: array of shape [3], one point on this surface
    :return:
    """
    # The intersection points for each array
    x = np.copy(s)

    # Do the math
    tmp = np.divide(np.dot(x0 - s, n), np.dot(k, n))
    x += tmp * k
    return x


# --------------------------------------------------------------
#          Geometric operation
# --------------------------------------------------------------
def get_total_path_length(intersect_list):
    """
    Get the path length of a series of points

    :param intersect_list:
    :return:
    """
    number = len(intersect_list)
    total_path = 0.
    for x in range(number - 1):
        total_path += l2_norm(intersect_list[x + 1] -
                              intersect_list[x])

    return total_path


# ---------------------------------------------------------------------------
#                     Grating
# ---------------------------------------------------------------------------
def get_grating_output_momentum(grating_wavenum, k_vec):
    """
    Calculate output momentum of the grating with the specified wave number and
    the corresponding incident k_vec

    :param grating_wavenum:
    :param k_vec:
    :return:
    """
    wavenum_reshape = np.reshape(grating_wavenum, (1, 3))
    return k_vec + wavenum_reshape


def get_grating_wavenumber_1d(direction, period, order):
    """

    :param direction:
    :param period:
    :param order:
    :return:
    """
    return order * direction * 2. * np.pi / period


def get_grating_period(dtheta, klen_in):
    """
    Derive the grating period based on the deviation angle and the incident wave number.
    Here, one assume that the incident wave vector is perpendicular to the the grating surface.

    :param dtheta:
    :param klen_in:
    :return:
    """
    period = 2 * np.pi / klen_in / np.tan(dtheta)
    return period


# ---------------------------------------------------------------------------
#                     IO
# ---------------------------------------------------------------------------
def save_branch_result_to_h5file(file_name, io_type, branch_name,
                                 result_3d_dict, result_2d_dict, check_dict):
    with h5.File(file_name, io_type) as h5file:
        group = h5file.create_group(branch_name)
        # Save the meta data
        group_check = group.create_group('check')
        for entry in list(check_dict.keys()):
            group_check.create_dataset(entry, data=check_dict[entry])

        group_2d = group.create_group('result_2d')
        for entry in list(result_2d_dict.keys()):
            group_2d.create_dataset(entry, data=result_2d_dict[entry])

        group_3d = group.create_group('result_3d')
        for entry in list(result_3d_dict.keys()):
            group_3d.create_dataset(entry, data=result_3d_dict[entry])


def time_stamp():
    """
    Get a time stamp
    :return: A time stamp of the form '%Y_%m_%d_%H_%M_%S'
    """
    stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')
    return stamp


# ---------------------------------------------------------------------------
#                  Get k mesh
# ---------------------------------------------------------------------------

def get_k_mesh_3d(number_x, number_y, number_z, delta_e_x, delta_e_y, delta_e_z):
    # Get the corresponding energy mesh
    energy_grid_x = np.linspace(start=- delta_e_x,
                                stop=+ delta_e_x,
                                num=number_x)
    energy_grid_y = np.linspace(start=- delta_e_y,
                                stop=+ delta_e_y,
                                num=number_y)
    energy_grid_z = np.linspace(start=- delta_e_z,
                                stop=+ delta_e_z,
                                num=number_z)

    # Get the k grid
    kx_grid = np.ascontiguousarray(kev_to_wave_number(energy=energy_grid_x))
    ky_grid = np.ascontiguousarray(kev_to_wave_number(energy=energy_grid_y))
    kz_grid = np.ascontiguousarray(kev_to_wave_number(energy=energy_grid_z))

    # Get the spatial mesh along x axis
    dkx = kev_to_wave_number(energy=energy_grid_x[1] - energy_grid_x[0])
    x_range = np.pi * 2 / dkx

    x_idx = np.linspace(start=-x_range / 2., stop=x_range / 2., num=number_x)
    x_idx_tick = ["{:.2f}".format(x) for x in x_idx]

    # Get the spatial mesh along y axis
    dky = kev_to_wave_number(energy=energy_grid_y[1] - energy_grid_y[0])
    y_range = np.pi * 2 / dky

    y_idx = np.linspace(start=-y_range / 2., stop=y_range / 2., num=number_y)
    y_idx_tick = ["{:.2f}".format(x) for x in y_idx]

    # Get the spatial mesh along z axis
    dkz = kev_to_wave_number(energy=energy_grid_z[1] - energy_grid_z[0])
    z_range = np.pi * 2 / dkz

    z_idx = np.linspace(start=-z_range / 2., stop=z_range / 2., num=number_z)
    z_idx_tick = ["{:.2f}".format(x) for x in z_idx]

    # Assemble the indexes and labels
    axis_info = {"x_range": x_range,
                 "x_idx": x_idx,
                 "x_idx_tick": x_idx_tick,
                 "dkx": dkx,
                 "energy_grid_x": energy_grid_x,

                 "y_range": y_range,
                 "y_idx": y_idx,
                 "y_idx_tick": y_idx_tick,
                 "dky": dky,
                 "energy_grid_y": energy_grid_y,

                 "z_range": z_range,
                 "z_idx": z_idx,
                 "z_idx_tick": z_idx_tick,
                 "dkz": dkz,
                 "energy_grid_z": energy_grid_z,
                 "z_time_idx": np.divide(z_idx, c),
                 "z_time_tick": ["{:.2f}".format(x) for x in np.divide(z_idx, c)],

                 "de_x_in_meV": np.linspace(start=- delta_e_x * 1e6,
                                            stop=+ delta_e_x * 1e6,
                                            num=number_x)}
    return kx_grid, ky_grid, kz_grid, axis_info


# ---------------------------------------------------
#              For DuMond Diagram
# ---------------------------------------------------
def get_klen_and_angular_mesh(k_num, theta_num, phi_num, energy_range, theta_range, phi_range):
    # Get the corresponding energy mesh
    energy_grid = np.linspace(start=energy_range[0], stop=energy_range[1], num=k_num)
    # Get the k grid
    klen_grid = np.ascontiguousarray(kev_to_wave_number(energy=energy_grid))

    # Get theta grid
    theta_grid = np.linspace(start=theta_range[0], stop=theta_range[1], num=theta_num)

    # Get phi grid
    phi_grid = np.linspace(start=phi_range[0], stop=phi_range[1], num=phi_num)

    info_dict = {"energy_grid": energy_grid,
                 "klen_grid": klen_grid,
                 "theta_grid": theta_grid,
                 "phi_grid": phi_grid}
    return info_dict


###############################################################################################
###############################################################################################
#
#    The following code handle bragg reflectivity with cpu in details
#
###############################################################################################
###############################################################################################
def get_bragg_reflection_array(kin_grid, d, h, n,
                               chi0, chih_sigma, chihbar_sigma,
                               chih_pi, chihbar_pi):
    """
    This function aims to get the info quickly with cpu.

    :param kin_grid:
    :param d:
    :param h:
    :param n:
    :param chi0:
    :param chih_sigma:
    :param chihbar_sigma:
    :param chih_pi:
    :param chihbar_pi:
    :return:
    """
    # Create holder to save the reflectivity and output momentum
    kout_grid = np.zeros_like(kin_grid, dtype=np.float64)

    # ------------------------------------------------------------
    #          Step 1: Get output momentum wave vector
    # ------------------------------------------------------------
    # Get some info to facilitate the calculation
    klen_grid = l2_norm_batch(kin_grid)
    dot_hn = np.dot(h, n)
    h_square = l2_square(h)

    # Get gamma and alpha and b
    dot_kn = np.dot(kin_grid, n)
    dot_kh = np.dot(kin_grid, h)

    gamma_0 = np.divide(dot_kn, klen_grid)
    gamma_h = np.divide(dot_kn + dot_hn, klen_grid)

    b = np.divide(gamma_0, gamma_h)
    b_cplx = b.astype(np.complex128)
    alpha = np.divide(2 * dot_kh + h_square, np.square(klen_grid))

    # Get momentum tranfer
    sqrt_gamma_alpha = np.sqrt(gamma_h ** 2 - alpha)

    mask = np.zeros_like(sqrt_gamma_alpha, dtype=np.bool)
    mask[np.abs(-gamma_h - sqrt_gamma_alpha) > np.abs(-gamma_h + sqrt_gamma_alpha)] = True

    m_trans = np.multiply(klen_grid, -gamma_h - sqrt_gamma_alpha)
    m_trans[mask] = np.multiply(klen_grid[mask], -gamma_h[mask] + sqrt_gamma_alpha[mask])

    # Update the kout_grid
    kout_grid[:, 0] = kin_grid[:, 0] + h[0] + m_trans * n[0]
    kout_grid[:, 1] = kin_grid[:, 1] + h[1] + m_trans * n[1]
    kout_grid[:, 2] = kin_grid[:, 2] + h[2] + m_trans * n[2]

    # ------------------------------------------------------------
    # Step 2: Get the reflectivity for input sigma polarization
    # ------------------------------------------------------------
    # Get alpha tidle
    alpha_tidle = (alpha * b + chi0 * (1. - b)) / 2.

    # Get sqrt(alpha**2 + beta**2) value
    sqrt_a2_b2 = np.sqrt(alpha_tidle ** 2 + np.multiply(b_cplx, chih_sigma * chihbar_sigma))

    # Change the imaginary part sign
    mask = np.zeros_like(sqrt_a2_b2, dtype=np.bool)
    mask[sqrt_a2_b2.imag < 0] = True
    sqrt_a2_b2[mask] = - sqrt_a2_b2[mask]

    # Calculate the phase term
    re = klen_grid * d / gamma_0 * sqrt_a2_b2.real
    im = klen_grid * d / gamma_0 * sqrt_a2_b2.imag

    magnitude = np.exp(-im).astype(np.complex128)
    phase = np.cos(re) + np.sin(re) * 1.j

    # Calculate some intermediate part
    numerator = 1. - magnitude * phase
    denominator = alpha_tidle * numerator + sqrt_a2_b2 * (2. - numerator)

    # Take care of the exponential
    mask = np.zeros_like(im, dtype=np.bool)
    mask[im <= 400] = True

    reflect_s = chih_sigma * b_cplx / denominator
    reflect_s[mask] = chih_sigma * b_cplx[mask] * numerator[mask] / denominator[mask]

    # ------------------------------------------------------------
    # Step 2: Get the reflectivity for pi polarization
    # ------------------------------------------------------------

    # Get the polarization factor with the asymmetric factor b.
    p_value = np.sum(np.multiply(kout_grid, kin_grid), axis=-1) / np.square(klen_grid)
    bp = b_cplx * p_value

    # Get sqrt(alpha**2 + beta**2) value
    sqrt_a2_b2 = np.sqrt(alpha_tidle ** 2 + bp * p_value * chih_pi * chihbar_pi)

    # Change the imaginary part sign
    mask = np.zeros_like(sqrt_a2_b2, dtype=np.bool)
    mask[sqrt_a2_b2.imag < 0] = True
    sqrt_a2_b2[mask] = - sqrt_a2_b2[mask]

    # Calculate the phase term
    re = klen_grid * d / gamma_0 * sqrt_a2_b2.real
    im = klen_grid * d / gamma_0 * sqrt_a2_b2.imag

    magnitude = np.exp(-im).astype(np.complex128)
    phase = np.cos(re) + np.sin(re) * 1.j

    # Calculate some intermediate part
    numerator = 1. - magnitude * phase
    denominator = alpha_tidle * numerator + sqrt_a2_b2 * (2. - numerator)

    # Take care of the exponential
    mask = np.zeros_like(im, dtype=np.bool)
    mask[im <= 400] = True

    reflect_p = bp * chih_pi / denominator
    reflect_p[mask] = bp[mask] * chih_pi * numerator[mask] / denominator[mask]

    return reflect_s, reflect_p, b, kout_grid


def get_rocking_curve(kin_list, crystal_list):
    """
    Get the reflectivity for each kin.

    :param kin_list:
    :param crystal_list:
    :return:
    """
    k_num = kin_list.shape[0]
    x_num = len(crystal_list)

    # Define some holder to save data
    kout_list = []
    reflect_p_list = []
    reflect_s_list = []
    reflect_p_total = np.ones(k_num, dtype=np.complex128)
    reflect_s_total = np.ones(k_num, dtype=np.complex128)
    b_total = np.ones(k_num, dtype=np.float64)

    kout_tmp = np.copy(kin_list)
    for x in range(x_num):
        # Get info
        (reflect_s_tmp,
         reflect_p_tmp,
         b_tmp,
         kout_tmp) = get_bragg_reflection_array(kin_grid=kout_tmp,
                                                d=crystal_list[x].thickness,
                                                h=crystal_list[x].h,
                                                n=crystal_list[x].normal,
                                                chi0=crystal_list[x].chi0,
                                                chih_sigma=crystal_list[x].chih_sigma,
                                                chihbar_sigma=crystal_list[x].chihbar_sigma,
                                                chih_pi=crystal_list[x].chih_pi,
                                                chihbar_pi=crystal_list[x].chihbar_pi)
        b_tmp = np.abs(b_tmp)

        # Save info to holders
        kout_list.append(np.copy(kout_tmp))
        reflect_p_list.append(np.square(np.abs(reflect_p_tmp)) / b_tmp)
        reflect_s_list.append(np.square(np.abs(reflect_s_tmp)) / b_tmp)

        # Update the total reflectivity
        reflect_s_total = np.multiply(reflect_s_total, reflect_s_tmp)
        reflect_p_total = np.multiply(reflect_p_total, reflect_p_tmp)
        b_total = np.multiply(b_total, b_tmp)

    reflect_s_total = np.square(np.abs(reflect_s_total)) / b_total
    reflect_p_total = np.square(np.abs(reflect_p_total)) / b_total

    return reflect_s_total, reflect_p_total, reflect_s_list, reflect_p_list, kout_list


###############################################################################################
###############################################################################################
#
#    The following code handle grating transmission with cpu
#
###############################################################################################
###############################################################################################
def get_square_grating_transmission(kin, height_vec, ab_ratio, base, refractive_index, order, grating_k):
    # The argument for exp(ik(n-1)h)
    nhk = np.dot(height_vec, kin).astype(np.complex128) * (refractive_index - complex(1.))

    # The argument for exp(ik(n-1)t) for the phase different and absorption from
    # the base of the grating
    thick_k_n = np.dot(base, kin).astype(np.complex128) * (refractive_index - complex(1.))

    first_factor = complex(1.
                           - np.cos(two_pi * order * ab_ratio),
                           - np.sin(two_pi * order * ab_ratio))
    second_factor = complex(1.) - complex(np.exp(-nhk.imag) * np.cos(nhk.real),
                                          np.exp(-nhk.imag) * np.sin(nhk.real))

    # Factor from the base
    factor_base = complex(np.cos(thick_k_n.real) * np.exp(-thick_k_n.imag),
                          np.sin(thick_k_n.real) * np.exp(-thick_k_n.imag))

    factor = 1.j / complex(2. * np.pi * order) * first_factor * second_factor * factor_base

    # Step 3: Update the momentum and the length of the momentum
    kout = kin + order * grating_k
    klen = l2_norm(kout)

    return factor, kout, klen


def get_square_grating_0th_transmission(kin, height_vec, refractive_index, ab_ratio, base):
    # The argument for exp(ik(n-1)h)
    nhk = np.dot(height_vec, kin).astype(np.complex128) * (refractive_index - complex(1.))

    # The argument for exp(ik(n-1)t) for the phase different and absorption from
    # the base of the grating
    thick_k_n = np.dot(base, kin).astype(np.complex128) * (refractive_index - complex(1.))

    # Factor from the base
    factor_base = complex(np.cos(thick_k_n.real) * np.exp(-thick_k_n.imag),
                          np.sin(thick_k_n.real) * np.exp(-thick_k_n.imag))

    pre_factor = complex(1.) - complex(np.exp(-nhk.imag) * np.cos(nhk.real),
                                       np.exp(-nhk.imag) * np.sin(nhk.real))

    factor = (complex(1.) - complex(ab_ratio) * pre_factor) * factor_base

    return factor


########################################################################################################################
########################################################################################################################
#
#          Functions created after 2020/2/14
#
########################################################################################################################
########################################################################################################################
def get_forward_bragg_reflection_array(kin_grid, d, h, n,
                                       chi0, chih_sigma, chihbar_sigma,
                                       chih_pi, chihbar_pi):
    """
    This function aims to get the info quickly with cpu.

    :param kin_grid:
    :param d:
    :param h:
    :param n:
    :param chi0:
    :param chih_sigma:
    :param chihbar_sigma:
    :param chih_pi:
    :param chihbar_pi:
    :return:
    """
    # Create holder to save the reflectivity and output momentum
    kout_grid = np.zeros_like(kin_grid, dtype=np.float64)

    # ------------------------------------------------------------
    #          Step 1: Get output momentum wave vector
    # ------------------------------------------------------------
    # Get some info to facilitate the calculation
    klen_grid = l2_norm_batch(kin_grid)
    dot_hn = np.dot(h, n)
    h_square = l2_square(h)

    # Get gamma and alpha and b
    dot_kn = np.dot(kin_grid, n)
    dot_kh = np.dot(kin_grid, h)

    gamma_0 = np.divide(dot_kn, klen_grid)
    gamma_h = np.divide(dot_kn + dot_hn, klen_grid)

    b = np.divide(gamma_0, gamma_h)
    b_cplx = b.astype(np.complex128)
    alpha = np.divide(2 * dot_kh + h_square, np.square(klen_grid))

    # Get momentum tranfer
    sqrt_gamma_alpha = np.sqrt(gamma_h ** 2 - alpha)
    m_trans = np.multiply(klen_grid, -gamma_h - sqrt_gamma_alpha)

    # Update the kout_grid
    kout_grid[:, 0] = kin_grid[:, 0] + h[0] + m_trans * n[0]
    kout_grid[:, 1] = kin_grid[:, 1] + h[1] + m_trans * n[1]
    kout_grid[:, 2] = kin_grid[:, 2] + h[2] + m_trans * n[2]

    # Get alpha tidle
    alpha_tidle = (alpha * b + chi0 * (1. - b)) / 2.

    # ------------------------------------------------------------
    # Step 2: Get the reflectivity for input sigma polarization
    # ------------------------------------------------------------
    # Get sqrt(alpha**2 + beta**2) value
    sqrt_a2_b2 = np.sqrt(alpha_tidle ** 2 + np.multiply(b_cplx, chih_sigma * chihbar_sigma))

    # Change the imaginary part sign
    mask = np.zeros_like(sqrt_a2_b2, dtype=np.bool)
    mask[sqrt_a2_b2.imag < 0] = True
    sqrt_a2_b2[mask] = - sqrt_a2_b2[mask]

    # Calculate the phase term
    normalized_l = np.divide(klen_grid * d / 2., gamma_0)
    tmp = normalized_l * (chi0 - alpha_tidle)
    re = tmp.real
    im = tmp.imag

    # Change the imaginary part sign
    mask = np.zeros_like(im, dtype=np.bool)
    mask[im >= 400.] = True

    # Calculate the values
    magnitude = np.exp(-im).astype(np.complex128)
    magnitude[mask] = complex(0., 0.)

    phase = np.cos(re) + np.sin(re) * 1.j

    # Calculate some intermediate part
    numerator = magnitude * phase
    denominator = (np.cos(normalized_l * sqrt_a2_b2) -
                   1.j * alpha_tidle * np.sin(normalized_l * sqrt_a2_b2) / sqrt_a2_b2)

    reflect_s = numerator / denominator

    # ------------------------------------------------------------
    # Step 2: Get the reflectivity for pi polarization
    # ------------------------------------------------------------

    # Get the polarization factor with the asymmetric factor b.
    p_value = np.sum(np.multiply(kout_grid, kin_grid), axis=-1) / np.square(klen_grid)
    bp = b_cplx * p_value

    # Get sqrt(alpha**2 + beta**2) value
    sqrt_a2_b2 = np.sqrt(alpha_tidle ** 2 + bp * p_value * chih_pi * chihbar_pi)

    # Change the imaginary part sign
    mask = np.zeros_like(sqrt_a2_b2, dtype=np.bool)
    mask[sqrt_a2_b2.imag < 0] = True
    sqrt_a2_b2[mask] = - sqrt_a2_b2[mask]

    # Calculate the phase term
    normalized_l = np.divide(klen_grid * d / 2., gamma_0)
    tmp = normalized_l * (chi0 - alpha_tidle)
    re = tmp.real
    im = tmp.imag

    # Change the imaginary part sign
    mask = np.zeros_like(im, dtype=np.bool)
    mask[im >= 400.] = True

    # Calculate the values
    magnitude = np.exp(-im).astype(np.complex128)
    magnitude[mask] = complex(0., 0.)

    phase = np.cos(re) + np.sin(re) * 1.j

    # Calculate some intermediate part
    numerator = magnitude * phase
    denominator = (np.cos(normalized_l * sqrt_a2_b2) -
                   1.j * alpha_tidle * np.sin(normalized_l * sqrt_a2_b2) / sqrt_a2_b2)

    reflect_p = numerator / denominator

    return reflect_s, reflect_p, b, kin_grid


########################################################################################################################
########################################################################################################################
#
#             New functions created after 2020/2/14
#
########################################################################################################################
########################################################################################################################
def get_diffraction_information(kin_grid,
                                h,
                                n):
    """
    This function is used to provide some intermediate variables that can be used to
    calculate the reflectivity.

    :param kin_grid:
    :param h:
    :param n:
    :return:
    """
    # Create holder to save the reflectivity and output momentum
    kout_grid = np.zeros_like(kin_grid, dtype=np.float64)

    # ------------------------------------------------------------
    #          Step 1: Get output momentum wave vector
    # ------------------------------------------------------------
    # Get some info to facilitate the calculation
    klen_grid = l2_norm_batch(kin_grid)
    dot_hn = np.dot(h, n)
    h_square = l2_square(h)

    # Get gamma and alpha and b
    dot_kn = np.dot(kin_grid, n)
    dot_kh = np.dot(kin_grid, h)

    gamma_0 = np.divide(dot_kn, klen_grid)
    gamma_h = np.divide(dot_kn + dot_hn, klen_grid)

    b = np.divide(gamma_0, gamma_h)
    b_cplx = b.astype(np.complex128)
    alpha = np.divide(2 * dot_kh + h_square, np.square(klen_grid))

    # Get momentum tranfer
    sqrt_gamma_alpha = np.sqrt(gamma_h ** 2 - alpha)
    m_trans = np.multiply(klen_grid, -gamma_h - sqrt_gamma_alpha)

    # Update the kout_grid
    kout_grid[:, 0] = kin_grid[:, 0] + h[0] + m_trans * n[0]
    kout_grid[:, 1] = kin_grid[:, 1] + h[1] + m_trans * n[1]
    kout_grid[:, 2] = kin_grid[:, 2] + h[2] + m_trans * n[2]

    return kout_grid, klen_grid, alpha, gamma_0, gamma_h, b_cplx, b


def get_laue_reflection_array(kin_grid, d, h, n,
                              chi0,
                              chih_sigma,
                              chihbar_sigma,
                              chih_pi,
                              chihbar_pi):
    """

    :param kin_grid:
    :param d:
    :param h:
    :param n:
    :param chi0:
    :param chih_sigma:
    :param chihbar_sigma:
    :param chih_pi:
    :param chihbar_pi:
    :return:
    """
    # Get intermediate variables to calculate reflectivity
    (kout_grid, klen_grid,
     alpha, gamma_0, gamma_h, b_cplx, b) = get_diffraction_information(kin_grid, h, n)

    # Get alpha tidle
    alpha_tidle = (alpha * b + chi0 * (1. - b)) / 2.

    # ------------------------------------------------------------
    # Step 2: Get the reflectivity for input sigma polarization
    # ------------------------------------------------------------
    # Get sqrt(alpha**2 + beta**2) value
    sqrt_a2_b2 = np.sqrt(alpha_tidle ** 2 + np.multiply(b_cplx, chih_sigma * chihbar_sigma))

    # Change the imaginary part sign
    mask = np.zeros_like(sqrt_a2_b2, dtype=np.bool)
    mask[sqrt_a2_b2.imag < 0] = True
    sqrt_a2_b2[mask] = - sqrt_a2_b2[mask]

    # Calculate the phase term
    normalized_l = np.divide(klen_grid * d / 2., gamma_0)
    tmp = normalized_l * (chi0 - alpha_tidle)
    re = tmp.real
    im = tmp.imag

    # Change the imaginary part sign
    mask = np.zeros_like(im, dtype=np.bool)
    mask[im >= 400.] = True

    # Calculate the values
    magnitude = np.exp(-im).astype(np.complex128)
    magnitude[mask] = complex(0., 0.)

    phase = np.cos(re) + np.sin(re) * 1.j

    # Calculate some intermediate part
    factor1 = 1.j * b * chih_sigma * magnitude * phase
    factor2 = np.sin(normalized_l * sqrt_a2_b2) / sqrt_a2_b2
    reflect_s = factor1 * factor2

    # ------------------------------------------------------------
    # Step 2: Get the reflectivity for pi polarization
    # ------------------------------------------------------------

    # Get the polarization factor with the asymmetric factor b.
    polar = np.sum(np.multiply(kout_grid, kin_grid), axis=-1) / np.square(klen_grid)
    bp = b_cplx * polar

    # Get sqrt(alpha**2 + beta**2) value
    sqrt_a2_b2 = np.sqrt(alpha_tidle ** 2 + bp * polar * chih_pi * chihbar_pi)

    # Change the imaginary part sign
    mask = np.zeros_like(sqrt_a2_b2, dtype=np.bool)
    mask[sqrt_a2_b2.imag < 0] = True
    sqrt_a2_b2[mask] = - sqrt_a2_b2[mask]

    # Calculate the phase term
    normalized_l = np.divide(klen_grid * d / 2., gamma_0)
    tmp = normalized_l * (chi0 - alpha_tidle)
    re = tmp.real
    im = tmp.imag

    # Change the imaginary part sign
    mask = np.zeros_like(im, dtype=np.bool)
    mask[im >= 400.] = True

    # Calculate the values
    magnitude = np.exp(-im).astype(np.complex128)
    magnitude[mask] = complex(0., 0.)

    phase = np.cos(re) + np.sin(re) * 1.j

    # Calculate some intermediate part
    factor1 = 1.j * polar * b * chih_pi * magnitude * phase
    factor2 = np.sin(normalized_l * sqrt_a2_b2) / sqrt_a2_b2

    reflect_p = factor1 * factor2

    return reflect_s, reflect_p, b, kout_grid


def get_forward_laue_reflection_array(kin_grid, d, h, n,
                                      chi0,
                                      chih_sigma,
                                      chihbar_sigma,
                                      chih_pi,
                                      chihbar_pi):
    """

    :param kin_grid:
    :param d:
    :param h:
    :param n:
    :param chi0:
    :param chih_sigma:
    :param chihbar_sigma:
    :param chih_pi:
    :param chihbar_pi:
    :return:
    """
    # Get intermediate variables to calculate reflectivity
    (kout_grid, klen_grid,
     alpha, gamma_0, gamma_h, b_cplx, b) = get_diffraction_information(kin_grid, h, n)

    # Get alpha tidle
    alpha_tidle = (alpha * b + chi0 * (1. - b)) / 2.

    # ------------------------------------------------------------
    # Step 2: Get the reflectivity for input sigma polarization
    # ------------------------------------------------------------
    # Get sqrt(alpha**2 + beta**2) value
    sqrt_a2_b2 = np.sqrt(alpha_tidle ** 2 + np.multiply(b_cplx, chih_sigma * chihbar_sigma))

    # Change the imaginary part sign
    mask = np.zeros_like(sqrt_a2_b2, dtype=np.bool)
    mask[sqrt_a2_b2.imag < 0] = True
    sqrt_a2_b2[mask] = - sqrt_a2_b2[mask]

    # Calculate the phase term
    normalized_l = np.divide(klen_grid * d / 2., gamma_0)
    tmp = normalized_l * (chi0 - alpha_tidle)
    re = tmp.real
    im = tmp.imag

    # Change the imaginary part sign
    mask = np.zeros_like(im, dtype=np.bool)
    mask[im >= 400.] = True

    # Calculate the values
    magnitude = np.exp(-im).astype(np.complex128)
    magnitude[mask] = complex(0., 0.)

    phase = np.cos(re) + np.sin(re) * 1.j

    # Calculate some intermediate part
    factor1 = chih_sigma * magnitude * phase
    factor2 = np.cos(normalized_l * sqrt_a2_b2) + 1.j * alpha_tidle * np.sin(normalized_l * sqrt_a2_b2) / sqrt_a2_b2

    reflect_s = factor1 * factor2

    # ------------------------------------------------------------
    # Step 2: Get the reflectivity for pi polarization
    # ------------------------------------------------------------

    # Get the polarization factor with the asymmetric factor b.
    polar = np.sum(np.multiply(kout_grid, kin_grid), axis=-1) / np.square(klen_grid)
    bp = b_cplx * polar

    # Get sqrt(alpha**2 + beta**2) value
    sqrt_a2_b2 = np.sqrt(alpha_tidle ** 2 + bp * polar * chih_pi * chihbar_pi)

    # Change the imaginary part sign
    mask = np.zeros_like(sqrt_a2_b2, dtype=np.bool)
    mask[sqrt_a2_b2.imag < 0] = True
    sqrt_a2_b2[mask] = - sqrt_a2_b2[mask]

    # Calculate the phase term
    normalized_l = np.divide(klen_grid * d / 2., gamma_0)
    tmp = normalized_l * (chi0 - alpha_tidle)
    re = tmp.real
    im = tmp.imag

    # Change the imaginary part sign
    mask = np.zeros_like(im, dtype=np.bool)
    mask[im >= 400.] = True

    # Calculate the values
    magnitude = np.exp(-im).astype(np.complex128)
    magnitude[mask] = complex(0., 0.)

    phase = np.cos(re) + np.sin(re) * 1.j

    # Calculate some intermediate part
    factor1 = chih_sigma * magnitude * phase
    factor2 = np.cos(normalized_l * sqrt_a2_b2) + 1.j * alpha_tidle * np.sin(normalized_l * sqrt_a2_b2) / sqrt_a2_b2

    reflect_p = factor1 * factor2

    return reflect_s, reflect_p, b, kin_grid


#######################################################################################################################
#                 Code from new package XRaySimulation
#######################################################################################################################
def get_kout(device, kin):
    """
    Get the output wave vector given the incident wave vector

    :param device:
    :param kin:
    :return:
    """
    # Get output wave vector
    if device.type == "Crystal: Bragg Reflection":
        kout = get_bragg_kout(kin=kin,
                              h=device.h,
                              normal=device.normal)
        return kout

    if device.type == "Transmissive Grating":
        kout = kin + device.momentum_transfer
        return kout


def get_intensity_efficiency_sigma_polarization(device, kin):
    """
    Get the output intensity efficiency for the given wave vector
    assuming a monochromatic plane incident wave.

    :param device:
    :param kin:
    :return:
    """
    # Get output wave vector
    if device.type == "Crystal: Bragg Reflection":
        tmp = np.zeros((1, 3))
        tmp[0, :] = kin

        (reflect_s,
         reflect_p,
         b,
         kout_grid) = get_bragg_reflection_array(kin_grid=tmp,
                                                 d=device.thickness,
                                                 h=device.h,
                                                 n=device.normal,
                                                 chi0=device.chi0,
                                                 chih_sigma=device.chih_sigma,
                                                 chihbar_sigma=device.chihbar_sigma,
                                                 chih_pi=device.chih_pi,
                                                 chihbar_pi=device.chihbar_pi)

        efficiency = np.square(np.abs(reflect_s)) / np.abs(b)
        return efficiency

    if device.type == "Transmissive Grating":

        # Determine the grating order
        if device.order == 0:
            efficiency = get_square_grating_0th_transmission(kin=kin,
                                                             height_vec=device.h,
                                                             refractive_index=device.n,
                                                             ab_ratio=device.ab_ratio,
                                                             base=device.thick_vec)
        else:
            efficiency, _, _ = get_square_grating_transmission(kin=kin,
                                                               height_vec=device.h,
                                                               ab_ratio=device.ab_ratio,
                                                               base=device.thick_vec,
                                                               refractive_index=device.n,
                                                               order=device.order,
                                                               grating_k=device.momentum_transfer)
        # Translate to the intensity efficiency
        return np.square(np.abs(efficiency))
