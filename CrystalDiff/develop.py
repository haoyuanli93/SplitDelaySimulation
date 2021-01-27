import cmath
import math

from numba import cuda


@cuda.jit('void(complex128[:], complex128[:], float64[:,:], complex128[:,:],'
          'float64[:], float64[:,:],'
          'float64, float64[:], float64[:],'
          'float64, float64,'
          'complex128, complex128, complex128, complex128, complex128,'
          'int64)')
def get_forward_bragg_reflection(reflectivity_sigma,
                                 reflectivity_pi,
                                 kout_grid,
                                 efield_grid,
                                 klen_grid,
                                 kin_grid,
                                 d, h, n,
                                 dot_hn,
                                 h_square,
                                 chi0,
                                 chih_sigma,
                                 chihbar_sigma,
                                 chih_pi,
                                 chihbar_pi,
                                 num):
    """
    Given the crystal info, the input electric field, this function returns the
    reflectivity for the sigma polarization and pi polarization and the
    diffracted electric field.

    :param reflectivity_sigma:
    :param reflectivity_pi:
    :param kout_grid:
    :param efield_grid:
    :param klen_grid:
    :param kin_grid:
    :param d:
    :param h:
    :param n:
    :param dot_hn:
    :param h_square:
    :param chi0:
    :param chih_sigma:
    :param chihbar_sigma:
    :param chih_pi:
    :param chihbar_pi:
    :param num:
    :return:
    """
    idx = cuda.grid(1)
    if idx < num:

        #####################################################################################################
        # Step 1: Get parameters for reflectivity and decompose input field
        #####################################################################################################
        # ------------------------------------
        #     Get the diffracted wave number
        # ------------------------------------
        # Get k components
        kin_x = kin_grid[idx, 0]
        kin_y = kin_grid[idx, 1]
        kin_z = kin_grid[idx, 2]
        klen = klen_grid[idx]

        # Get gamma and alpha and b
        dot_kn = kin_x * n[0] + kin_y * n[1] + kin_z * n[2]
        dot_kh = kin_x * h[0] + kin_y * h[1] + kin_z * h[2]

        gamma_0 = dot_kn / klen
        gamma_h = (dot_kn + dot_hn) / klen
        b = gamma_0 / gamma_h
        b_complex = complex(b)
        alpha = (2 * dot_kh + h_square) / (klen ** 2)

        # Get momentum tranfer
        sqrt_gamma_alpha = math.sqrt(gamma_h ** 2 - alpha)
        m_trans = klen * (-gamma_h - sqrt_gamma_alpha)

        # Get output wave vector
        kout_x = kin_x + h[0] + m_trans * n[0]
        kout_y = kin_y + h[1] + m_trans * n[1]
        kout_z = kin_z + h[2] + m_trans * n[2]

        # Update the kout_grid
        kout_grid[idx, 0] = kin_x
        kout_grid[idx, 1] = kin_y
        kout_grid[idx, 2] = kin_z

        # ----------------------------------------
        #     Get polarization component
        # ----------------------------------------
        """
        sigma_in = kH cross k0     sigma_out = sigma_in  
        pi_in = k0 cross sigma0    pi_out = k_out cross sigma_out        
        """
        sigma_in_x = kout_y * kin_z - kout_z * kin_y
        sigma_in_y = kout_z * kin_x - kout_x * kin_z
        sigma_in_z = kout_x * kin_y - kout_y * kin_x
        tmp_len = math.sqrt(sigma_in_x ** 2 + sigma_in_y ** 2 + sigma_in_z ** 2)
        sigma_in_x /= tmp_len
        sigma_in_y /= tmp_len
        sigma_in_z /= tmp_len

        pi_in_x = kin_y * sigma_in_z - kin_z * sigma_in_y
        pi_in_y = kin_z * sigma_in_x - kin_x * sigma_in_z
        pi_in_z = kin_x * sigma_in_y - kin_y * sigma_in_x
        tmp_len = math.sqrt(pi_in_x ** 2 + pi_in_y ** 2 + pi_in_z ** 2)
        pi_in_x /= tmp_len
        pi_in_y /= tmp_len
        pi_in_z /= tmp_len

        pi_out_x = kout_y * sigma_in_z - kout_z * sigma_in_y
        pi_out_y = kout_z * sigma_in_x - kout_x * sigma_in_z
        pi_out_z = kout_x * sigma_in_y - kout_y * sigma_in_x
        tmp_len = math.sqrt(pi_out_x ** 2 + pi_out_y ** 2 + pi_out_z ** 2)
        pi_out_x /= tmp_len
        pi_out_y /= tmp_len
        pi_out_z /= tmp_len

        # Decompose the input electric field
        efield_sigma = (efield_grid[idx, 0] * complex(sigma_in_x) +
                        efield_grid[idx, 1] * complex(sigma_in_y) +
                        efield_grid[idx, 2] * complex(sigma_in_z))
        efield_pi = (efield_grid[idx, 0] * complex(pi_in_x) +
                     efield_grid[idx, 1] * complex(pi_in_y) +
                     efield_grid[idx, 2] * complex(pi_in_z))

        #####################################################################################################
        # Step 2: Get the reflectivity for input sigma polarization
        #####################################################################################################
        # ----------------------------------------
        #    Get reflectivity
        # ----------------------------------------
        # Get alpha tidle
        alpha_tidle = complex((alpha * b + chi0.real * (1. - b)) / 2., chi0.imag * (1. - b) / 2.)

        # Get sqrt(alpha**2 + beta**2) value
        sqrt_a2_b2 = cmath.sqrt(alpha_tidle ** 2 + b_complex * chih_sigma * chihbar_sigma)

        if sqrt_a2_b2.imag < 0:
            sqrt_a2_b2 = - sqrt_a2_b2

        # Calculate the normalized lenght l
        normalized_l = klen * d / 2 / gamma_0
        re = normalized_l * (chi0.real - alpha_tidle.real)
        im = normalized_l * (chi0.imag - alpha_tidle.imag)

        # Take care of the exponential
        magnitude = complex(math.exp(-im))
        if math.isnan(magnitude.real):
            magnitude = complex(0.)
        phase = complex(math.cos(re), math.sin(re))

        # Calculate some intermediate part
        numerator = magnitude * phase
        denominator = (cmath.cos(normalized_l * sqrt_a2_b2)
                       - 1.j * alpha_tidle * cmath.sin(normalized_l * sqrt_a2_b2) / sqrt_a2_b2)

        # Assemble everything
        reflectivity_sigma[idx] = numerator / denominator

        # Get the output electric field due to this component
        efield_out_sigma_x = reflectivity_sigma[idx] * efield_sigma * complex(sigma_in_x)
        efield_out_sigma_y = reflectivity_sigma[idx] * efield_sigma * complex(sigma_in_y)
        efield_out_sigma_z = reflectivity_sigma[idx] * efield_sigma * complex(sigma_in_z)

        #####################################################################################################
        # Step 3: Get the reflectivity for input pi polarization
        #####################################################################################################
        # ----------------------------------------
        #    Get reflectivity
        # ----------------------------------------

        # Get the polarization factor with the asymmetric factor b.
        p_value = complex((kout_x * kin_x +
                           kout_y * kin_y +
                           kout_z * kin_z) / (klen ** 2))
        bp = b_complex * p_value

        # Get sqrt(alpha**2 + beta**2) value
        sqrt_a2_b2 = cmath.sqrt(alpha_tidle ** 2 + bp * p_value * chih_pi * chihbar_pi)

        # Because this is a thick crystal, only one mode will be activated.
        if sqrt_a2_b2.imag < 0.:
            # Because only one mode is activated,
            sqrt_a2_b2 = - sqrt_a2_b2

        # Calculate the normalized lenght l
        normalized_l = klen * d / 2 / gamma_0
        re = normalized_l * (chi0.real - alpha_tidle.real)
        im = normalized_l * (chi0.imag - alpha_tidle.imag)

        # Take care of the exponential
        magnitude = complex(math.exp(-im))
        if math.isnan(magnitude.real):
            magnitude = complex(0.)
        phase = complex(math.cos(re), math.sin(re))

        # Calculate some intermediate part
        numerator = magnitude * phase
        denominator = (cmath.cos(normalized_l * sqrt_a2_b2)
                       - 1.j * alpha_tidle * cmath.sin(normalized_l * sqrt_a2_b2) / sqrt_a2_b2)

        # Assemble everything
        reflectivity_sigma[idx] = numerator / denominator

        # Get the output electric field due to this component
        efield_out_pi_x = reflectivity_pi[idx] * efield_pi * complex(pi_out_x)
        efield_out_pi_y = reflectivity_pi[idx] * efield_pi * complex(pi_out_y)
        efield_out_pi_z = reflectivity_pi[idx] * efield_pi * complex(pi_out_z)

        #####################################################################################################
        # Step 4: Assemble to get the output electric field
        #####################################################################################################
        efield_grid[idx, 0] = efield_out_sigma_x + efield_out_pi_x
        efield_grid[idx, 1] = efield_out_sigma_y + efield_out_pi_y
        efield_grid[idx, 2] = efield_out_sigma_z + efield_out_pi_z


########################################################################################################################
#                    For laue diffraction
########################################################################################################################
@cuda.jit('void(complex128[:], complex128[:], float64[:,:], complex128[:,:],'
          'float64[:], float64[:,:],'
          'float64, float64[:], float64[:],'
          'float64, float64,'
          'complex128, complex128, complex128, complex128, complex128,'
          'int64)')
def get_laue_reflection(reflectivity_sigma, reflectivity_pi, kout_grid, efield_grid,
                        klen_grid, kin_grid,
                        d, h, n,
                        dot_hn, h_square,
                        chi0, chih_sigma, chihbar_sigma, chih_pi, chihbar_pi,
                        num):
    """
    Given the crystal info, the input electric field, this function returns the
    reflectivity for the sigma polarization and pi polarization and the
    diffracted electric field.

    :param reflectivity_sigma:
    :param reflectivity_pi:
    :param kout_grid:
    :param efield_grid:
    :param klen_grid:
    :param kin_grid:
    :param d:
    :param h:
    :param n:
    :param dot_hn:
    :param h_square:
    :param chi0:
    :param chih_sigma:
    :param chihbar_sigma:
    :param chih_pi:
    :param chihbar_pi:
    :param num:
    :return:
    """
    idx = cuda.grid(1)
    if idx < num:

        #####################################################################################################
        # Step 1: Get parameters for reflectivity and decompose input field
        #####################################################################################################
        # ------------------------------------
        #     Get the diffracted wave number
        # ------------------------------------
        # Get k components
        kin_x = kin_grid[idx, 0]
        kin_y = kin_grid[idx, 1]
        kin_z = kin_grid[idx, 2]
        klen = klen_grid[idx]

        # Get gamma and alpha and b
        dot_kn = kin_x * n[0] + kin_y * n[1] + kin_z * n[2]
        dot_kh = kin_x * h[0] + kin_y * h[1] + kin_z * h[2]

        gamma_0 = dot_kn / klen
        gamma_h = (dot_kn + dot_hn) / klen
        b = gamma_0 / gamma_h
        b_complex = complex(b)
        alpha = (2 * dot_kh + h_square) / (klen ** 2)

        # Get momentum tranfer
        sqrt_gamma_alpha = math.sqrt(gamma_h ** 2 - alpha)
        m_trans = klen * (-gamma_h - sqrt_gamma_alpha)

        # tmp_pos = abs(-gamma_h + sqrt_gamma_alpha)
        # tmp_neg = abs(-gamma_h - sqrt_gamma_alpha)
        # if tmp_pos > tmp_neg:
        #    m_trans = klen * (-gamma_h - sqrt_gamma_alpha)
        # else:
        #    m_trans = klen * (-gamma_h + sqrt_gamma_alpha)

        # Get output wave vector
        kout_x = kin_x + h[0] + m_trans * n[0]
        kout_y = kin_y + h[1] + m_trans * n[1]
        kout_z = kin_z + h[2] + m_trans * n[2]

        # Update the kout_grid
        kout_grid[idx, 0] = kout_x
        kout_grid[idx, 1] = kout_y
        kout_grid[idx, 2] = kout_z

        # ----------------------------------------
        #     Get polarization component
        # ----------------------------------------
        """
        sigma_in = kH cross k0     sigma_out = sigma_in  
        pi_in = k0 cross sigma0    pi_out = k_out cross sigma_out        
        """
        sigma_in_x = kout_y * kin_z - kout_z * kin_y
        sigma_in_y = kout_z * kin_x - kout_x * kin_z
        sigma_in_z = kout_x * kin_y - kout_y * kin_x
        tmp_len = math.sqrt(sigma_in_x ** 2 + sigma_in_y ** 2 + sigma_in_z ** 2)
        sigma_in_x /= tmp_len
        sigma_in_y /= tmp_len
        sigma_in_z /= tmp_len

        pi_in_x = kin_y * sigma_in_z - kin_z * sigma_in_y
        pi_in_y = kin_z * sigma_in_x - kin_x * sigma_in_z
        pi_in_z = kin_x * sigma_in_y - kin_y * sigma_in_x
        tmp_len = math.sqrt(pi_in_x ** 2 + pi_in_y ** 2 + pi_in_z ** 2)
        pi_in_x /= tmp_len
        pi_in_y /= tmp_len
        pi_in_z /= tmp_len

        pi_out_x = kout_y * sigma_in_z - kout_z * sigma_in_y
        pi_out_y = kout_z * sigma_in_x - kout_x * sigma_in_z
        pi_out_z = kout_x * sigma_in_y - kout_y * sigma_in_x
        tmp_len = math.sqrt(pi_out_x ** 2 + pi_out_y ** 2 + pi_out_z ** 2)
        pi_out_x /= tmp_len
        pi_out_y /= tmp_len
        pi_out_z /= tmp_len

        # Decompose the input electric field
        efield_sigma = (efield_grid[idx, 0] * complex(sigma_in_x) +
                        efield_grid[idx, 1] * complex(sigma_in_y) +
                        efield_grid[idx, 2] * complex(sigma_in_z))
        efield_pi = (efield_grid[idx, 0] * complex(pi_in_x) +
                     efield_grid[idx, 1] * complex(pi_in_y) +
                     efield_grid[idx, 2] * complex(pi_in_z))

        #####################################################################################################
        # Step 2: Get the reflectivity for input sigma polarization
        #####################################################################################################
        # ----------------------------------------
        #    Get reflectivity
        # ----------------------------------------
        # Get alpha tidle
        alpha_tidle = complex((alpha * b + chi0.real * (1. - b)) / 2., chi0.imag * (1. - b) / 2.)

        # Get sqrt(alpha**2 + beta**2) value
        sqrt_a2_b2 = cmath.sqrt(alpha_tidle ** 2 + b_complex * chih_sigma * chihbar_sigma)

        if sqrt_a2_b2.imag < 0:
            sqrt_a2_b2 = - sqrt_a2_b2

        # Calculate the normalized lenght l
        normalized_l = klen * d / 2 / gamma_0
        re = normalized_l * (chi0.real - alpha_tidle.real)
        im = normalized_l * (chi0.imag - alpha_tidle.imag)

        # Take care of the exponential
        magnitude = complex(math.exp(-im))
        if math.isnan(magnitude.real):
            magnitude = complex(0.)
        phase = complex(math.cos(re), math.sin(re))

        # Calculate some intermediate part
        factor1 = 1.j * b_complex * chih_sigma * magnitude * phase
        factor2 = cmath.sin(normalized_l * sqrt_a2_b2) / sqrt_a2_b2

        # Assemble everything
        reflectivity_sigma[idx] = factor1 * factor2

        # Get the output electric field due to this component
        efield_out_sigma_x = reflectivity_sigma[idx] * efield_sigma * complex(sigma_in_x)
        efield_out_sigma_y = reflectivity_sigma[idx] * efield_sigma * complex(sigma_in_y)
        efield_out_sigma_z = reflectivity_sigma[idx] * efield_sigma * complex(sigma_in_z)

        #####################################################################################################
        # Step 3: Get the reflectivity for input pi polarization
        #####################################################################################################
        # ----------------------------------------
        #    Get reflectivity
        # ----------------------------------------

        # Get the polarization factor with the asymmetric factor b.
        p_value = complex((kout_x * kin_x +
                           kout_y * kin_y +
                           kout_z * kin_z) / (klen ** 2))
        bp = b_complex * p_value

        # Get sqrt(alpha**2 + beta**2) value
        sqrt_a2_b2 = cmath.sqrt(alpha_tidle ** 2 + bp * p_value * chih_pi * chihbar_pi)

        # Because this is a thick crystal, only one mode will be activated.
        if sqrt_a2_b2.imag < 0.:
            # Because only one mode is activated,
            sqrt_a2_b2 = - sqrt_a2_b2

        # Calculate the normalized lenght l
        normalized_l = klen * d / 2 / gamma_0
        re = normalized_l * (chi0.real - alpha_tidle.real)
        im = normalized_l * (chi0.imag - alpha_tidle.imag)

        # Take care of the exponential
        magnitude = complex(math.exp(-im))
        if math.isnan(magnitude.real):
            magnitude = complex(0.)
        phase = complex(math.cos(re), math.sin(re))

        # Calculate some intermediate part
        factor1 = 1.j * p_value * b_complex * chih_sigma * magnitude * phase
        factor2 = cmath.sin(normalized_l * sqrt_a2_b2) / sqrt_a2_b2

        # Assemble everything
        reflectivity_sigma[idx] = factor1 * factor2

        # Get the output electric field due to this component
        efield_out_pi_x = reflectivity_pi[idx] * efield_pi * complex(pi_out_x)
        efield_out_pi_y = reflectivity_pi[idx] * efield_pi * complex(pi_out_y)
        efield_out_pi_z = reflectivity_pi[idx] * efield_pi * complex(pi_out_z)

        #####################################################################################################
        # Step 4: Assemble to get the output electric field
        #####################################################################################################
        efield_grid[idx, 0] = efield_out_sigma_x + efield_out_pi_x
        efield_grid[idx, 1] = efield_out_sigma_y + efield_out_pi_y
        efield_grid[idx, 2] = efield_out_sigma_z + efield_out_pi_z


@cuda.jit('void(complex128[:], complex128[:], float64[:,:], complex128[:,:],'
          'float64[:], float64[:,:],'
          'float64, float64[:], float64[:],'
          'float64, float64,'
          'complex128, complex128, complex128, complex128, complex128,'
          'int64)')
def get_laue_reflection(reflectivity_sigma,
                        reflectivity_pi,
                        kout_grid,
                        efield_grid,
                        klen_grid,
                        kin_grid,
                        d, h, n,
                        dot_hn, h_square,
                        chi0,
                        chih_sigma,
                        chihbar_sigma,
                        chih_pi,
                        chihbar_pi,
                        num):
    """
    Given the crystal info, the input electric field, this function returns the
    reflectivity for the sigma polarization and pi polarization and the
    diffracted electric field.

    :param reflectivity_sigma:
    :param reflectivity_pi:
    :param kout_grid:
    :param efield_grid:
    :param klen_grid:
    :param kin_grid:
    :param d:
    :param h:
    :param n:
    :param dot_hn:
    :param h_square:
    :param chi0:
    :param chih_sigma:
    :param chihbar_sigma:
    :param chih_pi:
    :param chihbar_pi:
    :param num:
    :return:
    """
    idx = cuda.grid(1)
    if idx < num:

        #####################################################################################################
        # Step 1: Get parameters for reflectivity and decompose input field
        #####################################################################################################
        # ------------------------------------
        #     Get the diffracted wave number
        # ------------------------------------
        # Get k components
        kin_x = kin_grid[idx, 0]
        kin_y = kin_grid[idx, 1]
        kin_z = kin_grid[idx, 2]
        klen = klen_grid[idx]

        # Get gamma and alpha and b
        dot_kn = kin_x * n[0] + kin_y * n[1] + kin_z * n[2]
        dot_kh = kin_x * h[0] + kin_y * h[1] + kin_z * h[2]

        gamma_0 = dot_kn / klen
        gamma_h = (dot_kn + dot_hn) / klen
        b = gamma_0 / gamma_h
        b_complex = complex(b)
        alpha = (2 * dot_kh + h_square) / (klen ** 2)

        # Get momentum tranfer
        sqrt_gamma_alpha = math.sqrt(gamma_h ** 2 - alpha)
        m_trans = klen * (-gamma_h - sqrt_gamma_alpha)

        # tmp_pos = abs(-gamma_h + sqrt_gamma_alpha)
        # tmp_neg = abs(-gamma_h - sqrt_gamma_alpha)
        # if tmp_pos > tmp_neg:
        #    m_trans = klen * (-gamma_h - sqrt_gamma_alpha)
        # else:
        #    m_trans = klen * (-gamma_h + sqrt_gamma_alpha)

        # Get output wave vector
        kout_x = kin_x + h[0] + m_trans * n[0]
        kout_y = kin_y + h[1] + m_trans * n[1]
        kout_z = kin_z + h[2] + m_trans * n[2]

        # Update the kout_grid
        kout_grid[idx, 0] = kout_x
        kout_grid[idx, 1] = kout_y
        kout_grid[idx, 2] = kout_z

        # ----------------------------------------
        #     Get polarization component
        # ----------------------------------------
        """
        sigma_in = kH cross k0     sigma_out = sigma_in  
        pi_in = k0 cross sigma0    pi_out = k_out cross sigma_out        
        """
        sigma_in_x = kout_y * kin_z - kout_z * kin_y
        sigma_in_y = kout_z * kin_x - kout_x * kin_z
        sigma_in_z = kout_x * kin_y - kout_y * kin_x
        tmp_len = math.sqrt(sigma_in_x ** 2 + sigma_in_y ** 2 + sigma_in_z ** 2)
        sigma_in_x /= tmp_len
        sigma_in_y /= tmp_len
        sigma_in_z /= tmp_len

        pi_in_x = kin_y * sigma_in_z - kin_z * sigma_in_y
        pi_in_y = kin_z * sigma_in_x - kin_x * sigma_in_z
        pi_in_z = kin_x * sigma_in_y - kin_y * sigma_in_x
        tmp_len = math.sqrt(pi_in_x ** 2 + pi_in_y ** 2 + pi_in_z ** 2)
        pi_in_x /= tmp_len
        pi_in_y /= tmp_len
        pi_in_z /= tmp_len

        pi_out_x = kout_y * sigma_in_z - kout_z * sigma_in_y
        pi_out_y = kout_z * sigma_in_x - kout_x * sigma_in_z
        pi_out_z = kout_x * sigma_in_y - kout_y * sigma_in_x
        tmp_len = math.sqrt(pi_out_x ** 2 + pi_out_y ** 2 + pi_out_z ** 2)
        pi_out_x /= tmp_len
        pi_out_y /= tmp_len
        pi_out_z /= tmp_len

        # Decompose the input electric field
        efield_sigma = (efield_grid[idx, 0] * complex(sigma_in_x) +
                        efield_grid[idx, 1] * complex(sigma_in_y) +
                        efield_grid[idx, 2] * complex(sigma_in_z))
        efield_pi = (efield_grid[idx, 0] * complex(pi_in_x) +
                     efield_grid[idx, 1] * complex(pi_in_y) +
                     efield_grid[idx, 2] * complex(pi_in_z))

        #####################################################################################################
        # Step 2: Get the reflectivity for input sigma polarization
        #####################################################################################################
        # ----------------------------------------
        #    Get reflectivity
        # ----------------------------------------
        # Get alpha tidle
        alpha_tidle = complex((alpha * b + chi0.real * (1. - b)) / 2., chi0.imag * (1. - b) / 2.)

        # Get sqrt(alpha**2 + beta**2) value
        sqrt_a2_b2 = cmath.sqrt(alpha_tidle ** 2 + b_complex * chih_sigma * chihbar_sigma)

        if sqrt_a2_b2.imag < 0:
            sqrt_a2_b2 = - sqrt_a2_b2

        # Calculate the normalized lenght l
        normalized_l = klen * d / 2 / gamma_0
        re = normalized_l * (chi0.real - alpha_tidle.real)
        im = normalized_l * (chi0.imag - alpha_tidle.imag)

        # Take care of the exponential
        magnitude = complex(math.exp(-im))
        if math.isnan(magnitude.real):
            magnitude = complex(0.)
        phase = complex(math.cos(re), math.sin(re))

        # Calculate some intermediate part
        factor1 = 1.j * b_complex * chih_sigma * magnitude * phase
        factor2 = cmath.sin(normalized_l * sqrt_a2_b2) / sqrt_a2_b2

        # Assemble everything
        reflectivity_sigma[idx] = factor1 * factor2

        # Get the output electric field due to this component
        efield_out_sigma_x = reflectivity_sigma[idx] * efield_sigma * complex(sigma_in_x)
        efield_out_sigma_y = reflectivity_sigma[idx] * efield_sigma * complex(sigma_in_y)
        efield_out_sigma_z = reflectivity_sigma[idx] * efield_sigma * complex(sigma_in_z)

        #####################################################################################################
        # Step 3: Get the reflectivity for input pi polarization
        #####################################################################################################
        # ----------------------------------------
        #    Get reflectivity
        # ----------------------------------------

        # Get the polarization factor with the asymmetric factor b.
        p_value = complex((kout_x * kin_x +
                           kout_y * kin_y +
                           kout_z * kin_z) / (klen ** 2))
        bp = b_complex * p_value

        # Get sqrt(alpha**2 + beta**2) value
        sqrt_a2_b2 = cmath.sqrt(alpha_tidle ** 2 + bp * p_value * chih_pi * chihbar_pi)

        # Because this is a thick crystal, only one mode will be activated.
        if sqrt_a2_b2.imag < 0.:
            # Because only one mode is activated,
            sqrt_a2_b2 = - sqrt_a2_b2

        # Calculate the normalized lenght l
        normalized_l = klen * d / 2 / gamma_0
        re = normalized_l * (chi0.real - alpha_tidle.real)
        im = normalized_l * (chi0.imag - alpha_tidle.imag)

        # Take care of the exponential
        magnitude = complex(math.exp(-im))
        if math.isnan(magnitude.real):
            magnitude = complex(0.)
        phase = complex(math.cos(re), math.sin(re))

        # Calculate some intermediate part
        factor1 = 1.j * p_value * b_complex * chih_sigma * magnitude * phase
        factor2 = cmath.sin(normalized_l * sqrt_a2_b2) / sqrt_a2_b2

        # Assemble everything
        reflectivity_sigma[idx] = factor1 * factor2

        # Get the output electric field due to this component
        efield_out_pi_x = reflectivity_pi[idx] * efield_pi * complex(pi_out_x)
        efield_out_pi_y = reflectivity_pi[idx] * efield_pi * complex(pi_out_y)
        efield_out_pi_z = reflectivity_pi[idx] * efield_pi * complex(pi_out_z)

        #####################################################################################################
        # Step 4: Assemble to get the output electric field
        #####################################################################################################
        efield_grid[idx, 0] = efield_out_sigma_x + efield_out_pi_x
        efield_grid[idx, 1] = efield_out_sigma_y + efield_out_pi_y
        efield_grid[idx, 2] = efield_out_sigma_z + efield_out_pi_z


@cuda.jit('void(complex128[:], complex128[:], float64[:,:], complex128[:,:],'
          'float64[:], float64[:,:],'
          'float64, float64[:], float64[:],'
          'float64, float64,'
          'complex128, complex128, complex128, complex128, complex128,'
          'int64)')
def get_forward_laue_reflection(reflectivity_sigma,
                                reflectivity_pi,
                                kout_grid,
                                efield_grid,
                                klen_grid,
                                kin_grid,
                                d, h, n,
                                dot_hn, h_square,
                                chi0,
                                chih_sigma,
                                chihbar_sigma,
                                chih_pi,
                                chihbar_pi,
                                num):
    """
    Given the crystal info, the input electric field, this function returns the
    reflectivity for the sigma polarization and pi polarization and the
    diffracted electric field.

    :param reflectivity_sigma:
    :param reflectivity_pi:
    :param kout_grid:
    :param efield_grid:
    :param klen_grid:
    :param kin_grid:
    :param d:
    :param h:
    :param n:
    :param dot_hn:
    :param h_square:
    :param chi0:
    :param chih_sigma:
    :param chihbar_sigma:
    :param chih_pi:
    :param chihbar_pi:
    :param num:
    :return:
    """
    idx = cuda.grid(1)
    if idx < num:

        #####################################################################################################
        # Step 1: Get parameters for reflectivity and decompose input field
        #####################################################################################################
        # ------------------------------------
        #     Get the diffracted wave number
        # ------------------------------------
        # Get k components
        kin_x = kin_grid[idx, 0]
        kin_y = kin_grid[idx, 1]
        kin_z = kin_grid[idx, 2]
        klen = klen_grid[idx]

        # Get gamma and alpha and b
        dot_kn = kin_x * n[0] + kin_y * n[1] + kin_z * n[2]
        dot_kh = kin_x * h[0] + kin_y * h[1] + kin_z * h[2]

        gamma_0 = dot_kn / klen
        gamma_h = (dot_kn + dot_hn) / klen
        b = gamma_0 / gamma_h
        b_complex = complex(b)
        alpha = (2 * dot_kh + h_square) / (klen ** 2)

        # Get momentum tranfer
        sqrt_gamma_alpha = math.sqrt(gamma_h ** 2 - alpha)
        m_trans = klen * (-gamma_h - sqrt_gamma_alpha)

        # tmp_pos = abs(-gamma_h + sqrt_gamma_alpha)
        # tmp_neg = abs(-gamma_h - sqrt_gamma_alpha)
        # if tmp_pos > tmp_neg:
        #    m_trans = klen * (-gamma_h - sqrt_gamma_alpha)
        # else:
        #    m_trans = klen * (-gamma_h + sqrt_gamma_alpha)

        # Get output wave vector
        kout_x = kin_x + h[0] + m_trans * n[0]
        kout_y = kin_y + h[1] + m_trans * n[1]
        kout_z = kin_z + h[2] + m_trans * n[2]

        # Update the kout_grid
        kout_grid[idx, 0] = kin_x
        kout_grid[idx, 1] = kin_y
        kout_grid[idx, 2] = kin_z

        # ----------------------------------------
        #     Get polarization component
        # ----------------------------------------
        """
        sigma_in = kH cross k0     sigma_out = sigma_in  
        pi_in = k0 cross sigma0    pi_out = k_out cross sigma_out        
        """
        sigma_in_x = kout_y * kin_z - kout_z * kin_y
        sigma_in_y = kout_z * kin_x - kout_x * kin_z
        sigma_in_z = kout_x * kin_y - kout_y * kin_x
        tmp_len = math.sqrt(sigma_in_x ** 2 + sigma_in_y ** 2 + sigma_in_z ** 2)
        sigma_in_x /= tmp_len
        sigma_in_y /= tmp_len
        sigma_in_z /= tmp_len

        pi_in_x = kin_y * sigma_in_z - kin_z * sigma_in_y
        pi_in_y = kin_z * sigma_in_x - kin_x * sigma_in_z
        pi_in_z = kin_x * sigma_in_y - kin_y * sigma_in_x
        tmp_len = math.sqrt(pi_in_x ** 2 + pi_in_y ** 2 + pi_in_z ** 2)
        pi_in_x /= tmp_len
        pi_in_y /= tmp_len
        pi_in_z /= tmp_len

        pi_out_x = kout_y * sigma_in_z - kout_z * sigma_in_y
        pi_out_y = kout_z * sigma_in_x - kout_x * sigma_in_z
        pi_out_z = kout_x * sigma_in_y - kout_y * sigma_in_x
        tmp_len = math.sqrt(pi_out_x ** 2 + pi_out_y ** 2 + pi_out_z ** 2)
        pi_out_x /= tmp_len
        pi_out_y /= tmp_len
        pi_out_z /= tmp_len

        # Decompose the input electric field
        efield_sigma = (efield_grid[idx, 0] * complex(sigma_in_x) +
                        efield_grid[idx, 1] * complex(sigma_in_y) +
                        efield_grid[idx, 2] * complex(sigma_in_z))
        efield_pi = (efield_grid[idx, 0] * complex(pi_in_x) +
                     efield_grid[idx, 1] * complex(pi_in_y) +
                     efield_grid[idx, 2] * complex(pi_in_z))

        #####################################################################################################
        # Step 2: Get the reflectivity for input sigma polarization
        #####################################################################################################
        # ----------------------------------------
        #    Get reflectivity
        # ----------------------------------------
        # Get alpha tidle
        alpha_tidle = complex((alpha * b + chi0.real * (1. - b)) / 2., chi0.imag * (1. - b) / 2.)

        # Get sqrt(alpha**2 + beta**2) value
        sqrt_a2_b2 = cmath.sqrt(alpha_tidle ** 2 + b_complex * chih_sigma * chihbar_sigma)

        if sqrt_a2_b2.imag < 0:
            sqrt_a2_b2 = - sqrt_a2_b2

        # Calculate the normalized lenght l
        normalized_l = klen * d / 2 / gamma_0
        re = normalized_l * (chi0.real - alpha_tidle.real)
        im = normalized_l * (chi0.imag - alpha_tidle.imag)

        # Take care of the exponential
        magnitude = complex(math.exp(-im))
        if math.isnan(magnitude.real):
            magnitude = complex(0.)
        phase = complex(math.cos(re), math.sin(re))

        # Calculate some intermediate part
        factor1 = magnitude * phase
        factor2 = (cmath.cos(normalized_l * sqrt_a2_b2) +
                   1.j * alpha_tidle * cmath.sin(normalized_l * sqrt_a2_b2) / sqrt_a2_b2)
        # Assemble everything
        reflectivity_sigma[idx] = factor1 * factor2

        # Get the output electric field due to this component
        efield_out_sigma_x = reflectivity_sigma[idx] * efield_sigma * complex(sigma_in_x)
        efield_out_sigma_y = reflectivity_sigma[idx] * efield_sigma * complex(sigma_in_y)
        efield_out_sigma_z = reflectivity_sigma[idx] * efield_sigma * complex(sigma_in_z)

        #####################################################################################################
        # Step 3: Get the reflectivity for input pi polarization
        #####################################################################################################
        # ----------------------------------------
        #    Get reflectivity
        # ----------------------------------------

        # Get the polarization factor with the asymmetric factor b.
        p_value = complex((kout_x * kin_x +
                           kout_y * kin_y +
                           kout_z * kin_z) / (klen ** 2))
        bp = b_complex * p_value

        # Get sqrt(alpha**2 + beta**2) value
        sqrt_a2_b2 = cmath.sqrt(alpha_tidle ** 2 + bp * p_value * chih_pi * chihbar_pi)

        # Because this is a thick crystal, only one mode will be activated.
        if sqrt_a2_b2.imag < 0.:
            # Because only one mode is activated,
            sqrt_a2_b2 = - sqrt_a2_b2

        # Calculate the normalized lenght l
        normalized_l = klen * d / 2 / gamma_0
        re = normalized_l * (chi0.real - alpha_tidle.real)
        im = normalized_l * (chi0.imag - alpha_tidle.imag)

        # Take care of the exponential
        magnitude = complex(math.exp(-im))
        if math.isnan(magnitude.real):
            magnitude = complex(0.)
        phase = complex(math.cos(re), math.sin(re))

        # Calculate some intermediate part
        factor1 = magnitude * phase
        factor2 = (cmath.cos(normalized_l * sqrt_a2_b2) +
                   1.j * alpha_tidle * cmath.sin(normalized_l * sqrt_a2_b2) / sqrt_a2_b2)
        # Assemble everything
        reflectivity_sigma[idx] = factor1 * factor2

        # Get the output electric field due to this component
        efield_out_pi_x = reflectivity_pi[idx] * efield_pi * complex(pi_out_x)
        efield_out_pi_y = reflectivity_pi[idx] * efield_pi * complex(pi_out_y)
        efield_out_pi_z = reflectivity_pi[idx] * efield_pi * complex(pi_out_z)

        #####################################################################################################
        # Step 4: Assemble to get the output electric field
        #####################################################################################################
        efield_grid[idx, 0] = efield_out_sigma_x + efield_out_pi_x
        efield_grid[idx, 1] = efield_out_sigma_y + efield_out_pi_y
        efield_grid[idx, 2] = efield_out_sigma_z + efield_out_pi_z
