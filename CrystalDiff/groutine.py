import time

import numpy as np
from numba import cuda
from pyculib import fft as cufft

from CrystalDiff import gfun


def get_diffraction_field(crystal_list,
                          total_path,
                          observation,
                          my_pulse,
                          pulse_delay_time,
                          pulse_k0_final,
                          grating_orders,
                          kx_grid,
                          ky_grid,
                          kz_grid,
                          number_x, number_y, number_z,
                          z_idx_range,
                          idx_start_1,
                          idx_start_2,
                          num1,
                          num2,
                          d_num=512):
    """

    :param crystal_list:
    :param total_path:
    :param observation:
    :param my_pulse:
    :param pulse_delay_time:
    :param pulse_k0_final: The output wave vector of the central wave length of the incident pulse.
    :param grating_orders:
    :param kx_grid:
    :param ky_grid:
    :param kz_grid:
    :param number_x:
    :param number_y:
    :param number_z:
    :param z_idx_range:
    :param idx_start_1:
    :param idx_start_2:
    :param num1:
    :param num2:
    :param d_num:
    :return:
    """
    crystal_num = len(crystal_list)

    tic = time.time()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #  [3D Blocks]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    efield_3d = np.zeros((number_x, number_y, z_idx_range, 3), dtype=np.complex128)
    efield_spec_3d = np.zeros((number_x, number_y, z_idx_range, 3), dtype=np.complex128)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #  [2D slices]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    x_field_2d = np.ascontiguousarray(np.zeros((number_y, z_idx_range), dtype=np.complex128))
    y_field_2d = np.ascontiguousarray(np.zeros((number_y, z_idx_range), dtype=np.complex128))
    z_field_2d = np.ascontiguousarray(np.zeros((number_y, z_idx_range), dtype=np.complex128))
    x_spec_2d = np.ascontiguousarray(np.zeros((number_y, z_idx_range), dtype=np.complex128))
    y_spec_2d = np.ascontiguousarray(np.zeros((number_y, z_idx_range), dtype=np.complex128))
    z_spec_2d = np.ascontiguousarray(np.zeros((number_y, z_idx_range), dtype=np.complex128))

    cuda_x_field_2d = cuda.to_device(x_field_2d)
    cuda_y_field_2d = cuda.to_device(y_field_2d)
    cuda_z_field_2d = cuda.to_device(z_field_2d)
    cuda_x_spec_2d = cuda.to_device(x_spec_2d)
    cuda_y_spec_2d = cuda.to_device(y_spec_2d)
    cuda_z_spec_2d = cuda.to_device(z_spec_2d)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #  [1D slices] Various intersection points, path length and phase
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    intersect_points = np.ascontiguousarray(np.zeros((number_z, 3), dtype=np.float64))
    component_final_points = np.ascontiguousarray(np.zeros((number_z, 3), dtype=np.float64))
    remaining_length = np.ascontiguousarray(np.zeros(number_z, dtype=np.float64))
    phase_grid = np.ascontiguousarray(np.ones(number_z, dtype=np.complex128))
    jacob_grid = np.ascontiguousarray(np.ones(number_z, dtype=np.complex128))

    cuda_intersect = cuda.to_device(intersect_points)
    cuda_final_points = cuda.to_device(component_final_points)
    cuda_remain_path = cuda.to_device(remaining_length)
    cuda_phase = cuda.to_device(phase_grid)
    cuda_jacob = cuda.to_device(jacob_grid)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # [1D slices] reflect and time response
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    reflect_pi = np.ascontiguousarray(np.ones(number_z, dtype=np.complex128))
    reflect_total_pi = np.ascontiguousarray(np.ones(number_z, dtype=np.complex128))
    reflect_sigma = np.ascontiguousarray(np.ones(number_z, dtype=np.complex128))
    reflect_total_sigma = np.ascontiguousarray(np.ones(number_z, dtype=np.complex128))

    cuda_reflect_pi = cuda.to_device(reflect_pi)
    cuda_reflect_total_pi = cuda.to_device(reflect_total_pi)
    cuda_reflect_sigma = cuda.to_device(reflect_sigma)
    cuda_reflect_total_sigma = cuda.to_device(reflect_total_sigma)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # [1D slices] Vector field
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # The reciprocal space
    coef_grid = np.ascontiguousarray(np.zeros(number_z, dtype=np.complex128))  # Input spectrum without Jacobian
    scalar_spec_holder = np.ascontiguousarray(np.ones(number_z, dtype=np.complex128))  # With Jacobian
    vector_spec_holder = np.ascontiguousarray(np.zeros((number_z, 3), dtype=np.complex128))
    x_spec_holder = np.ascontiguousarray(np.zeros(number_z, dtype=np.complex128))
    y_spec_holder = np.ascontiguousarray(np.zeros(number_z, dtype=np.complex128))
    z_spec_holder = np.ascontiguousarray(np.zeros(number_z, dtype=np.complex128))
    x_field_holder = np.ascontiguousarray(np.zeros(number_z, dtype=np.complex128))
    y_field_holder = np.ascontiguousarray(np.zeros(number_z, dtype=np.complex128))
    z_field_holder = np.ascontiguousarray(np.zeros(number_z, dtype=np.complex128))

    cuda_coef = cuda.to_device(coef_grid)
    cuda_spec_scalar = cuda.to_device(scalar_spec_holder)
    cuda_spec_vec = cuda.to_device(vector_spec_holder)
    cuda_spec_x = cuda.to_device(x_spec_holder)
    cuda_spec_y = cuda.to_device(y_spec_holder)
    cuda_spec_z = cuda.to_device(z_spec_holder)
    cuda_x_field = cuda.to_device(x_field_holder)
    cuda_y_field = cuda.to_device(y_field_holder)
    cuda_z_field = cuda.to_device(z_field_holder)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #  [1D slices] k grid
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    kin_grid = np.ascontiguousarray(np.zeros((number_z, 3), dtype=np.float64))
    klen_grid = np.ascontiguousarray(np.zeros(number_z, dtype=np.float64))

    kz_grid = np.ascontiguousarray(kz_grid)
    kz_square = np.ascontiguousarray(np.square(kz_grid))

    cuda_kin_grid = cuda.to_device(kin_grid)
    cuda_klen_grid = cuda.to_device(klen_grid)
    cuda_kz_grid = cuda.to_device(kz_grid)
    cuda_kz_square = cuda.to_device(kz_square)

    toc = time.time()
    print("It takes {:.2f} seconds to prepare the variables.".format(toc - tic))

    ############################################################################################################
    # ----------------------------------------------------------------------------------------------------------
    #
    #                           Step 2: Calculate the field and save the data
    #
    # ----------------------------------------------------------------------------------------------------------
    ############################################################################################################
    # d_num = 512
    b_num = (number_z + d_num - 1) // d_num

    for x_idx in range(number_x):
        kx = kx_grid[x_idx]

        for y_idx in range(number_y):
            ky = ky_grid[y_idx]

            # --------------------------------------------------------------------
            #  Step 1. Get k_out mesh
            # --------------------------------------------------------------------
            gfun.init_kvec[b_num, d_num](cuda_kin_grid,
                                         cuda_klen_grid,
                                         cuda_kz_grid,
                                         cuda_kz_square,
                                         kx,
                                         ky,
                                         ky ** 2 + kx ** 2,
                                         number_z)

            gfun.init_jacobian[b_num, d_num](cuda_jacob, number_z)
            gfun.init_scalar_grid[b_num, d_num](cuda_remain_path, total_path, number_z)
            gfun.init_vector_grid[b_num, d_num](cuda_intersect, my_pulse.x0, 3, number_z)

            # --------------------------------------------------------------------
            #  Step 2. Back propagate through all the crystals
            # --------------------------------------------------------------------
            # Define a variable to track the grating index for the grating order
            grating_idx = len(grating_orders) - 1

            for crystal_idx in range(crystal_num - 1, -1, -1):

                # Get the crystal
                my_crystal = crystal_list[crystal_idx]

                # Determine the type of the crystal
                if my_crystal.type == "Transmissive Grating":
                    gfun.add_vector[b_num, d_num](cuda_kin_grid,
                                                  cuda_kin_grid,
                                                  - grating_orders[grating_idx] * my_crystal.base_wave_vector,
                                                  number_z)

                    # Update the wave number
                    gfun.get_vector_length[b_num, d_num](cuda_klen_grid,
                                                         cuda_kin_grid,
                                                         3,
                                                         number_z)

                    # Update the grating idx
                    grating_idx -= 1

                if my_crystal.type == "Crystal: Bragg Reflection":
                    # Calculate the incident wave vector
                    gfun.get_kin_and_jacobian[b_num, d_num](cuda_kin_grid,
                                                            cuda_jacob,
                                                            cuda_klen_grid,
                                                            cuda_kin_grid,
                                                            my_crystal.h,
                                                            my_crystal.normal,
                                                            my_crystal.dot_hn,
                                                            my_crystal.h_square,
                                                            number_z)

                if my_crystal.type == "prism":
                    gfun.add_vector[b_num, d_num](cuda_kin_grid,
                                                  cuda_kin_grid,
                                                  - my_crystal.additional_wavevector,
                                                  number_z)

                    # Update the wave number
                    gfun.get_vector_length[b_num, d_num](cuda_klen_grid,
                                                         cuda_kin_grid,
                                                         3,
                                                         number_z)

            # --------------------------------------------------------------------
            #  Step 5. Get the coefficient of each monochromatic component
            # --------------------------------------------------------------------
            # Calculate the corresponding coefficient in the incident pulse
            gfun.get_gaussian_pulse_spectrum[b_num, d_num](cuda_coef,
                                                           cuda_kin_grid,
                                                           float(pulse_delay_time),
                                                           my_pulse.sigma_mat,
                                                           my_pulse.scaling,
                                                           np.zeros(3, dtype=np.float64),
                                                           my_pulse.k0,
                                                           my_pulse.omega0,
                                                           my_pulse.n,
                                                           number_z)

            # --------------------------------------------------------------------
            #  Step 6. Calculate the Jacobian weighted vector spectrum
            # --------------------------------------------------------------------
            # Add Jacobian
            gfun.scalar_scalar_multiply_complex[b_num, d_num](cuda_coef,
                                                              cuda_jacob,
                                                              cuda_spec_scalar,
                                                              number_z
                                                              )

            # Get the vector field
            gfun.scalar_vector_multiply_complex[b_num, d_num](cuda_spec_scalar,
                                                              my_pulse.polar,
                                                              cuda_spec_vec,
                                                              number_z)

            # --------------------------------------------------------------------
            #  Step 7. Forward propagation
            # --------------------------------------------------------------------
            grating_idx = 0
            for crystal_idx in range(crystal_num):
                my_crystal = crystal_list[crystal_idx]
                if my_crystal.type == "Transmissive Grating":
                    # Get the intersection point on the first grating from the initial point
                    gfun.get_intersection_point[b_num, d_num](cuda_remain_path,
                                                              cuda_intersect,
                                                              cuda_kin_grid,
                                                              cuda_klen_grid,
                                                              cuda_remain_path,
                                                              cuda_intersect,
                                                              my_crystal.surface_point,
                                                              my_crystal.normal,
                                                              number_z)

                    # Diffracted by the first grating
                    gfun.get_square_grating_effect_non_zero[b_num, d_num](cuda_kin_grid,
                                                                          cuda_spec_vec,
                                                                          cuda_klen_grid,
                                                                          cuda_kin_grid,
                                                                          my_crystal.h,
                                                                          my_crystal.n,
                                                                          my_crystal.ab_ratio,
                                                                          my_crystal.thick_vec,
                                                                          grating_orders[grating_idx],
                                                                          my_crystal.base_wave_vector,
                                                                          number_z)
                    # Update the grating_idx
                    grating_idx += 1

                if my_crystal.type == "Crystal: Bragg Reflection":
                    # Get the intersection point from the previous intersection point
                    gfun.get_intersection_point[b_num, d_num](cuda_remain_path,
                                                              cuda_intersect,
                                                              cuda_kin_grid,
                                                              cuda_klen_grid,
                                                              cuda_remain_path,
                                                              cuda_intersect,
                                                              my_crystal.surface_point,
                                                              my_crystal.normal,
                                                              number_z)

                    # Get the reflectivity
                    gfun.get_bragg_reflection[b_num, d_num](cuda_reflect_sigma,
                                                            cuda_reflect_pi,
                                                            cuda_kin_grid,
                                                            cuda_spec_vec,
                                                            cuda_klen_grid,
                                                            cuda_kin_grid,
                                                            my_crystal.thickness,
                                                            my_crystal.h,
                                                            my_crystal.normal,
                                                            my_crystal.dot_hn,
                                                            my_crystal.h_square,
                                                            my_crystal.chi0,
                                                            my_crystal.chih_sigma,
                                                            my_crystal.chihbar_sigma,
                                                            my_crystal.chih_pi,
                                                            my_crystal.chihbar_pi,
                                                            number_z)
                    gfun.scalar_scalar_multiply_complex[b_num, d_num](cuda_reflect_sigma,
                                                                      cuda_reflect_total_sigma,
                                                                      cuda_reflect_total_sigma,
                                                                      number_z)
                    gfun.scalar_scalar_multiply_complex[b_num, d_num](cuda_reflect_pi,
                                                                      cuda_reflect_total_pi,
                                                                      cuda_reflect_total_pi,
                                                                      number_z)
                if my_crystal.type == "prism":
                    # Assume that the prism do nothing but add an angular deviation
                    gfun.add_vector[b_num, d_num](cuda_kin_grid,
                                                  cuda_kin_grid,
                                                  my_crystal.additional_wavevector,
                                                  number_z)

                    # Update the wave number
                    gfun.get_vector_length[b_num, d_num](cuda_klen_grid,
                                                         cuda_kin_grid,
                                                         3,
                                                         number_z)

            # --------------------------------------------------------------------
            #  Step 8. Get the propagation phase
            # --------------------------------------------------------------------
            gfun.get_final_point[b_num, d_num](cuda_final_points,
                                               cuda_intersect,
                                               cuda_kin_grid,
                                               cuda_klen_grid,
                                               cuda_remain_path,
                                               number_z)

            # Get the propagational phase from the inital phase.
            gfun.get_relative_spatial_phase[b_num, d_num](cuda_phase,
                                                          cuda_final_points,
                                                          observation,
                                                          cuda_kin_grid,
                                                          pulse_k0_final,
                                                          number_z)
            # gfun.get_spatial_phase[b_num, d_num](cuda_phase,
            #                                      cuda_final_points,
            #                                      observation,
            #                                      cuda_kin_grid,
            #                                      pulse_k0_final,
            #                                      number_z)

            # Add the phase
            gfun.scalar_vector_elementwise_multiply_complex[b_num, d_num](cuda_phase,
                                                                          cuda_spec_vec,
                                                                          cuda_spec_vec,
                                                                          number_z)

            # --------------------------------------------------------------------
            #  Step 9. Goes from the reciprocal space to the real space
            # --------------------------------------------------------------------
            # Save the result to the total reflect
            gfun.vector_expansion[b_num, d_num](cuda_spec_vec,
                                                cuda_spec_x,
                                                cuda_spec_y,
                                                cuda_spec_z,
                                                number_z)
            # Save the spec of the field
            gfun.fill_column_complex_fftshift[b_num, d_num](cuda_x_spec_2d,
                                                            cuda_spec_x,
                                                            y_idx,
                                                            idx_start_1,
                                                            num1,
                                                            idx_start_2,
                                                            num2)
            gfun.fill_column_complex_fftshift[b_num, d_num](cuda_y_spec_2d,
                                                            cuda_spec_y,
                                                            y_idx,
                                                            idx_start_1,
                                                            num1,
                                                            idx_start_2,
                                                            num2)
            gfun.fill_column_complex_fftshift[b_num, d_num](cuda_z_spec_2d,
                                                            cuda_spec_z,
                                                            y_idx,
                                                            idx_start_1,
                                                            num1,
                                                            idx_start_2,
                                                            num2)

            # Take the fourier transformation
            cufft.ifft(cuda_spec_x, cuda_x_field)
            cufft.ifft(cuda_spec_y, cuda_y_field)
            cufft.ifft(cuda_spec_z, cuda_z_field)

            # Update the data holder
            gfun.fill_column_complex_fftshift[b_num, d_num](cuda_x_field_2d,
                                                            cuda_x_field,
                                                            y_idx,
                                                            idx_start_1,
                                                            num1,
                                                            idx_start_2,
                                                            num2)

            gfun.fill_column_complex_fftshift[b_num, d_num](cuda_y_field_2d,
                                                            cuda_y_field,
                                                            y_idx,
                                                            idx_start_1,
                                                            num1,
                                                            idx_start_2,
                                                            num2)

            gfun.fill_column_complex_fftshift[b_num, d_num](cuda_z_field_2d,
                                                            cuda_z_field,
                                                            y_idx,
                                                            idx_start_1,
                                                            num1,
                                                            idx_start_2,
                                                            num2)
        # """
        ###################################################################################################
        #                                  Finish
        ###################################################################################################
        # Move the 2D slices back to the host and then save them to the variables
        cuda_x_field_2d.to_host()
        cuda_y_field_2d.to_host()
        cuda_z_field_2d.to_host()
        cuda_x_spec_2d.to_host()
        cuda_y_spec_2d.to_host()
        cuda_z_spec_2d.to_host()

        # Update the 3D variables.
        efield_3d[x_idx, :, :, 0] = x_field_2d
        efield_3d[x_idx, :, :, 1] = y_field_2d
        efield_3d[x_idx, :, :, 2] = z_field_2d
        efield_spec_3d[x_idx, :, :, 0] = x_spec_2d
        efield_spec_3d[x_idx, :, :, 1] = y_spec_2d
        efield_spec_3d[x_idx, :, :, 2] = z_spec_2d

        # Move the variables back to the GPU
        cuda_x_field_2d = cuda.to_device(x_field_2d)
        cuda_y_field_2d = cuda.to_device(y_field_2d)
        cuda_z_field_2d = cuda.to_device(z_field_2d)
        cuda_x_spec_2d = cuda.to_device(x_spec_2d)
        cuda_y_spec_2d = cuda.to_device(y_spec_2d)
        cuda_z_spec_2d = cuda.to_device(z_spec_2d)

    # Move the arrays back to the device for debugging.
    cuda_final_points.to_host()
    cuda_remain_path.to_host()
    cuda_spec_scalar.to_host()
    cuda_intersect.to_host()
    cuda_phase.to_host()

    cuda_kin_grid.to_host()
    cuda_klen_grid.to_host()

    cuda_reflect_pi.to_host()
    cuda_reflect_sigma.to_host()
    cuda_reflect_total_pi.to_host()
    cuda_reflect_total_sigma.to_host()

    cuda_coef.to_host()
    cuda_spec_x.to_host()
    cuda_spec_y.to_host()
    cuda_spec_z.to_host()
    cuda_spec_vec.to_host()

    cuda_x_field.to_host()
    cuda_y_field.to_host()
    cuda_z_field.to_host()

    # Create result dictionary

    check_dict = {"intersect_points": intersect_points,
                  "component_final_points": component_final_points,
                  "remaining_length": remaining_length,
                  "phase_grid": phase_grid,
                  "final_point": component_final_points,
                  "scalar_spec": scalar_spec_holder,
                  "coef_grid": coef_grid,
                  "jacob_grid": jacob_grid,
                  "reflectivity_pi": reflect_pi,
                  "reflectivity_sigma": reflect_sigma,
                  "reflectivity_pi_tot": reflect_total_pi,
                  "reflectivity_sigma_tot": reflect_total_sigma,
                  "x_field": x_field_holder,
                  "y_field": y_field_holder,
                  "z_field": z_field_holder,
                  }

    result_3d_dict = {"efield_3d": efield_3d,
                      "efield_spec_3d": efield_spec_3d}

    result_2d_dict = {"x_field_2d": x_field_2d,
                      "y_field_2d": y_field_2d,
                      "z_field_2d": z_field_2d,
                      "x_spec_2d": x_spec_2d,
                      "y_spec_2d": y_spec_2d,
                      "z_spec_2d": z_spec_2d}

    return result_3d_dict, result_2d_dict, check_dict
