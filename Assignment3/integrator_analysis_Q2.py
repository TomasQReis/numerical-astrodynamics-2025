"""
Copyright (c) 2010-2020, Delft University of Technology
All rigths reserved

This file is part of the Tudat. Redistribution and use in source and
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.
"""

import os

from integrator_analysis_helper_functions import *

current_directory = os.getcwd()

# Load spice kernels.
spice.load_standard_kernels()
spice.load_kernel(current_directory + "/juice_mat_crema_5_1_150lb_v01.bsp")

# Create the bodies for the numerical simulation
bodies = create_bodies()

# Define list of step size for integrator to take
step_sizes = ...

# Iterate over phases
for current_phase in range(len(central_bodies_per_phase)):

    # Create initial state and time
    current_phase_start_time = initial_times_per_phase[current_phase]
    current_phase_end_time = (
        current_phase_start_time + propagation_times_per_phase[current_phase]
    )

    termination_settings = propagation_setup.propagator.time_termination(
        current_phase_end_time
    )

    # Define current centra
    current_central_body = central_bodies_per_phase[current_phase]

    # Retrieve JUICE initial state
    initial_state = spice.get_body_cartesian_state_at_epoch(
        target_body_name="JUICE",
        observer_body_name=current_central_body,
        reference_frame_name=global_frame_orientation,
        aberration_corrections="NONE",
        ephemeris_time=current_phase_start_time,
    )

    # Retrieve acceleration settings without perturbations
    acceleration_models = get_perturbed_accelerations(current_central_body, bodies)

    # Save propagation results for each time step into a list, for analysis after all propagations are done
    propagation_results_per_step_size = list()

    # Iterate over step size
    for step_size in step_sizes:

        # Define integrator settings
        integrator_settings = get_fixed_step_size_integrator_settings(
            current_phase_start_time, step_size
        )

        # Define propagator settings
        propagator_settings = ...

        # Propagate dynamics
        dynamics_simulator = ...
        state_history = dynamics_simulator.propagation_results.state_history
        propagation_results_per_step_size.append(state_history)

        # Write results to files
        file_output_identifier = (
            "Q2_step_size_" + str(step_size) + "_phase_index" + str(current_phase)
        )
        write_propagation_results_and_analytical_difference_to_file(
            dynamics_simulator,
            file_output_identifier,
            bodies.get_body(current_central_body).gravitational_parameter,
        )
