Core ADM1 Implementation
=========================

The core module provides the fundamental ADM1 (Anaerobic Digestion Model No. 1) implementation
as a pure ODE system without DAEs, specifically adapted for agricultural biogas plants.

ADM1 Class
----------

.. autoclass:: pyadm1.core.adm1.ADM1
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __weakref__

   Main class implementing the ADM1 model as a pure ODE system. This class manages the
   ADM1 state vector (37 variables), parameters, and provides methods for simulation
   including influent stream creation, gas production calculation, and state tracking.

   **Key Features:**

   - 37-state variable ODE system (no DAEs)
   - Substrate-dependent parameter calculation
   - Gas-liquid equilibrium
   - Acid-base chemistry
   - Biochemical process rates

   **State Variables (37 total):**

   Soluble Components (0-11):
      - S_su: Monosaccharides [kg COD/m³]
      - S_aa: Amino acids [kg COD/m³]
      - S_fa: Long chain fatty acids [kg COD/m³]
      - S_va: Valerate [kg COD/m³]
      - S_bu: Butyrate [kg COD/m³]
      - S_pro: Propionate [kg COD/m³]
      - S_ac: Acetate [kg COD/m³]
      - S_h2: Hydrogen gas [kg COD/m³]
      - S_ch4: Methane gas [kg COD/m³]
      - S_co2: Inorganic carbon [kmole C/m³]
      - S_nh4: Inorganic nitrogen [kmole N/m³]
      - S_I: Soluble inerts [kg COD/m³]

   Particulate Components (12-24):
      - X_xc: Composites [kg COD/m³]
      - X_ch: Carbohydrates [kg COD/m³]
      - X_pr: Proteins [kg COD/m³]
      - X_li: Lipids [kg COD/m³]
      - X_su through X_h2: Seven bacterial populations [kg COD/m³]
      - X_I: Particulate inerts [kg COD/m³]
      - X_p: Particulate products [kg COD/m³]

   Acid-Base Components (25-32):
      - S_cation, S_anion: Ion concentrations [kmole/m³]
      - S_va_ion through S_ac_ion: VFA ion forms [kg COD/m³]
      - S_hco3_ion: Bicarbonate [kmole C/m³]
      - S_nh3: Ammonia [kmole N/m³]

   Gas Phase (33-36):
      - pi_Sh2: Hydrogen partial pressure [bar]
      - pi_Sch4: Methane partial pressure [bar]
      - pi_Sco2: CO2 partial pressure [bar]
      - pTOTAL: Total pressure [bar]

   **Example Usage:**

   .. code-block:: python

      from pyadm1.core import ADM1
      from pyadm1.substrates import Feedstock

      # Create model
      feedstock = Feedstock(feeding_freq=48)
      adm1 = ADM1(feedstock, V_liq=2000, T_ad=308.15)

      # Create influent stream
      Q = [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]  # m³/d
      adm1.create_influent(Q, 0)

      # Calculate gas production
      state = [0.01] * 37
      q_gas, q_ch4, q_co2, p_gas = adm1.calc_gas(
          state[33], state[34], state[35], state[36]
      )

ADM Parameters
--------------

.. autoclass:: pyadm1.core.adm_params.ADMParams
   :members:
   :undoc-members:
   :show-inheritance:

   Static class providing all ADM1 stoichiometric, kinetic, and physical-chemical
   parameters. Parameters include:

   **Stoichiometric Parameters:**

   - Nitrogen content (N_xc, N_I, N_aa, N_bac)
   - Carbon content (C_xc, C_sI, C_ch, C_pr, C_li, etc.)
   - Yield coefficients (Y_su, Y_aa, Y_fa, Y_c4, Y_pro, Y_ac, Y_h2)
   - Product distribution factors (f_h2_su, f_bu_su, f_pro_su, f_ac_su, etc.)

   **Kinetic Parameters:**

   - Maximum uptake rates (k_m_su, k_m_aa, k_m_fa, k_m_c4, k_m_pro, k_m_ac, k_m_h2)
   - Half-saturation constants (K_S_su, K_S_aa, K_S_fa, K_S_c4, K_S_pro, K_S_ac, K_S_h2)
   - Inhibition constants (K_I_h2_fa, K_I_h2_c4, K_I_h2_pro, K_I_nh3)
   - Decay rates (k_dec_X_su through k_dec_X_h2)

   **Physical-Chemical Parameters:**

   - Acid dissociation constants (K_a_va, K_a_bu, K_a_pro, K_a_ac, K_a_co2, K_a_IN)
   - Acid-base kinetic rates (k_A_B_va, k_A_B_bu, k_A_B_pro, k_A_B_ac, k_A_B_co2, k_A_B_IN)
   - Henry's law constants (K_H_co2, K_H_ch4, K_H_h2)
   - Gas transfer coefficient (k_L_a)
   - pH limits for inhibition

   **Temperature Dependency:**

   Many parameters are temperature-dependent and use Arrhenius equations:

   .. math::

      K(T) = K(T_{base}) \cdot \exp\left(\frac{E_a}{R}\left(\frac{1}{T_{base}} - \frac{1}{T}\right)\right)

Process Equations
-----------------

Inhibition Functions
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pyadm1.core.adm_equations.InhibitionFunctions
   :members:
   :undoc-members:
   :show-inheritance:

   pH and substrate inhibition functions for ADM1 processes.

   **pH Inhibition:**

   .. math::

      I_{pH} = \frac{K_{pH}^n}{[H^+]^n + K_{pH}^n}

   **Substrate Limitation (Monod):**

   .. math::

      I_S = \frac{S}{K_S + S}

   **Hydrogen Inhibition:**

   .. math::

      I_{H_2} = \frac{1}{1 + \frac{S_{H_2}}{K_{I,H_2}}}

Process Rates
~~~~~~~~~~~~~

.. autoclass:: pyadm1.core.adm_equations.ProcessRates
   :members:
   :undoc-members:
   :show-inheritance:

   Biochemical process rate calculations including:

   - Disintegration of composites
   - Hydrolysis of carbohydrates, proteins, lipids
   - Monod uptake kinetics with inhibition
   - Biomass decay

   **General Rate Equation:**

   .. math::

      \rho = k_m \cdot \frac{S}{K_S + S} \cdot X \cdot I_{combined}

   where :math:`I_{combined}` includes pH, hydrogen, ammonia, and nitrogen inhibition factors.

Acid-Base Kinetics
~~~~~~~~~~~~~~~~~~

.. autoclass:: pyadm1.core.adm_equations.AcidBaseKinetics
   :members:
   :undoc-members:
   :show-inheritance:

   Acid-base equilibrium kinetics for VFAs and inorganic species.

   **Acid-Base Reaction:**

   .. math::

      S^- + H^+ \leftrightarrow S_{undissociated}

   **Rate Equation:**

   .. math::

      \rho_{AB} = k_{AB} \cdot ([S^-] \cdot [H^+] - K_a \cdot [S_{undissociated}])

Gas Transfer
~~~~~~~~~~~~

.. autoclass:: pyadm1.core.adm_equations.GasTransfer
   :members:
   :undoc-members:
   :show-inheritance:

   Gas-liquid transfer and gas outlet calculations.

   **Gas Transfer Rate:**

   .. math::

      \rho_{T,gas} = k_{L}a \cdot (S_{gas,liq} - \frac{p_{gas}}{K_H \cdot R \cdot T}) \cdot \frac{V_{liq}}{V_{gas}}

   **Gas Outlet:**

   .. math::

      \rho_{T,out} = k_p \cdot (p_{total} - p_{ext}) \cdot \frac{V_{liq}}{V_{gas}}

Biochemical Processes
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pyadm1.core.adm_equations.BiochemicalProcesses
   :members:
   :undoc-members:
   :show-inheritance:

   Combined biochemical process calculations orchestrating all 19 process rates:

   1. Disintegration of composites
   2. Hydrolysis of carbohydrates
   3. Hydrolysis of proteins
   4. Hydrolysis of lipids
   5. Uptake of sugars
   6. Uptake of amino acids
   7. Uptake of LCFA
   8. Uptake of valerate
   9. Uptake of butyrate
   10. Uptake of propionate
   11. Uptake of acetate
   12. Uptake of hydrogen
   13-19. Decay of biomass (7 populations)

ODE Solver
----------

Solver Configuration
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pyadm1.core.solver.SolverConfig
   :members:
   :undoc-members:
   :show-inheritance:

   Configuration for ODE solver with parameters for:

   - Integration method (BDF recommended for stiff systems)
   - Tolerances (rtol, atol)
   - Step size constraints
   - Adaptive time stepping

ODE Solver
~~~~~~~~~~

.. autoclass:: pyadm1.core.solver.ODESolver
   :members:
   :undoc-members:
   :show-inheritance:

   ODE solver wrapper for ADM1 system using scipy's solve_ivp with BDF method
   for stiff equations.

   **Solver Methods:**

   - **BDF**: Backward Differentiation Formula (recommended for ADM1)
   - **Radau**: Implicit Runge-Kutta (alternative)
   - **LSODA**: Automatic stiffness detection

   **Example:**

   .. code-block:: python

      from pyadm1.core.solver import ODESolver, SolverConfig

      config = SolverConfig(
          method='BDF',
          rtol=1e-6,
          atol=1e-8,
          max_step=0.1
      )
      solver = ODESolver(config)

      # Solve ODE system
      result = solver.solve(
          fun=adm1.ADM1_ODE,
          t_span=[0, 30],
          y0=initial_state
      )

Adaptive Solver
~~~~~~~~~~~~~~~

.. autoclass:: pyadm1.core.solver.AdaptiveODESolver
   :members:
   :undoc-members:
   :show-inheritance:

   Adaptive solver that adjusts tolerances based on solution behavior.
   Useful for long simulations with varying dynamics.

Factory Function
~~~~~~~~~~~~~~~~

.. autofunction:: pyadm1.core.solver.create_solver

   Factory function to create solver instances:

   .. code-block:: python

      # Standard solver
      solver = create_solver(method='BDF', rtol=1e-7)

      # Adaptive solver
      solver = create_solver(
          method='BDF',
          adaptive=True,
          min_rtol=1e-8,
          max_rtol=1e-4
      )

Helper Functions
----------------

.. autofunction:: pyadm1.core.adm1.get_state_zero_from_initial_state

   Load initial ADM1 state vector from CSV file.

   **CSV Format:**

   Single row with 37 columns representing the complete ADM1 state vector
   in order: S_su, S_aa, S_fa, ..., pi_Sco2, pTOTAL

   **Example:**

   .. code-block:: python

      from pyadm1.core import get_state_zero_from_initial_state

      # Load steady-state initial condition
      state = get_state_zero_from_initial_state(
          'data/initial_states/digester_initial8.csv'
      )

See Also
--------

- :doc:`../user_guide/quickstart` for practical examples
- :doc:`components` for plant component integration
- :doc:`calibration` for parameter estimation

References
----------

.. [Batstone2002] Batstone, D.J., et al. (2002). Anaerobic Digestion Model No. 1
   (ADM1). IWA Task Group for Mathematical Modelling of Anaerobic Digestion Processes.

.. [Rosen2006] Rosen, C., et al. (2006). Benchmark Simulation Model No. 2 (BSM2).
   IWA Task Group on Benchmarking of Control Strategies for WWTPs.

.. [Gaida2014] Gaida, D. (2014). Dynamic real-time substrate feed optimization of
   anaerobic co-digestion plants. PhD thesis, Universiteit Leiden.

.. [Sadrimajd2021] Sadrimajd, P., et al. (2021). PyADM1: a Python implementation of
   Anaerobic Digestion Model No. 1. bioRxiv. DOI: 10.1101/2021.03.03.433746
