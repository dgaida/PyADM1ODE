# Core ADM1 Model

Core ADM1 Implementation

This module contains the core implementation of the Anaerobic Digestion Model No. 1
(ADM1) as a pure ODE system without DAEs, specifically adapted for agricultural
biogas plants.

Modules:

    adm1: Main ADM1 class implementing the complete ODE system with 37 state variables,
          including methods for creating influent streams, calculating gas production,
          and managing simulation state.

    adm_params: Static parameter class providing all stoichiometric, kinetic, and
               physical-chemical parameters for ADM1, including temperature-dependent
               parameters and pH inhibition factors.

    adm_equations: Process rate equations, inhibition functions, and biochemical
                  transformations used in the ADM1 model, separated for clarity
                  and easier modification.

    solver: ODE solver wrapper providing interface to scipy solvers with
           appropriate settings for stiff systems (BDF method), time step
           management, and result handling.

Example:

```python
    >>> from pyadm1.core import ADM1, ADMParams, create_solver
    >>> from pyadm1.substrates import Feedstock
    >>>
    >>> # Create model
    >>> feedstock = Feedstock(feeding_freq=48)
    >>> adm1 = ADM1(feedstock, V_liq=2000, T_ad=308.15)
    >>>
    >>> # Get parameters
    >>> params = ADMParams.get_all_params(R=0.08314, T_base=298.15, T_ad=308.15)
    >>>
    >>> # Create custom solver
    >>> solver = create_solver(method='BDF', rtol=1e-7)
```

## Base Classes

- [ADM1](#adm1)
- [ADMParams](#admparams)
- [AcidBaseKinetics](#acidbasekinetics)
- [AdaptiveODESolver](#adaptiveodesolver)
- [BiochemicalProcesses](#biochemicalprocesses)
- [GasTransfer](#gastransfer)
- [InhibitionFunctions](#inhibitionfunctions)
- [ODESolver](#odesolver)
- [ProcessRates](#processrates)
- [SolverConfig](#solverconfig)

### ADM1

```python
from pyadm1.core import ADM1
```

Main class implementing ADM1 as pure ODE system.

This class manages the ADM1 state, parameters, and provides methods for
simulation including influent stream creation, gas production calculation,
and state tracking.

Attributes:

    V_liq: Liquid volume [m³]
    T_ad: Operating temperature [K]
    feedstock: Feedstock object for substrate management

Example:

```python
    >>> feedstock = Feedstock(feeding_freq=48)
    >>> adm1 = ADM1(feedstock, V_liq=2000, T_ad=308.15)
    >>> adm1.create_influent([15, 10, 0, 0, 0, 0, 0, 0, 0, 0], 0)
```

**Signature:**

```python
ADM1(
    feedstock,
    V_liq=1977.0,
    V_gas=304.0,
    T_ad=308.15
)
```

**Methods:**

#### `ADM1_ODE()`

```python
ADM1_ODE(t, state_zero)
```

Calculate derivatives for ADM1 ODE system.

This is the main ODE function that computes dy/dt for all 37 state
variables. Uses process rate equations and stoichiometric relationships.

Args:

    t: Current time [days] (not used, system is autonomous)
    state_zero: Current ADM1 state vector (37 elements)

Returns:

    Tuple of 37 derivatives (dy/dt)

Note:

    This method is called by the ODE solver and should not be called
    directly by users.

#### `calc_gas()`

```python
calc_gas(pi_Sh2, pi_Sch4, pi_Sco2, pTOTAL)
```

Calculate biogas production rates from partial pressures.

Uses the ideal gas law and Henry's constants to calculate gas flow rates
from the gas phase partial pressures.

Args:

    pi_Sh2: Hydrogen partial pressure [bar]
    pi_Sch4: Methane partial pressure [bar]
    pi_Sco2: CO2 partial pressure [bar]
    pTOTAL: Total gas pressure [bar]

Returns:

    Tuple containing:
        - q_gas: Total biogas flow rate [m³/d]
        - q_ch4: Methane flow rate [m³/d]
        - q_co2: CO2 flow rate [m³/d]
        - p_gas: Total gas partial pressure (excl. H2O) [bar]

Example:

```python
    >>> q_gas, q_ch4, q_co2, p_gas = adm1.calc_gas(5e-6, 0.55, 0.42, 0.98)
    >>> print(f"Biogas: {q_gas:.1f} m³/d, Methane: {q_ch4:.1f} m³/d")
```

#### `clear_calibration_parameters()`

```python
clear_calibration_parameters()
```

Clear all calibration parameters and revert to substrate-dependent calculations.

Example:

```python
    >>> adm1.clear_calibration_parameters()
```

#### `create_influent()`

```python
create_influent(Q, i)
```

Create ADM1 input stream from volumetric flow rates.

Calculates the ADM1 influent state by mixing substrate streams according
to their volumetric flow rates. The resulting influent composition is
stored internally for use in ODE calculations.

Args:

    Q: Volumetric flow rates for each substrate [m³/d]
       Length must equal number of substrates in feedstock
    i: Time step index for accessing influent dataframe

Example:

```python
    >>> adm1.create_influent([15, 10, 0, 0, 0, 0, 0, 0, 0, 0], 0)
```

#### `get_calibration_parameters()`

```python
get_calibration_parameters()
```

Get currently set calibration parameters.

Returns:

    dict: Current calibration parameters as {param_name: value}.

Example:

```python
    >>> params = adm1.get_calibration_parameters()
    >>> print(params)
    {'k_dis': 0.55, 'Y_su': 0.105}
```

#### `print_params_at_current_state()`

```python
print_params_at_current_state(state_ADM1xp)
```

Calculate and print process parameters from current state.

Computes and displays key process indicators including pH, VFA, TAC,
and gas production rates. Also stores values in tracking lists.

Args:

    state_ADM1xp: Current ADM1 state vector (37 elements)

Example:

```python
    >>> adm1.print_params_at_current_state(state_vector)
    pH(lib) = [7.2, 7.3]
    FOS/TAC = [0.25, 0.26]
    ...
```

#### `resume_from_broken_simulation()`

```python
resume_from_broken_simulation(Q_CH4)
```

#### `save_final_state_in_csv()`

```python
save_final_state_in_csv(simulate_results, filename='digester_final.csv')
```

Save final ADM1 state vector to CSV file.

Exports only the last state from simulation results, which can be used
as initial state for subsequent simulations.

Args:

    simulate_results: List of ADM1 state vectors from simulation
    filename: Output CSV filename

Example:

```python
    >>> results = [[0.01]*37, [0.02]*37, [0.03]*37]
    >>> adm1.save_final_state_in_csv(results, 'final_state.csv')
```

#### `set_calibration_parameters()`

```python
set_calibration_parameters(parameters)
```

Set calibration parameters that override substrate-dependent calculations.

Args:

    parameters: Parameter values as {param_name: value}.

Example:

```python
    >>> adm1.set_calibration_parameters({
    ...     'k_dis': 0.55,
    ...     'k_hyd_ch': 11.0,
    ...     'Y_su': 0.105
    ... })
```

**Attributes:**

- V_liq: Liquid volume [m³]
- T_ad: Operating temperature [K]
- feedstock: Feedstock object for substrate management


--------------------------------------
### ADMParams

```python
from pyadm1.core import ADMParams
```

Static class containing ADM1 model parameters.

**Signature:**

```python
ADMParams(
    args,
    kwargs
)
```

**Methods:**

#### `getADMgasparams()`

```python
getADMgasparams(R, T_base, T_ad)
```

Get gas phase parameters including Henry constants.

Parameters
----------
R : float
    Gas constant [bar·M^-1·K^-1]
T_base : float
    Base temperature [K]
T_ad : float
    Digester temperature [K]

Returns
-------
Tuple[float, float, float, float, float, float]
    p_gas_h2o, k_p, k_L_a, K_H_co2, K_H_ch4, K_H_h2

#### `getADMinhibitionparams()`

```python
getADMinhibitionparams()
```

Get pH inhibition parameters.

Returns
-------
Tuple[float, float, float, float, float, float]
    K_pH_aa, nn_aa, K_pH_ac, n_ac, K_pH_h2, n_h2

#### `getADMparams()`

```python
getADMparams(R, T_base, T_ad)
```

Get all ADM1 stoichiometric and kinetic parameters.

Parameters
----------
R : float
    Gas constant [bar·M^-1·K^-1]
T_base : float
    Base temperature [K]
T_ad : float
    Digester temperature [K]

Returns
-------
Tuple[float, ...]
    All ADM1 parameters (87 values)


--------------------------------------
### AcidBaseKinetics

```python
from pyadm1.core import AcidBaseKinetics
```

Acid-base equilibrium kinetics for ADM1.

**Signature:**

```python
AcidBaseKinetics(
    args,
    kwargs
)
```

**Methods:**

#### `acid_base_rate()`

```python
acid_base_rate(k_AB, S_ion, S_H_ion, K_a, S_undissociated)
```

Calculate acid-base reaction rate.

Implements: S_ion + H+ <-> S_undissociated

Args:

    k_AB: Acid-base kinetic constant [M^-1·d^-1]
    S_ion: Ionized form concentration [M or kg COD/m³]
    S_H_ion: Hydrogen ion concentration [M]
    K_a: Acid dissociation constant [M]
    S_undissociated: Undissociated form concentration [M or kg COD/m³]

Returns:

    Acid-base reaction rate [M/d or kg COD/(m³·d)]


--------------------------------------
### AdaptiveODESolver

```python
from pyadm1.core import AdaptiveODESolver
```

Adaptive ODE solver that adjusts tolerances based on solution behavior.

Monitors the solution and can tighten tolerances if instabilities are
detected, or relax them for faster computation when solution is smooth.

**Signature:**

```python
AdaptiveODESolver(
    config=None,
    adaptive=True,
    min_rtol=1e-08,
    max_rtol=0.0001
)
```

**Methods:**

#### `solve()`

```python
solve(fun, t_span, y0, t_eval=None, dense_output=False)
```

Solve with adaptive tolerance adjustment.

Same interface as parent class but monitors solution quality.

#### `solve_sequential()`

```python
solve_sequential(fun, t_points, y0)
```

Solve ODE system sequentially through multiple time points.

Useful for simulations where conditions change at specific times
(e.g., substrate feed changes).

Args:

    fun: Right-hand side of ODE system
    t_points: List of time points [days]
    y0: Initial state vector

Returns:

    List of state vectors at each time point

#### `solve_to_steady_state()`

```python
solve_to_steady_state(fun, y0, max_time=1000.0, steady_state_tol=1e-06, check_interval=10.0)
```

Integrate until steady state is reached or max time exceeded.

Args:

    fun: Right-hand side of ODE system
    y0: Initial state vector
    max_time: Maximum integration time [days]
    steady_state_tol: Tolerance for steady state detection
    check_interval: Interval for checking steady state [days]

Returns:

    Tuple of (final_state, final_time, converged)
    - final_state: State vector at end of integration
    - final_time: Time at end of integration [days]
    - converged: True if steady state was reached


--------------------------------------
### BiochemicalProcesses

```python
from pyadm1.core import BiochemicalProcesses
```

Combined biochemical process calculations for ADM1.

This class orchestrates the calculation of all process rates including
inhibition factors and stoichiometric relationships.

**Signature:**

```python
BiochemicalProcesses(
    args,
    kwargs
)
```

**Methods:**

#### `calculate_acid_base_rates()`

```python
calculate_acid_base_rates(state, acid_base_params)
```

Calculate acid-base reaction rates for ODE implementation.

Args:

    state: ADM1 state vector (37 elements)
    acid_base_params: Dictionary containing K_a and k_AB values

Returns:

    Tuple of 6 acid-base rates (Rho_A_4 through Rho_A_11)

#### `calculate_gas_transfer_rates()`

```python
calculate_gas_transfer_rates(state, gas_params, RT, V_liq, V_gas)
```

Calculate gas-liquid transfer and gas outlet rates.

Args:

    state: ADM1 state vector (37 elements)
    gas_params: Dictionary containing k_L_a, K_H constants, k_p
    RT: Gas constant × temperature [bar·m³/kmol]
    V_liq: Liquid volume [m³]
    V_gas: Gas volume [m³]

Returns:

    Tuple of 4 rates (Rho_T_8, Rho_T_9, Rho_T_10, Rho_T_11)

#### `calculate_inhibition_factors()`

```python
calculate_inhibition_factors(S_H_ion, S_h2, S_nh4_ion, S_nh3, K_pH_aa, nn_aa, K_pH_ac, n_ac, K_pH_h2, n_h2, K_S_IN, K_I_h2_fa, K_I_h2_c4, K_I_h2_pro, K_I_nh3)
```

Calculate all inhibition factors for ADM1 processes.

Args:

    S_H_ion: Hydrogen ion concentration [M]
    S_h2: Hydrogen gas concentration [kg COD/m³]
    S_nh4_ion: Ammonium concentration [M]
    S_nh3: Free ammonia concentration [M]
    K_pH_aa: pH inhibition constant for amino acid degraders [M]
    nn_aa: Hill coefficient for aa pH inhibition [-]
    K_pH_ac: pH inhibition constant for acetate degraders [M]
    n_ac: Hill coefficient for ac pH inhibition [-]
    K_pH_h2: pH inhibition constant for hydrogen degraders [M]
    n_h2: Hill coefficient for h2 pH inhibition [-]
    K_S_IN: Nitrogen half-saturation constant [M]
    K_I_h2_fa: H2 inhibition constant for LCFA degraders [kg COD/m³]
    K_I_h2_c4: H2 inhibition constant for C4 degraders [kg COD/m³]
    K_I_h2_pro: H2 inhibition constant for propionate degraders [kg COD/m³]
    K_I_nh3: Ammonia inhibition constant [M]

Returns:

    Tuple of inhibition factors (I_pH_aa, I_pH_ac, I_pH_h2, I_IN_lim,
    I_h2_fa, I_h2_c4, I_h2_pro, I_nh3, I_5 through I_12)

#### `calculate_process_rates()`

```python
calculate_process_rates(state, inhibitions, kinetic_params, substrate_params, hydro_factor=1.0)
```

Calculate all 19 biochemical process rates for ADM1.

Args:

    state: ADM1 state vector (37 elements)
    inhibitions: Tuple of inhibition factors from calculate_inhibition_factors
    kinetic_params: Dictionary of kinetic parameters (k_m, K_S, k_dec, etc.)
    substrate_params: Dictionary of substrate-dependent parameters
        (k_dis, k_hyd_ch, k_hyd_pr, k_hyd_li)
    hydro_factor: Optional TS-dependent hydrolysis factor [-]

Returns:

    Tuple of 19 process rates (Rho_1 through Rho_19)


--------------------------------------
### GasTransfer

```python
from pyadm1.core import GasTransfer
```

Gas-liquid transfer and gas outlet calculations.

**Signature:**

```python
GasTransfer(
    args,
    kwargs
)
```

**Methods:**

#### `gas_outlet_rate()`

```python
gas_outlet_rate(k_p, p_total, p_ext, V_liq, V_gas)
```

Calculate gas outlet flow rate.

Args:

    k_p: Gas outlet friction coefficient [m³/(m³·d·bar)]
    p_total: Total gas pressure [bar]
    p_ext: External pressure [bar]
    V_liq: Liquid volume [m³]
    V_gas: Gas volume [m³]

Returns:

    Gas outlet rate [1/d]

#### `gas_transfer_rate()`

```python
gas_transfer_rate(k_L_a, S_gas_liq, p_gas, K_H, RT, COD_per_mole, V_liq, V_gas)
```

Calculate gas-liquid transfer rate.

Args:

    k_L_a: Gas-liquid transfer coefficient [1/d]
    S_gas_liq: Gas concentration in liquid [kg COD/m³ or kmol/m³]
    p_gas: Partial pressure in gas phase [bar]
    K_H: Henry's law constant [M/bar]
    RT: Gas constant × temperature [bar·m³/kmol]
    COD_per_mole: COD per mole of gas [kg COD/kmol]
    V_liq: Liquid volume [m³]
    V_gas: Gas volume [m³]

Returns:

    Gas transfer rate to gas phase [kg COD/(m³_gas·d) or kmol/(m³_gas·d)]


--------------------------------------
### InhibitionFunctions

```python
from pyadm1.core import InhibitionFunctions
```

pH and substrate inhibition functions for ADM1 processes.

**Signature:**

```python
InhibitionFunctions(
    args,
    kwargs
)
```

**Methods:**

#### `ammonia_inhibition()`

```python
ammonia_inhibition(S_nh3, K_I_nh3)
```

Calculate ammonia inhibition.

Args:

    S_nh3: Free ammonia concentration [M]
    K_I_nh3: Ammonia inhibition constant [M]

Returns:

    Inhibition factor between 0 and 1

#### `hydrogen_inhibition()`

```python
hydrogen_inhibition(S_h2, K_I_h2)
```

Calculate non-competitive hydrogen inhibition.

Args:

    S_h2: Hydrogen concentration [kg COD/m³]
    K_I_h2: Hydrogen inhibition constant [kg COD/m³]

Returns:

    Inhibition factor between 0 and 1

#### `nitrogen_limitation()`

```python
nitrogen_limitation(S_nh4_ion, S_nh3, K_S_IN)
```

Calculate inorganic nitrogen limitation factor.

Args:

    S_nh4_ion: Ammonium ion concentration [M]
    S_nh3: Free ammonia concentration [M]
    K_S_IN: Nitrogen half-saturation constant [M]

Returns:

    Limitation factor between 0 and 1

#### `pH_inhibition()`

```python
pH_inhibition(S_H_ion, K_pH, n)
```

Calculate pH inhibition factor.

Args:

    S_H_ion: Hydrogen ion concentration [M]
    K_pH: pH inhibition constant [M]
    n: Hill coefficient for pH inhibition [-]

Returns:

    Inhibition factor between 0 and 1

#### `substrate_inhibition()`

```python
substrate_inhibition(S, K_S)
```

Calculate Monod substrate limitation factor.

Args:

    S: Substrate concentration [kg COD/m³]
    K_S: Half-saturation constant [kg COD/m³]

Returns:

    Limitation factor between 0 and 1


--------------------------------------
### ODESolver

```python
from pyadm1.core import ODESolver
```

ODE solver wrapper for ADM1 system.

Provides a clean interface to scipy's solve_ivp with appropriate settings
for stiff biogas process ODEs. Uses BDF (Backward Differentiation Formula)
method which is suitable for stiff systems.

Example:

```python
    >>> def ode_func(t, y):
    ...     return [-0.5 * y[0], 0.5 * y[0] - 0.1 * y[1]]
    >>> solver = ODESolver()
    >>> result = solver.solve(ode_func, [0, 10], [1.0, 0.0])
    >>> print(result.y[:, -1])  # Final state
```

**Signature:**

```python
ODESolver(
    config=None
)
```

**Methods:**

#### `solve()`

```python
solve(fun, t_span, y0, t_eval=None, dense_output=False)
```

Solve ODE system over time span.

Args:

    fun: Right-hand side of ODE system dy/dt = fun(t, y)
    t_span: Integration time span (t_start, t_end) [days]
    y0: Initial state vector
    t_eval: Times at which to store solution. If None, uses automatic
        time points with 0.05 day resolution
    dense_output: If True, returns continuous solution object

Returns:

    OdeResult object with solution (from scipy.integrate.solve_ivp)

Raises:

    RuntimeError: If integration fails

#### `solve_sequential()`

```python
solve_sequential(fun, t_points, y0)
```

Solve ODE system sequentially through multiple time points.

Useful for simulations where conditions change at specific times
(e.g., substrate feed changes).

Args:

    fun: Right-hand side of ODE system
    t_points: List of time points [days]
    y0: Initial state vector

Returns:

    List of state vectors at each time point

#### `solve_to_steady_state()`

```python
solve_to_steady_state(fun, y0, max_time=1000.0, steady_state_tol=1e-06, check_interval=10.0)
```

Integrate until steady state is reached or max time exceeded.

Args:

    fun: Right-hand side of ODE system
    y0: Initial state vector
    max_time: Maximum integration time [days]
    steady_state_tol: Tolerance for steady state detection
    check_interval: Interval for checking steady state [days]

Returns:

    Tuple of (final_state, final_time, converged)
    - final_state: State vector at end of integration
    - final_time: Time at end of integration [days]
    - converged: True if steady state was reached


--------------------------------------
### ProcessRates

```python
from pyadm1.core import ProcessRates
```

Biochemical process rate calculations for ADM1.

**Signature:**

```python
ProcessRates(
    args,
    kwargs
)
```

**Methods:**

#### `decay_rate()`

```python
decay_rate(k_dec, X_biomass)
```

Calculate biomass decay rate.

Args:

    k_dec: Decay rate constant [1/d]
    X_biomass: Biomass concentration [kg COD/m³]

Returns:

    Decay rate [kg COD/(m³·d)]

#### `disintegration_rate()`

```python
disintegration_rate(k_dis, X_xc)
```

Calculate disintegration rate of composites.

Args:

    k_dis: Disintegration rate constant [1/d]
    X_xc: Composite concentration [kg COD/m³]

Returns:

    Disintegration rate [kg COD/(m³·d)]

#### `hydrolysis_rate()`

```python
hydrolysis_rate(k_hyd, X_substrate, hydro_factor=1.0)
```

Calculate hydrolysis rate (carbohydrates, proteins, lipids).

Args:

    k_hyd: Hydrolysis rate constant [1/d]
    X_substrate: Particulate substrate concentration [kg COD/m³]
    hydro_factor: Optional TS-dependent factor [-]

Returns:

    Hydrolysis rate [kg COD/(m³·d)]

#### `uptake_rate()`

```python
uptake_rate(k_m, S_substrate, K_S, X_biomass, I_combined)
```

Calculate Monod uptake rate with inhibition.

Args:

    k_m: Maximum uptake rate constant [1/d]
    S_substrate: Substrate concentration [kg COD/m³]
    K_S: Half-saturation constant [kg COD/m³]
    X_biomass: Biomass concentration [kg COD/m³]
    I_combined: Combined inhibition factor [-]

Returns:

    Uptake rate [kg COD/(m³·d)]


--------------------------------------
### SolverConfig

```python
from pyadm1.core import SolverConfig
```

Configuration for ODE solver.

Attributes:

    method: Integration method ('BDF' for stiff ODEs)
    rtol: Relative tolerance for solver
    atol: Absolute tolerance for solver
    min_step: Minimum allowed time step [days]
    max_step: Maximum allowed time step [days]
    first_step: Initial step size [days]

**Signature:**

```python
SolverConfig(
    method='BDF',
    rtol=1e-06,
    atol=1e-08,
    min_step=1e-06,
    max_step=0.1,
    first_step=None
)
```

**Methods:**

**Attributes:**

- method: Integration method ('BDF' for stiff ODEs)
- rtol: Relative tolerance for solver
- atol: Absolute tolerance for solver
- min_step: Minimum allowed time step [days]
- max_step: Maximum allowed time step [days]
- first_step: Initial step size [days]


--------------------------------------
