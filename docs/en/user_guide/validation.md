# Validation

The framework has been validated against several reference models and real-world data:

- **SIMBA#**: Commercial biogas simulation software (ifak e.V.).
- **[ADM1F](https://github.com/lanl/ADM1F)**: LANL's Fortran ADM1 implementation.
- **Real plant data**: Multiple agricultural biogas plants.

## Comparison with SIMBA#
The implementation of the ADM1da model has been extensively compared with SIMBA# to ensure the correctness of stoichiometry, kinetics, and pH calculation.

## Comparison with ADM1F
The core ADM1 code was validated against the ADM1F implementation to confirm the mathematical consistency of the ODE solvers and process rates.

## Validation with Real Data
PyADM1ODE is continuously validated with data from real plant operations, particularly regarding gas production and substrate degradation rates under practical conditions.
