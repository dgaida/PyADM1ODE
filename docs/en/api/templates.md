# Templates

Plant Configuration Templates

Pre-defined biogas plant configurations for common layouts and designs.

Modules:

    single_stage: Single continuously stirred tank reactor (CSTR) with standard
                 peripheral equipment (feeding, CHP, heating), suitable for most
                 agricultural substrates with good degradability.

    two_stage: Two-stage system with separate hydrolysis and methanogenesis reactors,
              optimal for substrates with high lignocellulosic content or when
              process stability is critical, with independent temperature control.

    custom: Custom plant builder with guided configuration for specific requirements,
           providing interactive setup or programmatic configuration with validation
           at each step for complex plant designs.

Example:

```python
    >>> from pyadm1.configurator.templates import (
    ...     SingleStageTemplate,
    ...     TwoStageTemplate,
    ...     CustomPlantBuilder
    ... )
    >>> from pyadm1.substrates import Feedstock
    >>>
    >>> feedstock = Feedstock(feeding_freq=48)
    >>>
    >>> # Create single-stage plant
    >>> plant = SingleStageTemplate.create(
    ...     feedstock=feedstock,
    ...     V_liq=2000,
    ...     P_el_chp=500
    ... )
    >>>
    >>> # Create two-stage plant
    >>> plant = TwoStageTemplate.create(
    ...     feedstock=feedstock,
    ...     V_hydrolysis=500,
    ...     V_main=2000,
    ...     T_hydrolysis=318.15,
    ...     T_main=308.15
    ... )
```

