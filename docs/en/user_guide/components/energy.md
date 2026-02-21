# Energy Components

Components for energy generation and management.

## CHP (Combined Heat and Power)

Models a combined heat and power unit that consumes biogas and produces electricity and heat.

### Parameters

```python
configurator.add_chp(
    chp_id="chp1",
    P_el_nom=500.0,      # Nominal electrical power [kW]
    eta_el=0.40,         # Electrical efficiency
    eta_th=0.45          # Thermal efficiency
)
```

## Heating

Models a heating system for temperature control of the digester.

## Gas Storage

Models biogas storage with pressure management.

## Flare

Safety flare for excess gas combustion.

## Optimization Strategies
- Use waste heat for digester heating.
- Buffer gas during peak production.

## Performance Metrics
- Electrical efficiency.
- Heat utilization rate.

## Troubleshooting

### CHP Not Running
- Check gas availability and pressure.
