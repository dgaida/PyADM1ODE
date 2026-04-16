# Components Guide

PyADM1 uses a modular, component-based architecture. This guide covers all available components, their parameters, and usage patterns.

## Component Architecture

### Base Component Structure

All components inherit from the `Component` base class and implement:

```python
class Component(ABC):
    def __init__(self, component_id, component_type, name):
        """Initialize component with unique ID and type."""

    def initialize(self, initial_state):
        """Set initial state before simulation."""

    def step(self, t, dt, inputs):
        """Execute one simulation time step."""

    def to_dict(self):
        """Serialize to dictionary for JSON export."""

    @classmethod
    def from_dict(cls, config):
        """Create component from configuration dictionary."""
```

## Component Overview

PyADM1 provides several categories of components:

### [Biological Components](biological.md)

Components for biological conversion processes:  
- **Digester**: Main fermenter implementing the ADM1 model for anaerobic digestion.  
- **Hydrolysis**: Pre-treatment tank for hydrolysis processes.  
- **Separator**: Solid-liquid separation for digestate processing.  

### [Energy Components](energy.md)

Components for energy generation and storage:  
- **CHP**: Combined Heat and Power unit for electricity and heat generation.  
- **Heating**: Heating system for temperature control.  
- **GasStorage**: Biogas storage with pressure management.  
- **Flare**: Safety flare for excess gas.  

### [Mechanical Components](mechanical.md)

Mechanical plant components:  
- **Pump**: Pumps for substrate transport and recirculation.  
- **Mixer**: Agitators for homogenization in the fermenter.  

### [Feeding Components](feeding.md)

Substrate handling and dosing:  
- **SubstrateStorage**: Substrate storage tanks with quality tracking.  
- **Feeder**: Automated dosing systems.  

## Connection Types

- **Liquid Connections**: Transfer digestate between fermenters.  
- **Gas Connections**: Transfer biogas from storage to CHP.  
- **Heat Connections**: Transfer waste heat from CHP to heating systems.  

## Auto-Connection Helpers

```python
# Automatic gas routing: Digester → Storage → CHP → Flare
configurator.auto_connect_digester_to_chp("dig1", "chp1")

# Automatic heat routing: CHP → Heating
configurator.auto_connect_chp_to_heating("chp1", "heat1")
```
