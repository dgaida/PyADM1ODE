"""
PyADM1ODE MCP Server using FastMCP

Provides tools for biogas plant modeling: createBiogasPlant, addDigester, addCHP, add_connection, etc.

Requires:
    - fastmcp (pip install fastmcp)
    - pyadm1ode (repo in PYTHONPATH or install as package)
"""

from fastmcp import FastMCP
from typing import Optional
from pyadm1.configurator.plant_builder import BiogasPlant
from pyadm1.components.biological.digester import Digester
from pyadm1.components.energy.chp import CHP
from pyadm1.configurator.connection_manager import Connection

mcp = FastMCP(
    name="PyADM1ODE MCP Server",
    instructions="Use these tools to build, simulate, and optimize agricultural biogas plants with ADM1.",
)

# Global plant registry (by name for demo; consider session/state for prod)
plant_db = {}


@mcp.tool
def createBiogasPlant(plant_name: str = "Biogas Plant") -> str:
    """Create a new biogas plant object.

    Args:
        plant_name (str, optional): Name for the biogas plant. Defaults to "Biogas Plant".

    Returns:
        str: The plant ID string to use in subsequent tool queries.

    Example:
        >>> createBiogasPlant("FarmAB Plant")
        'FarmAB Plant'
    """
    plant = BiogasPlant(plant_name)
    plant_db[plant_name] = plant
    return plant_name


@mcp.tool
def addDigester(
    plant_id: str,
    digester_id: str,
    V_liq: float = 1977.0,
    V_gas: float = 304.0,
    T_ad: float = 308.15,
    feedstock_obj: Optional[object] = None,
    name: Optional[str] = None,
) -> str:
    """Add a new Digester component to an existing biogas plant.

    Args:
        plant_id (str): ID of the biogas plant.
        digester_id (str): Unique identifier for the new digester.
        V_liq (float, optional): Liquid volume in m3. Defaults to 1977.0.
        V_gas (float, optional): Gas volume in m3. Defaults to 304.0.
        T_ad (float, optional): Operating temperature in K. Defaults to 308.15.
        feedstock_obj (object, optional): Feedstock object, if None, default will be used.
        name (str, optional): Name for the digester.

    Returns:
        str: digester_id

    Raises:
        ValueError: If plant_id not found in registry.

    Example:
        >>> addDigester("FarmAB Plant", "D1")
        'D1'
    """
    plant = plant_db.get(plant_id)
    if not plant:
        raise ValueError(f"Plant '{plant_id}' not found.")

    if feedstock_obj is None:
        from pyadm1.substrates.feedstock import Feedstock

        feedstock_obj = Feedstock(feeding_freq=48)

    digester = Digester(digester_id, feedstock_obj, V_liq=V_liq, V_gas=V_gas, T_ad=T_ad, name=name)
    plant.add_component(digester)
    return digester_id


@mcp.tool
def addCHP(
    plant_id: str,
    chp_id: str,
    P_el_nom: float = 500.0,
    eta_el: float = 0.40,
    eta_th: float = 0.45,
    name: Optional[str] = None,
) -> str:
    """Add a CHP unit (Combined Heat and Power) to the biogas plant.

    Args:
        plant_id (str): ID of the biogas plant.
        chp_id (str): Unique identifier for the CHP unit.
        P_el_nom (float, optional): Nominal electrical power (kW). Defaults to 500.0.
        eta_el (float, optional): Electrical efficiency. Defaults to 0.40.
        eta_th (float, optional): Thermal efficiency. Defaults to 0.45.
        name (str, optional): Name for the CHP unit.

    Returns:
        str: chp_id

    Raises:
        ValueError: If plant_id not found.

    Example:
        >>> addCHP("FarmAB Plant", "CHP1")
        'CHP1'
    """
    plant = plant_db.get(plant_id)
    if not plant:
        raise ValueError(f"Plant '{plant_id}' not found.")

    chp = CHP(chp_id, P_el_nom, eta_el, eta_th, name=name)
    plant.add_component(chp)
    return chp_id


@mcp.tool
def add_connection(
    plant_id: str,
    from_component: str,
    to_component: str,
    connection_type: str = "default",
) -> str:
    """Add a connection between two components in the plant.

    Args:
        plant_id (str): ID of the biogas plant.
        from_component (str): Source component ID.
        to_component (str): Target component ID.
        connection_type (str, optional): Connection type (e.g., 'gas', 'liquid', 'electric'). Defaults to "default".

    Returns:
        str: Connection identifier (source->target (type))

    Raises:
        ValueError: If plant_id not found.

    Example:
        >>> add_connection("FarmAB Plant", "D1", "CHP1", "gas")
        'D1->CHP1 (gas)'
    """
    plant = plant_db.get(plant_id)
    if not plant:
        raise ValueError(f"Plant '{plant_id}' not found.")
    connection = Connection(from_component, to_component, connection_type)
    plant.add_connection(connection)
    return f"{from_component}->{to_component} ({connection_type})"


@mcp.tool
def plant_summary(plant_id: str) -> dict:
    """Return a summary of the plant showing components and connections.

    Args:
        plant_id (str): ID of the biogas plant.

    Returns:
        dict: Dictionary with keys "plant_name", "components", and "connections".

    Raises:
        ValueError: If plant_id not found.

    Example:
        >>> plant_summary("FarmAB Plant")
        {'plant_name': 'FarmAB Plant', 'components': [...], 'connections': [...]}
    """
    plant = plant_db.get(plant_id)
    if not plant:
        raise ValueError(f"Plant '{plant_id}' not found.")
    return {
        "plant_name": plant.plant_name,
        "components": list(plant.components.keys()),
        "connections": [str(conn) for conn in plant.connections],
    }


@mcp.tool
def step_simulation(plant_id: str, dt: float = 1.0) -> dict:
    """Advance the simulation for a specified biogas plant by dt days.

    Args:
        plant_id (str): ID of the biogas plant.
        dt (float, optional): Number of days to step. Defaults to 1.0.

    Returns:
        dict: Simulation results from all components.

    Raises:
        ValueError: If plant_id not found.

    Example:
        >>> step_simulation("FarmAB Plant", 1.0)
        {...}
    """
    plant = plant_db.get(plant_id)
    if not plant:
        raise ValueError(f"Plant '{plant_id}' not found.")
    plant.initialize()
    results = plant.step(dt)
    return results


if __name__ == "__main__":
    mcp.run()
