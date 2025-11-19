# pyadm1/configurator/mcp/schemas.py
"""
Pydantic Data Schemas for MCP Tool Request/Response Validation

This module defines Pydantic models for all MCP tool parameters and responses,
providing:
- Type safety and validation
- Clear documentation of expected formats
- Automatic JSON schema generation for LLM function calling
- Default values and constraints

Each tool has corresponding request and response schemas.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum


# ==============================================================================
# Enums for Constrained Values
# ==============================================================================


class ComponentTypeEnum(str, Enum):
    """Valid component types in the system."""

    DIGESTER = "digester"
    CHP = "chp"
    HEATING = "heating"
    STORAGE = "storage"
    SEPARATOR = "separator"
    MIXER = "mixer"


class ConnectionTypeEnum(str, Enum):
    """Valid connection types between components."""

    LIQUID = "liquid"
    GAS = "gas"
    HEAT = "heat"
    POWER = "power"
    CONTROL = "control"
    DEFAULT = "default"


class TemperatureRangeEnum(str, Enum):
    """Common temperature operating ranges."""

    PSYCHROPHILIC = "10°C (283.15 K)"
    MESOPHILIC = "35°C (308.15 K)"
    UPPER_MESOPHILIC = "40°C (313.15 K)"
    THERMOPHILIC_LOW = "45°C (318.15 K)"
    THERMOPHILIC_HIGH = "55°C (328.15 K)"


# ==============================================================================
# Plant Creation Schemas
# ==============================================================================


class PlantCreateRequest(BaseModel):
    """Request schema for creating a new biogas plant."""

    plant_id: str = Field(
        ...,
        description="Unique identifier for the plant (e.g., 'FarmAB_Plant', 'MyPlant')",
        min_length=1,
        max_length=50,
        json_schema_extra={"example": "MyFarm_Plant"},
    )

    description: Optional[str] = Field(
        None,
        description="Optional description of the plant purpose and design",
        max_length=500,
        json_schema_extra={"example": "Two-stage digestion plant with corn silage and cattle manure"},
    )

    feeding_freq: int = Field(
        48,
        description="Feeding frequency in hours (how often substrate feed can change)",
        ge=1,
        le=168,
        json_schema_extra={"example": 48},
    )

    @field_validator("plant_id")
    @classmethod
    def validate_plant_id(cls, v: str) -> str:
        """Ensure plant_id contains only valid characters."""
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("plant_id must contain only alphanumeric characters, underscores, and hyphens")
        return v


class PlantCreateResponse(BaseModel):
    """Response schema for plant creation."""

    plant_id: str = Field(..., description="The created plant ID")
    status: str = Field(..., description="Success or error status")
    message: str = Field(..., description="Detailed status message")
    components_count: int = Field(0, description="Number of components (initially 0)")
    connections_count: int = Field(0, description="Number of connections (initially 0)")


# ==============================================================================
# Digester Schemas
# ==============================================================================


class DigesterAddRequest(BaseModel):
    """Request schema for adding a digester component."""

    plant_id: str = Field(
        ..., description="Plant identifier where digester will be added", json_schema_extra={"example": "MyPlant"}
    )

    digester_id: str = Field(
        ..., description="Unique identifier for this digester", json_schema_extra={"example": "main_digester"}
    )

    V_liq: float = Field(
        1977.0,
        description="Liquid volume in cubic meters (m³)",
        ge=100.0,
        le=10000.0,
        json_schema_extra={"example": 2000.0},
    )

    V_gas: float = Field(
        304.0,
        description="Gas volume in cubic meters (m³), typically 10-20% of V_liq",
        ge=10.0,
        le=2000.0,
        json_schema_extra={"example": 300.0},
    )

    T_ad: float = Field(
        308.15,
        description="Operating temperature in Kelvin (308.15 K = 35°C mesophilic)",
        ge=283.15,
        le=333.15,
        json_schema_extra={"example": 308.15},
    )

    name: Optional[str] = Field(
        None,
        description="Human-readable name for the digester",
        max_length=100,
        json_schema_extra={"example": "Main Digester"},
    )

    load_initial_state: bool = Field(True, description="Load default initial state from CSV file")

    @field_validator("V_gas")
    @classmethod
    def validate_gas_volume(cls, v: float, info) -> float:
        """Check that gas volume is reasonable relative to liquid volume."""
        if "V_liq" in info.data:
            ratio = v / info.data["V_liq"]
            if ratio < 0.05 or ratio > 0.30:
                raise ValueError(
                    f"V_gas/V_liq ratio ({ratio:.2f}) should be between 0.05 and 0.30 "
                    "(typically 0.10-0.20 for biogas digesters)"
                )
        return v

    @field_validator("T_ad")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Check temperature is in valid operating range."""
        if v < 283.15:
            raise ValueError("Temperature too low for anaerobic digestion (< 10°C)")
        if v > 333.15:
            raise ValueError("Temperature too high for typical operation (> 60°C)")
        return v


class DigesterAddResponse(BaseModel):
    """Response schema for digester addition."""

    digester_id: str
    plant_id: str
    status: str
    message: str
    V_liq: float
    V_gas: float
    T_ad: float
    HRT_estimate: float = Field(..., description="Estimated hydraulic retention time in days")


# ==============================================================================
# CHP Schemas
# ==============================================================================


class CHPAddRequest(BaseModel):
    """Request schema for adding a CHP unit."""

    plant_id: str = Field(..., description="Plant identifier", json_schema_extra={"example": "MyPlant"})

    chp_id: str = Field(..., description="Unique identifier for this CHP unit", json_schema_extra={"example": "chp_main"})

    P_el_nom: float = Field(
        500.0,
        description="Nominal electrical power output in kilowatts (kW)",
        ge=10.0,
        le=5000.0,
        json_schema_extra={"example": 500.0},
    )

    eta_el: float = Field(
        0.40,
        description="Electrical efficiency (0-1), typical gas engines: 0.38-0.42",
        ge=0.20,
        le=0.60,
        json_schema_extra={"example": 0.40},
    )

    eta_th: float = Field(
        0.45, description="Thermal efficiency (0-1), typical: 0.40-0.50", ge=0.30, le=0.60, json_schema_extra={"example": 0.45}
    )

    name: Optional[str] = Field(
        None, description="Human-readable name", max_length=100, json_schema_extra={"example": "CHP Unit 1"}
    )

    @field_validator("eta_el", "eta_th")
    @classmethod
    def validate_efficiency(cls, v: float, info) -> float:
        """Validate efficiency is in reasonable range."""
        field_name = info.field_name
        if v < 0.20 or v > 0.60:
            raise ValueError(f"{field_name} should be between 0.20 and 0.60 for biogas CHP")
        return v

    @model_validator(mode="after")
    def validate_total_efficiency(self) -> "CHPAddRequest":
        """Check total efficiency is realistic."""
        total = self.eta_th + self.eta_el
        if total < 0.70:
            raise ValueError(
                f"Total efficiency (eta_el + eta_th = {total:.2f}) is too low. " "Typical total efficiency is 0.85-0.90"
            )
        if total > 0.95:
            raise ValueError(f"Total efficiency (eta_el + eta_th = {total:.2f}) is unrealistically high")
        return self


class CHPAddResponse(BaseModel):
    """Response schema for CHP addition."""

    chp_id: str
    plant_id: str
    status: str
    message: str
    P_el_nom: float
    P_th_nom: float = Field(..., description="Thermal power output in kW")
    eta_el: float
    eta_th: float
    eta_total: float = Field(..., description="Combined efficiency")
    biogas_demand_estimate: float = Field(..., description="Estimated biogas demand in m³/day at full load")


# ==============================================================================
# Heating Schemas
# ==============================================================================


class HeatingAddRequest(BaseModel):
    """Request schema for adding a heating system."""

    plant_id: str = Field(..., description="Plant identifier")

    heating_id: str = Field(
        ..., description="Unique identifier for this heating system", json_schema_extra={"example": "heating_main"}
    )

    target_temperature: float = Field(
        308.15,
        description="Target temperature in Kelvin (should match digester temperature)",
        ge=283.15,
        le=333.15,
        json_schema_extra={"example": 308.15},
    )

    heat_loss_coefficient: float = Field(
        0.5,
        description="Heat loss coefficient in kW/K (0.3-0.5 well insulated, 0.8-1.5 poor)",
        ge=0.1,
        le=3.0,
        json_schema_extra={"example": 0.5},
    )

    name: Optional[str] = Field(
        None, description="Human-readable name", json_schema_extra={"example": "Main Digester Heating"}
    )


class HeatingAddResponse(BaseModel):
    """Response schema for heating system addition."""

    heating_id: str
    plant_id: str
    status: str
    message: str
    target_temperature: float
    heat_loss_coefficient: float
    heat_demand_estimate: float = Field(..., description="Estimated heat demand in kW")


# ==============================================================================
# Connection Schemas
# ==============================================================================


class ConnectionAddRequest(BaseModel):
    """Request schema for adding a connection between components."""

    plant_id: str = Field(..., description="Plant identifier")

    from_component: str = Field(..., description="Source component ID", json_schema_extra={"example": "main_digester"})

    to_component: str = Field(..., description="Target component ID", json_schema_extra={"example": "chp_main"})

    connection_type: ConnectionTypeEnum = Field(
        ConnectionTypeEnum.DEFAULT, description="Type of connection defining what flows between components"
    )

    @field_validator("to_component")
    @classmethod
    def validate_not_self_connection(cls, v: str, info) -> str:
        """Prevent component from connecting to itself."""
        if "from_component" in info.data and v == info.data["from_component"]:
            raise ValueError("Cannot connect component to itself")
        return v


class ConnectionAddResponse(BaseModel):
    """Response schema for connection addition."""

    from_component: str
    to_component: str
    connection_type: str
    plant_id: str
    status: str
    message: str
    total_connections: int


# ==============================================================================
# Plant Initialization Schema
# ==============================================================================


class PlantInitializeRequest(BaseModel):
    """Request schema for initializing a plant."""

    plant_id: str = Field(..., description="Plant identifier to initialize")


class PlantInitializeResponse(BaseModel):
    """Response schema for plant initialization."""

    plant_id: str
    status: str
    message: str
    initialized: bool
    components_count: int
    connections_count: int
    warnings: List[str] = Field(default_factory=list, description="Configuration warnings")


# ==============================================================================
# Simulation Schemas
# ==============================================================================


class SimulationRequest(BaseModel):
    """Request schema for running a plant simulation."""

    plant_id: str = Field(..., description="Plant identifier to simulate")

    duration: float = Field(
        10.0, description="Simulation duration in days", ge=0.1, le=365.0, json_schema_extra={"example": 30.0}
    )

    dt: float = Field(
        0.04167,
        description="Time step in days (0.04167 = 1 hour, 0.02083 = 30 min)",
        ge=0.001,
        le=1.0,
        json_schema_extra={"example": 0.04167},
    )

    save_interval: float = Field(
        1.0,
        description="Interval for saving results in days (1.0 = daily snapshots)",
        ge=0.01,
        le=10.0,
        json_schema_extra={"example": 1.0},
    )

    @field_validator("dt")
    @classmethod
    def validate_timestep(cls, v: float, info) -> float:
        """Validate timestep is appropriate for duration."""
        if "duration" in info.data and v > info.data["duration"] / 10:
            raise ValueError(
                f"Time step ({v} days) is too large for duration ({info.data['duration']} days). "
                "Use at least 10 steps per simulation."
            )
        return v


class SimulationResponse(BaseModel):
    """Response schema for simulation results."""

    plant_id: str
    status: str
    message: str
    duration: float
    result_snapshots: int
    final_time: float
    performance_summary: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics by component")


# ==============================================================================
# Plant Status Schema
# ==============================================================================


class PlantStatusRequest(BaseModel):
    """Request schema for getting plant status."""

    plant_id: str = Field(..., description="Plant identifier")


class ComponentInfo(BaseModel):
    """Information about a single component."""

    component_id: str
    component_type: str
    name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ConnectionInfo(BaseModel):
    """Information about a single connection."""

    from_component: str
    to_component: str
    connection_type: str


class PlantStatusResponse(BaseModel):
    """Response schema for plant status."""

    plant_id: str
    description: Optional[str]
    initialized: bool
    simulation_runs: int
    components: List[ComponentInfo]
    connections: List[ConnectionInfo]
    status_message: str


# ==============================================================================
# Export Schema
# ==============================================================================


class PlantExportRequest(BaseModel):
    """Request schema for exporting plant configuration."""

    plant_id: str = Field(..., description="Plant identifier to export")

    filepath: Optional[str] = Field(
        None,
        description="Output filepath (auto-generated if not provided)",
        json_schema_extra={"example": "my_plant_config.json"},
    )


class PlantExportResponse(BaseModel):
    """Response schema for plant export."""

    plant_id: str
    filepath: str
    file_size: int = Field(..., description="File size in bytes")
    status: str
    message: str


# ==============================================================================
# List Plants Schema
# ==============================================================================


class PlantListResponse(BaseModel):
    """Response schema for listing all plants."""

    plants: List[str] = Field(description="List of plant IDs")
    count: int = Field(description="Total number of plants")
    status_summary: Dict[str, int] = Field(
        default_factory=dict, description="Summary of plant states (initialized, simulated, etc.)"
    )


# ==============================================================================
# Delete Plant Schema
# ==============================================================================


class PlantDeleteRequest(BaseModel):
    """Request schema for deleting a plant."""

    plant_id: str = Field(..., description="Plant identifier to delete")


class PlantDeleteResponse(BaseModel):
    """Response schema for plant deletion."""

    plant_id: str
    deleted: bool
    remaining_plants: int
    status: str
    message: str


# ==============================================================================
# Error Response Schema
# ==============================================================================


class ErrorResponse(BaseModel):
    """Standard error response schema."""

    error: str = Field(..., description="Error type or category")
    message: str = Field(..., description="Detailed error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    traceback: Optional[str] = Field(None, description="Stack trace (for debugging)")


# ==============================================================================
# Validation Helper Functions
# ==============================================================================


def validate_substrate_mix(Q: List[float]) -> bool:
    """
    Validate substrate flow rate mix.

    Args:
        Q: List of volumetric flow rates in m³/day

    Returns:
        True if valid, raises ValueError if not

    Raises:
        ValueError: If substrate mix is invalid
    """
    if len(Q) != 10:
        raise ValueError("Substrate mix Q must have exactly 10 values")

    if all(q == 0 for q in Q):
        raise ValueError("At least one substrate must have non-zero flow rate")

    total = sum(Q)
    if total < 5 or total > 200:
        raise ValueError(f"Total flow rate ({total:.1f} m³/d) should be between 5 and 200 m³/d " "for typical biogas plants")

    for i, q in enumerate(Q):
        if q < 0:
            raise ValueError(f"Substrate {i} has negative flow rate: {q}")

    return True


def calculate_hrt(V_liq: float, Q_total: float) -> float:
    """
    Calculate hydraulic retention time.

    Args:
        V_liq: Liquid volume in m³
        Q_total: Total volumetric flow rate in m³/day

    Returns:
        Hydraulic retention time in days
    """
    if Q_total <= 0:
        return float("inf")
    return V_liq / Q_total


def calculate_olr(Q: List[float], V_liq: float, VS_content: float = 0.08) -> float:
    """
    Calculate organic loading rate (simplified).

    Args:
        Q: Substrate flow rates in m³/day
        V_liq: Liquid volume in m³
        VS_content: Volatile solids content (kg VS/m³), default 80 kg/m³

    Returns:
        Organic loading rate in kg VS/(m³·day)
    """
    Q_total = sum(Q)
    # Simplified: assumes average VS content
    # Real calculation would use substrate-specific VS values
    return (Q_total * VS_content) / V_liq


def estimate_biogas_production(Q: List[float], specific_biogas: float = 400.0) -> float:
    """
    Estimate biogas production (simplified).

    Args:
        Q: Substrate flow rates in m³/day
        specific_biogas: Specific biogas yield in Nl/kg VS (default 400)

    Returns:
        Estimated biogas production in m³/day
    """
    Q_total = sum(Q)
    # Simplified: assumes average VS and biogas yield
    VS_total = Q_total * 80  # kg VS/day (80 kg/m³ average)
    biogas = VS_total * specific_biogas / 1000  # m³/day
    return biogas
