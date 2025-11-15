# pyadm1/configurator/mcp/client.py
"""
Intelligent MCP Client for PyADM1 Biogas Plant Modeling

This client provides a high-level interface for building biogas plants from
natural language descriptions. It intelligently:
- Parses plant requirements using LLM
- Selects appropriate tools and sequences
- Validates configurations using schemas
- Leverages server prompts for guidance
- Manages plant state through the building process

The client can work in two modes:
1. Automatic: Parse description and build plant automatically
2. Interactive: Provide tools for manual plant building

Example:
    >>> client = IntelligentBiogasClient("http://127.0.0.1:8000")
    >>> await client.connect()
    >>> result = await client.build_plant_from_description(
    ...     "Create a 500 kW plant with two-stage digestion"
    ... )
"""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from fastmcp import Client
from fastmcp.client.transports import SSETransport

# Import schemas for validation
from pyadm1.configurator.mcp.schemas import (
    PlantCreateRequest,
    DigesterAddRequest,
    CHPAddRequest,
    HeatingAddRequest,
    # ConnectionAddRequest,
    # validate_substrate_mix,
)

# Import prompts for context
from pyadm1.configurator.mcp.prompts import (
    get_prompt_for_plant_type,
    get_substrate_guidance,
    get_parameter_recommendation,
)


class PlantBuildState(Enum):
    """States in the plant building process."""

    NOT_CREATED = "not_created"
    CREATED = "created"
    COMPONENTS_ADDED = "components_added"
    CONNECTED = "connected"
    INITIALIZED = "initialized"
    SIMULATED = "simulated"


@dataclass
class PlantContext:
    """
    Tracks the current state of plant construction.

    Attributes:
        plant_id: Unique identifier for the plant
        state: Current build state
        components: Mapping of component IDs to types
        connections: List of connections
        description: Natural language description
        requirements: Parsed requirements dictionary
    """

    plant_id: Optional[str] = None
    state: PlantBuildState = PlantBuildState.NOT_CREATED
    components: Dict[str, str] = None
    connections: List[Dict[str, str]] = None
    description: str = ""
    requirements: Dict[str, Any] = None

    def __post_init__(self):
        if self.components is None:
            self.components = {}
        if self.connections is None:
            self.connections = []
        if self.requirements is None:
            self.requirements = {}

    def has_component_type(self, comp_type: str) -> bool:
        """Check if plant has component of given type."""
        return any(ct == comp_type for ct in self.components.values())

    def get_components_by_type(self, comp_type: str) -> List[str]:
        """Get all component IDs of a specific type."""
        return [cid for cid, ct in self.components.items() if ct == comp_type]

    def needs_connection(self, from_type: str, to_type: str) -> bool:
        """Check if connection between component types is needed."""
        if not (self.has_component_type(from_type) and self.has_component_type(to_type)):
            return False

        from_comps = self.get_components_by_type(from_type)
        to_comps = self.get_components_by_type(to_type)

        for from_id in from_comps:
            for to_id in to_comps:
                if any(c["from"] == from_id and c["to"] == to_id for c in self.connections):
                    return False

        return True


class IntelligentBiogasClient:
    """
    Intelligent MCP client for biogas plant design.

    This client understands plant building workflows and automatically:
    - Creates plants in the correct order
    - Adds appropriate components
    - Establishes necessary connections
    - Initializes and simulates

    It uses LLM for natural language parsing and leverages server prompts
    for design guidance.

    Attributes:
        mcp_server_url: URL of the MCP server
        transport: SSE transport for communication
        client: FastMCP client instance
        context: Current plant building context
        conversation_history: History of interactions
        available_prompts: Cached server prompts
        available_tools: Cached tool definitions
    """

    def __init__(self, mcp_server_url: str = "http://127.0.0.1:8000"):
        """
        Initialize the intelligent client.

        Args:
            mcp_server_url: URL of the PyADM1 MCP server
        """
        self.mcp_server_url = mcp_server_url
        self.transport = SSETransport(f"{mcp_server_url}/sse")
        self.client = Client(self.transport)
        self.context = PlantContext()
        self.conversation_history: List[Dict[str, str]] = []
        self.available_prompts: Dict[str, str] = {}
        self.available_tools: List[Dict[str, Any]] = []

    async def connect(self) -> None:
        """
        Connect to MCP server and fetch available prompts and tools.

        Raises:
            ConnectionError: If unable to connect to server
        """
        print("🔌 Connecting to PyADM1 MCP Server...")
        await self.client.__aenter__()

        # Get available tools
        tools = await self.client.list_tools()
        self.available_tools = [{"name": tool.name, "description": tool.description} for tool in tools]
        print(f"✓ Connected! Available tools: {len(self.available_tools)}")
        for tool in self.available_tools:
            print(f"  - {tool['name']}")

        # Fetch available prompts
        try:
            prompts = await self.client.list_prompts()
            self.available_prompts = {prompt.name: prompt.description for prompt in prompts}
            print(f"\n✓ Available prompts: {len(self.available_prompts)}")
            for name in self.available_prompts:
                print(f"  - {name}")
        except Exception as e:
            print(f"⚠️  Could not fetch prompts: {e}")

        print()

    async def disconnect(self) -> None:
        """Disconnect from MCP server."""
        await self.client.__aexit__(None, None, None)
        print("\n✓ Disconnected from MCP server")

    async def get_prompt(self, prompt_name: str) -> Optional[str]:
        """
        Retrieve a specific prompt from the server.

        Args:
            prompt_name: Name of the prompt to retrieve

        Returns:
            Prompt content or None if not available
        """
        try:
            prompt = await self.client.get_prompt(prompt_name)
            return prompt.messages[0].content.text if prompt.messages else None
        except Exception as e:
            print(f"⚠️  Could not retrieve prompt '{prompt_name}': {e}")
            return None

    async def build_plant_from_description(self, description: str) -> str:
        """
        Build a complete biogas plant from natural language description.

        This is the main entry point that orchestrates the entire plant
        building process using LLM for parsing and tool selection.

        Args:
            description: Natural language description of desired plant

        Returns:
            Summary of the built plant

        Example:
            >>> result = await client.build_plant_from_description(
            ...     "Create a 500 kW two-stage plant for dairy farm"
            ... )
        """
        print(f"📝 Plant Description: {description}\n")
        self.context.description = description

        # Get design guidance from server prompts
        component_guide = await self.get_prompt("component_selection")
        parameter_guide = await self.get_prompt("parameter_guidelines")

        # Parse description to extract requirements
        print("🔍 Parsing plant requirements...")
        requirements = await self._parse_description_with_llm(description, component_guide, parameter_guide)

        self.context.requirements = requirements
        print("\nIdentified requirements:")
        for key, value in requirements.items():
            if key not in ["digesters", "heating"]:  # Skip complex nested structures
                print(f"  - {key}: {value}")
        print()

        # Validate requirements against schemas
        await self._validate_requirements(requirements)

        # Step 1: Create plant
        await self._ensure_plant_created(requirements)

        # Step 2: Add components
        await self._add_required_components(requirements)

        # Step 3: Connect components
        await self._connect_components()

        # Step 4: Initialize
        await self._initialize_plant()

        # Step 5: Simulate (if requested)
        if requirements.get("simulate", True):
            await self._simulate_plant(requirements)

        # Step 6: Get final status
        result = await self._get_plant_status()

        return result

    async def _parse_description_with_llm(
        self, description: str, component_guide: Optional[str], parameter_guide: Optional[str]
    ) -> Dict[str, Any]:
        """
        Parse natural language description using LLM.

        Uses LLMClient to intelligently extract plant configuration from
        natural language descriptions, guided by server prompts.

        Args:
            description: Natural language description
            component_guide: Component selection guidelines from server
            parameter_guide: Parameter ranges from server

        Returns:
            Dictionary of extracted requirements
        """
        # Try to use LLMClient for intelligent parsing
        try:
            from llm_client import LLMClient
        except ImportError:
            print("⚠️  Warning: llm_client not installed. Using fallback keyword matching.")
            return self._parse_description_fallback(description)

        # Create LLM client
        try:
            llm = LLMClient(temperature=0.3, max_tokens=2048)
        except Exception as e:
            print(f"⚠️  Warning: Could not initialize LLMClient: {e}")
            print("Using fallback keyword matching.")
            return self._parse_description_fallback(description)

        # Build enhanced system prompt with server guidance
        system_prompt = (
            """You are an expert biogas plant engineer. Extract configuration parameters from plant descriptions.

"""
            + (f"Component Selection Guidelines:\n{component_guide}\n\n" if component_guide else "")
            + (f"Parameter Guidelines:\n{parameter_guide}\n\n" if parameter_guide else "")
            + """

Return ONLY a valid JSON object with this exact structure (no markdown, no explanation):
{
    "plant_id": "string (extract from description or generate like 'Plant_001')",
    "digesters": [
        {
            "id": "string (e.g., 'main_digester', 'hydrolysis_tank')",
            "V_liq": number (liquid volume in m³, default 2000.0 for single-stage),
            "V_gas": number (gas volume in m³, typically 15% of V_liq),
            "T_ad": number (temperature in Kelvin: 308.15 for 35°C, 318.15 for 45°C),
            "name": "string (descriptive name)"
        }
    ],
    "chp": {
        "id": "string (e.g., 'chp_main')",
        "P_el_nom": number (electrical power in kW, default 500.0),
        "eta_el": 0.40,
        "eta_th": 0.45,
        "name": "string"
    } or null,
    "heating": [
        {
            "id": "string (e.g., 'heating_1')",
            "target_temperature": number (same as corresponding digester T_ad),
            "heat_loss_coefficient": 0.5,
            "name": "string"
        }
    ],
    "simulate": true,
    "duration": number (days, default 10.0)
}

Guidelines:
- Single-stage: 1 digester at 308.15K (35°C)
- Two-stage: hydrolysis tank (318.15K/45°C) + main digester (308.15K/35°C)
- Extract volumes and powers from numbers in description
- Add CHP if mentioned (power generation, electricity, combined heat)
- Add heating for each digester if CHP present or heating mentioned
- Extract simulation duration if mentioned
- Validate against parameter ranges provided above"""
        )

        user_prompt = f"Extract biogas plant configuration from this description:\n\n{description}"

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        try:
            response = llm.chat_completion(messages)

            # Clean response (remove markdown if present)
            response_clean = response.strip()
            if response_clean.startswith("```"):
                lines = response_clean.split("\n")
                response_clean = "\n".join(lines[1:-1]) if len(lines) > 2 else response_clean
                response_clean = response_clean.replace("```json", "").replace("```", "").strip()

            # Parse JSON
            requirements = json.loads(response_clean)

            # Set defaults for missing fields
            if not requirements.get("plant_id"):
                requirements["plant_id"] = "BiogasPlant_001"

            if not requirements.get("digesters"):
                requirements["digesters"] = [
                    {"id": "main_digester", "V_liq": 2000.0, "V_gas": 300.0, "T_ad": 308.15, "name": "Main Digester"}
                ]

            # Ensure heating systems match digesters if CHP present
            if requirements.get("chp") and not requirements.get("heating"):
                requirements["heating"] = []
                for i, digester in enumerate(requirements["digesters"]):
                    requirements["heating"].append(
                        {
                            "id": f"heating_{i + 1}",
                            "target_temperature": digester["T_ad"],
                            "heat_loss_coefficient": 0.5,
                            "name": f"Heating System {i + 1}",
                        }
                    )

            requirements.setdefault("simulate", True)
            requirements.setdefault("duration", 10.0)

            return requirements

        except json.JSONDecodeError as e:
            print(f"⚠️  Warning: Could not parse LLM response as JSON: {e}")
            print(f"Response was: {response[:200]}...")
            print("Using fallback keyword matching.")
            return self._parse_description_fallback(description)
        except Exception as e:
            print(f"⚠️  Warning: Error using LLM for parsing: {e}")
            print("Using fallback keyword matching.")
            return self._parse_description_fallback(description)

    def _parse_description_fallback(self, description: str) -> Dict[str, Any]:
        """
        Fallback parser using simple keyword matching.

        Used when LLM is not available or fails.

        Args:
            description: Natural language description

        Returns:
            Dictionary of extracted requirements
        """
        desc_lower = description.lower()
        requirements = {"plant_id": None, "digesters": [], "chp": None, "heating": [], "simulate": True, "duration": 10.0}

        # Extract plant ID or generate one
        words = description.split()
        for i, word in enumerate(words):
            if word.lower() in ["plant", "farm", "facility"] and i > 0:
                requirements["plant_id"] = words[i - 1].strip(",").strip(".")
                break

        if not requirements["plant_id"]:
            requirements["plant_id"] = "BiogasPlant_001"

        # Detect digesters
        if "two-stage" in desc_lower or "two stage" in desc_lower:
            requirements["digesters"].append(
                {"id": "hydrolysis_tank", "V_liq": 500.0, "V_gas": 100.0, "T_ad": 318.15, "name": "Hydrolysis Tank"}
            )
            requirements["digesters"].append(
                {"id": "main_digester", "V_liq": 1500.0, "V_gas": 250.0, "T_ad": 308.15, "name": "Main Digester"}
            )
        else:
            # Single stage - extract volume if mentioned
            V_liq = 2000.0
            if "m³" in description or "m3" in description:
                import re

                matches = re.findall(r"(\d+(?:\.\d+)?)\s*m[³3]", description)
                if matches:
                    V_liq = float(matches[0])

            requirements["digesters"].append(
                {"id": "main_digester", "V_liq": V_liq, "V_gas": V_liq * 0.15, "T_ad": 308.15, "name": "Main Digester"}
            )

        # Detect CHP
        if "chp" in desc_lower or "combined heat" in desc_lower or "power" in desc_lower:
            P_el = 500.0
            import re

            matches = re.findall(r"(\d+(?:\.\d+)?)\s*kw", desc_lower)
            if matches:
                P_el = float(matches[0])

            requirements["chp"] = {"id": "chp_main", "P_el_nom": P_el, "eta_el": 0.40, "eta_th": 0.45, "name": "CHP Unit"}

        # Detect heating
        if "heating" in desc_lower or requirements["chp"]:
            for i, digester in enumerate(requirements["digesters"]):
                requirements["heating"].append(
                    {
                        "id": f"heating_{i + 1}",
                        "target_temperature": digester["T_ad"],
                        "heat_loss_coefficient": 0.5,
                        "name": f"Heating System {i + 1}",
                    }
                )

        # Detect simulation duration
        import re

        day_matches = re.findall(r"(\d+)\s*days?", desc_lower)
        if day_matches:
            requirements["duration"] = float(day_matches[0])

        return requirements

    async def _validate_requirements(self, requirements: Dict[str, Any]) -> None:
        """
        Validate parsed requirements using Pydantic schemas.

        Args:
            requirements: Parsed requirements dictionary

        Raises:
            ValidationError: If requirements don't match schemas
        """
        print("✅ Validating requirements against schemas...")

        try:
            # Validate plant creation
            PlantCreateRequest(
                plant_id=requirements["plant_id"],
                description=requirements.get("description", ""),
                feeding_freq=requirements.get("feeding_freq", 48),
            )

            # Validate digesters
            for digester in requirements.get("digesters", []):
                DigesterAddRequest(
                    plant_id=requirements["plant_id"],
                    digester_id=digester["id"],
                    V_liq=digester["V_liq"],
                    V_gas=digester["V_gas"],
                    T_ad=digester["T_ad"],
                    name=digester.get("name"),
                )

            # Validate CHP if present
            if requirements.get("chp"):
                chp = requirements["chp"]
                CHPAddRequest(
                    plant_id=requirements["plant_id"],
                    chp_id=chp["id"],
                    P_el_nom=chp["P_el_nom"],
                    eta_el=chp["eta_el"],
                    eta_th=chp["eta_th"],
                    name=chp.get("name"),
                )

            # Validate heating systems
            for heating in requirements.get("heating", []):
                HeatingAddRequest(
                    plant_id=requirements["plant_id"],
                    heating_id=heating["id"],
                    target_temperature=heating["target_temperature"],
                    heat_loss_coefficient=heating["heat_loss_coefficient"],
                    name=heating.get("name"),
                )

            print("✓ All requirements validated successfully\n")

        except Exception as e:
            print(f"⚠️  Validation warning: {e}")
            print("Continuing with potentially invalid configuration...\n")

    async def _ensure_plant_created(self, requirements: Dict[str, Any]) -> None:
        """Create plant if not already created."""
        if self.context.state != PlantBuildState.NOT_CREATED:
            return

        print("🏗️  Step 1: Creating plant...")

        plant_id = requirements["plant_id"]
        description = self.context.description

        result = await self.client.call_tool(
            "create_plant", {"plant_id": plant_id, "description": description, "feeding_freq": 48}
        )

        print(self._extract_text(result))
        print()

        self.context.plant_id = plant_id
        self.context.state = PlantBuildState.CREATED

    async def _add_required_components(self, requirements: Dict[str, Any]) -> None:
        """Add all required components to the plant."""
        if self.context.state.value not in ["created", "components_added"]:
            return

        print("🔧 Step 2: Adding components...")

        # Add digesters
        for digester_spec in requirements["digesters"]:
            await self._add_digester(digester_spec)
            self.context.components[digester_spec["id"]] = "digester"

        # Add CHP
        if requirements["chp"]:
            await self._add_chp(requirements["chp"])
            self.context.components[requirements["chp"]["id"]] = "chp"

        # Add heating systems
        for heating_spec in requirements["heating"]:
            await self._add_heating(heating_spec)
            self.context.components[heating_spec["id"]] = "heating"

        self.context.state = PlantBuildState.COMPONENTS_ADDED
        print()

    async def _add_digester(self, spec: Dict[str, Any]) -> None:
        """Add a digester component."""
        result = await self.client.call_tool(
            "add_digester_component",
            {
                "plant_id": self.context.plant_id,
                "digester_id": spec["id"],
                "V_liq": spec["V_liq"],
                "V_gas": spec["V_gas"],
                "T_ad": spec["T_ad"],
                "name": spec["name"],
                "load_initial_state": True,
            },
        )
        print(self._extract_text(result))

    async def _add_chp(self, spec: Dict[str, Any]) -> None:
        """Add a CHP unit."""
        result = await self.client.call_tool(
            "add_chp_unit",
            {
                "plant_id": self.context.plant_id,
                "chp_id": spec["id"],
                "P_el_nom": spec["P_el_nom"],
                "eta_el": spec["eta_el"],
                "eta_th": spec["eta_th"],
                "name": spec["name"],
            },
        )
        print(self._extract_text(result))

    async def _add_heating(self, spec: Dict[str, Any]) -> None:
        """Add a heating system."""
        result = await self.client.call_tool(
            "add_heating_system",
            {
                "plant_id": self.context.plant_id,
                "heating_id": spec["id"],
                "target_temperature": spec["target_temperature"],
                "heat_loss_coefficient": spec["heat_loss_coefficient"],
                "name": spec["name"],
            },
        )
        print(self._extract_text(result))

    async def _connect_components(self) -> None:
        """Intelligently connect all components."""
        if self.context.state != PlantBuildState.COMPONENTS_ADDED:
            return

        print("🔗 Step 3: Connecting components...")

        # Get connection guidance from server
        connection_guide = await self.get_prompt("connection_guidelines")
        if connection_guide:
            print("   Using server connection guidelines...")

        digesters = self.context.get_components_by_type("digester")
        chps = self.context.get_components_by_type("chp")
        heating_systems = self.context.get_components_by_type("heating")

        # Connect digesters in series if multiple
        if len(digesters) > 1:
            for i in range(len(digesters) - 1):
                await self._add_connection(digesters[i], digesters[i + 1], "liquid")

        # Connect digesters to CHP (gas flow)
        if chps:
            for digester in digesters:
                await self._add_connection(digester, chps[0], "gas")

        # Connect CHP to heating systems (heat flow)
        if chps and heating_systems:
            for heating in heating_systems:
                await self._add_connection(chps[0], heating, "heat")

        self.context.state = PlantBuildState.CONNECTED
        print()

    async def _add_connection(self, from_comp: str, to_comp: str, conn_type: str) -> None:
        """Add a connection between components."""
        result = await self.client.call_tool(
            "connect_components",
            {
                "plant_id": self.context.plant_id,
                "from_component": from_comp,
                "to_component": to_comp,
                "connection_type": conn_type,
            },
        )
        print(self._extract_text(result))

        self.context.connections.append({"from": from_comp, "to": to_comp, "type": conn_type})

    async def _initialize_plant(self) -> None:
        """Initialize the plant for simulation."""
        if self.context.state != PlantBuildState.CONNECTED:
            return

        print("⚙️  Step 4: Initializing plant...")

        result = await self.client.call_tool("initialize_biogas_plant", {"plant_id": self.context.plant_id})

        print(self._extract_text(result))
        print()

        self.context.state = PlantBuildState.INITIALIZED

    async def _simulate_plant(self, requirements: Dict[str, Any]) -> None:
        """Run plant simulation."""
        if self.context.state != PlantBuildState.INITIALIZED:
            return

        print("🚀 Step 5: Running simulation...")

        duration = requirements.get("duration", 10.0)

        result = await self.client.call_tool(
            "simulate_biogas_plant",
            {
                "plant_id": self.context.plant_id,
                "duration": duration,
                "dt": 0.04167,  # 1 hour
                "save_interval": 1.0,  # Daily snapshots
            },
        )

        print(self._extract_text(result))
        print()

        self.context.state = PlantBuildState.SIMULATED

    async def _get_plant_status(self) -> str:
        """Get final plant status."""
        print("📊 Step 6: Retrieving plant status...")

        result = await self.client.call_tool("get_biogas_plant_status", {"plant_id": self.context.plant_id})

        status_text = self._extract_text(result)
        print(status_text)

        return status_text

    def _extract_text(self, result: Any) -> str:
        """Extract text content from MCP result."""
        if hasattr(result, "content"):
            text_parts = [item.text for item in result.content if hasattr(item, "text")]
            return "\n".join(text_parts)
        return str(result)

    async def export_configuration(self, filepath: Optional[str] = None) -> str:
        """Export the current plant configuration."""
        if not self.context.plant_id:
            return "No plant to export"

        result = await self.client.call_tool(
            "export_biogas_plant_config", {"plant_id": self.context.plant_id, "filepath": filepath}
        )

        return self._extract_text(result)

    async def list_all_plants(self) -> str:
        """List all plants in the registry."""
        result = await self.client.call_tool("list_biogas_plants", {})
        return self._extract_text(result)

    async def get_substrate_recommendations(self, substrate_name: str) -> str:
        """
        Get substrate-specific recommendations.

        Args:
            substrate_name: Name of substrate (e.g., "corn_silage")

        Returns:
            Substrate guidance text
        """
        return get_substrate_guidance(substrate_name)

    async def get_plant_type_guidance(self, plant_type: str) -> str:
        """
        Get guidance for a specific plant type.

        Args:
            plant_type: Type of plant ("single_stage", "two_stage", etc.)

        Returns:
            Plant type guidance text
        """
        return get_prompt_for_plant_type(plant_type)

    async def get_parameter_recommendations(self, parameter_name: str, plant_size: str = "medium") -> Dict[str, float]:
        """
        Get parameter recommendations based on plant size.

        Args:
            parameter_name: Parameter name ("V_liq", "P_el_nom", etc.)
            plant_size: Plant size category ("small", "medium", "large")

        Returns:
            Dictionary with default, min, max values
        """
        return get_parameter_recommendation(parameter_name, plant_size)
