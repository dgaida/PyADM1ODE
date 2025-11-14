"""
Biogas Plant MCP Client: create and simulate biogas plants using natural language,
LLMClient for NLU, and FastMCP for communication.

Requires:
    - fastmcp (pip install fastmcp)
    - llm_client (pip install git+https://github.com/dgaida/llm_client.git)
"""

import asyncio
from pyadm1.configurator.mcp.client import IntelligentBiogasClient


async def main():
    """Main example demonstrating the intelligent client."""

    # Example plant descriptions
    examples = [
        "Create a MyFarm biogas plant with a single-stage 2000 m³ digester, "
        "500 kW CHP unit, and heating system. Simulate for 30 days.",
        "Build a TwoStage facility with two-stage digestion and combined heat " "and power generation. Run 14 day simulation.",
        "I need a biogas plant called FarmAB with one digester of 1500 m³ " "and a 400 kW CHP. Include heating.",
    ]

    # Use first example
    description = examples[0]

    # Create and connect client
    client = IntelligentBiogasClient("http://127.0.0.1:8000")

    try:
        await client.connect()

        # Build plant from description
        result = await client.build_plant_from_description(description)
        print(result)

        # Export configuration
        print("\n" + "=" * 70)
        export_result = await client.export_configuration()
        print(export_result)

        # List all plants
        print("\n" + "=" * 70)
        plants_list = await client.list_all_plants()
        print(plants_list)

    finally:
        await client.disconnect()


# python examples/biogas_mcp_client.py
if __name__ == "__main__":
    asyncio.run(main())
