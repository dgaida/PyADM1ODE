"""
Biogas Plant MCP Client: create and simulate biogas plants using natural language,
LLMClient for NLU, and FastMCP for communication.

Requires:
    - fastmcp (pip install fastmcp)
    - llm_client (pip install git+https://github.com/dgaida/llm_client.git)
"""

import asyncio
from fastmcp import Client
from llm_client.llm_client import LLMClient


async def main():
    # MCP server address
    mcp_server_url = "http://localhost:8000"  # change as needed

    # Initialize the LLM client (API key autodetect)
    llm_client = LLMClient()

    # Connect to the MCP server
    async with Client(mcp_server_url) as mcp_client:
        print("Connected to MCP MCP Server.")

        # User provides a natural language request for a new biogas plant
        user_prompt = (
            "Create a biogas plant for a farm, include a digester with "
            "2000 m³ liquid, 300 m³ gas, and a CHP unit with 500 kW electrical output."
        )

        # Step 1: Use LLM to parse prompt into tool calls
        messages = [
            {"role": "system", "content": "Extract actions for MCP tools from this natural language prompt."},
            {"role": "user", "content": user_prompt},
        ]
        llm_result = llm_client.chat_completion(messages)
        llm_output = llm_result["choices"][0]["message"]["content"]

        # Let's suppose the LLM output is a dictionary with fields for tool calls, e.g.:
        # {
        #   "plant_name": "FarmPlant01",
        #   "digester": {"V_liq": 2000, "V_gas": 300, "T_ad": 308.15, "digester_id": "D1"},
        #   "chp": {"P_el_nom": 500, "eta_el": 0.40, "eta_th": 0.45, "chp_id": "CHP1"}
        # }
        import json

        try:
            config = json.loads(llm_output)
        except Exception:
            print("LLM output could not be parsed; raw:", llm_output)
            return

        # Step 2: Create biogas plant on server
        plant_id = await mcp_client.call_tool("createBiogasPlant", {"plant_name": config["plant_name"]})
        print("Created plant:", plant_id.content[0].text)

        # Step 3: Add digester
        digester_args = config["digester"]
        digester_args["plant_id"] = config["plant_name"]
        digester_result = await mcp_client.call_tool("addDigester", digester_args)
        print("Added digester:", digester_result.content[0].text)

        # Step 4: Add CHP
        chp_args = config["chp"]
        chp_args["plant_id"] = config["plant_name"]
        chp_result = await mcp_client.call_tool("addCHP", chp_args)
        print("Added CHP:", chp_result.content[0].text)

        # Step 5: See plant summary
        summary = await mcp_client.call_tool("plant_summary", {"plant_id": config["plant_name"]})
        print("Summary:", summary.content[0].text)

        # Step 6: Step simulation
        sim_result = await mcp_client.call_tool("step_simulation", {"plant_id": config["plant_name"], "dt": 1.0})
        print("Simulation result:", sim_result.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())
