# -*- coding: utf-8 -*-
"""
Unit tests for the Intelligent Biogas MCP Client.

This module tests the IntelligentBiogasClient including:
- Client initialization and connection
- Natural language parsing with LLM
- Plant building workflow
- Component addition and connection
- Error handling and validation
- Integration with MCP tools
"""

import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock

from pyadm1.configurator.mcp.client import (
    IntelligentBiogasClient,
    PlantBuildState,
    PlantContext,
)


class TestPlantContext:
    """Test suite for PlantContext data class."""

    def test_plant_context_initialization_default(self) -> None:
        """Test PlantContext initializes with default values."""
        context = PlantContext()

        assert context.plant_id is None
        assert context.state == PlantBuildState.NOT_CREATED
        assert context.components == {}
        assert context.connections == []
        assert context.description == ""
        assert context.requirements == {}

    def test_plant_context_initialization_with_values(self) -> None:
        """Test PlantContext initializes with provided values."""
        context = PlantContext(plant_id="TestPlant", state=PlantBuildState.CREATED, description="Test plant description")

        assert context.plant_id == "TestPlant"
        assert context.state == PlantBuildState.CREATED
        assert context.description == "Test plant description"

    def test_plant_context_has_component_type(self) -> None:
        """Test has_component_type method."""
        context = PlantContext()
        context.components = {"dig1": "digester", "chp1": "chp"}

        assert context.has_component_type("digester") is True
        assert context.has_component_type("chp") is True
        assert context.has_component_type("heating") is False

    def test_plant_context_get_components_by_type(self) -> None:
        """Test get_components_by_type method."""
        context = PlantContext()
        context.components = {"dig1": "digester", "dig2": "digester", "chp1": "chp"}

        digesters = context.get_components_by_type("digester")
        assert len(digesters) == 2
        assert "dig1" in digesters
        assert "dig2" in digesters

        chps = context.get_components_by_type("chp")
        assert len(chps) == 1
        assert "chp1" in chps

    def test_plant_context_needs_connection(self) -> None:
        """Test needs_connection method."""
        context = PlantContext()
        context.components = {"dig1": "digester", "chp1": "chp"}

        # Before connection
        assert context.needs_connection("digester", "chp") is True

        # After connection
        context.connections = [{"from": "dig1", "to": "chp1", "type": "gas"}]
        assert context.needs_connection("digester", "chp") is False

    def test_plant_context_needs_connection_missing_components(self) -> None:
        """Test needs_connection with missing components."""
        context = PlantContext()
        context.components = {"dig1": "digester"}

        # Missing target component type
        assert context.needs_connection("digester", "chp") is False


class TestPlantBuildState:
    """Test suite for PlantBuildState enum."""

    def test_plant_build_state_enum_values(self) -> None:
        """Test that PlantBuildState has all expected values."""
        expected_states = ["NOT_CREATED", "CREATED", "COMPONENTS_ADDED", "CONNECTED", "INITIALIZED", "SIMULATED"]

        for state_name in expected_states:
            assert hasattr(PlantBuildState, state_name)

    def test_plant_build_state_values_are_strings(self) -> None:
        """Test that PlantBuildState values are strings."""
        for state in PlantBuildState:
            assert isinstance(state.value, str)


class TestIntelligentBiogasClientInitialization:
    """Test suite for client initialization."""

    def test_client_initialization_default_url(self) -> None:
        """Test client initializes with default URL."""
        client = IntelligentBiogasClient()

        assert client.mcp_server_url == "http://127.0.0.1:8000"
        assert client.context.state == PlantBuildState.NOT_CREATED

    def test_client_initialization_custom_url(self) -> None:
        """Test client initializes with custom URL."""
        client = IntelligentBiogasClient("http://localhost:9000")

        assert client.mcp_server_url == "http://localhost:9000"

    def test_client_initialization_creates_context(self) -> None:
        """Test client initializes with PlantContext."""
        client = IntelligentBiogasClient()

        assert isinstance(client.context, PlantContext)
        assert client.context.plant_id is None

    def test_client_initialization_creates_transport(self) -> None:
        """Test client creates SSE transport."""
        client = IntelligentBiogasClient("http://127.0.0.1:8000")

        assert client.transport is not None
        # Transport should have the SSE endpoint URL
        assert hasattr(client.transport, "url")

    def test_client_initialization_creates_mcp_client(self) -> None:
        """Test client creates FastMCP client instance."""
        from fastmcp import Client

        client = IntelligentBiogasClient()

        assert isinstance(client.client, Client)

    def test_client_initialization_empty_conversation_history(self) -> None:
        """Test client starts with empty conversation history."""
        client = IntelligentBiogasClient()

        assert client.conversation_history == []

    def test_client_initialization_empty_caches(self) -> None:
        """Test client starts with empty prompt/tool caches."""
        client = IntelligentBiogasClient()

        assert client.available_prompts == {}
        assert client.available_tools == []


class TestIntelligentBiogasClientConnection:
    """Test suite for client connection management."""

    @pytest.mark.asyncio
    async def test_connect_enters_client_context(self) -> None:
        """Test that connect enters the client async context."""
        client = IntelligentBiogasClient()

        with patch.object(client.client, "__aenter__", new_callable=AsyncMock) as mock_enter:
            with patch.object(client.client, "list_tools", new_callable=AsyncMock) as mock_tools:
                with patch.object(client.client, "list_prompts", new_callable=AsyncMock) as mock_prompts:
                    mock_tools.return_value = []
                    mock_prompts.return_value = []

                    await client.connect()

                    mock_enter.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_fetches_available_tools(self) -> None:
        """Test that connect fetches available tools from server."""
        client = IntelligentBiogasClient()

        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "Test tool description"

        with patch.object(client.client, "__aenter__", new_callable=AsyncMock):
            with patch.object(client.client, "list_tools", new_callable=AsyncMock) as mock_tools:
                with patch.object(client.client, "list_prompts", new_callable=AsyncMock) as mock_prompts:
                    mock_tools.return_value = [mock_tool]
                    mock_prompts.return_value = []

                    await client.connect()

                    assert len(client.available_tools) == 1
                    assert client.available_tools[0]["name"] == "test_tool"

    @pytest.mark.asyncio
    async def test_connect_fetches_available_prompts(self) -> None:
        """Test that connect fetches available prompts from server."""
        client = IntelligentBiogasClient()

        mock_prompt = MagicMock()
        mock_prompt.name = "test_prompt"
        mock_prompt.description = "Test prompt description"

        with patch.object(client.client, "__aenter__", new_callable=AsyncMock):
            with patch.object(client.client, "list_tools", new_callable=AsyncMock) as mock_tools:
                with patch.object(client.client, "list_prompts", new_callable=AsyncMock) as mock_prompts:
                    mock_tools.return_value = []
                    mock_prompts.return_value = [mock_prompt]

                    await client.connect()

                    assert "test_prompt" in client.available_prompts

    @pytest.mark.asyncio
    async def test_disconnect_exits_client_context(self) -> None:
        """Test that disconnect exits the client async context."""
        client = IntelligentBiogasClient()

        with patch.object(client.client, "__aexit__", new_callable=AsyncMock) as mock_exit:
            await client.disconnect()

            mock_exit.assert_called_once_with(None, None, None)

    @pytest.mark.asyncio
    async def test_get_prompt_retrieves_prompt_content(self) -> None:
        """Test that get_prompt retrieves prompt content from server."""
        client = IntelligentBiogasClient()

        mock_prompt = MagicMock()
        mock_message = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "Test prompt content"
        mock_message.content = mock_content
        mock_prompt.messages = [mock_message]

        with patch.object(client.client, "get_prompt", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_prompt

            result = await client.get_prompt("test_prompt")

            assert result == "Test prompt content"
            mock_get.assert_called_once_with("test_prompt")

    @pytest.mark.asyncio
    async def test_get_prompt_handles_errors(self) -> None:
        """Test that get_prompt handles errors gracefully."""
        client = IntelligentBiogasClient()

        with patch.object(client.client, "get_prompt", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = Exception("Test error")

            result = await client.get_prompt("test_prompt")

            assert result is None


class TestIntelligentBiogasClientParsing:
    """Test suite for natural language parsing."""

    @pytest.mark.asyncio
    async def test_parse_description_with_llm_success(self) -> None:
        """Test successful parsing with LLM."""
        client = IntelligentBiogasClient()

        description = "Create a MyFarm plant with 2000 m³ digester and 500 kW CHP"

        # Mock LLMClient
        with patch("pyadm1.configurator.mcp.client.LLMClient") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.return_value = mock_llm

            # Mock successful LLM response
            mock_response = json.dumps(
                {
                    "plant_id": "MyFarm",
                    "digesters": [{"id": "main_digester", "V_liq": 2000, "V_gas": 300, "T_ad": 308.15, "name": "Main"}],
                    "chp": {"id": "chp_main", "P_el_nom": 500, "eta_el": 0.40, "eta_th": 0.45, "name": "CHP"},
                    "heating": [],
                    "simulate": True,
                    "duration": 10.0,
                }
            )
            mock_llm.chat_completion.return_value = mock_response

            result = await client._parse_description_with_llm(description, None, None)

            assert result["plant_id"] == "MyFarm"
            assert len(result["digesters"]) == 1
            assert result["digesters"][0]["V_liq"] == 2000
            assert result["chp"]["P_el_nom"] == 500

    @pytest.mark.asyncio
    async def test_parse_description_fallback_on_llm_error(self) -> None:
        """Test fallback to keyword matching when LLM fails."""
        client = IntelligentBiogasClient()

        description = "Create a two-stage plant with 500 kW CHP"

        # Mock LLMClient to raise error
        with patch("pyadm1.configurator.mcp.client.LLMClient") as mock_llm_class:
            mock_llm_class.side_effect = ImportError("LLM not available")

            result = await client._parse_description_with_llm(description, None, None)

            # Should fall back to keyword parsing
            assert result["plant_id"] is not None
            assert len(result["digesters"]) > 0

    @pytest.mark.asyncio
    async def test_parse_description_fallback_detects_two_stage(self) -> None:
        """Test fallback parser detects two-stage configuration."""
        client = IntelligentBiogasClient()

        description = "Create a two-stage biogas plant"

        result = client._parse_description_fallback(description)

        # Should detect two digesters for two-stage
        assert len(result["digesters"]) == 2
        assert result["digesters"][0]["id"] == "hydrolysis_tank"
        assert result["digesters"][1]["id"] == "main_digester"

    @pytest.mark.asyncio
    async def test_parse_description_fallback_extracts_volume(self) -> None:
        """Test fallback parser extracts volume from description."""
        client = IntelligentBiogasClient()

        description = "Create a plant with 3000 m³ digester"

        result = client._parse_description_fallback(description)

        assert result["digesters"][0]["V_liq"] == 3000.0

    @pytest.mark.asyncio
    async def test_parse_description_fallback_extracts_chp_power(self) -> None:
        """Test fallback parser extracts CHP power from description."""
        client = IntelligentBiogasClient()

        description = "Create a plant with 750 kW CHP"

        result = client._parse_description_fallback(description)

        assert result["chp"]["P_el_nom"] == 750.0

    @pytest.mark.asyncio
    async def test_parse_description_fallback_extracts_duration(self) -> None:
        """Test fallback parser extracts simulation duration."""
        client = IntelligentBiogasClient()

        description = "Run simulation for 30 days"

        result = client._parse_description_fallback(description)

        assert result["duration"] == 30.0


class TestIntelligentBiogasClientValidation:
    """Test suite for requirements validation."""

    @pytest.mark.asyncio
    async def test_validate_requirements_valid_config(self) -> None:
        """Test validation passes for valid configuration."""
        client = IntelligentBiogasClient()

        requirements = {
            "plant_id": "TestPlant",
            "digesters": [{"id": "dig1", "V_liq": 2000, "V_gas": 300, "T_ad": 308.15, "name": "Main"}],
            "chp": {"id": "chp1", "P_el_nom": 500, "eta_el": 0.40, "eta_th": 0.45, "name": "CHP"},
            "heating": [],
            "simulate": True,
            "duration": 10.0,
        }

        # Should not raise exception
        await client._validate_requirements(requirements)

    @pytest.mark.asyncio
    async def test_validate_requirements_invalid_digester(self) -> None:
        """Test validation handles invalid digester configuration."""
        client = IntelligentBiogasClient()

        requirements = {
            "plant_id": "TestPlant",
            "digesters": [{"id": "dig1", "V_liq": -1000, "V_gas": 300, "T_ad": 308.15}],  # Invalid volume
        }

        # Should handle validation error gracefully
        await client._validate_requirements(requirements)


class TestIntelligentBiogasClientPlantBuilding:
    """Test suite for plant building workflow."""

    @pytest.mark.asyncio
    async def test_ensure_plant_created_creates_plant(self) -> None:
        """Test _ensure_plant_created creates plant."""
        client = IntelligentBiogasClient()

        requirements = {"plant_id": "TestPlant"}

        with patch.object(client.client, "call_tool", new_callable=AsyncMock) as mock_call:
            mock_result = MagicMock()
            mock_result.content = [MagicMock(text="Plant created")]
            mock_call.return_value = mock_result

            await client._ensure_plant_created(requirements)

            assert client.context.plant_id == "TestPlant"
            assert client.context.state == PlantBuildState.CREATED

    @pytest.mark.asyncio
    async def test_add_digester_calls_tool(self) -> None:
        """Test _add_digester calls the correct MCP tool."""
        client = IntelligentBiogasClient()
        client.context.plant_id = "TestPlant"

        digester_spec = {"id": "dig1", "V_liq": 2000, "V_gas": 300, "T_ad": 308.15, "name": "Main Digester"}

        with patch.object(client.client, "call_tool", new_callable=AsyncMock) as mock_call:
            mock_result = MagicMock()
            mock_result.content = [MagicMock(text="Digester added")]
            mock_call.return_value = mock_result

            await client._add_digester(digester_spec)

            mock_call.assert_called_once()
            call_args = mock_call.call_args
            assert call_args[0][0] == "add_digester_component"

    @pytest.mark.asyncio
    async def test_add_chp_calls_tool(self) -> None:
        """Test _add_chp calls the correct MCP tool."""
        client = IntelligentBiogasClient()
        client.context.plant_id = "TestPlant"

        chp_spec = {"id": "chp1", "P_el_nom": 500, "eta_el": 0.40, "eta_th": 0.45, "name": "Main CHP"}

        with patch.object(client.client, "call_tool", new_callable=AsyncMock) as mock_call:
            mock_result = MagicMock()
            mock_result.content = [MagicMock(text="CHP added")]
            mock_call.return_value = mock_result

            await client._add_chp(chp_spec)

            mock_call.assert_called_once()
            call_args = mock_call.call_args
            assert call_args[0][0] == "add_chp_unit"

    @pytest.mark.asyncio
    async def test_add_heating_calls_tool(self) -> None:
        """Test _add_heating calls the correct MCP tool."""
        client = IntelligentBiogasClient()
        client.context.plant_id = "TestPlant"

        heating_spec = {"id": "heat1", "target_temperature": 308.15, "heat_loss_coefficient": 0.5, "name": "Main Heating"}

        with patch.object(client.client, "call_tool", new_callable=AsyncMock) as mock_call:
            mock_result = MagicMock()
            mock_result.content = [MagicMock(text="Heating added")]
            mock_call.return_value = mock_result

            await client._add_heating(heating_spec)

            mock_call.assert_called_once()
            call_args = mock_call.call_args
            assert call_args[0][0] == "add_heating_system"

    @pytest.mark.asyncio
    async def test_add_connection_calls_tool(self) -> None:
        """Test _add_connection calls the correct MCP tool."""
        client = IntelligentBiogasClient()
        client.context.plant_id = "TestPlant"

        with patch.object(client.client, "call_tool", new_callable=AsyncMock) as mock_call:
            mock_result = MagicMock()
            mock_result.content = [MagicMock(text="Connection added")]
            mock_call.return_value = mock_result

            await client._add_connection("dig1", "chp1", "gas")

            mock_call.assert_called_once()
            call_args = mock_call.call_args
            assert call_args[0][0] == "connect_components"

    @pytest.mark.asyncio
    async def test_initialize_plant_calls_tool(self) -> None:
        """Test _initialize_plant calls the correct MCP tool."""
        client = IntelligentBiogasClient()
        client.context.plant_id = "TestPlant"
        client.context.state = PlantBuildState.CONNECTED

        with patch.object(client.client, "call_tool", new_callable=AsyncMock) as mock_call:
            mock_result = MagicMock()
            mock_result.content = [MagicMock(text="Plant initialized")]
            mock_call.return_value = mock_result

            await client._initialize_plant()

            mock_call.assert_called_once()
            call_args = mock_call.call_args
            assert call_args[0][0] == "initialize_biogas_plant"
            assert client.context.state == PlantBuildState.INITIALIZED

    @pytest.mark.asyncio
    async def test_simulate_plant_calls_tool(self) -> None:
        """Test _simulate_plant calls the correct MCP tool."""
        client = IntelligentBiogasClient()
        client.context.plant_id = "TestPlant"
        client.context.state = PlantBuildState.INITIALIZED

        requirements = {"duration": 30.0}

        with patch.object(client.client, "call_tool", new_callable=AsyncMock) as mock_call:
            mock_result = MagicMock()
            mock_result.content = [MagicMock(text="Simulation complete")]
            mock_call.return_value = mock_result

            await client._simulate_plant(requirements)

            mock_call.assert_called_once()
            call_args = mock_call.call_args
            assert call_args[0][0] == "simulate_biogas_plant"
            assert client.context.state == PlantBuildState.SIMULATED


class TestIntelligentBiogasClientUtilities:
    """Test suite for utility methods."""

    def test_extract_text_from_result(self) -> None:
        """Test _extract_text extracts text from MCP result."""
        client = IntelligentBiogasClient()

        mock_result = MagicMock()
        mock_item1 = MagicMock()
        mock_item1.text = "Part 1"
        mock_item2 = MagicMock()
        mock_item2.text = "Part 2"
        mock_result.content = [mock_item1, mock_item2]

        text = client._extract_text(mock_result)

        assert "Part 1" in text
        assert "Part 2" in text

    def test_extract_text_handles_no_content(self) -> None:
        """Test _extract_text handles result without content."""
        client = IntelligentBiogasClient()

        mock_result = "Simple string result"

        text = client._extract_text(mock_result)

        assert text == "Simple string result"

    @pytest.mark.asyncio
    async def test_export_configuration_calls_tool(self) -> None:
        """Test export_configuration calls the correct MCP tool."""
        client = IntelligentBiogasClient()
        client.context.plant_id = "TestPlant"

        with patch.object(client.client, "call_tool", new_callable=AsyncMock) as mock_call:
            mock_result = MagicMock()
            mock_result.content = [MagicMock(text="Export complete")]
            mock_call.return_value = mock_result

            result = await client.export_configuration("config.json")

            assert "Export complete" in result

    @pytest.mark.asyncio
    async def test_list_all_plants_calls_tool(self) -> None:
        """Test list_all_plants calls the correct MCP tool."""
        client = IntelligentBiogasClient()

        with patch.object(client.client, "call_tool", new_callable=AsyncMock) as mock_call:
            mock_result = MagicMock()
            mock_result.content = [MagicMock(text="Plant list")]
            mock_call.return_value = mock_result

            result = await client.list_all_plants()

            assert "Plant list" in result

    @pytest.mark.asyncio
    async def test_get_substrate_recommendations_returns_guidance(self) -> None:
        """Test get_substrate_recommendations returns guidance text."""
        client = IntelligentBiogasClient()

        result = await client.get_substrate_recommendations("corn_silage")

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_get_plant_type_guidance_returns_guidance(self) -> None:
        """Test get_plant_type_guidance returns guidance text."""
        client = IntelligentBiogasClient()

        result = await client.get_plant_type_guidance("single_stage")

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_get_parameter_recommendations_returns_dict(self) -> None:
        """Test get_parameter_recommendations returns parameter dictionary."""
        client = IntelligentBiogasClient()

        result = await client.get_parameter_recommendations("V_liq", "medium")

        assert isinstance(result, dict)


class TestIntelligentBiogasClientEndToEnd:
    """End-to-end integration tests for client workflow."""

    @pytest.mark.asyncio
    async def test_build_plant_from_description_complete_workflow(self) -> None:
        """Test complete plant building workflow from description."""
        client = IntelligentBiogasClient()

        description = "Create a TestPlant with 2000 m³ digester and 500 kW CHP"

        # Mock all necessary client methods
        with patch.object(client.client, "__aenter__", new_callable=AsyncMock):
            with patch.object(client.client, "list_tools", new_callable=AsyncMock) as mock_tools:
                with patch.object(client.client, "list_prompts", new_callable=AsyncMock) as mock_prompts:
                    with patch.object(client.client, "call_tool", new_callable=AsyncMock) as mock_call:
                        with patch.object(client, "_parse_description_with_llm", new_callable=AsyncMock) as mock_parse:
                            # Setup mocks
                            mock_tools.return_value = []
                            mock_prompts.return_value = []

                            mock_result = MagicMock()
                            mock_result.content = [MagicMock(text="Success")]
                            mock_call.return_value = mock_result

                            # Mock parsing result
                            mock_parse.return_value = {
                                "plant_id": "TestPlant",
                                "digesters": [{"id": "dig1", "V_liq": 2000, "V_gas": 300, "T_ad": 308.15, "name": "Main"}],
                                "chp": {"id": "chp1", "P_el_nom": 500, "eta_el": 0.40, "eta_th": 0.45, "name": "CHP"},
                                "heating": [],
                                "simulate": True,
                                "duration": 10.0,
                            }

                            await client.connect()
                            result = await client.build_plant_from_description(description)

                            # Verify workflow completed
                            assert isinstance(result, str)
                            assert client.context.plant_id == "TestPlant"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
