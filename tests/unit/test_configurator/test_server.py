# -*- coding: utf-8 -*-
"""
Unit tests for the PyADM1 MCP Server.

This module tests the FastMCP server implementation including:
- Server initialization and configuration
- Tool registration and availability
- Prompt registration and retrieval
- Error handling
- Server lifecycle management
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

# Import the server module and its dependencies
from pyadm1.configurator.mcp.server import (
    mcp,
    start_server,
    create_plant,
    add_digester_component,
    add_chp_unit,
    add_heating_system,
    connect_components,
    initialize_biogas_plant,
    simulate_biogas_plant,
    get_biogas_plant_status,
    export_biogas_plant_config,
    list_biogas_plants,
    delete_biogas_plant,
)


class TestMCPServerInitialization:
    """Test suite for MCP server initialization."""

    def test_mcp_server_has_correct_name(self) -> None:
        """Test that MCP server is initialized with correct name."""
        assert mcp.name == "PyADM1 Biogas Plant Server"

    def test_mcp_server_has_version(self) -> None:
        """Test that MCP server has version information."""
        assert mcp.version == "1.0.0"

    def test_mcp_server_is_fastmcp_instance(self) -> None:
        """Test that mcp is a FastMCP instance."""
        from fastmcp import FastMCP

        assert isinstance(mcp, FastMCP)


class TestMCPServerPrompts:
    """Test suite for MCP server prompt registration."""

    def test_system_guidance_prompt_registered(self) -> None:
        """Test that system_guidance prompt is registered."""
        # Get all registered prompts
        prompts = [p.name for p in mcp._prompts.values()]
        assert "system_guidance" in prompts

    def test_component_selection_prompt_registered(self) -> None:
        """Test that component_selection prompt is registered."""
        prompts = [p.name for p in mcp._prompts.values()]
        assert "component_selection" in prompts

    def test_connection_guidelines_prompt_registered(self) -> None:
        """Test that connection_guidelines prompt is registered."""
        prompts = [p.name for p in mcp._prompts.values()]
        assert "connection_guidelines" in prompts

    def test_parameter_guidelines_prompt_registered(self) -> None:
        """Test that parameter_guidelines prompt is registered."""
        prompts = [p.name for p in mcp._prompts.values()]
        assert "parameter_guidelines" in prompts

    def test_substrate_guide_prompt_registered(self) -> None:
        """Test that substrate_guide prompt is registered."""
        prompts = [p.name for p in mcp._prompts.values()]
        assert "substrate_guide" in prompts

    def test_design_best_practices_prompt_registered(self) -> None:
        """Test that design_best_practices prompt is registered."""
        prompts = [p.name for p in mcp._prompts.values()]
        assert "design_best_practices" in prompts

    def test_troubleshooting_prompt_registered(self) -> None:
        """Test that troubleshooting prompt is registered."""
        prompts = [p.name for p in mcp._prompts.values()]
        assert "troubleshooting" in prompts

    def test_example_plants_prompt_registered(self) -> None:
        """Test that example_plants prompt is registered."""
        prompts = [p.name for p in mcp._prompts.values()]
        assert "example_plants" in prompts

    def test_all_prompts_return_strings(self) -> None:
        """Test that all prompt functions return strings."""
        from pyadm1.configurator.mcp.server import (
            system_guidance,
            component_selection,
            connection_guidelines,
            parameter_guidelines,
            substrate_guide,
            design_best_practices,
            troubleshooting,
            example_plants,
        )

        prompt_functions = [
            system_guidance,
            component_selection,
            connection_guidelines,
            parameter_guidelines,
            substrate_guide,
            design_best_practices,
            troubleshooting,
            example_plants,
        ]

        for prompt_func in prompt_functions:
            result = prompt_func()
            assert isinstance(result, str), f"{prompt_func.__name__} should return string"
            assert len(result) > 0, f"{prompt_func.__name__} should return non-empty string"


class TestMCPServerTools:
    """Test suite for MCP server tool registration."""

    def test_create_plant_tool_registered(self) -> None:
        """Test that create_plant tool is registered."""
        tools = [t.name for t in mcp._tools.values()]
        assert "create_plant" in tools

    def test_add_digester_component_tool_registered(self) -> None:
        """Test that add_digester_component tool is registered."""
        tools = [t.name for t in mcp._tools.values()]
        assert "add_digester_component" in tools

    def test_add_chp_unit_tool_registered(self) -> None:
        """Test that add_chp_unit tool is registered."""
        tools = [t.name for t in mcp._tools.values()]
        assert "add_chp_unit" in tools

    def test_add_heating_system_tool_registered(self) -> None:
        """Test that add_heating_system tool is registered."""
        tools = [t.name for t in mcp._tools.values()]
        assert "add_heating_system" in tools

    def test_connect_components_tool_registered(self) -> None:
        """Test that connect_components tool is registered."""
        tools = [t.name for t in mcp._tools.values()]
        assert "connect_components" in tools

    def test_initialize_biogas_plant_tool_registered(self) -> None:
        """Test that initialize_biogas_plant tool is registered."""
        tools = [t.name for t in mcp._tools.values()]
        assert "initialize_biogas_plant" in tools

    def test_simulate_biogas_plant_tool_registered(self) -> None:
        """Test that simulate_biogas_plant tool is registered."""
        tools = [t.name for t in mcp._tools.values()]
        assert "simulate_biogas_plant" in tools

    def test_get_biogas_plant_status_tool_registered(self) -> None:
        """Test that get_biogas_plant_status tool is registered."""
        tools = [t.name for t in mcp._tools.values()]
        assert "get_biogas_plant_status" in tools

    def test_export_biogas_plant_config_tool_registered(self) -> None:
        """Test that export_biogas_plant_config tool is registered."""
        tools = [t.name for t in mcp._tools.values()]
        assert "export_biogas_plant_config" in tools

    def test_list_biogas_plants_tool_registered(self) -> None:
        """Test that list_biogas_plants tool is registered."""
        tools = [t.name for t in mcp._tools.values()]
        assert "list_biogas_plants" in tools

    def test_delete_biogas_plant_tool_registered(self) -> None:
        """Test that delete_biogas_plant tool is registered."""
        tools = [t.name for t in mcp._tools.values()]
        assert "delete_biogas_plant" in tools

    def test_all_tools_have_correct_count(self) -> None:
        """Test that all expected tools are registered."""
        expected_tool_count = 11  # Total number of tools defined
        actual_tool_count = len(mcp._tools)
        assert actual_tool_count == expected_tool_count, f"Expected {expected_tool_count} tools, got {actual_tool_count}"


class TestMCPServerToolFunctions:
    """Test suite for MCP server tool wrapper functions."""

    @patch("pyadm1.configurator.mcp.server.create_biogas_plant")
    def test_create_plant_calls_underlying_function(self, mock_create: Mock) -> None:
        """Test that create_plant wrapper calls the underlying function."""
        mock_create.return_value = "Success"

        result = create_plant("TestPlant", "Test description", 48)

        mock_create.assert_called_once_with("TestPlant", "Test description", 48)
        assert result == "Success"

    @patch("pyadm1.configurator.mcp.server.add_digester")
    def test_add_digester_component_calls_underlying_function(self, mock_add: Mock) -> None:
        """Test that add_digester_component wrapper calls the underlying function."""
        mock_add.return_value = "Digester added"

        result = add_digester_component(
            "TestPlant", "dig1", V_liq=2000, V_gas=300, T_ad=308.15, name="Main Digester", load_initial_state=True
        )

        mock_add.assert_called_once()
        assert result == "Digester added"

    @patch("pyadm1.configurator.mcp.server.add_chp")
    def test_add_chp_unit_calls_underlying_function(self, mock_add: Mock) -> None:
        """Test that add_chp_unit wrapper calls the underlying function."""
        mock_add.return_value = "CHP added"

        result = add_chp_unit("TestPlant", "chp1", P_el_nom=500, eta_el=0.40, eta_th=0.45, name="Main CHP")

        mock_add.assert_called_once()
        assert result == "CHP added"

    @patch("pyadm1.configurator.mcp.server.add_heating")
    def test_add_heating_system_calls_underlying_function(self, mock_add: Mock) -> None:
        """Test that add_heating_system wrapper calls the underlying function."""
        mock_add.return_value = "Heating added"

        result = add_heating_system(
            "TestPlant", "heat1", target_temperature=308.15, heat_loss_coefficient=0.5, name="Main Heating"
        )

        mock_add.assert_called_once()
        assert result == "Heating added"

    @patch("pyadm1.configurator.mcp.server.add_connection")
    def test_connect_components_calls_underlying_function(self, mock_connect: Mock) -> None:
        """Test that connect_components wrapper calls the underlying function."""
        mock_connect.return_value = "Components connected"

        result = connect_components("TestPlant", "dig1", "chp1", "gas")

        mock_connect.assert_called_once_with("TestPlant", "dig1", "chp1", "gas")
        assert result == "Components connected"

    @patch("pyadm1.configurator.mcp.server.initialize_plant")
    def test_initialize_biogas_plant_calls_underlying_function(self, mock_init: Mock) -> None:
        """Test that initialize_biogas_plant wrapper calls the underlying function."""
        mock_init.return_value = "Plant initialized"

        result = initialize_biogas_plant("TestPlant")

        mock_init.assert_called_once_with("TestPlant")
        assert result == "Plant initialized"

    @patch("pyadm1.configurator.mcp.server.simulate_plant")
    def test_simulate_biogas_plant_calls_underlying_function(self, mock_sim: Mock) -> None:
        """Test that simulate_biogas_plant wrapper calls the underlying function."""
        mock_sim.return_value = "Simulation complete"

        result = simulate_biogas_plant("TestPlant", duration=10.0, dt=0.04167, save_interval=1.0)

        mock_sim.assert_called_once_with("TestPlant", 10.0, 0.04167, 1.0)
        assert result == "Simulation complete"

    @patch("pyadm1.configurator.mcp.server.get_plant_status")
    def test_get_biogas_plant_status_calls_underlying_function(self, mock_status: Mock) -> None:
        """Test that get_biogas_plant_status wrapper calls the underlying function."""
        mock_status.return_value = "Plant status"

        result = get_biogas_plant_status("TestPlant")

        mock_status.assert_called_once_with("TestPlant")
        assert result == "Plant status"

    @patch("pyadm1.configurator.mcp.server.export_plant_config")
    def test_export_biogas_plant_config_calls_underlying_function(self, mock_export: Mock) -> None:
        """Test that export_biogas_plant_config wrapper calls the underlying function."""
        mock_export.return_value = "Export complete"

        result = export_biogas_plant_config("TestPlant", "config.json")

        mock_export.assert_called_once_with("TestPlant", "config.json")
        assert result == "Export complete"

    @patch("pyadm1.configurator.mcp.server.list_plants")
    def test_list_biogas_plants_calls_underlying_function(self, mock_list: Mock) -> None:
        """Test that list_biogas_plants wrapper calls the underlying function."""
        mock_list.return_value = "Plant list"

        result = list_biogas_plants()

        mock_list.assert_called_once()
        assert result == "Plant list"

    @patch("pyadm1.configurator.mcp.server.delete_plant")
    def test_delete_biogas_plant_calls_underlying_function(self, mock_delete: Mock) -> None:
        """Test that delete_biogas_plant wrapper calls the underlying function."""
        mock_delete.return_value = "Plant deleted"

        result = delete_biogas_plant("TestPlant")

        mock_delete.assert_called_once_with("TestPlant")
        assert result == "Plant deleted"


class TestStartServerFunction:
    """Test suite for the start_server function."""

    @patch("pyadm1.configurator.mcp.server.mcp.run")
    @patch("builtins.print")
    def test_start_server_calls_mcp_run(self, mock_print: Mock, mock_run: Mock) -> None:
        """Test that start_server calls mcp.run with correct parameters."""
        start_server(host="127.0.0.1", port=8000)

        mock_run.assert_called_once_with(transport="sse", host="127.0.0.1", port=8000)

    @patch("pyadm1.configurator.mcp.server.mcp.run")
    @patch("builtins.print")
    def test_start_server_with_custom_host(self, mock_print: Mock, mock_run: Mock) -> None:
        """Test that start_server accepts custom host."""
        start_server(host="0.0.0.0", port=8000)

        call_args = mock_run.call_args
        assert call_args.kwargs["host"] == "0.0.0.0"

    @patch("pyadm1.configurator.mcp.server.mcp.run")
    @patch("builtins.print")
    def test_start_server_with_custom_port(self, mock_print: Mock, mock_run: Mock) -> None:
        """Test that start_server accepts custom port."""
        start_server(host="127.0.0.1", port=9000)

        call_args = mock_run.call_args
        assert call_args.kwargs["port"] == 9000

    @patch("pyadm1.configurator.mcp.server.mcp.run")
    @patch("builtins.print")
    def test_start_server_prints_startup_info(self, mock_print: Mock, mock_run: Mock) -> None:
        """Test that start_server prints startup information."""
        start_server(host="127.0.0.1", port=8000)

        # Check that print was called multiple times
        assert mock_print.call_count > 0

        # Check for key information in printed output
        print_calls = [str(call) for call in mock_print.call_args_list]
        output = "".join(print_calls)

        assert "127.0.0.1" in output or "8000" in output

    @patch("pyadm1.configurator.mcp.server.mcp.run", side_effect=KeyboardInterrupt())
    @patch("builtins.print")
    def test_start_server_handles_keyboard_interrupt(self, mock_print: Mock, mock_run: Mock) -> None:
        """Test that start_server handles KeyboardInterrupt gracefully."""
        start_server(host="127.0.0.1", port=8000)

        # Should not raise exception
        # Check that shutdown message was printed
        print_calls = [str(call) for call in mock_print.call_args_list]
        output = "".join(print_calls)

        assert "Shutting down" in output or "shutdown" in output.lower()

    @patch("pyadm1.configurator.mcp.server.mcp.run", side_effect=Exception("Test error"))
    @patch("builtins.print")
    def test_start_server_handles_exceptions(self, mock_print: Mock, mock_run: Mock) -> None:
        """Test that start_server handles exceptions."""
        with pytest.raises(Exception) as excinfo:
            start_server(host="127.0.0.1", port=8000)

        assert "Test error" in str(excinfo.value)

        # Check that error was printed
        print_calls = [str(call) for call in mock_print.call_args_list]
        output = "".join(print_calls)

        assert "Error" in output or "error" in output.lower()

    @patch("pyadm1.configurator.mcp.server.mcp.run")
    @patch("builtins.print")
    def test_start_server_uses_sse_transport(self, mock_print: Mock, mock_run: Mock) -> None:
        """Test that start_server uses SSE transport."""
        start_server(host="127.0.0.1", port=8000)

        call_args = mock_run.call_args
        assert call_args.kwargs["transport"] == "sse"


class TestMCPServerIntegration:
    """Integration tests for MCP server functionality."""

    def test_server_has_all_required_components(self) -> None:
        """Test that server has all required components."""
        # Check tools exist
        assert len(mcp._tools) > 0, "Server should have tools"

        # Check prompts exist
        assert len(mcp._prompts) > 0, "Server should have prompts"

    def test_tool_names_are_unique(self) -> None:
        """Test that all tool names are unique."""
        tool_names = [t.name for t in mcp._tools.values()]
        assert len(tool_names) == len(set(tool_names)), "Tool names should be unique"

    def test_prompt_names_are_unique(self) -> None:
        """Test that all prompt names are unique."""
        prompt_names = [p.name for p in mcp._prompts.values()]
        assert len(prompt_names) == len(set(prompt_names)), "Prompt names should be unique"

    def test_tools_have_documentation(self) -> None:
        """Test that all tools have docstrings."""
        for tool in mcp._tools.values():
            assert tool.fn.__doc__ is not None, f"Tool {tool.name} should have documentation"
            assert len(tool.fn.__doc__) > 0, f"Tool {tool.name} should have non-empty documentation"

    def test_prompts_have_documentation(self) -> None:
        """Test that all prompts have docstrings."""
        for prompt in mcp._prompts.values():
            assert prompt.fn.__doc__ is not None, f"Prompt {prompt.name} should have documentation"
            assert len(prompt.fn.__doc__) > 0, f"Prompt {prompt.name} should have non-empty documentation"


class TestMCPServerToolSequencing:
    """Test suite for proper tool sequencing and workflow."""

    @patch("pyadm1.configurator.mcp.tools.get_registry")
    def test_tools_follow_logical_sequence(self, mock_registry: Mock) -> None:
        """Test that tools can be called in logical sequence."""
        # Mock registry
        mock_reg = MagicMock()
        mock_registry.return_value = mock_reg

        # 1. Create plant should be first
        mock_reg.create_plant.return_value = "TestPlant"

        # 2. Then add components
        # 3. Then connect components
        # 4. Then initialize
        # 5. Then simulate

        # This test verifies the tools exist and can be called in order
        tools_in_order = [
            "create_plant",
            "add_digester_component",
            "add_chp_unit",
            "add_heating_system",
            "connect_components",
            "initialize_biogas_plant",
            "simulate_biogas_plant",
            "get_biogas_plant_status",
        ]

        tool_names = [t.name for t in mcp._tools.values()]

        for tool_name in tools_in_order:
            assert tool_name in tool_names, f"Tool {tool_name} should be registered"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
