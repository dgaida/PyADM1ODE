# -*- coding: utf-8 -*-
"""
Unit tests for the BiogasPlant class.

This module tests the BiogasPlant class which manages multiple components
and their connections to build complete biogas plant configurations.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch

from pyadm1.configurator.plant_builder import BiogasPlant
from pyadm1.configurator.connection_manager import Connection
from pyadm1.components.biological.digester import Digester
from pyadm1.components.energy.chp import CHP
from pyadm1.components.energy.heating import HeatingSystem
from pyadm1.substrates.feedstock import Feedstock


class TestBiogasPlantInitialization:
    """Test suite for BiogasPlant initialization."""

    def test_plant_initialization_sets_name(self) -> None:
        """Test that plant initialization sets the plant name."""
        plant = BiogasPlant("My Biogas Plant")

        assert plant.plant_name == "My Biogas Plant", "Plant name should be set"

    def test_plant_initialization_default_name(self) -> None:
        """Test that plant has default name if not provided."""
        plant = BiogasPlant()

        assert plant.plant_name == "Biogas Plant", "Should have default name"

    def test_plant_initialization_creates_empty_components(self) -> None:
        """Test that plant initializes with empty components dictionary."""
        plant = BiogasPlant("Test Plant")

        assert isinstance(plant.components, dict), "components should be a dictionary"
        assert len(plant.components) == 0, "components should be empty initially"

    def test_plant_initialization_creates_empty_connections(self) -> None:
        """Test that plant initializes with empty connections list."""
        plant = BiogasPlant("Test Plant")

        assert isinstance(plant.connections, list), "connections should be a list"
        assert len(plant.connections) == 0, "connections should be empty initially"

    def test_plant_initialization_sets_simulation_time_to_zero(self) -> None:
        """Test that simulation time starts at zero."""
        plant = BiogasPlant("Test Plant")

        assert plant.simulation_time == 0.0, "simulation_time should start at 0.0"


class TestBiogasPlantAddComponent:
    """Test suite for adding components to the plant."""

    @pytest.fixture
    def mock_feedstock(self) -> Mock:
        """Create a mock Feedstock object."""
        feedstock = Mock(spec=Feedstock)
        feedstock.mySubstrates = Mock()
        return feedstock

    @pytest.fixture
    def mock_digester(self, mock_feedstock: Mock) -> Digester:
        """Create a mock Digester component."""
        digester = Digester("dig_1", mock_feedstock)
        digester.initialize()
        return digester

    @pytest.fixture
    def mock_chp(self) -> CHP:
        """Create a mock CHP component."""
        chp = CHP("chp_1")
        chp.initialize()
        return chp

    def test_add_component_stores_component(self, mock_digester: Digester) -> None:
        """Test that add_component stores the component."""
        plant = BiogasPlant("Test Plant")

        plant.add_component(mock_digester)

        assert "dig_1" in plant.components, "Component should be stored with correct ID"
        assert plant.components["dig_1"] == mock_digester, "Stored component should be the same object"

    def test_add_multiple_components(self, mock_digester: Digester, mock_chp: CHP) -> None:
        """Test that multiple components can be added."""
        plant = BiogasPlant("Test Plant")

        plant.add_component(mock_digester)
        plant.add_component(mock_chp)

        assert len(plant.components) == 2, "Plant should have 2 components"
        assert "dig_1" in plant.components, "Digester should be in components"
        assert "chp_1" in plant.components, "CHP should be in components"

    def test_add_component_duplicate_raises_error(self, mock_digester: Digester) -> None:
        """Test that adding duplicate component ID raises error."""
        plant = BiogasPlant("Test Plant")

        plant.add_component(mock_digester)

        with pytest.raises(ValueError) as excinfo:
            plant.add_component(mock_digester)

        assert "already exists" in str(excinfo.value), "Should raise error for duplicate ID"

    def test_add_component_increments_count(self, mock_digester: Digester) -> None:
        """Test that adding components increases the count."""
        plant = BiogasPlant("Test Plant")

        initial_count = len(plant.components)
        plant.add_component(mock_digester)

        assert len(plant.components) == initial_count + 1, "Component count should increase"


class TestBiogasPlantAddConnection:
    """Test suite for adding connections between components."""

    @pytest.fixture
    def mock_feedstock(self) -> Mock:
        """Create a mock Feedstock object."""
        feedstock = Mock(spec=Feedstock)
        feedstock.mySubstrates = Mock()
        return feedstock

    @pytest.fixture
    def plant_with_components(self, mock_feedstock: Mock) -> BiogasPlant:
        """Create a plant with some components."""
        plant = BiogasPlant("Test Plant")

        digester = Digester("dig_1", mock_feedstock)
        digester.initialize()
        chp = CHP("chp_1")
        chp.initialize()

        plant.add_component(digester)
        plant.add_component(chp)

        return plant

    def test_add_connection_stores_connection(self, plant_with_components: BiogasPlant) -> None:
        """Test that add_connection stores the connection."""
        connection = Connection("dig_1", "chp_1", "gas")

        plant_with_components.add_connection(connection)

        assert len(plant_with_components.connections) == 1, "Connection should be stored"
        assert plant_with_components.connections[0] == connection, "Stored connection should match"

    def test_add_connection_updates_component_connections(self, plant_with_components: BiogasPlant) -> None:
        """Test that add_connection updates component input/output lists."""
        connection = Connection("dig_1", "chp_1", "gas")

        plant_with_components.add_connection(connection)

        digester = plant_with_components.components["dig_1"]
        chp = plant_with_components.components["chp_1"]

        assert "chp_1" in digester.outputs, "Digester should have CHP as output"
        assert "dig_1" in chp.inputs, "CHP should have digester as input"

    def test_add_connection_nonexistent_source_raises_error(self, plant_with_components: BiogasPlant) -> None:
        """Test that connecting from nonexistent source raises error."""
        connection = Connection("nonexistent", "chp_1", "gas")

        with pytest.raises(ValueError) as excinfo:
            plant_with_components.add_connection(connection)

        assert "not found" in str(excinfo.value), "Should raise error for missing source"

    def test_add_connection_nonexistent_target_raises_error(self, plant_with_components: BiogasPlant) -> None:
        """Test that connecting to nonexistent target raises error."""
        connection = Connection("dig_1", "nonexistent", "gas")

        with pytest.raises(ValueError) as excinfo:
            plant_with_components.add_connection(connection)

        assert "not found" in str(excinfo.value), "Should raise error for missing target"

    def test_add_multiple_connections(self, plant_with_components: BiogasPlant) -> None:
        """Test that multiple connections can be added."""
        heating = HeatingSystem("heat_1")
        heating.initialize()
        plant_with_components.add_component(heating)

        conn1 = Connection("dig_1", "chp_1", "gas")
        conn2 = Connection("chp_1", "heat_1", "heat")

        plant_with_components.add_connection(conn1)
        plant_with_components.add_connection(conn2)

        assert len(plant_with_components.connections) == 2, "Plant should have 2 connections"


class TestBiogasPlantInitialize:
    """Test suite for plant initialization."""

    @pytest.fixture
    def mock_feedstock(self) -> Mock:
        """Create a mock Feedstock object."""
        feedstock = Mock(spec=Feedstock)
        feedstock.mySubstrates = Mock()
        return feedstock

    @pytest.fixture
    def simple_plant(self, mock_feedstock: Mock) -> BiogasPlant:
        """Create a simple plant with basic components."""
        plant = BiogasPlant("Test Plant")

        digester = Digester("dig_1", mock_feedstock)
        chp = CHP("chp_1")

        plant.add_component(digester)
        plant.add_component(chp)

        return plant

    def test_initialize_initializes_all_components(self, simple_plant: BiogasPlant) -> None:
        """Test that initialize initializes all components."""
        simple_plant.initialize()

        for component in simple_plant.components.values():
            assert component._initialized, f"{component.name} should be initialized"

    def test_initialize_can_be_called_multiple_times(self, simple_plant: BiogasPlant) -> None:
        """Test that initialize can be called multiple times safely."""
        simple_plant.initialize()
        simple_plant.initialize()  # Should not raise error

        assert all(c._initialized for c in simple_plant.components.values()), "Components should remain initialized"


class TestBiogasPlantSerialization:
    """Test suite for plant serialization and deserialization."""

    @pytest.fixture
    def mock_feedstock(self) -> Mock:
        """Create a mock Feedstock object."""
        feedstock = Mock(spec=Feedstock)
        feedstock.mySubstrates = Mock()
        return feedstock

    @pytest.fixture
    def plant_for_serialization(self, mock_feedstock: Mock) -> BiogasPlant:
        """Create a plant for serialization testing."""
        plant = BiogasPlant("Test Plant")

        digester = Digester("dig_1", mock_feedstock, V_liq=2000, V_gas=300)
        digester.initialize()
        chp = CHP("chp_1", P_el_nom=500)
        chp.initialize()

        plant.add_component(digester)
        plant.add_component(chp)
        plant.add_connection(Connection("dig_1", "chp_1", "gas"))

        return plant

    def test_to_json_creates_file(self, plant_for_serialization: BiogasPlant, tmp_path: Path) -> None:
        """Test that to_json creates a JSON file."""
        filepath = tmp_path / "plant_config.json"

        plant_for_serialization.to_json(str(filepath))

        assert filepath.exists(), "JSON file should be created"

    def test_to_json_file_is_valid_json(self, plant_for_serialization: BiogasPlant, tmp_path: Path) -> None:
        """Test that created JSON file is valid JSON."""
        filepath = tmp_path / "plant_config.json"

        plant_for_serialization.to_json(str(filepath))

        with open(filepath) as f:
            data = json.load(f)

        assert isinstance(data, dict), "JSON should be a valid dictionary"

    def test_to_json_contains_required_fields(self, plant_for_serialization: BiogasPlant, tmp_path: Path) -> None:
        """Test that JSON contains required fields."""
        filepath = tmp_path / "plant_config.json"

        plant_for_serialization.to_json(str(filepath))

        with open(filepath) as f:
            data = json.load(f)

        required_fields = ["plant_name", "components", "connections", "simulation_time"]
        for field in required_fields:
            assert field in data, f"JSON should contain '{field}'"

    def test_from_json_recreates_plant(
        self, plant_for_serialization: BiogasPlant, mock_feedstock: Mock, tmp_path: Path
    ) -> None:
        """Test that from_json can recreate a plant."""
        filepath = tmp_path / "plant_config.json"

        plant_for_serialization.to_json(str(filepath))
        recreated_plant = BiogasPlant.from_json(str(filepath), mock_feedstock)

        assert recreated_plant.plant_name == plant_for_serialization.plant_name, "Plant name should match"
        assert len(recreated_plant.components) == len(plant_for_serialization.components), "Component count should match"
        assert len(recreated_plant.connections) == len(plant_for_serialization.connections), "Connection count should match"

    def test_roundtrip_preserves_structure(
        self, plant_for_serialization: BiogasPlant, mock_feedstock: Mock, tmp_path: Path
    ) -> None:
        """Test that serialization roundtrip preserves plant structure."""
        filepath = tmp_path / "plant_config.json"

        plant_for_serialization.to_json(str(filepath))
        recreated = BiogasPlant.from_json(str(filepath), mock_feedstock)

        assert "dig_1" in recreated.components, "Digester should be in recreated plant"
        assert "chp_1" in recreated.components, "CHP should be in recreated plant"


class TestBiogasPlantProperties:
    """Test suite for BiogasPlant properties and summaries."""

    @pytest.fixture
    def mock_feedstock(self) -> Mock:
        """Create a mock Feedstock object."""
        feedstock = Mock(spec=Feedstock)
        feedstock.mySubstrates = Mock()
        return feedstock

    @pytest.fixture
    def sample_plant(self, mock_feedstock: Mock) -> BiogasPlant:
        """Create a sample plant for testing."""
        plant = BiogasPlant("Sample Plant")

        digester = Digester("dig_1", mock_feedstock)
        digester.initialize()
        chp = CHP("chp_1")
        chp.initialize()

        plant.add_component(digester)
        plant.add_component(chp)

        return plant

    def test_get_summary_returns_string(self, sample_plant: BiogasPlant) -> None:
        """Test that get_summary returns a string."""
        summary = sample_plant.get_summary()

        assert isinstance(summary, str), "get_summary should return a string"

    def test_get_summary_contains_plant_name(self, sample_plant: BiogasPlant) -> None:
        """Test that summary contains plant name."""
        summary = sample_plant.get_summary()

        assert "Sample Plant" in summary, "Summary should contain plant name"

    def test_get_summary_contains_component_count(self, sample_plant: BiogasPlant) -> None:
        """Test that summary contains component information."""
        summary = sample_plant.get_summary()

        assert "2" in summary, "Summary should mention number of components"

    def test_component_count(self, sample_plant: BiogasPlant) -> None:
        """Test that component count is accurate."""
        assert len(sample_plant.components) == 2, "Plant should have 2 components"

    def test_connection_count(self, sample_plant: BiogasPlant) -> None:
        """Test that connection count is accurate."""
        # Before adding connections
        assert len(sample_plant.connections) == 0, "Plant should have no connections initially"

        # After adding connection
        sample_plant.add_connection(Connection("dig_1", "chp_1", "gas"))
        assert len(sample_plant.connections) == 1, "Plant should have 1 connection"


class TestBiogasPlantSimulation:
    """Test suite for plant simulation capabilities."""

    @pytest.fixture
    def mock_feedstock(self) -> Mock:
        """Create a mock Feedstock object."""
        feedstock = Mock(spec=Feedstock)
        feedstock.mySubstrates = Mock()
        return feedstock

    @pytest.fixture
    def plant_ready_to_simulate(self, mock_feedstock: Mock) -> BiogasPlant:
        """Create a plant ready for simulation."""
        plant = BiogasPlant("Simulation Test Plant")

        digester = Digester("dig_1", mock_feedstock)
        digester.initialize()
        chp = CHP("chp_1")
        chp.initialize()

        plant.add_component(digester)
        plant.add_component(chp)
        plant.add_connection(Connection("dig_1", "chp_1", "gas"))

        plant.initialize()

        return plant

    def test_step_returns_dict(self, plant_ready_to_simulate: BiogasPlant) -> None:
        """Test that step method returns a dictionary."""
        with patch.object(Digester, "step", return_value={"Q_gas": 1500}):
            with patch.object(CHP, "step", return_value={"P_el": 400}):
                result = plant_ready_to_simulate.step(dt=1.0 / 24.0)

        assert isinstance(result, dict), "step should return a dictionary"

    def test_step_contains_component_results(self, plant_ready_to_simulate: BiogasPlant) -> None:
        """Test that step results contain component IDs."""
        with patch.object(Digester, "step", return_value={"Q_gas": 1500}):
            with patch.object(CHP, "step", return_value={"P_el": 400}):
                result = plant_ready_to_simulate.step(dt=1.0 / 24.0)

        assert "dig_1" in result, "Results should contain digester"

    def test_step_updates_simulation_time(self, plant_ready_to_simulate: BiogasPlant) -> None:
        """Test that step updates simulation time."""
        initial_time = plant_ready_to_simulate.simulation_time

        with patch.object(Digester, "step", return_value={}):
            with patch.object(CHP, "step", return_value={}):
                plant_ready_to_simulate.step(dt=1.0)

        assert plant_ready_to_simulate.simulation_time > initial_time, "Simulation time should increase"

    def test_simulate_returns_list(self, plant_ready_to_simulate: BiogasPlant) -> None:
        """Test that simulate returns a list of results."""
        with patch.object(Digester, "step", return_value={}):
            with patch.object(CHP, "step", return_value={}):
                results = plant_ready_to_simulate.simulate(duration=1.0, dt=1.0 / 24.0)

        assert isinstance(results, list), "simulate should return a list"

    def test_simulate_returns_timestamped_results(self, plant_ready_to_simulate: BiogasPlant) -> None:
        """Test that simulate returns timestamped results."""
        with patch.object(Digester, "step", return_value={}):
            with patch.object(CHP, "step", return_value={}):
                results = plant_ready_to_simulate.simulate(duration=2.0, dt=1.0, save_interval=1.0)

        assert len(results) > 0, "Should have some results"
        assert all("time" in r for r in results), "Each result should have time"
        assert all("components" in r for r in results), "Each result should have components"
