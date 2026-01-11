# tests/unit/test_configurator/test_plant_configurator.py
# -*- coding: utf-8 -*-
"""
Unit tests for PlantConfigurator class.

This module tests the PlantConfigurator class which provides high-level
methods for building biogas plants with sensible defaults and validation.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from pyadm1.configurator.plant_configurator import PlantConfigurator
from pyadm1.configurator.plant_builder import BiogasPlant
from pyadm1.configurator.connection_manager import Connection
from pyadm1.components.biological.digester import Digester
from pyadm1.components.energy.chp import CHP
from pyadm1.components.energy.heating import HeatingSystem
from pyadm1.components.energy.gas_storage import GasStorage
from pyadm1.components.energy.flare import Flare
from pyadm1.substrates.feedstock import Feedstock


class TestPlantConfiguratorInitialization:
    """Test suite for PlantConfigurator initialization."""

    @pytest.fixture
    def mock_feedstock(self) -> Mock:
        """
        Create a mock Feedstock object.

        Returns:
            Mock Feedstock object with necessary attributes.
        """
        feedstock = Mock(spec=Feedstock)
        feedstock.mySubstrates = Mock()
        return feedstock

    @pytest.fixture
    def plant(self) -> BiogasPlant:
        """
        Create a BiogasPlant instance.

        Returns:
            BiogasPlant instance for testing.
        """
        return BiogasPlant("Test Plant")

    def test_initialization_stores_plant(self, plant: BiogasPlant, mock_feedstock: Mock) -> None:
        """
        Test that initialization stores plant reference.

        Args:
            plant: BiogasPlant fixture.
            mock_feedstock: Mock Feedstock fixture.
        """
        configurator = PlantConfigurator(plant, mock_feedstock)

        assert configurator.plant == plant, "Should store plant reference"

    def test_initialization_stores_feedstock(self, plant: BiogasPlant, mock_feedstock: Mock) -> None:
        """
        Test that initialization stores feedstock reference.

        Args:
            plant: BiogasPlant fixture.
            mock_feedstock: Mock Feedstock fixture.
        """
        configurator = PlantConfigurator(plant, mock_feedstock)

        assert configurator.feedstock == mock_feedstock, "Should store feedstock reference"


class TestAddDigester:
    """Test suite for add_digester method."""

    @pytest.fixture
    def mock_feedstock(self) -> Mock:
        """Create a mock Feedstock object."""
        feedstock = Mock(spec=Feedstock)
        feedstock.mySubstrates = Mock()
        return feedstock

    @pytest.fixture
    def plant(self) -> BiogasPlant:
        """Create a BiogasPlant instance."""
        return BiogasPlant("Test Plant")

    @pytest.fixture
    def configurator(self, plant: BiogasPlant, mock_feedstock: Mock) -> PlantConfigurator:
        """Create a PlantConfigurator instance."""
        return PlantConfigurator(plant, mock_feedstock)

    def test_add_digester_returns_digester_and_info(self, configurator: PlantConfigurator) -> None:
        """
        Test that add_digester returns digester and state info.

        Args:
            configurator: PlantConfigurator fixture.
        """
        with patch("pyadm1.configurator.plant_configurator.get_state_zero_from_initial_state") as mock_get_state:
            mock_get_state.return_value = [0.01] * 37

            result = configurator.add_digester("dig1")

            assert isinstance(result, tuple), "Should return a tuple"
            assert len(result) == 2, "Should return (digester, state_info)"

            digester, state_info = result
            assert isinstance(digester, Digester), "First element should be Digester"
            assert isinstance(state_info, str), "Second element should be state info string"

    def test_add_digester_adds_to_plant(self, configurator: PlantConfigurator) -> None:
        """
        Test that add_digester adds component to plant.

        Args:
            configurator: PlantConfigurator fixture.
        """
        with patch("pyadm1.configurator.plant_configurator.get_state_zero_from_initial_state") as mock_get_state:
            mock_get_state.return_value = [0.01] * 37

            configurator.add_digester("dig1")

            assert "dig1" in configurator.plant.components, "Digester should be added to plant"

    def test_add_digester_creates_gas_storage(self, configurator: PlantConfigurator) -> None:
        """
        Test that add_digester creates associated gas storage.

        Args:
            configurator: PlantConfigurator fixture.
        """
        with patch("pyadm1.configurator.plant_configurator.get_state_zero_from_initial_state") as mock_get_state:
            mock_get_state.return_value = [0.01] * 37

            configurator.add_digester("dig1")

            storage_id = "dig1_storage"
            assert storage_id in configurator.plant.components, "Gas storage should be created"
            assert isinstance(configurator.plant.components[storage_id], GasStorage), "Storage should be GasStorage instance"

    def test_add_digester_connects_to_storage(self, configurator: PlantConfigurator) -> None:
        """
        Test that add_digester connects digester to storage.

        Args:
            configurator: PlantConfigurator fixture.
        """
        with patch("pyadm1.configurator.plant_configurator.get_state_zero_from_initial_state") as mock_get_state:
            mock_get_state.return_value = [0.01] * 37

            configurator.add_digester("dig1")

            # Check if connection exists
            connection_exists = any(
                conn.from_component == "dig1" and conn.to_component == "dig1_storage" and conn.connection_type == "gas"
                for conn in configurator.plant.connections
            )
            assert connection_exists, "Digester should be connected to storage"

    def test_add_digester_with_custom_volumes(self, configurator: PlantConfigurator) -> None:
        """
        Test add_digester with custom liquid and gas volumes.

        Args:
            configurator: PlantConfigurator fixture.
        """
        with patch("pyadm1.configurator.plant_configurator.get_state_zero_from_initial_state") as mock_get_state:
            mock_get_state.return_value = [0.01] * 37

            V_liq = 2500.0
            V_gas = 350.0

            digester, _ = configurator.add_digester("dig1", V_liq=V_liq, V_gas=V_gas)

            assert digester.V_liq == V_liq, f"V_liq should be {V_liq}"
            assert digester.V_gas == V_gas, f"V_gas should be {V_gas}"

    def test_add_digester_with_custom_temperature(self, configurator: PlantConfigurator) -> None:
        """
        Test add_digester with custom operating temperature.

        Args:
            configurator: PlantConfigurator fixture.
        """
        with patch("pyadm1.configurator.plant_configurator.get_state_zero_from_initial_state") as mock_get_state:
            mock_get_state.return_value = [0.01] * 37

            T_ad = 318.15  # 45°C (thermophilic)

            digester, _ = configurator.add_digester("dig1", T_ad=T_ad)

            assert digester.T_ad == T_ad, f"T_ad should be {T_ad}"

    def test_add_digester_with_custom_name(self, configurator: PlantConfigurator) -> None:
        """
        Test add_digester with custom name.

        Args:
            configurator: PlantConfigurator fixture.
        """
        with patch("pyadm1.configurator.plant_configurator.get_state_zero_from_initial_state") as mock_get_state:
            mock_get_state.return_value = [0.01] * 37

            custom_name = "Main Digester"

            digester, _ = configurator.add_digester("dig1", name=custom_name)

            assert digester.name == custom_name, f"Name should be '{custom_name}'"

    def test_add_digester_loads_default_initial_state(self, configurator: PlantConfigurator) -> None:
        """
        Test that add_digester loads default initial state.

        Args:
            configurator: PlantConfigurator fixture.
        """
        with patch("pyadm1.configurator.plant_configurator.get_state_zero_from_initial_state") as mock_get_state:
            expected_state = [0.015] * 37
            mock_get_state.return_value = expected_state

            digester, state_info = configurator.add_digester("dig1")

            assert "Loaded from" in state_info, "State info should mention loading"
            mock_get_state.assert_called_once()

    def test_add_digester_with_custom_initial_state_file(self, configurator: PlantConfigurator, tmp_path: Path) -> None:
        """
        Test add_digester with custom initial state file.

        Args:
            configurator: PlantConfigurator fixture.
            tmp_path: pytest fixture providing temporary directory.
        """
        custom_file = str(tmp_path / "custom_state.csv")

        with patch("pyadm1.configurator.plant_configurator.get_state_zero_from_initial_state") as mock_get_state:
            mock_get_state.return_value = [0.02] * 37

            digester, state_info = configurator.add_digester("dig1", initial_state_file=custom_file)

            mock_get_state.assert_called_once_with(custom_file)
            assert custom_file in state_info, "State info should mention custom file"

    def test_add_digester_with_custom_Q_substrates(self, configurator: PlantConfigurator) -> None:
        """
        Test add_digester with custom substrate feed rates.

        Args:
            configurator: PlantConfigurator fixture.
        """
        with patch("pyadm1.configurator.plant_configurator.get_state_zero_from_initial_state") as mock_get_state:
            mock_get_state.return_value = [0.01] * 37

            Q_substrates = [20, 15, 5, 0, 0, 0, 0, 0, 0, 0]

            digester, _ = configurator.add_digester("dig1", Q_substrates=Q_substrates)

            assert digester.Q_substrates == Q_substrates, "Q_substrates should match"

    def test_add_digester_without_loading_state(self, configurator: PlantConfigurator) -> None:
        """
        Test add_digester with load_initial_state=False.

        Args:
            configurator: PlantConfigurator fixture.
        """
        with patch("pyadm1.configurator.plant_configurator.get_state_zero_from_initial_state") as mock_get_state:
            digester, state_info = configurator.add_digester("dig1", load_initial_state=False)

            mock_get_state.assert_not_called()
            assert "Not initialized" in state_info, "State info should indicate no initialization"


class TestAddCHP:
    """Test suite for add_chp method."""

    @pytest.fixture
    def mock_feedstock(self) -> Mock:
        """Create a mock Feedstock object."""
        feedstock = Mock(spec=Feedstock)
        feedstock.mySubstrates = Mock()
        return feedstock

    @pytest.fixture
    def plant(self) -> BiogasPlant:
        """Create a BiogasPlant instance."""
        return BiogasPlant("Test Plant")

    @pytest.fixture
    def configurator(self, plant: BiogasPlant, mock_feedstock: Mock) -> PlantConfigurator:
        """Create a PlantConfigurator instance."""
        return PlantConfigurator(plant, mock_feedstock)

    def test_add_chp_returns_chp_instance(self, configurator: PlantConfigurator) -> None:
        """
        Test that add_chp returns a CHP instance.

        Args:
            configurator: PlantConfigurator fixture.
        """
        chp = configurator.add_chp("chp1")

        assert isinstance(chp, CHP), "Should return a CHP instance"

    def test_add_chp_adds_to_plant(self, configurator: PlantConfigurator) -> None:
        """
        Test that add_chp adds component to plant.

        Args:
            configurator: PlantConfigurator fixture.
        """
        configurator.add_chp("chp1")

        assert "chp1" in configurator.plant.components, "CHP should be added to plant"

    def test_add_chp_creates_flare(self, configurator: PlantConfigurator) -> None:
        """
        Test that add_chp creates associated flare.

        Args:
            configurator: PlantConfigurator fixture.
        """
        configurator.add_chp("chp1")

        flare_id = "chp1_flare"
        assert flare_id in configurator.plant.components, "Flare should be created"
        assert isinstance(configurator.plant.components[flare_id], Flare), "Should be a Flare instance"

    def test_add_chp_connects_to_flare(self, configurator: PlantConfigurator) -> None:
        """
        Test that add_chp connects CHP to flare.

        Args:
            configurator: PlantConfigurator fixture.
        """
        configurator.add_chp("chp1")

        connection_exists = any(
            conn.from_component == "chp1" and conn.to_component == "chp1_flare" and conn.connection_type == "gas"
            for conn in configurator.plant.connections
        )
        assert connection_exists, "CHP should be connected to flare"

    def test_add_chp_with_custom_power(self, configurator: PlantConfigurator) -> None:
        """
        Test add_chp with custom electrical power.

        Args:
            configurator: PlantConfigurator fixture.
        """
        P_el_nom = 750.0

        chp = configurator.add_chp("chp1", P_el_nom=P_el_nom)

        assert chp.P_el_nom == P_el_nom, f"P_el_nom should be {P_el_nom}"

    def test_add_chp_with_custom_efficiencies(self, configurator: PlantConfigurator) -> None:
        """
        Test add_chp with custom efficiency values.

        Args:
            configurator: PlantConfigurator fixture.
        """
        eta_el = 0.42
        eta_th = 0.48

        chp = configurator.add_chp("chp1", eta_el=eta_el, eta_th=eta_th)

        assert chp.eta_el == eta_el, f"eta_el should be {eta_el}"
        assert chp.eta_th == eta_th, f"eta_th should be {eta_th}"

    def test_add_chp_with_custom_name(self, configurator: PlantConfigurator) -> None:
        """
        Test add_chp with custom name.

        Args:
            configurator: PlantConfigurator fixture.
        """
        custom_name = "Main CHP Unit"

        chp = configurator.add_chp("chp1", name=custom_name)

        assert chp.name == custom_name, f"Name should be '{custom_name}'"


class TestAddHeating:
    """Test suite for add_heating method."""

    @pytest.fixture
    def mock_feedstock(self) -> Mock:
        """Create a mock Feedstock object."""
        feedstock = Mock(spec=Feedstock)
        feedstock.mySubstrates = Mock()
        return feedstock

    @pytest.fixture
    def plant(self) -> BiogasPlant:
        """Create a BiogasPlant instance."""
        return BiogasPlant("Test Plant")

    @pytest.fixture
    def configurator(self, plant: BiogasPlant, mock_feedstock: Mock) -> PlantConfigurator:
        """Create a PlantConfigurator instance."""
        return PlantConfigurator(plant, mock_feedstock)

    def test_add_heating_returns_heating_instance(self, configurator: PlantConfigurator) -> None:
        """
        Test that add_heating returns a HeatingSystem instance.

        Args:
            configurator: PlantConfigurator fixture.
        """
        heating = configurator.add_heating("heat1")

        assert isinstance(heating, HeatingSystem), "Should return a HeatingSystem instance"

    def test_add_heating_adds_to_plant(self, configurator: PlantConfigurator) -> None:
        """
        Test that add_heating adds component to plant.

        Args:
            configurator: PlantConfigurator fixture.
        """
        configurator.add_heating("heat1")

        assert "heat1" in configurator.plant.components, "Heating should be added to plant"

    def test_add_heating_with_custom_temperature(self, configurator: PlantConfigurator) -> None:
        """
        Test add_heating with custom target temperature.

        Args:
            configurator: PlantConfigurator fixture.
        """
        target_temp = 318.15  # 45°C

        heating = configurator.add_heating("heat1", target_temperature=target_temp)

        assert heating.target_temperature == target_temp, f"Target temperature should be {target_temp}"

    def test_add_heating_with_custom_heat_loss(self, configurator: PlantConfigurator) -> None:
        """
        Test add_heating with custom heat loss coefficient.

        Args:
            configurator: PlantConfigurator fixture.
        """
        heat_loss = 0.75

        heating = configurator.add_heating("heat1", heat_loss_coefficient=heat_loss)

        assert heating.heat_loss_coefficient == heat_loss, f"Heat loss should be {heat_loss}"

    def test_add_heating_with_custom_name(self, configurator: PlantConfigurator) -> None:
        """
        Test add_heating with custom name.

        Args:
            configurator: PlantConfigurator fixture.
        """
        custom_name = "Digester Heating System"

        heating = configurator.add_heating("heat1", name=custom_name)

        assert heating.name == custom_name, f"Name should be '{custom_name}'"


class TestConnect:
    """Test suite for connect method."""

    @pytest.fixture
    def mock_feedstock(self) -> Mock:
        """Create a mock Feedstock object."""
        feedstock = Mock(spec=Feedstock)
        feedstock.mySubstrates = Mock()
        return feedstock

    @pytest.fixture
    def plant_with_components(self, mock_feedstock: Mock) -> BiogasPlant:
        """Create a plant with pre-added components."""
        plant = BiogasPlant("Test Plant")

        with patch("pyadm1.configurator.plant_configurator.get_state_zero_from_initial_state") as mock_get_state:
            mock_get_state.return_value = [0.01] * 37

            digester = Digester("dig1", mock_feedstock)
            digester.initialize()
            chp = CHP("chp1")
            chp.initialize()

            plant.add_component(digester)
            plant.add_component(chp)

        return plant

    @pytest.fixture
    def configurator(self, plant_with_components: BiogasPlant, mock_feedstock: Mock) -> PlantConfigurator:
        """Create a PlantConfigurator with components."""
        return PlantConfigurator(plant_with_components, mock_feedstock)

    def test_connect_returns_connection(self, configurator: PlantConfigurator) -> None:
        """
        Test that connect returns a Connection instance.

        Args:
            configurator: PlantConfigurator fixture.
        """
        connection = configurator.connect("dig1", "chp1", "gas")

        assert isinstance(connection, Connection), "Should return a Connection instance"

    def test_connect_adds_connection_to_plant(self, configurator: PlantConfigurator) -> None:
        """
        Test that connect adds connection to plant.

        Args:
            configurator: PlantConfigurator fixture.
        """
        initial_count = len(configurator.plant.connections)

        configurator.connect("dig1", "chp1", "gas")

        assert len(configurator.plant.connections) == initial_count + 1, "Connection count should increase"

    def test_connect_with_different_types(self, configurator: PlantConfigurator) -> None:
        """
        Test connect with different connection types.

        Args:
            configurator: PlantConfigurator fixture.
        """
        connection_types = ["liquid", "gas", "heat", "power", "default"]

        for i, conn_type in enumerate(connection_types):
            # Note: we're creating multiple connections for testing
            configurator.connect("dig1", "chp1", conn_type)

        assert len(configurator.plant.connections) == len(connection_types), "Should have created all connection types"


class TestAutoConnectDigesterToChp:
    """Test suite for auto_connect_digester_to_chp method."""

    @pytest.fixture
    def mock_feedstock(self) -> Mock:
        """Create a mock Feedstock object."""
        feedstock = Mock(spec=Feedstock)
        feedstock.mySubstrates = Mock()
        return feedstock

    @pytest.fixture
    def plant(self) -> BiogasPlant:
        """Create a BiogasPlant instance."""
        return BiogasPlant("Test Plant")

    @pytest.fixture
    def configurator(self, plant: BiogasPlant, mock_feedstock: Mock) -> PlantConfigurator:
        """Create a PlantConfigurator instance."""
        return PlantConfigurator(plant, mock_feedstock)

    def test_auto_connect_creates_storage_connection(self, configurator: PlantConfigurator) -> None:
        """
        Test that auto_connect_digester_to_chp creates storage connection.

        Args:
            configurator: PlantConfigurator fixture.
        """
        with patch("pyadm1.configurator.plant_configurator.get_state_zero_from_initial_state") as mock_get_state:
            mock_get_state.return_value = [0.01] * 37

            # Add digester (which creates storage)
            configurator.add_digester("dig1")
            # Add CHP
            configurator.add_chp("chp1")

            # Auto-connect through storage
            configurator.auto_connect_digester_to_chp("dig1", "chp1")

            # Check storage -> chp connection exists
            connection_exists = any(
                conn.from_component == "dig1_storage" and conn.to_component == "chp1" and conn.connection_type == "gas"
                for conn in configurator.plant.connections
            )
            assert connection_exists, "Storage should be connected to CHP"

    def test_auto_connect_raises_error_if_storage_missing(self, configurator: PlantConfigurator) -> None:
        """
        Test that auto_connect raises error if gas storage not found.

        Args:
            configurator: PlantConfigurator fixture.
        """
        # Add CHP without digester (no storage)
        configurator.add_chp("chp1")

        with pytest.raises(ValueError) as excinfo:
            configurator.auto_connect_digester_to_chp("dig1", "chp1")

        assert "not found" in str(excinfo.value).lower(), "Should mention storage not found"


class TestAutoConnectChpToHeating:
    """Test suite for auto_connect_chp_to_heating method."""

    @pytest.fixture
    def mock_feedstock(self) -> Mock:
        """Create a mock Feedstock object."""
        feedstock = Mock(spec=Feedstock)
        feedstock.mySubstrates = Mock()
        return feedstock

    @pytest.fixture
    def plant(self) -> BiogasPlant:
        """Create a BiogasPlant instance."""
        return BiogasPlant("Test Plant")

    @pytest.fixture
    def configurator(self, plant: BiogasPlant, mock_feedstock: Mock) -> PlantConfigurator:
        """Create a PlantConfigurator instance."""
        return PlantConfigurator(plant, mock_feedstock)

    def test_auto_connect_creates_heat_connection(self, configurator: PlantConfigurator) -> None:
        """
        Test that auto_connect_chp_to_heating creates heat connection.

        Args:
            configurator: PlantConfigurator fixture.
        """
        configurator.add_chp("chp1")
        configurator.add_heating("heat1")

        configurator.auto_connect_chp_to_heating("chp1", "heat1")

        connection_exists = any(
            conn.from_component == "chp1" and conn.to_component == "heat1" and conn.connection_type == "heat"
            for conn in configurator.plant.connections
        )
        assert connection_exists, "CHP should be connected to heating with heat type"


class TestCreateSingleStagePlant:
    """Test suite for create_single_stage_plant method."""

    @pytest.fixture
    def mock_feedstock(self) -> Mock:
        """Create a mock Feedstock object."""
        feedstock = Mock(spec=Feedstock)
        feedstock.mySubstrates = Mock()
        return feedstock

    @pytest.fixture
    def plant(self) -> BiogasPlant:
        """Create a BiogasPlant instance."""
        return BiogasPlant("Test Plant")

    @pytest.fixture
    def configurator(self, plant: BiogasPlant, mock_feedstock: Mock) -> PlantConfigurator:
        """Create a PlantConfigurator instance."""
        return PlantConfigurator(plant, mock_feedstock)

    def test_create_single_stage_returns_dict(self, configurator: PlantConfigurator) -> None:
        """
        Test that create_single_stage_plant returns a dictionary.

        Args:
            configurator: PlantConfigurator fixture.
        """
        with patch("pyadm1.configurator.plant_configurator.get_state_zero_from_initial_state") as mock_get_state:
            mock_get_state.return_value = [0.01] * 37

            result = configurator.create_single_stage_plant()

            assert isinstance(result, dict), "Should return a dictionary"

    def test_create_single_stage_creates_digester(self, configurator: PlantConfigurator) -> None:
        """
        Test that create_single_stage_plant creates a digester.

        Args:
            configurator: PlantConfigurator fixture.
        """
        with patch("pyadm1.configurator.plant_configurator.get_state_zero_from_initial_state") as mock_get_state:
            mock_get_state.return_value = [0.01] * 37

            components = configurator.create_single_stage_plant()

            assert "digester" in components, "Should have digester in components"
            assert components["digester"] in configurator.plant.components, "Digester should be in plant"

    def test_create_single_stage_with_chp(self, configurator: PlantConfigurator) -> None:
        """
        Test that create_single_stage_plant creates CHP if configured.

        Args:
            configurator: PlantConfigurator fixture.
        """
        with patch("pyadm1.configurator.plant_configurator.get_state_zero_from_initial_state") as mock_get_state:
            mock_get_state.return_value = [0.01] * 37

            components = configurator.create_single_stage_plant(chp_config={"P_el_nom": 500})

            assert "chp" in components, "Should have CHP in components"
            assert components["chp"] in configurator.plant.components, "CHP should be in plant"

    def test_create_single_stage_with_heating(self, configurator: PlantConfigurator) -> None:
        """
        Test that create_single_stage_plant creates heating if configured.

        Args:
            configurator: PlantConfigurator fixture.
        """
        with patch("pyadm1.configurator.plant_configurator.get_state_zero_from_initial_state") as mock_get_state:
            mock_get_state.return_value = [0.01] * 37

            components = configurator.create_single_stage_plant(
                chp_config={"P_el_nom": 500}, heating_config={"target_temperature": 308.15}
            )

            assert "heating" in components, "Should have heating in components"
            assert components["heating"] in configurator.plant.components, "Heating should be in plant"

    def test_create_single_stage_auto_connects(self, configurator: PlantConfigurator) -> None:
        """
        Test that create_single_stage_plant auto-connects components.

        Args:
            configurator: PlantConfigurator fixture.
        """
        with patch("pyadm1.configurator.plant_configurator.get_state_zero_from_initial_state") as mock_get_state:
            mock_get_state.return_value = [0.01] * 37

            configurator.create_single_stage_plant(
                chp_config={"P_el_nom": 500}, heating_config={"target_temperature": 308.15}, auto_connect=True
            )

            # Should have connections
            assert len(configurator.plant.connections) > 0, "Should have connections"

    def test_create_single_stage_no_auto_connect(self, configurator: PlantConfigurator) -> None:
        """
        Test create_single_stage_plant without auto-connection.

        Args:
            configurator: PlantConfigurator fixture.
        """
        with patch("pyadm1.configurator.plant_configurator.get_state_zero_from_initial_state") as mock_get_state:
            mock_get_state.return_value = [0.01] * 37

            configurator.create_single_stage_plant(chp_config={"P_el_nom": 500}, auto_connect=False)

            # Should have no connections (only auto-created storage connections)
            # Storage is always auto-connected
            assert all(
                "storage" in conn.to_component or "storage" in conn.from_component or "flare" in conn.to_component
                for conn in configurator.plant.connections
            ), "Only storage/flare connections should exist"

    def test_create_single_stage_with_custom_config(self, configurator: PlantConfigurator) -> None:
        """
        Test create_single_stage_plant with custom configurations.

        Args:
            configurator: PlantConfigurator fixture.
        """
        with patch("pyadm1.configurator.plant_configurator.get_state_zero_from_initial_state") as mock_get_state:
            mock_get_state.return_value = [0.01] * 37

            components = configurator.create_single_stage_plant(
                digester_config={"V_liq": 2500, "T_ad": 318.15}, chp_config={"P_el_nom": 750, "eta_el": 0.42}
            )

            digester_id = components["digester"]
            digester = configurator.plant.components[digester_id]

            assert digester.V_liq == 2500, "Digester should have custom volume"
            assert digester.T_ad == 318.15, "Digester should have custom temperature"


class TestCreateTwoStagePlant:
    """Test suite for create_two_stage_plant method."""

    @pytest.fixture
    def mock_feedstock(self) -> Mock:
        """Create a mock Feedstock object."""
        feedstock = Mock(spec=Feedstock)
        feedstock.mySubstrates = Mock()
        return feedstock

    @pytest.fixture
    def plant(self) -> BiogasPlant:
        """Create a BiogasPlant instance."""
        return BiogasPlant("Test Plant")

    @pytest.fixture
    def configurator(self, plant: BiogasPlant, mock_feedstock: Mock) -> PlantConfigurator:
        """Create a PlantConfigurator instance."""
        return PlantConfigurator(plant, mock_feedstock)

    def test_create_two_stage_returns_dict(self, configurator: PlantConfigurator) -> None:
        """
        Test that create_two_stage_plant returns a dictionary.

        Args:
            configurator: PlantConfigurator fixture.
        """
        with patch("pyadm1.configurator.plant_configurator.get_state_zero_from_initial_state") as mock_get_state:
            mock_get_state.return_value = [0.01] * 37

            result = configurator.create_two_stage_plant()

            assert isinstance(result, dict), "Should return a dictionary"

    def test_create_two_stage_creates_two_digesters(self, configurator: PlantConfigurator) -> None:
        """
        Test that create_two_stage_plant creates two digesters.

        Args:
            configurator: PlantConfigurator fixture.
        """
        with patch("pyadm1.configurator.plant_configurator.get_state_zero_from_initial_state") as mock_get_state:
            mock_get_state.return_value = [0.01] * 37

            components = configurator.create_two_stage_plant()

            assert "hydrolysis" in components, "Should have hydrolysis digester"
            assert "digester" in components, "Should have main digester"

    def test_create_two_stage_connects_digesters_in_series(self, configurator: PlantConfigurator) -> None:
        """
        Test that create_two_stage_plant connects digesters in series.

        Args:
            configurator: PlantConfigurator fixture.
        """
        with patch("pyadm1.configurator.plant_configurator.get_state_zero_from_initial_state") as mock_get_state:
            mock_get_state.return_value = [0.01] * 37

            components = configurator.create_two_stage_plant()

            hydro_id = components["hydrolysis"]
            main_id = components["digester"]

            # Check liquid connection between digesters
            connection_exists = any(
                conn.from_component == hydro_id and conn.to_component == main_id and conn.connection_type == "liquid"
                for conn in configurator.plant.connections
            )
            assert connection_exists, "Digesters should be connected in series with liquid"

    def test_create_two_stage_with_thermophilic_hydrolysis(self, configurator: PlantConfigurator) -> None:
        """
        Test create_two_stage_plant with thermophilic hydrolysis.

        Args:
            configurator: PlantConfigurator fixture.
        """
        with patch("pyadm1.configurator.plant_configurator.get_state_zero_from_initial_state") as mock_get_state:
            mock_get_state.return_value = [0.01] * 37

            components = configurator.create_two_stage_plant(hydrolysis_config={"T_ad": 318.15})  # 45°C

            hydro_id = components["hydrolysis"]
            hydrolysis = configurator.plant.components[hydro_id]

            assert hydrolysis.T_ad == 318.15, "Hydrolysis should be thermophilic"

    def test_create_two_stage_with_chp(self, configurator: PlantConfigurator) -> None:
        """
        Test create_two_stage_plant with CHP.

        Args:
            configurator: PlantConfigurator fixture.
        """
        with patch("pyadm1.configurator.plant_configurator.get_state_zero_from_initial_state") as mock_get_state:
            mock_get_state.return_value = [0.01] * 37

            components = configurator.create_two_stage_plant(chp_config={"P_el_nom": 500})

            assert "chp" in components, "Should have CHP"

    def test_create_two_stage_connects_both_digesters_to_chp(self, configurator: PlantConfigurator) -> None:
        """
        Test that both digesters are connected to CHP via storage.

        Args:
            configurator: PlantConfigurator fixture.
        """
        with patch("pyadm1.configurator.plant_configurator.get_state_zero_from_initial_state") as mock_get_state:
            mock_get_state.return_value = [0.01] * 37

            components = configurator.create_two_stage_plant(chp_config={"P_el_nom": 500})

            chp_id = components["chp"]

            # Both storages should connect to CHP
            storage_connections = [
                conn
                for conn in configurator.plant.connections
                if "storage" in conn.from_component and conn.to_component == chp_id
            ]

            assert len(storage_connections) >= 2, "Both digesters' storages should connect to CHP"

    def test_create_two_stage_with_multiple_heating_systems(self, configurator: PlantConfigurator) -> None:
        """
        Test create_two_stage_plant with multiple heating systems.

        Args:
            configurator: PlantConfigurator fixture.
        """
        with patch("pyadm1.configurator.plant_configurator.get_state_zero_from_initial_state") as mock_get_state:
            mock_get_state.return_value = [0.01] * 37

            components = configurator.create_two_stage_plant(
                chp_config={"P_el_nom": 500}, heating_configs=[{"target_temperature": 318.15}, {"target_temperature": 308.15}]
            )

            assert "heating" in components, "Should have heating systems"
            assert isinstance(components["heating"], list), "Heating should be a list"
            assert len(components["heating"]) == 2, "Should have 2 heating systems"

    def test_create_two_stage_with_custom_volumes(self, configurator: PlantConfigurator) -> None:
        """
        Test create_two_stage_plant with custom volumes.

        Args:
            configurator: PlantConfigurator fixture.
        """
        with patch("pyadm1.configurator.plant_configurator.get_state_zero_from_initial_state") as mock_get_state:
            mock_get_state.return_value = [0.01] * 37

            components = configurator.create_two_stage_plant(hydrolysis_config={"V_liq": 500}, digester_config={"V_liq": 1500})

            hydro_id = components["hydrolysis"]
            main_id = components["digester"]

            hydrolysis = configurator.plant.components[hydro_id]
            main_digester = configurator.plant.components[main_id]

            assert hydrolysis.V_liq == 500, "Hydrolysis should have custom volume"
            assert main_digester.V_liq == 1500, "Main digester should have custom volume"


class TestPlantConfiguratorIntegration:
    """Integration tests for PlantConfigurator."""

    @pytest.fixture
    def mock_feedstock(self) -> Mock:
        """Create a mock Feedstock object."""
        feedstock = Mock(spec=Feedstock)
        feedstock.mySubstrates = Mock()
        return feedstock

    @pytest.fixture
    def plant(self) -> BiogasPlant:
        """Create a BiogasPlant instance."""
        return BiogasPlant("Integration Test Plant")

    @pytest.fixture
    def configurator(self, plant: BiogasPlant, mock_feedstock: Mock) -> PlantConfigurator:
        """Create a PlantConfigurator instance."""
        return PlantConfigurator(plant, mock_feedstock)

    def test_build_complete_single_stage_plant(self, configurator: PlantConfigurator) -> None:
        """
        Test building a complete single-stage plant.

        Args:
            configurator: PlantConfigurator fixture.
        """
        with patch("pyadm1.configurator.plant_configurator.get_state_zero_from_initial_state") as mock_get_state:
            mock_get_state.return_value = [0.01] * 37

            components = configurator.create_single_stage_plant(
                digester_config={"V_liq": 2000, "V_gas": 300},
                chp_config={"P_el_nom": 500, "eta_el": 0.40},
                heating_config={"target_temperature": 308.15},
            )

            # Verify all components exist
            assert len(configurator.plant.components) >= 5, "Should have digester, storage, CHP, flare, heating"
            assert len(components) >= 5, "Should have digester, storage, CHP, flare, heating"

            # Verify connections exist
            assert len(configurator.plant.connections) > 0, "Should have connections"

    def test_build_complete_two_stage_plant(self, configurator: PlantConfigurator) -> None:
        """
        Test building a complete two-stage plant.

        Args:
            configurator: PlantConfigurator fixture.
        """
        with patch("pyadm1.configurator.plant_configurator.get_state_zero_from_initial_state") as mock_get_state:
            mock_get_state.return_value = [0.01] * 37

            components = configurator.create_two_stage_plant(
                hydrolysis_config={"V_liq": 500, "T_ad": 318.15},
                digester_config={"V_liq": 1500, "T_ad": 308.15},
                chp_config={"P_el_nom": 500},
            )

            # Verify all components exist
            # 2 digesters + 2 storages + 1 CHP + 1 flare = 6
            assert len(configurator.plant.components) >= 6, "Should have all required components"
            assert len(components) >= 6, "Should have all required components"

    def test_manual_plant_building(self, configurator: PlantConfigurator) -> None:
        """
        Test manually building a plant step-by-step.

        Args:
            configurator: PlantConfigurator fixture.
        """
        with patch("pyadm1.configurator.plant_configurator.get_state_zero_from_initial_state") as mock_get_state:
            mock_get_state.return_value = [0.01] * 37

            # Add components manually
            digester, _ = configurator.add_digester("dig1", V_liq=2000)
            configurator.add_chp("chp1", P_el_nom=500)
            configurator.add_heating("heat1")

            # Connect manually
            configurator.auto_connect_digester_to_chp("dig1", "chp1")
            configurator.auto_connect_chp_to_heating("chp1", "heat1")

            # Verify structure
            assert "dig1" in configurator.plant.components
            assert "chp1" in configurator.plant.components
            assert "heat1" in configurator.plant.components
            assert len(configurator.plant.connections) > 0

    def test_complex_plant_configuration(self, configurator: PlantConfigurator) -> None:
        """
        Test building a complex plant with multiple digesters and CHPs.

        Args:
            configurator: PlantConfigurator fixture.
        """
        with patch("pyadm1.configurator.plant_configurator.get_state_zero_from_initial_state") as mock_get_state:
            mock_get_state.return_value = [0.01] * 37

            # Create two-stage plant
            components = configurator.create_two_stage_plant(
                hydrolysis_config={"V_liq": 500}, digester_config={"V_liq": 1500}, chp_config={"P_el_nom": 500}
            )

            # Add additional digester
            extra_digester, _ = configurator.add_digester("dig_extra", V_liq=1000)

            # Connect to existing CHP
            chp_id = components["chp"]
            configurator.auto_connect_digester_to_chp("dig_extra", chp_id)

            # Verify complex structure
            assert len(configurator.plant.components) >= 7, "Should have 3 digesters + storages + CHP + flare"


class TestPlantConfiguratorEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def mock_feedstock(self) -> Mock:
        """Create a mock Feedstock object."""
        feedstock = Mock(spec=Feedstock)
        feedstock.mySubstrates = Mock()
        return feedstock

    @pytest.fixture
    def plant(self) -> BiogasPlant:
        """Create a BiogasPlant instance."""
        return BiogasPlant("Test Plant")

    @pytest.fixture
    def configurator(self, plant: BiogasPlant, mock_feedstock: Mock) -> PlantConfigurator:
        """Create a PlantConfigurator instance."""
        return PlantConfigurator(plant, mock_feedstock)

    def test_add_digester_with_zero_volumes(self, configurator: PlantConfigurator) -> None:
        """
        Test that zero volumes are handled (though not realistic).

        Args:
            configurator: PlantConfigurator fixture.
        """
        with patch("pyadm1.configurator.plant_configurator.get_state_zero_from_initial_state") as mock_get_state:
            mock_get_state.return_value = [0.01] * 37

            # This should work even with unrealistic values
            digester, _ = configurator.add_digester("dig1", V_liq=0.1, V_gas=0.1)

            assert digester.V_liq == 0.1
            assert digester.V_gas == 0.1

    def test_add_chp_with_zero_power(self, configurator: PlantConfigurator) -> None:
        """
        Test adding CHP with zero power.

        Args:
            configurator: PlantConfigurator fixture.
        """
        chp = configurator.add_chp("chp1", P_el_nom=0.0)

        assert chp.P_el_nom == 0.0, "Should accept zero power"

    def test_auto_connect_nonexistent_digester(self, configurator: PlantConfigurator) -> None:
        """
        Test auto-connecting non-existent digester raises error.

        Args:
            configurator: PlantConfigurator fixture.
        """
        configurator.add_chp("chp1")

        with pytest.raises(ValueError):
            configurator.auto_connect_digester_to_chp("nonexistent", "chp1")

    def test_create_plant_with_empty_configs(self, configurator: PlantConfigurator) -> None:
        """
        Test creating plant with empty configuration dictionaries.

        Args:
            configurator: PlantConfigurator fixture.
        """
        with patch("pyadm1.configurator.plant_configurator.get_state_zero_from_initial_state") as mock_get_state:
            mock_get_state.return_value = [0.01] * 37

            # Empty dicts should use defaults
            components = configurator.create_single_stage_plant(digester_config={}, chp_config={}, heating_config={})

            assert "digester" in components
            assert "chp" in components
            assert "heating" in components
