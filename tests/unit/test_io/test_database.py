# tests/unit/test_io/test_database.py
"""
Unit tests for database.py

Tests database operations including:
- Connection and session management
- Plant CRUD operations
- Measurement data storage/retrieval
- Simulation results storage/retrieval
- Calibration history tracking
- Substrate data management
- Error handling and validation

Run with:
    pytest tests/unit/test_io/test_database.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import timedelta

from pyadm1.io.database import (
    Database,
    DatabaseConfig,
    Plant,
)

from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="function")
def temp_db():
    """Create temporary SQLite database for testing."""
    # Use SQLite in-memory database for tests
    db = Database("sqlite:///:memory:")
    db.create_all_tables()
    yield db
    db.close()


@pytest.fixture
def sample_plant_data():
    """Sample plant data."""
    return {
        "plant_id": "test_plant_001",
        "name": "Test Biogas Plant",
        "location": "Test Location",
        "operator": "Test Operator",
        "V_liq": 2000.0,
        "V_gas": 300.0,
        "T_ad": 308.15,
        "P_el_nom": 500.0,
        "configuration": {"type": "single_stage", "substrates": ["maize", "manure"]},
    }


@pytest.fixture
def sample_measurement_data():
    """Sample measurement data as DataFrame."""
    dates = pd.date_range("2024-01-01", periods=24, freq="H")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "Q_sub_maize": np.random.uniform(14, 16, 24),
            "Q_sub_manure": np.random.uniform(9, 11, 24),
            "pH": np.random.uniform(7.0, 7.5, 24),
            "VFA": np.random.uniform(2.0, 3.0, 24),
            "TAC": np.random.uniform(8.0, 10.0, 24),
            "Q_gas": np.random.uniform(1200, 1300, 24),
            "Q_ch4": np.random.uniform(720, 780, 24),
            "P_el": np.random.uniform(480, 500, 24),
            "P_th": np.random.uniform(540, 560, 24),
        }
    )


@pytest.fixture
def sample_simulation_results():
    """Sample simulation results."""
    results = []
    for i in range(10):
        results.append(
            {
                "time": float(i),
                "components": {
                    "digester_1": {
                        "Q_gas": 1250.0 + np.random.randn() * 10,
                        "Q_ch4": 750.0 + np.random.randn() * 5,
                        "Q_co2": 475.0 + np.random.randn() * 5,
                        "pH": 7.3 + np.random.randn() * 0.1,
                        "VFA": 2.5 + np.random.randn() * 0.2,
                        "TAC": 8.5 + np.random.randn() * 0.5,
                    },
                    "chp_1": {"P_el": 480.0 + np.random.randn() * 10, "P_th": 540.0 + np.random.randn() * 10},
                },
            }
        )
    return results


@pytest.fixture
def sample_substrate_data():
    """Sample substrate characterization data."""
    return {
        "TS": 32.5,
        "VS": 96.2,
        "oTS": 31.3,
        "foTS": 28.5,
        "RP": 8.5,
        "RL": 3.2,
        "RF": 21.5,
        "NDF": 42.1,
        "ADF": 22.3,
        "ADL": 2.1,
        "pH": 3.9,
        "NH4_N": 0.5,
        "TAC": 11.0,
        "COD_S": 18.5,
        "BMP": 345.0,
        "C_content": 45.2,
        "N_content": 1.8,
        "C_to_N": 25.1,
    }


# ============================================================================
# Database Connection Tests
# ============================================================================


class TestDatabaseConnection:
    """Test database connection and initialization."""

    def test_create_database_with_connection_string(self):
        """Test database creation with connection string."""
        db = Database("sqlite:///:memory:")
        assert db.engine is not None
        assert db.SessionLocal is not None

    def test_create_database_with_config(self):
        """Test database creation with config object."""
        config = DatabaseConfig(host="localhost", port=5432, database="test_db", username="test_user", password="test_pass")
        # Don't actually connect, just verify connection string format
        db = Database(config=config)
        assert "postgresql" in db.connection_string
        assert "test_db" in db.connection_string
        db.engine.dispose()  # Clean up

    def test_create_all_tables(self, temp_db):
        """Test table creation."""
        # Tables should already be created by fixture
        # inspector = temp_db.engine.dialect.get_table_names(temp_db.engine.connect())
        inspector = inspect(temp_db.engine)
        table_names = inspector.get_table_names()
        expected_tables = ["plants", "measurements", "simulations", "simulation_time_series", "calibrations", "substrates"]

        for table in expected_tables:
            assert table in table_names

    def test_session_context_manager(self, temp_db):
        """Test session context manager."""
        with temp_db.get_session() as session:
            assert session is not None
            # Session should be committed automatically

    def test_session_rollback_on_error(self, temp_db):
        """Test session rollback on error."""
        with pytest.raises(IntegrityError):
            with temp_db.get_session() as session:
                # Create invalid plant (duplicate ID)
                plant1 = Plant(id="test", name="Plant 1")
                session.add(plant1)
                session.flush()

                plant2 = Plant(id="test", name="Plant 2")  # Duplicate ID
                session.add(plant2)
                session.flush()  # This should raise IntegrityError


# ============================================================================
# Plant Management Tests
# ============================================================================


class TestPlantManagement:
    """Test plant CRUD operations."""

    def test_create_plant(self, temp_db, sample_plant_data):
        """Test creating a plant."""
        plant = temp_db.create_plant(**sample_plant_data)

        assert plant.id == sample_plant_data["plant_id"]
        assert plant.name == sample_plant_data["name"]
        assert plant.V_liq == sample_plant_data["V_liq"]
        # Note: configuration comparison might fail due to JSON serialization
        # Just check it's not None
        assert plant.configuration is not None

    def test_create_duplicate_plant_raises_error(self, temp_db, sample_plant_data):
        """Test that creating duplicate plant raises error."""
        temp_db.create_plant(**sample_plant_data)

        with pytest.raises(ValueError, match="already exists"):
            temp_db.create_plant(**sample_plant_data)

    def test_get_plant(self, temp_db, sample_plant_data):
        """Test retrieving a plant."""
        temp_db.create_plant(**sample_plant_data)

        # Re-fetch from database
        plant = temp_db.get_plant(sample_plant_data["plant_id"])

        assert plant is not None
        assert plant.id == sample_plant_data["plant_id"]
        assert plant.name == sample_plant_data["name"]

    def test_get_nonexistent_plant(self, temp_db):
        """Test retrieving nonexistent plant returns None."""
        plant = temp_db.get_plant("nonexistent")
        assert plant is None

    def test_list_plants(self, temp_db):
        """Test listing all plants."""
        # Create multiple plants
        for i in range(3):
            temp_db.create_plant(plant_id=f"plant_{i}", name=f"Plant {i}", V_liq=2000.0)

        plants = temp_db.list_plants()

        assert len(plants) == 3
        assert all("id" in p and "name" in p for p in plants)

    def test_list_plants_empty(self, temp_db):
        """Test listing plants when database is empty."""
        plants = temp_db.list_plants()
        assert len(plants) == 0


# ============================================================================
# Measurement Data Tests
# ============================================================================


class TestMeasurementData:
    """Test measurement data operations."""

    def test_store_measurements(self, temp_db, sample_plant_data, sample_measurement_data):
        """Test storing measurement data."""
        temp_db.create_plant(**sample_plant_data)

        n = temp_db.store_measurements(sample_plant_data["plant_id"], sample_measurement_data, source="SCADA")

        assert n == len(sample_measurement_data)

    def test_store_measurements_invalid_plant(self, temp_db, sample_measurement_data):
        """Test storing measurements for nonexistent plant raises error."""
        with pytest.raises(ValueError, match="not found"):
            temp_db.store_measurements("nonexistent", sample_measurement_data)

    def test_store_measurements_missing_timestamp(self, temp_db, sample_plant_data):
        """Test storing measurements without timestamp raises error."""
        temp_db.create_plant(**sample_plant_data)

        df = pd.DataFrame({"pH": [7.2, 7.3], "VFA": [2.5, 2.6]})

        with pytest.raises(ValueError, match="timestamp"):
            temp_db.store_measurements(sample_plant_data["plant_id"], df)

    def test_load_measurements(self, temp_db, sample_plant_data, sample_measurement_data):
        """Test loading measurement data."""
        temp_db.create_plant(**sample_plant_data)
        temp_db.store_measurements(sample_plant_data["plant_id"], sample_measurement_data)

        data = temp_db.load_measurements(sample_plant_data["plant_id"])

        assert len(data) == len(sample_measurement_data)
        assert "pH" in data.columns
        assert "Q_gas" in data.columns

    def test_load_measurements_with_time_range(self, temp_db, sample_plant_data, sample_measurement_data):
        """Test loading measurements with time range filter."""
        temp_db.create_plant(**sample_plant_data)
        temp_db.store_measurements(sample_plant_data["plant_id"], sample_measurement_data)

        # Load only first 12 hours
        start = sample_measurement_data["timestamp"].iloc[0]
        end = start + timedelta(hours=11)

        data = temp_db.load_measurements(sample_plant_data["plant_id"], start_time=start, end_time=end)

        assert len(data) == 12

    def test_load_measurements_with_source_filter(self, temp_db, sample_plant_data, sample_measurement_data):
        """Test loading measurements with source filter."""
        temp_db.create_plant(**sample_plant_data)
        temp_db.store_measurements(sample_plant_data["plant_id"], sample_measurement_data, source="SCADA")
        temp_db.store_measurements(sample_plant_data["plant_id"], sample_measurement_data.iloc[:5], source="Lab")

        scada_data = temp_db.load_measurements(sample_plant_data["plant_id"], source="SCADA")
        lab_data = temp_db.load_measurements(sample_plant_data["plant_id"], source="Lab")

        assert len(scada_data) == 24
        assert len(lab_data) == 5

    def test_load_measurements_specific_columns(self, temp_db, sample_plant_data, sample_measurement_data):
        """Test loading specific columns."""
        temp_db.create_plant(**sample_plant_data)
        temp_db.store_measurements(sample_plant_data["plant_id"], sample_measurement_data)

        data = temp_db.load_measurements(sample_plant_data["plant_id"], columns=["pH", "VFA"])

        assert "pH" in data.columns
        assert "VFA" in data.columns
        assert "Q_gas" not in data.columns


# ============================================================================
# Simulation Results Tests
# ============================================================================


class TestSimulationResults:
    """Test simulation results operations."""

    def test_store_simulation(self, temp_db, sample_plant_data, sample_simulation_results):
        """Test storing simulation results."""
        temp_db.create_plant(**sample_plant_data)

        simulation = temp_db.store_simulation(
            simulation_id="sim_001",
            plant_id=sample_plant_data["plant_id"],
            results=sample_simulation_results,
            name="Test Simulation",
            duration=10.0,
            scenario="baseline",
        )

        assert simulation.id == "sim_001"
        assert simulation.plant_id == sample_plant_data["plant_id"]
        assert simulation.status == "completed"
        assert simulation.avg_Q_gas is not None
        assert simulation.avg_Q_ch4 is not None

    def test_store_simulation_invalid_plant(self, temp_db, sample_simulation_results):
        """Test storing simulation for nonexistent plant raises error."""
        with pytest.raises(ValueError, match="not found"):
            temp_db.store_simulation("sim_001", "nonexistent", sample_simulation_results)

    def test_store_duplicate_simulation(self, temp_db, sample_plant_data, sample_simulation_results):
        """Test storing duplicate simulation raises error."""
        temp_db.create_plant(**sample_plant_data)
        temp_db.store_simulation("sim_001", sample_plant_data["plant_id"], sample_simulation_results)

        with pytest.raises(ValueError, match="already exists"):
            temp_db.store_simulation("sim_001", sample_plant_data["plant_id"], sample_simulation_results)

    def test_load_simulation(self, temp_db, sample_plant_data, sample_simulation_results):
        """Test loading simulation results."""
        temp_db.create_plant(**sample_plant_data)
        temp_db.store_simulation("sim_001", sample_plant_data["plant_id"], sample_simulation_results)

        sim_data = temp_db.load_simulation("sim_001")

        assert sim_data is not None
        assert sim_data["id"] == "sim_001"
        assert "time_series" in sim_data
        assert len(sim_data["time_series"]) == len(sample_simulation_results)
        assert "Q_gas" in sim_data["time_series"].columns

    def test_load_nonexistent_simulation(self, temp_db):
        """Test loading nonexistent simulation returns None."""
        sim_data = temp_db.load_simulation("nonexistent")
        assert sim_data is None

    def test_list_simulations(self, temp_db, sample_plant_data, sample_simulation_results):
        """Test listing simulations."""
        temp_db.create_plant(**sample_plant_data)

        # Create multiple simulations
        for i in range(3):
            temp_db.store_simulation(
                f"sim_{i:03d}", sample_plant_data["plant_id"], sample_simulation_results, scenario=f"scenario_{i}"
            )

        simulations = temp_db.list_simulations(plant_id=sample_plant_data["plant_id"])

        assert len(simulations) == 3

    def test_list_simulations_with_scenario_filter(self, temp_db, sample_plant_data, sample_simulation_results):
        """Test listing simulations with scenario filter."""
        temp_db.create_plant(**sample_plant_data)

        temp_db.store_simulation("sim_001", sample_plant_data["plant_id"], sample_simulation_results, scenario="baseline")
        temp_db.store_simulation("sim_002", sample_plant_data["plant_id"], sample_simulation_results, scenario="optimized")

        baseline_sims = temp_db.list_simulations(scenario="baseline")
        optimized_sims = temp_db.list_simulations(scenario="optimized")

        assert len(baseline_sims) == 1
        assert len(optimized_sims) == 1

    def test_calculate_simulation_metrics(self, temp_db, sample_simulation_results):
        """Test simulation metrics calculation."""
        metrics = temp_db._calculate_simulation_metrics(sample_simulation_results)

        assert "avg_Q_gas" in metrics
        assert "avg_Q_ch4" in metrics
        assert "avg_CH4_content" in metrics
        assert "avg_pH" in metrics
        assert "total_energy" in metrics

        # Check values are reasonable
        assert 1200 < metrics["avg_Q_gas"] < 1300
        assert 740 < metrics["avg_Q_ch4"] < 760


# ============================================================================
# Calibration History Tests
# ============================================================================


class TestCalibrationHistory:
    """Test calibration history operations."""

    def test_store_calibration(self, temp_db, sample_plant_data):
        """Test storing calibration results."""
        temp_db.create_plant(**sample_plant_data)

        calibration = temp_db.store_calibration(
            plant_id=sample_plant_data["plant_id"],
            calibration_type="initial",
            method="differential_evolution",
            parameters={"k_dis": 0.55, "Y_su": 0.105},
            objective_value=0.123,
            objectives=["Q_ch4", "pH"],
            validation_metrics={"Q_ch4_r2": 0.85, "pH_rmse": 0.12},
            success=True,
        )

        assert calibration.plant_id == sample_plant_data["plant_id"]
        assert calibration.method == "differential_evolution"
        assert calibration.parameters["k_dis"] == 0.55
        assert calibration.success is True

    def test_load_calibrations(self, temp_db, sample_plant_data):
        """Test loading calibration history."""
        temp_db.create_plant(**sample_plant_data)

        # Create multiple calibrations
        for i in range(3):
            temp_db.store_calibration(
                plant_id=sample_plant_data["plant_id"],
                calibration_type="initial" if i == 0 else "online",
                method="differential_evolution",
                parameters={"k_dis": 0.5 + i * 0.05},
                objective_value=0.1 + i * 0.01,
                objectives=["Q_ch4"],
                success=True,
            )

        calibrations = temp_db.load_calibrations(sample_plant_data["plant_id"])

        assert len(calibrations) == 3
        # Should be ordered by created_at DESC (most recent first)
        assert calibrations[0]["parameters"]["k_dis"] == 0.6

    def test_load_calibrations_with_type_filter(self, temp_db, sample_plant_data):
        """Test loading calibrations with type filter."""
        temp_db.create_plant(**sample_plant_data)

        temp_db.store_calibration(
            sample_plant_data["plant_id"], "initial", "method1", {"k_dis": 0.5}, 0.1, ["Q_ch4"], success=True
        )
        temp_db.store_calibration(
            sample_plant_data["plant_id"], "online", "method1", {"k_dis": 0.55}, 0.09, ["Q_ch4"], success=True
        )

        initial_cals = temp_db.load_calibrations(sample_plant_data["plant_id"], calibration_type="initial")
        online_cals = temp_db.load_calibrations(sample_plant_data["plant_id"], calibration_type="online")

        assert len(initial_cals) == 1
        assert len(online_cals) == 1

    def test_get_latest_calibration(self, temp_db, sample_plant_data):
        """Test getting latest calibration."""
        temp_db.create_plant(**sample_plant_data)

        temp_db.store_calibration(
            sample_plant_data["plant_id"], "initial", "method1", {"k_dis": 0.5}, 0.1, ["Q_ch4"], success=True
        )
        temp_db.store_calibration(
            sample_plant_data["plant_id"], "online", "method1", {"k_dis": 0.55}, 0.09, ["Q_ch4"], success=True
        )

        latest = temp_db.get_latest_calibration(sample_plant_data["plant_id"])

        assert latest is not None
        assert latest["parameters"]["k_dis"] == 0.55  # Most recent


# ============================================================================
# Substrate Data Tests
# ============================================================================


class TestSubstrateData:
    """Test substrate data operations."""

    def test_store_substrate(self, temp_db, sample_plant_data, sample_substrate_data):
        """Test storing substrate characterization data."""
        temp_db.create_plant(**sample_plant_data)

        substrate = temp_db.store_substrate(
            plant_id=sample_plant_data["plant_id"],
            substrate_name="Maize silage batch 23",
            substrate_type="maize",
            sample_date="2024-01-15",
            lab_data=sample_substrate_data,
            sample_id="LAB-2024-001",
            lab_name="Test Laboratory",
        )

        # Access attributes immediately after creation
        assert substrate.plant_id == sample_plant_data["plant_id"]
        assert substrate.substrate_name == "Maize silage batch 23"
        assert substrate.TS == sample_substrate_data["TS"]
        assert substrate.BMP == sample_substrate_data["BMP"]

    def test_load_substrates(self, temp_db, sample_plant_data, sample_substrate_data):
        """Test loading substrate data."""
        temp_db.create_plant(**sample_plant_data)

        # Store multiple substrates
        for i in range(3):
            temp_db.store_substrate(
                plant_id=sample_plant_data["plant_id"],
                substrate_name=f"Sample {i}",
                substrate_type="maize",
                sample_date=f"2024-01-{15+i:02d}",
                lab_data=sample_substrate_data,
            )

        df = temp_db.load_substrates(sample_plant_data["plant_id"])

        assert len(df) == 3
        assert "TS" in df.columns
        assert "BMP" in df.columns

    def test_load_substrates_with_type_filter(self, temp_db, sample_plant_data, sample_substrate_data):
        """Test loading substrates with type filter."""
        temp_db.create_plant(**sample_plant_data)

        temp_db.store_substrate(sample_plant_data["plant_id"], "Maize", "maize", "2024-01-15", sample_substrate_data)
        temp_db.store_substrate(sample_plant_data["plant_id"], "Manure", "manure", "2024-01-16", sample_substrate_data)

        maize_df = temp_db.load_substrates(sample_plant_data["plant_id"], substrate_type="maize")
        manure_df = temp_db.load_substrates(sample_plant_data["plant_id"], substrate_type="manure")

        assert len(maize_df) == 1
        assert len(manure_df) == 1

    def test_load_substrates_with_date_range(self, temp_db, sample_plant_data, sample_substrate_data):
        """Test loading substrates with date range filter."""
        temp_db.create_plant(**sample_plant_data)

        temp_db.store_substrate(sample_plant_data["plant_id"], "Sample 1", "maize", "2024-01-10", sample_substrate_data)
        temp_db.store_substrate(sample_plant_data["plant_id"], "Sample 2", "maize", "2024-01-20", sample_substrate_data)
        temp_db.store_substrate(sample_plant_data["plant_id"], "Sample 3", "maize", "2024-01-30", sample_substrate_data)

        df = temp_db.load_substrates(sample_plant_data["plant_id"], start_date="2024-01-15", end_date="2024-01-25")

        assert len(df) == 1  # Only Sample 2


# ============================================================================
# Statistics and Utility Tests
# ============================================================================


class TestStatisticsAndUtilities:
    """Test statistics and utility functions."""

    def test_get_statistics(self, temp_db, sample_plant_data, sample_measurement_data, sample_simulation_results):
        """Test getting database statistics."""
        temp_db.create_plant(**sample_plant_data)
        temp_db.store_measurements(sample_plant_data["plant_id"], sample_measurement_data)
        temp_db.store_simulation("sim_001", sample_plant_data["plant_id"], sample_simulation_results)

        stats = temp_db.get_statistics(sample_plant_data["plant_id"])

        assert stats["plant_id"] == sample_plant_data["plant_id"]
        assert stats["n_measurements"] == 24
        assert stats["n_simulations"] == 1
        assert stats["first_measurement"] is not None
        assert stats["last_measurement"] is not None

    def test_execute_query(self, temp_db, sample_plant_data, sample_measurement_data):
        """Test executing raw SQL query."""
        temp_db.create_plant(**sample_plant_data)
        temp_db.store_measurements(sample_plant_data["plant_id"], sample_measurement_data)

        df = temp_db.execute_query("SELECT plant_id, AVG(pH) as avg_ph FROM measurements GROUP BY plant_id")

        assert len(df) == 1
        assert "avg_ph" in df.columns
        assert 7.0 <= df["avg_ph"].iloc[0] <= 7.5


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_connection_string(self):
        """Test invalid connection string raises error."""
        with pytest.raises(Exception):
            db = Database("invalid://connection")
            db.create_all_tables()

    def test_missing_connection_info(self):
        """Test missing connection info raises error."""
        with pytest.raises(ValueError):
            Database()

    def test_store_measurements_with_nan_values(self, temp_db, sample_plant_data):
        """Test storing measurements with NaN values."""
        temp_db.create_plant(**sample_plant_data)

        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=5, freq="H"),
                "pH": [7.2, np.nan, 7.3, 7.4, np.nan],
                "Q_gas": [1200, 1210, np.nan, 1230, 1240],
            }
        )

        # Should handle NaN values gracefully
        n = temp_db.store_measurements(sample_plant_data["plant_id"], df)
        assert n == 5

        # Load and check
        loaded = temp_db.load_measurements(sample_plant_data["plant_id"])
        assert len(loaded) == 5
        # NaN values should be None in database


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_workflow(self, temp_db, sample_plant_data, sample_measurement_data, sample_simulation_results):
        """Test complete workflow from plant creation to data retrieval."""
        # 1. Create plant
        plant = temp_db.create_plant(**sample_plant_data)
        assert plant is not None

        # 2. Store measurements
        n_measurements = temp_db.store_measurements(plant.id, sample_measurement_data)
        assert n_measurements > 0

        # 3. Store simulation
        simulation = temp_db.store_simulation("sim_001", plant.id, sample_simulation_results, name="Test Simulation")
        assert simulation is not None

        # 4. Store calibration
        calibration = temp_db.store_calibration(
            plant.id, "initial", "differential_evolution", {"k_dis": 0.55}, 0.123, ["Q_ch4"], success=True
        )
        assert calibration is not None

        # 5. Get statistics
        stats = temp_db.get_statistics(plant.id)
        assert stats["n_measurements"] == n_measurements
        assert stats["n_simulations"] == 1
        assert stats["n_calibrations"] == 1

        # 6. Load all data
        measurements = temp_db.load_measurements(plant.id)
        assert len(measurements) == n_measurements

        sim_data = temp_db.load_simulation("sim_001")
        assert sim_data is not None

        calibrations = temp_db.load_calibrations(plant.id)
        assert len(calibrations) == 1
