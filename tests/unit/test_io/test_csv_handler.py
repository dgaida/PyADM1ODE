# tests/unit/test_io/test_csv_handler.py
"""
Unit tests for csv_handler.py

Tests CSV import/export functionality including:
- Substrate laboratory data loading
- Measurement data import/export
- Simulation results export
- Column name mapping (German/English)
- Data validation
- Template generation

Run with:
    pytest tests/unit/test_io/test_csv_handler.py -v
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os

from pyadm1.io.csv_handler import CSVHandler


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def handler():
    """Create CSV handler instance."""
    return CSVHandler()


@pytest.fixture
def temp_csv_file():
    """Create temporary CSV file."""
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def sample_substrate_data_horizontal():
    """Sample substrate data in horizontal format (one row per sample)."""
    return pd.DataFrame(
        {
            "substrate_name": ["Maize silage"],
            "substrate_type": ["maize"],
            "sample_date": ["2024-01-15"],
            "TS": [32.5],
            "VS": [96.2],
            "oTS": [31.3],
            "foTS": [28.5],
            "RP": [8.5],
            "RL": [3.2],
            "RF": [21.5],
            "NDF": [42.1],
            "ADF": [22.3],
            "ADL": [2.1],
            "pH": [3.9],
            "NH4_N": [0.5],
            "TAC": [11.0],
            "COD_S": [18.5],
            "BMP": [345.0],
            "C_content": [45.2],
            "N_content": [1.8],
            "C_to_N": [25.1],
        }
    )


@pytest.fixture
def sample_substrate_data_vertical():
    """Sample substrate data in vertical format (parameter, value, unit)."""
    return pd.DataFrame(
        {
            "Parameter": ["TS", "VS", "RP", "RL", "NDF", "ADF", "ADL", "pH", "NH4-N", "TAC", "BMP"],
            "Value": [32.5, 96.2, 8.5, 3.2, 42.1, 22.3, 2.1, 3.9, 0.5, 11.0, 345.0],
            "Unit": ["% FM", "% TS", "% TS", "% TS", "% TS", "% TS", "% TS", "-", "g/L", "mmol/L", "L CH4/kg oTS"],
        }
    )


@pytest.fixture
def sample_substrate_data_german():
    """Sample substrate data with German column names."""
    return pd.DataFrame(
        {
            "Substratname": ["Maissilage"],
            "Substrattyp": ["maize"],
            "Probendatum": ["2024-01-15"],
            "Trockensubstanzgehalt": [32.5],
            "Organische Trockensubstanz": [96.2],
            "Rohprotein": [8.5],
            "Rohfett": [3.2],
            "Rohfaser": [21.5],
            "pH-Wert": [3.9],
            "Ammoniumstickstoff": [0.5],
            "Alkalinität": [11.0],
            "Biochemisches Methanpotential": [345.0],
        }
    )


@pytest.fixture
def sample_measurement_data():
    """Sample measurement time series data."""
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
                        "Q_gas": 1250.0,
                        "Q_ch4": 750.0,
                        "Q_co2": 475.0,
                        "pH": 7.3,
                        "VFA": 2.5,
                        "TAC": 8.5,
                    },
                    "chp_1": {"P_el": 480.0, "P_th": 540.0},
                },
            }
        )
    return results


# ============================================================================
# Substrate Data Loading Tests
# ============================================================================


class TestSubstrateDataLoading:
    """Test substrate laboratory data loading."""

    def test_load_horizontal_format(self, handler, temp_csv_file, sample_substrate_data_horizontal):
        """Test loading substrate data in horizontal format."""
        sample_substrate_data_horizontal.to_csv(temp_csv_file, index=False)

        data = handler.load_substrate_lab_data(
            temp_csv_file, substrate_name="Maize silage", substrate_type="maize", validate=False
        )

        assert data["substrate_name"] == "Maize silage"
        assert data["TS"] == 32.5
        assert data["VS"] == 96.2
        assert data["BMP"] == 345.0

    def test_load_vertical_format(self, handler, temp_csv_file, sample_substrate_data_vertical):
        """Test loading substrate data in vertical format."""
        sample_substrate_data_vertical.to_csv(temp_csv_file, index=False)

        data = handler.load_substrate_lab_data(
            temp_csv_file, substrate_name="Maize silage", substrate_type="maize", validate=False
        )

        assert data["TS"] == 32.5
        assert data["VS"] == 96.2
        assert data["BMP"] == 345.0

    def test_load_german_column_names(self, handler, temp_csv_file, sample_substrate_data_german):
        """Test loading substrate data with German column names."""
        sample_substrate_data_german.to_csv(temp_csv_file, index=False)

        data = handler.load_substrate_lab_data(
            temp_csv_file, substrate_name="Maize silage", substrate_type="maize", validate=False
        )

        # German columns should be mapped to English
        assert "TS" in data
        assert "VS" in data
        assert "RP" in data
        assert "pH" in data
        assert "NH4_N" in data
        assert "TAC" in data
        assert "BMP" in data

    def test_load_with_semicolon_separator(self, handler, temp_csv_file, sample_substrate_data_horizontal):
        """Test loading CSV with semicolon separator."""
        sample_substrate_data_horizontal.to_csv(temp_csv_file, sep=";", index=False)

        data = handler.load_substrate_lab_data(temp_csv_file, sep=";", substrate_name="Maize silage", validate=False)

        assert data["TS"] == 32.5

    def test_load_with_validation(self, handler, temp_csv_file, sample_substrate_data_horizontal):
        """Test loading with data validation."""
        sample_substrate_data_horizontal.to_csv(temp_csv_file, index=False)

        # Should not raise error for valid data
        data = handler.load_substrate_lab_data(temp_csv_file, substrate_name="Maize silage", validate=True)

        assert data["TS"] == 32.5

    def test_load_with_invalid_values_warns(self, handler, temp_csv_file):
        """Test that invalid values produce warnings."""
        # Create data with out-of-range values
        df = pd.DataFrame(
            {
                "substrate_name": ["Test"],
                "substrate_type": ["maize"],
                "TS": [150.0],  # Invalid: > 100%
                "VS": [50.0],
                "pH": [15.0],  # Invalid: > 14
            }
        )
        df.to_csv(temp_csv_file, index=False)

        with pytest.warns(UserWarning):
            handler.load_substrate_lab_data(temp_csv_file, validate=True)

    def test_load_multiple_substrate_samples(self, handler, temp_csv_file):
        """Test loading multiple substrate samples."""
        # Create multiple samples
        df = pd.DataFrame(
            {
                "substrate_name": ["Sample 1", "Sample 2", "Sample 3"],
                "substrate_type": ["maize", "maize", "grass"],
                "sample_date": ["2024-01-10", "2024-01-20", "2024-01-30"],
                "TS": [32.5, 33.0, 28.5],
                "VS": [96.2, 96.5, 92.1],
                "BMP": [345.0, 350.0, 320.0],
            }
        )
        df.to_csv(temp_csv_file, index=False)

        samples = handler.load_multiple_substrate_samples(temp_csv_file)

        assert len(samples) == 3
        assert "TS" in samples.columns
        assert "BMP" in samples.columns

    def test_export_substrate_data(self, handler, temp_csv_file):
        """Test exporting substrate data."""
        data = {"substrate_name": "Maize silage", "TS": 32.5, "VS": 96.2, "BMP": 345.0}

        handler.export_substrate_data(data, temp_csv_file)

        # Load and verify
        df = pd.read_csv(temp_csv_file)
        assert len(df) == 1
        assert df["TS"].iloc[0] == 32.5


# ============================================================================
# Measurement Data Tests
# ============================================================================


class TestMeasurementData:
    """Test measurement data import/export."""

    def test_load_measurement_data(self, handler, temp_csv_file, sample_measurement_data):
        """Test loading measurement data."""
        sample_measurement_data.to_csv(temp_csv_file, index=False)

        df = handler.load_measurement_data(temp_csv_file)

        assert len(df) == 24
        assert "pH" in df.columns
        assert "Q_gas" in df.columns
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_load_measurement_data_with_resample(self, handler, temp_csv_file, sample_measurement_data):
        """Test loading measurement data with resampling."""
        sample_measurement_data.to_csv(temp_csv_file, index=False)

        df = handler.load_measurement_data(temp_csv_file, resample="2H")

        assert len(df) == 12  # Resampled from 24 hourly to 12 2-hourly

    def test_load_measurement_data_german_columns(self, handler, temp_csv_file):
        """Test loading measurement data with German column names."""
        dates = pd.date_range("2024-01-01", periods=5, freq="H")
        df = pd.DataFrame(
            {
                "Zeitstempel": dates,
                "pH-Wert": [7.2, 7.3, 7.2, 7.4, 7.3],
                "Biogasproduktion": [1200, 1210, 1205, 1215, 1220],
                "Methanproduktion": [720, 726, 723, 729, 732],
            }
        )
        df.to_csv(temp_csv_file, index=False)

        loaded = handler.load_measurement_data(temp_csv_file, timestamp_column="Zeitstempel")

        # Columns should be mapped
        assert "pH" in loaded.columns
        assert "Q_gas" in loaded.columns
        assert "Q_ch4" in loaded.columns

    def test_export_measurement_data(self, handler, temp_csv_file, sample_measurement_data):
        """Test exporting measurement data."""
        handler.export_measurement_data(sample_measurement_data, temp_csv_file, include_index=True)

        # Load and verify
        df = pd.read_csv(temp_csv_file)
        assert len(df) == 24
        assert "pH" in df.columns

    def test_auto_detect_separator(self, handler, temp_csv_file, sample_measurement_data):
        """Test automatic separator detection."""
        # Save with semicolon
        sample_measurement_data.to_csv(temp_csv_file, sep=";", index=False)

        # Load with auto-detect
        df = handler.load_measurement_data(temp_csv_file, sep="auto")

        assert len(df) == 24


# ============================================================================
# Simulation Results Tests
# ============================================================================


class TestSimulationResults:
    """Test simulation results export/import."""

    def test_export_simulation_results_flattened(self, handler, temp_csv_file, sample_simulation_results):
        """Test exporting simulation results with flattened components."""
        handler.export_simulation_results(sample_simulation_results, temp_csv_file, flatten_components=True)

        # Load and verify
        df = pd.read_csv(temp_csv_file)
        assert len(df) == 10
        assert "time" in df.columns
        assert "digester_1_Q_gas" in df.columns
        assert "digester_1_pH" in df.columns
        assert "chp_1_P_el" in df.columns

    def test_export_simulation_results_simple(self, handler, temp_csv_file, sample_simulation_results):
        """Test exporting simulation results in simple format."""
        handler.export_simulation_results(sample_simulation_results, temp_csv_file, flatten_components=False)

        # Load and verify
        df = pd.read_csv(temp_csv_file)
        assert len(df) == 10
        assert "time" in df.columns
        assert "Q_gas" in df.columns  # From first component

    def test_load_simulation_results(self, handler, temp_csv_file, sample_simulation_results):
        """Test loading simulation results."""
        handler.export_simulation_results(sample_simulation_results, temp_csv_file)

        loaded_results = handler.load_simulation_results(temp_csv_file)

        assert len(loaded_results) == 10
        assert "time" in loaded_results[0]
        assert "components" in loaded_results[0]

    def test_export_empty_results(self, handler, temp_csv_file):
        """Test exporting empty results produces warning."""
        with pytest.warns(UserWarning):
            handler.export_simulation_results([], temp_csv_file)


# ============================================================================
# Parameter Tables Tests
# ============================================================================


class TestParameterTables:
    """Test parameter table import/export."""

    def test_load_parameter_table(self, handler, temp_csv_file):
        """Test loading parameter table."""
        df = pd.DataFrame(
            {
                "Parameter": ["k_dis", "Y_su", "k_hyd_ch"],
                "Scenario_1": [0.5, 0.10, 10.0],
                "Scenario_2": [0.6, 0.11, 11.0],
                "Scenario_3": [0.7, 0.12, 12.0],
            }
        )
        df.to_csv(temp_csv_file, index=False)

        params = handler.load_parameter_table(temp_csv_file, index_col="Parameter")

        assert len(params) == 3
        assert "Scenario_1" in params.columns
        assert params.loc["k_dis", "Scenario_1"] == 0.5

    def test_export_parameter_table(self, handler, temp_csv_file):
        """Test exporting parameter table."""
        df = pd.DataFrame(
            {"Scenario_1": [0.5, 0.10, 10.0], "Scenario_2": [0.6, 0.11, 11.0]}, index=["k_dis", "Y_su", "k_hyd_ch"]
        )

        handler.export_parameter_table(df, temp_csv_file)

        # Load and verify
        loaded = pd.read_csv(temp_csv_file, index_col=0)
        assert len(loaded) == 3
        assert "Scenario_1" in loaded.columns


# ============================================================================
# Helper Methods Tests
# ============================================================================


class TestHelperMethods:
    """Test helper methods."""

    def test_detect_separator_comma(self, handler, temp_csv_file):
        """Test separator detection for comma."""
        with open(temp_csv_file, "w") as f:
            f.write("col1,col2,col3\n")
            f.write("1,2,3\n")

        sep = handler._detect_separator(temp_csv_file)
        assert sep == ","

    def test_detect_separator_semicolon(self, handler, temp_csv_file):
        """Test separator detection for semicolon."""
        with open(temp_csv_file, "w") as f:
            f.write("col1;col2;col3\n")
            f.write("1;2;3\n")

        sep = handler._detect_separator(temp_csv_file)
        assert sep == ";"

    def test_detect_separator_tab(self, handler, temp_csv_file):
        """Test separator detection for tab."""
        with open(temp_csv_file, "w") as f:
            f.write("col1\tcol2\tcol3\n")
            f.write("1\t2\t3\n")

        sep = handler._detect_separator(temp_csv_file)
        assert sep == "\t"

    def test_map_column_names(self, handler):
        """Test column name mapping."""
        df = pd.DataFrame(
            {
                "Trockensubstanzgehalt": [32.5],
                "pH-Wert": [7.2],
                "Rohprotein": [8.5],
                "Biochemisches Methanpotential": [345.0],
            }
        )

        mapped = handler._map_column_names(df)

        assert "TS" in mapped.columns
        assert "pH" in mapped.columns
        assert "RP" in mapped.columns
        assert "BMP" in mapped.columns

    def test_map_column_names_case_insensitive(self, handler):
        """Test column name mapping is case-insensitive."""
        df = pd.DataFrame({"trockensubstanzgehalt": [32.5], "PH-WERT": [7.2]})

        mapped = handler._map_column_names(df)

        assert "TS" in mapped.columns
        assert "pH" in mapped.columns

    def test_parse_vertical_format(self, handler, sample_substrate_data_vertical):
        """Test parsing vertical format."""
        parsed = handler._parse_vertical_format(sample_substrate_data_vertical)

        assert len(parsed) == 1  # Single row
        assert "TS" in parsed.columns
        assert parsed["TS"].iloc[0] == 32.5

    def test_validate_substrate_data(self, handler):
        """Test substrate data validation."""
        # Valid data
        valid_data = {"TS": 32.5, "VS": 96.2, "pH": 7.2, "BMP": 345.0}

        validated = handler._validate_substrate_data(valid_data)
        assert validated == valid_data

    def test_validate_substrate_data_warns_out_of_range(self, handler):
        """Test validation warns for out-of-range values."""
        invalid_data = {"TS": 150.0, "pH": 15.0}  # Both out of range

        with pytest.warns(UserWarning):
            handler._validate_substrate_data(invalid_data)


# ============================================================================
# Template Generation Tests
# ============================================================================


class TestTemplateGeneration:
    """Test CSV template generation."""

    def test_create_horizontal_template(self, handler, temp_csv_file):
        """Test creating horizontal template."""
        handler.create_template_substrate_csv(temp_csv_file, format_type="horizontal")

        # Load and verify
        df = pd.read_csv(temp_csv_file)
        assert len(df) == 1  # One example row
        assert "substrate_name" in df.columns
        assert "TS" in df.columns
        assert "BMP" in df.columns

    def test_create_vertical_template(self, handler, temp_csv_file):
        """Test creating vertical template."""
        handler.create_template_substrate_csv(temp_csv_file, format_type="vertical")

        # Load and verify
        df = pd.read_csv(temp_csv_file)
        assert "Parameter" in df.columns
        assert "Value" in df.columns
        assert "Unit" in df.columns
        assert "TS" in df["Parameter"].values


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_substrate_workflow(self, handler, temp_csv_file):
        """Test complete substrate data workflow."""
        # 1. Create template
        handler.create_template_substrate_csv(temp_csv_file)

        # 2. Load template
        data = handler.load_substrate_lab_data(temp_csv_file, validate=True)
        assert data["TS"] == 32.5

        # 3. Modify data
        data["TS"] = 35.0
        data["BMP"] = 360.0

        # 4. Export modified data
        temp_export = temp_csv_file.replace(".csv", "_export.csv")
        handler.export_substrate_data(data, temp_export)

        # 5. Load exported data
        loaded = handler.load_substrate_lab_data(temp_export, validate=False)
        assert loaded["TS"] == 35.0
        assert loaded["BMP"] == 360.0

        # Cleanup
        if os.path.exists(temp_export):
            os.unlink(temp_export)

    def test_measurement_data_roundtrip(self, handler, temp_csv_file, sample_measurement_data):
        """Test measurement data export and import roundtrip."""
        # Export
        handler.export_measurement_data(sample_measurement_data, temp_csv_file, include_index=False)

        # Import
        loaded = handler.load_measurement_data(temp_csv_file)

        # Verify
        assert len(loaded) == len(sample_measurement_data)
        assert list(loaded.columns) == [c for c in sample_measurement_data.columns if c != "timestamp"]

    def test_simulation_results_roundtrip(self, handler, temp_csv_file, sample_simulation_results):
        """Test simulation results export and import roundtrip."""
        # Export
        handler.export_simulation_results(sample_simulation_results, temp_csv_file)

        # Import
        loaded = handler.load_simulation_results(temp_csv_file)

        # Verify
        assert len(loaded) == len(sample_simulation_results)
        assert loaded[0]["time"] == sample_simulation_results[0]["time"]


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_load_empty_csv(self, handler, temp_csv_file):
        """Test loading empty CSV."""
        # Create empty CSV with headers only
        df = pd.DataFrame(columns=["substrate_name", "TS", "VS"])
        df.to_csv(temp_csv_file, index=False)

        with pytest.raises(Exception):  # Should raise IndexError or similar
            handler.load_substrate_lab_data(temp_csv_file, validate=False)

    def test_load_csv_missing_columns(self, handler, temp_csv_file):
        """Test loading CSV with missing expected columns."""
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        df.to_csv(temp_csv_file, index=False)

        # Should still work, just with no recognized parameters
        data = handler.load_substrate_lab_data(temp_csv_file, substrate_name="Test", validate=False)
        assert data["substrate_name"] == "Test"

    def test_load_csv_with_special_characters(self, handler, temp_csv_file):
        """Test loading CSV with special characters in data."""
        df = pd.DataFrame({"substrate_name": ["Maize silage (äöü)"], "TS": [32.5], "VS": [96.2], "pH": [7.2], "BMP": [345.0]})
        df.to_csv(temp_csv_file, index=False, encoding="utf-8")

        data = handler.load_substrate_lab_data(temp_csv_file, encoding="utf-8", validate=False)
        assert "äöü" in data["substrate_name"]

    def test_load_csv_with_missing_timestamp(self, handler, temp_csv_file):
        """Test loading measurement CSV without timestamp column."""
        df = pd.DataFrame({"pH": [7.2, 7.3], "Q_gas": [1200, 1210]})
        df.to_csv(temp_csv_file, index=False)

        # Should work but not have datetime index
        loaded = handler.load_measurement_data(temp_csv_file, parse_dates=False)
        assert len(loaded) == 2

    def test_column_mapping_with_unmapped_columns(self, handler):
        """Test that unmapped columns are preserved."""
        df = pd.DataFrame({"TS": [32.5], "CustomColumn": [123], "AnotherColumn": ["text"]})

        mapped = handler._map_column_names(df)

        assert "TS" in mapped.columns
        assert "CustomColumn" in mapped.columns
        assert "AnotherColumn" in mapped.columns


# ============================================================================
# Performance Tests (Optional)
# ============================================================================


class TestPerformance:
    """Test performance with large datasets."""

    @pytest.mark.slow
    def test_load_large_measurement_file(self, handler, temp_csv_file):
        """Test loading large measurement file."""
        # Create large dataset (1 year of hourly data)
        dates = pd.date_range("2024-01-01", periods=8760, freq="H")
        df = pd.DataFrame(
            {
                "timestamp": dates,
                "pH": np.random.uniform(7.0, 7.5, 8760),
                "Q_gas": np.random.uniform(1200, 1300, 8760),
                "Q_ch4": np.random.uniform(720, 780, 8760),
            }
        )
        df.to_csv(temp_csv_file, index=False)

        # Should load without issues
        loaded = handler.load_measurement_data(temp_csv_file)
        assert len(loaded) == 8760

    @pytest.mark.slow
    def test_export_large_simulation(self, handler, temp_csv_file):
        """Test exporting large simulation results."""
        # Create large simulation (1000 time points)
        results = []
        for i in range(1000):
            results.append(
                {
                    "time": float(i) * 0.01,
                    "components": {
                        "digester_1": {"Q_gas": 1250.0, "Q_ch4": 750.0, "pH": 7.3},
                        "chp_1": {"P_el": 480.0, "P_th": 540.0},
                    },
                }
            )

        # Should export without issues
        handler.export_simulation_results(results, temp_csv_file)

        # Verify
        df = pd.read_csv(temp_csv_file)
        assert len(df) == 1000
