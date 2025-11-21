# pyadm1/io/csv_handler.py
"""
CSV Import/Export Handler for PyADM1

Provides functionality for reading and writing CSV files including:
- Substrate characterization data from laboratory analysis
- Time series measurement data
- Simulation results
- Parameter tables
- Substrate schedules

Features:
- Automatic type detection and conversion
- Multiple CSV formats supported (German/English separators)
- Header validation and mapping
- Missing value handling
- Unit conversion support
- Data validation and quality checks

Example:
    >>> from pyadm1.io import CSVHandler
    >>>
    >>> # Load substrate lab data
    >>> handler = CSVHandler()
    >>> substrate_data = handler.load_substrate_lab_data(
    ...     "lab_results.csv",
    ...     substrate_name="Maize silage"
    ... )
    >>>
    >>> # Export simulation results
    >>> handler.export_simulation_results(
    ...     results,
    ...     "simulation_output.csv"
    ... )
"""

import pandas as pd
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import warnings


class CSVHandler:
    """
    Handler for CSV file operations in PyADM1.

    Supports reading and writing various CSV formats used in biogas
    plant operation and laboratory analysis.

    Example:
        >>> handler = CSVHandler()
        >>> data = handler.load_substrate_lab_data("lab_results.csv")
    """

    # Standard column mappings (German -> English)
    COLUMN_MAPPINGS = {
        # Dry matter and organic content
        "Trockensubstanz": "TS",
        "Trockensubstanzgehalt": "TS",
        "TS-Gehalt": "TS",
        "Organische Trockensubstanz": "VS",
        "oTS": "VS",
        "Organischer Trockensubstanzgehalt": "oTS",
        "Fermentierbare organische Trockensubstanz": "foTS",
        "Flüchtige Feststoffe": "VS",
        # Weender analysis
        "Rohprotein": "RP",
        "Rohfett": "RL",
        "Rohfaser": "RF",
        "Rohasche": "RA",
        "N-freie Extraktstoffe": "NfE",
        # Van Soest
        "Neutral Detergent Fiber": "NDF",
        "Acid Detergent Fiber": "ADF",
        "Acid Detergent Lignin": "ADL",
        # Chemical properties
        "pH-Wert": "pH",
        "Ammoniumstickstoff": "NH4_N",
        "Ammonium-N": "NH4_N",
        "NH4-N": "NH4_N",
        "Alkalinität": "TAC",
        "Pufferkapazität": "TAC",
        "CSB": "COD",
        "CSB-Filtrat": "COD_S",
        "Chemischer Sauerstoffbedarf": "COD",
        "Chemischer Sauerstoffbedarf des Filtrats": "COD_S",
        # Biogas potential
        "Biogaspotential": "BMP",
        "Methanpotential": "BMP",
        "Biochemisches Methanpotential": "BMP",
        "Gasausbeute": "BMP",
        # Carbon and nitrogen
        "Kohlenstoffgehalt": "C_content",
        "Stickstoffgehalt": "N_content",
        "C/N-Verhältnis": "C_to_N",
        "C-N-Verhältnis": "C_to_N",
        "Gesamt-Kjeldahl-Stickstoff": "TKN",
        # Measurement data
        "Zeitstempel": "timestamp",
        "Zeit": "timestamp",
        "Datum": "timestamp",
        "Biogasproduktion": "Q_gas",
        "Methanproduktion": "Q_ch4",
        "Elektrische Leistung": "P_el",
        "Thermische Leistung": "P_th",
        "Temperatur": "T_digester",
    }

    # Unit conversions (from -> to, factor)
    UNIT_CONVERSIONS = {
        # Dry matter: % FM -> % FM (no conversion needed, just validation)
        ("TS", "% FM", "% FM"): 1.0,
        ("TS", "%FM", "% FM"): 1.0,
        ("TS", "g/100g", "% FM"): 1.0,
        # Volatile solids: % TS -> % TS
        ("VS", "% TS", "% TS"): 1.0,
        ("VS", "%TS", "% TS"): 1.0,
        # BMP conversions
        ("BMP", "L/kg VS", "L CH4/kg oTS"): 1.0,  # Assuming same
        ("BMP", "mL/g oTS", "L CH4/kg oTS"): 1.0,  # mL/g = L/kg
        ("BMP", "Nm³/t oTS", "L CH4/kg oTS"): 1.0,  # Nm³/t = L/kg
        # Nitrogen content
        ("NH4_N", "mg/L", "g/L"): 0.001,
        ("NH4_N", "g/kg", "g/L"): 1.0,  # Approximate for density ~1
        # Alkalinity
        ("TAC", "mmol/L", "mmol/L"): 1.0,
        ("TAC", "meq/L", "mmol/L"): 1.0,  # Equivalent
        ("TAC", "g CaCO3/L", "mmol/L"): 20.0,  # 1 g CaCO3 = 20 mmol
        # COD
        ("COD_S", "mg/L", "g/L"): 0.001,
        ("COD_S", "g/L", "g/L"): 1.0,
    }

    def __init__(self, decimal_separator: str = ".", thousands_separator: str = ","):
        """
        Initialize CSV handler.

        Args:
            decimal_separator: Decimal separator ("." or ",")
            thousands_separator: Thousands separator ("," or "." or "")
        """
        self.decimal_separator = decimal_separator
        self.thousands_separator = thousands_separator

    # ========================================================================
    # Substrate Laboratory Data
    # ========================================================================

    def load_substrate_lab_data(
        self,
        filepath: str,
        substrate_name: Optional[str] = None,
        substrate_type: Optional[str] = None,
        sample_date: Optional[Union[str, datetime]] = None,
        sep: str = ",",
        encoding: str = "utf-8",
        validate: bool = True,
    ) -> Dict[str, Any]:
        """
        Load substrate characterization data from laboratory CSV.

        Expected columns (German or English):
        - Trockensubstanzgehalt (TS) [% FM]
        - Organische Trockensubstanz (VS) [% TS]
        - Fermentierbare organische Trockensubstanz (foTS) [% TS]
        - Rohprotein (RP) [% TS]
        - Rohfett (RL) [% TS]
        - Rohfaser (RF) [% TS]
        - NDF, ADF, ADL [% TS]
        - pH-Wert (pH)
        - Ammoniumstickstoff (NH4-N) [g/L or mg/L]
        - Alkalinität (TAC) [mmol/L]
        - Biochemisches Methanpotential (BMP) [L CH4/kg oTS]
        - CSB des Filtrats (COD_S) [g/L]

        Args:
            filepath: Path to CSV file
            substrate_name: Substrate name (if not in file)
            substrate_type: Substrate type (maize, manure, grass, etc.)
            sample_date: Sample date (if not in file)
            sep: Column separator
            encoding: File encoding
            validate: Validate data ranges

        Returns:
            Dict with substrate data

        Example:
            >>> handler = CSVHandler()
            >>> data = handler.load_substrate_lab_data(
            ...     "maize_analysis.csv",
            ...     substrate_name="Maize silage batch 23",
            ...     substrate_type="maize",
            ...     sample_date="2024-01-15"
            ... )
            >>> print(f"TS: {data['TS']:.1f}% FM")
        """
        # Auto-detect separator if needed
        if sep == "auto":
            sep = self._detect_separator(filepath)

        # Read CSV
        df = pd.read_csv(filepath, sep=sep, encoding=encoding)

        # Try to detect if file is in "vertical" format (parameter, value, unit)
        if len(df.columns) <= 3 and "Parameter" in df.columns or "Messgröße" in df.columns:
            df = self._parse_vertical_format(df)

        # Map column names
        df = self._map_column_names(df)

        # If multiple rows, take the first one (or could aggregate)
        if len(df) > 1:
            warnings.warn(f"CSV contains {len(df)} rows, using first row only")

        row = df.iloc[0]

        # Extract data
        result = {
            "substrate_name": substrate_name or row.get("substrate_name", "Unknown"),
            "substrate_type": substrate_type or row.get("substrate_type", "unknown"),
            "sample_date": sample_date or row.get("sample_date", datetime.now()),
        }

        # Add all available parameters
        for param in [
            "TS",
            "VS",
            "oTS",
            "foTS",
            "RP",
            "RL",
            "RF",
            "RA",
            "NfE",
            "NDF",
            "ADF",
            "ADL",
            "pH",
            "NH4_N",
            "TAC",
            "COD",
            "COD_S",
            "BMP",
            "C_content",
            "N_content",
            "C_to_N",
            "TKN",
        ]:
            if param in df.columns:
                value = row[param]
                # Handle both scalar values and Series
                if isinstance(value, pd.Series):
                    value = value.iloc[0] if len(value) > 0 else None
                if pd.notna(value):
                    result[param] = float(value)

        # Validate if requested
        if validate:
            result = self._validate_substrate_data(result)

        # TODO: diese Substratparameter müssen in die substrate_....xml geschrieben werden. evtl. gibt es in einer
        #  c# DLL auch bereits eine Methode die man aufrufen kann. glaube aber eher nicht

        return result

    def load_multiple_substrate_samples(
        self,
        filepath: str,
        sep: str = ",",
        encoding: str = "utf-8",
        date_column: str = "sample_date",
        name_column: str = "substrate_name",
    ) -> pd.DataFrame:
        """
        Load multiple substrate samples from CSV.

        Expected format: Each row is one sample with columns for all parameters.

        Args:
            filepath: Path to CSV file
            sep: Column separator
            encoding: File encoding
            date_column: Name of date column
            name_column: Name of substrate name column

        Returns:
            DataFrame with substrate data

        Example:
            >>> handler = CSVHandler()
            >>> samples = handler.load_multiple_substrate_samples(
            ...     "substrate_database.csv"
            ... )
            >>> print(samples.head())
        """
        df = pd.read_csv(filepath, sep=sep, encoding=encoding)

        # Map column names
        df = self._map_column_names(df)

        # Parse date column
        if date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column])

        return df

    def export_substrate_data(
        self, data: Union[Dict[str, Any], pd.DataFrame], filepath: str, sep: str = ",", encoding: str = "utf-8"
    ) -> None:
        """
        Export substrate data to CSV.

        Args:
            data: Dict or DataFrame with substrate data
            filepath: Output file path
            sep: Column separator
            encoding: File encoding

        Example:
            >>> handler.export_substrate_data(substrate_data, "export.csv")
        """
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data

        df.to_csv(filepath, sep=sep, encoding=encoding, index=False)
        print(f"✓ Exported substrate data to {filepath}")

    # ========================================================================
    # Measurement Data
    # ========================================================================

    # TODO: so eine Methode gibt es bereits in calibration.py. Dort evtl. löschen. Vorher beide vergleichen.
    def load_measurement_data(
        self,
        filepath: str,
        timestamp_column: str = "timestamp",
        sep: str = ",",
        encoding: str = "utf-8",
        parse_dates: bool = True,
        resample: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load time series measurement data from CSV.

        Expected columns:
        - timestamp (or Zeit, Zeitstempel)
        - Q_sub_* (substrate feeds)
        - pH, VFA, TAC, FOS_TAC
        - T_digester
        - Q_gas, Q_ch4, Q_co2, CH4_content, P_gas
        - P_el, P_th

        Args:
            filepath: Path to CSV file
            timestamp_column: Name of timestamp column
            sep: Column separator
            encoding: File encoding
            parse_dates: Parse timestamp column
            resample: Resample frequency (e.g., "1H", "1D")

        Returns:
            DataFrame with measurements

        Example:
            >>> handler = CSVHandler()
            >>> data = handler.load_measurement_data(
            ...     "plant_data.csv",
            ...     resample="1H"
            ... )
        """
        # Auto-detect separator
        if sep == "auto":
            sep = self._detect_separator(filepath)

        # Read CSV
        df = pd.read_csv(filepath, sep=sep, encoding=encoding)

        # Map column names
        df = self._map_column_names(df)

        # Parse timestamp
        if timestamp_column in df.columns:
            if parse_dates:
                df[timestamp_column] = pd.to_datetime(df[timestamp_column])
            df = df.set_index(timestamp_column).sort_index()

        # Resample if requested
        if resample is not None:
            df = df.resample(resample).mean()

        return df

    def export_measurement_data(
        self, data: pd.DataFrame, filepath: str, sep: str = ",", encoding: str = "utf-8", include_index: bool = True
    ) -> None:
        """
        Export measurement data to CSV.

        Args:
            data: DataFrame with measurements
            filepath: Output file path
            sep: Column separator
            encoding: File encoding
            include_index: Include index (timestamp) in output

        Example:
            >>> handler.export_measurement_data(measurements, "export.csv")
        """
        data.to_csv(filepath, sep=sep, encoding=encoding, index=include_index)
        print(f"✓ Exported measurement data to {filepath} ({len(data)} rows)")

    # ========================================================================
    # Simulation Results
    # ========================================================================

    def export_simulation_results(
        self,
        results: List[Dict[str, Any]],
        filepath: str,
        sep: str = ",",
        encoding: str = "utf-8",
        flatten_components: bool = True,
    ) -> None:
        """
        Export simulation results to CSV.

        Args:
            results: List of result dicts from plant.simulate()
            filepath: Output file path
            sep: Column separator
            encoding: File encoding
            flatten_components: Flatten component results into columns

        Example:
            >>> results = plant.simulate(duration=30, dt=1/24)
            >>> handler.export_simulation_results(results, "simulation.csv")
        """
        if not results:
            warnings.warn("No results to export")
            return

        # Convert to DataFrame
        if flatten_components:
            # Flatten structure: time, component1_metric1, component1_metric2, ...
            rows = []
            for result in results:
                row = {"time": result["time"]}

                for comp_id, comp_data in result["components"].items():
                    for metric, value in comp_data.items():
                        # Skip nested dicts (like gas_storage)
                        if isinstance(value, dict):
                            continue
                        col_name = f"{comp_id}_{metric}"
                        row[col_name] = value

                rows.append(row)

            df = pd.DataFrame(rows)
        else:
            # Simple format: just time and first component's data
            first_comp_id = list(results[0]["components"].keys())[0]
            rows = []
            for result in results:
                row = {"time": result["time"]}
                row.update(result["components"][first_comp_id])
                # Remove nested dicts
                row = {k: v for k, v in row.items() if not isinstance(v, dict)}
                rows.append(row)

            df = pd.DataFrame(rows)

        # Export
        df.to_csv(filepath, sep=sep, encoding=encoding, index=False)
        print(f"✓ Exported simulation results to {filepath} ({len(df)} time points)")

    def load_simulation_results(self, filepath: str, sep: str = ",", encoding: str = "utf-8") -> List[Dict[str, Any]]:
        """
        Load simulation results from CSV.

        Args:
            filepath: Path to CSV file
            sep: Column separator
            encoding: File encoding

        Returns:
            List of result dicts

        Example:
            >>> results = handler.load_simulation_results("simulation.csv")
        """
        df = pd.read_csv(filepath, sep=sep, encoding=encoding)

        # Convert back to results format
        results = []
        for _, row in df.iterrows():
            result = {"time": row["time"], "components": {}}

            # Group columns by component
            for col in df.columns:
                if col == "time":
                    continue

                if "_" in col:
                    comp_id, metric = col.split("_", 1)
                    if comp_id not in result["components"]:
                        result["components"][comp_id] = {}
                    result["components"][comp_id][metric] = row[col]

            results.append(result)

        return results

    # ========================================================================
    # Parameter Tables
    # ========================================================================

    def load_parameter_table(
        self, filepath: str, sep: str = ",", encoding: str = "utf-8", index_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load parameter table from CSV.

        Expected format:
        - Rows: Parameters
        - Columns: Different scenarios/substrates

        Args:
            filepath: Path to CSV file
            sep: Column separator
            encoding: File encoding
            index_col: Column to use as index (usually parameter name)

        Returns:
            DataFrame with parameters

        Example:
            >>> params = handler.load_parameter_table("parameters.csv")
        """
        df = pd.read_csv(filepath, sep=sep, encoding=encoding, index_col=index_col)
        return df

    def export_parameter_table(self, data: pd.DataFrame, filepath: str, sep: str = ",", encoding: str = "utf-8") -> None:
        """
        Export parameter table to CSV.

        Args:
            data: DataFrame with parameters
            filepath: Output file path
            sep: Column separator
            encoding: File encoding

        Example:
            >>> handler.export_parameter_table(params_df, "parameters.csv")
        """
        data.to_csv(filepath, sep=sep, encoding=encoding)
        print(f"✓ Exported parameter table to {filepath}")

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _detect_separator(self, filepath: str) -> str:
        """
        Auto-detect CSV separator.

        Args:
            filepath: Path to CSV file

        Returns:
            Detected separator
        """
        with open(filepath, "r") as f:
            first_line = f.readline()

        # Count occurrences of common separators
        comma_count = first_line.count(",")
        semicolon_count = first_line.count(";")
        tab_count = first_line.count("\t")

        if semicolon_count > comma_count and semicolon_count > tab_count:
            return ";"
        elif tab_count > comma_count:
            return "\t"
        else:
            return ","

    def _map_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map German column names to English standard names.

        Args:
            df: DataFrame with original column names

        Returns:
            DataFrame with mapped column names
        """
        # Create mapping dict for this DataFrame
        mapping = {}
        for col in df.columns:
            # Check if column is in mapping dict
            if col in self.COLUMN_MAPPINGS:
                mapping[col] = self.COLUMN_MAPPINGS[col]
            # Also check case-insensitive
            elif col.lower().strip() in {k.lower(): v for k, v in self.COLUMN_MAPPINGS.items()}:
                original_key = [k for k in self.COLUMN_MAPPINGS if k.lower() == col.lower().strip()][0]
                mapping[col] = self.COLUMN_MAPPINGS[original_key]

        if mapping:
            df = df.rename(columns=mapping)

        return df

    def _parse_vertical_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse vertical CSV format (parameter, value, unit columns).

        Args:
            df: DataFrame in vertical format

        Returns:
            DataFrame in horizontal format
        """
        # Expected columns: Parameter/Messgröße, Wert/Value, Einheit/Unit
        param_col = None
        value_col = None
        unit_col = None

        for col in df.columns:
            col_lower = col.lower().strip()
            if "parameter" in col_lower or "messgröße" in col_lower:
                param_col = col
            elif "wert" in col_lower or "value" in col_lower:
                value_col = col
            elif "einheit" in col_lower or "unit" in col_lower:
                unit_col = col
                print("_parse_vertical_format: ", unit_col)

        if not (param_col and value_col):
            raise ValueError("Cannot parse vertical format: missing Parameter or Value column")

        # Create horizontal DataFrame
        data = {}
        for _, row in df.iterrows():
            param = str(row[param_col]).strip()
            value = row[value_col]

            # Map parameter name
            if param in self.COLUMN_MAPPINGS:
                param = self.COLUMN_MAPPINGS[param]

            data[param] = value

        return pd.DataFrame([data])

    def _validate_substrate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate substrate data ranges.

        Args:
            data: Substrate data dict

        Returns:
            Validated data dict (with warnings for out-of-range values)
        """
        # Expected ranges
        ranges = {
            "TS": (5.0, 95.0),  # % FM
            "VS": (50.0, 100.0),  # % TS
            "RP": (0.0, 40.0),  # % TS
            "RL": (0.0, 20.0),  # % TS
            "RF": (0.0, 50.0),  # % TS
            "NDF": (10.0, 90.0),  # % TS
            "ADF": (5.0, 70.0),  # % TS
            "ADL": (0.0, 30.0),  # % TS
            "pH": (3.0, 9.0),
            "NH4_N": (0.0, 10.0),  # g/L
            "TAC": (0.0, 500.0),  # mmol/L
            "BMP": (50.0, 800.0),  # L CH4/kg oTS
            "C_to_N": (5.0, 100.0),
        }

        for param, (min_val, max_val) in ranges.items():
            if param in data:
                value = data[param]
                if value < min_val or value > max_val:
                    warnings.warn(f"Parameter '{param}' = {value} is outside expected " f"range [{min_val}, {max_val}]")

        return data

    def create_template_substrate_csv(self, filepath: str, format_type: str = "horizontal") -> None:
        """
        Create template CSV file for substrate data entry.

        Args:
            filepath: Output file path
            format_type: "horizontal" or "vertical"

        Example:
            >>> handler.create_template_substrate_csv("template.csv")
        """
        if format_type == "horizontal":
            # One row per sample
            template = pd.DataFrame(
                columns=[
                    "substrate_name",
                    "substrate_type",
                    "sample_date",
                    "TS",
                    "VS",
                    "oTS",
                    "foTS",
                    "RP",
                    "RL",
                    "RF",
                    "NDF",
                    "ADF",
                    "ADL",
                    "pH",
                    "NH4_N",
                    "TAC",
                    "COD_S",
                    "BMP",
                    "C_content",
                    "N_content",
                    "C_to_N",
                ]
            )

            # Add example row
            template.loc[0] = [
                "Maize silage",
                "maize",
                "2024-01-15",
                32.5,
                96.2,
                31.3,
                28.5,
                8.5,
                3.2,
                21.5,
                42.1,
                22.3,
                2.1,
                3.9,
                0.5,
                11.0,
                18.5,
                345.0,
                45.2,
                1.8,
                25.1,
            ]

        else:  # vertical
            template = pd.DataFrame(
                {
                    "Parameter": [
                        "Substrate name",
                        "Substrate type",
                        "TS",
                        "VS",
                        "RP",
                        "RL",
                        "NDF",
                        "ADF",
                        "ADL",
                        "pH",
                        "NH4-N",
                        "TAC",
                        "COD_S",
                        "BMP",
                    ],
                    "Value": ["Maize silage", "maize", 32.5, 96.2, 8.5, 3.2, 42.1, 22.3, 2.1, 3.9, 0.5, 11.0, 18.5, 345.0],
                    "Unit": [
                        "",
                        "",
                        "% FM",
                        "% TS",
                        "% TS",
                        "% TS",
                        "% TS",
                        "% TS",
                        "% TS",
                        "-",
                        "g/L",
                        "mmol/L",
                        "g/L",
                        "L CH4/kg oTS",
                    ],
                }
            )

        template.to_csv(filepath, index=False)
        print(f"✓ Created template CSV at {filepath}")
