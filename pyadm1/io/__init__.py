"""
Input/Output and Data Management

Modules for importing, exporting, and managing simulation data in various formats.

Modules:
    json_handler: JSON serialization and deserialization for plant configurations,
                 simulation results, and parameter sets with schema validation,
                 pretty printing, and compression support for large datasets.

    csv_handler: CSV import/export for time series data (measurements, simulation
                results), substrate schedules, and parameter tables with automatic
                type detection, missing value handling, and header validation.

    database: Database interface for persistent storage of simulation results,
             calibration history, and plant configurations supporting SQLite
             (default), PostgreSQL, and other SQL databases with ORM layer.

    measurement_data: Measurement data loader for real plant data including parsers
                     for common SCADA formats, data validation, outlier detection,
                     resampling to consistent time intervals, and gap filling.

Example:
    >>> from pyadm1.io import (
    ...     JSONHandler,
    ...     CSVHandler,
    ...     Database,
    ...     MeasurementData
    ... )
    >>>
    >>> # Load plant from JSON
    >>> plant = JSONHandler.load_plant("plant.json", feedstock)
    >>>
    >>> # Export results to CSV
    >>> CSVHandler.export_results(results, "results.csv")
    >>>
    >>> # Store in database
    >>> db = Database("sqlite:///biogas.db")
    >>> db.store_simulation(plant_id="plant1", results=results)
    >>>
    >>> # Load measurement data
    >>> measurements = MeasurementData.from_csv(
    ...     "plant_data.csv",
    ...     timestamp_column="time",
    ...     resample="1H"
    ... )
"""

# from pyadm1.io.json_handler import JSONHandler
from pyadm1.io.csv_handler import CSVHandler
from pyadm1.io.database import Database, DatabaseConfig
from pyadm1.io.measurement_data import (
    MeasurementData,
    DataValidator,
    OutlierDetector,
)

__all__ = [
    # "JSONHandler",
    "CSVHandler",
    "Database",
    "DatabaseConfig",
    "MeasurementData",
    "DataValidator",
    "OutlierDetector",
]
