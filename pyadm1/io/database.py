# pyadm1/io/database.py
"""
Database Interface for PyADM1

Provides PostgreSQL database interface for storing and retrieving:
- Measurement data from biogas plants
- Simulation results (time series, metrics)
- Calibration history and parameters
- Plant configurations
- Substrate characterization data

Features:
- SQLAlchemy ORM for database abstraction
- Automatic schema creation and migration
- Bulk insert for time series data
- Query builders for common operations
- Connection pooling and error handling

Example:
    >>> from pyadm1.io import Database
    >>>
    >>> # Connect to database
    >>> db = Database("postgresql://user:pass@localhost/biogas")
    >>>
    >>> # Store measurement data
    >>> db.store_measurements(
    ...     plant_id="plant1",
    ...     data=measurements_df,
    ...     source="SCADA"
    ... )
    >>>
    >>> # Load measurement data
    >>> data = db.load_measurements(
    ...     plant_id="plant1",
    ...     start_time="2024-01-01",
    ...     end_time="2024-01-31"
    ... )
    >>>
    >>> # Store simulation results
    >>> db.store_simulation(
    ...     simulation_id="sim_001",
    ...     plant_id="plant1",
    ...     results=results
    ... )
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import pandas as pd
import numpy as np
from dataclasses import dataclass
from contextlib import contextmanager

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    Float,
    String,
    DateTime,
    Text,
    Boolean,
    ForeignKey,
    Index,
    JSON,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

Base = declarative_base()


# ============================================================================
# Database Models (ORM Tables)
# ============================================================================


class Plant(Base):
    """Plant configuration and metadata."""

    __tablename__ = "plants"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    location = Column(String)
    operator = Column(String)
    commissioned = Column(DateTime)
    V_liq = Column(Float)  # Liquid volume [m³]
    V_gas = Column(Float)  # Gas volume [m³]
    T_ad = Column(Float)  # Operating temperature [K]
    P_el_nom = Column(Float)  # Nominal electrical power [kW]
    configuration = Column(JSON)  # Plant configuration JSON
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    measurements = relationship("Measurement", back_populates="plant", cascade="all, delete-orphan")
    simulations = relationship("Simulation", back_populates="plant", cascade="all, delete-orphan")
    calibrations = relationship("Calibration", back_populates="plant", cascade="all, delete-orphan")


class Measurement(Base):
    """Time series measurement data."""

    __tablename__ = "measurements"

    id = Column(Integer, primary_key=True)
    plant_id = Column(String, ForeignKey("plants.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    source = Column(String)  # SCADA, lab, manual, etc.

    # Substrate feeds [m³/d]
    Q_sub_maize = Column(Float)
    Q_sub_manure = Column(Float)
    Q_sub_grass = Column(Float)
    Q_sub_other = Column(Float)

    # Process indicators
    pH = Column(Float)
    VFA = Column(Float)  # [g HAc eq/L]
    TAC = Column(Float)  # [g CaCO3 eq/L]
    FOS_TAC = Column(Float)  # VFA/TAC ratio
    T_digester = Column(Float)  # [K]

    # Gas production
    Q_gas = Column(Float)  # [m³/d]
    Q_ch4 = Column(Float)  # [m³/d]
    Q_co2 = Column(Float)  # [m³/d]
    CH4_content = Column(Float)  # [%]
    P_gas = Column(Float)  # [bar]

    # Energy output
    P_el = Column(Float)  # [kW]
    P_th = Column(Float)  # [kW]

    # Quality flags
    is_validated = Column(Boolean, default=False)
    quality_flag = Column(String)  # good, suspect, bad

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    plant = relationship("Plant", back_populates="measurements")

    # Composite index for efficient time-range queries
    __table_args__ = (Index("idx_plant_timestamp", "plant_id", "timestamp"),)


class Simulation(Base):
    """Simulation run metadata and summary results."""

    __tablename__ = "simulations"

    id = Column(String, primary_key=True)
    plant_id = Column(String, ForeignKey("plants.id"), nullable=False)
    name = Column(String)
    description = Column(Text)
    duration = Column(Float)  # [days]
    dt = Column(Float)  # Time step [days]
    scenario = Column(String)  # baseline, optimized, calibrated, etc.
    parameters = Column(JSON)  # Simulation parameters
    initial_state = Column(JSON)  # Initial ADM1 state

    # Summary metrics
    avg_Q_gas = Column(Float)  # Average biogas [m³/d]
    avg_Q_ch4 = Column(Float)  # Average methane [m³/d]
    avg_CH4_content = Column(Float)  # Average CH4 content [%]
    avg_pH = Column(Float)
    avg_VFA = Column(Float)
    total_energy = Column(Float)  # Total energy [kWh]

    # Status
    status = Column(String)  # pending, running, completed, failed
    error_message = Column(Text)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    plant = relationship("Plant", back_populates="simulations")
    time_series = relationship("SimulationTimeSeries", back_populates="simulation", cascade="all, delete-orphan")


class SimulationTimeSeries(Base):
    """Time series data from simulation."""

    __tablename__ = "simulation_time_series"

    id = Column(Integer, primary_key=True)
    simulation_id = Column(String, ForeignKey("simulations.id"), nullable=False)
    time = Column(Float, nullable=False)  # [days]

    # Biogas production
    Q_gas = Column(Float)  # [m³/d]
    Q_ch4 = Column(Float)  # [m³/d]
    Q_co2 = Column(Float)  # [m³/d]
    CH4_content = Column(Float)  # [%]

    # Process indicators
    pH = Column(Float)
    VFA = Column(Float)  # [g/L]
    TAC = Column(Float)  # [g CaCO3/L]
    FOS_TAC = Column(Float)

    # Energy
    P_el = Column(Float)  # [kW]
    P_th = Column(Float)  # [kW]

    # Relationships
    simulation = relationship("Simulation", back_populates="time_series")

    __table_args__ = (Index("idx_simulation_time", "simulation_id", "time"),)


class Calibration(Base):
    """Calibration history and results."""

    __tablename__ = "calibrations"

    id = Column(Integer, primary_key=True)
    plant_id = Column(String, ForeignKey("plants.id"), nullable=False)
    calibration_type = Column(String)  # initial, online
    method = Column(String)  # differential_evolution, nelder_mead, etc.

    # Parameters calibrated
    parameters = Column(JSON)  # {param_name: value}
    parameter_bounds = Column(JSON)  # {param_name: [min, max]}

    # Objectives and metrics
    objectives = Column(JSON)  # [Q_ch4, pH, VFA]
    objective_weights = Column(JSON)  # {Q_ch4: 0.6, pH: 0.2, VFA: 0.2}
    objective_value = Column(Float)  # Final objective value

    # Validation metrics
    validation_metrics = Column(JSON)  # {Q_ch4_r2: 0.85, pH_rmse: 0.12, ...}

    # Data used
    data_start = Column(DateTime)
    data_end = Column(DateTime)
    n_measurements = Column(Integer)

    # Status
    success = Column(Boolean)
    message = Column(Text)
    n_iterations = Column(Integer)
    execution_time = Column(Float)  # [seconds]

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    plant = relationship("Plant", back_populates="calibrations")


class Substrate(Base):
    """Substrate characterization data from laboratory analysis."""

    __tablename__ = "substrates"

    id = Column(Integer, primary_key=True)
    plant_id = Column(String, ForeignKey("plants.id"), nullable=False)
    substrate_name = Column(String, nullable=False)
    substrate_type = Column(String)  # maize, manure, grass, etc.
    sample_date = Column(DateTime, nullable=False)
    sample_id = Column(String)  # Laboratory sample ID

    # Dry matter and organic content
    TS = Column(Float)  # Total solids [% FM]
    VS = Column(Float)  # Volatile solids [% TS]
    oTS = Column(Float)  # Organic total solids [% FM]
    foTS = Column(Float)  # Fermentable organic total solids [% TS]

    # Weender analysis [% TS]
    RP = Column(Float)  # Raw protein (Rohprotein)
    RL = Column(Float)  # Raw lipids (Rohfett)
    RF = Column(Float)  # Raw fiber (Rohfaser)
    RA = Column(Float)  # Raw ash (Rohasche)
    NfE = Column(Float)  # N-free extract (N-freie Extraktstoffe)

    # Van Soest fiber fractions [% TS]
    NDF = Column(Float)  # Neutral detergent fiber
    ADF = Column(Float)  # Acid detergent fiber
    ADL = Column(Float)  # Acid detergent lignin

    # Chemical properties
    pH = Column(Float)
    NH4_N = Column(Float)  # Ammonium nitrogen [g/L or g/kg]
    TAC = Column(Float)  # Total alkalinity [mmol/L or meq/L]
    COD_S = Column(Float)  # COD of filtrate [g/L]

    # Biogas potential
    BMP = Column(Float)  # Biochemical methane potential [L CH4/kg oTS or VS]
    BMP_unit = Column(String, default="L_CH4/kg_oTS")

    # Carbon and nitrogen
    C_content = Column(Float)  # Total carbon [% TS]
    N_content = Column(Float)  # Total nitrogen [% TS]
    C_to_N = Column(Float)  # C/N ratio
    TKN = Column(Float)  # Total Kjeldahl Nitrogen [% FM]

    # ADM1 parameters (if calculated)
    f_ch_xc = Column(Float)  # Carbohydrate fraction
    f_pr_xc = Column(Float)  # Protein fraction
    f_li_xc = Column(Float)  # Lipid fraction
    k_dis = Column(Float)  # Disintegration rate [1/d]
    k_hyd_ch = Column(Float)  # Hydrolysis rate carbohydrates [1/d]
    k_hyd_pr = Column(Float)  # Hydrolysis rate proteins [1/d]
    k_hyd_li = Column(Float)  # Hydrolysis rate lipids [1/d]

    # Metadata
    lab_name = Column(String)  # Laboratory that performed analysis
    analysis_method = Column(String)  # DIN, VDLUFA, etc.
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("idx_plant_substrate_date", "plant_id", "substrate_name", "sample_date"),)


# ============================================================================
# Database Configuration
# ============================================================================


@dataclass
class DatabaseConfig:
    """Database connection configuration."""

    host: str = "localhost"
    port: int = 5432
    database: str = "biogas"
    username: str = "postgres"
    password: str = ""
    pool_size: int = 5
    max_overflow: int = 10
    echo: bool = False  # Echo SQL statements


# ============================================================================
# Main Database Class
# ============================================================================


class Database:
    """
    PostgreSQL database interface for PyADM1.

    Handles all database operations including connection management,
    data storage/retrieval, and query execution.

    Example:
        >>> db = Database("postgresql://user:pass@localhost/biogas")
        >>> db.create_all_tables()
        >>> db.store_measurements(plant_id="plant1", data=df)
    """

    def __init__(self, connection_string: Optional[str] = None, config: Optional[DatabaseConfig] = None):
        """
        Initialize database connection.

        Args:
            connection_string: SQLAlchemy connection string
                Format: postgresql://username:password@host:port/database
            config: DatabaseConfig object (alternative to connection_string)

        Example:
            >>> db = Database("postgresql://user:pass@localhost/biogas")
            >>> # or
            >>> config = DatabaseConfig(host="localhost", database="biogas")
            >>> db = Database(config=config)
        """
        if connection_string:
            self.connection_string = connection_string
        elif config:
            self.connection_string = (
                f"postgresql://{config.username}:{config.password}@" f"{config.host}:{config.port}/{config.database}"
            )
            self.config = config
        else:
            raise ValueError("Either connection_string or config must be provided")

        # Create engine with connection pooling
        pool_size = config.pool_size if config else 5
        max_overflow = config.max_overflow if config else 10
        echo = config.echo if config else False

        self.engine = create_engine(
            self.connection_string, poolclass=QueuePool, pool_size=pool_size, max_overflow=max_overflow, echo=echo
        )

        # Create session factory
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    @contextmanager
    def get_session(self) -> Session:
        """
        Context manager for database sessions.

        Automatically handles commit/rollback and session cleanup.

        Example:
            >>> with db.get_session() as session:
            ...     plant = session.query(Plant).first()
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def create_all_tables(self) -> None:
        """
        Create all database tables.

        Should be called once when setting up a new database.

        Example:
            >>> db.create_all_tables()
        """
        Base.metadata.create_all(bind=self.engine)
        print("✓ All database tables created")

    def drop_all_tables(self) -> None:
        """
        Drop all database tables.

        WARNING: This will delete all data!

        Example:
            >>> db.drop_all_tables()  # Use with caution!
        """
        Base.metadata.drop_all(bind=self.engine)
        print("✓ All database tables dropped")

    # ========================================================================
    # Plant Management
    # ========================================================================

    def create_plant(
        self,
        plant_id: str,
        name: str,
        location: Optional[str] = None,
        operator: Optional[str] = None,
        V_liq: Optional[float] = None,
        V_gas: Optional[float] = None,
        T_ad: Optional[float] = None,
        P_el_nom: Optional[float] = None,
        configuration: Optional[Dict] = None,
    ) -> Plant:
        """
        Create new plant entry.

        Args:
            plant_id: Unique plant identifier
            name: Plant name
            location: Plant location
            operator: Operating company
            V_liq: Liquid volume [m³]
            V_gas: Gas volume [m³]
            T_ad: Operating temperature [K]
            P_el_nom: Nominal electrical power [kW]
            configuration: Plant configuration dict

        Returns:
            Created Plant object

        Example:
            >>> plant = db.create_plant(
            ...     plant_id="plant1",
            ...     name="Demo Plant",
            ...     V_liq=2000,
            ...     T_ad=308.15
            ... )
        """
        session = self.SessionLocal()
        try:
            plant = Plant(
                id=plant_id,
                name=name,
                location=location,
                operator=operator,
                V_liq=V_liq,
                V_gas=V_gas,
                T_ad=T_ad,
                P_el_nom=P_el_nom,
                configuration=configuration,
            )

            session.add(plant)
            session.commit()
            session.refresh(plant)

            # Create a detached copy with all attributes loaded
            plant_dict = {
                "id": plant.id,
                "name": plant.name,
                "location": plant.location,
                "operator": plant.operator,
                "V_liq": plant.V_liq,
                "V_gas": plant.V_gas,
                "T_ad": plant.T_ad,
                "P_el_nom": plant.P_el_nom,
                "configuration": plant.configuration,
                "created_at": plant.created_at,
                "updated_at": plant.updated_at,
            }
            print(plant_dict)

            session.expunge(plant)

            print(f"✓ Plant '{plant_id}' created")
            return plant

        except IntegrityError:
            session.rollback()
            raise ValueError(f"Plant with ID '{plant_id}' already exists")
        finally:
            session.close()

    def get_plant(self, plant_id: str) -> Optional[Plant]:
        """
        Get plant by ID.

        Args:
            plant_id: Plant identifier

        Returns:
            Plant object or None if not found
        """
        session = self.SessionLocal()
        try:
            plant = session.query(Plant).filter(Plant.id == plant_id).first()
            if plant:
                session.expunge(plant)  # Detach before closing
            return plant
        finally:
            session.close()

    def list_plants(self) -> List[Dict[str, Any]]:
        """
        List all plants.

        Returns:
            List of plant information dicts
        """
        with self.get_session() as session:
            plants = session.query(Plant).all()

            return [
                {
                    "id": p.id,
                    "name": p.name,
                    "location": p.location,
                    "V_liq": p.V_liq,
                    "V_gas": p.V_gas,
                    "T_ad": p.T_ad,
                    "created_at": p.created_at,
                }
                for p in plants
            ]

    # ========================================================================
    # Measurement Data
    # ========================================================================

    def store_measurements(self, plant_id: str, data: pd.DataFrame, source: str = "SCADA", validate: bool = True) -> int:
        """
        Store measurement data from DataFrame.

        Args:
            plant_id: Plant identifier
            data: DataFrame with measurement data
                Required column: timestamp
                Optional columns: Q_sub_*, pH, VFA, TAC, T_digester,
                                Q_gas, Q_ch4, P_el, P_th, etc.
            source: Data source identifier
            validate: Validate data before storing

        Returns:
            Number of records stored

        Example:
            >>> data = pd.read_csv("measurements.csv")
            >>> n = db.store_measurements("plant1", data, source="SCADA")
            >>> print(f"Stored {n} measurements")
        """
        if "timestamp" not in data.columns:
            raise ValueError("DataFrame must have 'timestamp' column")

        # Verify plant exists
        if not self.get_plant(plant_id):
            raise ValueError(f"Plant '{plant_id}' not found")

        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(data["timestamp"]):
            data["timestamp"] = pd.to_datetime(data["timestamp"])

        # Validate data if requested
        if validate:
            from pyadm1.io.measurement_data import DataValidator

            validation = DataValidator.validate(data)
            if not validation.is_valid:
                print("⚠ Data validation warnings:")
                for issue in validation.issues:
                    print(f"  - {issue}")

        # Prepare records for bulk insert
        records = []
        for _, row in data.iterrows():
            record = {
                "plant_id": plant_id,
                "timestamp": row["timestamp"],
                "source": source,
            }

            # Add all available columns
            for col in data.columns:
                if col != "timestamp" and col in Measurement.__table__.columns:
                    value = row[col]
                    if pd.notna(value):
                        record[col] = float(value) if isinstance(value, (int, float, np.number)) else value

            records.append(record)

        # Bulk insert
        with self.get_session() as session:
            try:
                session.bulk_insert_mappings(Measurement, records)
                session.commit()
                print(f"✓ Stored {len(records)} measurements for plant '{plant_id}'")
                return len(records)
            except SQLAlchemyError as e:
                print(f"✗ Error storing measurements: {e}")
                raise

    def load_measurements(
        self,
        plant_id: str,
        start_time: Optional[Union[str, datetime]] = None,
        end_time: Optional[Union[str, datetime]] = None,
        columns: Optional[List[str]] = None,
        source: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load measurement data as DataFrame.

        Args:
            plant_id: Plant identifier
            start_time: Start time (ISO string or datetime)
            end_time: End time (ISO string or datetime)
            columns: Specific columns to load (None = all)
            source: Filter by data source

        Returns:
            DataFrame with measurements

        Example:
            >>> data = db.load_measurements(
            ...     "plant1",
            ...     start_time="2024-01-01",
            ...     end_time="2024-01-31"
            ... )
        """
        # Convert string dates to datetime
        if isinstance(start_time, str):
            start_time = pd.to_datetime(start_time)
        if isinstance(end_time, str):
            end_time = pd.to_datetime(end_time)

        with self.get_session() as session:
            query = session.query(Measurement).filter(Measurement.plant_id == plant_id)

            # Apply filters
            if start_time:
                query = query.filter(Measurement.timestamp >= start_time)
            if end_time:
                query = query.filter(Measurement.timestamp <= end_time)
            if source:
                query = query.filter(Measurement.source == source)

            # Order by timestamp
            query = query.order_by(Measurement.timestamp)

            # Execute query
            results = query.all()

            if not results:
                return pd.DataFrame()

            # Convert to DataFrame
            data_dict = {"timestamp": [r.timestamp for r in results]}

            # Determine which columns to include
            if columns is None:
                columns = [
                    c.name
                    for c in Measurement.__table__.columns
                    if c.name not in ["id", "plant_id", "timestamp", "source", "created_at"]
                ]

            for col in columns:
                data_dict[col] = [getattr(r, col) for r in results]

            df = pd.DataFrame(data_dict)
            df = df.set_index("timestamp")

            return df

    # ========================================================================
    # Simulation Results
    # ========================================================================

    def store_simulation(
        self,
        simulation_id: str,
        plant_id: str,
        results: List[Dict[str, Any]],
        name: Optional[str] = None,
        description: Optional[str] = None,
        duration: Optional[float] = None,
        parameters: Optional[Dict] = None,
        scenario: str = "baseline",
    ) -> Simulation:
        """
        Store simulation results.

        Args:
            simulation_id: Unique simulation identifier
            plant_id: Plant identifier
            results: List of result dicts with 'time' and 'components' keys
            name: Simulation name
            description: Description
            duration: Simulation duration [days]
            parameters: Simulation parameters
            scenario: Scenario name

        Returns:
            Created Simulation object

        Example:
            >>> results = plant.simulate(duration=30, dt=1/24)
            >>> db.store_simulation(
            ...     "sim_001",
            ...     "plant1",
            ...     results,
            ...     name="Baseline simulation"
            ... )
        """
        # Verify plant exists
        if not self.get_plant(plant_id):
            raise ValueError(f"Plant '{plant_id}' not found")

        # Calculate summary metrics
        metrics = self._calculate_simulation_metrics(results)

        with self.get_session() as session:
            # Create simulation record
            simulation = Simulation(
                id=simulation_id,
                plant_id=plant_id,
                name=name,
                description=description,
                duration=duration or (results[-1]["time"] if results else 0),
                scenario=scenario,
                parameters=parameters,
                avg_Q_gas=metrics.get("avg_Q_gas"),
                avg_Q_ch4=metrics.get("avg_Q_ch4"),
                avg_CH4_content=metrics.get("avg_CH4_content"),
                avg_pH=metrics.get("avg_pH"),
                avg_VFA=metrics.get("avg_VFA"),
                total_energy=metrics.get("total_energy"),
                status="completed",
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
            )

            try:
                session.add(simulation)
                session.flush()

                # Store time series data
                time_series_records = []
                for result in results:
                    time = result["time"]
                    components = result["components"]

                    # Extract data from first component (usually digester)
                    comp_data = next(iter(components.values()))

                    record = {
                        "simulation_id": simulation_id,
                        "time": time,
                        "Q_gas": comp_data.get("Q_gas"),
                        "Q_ch4": comp_data.get("Q_ch4"),
                        "Q_co2": comp_data.get("Q_co2"),
                        "pH": comp_data.get("pH"),
                        "VFA": comp_data.get("VFA"),
                        "TAC": comp_data.get("TAC"),
                    }

                    # Add CHP data if present
                    for comp_id, comp_result in components.items():
                        if "P_el" in comp_result:
                            record["P_el"] = comp_result["P_el"]
                        if "P_th" in comp_result:
                            record["P_th"] = comp_result["P_th"]

                    # Calculate CH4 content
                    if record["Q_gas"] and record["Q_ch4"] and record["Q_gas"] > 0:
                        record["CH4_content"] = (record["Q_ch4"] / record["Q_gas"]) * 100

                    # Calculate FOS/TAC
                    if record["VFA"] and record["TAC"] and record["TAC"] > 0:
                        record["FOS_TAC"] = record["VFA"] / record["TAC"]

                    time_series_records.append(record)

                # Bulk insert time series
                session.bulk_insert_mappings(SimulationTimeSeries, time_series_records)
                session.commit()

                print(f"✓ Stored simulation '{simulation_id}' with {len(time_series_records)} time points")
                return simulation

            except IntegrityError:
                raise ValueError(f"Simulation with ID '{simulation_id}' already exists")

    def load_simulation(self, simulation_id: str) -> Optional[Dict[str, Any]]:
        """
        Load simulation results.

        Args:
            simulation_id: Simulation identifier

        Returns:
            Dict with simulation metadata and time series data

        Example:
            >>> sim = db.load_simulation("sim_001")
            >>> print(f"Average CH4: {sim['avg_Q_ch4']:.1f} m³/d")
        """
        with self.get_session() as session:
            simulation = session.query(Simulation).filter(Simulation.id == simulation_id).first()

            if not simulation:
                return None

            # Load time series
            time_series = (
                session.query(SimulationTimeSeries)
                .filter(SimulationTimeSeries.simulation_id == simulation_id)
                .order_by(SimulationTimeSeries.time)
                .all()
            )

            # Convert to DataFrame
            if time_series:
                ts_data = {
                    "time": [ts.time for ts in time_series],
                    "Q_gas": [ts.Q_gas for ts in time_series],
                    "Q_ch4": [ts.Q_ch4 for ts in time_series],
                    "Q_co2": [ts.Q_co2 for ts in time_series],
                    "CH4_content": [ts.CH4_content for ts in time_series],
                    "pH": [ts.pH for ts in time_series],
                    "VFA": [ts.VFA for ts in time_series],
                    "TAC": [ts.TAC for ts in time_series],
                    "FOS_TAC": [ts.FOS_TAC for ts in time_series],
                    "P_el": [ts.P_el for ts in time_series],
                    "P_th": [ts.P_th for ts in time_series],
                }
                df = pd.DataFrame(ts_data)
            else:
                df = pd.DataFrame()

            return {
                "id": simulation.id,
                "plant_id": simulation.plant_id,
                "name": simulation.name,
                "description": simulation.description,
                "duration": simulation.duration,
                "scenario": simulation.scenario,
                "parameters": simulation.parameters,
                "avg_Q_gas": simulation.avg_Q_gas,
                "avg_Q_ch4": simulation.avg_Q_ch4,
                "avg_CH4_content": simulation.avg_CH4_content,
                "avg_pH": simulation.avg_pH,
                "avg_VFA": simulation.avg_VFA,
                "total_energy": simulation.total_energy,
                "status": simulation.status,
                "created_at": simulation.created_at,
                "completed_at": simulation.completed_at,
                "time_series": df,
            }

    def list_simulations(self, plant_id: Optional[str] = None, scenario: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all simulations.

        Args:
            plant_id: Filter by plant ID
            scenario: Filter by scenario

        Returns:
            List of simulation summaries
        """
        with self.get_session() as session:
            query = session.query(Simulation)

            if plant_id:
                query = query.filter(Simulation.plant_id == plant_id)
            if scenario:
                query = query.filter(Simulation.scenario == scenario)

            simulations = query.order_by(Simulation.created_at.desc()).all()

            return [
                {
                    "id": s.id,
                    "plant_id": s.plant_id,
                    "name": s.name,
                    "scenario": s.scenario,
                    "duration": s.duration,
                    "avg_Q_ch4": s.avg_Q_ch4,
                    "status": s.status,
                    "created_at": s.created_at,
                }
                for s in simulations
            ]

    # ========================================================================
    # Calibration History
    # ========================================================================

    def store_calibration(
        self,
        plant_id: str,
        calibration_type: str,
        method: str,
        parameters: Dict[str, float],
        objective_value: float,
        objectives: List[str],
        validation_metrics: Optional[Dict[str, float]] = None,
        data_start: Optional[datetime] = None,
        data_end: Optional[datetime] = None,
        success: bool = True,
        message: Optional[str] = None,
    ) -> Calibration:
        """
        Store calibration results.

        Args:
            plant_id: Plant identifier
            calibration_type: 'initial' or 'online'
            method: Optimization method used
            parameters: Calibrated parameter values
            objective_value: Final objective function value
            objectives: List of objectives optimized
            validation_metrics: Validation metrics dict
            data_start: Start of calibration data
            data_end: End of calibration data
            success: Whether calibration succeeded
            message: Status message

        Returns:
            Created Calibration object

        Example:
            >>> db.store_calibration(
            ...     "plant1",
            ...     "initial",
            ...     "differential_evolution",
            ...     {"k_dis": 0.55, "Y_su": 0.105},
            ...     0.123,
            ...     ["Q_ch4", "pH"],
            ...     validation_metrics={"Q_ch4_r2": 0.85}
            ... )
        """
        with self.get_session() as session:
            calibration = Calibration(
                plant_id=plant_id,
                calibration_type=calibration_type,
                method=method,
                parameters=parameters,
                objective_value=objective_value,
                objectives=objectives,
                validation_metrics=validation_metrics,
                data_start=data_start,
                data_end=data_end,
                success=success,
                message=message,
            )

            session.add(calibration)
            session.commit()

            print(f"✓ Stored calibration for plant '{plant_id}'")
            return calibration

    def load_calibrations(
        self, plant_id: str, calibration_type: Optional[str] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Load calibration history.

        Args:
            plant_id: Plant identifier
            calibration_type: Filter by type ('initial' or 'online')
            limit: Maximum number of records to return

        Returns:
            List of calibration records
        """
        with self.get_session() as session:
            query = session.query(Calibration).filter(Calibration.plant_id == plant_id)

            if calibration_type:
                query = query.filter(Calibration.calibration_type == calibration_type)

            calibrations = query.order_by(Calibration.created_at.desc()).limit(limit).all()

            return [
                {
                    "id": c.id,
                    "calibration_type": c.calibration_type,
                    "method": c.method,
                    "parameters": c.parameters,
                    "objective_value": c.objective_value,
                    "objectives": c.objectives,
                    "validation_metrics": c.validation_metrics,
                    "success": c.success,
                    "created_at": c.created_at,
                }
                for c in calibrations
            ]

    def get_latest_calibration(self, plant_id: str) -> Optional[Dict[str, Any]]:
        """
        Get most recent calibration for plant.

        Args:
            plant_id: Plant identifier

        Returns:
            Latest calibration dict or None
        """
        calibrations = self.load_calibrations(plant_id, limit=1)
        return calibrations[0] if calibrations else None

    # ========================================================================
    # Substrate Data
    # ========================================================================

    def store_substrate(
        self,
        plant_id: str,
        substrate_name: str,
        substrate_type: str,
        sample_date: Union[str, datetime],
        lab_data: Dict[str, float],
        sample_id: Optional[str] = None,
        lab_name: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Substrate:
        """
        Store substrate characterization data.

        Args:
            plant_id: Plant identifier
            substrate_name: Substrate name (e.g., "Maize silage batch 23")
            substrate_type: Substrate type (maize, manure, grass, etc.)
            sample_date: Sample date
            lab_data: Dict with lab analysis results
            sample_id: Laboratory sample ID
            lab_name: Laboratory name
            notes: Additional notes

        Returns:
            Created Substrate object

        Example:
            >>> lab_data = {
            ...     "TS": 32.5,
            ...     "VS": 96.2,
            ...     "pH": 3.9,
            ...     "RP": 8.5,
            ...     "RL": 3.2,
            ...     "NDF": 42.1,
            ...     "BMP": 345.2
            ... }
            >>> db.store_substrate(
            ...     "plant1",
            ...     "Maize silage",
            ...     "maize",
            ...     "2024-01-15",
            ...     lab_data
            ... )
        """
        if isinstance(sample_date, str):
            sample_date = pd.to_datetime(sample_date)

        session = self.SessionLocal()
        try:
            substrate = Substrate(
                plant_id=plant_id,
                substrate_name=substrate_name,
                substrate_type=substrate_type,
                sample_date=sample_date,
                sample_id=sample_id,
                lab_name=lab_name,
                notes=notes,
            )

            # Set all lab data attributes
            for key, value in lab_data.items():
                if hasattr(substrate, key):
                    setattr(substrate, key, value)

            session.add(substrate)
            session.commit()
            session.refresh(substrate)
            session.expunge(substrate)

            print(f"✓ Stored substrate data for '{substrate_name}'")
            return substrate
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def load_substrates(
        self,
        plant_id: str,
        substrate_type: Optional[str] = None,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
    ) -> pd.DataFrame:
        """
        Load substrate characterization data.

        Args:
            plant_id: Plant identifier
            substrate_type: Filter by substrate type
            start_date: Start date for samples
            end_date: End date for samples

        Returns:
            DataFrame with substrate data
        """
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        with self.get_session() as session:
            query = session.query(Substrate).filter(Substrate.plant_id == plant_id)

            if substrate_type:
                query = query.filter(Substrate.substrate_type == substrate_type)
            if start_date:
                query = query.filter(Substrate.sample_date >= start_date)
            if end_date:
                query = query.filter(Substrate.sample_date <= end_date)

            substrates = query.order_by(Substrate.sample_date).all()

            if not substrates:
                return pd.DataFrame()

            # Convert to DataFrame
            data = []
            for s in substrates:
                record = {
                    "sample_date": s.sample_date,
                    "substrate_name": s.substrate_name,
                    "substrate_type": s.substrate_type,
                    "sample_id": s.sample_id,
                    "TS": s.TS,
                    "VS": s.VS,
                    "oTS": s.oTS,
                    "foTS": s.foTS,
                    "RP": s.RP,
                    "RL": s.RL,
                    "RF": s.RF,
                    "NDF": s.NDF,
                    "ADF": s.ADF,
                    "ADL": s.ADL,
                    "pH": s.pH,
                    "NH4_N": s.NH4_N,
                    "TAC": s.TAC,
                    "COD_S": s.COD_S,
                    "BMP": s.BMP,
                    "C_to_N": s.C_to_N,
                    "lab_name": s.lab_name,
                }
                data.append(record)

            return pd.DataFrame(data)

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def _calculate_simulation_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate summary metrics from simulation results."""
        if not results:
            return {}

        Q_gas_values = []
        Q_ch4_values = []
        pH_values = []
        VFA_values = []
        P_el_values = []

        for result in results:
            components = result["components"]
            comp_data = next(iter(components.values()))

            if "Q_gas" in comp_data:
                Q_gas_values.append(comp_data["Q_gas"])
            if "Q_ch4" in comp_data:
                Q_ch4_values.append(comp_data["Q_ch4"])
            if "pH" in comp_data:
                pH_values.append(comp_data["pH"])
            if "VFA" in comp_data:
                VFA_values.append(comp_data["VFA"])

            # Look for CHP data
            for comp_id, comp_result in components.items():
                if "P_el" in comp_result:
                    P_el_values.append(comp_result["P_el"])

        metrics = {}

        if Q_gas_values:
            metrics["avg_Q_gas"] = float(np.mean(Q_gas_values))
        if Q_ch4_values:
            metrics["avg_Q_ch4"] = float(np.mean(Q_ch4_values))
            if Q_gas_values:
                metrics["avg_CH4_content"] = float(np.mean(Q_ch4_values) / np.mean(Q_gas_values) * 100)
        if pH_values:
            metrics["avg_pH"] = float(np.mean(pH_values))
        if VFA_values:
            metrics["avg_VFA"] = float(np.mean(VFA_values))
        if P_el_values:
            # Total energy = average power * duration * 24 hours
            duration = results[-1]["time"] if results else 0
            metrics["total_energy"] = float(np.mean(P_el_values) * duration * 24)

        return metrics

    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute raw SQL query and return results as DataFrame.

        Args:
            query: SQL query string

        Returns:
            DataFrame with query results

        Example:
            >>> df = db.execute_query(
            ...     "SELECT plant_id, AVG(Q_ch4) FROM measurements "
            ...     "GROUP BY plant_id"
            ... )
        """
        return pd.read_sql(query, self.engine)

    def get_statistics(self, plant_id: str) -> Dict[str, Any]:
        """
        Get database statistics for a plant.

        Args:
            plant_id: Plant identifier

        Returns:
            Dict with statistics
        """
        with self.get_session() as session:
            n_measurements = session.query(Measurement).filter(Measurement.plant_id == plant_id).count()

            n_simulations = session.query(Simulation).filter(Simulation.plant_id == plant_id).count()

            n_calibrations = session.query(Calibration).filter(Calibration.plant_id == plant_id).count()

            n_substrates = session.query(Substrate).filter(Substrate.plant_id == plant_id).count()

            # Get date ranges
            first_measurement = (
                session.query(Measurement.timestamp)
                .filter(Measurement.plant_id == plant_id)
                .order_by(Measurement.timestamp)
                .first()
            )

            last_measurement = (
                session.query(Measurement.timestamp)
                .filter(Measurement.plant_id == plant_id)
                .order_by(Measurement.timestamp.desc())
                .first()
            )

            return {
                "plant_id": plant_id,
                "n_measurements": n_measurements,
                "n_simulations": n_simulations,
                "n_calibrations": n_calibrations,
                "n_substrates": n_substrates,
                "first_measurement": first_measurement[0] if first_measurement else None,
                "last_measurement": last_measurement[0] if last_measurement else None,
            }

    def backup_database(self, filepath: str) -> None:
        """
        Create database backup (PostgreSQL dump).

        Args:
            filepath: Path for backup file

        Note:
            Requires pg_dump to be available in PATH
        """
        import subprocess

        # Extract connection info from connection string
        # postgresql://username:password@host:port/database
        parts = self.connection_string.replace("postgresql://", "").split("@")
        user_pass = parts[0].split(":")
        host_db = parts[1].split("/")
        host_port = host_db[0].split(":")

        username = user_pass[0]
        password = user_pass[1] if len(user_pass) > 1 else ""
        host = host_port[0]
        port = host_port[1] if len(host_port) > 1 else "5432"
        database = host_db[1]

        # Set password environment variable
        env = {"PGPASSWORD": password}

        # Execute pg_dump
        cmd = ["pg_dump", "-h", host, "-p", port, "-U", username, "-F", "c", "-f", filepath, database]

        try:
            subprocess.run(cmd, env=env, check=True, capture_output=True)
            print(f"✓ Database backed up to {filepath}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Backup failed: {e.stderr.decode()}")
            raise

    def close(self) -> None:
        """Close database connection and dispose of engine."""
        self.engine.dispose()
        print("✓ Database connection closed")
