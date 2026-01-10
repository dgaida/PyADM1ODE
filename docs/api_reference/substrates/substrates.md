# Substrate Management and Characterization

Substrate Management and Characterization

This module handles substrate definitions, characterization, and ADM1 input
stream calculation for agricultural biogas substrates.

Modules:

    feedstock: Main Feedstock class managing substrate mixing and ADM1 input stream
              generation, with support for time-varying substrate feeds and automatic
              weighting of substrate properties based on volumetric flow rates.

    substrate_db: Database interface for substrate properties including built-in
                 database of common agricultural substrates (energy crops, manures,
                 organic waste) with literature values and local calibrations.

    xml_loader: Parser for substrate definition XML files following the schema used
               in SIMBA and other biogas simulation tools, with validation and
               error handling for malformed substrate definitions.

    characterization: Substrate characterization methods for converting laboratory
                     analysis data (Weender, Van Soest, BMP) into ADM1 model
                     parameters, including COD fractionation and stoichiometry.

Example:

```python
    >>> from pyadm1.substrates import Feedstock, SubstrateDB
    >>>
    >>> # Load substrates from database
    >>> db = SubstrateDB()
    >>> corn_silage = db.get_substrate("corn_silage")
    >>>
    >>> # Create feedstock for simulation
    >>> feedstock = Feedstock(feeding_freq=48)
    >>> Q = [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]  # m³/d for each substrate
    >>> influent_df = feedstock.get_influent_dataframe(Q)
```

## Base Classes

### Feedstock

```python
from pyadm1.substrates import Feedstock
```

Manages substrate information and creates ADM1 input streams.

Substrate parameters are loaded from XML files and processed via C# DLLs
to generate ADM1-compatible input streams.

**Signature:**

```python
Feedstock(
    feeding_freq,
    total_simtime=60,
    substrate_xml='substrate_gummersbach.xml'
)
```

**Methods:**

#### `get_influent_dataframe()`

```python
get_influent_dataframe(Q)
```

Generate ADM1 input stream as DataFrame for entire simulation.

The input stream is constant over the simulation duration and depends
on the volumetric flow rate of each substrate.

Parameters
----------
Q : List[float]
    Volumetric flow rates [m³/d], e.g., [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]

Returns
-------
pd.DataFrame
    ADM1 input stream with columns: time, S_su, S_aa, ..., Q

#### `get_substrate_feed_mixtures()`

```python
get_substrate_feed_mixtures(Q, n=13)
```

#### `header()`

```python
header()
```

Names of ADM1 input stream components.

#### `mySubstrates()`

```python
mySubstrates()
```

Substrates object from C# DLL.

#### `simtime()`

```python
simtime()
```

Simulation time array [days].


