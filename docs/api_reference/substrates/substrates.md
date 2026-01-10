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

