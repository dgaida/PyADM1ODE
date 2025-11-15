# pyadm1/configurator/mcp/prompts.py
"""
System Prompts and Templates for LLM-Driven Plant Design

This module provides system prompts, templates, and guidelines to help LLMs
effectively design and configure biogas plants using the PyADM1 MCP tools.

The prompts include:
- Component selection criteria
- Connection rules and validation
- Parameter ranges and defaults
- Best practices for different plant types
- Substrate-specific recommendations
- Troubleshooting guidance
"""

from typing import Dict


# ==============================================================================
# Main System Prompt
# ==============================================================================

SYSTEM_PROMPT = """You are an expert biogas plant engineer specializing in the design and
optimization of agricultural anaerobic digestion systems. You help users design,
configure, and simulate biogas plants using the PyADM1 framework.

Your expertise includes:
- Anaerobic digestion fundamentals (ADM1 model)
- Biogas plant component sizing and selection
- Process optimization and control
- Substrate characterization and feeding strategies
- Energy balance and CHP integration
- Thermal management systems

When helping users:
1. Ask clarifying questions if requirements are unclear
2. Suggest appropriate configurations based on substrate types
3. Validate design decisions (volumes, temperatures, power ratings)
4. Provide realistic parameter values
5. Explain the reasoning behind your recommendations
6. Use the MCP tools in the correct sequence (create → add → connect → initialize → simulate)

Always consider:
- Hydraulic retention time (HRT) appropriate for substrate
- Organic loading rate (OLR) limits
- Temperature stability (mesophilic vs thermophilic)
- Gas production expectations
- Heat demand and supply balance
- Process stability indicators (pH, VFA/TAC ratio)
"""


# ==============================================================================
# Component Selection Guidelines
# ==============================================================================

COMPONENT_SELECTION_GUIDE = """
Component Selection Guidelines
================================

Digesters:
----------
Sizing Rules:
- Hydraulic retention time (HRT): 15-60 days typical for agricultural substrates
- Organic loading rate (OLR): 2-5 kg VS/(m³·d) for mesophilic, up to 8 for thermophilic
- Gas volume: 10-20% of liquid volume (typically 15%)
- Total volume = liquid volume ÷ (1 - 0.15)

Temperature Selection:
- Mesophilic (35°C / 308.15 K): Standard, lower heat demand, stable
- Thermophilic (45-55°C / 318-328 K): Higher gas yield, faster degradation, higher heat demand

Two-Stage Systems:
- Hydrolysis tank: 20-30% of total volume, can be thermophilic
- Main digester: 70-80% of total volume, typically mesophilic
- Better for difficult substrates (high lignocellulosic content)

CHP Units:
----------
Sizing:
- Electrical efficiency: 38-42% for gas engines, 25-30% for micro-turbines
- Thermal efficiency: 40-50% (reject heat)
- Total efficiency: 85-90%
- Size based on expected biogas: ~10 kWh electrical per m³ CH₄

Gas Demand:
- 1 kW electrical ≈ 0.25 m³/h biogas (at 60% CH₄, full load)
- 500 kW CHP needs ~300 m³/h biogas = 7,200 m³/d at full load

Heating Systems:
----------------
Heat Demand:
- Typical: 0.5-1.0 kW/K heat loss coefficient
- Heat demand = k × (T_digester - T_ambient)
- At 35°C digester, 15°C ambient: 10-20 kW per 1000 m³

Heat Supply:
- CHP waste heat should cover 80-100% of demand
- Size CHP to provide sufficient waste heat
- Include auxiliary heating for startup and peak demand
"""


# ==============================================================================
# Connection Rules
# ==============================================================================

CONNECTION_RULES = """
Connection Rules and Best Practices
====================================

Connection Types:
-----------------
1. liquid: Material flow between digesters
   - Use for: Digester chains (series connection)
   - Flow direction: Hydrolysis → Main digester → Post-digester

2. gas: Biogas flow to consumers
   - Use for: Digester → CHP, Digester → Gas storage
   - Multiple digesters can feed same CHP (parallel connection)

3. heat: Thermal energy flow
   - Use for: CHP → Heating systems
   - One CHP can supply multiple heating systems

4. power: Electrical connections
   - Use for: CHP → Grid, CHP → Internal consumers

5. control: Control signals
   - Use for: Sensor → Controller → Actuator

Required Connections:
---------------------
Single-Stage Plant:
- None required (digester can operate standalone)
- Recommended: Digester --[gas]--> CHP (if CHP present)
- Recommended: CHP --[heat]--> Heating (if both present)

Two-Stage Plant:
- Required: Hydrolysis --[liquid]--> Main digester
- Recommended: Both digesters --[gas]--> CHP
- Recommended: CHP --[heat]--> All heating systems

Validation Rules:
-----------------
1. No circular liquid connections (prevents infinite loops)
2. Gas consumers must have gas inputs
3. Heating systems should have heat sources (or warn about auxiliary)
4. At least one digester required for functional plant
5. CHP without gas input will not operate

Common Patterns:
----------------
Pattern 1: Simple single-stage with CHP
  Digester --[gas]--> CHP --[heat]--> Heating

Pattern 2: Two-stage with integrated CHP
  Hydrolysis --[liquid]--> Main_Digester
  Hydrolysis --[gas]--> CHP
  Main_Digester --[gas]--> CHP
  CHP --[heat]--> Heating_1
  CHP --[heat]--> Heating_2

Pattern 3: Parallel digesters with shared CHP
  Digester_1 --[gas]--> CHP
  Digester_2 --[gas]--> CHP
  CHP --[heat]--> Heating_1
  CHP --[heat]--> Heating_2
"""


# ==============================================================================
# Parameter Ranges and Defaults
# ==============================================================================

PARAMETER_RANGES = """
Parameter Ranges and Default Values
====================================

Digester Parameters:
--------------------
V_liq (Liquid Volume):
- Range: 500 - 5000 m³ typical, up to 10,000 m³ for large plants
- Small farm: 500-1500 m³
- Medium farm: 1500-3000 m³
- Large plant: 3000-5000 m³
- Default: 1977 m³ (validated reference plant)

V_gas (Gas Volume):
- Typically: 10-20% of V_liq
- Range: 0.10 × V_liq to 0.20 × V_liq
- Default: 304 m³ (15.4% of 1977 m³)
- Larger for intermittent gas use, smaller with continuous CHP

T_ad (Operating Temperature):
- Mesophilic: 308.15 K (35°C) - MOST COMMON
- Upper mesophilic: 313.15 K (40°C)
- Thermophilic: 318.15 K (45°C) to 328.15 K (55°C)
- Psychrophilic: 283.15 K (10°C) - rare, low-temp applications
- Default: 308.15 K (35°C)

CHP Parameters:
---------------
P_el_nom (Electrical Power):
- Range: 50 - 2000 kW for agricultural plants
- Small: 75-250 kW
- Medium: 250-500 kW
- Large: 500-1000 kW
- Default: 500 kW
- Sizing: Match to expected biogas (100-150 m³/h per 500 kW)

eta_el (Electrical Efficiency):
- Gas engines: 0.38 - 0.42 (typical: 0.40)
- Micro turbines: 0.25 - 0.30
- Fuel cells: 0.45 - 0.55 (rare)
- Default: 0.40

eta_th (Thermal Efficiency):
- Range: 0.40 - 0.50
- Typical: 0.45
- Total efficiency (eta_el + eta_th): 0.85 - 0.90
- Default: 0.45

Heating Parameters:
-------------------
target_temperature:
- Should match digester T_ad
- Typical: 308.15 K (35°C) for mesophilic
- Range: 303.15 K (30°C) to 328.15 K (55°C)
- Default: 308.15 K

heat_loss_coefficient:
- Well insulated: 0.3 - 0.5 kW/K
- Average insulation: 0.5 - 0.8 kW/K
- Poor insulation: 0.8 - 1.5 kW/K
- Default: 0.5 kW/K
- Increases with tank surface area

Operational Parameters:
-----------------------
feeding_freq (Feeding Frequency):
- Continuous: 1-4 hours
- Daily: 24 hours
- Intermittent: 48-72 hours
- Default: 48 hours
- More frequent = more stable process

Substrate Feed (Q):
- Total: 20-40 m³/d typical for 2000 m³ digester
- HRT = V_liq / Q_total
- Target HRT: 20-40 days for energy crops
- Target HRT: 30-60 days for manure-heavy mixes
"""


# ==============================================================================
# Substrate-Specific Recommendations
# ==============================================================================

SUBSTRATE_RECOMMENDATIONS = """
Substrate-Specific Design Recommendations
==========================================

Energy Crops (Corn Silage, Grass, Whole Crop Silage):
------------------------------------------------------
Characteristics:
- High VS content (28-35% TS)
- Good biogas potential (500-700 Nl/kg VS)
- Can cause foaming at high loading

Recommended Configuration:
- Single-stage mesophilic (35°C)
- HRT: 25-40 days
- OLR: 3-4 kg VS/(m³·d)
- Consider two-stage if OLR > 4

Feed Strategy:
- Mix 40-70% energy crops with 30-60% manure
- Example: 15 m³/d corn silage + 10 m³/d cattle manure

Animal Manure (Cattle, Pig, Poultry):
--------------------------------------
Characteristics:
- Lower VS (6-12% TS for liquid manure)
- Moderate biogas potential (200-400 Nl/kg VS)
- High ammonia content (potential inhibition)
- Good buffering capacity

Recommended Configuration:
- Single-stage mesophilic (35°C)
- HRT: 30-60 days
- OLR: 2-3 kg VS/(m³·d)
- Larger volume due to low TS

Feed Strategy:
- Use as base substrate (50-80% of mix)
- Add energy crops for higher gas yield
- Monitor ammonia (keep < 4 g/L)

Mixed Substrates (Co-Digestion):
---------------------------------
Typical Mixture Example:
- 60% corn silage (energy)
- 35% cattle manure (base, buffering)
- 5% other (grass, food waste)

Benefits:
- Balanced C/N ratio
- Stable process
- Higher gas yield than manure alone
- Better economics

Recommended Configuration:
- Single-stage mesophilic (35°C)
- HRT: 30-40 days
- OLR: 3-4 kg VS/(m³·d)
- Volume: 1500-2500 m³ for 400-500 kW CHP

Challenging Substrates:
-----------------------
High Lignocellulose (Straw, Wood):
- Two-stage: thermophilic hydrolysis (50°C) + mesophilic main
- Longer HRT: 40-60 days
- Pre-treatment may be needed

High Protein (Chicken manure):
- Risk of ammonia inhibition
- Dilute with C-rich substrates
- Monitor pH and ammonia closely
- Consider lower OLR

High Lipids (Food industry waste):
- Risk of LCFA accumulation
- Limit to < 20% of mix
- Longer HRT needed
- Monitor VFA/TAC ratio
"""


# ==============================================================================
# Best Practices
# ==============================================================================

BEST_PRACTICES = """
Design Best Practices
=====================

Plant Sizing:
-------------
1. Start with substrate availability
   - Calculate total substrate: 8,000-12,000 t/year for 500 kW CHP
   - Account for seasonal variations

2. Calculate expected biogas
   - Energy crops: 600 Nl/kg VS × VS content
   - Manure: 300 Nl/kg VS × VS content
   - Total biogas = Σ(substrate × VS × biogas potential)

3. Size digester volume
   - V_liq = Q_total × HRT
   - Q_total = total feed (m³/d)
   - HRT = 30-40 days typical

4. Size CHP
   - Expected CH₄ = 60% of biogas
   - Energy = CH₄ (m³/d) × 10 kWh/m³
   - P_el = Energy × eta_el / 24 hours

Process Stability:
------------------
Monitor these indicators:
- pH: 6.8 - 7.5 optimal (alarm if < 6.5 or > 8.0)
- VFA: < 2 g/L stable, 2-4 g/L caution, > 4 g/L critical
- VFA/TAC: < 0.3 stable, 0.3-0.4 caution, > 0.4 critical
- Ammonia: < 3 g/L safe, 3-4 g/L monitor, > 4 g/L risk

Safety margin:
- Don't run at maximum OLR continuously
- Keep 20% capacity reserve
- Have auxiliary heating backup

Economic Considerations:
------------------------
1. CHP sizing
   - Minimum: 4000-5000 hours/year operation
   - Optimal: 7000-8000 hours/year
   - Don't oversize (poor part-load efficiency)

2. Heat utilization
   - Use CHP heat for digester heating first
   - Additional heat: buildings, drying, district heating
   - Heat utilization > 60% improves economics

3. Substrate costs
   - Balance cheap substrates (manure) with high-yield (crops)
   - Consider transport distances
   - Account for seasonal availability

Startup and Operation:
----------------------
1. Initial startup (cold start)
   - Load inoculum (20-30% of volume)
   - Start with low OLR (1-2 kg VS/(m³·d))
   - Gradually increase over 2-3 months
   - Use auxiliary heating

2. Substrate changes
   - Change gradually (< 10% per week)
   - Monitor process indicators closely
   - Have backup substrate available

3. Maintenance
   - CHP: 2000-4000 hour service intervals
   - Mixing: Check daily
   - Sensors: Calibrate monthly
   - Clean gas lines quarterly
"""


# ==============================================================================
# Troubleshooting Guide
# ==============================================================================

TROUBLESHOOTING_GUIDE = """
Common Issues and Solutions
============================

Low Gas Production:
-------------------
Possible Causes:
1. Overloading (OLR too high)
   → Reduce feed rate
   → Monitor VFA/TAC ratio

2. Temperature too low
   → Check heating system
   → Increase heat supply

3. Substrate quality issues
   → Check VS content
   → Verify substrate degradability

4. Inhibition (ammonia, VFA)
   → Reduce feed rate
   → Adjust substrate mix
   → Add trace elements

Process Instability (High VFA):
--------------------------------
Symptoms: VFA > 4 g/L, VFA/TAC > 0.4, pH drop

Solutions:
1. Stop or reduce feeding immediately
2. Add buffer (if available in model)
3. Increase temperature if possible
4. Wait for recovery (may take weeks)
5. Check for substrate contamination

Prevention:
- Gradual feed increases
- Monitor VFA daily
- Keep OLR below limits
- Maintain temperature stability

Low CHP Efficiency:
-------------------
Causes:
1. Part-load operation
   → CHP oversized for available gas
   → Consider smaller CHP

2. Poor maintenance
   → Service CHP regularly
   → Check for leaks in gas system

3. Low methane content
   → Check digester operation
   → Look for air leaks

High Heat Demand:
-----------------
Causes:
1. Poor insulation
   → Increase heat_loss_coefficient in model
   → In reality: improve insulation

2. Low ambient temperature
   → Normal in winter
   → Ensure adequate CHP heat

3. Temperature too high
   → Reassess if 55°C is necessary
   → Consider reducing to 35-40°C

Temperature Fluctuations:
-------------------------
Causes:
1. Inadequate heating capacity
   → Size heating system larger
   → Reduce heat loss coefficient

2. Heating system issues
   → Check CHP operation
   → Verify heat exchanger

3. Substrate temperature variations
   → Pre-heat substrates in winter
   → Store substrates indoors
"""


# ==============================================================================
# Example Configurations
# ==============================================================================

EXAMPLE_CONFIGURATIONS = """
Example Plant Configurations
=============================

Example 1: Small Farm Single-Stage Plant
-----------------------------------------
Application: 100-cow dairy farm with 20 ha energy crops

Configuration:
- Main digester: 1500 m³ liquid, 250 m³ gas, 35°C
- CHP: 250 kW electrical (0.40/0.45 efficiency)
- Heating: 1 system (308.15 K target)

Substrates:
- 25 m³/d cattle manure (base)
- 8 m³/d corn silage (energy)
- Total: 33 m³/d → HRT = 45 days

Expected Performance:
- Biogas: ~900 m³/d
- Methane: ~540 m³/d (60%)
- Electrical: ~225 kW average (90% capacity)
- Operating hours: ~8000 hours/year

Tool Sequence:
1. create_biogas_plant("SmallFarm")
2. add_digester("main", V_liq=1500, V_gas=250)
3. add_chp("chp1", P_el_nom=250)
4. add_heating("heat1")
5. add_connection("main", "chp1", "gas")
6. add_connection("chp1", "heat1", "heat")
7. initialize_plant("SmallFarm")
8. simulate_plant("SmallFarm", duration=30)

Example 2: Medium Two-Stage Plant
----------------------------------
Application: 300-cow farm with 80 ha crops

Configuration:
- Hydrolysis tank: 600 m³ liquid, 100 m³ gas, 45°C
- Main digester: 2400 m³ liquid, 400 m³ gas, 35°C
- CHP: 500 kW electrical (0.40/0.45 efficiency)
- Heating: 2 systems (one per digester)

Substrates:
- 50 m³/d cattle manure
- 25 m³/d corn silage
- 5 m³/d grass silage
- Total: 80 m³/d → HRT hydro = 7.5 days, main = 30 days

Expected Performance:
- Biogas: ~2000 m³/d
- Methane: ~1200 m³/d
- Electrical: ~480 kW average (96% capacity)

Tool Sequence:
1. create_biogas_plant("MediumFarm")
2. add_digester("hydro", V_liq=600, V_gas=100, T_ad=318.15, name="Hydrolysis")
3. add_digester("main", V_liq=2400, V_gas=400, T_ad=308.15, name="Main Digester")
4. add_chp("chp1", P_el_nom=500)
5. add_heating("heat_hydro", target_temperature=318.15)
6. add_heating("heat_main", target_temperature=308.15)
7. add_connection("hydro", "main", "liquid")
8. add_connection("hydro", "chp1", "gas")
9. add_connection("main", "chp1", "gas")
10. add_connection("chp1", "heat_hydro", "heat")
11. add_connection("chp1", "heat_main", "heat")
12. initialize_plant("MediumFarm")
13. simulate_plant("MediumFarm", duration=30)

Example 3: Large Industrial Plant
----------------------------------
Application: Centralized plant with multiple feedstock sources

Configuration:
- Main digester: 5000 m³ liquid, 750 m³ gas, 38°C
- Post-digester: 2000 m³ liquid, 300 m³ gas, 35°C
- CHP: 1000 kW electrical
- Heating: 2 systems

Substrates:
- 80 m³/d various manures
- 60 m³/d corn silage
- 20 m³/d grass silage
- 10 m³/d food waste
- Total: 170 m³/d → HRT main = 29 days, post = 12 days

Expected Performance:
- Biogas: ~4500 m³/d
- Methane: ~2700 m³/d
- Electrical: ~950 kW average

[Tool sequence similar to Example 2, scaled up]
"""


# ==============================================================================
# Helper Functions
# ==============================================================================


def get_prompt_for_plant_type(plant_type: str) -> str:
    """
    Get specific guidance for a plant type.

    Args:
        plant_type: Type of plant ("single_stage", "two_stage", "parallel", "custom")

    Returns:
        Specific guidance text for that plant type
    """
    prompts = {
        "single_stage": """
Single-Stage Plant Design:
- Use one main digester
- Simplest configuration
- Best for: Manure-heavy mixes, stable substrates
- Typical HRT: 30-40 days
- Typical OLR: 2-4 kg VS/(m³·d)
        """,
        "two_stage": """
Two-Stage Plant Design:
- Hydrolysis tank + Main digester
- Better for difficult substrates
- Hydrolysis: Higher temperature possible (45-50°C)
- Main: Typically mesophilic (35°C)
- Total HRT split: 20% hydrolysis, 80% main
- Higher operational complexity
        """,
        "parallel": """
Parallel Digester Design:
- Multiple digesters feeding same CHP
- Good for: Large capacity, operational flexibility
- Can take one offline for maintenance
- Balance loading between digesters
- Share common gas and heating infrastructure
        """,
        "custom": """
Custom Plant Design:
- Define requirements first (substrates, capacity, energy needs)
- Consider: Available space, infrastructure, substrates
- Start simple, add complexity if needed
- Validate design with simulations
        """,
    }
    return prompts.get(plant_type, prompts["custom"])


def get_substrate_guidance(substrate_name: str) -> str:
    """
    Get specific guidance for a substrate type.

    Args:
        substrate_name: Name of substrate (e.g., "corn_silage", "cattle_manure")

    Returns:
        Specific guidance for handling that substrate
    """
    guidance = {
        "corn_silage": """
Corn Silage:
- TS: 28-35%
- VS: 95-97% of TS
- Biogas: 600-700 Nl/kg VS
- Good C/N ratio (~25:1)
- Risk of foaming at high loads
- Mix with manure (40-60% of total mix)
        """,
        "cattle_manure": """
Cattle Manure:
- TS: 6-12% (liquid), 20-25% (solid)
- VS: 75-85% of TS
- Biogas: 200-300 Nl/kg VS
- Good buffer capacity
- Use as base substrate (40-80%)
- Dilutes high-energy substrates
        """,
        "grass_silage": """
Grass Silage:
- TS: 25-35%
- VS: 85-90% of TS
- Biogas: 500-600 Nl/kg VS
- Cut stage affects degradability
- May need longer HRT than corn
- Good alternative energy crop
        """,
        "pig_manure": """
Pig Manure:
- TS: 5-8% (liquid), 20-25% (solid)
- VS: 70-80% of TS
- Biogas: 300-450 Nl/kg VS
- Higher ammonia than cattle
- Monitor ammonia levels
- Good buffering capacity
        """,
    }

    return guidance.get(substrate_name.lower(), "No specific guidance available for this substrate.")


def get_parameter_recommendation(parameter_name: str, plant_size: str = "medium") -> Dict[str, float]:
    """
    Get recommended parameter values based on plant size.

    Args:
        parameter_name: Name of parameter ("V_liq", "P_el_nom", etc.)
        plant_size: Size category ("small", "medium", "large")

    Returns:
        Dictionary with "default", "min", "max" values
    """
    recommendations = {
        "V_liq": {
            "small": {"default": 1000, "min": 500, "max": 1500},
            "medium": {"default": 2000, "min": 1500, "max": 3000},
            "large": {"default": 4000, "min": 3000, "max": 6000},
        },
        "P_el_nom": {
            "small": {"default": 150, "min": 75, "max": 250},
            "medium": {"default": 500, "min": 250, "max": 750},
            "large": {"default": 1000, "min": 750, "max": 1500},
        },
        "T_ad": {
            "small": {"default": 308.15, "min": 303.15, "max": 313.15},
            "medium": {"default": 308.15, "min": 303.15, "max": 318.15},
            "large": {"default": 308.15, "min": 303.15, "max": 318.15},
        },
    }

    param_dict = recommendations.get(parameter_name, {})
    return param_dict.get(plant_size, param_dict.get("medium", {}))
