# pyadm1/substrates/gummersbach.py
"""
Substrates described by the measured parameters of the legacy Gummersbach library.

The ADM1da fractionation of :class:`~pyadm1.substrates.feedstock.SubstrateParams`
(``aXI``, ``fOTSrf``, ``fsOTS``, ``ffOTS``, ``aSi``) follows SIMBA and is not
measured directly. The legacy Gummersbach library of the MATLAB / C# toolbox
describes a substrate by laboratory values instead: Weender and van Soest
analysis, degradation level, single volatile fatty acids, ammonium and the
titrated acid capacity. This module accepts that parameter set and maps it to
``SubstrateParams``, whose optional measured fields then drive the calculation.

Three ways in:

* ``load_gummersbach_library()`` reads a legacy library file with several
  ``<substrate>`` entries (default: the bundled ``substrate_gummersbach.xml``).
* A substrate ID ``"gummersbach:<id>"`` selects one entry of the bundled
  library wherever a substrate ID is accepted, e.g.
  ``Feedstock(["gummersbach:maize", "gummersbach:swinemanure"])``.
* A flat YAML / TOML / XML substrate file with ``schema: measured`` (or the
  older spelling ``schema: gummersbach``) uses the names and units of
  :data:`MEASURED_PARAMETERS`. The bundled substrate files are written this way.

The inert share comes from one of three keys, in this order: the degradation
level ``D_VS``, the inert share ``aXI``, or a methane potential ``BMP``
measured in a batch test, to which the degradability is then fitted
(:func:`~pyadm1.substrates.feedstock.fit_degradability_to_bmp`).

Differences to the legacy model, all deliberate:

* The legacy model has one disintegration pool with a rate per substrate. Here
  the cell wall decides between the slow and the fast pool, and the rates stay
  parameters of the digester. ``kdis``, ``khyd_*`` and ``km_*`` of the library
  are read and ignored.
* The bicarbonate is the titrated acid capacity minus the acid anions that are
  titrated down to pH 4.3. The legacy model subtracts all acid anions.
* Density uses additive volumes, as everywhere in ADM1da.

Reference: Koch, K., Lübken, M., Gehring, T., Wichern, M., Horn, H. (2010):
Biogas from grass silage - Measurements and modeling with ADM1. Bioresource
Technology 101, 8158-8165.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import fields, replace
from functools import lru_cache
from itertools import pairwise
from pathlib import Path

from .feedstock import (
    _BMP_RECOVERY_KEY,
    _DEFAULT_DATA_DIR,
    DEFAULT_BMP_RECOVERY,
    Feedstock,
    SubstrateParams,
    fit_degradability_to_bmp,
)

#: Value of the ``schema`` key that selects this parameter set in a substrate file.
SCHEMA_NAME = "measured"

#: All accepted spellings of that value; ``gummersbach`` is the older one.
SCHEMA_NAMES = ("measured", "gummersbach")

#: The legacy library bundled with the package.
DEFAULT_LIBRARY = _DEFAULT_DATA_DIR / "legacy" / "substrate_gummersbach.xml"

#: Parameters of the schema: name -> (unit, meaning). ``TS``, ``VS``, ``RP``,
#: ``RL`` and ``pH`` are required; ``RF`` is required unless ``NDF`` and
#: ``ADL`` are given.
MEASURED_PARAMETERS: dict[str, tuple[str, str]] = {
    "TS": ("% FM", "total solids"),
    "VS": ("% TS", "volatile solids"),
    "RF": ("% TS", "crude fibre (Weender)"),
    "RP": ("% TS", "crude protein"),
    "RL": ("% TS", "crude lipids"),
    "NDF": ("% TS", "neutral detergent fibre: whole cell wall"),
    "ADF": ("% TS", "acid detergent fibre: cellulose and lignin (stored only)"),
    "ADL": ("% TS", "acid detergent lignin"),
    "NDICP": ("% TS", "neutral detergent insoluble crude protein: protein bound in the cell wall"),
    "D_VS": ("100 %", "degradation level of the volatile solids"),
    "BMP": ("Nm³ CH4/t VS", "methane potential of a batch test; sets the inert share without D_VS and aXI"),
    "pH": ("-", "pH value"),
    "Sac": ("g/l", "acetic acid and acetate"),
    "Spro": ("g/l", "propionic acid and propionate"),
    "Sbu": ("g/l", "butyric acid and butyrate"),
    "Sva": ("g/l", "valeric acid and valerate"),
    "Snh4": ("g/l", "ammonium, as NH4"),
    "TAC": ("mmol/l", "titrated acid capacity to pH 4.3"),
    "T": ("°C", "storage temperature"),
    "COD_S": ("gCOD/l", "COD of the filtrate"),
    "SIin": ("gCOD/l", "soluble inert COD"),
}

#: Further keys a flat file may carry, in the units of ``SubstrateParams``.
_NATIVE_KEYS = (
    "aXI",
    "aSi",
    "fOTSrf",
    "fsOTS",
    "ffOTS",
    "NH4",
    "KS43",
    "FFS",
    "Temp",
    "BGP",
    "cost",
    _BMP_RECOVERY_KEY,
    "X_bac",
    "X_meth",
)

#: Keys of the legacy library without a counterpart here: the kinetic rates
#: are digester parameters in ADM1da.
_IGNORED_KEYS = ("kdis", "khyd_ch", "khyd_pr", "khyd_li", "km_c4", "km_pro", "km_ac", "km_h2")

#: Model constants of ``SubstrateParams`` a file may override: densities, COD
#: factors, methane potentials and equilibrium constants.
_CONSTANT_PREFIXES = ("roh_", "M_", "MP_", "K", "V_m", "CH4_cod_2_mol", "N_aa")

#: Keys that steer the substrate-borne biomass.
_BIOMASS_KEYS = ("substrate_class", "age")

#: Units the library file must state for the values to be read correctly.
#: ``T``, ``BMP`` and ``cost`` are left out: their unit strings carry non-ASCII
#: characters that old files store in differing encodings.
_CHECKED_UNITS = {key: unit for key, (unit, _) in MEASURED_PARAMETERS.items() if key not in ("T", "BMP")}

# Substrate-borne biomass of manure in the legacy model (after Lübken et al.).
_CELLS_PER_ML = 1.47e10
_BACTERIA_SHARE = 0.85
_METHANOGEN_SHARE = 0.01
_CELL_MASS_MG = 14.0e-11
_COD_PER_BIOMASS = 1.416  # gCOD/g biomass
_BACTERIA_DECAY_DAYS = 10.0  # one decade per 10 days
_METHANOGEN_DECAY_DAYS = 5.0

#: Storage age of manure the legacy model assumes when none is given [d].
LEGACY_MANURE_AGE = 10.0

_METHANE_SHARE_OF_BIOGAS = 0.55
_DEFAULT_TEMPERATURE = 15.0  # °C

_MANURE_PATTERN = re.compile(r"mist|g\S{1,3}lle|manure|slurry|dung", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def is_manure_class(substrate_class: str | None) -> bool:
    """
    Tell whether a substrate class names a manure.

    The legacy model keys the substrate-borne biomass to the EEG classes
    "Rindergülle", "Schweinegülle", "Rinderfestmist", "Schweinefestmist",
    "Schafs-/Ziegenmist" and "Pferdemist". Matching is by word stem, so a
    wrongly decoded umlaut does not matter.

    Args:
        substrate_class: Class text of the substrate, or ``None``.

    Returns:
        True for manure and slurry classes.
    """
    return bool(substrate_class) and _MANURE_PATTERN.search(str(substrate_class)) is not None


def manure_biomass(age_days: float = LEGACY_MANURE_AGE) -> tuple[float, float]:
    """
    Living biomass in manure after the legacy model.

    Fresh manure carries 1.47e10 cells/ml of 14e-11 mg each, 85 % bacteria and
    1 % methanogens. Both decay by one decade per 10 and 5 days of storage.

    Args:
        age_days: Storage age of the manure [d].

    Returns:
        Bacteria and methanogens [kgCOD/m³].
    """
    fresh = _CELLS_PER_ML * _CELL_MASS_MG * _COD_PER_BIOMASS  # mg/ml equals kg/m³
    bacteria = fresh * _BACTERIA_SHARE * 10.0 ** (-age_days / _BACTERIA_DECAY_DAYS)
    methanogens = fresh * _METHANOGEN_SHARE * 10.0 ** (-age_days / _METHANOGEN_DECAY_DAYS)
    return bacteria, methanogens


def fibre_degradability(NDF: float, ADL: float, VS: float, D_VS: float) -> float:
    """
    Degradable share ``d`` of the structural carbohydrates (Koch et al. 2010).

    The non-degradable volatile solids ``VS * (1 - D_VS)`` consist of the
    lignin and of the share ``1 - d`` of the structural carbohydrates:
    ``d = (NDF - VS * (1 - D_VS)) / (NDF - ADL)``. The result is limited to
    0..1; the legacy model raises an error outside that range.

    Args:
        NDF: Neutral detergent fibre [% TS or fraction of TS].
        ADL: Acid detergent lignin, same unit.
        VS: Volatile solids, same unit.
        D_VS: Degradation level of the volatile solids [-].

    Returns:
        ``d`` between 0 and 1; 0 without structural carbohydrates.
    """
    if NDF - ADL <= 0.0:
        return 0.0
    return min(max((NDF - VS * (1.0 - D_VS)) / (NDF - ADL), 0.0), 1.0)


def _number(raw: Mapping[str, object], key: str, default: float | None = None) -> float | None:
    """Read *key* as float; ``None`` and missing keys give *default*."""
    value = raw.get(key)
    return default if value is None else float(value)  # type: ignore[arg-type]


def _check_keys(raw: Mapping[str, object], source: str) -> None:
    """Reject keys that belong to neither parameter set, so a typo is not silently dropped."""
    constants = {f.name for f in fields(SubstrateParams) if f.name.startswith(_CONSTANT_PREFIXES)}
    known = set(MEASURED_PARAMETERS) | set(_NATIVE_KEYS) | set(_IGNORED_KEYS) | set(_BIOMASS_KEYS) | constants
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(
            f"Unknown parameter(s) {unknown} in substrate {source or 'definition'} "
            f"(schema '{SCHEMA_NAME}'); measured parameters are {sorted(MEASURED_PARAMETERS)}."
        )


def _require(raw: Mapping[str, object], key: str, source: str) -> float:
    """Read a required parameter."""
    value = _number(raw, key)
    if value is None:
        raise ValueError(
            f"Required parameter '{key}' not found in substrate {source or 'definition'} (schema '{SCHEMA_NAME}')."
        )
    return value


# ---------------------------------------------------------------------------
# Measured parameters -> SubstrateParams
# ---------------------------------------------------------------------------


def params_from_measured(name: str, raw: Mapping[str, object], source: str = "") -> SubstrateParams:
    """
    Build :class:`SubstrateParams` from the measured parameters of the Gummersbach schema.

    Names and units follow :data:`MEASURED_PARAMETERS`. The fields of the
    SIMBA fractionation that the calculation no longer reads are filled with
    their derived equivalents (``aXI = 1 - D_VS``, ``fOTSrf`` = Koch ``d``,
    ``KS43 = TAC``, ``FFS`` = sum of the acids, ``BMP`` = model potential), so
    code that reads them keeps working.

    The inert share comes from ``D_VS``, else from ``aXI``, else it is fitted
    to the measured ``BMP`` (``BMP_recovery`` sets the share of the degradable
    COD a batch test recovers as methane, default 1).

    What remains unmeasured:

    * ``fsOTS`` / ``ffOTS`` split protein and lipids between the slow and the
      fast pool. Default: slow for manure classes, fast for everything else,
      as in the SIMBA class values. With ``NDICP`` the protein follows the
      measurement and the two shares only apply to the lipids.

    Args:
        name: Human-readable substrate label.
        raw: Parameter name -> value. May also carry ``substrate_class`` and
            ``age`` [d]; a manure class with an age gets the substrate-borne
            biomass of :func:`manure_biomass` unless ``X_bac`` / ``X_meth``
            are given. ``cost`` is in EUR/t FM.
        source: File name for error messages.

    Returns:
        The substrate parameters.

    Raises:
        ValueError: A required parameter is missing, a key is unknown or a
            value is out of range.
    """
    _check_keys(raw, source)

    ts = _require(raw, "TS", source)
    vs = _require(raw, "VS", source)
    if not 0.0 <= ts <= 100.0 or not 0.0 <= vs <= 100.0:
        raise ValueError(f"Substrate {source or name}: TS [% FM] and VS [% TS] must lie between 0 and 100.")

    ndf, adl, adf = _number(raw, "NDF"), _number(raw, "ADL"), _number(raw, "ADF")
    cell_wall = ndf is not None and adl is not None
    rf = _number(raw, "RF", 0.0) if cell_wall else _require(raw, "RF", source)

    # Organic fractions cannot exceed the organic matter, and the detergent
    # fibres are nested (lignin within ADF within NDF). Half a percentage
    # point is allowed for rounding in analysis reports.
    organic = _number(raw, "RP", 0.0) + _number(raw, "RL", 0.0) + (ndf if cell_wall else rf)
    if organic > vs + 0.5:
        raise ValueError(
            f"Substrate {source or name}: protein, lipids and fibre add up to {organic:.1f} % TS, "
            f"more than the volatile solids ({vs:.1f} % TS)."
        )
    nested = [value for value in (adl, adf, ndf) if value is not None]
    if any(low > high + 0.5 for low, high in pairwise(nested)):
        raise ValueError(f"Substrate {source or name}: the detergent fibres must satisfy ADL <= ADF <= NDF.")
    d_vs = _number(raw, "D_VS")
    ndicp = _number(raw, "NDICP")
    if ndicp is not None and not cell_wall:
        raise ValueError(f"Substrate {source or name}: 'NDICP' is part of the cell wall and needs 'NDF' and 'ADL'.")

    substrate_class = raw.get("substrate_class")
    substrate_class = None if substrate_class is None else str(substrate_class)
    manure = is_manure_class(substrate_class)

    # --- Fractionation that is not measured --------------------------------
    a_si = _number(raw, "aSi", 0.0)
    measured_bmp = _number(raw, "BMP")
    fit_to_bmp = False
    if d_vs is not None:
        a_xi = max(1.0 - d_vs - (0.0 if cell_wall else a_si), 0.0)
    else:
        a_xi = _number(raw, "aXI")
        if a_xi is None:
            if measured_bmp is None:
                raise ValueError(
                    f"Substrate {source or name}: give 'D_VS', 'aXI' or a measured 'BMP'; "
                    "without any of them the inert share is unknown."
                )
            fit_to_bmp = True
            a_xi = 0.0  # placeholder until the fit below
    koch = cell_wall and d_vs is not None
    f_ots_rf = fibre_degradability(ndf, adl, vs, d_vs) if koch else _number(raw, "fOTSrf", 1.0)
    fs_ots = _number(raw, "fsOTS", 1.0 if manure else 0.0)
    ff_ots = _number(raw, "ffOTS", 1.0 - fs_ots)

    # --- Acids, buffer, ammonium ---------------------------------------------
    acids = {key: _number(raw, key) for key in ("Sac", "Spro", "Sbu", "Sva")}
    has_acids = any(value is not None for value in acids.values())
    ffs = sum(value or 0.0 for value in acids.values()) if has_acids else _number(raw, "FFS", 0.0)

    tac = _number(raw, "TAC")
    ks43 = tac if tac is not None else _number(raw, "KS43", 0.0)

    # The legacy model converts Snh4 with the molar mass of NH4 (18 g/mol).
    snh4 = _number(raw, "Snh4")
    nh4 = snh4 * 14.0 / 18.0 if snh4 is not None else _number(raw, "NH4", 0.0)

    # --- Substrate-borne biomass -----------------------------------------------
    x_bac, x_meth = _number(raw, "X_bac"), _number(raw, "X_meth")
    age = _number(raw, "age")
    if manure and age is not None:
        legacy_bac, legacy_meth = manure_biomass(age)
        x_bac = legacy_bac if x_bac is None else x_bac
        x_meth = legacy_meth if x_meth is None else x_meth

    kwargs: dict = {
        "name": name,
        "TS": ts * 10.0,  # % FM -> kg/t FM
        "NH4": nh4,
        "BGP": _number(raw, "BGP", 0.0),
        "BMP": measured_bmp or 0.0,
        "aXI": a_xi,
        "fOTSrf": f_ots_rf,
        "fsOTS": fs_ots,
        "ffOTS": ff_ots,
        "aSi": a_si,
        "fRF": rf / 100.0,
        "fRP": _require(raw, "RP", source) / 100.0,
        "fRFe": _require(raw, "RL", source) / 100.0,
        "fRA": 1.0 - vs / 100.0,
        "Temp": _number(raw, "T", _number(raw, "Temp", _DEFAULT_TEMPERATURE)),
        "pH": _require(raw, "pH", source),
        "KS43": ks43,
        "FFS": ffs,
        "cost": _number(raw, "cost", 0.0),
        "NDF": ndf / 100.0 if cell_wall else None,
        "ADF": adf / 100.0 if adf is not None else None,
        "ADL": adl / 100.0 if cell_wall else None,
        "D_VS": d_vs,
        "TAC": tac,
        "COD_S": _number(raw, "COD_S"),
        "SIin": _number(raw, "SIin"),
        "X_bac": x_bac or 0.0,
        "X_meth": x_meth or 0.0,
        "NDICP": ndicp / 100.0 if ndicp is not None else None,
        "substrate_class": substrate_class,
    }
    if has_acids:
        kwargs.update(acids)

    # Model constants (densities, COD factors, equilibrium constants) may be overridden.
    for f in fields(SubstrateParams):
        if f.name not in kwargs and raw.get(f.name) is not None:
            kwargs[f.name] = float(raw[f.name])  # type: ignore[arg-type]

    params = SubstrateParams(**kwargs)

    if fit_to_bmp:
        params = fit_degradability_to_bmp(params, recovery=_number(raw, _BMP_RECOVERY_KEY, DEFAULT_BMP_RECOVERY))
        if cell_wall:
            # Keep the unused SIMBA fields in step with the fitted degradation level.
            params = replace(params, aXI=max(1.0 - params.D_VS, 0.0), fOTSrf=fibre_degradability(ndf, adl, vs, params.D_VS))

    # Without a measured BMP the field is a reference value: the model potential.
    if measured_bmp is None:
        params = replace(params, BMP=Feedstock(params).bmp_theoretical())
    if raw.get("BGP") is None:
        params = replace(params, BGP=params.BMP / _METHANE_SHARE_OF_BIOGAS)
    return params


# ---------------------------------------------------------------------------
# Legacy library file
# ---------------------------------------------------------------------------


def load_gummersbach_library(
    path: str | Path | None = None, manure_age: float | None = LEGACY_MANURE_AGE
) -> dict[str, SubstrateParams]:
    """
    Read a legacy Gummersbach substrate library.

    The file holds several ``<substrate id="...">`` entries, each a list of
    ``<physValue symbol="...">`` elements with ``<value>`` and ``<unit>``.

    Args:
        path: Library file; defaults to the bundled ``substrate_gummersbach.xml``.
        manure_age: Storage age [d] for the substrate-borne biomass of manure
            classes, as in the legacy model. ``None`` feeds no biomass.

    Returns:
        Substrate ID -> parameters, in file order. ``cost`` is converted from
        EUR/m³ of the legacy file to EUR/t FM.

    Raises:
        FileNotFoundError: The file does not exist.
        ValueError: A value carries another unit than the schema expects.
    """
    return {sid: replace(params) for sid, params in _load_library(str(Path(path or DEFAULT_LIBRARY)), manure_age).items()}


@lru_cache(maxsize=8)
def _load_library(path: str, manure_age: float | None) -> dict[str, SubstrateParams]:
    """Parse the library once per file and manure age."""
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"Gummersbach substrate library not found: {file}")

    root = ET.parse(file).getroot()  # noqa: S314 - trusted local substrate definition files, not untrusted input
    library: dict[str, SubstrateParams] = {}
    for element in root.iter("substrate"):
        sid = element.get("id")
        if sid is None:
            continue
        raw: dict[str, object] = {}
        for phys in element.iter("physValue"):
            symbol = phys.get("symbol")
            value = phys.findtext("value")
            if symbol is None or value is None:
                continue
            unit = (phys.findtext("unit") or "").strip()
            expected = _CHECKED_UNITS.get(symbol)
            if expected is not None and unit != expected:
                raise ValueError(f"Substrate '{sid}' in {file.name}: {symbol} is given in '{unit}', expected '{expected}'.")
            raw[symbol] = float(value)

        cost_per_m3 = float(raw.pop("cost", 0.0))  # type: ignore[arg-type]
        raw["substrate_class"] = element.findtext("substrate_class")
        if manure_age is not None:
            raw["age"] = manure_age

        label = (element.findtext("name") or sid).strip()
        params = params_from_measured(f"{label} (Gummersbach library)", raw, source=f"'{sid}' in {file.name}")
        density = Feedstock._calc_density(params)
        library[sid] = replace(params, cost=cost_per_m3 * 1000.0 / density)
    return library


def gummersbach_substrate(substrate_id: str, path: str | Path | None = None) -> SubstrateParams:
    """
    Return one substrate of a legacy Gummersbach library.

    Args:
        substrate_id: ID of the ``<substrate>`` entry, e.g. ``"maize"``.
        path: Library file; defaults to the bundled library.

    Returns:
        The substrate parameters.

    Raises:
        KeyError: The library has no such substrate.
    """
    library = load_gummersbach_library(path)
    if substrate_id not in library:
        raise KeyError(f"Substrate '{substrate_id}' not found in the Gummersbach library. Available: {sorted(library)}")
    return library[substrate_id]
