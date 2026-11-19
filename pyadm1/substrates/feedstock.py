# pyadm1/substrates/feedstock.py
"""
Pure-Python substrate characterization for the ADM1da model.

Substrate definitions are stored under ``data/substrates/`` and may be
written as YAML (canonical), XML, or TOML.  ``load_substrate`` dispatches on
the file extension; format-specific helpers (``load_substrate_yaml``,
``load_substrate_xml``, ``load_substrate_toml``) are also exposed.
``SubstrateRegistry`` discovers all supported files in the directory.

``Feedstock`` converts the characterization data into the 38-column influent
DataFrame expected by ``ADM1.set_influent_dataframe()``.

No C#/.NET DLL dependency — works in any Python environment.

Reference: Schlattmann (2011); SIMBA# biogas 4.2 Tutorial.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np
import pandas as pd

from pyadm1.core.adm1 import INFLUENT_COLUMNS

# Default location of substrate definition files.
_DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "substrates"
_DEFAULT_XML_DIR = _DEFAULT_DATA_DIR  # backwards-compatible alias

# Supported substrate file extensions, in registry-lookup priority order.
_SUBSTRATE_EXTENSIONS = (".yaml", ".yml", ".xml", ".toml")

# Canonical default ordering for the bundled substrate library. Used by
# ``Feedstock(substrates=None)`` so the Q array indices are stable and
# meaningful (most-frequently-used substrates first; legacy/variants at
# the end). Substrate IDs not listed here are appended afterwards in
# alphabetical order, so adding new files never silently drops them.
_DEFAULT_SUBSTRATE_ORDER: tuple = (
    "maize_silage_milk_ripeness",
    "cattle_manure",
    "swine_manure",
    "corn_cob_mix",
    "grass_silage",
    "green_rye_silage",
    "cereal_gps_silage",
    "onion_waste",
    "maize_silage_gummersbach",
    "cattle_manure_solid",
    "chicken_manure_dry",
    "swine_manure_gummersbach",
    "wheat_whole_plant_silage",
)


def _order_substrates(available: Sequence[str]) -> list[str]:
    """Reorder substrate IDs by :data:`_DEFAULT_SUBSTRATE_ORDER`."""
    available_set = set(available)
    ordered = [sid for sid in _DEFAULT_SUBSTRATE_ORDER if sid in available_set]
    extras = sorted(available_set - set(ordered))
    return ordered + extras


# ---------------------------------------------------------------------------
# Substrate parameter dataclass
# ---------------------------------------------------------------------------


@dataclass
class SubstrateParams:
    """
    Complete characterization of one substrate for the ADM1 model.

    Measured substrate properties (per ton or m³ of fresh matter)
    -------------------------------------------------------------
    name  : str    Human-readable label.
    TS    : float  Total solids [kg/t FM]  (liquid substrates: kg/m³ FM)
    NH4   : float  Ammonia nitrogen [kg N/m³ FM]
    BGP   : float  Biogas potential [Nm³/t VS]  (reference)
    BMP   : float  Biomethane potential [Nm³ CH₄/t VS]  (reference)

    COD fractionation
    -----------------
    aXI    : Particulate inert fraction of degradable organic COD [-]
    fOTSrf : Biodegradable fraction of crude fibre [-]
    fsOTS  : VS fraction entering slow disintegration pool (XPS) [-]
    ffOTS  : VS fraction entering fast disintegration pool (XPF) [-]
    aSi    : Dissolved inert fraction of degradable organic COD [-]

    Weender analysis (fractions of TS)
    ----------------------------------
    fRF, fRP, fRFe, fRA : Crude fibre, protein, lipid, ash fractions [-]

    Physical / chemical state
    -------------------------
    Temp : Temperature [°C]
    pH   : pH [-]
    KS43 : Acid capacity to pH 4.3 [mol/m³]
    FFS  : Volatile fatty acids as acetic-acid equivalent [kg HAc/m³]

    Economics
    ---------
    cost : Substrate cost free plant [EUR/t FM]. Optional, default 0.

    Optional measured parameters
    ----------------------------
    The COD fractionation above (``aXI``, ``fOTSrf``, ``fsOTS``, ``ffOTS``,
    ``aSi``) is not measured directly. The parameters below are, and each one
    replaces part of that fractionation when given. They are the parameters of
    the legacy Gummersbach substrate library; see
    :mod:`pyadm1.substrates.gummersbach` for its units and file schema.

    NDF, ADL : Cell wall and lignin after van Soest [fraction of TS]. When
        both are given, the carbohydrates are split by the cell wall instead
        of the crude fibre: structural carbohydrates (``NDF - ADL``) enter the
        slow pool, non-fibre carbohydrates the fast pool, lignin is inert.
        ``fOTSrf`` is then not used and ``fsOTS`` / ``ffOTS`` only split
        protein and lipids.
    ADF  : Acid detergent fibre [fraction of TS]. Stored, not used.
    D_VS : Degradation level of the volatile solids [-]. With ``NDF`` and
        ``ADL`` it sets the degradable share of the structural carbohydrates
        (Koch et al. 2010); ``aXI`` and ``aSi`` are then not used.
    Sac, Spro, Sbu, Sva : Acetic, propionic, butyric and valeric acid
        [kg/m³]. When at least one is given they replace ``FFS``.
    TAC  : Titrated acid capacity to pH 4.3 [mol/m³]. Replaces ``KS43``; the
        acid anions titrated along with the bicarbonate are subtracted.
    COD_S : COD of the filtrate [kgCOD/m³]. What exceeds the acids and the
        soluble inerts is already disintegrated and enters the hydrolysable
        pools ``X_S_*``.
    SIin : Soluble inert COD [kgCOD/m³]. Replaces the ``aSi`` share and is
        taken out of the inert COD.
    X_bac, X_meth : Living bacteria and methanogens [kgCOD/m³]. Taken out of
        the disintegration pools, so the total COD is unchanged.
    NDICP : Neutral detergent insoluble crude protein [fraction of TS]: the
        protein bound in the cell wall (Licitra et al. 1996). Needs ``NDF``
        and ``ADL``. It enters the slow pool, the remaining protein the fast
        pool, and it is taken out of the structural carbohydrates, as in the
        non-fibre carbohydrates of NRC (2001). ``fsOTS`` / ``ffOTS`` then only
        split the lipids.

    Inert share from a measured methane potential
    ---------------------------------------------
    ``aXI`` (or ``D_VS``) can be fitted to a methane potential measured in a
    batch test with :func:`fit_degradability_to_bmp`; the substrate loaders do
    so when a file gives ``BMP`` but neither ``aXI`` nor ``D_VS``.

    Classification
    --------------
    substrate_class : Remuneration class of the substrate as free text, e.g.
        ``"Mais (GPS) (EK I)"`` after the German EEG 2012 / BiomasseV. Not used
        by the model; a manure class selects the manure defaults of the
        measured schema. Optional.
    """

    name: str

    # Proximate analysis
    TS: float
    NH4: float
    BGP: float
    BMP: float

    # COD fractionation
    aXI: float
    fOTSrf: float
    fsOTS: float
    ffOTS: float
    aSi: float

    # Weender analysis
    fRF: float
    fRP: float
    fRFe: float
    fRA: float

    # Physical / chemical state
    Temp: float
    pH: float
    KS43: float
    FFS: float

    # Component densities [kg/m³]
    roh_CH: float = 1550.0
    roh_PR: float = 1370.0
    roh_LI: float = 920.0
    roh_MI: float = 2420.0
    roh_AC: float = 1050.0
    roh_H2O: float = 1000.0

    # COD conversion factors [kg mass / kg COD]
    M_Xch: float = 0.9375
    M_Xpr: float = 0.7736
    M_Xli: float = 0.34741379310344828
    M_Sac: float = 0.9375
    M_Spro: float = 0.6607142857142857
    M_Sbu: float = 0.55
    M_Sva: float = 0.49038461538461536
    M_Sh2: float = 0.125
    M_XB: float = 0.76376137931034471

    # Methane potentials
    MP_CH: float = 0.5
    MP_PR: float = 0.71
    MP_LI: float = 0.68
    MP_AC: float = 0.5

    # Physical constants
    V_m: float = 0.022413
    CH4_cod_2_mol: float = 64.0

    # Acid-base equilibrium constants at 35 °C [kmol/m³]
    Kw_35: float = 1.0e-14
    N_aa: float = 0.0076475885714285706
    Kava: float = 1.3803842646028839e-05
    Kabu: float = 1.5135612484362071e-05
    Kapro: float = 1.3182567385564074e-05
    Kaac: float = 1.7378008287493764e-05
    Kaco2_35: float = 4.4668359215096349e-07
    Kain_35: float = 5.623413251903491e-10

    # Economics
    cost: float = 0.0

    # Optional measured parameters; ``None`` means "not measured".
    NDF: float | None = None
    ADF: float | None = None
    ADL: float | None = None
    D_VS: float | None = None
    Sac: float | None = None
    Spro: float | None = None
    Sbu: float | None = None
    Sva: float | None = None
    TAC: float | None = None
    COD_S: float | None = None
    SIin: float | None = None
    X_bac: float = 0.0
    X_meth: float = 0.0

    # Lignin C10.92 H14.24 O5.76 [kg mass / kg COD], as in the legacy model.
    M_Xlig: float = 0.63965517241379311

    # Protein bound in the cell wall [fraction of TS]; ``None`` means "not measured".
    NDICP: float | None = None

    # Remuneration class (free text); not used by the model.
    substrate_class: str | None = None

    def __post_init__(self) -> None:
        """Reject parameters that are out of range or given in the wrong unit."""
        if not 0.0 <= self.TS <= 1000.0:
            raise ValueError(f"Substrate '{self.name}': TS={self.TS} must lie between 0 and 1000 g/kg fresh matter.")
        if not 0.0 <= self.pH <= 14.0:
            raise ValueError(f"Substrate '{self.name}': pH={self.pH} must lie between 0 and 14.")
        weender = {"fRF": self.fRF, "fRP": self.fRP, "fRFe": self.fRFe, "fRA": self.fRA}
        for key, value in weender.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Substrate '{self.name}': {key}={value} must be a fraction of TS between 0 and 1.")
        if sum(weender.values()) > 1.0 + 1.0e-6:
            raise ValueError(
                f"Substrate '{self.name}': crude fibre, protein, lipids and ash add up to "
                f"{sum(weender.values()):.3f} of TS, more than the dry matter holds."
            )
        for key in ("aXI", "aSi", "fOTSrf", "fsOTS", "ffOTS"):
            value = getattr(self, key)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Substrate '{self.name}': {key}={value} must lie between 0 and 1.")
        for key in ("NH4", "KS43", "FFS", "cost"):
            value = getattr(self, key)
            if value < 0.0:
                raise ValueError(f"Substrate '{self.name}': {key}={value} must not be negative.")
        for key in ("NDF", "ADF", "ADL", "NDICP"):
            value = getattr(self, key)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Substrate '{self.name}': {key}={value} must be a fraction of TS between 0 and 1 "
                    "(the Gummersbach schema takes % TS instead)."
                )
        if self.D_VS is not None and not 0.0 <= self.D_VS <= 1.0:
            raise ValueError(f"Substrate '{self.name}': D_VS={self.D_VS} must lie between 0 and 1.")
        if (self.NDF is None) != (self.ADL is None):
            raise ValueError(f"Substrate '{self.name}': NDF and ADL must be given together.")
        if self.NDICP is not None and self.NDF is None:
            raise ValueError(f"Substrate '{self.name}': NDICP is part of the cell wall and needs NDF and ADL.")
        for key in ("Sac", "Spro", "Sbu", "Sva", "TAC", "COD_S", "SIin", "X_bac", "X_meth"):
            value = getattr(self, key)
            if value is not None and value < 0.0:
                raise ValueError(f"Substrate '{self.name}': {key}={value} must not be negative.")

    @property
    def uses_cell_wall_analysis(self) -> bool:
        """True when the carbohydrates are split by ``NDF`` / ``ADL`` instead of the crude fibre."""
        return self.NDF is not None and self.ADL is not None

    @property
    def vfa_masses(self) -> tuple[float, float, float, float]:
        """Acetic, propionic, butyric and valeric acid [kg/m³]; ``FFS`` counts as acetic acid."""
        singles = (self.Sac, self.Spro, self.Sbu, self.Sva)
        if all(value is None for value in singles):
            return (self.FFS, 0.0, 0.0, 0.0)
        return tuple(0.0 if value is None else float(value) for value in singles)  # type: ignore[return-value]

    @property
    def vfa_mass(self) -> float:
        """Total volatile fatty acids [kg/m³]."""
        return float(sum(self.vfa_masses))


# ---------------------------------------------------------------------------
# Substrate loaders (XML, YAML, TOML)
# ---------------------------------------------------------------------------

#: Key that selects the parameter set of a substrate file. Without it the file
#: uses the :class:`SubstrateParams` names and units.
_SCHEMA_KEY = "schema"

#: Fields of :class:`SubstrateParams` that hold text instead of a number.
_TEXT_FIELDS = ("substrate_class",)

#: Key of a substrate file: share of the degradable COD that a batch test
#: recovers as methane. See :func:`fit_degradability_to_bmp`.
_BMP_RECOVERY_KEY = "BMP_recovery"

#: Default of that share: the model potential is matched to the measured one.
#: Two effects of opposite sign are left out by this. A batch test recovers
#: about 5 % less than the degradable COD holds, because biomass grows
#: (Weißbach 2009). The model counts carbohydrates as glucose, which is about
#: 10 % below the COD of starch and cellulose. A value below 1 accounts for the
#: first effect alone.
DEFAULT_BMP_RECOVERY = 1.0


def fit_degradability_to_bmp(
    params: SubstrateParams, bmp: float | None = None, recovery: float = DEFAULT_BMP_RECOVERY
) -> SubstrateParams:
    """
    Fit the inert share of a substrate to a measured methane potential.

    The model potential :meth:`Feedstock.bmp_theoretical` is matched to
    ``bmp / recovery``. A batch test (VDI 4630) yields less methane than the
    degradable COD holds, because part of it becomes biomass; *recovery* is
    that share. With a cell-wall analysis (``NDF``, ``ADL``) the
    degradation level ``D_VS`` is fitted, otherwise the inert share ``aXI``;
    both change the potential monotonically, so bisection finds the value.

    Args:
        params: Substrate whose other parameters are final.
        bmp: Measured potential [Nm³ CH4/t VS]; defaults to ``params.BMP``.
        recovery: Share of the degradable COD recovered as methane in the
            batch test, 0 < recovery <= 1.

    Returns:
        A copy of *params* with the fitted ``D_VS`` or ``aXI``.

    Raises:
        ValueError: *bmp* or *recovery* is out of range, or *bmp* exceeds what
            the composition of the substrate can yield.
    """
    target_bmp = float(params.BMP if bmp is None else bmp)
    if not 0.0 < recovery <= 1.0:
        raise ValueError(f"Substrate '{params.name}': BMP recovery must lie in (0, 1], got {recovery}.")
    if target_bmp <= 0.0:
        raise ValueError(f"Substrate '{params.name}': a measured BMP must be positive, got {target_bmp}.")
    target = target_bmp / recovery

    by_cell_wall = params.uses_cell_wall_analysis

    def with_degradable_share(share: float) -> SubstrateParams:
        if by_cell_wall:
            return replace(params, D_VS=share)
        return replace(params, aXI=max(1.0 - params.aSi - share, 0.0))

    def potential(share: float) -> float:
        return Feedstock(with_degradable_share(share)).bmp_theoretical()

    upper = 1.0 if by_cell_wall else max(1.0 - params.aSi, 0.0)
    lowest, highest = potential(0.0), potential(upper)
    if not lowest <= target <= highest:
        raise ValueError(
            f"Substrate '{params.name}': a BMP of {target_bmp:.0f} Nm³ CH4/t VS (recovery {recovery:.2f}) needs a "
            f"model potential of {target:.0f}, but the composition allows {lowest:.0f} to {highest:.0f} only."
        )

    low, high = 0.0, upper
    for _ in range(60):
        middle = 0.5 * (low + high)
        if potential(middle) < target:
            low = middle
        else:
            high = middle
    return with_degradable_share(0.5 * (low + high))


def _build_substrate_params(substrate_name: str, raw: dict[str, object], source: Path) -> SubstrateParams:
    """Common dict -> SubstrateParams construction used by all loaders."""
    schema = raw.get(_SCHEMA_KEY)
    if schema is not None:
        from .gummersbach import SCHEMA_NAMES, params_from_measured

        if str(schema).strip().lower() not in SCHEMA_NAMES:
            raise ValueError(f"Unknown substrate schema '{schema}' in {source.name}; supported: {list(SCHEMA_NAMES)}.")
        measured = {key: value for key, value in raw.items() if key != _SCHEMA_KEY}
        return params_from_measured(substrate_name, measured, source=source.name)

    # A misspelt key (``aXi`` for ``aXI``) would otherwise be dropped silently
    # and the default or the BMP fit would take its place.
    known = {f.name for f in fields(SubstrateParams)} | {_SCHEMA_KEY, _BMP_RECOVERY_KEY}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"Unknown parameter(s) {unknown} in substrate {source.name}.")

    # Without aXI the inert share is fitted to the methane potential of the file.
    fit_to_bmp = raw.get("aXI") is None and raw.get("D_VS") is None and raw.get("BMP") is not None

    kwargs: dict = {"name": substrate_name}
    for f in fields(SubstrateParams):
        if f.name == "name":
            continue
        if raw.get(f.name) is not None:
            kwargs[f.name] = str(raw[f.name]) if f.name in _TEXT_FIELDS else float(raw[f.name])
        elif f.default.__class__.__name__ != "_MISSING_TYPE":
            kwargs[f.name] = f.default
        elif f.name == "aXI" and fit_to_bmp:
            kwargs[f.name] = 0.0
        else:
            raise ValueError(f"Required parameter '{f.name}' not found in {source.name}")
    params = SubstrateParams(**kwargs)
    if fit_to_bmp:
        recovery = raw.get(_BMP_RECOVERY_KEY)
        params = fit_degradability_to_bmp(params, recovery=DEFAULT_BMP_RECOVERY if recovery is None else float(recovery))
    return params


def load_substrate_xml(path: str | Path) -> SubstrateParams:
    """
    Load a substrate definition from an XML file.

    Schema: ``<substrate name="...">`` root with ``<param name="..."
    value="..."/>`` children. ``<substrate schema="gummersbach" ...>``
    selects the measured parameter set of
    :mod:`pyadm1.substrates.gummersbach`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Substrate XML not found: {path}")

    root = ET.parse(path).getroot()  # noqa: S314 - trusted local substrate definition files, not untrusted input
    substrate_name = root.get("name", path.stem)

    raw: dict[str, str] = {}
    if root.get(_SCHEMA_KEY) is not None:
        raw[_SCHEMA_KEY] = root.get(_SCHEMA_KEY)
    for elem in root.findall("param"):
        pname = elem.get("name")
        pvalue = elem.get("value")
        if pname is not None and pvalue is not None:
            raw[pname] = pvalue

    return _build_substrate_params(substrate_name, raw, path)


def load_substrate_yaml(path: str | Path) -> SubstrateParams:
    """
    Load a substrate definition from a YAML file.

    Schema: a flat top-level mapping of parameter name -> value, with one
    optional ``name`` key for the human-readable substrate label. A
    ``schema: gummersbach`` key selects the measured parameter set of
    :mod:`pyadm1.substrates.gummersbach`.
    """
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Loading YAML substrate files requires PyYAML. Install it with `pip install PyYAML`.") from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Substrate YAML not found: {path}")

    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Substrate YAML must be a mapping at the top level: {path}")

    substrate_name = str(data.pop("name", path.stem))
    return _build_substrate_params(substrate_name, data, path)


def load_substrate_toml(path: str | Path) -> SubstrateParams:
    """
    Load a substrate definition from a TOML file.

    Schema: top-level table with ``name = "..."`` plus one ``key = value``
    per substrate parameter. ``schema = "gummersbach"`` selects the measured
    parameter set of :mod:`pyadm1.substrates.gummersbach`.
    """
    try:
        import tomllib  # Python 3.11+
    except ImportError:  # pragma: no cover
        try:
            import tomli as tomllib  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Loading TOML substrate files on Python < 3.11 requires `tomli`. Install it with `pip install tomli`."
            ) from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Substrate TOML not found: {path}")

    with path.open("rb") as fp:
        data = tomllib.load(fp)
    if not isinstance(data, dict):  # pragma: no cover - TOML's grammar guarantees a top-level table
        raise ValueError(f"Substrate TOML must be a table at the top level: {path}")

    substrate_name = str(data.pop("name", path.stem))
    return _build_substrate_params(substrate_name, data, path)


def load_substrate(path: str | Path) -> SubstrateParams:
    """
    Load a substrate definition from any supported file format.

    The format is selected by the file extension:

    * ``.yaml`` / ``.yml`` -> :func:`load_substrate_yaml` (canonical)
    * ``.xml``             -> :func:`load_substrate_xml`
    * ``.toml``            -> :func:`load_substrate_toml`
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        return load_substrate_yaml(path)
    if suffix == ".xml":
        return load_substrate_xml(path)
    if suffix == ".toml":
        return load_substrate_toml(path)
    raise ValueError(f"Unsupported substrate file extension '{suffix}' for {path}. Supported: {_SUBSTRATE_EXTENSIONS}")


# ---------------------------------------------------------------------------
# Substrate registry
# ---------------------------------------------------------------------------


class SubstrateRegistry:
    """
    Discovers and lazy-loads substrate files from a directory.

    Supported file formats: YAML (canonical), XML, TOML. When the same
    substrate ID is present in multiple formats, lookup priority is
    ``.yaml > .yml > .xml > .toml``.

    Usage
    -----
    >>> registry = SubstrateRegistry()
    >>> print(registry.available())
    ['cattle_manure', 'maize_silage_milk_ripeness', 'swine_manure']
    >>> sub = registry.get("swine_manure")
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        xml_dir: str | Path | None = None,
    ) -> None:
        # ``xml_dir`` is the legacy keyword from when substrates were
        # XML-only; kept as a back-compat alias so existing callers don't
        # break. New code should use ``data_dir``.
        if xml_dir is not None:
            if data_dir is not None:
                raise TypeError("SubstrateRegistry() accepts 'data_dir' or 'xml_dir', not both.")
            data_dir = xml_dir
        self._dir = Path(data_dir) if data_dir is not None else _DEFAULT_DATA_DIR
        self._cache: dict[str, SubstrateParams] = {}

    def available(self) -> list[str]:
        """Return substrate IDs (file stems) found in the directory."""
        if not self._dir.exists():
            return []
        seen: set = set()
        for ext in _SUBSTRATE_EXTENSIONS:
            for p in self._dir.glob(f"*{ext}"):
                seen.add(p.stem)
        return sorted(seen)

    def _find_path(self, substrate_id: str) -> Path | None:
        for ext in _SUBSTRATE_EXTENSIONS:
            candidate = self._dir / f"{substrate_id}{ext}"
            if candidate.exists():
                return candidate
        return None

    def get(self, substrate_id: str) -> SubstrateParams:
        """Return the substrate with the given ID, loading it on first access."""
        if substrate_id not in self._cache:
            path = self._find_path(substrate_id)
            if path is None:
                raise KeyError(f"Substrate '{substrate_id}' not found in {self._dir}. Available: {self.available()}")
            self._cache[substrate_id] = load_substrate(path)
        return self._cache[substrate_id]

    def load_all(self) -> dict[str, SubstrateParams]:
        """Load every substrate file in the directory."""
        for sid in self.available():
            self.get(sid)
        return dict(self._cache)


# ---------------------------------------------------------------------------
# Feedstock class
# ---------------------------------------------------------------------------


_SubstrateInput = SubstrateParams | str | Path

#: Prefix of substrate IDs that refer to the bundled legacy Gummersbach library.
_LEGACY_LIBRARY_PREFIX = "gummersbach:"

#: Prefix of substrate IDs that refer to the superseded substrate files in
#: ``data/substrates/legacy/``, e.g. ``"legacy:cattle_manure"``.
_LEGACY_FILE_PREFIX = "legacy:"
_LEGACY_FILE_DIR = _DEFAULT_DATA_DIR / "legacy"

#: Disintegration pools, by substrate class (carbohydrates, proteins, lipids).
_DISINTEGRATION_POOLS = {
    "ch": ("X_PS_ch", "X_PF_ch"),
    "pr": ("X_PS_pr", "X_PF_pr"),
    "li": ("X_PS_li", "X_PF_li"),
}

#: Split of substrate-borne biomass over the ADM1 groups. Same ratios as the
#: pre-inoculated initial state of the digester.
_BACTERIA_SPLIT = {"X_su": 0.25, "X_aa": 0.25, "X_fa": 0.15, "X_c4": 0.20, "X_pro": 0.15}
_METHANOGEN_SPLIT = {"X_ac": 0.80, "X_h2": 0.20}

#: COD per kmol of the volatile fatty acids [kgCOD/kmol].
_VFA_COD_PER_KMOL = {"S_ac": 64.0, "S_pro": 112.0, "S_bu": 160.0, "S_va": 208.0}

#: End point of the acid-capacity titration.
_PH_TITRATION_END = 4.3


class Feedstock:
    """
    Computes ADM1 influent concentrations from substrate characterization.

    Accepts either a single substrate (``SubstrateParams``, XML path, or bare
    XML stem ID) or a list of substrates for co-digestion.  In multi-substrate
    mode the influent DataFrame is generated from a volumetric-flow-weighted
    blend of per-substrate concentrations.

    Usage (single substrate)
    ------------------------
    >>> from pyadm1 import Feedstock
    >>> fs = Feedstock("maize_silage_milk_ripeness", feeding_freq=48, total_simtime=60)
    >>> df = fs.get_influent_dataframe(Q=15.0)

    Usage (co-digestion, up to 10 substrates)
    -----------------------------------------
    >>> fs = Feedstock(
    ...     ["maize_silage_milk_ripeness", "swine_manure"],
    ...     feeding_freq=24,
    ...     total_simtime=160,
    ... )
    >>> Q = [11.4, 6.1, 0, 0, 0, 0, 0, 0, 0, 0]  # m³/d, up to 10 slots
    >>> df = fs.get_influent_dataframe(Q=Q)
    """

    def __init__(
        self,
        substrates: _SubstrateInput | Sequence[_SubstrateInput] | None = None,
        feeding_freq: int = 48,
        total_simtime: int = 60,
        simba_q_convention: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        substrates : SubstrateParams | str | Path | list of those, optional
            A single substrate object/file path/ID, or a list of substrates
            to co-digest.  When ``None`` (the default), every substrate file
            under ``data/substrates/`` is loaded, ordered by the canonical
            default in :data:`_DEFAULT_SUBSTRATE_ORDER` (frequently-used
            substrates first, then variants; unknown IDs are appended in
            alphabetical order). Handy for demos and tests where the exact
            mix doesn't matter; pass an explicit list whenever the index
            of ``Q`` must be stable across releases.
        feeding_freq : int
            Time between feeding events [hours].
        total_simtime : int
            Total simulation duration [days].
        simba_q_convention : bool, default True
            How to interpret ``Q`` in ``get_influent_dataframe(Q=...)``.

            * ``True`` (default, ADM1da convention): each ``Q_i``
              [m³/d] is interpreted as a mass-equivalent flow.  Internally
              ``Q_actual_i = Q_input_i · 1000 / ρ_FM_i``.  For liquid
              substrates (TS < 200) this is a no-op (ρ_FM = 1000 by
              convention).  For solid substrates (e.g. maize silage) this
              produces a slightly smaller actual liquid volume.
            * ``False``: ``Q`` is taken literally as the actual liquid
              volume added to the reactor [m³/d].
        """
        if substrates is None:
            available = SubstrateRegistry().available()
            if not available:
                raise ValueError(
                    f"No substrate files found in {_DEFAULT_DATA_DIR}; pass an explicit substrate list or add files."
                )
            self._multi = True
            raw_subs: list[_SubstrateInput] = list(_order_substrates(available))
        elif isinstance(substrates, (list, tuple)):
            if len(substrates) == 0:
                raise ValueError("At least one substrate must be provided.")
            self._multi = True
            raw_subs = list(substrates)
        else:
            self._multi = False
            raw_subs = [substrates]

        # Remember the original input identifiers (XML stems / paths / params).
        # ``substrate_ids`` is used for serialisation (e.g. parallel workers
        # rebuilding the feedstock in a fresh process) where the human-readable
        # ``SubstrateParams.name`` may not match the XML file stem.
        self._raw_inputs = list(raw_subs)
        self._substrate_ids: list[str] = self._unique_ids([self._raw_input_id(item) for item in raw_subs])

        self._subs: list[SubstrateParams] = [self._resolve_substrate(item) for item in raw_subs]
        self._simtime = np.arange(0, total_simtime, float(feeding_freq) / 24.0)
        self._feeding_freq = int(feeding_freq)

        self._densities: list[float] = [self._calc_density(s) for s in self._subs]
        self._conc_list: list[dict] = [self._calc_concentrations(s, rho) for s, rho in zip(self._subs, self._densities)]

        self._simba_q_convention = bool(simba_q_convention)
        if self._simba_q_convention:
            self._q_factors: list[float] = [1000.0 / rho for rho in self._densities]
        else:
            self._q_factors = [1.0] * len(self._subs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_influent_dataframe(self, Q: float | Sequence[float]) -> pd.DataFrame:
        """
        Generate an ADM1 influent DataFrame for the full simulation period.

        Substrate concentrations are constant in time (steady-state feed
        composition assumption).  Pass the result to
        ``ADM1.set_influent_dataframe()``.
        """
        Q_arr = self._validate_Q(Q)
        q_total = float(np.sum(Q_arr))
        row = self._blended_concentrations(Q_arr)
        row["Q"] = q_total

        df = pd.DataFrame([row] * len(self._simtime))
        return df[INFLUENT_COLUMNS]

    def simtime(self) -> np.ndarray:
        """Simulation time array [days]."""
        return self._simtime

    def header(self) -> list[str]:
        """Names of ADM1 input stream columns."""
        return list(INFLUENT_COLUMNS)

    # ---- Single-substrate convenience accessors -----------------------

    @property
    def substrate(self) -> SubstrateParams:
        """Single substrate (raises if multiple substrates are configured)."""
        self._require_single("substrate", hint="use .substrates[i]")
        return self._subs[0]

    @property
    def density(self) -> float:
        """Fresh-matter density [kg/m³] (single-substrate mode)."""
        self._require_single("density", hint="use .densities[i] or .blended_density(Q)")
        return self._densities[0]

    @property
    def concentrations(self) -> dict:
        """Influent concentrations (single-substrate mode)."""
        self._require_single(
            "concentrations",
            hint="use .concentrations_list[i] or .blended_concentrations(Q)",
        )
        return dict(self._conc_list[0])

    # ---- Multi-substrate accessors ------------------------------------

    @property
    def substrates(self) -> list[SubstrateParams]:
        """All configured substrates, in feed-index order."""
        return list(self._subs)

    @property
    def densities(self) -> list[float]:
        """Per-substrate fresh-matter densities [kg/m³]."""
        return list(self._densities)

    def actual_Q(self, Q: float | Sequence[float]) -> list[float]:
        """
        Return per-substrate actual liquid volume flows [m³/d].

        Applies the ADM1da mass-to-volume conversion when
        ``simba_q_convention=True``; otherwise returns *Q* unchanged.
        """
        return self._validate_Q(Q).tolist()

    @property
    def substrate_costs(self) -> dict[str, float]:
        """Cost of each configured substrate [EUR/t FM], keyed by substrate ID."""
        return {sid: float(s.cost) for sid, s in zip(self._substrate_ids, self._subs)}

    def set_substrate_costs(self, costs: dict[str, float]) -> None:
        """
        Set or override substrate costs [EUR/t FM].

        Costs may also be given as ``cost:`` in the substrate definition file;
        this overrides them for this feedstock only.

        Parameters
        ----------
        costs : dict
            ``{substrate_id: cost}``. Substrates not listed keep their cost.
        """
        unknown = set(costs) - set(self._substrate_ids)
        if unknown:
            raise ValueError(f"Unknown substrate(s) {sorted(unknown)}; configured are {self._substrate_ids}.")
        for index, sid in enumerate(self._substrate_ids):
            if sid in costs:
                if float(costs[sid]) < 0.0:
                    raise ValueError(f"Cost of '{sid}' must not be negative, got {costs[sid]}")
                # Replace rather than mutate: the parameter object may be shared.
                self._subs[index] = replace(self._subs[index], cost=float(costs[sid]))

    def mass_flows(self, Q: float | Sequence[float]) -> list[float]:
        """Per-substrate fresh-matter mass flows [t/d] for feed rates *Q*."""
        return [q * rho / 1000.0 for q, rho in zip(self._validate_Q(Q), self._densities)]

    def calc_substrate_costs(self, Q: float | Sequence[float] | Sequence[Sequence[float]], dt: float = 1.0) -> dict:
        """
        Substrate costs for a given feed.

        Works on any feed record: the configured rates, a feed log, a measured
        flow from a sensor or an estimate. For the costs of a simulated plant
        see :meth:`BiogasPlant.get_substrate_costs`, which applies this to the
        feed the digesters actually received.

        Parameters
        ----------
        Q : sequence of float, or sequence of such sequences
            Feed rates per substrate, in the unit of ``Q_substrates``. Either
            one set of rates held for ``dt`` days, or a time series with one
            row per interval of ``dt`` days.
        dt : float, default 1.0
            Length of the interval each set of rates applies to [d].

        Returns
        -------
        dict
            ``total`` [EUR], ``by_substrate`` ({id: EUR}) and ``mass_t``
            ({id: t FM fed}).
        """
        rows = np.atleast_2d(np.asarray(Q, dtype=float))
        mass = np.zeros(len(self._subs))
        for row in rows:
            mass += np.asarray(self.mass_flows(row)) * float(dt)

        mass_t = {sid: float(m) for sid, m in zip(self._substrate_ids, mass)}
        by_substrate = {sid: mass_t[sid] * float(s.cost) for sid, s in zip(self._substrate_ids, self._subs)}
        return {"total": float(sum(by_substrate.values())), "by_substrate": by_substrate, "mass_t": mass_t}

    @property
    def q_conversion_factors(self) -> list[float]:
        """Per-substrate ADM1da Q-conversion factors [-]."""
        return list(self._q_factors)

    @property
    def concentrations_list(self) -> list[dict]:
        """Per-substrate influent concentrations."""
        return [dict(c) for c in self._conc_list]

    def vs_content(self, index: int = 0) -> float:
        """Volatile-solids content of the i-th substrate [kg VS/m³]."""
        s = self._subs[index]
        fTS = s.TS / 1000.0
        return fTS * (1.0 - s.fRA) * self._densities[index]

    def total_cod(self, index: int = 0) -> float:
        """Total COD concentration of the i-th substrate [kg COD/m³]."""
        c = self._conc_list[index]
        return self._degradable_cod(c) + c["X_I"] + c["S_I"]

    @staticmethod
    def _degradable_cod(c: dict) -> float:
        """Degradable COD of one substrate [kg COD/m³]: particulate pools, acids and biomass."""
        keys = [pool for pools in _DISINTEGRATION_POOLS.values() for pool in pools]
        keys += ["X_S_ch", "X_S_pr", "X_S_li"]
        keys += list(_VFA_COD_PER_KMOL)
        keys += list(_BACTERIA_SPLIT) + list(_METHANOGEN_SPLIT)
        return float(sum(c[key] for key in keys))

    def bmp_theoretical(self, index: int = 0) -> float:
        """Theoretical biomethane potential of the i-th substrate [Nm³ CH₄/t VS]."""
        c = self._conc_list[index]
        s = self._subs[index]
        th_yield = s.V_m / (s.CH4_cod_2_mol / 1000.0)
        degradable_cod = self._degradable_cod(c)
        vs = self.vs_content(index)
        if vs <= 0.0:
            return 0.0
        return degradable_cod * th_yield / vs * 1000.0

    def blended_density(self, Q: float | Sequence[float]) -> float:
        """Volumetric-flow-weighted fresh-matter density [kg/m³]."""
        Q_arr = self._validate_Q(Q)
        q_tot = float(np.sum(Q_arr))
        if q_tot <= 0.0:
            return 1000.0
        return float(np.dot(Q_arr, self._densities) / q_tot)

    def blended_vs_content(self, Q: float | Sequence[float]) -> float:
        """Volumetric-flow-weighted VS content [kg VS/m³]."""
        Q_arr = self._validate_Q(Q)
        q_tot = float(np.sum(Q_arr))
        if q_tot <= 0.0:
            return 0.0
        vs = np.array([self.vs_content(i) for i in range(len(self._subs))])
        return float(np.dot(Q_arr, vs) / q_tot)

    def blended_concentrations(self, Q: float | Sequence[float]) -> dict:
        """Volumetric-flow-weighted influent concentrations (no Q field)."""
        Q_arr = self._validate_Q(Q)
        return self._blended_concentrations(Q_arr)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @property
    def substrate_ids(self) -> list[str]:
        """
        Stable identifiers for the configured substrates.

        For inputs given as substrate IDs / XML paths, this returns the file
        stem; for raw :class:`SubstrateParams` instances, it returns
        ``substrate.name``.  Useful for serialisation and worker rebuild.
        """
        return list(self._substrate_ids)

    @property
    def feeding_freq(self) -> int:
        """Feeding frequency [hours]."""
        return self._feeding_freq

    @staticmethod
    def _unique_ids(ids: list[str]) -> list[str]:
        """
        Make substrate IDs unique by numbering repeats (``maize``, ``maize_2``).

        Costs and masses are reported per ID, so two substrates sharing one
        (the same file twice, or two paths with the same stem) would
        otherwise be merged into a single entry.

        Args:
            ids (List[str]): IDs in feedstock order.

        Returns:
            List[str]: The IDs with a ``_<n>`` suffix on every repeat.
        """
        seen: dict[str, int] = {}
        unique: list[str] = []
        for sid in ids:
            seen[sid] = seen.get(sid, 0) + 1
            unique.append(sid if seen[sid] == 1 else f"{sid}_{seen[sid]}")
        return unique

    @staticmethod
    def _raw_input_id(item: _SubstrateInput) -> str:
        """Return a stable identifier for a substrate input item."""
        if isinstance(item, SubstrateParams):
            return item.name
        if isinstance(item, str) and item.startswith((_LEGACY_LIBRARY_PREFIX, _LEGACY_FILE_PREFIX)):
            return item
        return Path(item).stem

    @staticmethod
    def _resolve_substrate(item: _SubstrateInput) -> SubstrateParams:
        """
        Accept a params object, a filesystem path, or a bare substrate ID
        (file stem in the default ``data/substrates/`` directory).
        Any supported format (.yaml/.yml/.xml/.toml) is recognised. An ID of
        the form ``gummersbach:<id>`` selects a substrate of the bundled
        legacy Gummersbach library, ``legacy:<id>`` a superseded substrate
        file in ``data/substrates/legacy/``.
        """
        if isinstance(item, SubstrateParams):
            return item
        if isinstance(item, str) and item.startswith(_LEGACY_LIBRARY_PREFIX):
            from .gummersbach import gummersbach_substrate

            return gummersbach_substrate(item[len(_LEGACY_LIBRARY_PREFIX) :])
        if isinstance(item, str) and item.startswith(_LEGACY_FILE_PREFIX):
            stem = item[len(_LEGACY_FILE_PREFIX) :]
            for ext in _SUBSTRATE_EXTENSIONS:
                legacy_path = _LEGACY_FILE_DIR / f"{stem}{ext}"
                if legacy_path.exists():
                    return load_substrate(legacy_path)
            raise FileNotFoundError(f"Legacy substrate '{stem}' not found in {_LEGACY_FILE_DIR}")
        p = Path(item)
        if p.exists():
            return load_substrate(p)
        for ext in _SUBSTRATE_EXTENSIONS:
            reg_path = _DEFAULT_DATA_DIR / f"{p.stem}{ext}"
            if reg_path.exists():
                return load_substrate(reg_path)
        raise FileNotFoundError(f"Substrate '{item}' not found as a path or as an ID in {_DEFAULT_DATA_DIR}")

    def _require_single(self, prop: str, hint: str) -> None:
        """Raise ValueError if the single-substrate accessor *prop* is called on a multi-substrate feedstock."""
        if len(self._subs) != 1:
            raise ValueError(
                f"'{prop}' is a single-substrate accessor; this feedstock has {len(self._subs)} substrates. {hint}."
            )

    def _validate_Q(self, Q: float | Sequence[float]) -> np.ndarray:
        """Normalise *Q* to a per-substrate numpy array."""
        Q_arr = np.array([float(Q)], dtype=float) if np.isscalar(Q) else np.asarray(list(Q), dtype=float)

        n_subs = len(self._subs)
        if Q_arr.size < n_subs:
            Q_arr = np.concatenate([Q_arr, np.zeros(n_subs - Q_arr.size)])
        elif Q_arr.size > n_subs:
            extras = Q_arr[n_subs:]
            if np.any(extras != 0.0):
                raise ValueError(
                    f"Q has {Q_arr.size} entries with non-zero values beyond "
                    f"the {n_subs} configured substrates: {list(extras)}"
                )
            Q_arr = Q_arr[:n_subs]

        if not np.all(np.isfinite(Q_arr)) or np.any(Q_arr < 0.0):
            raise ValueError(f"Feed rates must be finite and not negative, got {list(Q_arr)}")

        factors = np.asarray(self._q_factors, dtype=float)
        return Q_arr * factors

    def _blended_concentrations(self, Q_arr: np.ndarray) -> dict:
        """Return flow-weighted blended influent concentrations from per-substrate flows *Q_arr*."""
        q_tot = float(np.sum(Q_arr))
        keys = INFLUENT_COLUMNS[:-1]  # exclude "Q"
        if q_tot <= 0.0:
            return {k: 0.0 for k in keys}
        row = {k: 0.0 for k in keys}
        for q, conc in zip(Q_arr, self._conc_list):
            if q <= 0.0:
                continue
            w = q / q_tot
            for k in keys:
                row[k] += conc.get(k, 0.0) * w
        return row

    @staticmethod
    def _calc_density(s: SubstrateParams) -> float:
        """Estimate fresh-matter density [kg/m³] (ADM1da convention)."""
        if s.TS < 200.0:
            return 1000.0

        fTS = s.TS / 1000.0

        f_fiber_total = fTS * s.fRF
        f_protein = fTS * s.fRP
        f_lipid = fTS * s.fRFe
        f_ash = fTS * s.fRA
        f_NFE = fTS - f_fiber_total - f_protein - f_lipid - f_ash
        f_CH = f_fiber_total + f_NFE

        f_AC = s.vfa_mass / 1000.0
        f_H2O = max(0.0, 1.0 - fTS - f_AC)

        v_spec = (
            f_CH / s.roh_CH
            + f_protein / s.roh_PR
            + f_lipid / s.roh_LI
            + f_ash / s.roh_MI
            + f_H2O / s.roh_H2O
            + f_AC / s.roh_AC
        )
        return 1.0 / max(v_spec, 1.0e-10)

    @staticmethod
    def _calc_concentrations(s: SubstrateParams, rho: float) -> dict:
        """Compute all ADM1 influent concentrations per m³ of fresh substrate."""
        pools = Feedstock._calc_organic_pools(s, rho)

        # A measured soluble inert replaces the aSi share. It is taken out of
        # the inert COD, so the total COD does not change.
        if s.SIin is not None:
            inert = pools["X_I"] + pools["S_I"]
            pools["S_I"] = min(float(s.SIin), inert)
            pools["X_I"] = inert - pools["S_I"]

        m_ac, m_pro, m_bu, m_va = s.vfa_masses
        acids = {
            "S_ac": m_ac / s.M_Sac,
            "S_pro": m_pro / s.M_Spro,
            "S_bu": m_bu / s.M_Sbu,
            "S_va": m_va / s.M_Sva,
        }

        hydrolysable, biomass = Feedstock._split_off_filtrate_and_biomass(s, pools, sum(acids.values()))

        S_nh4 = s.NH4 / 14.0

        S_H = 10.0 ** (-s.pH)
        alpha = {
            "S_ac": s.Kaac / (s.Kaac + S_H),
            "S_pro": s.Kapro / (s.Kapro + S_H),
            "S_bu": s.Kabu / (s.Kabu + S_H),
            "S_va": s.Kava / (s.Kava + S_H),
        }
        alpha_IN = s.Kain_35 / (s.Kain_35 + S_H)

        S_hco3 = Feedstock._calc_bicarbonate(s, acids, alpha)
        S_co2 = max(S_hco3, 0.0)

        ions = {key: alpha[key] * acids[key] for key in acids}
        S_nh3 = alpha_IN * S_nh4

        # Charge balance, closed by the strong ions. The closure is split by
        # sign so that both pools stay non-negative: a net-anionic material
        # (silage at its own pH) carries the excess as strong-acid anions, a
        # net-cationic one (slurry, solid manure: HCO3- and VFA- exceed NH4+)
        # as strong-base cations (K+, Na+, Ca2+, Mg2+). Only the difference
        # S_cation - S_anion enters pH, speciation and the acid capacity, so
        # the chemistry equals the ADM1da closure that fixes S_cation = 0 and
        # lets S_anion go negative (Schlattmann 2011); the split keeps the
        # influent inside the non-negative state box that state estimators
        # and the benchmark truth are clipped to.
        vfa_kmol = (
            ions["S_ac"] / _VFA_COD_PER_KMOL["S_ac"]
            + ions["S_pro"] / _VFA_COD_PER_KMOL["S_pro"]
            + ions["S_bu"] / _VFA_COD_PER_KMOL["S_bu"]
            + ions["S_va"] / _VFA_COD_PER_KMOL["S_va"]
        )
        closure = S_H + (S_nh4 - S_nh3) - S_hco3 - vfa_kmol - s.Kw_35 / (S_H + 1.0e-30)
        S_anion = max(closure, 0.0)
        S_cation = max(-closure, 0.0)

        return {
            "S_su": 0.0,
            "S_aa": 0.0,
            "S_fa": 0.0,
            "S_va": acids["S_va"],
            "S_bu": acids["S_bu"],
            "S_pro": acids["S_pro"],
            "S_ac": acids["S_ac"],
            "S_h2": 0.0,
            "S_ch4": 0.0,
            "S_co2": S_co2,
            "S_nh4": S_nh4,
            "S_I": pools["S_I"],
            "X_PS_ch": pools["X_PS_ch"],
            "X_PS_pr": pools["X_PS_pr"],
            "X_PS_li": pools["X_PS_li"],
            "X_PF_ch": pools["X_PF_ch"],
            "X_PF_pr": pools["X_PF_pr"],
            "X_PF_li": pools["X_PF_li"],
            "X_S_ch": hydrolysable["ch"],
            "X_S_pr": hydrolysable["pr"],
            "X_S_li": hydrolysable["li"],
            "X_I": pools["X_I"],
            "X_su": biomass["X_su"],
            "X_aa": biomass["X_aa"],
            "X_fa": biomass["X_fa"],
            "X_c4": biomass["X_c4"],
            "X_pro": biomass["X_pro"],
            "X_ac": biomass["X_ac"],
            "X_h2": biomass["X_h2"],
            "S_cation": S_cation,
            "S_anion": S_anion,
            "S_va_ion": ions["S_va"],
            "S_bu_ion": ions["S_bu"],
            "S_pro_ion": ions["S_pro"],
            "S_ac_ion": ions["S_ac"],
            "S_hco3_ion": S_hco3,
            "S_nh3": S_nh3,
        }

    @staticmethod
    def _calc_organic_pools(s: SubstrateParams, rho: float) -> dict:
        """Disintegration pools and inerts [kgCOD/m³], by crude fibre or by cell wall."""
        if s.uses_cell_wall_analysis:
            return Feedstock._pools_from_cell_wall(s, rho)
        return Feedstock._pools_from_crude_fibre(s, rho)

    @staticmethod
    def _pools_from_crude_fibre(s: SubstrateParams, rho: float) -> dict:
        """
        ADM1da fractionation from the Weender analysis.

        The degradable crude fibre enters the slow pool; nitrogen-free
        extracts, protein and lipids are split by ``fsOTS`` / ``ffOTS``;
        ``aXI`` and ``aSi`` of the organic COD are inert.
        """
        fTS = s.TS / 1000.0

        f_fiber_deg = fTS * s.fRF * s.fOTSrf
        f_protein = fTS * s.fRP
        f_lipid = fTS * s.fRFe
        f_NFE = fTS - fTS * s.fRF - f_protein - f_lipid - fTS * s.fRA
        f_CH = f_fiber_deg + f_NFE

        X_ch_raw = f_CH * rho / s.M_Xch
        X_pr_raw = f_protein * rho / s.M_Xpr
        X_li_raw = f_lipid * rho / s.M_Xli
        X_org_raw = X_ch_raw + X_pr_raw + X_li_raw

        f_deg = max(1.0 - s.aXI - s.aSi, 0.0)

        X_ch_raw_NFE = f_NFE * rho / s.M_Xch
        X_ch_raw_fiber = f_fiber_deg * rho / s.M_Xch
        return {
            "X_PS_ch": (X_ch_raw_fiber + X_ch_raw_NFE * s.fsOTS) * f_deg,
            "X_PS_pr": X_pr_raw * f_deg * s.fsOTS,
            "X_PS_li": X_li_raw * f_deg * s.fsOTS,
            "X_PF_ch": X_ch_raw_NFE * f_deg * s.ffOTS,
            "X_PF_pr": X_pr_raw * f_deg * s.ffOTS,
            "X_PF_li": X_li_raw * f_deg * s.ffOTS,
            "X_I": X_org_raw * s.aXI,
            "S_I": X_org_raw * s.aSi,
        }

    @staticmethod
    def _pools_from_cell_wall(s: SubstrateParams, rho: float) -> dict:
        """
        Fractionation from the van Soest cell-wall analysis (``NDF``, ``ADL``).

        Structural carbohydrates (``NDF - ADL``) enter the slow pool,
        non-fibre carbohydrates the fast pool, lignin is inert. Protein and
        lipids are split by ``fsOTS`` / ``ffOTS``.

        With ``NDICP`` the protein bound in the cell wall enters the slow
        pool and the remaining protein the fast pool; ``fsOTS`` / ``ffOTS``
        then only split the lipids. The bound protein is part of the measured
        ``NDF`` and is taken out of the structural carbohydrates.

        With ``D_VS`` the non-degradable volatile solids ``VS * (1 - D_VS)``
        are assigned to the cell wall, lignin first (Koch et al. 2010,
        Bioresource Technology 101, 8158-8165). ``aXI`` and ``aSi`` are not
        used. Without ``D_VS`` the structural carbohydrates count as
        degradable and ``aXI`` / ``aSi`` apply to all organics except lignin.
        """
        m_ts = s.TS / 1000.0 * rho  # kg TS per m³ fresh matter

        f_carb = max(1.0 - s.fRA - s.fRP - s.fRFe, 0.0)  # all carbohydrates, lignin included
        # Protein bound in the cell wall is weighed with the NDF but is no carbohydrate.
        f_bound_pr = min(float(s.NDICP), s.fRP, float(s.NDF)) if s.NDICP is not None else 0.0
        f_ndf = min(float(s.NDF) - f_bound_pr, f_carb)
        f_lignin = min(float(s.ADL), f_ndf)
        f_struct = f_ndf - f_lignin
        f_nfc = f_carb - f_ndf
        f_other = f_nfc + s.fRP + s.fRFe  # cell contents

        if s.D_VS is not None:
            # Non-degradable VS: lignin first, then structural carbohydrates.
            # Only what exceeds the whole cell wall reaches the cell contents.
            f_nondeg = (1.0 - s.fRA) * (1.0 - s.D_VS)
            f_struct_inert = min(max(f_nondeg - f_lignin, 0.0), f_struct)
            f_rest = max(f_nondeg - f_lignin - f_struct, 0.0)
            xi_struct = f_struct_inert / f_struct if f_struct > 0.0 else 0.0
            xi_other = min(f_rest / f_other, 1.0) if f_other > 0.0 else 0.0
            si_struct = si_other = 0.0
        else:
            xi_struct = xi_other = s.aXI
            si_struct = si_other = s.aSi

        deg_struct = max(1.0 - xi_struct - si_struct, 0.0)
        deg_other = max(1.0 - xi_other - si_other, 0.0)

        X_struct = f_struct * m_ts / s.M_Xch
        X_nfc = f_nfc * m_ts / s.M_Xch
        X_pr = s.fRP * m_ts / s.M_Xpr
        X_li = s.fRFe * m_ts / s.M_Xli
        X_lignin = f_lignin * m_ts / s.M_Xlig
        X_other = X_nfc + X_pr + X_li

        if s.NDICP is not None:
            slow_pr = f_bound_pr / s.fRP if s.fRP > 0.0 else 0.0
            fast_pr = 1.0 - slow_pr
        else:
            slow_pr, fast_pr = s.fsOTS, s.ffOTS

        return {
            "X_PS_ch": X_struct * deg_struct,
            "X_PS_pr": X_pr * deg_other * slow_pr,
            "X_PS_li": X_li * deg_other * s.fsOTS,
            "X_PF_ch": X_nfc * deg_other,
            "X_PF_pr": X_pr * deg_other * fast_pr,
            "X_PF_li": X_li * deg_other * s.ffOTS,
            "X_I": X_lignin + X_struct * xi_struct + X_other * xi_other,
            "S_I": X_struct * si_struct + X_other * si_other,
        }

    @staticmethod
    def _split_off_filtrate_and_biomass(s: SubstrateParams, pools: dict, acid_cod: float) -> tuple[dict, dict]:
        """
        Move filtrate COD and living biomass out of the disintegration pools.

        Filtrate COD beyond the acids and the soluble inerts is particulate
        matter that is already disintegrated; it enters the hydrolysable pools
        ``X_S_*``. Biomass is part of the measured volatile solids as well.
        Both are taken from all six disintegration pools in proportion, so the
        total COD stays the same. *pools* is updated in place.

        Returns:
            The hydrolysable COD by substrate class and the biomass by ADM1 group.
        """
        class_cod = {cls: sum(pools[key] for key in keys) for cls, keys in _DISINTEGRATION_POOLS.items()}
        total = sum(class_cod.values())

        filtrate = max(float(s.COD_S) - acid_cod - pools["S_I"], 0.0) if s.COD_S is not None else 0.0
        bacteria, methanogens = float(s.X_bac), float(s.X_meth)
        moved = filtrate + bacteria + methanogens

        scale = 0.0
        if moved > 0.0 and total > 0.0:
            scale = min(total / moved, 1.0)  # never move more than the pools hold
            remaining = 1.0 - scale * moved / total
            for keys in _DISINTEGRATION_POOLS.values():
                for key in keys:
                    pools[key] *= remaining

        hydrolysable = {cls: scale * filtrate * (cod / total if total > 0.0 else 0.0) for cls, cod in class_cod.items()}
        biomass = {key: scale * bacteria * share for key, share in _BACTERIA_SPLIT.items()}
        biomass.update({key: scale * methanogens * share for key, share in _METHANOGEN_SPLIT.items()})
        return hydrolysable, biomass

    @staticmethod
    def _calc_bicarbonate(s: SubstrateParams, acids: dict, alpha: dict) -> float:
        """
        Bicarbonate of the substrate [kmol/m³].

        ``KS43`` is taken as bicarbonate directly. A titrated acid capacity
        ``TAC`` also contains the acid anions that are protonated on the way
        from the substrate pH down to pH 4.3; they are subtracted.
        """
        if s.TAC is None:
            return s.KS43 * 1.0e-3

        S_H_end = 10.0 ** (-_PH_TITRATION_END)
        K_a = {"S_ac": s.Kaac, "S_pro": s.Kapro, "S_bu": s.Kabu, "S_va": s.Kava}
        titrated = 0.0
        for key, cod in acids.items():
            alpha_end = K_a[key] / (K_a[key] + S_H_end)
            titrated += cod / _VFA_COD_PER_KMOL[key] * max(alpha[key] - alpha_end, 0.0)
        return max(float(s.TAC) * 1.0e-3 - titrated, 0.0)
