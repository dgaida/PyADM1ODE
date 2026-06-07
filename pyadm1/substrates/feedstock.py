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

import xml.etree.ElementTree as ET
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, List, Sequence, Union

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


def _order_substrates(available: Sequence[str]) -> List[str]:
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


# ---------------------------------------------------------------------------
# Substrate loaders (XML, YAML, TOML)
# ---------------------------------------------------------------------------


def _build_substrate_params(substrate_name: str, raw: Dict[str, object], source: Path) -> SubstrateParams:
    """Common dict -> SubstrateParams construction used by all loaders."""
    kwargs: dict = {"name": substrate_name}
    for f in fields(SubstrateParams):
        if f.name == "name":
            continue
        if f.name in raw:
            kwargs[f.name] = float(raw[f.name])
        elif f.default.__class__.__name__ != "_MISSING_TYPE":
            kwargs[f.name] = f.default
        else:
            raise ValueError(f"Required parameter '{f.name}' not found in {source.name}")
    return SubstrateParams(**kwargs)


def load_substrate_xml(path: Union[str, Path]) -> SubstrateParams:
    """
    Load a substrate definition from an XML file.

    Schema: ``<substrate name="...">`` root with ``<param name="..."
    value="..."/>`` children.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Substrate XML not found: {path}")

    root = ET.parse(path).getroot()
    substrate_name = root.get("name", path.stem)

    raw: Dict[str, str] = {}
    for elem in root.findall("param"):
        pname = elem.get("name")
        pvalue = elem.get("value")
        if pname is not None and pvalue is not None:
            raw[pname] = pvalue

    return _build_substrate_params(substrate_name, raw, path)


def load_substrate_yaml(path: Union[str, Path]) -> SubstrateParams:
    """
    Load a substrate definition from a YAML file.

    Schema: a flat top-level mapping of parameter name -> value, with one
    optional ``name`` key for the human-readable substrate label.
    """
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Loading YAML substrate files requires PyYAML. " "Install it with `pip install PyYAML`.") from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Substrate YAML not found: {path}")

    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Substrate YAML must be a mapping at the top level: {path}")

    substrate_name = str(data.pop("name", path.stem))
    return _build_substrate_params(substrate_name, data, path)


def load_substrate_toml(path: Union[str, Path]) -> SubstrateParams:
    """
    Load a substrate definition from a TOML file.

    Schema: top-level table with ``name = "..."`` plus one ``key = value``
    per substrate parameter.
    """
    try:
        import tomllib  # Python 3.11+
    except ImportError:  # pragma: no cover
        try:
            import tomli as tomllib  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Loading TOML substrate files on Python < 3.11 requires `tomli`. " "Install it with `pip install tomli`."
            ) from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Substrate TOML not found: {path}")

    with path.open("rb") as fp:
        data = tomllib.load(fp)
    if not isinstance(data, dict):
        raise ValueError(f"Substrate TOML must be a table at the top level: {path}")

    substrate_name = str(data.pop("name", path.stem))
    return _build_substrate_params(substrate_name, data, path)


def load_substrate(path: Union[str, Path]) -> SubstrateParams:
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
    raise ValueError(f"Unsupported substrate file extension '{suffix}' for {path}. " f"Supported: {_SUBSTRATE_EXTENSIONS}")


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
        data_dir: Union[str, Path, None] = None,
        xml_dir: Union[str, Path, None] = None,
    ) -> None:
        # ``xml_dir`` is the legacy keyword from when substrates were
        # XML-only; kept as a back-compat alias so existing callers don't
        # break. New code should use ``data_dir``.
        if xml_dir is not None:
            if data_dir is not None:
                raise TypeError("SubstrateRegistry() accepts 'data_dir' or 'xml_dir', not both.")
            data_dir = xml_dir
        self._dir = Path(data_dir) if data_dir is not None else _DEFAULT_DATA_DIR
        self._cache: Dict[str, SubstrateParams] = {}

    def available(self) -> List[str]:
        """Return substrate IDs (file stems) found in the directory."""
        if not self._dir.exists():
            return []
        seen: set = set()
        for ext in _SUBSTRATE_EXTENSIONS:
            for p in self._dir.glob(f"*{ext}"):
                seen.add(p.stem)
        return sorted(seen)

    def _find_path(self, substrate_id: str) -> Union[Path, None]:
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
                raise KeyError(f"Substrate '{substrate_id}' not found in {self._dir}. " f"Available: {self.available()}")
            self._cache[substrate_id] = load_substrate(path)
        return self._cache[substrate_id]

    def load_all(self) -> Dict[str, SubstrateParams]:
        """Load every substrate file in the directory."""
        for sid in self.available():
            self.get(sid)
        return dict(self._cache)


# ---------------------------------------------------------------------------
# Feedstock class
# ---------------------------------------------------------------------------


_SubstrateInput = Union[SubstrateParams, str, Path]


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
        substrates: Union[_SubstrateInput, Sequence[_SubstrateInput], None] = None,
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
                    f"No substrate files found in {_DEFAULT_DATA_DIR}; " "pass an explicit substrate list or add files."
                )
            self._multi = True
            raw_subs: List[_SubstrateInput] = list(_order_substrates(available))
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
        self._substrate_ids: List[str] = [self._raw_input_id(item) for item in raw_subs]

        self._subs: List[SubstrateParams] = [self._resolve_substrate(item) for item in raw_subs]
        self._simtime = np.arange(0, total_simtime, float(feeding_freq) / 24.0)
        self._feeding_freq = int(feeding_freq)

        self._densities: List[float] = [self._calc_density(s) for s in self._subs]
        self._conc_list: List[dict] = [self._calc_concentrations(s, rho) for s, rho in zip(self._subs, self._densities)]

        self._simba_q_convention = bool(simba_q_convention)
        if self._simba_q_convention:
            self._q_factors: List[float] = [1000.0 / rho for rho in self._densities]
        else:
            self._q_factors = [1.0] * len(self._subs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_influent_dataframe(self, Q: Union[float, Sequence[float]]) -> pd.DataFrame:
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

    def header(self) -> List[str]:
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
    def substrates(self) -> List[SubstrateParams]:
        """All configured substrates, in feed-index order."""
        return list(self._subs)

    @property
    def densities(self) -> List[float]:
        """Per-substrate fresh-matter densities [kg/m³]."""
        return list(self._densities)

    def actual_Q(self, Q: Union[float, Sequence[float]]) -> List[float]:
        """
        Return per-substrate actual liquid volume flows [m³/d].

        Applies the ADM1da mass-to-volume conversion when
        ``simba_q_convention=True``; otherwise returns *Q* unchanged.
        """
        return self._validate_Q(Q).tolist()

    @property
    def q_conversion_factors(self) -> List[float]:
        """Per-substrate ADM1da Q-conversion factors [-]."""
        return list(self._q_factors)

    @property
    def concentrations_list(self) -> List[dict]:
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
        return (
            c["X_PS_ch"]
            + c["X_PS_pr"]
            + c["X_PS_li"]
            + c["X_PF_ch"]
            + c["X_PF_pr"]
            + c["X_PF_li"]
            + c["X_I"]
            + c["S_ac"]
            + c["S_I"]
        )

    def bmp_theoretical(self, index: int = 0) -> float:
        """Theoretical biomethane potential of the i-th substrate [Nm³ CH₄/t VS]."""
        c = self._conc_list[index]
        s = self._subs[index]
        th_yield = s.V_m / (s.CH4_cod_2_mol / 1000.0)
        degradable_cod = c["X_PS_ch"] + c["X_PS_pr"] + c["X_PS_li"] + c["X_PF_ch"] + c["X_PF_pr"] + c["X_PF_li"] + c["S_ac"]
        vs = self.vs_content(index)
        if vs <= 0.0:
            return 0.0
        return degradable_cod * th_yield / vs * 1000.0

    def blended_density(self, Q: Union[float, Sequence[float]]) -> float:
        """Volumetric-flow-weighted fresh-matter density [kg/m³]."""
        Q_arr = self._validate_Q(Q)
        q_tot = float(np.sum(Q_arr))
        if q_tot <= 0.0:
            return 1000.0
        return float(np.dot(Q_arr, self._densities) / q_tot)

    def blended_vs_content(self, Q: Union[float, Sequence[float]]) -> float:
        """Volumetric-flow-weighted VS content [kg VS/m³]."""
        Q_arr = self._validate_Q(Q)
        q_tot = float(np.sum(Q_arr))
        if q_tot <= 0.0:
            return 0.0
        vs = np.array([self.vs_content(i) for i in range(len(self._subs))])
        return float(np.dot(Q_arr, vs) / q_tot)

    def blended_concentrations(self, Q: Union[float, Sequence[float]]) -> dict:
        """Volumetric-flow-weighted influent concentrations (no Q field)."""
        Q_arr = self._validate_Q(Q)
        return self._blended_concentrations(Q_arr)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @property
    def substrate_ids(self) -> List[str]:
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
    def _raw_input_id(item: _SubstrateInput) -> str:
        """Return a stable identifier for a substrate input item."""
        if isinstance(item, SubstrateParams):
            return item.name
        return Path(item).stem

    @staticmethod
    def _resolve_substrate(item: _SubstrateInput) -> SubstrateParams:
        """
        Accept a params object, a filesystem path, or a bare substrate ID
        (file stem in the default ``data/substrates/`` directory).
        Any supported format (.yaml/.yml/.xml/.toml) is recognised.
        """
        if isinstance(item, SubstrateParams):
            return item
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
                f"'{prop}' is a single-substrate accessor; this feedstock has " f"{len(self._subs)} substrates. {hint}."
            )

    def _validate_Q(self, Q: Union[float, Sequence[float]]) -> np.ndarray:
        """Normalise *Q* to a per-substrate numpy array."""
        if np.isscalar(Q):
            Q_arr = np.array([float(Q)], dtype=float)
        else:
            Q_arr = np.asarray(list(Q), dtype=float)

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

        f_AC = s.FFS / 1000.0
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

        X_I = X_org_raw * s.aXI
        S_I = X_org_raw * s.aSi
        f_deg = max(1.0 - s.aXI - s.aSi, 0.0)

        X_ch_raw_NFE = f_NFE * rho / s.M_Xch
        X_ch_raw_fiber = f_fiber_deg * rho / s.M_Xch
        X_PS_ch = (X_ch_raw_fiber + X_ch_raw_NFE * s.fsOTS) * f_deg
        X_PS_pr = X_pr_raw * f_deg * s.fsOTS
        X_PS_li = X_li_raw * f_deg * s.fsOTS
        X_PF_ch = X_ch_raw_NFE * f_deg * s.ffOTS
        X_PF_pr = X_pr_raw * f_deg * s.ffOTS
        X_PF_li = X_li_raw * f_deg * s.ffOTS

        S_ac = s.FFS / s.M_Sac
        S_nh4 = s.NH4 / 14.0

        S_hco3 = s.KS43 * 1.0e-3
        S_co2 = max(S_hco3, 0.0)

        S_H = 10.0 ** (-s.pH)
        alpha_ac = s.Kaac / (s.Kaac + S_H)
        alpha_pro = s.Kapro / (s.Kapro + S_H)
        alpha_bu = s.Kabu / (s.Kabu + S_H)
        alpha_va = s.Kava / (s.Kava + S_H)
        alpha_IN = s.Kain_35 / (s.Kain_35 + S_H)

        S_ac_ion = alpha_ac * S_ac
        S_pro_ion = alpha_pro * 0.0
        S_bu_ion = alpha_bu * 0.0
        S_va_ion = alpha_va * 0.0
        S_nh3 = alpha_IN * S_nh4

        # Charge balance
        vfa_kmol = S_ac_ion / 64.0 + S_pro_ion / 112.0 + S_bu_ion / 160.0 + S_va_ion / 208.0
        S_cation = 0.0
        S_anion = S_cation + S_H + (S_nh4 - S_nh3) - S_hco3 - vfa_kmol - s.Kw_35 / (S_H + 1.0e-30)

        return {
            "S_su": 0.0,
            "S_aa": 0.0,
            "S_fa": 0.0,
            "S_va": 0.0,
            "S_bu": 0.0,
            "S_pro": 0.0,
            "S_ac": S_ac,
            "S_h2": 0.0,
            "S_ch4": 0.0,
            "S_co2": S_co2,
            "S_nh4": S_nh4,
            "S_I": S_I,
            "X_PS_ch": X_PS_ch,
            "X_PS_pr": X_PS_pr,
            "X_PS_li": X_PS_li,
            "X_PF_ch": X_PF_ch,
            "X_PF_pr": X_PF_pr,
            "X_PF_li": X_PF_li,
            "X_S_ch": 0.0,
            "X_S_pr": 0.0,
            "X_S_li": 0.0,
            "X_I": X_I,
            "X_su": 0.0,
            "X_aa": 0.0,
            "X_fa": 0.0,
            "X_c4": 0.0,
            "X_pro": 0.0,
            "X_ac": 0.0,
            "X_h2": 0.0,
            "S_cation": S_cation,
            "S_anion": S_anion,
            "S_va_ion": S_va_ion,
            "S_bu_ion": S_bu_ion,
            "S_pro_ion": S_pro_ion,
            "S_ac_ion": S_ac_ion,
            "S_hco3_ion": S_hco3,
            "S_nh3": S_nh3,
        }
