# pyadm1/substrates/adm1da_feedstock.py
"""
Pure-Python substrate characterization for the ADM1da model.

Substrate definitions are stored as XML files under
``data/substrates/adm1da/``.  The ``load_substrate_xml`` function reads one
file and returns an ``ADM1daSubstrateParams`` dataclass.  The
``SubstrateRegistry`` class discovers and indexes all XML files in that
directory.

``ADM1daFeedstock`` converts the characterization data into the 38-column
influent DataFrame expected by ``ADM1da.set_influent_dataframe()``.

No C#/.NET DLL dependency — works with any Python environment.

Reference: SIMBA# biogas 4.2 Tutorial, ifak e.V. Magdeburg (Schlattmann 2011)

Calculation overview
--------------------
1. Compute fresh-matter density from component densities (Weender fractions).
2. Convert organic dry-matter fractions to COD concentrations [kg COD m⁻³]:
       X_ch = m_CH × ρ / M_Xch  (degradable carbohydrates)
       X_pr = m_PR × ρ / M_Xpr  (crude proteins)
       X_li = m_LI × ρ / M_Xli  (crude lipids)
3. Calculate particulate and dissolved inerts from total degradable COD:
       X_I = X_org × aXI / (1 − aXI)
       S_I = X_org × aSi / (1 − aSi)
4. Distribute particulate organics between slow (XPS, fsOTS) and fast
   (XPF, ffOTS) disintegration pools.
5. Set dissolved concentrations from FFS (acetate), NH4, KS43 (DIC).
6. Derive acid-base ion concentrations at the substrate pH.
7. Derive S_anion (or S_cation) from the charge balance at the given pH.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, List, Sequence, Union

import numpy as np
import pandas as pd

# Maximum number of substrates that can be fed simultaneously (matches ADM1
# convention in pyadm1.substrates.feedstock.Feedstock.get_influent_dataframe).
MAX_SUBSTRATES = 10

# Default location of ADM1da substrate XML files
_DEFAULT_XML_DIR = Path(__file__).parent.parent.parent / "data" / "substrates" / "adm1da"

# ---------------------------------------------------------------------------
# Influent column names for the ADM1da model
# (defined here to avoid a circular import with pyadm1.core.adm1da)
# ---------------------------------------------------------------------------
INFLUENT_COLUMNS = [
    "S_su",
    "S_aa",
    "S_fa",
    "S_va",
    "S_bu",
    "S_pro",
    "S_ac",
    "S_h2",
    "S_ch4",
    "S_co2",
    "S_nh4",
    "S_I",
    "X_PS_ch",
    "X_PS_pr",
    "X_PS_li",
    "X_PF_ch",
    "X_PF_pr",
    "X_PF_li",
    "X_S_ch",
    "X_S_pr",
    "X_S_li",
    "X_I",
    "X_su",
    "X_aa",
    "X_fa",
    "X_c4",
    "X_pro",
    "X_ac",
    "X_h2",
    "S_cation",
    "S_anion",
    "S_va_ion",
    "S_bu_ion",
    "S_pro_ion",
    "S_ac_ion",
    "S_hco3_ion",
    "S_nh3",
    "Q",
]


# ---------------------------------------------------------------------------
# Substrate parameter dataclass
# ---------------------------------------------------------------------------


@dataclass
class ADM1daSubstrateParams:
    """
    Complete characterization of one substrate for the ADM1da model.

    All substrate-specific measurements and ADM1da model constants needed
    to compute the influent concentration vector.  Instances are normally
    created via ``load_substrate_xml()`` rather than constructed manually.

    Measured substrate properties (per ton or m³ of fresh matter)
    -------------------------------------------------------------
    name  : str    Human-readable label.
    TS    : float  Total solids [kg/t FM]  (liquid substrates: kg/m³ FM)
    NH4   : float  Ammonia nitrogen [kg N/m³ FM]  (SIMBA# convention: always per m³)
    BGP   : float  Biogas potential [Nm³/t VS]  (reference)
    BMP   : float  Biomethane potential [Nm³ CH₄/t VS]  (reference)

    COD fractionation
    -----------------
    aXI    : float  Particulate inert fraction of degradable organic COD [-]
    fOTSrf : float  Biodegradable fraction of crude fibre [-]
    fsOTS  : float  VS fraction entering slow disintegration pool (XPS) [-]
    ffOTS  : float  VS fraction entering fast disintegration pool (XPF) [-]
    aSi    : float  Dissolved inert fraction of degradable organic COD [-]

    Weender analysis (fractions of TS)
    ------------------------------------
    fRF  : float  Crude fibre fraction [-]
    fRP  : float  Crude protein fraction [-]
    fRFe : float  Crude lipid fraction [-]
    fRA  : float  Ash fraction [-]

    Physical/chemical state
    -----------------------
    Temp : float  Temperature [°C]
    pH   : float  pH [-]
    KS43 : float  Acid capacity to pH 4.3 [mol/m³]
    FFS  : float  Volatile fatty acids as acetic-acid equivalent [kg HAc/m³]

    Component densities [kg/m³]  (ADM1da model defaults)
    -----------------------------------------------------
    roh_CH, roh_PR, roh_LI, roh_MI, roh_AC, roh_H2O

    COD conversion factors [kg dry mass / kg COD]  (ADM1da model defaults)
    -----------------------------------------------------------------------
    M_Xch, M_Xpr, M_Xli  — particulate fractions
    M_Sac, M_Spro, M_Sbu, M_Sva, M_Sh2 — dissolved components
    M_XB  — active biomass

    Methane potentials [fraction of theoretical, -]  (ADM1da defaults)
    -------------------------------------------------------------------
    MP_CH, MP_PR, MP_LI, MP_AC

    Physical / acid-base constants  (ADM1da model defaults)
    --------------------------------------------------------
    V_m, CH4_cod_2_mol, Kw_35, N_aa,
    Kava, Kabu, Kapro, Kaac, Kaco2_35, Kain_35
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
# XML loader
# ---------------------------------------------------------------------------


def load_substrate_xml(path: Union[str, Path]) -> ADM1daSubstrateParams:
    """
    Load an ADM1da substrate definition from an XML file.

    The XML file must follow the schema used in
    ``data/substrates/adm1da/*.xml``:  a ``<substrate>`` root element
    with a ``name`` attribute and ``<param name="..." value="..."/>``
    child elements.

    Parameters
    ----------
    path : str or Path
        Path to the substrate XML file.

    Returns
    -------
    ADM1daSubstrateParams
        Populated substrate parameter dataclass.

    Raises
    ------
    FileNotFoundError
        If the XML file does not exist.
    ValueError
        If a required parameter is missing from the XML file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Substrate XML not found: {path}")

    tree = ET.parse(path)
    root = tree.getroot()

    substrate_name = root.get("name", path.stem)

    # Collect all <param> elements into a flat dict
    raw: Dict[str, str] = {}
    for elem in root.findall("param"):
        pname = elem.get("name")
        pvalue = elem.get("value")
        if pname is not None and pvalue is not None:
            raw[pname] = pvalue

    all_fields = fields(ADM1daSubstrateParams)

    # Build constructor kwargs
    kwargs: dict = {"name": substrate_name}
    for f in all_fields:
        if f.name == "name":
            continue
        if f.name in raw:
            kwargs[f.name] = float(raw[f.name])
        elif f.default.__class__.__name__ != "_MISSING_TYPE":
            kwargs[f.name] = f.default
        else:
            raise ValueError(f"Required parameter '{f.name}' not found in {path.name}")

    return ADM1daSubstrateParams(**kwargs)


# ---------------------------------------------------------------------------
# Substrate registry
# ---------------------------------------------------------------------------


class SubstrateRegistry:
    """
    Discovers and lazy-loads all ADM1da substrate XML files from a directory.

    Parameters
    ----------
    xml_dir : str or Path, optional
        Directory containing ``*.xml`` substrate files.
        Defaults to ``data/substrates/adm1da/`` relative to the project root.

    Usage
    -----
    >>> registry = SubstrateRegistry()
    >>> print(registry.available())
    ['maize_silage_milk_ripeness', 'swine_manure']
    >>> sub = registry.get("swine_manure")
    >>> print(sub.name, sub.TS)
    """

    def __init__(self, xml_dir: Union[str, Path, None] = None) -> None:
        self._dir = Path(xml_dir) if xml_dir is not None else _DEFAULT_XML_DIR
        self._cache: Dict[str, ADM1daSubstrateParams] = {}

    def available(self) -> List[str]:
        """Return substrate IDs (= XML file stems) found in the directory."""
        if not self._dir.exists():
            return []
        return sorted(p.stem for p in self._dir.glob("*.xml"))

    def get(self, substrate_id: str) -> ADM1daSubstrateParams:
        """
        Return the substrate with the given ID, loading it if necessary.

        Parameters
        ----------
        substrate_id : str
            File stem of the XML file (without ``.xml``).

        Returns
        -------
        ADM1daSubstrateParams

        Raises
        ------
        KeyError
            If no XML file matching *substrate_id* exists.
        """
        if substrate_id not in self._cache:
            path = self._dir / f"{substrate_id}.xml"
            if not path.exists():
                available = self.available()
                raise KeyError(f"Substrate '{substrate_id}' not found in {self._dir}. " f"Available: {available}")
            self._cache[substrate_id] = load_substrate_xml(path)
        return self._cache[substrate_id]

    def load_all(self) -> Dict[str, ADM1daSubstrateParams]:
        """Load all XML files in the directory and return a dict id → params."""
        for sid in self.available():
            self.get(sid)
        return dict(self._cache)


# ---------------------------------------------------------------------------
# Feedstock class
# ---------------------------------------------------------------------------


_SubstrateInput = Union[ADM1daSubstrateParams, str, Path]


class ADM1daFeedstock:
    """
    Computes ADM1da influent concentrations from substrate characterization.

    Accepts either a single substrate (``ADM1daSubstrateParams``, XML path)
    or a list of up to ``MAX_SUBSTRATES`` substrates for co-digestion.
    In multi-substrate mode the influent DataFrame is generated from a
    volumetric-flow-weighted blend of per-substrate concentrations — the
    same convention used by the ADM1 ``Feedstock`` class.

    Usage (single substrate)
    ------------------------
    >>> from pyadm1.substrates import ADM1daFeedstock, load_substrate_xml
    >>> sub = load_substrate_xml("data/substrates/adm1da/maize_silage_milk_ripeness.xml")
    >>> fs = ADM1daFeedstock(sub, feeding_freq=48, total_simtime=60)
    >>> df = fs.get_influent_dataframe(Q=15.0)

    Usage (co-digestion, up to 10 substrates)
    -----------------------------------------
    >>> from pyadm1.substrates import ADM1daFeedstock, SubstrateRegistry
    >>> reg = SubstrateRegistry()
    >>> fs = ADM1daFeedstock(
    ...     [reg.get("maize_silage_milk_ripeness"), reg.get("swine_manure")],
    ...     feeding_freq=24,
    ...     total_simtime=160,
    ... )
    >>> Q = [11.4, 6.1, 0, 0, 0, 0, 0, 0, 0, 0]  # m³/d, up to 10 slots
    >>> df = fs.get_influent_dataframe(Q=Q)
    """

    def __init__(
        self,
        substrates: Union[_SubstrateInput, Sequence[_SubstrateInput]],
        feeding_freq: int = 48,
        total_simtime: int = 60,
        simba_q_convention: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        substrates : ADM1daSubstrateParams | str | Path | list of those
            A single substrate object/XML path, or a list of up to
            ``MAX_SUBSTRATES`` substrates (or XML paths) to co-digest.
        feeding_freq : int
            Time between feeding events [hours].
        total_simtime : int
            Total simulation duration [days].
        simba_q_convention : bool, default True
            How to interpret ``Q`` in ``get_influent_dataframe(Q=...)``.

            * ``True`` (default, SIMBA# biogas convention): each Q_i [m³/d]
              is interpreted as a **mass-equivalent** flow — i.e. SIMBA#
              internally multiplies by the water density (1000 kg/m³) to
              get a mass flow in t/d, then divides by the true fresh-matter
              density to obtain the actual liquid volume added to the
              reactor.  Effective conversion factor is
              ``Q_actual_i = Q_input_i · 1000 / ρ_FM_i``.
              For liquid substrates (TS < 200) ρ_FM ≈ 1000 so no change.
              For solid substrates like maize silage (ρ_FM ≈ 1134) the
              actual volume is ~12 % smaller than the user input — this
              reproduces SIMBA#'s volumetric behaviour so that the same
              Q value from a SIMBA# study gives the same reactor loading.
            * ``False``: Q is taken literally as the actual liquid volume
              added to the reactor [m³/d], with no mass-to-volume
              conversion.  Use this when you want full control over the
              volumetric feed and don't care about SIMBA# parity.
        """
        # Detect multi-substrate vs single-substrate input and normalise
        # everything to an internal list representation.
        if isinstance(substrates, (list, tuple)):
            if len(substrates) == 0:
                raise ValueError("At least one substrate must be provided.")
            if len(substrates) > MAX_SUBSTRATES:
                raise ValueError(f"Up to {MAX_SUBSTRATES} substrates are supported; got {len(substrates)}.")
            self._multi = True
            raw_subs = list(substrates)
        else:
            self._multi = False
            raw_subs = [substrates]

        self._subs: List[ADM1daSubstrateParams] = [self._resolve_substrate(item) for item in raw_subs]
        self._simtime = np.arange(0, total_simtime, float(feeding_freq) / 24.0)

        # Pre-compute once per substrate; reused for every get_influent_dataframe call
        self._densities: List[float] = [self._calc_density(s) for s in self._subs]
        self._conc_list: List[dict] = [self._calc_concentrations(s, rho) for s, rho in zip(self._subs, self._densities)]

        # SIMBA# mass-to-volume conversion factors (1 per substrate).
        # For each substrate, a user-supplied Q_i is scaled by this factor
        # before being used as the actual liquid volume flow.
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
        Generate an ADM1da influent DataFrame for the full simulation period.

        Substrate concentrations are constant in time (steady-state feed
        composition assumption).  Pass the result to
        ``ADM1da.set_influent_dataframe()``.

        Parameters
        ----------
        Q : float or sequence of float
            Volumetric flow rates [m³/d].

            * Single-substrate mode: a scalar (or 1-element list).
            * Multi-substrate mode: a list with one flow rate per substrate.
              May include trailing zeros up to ``MAX_SUBSTRATES`` slots
              (matching the ADM1 ``Feedstock`` convention), e.g.
              ``[11.4, 6.1, 0, 0, 0, 0, 0, 0, 0, 0]``.

        Returns
        -------
        pd.DataFrame
            Columns match ``INFLUENT_COLUMNS`` (37 state variables + Q).
        """
        Q_arr = self._validate_Q(Q)
        q_total = float(np.sum(Q_arr))
        row = self._blended_concentrations(Q_arr)
        row["Q"] = q_total

        df = pd.DataFrame([row] * len(self._simtime))
        return df[INFLUENT_COLUMNS]

    # ---- Single-substrate convenience accessors -----------------------

    @property
    def substrate(self) -> ADM1daSubstrateParams:
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
        """Influent concentrations [kg COD/m³ or kmol/m³] (single-substrate mode)."""
        self._require_single("concentrations", hint="use .concentrations_list[i] or .blended_concentrations(Q)")
        return dict(self._conc_list[0])

    # ---- Multi-substrate accessors ------------------------------------

    @property
    def substrates(self) -> List[ADM1daSubstrateParams]:
        """All configured substrates, in feed-index order."""
        return list(self._subs)

    @property
    def densities(self) -> List[float]:
        """Per-substrate fresh-matter densities [kg/m³]."""
        return list(self._densities)

    def actual_Q(self, Q: Union[float, Sequence[float]]) -> List[float]:
        """
        Return the per-substrate actual liquid volume flows [m³/d].

        Applies the SIMBA# mass-to-volume conversion when the feedstock was
        created with ``simba_q_convention=True``; otherwise returns *Q* as
        a list unchanged.  Use this anywhere you need the volumetric feed
        rate that actually enters the reactor (volume balance, HRT, etc.).
        """
        return self._validate_Q(Q).tolist()

    @property
    def q_conversion_factors(self) -> List[float]:
        """
        Per-substrate SIMBA# Q-conversion factors [-].

        With ``simba_q_convention=True``, each user-supplied Q_i is multiplied
        by this factor to obtain the actual liquid volume added to the reactor.
        Equals ``1000 / ρ_FM_i`` for solid substrates (< 1.0) and 1.0 for
        liquid substrates (ρ_FM = 1000 by convention).  All 1.0 when
        ``simba_q_convention=False``.
        """
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
        """
        Theoretical biomethane potential of the i-th substrate from COD
        stoichiometry [Nm³ CH₄/t VS].  Assumes 100 % conversion of
        degradable COD — the given BMP is lower due to incomplete
        biodegradation and kinetic limitations.
        """
        c = self._conc_list[index]
        s = self._subs[index]
        th_yield = s.V_m / (s.CH4_cod_2_mol / 1000.0)  # Nm³ CH4 / kg COD
        degradable_cod = c["X_PS_ch"] + c["X_PS_pr"] + c["X_PS_li"] + c["X_PF_ch"] + c["X_PF_pr"] + c["X_PF_li"] + c["S_ac"]
        vs = self.vs_content(index)
        if vs <= 0.0:
            return 0.0
        return degradable_cod * th_yield / vs * 1000.0  # Nm³ CH4 / t VS

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

    @staticmethod
    def _resolve_substrate(item: _SubstrateInput) -> ADM1daSubstrateParams:
        """
        Accept a params object, a filesystem path, or a bare substrate ID
        (XML file stem in the default ``data/substrates/adm1da/`` directory).
        """
        if isinstance(item, ADM1daSubstrateParams):
            return item
        p = Path(item)
        if p.exists():
            return load_substrate_xml(p)
        # Treat as a substrate ID resolvable from the default XML directory
        reg_path = _DEFAULT_XML_DIR / f"{p.stem}.xml"
        if reg_path.exists():
            return load_substrate_xml(reg_path)
        raise FileNotFoundError(f"Substrate '{item}' not found as a path or as an ID in {_DEFAULT_XML_DIR}")

    def _require_single(self, prop: str, hint: str) -> None:
        if len(self._subs) != 1:
            raise ValueError(
                f"'{prop}' is a single-substrate accessor; this feedstock has " f"{len(self._subs)} substrates. {hint}."
            )

    def _validate_Q(self, Q: Union[float, Sequence[float]]) -> np.ndarray:
        """Normalise *Q* to a per-substrate numpy array, padding trailing zeros.

        When ``simba_q_convention=True`` (constructor default), each Q_i is
        also scaled by ``1000 / ρ_FM_i`` so that downstream users always see
        the actual liquid-volume flow rate [m³/d].
        """
        if np.isscalar(Q):
            Q_arr = np.array([float(Q)], dtype=float)
        else:
            Q_arr = np.asarray(list(Q), dtype=float)

        n_subs = len(self._subs)
        if Q_arr.size < n_subs:
            Q_arr = np.concatenate([Q_arr, np.zeros(n_subs - Q_arr.size)])
        elif Q_arr.size > n_subs:
            # Allow trailing zero slots (e.g. up to MAX_SUBSTRATES placeholders)
            extras = Q_arr[n_subs:]
            if np.any(extras != 0.0):
                raise ValueError(
                    f"Q has {Q_arr.size} entries with non-zero values beyond "
                    f"the {n_subs} configured substrates: {list(extras)}"
                )
            Q_arr = Q_arr[:n_subs]

        # Apply SIMBA# mass-to-volume conversion (no-op when factors are 1.0).
        factors = np.asarray(self._q_factors, dtype=float)
        return Q_arr * factors

    def _blended_concentrations(self, Q_arr: np.ndarray) -> dict:
        """Flow-weighted blend of per-substrate concentrations."""
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
    def _calc_density(s: ADM1daSubstrateParams) -> float:
        """
        Estimate fresh-matter density from component densities [kg/m³].

        For liquid substrates (TS < 200 kg/m³ or kg/t FM), SIMBA# uses
        ρ = 1000 kg/m³ by convention (density ≈ water).  This avoids an
        ~2–3% COD overestimate that the specific-volume mixing rule would
        introduce for dilute slurries.

        For solid/semi-solid substrates (TS ≥ 200), the specific-volume
        mixing rule is used:
            1/ρ_FM = Σ (mass_fraction_i / ρ_i)
        """
        # Liquid substrate convention (SIMBA# biogas 4.2)
        if s.TS < 200.0:
            return 1000.0

        fTS = s.TS / 1000.0

        f_fiber_total = fTS * s.fRF
        f_protein = fTS * s.fRP
        f_lipid = fTS * s.fRFe
        f_ash = fTS * s.fRA
        f_NFE = fTS - f_fiber_total - f_protein - f_lipid - f_ash
        f_CH = f_fiber_total + f_NFE

        # FFS (acetic acid) is dissolved in the water phase; subtract from water
        f_AC = s.FFS / 1000.0  # approximate: uses ρ_FM ≈ 1000 initially
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
    def _calc_concentrations(s: ADM1daSubstrateParams, rho: float) -> dict:
        """
        Compute all ADM1da influent concentrations per m³ of fresh substrate.

        Returns
        -------
        dict
            Keys match INFLUENT_COLUMNS (without Q).
            Organic fractions in kg COD/m³; ionic species in kmol/m³.
        """
        fTS = s.TS / 1000.0

        # --- Organic dry-mass fractions per kg FM ---
        f_fiber_deg = fTS * s.fRF * s.fOTSrf  # degradable crude fibre
        f_protein = fTS * s.fRP
        f_lipid = fTS * s.fRFe
        f_NFE = fTS - fTS * s.fRF - f_protein - f_lipid - fTS * s.fRA
        f_CH = f_fiber_deg + f_NFE  # total degradable carbohydrates

        # --- Raw organic COD per m³ from Weender fractions [kg COD/m³] ---
        # These include both degradable and inert organic matter.
        X_ch_raw = f_CH * rho / s.M_Xch
        X_pr_raw = f_protein * rho / s.M_Xpr
        X_li_raw = f_lipid * rho / s.M_Xli
        X_org_raw = X_ch_raw + X_pr_raw + X_li_raw

        # --- Fractionation (SIMBA# convention) ---
        # aXI = X_I  / X_org_raw  →  fraction of raw organic COD that is inert
        # aSi = S_I  / X_org_raw  →  fraction of raw organic COD becoming dissolved inerts
        # The remainder (1 - aXI - aSi) is truly degradable particulate COD.
        X_I = X_org_raw * s.aXI
        S_I = X_org_raw * s.aSi
        f_deg = max(1.0 - s.aXI - s.aSi, 0.0)

        # --- Distribute degradable fraction to slow (XPS) and fast (XPF) pools ---
        # SIMBA# convention: crude fibre (f_fiber_deg) always routes to the slow
        # disintegration pool (X_PS) regardless of fsOTS/ffOTS.  Only the NFE
        # fraction is split between X_PS and X_PF via fsOTS/ffOTS.
        # Proteins and lipids follow fsOTS/ffOTS directly (no fixed-pool routing).
        X_ch_raw_NFE = f_NFE * rho / s.M_Xch  # NFE only → PS/PF split
        X_ch_raw_fiber = f_fiber_deg * rho / s.M_Xch  # crude fibre → always PS
        X_PS_ch = (X_ch_raw_fiber + X_ch_raw_NFE * s.fsOTS) * f_deg
        X_PS_pr = X_pr_raw * f_deg * s.fsOTS
        X_PS_li = X_li_raw * f_deg * s.fsOTS
        X_PF_ch = X_ch_raw_NFE * f_deg * s.ffOTS
        X_PF_pr = X_pr_raw * f_deg * s.ffOTS
        X_PF_li = X_li_raw * f_deg * s.ffOTS

        # --- Acetic acid (FFS) → S_ac [kg COD/m³] ---
        S_ac = s.FFS / s.M_Sac

        # --- Ammonium: NH4 [kg N/m³ FM] → S_nh4 [kmol N/m³] ---
        # SIMBA# convention: NH4 is always kg N per m³ of fresh matter for
        # all substrate types.  No density correction is applied (for liquid
        # substrates with ρ≈1000 this is numerically equivalent to kg N/t FM).
        S_nh4 = s.NH4 / 14.0

        # --- Inorganic carbon from acid capacity KS43 ---
        # KS43 [mol/m³] ≈ alkalinity ≈ [HCO3⁻] at pH > 4.3
        S_hco3 = s.KS43 * 1.0e-3  # mol/m³ → kmol/m³
        S_co2 = max(S_hco3, 0.0)

        # --- Acid-base equilibrium at substrate pH ---
        S_H = 10.0 ** (-s.pH)  # [kmol/m³] = [M]

        alpha_ac = s.Kaac / (s.Kaac + S_H)
        alpha_pro = s.Kapro / (s.Kapro + S_H)
        alpha_bu = s.Kabu / (s.Kabu + S_H)
        alpha_va = s.Kava / (s.Kava + S_H)
        alpha_IN = s.Kain_35 / (s.Kain_35 + S_H)

        S_ac_ion = alpha_ac * S_ac  # [kg COD/m³]
        S_pro_ion = alpha_pro * 0.0  # no other VFAs in fresh substrate
        S_bu_ion = alpha_bu * 0.0
        S_va_ion = alpha_va * 0.0
        S_nh3 = alpha_IN * S_nh4  # [kmol N/m³]

        # --- Charge balance at substrate pH → S_anion (or S_cation) ---
        # Balance:  S_cat + S_H + (S_nh4−S_nh3) = S_an + Kw/S_H + S_hco3 + vfa_anions
        # Set S_cation = 0; solve for S_anion.  If negative → use as S_cation instead.
        vfa_kmol = S_ac_ion / 64.0 + S_pro_ion / 112.0 + S_bu_ion / 160.0 + S_va_ion / 208.0

        S_anion = (S_nh4 - S_nh3) - S_hco3 - vfa_kmol + S_H - s.Kw_35 / (S_H + 1.0e-30)
        S_cation = 0.0

        if S_anion < 0.0:
            S_cation = -S_anion
            S_anion = 0.0

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
