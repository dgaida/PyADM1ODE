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
from typing import Dict, List, Union

import numpy as np
import pandas as pd

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


class ADM1daFeedstock:
    """
    Computes ADM1da influent concentrations from substrate characterization.

    Accepts either an ``ADM1daSubstrateParams`` object (obtained from
    ``load_substrate_xml``) or a path to an XML file.

    Usage
    -----
    >>> from pyadm1.substrates import ADM1daFeedstock, load_substrate_xml
    >>> sub = load_substrate_xml("data/substrates/adm1da/maize_silage_milk_ripeness.xml")
    >>> fs = ADM1daFeedstock(sub, feeding_freq=48, total_simtime=60)
    >>> df = fs.get_influent_dataframe(Q=15.0)

    Or using the registry shortcut:

    >>> from pyadm1.substrates import SubstrateRegistry, ADM1daFeedstock
    >>> reg = SubstrateRegistry()
    >>> fs = ADM1daFeedstock(reg.get("swine_manure"))
    >>> df = fs.get_influent_dataframe(Q=10.0)
    """

    def __init__(
        self,
        substrate: Union[ADM1daSubstrateParams, str, Path],
        feeding_freq: int = 48,
        total_simtime: int = 60,
    ) -> None:
        """
        Parameters
        ----------
        substrate : ADM1daSubstrateParams | str | Path
            Substrate data object **or** path to a substrate XML file.
        feeding_freq : int
            Time between feeding events [hours].
        total_simtime : int
            Total simulation duration [days].
        """
        if isinstance(substrate, (str, Path)):
            substrate = load_substrate_xml(substrate)
        self._sub = substrate
        self._simtime = np.arange(0, total_simtime, float(feeding_freq) / 24.0)

        # Pre-compute once; reused for every get_influent_dataframe call
        self._density = self._calc_density()
        self._conc = self._calc_concentrations()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_influent_dataframe(self, Q: Union[float, List[float]]) -> pd.DataFrame:
        """
        Generate an ADM1da influent DataFrame for the full simulation period.

        The substrate concentrations are constant over time (steady-state
        feed composition assumption).  Pass the result to
        ``ADM1da.set_influent_dataframe()``.

        Parameters
        ----------
        Q : float or List[float]
            Total volumetric substrate flow rate(s) [m³/d].

        Returns
        -------
        pd.DataFrame
            Columns match ``INFLUENT_COLUMNS`` (37 state variables + Q).
        """
        q_total = float(np.sum(Q))
        row = dict(self._conc)
        row["Q"] = q_total

        df = pd.DataFrame([row] * len(self._simtime))
        return df[INFLUENT_COLUMNS]

    @property
    def substrate(self) -> ADM1daSubstrateParams:
        """The substrate characterization data."""
        return self._sub

    @property
    def density(self) -> float:
        """Estimated fresh-matter density [kg/m³]."""
        return self._density

    @property
    def concentrations(self) -> dict:
        """ADM1da influent concentrations [kg COD/m³ or kmol/m³]."""
        return dict(self._conc)

    def vs_content(self) -> float:
        """Volatile-solids content of the substrate [kg VS/m³]."""
        s = self._sub
        fTS = s.TS / 1000.0
        return fTS * (1.0 - s.fRA) * self._density

    def total_cod(self) -> float:
        """Total COD concentration (particulate + dissolved) [kg COD/m³]."""
        c = self._conc
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

    def bmp_theoretical(self) -> float:
        """
        Theoretical biomethane potential from COD stoichiometry [Nm³ CH₄/t VS].

        Assumes 100 % conversion of degradable COD.  The given BMP is lower
        due to incomplete biodegradation and kinetic limitations.
        """
        c = self._conc
        s = self._sub
        th_yield = s.V_m / (s.CH4_cod_2_mol / 1000.0)  # Nm³ CH4 / kg COD
        degradable_cod = c["X_PS_ch"] + c["X_PS_pr"] + c["X_PS_li"] + c["X_PF_ch"] + c["X_PF_pr"] + c["X_PF_li"] + c["S_ac"]
        vs = self.vs_content()
        if vs <= 0.0:
            return 0.0
        return degradable_cod * th_yield / vs * 1000.0  # Nm³ CH4 / t VS

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _calc_density(self) -> float:
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
        s = self._sub

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

    def _calc_concentrations(self) -> dict:
        """
        Compute all ADM1da influent concentrations per m³ of fresh substrate.

        Returns
        -------
        dict
            Keys match INFLUENT_COLUMNS (without Q).
            Organic fractions in kg COD/m³; ionic species in kmol/m³.
        """
        s = self._sub
        rho = self._density
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
