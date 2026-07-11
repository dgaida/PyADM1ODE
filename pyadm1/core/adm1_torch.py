# pyadm1/core/adm1_torch.py
"""Differentiable PyTorch backend for the ADM1da right-hand side.

This module mirrors :meth:`pyadm1.core.adm1.ADM1.ADM_ODE` operation for
operation, but in pure PyTorch so the state derivatives ``dx/dt = f(x, u)``
are autograd-differentiable with respect to the state ``x``. It is used

* as an optional simulation backend (selected via ``backend="torch"`` on the
  digester / plant; integrated by the same scipy solver as the numpy path), and
* wherever gradients of the dynamics w.r.t. the state are needed e.g.
  gradient-based state estimation, sensitivity analysis or optimisation.

Design notes
------------
* The kinetic / stoichiometric / inhibition constants are scalars and stay
  plain Python floats. Autograd only needs to flow through the state ``x``, so
  keeping the parameters as floats (rather than tensors) is both simpler and
  faster while preserving differentiability w.r.t. ``x``.
* The charge-balance pH solve of the numpy model is a scalar Newton iteration
  on ``fixed + S_H - K_w / S_H = 0``. That is the positive root of the
  quadratic ``S_H^2 + fixed * S_H - K_w = 0``, so here it is evaluated in
  closed form - no iteration, fully differentiable.
* ``max(...)`` guards become :func:`torch.clamp`; the ``1e-20`` / ``1e-30``
  epsilons of the numpy model are kept verbatim to reproduce its values and to
  avoid division-by-zero NaN gradients.
* The right-hand side is batch-agnostic: ``x`` may be shaped ``(41,)`` or
  ``(B, 41)``; indexing via ``x[..., i]`` and a final ``stack(dim=-1)`` handle
  both. The parity is defined at ``float64``.

Reference: identical formulation to :meth:`ADM1.ADM_ODE`; see
:mod:`pyadm1.core.adm1` for the state-vector index map and units.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import numpy as np
import torch

from pyadm1.core.adm1 import (
    _IDX_S_SU,
    _IDX_S_AA,
    _IDX_S_FA,
    _IDX_S_VA,
    _IDX_S_BU,
    _IDX_S_PRO,
    _IDX_S_AC,
    _IDX_S_H2,
    _IDX_S_CH4,
    _IDX_S_CO2,
    _IDX_S_NH4,
    _IDX_S_I,
    _IDX_X_PS_CH,
    _IDX_X_PS_PR,
    _IDX_X_PS_LI,
    _IDX_X_PF_CH,
    _IDX_X_PF_PR,
    _IDX_X_PF_LI,
    _IDX_X_S_CH,
    _IDX_X_S_PR,
    _IDX_X_S_LI,
    _IDX_X_I,
    _IDX_X_SU,
    _IDX_X_AA,
    _IDX_X_FA,
    _IDX_X_C4,
    _IDX_X_PRO,
    _IDX_X_AC,
    _IDX_X_H2,
    _IDX_S_CATION,
    _IDX_S_ANION,
    _IDX_S_VA_ION,
    _IDX_S_BU_ION,
    _IDX_S_PRO_ION,
    _IDX_S_AC_ION,
    _IDX_S_HCO3,
    _IDX_S_NH3,
    _IDX_P_H2,
    _IDX_P_CH4,
    _IDX_P_CO2,
    _IDX_P_TOTAL,
)

if TYPE_CHECKING:
    from pyadm1.core.adm1 import ADM1

#: Number of soluble/liquid influent components (indices 0-36).
_N_INFLUENT = 37


@dataclass
class Adm1TorchParams:
    """Snapshot of the ADM1 parameters the right-hand side reads from ``self``.

    Mirrors exactly what :meth:`ADM1.ADM_ODE` consumes: the four constant
    dictionaries, the resolved gas/transport scalars (with any calibration
    overrides already applied, as the numpy model does at call time), the
    influent composition and flow, and the optional outflow override.

    Build one with :meth:`from_adm1` at the start of an integration step; the
    values are constant within a scipy step, exactly as the numpy model treats
    them.
    """

    kinetic: Dict[str, float]
    stoich: Dict[str, float]
    fractions: Dict[str, float]
    inhib: Dict[str, float]

    V_liq: float
    V_gas: float
    RT: float
    K_H_h2: float
    K_H_ch4: float
    K_H_co2: float
    k_L_a: float
    k_p: float
    p_gas_h2o: float
    p_ext: float
    NQ: float

    q_ad: float
    s_in: List[float] = field(default_factory=lambda: [0.0] * _N_INFLUENT)
    Q_out_override: Optional[float] = None

    def with_q_ad(self, q_ad: float) -> "Adm1TorchParams":
        """Return a copy with the total influent flow ``q_ad`` replaced.

        Makes the physics feed-aware for a different operating point: the ODE
        influent term is ``D_in·s_in`` with ``D_in = q_ad/V_liq``, so scaling
        ``q_ad`` to the actual live feed corrects the physics without changing
        the influent *composition* ``s_in`` (exact for a proportional feed
        change, the same mix at a higher/lower total rate).
        """
        import dataclasses

        return dataclasses.replace(self, q_ad=float(q_ad))

    @classmethod
    def from_adm1(cls, adm1: "ADM1") -> "Adm1TorchParams":
        """Snapshot the current parameter state of an :class:`ADM1` instance.

        Resolves the ``k_L_a`` / ``k_p`` calibration overrides the same way
        :meth:`ADM1.ADM_ODE` does, and reads back the (possibly calibrated)
        Henry constants stored on the instance.
        """
        cal = adm1._calibration_params

        k_L_a = adm1._k_L_a
        if cal.get("k_L_a") is not None:
            k_L_a = float(cal["k_L_a"])
        k_p = adm1._k_p
        if cal.get("k_p") is not None:
            k_p = float(cal["k_p"])

        q_ad = float(sum(adm1._Q)) if adm1._Q is not None else 0.0
        s_in = list(adm1._state_input) if adm1._state_input is not None else [0.0] * _N_INFLUENT

        return cls(
            kinetic=dict(adm1._kinetic),
            stoich=dict(adm1._stoich),
            fractions=dict(adm1._fractions),
            inhib=dict(adm1._inhib_params),
            V_liq=float(adm1.V_liq),
            V_gas=float(adm1._V_gas),
            RT=float(adm1._RT),
            K_H_h2=float(adm1._K_H_h2),
            K_H_ch4=float(adm1._K_H_ch4),
            K_H_co2=float(adm1._K_H_co2),
            k_L_a=float(k_L_a),
            k_p=float(k_p),
            p_gas_h2o=float(adm1._p_gas_h2o),
            p_ext=float(adm1._p_ext),
            NQ=float(adm1._NQ),
            q_ad=q_ad,
            s_in=s_in,
            Q_out_override=(None if adm1._Q_out_override is None else float(adm1._Q_out_override)),
        )


def _ph_inhib(S_H: torch.Tensor, K_pH: float, n: int) -> torch.Tensor:
    """Hill-type pH inhibition ``K_pH^n / (K_pH^n + S_H^n)`` (see ADM1._pH_inhib)."""
    Kn = K_pH**n
    SHn = S_H**n
    return Kn / (Kn + SHn)


def _calc_ph(
    S_nh4: torch.Tensor,
    S_nh3: torch.Tensor,
    S_hco3: torch.Tensor,
    S_ac_ion: torch.Tensor,
    S_pro_ion: torch.Tensor,
    S_bu_ion: torch.Tensor,
    S_va_ion: torch.Tensor,
    S_cation: torch.Tensor,
    S_anion: torch.Tensor,
    K_w: float,
) -> torch.Tensor:
    """Closed-form charge-balance solve for [H+].

    Positive root of ``S_H^2 + fixed * S_H - K_w = 0``, which is the fixed
    point the numpy Newton iteration in :meth:`ADM1._calc_ph` converges to.
    """
    vfa_anions = S_ac_ion / 64.0 + S_pro_ion / 112.0 + S_bu_ion / 160.0 + S_va_ion / 208.0
    fixed = S_cation - S_anion + (S_nh4 - S_nh3) - S_hco3 - vfa_anions
    S_H = 0.5 * (-fixed + torch.sqrt(fixed * fixed + 4.0 * K_w))
    return torch.clamp(S_H, min=1.0e-14)


def adm1da_rhs_torch(x: torch.Tensor, p: Adm1TorchParams) -> torch.Tensor:
    """Compute ``dx/dt`` for the 41-state ADM1da vector in PyTorch.

    Args:
        x: State tensor shaped ``(41,)`` or ``(B, 41)``.
        p: Parameter snapshot (see :class:`Adm1TorchParams`).

    Returns:
        Tensor of the same shape as ``x`` holding the state derivatives.
    """
    # ---- Unpack state ------------------------------------------------
    S_su = x[..., _IDX_S_SU]
    S_aa = x[..., _IDX_S_AA]
    S_fa = x[..., _IDX_S_FA]
    S_va = x[..., _IDX_S_VA]
    S_bu = x[..., _IDX_S_BU]
    S_pro = x[..., _IDX_S_PRO]
    S_ac = x[..., _IDX_S_AC]
    S_h2 = x[..., _IDX_S_H2]
    S_ch4 = x[..., _IDX_S_CH4]
    S_co2 = x[..., _IDX_S_CO2]
    S_nh4 = x[..., _IDX_S_NH4]
    S_I = x[..., _IDX_S_I]

    X_PS_ch = x[..., _IDX_X_PS_CH]
    X_PS_pr = x[..., _IDX_X_PS_PR]
    X_PS_li = x[..., _IDX_X_PS_LI]
    X_PF_ch = x[..., _IDX_X_PF_CH]
    X_PF_pr = x[..., _IDX_X_PF_PR]
    X_PF_li = x[..., _IDX_X_PF_LI]
    X_S_ch = x[..., _IDX_X_S_CH]
    X_S_pr = x[..., _IDX_X_S_PR]
    X_S_li = x[..., _IDX_X_S_LI]
    X_I = x[..., _IDX_X_I]

    X_su = x[..., _IDX_X_SU]
    X_aa = x[..., _IDX_X_AA]
    X_fa = x[..., _IDX_X_FA]
    X_c4 = x[..., _IDX_X_C4]
    X_pro = x[..., _IDX_X_PRO]
    X_ac = x[..., _IDX_X_AC]
    X_h2 = x[..., _IDX_X_H2]

    S_cation = x[..., _IDX_S_CATION]
    S_anion = x[..., _IDX_S_ANION]
    S_va_ion = x[..., _IDX_S_VA_ION]
    S_bu_ion = x[..., _IDX_S_BU_ION]
    S_pro_ion = x[..., _IDX_S_PRO_ION]
    S_ac_ion = x[..., _IDX_S_AC_ION]
    S_hco3 = x[..., _IDX_S_HCO3]
    S_nh3 = x[..., _IDX_S_NH3]

    p_gas_h2 = x[..., _IDX_P_H2]
    p_gas_ch4 = x[..., _IDX_P_CH4]
    p_gas_co2 = x[..., _IDX_P_CO2]
    pTOTAL = x[..., _IDX_P_TOTAL]

    # ---- Influent and hydraulic flow ---------------------------------
    q_ad = p.q_ad
    s_in = p.s_in

    # ---- Parameters --------------------------------------------------
    k = p.kinetic
    st = p.stoich
    fr = p.fractions
    ip = p.inhib

    f_fa_li = st["f_fa_li"]
    f_ch_bac = st["f_ch_bac"]
    f_pr_bac = st["f_pr_bac"]
    f_li_bac = st["f_li_bac"]
    f_p_bac = st["f_p_bac"]
    fXI_PS = st["fXI_PS"]
    fXI_PF = st["fXI_PF"]
    fSI = st["fSI_hyd"]

    Y_su = k["Y_su"]
    Y_aa = k["Y_aa"]
    Y_fa = k["Y_fa"]
    Y_c4 = k["Y_c4"]
    Y_pro = k["Y_pro"]
    Y_ac = k["Y_ac"]
    Y_h2 = k["Y_h2"]

    # ---- pH (closed-form charge balance) -----------------------------
    S_H = _calc_ph(
        S_nh4,
        S_nh3,
        S_hco3,
        S_ac_ion,
        S_pro_ion,
        S_bu_ion,
        S_va_ion,
        S_cation,
        S_anion,
        ip["K_w"],
    )

    # ---- Inhibition factors ------------------------------------------
    S_IN = S_nh4 + S_nh3
    I_IN = S_IN / (ip["K_S_IN"] + S_IN + 1.0e-20)

    I_pH_aa = _ph_inhib(S_H, ip["K_pH_aa"], n=1)
    I_pH_fa = _ph_inhib(S_H, ip["K_pH_aa"], n=2)
    I_pH_c4 = _ph_inhib(S_H, ip["K_pH_aa"], n=2)
    I_pH_pro = _ph_inhib(S_H, ip["K_pH_aa"], n=2)
    I_pH_ac = _ph_inhib(S_H, ip["K_pH_ac"], n=3)
    I_pH_h2 = _ph_inhib(S_H, ip["K_pH_h2"], n=3)

    I_h2_fa = ip["K_I_h2_fa"] / (ip["K_I_h2_fa"] + S_h2)
    I_h2_c4 = ip["K_I_h2_c4"] / (ip["K_I_h2_c4"] + S_h2)
    I_h2_pro = ip["K_I_h2_pro"] / (ip["K_I_h2_pro"] + S_h2)

    K_nh3_ac_sq = ip["K_I_nh3"] ** 2
    K_nh3_pro_sq = ip["K_I_nh3_pro"] ** 2
    S_nh3_sq = S_nh3 * S_nh3
    I_nh3 = K_nh3_ac_sq / (K_nh3_ac_sq + S_nh3_sq)
    I_nh3_pro = K_nh3_pro_sq / (K_nh3_pro_sq + S_nh3_sq)

    K_co2_h2_sq = ip["K_S_co2_h2"] * ip["K_S_co2_h2"]
    S_co2_sq = S_co2 * S_co2
    I_co2_h2 = S_co2_sq / (K_co2_h2_sq + S_co2_sq + 1.0e-30)

    Ka_pro = ip["K_a_pro"]
    S_HPr = (S_pro / 112.0) * S_H / (S_H + Ka_pro + 1.0e-20)
    I_HPr = ip["K_IH_pro"] / (ip["K_IH_pro"] + S_HPr + 1.0e-20)

    Ka_ac = ip["K_a_ac"]
    S_HAc = (S_ac / 64.0) * S_H / (S_H + Ka_ac + 1.0e-20)
    I_HAc = ip["K_IH_ac"] / (ip["K_IH_ac"] + S_HAc + 1.0e-20)

    I_su = I_pH_aa * I_IN
    I_aa = I_pH_aa * I_IN
    I_fa = I_pH_fa * I_IN * I_h2_fa
    I_c4 = I_pH_c4 * I_IN * I_h2_c4
    I_pro = I_pH_pro * I_IN * I_h2_pro * I_HPr * I_nh3_pro
    I_ac = I_pH_ac * I_IN * I_nh3 * I_HAc
    I_h2 = I_pH_h2 * I_IN * I_co2_h2

    # ---- Process rates -----------------------------------------------
    Rho_dis_PS_ch = k["k_dis_PS"] * X_PS_ch
    Rho_dis_PS_pr = k["k_dis_PS"] * X_PS_pr
    Rho_dis_PS_li = k["k_dis_PS"] * X_PS_li
    Rho_dis_PF_ch = k["k_dis_PF"] * X_PF_ch
    Rho_dis_PF_pr = k["k_dis_PF"] * X_PF_pr
    Rho_dis_PF_li = k["k_dis_PF"] * X_PF_li

    Rho_hyd_ch = k["k_hyd_ch"] * X_S_ch
    Rho_hyd_pr = k["k_hyd_pr"] * X_S_pr
    Rho_hyd_li = k["k_hyd_li"] * X_S_li

    Rho_su = k["k_m_su"] * S_su / (k["K_S_su"] + S_su + 1.0e-20) * X_su * I_su
    Rho_aa = k["k_m_aa"] * S_aa / (k["K_S_aa"] + S_aa + 1.0e-20) * X_aa * I_aa
    Rho_fa = k["k_m_fa"] * S_fa / (k["K_S_fa"] + S_fa + 1.0e-20) * X_fa * I_fa

    S_vbu = S_va + S_bu + 1.0e-20
    Rho_c4_va = k["k_m_c4"] * S_va / (k["K_S_c4"] + S_va + 1.0e-20) * X_c4 * (S_va / S_vbu) * I_c4
    Rho_c4_bu = k["k_m_c4"] * S_bu / (k["K_S_c4"] + S_bu + 1.0e-20) * X_c4 * (S_bu / S_vbu) * I_c4

    Rho_pro = k["k_m_pro"] * S_pro / (k["K_S_pro"] + S_pro + 1.0e-20) * X_pro * I_pro
    Rho_ac = k["k_m_ac"] * S_ac / (k["K_S_ac"] + S_ac + 1.0e-20) * X_ac * I_ac
    Rho_h2 = k["k_m_h2"] * S_h2 / (k["K_S_h2"] + S_h2 + 1.0e-20) * X_h2 * I_h2

    Rho_dec_su = k["k_dec_su"] * X_su
    Rho_dec_aa = k["k_dec_aa"] * X_aa
    Rho_dec_fa = k["k_dec_fa"] * X_fa
    Rho_dec_c4 = k["k_dec_c4"] * X_c4
    Rho_dec_pro = k["k_dec_pro"] * X_pro
    Rho_dec_ac = k["k_dec_ac"] * X_ac
    Rho_dec_h2 = k["k_dec_h2"] * X_h2
    sum_decay = Rho_dec_su + Rho_dec_aa + Rho_dec_fa + Rho_dec_c4 + Rho_dec_pro + Rho_dec_ac + Rho_dec_h2

    # ---- Acid-base rates --------------------------------------------
    k_AB = ip["k_A_B"]
    Rho_A_va = k_AB * (S_va_ion * S_H - ip["K_a_va"] * (S_va - S_va_ion))
    Rho_A_bu = k_AB * (S_bu_ion * S_H - ip["K_a_bu"] * (S_bu - S_bu_ion))
    Rho_A_pro = k_AB * (S_pro_ion * S_H - ip["K_a_pro"] * (S_pro - S_pro_ion))
    Rho_A_ac = k_AB * (S_ac_ion * S_H - ip["K_a_ac"] * (S_ac - S_ac_ion))
    Rho_A_co2 = k_AB * (S_hco3 * S_H - ip["K_a_co2"] * (S_co2 - S_hco3))
    Rho_A_IN = k_AB * (S_nh3 * S_H - ip["K_a_IN"] * (S_nh4 - S_nh3))

    # ---- Gas transfer rates -----------------------------------------
    k_L_a = p.k_L_a
    Rho_T_h2 = k_L_a * (S_h2 - 16.0 * p_gas_h2 / (p.RT * p.K_H_h2)) * (p.V_liq / p.V_gas)
    Rho_T_ch4 = k_L_a * (S_ch4 - 64.0 * p_gas_ch4 / (p.RT * p.K_H_ch4)) * (p.V_liq / p.V_gas)
    S_co2_free = torch.clamp(S_co2 - S_hco3, min=0.0)
    Rho_T_co2 = k_L_a * (S_co2_free - p_gas_co2 / (p.RT * p.K_H_co2)) * (p.V_liq / p.V_gas)

    k_p = p.k_p
    Rho_T_11 = torch.clamp(
        k_p * (pTOTAL + p.p_gas_h2o - p.p_ext) * (p.V_liq / p.V_gas),
        min=0.0,
    )

    # ---- Carbon stoichiometry coefficients for S_co2 balance --------
    C = st
    s_hyd_ch = -C["C_ch"] + (1.0 - fSI) * C["C_su"] + fSI * C["C_I_s"]
    s_hyd_pr = -C["C_pr"] + (1.0 - fSI) * C["C_aa"] + fSI * C["C_I_s"]
    s_hyd_li = -C["C_li"] + (1.0 - fSI) * (f_fa_li * C["C_fa"] + (1.0 - f_fa_li) * C["C_su"]) + fSI * C["C_I_s"]

    s_su = (
        -C["C_su"]
        + (1.0 - Y_su) * (fr["f_bu_su"] * C["C_bu"] + fr["f_pro_su"] * C["C_pro"] + fr["f_ac_su"] * C["C_ac"])
        + Y_su * C["C_bac"]
    )
    s_aa = (
        -C["C_aa"]
        + (1.0 - Y_aa)
        * (fr["f_va_aa"] * C["C_va"] + fr["f_bu_aa"] * C["C_bu"] + fr["f_pro_aa"] * C["C_pro"] + fr["f_ac_aa"] * C["C_ac"])
        + Y_aa * C["C_bac"]
    )
    s_fa = -C["C_fa"] + (1.0 - Y_fa) * 0.7 * C["C_ac"] + Y_fa * C["C_bac"]
    s_c4_va = -C["C_va"] + (1.0 - Y_c4) * (0.54 * C["C_pro"] + 0.31 * C["C_ac"]) + Y_c4 * C["C_bac"]
    s_c4_bu = -C["C_bu"] + (1.0 - Y_c4) * 0.8 * C["C_ac"] + Y_c4 * C["C_bac"]
    s_pro = -C["C_pro"] + (1.0 - Y_pro) * 0.57 * C["C_ac"] + Y_pro * C["C_bac"]
    s_ac = -C["C_ac"] + (1.0 - Y_ac) * C["C_ch4"] + Y_ac * C["C_bac"]
    s_h2 = (1.0 - Y_h2) * C["C_ch4"] + Y_h2 * C["C_bac"]
    s_dec = -C["C_bac"] + f_ch_bac * C["C_ch"] + f_pr_bac * C["C_pr"] + f_li_bac * C["C_li"] + f_p_bac * C["C_I_x"]

    Sigma = (
        s_hyd_ch * Rho_hyd_ch
        + s_hyd_pr * Rho_hyd_pr
        + s_hyd_li * Rho_hyd_li
        + s_su * Rho_su
        + s_aa * Rho_aa
        + s_fa * Rho_fa
        + s_c4_va * Rho_c4_va
        + s_c4_bu * Rho_c4_bu
        + s_pro * Rho_pro
        + s_ac * Rho_ac
        + s_h2 * Rho_h2
        + s_dec * sum_decay
    )

    # ---- Differential equations -------------------------------------
    D_in = q_ad / p.V_liq

    # Sludge volume loss (hydrolysis-rate volume balance, after Schlattmann 2011).
    _q_S_loss = p.V_liq * (Rho_hyd_ch * (0.9375 / 1550.0) + Rho_hyd_pr * (0.6125 / 1370.0) + Rho_hyd_li * (0.3474 / 920.0))

    if p.Q_out_override is not None:
        _Q_out = max(float(p.Q_out_override), 0.0)
        D_out = _Q_out / p.V_liq
    else:
        _Q_out = torch.clamp(q_ad - _q_S_loss, min=0.0)
        D_out = _Q_out / p.V_liq

    # --- Dissolved (0-11) ---
    diff_S_su = D_in * s_in[0] - D_out * S_su + (1.0 - fSI) * Rho_hyd_ch + (1.0 - fSI) * (1.0 - f_fa_li) * Rho_hyd_li - Rho_su
    diff_S_aa = D_in * s_in[1] - D_out * S_aa + (1.0 - fSI) * Rho_hyd_pr - Rho_aa
    diff_S_fa = D_in * s_in[2] - D_out * S_fa + (1.0 - fSI) * f_fa_li * Rho_hyd_li - Rho_fa
    diff_S_va = D_in * s_in[3] - D_out * S_va + (1.0 - Y_aa) * fr["f_va_aa"] * Rho_aa - Rho_c4_va
    diff_S_bu = (
        D_in * s_in[4]
        - D_out * S_bu
        + (1.0 - Y_su) * fr["f_bu_su"] * Rho_su
        + (1.0 - Y_aa) * fr["f_bu_aa"] * Rho_aa
        - Rho_c4_bu
    )
    diff_S_pro = (
        D_in * s_in[5]
        - D_out * S_pro
        + (1.0 - Y_su) * fr["f_pro_su"] * Rho_su
        + (1.0 - Y_aa) * fr["f_pro_aa"] * Rho_aa
        + (1.0 - Y_c4) * 0.54 * Rho_c4_va
        - Rho_pro
    )
    diff_S_ac = (
        D_in * s_in[6]
        - D_out * S_ac
        + (1.0 - Y_su) * fr["f_ac_su"] * Rho_su
        + (1.0 - Y_aa) * fr["f_ac_aa"] * Rho_aa
        + (1.0 - Y_fa) * 0.7 * Rho_fa
        + (1.0 - Y_c4) * 0.31 * Rho_c4_va
        + (1.0 - Y_c4) * 0.8 * Rho_c4_bu
        + (1.0 - Y_pro) * 0.57 * Rho_pro
        - Rho_ac
    )
    diff_S_h2 = (
        D_in * s_in[7]
        - D_out * S_h2
        + (1.0 - Y_su) * fr["f_h2_su"] * Rho_su
        + (1.0 - Y_aa) * fr["f_h2_aa"] * Rho_aa
        + (1.0 - Y_fa) * 0.3 * Rho_fa
        + (1.0 - Y_c4) * 0.15 * Rho_c4_va
        + (1.0 - Y_c4) * 0.2 * Rho_c4_bu
        + (1.0 - Y_pro) * 0.43 * Rho_pro
        - Rho_h2
        - p.V_gas / p.V_liq * Rho_T_h2
    )
    diff_S_ch4 = D_in * s_in[8] - D_out * S_ch4 + (1.0 - Y_ac) * Rho_ac + (1.0 - Y_h2) * Rho_h2 - p.V_gas / p.V_liq * Rho_T_ch4
    diff_S_co2 = D_in * s_in[9] - D_out * S_co2 - Sigma - p.V_gas / p.V_liq * Rho_T_co2 + Rho_A_co2

    N_bac = st["N_bac"]
    N_aa = st["N_aa"]
    N_I = st["N_I"]
    diff_S_nh4 = (
        D_in * s_in[10]
        - D_out * S_nh4
        - Y_su * N_bac * Rho_su
        + (N_aa - Y_aa * N_bac) * Rho_aa
        - Y_fa * N_bac * Rho_fa
        - Y_c4 * N_bac * Rho_c4_va
        - Y_c4 * N_bac * Rho_c4_bu
        - Y_pro * N_bac * Rho_pro
        - Y_ac * N_bac * Rho_ac
        - Y_h2 * N_bac * Rho_h2
        + (N_bac - f_pr_bac * N_aa - f_p_bac * N_I) * sum_decay
        + Rho_A_IN
    )
    diff_S_I = D_in * s_in[11] - D_out * S_I + fSI * (Rho_hyd_ch + Rho_hyd_pr + Rho_hyd_li)

    # --- Particulate sub-fractions (12-21) ---
    diff_X_PS_ch = D_in * s_in[12] - D_out * X_PS_ch - Rho_dis_PS_ch
    diff_X_PS_pr = D_in * s_in[13] - D_out * X_PS_pr - Rho_dis_PS_pr
    diff_X_PS_li = D_in * s_in[14] - D_out * X_PS_li - Rho_dis_PS_li
    diff_X_PF_ch = D_in * s_in[15] - D_out * X_PF_ch - Rho_dis_PF_ch
    diff_X_PF_pr = D_in * s_in[16] - D_out * X_PF_pr - Rho_dis_PF_pr
    diff_X_PF_li = D_in * s_in[17] - D_out * X_PF_li - Rho_dis_PF_li
    diff_X_S_ch = (
        D_in * s_in[18]
        - D_out * X_S_ch
        + (1.0 - fXI_PS) * Rho_dis_PS_ch
        + (1.0 - fXI_PF) * Rho_dis_PF_ch
        + f_ch_bac * sum_decay
        - Rho_hyd_ch
    )
    diff_X_S_pr = (
        D_in * s_in[19]
        - D_out * X_S_pr
        + (1.0 - fXI_PS) * Rho_dis_PS_pr
        + (1.0 - fXI_PF) * Rho_dis_PF_pr
        + f_pr_bac * sum_decay
        - Rho_hyd_pr
    )
    diff_X_S_li = (
        D_in * s_in[20]
        - D_out * X_S_li
        + (1.0 - fXI_PS) * Rho_dis_PS_li
        + (1.0 - fXI_PF) * Rho_dis_PF_li
        + f_li_bac * sum_decay
        - Rho_hyd_li
    )
    diff_X_I = (
        D_in * s_in[21]
        - D_out * X_I
        + fXI_PS * (Rho_dis_PS_ch + Rho_dis_PS_pr + Rho_dis_PS_li)
        + fXI_PF * (Rho_dis_PF_ch + Rho_dis_PF_pr + Rho_dis_PF_li)
        + f_p_bac * sum_decay
    )

    # --- Biomass (22-28) ---
    diff_X_su = D_in * s_in[22] - D_out * X_su + Y_su * Rho_su - Rho_dec_su
    diff_X_aa = D_in * s_in[23] - D_out * X_aa + Y_aa * Rho_aa - Rho_dec_aa
    diff_X_fa = D_in * s_in[24] - D_out * X_fa + Y_fa * Rho_fa - Rho_dec_fa
    diff_X_c4 = D_in * s_in[25] - D_out * X_c4 + Y_c4 * Rho_c4_va + Y_c4 * Rho_c4_bu - Rho_dec_c4
    diff_X_pro = D_in * s_in[26] - D_out * X_pro + Y_pro * Rho_pro - Rho_dec_pro
    diff_X_ac = D_in * s_in[27] - D_out * X_ac + Y_ac * Rho_ac - Rho_dec_ac
    diff_X_h2 = D_in * s_in[28] - D_out * X_h2 + Y_h2 * Rho_h2 - Rho_dec_h2

    # --- Charge balance (29-36) ---
    diff_S_cation = D_in * s_in[29] - D_out * S_cation
    diff_S_anion = D_in * s_in[30] - D_out * S_anion
    diff_S_va_ion = D_in * s_in[31] - D_out * S_va_ion - Rho_A_va
    diff_S_bu_ion = D_in * s_in[32] - D_out * S_bu_ion - Rho_A_bu
    diff_S_pro_ion = D_in * s_in[33] - D_out * S_pro_ion - Rho_A_pro
    diff_S_ac_ion = D_in * s_in[34] - D_out * S_ac_ion - Rho_A_ac
    diff_S_hco3 = D_in * s_in[35] - D_out * S_hco3 - Rho_A_co2
    diff_S_nh3 = D_in * s_in[36] - D_out * S_nh3 - Rho_A_IN

    # --- Gas phase (37-40) ---
    diff_p_h2 = Rho_T_h2 * p.RT / 16.0 - p_gas_h2 / pTOTAL * Rho_T_11
    diff_p_ch4 = Rho_T_ch4 * p.RT / 64.0 - p_gas_ch4 / pTOTAL * Rho_T_11
    diff_p_co2 = Rho_T_co2 * p.RT - p_gas_co2 / pTOTAL * Rho_T_11
    diff_pTOT = p.RT / 16.0 * Rho_T_h2 + p.RT / 64.0 * Rho_T_ch4 + p.RT * Rho_T_co2 - Rho_T_11

    return torch.stack(
        [
            diff_S_su,
            diff_S_aa,
            diff_S_fa,
            diff_S_va,
            diff_S_bu,
            diff_S_pro,
            diff_S_ac,
            diff_S_h2,
            diff_S_ch4,
            diff_S_co2,
            diff_S_nh4,
            diff_S_I,
            diff_X_PS_ch,
            diff_X_PS_pr,
            diff_X_PS_li,
            diff_X_PF_ch,
            diff_X_PF_pr,
            diff_X_PF_li,
            diff_X_S_ch,
            diff_X_S_pr,
            diff_X_S_li,
            diff_X_I,
            diff_X_su,
            diff_X_aa,
            diff_X_fa,
            diff_X_c4,
            diff_X_pro,
            diff_X_ac,
            diff_X_h2,
            diff_S_cation,
            diff_S_anion,
            diff_S_va_ion,
            diff_S_bu_ion,
            diff_S_pro_ion,
            diff_S_ac_ion,
            diff_S_hco3,
            diff_S_nh3,
            diff_p_h2,
            diff_p_ch4,
            diff_p_co2,
            diff_pTOT,
        ],
        dim=-1,
    )


def make_scipy_rhs(adm1: "ADM1") -> Callable[[float, np.ndarray], np.ndarray]:
    """Return a scipy ``solve_ivp``-compatible ``fun(t, y)`` using the torch RHS.

    The parameters are snapshotted once (constant within a scipy step), so
    build a fresh callable at the start of each integration step. To stay a
    drop-in for the numpy path, the adapter also updates
    ``adm1._q_S_loss_last`` on every evaluation, the sludge-volume side effect
    that :meth:`ADM1.ADM_ODE` writes for the external dynamic-volume balance.
    """
    params = Adm1TorchParams.from_adm1(adm1)
    k_hyd_ch = params.kinetic["k_hyd_ch"]
    k_hyd_pr = params.kinetic["k_hyd_pr"]
    k_hyd_li = params.kinetic["k_hyd_li"]
    V_liq = params.V_liq

    def fun(t: float, y: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(np.asarray(y, dtype=np.float64), dtype=torch.float64)
        dxdt = adm1da_rhs_torch(x, params)
        # Mirror ADM_ODE's cached sludge-volume loss (Rho_hyd = k_hyd * X_S).
        adm1._q_S_loss_last = float(
            V_liq
            * (
                k_hyd_ch * y[_IDX_X_S_CH] * (0.9375 / 1550.0)
                + k_hyd_pr * y[_IDX_X_S_PR] * (0.6125 / 1370.0)
                + k_hyd_li * y[_IDX_X_S_LI] * (0.3474 / 920.0)
            )
        )
        return dxdt.detach().numpy()

    return fun


_GAS_LEAK_SLOPE = 1.0e-2


def _leaky_floor(z: torch.Tensor) -> torch.Tensor:
    """Identity for ``z >= 0`` (so it matches the hard ``clamp(min=0)`` in the
    physical region), a small negative slope for ``z < 0``."""
    return torch.where(z > 0.0, z, _GAS_LEAK_SLOPE * z)


def calc_gas_torch(x: torch.Tensor, p: Adm1TorchParams, soft: bool = False):
    """Biogas / methane / CO2 volumetric flows [m^3/d] from the gas-phase state.

    Mirrors :meth:`ADM1.calc_gas`. Returns ``(q_gas, q_ch4, q_co2)``, each the
    same batch shape as ``x[..., 0]``.

    Args:
        soft: When ``False`` (default) the flows are floored with a hard
            ``clamp(min=0)`` - bit-matching the numpy model. When ``True`` the
            floor is *leaky* (small non-zero slope below zero), removing the
            zero-gradient dead-zone at ``q_gas = 0``. Use ``True`` only for
            gradient-based use; the values are identical wherever the gas
            actually flows (``q_gas >= 0``).
    """
    p_h2 = x[..., _IDX_P_H2]
    p_ch4 = x[..., _IDX_P_CH4]
    p_co2 = x[..., _IDX_P_CO2]
    pTOTAL = x[..., _IDX_P_TOTAL]

    p_total_wet = pTOTAL + p.p_gas_h2o
    q_gas_raw = p.k_p * (p_total_wet - p.p_ext) / (p.RT / 1000.0 * p.NQ) * p.V_liq

    p_gas = p_h2 + p_ch4 + p_co2
    p_gas_wet = p_gas + p.p_gas_h2o
    safe = torch.clamp(p_gas_wet, min=1.0e-30)

    if soft:
        q_gas = _leaky_floor(q_gas_raw)
        q_ch4 = _leaky_floor(q_gas * (p_ch4 / safe))
        q_co2 = _leaky_floor(q_gas * (p_co2 / safe))
    else:
        q_gas = torch.clamp(q_gas_raw, min=0.0)
        zeros = torch.zeros_like(q_gas)
        pos = p_gas_wet > 0.0
        q_ch4 = torch.where(pos, torch.clamp(q_gas * (p_ch4 / safe), min=0.0), zeros)
        q_co2 = torch.where(pos, torch.clamp(q_gas * (p_co2 / safe), min=0.0), zeros)
    return q_gas, q_ch4, q_co2


def _solve_gas_phase_torch(x: torch.Tensor, p: Adm1TorchParams, n_iter: int = 25):
    """Quasi-steady gas phase: solve the 4 gas ODEs = 0 for the pressures + flows.

    The gas phase relaxes far faster than the liquid dynamics, so it is
    algebraically slaved to the liquid state. Setting ``diff_p_{h2,ch4,co2}`` to
    zero gives, per species, ``c_i·Rho_T_i = (p_i/pTOTAL)·Rho_T_11``. Writing
    ``Rho_T_i = k_La·V·(S_i - a_i·p_i)`` and ``B := Rho_T_11/pTOTAL`` closes each
    pressure explicitly, ``p_i = c_i·k_La·V·S_i / (c_i·k_La·V·a_i + B)``, and the
    single scalar ``B`` is found from ``Σ p_i = pTOTAL`` by a few Newton steps.
    ``pTOTAL`` is pinned near ``p_ext - p_h2o`` (the outlet balance), so it is no
    longer a free knife-edge variable.

    Crucially the total gas flow is read **directly from the transfer**
    ``Q_gas = Rho_T_11·V_gas/(RT/1000·NQ)`` (= ``B·pTOTAL·…``), never from
    ``k_p·(pTOTAL + p_h2o - p_ext)`` - so the catastrophic cancellation of
    :func:`calc_gas_torch` does not reappear.

    Returns ``(p_h2, p_ch4, p_co2, pTOTAL, q_gas, q_ch4, q_co2)``.
    """
    S_h2 = x[..., _IDX_S_H2]
    S_ch4 = x[..., _IDX_S_CH4]
    S_co2_free = torch.clamp(x[..., _IDX_S_CO2] - x[..., _IDX_S_HCO3], min=0.0)

    RT = p.RT
    kLa_V = p.k_L_a * (p.V_liq / p.V_gas)
    P0 = p.p_ext - p.p_gas_h2o  # pTOTAL is pinned near this

    c = (RT / 16.0, RT / 64.0, RT)
    a = (16.0 / (RT * p.K_H_h2), 64.0 / (RT * p.K_H_ch4), 1.0 / (RT * p.K_H_co2))
    S = (S_h2, S_ch4, S_co2_free)
    d = tuple(c[i] * kLa_V * a[i] for i in range(3))  # constant denom part
    num = tuple(c[i] * kLa_V * S[i] for i in range(3))  # numerators (∝ S_i)

    # Newton on  F(B) = Σ num_i/(d_i + B) - P0 = 0  (monotone decreasing).
    # B is NOT clamped at 0: when the dissolved gas cannot even sustain the
    # ambient pressure the root is negative, giving pTOTAL = P0 still (the sum
    # constraint holds) and a small negative Q_gas - a leaky "no-flow" boundary
    # that keeps a live gradient instead of a zero-gradient dead-zone. Only the
    # denominators are kept positive to avoid the pole at ``d_i + B = 0``.
    b_min = -0.9 * min(d)

    def _F_Fp(Bv):
        s0, s1, s2 = d[0] + Bv, d[1] + Bv, d[2] + Bv
        f = num[0] / s0 + num[1] / s1 + num[2] / s2 - P0
        fp = -(num[0] / s0**2 + num[1] / s1**2 + num[2] / s2**2)
        # Floor |fp| away from zero so a flat spot (all S_i ≈ 0) can't produce an
        # inf/NaN Newton step; keep the sign (F is monotone decreasing → fp ≤ 0).
        # The floor is float32-safe: too small a value lets ``fp**2`` underflow to
        # 0 in the implicit-gradient backward, and ``0·inf`` then poisons the whole
        # gradient with NaN. ``1e-8`` keeps ``fp**2 = 1e-16`` well above underflow
        # while staying far below any real ``|fp|`` (so the forward is unchanged).
        fp = torch.clamp(fp, max=-1.0e-8)
        return f, fp

    # Find the root B* under no_grad (forward value identical to a plain unrolled
    # Newton), then take ONE differentiable correction step. At the root F(B*) ≈ 0
    # so the value stays B*, but the gradient now flows via the implicit function
    # theorem (dB/dstate = -F_state/F_B) instead of through every iteration -
    # back-propagating the stiff unrolled solve can explode to NaN on extreme
    # states, whereas the implicit gradient is bounded.
    with torch.no_grad():
        B = torch.ones_like(S_h2)
        for _ in range(n_iter):
            f, fp = _F_Fp(B)
            B = torch.clamp(B - f / fp, min=b_min, max=1.0e6)
    if torch.is_grad_enabled() and any(t.requires_grad for t in (S_h2, S_ch4, S_co2_free)):
        f, fp = _F_Fp(B)  # num/d carry the graph back to the liquid state; B is detached
        B = torch.clamp(B - f / fp, min=b_min, max=1.0e6)

    p_h2 = num[0] / (d[0] + B)
    p_ch4 = num[1] / (d[1] + B)
    p_co2 = num[2] / (d[2] + B)
    pTOTAL = p_h2 + p_ch4 + p_co2

    rho_11 = B * pTOTAL  # total transfer to headspace = Rho_T_11
    q_gas = rho_11 * p.V_gas / (RT / 1000.0 * p.NQ)
    p_gas_wet = torch.clamp(pTOTAL + p.p_gas_h2o, min=1.0e-30)
    q_ch4 = q_gas * (p_ch4 / p_gas_wet)
    q_co2 = q_gas * (p_co2 / p_gas_wet)
    return p_h2, p_ch4, p_co2, pTOTAL, q_gas, q_ch4, q_co2


def gas_equilibrium_torch(x: torch.Tensor, p: Adm1TorchParams, n_iter: int = 25) -> torch.Tensor:
    """Quasi-steady gas-phase pressures ``[p_h2, p_ch4, p_co2, pTOTAL]`` (see
    :func:`_solve_gas_phase_torch`)."""
    p_h2, p_ch4, p_co2, pTOTAL, *_ = _solve_gas_phase_torch(x, p, n_iter)
    return torch.stack([p_h2, p_ch4, p_co2, pTOTAL], dim=-1)


def calc_gas_quasi_steady_torch(x: torch.Tensor, p: Adm1TorchParams, n_iter: int = 25):
    """Well-conditioned ``(q_gas, q_ch4, q_co2)`` from the liquid state via the
    quasi-steady gas phase - the fix for the knife-edge :func:`calc_gas_torch`
    (see :func:`_solve_gas_phase_torch`). Only the liquid slots of ``x`` are read."""
    *_, q_gas, q_ch4, q_co2 = _solve_gas_phase_torch(x, p, n_iter)
    return q_gas, q_ch4, q_co2


def ph_torch(x: torch.Tensor, p: Adm1TorchParams) -> torch.Tensor:
    """Liquid-phase pH from the charge-balance state (``-log10[H+]``)."""
    S_H = _calc_ph(
        x[..., _IDX_S_NH4],
        x[..., _IDX_S_NH3],
        x[..., _IDX_S_HCO3],
        x[..., _IDX_S_AC_ION],
        x[..., _IDX_S_PRO_ION],
        x[..., _IDX_S_BU_ION],
        x[..., _IDX_S_VA_ION],
        x[..., _IDX_S_CATION],
        x[..., _IDX_S_ANION],
        p.inhib["K_w"],
    )
    return -torch.log10(torch.clamp(S_H, min=1.0e-14))


# COD-to-mass conversion factors [kg COD / kmol] used by the VFA / TAC formulas.
_M_HAc = 60.0
_COD_AC, _COD_PRO, _COD_BU, _COD_VA = 64.0, 112.0, 160.0, 208.0


def vfa_torch(x: torch.Tensor) -> torch.Tensor:
    """Total VFA concentration [kg HAc-eq / m^3] (Schlattmann 2011)."""
    return _M_HAc * (
        x[..., _IDX_S_AC] / _COD_AC + x[..., _IDX_S_PRO] / _COD_PRO + x[..., _IDX_S_BU] / _COD_BU + x[..., _IDX_S_VA] / _COD_VA
    )


def tac_torch(x: torch.Tensor, p: Adm1TorchParams) -> torch.Tensor:
    """Total alkalinity (TAC) [kg CaCO3 / m^3], titration endpoint pH 5."""
    ip = p.inhib
    H_pH5 = 1.0e-5
    a_nh4 = ip["K_a_IN"] / (H_pH5 + ip["K_a_IN"])
    a_co2 = ip["K_a_co2"] / (H_pH5 + ip["K_a_co2"])
    a_ac = ip["K_a_ac"] / (H_pH5 + ip["K_a_ac"])
    a_pro = ip["K_a_pro"] / (H_pH5 + ip["K_a_pro"])
    a_bu = ip["K_a_bu"] / (H_pH5 + ip["K_a_bu"])
    a_va = ip["K_a_va"] / (H_pH5 + ip["K_a_va"])

    total_N = x[..., _IDX_S_NH4] + x[..., _IDX_S_NH3]
    total_IC = x[..., _IDX_S_CO2]
    tac_mol = (
        (x[..., _IDX_S_NH3] - a_nh4 * total_N)
        + (x[..., _IDX_S_HCO3] - a_co2 * total_IC)
        + (x[..., _IDX_S_AC_ION] / _COD_AC - a_ac * x[..., _IDX_S_AC] / _COD_AC)
        + (x[..., _IDX_S_PRO_ION] / _COD_PRO - a_pro * x[..., _IDX_S_PRO] / _COD_PRO)
        + (x[..., _IDX_S_BU_ION] / _COD_BU - a_bu * x[..., _IDX_S_BU] / _COD_BU)
        + (x[..., _IDX_S_VA_ION] / _COD_VA - a_va * x[..., _IDX_S_VA] / _COD_VA)
        + x[..., _IDX_S_ANION]
        - x[..., _IDX_S_CATION]
    )
    return 50.0 * tac_mol
