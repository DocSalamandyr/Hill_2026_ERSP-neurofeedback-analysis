"""Statistical battery for the ERSP mechanism paper (PAPER.md §2b).

Implements:
- Cluster-based permutation tests (time-frequency and scalp)
- Linear mixed-effects models (Group × Session, random intercept)
- Planned contrasts with Bonferroni correction
- Cohen's d with 95% CI
- Bayesian evidence for absence (BF01, sham ERD vs zero)
- FDR correction for secondary analyses
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import STATS as STATS_CFG, StatsConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class ClusterResult:
    """Output from a cluster permutation test."""
    t_obs: np.ndarray              # observed test statistic map
    clusters: list                 # list of index arrays
    cluster_pv: np.ndarray         # p-value per cluster
    h0: np.ndarray                 # null distribution of cluster stats
    n_sig_clusters: int


@dataclass
class MixedModelResult:
    """Key outputs from a linear mixed-effects model."""
    formula: str
    coefficients: Dict[str, float]
    std_errors: Dict[str, float]
    p_values: Dict[str, float]
    aic: float
    bic: float
    n_obs: int
    n_groups: int
    eta_squared: Optional[Dict[str, float]] = None


@dataclass
class ContrastResult:
    """Result of a planned contrast or pairwise comparison."""
    name: str
    statistic: float
    p_value: float
    cohens_d: float
    ci_low: float
    ci_high: float
    bf01: Optional[float] = None   # Bayesian evidence for H0
    p_adjusted: Optional[float] = None  # FDR-corrected p-value


# ---------------------------------------------------------------------------
# Cluster-based permutation (wraps MNE)
# ---------------------------------------------------------------------------


def cluster_permutation_tfr(
    data_a: np.ndarray,
    data_b: np.ndarray,
    cfg: StatsConfig = STATS_CFG,
    tail: int = 0,
) -> ClusterResult:
    """Cluster permutation test on time-frequency arrays.

    Parameters
    ----------
    data_a, data_b : ndarray, shape (n_subjects, n_freqs, n_times)
        Per-subject ERSP matrices for two conditions/groups.
    cfg : StatsConfig
    tail : int
        0 = two-tailed, 1 = greater, -1 = less.

    Returns
    -------
    ClusterResult
    """
    from mne.stats import permutation_cluster_test

    t_obs, clusters, cluster_pv, h0 = permutation_cluster_test(
        [data_a, data_b],
        n_permutations=cfg.cluster_n_permutations,
        threshold=None,      # use default t-threshold from alpha
        tail=tail,
        n_jobs=-1,
        verbose="WARNING",
    )

    n_sig = int((cluster_pv < cfg.cluster_alpha).sum())
    logger.info(
        "Cluster permutation: %d significant clusters (p < %.3f) out of %d",
        n_sig, cfg.cluster_alpha, len(clusters),
    )

    return ClusterResult(
        t_obs=t_obs,
        clusters=clusters,
        cluster_pv=cluster_pv,
        h0=h0,
        n_sig_clusters=n_sig,
    )


def cluster_permutation_1samp(
    data: np.ndarray,
    cfg: StatsConfig = STATS_CFG,
    tail: int = 0,
) -> ClusterResult:
    """One-sample cluster permutation (e.g. ERSP vs zero for one group).

    Parameters
    ----------
    data : ndarray, shape (n_subjects, n_freqs, n_times)
    """
    from mne.stats import permutation_cluster_1samp_test

    t_obs, clusters, cluster_pv, h0 = permutation_cluster_1samp_test(
        data,
        n_permutations=cfg.cluster_n_permutations,
        threshold=None,
        tail=tail,
        n_jobs=-1,
        verbose="WARNING",
    )
    n_sig = int((cluster_pv < cfg.cluster_alpha).sum())
    return ClusterResult(
        t_obs=t_obs, clusters=clusters, cluster_pv=cluster_pv,
        h0=h0, n_sig_clusters=n_sig,
    )


# ---------------------------------------------------------------------------
# Linear mixed-effects models
# ---------------------------------------------------------------------------


def fit_mixed_model(
    df,
    dv: str = "erd",
    group_col: str = "group",
    session_col: str = "session",
    subject_col: str = "subject",
    formula: Optional[str] = None,
) -> MixedModelResult:
    """Fit Group × Session mixed model with random subject intercept.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format data with columns for DV, group, session, subject.
    dv : str
        Dependent variable column name.
    group_col, session_col, subject_col : str
        Column names.
    formula : str, optional
        Override the default ``"dv ~ C(group) * C(session)"`` formula.

    Returns
    -------
    MixedModelResult
    """
    import statsmodels.formula.api as smf

    if formula is None:
        formula = f"{dv} ~ C({group_col}) * C({session_col})"

    model = smf.mixedlm(
        formula, data=df, groups=df[subject_col],
    )
    result = model.fit(reml=True)

    coefficients = dict(result.fe_params)
    std_errors = dict(result.bse_fe)
    p_values = dict(result.pvalues)

    n_groups = int(getattr(result, "n_groups", 0) or df[subject_col].nunique())
    logger.info("Mixed model: AIC=%.1f, BIC=%.1f", result.aic, result.bic)
    return MixedModelResult(
        formula=formula,
        coefficients=coefficients,
        std_errors=std_errors,
        p_values=p_values,
        aic=result.aic,
        bic=result.bic,
        n_obs=int(result.nobs),
        n_groups=n_groups,
    )


def fit_mixed_model_with_slope(
    df,
    dv: str = "erd",
    group_col: str = "group",
    session_col: str = "session",
    subject_col: str = "subject",
    formula: Optional[str] = None,
) -> MixedModelResult:
    """Fit Group x Session LME with random intercept *and* slope for session.

    Falls back to intercept-only if the random-slope model fails to converge.
    Also computes partial eta-squared from a Type III mixed ANOVA via pingouin.
    """
    import math
    import statsmodels.formula.api as smf

    if formula is None:
        formula = f"{dv} ~ C({group_col}) * C({session_col})"

    re_formula = f"~{session_col}"
    converged = False

    try:
        model = smf.mixedlm(
            formula, data=df, groups=df[subject_col],
            re_formula=re_formula,
        )
        result = model.fit(reml=True)
        if not math.isnan(result.aic):
            logger.info("Random-slope model converged: AIC=%.1f", result.aic)
            converged = True
        else:
            raise ValueError("AIC is NaN, model did not converge properly")
    except Exception:
        logger.warning(
            "Random-slope model failed to converge; falling back to intercept-only"
        )

    if not converged:
        model = smf.mixedlm(
            formula, data=df, groups=df[subject_col],
        )
        result = model.fit(reml=True)
        logger.info("Intercept-only model: AIC=%.1f", result.aic)

    coefficients = dict(result.fe_params)
    std_errors = dict(result.bse_fe)
    p_values = dict(result.pvalues)

    eta_sq = _compute_eta_squared(df, dv, group_col, session_col, subject_col)
    n_groups = int(getattr(result, "n_groups", 0) or df[subject_col].nunique())

    return MixedModelResult(
        formula=formula,
        coefficients=coefficients,
        std_errors=std_errors,
        p_values=p_values,
        aic=result.aic,
        bic=result.bic,
        n_obs=int(result.nobs),
        n_groups=n_groups,
        eta_squared=eta_sq,
    )


def _compute_eta_squared(
    df, dv: str, group_col: str, session_col: str, subject_col: str,
) -> Optional[Dict[str, float]]:
    """Partial eta-squared from a mixed ANOVA (pingouin)."""
    try:
        import pingouin as pg
        aov = pg.mixed_anova(
            data=df, dv=dv, within=session_col,
            between=group_col, subject=subject_col,
        )
        eta = {}
        for _, row in aov.iterrows():
            eta[row["Source"]] = float(row["np2"])
        return eta
    except Exception as exc:
        logger.warning("Could not compute eta-squared: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------


def cohens_d(
    a: np.ndarray,
    b: np.ndarray,
) -> Tuple[float, float, float]:
    """Compute Cohen's d with 95% CI (independent samples).

    Returns
    -------
    (d, ci_low, ci_high)
    """
    na, nb = len(a), len(b)
    mean_diff = a.mean() - b.mean()
    pooled_var = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    pooled_sd = np.sqrt(pooled_var)

    d = mean_diff / pooled_sd if pooled_sd > 0 else 0.0

    # Hedges' correction for small samples
    correction = 1.0 - 3.0 / (4.0 * (na + nb) - 9.0)
    d *= correction

    # SE of d (Hedges & Olkin)
    se_d = np.sqrt((na + nb) / (na * nb) + d ** 2 / (2 * (na + nb)))
    ci_low = d - 1.96 * se_d
    ci_high = d + 1.96 * se_d

    return d, ci_low, ci_high


# ---------------------------------------------------------------------------
# Bayesian null testing
# ---------------------------------------------------------------------------


def bayes_factor_ttest(
    a: np.ndarray,
    b: Optional[np.ndarray] = None,
    paired: bool = False,
    prior: float = STATS_CFG.bayes_prior,
) -> float:
    """Bayesian t-test returning BF01 (evidence for H0).

    Uses ``pingouin.bayesfactor_ttest`` when available, otherwise falls
    back to a JZS Bayes factor approximation.

    Parameters
    ----------
    a : array
        First sample (or differences if one-sample).
    b : array, optional
        Second sample.  If None, performs a one-sample test vs zero.
    paired : bool
    prior : float
        Cauchy prior width (default 0.707, JASP default).

    Returns
    -------
    bf01 : float
        Values > 1 favour H0 (absence of effect).
    """
    try:
        import pingouin as pg

        if b is None:
            # One-sample: test a against 0
            from scipy.stats import ttest_1samp
            t_stat, _ = ttest_1samp(a, 0)
            n = len(a)
        elif paired:
            from scipy.stats import ttest_rel
            t_stat, _ = ttest_rel(a, b)
            n = len(a)
        else:
            from scipy.stats import ttest_ind
            t_stat, _ = ttest_ind(a, b)
            n = len(a) + len(b)

        bf10 = pg.bayesfactor_ttest(t_stat, n, paired=paired, r=prior)
        bf01 = 1.0 / bf10 if bf10 > 0 else float("inf")
        return bf01

    except ImportError:
        logger.warning("pingouin not installed; returning NaN for BF01")
        return float("nan")


# ---------------------------------------------------------------------------
# Planned contrasts
# ---------------------------------------------------------------------------


def planned_contrast(
    active: np.ndarray,
    sham: np.ndarray,
    name: str = "",
    cfg: StatsConfig = STATS_CFG,
) -> ContrastResult:
    """Run a single planned contrast (independent t-test + d + BF01).

    Parameters
    ----------
    active, sham : 1-D arrays
        Per-subject scalar ERD (or other DV) for each group.
    name : str
        Label for this contrast.
    cfg : StatsConfig
    """
    from scipy.stats import ttest_ind

    t_stat, p_val = ttest_ind(active, sham)
    d, ci_lo, ci_hi = cohens_d(active, sham)
    bf01 = bayes_factor_ttest(active, sham)

    logger.info(
        "Contrast '%s': t=%.3f, p=%.4f, d=%.3f [%.3f, %.3f], BF01=%.2f",
        name, t_stat, p_val, d, ci_lo, ci_hi, bf01,
    )

    return ContrastResult(
        name=name,
        statistic=float(t_stat),
        p_value=float(p_val),
        cohens_d=d,
        ci_low=ci_lo,
        ci_high=ci_hi,
        bf01=bf01,
    )


def run_all_planned_contrasts(
    group_data: Dict[str, np.ndarray],
    cfg: StatsConfig = STATS_CFG,
) -> List[ContrastResult]:
    """Execute the four planned contrasts from PAPER.md §2b.

    Parameters
    ----------
    group_data : dict
        Keys: "c3_smr", "c3_beta", "c4_smr", "sham".
        Values: 1-D arrays of per-subject scalar ERD.

    Returns
    -------
    list of ContrastResult
    """
    sham = group_data["sham"]

    # 1. Active pooled vs sham
    active_pooled = np.concatenate([
        group_data["c3_smr"], group_data["c3_beta"], group_data["c4_smr"],
    ])
    results = [planned_contrast(active_pooled, sham, "Active (pooled) vs Sham", cfg)]

    # 2. C3 SMR vs C3 Beta (frequency specificity)
    results.append(planned_contrast(
        group_data["c3_smr"], group_data["c3_beta"],
        "C3 SMR vs C3 Beta (freq specificity)", cfg,
    ))

    # 3. C3 SMR vs C4 SMR (site specificity)
    results.append(planned_contrast(
        group_data["c3_smr"], group_data["c4_smr"],
        "C3 SMR vs C4 SMR (site specificity)", cfg,
    ))

    # 4-6. Each active vs sham (Bonferroni p < .017)
    for gkey in ("c3_smr", "c3_beta", "c4_smr"):
        results.append(planned_contrast(
            group_data[gkey], sham,
            f"{gkey} vs Sham", cfg,
        ))

    p_vals = np.array([c.p_value for c in results])
    _, p_adj = fdr_correct(p_vals)
    for c, pa in zip(results, p_adj):
        c.p_adjusted = float(pa)
    logger.info(
        "FDR correction applied to %d contrasts (method=%s)",
        len(results), cfg.fdr_method,
    )

    return results


# ---------------------------------------------------------------------------
# FDR correction
# ---------------------------------------------------------------------------


def fdr_correct(
    p_values: np.ndarray,
    method: str = STATS_CFG.fdr_method,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply FDR correction (Benjamini-Hochberg).

    Returns
    -------
    reject : bool array
    p_corrected : float array
    """
    from mne.stats import fdr_correction
    reject, p_corr = fdr_correction(p_values, alpha=alpha, method="indep")
    return reject, p_corr


# ---------------------------------------------------------------------------
# ANOVA helper (for ERP repeated-measures)
# ---------------------------------------------------------------------------


def theta_trajectory_model(df) -> MixedModelResult:
    """LME for theta ERS: theta_ers ~ C(group) * C(session) + (1 + session|subject).

    Tests the THEORY.md prediction: non-significant Group x Session interaction
    on theta ERS (stable sensory component) combined with significant interaction
    on reward-band ERD supports the two-process model.

    Parameters
    ----------
    df : DataFrame
        Long-format with columns: subject, group, session, theta_ers.
    """
    return fit_mixed_model_with_slope(
        df, dv="theta_ers", group_col="group",
        session_col="session", subject_col="subject",
    )


def retention_paired_contrasts(
    df,
    dv: str = "erd",
    session_a: int = 5,
    session_b: int = 6,
    group_col: str = "group",
    session_col: str = "session",
    subject_col: str = "subject",
    active_groups: Sequence[str] = ("c3_smr", "c3_beta", "c4_smr"),
) -> List[ContrastResult]:
    """Paired t-tests on ERD (session A vs B) per active group.

    Default: test whether ERD is maintained from session 5 to session 6
    (retention).  Returns one ContrastResult per active group.
    """
    from scipy.stats import ttest_rel

    results: list[ContrastResult] = []
    for grp in active_groups:
        g = df[df[group_col] == grp]
        sa = g[g[session_col] == session_a].set_index(subject_col)[dv]
        sb = g[g[session_col] == session_b].set_index(subject_col)[dv]
        common = sa.index.intersection(sb.index)
        if len(common) < 3:
            logger.warning("Too few subjects for retention contrast: %s (n=%d)",
                           grp, len(common))
            continue

        a_vals = sa.loc[common].values
        b_vals = sb.loc[common].values

        t_stat, p_val = ttest_rel(a_vals, b_vals)
        diff = a_vals - b_vals
        d_val = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else 0.0
        se_d = np.sqrt(1.0 / len(common) + d_val ** 2 / (2.0 * len(common)))
        ci_lo = d_val - 1.96 * se_d
        ci_hi = d_val + 1.96 * se_d

        bf01 = bayes_factor_ttest(a_vals, b_vals, paired=True)

        results.append(ContrastResult(
            name=f"Retention S{session_a}→S{session_b} {grp}",
            statistic=float(t_stat),
            p_value=float(p_val),
            cohens_d=d_val,
            ci_low=ci_lo,
            ci_high=ci_hi,
            bf01=bf01,
        ))

    if results:
        p_vals = np.array([c.p_value for c in results])
        _, p_adj = fdr_correct(p_vals)
        for c, pa in zip(results, p_adj):
            c.p_adjusted = float(pa)

    return results


def resting_persistence(
    df,
    dv: str = "delta",
    group_col: str = "group",
    cfg: StatsConfig = STATS_CFG,
) -> List[ContrastResult]:
    """Between-group comparisons on resting-state PSD change (S6 - S1).

    Each active group is compared to sham.

    Parameters
    ----------
    df : DataFrame
        Output of ``group.assemble_resting_change()`` with columns:
        subject, group, delta, ...
    """
    sham = df[df[group_col] == "sham"][dv].values
    results: list[ContrastResult] = []
    for grp in ("c3_smr", "c3_beta", "c4_smr"):
        active = df[df[group_col] == grp][dv].values
        if len(active) < 2 or len(sham) < 2:
            continue
        results.append(planned_contrast(active, sham, f"Resting Δ {grp} vs sham", cfg))

    if results:
        p_vals = np.array([c.p_value for c in results])
        _, p_adj = fdr_correct(p_vals)
        for c, pa in zip(results, p_adj):
            c.p_adjusted = float(pa)

    return results


def erd_predicts_resting_change(
    erd_df,
    resting_df,
    erd_col: str = "value",
    resting_col: str = "delta",
    subject_col: str = "subject",
) -> Dict[str, float]:
    """Linear regression: mean training ERD predicts S6-S1 resting change.

    Restricted to active subjects (PAPER.md §2d.4).

    Parameters
    ----------
    erd_df : DataFrame
        ERSP scalar data (from ``group.assemble_ersp_scalars()``) filtered
        to active subjects, with columns subject, value (mean ERD).
    resting_df : DataFrame
        Resting change data (from ``group.assemble_resting_change()``),
        filtered to active subjects, with columns subject, delta.

    Returns
    -------
    dict with keys: r_squared, beta, p_value, n.
    """
    from scipy.stats import linregress

    erd_means = erd_df.groupby(subject_col)[erd_col].mean()
    rest_vals = resting_df.groupby(subject_col)[resting_col].mean()
    common = erd_means.index.intersection(rest_vals.index)

    if len(common) < 5:
        logger.warning("Regression needs ≥5 subjects, got %d", len(common))
        return {"r_squared": float("nan"), "beta": 0.0, "p_value": 1.0, "n": len(common)}

    x = erd_means.loc[common].values
    y = rest_vals.loc[common].values

    slope, intercept, r, p, se = linregress(x, y)
    logger.info(
        "ERD→resting regression: r²=%.3f, β=%.3f, p=%.4f, n=%d",
        r ** 2, slope, p, len(common),
    )
    return {"r_squared": float(r ** 2), "beta": float(slope),
            "p_value": float(p), "n": len(common)}


def rm_anova(
    df,
    dv: str,
    within: str = "session",
    between: str = "group",
    subject: str = "subject",
):
    """Mixed (split-plot) ANOVA via pingouin.

    Returns the pingouin ANOVA table as a DataFrame.
    """
    import pingouin as pg

    return pg.mixed_anova(
        data=df, dv=dv, within=within, between=between, subject=subject,
    )
