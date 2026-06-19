"""
============================================================================
UMM for Pharma — Unified Bayesian Model Starter
============================================================================
A starter skeleton that combines:
  1. Bayesian Hierarchical MMM across multiple geos (monthly granularity)
  2. HCP-level MTA on prescription claims data (path → Rx outcome)

Architecture:
    MMM provides structural skeleton: adstock + Hill saturation + geo pooling
    MTA provides granular likelihood evidence: HCP touchpoint paths → Rx outcome
    Unified posterior = single source of truth for channel contribution

This file is meant to be ADAPTED, not run as-is on production data.
Replace the synthetic data section with your real data loaders.

Tested with: pymc>=5.10, arviz>=0.16, numpy>=1.24, pandas>=2.0
============================================================================
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from scipy.special import expit
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# SECTION 0: CONFIGURATION
# ============================================================================
# These are the knobs you'll tune for your brand / geo setup.

CONFIG = {
    "n_geos": 5,                      # number of geos (countries/regions)
    "n_months": 36,                   # months of MMM history (>=24 recommended)
    "n_channels": 6,                  # number of marketing channels
    "n_hcps": 2000,                   # HCPs in the MTA panel
    "avg_path_length": 8,             # avg touchpoints per HCP over window
    "rx_window_days": 180,            # MTA lookback window (pharma: 60-180)
    "adstock_max_lag": 8,             # months of carryover to model
    "seed": 42,
}

CHANNEL_NAMES = [
    "Paid_Digital",      # programmatic, search, display on HCP sites
    "Sales_Force",       # rep detailing calls
    "Conference",        # ASCO/ACC/etc. sponsorships + booth
    "CME",               # sponsored CME modules
    "KOL_Program",       # speaker bureau + advisory boards
    "Email",             # HCP email nurture
]

GEO_NAMES = [f"GEO_{i+1}" for i in range(CONFIG["n_geos"])]

np.random.seed(CONFIG["seed"])

# ============================================================================
# SECTION 1: SYNTHETIC DATA GENERATION
# ============================================================================
# Replace this entire section with real data loaders.
# Real data shape requirements are documented inline below.

def generate_synthetic_data():
    """
    Generates realistic-looking pharma marketing data:
      - spend_df: monthly channel spend per geo (MMM-side)
      - rx_df: monthly Rx outcomes per geo (MMM-side response variable)
      - hcp_paths_df: HCP-level touchpoint sequences with Rx outcomes (MTA-side)

    Returns three DataFrames. Replace with your real data loaders.
    """

    n_g = CONFIG["n_geos"]
    n_t = CONFIG["n_months"]
    n_c = CONFIG["n_channels"]
    n_h = CONFIG["n_hcps"]

    # --- True channel effects (unknown to the model — what we're trying to recover) ---
    # Geo-specific deviations around a global mean
    true_mu = np.array([0.8, 1.2, 0.6, 0.4, 0.9, 0.3])  # global channel effects
    true_sigma_geo = 0.15  # geo-level heterogeneity
    true_beta = np.exp(
        true_mu[None, :] + true_sigma_geo * np.random.randn(n_g, n_c)
    )  # shape: (n_geos, n_channels)

    # Adstock decay parameters per channel
    true_lambda = np.array([0.3, 0.6, 0.4, 0.5, 0.45, 0.2])

    # Hill saturation: alpha (shape) and K (half-saturation point)
    true_alpha = np.array([1.2, 1.0, 1.3, 1.1, 1.0, 1.4])
    true_K = np.array([50, 30, 25, 20, 35, 15])

    # Generate spend with realistic pharma distribution: skewed, with zeros
    spend = np.random.gamma(shape=2.0, scale=20.0, size=(n_g, n_t, n_c))
    # Some channels have sparse activity in some geos (e.g., no conferences in GEO_1)
    spend[0, :, 2] *= 0.1  # less conference in GEO_1

    # --- MMM-side response variable: monthly Rx volume per geo ---
    baseline = 1000 + 200 * np.random.randn(n_g)  # geo-specific baselines
    seasonality = 50 * np.sin(2 * np.pi * np.arange(n_t) / 12)  # yearly cycle
    rx_volume = np.zeros((n_g, n_t))

    for g in range(n_g):
        for c in range(n_c):
            # Apply adstock
            adstocked = np.zeros(n_t)
            for t in range(n_t):
                adstocked[t] = spend[g, t, c] + true_lambda[c] * (adstocked[t-1] if t > 0 else 0)
            # Apply Hill saturation
            saturated = (adstocked ** true_alpha[c]) / (adstocked ** true_alpha[c] + true_K[c] ** true_alpha[c])
            # Scale by channel effect
            rx_volume[g] += true_beta[g, c] * saturated * 100

        rx_volume[g] += baseline[g] + seasonality + 20 * np.random.randn(n_t)

    # Build MMM DataFrames
    spend_long = []
    rx_long = []
    for g in range(n_g):
        for t in range(n_t):
            for c in range(n_c):
                spend_long.append({
                    "geo": GEO_NAMES[g],
                    "month": t,
                    "channel": CHANNEL_NAMES[c],
                    "spend": spend[g, t, c],
                })
            rx_long.append({
                "geo": GEO_NAMES[g],
                "month": t,
                "rx_volume": rx_volume[g, t],
            })
    spend_df = pd.DataFrame(spend_long)
    rx_df = pd.DataFrame(rx_long)

    # --- MTA-side data: HCP paths and Rx outcomes ---
    # Each HCP has a sequence of touchpoints and a binary Rx outcome
    hcp_records = []
    for h in range(n_h):
        geo_idx = np.random.randint(n_g)
        path_length = np.random.poisson(CONFIG["avg_path_length"])
        # Random touchpoints weighted by channel activity
        touches = np.random.choice(n_c, size=path_length, p=spend[geo_idx].sum(axis=0)/spend[geo_idx].sum())
        # Position matters
        positions = np.random.uniform(0, 1, size=path_length)
        # Outcome probability depends on touches (MTA evidence for the model)
        # Higher-decile prescribers respond more to rep calls, etc.
        # Here we just use a simple sum-based scoring
        score = sum(true_beta[geo_idx, c] for c in touches) / 100
        p_rx = expit(score - 2.0 + np.random.randn() * 0.3)
        rx_outcome = np.random.binomial(1, p_rx)

        for i, c in enumerate(touches):
            hcp_records.append({
                "hcp_id": f"HCP_{h:05d}",
                "geo": GEO_NAMES[geo_idx],
                "touch_seq": i,
                "channel": CHANNEL_NAMES[c],
                "position_score": positions[i],
                "rx_outcome": rx_outcome,  # same for all rows of this HCP's path
            })

    hcp_paths_df = pd.DataFrame(hcp_records)

    return spend_df, rx_df, hcp_paths_df


# ============================================================================
# SECTION 2: PREPROCESSING — ADSTOCK & HILL TRANSFORMS
# ============================================================================

def geometric_adstock(spend_array, lam):
    """
    Apply geometric adstock decay to a spend time series.

    Parameters
    ----------
    spend_array : np.ndarray, shape (T,)
        Spend over T time periods.
    lam : float in [0, 1)
        Decay parameter. Higher = longer carryover.

    Returns
    -------
    np.ndarray, shape (T,)
        Adstocked spend.
    """
    T = len(spend_array)
    adstocked = np.zeros(T)
    for t in range(T):
        adstocked[t] = spend_array[t] + lam * (adstocked[t - 1] if t > 0 else 0.0)
    return adstocked


def hill_saturation(x, alpha, K):
    """
    Hill function: x^alpha / (x^alpha + K^alpha).
    Captures diminishing returns at high spend.

    Parameters
    ----------
    x : np.ndarray
        Input (typically adstocked spend).
    alpha : float > 0
        Shape parameter. alpha=1 is standard Michaelis-Menten.
    K : float > 0
        Half-saturation point (spend level at which effect = 50% of max).
    """
    xa = np.power(np.maximum(x, 1e-8), alpha)
    Ka = np.power(K, alpha)
    return xa / (xa + Ka)


def preprocess_mmm_data(spend_df, rx_df, lambdas):
    """
    Pivot spend to (geo, month, channel) tensor and apply adstock per channel.

    Returns
    -------
    spend_tensor : np.ndarray, shape (n_geos, n_months, n_channels), raw
    adstocked_tensor : np.ndarray, shape (n_geos, n_months, n_channels), adstocked
    rx_matrix : np.ndarray, shape (n_geos, n_months), Rx outcome
    geo_index : dict mapping geo name to integer index
    """
    n_g = CONFIG["n_geos"]
    n_t = CONFIG["n_months"]
    n_c = CONFIG["n_channels"]

    geo_index = {g: i for i, g in enumerate(GEO_NAMES)}

    spend_tensor = np.zeros((n_g, n_t, n_c))
    for _, row in spend_df.iterrows():
        g = geo_index[row["geo"]]
        t = int(row["month"])
        c = CHANNEL_NAMES.index(row["channel"])
        spend_tensor[g, t, c] = row["spend"]

    rx_matrix = np.zeros((n_g, n_t))
    for _, row in rx_df.iterrows():
        g = geo_index[row["geo"]]
        t = int(row["month"])
        rx_matrix[g, t] = row["rx_volume"]

    # Apply adstock per channel
    adstocked_tensor = np.zeros_like(spend_tensor)
    for g in range(n_g):
        for c in range(n_c):
            adstocked_tensor[g, :, c] = geometric_adstock(spend_tensor[g, :, c], lambdas[c])

    return spend_tensor, adstocked_tensor, rx_matrix, geo_index


def preprocess_mta_data(hcp_paths_df):
    """
    Build the HCP-level path evidence matrix.

    Returns
    -------
    hcp_features : np.ndarray, shape (n_hcps, n_channels)
        Per-HCP touch count per channel (MTA evidence).
    hcp_outcomes : np.ndarray, shape (n_hcps,)
        Binary Rx outcome per HCP.
    hcp_geo_idx : np.ndarray, shape (n_hcps,)
        Geo index per HCP.
    """
    hcp_features = np.zeros((CONFIG["n_hcps"], CONFIG["n_channels"]))
    hcp_outcomes = np.zeros(CONFIG["n_hcps"])
    hcp_geo_idx = np.zeros(CONFIG["n_hcps"], dtype=int)

    geo_index = {g: i for i, g in enumerate(GEO_NAMES)}

    for _, row in hcp_paths_df.iterrows():
        h = int(row["hcp_id"].split("_")[1])
        c = CHANNEL_NAMES.index(row["channel"])
        hcp_features[h, c] += 1
        hcp_outcomes[h] = row["rx_outcome"]
        hcp_geo_idx[h] = geo_index[row["geo"]]

    return hcp_features, hcp_outcomes, hcp_geo_idx


# ============================================================================
# SECTION 3: UNIFIED BAYESIAN MODEL — THE HEART OF UMM
# ============================================================================

def build_unified_model(adstocked_tensor, rx_matrix,
                        hcp_features, hcp_outcomes, hcp_geo_idx):
    """
    Build the unified Bayesian model that combines:
      - MMM structural skeleton (geo-hierarchical adstocked spend → Rx volume)
      - MTA granular evidence (HCP touch patterns → Rx binary outcome)

    Parameters
    ----------
    adstocked_tensor : (n_geos, n_months, n_channels)
        Already adstocked spend (use geometric_adstock with your lambdas).
        In production, treat lambda as a parameter to learn — see TODO below.
    rx_matrix : (n_geos, n_months)
        Monthly Rx volume per geo.
    hcp_features : (n_hcps, n_channels)
        Touch count per HCP per channel.
    hcp_outcomes : (n_hcps,)
        Binary Rx outcome per HCP.
    hcp_geo_idx : (n_hcps,)
        Geo index per HCP.

    Returns
    -------
    pymc.Model
    """
    n_g, n_t, n_c = adstocked_tensor.shape
    n_h = hcp_features.shape[0]

    # Coordinate labels for posterior interpretation
    geo_labels = GEO_NAMES
    channel_labels = CHANNEL_NAMES

    with pm.Model() as model:
        # ------------------------------------------------------------------
        # TIER 1: GLOBAL HYPERPRIORS (shared across all geos)
        # ------------------------------------------------------------------
        # Log-scale channel effect means — these are weakly informative
        mu_channel = pm.Normal("mu_channel", mu=0.0, sigma=0.5, shape=n_c)

        # Between-geo heterogeneity (partial pooling strength)
        sigma_geo = pm.HalfNormal("sigma_geo", sigma=0.3, shape=n_c)

        # ------------------------------------------------------------------
        # TIER 2: GEO-SPECIFIC CHANNEL EFFECTS (partial pooling)
        # ------------------------------------------------------------------
        # Non-centered parameterization for better sampling
        z_geo = pm.Normal("z_geo", mu=0, sigma=1, shape=(n_g, n_c))
        beta_geo = pm.Deterministic(
            "beta_geo",
            mu_channel[None, :] + sigma_geo[None, :] * z_geo,
        )

        # Hill saturation parameters per channel (global, weakly informed)
        alpha = pm.Gamma("alpha", alpha=2, beta=2, shape=n_c)  # around 1.0
        K = pm.LogNormal("K", mu=np.log(20.0), sigma=0.5, shape=n_c)

        # Apply Hill saturation to adstocked spend
        # Shape: (n_geos, n_months, n_channels)
        saturated = pm.Deterministic(
            "saturated",
            hill_saturation(adstocked_tensor, alpha[None, None, :], K[None, None, :]),
        )

        # ------------------------------------------------------------------
        # TIER 3: MMM LIKELIHOOD (aggregated Rx)
        # ------------------------------------------------------------------
        # Compute geo-level expected contribution
        # contribution[g, t] = sum_c beta_geo[g, c] * saturated[g, t, c]
        contribution = pm.math.dot(saturated, beta_geo.T)  # shape: (n_g, n_t)

        # Baseline (geo-specific intercept)
        baseline = pm.Normal("baseline", mu=1000, sigma=300, shape=n_g)

        # Seasonality (Fourier terms)
        t_idx = np.arange(n_t)
        season_amp = pm.HalfNormal("season_amp", sigma=50)
        season = pm.Deterministic(
            "season",
            season_amp * np.sin(2 * np.pi * t_idx / 12),
        )

        # MMM expected Rx
        mu_mmm = baseline[:, None] + season[None, :] + contribution

        # MMM observation noise
        sigma_mmm = pm.HalfNormal("sigma_mmm", sigma=100)

        # MMM likelihood
        pm.Normal("rx_obs", mu=mu_mmm, sigma=sigma_mmm, observed=rx_matrix)

        # ------------------------------------------------------------------
        # TIER 4: MTA LIKELIHOOD (HCP-level Rx outcome)
        # ------------------------------------------------------------------
        # For each HCP, model probability of Rx given their touch pattern.
        # This is where MTA evidence enters the unified posterior.

        # HCP-level channel effect (use geo-specific betas)
        hcp_touch_logit = pm.math.dot(hcp_features, beta_geo[hcp_geo_idx].T)
        # Shape: (n_hcps,) — logit contribution from each HCP's touches

        # HCP-level intercept (prescriber baseline)
        hcp_intercept = pm.Normal("hcp_intercept", mu=-2.0, sigma=0.5)

        # MTA logit
        hcp_logit = hcp_intercept + hcp_touch_logit / 10.0  # scale down

        # MTA likelihood
        pm.Bernoulli("rx_hcp_obs", logit_p=hcp_logit, observed=hcp_outcomes)

        # ------------------------------------------------------------------
        # CROSS-CHANNEL INTERACTIONS (optional but powerful)
        # ------------------------------------------------------------------
        # Example: Sales Force × Paid Digital interaction
        # Uncomment to enable — adds identifiability cost

        # interaction_sf_digital = pm.Normal("int_sf_digital", mu=0, sigma=0.2)
        # interaction_term = interaction_sf_digital * (
        #     saturated[:, :, 0] * saturated[:, :, 1]  # digital × sales_force
        # ).sum(axis=1)
        # pm.Deterministic("interaction_contrib", interaction_term)

    return model


# ============================================================================
# SECTION 4: SAMPLING
# ============================================================================

def sample_model(model, draws=2000, tune=1000, chains=4, target_accept=0.95):
    """
    Sample the unified model. Returns an ArviZ InferenceData object.

    Production tips:
      - Start with draws=500, tune=500, chains=2 for a smoke test
      - For final quarterly fit: draws=2000, tune=1000, chains=4
      - If divergences: increase target_accept to 0.99
      - For very large models: consider pm.sample(..., nuts_sampler="numpyro")
    """
    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=CONFIG["seed"],
            progressbar=False,
            return_inferencedata=True,
        )
    return trace


# ============================================================================
# SECTION 5: VALIDATION & DIAGNOSTICS
# ============================================================================

def validate_model(trace):
    """Print standard MCMC diagnostics."""
    print("\n=== CONVERGENCE DIAGNOSTICS ===")
    summary = az.summary(trace, var_names=["mu_channel", "sigma_geo", "alpha", "K", "baseline"])
    print(summary)

    print("\n=== R-HAT CHECK (should all be < 1.01) ===")
    rhat = az.rhat(trace)
    print(rhat)

    print("\n=== EFFECTIVE SAMPLE SIZE ===")
    ess = az.ess(trace)
    print(ess)


def extract_unified_results(trace):
    """
    Extract the unified outputs:
      - Per-channel per-geo effect (posterior mean + 95% HDI)
      - Response curves
      - Cross-channel interactions
    """
    # Channel effects per geo
    beta_post = trace.posterior["beta_geo"].mean(dim=["chain", "draw"]).values
    beta_hdi = az.hdi(trace, var_names=["beta_geo"], hdi_prob=0.95)["beta_geo"].values

    # Hill parameters
    alpha_post = trace.posterior["alpha"].mean(dim=["chain", "draw"]).values
    K_post = trace.posterior["K"].mean(dim=["chain", "draw"]).values

    # Build results DataFrame
    rows = []
    for g in range(CONFIG["n_geos"]):
        for c in range(CONFIG["n_channels"]):
            rows.append({
                "geo": GEO_NAMES[g],
                "channel": CHANNEL_NAMES[c],
                "beta_mean": beta_post[g, c],
                "beta_hdi_low": beta_hdi[g, c, 0],
                "beta_hdi_high": beta_hdi[g, c, 1],
                "alpha": alpha_post[c],
                "K": K_post[c],
            })
    results_df = pd.DataFrame(rows)
    return results_df


# ============================================================================
# SECTION 6: VISUALIZATION
# ============================================================================

def plot_response_curves(trace, channel_idx=0, geo_idx=0):
    """Plot the Hill response curve for a given channel/geo with uncertainty."""
    alpha_samples = trace.posterior["alpha"].values[:, :, channel_idx].flatten()
    K_samples = trace.posterior["K"].values[:, :, channel_idx].flatten()
    beta_samples = trace.posterior["beta_geo"].values[:, :, geo_idx, channel_idx].flatten()

    x = np.linspace(0, 100, 200)
    curves = np.zeros((len(alpha_samples), len(x)))
    for i in range(len(alpha_samples)):
        s = hill_saturation(x, alpha_samples[i], K_samples[i]) * beta_samples[i]
        curves[i] = s

    mean_curve = curves.mean(axis=0)
    low_curve = np.percentile(curves, 2.5, axis=0)
    high_curve = np.percentile(curves, 97.5, axis=0)

    plt.figure(figsize=(10, 6))
    plt.fill_between(x, low_curve, high_curve, alpha=0.3, label="95% CI")
    plt.plot(x, mean_curve, "b-", lw=2, label="Posterior mean")
    plt.xlabel("Adstocked Spend")
    plt.ylabel("Rx Lift Contribution")
    plt.title(f"Response Curve: {CHANNEL_NAMES[channel_idx]} in {GEO_NAMES[geo_idx]}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"response_curve_{CHANNEL_NAMES[channel_idx]}_{GEO_NAMES[geo_idx]}.png", dpi=150)
    plt.close()
    print(f"Saved response_curve_{CHANNEL_NAMES[channel_idx]}_{GEO_NAMES[geo_idx]}.png")


def plot_channel_contribution(results_df):
    """Bar plot of channel contribution per geo with uncertainty."""
    fig, axes = plt.subplots(1, CONFIG["n_geos"], figsize=(20, 5), sharey=True)
    for g, ax in enumerate(axes):
        sub = results_df[results_df["geo"] == GEO_NAMES[g]]
        ax.bar(sub["channel"], sub["beta_mean"],
               yerr=[sub["beta_mean"] - sub["beta_hdi_low"],
                     sub["beta_hdi_high"] - sub["beta_mean"]],
               capsize=4)
        ax.set_title(GEO_NAMES[g])
        ax.tick_params(axis="x", rotation=45)
    axes[0].set_ylabel("Channel Effect (β)")
    plt.suptitle("Unified Channel Contributions per Geo (with 95% HDI)")
    plt.tight_layout()
    plt.savefig("channel_contribution_per_geo.png", dpi=150)
    plt.close()
    print("Saved channel_contribution_per_geo.png")


# ============================================================================
# SECTION 7: MAIN — RUN END-TO-END
# ============================================================================

def main():
    print("=" * 70)
    print("UMM for Pharma — Unified Bayesian Model")
    print("=" * 70)

    # Step 1: Generate (or load) data
    print("\n[1/6] Generating synthetic data...")
    spend_df, rx_df, hcp_paths_df = generate_synthetic_data()
    print(f"  MMM data: {len(spend_df)} spend rows, {len(rx_df)} Rx rows")
    print(f"  MTA data: {len(hcp_paths_df)} touchpoints across {hcp_paths_df['hcp_id'].nunique()} HCPs")

    # Step 2: Preprocess — apply adstock
    print("\n[2/6] Preprocessing (adstock + feature engineering)...")
    # In production: learn lambda as a parameter, or set per business knowledge
    initial_lambdas = np.array([0.3, 0.6, 0.4, 0.5, 0.45, 0.2])
    spend_tensor, adstocked_tensor, rx_matrix, geo_index = preprocess_mmm_data(
        spend_df, rx_df, initial_lambdas
    )
    hcp_features, hcp_outcomes, hcp_geo_idx = preprocess_mta_data(hcp_paths_df)
    print(f"  Adstocked tensor shape: {adstocked_tensor.shape}")
    print(f"  HCP features shape: {hcp_features.shape}")

    # Step 3: Build unified model
    print("\n[3/6] Building unified Bayesian model...")
    model = build_unified_model(
        adstocked_tensor, rx_matrix,
        hcp_features, hcp_outcomes, hcp_geo_idx,
    )
    print(f"  Free RVs: {len(model.free_RVs)}")

    # Step 4: Sample
    print("\n[4/6] Sampling (this may take a few minutes)...")
    print("  For production: draws=2000, tune=1000, chains=4")
    print("  For smoke test: draws=500, tune=500, chains=2")
    trace = sample_model(model, draws=500, tune=500, chains=2)

    # Step 5: Validate
    print("\n[5/6] Validating model...")
    validate_model(trace)

    # Step 6: Extract and visualize results
    print("\n[6/6] Extracting unified results...")
    results_df = extract_unified_results(trace)
    print("\n=== UNIFIED CHANNEL CONTRIBUTIONS ===")
    print(results_df.to_string(index=False))

    # Save results
    results_df.to_csv("unified_channel_contributions.csv", index=False)
    print("\nSaved unified_channel_contributions.csv")

    # Plot
    plot_channel_contribution(results_df)
    plot_response_curves(trace, channel_idx=0, geo_idx=0)

    print("\n" + "=" * 70)
    print("DONE — Unified model fit complete.")
    print("Next steps:")
    print("  1. Review R-hat diagnostics — should all be < 1.01")
    print("  2. Inspect channel contributions for business sense")
    print("  3. Validate against geo experiment results (Phase 5)")
    print("  4. Update priors for next quarter's fit")
    print("=" * 70)


if __name__ == "__main__":
    main()