# UMM Joint-Likelihood Architecture Specification
### Shared-Parameter Bayesian Blending of MMM and MTA ("Method 2")

| | |
|---|---|
| **Document status** | Draft v0.1 — for technical review |
| **Scope** | Architecture + mathematical spec for the shared-parameter joint-likelihood UMM variant |
| **Companion docs** | `The Unified Blueprint` (pptx/pdf/html) — documents the sequential prior-transfer variant ("Method 1") already built |
| **Relationship to prior work** | Method 1 in the existing blueprint follows the Objective Platform / Think with Google paper's recipe almost exactly: build MMM and MTA separately, then regress MTA-attributed online sales against aggregated upper-funnel effort to estimate an interaction term, and feed that in as a Bayesian prior. Method 2, specified here, is structurally different — there is no sequential prior hand-off. Both likelihoods are evaluated inside one model and tied together through parameters they are forced to share. |

---

## 1. Purpose

This spec defines the architecture for a **single Bayesian model that reads the MMM aggregate table and the MTA touchpoint-log table simultaneously**, without merging them into a common row structure, and without a customer-level join key. Channel-level effectiveness is estimated once, jointly, by a parameter structure that both data streams are required to be consistent with.

This is a materially harder system to build, validate, and explain than Method 1. Section 2 says so plainly before the spec proceeds, because building this without first agreeing on the open questions in Section 9 is how joint models silently produce confident, wrong numbers.

---

## 2. Critical Read Before Building This

**[certain]** The phrase "shared parameters" in the brief, taken literally, doesn't work. The MMM stream is a Normal-likelihood regression of *daily aggregate sales* on *spend* — its channel coefficients have units of incremental dollars (or units) per dollar spent. The MTA stream is a Bernoulli/logistic-likelihood model of *individual conversion* on *touchpoint exposure* — its channel coefficients are log-odds per exposure. You cannot set `θ_MMM,c = θ_MTA,c` and have that equality mean anything; the two numbers don't live in the same space. Section 4.4 below replaces the literal-identity claim with the actual mechanism: a shared **latent** channel-effectiveness parameter that each stream maps into its own native scale through a separate, fitted link. That distinction is not pedantic — it is the entire technical content of "Method 2," and a statistically literate reviewer will ask about it first.

**[certain, from inspecting the uploaded files]** `simulated_spend_data.csv` spans 180 daily rows, 2023-01-01 → 2023-06-29. `MTA.csv` — once its `Timestamp` field is parsed correctly (`DD-MM-YYYY`, not `MM-DD-YYYY`) — spans **only 5 calendar days, 2023-01-01 → 2023-01-05**. A joint model that ties MMM and MTA together through a single time-invariant channel-effect parameter is implicitly assuming channel effectiveness didn't move between January and June. A joint model that instead lets effectiveness vary by day has no MTA evidence for 175 of the 180 days. Either way, this is a load-bearing assumption, not a footnote, and it should be agreed with stakeholders before this is built — see Decision D1 in Section 9.

**[likely]** The MTA file's conversion signal is thin: 954 conversion-flagged rows out of 30,399 touchpoint logs, concentrated in only 25 of 599 unique customers. A logistic likelihood fit on that alone is not going to move much in a joint posterior — which is fine, that's what the Bayesian argument in the source paper is actually for (shrinkage toward the prior when evidence is weak), but it means you should expect the MMM stream to dominate the shared posterior unless you deliberately reweight or use a wider MTA-side likelihood (e.g. modeling touchpoint *position-weighted attributed revenue* rather than a sparse binary conversion flag — see Section 4.3).

**[guessing, flagging for your judgment]** "Completely bypassing the need for a customer-level primary key link" is true, but it isn't free — you're trading a deterministic join for joint MCMC over two structurally different likelihoods, which is harder to fit, harder to diagnose when it goes wrong (label-switching-style scale ambiguity between `μ_θ`/`σ_θ` and `μ_φ`/`σ_φ` is a real risk — see Section 6.2), and meaningfully more compute and engineering time than Method 1. Before building this, it's worth being explicit with yourself about what Method 2 buys you over Method 1 that justifies the cost. Section 8 lays out that trade-off directly.

---

## 3. System Architecture

### 3.1 High-Level Diagram

```
                          ┌───────────────────────────────────┐
                          │     Joint Bayesian Posterior       │
                          │   p(z, θ, φ, η | y_MMM, conv_MTA)  │
                          └─────────────────┬───────────────────┘
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    │   Shared Latent Layer: channel effectiveness  │
                    │   z_c ~ Hierarchical Normal(0, τ_channel)      │
                    │   (defined ONLY for the 6 channels that exist  │
                    │    in both streams — see §4.5)                 │
                    └───────────────────────┬───────────────────────┘
                       ┌─────────────────────┴─────────────────────┐
                       │  link: θ_c = g_MMM(z_c)   φ_c = g_MTA(z_c) │
                       └─────┬───────────────────────────────┬─────┘
                             ▼                                 ▼
            ┌─────────────────────────────┐     ┌─────────────────────────────────┐
            │   STREAM A — MMM Likelihood │     │   STREAM B — MTA Likelihood      │
            │   Normal(y_t | μ_t, σ)      │     │   Bernoulli(conv_i | p_i)        │
            │   180 daily rows, 1 brand   │     │   30,399 touchpoints, 599 cust.  │
            │   11 channels (incl. 5      │     │   6 digital channels             │
            │   offline-only)             │     │                                  │
            └─────────────┬───────────────┘     └───────────────┬───────────────────┘
                          │                                     │
                          ▼                                     ▼
              [simulated_spend_data.csv]                  [MTA.csv]
              (read as-is — no merge,                (read as-is — no merge,
               no row-level join)                      no row-level join)
```

The two source tables never touch each other. They are both arguments to one `pm.Model()` call. The only thing connecting them is that `θ_c` and `φ_c` are both deterministic transforms of the same `z_c` draw on every MCMC step.

### 3.2 Component Responsibilities

| Component | Responsibility | Reads | Writes |
|---|---|---|---|
| **Data Layer** | Loads and validates the two source tables independently; no join logic exists here by design | `simulated_spend_data.csv`, `MTA.csv` | Two validated, in-memory frames |
| **Transform Layer (MMM side)** | Adstock + saturation transforms per channel; control-variable construction (price, seasonality) | MMM frame | `X_mmm` design array |
| **Transform Layer (MTA side)** | Position-decay weighting of touchpoints; per-customer exposure aggregation; channel one-hot encoding | MTA frame | `X_mta` exposure array |
| **Shared Latent Layer** | Defines `z_c` and the hierarchical prior over it; defines `g_MMM`, `g_MTA` link functions | Channel crosswalk (§4.5) | `z_c`, `μ_θ, σ_θ, μ_φ, σ_φ` |
| **Inference Engine** | Runs joint NUTS sampling over the full model (both likelihoods + shared layer + nuisance params) | All of the above | `InferenceData` (posterior trace) |
| **Validation Layer** | Per-stream posterior predictive checks, R-hat/divergence diagnostics, prior-sensitivity sweep, holdout backtests | Posterior trace | Diagnostics report |
| **Output Layer** | Unified channel attribution, marginal ROI curves, budget-reallocation recommendation | Posterior trace | Decision-ready output table |
| **Feedback Loop** | Logs realized outcomes from any reallocation as new evidence for the next refit cycle | Output layer + actuals | Updated prior config for next cycle |

---

## 4. Mathematical Specification

### 4.1 Notation

| Symbol | Meaning |
|---|---|
| `t = 1...180` | Day index, MMM stream |
| `i = 1...599` | Customer index, MTA stream |
| `c` | Channel index — see §4.5 for which channels are in scope where |
| `y_t` | MMM-side KPI on day `t` (candidate: `DollarSales`) |
| `conv_i` | MTA-side binary conversion flag for customer `i` |
| `x_{c,t}` | Spend on channel `c`, day `t` (MMM table) |
| `e_{i,c}` | Position-decay-weighted exposure count to channel `c` for customer `i` (derived from MTA table) |
| `z_c` | **Shared latent** relative effectiveness of channel `c` |
| `θ_c` | Channel `c` effect, MMM (elasticity) scale |
| `φ_c` | Channel `c` effect, MTA (log-odds) scale |

### 4.2 Stream A — MMM Likelihood

```
y_t ~ Normal(μ_t, σ_MMM)

μ_t = τ + Σ_c θ_c · Sat( Adstock(x_{c,t}; λ_c) ; κ_c, ν_c )  +  γ_price · price_t  +  γ_season · season_t

Adstock(x_{c,t}; λ_c) = Σ_{l=0}^{L} λ_c^l · x_{c,t-l}            (geometric decay, λ_c ∈ [0,1))
Sat(a; κ_c, ν_c)       = a^ν_c / (κ_c^ν_c + a^ν_c)                (Hill saturation)
```

This is a conventional Bayesian MMM specification — nothing about it is novel to "Method 2." It runs over all **11** channels present in `simulated_spend_data.csv`.

### 4.3 Stream B — MTA Likelihood

Two reasonable specifications exist; pick one and record the decision (Decision D2, §9):

**Option B1 — binary conversion (matches `Is_Conversion` directly):**
```
conv_i ~ Bernoulli(p_i)
logit(p_i) = α + Σ_c φ_c · e_{i,c}
```
Given only 954 positive rows across 25 converting customers, this likelihood will be weakly identified on its own — expected, and not disqualifying, but worth knowing going in.

**Option B2 — attributed revenue as a continuous outcome (uses `Attributed_Revenue`):**
```
rev_i ~ Gamma(μ_i, σ_MTA)        [or LogNormal — revenue is strictly positive]
log(μ_i) = α + Σ_c φ_c · e_{i,c}
```
This uses more of the file (`Attributed_Revenue` is populated for 1,072 rows, not just the 954 conversion flags) and avoids throwing away magnitude information. **[likely]** this is the better-identified choice given the data as uploaded — flagging it as the recommended default, not a unilateral decision.

Exposure weighting by touchpoint position (`First Touch` / `Middle Touch` / `Last Touch`) should use a fitted decay weight, not a fixed heuristic split (e.g. the common 40/20/40 rule) — let the model learn `w_position` rather than hand-set it, since hand-set position weights are exactly the kind of "credit-based algorithm" assumption this whole exercise is trying to move away from.

### 4.4 The Shared-Parameter Mechanism

This is the part of the brief that needs to be made mathematically explicit rather than asserted.

```
# Hierarchical shared latent, non-centered parameterization
z_raw_c ~ Normal(0, 1)
z_c = z_raw_c                       # z_c is the channel's relative effectiveness, unitless

# Stream-specific links — same z_c, different scale/shape per stream
θ_c = exp( μ_θ + σ_θ · z_c )        # MMM elasticity — lognormal keeps it positive
φ_c = μ_φ + σ_φ · z_c                # MTA log-odds — unconstrained sign is fine here

μ_θ, σ_θ, μ_φ, σ_φ  ~ weakly informative priors (HalfNormal on the σ's)
```

A channel that the MTA stream's data pushes toward a high `z_c` is *mechanically* required to also produce a higher MMM-side elasticity `θ_c`, and vice versa, because both are deterministic functions of the same draw. That is the actual content of "forcing the channel efficiency parameters to reconcile" — it is a **correlation imposed through a shared hierarchical prior**, not literal parameter equality, and it is only as strong as `σ_θ` and `σ_φ` let it be. If `σ_θ` and `σ_φ` are estimated to be large, the two streams are effectively independent in practice and you've bought yourself a hard inference problem for very little reconciliation. This is something to check in the posterior, not assume.

Full joint log-posterior the sampler actually evaluates:

```
log p(z, θ, φ, η | y, conv) ∝  LL_MMM(y | θ(z), η_MMM)
                               + LL_MTA(conv | φ(z), η_MTA)
                               + log p(z)
                               + log p(η_MMM)
                               + log p(η_MTA)
```
where `η_MMM` = {`τ, σ_MMM, λ_c, κ_c, ν_c, γ_price, γ_season`} and `η_MTA` = {`α, w_position, σ_MTA`} (if using B2).

### 4.5 Channel Crosswalk & Scope Boundary

The brief's diagram implies all channels are jointly modeled. **They can't be** — `MTA.csv` has no offline touchpoints at all, so there is no likelihood evidence to tie offline channels to anything. The shared latent layer is only defined where both tables actually overlap:

| MMM channel (`simulated_spend_data.csv`) | MTA channel (`MTA.csv`) | In shared latent layer? |
|---|---|---|
| `Search_Spend` | `Paid Search` | ✅ Yes |
| `Social_Spend` | `Social Media` | ✅ Yes |
| `Display_Spend` | `Programmatic Display` | ✅ Yes |
| `Email_Spend` | `Email` | ✅ Yes |
| `Video_Spend` | `Online Video` | ✅ Yes |
| `Affiliate_Spend` | `Affiliate` | ✅ Yes |
| `Broad_Spend` | — none — | ❌ No — MMM-only `θ_c`, independent prior |
| `Print_Spend` | — none — | ❌ No — MMM-only `θ_c`, independent prior |
| `Mag_Spend` | — none — | ❌ No — MMM-only `θ_c`, independent prior |
| `News_Spend` | — none — | ❌ No — MMM-only `θ_c`, independent prior |
| `Outdoor_Spend` | — none — | ❌ No — MMM-only `θ_c`, independent prior |

`Broad_Spend` is also, by a wide margin, the single largest line item in the file ($2.25M of roughly $5.86M total simulated spend — over a third of the budget) and it sits entirely outside the joint structure. Any stakeholder deck built on top of this model needs to say plainly that the "unified" estimate for TV/broadcast is exactly as unified as it was in a standalone MMM — Method 2 does not touch it. If cross-effects from offline-to-online (the TV-drives-Paid-Search story the source paper centers on) are the thing you actually want, that is a **different** mechanism — an interaction term of aggregated upper-funnel effort on online-channel outcomes — which is what Method 1 already implements. Method 2 and Method 1 are not solving the same problem; see Section 8.

---

## 5. Data Specification

### 5.1 MMM Source Schema — `simulated_spend_data.csv`

| Field | Type | Notes |
|---|---|---|
| `Date` | date (string, `YYYY-MM-DD`) | 180 rows, 2023-01-01 → 2023-06-29, daily, no gaps observed |
| `BrandName` | string | Single value (`BrandX`) — not a hierarchical/geo MMM in this file, despite the production model being geo-level; treat this dataset as a single-market pilot, not a like-for-like stand-in |
| `TotalSales` | int | Unit volume |
| `DollarSales` | float | Revenue — **recommended `y_t` target** |
| `PriceRerUnit` | float | Sic — header has a typo (`Rer` not `Per`); carry through as-is for traceability, fix only in a renamed derived column |
| `Broad_Spend` … `Affiliate_Spend` | float ×11 | Daily spend by channel, see §4.5 for crosswalk |

### 5.2 MTA Source Schema — `MTA.csv`

| Field | Type | Notes |
|---|---|---|
| `Customer_ID` | string | 599 unique customers across 30,399 rows |
| `BrandName` | string | `BrandX`, with some null rows |
| `Timestamp` | string, **inconsistent format** | Two raw formats present (`DD-MM-YYYY` and `DD-MM-YYYY HH:MM`); parses to only **5 distinct calendar days** (2023-01-01 → 2023-01-05) |
| `Digital_Channel` | string, categorical | 6 levels (see §4.5); **~9,000 / 30,399 rows (≈30%) are null** |
| `Touchpoint_Position` | string, categorical | `First Touch` / `Middle Touch` / `Last Touch`; **~5,200 rows (≈17%) null** |
| `Is_Conversion` | int (0/1) | 954 positive rows |
| `Attributed_Revenue` | float | Populated for 1,072 rows, $45.65–$163.07 range, mean ≈$99.79 |
| `Unnamed: 7` | — | Fully empty — trailing comma in the source header, drop on load |

### 5.3 Known Data-Quality Issues (engineering checklist before model build)

- [ ] **Timestamp format is mixed within the same column** (`01-01-2023 00:00` vs `02-01-2023`) — must be parsed with an explicit format map, not a single `pd.to_datetime` call with one `dayfirst` setting, or rows will silently fail to parse (a naive `dayfirst=True` parse drops 29,399 of 30,399 rows to `NaT`).
- [ ] **5-day MTA window vs 180-day MMM window** — this is Decision D1 (§9), not a cleaning step, but it has to be resolved before §4.4 can be implemented.
- [ ] **~30% null `Digital_Channel`** — confirm with whoever generated this file whether null means "no exposure recorded" (drop the row) or "non-digital/offline touch logged in the wrong table" (would need re-routing). Don't impute a channel.
- [ ] **`KPI` definition mismatch across streams** — `DollarSales` (MMM, ~$354k/day average) and `Attributed_Revenue` (MTA, ~$100/conversion average) are not interchangeable scales or even necessarily the same revenue definition (net vs. gross — the exact ambiguity the source paper calls out by name as a prerequisite to fix before blending). Confirm both are defined consistently (same revenue recognition, same currency/unit basis) before either enters a likelihood.
- [ ] **`Unnamed: 7`** — artifact of a trailing comma in the CSV header; drop on load, don't carry forward.

### 5.4 Required Pre-Model Transformations

1. MMM: adstock + saturation per channel (parameters `λ_c, κ_c, ν_c` — estimate jointly, don't grid-search and fix, or you reintroduce a non-Bayesian point-estimate step upstream of an otherwise fully Bayesian model).
2. MTA: collapse touchpoint logs to one feature row per customer — `e_{i,c}` = position-decay-weighted count of exposures to channel `c`. Decay weights `w_position` are free parameters fit jointly (§4.3), not hand-set.
3. Both: confirm channel crosswalk (§4.5) before constructing the shared latent layer — get this list signed off, it determines which channels are estimable at all under Method 2.

---

## 6. Inference & Implementation

### 6.1 Stack Recommendation

**[likely]** PyMC (≥5.x) with the default NUTS sampler is adequate at this data scale (180 MMM rows, 599 MTA customers — small by any standard). NumPyro/JAX would only be worth the switch if this scales to a production geo-level MMM crossed with a much larger MTA log (e.g. real HCP-level touchpoint volume), where JIT-compiled NUTS meaningfully outperforms PyMC's default backend. Don't reach for it yet on data this size — the bottleneck here is identifiability, not sampler throughput.

### 6.2 Parameterization Notes

- **Use the non-centered parameterization for `z_c`** (`z_raw_c ~ Normal(0,1)`, then `z_c` derived) exactly as shown in §4.4 — centered hierarchical parameterizations are a known source of divergent transitions in exactly this kind of funnel geometry, and you have few enough channels (6 shared) that the non-centered form costs nothing.
- **Scale ambiguity risk**: `(μ_θ, σ_θ)` and `(μ_φ, σ_φ)` can trade off against the scale of `z_c` itself in ways that are individually unidentified even though `θ_c` and `φ_c` are fine — this shows up as high correlation in the trace between `σ_θ` and `σ_φ`, not necessarily as a divergence. Check pair plots for this specifically; R-hat alone won't catch it.
- **Prior predictive check is mandatory before fitting** given two heterogeneous likelihoods — simulate from the priors alone and confirm the implied `μ_t` range and `p_i` range are both plausible before touching real data.

### 6.3 Illustrative Pseudocode (PyMC-style — not runnable as-is)

```python
import pymc as pm

with pm.Model() as umm_joint:
    # --- Shared latent layer (6 overlapping channels only) ---
    z_raw = pm.Normal("z_raw", 0, 1, shape=n_shared_channels)

    mu_theta  = pm.Normal("mu_theta", 0, 1)
    sigma_theta = pm.HalfNormal("sigma_theta", 1)
    mu_phi    = pm.Normal("mu_phi", 0, 1)
    sigma_phi = pm.HalfNormal("sigma_phi", 1)

    theta_shared = pm.Deterministic("theta_shared", pm.math.exp(mu_theta + sigma_theta * z_raw))
    phi_shared   = pm.Deterministic("phi_shared", mu_phi + sigma_phi * z_raw)

    # --- MMM-only channels (offline, no shared structure) ---
    theta_offline = pm.Lognormal("theta_offline", 0, 1, shape=n_offline_channels)

    # --- Stream A: MMM likelihood ---
    mu_t = mmm_baseline(theta_shared, theta_offline, X_mmm_transformed, controls)
    sigma_mmm = pm.HalfNormal("sigma_mmm", 1)
    pm.Normal("y_obs", mu=mu_t, sigma=sigma_mmm, observed=y_mmm)

    # --- Stream B: MTA likelihood (Option B2, revenue-based) ---
    log_mu_i = mta_baseline(phi_shared, X_mta_exposure)
    sigma_mta = pm.HalfNormal("sigma_mta", 1)
    pm.Gamma("rev_obs", mu=pm.math.exp(log_mu_i), sigma=sigma_mta, observed=rev_mta)

    trace = pm.sample(2000, tune=2000, target_accept=0.95, chains=4)
```

---

## 7. Validation & Diagnostics Plan

| Check | Why it matters here specifically |
|---|---|
| Per-stream posterior predictive checks (separately, not just jointly) | A joint model can fit one stream well by overfitting `z_c` to it and effectively ignoring the other — check each stream's PPC independently |
| R-hat < 1.01, zero divergences | Standard, but watch divergences in `z_raw`, `sigma_theta`, `sigma_phi` specifically — see §6.2 |
| Pair plots: `sigma_theta` × `sigma_phi`, `mu_theta` × `mu_phi` | Targeted check for the scale-ambiguity risk flagged in §6.2 |
| Prior sensitivity sweep on `σ_θ`, `σ_φ` | The source paper itself names prior-driven manipulation as a known, fair critique of Bayesian MMM/UMM — show the posterior is stable across a reasonable prior range before presenting a single point estimate to stakeholders |
| Holdout: last N days for MMM, held-out customer set for MTA | Standard backtest, run on each stream independently |
| Independent experiment (geo-lift / holdout test), if available | The one calibration check that doesn't depend on either model being right — per the source paper's "Experiments" leg, this is what the shared-latent estimate should ultimately be checked against, not just internal cross-validation |

---

## 8. Method 2 vs. Method 1 — Decision Table

| | **Method 1** (sequential prior-transfer, already built) | **Method 2** (this spec) |
|---|---|---|
| Mechanism | Fit MMM, fit MTA, regress MTA-attributed online sales on aggregated upper-funnel effort, feed result in as a prior | Single joint model, one MCMC run, two likelihoods tied by a shared latent parameter |
| Customer-level join required? | No | No |
| Captures offline→online interaction (TV lifts Paid Search)? | **Yes — this is its core mechanism** | **No** — offline channels sit outside the shared layer entirely (§4.5) |
| Computational complexity | Two sequential, individually tractable fits | One joint fit over heterogeneous likelihoods; harder to diagnose |
| Data requirement to be well-identified | Each model needs to be individually well-specified | Additionally needs temporal/coverage overlap between the two source tables, which the current files don't have (§2) |
| Interpretability for a stakeholder review | High — each step maps to a paper-documented best practice | Lower — "shared latent effectiveness" requires more explanation than "prior transfer" |
| What it's actually good for | Estimating cross-channel (especially upper-to-lower-funnel) interaction effects | Pooling statistical strength across two granularities for the *same* set of addressable digital channels, when no interaction term is the goal |

**[likely]** Given that the offline→online interaction story is the headline result in both the source paper's case study and your existing `unified_blueprint` deck, Method 1 is doing the thing you've already told stakeholders UMM does. Method 2 answers a narrower, different question — "do my digital-channel MMM elasticities and MTA log-odds agree, and can I pool them?" — which is a legitimate question, but worth confirming is actually the question before investing in the harder build.

---

## 9. Open Decisions Requiring Sign-off

| ID | Decision | Options | Recommendation |
|---|---|---|---|
| D1 | How to handle the 5-day MTA / 180-day MMM coverage mismatch | (a) assume time-invariant `z_c` over the full window and flag it explicitly as an assumption; (b) restrict the joint model to the 5 overlapping days only, with MMM also subset; (c) wait for a longer MTA extract before building Method 2 | **[guessing]** — this is a business call on data availability, not a modeling one; flagging (c) as the cleanest if a longer MTA pull is feasible |
| D2 | MTA likelihood target | Option B1 (binary `Is_Conversion`) vs Option B2 (`Attributed_Revenue`, Gamma/LogNormal) | **[likely]** B2 — better identified given the sparsity in B1 |
| D3 | Common KPI definition across streams | Confirm `DollarSales` (MMM) and `Attributed_Revenue` (MTA) are on a consistent revenue basis | Must be resolved before §4 is implemented, not after |
| D4 | Scope of "unified" claim in any resulting stakeholder deck | Explicit caveat that offline channels (`Broad`, `Print`, `Mag`, `News`, `Outdoor` — ~38% of simulated spend) are outside the shared-parameter structure under Method 2 | Recommend stating this plainly rather than letting "unified" imply full coverage |

---

## 10. Glossary

- **Shared latent parameter (`z_c`)** — a single per-channel random variable that both likelihoods' channel coefficients are deterministic functions of; the actual mechanism behind "reconciling" the two streams.
- **Stream** — one of the two data sources (MMM or MTA) and its associated likelihood, transform layer, and nuisance parameters.
- **Crosswalk** — the explicit mapping (§4.5) of which MMM channels correspond to which MTA channels; defines the boundary of the shared latent layer.
- **Non-centered parameterization** — a reparameterization of hierarchical priors (`z_raw ~ N(0,1)`, then transformed) that improves NUTS sampling geometry; not optional at this model's funnel structure.
