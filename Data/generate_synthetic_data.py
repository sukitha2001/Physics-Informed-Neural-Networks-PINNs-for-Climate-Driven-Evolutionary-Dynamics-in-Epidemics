"""
generate_synthetic_data.py
──────────────────────────
Generates synthetic monthly dengue data for Colombo using an SIRS model.

Why SIRS, not SIR?
──────────────────
The original SIR model depletes susceptibles to zero over ~5 years.
Cases then collapse permanently to zero, producing a near-flat test region
that breaks the combined training/testing pipeline.

Dengue biologically justifies SIRS:
  - 4 serotypes; cross-serotype immunity wanes after ~2 years (delta = 1/24)
  - Births continuously replenish susceptibles (mu = Sri Lanka birth rate)
  These two terms sustain recurring annual epidemic cycles indefinitely.

SIRS equations (Euler integration):
  dS/dt =  mu*(1-S) + delta*R  -  beta(t)*S*I
  dI/dt =  beta(t)*S*I         -  (gamma + mu)*I
  dR/dt =  gamma*I             -  (delta + mu)*R

Calibration targets (2010-2020 Colombo data):
  monthly cases: mean ≈ 1119,  std ≈ 1022,  max ≈ 7471
  Cases_Scaled:  mean ≈ 0.146, std ≈ 0.137

Two mechanisms achieve realistic variance:
  1. N_POP_EFFECTIVE ≈ 10 000 gives mean cases ≈ 1185 (SIRS I-fraction × N)
  2. Super-outbreak amplifier: ~2 events/decade that 3-6× normal cases for
     2-4 consecutive months (matches the 2017 / 2019 patterns in real data)
"""

import argparse
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (derived from real Colombo 2010-2020 data)
# ─────────────────────────────────────────────────────────────────────────────

TEMP_MONTHLY_MEAN = np.array([
    26.42, 27.22, 27.81, 27.64, 27.24, 26.86,
    26.65, 26.42, 26.28, 26.23, 26.02, 25.81
])
RAIN_MONTHLY_MEAN = np.array([
    71.36, 91.52, 110.78, 190.68, 302.37, 262.59,
    200.56, 217.21, 264.44, 356.31, 333.04, 230.24
])

TEMP_NOISE_STD = 0.80   # °C   — matches real std
RAIN_NOISE_STD = 80.0   # mm   — matches real rain variability

# SIRS epidemiological parameters
N_POP_EFFECTIVE = 10_000  # calibrated so mean simulated cases ≈ real mean (1119)
GAMMA   = 0.14            # recovery rate: ~7-month infectious period
MU      = 0.0014          # monthly birth = death rate (Sri Lanka ≈ 1.7% annual)
DELTA   = 0.042           # waning immunity: ~2-year cross-protection period

# Beta (transmission) — weather-driven, calibrated to lag correlation structure
BETA_BASE        = 0.18
BETA_RAIN_LAG2_W = 0.25   # rain 2 months ago (strongest signal: r=0.24 in real data)
BETA_RAIN_LAG1_W = 0.12   # rain 1 month ago  (r=0.18)
BETA_TEMP_W      = 0.04   # current temperature (weak: r=-0.04)

# Super-outbreak parameters — replicates extreme years like 2017 & 2019
OUTBREAK_PROB_PER_YEAR  = 0.20   # ~2 outbreaks per decade
OUTBREAK_DURATION_RANGE = (2, 4) # months the amplification lasts
OUTBREAK_AMP_RANGE      = (3.0, 6.5) # multiplier on top of normal cases

# Initial conditions (endemic start — not naive population)
S0, I0 = 0.70, 0.005
R0_SIR = 1.0 - S0 - I0


# ─────────────────────────────────────────────────────────────────────────────
# WEATHER GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_weather(n_months: int, seed: int):
    rng    = np.random.default_rng(seed)
    months = np.arange(n_months) % 12

    temp = TEMP_MONTHLY_MEAN[months] + rng.normal(0, TEMP_NOISE_STD, n_months)
    rain = RAIN_MONTHLY_MEAN[months] + rng.normal(0, RAIN_NOISE_STD, n_months)

    # Extreme rainfall events (~5% of months)
    extreme = rng.random(n_months) < 0.05
    rain[extreme] *= rng.uniform(2.0, 3.0, extreme.sum())

    return np.clip(temp, 23.0, 32.0).astype(np.float32), \
           np.clip(rain, 5.0, 900.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# SIRS SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_beta(t: int, rain_sc: np.ndarray, temp_sc: np.ndarray) -> float:
    rl2 = rain_sc[t - 2] if t >= 2 else 0.0
    rl1 = rain_sc[t - 1] if t >= 1 else 0.0
    return max(BETA_BASE
               + BETA_RAIN_LAG2_W * rl2
               + BETA_RAIN_LAG1_W * rl1
               + BETA_TEMP_W      * temp_sc[t], 0.01)


def build_outbreak_mask(n_months: int, seed: int) -> np.ndarray:
    """
    Creates a multiplier array (1.0 = normal, >1 = outbreak amplification).
    Outbreaks are placed randomly but cluster for 2-4 months like real events.
    """
    rng = np.random.default_rng(seed + 99)
    mask = np.ones(n_months, dtype=np.float32)

    n_years = n_months // 12
    for year in range(n_years):
        if rng.random() < OUTBREAK_PROB_PER_YEAR:
            start_month = year * 12 + int(rng.integers(0, 12))
            duration    = int(rng.integers(*OUTBREAK_DURATION_RANGE))
            amplitude   = rng.uniform(*OUTBREAK_AMP_RANGE)
            end_month   = min(start_month + duration, n_months)
            mask[start_month:end_month] = amplitude
    return mask


def run_sirs(n_months: int, temp_raw: np.ndarray, rain_raw: np.ndarray, seed: int):
    """
    Integrates SIRS ODEs (Euler method) with weather-driven beta and
    super-outbreak amplification. Returns monthly reported case counts.

    The mu and delta terms are the critical additions over plain SIR:
      mu    — birth inflow replenishes S continuously
      delta — waning immunity returns R back to S every ~2 years
    Together they sustain endemic cycling for any number of years.
    """
    rng = np.random.default_rng(seed + 1)

    def sc(arr):
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo + 1e-8)

    temp_sc = sc(temp_raw)
    rain_sc = sc(rain_raw)

    outbreak_amp = build_outbreak_mask(n_months, seed)

    S = np.zeros(n_months)
    I = np.zeros(n_months)
    R = np.zeros(n_months)
    S[0], I[0], R[0] = S0, I0, R0_SIR

    for t in range(1, n_months):
        beta = compute_beta(t, rain_sc, temp_sc)
        St, It, Rt = S[t-1], I[t-1], R[t-1]

        new_inf = beta * St * It
        dS = MU*(1 - St) + DELTA*Rt - new_inf
        dI = new_inf - (GAMMA + MU)*It
        dR = GAMMA*It - (DELTA + MU)*Rt

        S[t] = np.clip(St + dS, 0, 1)
        I[t] = np.clip(It + dI, 0, 1)
        R[t] = np.clip(Rt + dR, 0, 1)

        # Renormalise to prevent Euler drift
        total = S[t] + I[t] + R[t]
        if total > 0:
            S[t] /= total; I[t] /= total; R[t] /= total

    # Convert to case counts
    base_cases  = I * N_POP_EFFECTIVE
    amplified   = base_cases * outbreak_amp
    noise       = rng.normal(0, amplified * 0.08)
    cases       = np.round(np.clip(amplified + noise, 1, None)).astype(int)
    return cases


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def minmax(arr):
    lo, hi = arr.min(), arr.max()
    return ((arr - lo) / (hi - lo + 1e-8)).astype(np.float32)


def generate_dataset(n_years: int   = 25,
                     start_year: int = 2000,
                     seed: int       = 42,
                     city: str       = "Colombo",
                     output_path: str = "Synthetic_Dataset.csv") -> pd.DataFrame:

    n_months = n_years * 12
    print(f"Generating {n_months} months ({n_years} years) — SIRS model")
    print(f"  Seed={seed}  City={city}  Start={start_year}")

    temp, rain = generate_weather(n_months, seed)
    cases      = run_sirs(n_months, temp, rain, seed)

    dates = pd.date_range(start=f"{start_year}-01-01", periods=n_months, freq="MS")
    df = pd.DataFrame({
        "Date"         : dates.strftime("%Y-%m-%d"),
        "City"         : city,
        "Time_Step"    : np.arange(n_months, dtype=float),
        "Temperature"  : np.round(temp, 4),
        "Rainfall"     : np.round(rain, 2),
        "Dengue_Cases" : cases,
        "Temp_Scaled"  : np.round(minmax(temp),  6),
        "Rain_Scaled"  : np.round(minmax(rain),  6),
        "Cases_Scaled" : np.round(minmax(cases.astype(np.float32)), 6),
    })

    df.to_csv(output_path, index=False)
    print(f"Saved → {output_path}")

    cs = df["Cases_Scaled"].values
    print("\n── Summary ───────────────────────────────────────────────────")
    print(f"  Rows            : {len(df)}")
    print(f"  Cases raw       : mean={cases.mean():.0f}  std={cases.std():.0f}  "
          f"max={cases.max()}  (real: mean=1119, std=1022, max=7471)")
    print(f"  Cases_Scaled    : mean={cs.mean():.4f}  std={cs.std():.4f}  "
          f"(real: mean=0.1464, std=0.1374)")
    years_with_peak = sum(
        df.groupby(pd.to_datetime(df.Date).dt.year)["Dengue_Cases"].max() > cases.mean()
    )
    print(f"  Years with peak : {years_with_peak}/{n_years}  (all {n_years} expected)")
    print("──────────────────────────────────────────────────────────────\n")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_years",    type=int, default=25)
    parser.add_argument("--start_year", type=int, default=2000)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--output",     type=str, default="Synthetic_Dataset.csv")
    args = parser.parse_args()
    generate_dataset(n_years=args.n_years, start_year=args.start_year,
                     seed=args.seed, output_path=args.output)