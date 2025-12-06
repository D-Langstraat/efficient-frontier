"""
efficient_frontier.py

This script does all of the following in one go:

1) Reads historical PRICE data from a CSV file called "prices.csv".
   - The file must be in the SAME folder as this script.
   - The data should be in "wide" format, for example:

       Date,BA,MCD,TRV,CVX,MSFT
       12/01/2020,214.06,192.58,126.66,68.28,213.42
       01/01/2021,194.19,186.53,123.76,68.89,222.57
       ...
       (dates go from oldest to newest OR newest to oldest;
        the script will sort by date)

2) Converts prices to monthly RETURNS for each stock:
   - return_t = (Price_t / Price_(t-1)) - 1

3) Calculates:
   - Mean return for each stock (average monthly return)
   - Covariance matrix of stock returns (risk & co-movements)

4) Uses Markowitz mean–variance optimization to:
   - Generate many RANDOM portfolios (random weights for each stock)
   - Compute the MINIMUM-VARIANCE portfolio for many target returns
     (this traces out the "Markowitz bullet" / efficient frontier)
   - Find the MAX-SHARPE (tangency) portfolio

5) Plots TWO figures:
   - One WITHOUT SHORTING (long-only weights: each between 0 and 1)
   - One WITH SHORTING allowed (weights can be negative or > 1)

   Each figure shows:
   - A cloud of random portfolios
   - The bottom part of the frontier (inefficient) in gray dashed
   - The top part of the frontier (efficient) in orange
   - A black star at the max-Sharpe portfolio
   - A text box with the annual stats and weights of the max-Sharpe portfolio

6) Automatically saves PNG files (one for each case) to:
   - The user's Desktop, if it exists
   - Otherwise, the folder containing this script

   Filenames:
   - efficient_frontier_without_shorting.png
   - efficient_frontier_with_shorting.png

HOW TO RUN (from PowerShell, Git Bash, or CMD):

   cd C:\(your path to this script)
   python "efficient_frontier.py"

REQUIREMENTS:
   - Python 3
   - numpy
   - pandas
   - matplotlib
   - scipy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pathlib import Path

# =======================
# CONFIGURATION SECTION
# =======================
# Think of this section as "knobs" you can turn without touching the math.

# Name of your CSV file in the SAME folder as this script.
# If you want to use another filename (e.g., "my_prices.csv"),
# just change this string.
CSV_PATH = "prices.csv"

# Annual risk-free rate (for example, 3.8% = 0.038).
# This is used to compute the Sharpe ratio.
RISK_FREE_RATE_ANNUAL = 0.038

# Number of return periods per year in your data.
# For monthly data: 12
# For daily data:   252 (trading days)  <-- not used here, but good to know.
PERIODS_PER_YEAR = 12

# Convert annual risk-free rate to per-period (monthly) rate.
# We use geometric conversion, not simple division:
#   (1 + R_annual)^(1/12) - 1
RISK_FREE_RATE_PERIOD = (1.0 + RISK_FREE_RATE_ANNUAL) ** (1.0 / PERIODS_PER_YEAR) - 1.0

# How many random portfolios to simulate.
# More portfolios = prettier cloud, but slower.
N_RANDOM_PORTFOLIOS = 5000

# Figure saving options.
SAVE_FIGURES = True
FIGURE_DPI = 300

# Determine where to save the PNG files.
# 1) Try the user's Desktop (very convenient for viewing).
# 2) If that does not exist, fall back to the folder containing this script.
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    # __file__ may not exist in some interactive environments;
    # in that case, use the current working directory.
    SCRIPT_DIR = Path.cwd()

DESKTOP_DIR = Path.home() / "Desktop"
if DESKTOP_DIR.is_dir():
    SAVE_DIR = DESKTOP_DIR
else:
    SAVE_DIR = SCRIPT_DIR


# =======================
# DATA HANDLING FUNCTIONS
# =======================

def load_price_data(path: str) -> pd.DataFrame:
    """
    Load price data from a CSV file and convert the Date column to a DateTime index.

    What this function does, step-by-step:
    1) Reads the CSV file.
    2) Cleans column names (removes whitespace and BOM characters).
    3) If there is a "Date" column:
       - Converts it to actual datetime objects.
       - Drops rows where the date could not be parsed.
       - Sorts by date from oldest to newest.
       - Sets the date as the index.
    4) Drops empty rows (if any).

    Returns:
        A pandas DataFrame with:
        - Index: dates (if "Date" existed)
        - Columns: stocks (e.g., BA, MCD, TRV, ...)
        - Values: prices
    """
    # "utf-8-sig" automatically strips the UTF-8 BOM if present,
    # which prevents weird header names like "ï»¿Date".
    df = pd.read_csv(path, encoding="utf-8-sig")

    # Clean up column names: remove spaces and BOM characters.
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]

    if "Date" in df.columns:
        # Ensure "Date" column is string-like and trim spaces.
        df["Date"] = df["Date"].astype(str).str.strip()

        # Let pandas guess the date format (month/day/year vs year-month-day, etc.).
        # We do NOT set dayfirst=True because your data is US-style (MM/DD/YYYY).
        df["Date"] = pd.to_datetime(
            df["Date"],
            infer_datetime_format=True,
            dayfirst=False,
            errors="coerce",  # rows that cannot parse become NaT
        )

        # Drop any rows where the date could not be parsed.
        df = df[~df["Date"].isna()]

        # Sort from oldest date to newest date.
        df = df.sort_values("Date")

        # Use the Date column as the DataFrame's index.
        df = df.set_index("Date")

    # Remove rows that are completely empty.
    df = df.dropna(how="all")

    return df


def compute_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert PRICE data into RETURN data.

    Given a DataFrame of prices (rows = dates, columns = stocks), we:
    1) Remove any currency symbols or commas from the data.
    2) Convert everything to numeric (non-numeric entries become NaN).
    3) Drop any columns that are entirely NaN (no valid numbers).
    4) Sort the index (dates) from oldest to newest (just in case).
    5) Compute simple percentage returns using pct_change():
          return_t = (Price_t / Price_(t-1)) - 1
    6) Drop any rows where ALL returns are NaN.

    Returns:
        A DataFrame of returns, same shape as prices minus 1 row.
    """
    # Make a copy so we don't accidentally modify the original.
    df = price_df.copy()

    # Remove symbols like $ and commas that might appear in the raw CSV.
    df = df.replace(r"[\$,]", "", regex=True)
    df = df.replace(r"%", "", regex=True)

    # Convert everything to numeric. Non-numeric values become NaN.
    numeric_df = df.apply(pd.to_numeric, errors="coerce")

    # Drop columns that have no numeric data at all.
    numeric_df = numeric_df.dropna(axis=1, how="all")

    # Ensure dates are in ascending order.
    numeric_df = numeric_df.sort_index()

    # Compute percentage returns from prices.
    returns = numeric_df.pct_change()

    # Drop rows where all assets have NaN returns (usually the first row).
    returns = returns.dropna(how="all")

    return returns


# =======================
# PORTFOLIO MATH HELPERS
# =======================

def portfolio_performance(weights: np.ndarray,
                          mean_returns: np.ndarray,
                          cov_matrix: np.ndarray) -> tuple[float, float]:
    """
    Compute the expected return and volatility (standard deviation) of a portfolio.

    Inputs:
        weights      : 1D numpy array of portfolio weights (one per stock).
        mean_returns : 1D numpy array of mean returns (same order as weights).
        cov_matrix   : 2D numpy array covariance matrix of returns.

    Returns:
        (port_return, port_vol)
        where:
            port_return = sum(weights_i * mean_return_i)
            port_vol    = sqrt(weights^T * Cov * weights)
    """
    # Expected return = dot product of weights and mean returns.
    port_return = float(np.dot(weights, mean_returns))

    # Portfolio variance = w^T * Cov * w
    port_var = float(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Portfolio volatility (standard deviation) = sqrt(variance).
    port_vol = np.sqrt(port_var)

    return port_return, port_vol


def min_vol_for_target_return(target_return: float,
                              mean_returns: np.ndarray,
                              cov_matrix: np.ndarray,
                              allow_short: bool):
    """
    For a given TARGET RETURN, find the portfolio with MINIMUM VOLATILITY.

    Mathematically:
        minimize   sigma_p(w)
        subject to sum_i w_i = 1
                   sum_i w_i * mean_return_i = target_return
                   and, depending on allow_short:
                     - if allow_short is False: 0 <= w_i <= 1 (long-only)
                     - if allow_short is True : w_i is unbounded above/below (but must still sum to 1)

    This uses the SLSQP optimizer from SciPy.
    """
    num_assets = len(mean_returns)

    # Objective: portfolio volatility given weights vector w.
    def portfolio_volatility(weights):
        return portfolio_performance(weights, mean_returns, cov_matrix)[1]

    # Constraints:
    # 1) sum of weights = 1  (full investment, no leftover cash)
    # 2) expected return of portfolio = target_return
    constraints = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "eq", "fun": lambda w, tr=target_return: np.dot(w, mean_returns) - tr},
    )

    # Bounds on each weight:
    if allow_short:
        # Shorting allowed: weights can be negative or > 1.
        bounds = tuple((None, None) for _ in range(num_assets))
    else:
        # Long-only: each weight is between 0 and 1.
        bounds = tuple((0.0, 1.0) for _ in range(num_assets))

    # Initial guess: equal weights.
    x0 = np.ones(num_assets) / num_assets

    # Run the optimization.
    result = minimize(
        portfolio_volatility,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    return result


def efficient_frontier(mean_returns: np.ndarray,
                       cov_matrix: np.ndarray,
                       allow_short: bool,
                       n_points: int = 80,
                       ret_min: float | None = None,
                       ret_max: float | None = None):
    """
    Build the efficient frontier in RETURN-VOLATILITY space (per-period).

    The idea:
    - Pick a grid of target returns between ret_min and ret_max.
    - For each target return, find the minimum-volatility portfolio.
    - The collection of (vol, return) points traces out the frontier.

    Inputs:
        mean_returns : array of mean returns per asset (per period).
        cov_matrix   : covariance matrix.
        allow_short  : whether shorting is allowed (affects constraints).
        n_points     : how many target returns to try.
        ret_min      : minimum target return (if None, use min(mean_returns)).
        ret_max      : maximum target return (if None, use max(mean_returns)).

    Returns:
        frontier_vols_period  : 1D array of volatilities (per period).
        frontier_rets_period  : 1D array of returns (per period).
        frontier_weights_list : list of weight vectors that generate each point.
    """
    mean_returns = np.asarray(mean_returns, dtype=float)

    # If user did not specify ret_min/ret_max, use the range of asset means.
    if ret_min is None:
        ret_min = float(np.min(mean_returns))
    if ret_max is None:
        ret_max = float(np.max(mean_returns))

    # Create a grid of target returns between ret_min and ret_max.
    target_returns = np.linspace(ret_min, ret_max, n_points)

    frontier_vols = []
    frontier_rets = []
    frontier_weights = []

    # For each desired target return, find the minimum-vol portfolio.
    for tr in target_returns:
        res = min_vol_for_target_return(tr, mean_returns, cov_matrix, allow_short)
        if res.success:
            vol_period = portfolio_performance(res.x, mean_returns, cov_matrix)[1]
            frontier_vols.append(vol_period)
            frontier_rets.append(tr)
            frontier_weights.append(res.x)

    return (
        np.array(frontier_vols),
        np.array(frontier_rets),
        frontier_weights,
    )


def max_sharpe_ratio(mean_returns: np.ndarray,
                     cov_matrix: np.ndarray,
                     risk_free_rate_period: float,
                     allow_short: bool):
    """
    Find the portfolio that MAXIMIZES the Sharpe ratio (per period):

        Sharpe(w) = [E(R_p(w)) - R_f] / sigma_p(w)

    where:
        E(R_p(w)) is the expected portfolio return given weights w,
        R_f        is the per-period risk-free rate,
        sigma_p(w) is the portfolio volatility.

    Subject to:
        sum_i w_i = 1
        and bounds:
            - if allow_short is False: 0 <= w_i <= 1
            - if allow_short is True : no bounds on w_i (but they still must sum to 1)
    """
    num_assets = len(mean_returns)

    def neg_sharpe(weights):
        # Compute portfolio return and volatility for given weights.
        ret_period, vol_period = portfolio_performance(weights, mean_returns, cov_matrix)
        if vol_period == 0:
            # Avoid division by zero.
            return np.inf
        # Negative Sharpe because we want to MAXIMIZE Sharpe but
        # the optimizer only MINIMIZES things.
        return -(ret_period - risk_free_rate_period) / vol_period

    # Constraint: fully invested (weights sum to 1).
    constraints = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    )

    # Bounds on each asset's weight.
    if allow_short:
        bounds = tuple((None, None) for _ in range(num_assets))
    else:
        bounds = tuple((0.0, 1.0) for _ in range(num_assets))

    # Start from equal weights.
    x0 = np.ones(num_assets) / num_assets

    # Run the optimization.
    result = minimize(
        neg_sharpe,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    return result


def simulate_random_portfolios(n_portfolios: int,
                               mean_returns: np.ndarray,
                               cov_matrix: np.ndarray,
                               risk_free_rate_period: float,
                               allow_short: bool):
    """
    Create a cloud of random portfolios to visualize the feasible region.

    For each random portfolio:
    1) Draw random weights.
       - If shorting is NOT allowed: draw from Uniform(0,1).
       - If shorting IS allowed  : draw from Normal(0,1) (can be negative).
    2) Normalize weights so they sum to 1.
    3) Compute portfolio return and volatility.
    4) Compute its Sharpe ratio.

    Inputs:
        n_portfolios         : how many random portfolios to simulate.
        mean_returns         : mean returns (per period).
        cov_matrix           : covariance matrix.
        risk_free_rate_period: risk-free rate (per period).
        allow_short          : whether to allow negative weights.

    Returns:
        returns_array : array of portfolio returns (per period).
        vols_array    : array of portfolio volatilities (per period).
        sharpes_array : array of portfolio Sharpe ratios (per period).
    """
    num_assets = len(mean_returns)
    rets = []
    vols = []
    sharpes = []

    for _ in range(n_portfolios):
        if allow_short:
            # Normal draws allow negative and positive values.
            weights = np.random.randn(num_assets)
        else:
            # Uniform draws in [0, 1], we will normalize to sum to 1.
            weights = np.random.rand(num_assets)

        # Normalize so that the sum of weights = 1.
        weights = weights / np.sum(weights)

        ret_p, vol_p = portfolio_performance(weights, mean_returns, cov_matrix)

        # Compute Sharpe ratio. If vol is zero, set Sharpe to NaN to avoid errors.
        sharpe_p = (ret_p - risk_free_rate_period) / vol_p if vol_p > 0 else np.nan

        rets.append(ret_p)
        vols.append(vol_p)
        sharpes.append(sharpe_p)

    return np.array(rets), np.array(vols), np.array(sharpes)


# =======================
# PLOTTING FUNCTION
# =======================

def plot_case(mean_returns_period: np.ndarray,
              cov_matrix_period: np.ndarray,
              asset_names: list[str],
              allow_short: bool):
    """
    Create ONE efficient frontier plot for a single case (either with or without shorting).

    This function:
    - Simulates random portfolios.
    - Finds the max-Sharpe portfolio.
    - Constructs the efficient frontier.
    - Splits the frontier into:
        * bottom branch (inefficient) -> gray dashed
        * top branch (efficient)      -> orange solid
    - Draws a star at the max-Sharpe point.
    - Shows a text box with the annual stats and weights of the max-Sharpe portfolio.
    - Automatically saves the figure as a PNG.
    """
    # -----------------------
    # 1) Random portfolios
    # -----------------------
    rand_rets_m, rand_vols_m, _ = simulate_random_portfolios(
        N_RANDOM_PORTFOLIOS,
        mean_returns_period,
        cov_matrix_period,
        RISK_FREE_RATE_PERIOD,
        allow_short,
    )

    # -----------------------
    # 2) Max-Sharpe portfolio
    # -----------------------
    max_sharpe_res = max_sharpe_ratio(
        mean_returns_period,
        cov_matrix_period,
        RISK_FREE_RATE_PERIOD,
        allow_short,
    )

    if max_sharpe_res.success:
        ms_weights = max_sharpe_res.x
        ms_ret_m, ms_vol_m = portfolio_performance(
            ms_weights,
            mean_returns_period,
            cov_matrix_period,
        )
        ms_sharpe_m = (ms_ret_m - RISK_FREE_RATE_PERIOD) / ms_vol_m if ms_vol_m > 0 else np.nan

        # Convert monthly stats to annual stats for display.
        ms_ret_ann = ms_ret_m * PERIODS_PER_YEAR
        ms_vol_ann = ms_vol_m * np.sqrt(PERIODS_PER_YEAR)
        ms_sharpe_ann = ms_sharpe_m * np.sqrt(PERIODS_PER_YEAR)
    else:
        # If the optimizer fails (rare), we treat this as "no max Sharpe found".
        ms_weights = None
        ms_ret_ann = ms_vol_ann = ms_sharpe_ann = np.nan

    # -----------------------
    # 3) Efficient frontier
    # -----------------------
    base_ret_min = float(np.min(mean_returns_period))
    base_ret_max = float(np.max(mean_returns_period))

    ret_min = base_ret_min
    ret_max = base_ret_max

    # For the case with shorting, the max-Sharpe portfolio can have a higher
    # expected return than any single asset. We extend ret_max slightly above it
    # so we see the full "top arc" of the frontier.
    if ms_weights is not None and np.isfinite(ms_ret_ann):
        # Convert annual tangency return back to per-period and then extend it by 5%.
        ret_max = max(base_ret_max, ms_ret_ann / PERIODS_PER_YEAR) * 1.05

    front_vols_m, front_rets_m, _ = efficient_frontier(
        mean_returns_period,
        cov_matrix_period,
        allow_short,
        n_points=120,
        ret_min=ret_min,
        ret_max=ret_max,
    )

    # -----------------------
    # 4) Convert everything to annual units for plotting.
    # -----------------------
    rand_rets_ann = rand_rets_m * PERIODS_PER_YEAR
    rand_vols_ann = rand_vols_m * np.sqrt(PERIODS_PER_YEAR)
    front_rets_ann = front_rets_m * PERIODS_PER_YEAR
    front_vols_ann = front_vols_m * np.sqrt(PERIODS_PER_YEAR)

    # Split frontier at the minimum-volatility point.
    # Everything up to that point is the bottom branch (inefficient),
    # and everything from that point up is the top branch (efficient).
    if front_vols_ann.size > 0:
        min_idx = int(np.argmin(front_vols_ann))
        lower_vols_ann = front_vols_ann[:min_idx + 1]
        lower_rets_ann = front_rets_ann[:min_idx + 1]
        upper_vols_ann = front_vols_ann[min_idx:]
        upper_rets_ann = front_rets_ann[min_idx:]
    else:
        lower_vols_ann = np.array([])
        lower_rets_ann = np.array([])
        upper_vols_ann = np.array([])
        upper_rets_ann = np.array([])

    # -----------------------
    # 5) Build the plot
    # -----------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    # Leave extra space on the right for the legend and text box.
    fig.subplots_adjust(right=0.75)

    # Random portfolio cloud
    ax.scatter(
        rand_vols_ann,
        rand_rets_ann,
        alpha=0.3,
        label="Random portfolios",
        color="tab:blue",
    )

    # Bottom branch: inefficient part of the frontier (dominated portfolios).
    if lower_vols_ann.size > 0:
        ax.plot(
            lower_vols_ann,
            lower_rets_ann,
            linestyle="--",
            color="tab:gray",
            label="Inefficient frontier",
        )

    # Top branch: efficient part of the frontier.
    if upper_vols_ann.size > 0:
        ax.plot(
            upper_vols_ann,
            upper_rets_ann,
            linewidth=2,
            color="tab:orange",
            label="Efficient frontier",
        )

    # Mark the max-Sharpe portfolio with a star, if it exists.
    if ms_weights is not None:
        ax.scatter(
            ms_vol_ann,
            ms_ret_ann,
            marker="*",
            s=200,
            label=f"Max Sharpe (Annual SR={ms_sharpe_ann:.2f})",
            color="black",
        )

    ax.set_xlabel("Annualized volatility (standard deviation)")
    ax.set_ylabel("Annualized expected return")

    # Label and slug depend on whether shorting is allowed.
    title_label = "With Shorting" if allow_short else "Without Shorting"
    slug_label = "with_shorting" if allow_short else "without_shorting"
    ax.set_title(f"Efficient Frontier ({title_label})")

    ax.grid(True)

    # Place the legend just outside the plot area, top-right.
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )

    # Build a text box with max-Sharpe stats and weights, and place it under the legend.
    if ms_weights is not None:
        lines = []
        lines.append("Max Sharpe portfolio (annual):")
        lines.append(f"  Return:    {ms_ret_ann*100:.2f}%")
        lines.append(f"  Std. Dev.: {ms_vol_ann*100:.2f}%")
        lines.append(f"  Sharpe:    {ms_sharpe_ann:.2f}")
        lines.append("")
        lines.append("Max Sharpe weights:")
        for name, w in zip(asset_names, ms_weights):
            lines.append(f"  {name}: {w*100:5.2f}%")
        weights_text = "\n".join(lines)

        ax.text(
            1.02,
            0.60,  # y-position in Axes coordinates (0 bottom, 1 top)
            weights_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round", alpha=0.15),
        )

    # -----------------------
    # 6) Axis limits
    # -----------------------
    # We want to zoom the axes around the frontier and the max-Sharpe point,
    # and NOT let extreme random portfolios blow up the scale.

    vol_core_list = [front_vols_ann]
    ret_core_list = [front_rets_ann]

    if ms_weights is not None and np.isfinite(ms_vol_ann) and np.isfinite(ms_ret_ann):
        vol_core_list.append(np.array([ms_vol_ann]))
        ret_core_list.append(np.array([ms_ret_ann]))

    vol_core = np.concatenate([v for v in vol_core_list if v.size > 0])
    ret_core = np.concatenate([r for r in ret_core_list if r.size > 0])

    if vol_core.size > 0:
        x_min = 0.0
        x_max = vol_core.max() * 1.10  # 10% padding to the right
        ax.set_xlim(x_min, x_max)

    if ret_core.size > 0:
        y_min = ret_core.min()
        y_max = ret_core.max()
        spread = y_max - y_min
        if spread <= 0:
            spread = max(abs(y_max), 0.01)
        pad = 0.1 * spread
        ax.set_ylim(y_min - pad, y_max + pad)

    # -----------------------
    # 7) Save the figure
    # -----------------------
    if SAVE_FIGURES:
        filename = f"efficient_frontier_{slug_label}.png"
        filepath = SAVE_DIR / filename
        fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight")

    # Finally, show the plot on screen.
    plt.show()


# =======================
# MAIN ENTRY POINT
# =======================

def main():
    """
    Main driver function.

    High-level flow:
    1) Load price data from CSV.
    2) Convert prices to returns.
    3) Compute average returns and covariance matrix.
    4) Plot two cases:
       - Case A: Without shorting (long-only).
       - Case B: With shorting.
    """
    # 1) Load CSV data.
    prices = load_price_data(CSV_PATH)
    if prices.empty:
        raise ValueError("Price data is empty. Check that prices.csv has data in it and is in the same folder as this script.")

    # Keep the asset (column) names for labeling the weights later.
    asset_names = list(prices.columns)

    # 2) Convert prices to returns.
    returns = compute_returns(prices)
    if returns.empty:
        raise ValueError(
            "Return data is empty after processing. "
            "Check that the price columns are numeric and not all missing."
        )

    # 3) Compute mean returns (per period) and covariance matrix.
    mean_returns_period = returns.mean().values
    cov_matrix_period = returns.cov().values

    # 4A) Plot the case WITHOUT shorting (long-only constraints).
    plot_case(mean_returns_period, cov_matrix_period, asset_names, allow_short=False)

    # 4B) Plot the case WITH shorting allowed.
    plot_case(mean_returns_period, cov_matrix_period, asset_names, allow_short=True)


# Standard Python entry point check.
if __name__ == "__main__":
    main()
