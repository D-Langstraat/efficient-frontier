# Efficient Frontier Visualizer

This repo contains a small, self-contained Python tool that:

- Reads historical **stock price data** from `prices.csv`
- Converts prices to **returns**
- Builds the **Markowitz efficient frontier** (the “bullet”)
- Finds the **max-Sharpe (tangency) portfolio**
- Plots and saves **two figures**:
  - Efficient frontier **without shorting** (long-only)
  - Efficient frontier **with shorting**

---

## What the script does (in plain English)

The script:

1. **Loads price data** for several stocks from `prices.csv`.
2. **Computes monthly returns**:
   \[
   r_t = \frac{P_t}{P_{t-1}} - 1
   \]
3. **Computes risk and co-movement**:
   - Mean return of each stock  
   - Covariance matrix of the stock returns
4. Uses that information to:
   - Generate many **random portfolios** (random weights)
   - Build the **efficient frontier** by minimizing volatility for a range of target returns
   - Find the **max-Sharpe portfolio**:
     \[
     \text{Sharpe} = \frac{E[R_p] - R_f}{\sigma_p}
     \]
5. **Plots and saves** two charts:
   - One assuming **no shorting** (all weights between 0 and 1)
   - One **with shorting** allowed (weights can be negative or > 1)

Each chart shows:

- A cloud of random portfolios (risk vs. return)
- The **bottom** branch of the frontier (inefficient) in **dashed gray**
- The **top** branch of the frontier (efficient) in **solid orange**
- A **black star** at the **max-Sharpe portfolio**
- A text box listing:
  - Annual **expected return**
  - Annual **standard deviation**
  - Annual **Sharpe ratio**
  - The **exact weights** for each stock in the max-Sharpe portfolio

---

## Repository contents

- `efficient_frontier.py`  
  Main Python script that does all the calculations and plotting.

- `prices.csv`  
  Historical **price** data (not returns) for the stocks used in the analysis.

If you want to customize the universe of assets, you can replace `prices.csv` with another file that has the same layout (a `Date` column + one column per asset).

---

## Requirements

You need:

- **Python 3.x** (any recent 3.x is fine)
- Python packages:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scipy`

### 1. Install Python

1. Go to [python.org](https://www.python.org/)  
2. Download and run the latest **Python 3** installer.
3. On the first screen, check:

   > ✅ **Add Python 3.x to PATH**

4. Click **Install Now** and finish the setup.

To confirm it worked, open **Windows PowerShell** and run:

```powershell
python --version
pip --version
