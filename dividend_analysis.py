import pandas as pd
import numpy as np


def analyze_dividends():
    """
    Reads 'data.csv' in the same directory and computes:
      - Longest period (in days) without a dividend for each ticker.
      - Number of dividend payouts observed.
      - Average gap between consecutive dividends.
      - Earliest (first) dividend date in the dataset for each ticker.
    Prints a concise summary DataFrame to the console.
    """
    # 1. Load the CSV
    df = pd.read_csv(
        'pipeline_output_final/raw_data/final_equity_events.csv')
    print(df.head())
    # 2. Filter to keep only dividend records
    # event_type == 'dividend' usually indicates a dividend row.
    # We also ensure pay_date or ex_dividend_date is not null.
    div_df = df[df['event_type'] == 'dividend'].copy()
    if div_df.empty:
        print("No dividend rows found in the dataset.")
        return

    # 3. Create a single 'div_date' column (prefer pay_date if present, else ex_dividend_date)
    # Some rows might have blank pay_date. We'll also drop rows where both are missing.
    div_df['pay_date'] = pd.to_datetime(div_df['pay_date'], errors='coerce')
    div_df['ex_dividend_date'] = pd.to_datetime(
        div_df['ex_dividend_date'], errors='coerce')

    # We choose pay_date as primary, fallback to ex_dividend_date if pay_date is NaT
    div_df['div_date'] = div_df['pay_date'].fillna(div_df['ex_dividend_date'])
    div_df.dropna(subset=['div_date'], inplace=True)

    if div_df.empty:
        print("No valid dividend dates found after filtering.")
        return

    # 4. For each ticker, sort by div_date and compute day-gaps
    results = []
    grouped = div_df.groupby('ticker')
    for ticker, group in grouped:
        group = group.sort_values('div_date')
        dates = group['div_date'].values

        if len(dates) == 1:
            # Only one dividend date -> no "gap" can be computed
            gap_max = 0
            gap_avg = 0
        else:
            # Compute consecutive differences in days
            diffs = np.diff(dates) / np.timedelta64(1, 'D')
            gap_max = diffs.max()
            gap_avg = diffs.mean()

        first_date = group['div_date'].iloc[0]
        total_dividends = len(group)

        results.append({
            'Ticker': ticker,
            'EarliestDividendDate': first_date.strftime('%Y-%m-%d'),
            'TotalDividends': total_dividends,
            'LongestGapDays': round(gap_max, 1),
            'AverageGapDays': round(gap_avg, 1)
        })

    # 5. Convert to DataFrame and print
    results_df = pd.DataFrame(results)
    results_df.sort_values('Ticker', inplace=True)

    print("\n=== Dividend Analysis: Longest Period Without Dividends ===")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    analyze_dividends()
