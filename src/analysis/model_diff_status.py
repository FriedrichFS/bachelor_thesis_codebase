import pandas as pd


def calculate_model_diff_stats():
    """
    Reads 'model_diff_status.csv' from the local directory, calculates the average
    calculated_price grouped by (target_dte_group, model), and shows
    percentage differences vs. a reference model (CRR).
    """
    df = pd.read_csv('model_diff_status.csv')

    grouped = df.groupby(['target_dte_group', 'model'])[
        'calculated_price'].mean().reset_index(name='avg_price')

    pivot_df = grouped.pivot(index='target_dte_group',
                             columns='model', values='avg_price')

    reference_model = 'CRR'
    if reference_model in pivot_df.columns:
        for model_name in pivot_df.columns:
            if model_name != reference_model:
                pivot_df[f'{model_name}_diff_%'] = 100 * (
                    pivot_df[model_name] - pivot_df[reference_model]) / pivot_df[reference_model]

    print("\n=== Average Calculated Price by DTE and Model ===")
    print(pivot_df)

    diff_cols = [c for c in pivot_df.columns if c.endswith('_diff_%')]
    if diff_cols:
        stats = pivot_df[diff_cols].agg(['mean', 'std', 'min', 'max'])
        print("\n=== Statistics for % Differences vs. Reference Model ===")
        print(stats)


if __name__ == "__main__":
    calculate_model_diff_stats()
