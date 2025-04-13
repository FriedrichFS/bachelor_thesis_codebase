import pandas as pd
import matplotlib.pyplot as plt


def compare_model_performance():
    # NOTE: Manually define this for active analysis! Code itself not working without manual prep! Onyl used for temp analysis!!
    df = pd.read_csv('data.csv')

    summary_table = df.groupby(['target_dte_group', 'option_type'])['LR_vs_CRR_Diff (%)'].agg(
        MeanDiffPct='mean',
        StdDiffPct='std',
        MinDiffPct='min',
        MaxDiffPct='max'
    )
    print("=== Statistical Summary of LR vs CRR Differences (%) ===")
    print(summary_table.reset_index())
    print()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

    calls_df = df[df['option_type'] == 'call'].copy()
    puts_df = df[df['option_type'] == 'put'].copy()

    # Markers for better deferientiation
    markers = ['o', 'v', '^', '<', '>', 's', 'p', 'X', 'D']

    def plot_model_times(ax, dataframe, option_type_label):
        """
        Plots CRR vs. LR average time in milliseconds vs. N_steps on ax
        using log-scale y-axis.
        Each target_dte_group is plotted with its own marker.
        """
        ax.set_title(
            f"{option_type_label.capitalize()} - CRR vs. LR Runtimes (Log-Scale Y)")

        unique_dtes = sorted(dataframe['target_dte_group'].unique())

        for i, dte in enumerate(unique_dtes):
            subset = dataframe[dataframe['target_dte_group'] == dte].copy()
            subset.sort_values('N_steps', inplace=True)
            marker = markers[i % len(markers)]

            ax.plot(
                subset['N_steps'],
                subset['CRR Avg Time (ms)'],
                marker=marker,
                label=f"DTE={dte} CRR"
            )
            ax.plot(
                subset['N_steps'],
                subset['LR Avg Time (ms)'],
                marker=marker,
                linestyle='--',
                label=f"DTE={dte} LR"
            )

        ax.set_xlabel("Number of Steps (N)")
        ax.set_ylabel("Avg Calculation Time (ms)")

        ax.set_yscale('log')
        ax.grid(True, which="both", ls="--", linewidth=0.7)

    plot_model_times(axes[0], calls_df, 'call')

    plot_model_times(axes[1], puts_df, 'put')

    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l

    unique_pairs = []
    unique_handles = []
    for h, l in zip(handles, labels):
        if l not in unique_pairs:
            unique_pairs.append(l)
            unique_handles.append(h)

    fig.legend(
        unique_handles,
        unique_pairs,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.02),
        ncol=4
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    plt.savefig("model_performance_side_by_side.png",
                dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    compare_model_performance()
