import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    print("Multi-Start Restart plotting started...")
    if not os.path.exists("../scatter_data.csv"):
        print("Error: scatter_data.csv not found! Run the scatter Rust script first.")
        return

    df = pd.read_csv("../scatter_data.csv", skipinitialspace=True)

    instances = ["els19", "chr20a", "esc32a"]
    
    # Calculate Quality Ratio (Higher is better, 1.0 is optimal)
    df['Quality'] = df['OptCost'] / df['FinalCost']

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    palette = {'G': '#984ea3', 'S': '#ff7f00'}

    # Create a single figure with 1 row and exactly as many columns as instances.
    # sharey=True forces them all to use the exact same vertical scale.
    fig, axes = plt.subplots(1, len(instances), figsize=(18, 6), sharey=True)
    
    # Safety catch in case you ever change it to just 1 instance
    if len(instances) == 1:
        axes = [axes]

    for ax, instance in zip(axes, instances):
        instance_df = df[df['Instance'] == instance].copy()
        
        # Sort by run number to simulate the timeline of a Multi-Start algorithm
        instance_df = instance_df.sort_values(by=['Algorithm', 'Run'])

        for alg in ['G', 'S']:
            alg_df = instance_df[instance_df['Algorithm'] == alg].copy()
            
            # Skip if no data for this algorithm/instance combo
            if alg_df.empty:
                continue
            
            # Calculate the metrics as we progress through the runs (1 to 450)
            alg_df['Best_So_Far'] = alg_df['Quality'].cummax()
            alg_df['Average_So_Far'] = alg_df['Quality'].expanding().mean()

            # Plot Best So Far (Thick solid line) - Note the ax=ax parameter!
            sns.lineplot(
                data=alg_df, 
                x='Run', 
                y='Best_So_Far', 
                color=palette[alg], 
                linewidth=3, 
                label=f"{alg} - Best",
                ax=ax
            )

            # Plot Average So Far (Thin dashed line)
            sns.lineplot(
                data=alg_df, 
                x='Run', 
                y='Average_So_Far', 
                color=palette[alg], 
                linewidth=1.5, 
                linestyle='--', 
                label=f"{alg} - Average",
                ax=ax
            )

        ax.set_title(f"Instance: {instance}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Number of Repetitions", fontsize=12)
        
        # Clear the individual Y labels so we can set one master label later
        ax.set_ylabel("")
        
        ax.set_xlim(1, 400)
        ax.set_ylim(bottom=0.5, top=1.01)

        # Legend Management: Only keep the legend on the very last subplot
        if ax == axes[-1]:
            ax.legend(title="Metric", loc='lower right', fontsize=10)
        else:
            if ax.get_legend() is not None:
                ax.get_legend().remove()

    # Set the Y-axis label only on the first (leftmost) plot
    axes[0].set_ylabel("Algorithm Quality", fontsize=14)
    
    # Set a master title for the entire figure
    plt.suptitle("Multi-Start Efficiency: Quality vs. Restarts", fontsize=18, fontweight='bold', y=1.05)
    
    # bbox_inches='tight' prevents the master title from getting cut off
    plt.tight_layout()
    plt.savefig("../plots/plot_8_restarts_combined.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Combined Restart Plot Generated!")

if __name__ == "__main__":
    main()