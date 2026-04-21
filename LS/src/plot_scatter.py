import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import spearmanr

def main():
    print("Scatter plotting started...")
    if not os.path.exists("../scatter_data.csv"):
        print("Error: scatter_data.csv not found!")
        return

    df = pd.read_csv("../scatter_data.csv", skipinitialspace=True)

    # Calculate Quality Ratio (Higher is better, 1.0 is optimal)
    df['InitQuality'] = df['OptCost'] / df['InitCost']
    df['FinalQuality'] = df['OptCost'] / df['FinalCost']

    instances = ["els19", "chr20c", "esc32a"]
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    # Group G+S with shades of blue/purple and same marker
    palette = {'G': '#9999ff', 'S': '#984ea3'} # S now uses TS's old color
    markers = {'G': 'D', 'S': 'D'}

    for instance in instances:
        instance_df = df[df['Instance'] == instance]
        
        # Create a side-by-side plot for G and S
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
        
        
        for ax, alg in zip(axes, ['G', 'S']):
            alg_df = instance_df[instance_df['Algorithm'] == alg]
            
            if len(alg_df) > 1:
                rho, _ = spearmanr(alg_df['InitQuality'], alg_df['FinalQuality'])
            else:
                rho = 0.0

            # Draw scatter plot with a linear regression line
            sns.scatterplot(
                data=alg_df, 
                x='InitQuality', 
                y='FinalQuality', 
                ax=ax,
                color=palette[alg],
                marker=markers[alg],
                s=30,          # Rozmiar punktów
                alpha=0.6      # Przezroczystość
            )

            ax.set_title(f"{alg} (ρ = {rho:.3f})", fontsize=24, fontweight='bold', pad=10)
            ax.set_xlabel("Initial Quality", fontsize=20, fontweight='bold', labelpad=25)
            if ax == axes[0]:
                ax.set_ylabel("Final Quality", fontsize=20, fontweight='bold', labelpad=25)
            
        # plt.suptitle(f"Search Space Structure: {instance}", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.ylim(0.3, 1.01)
        plt.savefig(f"../plots/plot_7_scatter_{instance}.pdf", format='pdf', dpi=300)
        plt.close()
        print(f"✅ Scatter Plot Generated for {instance}")

if __name__ == "__main__":
    main()