import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    print("Generating Multi-Start Plot (Fixed Labels & Large Fonts)...")
    csv_path = "../scatter_data.csv"
    if not os.path.exists(csv_path): return

    df = pd.read_csv(csv_path, skipinitialspace=True)
    df['Instance'] = df['Instance'].astype(str).str.strip()
    df['Quality'] = df['OptCost'] / df['FinalCost']

    instances = ["els19", "chr20a", "esc32a"]
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.6)

    # Group G+S with shades of blue/purple
    palette = {'G': '#9999ff', 'S': '#984ea3'} # S now uses TS's old color

    fig, axes = plt.subplots(1, len(instances), figsize=(22, 9), sharey=True)
    if len(instances) == 1: axes = [axes]

    for ax, instance in zip(axes, instances):
        instance_df = df[df['Instance'] == instance].copy()
        instance_df = instance_df.sort_values(by=['Algorithm', 'Run'])

        any_plotted = False
        for alg in ['G', 'S']:
            alg_df = instance_df[instance_df['Algorithm'] == alg].copy()
            if alg_df.empty: continue
            
            any_plotted = True
            alg_df['Best_So_Far'] = alg_df['Quality'].cummax()
            alg_df['Average_So_Far'] = alg_df['Quality'].expanding().mean()

            # Rysujemy z jawnymi etykietami
            ax.plot(alg_df['Run'], alg_df['Best_So_Far'], color=palette[alg], linewidth=5.0, label=f"{alg} - Best")
            ax.plot(alg_df['Run'], alg_df['Average_So_Far'], color=palette[alg], linewidth=2.5, linestyle='--', label=f"{alg} - Average")

        ax.set_title(f"Instance: {instance}", fontsize=24, fontweight='bold', pad=10)
        ax.set_xlabel("Number of Repetitions", fontsize=20, fontweight='bold', labelpad=25)
        ax.tick_params(axis='both', which='major', labelsize=16, pad=15)
        
        ax.set_ylabel("")
        ax.set_xlim(1, 400)
        ax.set_ylim(bottom=0.5, top=1.01)

        # Wywołujemy legendę tylko jeśli coś faktycznie narysowaliśmy na ostatnim wykresie
        if ax == axes[-1] and any_plotted:
            ax.legend(title="", loc='lower right', fontsize=18, title_fontsize=20, frameon=True)
        elif ax.get_legend() is not None:
            ax.get_legend().remove()

    axes[0].set_ylabel("Algorithm Quality", fontsize=20, fontweight='bold', labelpad=25)
    
    plt.subplots_adjust(left=0.1, right=0.95, top=0.88, bottom=0.18, wspace=0.15)
    plt.savefig("../plots/plot_8_restarts_combined.pdf", dpi=300, format='pdf', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()