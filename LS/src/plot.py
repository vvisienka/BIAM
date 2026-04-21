import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

def extract_size(instance_name):
    """Extracts the integer size N from the instance name for sorting."""
    match = re.search(r'\d+', str(instance_name))
    return int(match.group()) if match else 0

def main():
    print("Plotting started (High-Visibility Style)...")

    if not os.path.exists("../results.csv"):
        print("❌ Error: results.csv not found!")
        return

    df = pd.read_csv("../results.csv", skipinitialspace=True)

    # --- CLEAN INSTANCE NAMES ---
    df['Instance'] = df['Instance'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])

    # --- RENAME ALGORITHMS AND SET ORDER ---
    alg_mapping = {
        'RandomSearch': 'RS',
        'RandomWalk': 'RW',
        'Heuristic': 'H',
        'Greedy': 'G',
        'Steepest': 'S',
        'SA': 'SA',
        'TS': 'TS'
    }
    df['Algorithm'] = df['Algorithm'].replace(alg_mapping)
    
    algorithms_order = ['RS', 'RW', 'H', 'G', 'S', 'SA', 'TS']
    algorithms = [alg for alg in algorithms_order if alg in df['Algorithm'].unique()]

    df['OptCost'] = df['OptCost'].replace(0, np.nan)
    
    # 1. QUALITY METRIC
    # f_OPT/f_Best (Higher is better, 1.0 is optimum)
    df['Quality'] = df['OptCost'] / df['BestCost'] 

    # Sort instances by extracted size N
    df['Size'] = df['Instance'].apply(extract_size)
    df = df.sort_values(by=['Size', 'Instance'])
    ordered_instances = df['Instance'].unique()

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)

    # --- UNIFIED PALETTE AND MARKERS ---
    palette = {
        'RS': '#ff9999', 'RW': '#e41a1c',  # Light/Dark Red for Random
        'H': 'grey',                       # Neutral for Heuristic
        'G': '#9999ff', 'S': '#984ea3',    # Light Blue for Greedy, Purple (old TS) for Steepest
        'SA': '#4daf4a', 'TS': '#2e7d32'   # Green for SA, Darker Green for TS
    }
    markers = {
        'RS': 'o', 'RW': 'o', 'H': '^', 'G': 'D', 'S': 'D', 'SA': 'P', 'TS': 'P' # TS now uses SA's marker
    }

    # ==========================================
    # PLOT 1: QUALITY TRENDS (Min, Mean, Max) - Layout 2x3 (Legend in Slot 3)
    # ==========================================
    agg_q = df.groupby(['Algorithm', 'Instance', 'Size']).agg(
        Quality_Min=('Quality', 'min'),
        Quality_Mean=('Quality', 'mean'),
        Quality_Max=('Quality', 'max')
    ).reset_index()

    agg_q = agg_q.sort_values(by=['Size', 'Instance'])

    fig, axes = plt.subplots(2, 4, figsize=(32, 16), sharey=True)
    axes_flat = axes.flatten()

    plot_indices = [0, 1, 2, 4, 5, 6, 7] # Reserve index 3 for the legend

    for i, alg in enumerate(algorithms):
        if i >= len(plot_indices):
            print(f"Warning: Not enough plot slots for algorithm {alg}. Skipping.")
            continue
        
        ax = axes_flat[plot_indices[i]]
        subset = agg_q[agg_q['Algorithm'] == alg]

        ax.plot(subset['Instance'], subset['Quality_Max'], marker='^', color='#4daf4a', 
                linestyle='--', linewidth=2, markersize=12, label='Max')
        ax.plot(subset['Instance'], subset['Quality_Mean'], marker='o', color='#377eb8', 
                linestyle='-', linewidth=3, markersize=12, label='Average')
        ax.plot(subset['Instance'], subset['Quality_Min'], marker='v', color='#e41a1c', 
                linestyle='--', linewidth=2, markersize=12, label='Min')
        

        ax.set_title(f"{alg}", fontweight='bold', fontsize=22, pad=20)
        ax.set_xticks(range(len(ordered_instances)))
        ax.set_xticklabels(ordered_instances, rotation=45, ha='right')
        ax.set_xlabel("Instance", fontsize=20, fontweight='bold', labelpad=25)
        ax.tick_params(axis='both', which='major', labelsize=16, pad=15)
        ax.grid(True, linestyle='--', alpha=0.6)

    # Hide any axes that were not used for plotting and are not the legend slot
    used_indices = set(plot_indices[:len(algorithms)])
    for i, ax in enumerate(axes_flat):
        if i not in used_indices and i != 3: # Keep legend slot 3
            ax.axis('off')

    legend_ax = axes_flat[3]
    legend_ax.axis('off')
    handles, labels = axes_flat[0].get_legend_handles_labels()

    legend_ax.legend(handles, labels, loc='center', fontsize=22, title=None, 
                    frameon=True, shadow=True, markerscale=1.5)

    # Adjust y-labels for all rows in the new grid
    for row in range(2):
        axes[row, 0].set_ylabel("Algorithm Quality", fontsize=20, fontweight='bold', labelpad=25)

    plt.tight_layout()
    plt.savefig("../plots/plot_1_quality_trends.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ PLOT 1: Generated")
    # ==========================================
    # PLOT 2: COMPREHENSIVE PERFORMANCE COMPARISON
    # ==========================================
    plt.figure(figsize=(16, 9))
    
    sns.pointplot(data=df, x='Instance', y='Quality', hue='Algorithm',
                  hue_order=algorithms, order=ordered_instances, palette=palette,
                  markers=[markers.get(alg, 'o') for alg in algorithms], linestyle='-', linewidth=3, markersize=10,
                  errorbar='sd', capsize=0.1, dodge=0.3)
    
    plt.ylabel("Average Algorithm Quality", fontsize=20, fontweight='bold', labelpad=25)
    plt.xlabel("Instance", fontsize=20, fontweight='bold', labelpad=25)
    plt.tick_params(axis='both', which='major', labelsize=16, pad=15)
    plt.xticks(rotation=45)
    plt.legend(title="", loc='lower left', fontsize=16)
    plt.tight_layout()
    plt.savefig("../plots/plot_2_performance_comparison.pdf",format='pdf', dpi=300)
    plt.close()
    print("✅ PLOT 2: Generated")

    # ==========================================
    # PLOT 3: AVERAGE RUNNING TIME
    # ==========================================
    plt.figure(figsize=(16, 9))
    sns.pointplot(data=df, x='Instance', y='TimeMicros', hue='Algorithm',
                  hue_order=algorithms, order=ordered_instances, palette=palette,
                  errorbar='sd', capsize=0.1, markers=[markers.get(alg, 'o') for alg in algorithms], 
                  linestyle='-', linewidth=3, markersize=10, dodge=0.3)
    
    plt.ylabel("Algorithm Running Time [μs]", fontsize=20, fontweight='bold', labelpad=25)
    plt.xlabel("Instance", fontsize=20, fontweight='bold', labelpad=25)
    plt.yscale('log')
    plt.tick_params(axis='both', which='major', labelsize=16, pad=15)
    plt.xticks(rotation=45)
    plt.legend(title="", loc='upper left', fontsize=16)
    plt.tight_layout()
    plt.savefig("../plots/plot_3_running_time.pdf", format='pdf', dpi=300)
    plt.close()
    print("✅ PLOT 3: Generated")

    # ==========================================
    # PLOT 4: EFFICIENCY
    # ==========================================
    df['Efficiency'] = df['Quality'] / np.log(df['TimeMicros']+1)
    plt.figure(figsize=(16, 9))
    
    sns.pointplot(data=df, x='Instance', y='Efficiency', hue='Algorithm',
                  hue_order=algorithms, order=ordered_instances, palette=palette,
                  errorbar='sd', capsize=0.1, markers=[markers.get(alg, 'o') for alg in algorithms], 
                  linestyle='-', linewidth=3, markersize=10)
    
    plt.ylabel("Algorithm Efficiency", fontsize=20, fontweight='bold', labelpad=25)
    plt.xlabel("Instance", fontsize=20, fontweight='bold', labelpad=25)
    plt.tick_params(axis='both', which='major', labelsize=16, pad=15)
    plt.xticks(rotation=45)
    plt.legend(title="", loc='upper right', fontsize=16)
    plt.tight_layout()
    plt.savefig("../plots/plot_4_efficiency.pdf", format='pdf', dpi=300)
    plt.close()
    print("✅ PLOT 4: Generated")

    # ==========================================
    # PLOT 5: AVERAGE NUMBER OF STEPS
    # ==========================================
    algorithms_steps = ['G', 'S']
    df_steps = df[df['Algorithm'].isin(algorithms_steps)]

    plt.figure(figsize=(16, 9))
    sns.pointplot(data=df_steps, x='Instance', y='Steps', hue='Algorithm',
                  hue_order=algorithms_steps, order=ordered_instances, palette=palette,
                  errorbar='sd', capsize=0.1, markers=[markers.get(alg, 'o') for alg in algorithms_steps], 
                  linestyle='-', linewidth=3, markersize=11)
    
    plt.ylabel("Average Number of Steps", fontsize=20, fontweight='bold', labelpad=25)
    plt.xlabel("Instance", fontsize=20, fontweight='bold', labelpad=25)
    plt.tick_params(axis='both', which='major', labelsize=16, pad=15)
    plt.xticks(rotation=45)
    plt.legend(title="", loc='upper left', fontsize=16)
    plt.tight_layout()
    plt.savefig("../plots/plot_5_algorithm_steps.pdf", format='pdf', dpi=300)
    plt.close()
    print("✅ PLOT 5: Generated")

    # ==========================================
    # PLOT 6: AVERAGE NUMBER OF EVALUATIONS
    # ==========================================
    algorithms_eval = ['RS', 'RW', 'G', 'S', 'SA', 'TS']
    df_evals = df[df['Algorithm'].isin(algorithms_eval)]

    plt.figure(figsize=(16, 9))
    sns.pointplot(data=df_evals, x='Instance', y='Evaluations', hue='Algorithm',
                  hue_order=algorithms_eval, order=ordered_instances, palette=palette,
                  errorbar='sd', capsize=0.1, markers=[markers.get(alg, 'o') for alg in algorithms_eval], 
                  linestyle='-', linewidth=3, markersize=11)
    
    plt.ylabel("Average Number of Evaluations", fontsize=20, fontweight='bold', labelpad=25)
    plt.xlabel("Instance", fontsize=20, fontweight='bold', labelpad=25)
    plt.yscale('log')
    plt.tick_params(axis='both', which='major', labelsize=16, pad=15)
    plt.xticks(rotation=45)
    plt.legend(title="", loc='upper left', fontsize=16)
    plt.tight_layout()
    plt.savefig("../plots/plot_6_evaluations.pdf", format='pdf', dpi=300)
    plt.close()
    print("✅ PLOT 6: Generated")
    
if __name__ == "__main__":
    main()