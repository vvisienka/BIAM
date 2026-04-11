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
        'Steepest': 'S'
    }
    df['Algorithm'] = df['Algorithm'].replace(alg_mapping)
    
    algorithms_order = ['RS', 'RW', 'H', 'G', 'S']
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
    palette = "Set1"

    # ==========================================
    # PLOT 1: QUALITY TRENDS (Min, Mean, Max) - Layout 2x3 (Legend in Slot 3)
    # ==========================================
    agg_q = df.groupby(['Algorithm', 'Instance', 'Size']).agg(
        Quality_Min=('Quality', 'min'),
        Quality_Mean=('Quality', 'mean'),
        Quality_Max=('Quality', 'max')
    ).reset_index()

    agg_q = agg_q.sort_values(by=['Size', 'Instance'])

    fig, axes = plt.subplots(2, 3, figsize=(20, 16), sharey=True)
    axes_flat = axes.flatten()

    plot_map = [0, 1, 3, 4, 5]

    for i, alg in enumerate(algorithms):
        target_idx = plot_map[i]
        ax = axes_flat[target_idx]
        subset = agg_q[agg_q['Algorithm'] == alg]

        # Zwiększone markersize=12
        ax.plot(subset['Instance'], subset['Quality_Min'], marker='^', color='#e41a1c', 
                linestyle='--', linewidth=2, markersize=12, label='Min')
        ax.plot(subset['Instance'], subset['Quality_Mean'], marker='o', color='#377eb8', 
                linestyle='-', linewidth=3, markersize=12, label='Average')
        ax.plot(subset['Instance'], subset['Quality_Max'], marker='v', color='#4daf4a', 
                linestyle='--', linewidth=2, markersize=12, label='Max')

        ax.set_title(f"{alg}", fontweight='bold', fontsize=22, pad=20)
        ax.set_xticks(range(len(ordered_instances)))
        ax.set_xticklabels(ordered_instances, rotation=45, ha='right')
        ax.set_xlabel("Instance", fontsize=20, fontweight='bold', labelpad=25)
        ax.tick_params(axis='both', which='major', labelsize=16, pad=15)
        ax.grid(True, linestyle='--', alpha=0.6)

    legend_ax = axes_flat[2]
    legend_ax.axis('off')
    handles, labels = axes_flat[0].get_legend_handles_labels()

    # Legend bez tytułu, z powiększonymi punktami (markerscale=1.5)
    legend_ax.legend(handles, labels, loc='center', fontsize=22, title=None, 
                    frameon=True, shadow=True, markerscale=1.5)

    for row in range(2):
        axes[row, 0].set_ylabel("Algorithm Quality", fontsize=20, fontweight='bold', labelpad=25)

    plt.tight_layout()
    plt.savefig("../plots/plot_1_quality_trends.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ PLOT 1: Generated (Large points, cleaned legend)")
    # # ==========================================
    # # PLOT 2: COMPREHENSIVE PERFORMANCE COMPARISON
    # # ==========================================
    # performance_agg = df.groupby(['Algorithm', 'Instance', 'Size']).agg(AverageQuality=('Quality', 'mean')).reset_index()
    # plt.figure(figsize=(16, 9))
    
    # sns.pointplot(data=performance_agg, x='Instance', y='AverageQuality', hue='Algorithm',
    #               hue_order=algorithms, order=ordered_instances, palette=palette,
    #               markers=['o', 's', 'D', 'v', '^'], linestyle='-', linewidth=3, markersize=10)
    
    # plt.ylabel("Average Algorithm Quality", fontsize=20, fontweight='bold', labelpad=25)
    # plt.xlabel("Instance", fontsize=20, fontweight='bold', labelpad=25)
    # plt.tick_params(axis='both', which='major', labelsize=16, pad=15)
    # plt.xticks(rotation=45)
    # plt.legend(title="", loc='lower left', fontsize=16)
    # plt.tight_layout()
    # plt.savefig("../plots/plot_2_performance_comparison.png", dpi=300)
    # plt.close()
    # print("✅ PLOT 2: Generated")

    # # ==========================================
    # # PLOT 3: AVERAGE RUNNING TIME
    # # ==========================================
    # plt.figure(figsize=(16, 9))
    # sns.barplot(data=df, x='Instance', y='TimeMicros', hue='Algorithm',
    #             hue_order=algorithms, order=ordered_instances, palette=palette, errorbar='sd', capsize=0.1)
    
    # plt.ylabel("Algorithm Running Time [μs]", fontsize=20, fontweight='bold', labelpad=25)
    # plt.xlabel("Instance", fontsize=20, fontweight='bold', labelpad=25)
    # plt.yscale('log')
    # plt.tick_params(axis='both', which='major', labelsize=16, pad=15)
    # plt.xticks(rotation=45)
    # plt.legend(title="", loc='upper left', fontsize=16)
    # plt.tight_layout()
    # plt.savefig("../plots/plot_3_running_time.png", dpi=300)
    # plt.close()
    # print("✅ PLOT 3: Generated")

    # # ==========================================
    # # PLOT 4: EFFICIENCY
    # # ==========================================
    # df['Efficiency'] = df['Quality'] / np.log(df['TimeMicros']+1)
    # plt.figure(figsize=(16, 9))
    
    # sns.pointplot(data=df, x='Instance', y='Efficiency', hue='Algorithm',
    #               hue_order=algorithms, order=ordered_instances, palette=palette,
    #               errorbar='sd', capsize=0.1, markers=['o', 's', 'D', 'v', '^'], 
    #               linestyle='-', linewidth=3, markersize=10)
    
    # plt.ylabel("Algorithm Efficiency", fontsize=20, fontweight='bold', labelpad=25)
    # plt.xlabel("Instance", fontsize=20, fontweight='bold', labelpad=25)
    # plt.tick_params(axis='both', which='major', labelsize=16, pad=15)
    # plt.xticks(rotation=45)
    # plt.legend(title="", loc='upper right', fontsize=16)
    # plt.tight_layout()
    # plt.savefig("../plots/plot_4_efficiency.png", dpi=300)
    # plt.close()
    # print("✅ PLOT 4: Generated")

    # # ==========================================
    # # PLOT 5: AVERAGE NUMBER OF STEPS (G vs S)
    # # ==========================================
    # df_gs = df[df['Algorithm'].isin(['G', 'S'])]
    # gs_palette = {'G': '#984ea3', 'S': '#ff7f00'}

    # plt.figure(figsize=(16, 9))
    # sns.pointplot(data=df_gs, x='Instance', y='Steps', hue='Algorithm',
    #               hue_order=['G', 'S'], order=ordered_instances, palette=gs_palette,
    #               errorbar='sd', capsize=0.1, markers=['v', '^'], 
    #               linestyle='-', linewidth=3, markersize=11)
    
    # plt.ylabel("Average Number of Steps", fontsize=20, fontweight='bold', labelpad=25)
    # plt.xlabel("Instance", fontsize=20, fontweight='bold', labelpad=25)
    # plt.tick_params(axis='both', which='major', labelsize=16, pad=15)
    # plt.xticks(rotation=45)
    # plt.legend(title="", loc='upper left', fontsize=16)
    # plt.tight_layout()
    # plt.savefig("../plots/plot_5_algorithm_steps.png", dpi=300)
    # plt.close()
    # print("✅ PLOT 5: Generated")

    # # ==========================================
    # # PLOT 6: AVERAGE NUMBER OF EVALUATIONS
    # # ==========================================
    # algorithms_eval = ['RS', 'RW', 'G', 'S']
    # df_evals = df[df['Algorithm'].isin(algorithms_eval)]
    # eval_palette = {'RS': '#e41a1c', 'RW': '#377eb8', 'G': '#984ea3', 'S': '#ff7f00'}

    # plt.figure(figsize=(16, 9))
    # sns.pointplot(data=df_evals, x='Instance', y='Evaluations', hue='Algorithm',
    #               hue_order=algorithms_eval, order=ordered_instances, palette=eval_palette,
    #               errorbar='sd', capsize=0.1, markers=['o', 's', 'v', '^'], 
    #               linestyle='-', linewidth=3, markersize=11)
    
    # plt.ylabel("Average Number of Evaluations", fontsize=20, fontweight='bold', labelpad=25)
    # plt.xlabel("Instance", fontsize=20, fontweight='bold', labelpad=25)
    # plt.yscale('log')
    # plt.tick_params(axis='both', which='major', labelsize=16, pad=15)
    # plt.xticks(rotation=45)
    # plt.legend(title="", loc='upper left', fontsize=16)
    # plt.tight_layout()
    # plt.savefig("../plots/plot_6_evaluations.png", dpi=300)
    # plt.close()
    # print("✅ PLOT 6: Generated")
    
if __name__ == "__main__":
    main()