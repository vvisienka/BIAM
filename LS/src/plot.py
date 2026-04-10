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
    print("Plotting started...")

    # Update the path if your results.csv is in another directory
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

    # Protect against missing/zero OptCost to prevent division by zero
    df['OptCost'] = df['OptCost'].replace(0, np.nan)

    # 1. QUALITY METRIC
    # Quality Gap: (f_A - f_OPT) / f_OPT (Lower is better)
    df['Quality_Gap'] = (df['BestCost'] - df['OptCost']) / df['OptCost']

    # Sort instances by extracted size N, then by name
    df['Size'] = df['Instance'].apply(extract_size)
    df = df.sort_values(by=['Size', 'Instance'])
    # This list guarantees the X-axis order across all plots
    ordered_instances = df['Instance'].unique()

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    palette = "Set1"

    # ==========================================
    # PLOT 1: QUALITY TRENDS (Min, Mean, Max)
    # ==========================================
    agg_q = df.groupby(['Algorithm', 'Instance', 'Size']).agg(
        Gap_Min=('Quality_Gap', 'min'),
        Gap_Mean=('Quality_Gap', 'mean'),
        Gap_Max=('Quality_Gap', 'max')
    ).reset_index()

    agg_q = agg_q.sort_values(by=['Size', 'Instance'])

    fig, axes = plt.subplots(1, len(algorithms), figsize=(5 * len(algorithms), 6), sharey=True)
    if len(algorithms) == 1:
        axes = [axes]

    for ax, alg in zip(axes, algorithms):
        subset = agg_q[agg_q['Algorithm'] == alg]

        # MAX (Worst case)
        ax.plot(subset['Instance'], subset['Gap_Max'], marker='^', color='#e41a1c', 
                linestyle='--', linewidth=2, markersize=8, label='Max')
        # AVERAGE
        ax.plot(subset['Instance'], subset['Gap_Mean'], marker='o', color='#377eb8', 
                linestyle='-', linewidth=2.5, markersize=8, label='Average')
        # MIN (Best case)
        ax.plot(subset['Instance'], subset['Gap_Min'], marker='v', color='#4daf4a', 
                linestyle='--', linewidth=2, markersize=8, label='Min')

        ax.set_title(f"{alg}", fontweight='bold', fontsize=14)
        # Using ordered_instances to ensure strict sorting
        ax.set_xticks(range(len(ordered_instances)))
        ax.set_xticklabels(ordered_instances, rotation=45, ha='right')
        
        ax.set_xlabel("Instance", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)

    axes[0].set_ylabel("Quality Gap", fontsize=14)
    plt.suptitle("Quality Comparison", fontsize=18, fontweight='bold', y=1.05)
    axes[-1].legend(loc='upper right', title="")

    plt.tight_layout()
    plt.savefig("plot_1_quality_trends.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ PLOT 1: Generated (Quality comparison)")

    # ==========================================
    # PLOT 2: COMPREHENSIVE PERFORMANCE COMPARISON
    # Comparing average Quality Gap of all algorithms on each instance
    # ==========================================
    performance_agg = df.groupby(['Algorithm', 'Instance', 'Size']).agg(
        AverageGap=('Quality_Gap', 'mean')
    ).reset_index()

    # Sort the aggregated data just to be safe
    performance_agg = performance_agg.sort_values(by=['Size', 'Instance'])

    plt.figure(figsize=(16, 8)) # Wide format to fit all instances nicely
    
    # We use pointplot: dots connected by lines, perfect for showing trends
    sns.pointplot(
        data=performance_agg, 
        x='Instance', 
        y='AverageGap', 
        hue='Algorithm',
        hue_order=algorithms,         # Forces legend order (RS, RW, H, G, S)
        order=ordered_instances,      # FORCES X-AXIS SORTING BY N SIZE
        palette=palette,
        markers=['o', 's', 'D', 'v', '^'], # Different shapes for each algorithm
        linestyle='-',
        linewidth=2,
        markersize=8
    )
    
    plt.title("Average Quality Comparison", fontsize=18, fontweight='bold')
    plt.ylabel("Average Quality Gap", fontsize=14)
    plt.xlabel("Instance", fontsize=14)
    
    # Using a logarithmic scale on Y axis can be very helpful here because
    # Random Search might have a gap of 2.0 (200%), while Steepest might have 0.05 (5%).
    # If the lines look too squashed at the bottom, uncomment the line below:
    # plt.yscale('log')

    plt.xticks(rotation=45, fontsize=12)
    
    # Move legend outside the plot so it doesn't cover the data points
    plt.legend(title="", bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=12)
    
    plt.tight_layout()
    plt.savefig("plot_2_performance_comparison.png", dpi=300)
    plt.close()
    print("✅ PLOT 2: Generated (Average Performance Pointplot)")

    # ==========================================
    # PLOT 3: AVERAGE RUNNING TIME
    # Average execution time with standard deviation
    # ==========================================
    plt.figure(figsize=(16, 8))
    
    sns.barplot(
        data=df, 
        x='Instance', 
        y='TimeMicros', 
        hue='Algorithm',
        hue_order=algorithms,    # Zachowuje kolejność RS, RW, H, G, S
        order=ordered_instances, # Posortowane rosnąco według N
        palette=palette,
        errorbar='sd',           # Rysuje odchylenie standardowe (Standard Deviation)
        capsize=0.1              # Dodaje poziome kreski na końcach wąsów błędu
    )
    
    plt.title("Average Running Time With Standard Deviation", fontsize=18, fontweight='bold')
    plt.ylabel("Time [μs]", fontsize=14)
    plt.xlabel("Instance", fontsize=14)
    
    # Skala logarytmiczna - kluczowa dla czasów wykonania algorytmów
    plt.yscale('log')
    
    plt.xticks(rotation=45, fontsize=12)
    plt.legend(title="", bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=12)
    
    plt.tight_layout()
    plt.savefig("plot_3_running_time.png", dpi=300)
    plt.close()
    print("✅ PLOT 3: Generated (Average Running Time)")

    
if __name__ == "__main__":
    main()