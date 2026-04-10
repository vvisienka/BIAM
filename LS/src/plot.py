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
    # Quality: f_OPT/f_Best (Higher is better)
    df['Quality'] = df['OptCost']/df['BestCost']

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
        Quality_Min=('Quality', 'min'),
        Quality_Mean=('Quality', 'mean'),
        Quality_Max=('Quality', 'max')
    ).reset_index()

    agg_q = agg_q.sort_values(by=['Size', 'Instance'])

    fig, axes = plt.subplots(1, len(algorithms), figsize=(5 * len(algorithms), 6), sharey=True)
    if len(algorithms) == 1:
        axes = [axes]

    for ax, alg in zip(axes, algorithms):
        subset = agg_q[agg_q['Algorithm'] == alg]

        # MAX (Worst case) - min quality
        ax.plot(subset['Instance'], subset['Quality_Min'], marker='^', color='#e41a1c', 
                linestyle='--', linewidth=2, markersize=8, label='Max')
        # AVERAGE
        ax.plot(subset['Instance'], subset['Quality_Mean'], marker='o', color='#377eb8', 
                linestyle='-', linewidth=2.5, markersize=8, label='Average')
        # MIN (Best case) - max quality
        ax.plot(subset['Instance'], subset['Quality_Max'], marker='v', color='#4daf4a', 
                linestyle='--', linewidth=2, markersize=8, label='Min')

        ax.set_title(f"{alg}", fontweight='bold', fontsize=14)
        # Using ordered_instances to ensure strict sorting
        ax.set_xticks(range(len(ordered_instances)))
        ax.set_xticklabels(ordered_instances, rotation=45, ha='right')
        
        ax.set_xlabel("Instance", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)

    axes[0].set_ylabel("Algorithm Quality", fontsize=14)
    plt.suptitle("Algorithm Quality Comparison", fontsize=18, fontweight='bold', y=1.05)
    axes[-1].legend(loc='lower right', title="")

    plt.tight_layout()
    plt.savefig("../plots/plot_1_quality_trends.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ PLOT 1: Generated (Quality comparison)")

    # ==========================================
    # PLOT 2: COMPREHENSIVE PERFORMANCE COMPARISON
    # Comparing average Quality of all algorithms on each instance
    # ==========================================
    performance_agg = df.groupby(['Algorithm', 'Instance', 'Size']).agg(
        AverageQuality=('Quality', 'mean')
    ).reset_index()

    # Sort the aggregated data just to be safe
    performance_agg = performance_agg.sort_values(by=['Size', 'Instance'])

    plt.figure(figsize=(16, 8)) # Wide format to fit all instances nicely
    
    # We use pointplot: dots connected by lines, perfect for showing trends
    sns.pointplot(
        data=performance_agg, 
        x='Instance', 
        y='AverageQuality', 
        hue='Algorithm',
        hue_order=algorithms,         # Forces legend order (RS, RW, H, G, S)
        order=ordered_instances,      # FORCES X-AXIS SORTING BY N SIZE
        palette=palette,
        markers=['o', 's', 'D', 'v', '^'], # Different shapes for each algorithm
        linestyle='-',
        linewidth=2,
        markersize=8
    )
    
    plt.title("Average Algorithm Quality Comparison", fontsize=18, fontweight='bold')
    plt.ylabel("Average Algorithm Quality", fontsize=14)
    plt.xlabel("Instance", fontsize=14)
    

    plt.xticks(rotation=45, fontsize=12)
    
    # Move legend outside the plot so it doesn't cover the data points
    plt.legend(title="", bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=12)
    
    plt.tight_layout()
    plt.savefig("../plots/plot_2_performance_comparison.png", dpi=300)
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
        hue_order=algorithms,    # Keep the order
        order=ordered_instances, # Sorted instances
        palette=palette,
        errorbar='sd',           # tandard Deviation
        capsize=0.1              # Add caps to error bars
    )
    
    plt.title("Average Algorithm Running Time With Standard Deviation", fontsize=18, fontweight='bold')
    plt.ylabel("Algorithm Running Time [μs]", fontsize=14)
    plt.xlabel("Instance", fontsize=14)
    plt.yscale('log')
    
    plt.xticks(rotation=45, fontsize=12)
    plt.legend(title="", bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=12)
    
    plt.tight_layout()
    plt.savefig("../plots/plot_3_running_time.png", dpi=300)
    plt.close()
    print("✅ PLOT 3: Generated (Average Running Time)")

    # ==========================================
    # PLOT 4: EFFICIENCY (Raw Ratio Version)
    # Efficiency = Quality_Ratio / ln(TimeMicros+1)
    # Higher is better
    # ==========================================

    
    # Efficiency: Ratio divided by Time (with epsilon=1 to prevent div by zero)
    df['Efficiency'] = df['Quality'] / np.log(df['TimeMicros']+1)

    plt.figure(figsize=(16, 8))
    # plt.yscale('log')
    
    # Using pointplot to show the trend of efficiency across instance sizes
    sns.pointplot(
        data=df, 
        x='Instance', 
        y='Efficiency', 
        hue='Algorithm',
        hue_order=algorithms,    
        order=ordered_instances, 
        palette=palette,
        errorbar='sd',           
        capsize=0.1,
        markers=['o', 's', 'D', 'v', '^'], 
        linestyle='-',
        linewidth=2,
        markersize=8
    )
    
    plt.title("Algorithm Efficiency Comparison", fontsize=18, fontweight='bold')
    plt.ylabel("Algorithm Efficiency", fontsize=14)
    plt.xlabel("Instance", fontsize=14)
    
    plt.xticks(rotation=45, fontsize=12)
    plt.legend(title="", bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=12)
    
    plt.tight_layout()
    plt.savefig("../plots/plot_4_efficiency.png", dpi=300)
    plt.close()
    print("✅ PLOT 4: Generated (Efficiency - Pointplot)")

    # ==========================================
    # PLOT 5: AVERAGE NUMBER OF STEPS (G vs S)
    # Comparing how many times the solution was changed
    # ==========================================
    
    # Filter data to only include Greedy and Steepest
    df_gs = df[df['Algorithm'].isin(['G', 'S'])]
    
    # Hardcode the Set1 palette colors to maintain consistency
    gs_palette = {'G': '#984ea3', 'S': '#ff7f00'}

    plt.figure(figsize=(16, 8))
    
    # Using pointplot to show the trend as instance size grows
    sns.pointplot(
        data=df_gs, 
        x='Instance', 
        y='Steps', 
        hue='Algorithm',
        hue_order=['G', 'S'],         # Only G and S in the legend
        order=ordered_instances,      # Keep the N-size sorting
        palette=gs_palette,           # Enforce consistent colors
        errorbar='sd',           
        capsize=0.1,
        markers=['v', '^'],           # Down-triangle for G, Up-triangle for S
        linestyle='-',
        linewidth=2,
        markersize=9
    )
    
    plt.title("Algorithm Steps: Greedy vs. Steepest", fontsize=18, fontweight='bold')
    plt.ylabel("Average Number of Steps", fontsize=14)
    plt.xlabel("Instance", fontsize=14)
    
    plt.xticks(rotation=45, fontsize=12)
    plt.legend(title="", bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=12)
    
    plt.tight_layout()
    plt.savefig("../plots/plot_5_algorithm_steps.png", dpi=300)
    plt.close()
    print("✅ PLOT 5: Generated (Algorithm Steps - G vs S)")

    # ==========================================
    # PLOT 6: AVERAGE NUMBER OF EVALUATIONS (RS, RW, G, S)
    # Comparing how many solutions were visited/evaluated
    # ==========================================
    
    # 1. Exclude Heuristic (H) since we are only interested in search algorithms
    algorithms_eval = ['RS', 'RW', 'G', 'S']
    df_evals = df[df['Algorithm'].isin(algorithms_eval)]
    
    # 2. Hardcode colors and markers to maintain consistency with previous plots
    eval_palette = {'RS': '#e41a1c', 'RW': '#377eb8', 'G': '#984ea3', 'S': '#ff7f00'}
    eval_markers = ['o', 's', 'v', '^'] 

    plt.figure(figsize=(16, 8))
    
    sns.pointplot(
        data=df_evals, 
        x='Instance', 
        y='Evaluations', 
        hue='Algorithm',
        hue_order=algorithms_eval,
        order=ordered_instances,
        palette=eval_palette,
        errorbar='sd',           
        capsize=0.1,
        markers=eval_markers,
        linestyle='-',
        linewidth=2,
        markersize=9
    )
    
    plt.title("Algorithm Evaluations: Number of Solutions Visited", fontsize=18, fontweight='bold')
    plt.ylabel("Average Number of Evaluations", fontsize=14)
    plt.xlabel("Instance", fontsize=14)
    
    # Logarithmic scale - necessary because RS/RW can have millions of evaluations, while G only has hundreds
    plt.yscale('log')
    
    plt.xticks(rotation=45, fontsize=12)
    plt.legend(title="", bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=12)
    
    plt.tight_layout()
    plt.savefig("../plots/plot_6_evaluations.png", dpi=300)
    plt.close()
    print("✅ PLOT 6: Generated (Evaluations - RS, RW, G, S)")
    
if __name__ == "__main__":
    main()