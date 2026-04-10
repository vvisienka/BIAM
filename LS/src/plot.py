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
    print("🕵️‍♀️ QAP Expert Analysis (Refined Plot 1) started...")

    # Update the path if your results.csv is in another directory
    if not os.path.exists("../results.csv"):
        print("❌ Error: results.csv not found!")
        return

    df = pd.read_csv("../results.csv", skipinitialspace=True)

    # --- CHANGE 1: CLEAN INSTANCE NAMES ---
    # Extract just the base name without path or .dat extension (e.g., /nug12.dat -> nug12)
    df['Instance'] = df['Instance'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])

    # --- CHANGE 2: RENAME ALGORITHMS AND SET ORDER ---
    alg_mapping = {
        'RandomSearch': 'RS',
        'RandomWalk': 'RW',
        'Heuristic': 'H',
        'Greedy': 'G',
        'Steepest': 'S'
    }
    df['Algorithm'] = df['Algorithm'].replace(alg_mapping)
    
    # Define the strict order for the plots
    algorithms_order = ['RS', 'RW', 'H', 'G', 'S']
    # Filter only algorithms that actually exist in the dataframe to avoid errors
    algorithms = [alg for alg in algorithms_order if alg in df['Algorithm'].unique()]

    # Protect against missing/zero OptCost to prevent division by zero
    df['OptCost'] = df['OptCost'].replace(0, np.nan)

    # 1. QUALITY METRICS
    # Quality Gap: (f_A - f_OPT) / f_OPT (Lower is better)
    df['Quality_Gap'] = (df['BestCost'] - df['OptCost']) / df['OptCost']
    
    # Q-Score: f_OPT / f_A (Higher is better, max 1.0)
    df['Q_Score'] = df['OptCost'] / df['BestCost']

    # 2. EFFICIENCY METRIC (E = Q / ln(T))
    df['Efficiency'] = df['Q_Score'] / np.log(np.maximum(df['TimeMicros'], 2))

    # Sort instances by extracted size N, then by name
    df['Size'] = df['Instance'].apply(extract_size)
    df = df.sort_values(by=['Size', 'Instance'])
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

    # Create subplots with shared Y axis
    fig, axes = plt.subplots(1, len(algorithms), figsize=(5 * len(algorithms), 6), sharey=True)
    
    # Handle case if there's only 1 algorithm tested
    if len(algorithms) == 1:
        axes = [axes]

    for ax, alg in zip(axes, algorithms):
        subset = agg_q[agg_q['Algorithm'] == alg]

        # MAX (Worst case)
        ax.plot(subset['Instance'], subset['Gap_Max'], marker='^', color='#e41a1c', 
                linestyle=':', linewidth=2, markersize=8, label='Max (Worst)')
        # AVERAGE
        ax.plot(subset['Instance'], subset['Gap_Mean'], marker='o', color='#377eb8', 
                linestyle='-', linewidth=2.5, markersize=8, label='Average')
        # MIN (Best case)
        ax.plot(subset['Instance'], subset['Gap_Min'], marker='v', color='#4daf4a', 
                linestyle='--', linewidth=2, markersize=8, label='Min (Best)')

        ax.set_title(f"{alg}", fontweight='bold', fontsize=14)
        ax.set_xticks(range(len(subset['Instance'])))
        ax.set_xticklabels(subset['Instance'], rotation=45, ha='right')
        
        # --- CHANGE 3: X-AXIS NAME ---
        ax.set_xlabel("Instance", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)

    # --- CHANGE 4: Y-AXIS NAME & PLOT TITLE ---
    axes[0].set_ylabel("Quality", fontsize=12)
    plt.suptitle("Quality comparison", fontsize=16, fontweight='bold', y=1.05)
    
    # --- CHANGE 5: LEGEND IN THE TOP RIGHT CORNER OF THE LAST PLOT ---
    # loc='upper right' places it strictly inside the top-right corner of the axes bounds
    axes[-1].legend(loc='upper right', title="10-run summary:")

    plt.tight_layout()
    plt.savefig("plot_1_quality_trends.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ PLOT 1: Generated (Quality comparison)")

    # # ==========================================
    # # PLOT 2: EFFICIENCY (E = Q / ln(T))
    # # ==========================================
    # plt.figure(figsize=(14, 7))
    # sns.barplot(
    #     data=df[df['Algorithm'] != 'H'], # Exclude Heuristic using new acronym
    #     x='Instance', y='Efficiency', hue='Algorithm', hue_order=[a for a in algorithms if a != 'H'],
    #     order=ordered_instances, palette=palette, errorbar='sd', capsize=0.1
    # )
    # plt.title("Algorithm Efficiency: E = Q / ln(T)", fontsize=16, fontweight='bold')
    # plt.ylabel("Efficiency Score (Higher is Better)")
    # plt.xlabel("Instance")
    # plt.xticks(rotation=45)
    # plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    # plt.tight_layout()
    # plt.savefig("plot_2_efficiency_barplot.png", dpi=300)
    # plt.close()
    # print("✅ PLOT 2: Generated (Efficiency Barplot)")

    # # ==========================================
    # # PLOT 3: PARETO FRONT (Time vs Quality)
    # # ==========================================
    # plt.figure(figsize=(12, 8))
    
    # pareto_df = df[df['Algorithm'] != 'H'].groupby(['Algorithm', 'Instance']).agg(
    #     Avg_Time=('TimeMicros', 'mean'),
    #     Avg_Gap=('Quality_Gap', 'mean')
    # ).reset_index()

    # sns.scatterplot(
    #     data=pareto_df, x='Avg_Time', y='Avg_Gap', hue='Algorithm', hue_order=[a for a in algorithms if a != 'H'],
    #     style='Algorithm', s=150, palette=palette, alpha=0.8
    # )
    
    # pareto_df = pareto_df.sort_values('Avg_Time')
    # front_x, front_y = [], []
    # min_gap = float('inf')
    
    # for _, row in pareto_df.iterrows():
    #     if row['Avg_Gap'] < min_gap:
    #         front_x.append(row['Avg_Time'])
    #         front_y.append(row['Avg_Gap'])
    #         min_gap = row['Avg_Gap']
            
    # plt.plot(front_x, front_y, color='black', linestyle='--', linewidth=2, label='Pareto Front')

    # plt.title("Pareto Analysis: Average Time vs Average Quality", fontsize=16, fontweight='bold')
    # plt.xlabel("Average Time [microseconds] (Log Scale)\n<-- FASTER")
    # plt.ylabel("Quality (Relative Gap)\n<-- BETTER")
    # plt.xscale('log')
    
    # handles, labels = plt.gca().get_legend_handles_labels()
    # plt.legend(handles=handles, labels=labels, bbox_to_anchor=(1.01, 1), loc='upper left')
    
    # plt.tight_layout()
    # plt.savefig("plot_3_pareto_scatter.png", dpi=300)
    # plt.close()
    # print("✅ PLOT 3: Generated (Pareto Scatter Plot)")

    # # ==========================================
    # # PLOT 4: EVALUATIONS (Exploration)
    # # ==========================================
    # plt.figure(figsize=(14, 7))
    # sns.boxplot(
    #     data=df[df['Algorithm'] != 'H'], 
    #     x='Instance', y='Evaluations', hue='Algorithm', hue_order=[a for a in algorithms if a != 'H'],
    #     order=ordered_instances, palette=palette
    # )
    # plt.title("Exploration: Visited Solutions (Boxplot)", fontsize=16, fontweight='bold')
    # plt.ylabel("Number of Evaluations (Log Scale)")
    # plt.xlabel("Instance")
    # plt.yscale('log')
    # plt.xticks(rotation=45)
    # plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    # plt.tight_layout()
    # plt.savefig("plot_4_evaluations_boxplot.png", dpi=300)
    # plt.close()
    # print("✅ PLOT 4: Generated (Evaluations Boxplot)")

    # # ==========================================
    # # PLOT 5: STEPS (Exploitation - Local Search Only)
    # # ==========================================
    # plt.figure(figsize=(14, 7))
    # steps_df = df[df['Algorithm'].isin(['G', 'S'])]
    # sns.barplot(
    #     data=steps_df, x='Instance', y='Steps', hue='Algorithm', hue_order=['G', 'S'],
    #     order=ordered_instances, palette=['#e41a1c', '#377eb8'], errorbar='sd', capsize=0.1
    # )
    # plt.title("Exploitation: Average Number of Executed Steps (Local Algorithms)", fontsize=16, fontweight='bold')
    # plt.ylabel("Number of Executed Steps (Swaps)")
    # plt.xlabel("Instance")
    # plt.xticks(rotation=45)
    # plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    # plt.tight_layout()
    # plt.savefig("plot_5_steps_barplot.png", dpi=300)
    # plt.close()
    # print("✅ PLOT 5: Generated (Steps Barplot)")

    # print("\n🎉 All plots have been successfully generated and saved!")

if __name__ == "__main__":
    main()