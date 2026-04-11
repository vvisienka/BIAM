import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def load_global_optimum(instance_name):
    """
    Loads the global optimum from the _solution.dat file.
    """
    filepath = f"../../data/{instance_name}_solution.dat"
    if not os.path.exists(filepath):
        return None
        
    with open(filepath, 'r') as f:
        tokens = f.read().split() 
        
    if len(tokens) < 2:
        return None
        
    N = int(tokens[0])
    opt_perm = np.array([int(x) for x in tokens[2:2+N]])
    return opt_perm

def main():
    print("Similarity plotting started...")
    
    csv_path = "../scatter_data.csv"
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path, skipinitialspace=True)
    df['Instance'] = df['Instance'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    
    alg_mapping = {'RandomSearch': 'RS', 'RandomWalk': 'RW', 'Heuristic': 'H', 'Greedy': 'G', 'Steepest': 'S'}
    df['Algorithm'] = df['Algorithm'].replace(alg_mapping)
    algorithms = ['RS', 'RW', 'H', 'G', 'S']
    df = df[df['Algorithm'].isin(algorithms)].copy()

    def parse_solution(sol_str):
        return np.fromstring(str(sol_str), dtype=int, sep=' ')

    df['SolutionArray'] = df['Solution'].apply(parse_solution)
    df['Quality'] = df['OptCost'] / df['FinalCost']

    instances = ["els19", "chr20c"] 
    
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    palette = {'RS': '#e41a1c', 'RW': '#377eb8', 'H': '#4daf4a', 'G': '#984ea3', 'S': '#ff7f00'}

    os.makedirs("../plots", exist_ok=True)

    for instance in instances:
        instance_df = df[df['Instance'] == instance].copy()
        if instance_df.empty: continue
            
        optimum_solution = load_global_optimum(instance)
        if optimum_solution is None: continue
        N = len(optimum_solution)
        
        instance_df['Sim_to_Optimum'] = instance_df['SolutionArray'].apply(lambda x: np.sum(x == optimum_solution) / N)

        sim_to_others_list = []
        for index, row in instance_df.iterrows():
            alg, current_sol = row['Algorithm'], row['SolutionArray']
            alg_sols = instance_df[instance_df['Algorithm'] == alg]['SolutionArray'].tolist()
            matches, count = 0, 0
            for other_sol in alg_sols:
                if not np.array_equal(current_sol, other_sol): 
                    matches += np.sum(current_sol == other_sol)
                    count += 1
            sim_to_others_list.append((matches / count) / N if count > 0 else 0)
        instance_df['Sim_to_Others'] = sim_to_others_list

        fig, axes = plt.subplots(5, 2, figsize=(14, 22), sharex=True, sharey=False)

        for idx, alg in enumerate(algorithms):
            alg_df = instance_df[instance_df['Algorithm'] == alg]
            sns.scatterplot(data=alg_df, x='Quality', y='Sim_to_Others', ax=axes[idx, 0], color=palette[alg], s=80, alpha=0.6)
            sns.scatterplot(data=alg_df, x='Quality', y='Sim_to_Optimum', ax=axes[idx, 1], color=palette[alg], s=80, alpha=0.6)

        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[alg], markersize=14, label=alg) for alg in algorithms]

        for i in range(5):
            for j in range(2):
                ax = axes[i, j]
                ax.axvline(1.0, color='gray', linestyle='--', alpha=0.5)

                ax.tick_params(axis='both', which='major', labelsize=15, pad=12)
                ax.set_xlabel(""); ax.set_ylabel("")

        axes[2, 1].legend(handles=legend_handles, labels=algorithms, title="", 
                          loc='center right', fontsize=16, title_fontsize=18, borderpad=1)

        axes[0, 0].set_title("Average Similarity to Others", fontsize=20, fontweight='bold', pad=0)
        axes[0, 1].set_title("Similarity to Global Optimum", fontsize=20, fontweight='bold', pad=0)

        for idx in range(5):
            axes[idx, 0].set_ylabel("Similarity", fontsize=18, fontweight='bold', labelpad=20)
        
        axes[4, 0].set_xlabel("Quality", fontsize=18, fontweight='bold', labelpad=20)
        axes[4, 1].set_xlabel("Quality", fontsize=18, fontweight='bold', labelpad=20)

        plt.subplots_adjust(left=0.12, right=0.95, top=0.94, bottom=0.06, hspace=0.3, wspace=0.3)
        
        plt.savefig(f"../plots/plot_9_similarity_grid_{instance}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Final Grid plot with internal legend generated for: {instance}")

if __name__ == "__main__":
    main()