import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def load_global_optimum(instance_name):
    """
    Loads the global optimum from the _solution.dat file.
    QAPLIB standard format: [N] [OptCost] [perm_1] [perm_2] ... [perm_N]
    """
    filepath = f"../../data/{instance_name}_solution.dat"
    if not os.path.exists(filepath):
        print(f"⚠️ Warning: Global optimum file not found: {filepath}")
        return None
        
    with open(filepath, 'r') as f:
        tokens = f.read().split() 
        
    if len(tokens) < 2:
        return None
        
    N = int(tokens[0])
    opt_perm = np.array([int(x) for x in tokens[2:2+N]])
    return opt_perm

def main():
    print("Similarity plotting started (Adaptive Y-Scale & Global Legend)...")
    
    csv_path = "../scatter_data.csv"
    if not os.path.exists(csv_path):
        print(f"❌ Error: File not found {csv_path}!")
        return

    df = pd.read_csv(csv_path, skipinitialspace=True)
    
    df['Instance'] = df['Instance'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    
    alg_mapping = {
        'RandomSearch': 'RS', 'RandomWalk': 'RW', 'Heuristic': 'H', 'Greedy': 'G', 'Steepest': 'S'
    }
    df['Algorithm'] = df['Algorithm'].replace(alg_mapping)

    algorithms = ['RS', 'RW', 'H', 'G', 'S']
    df = df[df['Algorithm'].isin(algorithms)].copy()

    def parse_solution(sol_str):
        return np.fromstring(str(sol_str), dtype=int, sep=' ')

    df['SolutionArray'] = df['Solution'].apply(parse_solution)
    df['Quality'] = df['OptCost'] / df['FinalCost']

    instances = ["els19", "chr20c"] 
    
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    palette = {'RS': '#e41a1c', 'RW': '#377eb8', 'H': '#4daf4a', 'G': '#984ea3', 'S': '#ff7f00'}

    os.makedirs("../plots", exist_ok=True)

    for instance in instances:
        instance_df = df[df['Instance'] == instance].copy()
        if instance_df.empty:
            continue
            
        optimum_solution = load_global_optimum(instance)
        if optimum_solution is None:
            continue
            
        N = len(optimum_solution)
        
        # 1. Similarity to Optimum (Normalized: 0.0 to 1.0)
        def sim_to_global(row_sol):
            return np.sum(row_sol == optimum_solution) / N
            
        instance_df['Sim_to_Optimum'] = instance_df['SolutionArray'].apply(sim_to_global)

        # 2. Average Similarity to Others (Normalized: 0.0 to 1.0)
        sim_to_others_list = []
        for index, row in instance_df.iterrows():
            alg = row['Algorithm']
            current_sol = row['SolutionArray']
            
            alg_sols = instance_df[instance_df['Algorithm'] == alg]['SolutionArray'].tolist()
            
            matches = 0
            count = 0
            for other_sol in alg_sols:
                if not np.array_equal(current_sol, other_sol): 
                    matches += np.sum(current_sol == other_sol)
                    count += 1
                    
            avg_match = (matches / count) / N if count > 0 else 0
            sim_to_others_list.append(avg_match)
            
        instance_df['Sim_to_Others'] = sim_to_others_list

        # --- DRAWING THE 5x2 GRID ---
        # Zmieniono: sharey=False (każdy wykres ma swoją własną, idealnie dopasowaną skalę Y)
        fig, axes = plt.subplots(5, 2, figsize=(12, 18), sharex=True, sharey=False)

        for idx, alg in enumerate(algorithms):
            alg_df = instance_df[instance_df['Algorithm'] == alg]
            
            ax_others = axes[idx, 0]
            ax_opt = axes[idx, 1]
            
            if alg_df.empty:
                continue
            
            # Kolumna 0: Podobieństwo do innych
            sns.scatterplot(
                data=alg_df, x='Quality', y='Sim_to_Others', ax=ax_others,
                color=palette[alg], marker='o', s=40, alpha=0.5
            )
            
            # Kolumna 1: Podobieństwo do optimum
            sns.scatterplot(
                data=alg_df, x='Quality', y='Sim_to_Optimum', ax=ax_opt,
                color=palette[alg], marker='o', s=40, alpha=0.5
            )

        # Formatowanie osi i linii pomocniczych
        for i in range(5):
            for j in range(2):
                ax = axes[i, j]
                # Zostawiamy tylko pionową linię optimum (1.0). 
                # Poziomą (Y=1.0) usuwamy, żeby nie psuła auto-skalowania Y.
                ax.axvline(1.0, color='gray', linestyle='--', alpha=0.5)
                
                min_q = instance_df['Quality'].min()
                ax.set_xlim(left=min_q - 0.02 if not np.isnan(min_q) else 0.8, right=1.02)
                
                # Czyścimy domyślne podpisy
                ax.set_xlabel("")
                ax.set_ylabel("")

        # Dodajemy opisy Kolumn na samej górze
        axes[0, 0].set_title("Average Similarity to Others (Ratio)", fontsize=14, fontweight='bold')
        axes[0, 1].set_title("Similarity to Global Optimum (Ratio)", fontsize=14, fontweight='bold')

        # Dodajemy CZYSTY opis osi Y (bez nazw algorytmów)
        for idx in range(5):
            axes[idx, 0].set_ylabel("Sim Ratio", fontsize=12, fontweight='bold')
        
        # Dodajemy opisy osi X na samym dole
        axes[4, 0].set_xlabel("Quality (1.0 = Opt)", fontsize=12)
        axes[4, 1].set_xlabel("Quality (1.0 = Opt)", fontsize=12)

        # Dodawanie globalnej Legendy po prawej stronie
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[alg], markersize=12, alpha=0.8) for alg in algorithms]
        fig.legend(handles=legend_handles, labels=algorithms, title="Algorithms", loc='center right', bbox_to_anchor=(0.98, 0.5), fontsize=12, title_fontsize=14)

        plt.suptitle(f"Search Space Structure: {instance} (Size N={N})", fontsize=18, fontweight='bold', y=0.98)
        
        # Używamy suplots_adjust zamiast tight_layout dla precyzyjnego zostawienia miejsca na legendę (prawy margines = 85%)
        plt.subplots_adjust(left=0.08, right=0.85, top=0.94, bottom=0.05, hspace=0.15)
        
        plt.savefig(f"../plots/plot_9_similarity_grid_{instance}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Adaptive Vertical Grid plot generated for: {instance}")

if __name__ == "__main__":
    main()