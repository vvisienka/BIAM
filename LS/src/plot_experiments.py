import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Ignore technical PDF backend warnings
warnings.filterwarnings("ignore", message=".*meta NOT subset.*")

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42})

def run_analysis():
    try:
        # Load data, skipping initial spaces
        df_results = pd.read_csv('../results.csv', skipinitialspace=True)
        df_exp = pd.read_csv('../experiment_data.csv', skipinitialspace=True)
        
        # Clean instance names to ensure matching
        df_results['Instance'] = df_results['Instance'].apply(lambda x: x.split('\\')[-1].split('/')[-1])
        df_exp['Instance'] = df_exp['Instance'].apply(lambda x: x.split('\\')[-1].split('/')[-1])
        
        # Get OptCost map
        opt_map = df_results.groupby('Instance')['OptCost'].first().to_dict()
        df_exp['OptCost'] = df_exp['Instance'].map(opt_map)
        
        # Calculate Quality = f_OPT / f_BEST
        df_exp['Quality'] = df_exp['OptCost'] / df_exp['BestCost']
        
        metrics = {
            'Quality': 'Quality Comparison',
            # 'Evaluations': 'Number of Evaluations Comparison',
            'TimeMicros': 'Running Time Comparison'
        }

        # Target instances for the 2x3 comparison
        target_instances = ['chr20c.dat', 'esc32a.dat']

        # Define palette and markers specifically for SA and TS in this plot
        # SA: green, P marker; TS: darker green, P marker
        alg_palette = {
            'SA': '#4daf4a', 'TS': '#2e7d32' # SA green, TS darker green
        }
        alg_markers = {'SA': 'P', 'TS': 'P'} # Both SA and TS use P marker

        for alg in ['SA', 'TS']:
            alg_subset = df_exp[df_exp['Algorithm'] == alg].copy()
            if alg_subset.empty: continue
            
            p1_name = alg_subset['P1_Name'].iloc[0]
            p2_name = alg_subset['P2_Name'].iloc[0]
            p1_ticks = sorted(alg_subset['P1_Val'].unique())

            # Create 2 rows (instances) x 3 columns (metrics)
            fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex='col')
            fig.suptitle(f'Parametric Analysis: {alg} Comparison', fontsize=18, fontweight='bold')

            for row_idx, instance in enumerate(target_instances):
                inst_subset = alg_subset[alg_subset['Instance'] == instance]
                if inst_subset.empty:
                    print(f"Warning: No data for {instance}")
                    continue

                for col_idx, (col, title) in enumerate(metrics.items()):
                    ax = axes[row_idx, col_idx]
                    sns.lineplot(data=inst_subset, x='P1_Val', y=col, hue='P2_Val', 
                                 marker=alg_markers[alg], ax=ax, palette='tab10') # Use specific marker for alg
                    
                    # Row-specific titles (Instance name)
                    if col_idx == 0:
                        ax.set_ylabel(f"{instance.split('.')[0]} Instance\n\n{ax.get_ylabel()}", fontweight='bold')
                    
                    ax.set_title(f"{title}")
                    ax.set_xlabel(p1_name if row_idx == 1 else "")
                    ax.legend(title=p2_name, loc='best')
                    ax.set_xticks(p1_ticks)

                    
                    if col == 'Quality':
                        ax.set_ylim(0.7, 1.01)
                    elif col == 'TimeMicros':
                        ax.set_yscale('log')
                        ax.set_ylim(10**3, 10**6)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f'../plots/experiment_{alg.lower()}_comparison.pdf', format='pdf', bbox_inches='tight')
            print(f"✅ Generated comparison plot for {alg}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_analysis()