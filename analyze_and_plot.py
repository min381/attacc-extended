import pandas as pd
import matplotlib.pyplot as plt
import os
import subprocess
import numpy as np
import glob
from itertools import product

# ==============================================================================
# --- 1. EXPERIMENT CONFIGURATION ---
# ==============================================================================

MODELS_TO_TEST = [
    "GPT-13B",
    "GPT-89B",
    "GPT-175B",
    "LLAMA-7B",
    "LLAMA-65B",
    "LLAMA2-7B",
    "LLAMA2-13B",
    "LLAMA2-70B",
    "LLAMA3-8B",
    "LLAMA3-70B", 
    "Mixtral-8x7B",
    "Mixtral-8x22B"
]

SYSTEMS_TO_TEST = [
    "dgx",
    "dgx-attacc"
]

COMMON_PARAMS = {
    "batch": 32,
    "lout": 8
}

def run_experiments():
    """Runs all defined simulations, saving results to a temporary directory."""
    results_dir = 'temp_results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    print("--- Running Simulations ---")

    experiments = list(product(MODELS_TO_TEST, SYSTEMS_TO_TEST))
    for i, (model, system) in enumerate(experiments):
        output_file = os.path.join(results_dir, f"run_{i}.csv")
        command = [
            "python", "main.py",
            "--system", system,
            "--model", model,
            "--batch", str(COMMON_PARAMS["batch"]),
            "--lout", str(COMMON_PARAMS["lout"]),
            "--output", output_file
        ]
        if system == "dgx-attacc":
            command.extend(["--pipeopt", "--ffopt"])

        print(f"Running ({i+1}/{len(experiments)}): {' '.join(command)}")
        subprocess.run(command, check=True)
    
    print("\n--- All simulations complete ---\n")
    return results_dir


# ==============================================================================
# --- 2. DATA PROCESSING ---
# ==============================================================================

def process_data(results_dir):
    """Loads all CSVs, calculates derived metrics, and adds clean labels."""
    csv_files = glob.glob(os.path.join(results_dir, 'run_*.csv'))
    if not csv_files:
        print(f"Error: No result files found in {results_dir}.")
        return None

    df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)

    # --- Calculate Derived Metrics ---
    df['throughput'] = df['bs'] / (df['g_time (ms)'] / 1000)
    
    # Energy breakdown
    df['mem_energy'] = df['g_dram_energy'] + df['g_l2_energy'] + df['g_l1_energy'] + df['g_reg_energy']
    df['comp_energy'] = df['g_alu_energy']
    
    # Max Batch Size Calculation
    # Note: This is an approximation since we don't have the raw memory numbers here.
    # A more accurate way would be to run a bs=1 simulation for each.
    # For now, we'll assume a linear scaling from the bs=32 memory usage.
    df['total_required_mem_gb'] = df['required_cap'] / (1024**3)
    df['gpu_mem_gb'] = (df['cap'] / (df['bs'] / 32)) # Estimate total available memory
    df['max_batch_size'] = (df['gpu_mem_gb'] / (df['total_required_mem_gb'] / df['bs'])).astype(int)


    # --- Add Clean Labels ---
    df['num_kv_heads'] = df['cores'] // df['gqa_size']
    df['attn_type'] = np.where(df['cores'] == df['num_kv_heads'], 'MHA', 'GQA')
    df['model_label'] = df['model']
    # df['system_label'] = df['hw'].apply(lambda hw_name: 'DGX' if hw_name == 'NONE' else  f'DGX-AttAcc ({df["hw"]})')
    
    def get_system_label(hw_name):
        if hw_name == 'NONE': return 'DGX (Baseline)'
        else: return f'DGX-AttAcc ({hw_name})'
    df['system_label'] = df['hw'].apply(get_system_label)
    return df


# ==============================================================================
# --- 3. PLOTTING FUNCTIONS ---
# ==============================================================================

def plot_throughput(df, output_dir):
    """Generates a grouped bar chart for throughput."""
    pivot_df = df.pivot(index='model_label', columns='system_label', values='throughput')
    ax = pivot_df.plot(kind='bar', figsize=(14, 8), rot=0, width=0.7)
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f', padding=3)

    ax.set_ylabel('Throughput (tokens/s)')
    ax.set_xlabel('')
    ax.set_title(f'System Performance by Model (Batch Size {COMMON_PARAMS["batch"]})')
    ax.legend(title="System Type")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, '1_throughput_comparison.png')
    plt.savefig(output_path)
    print(f"Saved throughput plot to {output_path}")
    plt.close()

def plot_latency_breakdown(df, output_dir):
    """Generates a stacked bar chart for latency breakdown."""
    latency_df = df.pivot(index='model_label', columns='system_label', values=['g_matmul', 'g_fc', 'g_comm', 'g_etc'])
    
    ax = latency_df.plot(kind='bar', stacked=True, figsize=(14, 8), rot=0)

    ax.set_ylabel('Latency per Token (ms)')
    ax.set_xlabel('')
    ax.set_title(f'Latency Breakdown by Operation Type (Batch Size {COMMON_PARAMS["batch"]})')
    ax.legend(title="Layer Type", labels=['Attention (matmul)', 'Feed-Forward (fc)', 'Communication (comm)', 'Other (etc)'])
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    output_path = os.path.join(output_dir, '2_latency_breakdown.png')
    plt.savefig(output_path)
    print(f"Saved latency breakdown plot to {output_path}")
    plt.close()


def plot_energy_consumption(df, output_dir):
    """Generates grouped and stacked bar charts for energy."""
    # Plot 1: Total Energy Comparison (Grouped Bar)
    pivot_df = df.pivot(index='model_label', columns='system_label', values='g_energy (nJ)')
    ax = pivot_df.plot(kind='bar', figsize=(14, 8), rot=0, width=0.7)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=3)
    ax.set_ylabel('Energy per Token (nJ)')
    ax.set_xlabel('')
    ax.set_title(f'Total Energy Consumption (Batch Size {COMMON_PARAMS["batch"]})')
    ax.legend(title="System Type")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    output_path = os.path.join(output_dir, '3a_total_energy.png')
    plt.savefig(output_path)
    print(f"Saved total energy plot to {output_path}")
    plt.close()

    # Plot 2: Energy Breakdown (Stacked Bar)
    energy_df = df.pivot(index='model_label', columns='system_label', values=['mem_energy', 'comp_energy', 'g_comm_energy'])
    ax = energy_df.plot(kind='bar', stacked=True, figsize=(14, 8), rot=0)
    ax.set_ylabel('Energy per Token (nJ)')
    ax.set_xlabel('')
    ax.set_title(f'Energy Breakdown: Memory vs. Compute (Batch Size {COMMON_PARAMS["batch"]})')
    ax.legend(title="Energy Source", labels=['Memory', 'Compute', 'Communication'])
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    output_path = os.path.join(output_dir, '3b_energy_breakdown.png')
    plt.savefig(output_path)
    print(f"Saved energy breakdown plot to {output_path}")
    plt.close()

def plot_max_batch_size(df, output_dir):
    """Generates a bar chart for maximum theoretical batch size."""
    # We only need one entry per model, as max batch is a system property.
    # Let's just use the DGX-AttAcc numbers for this plot as it's the more capable system.
    max_batch_df = df[df['system_label'].str.contains('AttAcc')].set_index('model_label')

    ax = max_batch_df['max_batch_size'].plot(kind='bar', figsize=(14, 8), rot=0)
    ax.bar_label(ax.containers[0], padding=3)
    
    ax.set_ylabel('Maximum Theoretical Batch Size')
    ax.set_xlabel('')
    ax.set_title('Memory Capacity: Maximum Batch Size on DGX-AttAcc')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    output_path = os.path.join(output_dir, '4_max_batch_size.png')
    plt.savefig(output_path)
    print(f"Saved max batch size plot to {output_path}")
    plt.close()


# ==============================================================================
# --- 4. MAIN EXECUTION ---
# ==============================================================================

if __name__ == "__main__":
    try:
        import pandas
        import matplotlib
    except ImportError:
        print("Please install required libraries: conda install pandas matplotlib")
    else:
        # Create the main visualizations directory
        vis_dir = 'visualizations'
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

        # Step 1: Run all the simulations
        results_directory = run_experiments()

        # Step 2: Process the collected data
        processed_df = process_data(results_directory)

        # Step 3: Generate each of the plots
        if processed_df is not None:
            plot_throughput(processed_df, vis_dir)
            plot_latency_breakdown(processed_df, vis_dir)
            plot_energy_consumption(processed_df, vis_dir)
            plot_max_batch_size(processed_df, vis_dir)
            print("\nAll visualizations created successfully.")
