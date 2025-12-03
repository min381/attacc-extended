import pandas as pd
import matplotlib.pyplot as plt
import os
import subprocess
import numpy as np
import glob

# --- 1. Experiment Configuration ---
# Define the experiments we want to run
EXPERIMENTS = [
    {
        "model": "LLAMA-65B",
        "system": "dgx",
        "label": "LLAMA-65B (MHA) on DGX"
    },
    {
        "model": "LLAMA-65B",
        "system": "dgx-attacc",
        "label": "LLAMA-65B (MHA) on DGX-AttAcc"
    },
    {
        "model": "LLAMA2-70B",
        "system": "dgx",
        "label": "LLAMA2-70B (GQA) on DGX"
    },
    {
        "model": "LLAMA2-70B",
        "system": "dgx-attacc",
        "label": "LLAMA2-70B (GQA) on DGX-AttAcc"
    }
]

# Shared simulation parameters
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

    for i, exp in enumerate(EXPERIMENTS):
        output_file = os.path.join(results_dir, f"run_{i}.csv")
        
        # Construct the command-line arguments for main.py
        command = [
            "python", "main.py",
            "--system", exp["system"],
            "--model", exp["model"],
            "--batch", str(COMMON_PARAMS["batch"]),
            "--lout", str(COMMON_PARAMS["lout"]),
            "--output", output_file
        ]
        
        # Add optimization flags only for the attacc system
        if exp["system"] == "dgx-attacc":
            command.extend(["--pipeopt", "--ffopt"])

        print(f"Running: {' '.join(command)}")
        # Run the simulation as a subprocess
        subprocess.run(command, check=True)
    
    print("\n--- All simulations complete ---\n")
    return results_dir

def analyze_and_visualize(results_dir):
    """Analyzes results and generates the plot."""
    # --- Data Loading and Processing (same as before) ---
    output_dir = 'visualizations'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    csv_files = glob.glob(os.path.join(results_dir, 'run_*.csv'))
    if not csv_files:
        print(f"Error: No result files found in {results_dir}.")
        return

    df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)
    
    df['throughput'] = df['bs'] / (df['g_time (ms)'] / 1000)
    df['model_label'] = df['model'].apply(lambda x: f"{x} (GQA)" if '2' in x or '3' in x else f"{x} (MHA)")
    
    # --- Visualization (same as before) ---
    pivot_df = df.pivot(index='model_label', columns='hw', values='throughput')
    
    x = np.arange(len(pivot_df.index))
    width = 0.35
    num_bars = len(pivot_df.columns)

    fig, ax = plt.subplots(figsize=(12, 7))
    
    for i, system_name in enumerate(pivot_df.columns):
        offset = (i - (num_bars - 1) / 2) * width
        rects = ax.bar(x + offset, pivot_df[system_name], width, label=system_name.replace('_', '-')) # Nicer label
        ax.bar_label(rects, padding=3, fmt='%.0f')

    ax.set_ylabel('Throughput (tokens/s)')
    ax.set_title(f'System Performance: MHA vs. GQA (Batch Size {COMMON_PARAMS["batch"]})')
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_df.index)
    ax.legend(title="System Type")
    fig.tight_layout()
    
    output_path = os.path.join(output_dir, 'mha_vs_gqa_performance.png')
    plt.savefig(output_path)
    print(f"Visualization saved to: {output_path}")


if __name__ == "__main__":
    # Run the full workflow
    temp_results_dir = run_experiments()
    analyze_and_visualize(temp_results_dir)
