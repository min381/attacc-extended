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
        "model": "GPT-175B",
        "system": "dgx",
        "label": "GPT-175B (MHA) on DGX"
    },
    {
        "model": "GPT-175B",
        "system": "dgx-attacc",
        "label": "GPT-175BB (MHA) on DGX-AttAcc"
    },
    {
        "model": "LLAMA-7B",
        "system": "dgx",
        "label": "LLAMA-7B (MHA) on DGX"
    },
    {
        "model": "LLAMA-7B",
        "system": "dgx-attacc",
        "label": "LLAMA-7B (MHA) on DGX-AttAcc"
    },
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
        "model": "LLAMA2-7B",
        "system": "dgx",
        "label": "LLAMA2-7B (GQA) on DGX"
    },
    {
        "model": "LLAMA2-7B",
        "system": "dgx-attacc",
        "label": "LLAMA2-7B (GQA) on DGX-AttAcc"
    },
    {
        "model": "LLAMA2-13B",
        "system": "dgx",
        "label": "LLAMA2-13B (GQA) on DGX"
    },
    {
        "model": "LLAMA2-13B",
        "system": "dgx-attacc",
        "label": "LLAMA2-13B (GQA) on DGX-AttAcc"
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
    },
    {
        "model": "LLAMA3-8B",
        "system": "dgx",
        "label": "LLAMA3-8B (GQA) on DGX"
    },
    {
        "model": "LLAMA3-8B",
        "system": "dgx-attacc",
        "label": "LLAMA3-8B (GQA) on DGX-AttAcc"
    },
    {
        "model": "LLAMA3-70B",
        "system": "dgx",
        "label": "LLAMA3-70B (GQA) on DGX"
    },
    {
        "model": "LLAMA3-70B",
        "system": "dgx-attacc",
        "label": "LLAMA3-70B (GQA) on DGX-AttAcc"
    },
    {
        "model": "Mixtral-8x7B",
        "system": "dgx",
        "label": "Mixtral-8x7B (GQA) on DGX"
    },
    {
        "model": "Mixtral-8x7B",
        "system": "dgx-attacc",
        "label": "Mixtral-8x7B (GQA) on DGX-AttAcc"
    },
    {
        "model": "Mixtral-8x22B",
        "system": "dgx",
        "label": "Mixtral-8x22B (GQA) on DGX"
    },
    {
        "model": "Mixtral-8x22B",
        "system": "dgx-attacc",
        "label": "Mixtral-8x22B (GQA) on DGX-AttAcc"
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
    # df['model_label'] = df['model']
    def get_system_label(hw_name):
        if hw_name == 'NONE': return 'DGX (Baseline)'
        else: return f'DGX-AttAcc ({hw_name})'
    df['system_label'] = df['hw'].apply(get_system_label)

    # --- Visualization (same as before) ---
    pivot_df = df.pivot(index='model', columns='system_label', values='throughput')
    
    x = np.arange(len(pivot_df.index))
    width = 0.35
    num_bars = len(pivot_df.columns)

    fig, ax = plt.subplots(figsize=(12, 7))
    
    for i, system_name in enumerate(pivot_df.columns):
        offset = (i - (num_bars - 1) / 2) * width
        rects = ax.bar(x + offset, pivot_df[system_name], width, label=system_name) # Nicer label
        ax.bar_label(rects, padding=3, fmt='%.0f')

    ax.set_ylabel('Throughput (tokens/s)')
    ax.set_title(f'System Performance: MoE (Batch Size {COMMON_PARAMS["batch"]})')
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_df.index)
    ax.legend(title="System Type")
    fig.tight_layout()
    
    output_path = os.path.join(output_dir, 'moe_performance.png')
    plt.savefig(output_path)
    print(f"Visualization saved to: {output_path}")


if __name__ == "__main__":
    # Run the full workflow
    temp_results_dir = run_experiments()
    analyze_and_visualize(temp_results_dir)
