import argparse
import csv
import os
from src.system import *
from src.type import *
from src.config import *
from src.ramulator_wrapper import *

RAMULATOR = False


def write_csv(logfile, perfs):
    if logfile is not None:
        firstrow = False
        if not os.path.exists(logfile):
            firstrow = True

        f = open(logfile, 'a')
        wrt = csv.writer(f)
        if firstrow:
            col_name = [
                'model', 'dtype', 'xpu', 'cap', 'bw', 'sys_opb', 'hw', 'cores',
                'pipe_level', 'is parallel', 'power constraint', 'gqa_size',
                'Lin', 'Lout', 'bs', 'required_cap', 's_flops',
                'g_flops', 's_time', 's_matmul', 's_fc', 's_comm', 's_softmax',
                's_act', 's_lnorm', 'g_time (ms)', 'g_matmul', 'g_fc', 'g_comm',
                'g_etc', 'g_qkv_time', 'g_prj_time', 'g_ff_time', 'g2g_comm',
                'c2g_comm', 'g_softmax', 'g_act', 'g_lnorm', 'g_energy (nJ)',
                'g_dram_energy', 'g_l2_energy', 'g_l1_energy', 'g_reg_energy',
                'g_alu_energy', 'g_fc_mem_energy', 'g_fc_comp_energy',
                'g_attn_mem_energy', 'g_attn_comp_energy', 'g_etc_mem_energy',
                'g_etc_comp_energy', 'g_comm_energy'
            ]
            wrt.writerow(col_name)

        for perf in perfs:
            tag, config, time, energy = perf
            info = tag + config + time + energy
            wrt.writerow(info)
        f.close()

def verify_memory_capacity(system, args):
    # Analyze memory requirements for the given configuration. 
    print("\n--- Verifying Per-Device Memory Capacity ---")

    # Step 1: Calculate memory components for a SINGLE batch (bs=1)
    weight_mem, kv_mem_per_batch, act_mem_per_batch = system.get_required_mem_capacity(1, args.lin, args.lout)

    # Step 2: Get available capacity for each device in BYTES
    gpu_mem_avail = system.devices['GPU'].aggregate_memory_capacity
    acc_mem_avail = 0
    is_heterogeneous = system.devices['Acc'] is not system.devices['GPU']
    if is_heterogeneous:
        acc_mem_avail = system.devices['Acc'].aggregate_memory_capacity

    # Step 3: Calculate the maximum possible batch size
    max_batch_gpu = float('inf')
    max_batch_acc = float('inf')

    if is_heterogeneous:
        # GPU is limited by Weights + Activations
        if gpu_mem_avail > weight_mem:
            max_batch_gpu = (gpu_mem_avail - weight_mem) // act_mem_per_batch
        # Accelerator is limited by KV Cache
        if acc_mem_avail > 0:
            max_batch_acc = acc_mem_avail // kv_mem_per_batch
    else:
        # GPU is limited by Weights + Activations + KV Cache
        gpu_per_batch_mem = act_mem_per_batch + kv_mem_per_batch
        if gpu_mem_avail > weight_mem:
            max_batch_gpu = (gpu_mem_avail - weight_mem) // gpu_per_batch_mem

    max_batch_size = int(min(max_batch_gpu, max_batch_acc))

    # Step 4: Print the detailed report
    # Convert bytes to GB for printing
    gpu_mem_avail_gb = gpu_mem_avail / (1024**3)
    acc_mem_avail_gb = acc_mem_avail / (1024**3)
    weight_mem_gb = weight_mem / (1024**3)
    act_mem_gb = (act_mem_per_batch * args.batch) / (1024**3)
    kv_mem_gb = (kv_mem_per_batch * args.batch) / (1024**3)

    print(f"System Type: {args.system}, Model: {args.model}, Batch: {args.batch}")
    if is_heterogeneous:
        gpu_req = weight_mem_gb + act_mem_gb
        print(f"  GPU Memory: Required = {gpu_req:.2f} GB (Weights+Activations), Available = {gpu_mem_avail_gb:.2f} GB")
        print(f"  Accelerator Memory: Required = {kv_mem_gb:.2f} GB (KV Cache), Available = {acc_mem_avail_gb:.2f} GB")
    else:
        gpu_req = weight_mem_gb + act_mem_gb + kv_mem_gb
        print(f"  GPU Memory: Required = {gpu_req:.2f} GB (All), Available = {gpu_mem_avail_gb:.2f} GB")

    print(f"\nMaximum Theoretical Batch Size: {max_batch_size}")
    print("------------------------------------------\n")
    
    # Step 5: Return the result of the check
    return args.batch > max_batch_size, max_batch_size


def run(system: System,
        batch,
        lin,
        lout,
        power_constraint=False,
        pipe=0,
        parallel=False,
        output_file=None):
    print("---Run simple mode Batch {} Lin {} Lout {} pipe {} parall {}---".
          format(batch, lin, lout, pipe, parallel))
    assert system.model_set, "Need to SetModel"
    perfs = []
    system.simulate(batch,
                    lin,
                    lout,
                    perfs=perfs,
                    pipe=pipe,
                    parallel_ff=parallel,
                    power_constraint=power_constraint)
    if output_file is not None:
        write_csv(output_file, perfs)


def main():
    parser = argparse.ArgumentParser(
        description="Model configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## set system configuration
    parser.add_argument(
        "--system",
        type=str,
        default="dgx",
        help="dgx (each GPU has 80GB HBM), \
              dgx-cpu (In dgx, offloading the attention layer to cpu), \
              dgx-attacc (dgx + attacc)")
    parser.add_argument(
        "--gpu",
        type=str,
        default='A100a',
        help="GPU type (A100a and H100), A100a is A100 with HBM3")
    parser.add_argument("--ngpu",
                        type=int,
                        default=8,
                        help="number of GPUs in DGX system. default=8")
    parser.add_argument("--gmemcap",
                        type=int,
                        default=80,
                        help="memory capacity per GPU (GB). default=80")



    ## set attacc configuration
    parser.add_argument("--pim",
                        type=str,
                        default='bank',
                        help="pim mode. list: bank, bg, buffer")
    parser.add_argument("--powerlimit",
                        action='store_true',
                        help="power constraint for PIM ")
    parser.add_argument("--ffopt",
                        action='store_true',
                        help="apply feedforward parallel optimization")
    parser.add_argument("--pipeopt",
                        action='store_true',
                        help="apply pipeline optimization ")

    ## set model and service environment
    parser.add_argument(
        "--model",
        type=str,
        default='GPT-175B',
        help="model list: GPT-175B, LLAMA-65B, MT-530B, OPT-66B")
    parser.add_argument("--word",
                        type=int,
                        default='2',
                        help="word size (precision): 1(INT8), 2(FP16)")
    parser.add_argument("--lin",
                        type=int,
                        default=2048,
                        help="input sequence length")
    parser.add_argument("--lout",
                        type=int,
                        default=128,
                        help="number of generated tokens")
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help=
        "batch size, default = 1"
    )

    args = parser.parse_args()

    system=System.create(args)

    # Baseline (GPU-only)
    if args.system == 'dgx-attacc':
         print("{}: ({} x {}), PIM:{}, [Lin, Lout, batch]: {}".format(
            args.system, args.gpu, args.ngpu, args.pim,
            [args.lin, args.lout, args.batch]))
    else:
         print("{}: ({} x {}), [Lin, Lout, batch]: {}".format(
            args.system, args.gpu, args.ngpu,
            [args.lin, args.lout, args.batch]))
    
    oom_detected, max_batch = verify_memory_capacity(system, args)

    # Display the warning if needed
    if oom_detected:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"!!! WARNING: Chosen batch size ({args.batch}) exceeds the       !!!")
        print(f"!!!      maximum possible batch size of {max_batch}.        !!!")
        print("!!!      Simulation results are theoretical and        !!!")
        print("!!!      this would fail on real hardware.             !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

    output_path = "output.csv"
    if os.path.exists(output_path):
        os.remove(output_path) # Use os.remove instead of os.system
    
    run(system,
        args.batch,
        args.lin,
        args.lout,
        pipe=args.pipeopt,
        parallel=args.ffopt,
        output_file=output_path,
        power_constraint=args.powerlimit)


if __name__ == "__main__":
    main()
