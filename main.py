import os
import gc
import traceback

import torch

from training_grade import main as training_main, RLVRConfig, DeviceManager
from analysis_script import main as analysis_main


OUTPUT_DIR = os.path.abspath("./results")
EVAL_EVERY = 100

# ============================================================================
# CUDA memory stability: prevent fragmentation-induced OOM in GRPO
# ============================================================================
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def cleanup_between_modes():
    """
    Full CUDA state cleanup between training modes.

    After a CUDA error (especially ECC errors), the CUDA context can be left
    in a bad state. This function:
    1. Forces Python garbage collection to release all GPU tensors
    2. Empties the CUDA memory cache on all devices
    3. Resets the DeviceManager singleton so it re-probes healthy GPUs
    """
    print("\n--- Cleaning up CUDA state between modes ---")

    # Force garbage collection to release all references to GPU tensors
    gc.collect()

    # Empty CUDA cache on every visible device
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    # Reset DeviceManager singleton so next mode re-initializes cleanly
    DeviceManager._instance = None
    DeviceManager._initialised = False

    print("--- Cleanup complete ---\n")


def main():

    print("="*60)
    print("Running GRADE vs GRPO experiment")
    print("="*60)

    print(f"Results directory: {OUTPUT_DIR}")
    print(f"Evaluation interval: {EVAL_EVERY}")
    print(f"PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'not set')}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    completed_modes = []

    # Run each training mode with proper config
    for mode in ["grade_only", "grpo_only"]:
        try:
            print(f"\nStarting training for {mode}...")
            config = RLVRConfig(
                training_mode=mode,
                output_dir=OUTPUT_DIR,
                eval_every=EVAL_EVERY,
            )
            training_main(config)
            print(f"\nTraining finished for {mode}")
            completed_modes.append(mode)

        except Exception as e:
            print(f"\nTraining {mode} crashed!")
            print(e)
            traceback.print_exc()

        # Always clean up between modes, whether success or crash
        cleanup_between_modes()

    # Run analysis on whatever data is available (even if one mode crashed)
    if completed_modes:
        try:
            print(f"\nRunning analysis on completed modes: {completed_modes}...")
            analysis_main(results_dir=OUTPUT_DIR, eval_every=EVAL_EVERY)
            print("\nAll done!")
            print(f"Results saved in {OUTPUT_DIR}")
        except Exception as e:
            print("\nAnalysis crashed!")
            print(e)
            traceback.print_exc()
    else:
        print("\nNo training modes completed successfully. Skipping analysis.")

if __name__ == "__main__":
    main()