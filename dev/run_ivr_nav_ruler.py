"""Launch the IVR nav GRPO+RULER training run on SkyPilot."""

import argparse
import json
import os
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path

from dotenv import load_dotenv
import sky
from sky import ClusterStatus

load_dotenv()

DEFAULT_IMAGE_ID = "docker:nvidia/cuda:12.8.1-devel-ubuntu22.04"
METHOD_DIR = os.path.join(os.path.dirname(__file__), "../../method")
DEFAULT_TRAIN_FILE = os.path.join(
    METHOD_DIR,
    "Method - April 2, 2026 10_01_01 PM - ad874ed4-2852-42b2-b856-a4840dc473f3.jsonl",
)
DEFAULT_AUX_FILE = os.path.join(
    METHOD_DIR,
    "nav_1_3_24_gpt4o_relabeled - March 31, 2026 9_16_01 PM - c0fba847-9df3-4b8a-8190-5001cde7cc2e.jsonl",
)
VALIDATION_SCRIPT = os.path.join(METHOD_DIR, "scripts", "validate_training_data.py")

parser = argparse.ArgumentParser(
    description="Launch IVR nav GRPO+RULER fine-tuning on Qwen3.5-35B-A3B."
)
parser.add_argument("--fast", action="store_true")
parser.add_argument("--base-model", type=str, default="Qwen/Qwen3.5-35B-A3B")
parser.add_argument("--accelerator", type=str, default="H200:2")
parser.add_argument("--cluster-name", type=str, default="art-ivr-nav-ruler")
parser.add_argument("--image-id", type=str, default=DEFAULT_IMAGE_ID)
parser.add_argument("--project", type=str, default="genesisfi")
parser.add_argument("--model-name", type=str, default="ivr-nav-ruler-qwen3.5-054")
parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
parser.add_argument("--max-model-len", type=int, default=8192)
parser.add_argument("--max-seq-length", type=int, default=8192)
parser.add_argument("--max-num-seqs", type=int, default=8)
parser.add_argument("--grpo-group-size", type=int, default=16)
parser.add_argument("--batch-size", type=int, default=60)
parser.add_argument("--num-epochs", type=int, default=3)
parser.add_argument("--train-limit", type=int, default=None)
parser.add_argument("--max-steps", type=int, default=None)
parser.add_argument("--learning-rate", type=float, default=1e-6)
parser.add_argument("--eval-every", type=int, default=20)
parser.add_argument("--n-holdout-rows", type=int, default=100)
parser.add_argument("--n-test-rows", type=int, default=100)
parser.add_argument("--max-tokens", type=int, default=128)
parser.add_argument("--min-reward-std", type=float, default=0.1)
parser.add_argument("--train-file", type=str, default=DEFAULT_TRAIN_FILE)
parser.add_argument("--aux-file", type=str, default=DEFAULT_AUX_FILE)
parser.add_argument("--skip-validation", action="store_true")
parser.add_argument("--dry-run", action="store_true")
parser.add_argument(
    "--tail-logs",
    action=argparse.BooleanOptionalAction,
    default=False,
)
parser.add_argument(
    "--cancel-existing-jobs",
    action=argparse.BooleanOptionalAction,
    default=False,
)
parser.add_argument(
    "--save-checkpoint",
    action=argparse.BooleanOptionalAction,
    default=True,
)
parser.add_argument("--trainer-gpu-ids", type=int, nargs="+", default=[0])
parser.add_argument("--inference-gpu-ids", type=int, nargs="+", default=[1])
args = parser.parse_args()

cluster_name = args.cluster_name
cluster_prefix = os.environ.get("CLUSTER_PREFIX")
if cluster_prefix:
    cluster_name = f"{cluster_prefix}-{cluster_name}"

train_file = str(Path(args.train_file).expanduser().resolve())
aux_file = str(Path(args.aux_file).expanduser().resolve())

setup_script = textwrap.dedent("""\
    echo 'Setting up environment...'
    apt-get update
    apt-get install -y python3 python3-pip python-is-python3 git curl ninja-build
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
""")

trainer_gpu_ids = ",".join(str(i) for i in args.trainer_gpu_ids)
inference_gpu_ids = ",".join(str(i) for i in args.inference_gpu_ids)
runtime_env: dict[str, str] = {
    "PROJECT": args.project,
    "MODEL_NAME": args.model_name,
    "BASE_MODEL": args.base_model,
    "GPU_MEMORY_UTILIZATION": str(args.gpu_memory_utilization),
    "MAX_MODEL_LEN": str(args.max_model_len),
    "MAX_SEQ_LENGTH": str(args.max_seq_length),
    "MAX_NUM_SEQS": str(args.max_num_seqs),
    "ENFORCE_EAGER": "true",
    "LOAD_IN_4BIT": "false",
    "LOAD_IN_16BIT": "true",
    "ENABLE_THINKING": "false",
    "ROLLOUT_WEIGHTS_MODE": "merged",
    "TRAINER_GPU_IDS": trainer_gpu_ids,
    "INFERENCE_GPU_IDS": inference_gpu_ids,
    "GRPO_GROUP_SIZE": str(args.grpo_group_size),
    "BATCH_SIZE": str(args.batch_size),
    "NUM_EPOCHS": str(args.num_epochs),
    "SAVE_CHECKPOINT": "true" if args.save_checkpoint else "false",
    "LEARNING_RATE": str(args.learning_rate),
    "EVAL_EVERY": str(args.eval_every),
    "N_HOLDOUT_ROWS": str(args.n_holdout_rows),
    "N_TEST_ROWS": str(args.n_test_rows),
    "MAX_TOKENS": str(args.max_tokens),
    "MIN_REWARD_STD": str(args.min_reward_std),
    "TRAIN_FILE": "/tmp/method-data/method.jsonl",
    "AUX_FILE": "/tmp/method-data/nav.jsonl",
    "ART_SERVER_MONITOR_TIMEOUT": "120",
}
if args.train_limit is not None:
    runtime_env["TRAIN_LIMIT"] = str(args.train_limit)
if args.max_steps is not None:
    runtime_env["MAX_STEPS"] = str(args.max_steps)


def render_env_block(env: dict[str, str]) -> str:
    return "".join(
        f"    {key}={shlex.quote(str(value))} \\\n"
        for key, value in env.items()
    )


runtime_env_block = render_env_block(runtime_env)

run_script = textwrap.dedent(
    f"""\
    source $HOME/.local/bin/env
    cd ~/sky_workdir
    ~/.local/bin/uv sync --extra backend
    # Patch unsloth grouped_mm for mixed-dtype LoRA training (bf16 weights + fp32 activations).
    # torch._grouped_mm doesn't support mixed dtypes; cast activations down to weight dtype.
    # See: lab-log-2026-04-01.md Bug 3 in PeterWofford/method
    python3 -c "
import pathlib
for p in pathlib.Path('.venv').rglob('moe_utils.py'):
    t = p.read_text()
    old = 'return torch._grouped_mm(inputs, weight, offs=offsets)'
    new = 'inputs = inputs.to(weight.dtype); return torch._grouped_mm(inputs, weight, offs=offsets)'
    if old in t:
        p.write_text(t.replace(old, new))
        print(f'Patched {{p}}')
"
    rm -rf unsloth_compiled_cache/
{runtime_env_block}\
    ~/.local/bin/uv run dev/train_ruler.py
"""
)

task = sky.Task(
    name="ivr-nav-ruler",
    setup=setup_script,
    run=run_script,
    workdir="/Users/pwofford/src/ART",
)
task.set_resources(
    sky.Resources(
        accelerators=args.accelerator,
        cloud=sky.clouds.Kubernetes(),
        image_id=args.image_id,
    )
)
task.set_file_mounts({
    "~/sky_workdir/.env": "/Users/pwofford/src/method/.env",
    "/tmp/method-data/method.jsonl": train_file,
    "/tmp/method-data/nav.jsonl": aux_file,
})

effective_config = {
    "cluster_name": cluster_name,
    "base_model": args.base_model,
    "project": args.project,
    "accelerator": args.accelerator,
    "image_id": args.image_id,
    "gpu_memory_utilization": args.gpu_memory_utilization,
    "max_model_len": args.max_model_len,
    "max_seq_length": args.max_seq_length,
    "max_num_seqs": args.max_num_seqs,
    "grpo_group_size": args.grpo_group_size,
    "batch_size": args.batch_size,
    "num_epochs": args.num_epochs,
    "train_limit": args.train_limit,
    "max_steps": args.max_steps,
    "learning_rate": args.learning_rate,
    "eval_every": args.eval_every,
    "n_holdout_rows": args.n_holdout_rows,
    "n_test_rows": args.n_test_rows,
    "max_tokens": args.max_tokens,
    "min_reward_std": args.min_reward_std,
    "save_checkpoint": args.save_checkpoint,
    "skip_validation": args.skip_validation,
    "dry_run": args.dry_run,
    "tail_logs": args.tail_logs,
    "cancel_existing_jobs": args.cancel_existing_jobs,
    "train_file": train_file,
    "aux_file": aux_file,
    "trainer_gpu_ids": args.trainer_gpu_ids,
    "inference_gpu_ids": args.inference_gpu_ids,
    "runtime_env": runtime_env,
}

print("Effective launch config:")
print(json.dumps(effective_config, indent=2, sort_keys=True))
print("Runtime env block:")
print(runtime_env_block, end="")
sys.stdout.flush()

if not args.skip_validation:
    print("\nRunning training data validation...", flush=True)
    subprocess.run(
        [
            sys.executable,
            VALIDATION_SCRIPT,
            "--train-file",
            train_file,
            "--eval-file",
            aux_file,
        ],
        check=True,
    )
else:
    print("\nSkipping training data validation.")

if args.dry_run:
    print("\nDry run complete; not launching.")
    raise SystemExit(0)

cluster_status = sky.stream_and_get(sky.status(cluster_names=[cluster_name]))
if cluster_status:
    existing_status = cluster_status[0]["status"]
    if existing_status == ClusterStatus.UP:
        if args.cancel_existing_jobs:
            print(f"Cluster {cluster_name} is UP. Canceling active jobs because --cancel-existing-jobs was set...")
            sky.stream_and_get(sky.cancel(cluster_name, all=True))
        else:
            raise SystemExit(
                f"Cluster {cluster_name} is already UP. "
                "Refusing to cancel or reuse it by default. "
                "Choose a new --cluster-name, wait for the existing run to finish, "
                "or pass --cancel-existing-jobs if you intentionally want to cancel the existing work."
            )
    if existing_status == ClusterStatus.INIT:
        raise SystemExit(
            f"Cluster {cluster_name} is currently INIT. "
            "Refusing to launch onto a cluster that is still provisioning or setting up. "
            "Choose a new --cluster-name or wait for the existing launch to settle first."
        )

job_id, _ = sky.stream_and_get(
    sky.launch(
        task,
        cluster_name=cluster_name,
        retry_until_up=True,
        idle_minutes_to_autostop=60,
        down=True,
        fast=args.fast,
    )
)

print(f"Job submitted (ID: {job_id}).")
if args.tail_logs:
    print("Streaming logs...")
    exit_code = sky.tail_logs(cluster_name=cluster_name, job_id=job_id, follow=True)
    print(f"Job {job_id} finished with exit code {exit_code}.")
else:
    print("Not streaming logs by default; local log/API interruptions will not affect the remote job.")
