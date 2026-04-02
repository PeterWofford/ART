"""Launch the IVR nav GRPO+RULER training run on SkyPilot."""

import argparse
import os
import textwrap

from dotenv import load_dotenv
import sky
from sky import ClusterStatus

load_dotenv()

DEFAULT_IMAGE_ID = "docker:nvidia/cuda:12.8.1-devel-ubuntu22.04"
METHOD_DIR = os.path.join(os.path.dirname(__file__), "../../method")

parser = argparse.ArgumentParser(
    description="Launch IVR nav GRPO+RULER fine-tuning on Qwen3.5-35B-A3B."
)
parser.add_argument("--fast", action="store_true")
parser.add_argument("--base-model", type=str, default="Qwen/Qwen3.5-35B-A3B")
parser.add_argument("--accelerator", type=str, default="H200:4")
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
parser.add_argument("--learning-rate", type=float, default=1e-6)
parser.add_argument("--eval-every", type=int, default=20)
parser.add_argument("--n-holdout-rows", type=int, default=100)
parser.add_argument("--n-test-rows", type=int, default=100)
parser.add_argument("--max-tokens", type=int, default=128)
parser.add_argument("--min-reward-std", type=float, default=0.1)
parser.add_argument("--trainer-gpu-ids", type=int, nargs="+", default=[0, 1, 2])
parser.add_argument("--inference-gpu-ids", type=int, nargs="+", default=[3])
args = parser.parse_args()

cluster_name = args.cluster_name
cluster_prefix = os.environ.get("CLUSTER_PREFIX")
if cluster_prefix:
    cluster_name = f"{cluster_prefix}-{cluster_name}"

setup_script = textwrap.dedent("""\
    echo 'Setting up environment...'
    apt-get update
    apt-get install -y python3 python3-pip python-is-python3 git curl ninja-build
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
""")

trainer_gpu_ids = ",".join(str(i) for i in args.trainer_gpu_ids)
inference_gpu_ids = ",".join(str(i) for i in args.inference_gpu_ids)

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
    PROJECT={args.project} \\
    MODEL_NAME={args.model_name} \\
    BASE_MODEL={args.base_model} \\
    GPU_MEMORY_UTILIZATION={args.gpu_memory_utilization} \\
    MAX_MODEL_LEN={args.max_model_len} \\
    MAX_SEQ_LENGTH={args.max_seq_length} \\
    MAX_NUM_SEQS={args.max_num_seqs} \\
    ENFORCE_EAGER=true \\
    LOAD_IN_4BIT=false \\
    LOAD_IN_16BIT=true \\
    ENABLE_THINKING=false \\
    ROLLOUT_WEIGHTS_MODE=merged \\
    TRAINER_GPU_IDS={trainer_gpu_ids} \\
    INFERENCE_GPU_IDS={inference_gpu_ids} \\
    GRPO_GROUP_SIZE={args.grpo_group_size} \\
    BATCH_SIZE={args.batch_size} \\
    NUM_EPOCHS={args.num_epochs} \\
    LEARNING_RATE={args.learning_rate} \\
    EVAL_EVERY={args.eval_every} \\
    N_HOLDOUT_ROWS={args.n_holdout_rows} \\
    N_TEST_ROWS={args.n_test_rows} \\
    MAX_TOKENS={args.max_tokens} \\
    MIN_REWARD_STD={args.min_reward_std} \\
    TRAIN_FILE=/tmp/method-data/method.jsonl \\
    AUX_FILE=/tmp/method-data/nav.jsonl \\
    ART_SERVER_MONITOR_TIMEOUT=120 \\
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
    "/tmp/method-data/method.jsonl": (
        "/Users/pwofford/src/method/"
        "Method - March 31, 2026 10_17_31 PM"
        " - ad874ed4-2852-42b2-b856-a4840dc473f3.jsonl"
    ),
    "/tmp/method-data/nav.jsonl": (
        "/Users/pwofford/src/method/"
        "nav_1_3_24_gpt4o_relabeled - March 31, 2026 9_16_01 PM"
        " - c0fba847-9df3-4b8a-8190-5001cde7cc2e.jsonl"
    ),
})

print(f"Launching on cluster: {cluster_name}")
print(f"  base_model:              {args.base_model}")
print(f"  project:                 {args.project}")
print(f"  accelerator:             {args.accelerator}")
print(f"  gpu_memory_utilization:  {args.gpu_memory_utilization}")
print(f"  max_model_len:           {args.max_model_len}")
print(f"  grpo_group_size:         {args.grpo_group_size}")
print(f"  batch_size:              {args.batch_size}")
print(f"  num_epochs:              {args.num_epochs}")
print(f"  learning_rate:           {args.learning_rate}")
print(f"  eval_every:              {args.eval_every}")
print(f"  trainer_gpu_ids:         {args.trainer_gpu_ids}")
print(f"  inference_gpu_ids:       {args.inference_gpu_ids}")

cluster_status = sky.stream_and_get(sky.status(cluster_names=[cluster_name]))
if cluster_status and cluster_status[0]["status"] == ClusterStatus.UP:
    print(f"Cluster {cluster_name} is UP. Canceling any active jobs...")
    sky.stream_and_get(sky.cancel(cluster_name, all=True))

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

print(f"Job submitted (ID: {job_id}). Streaming logs...")
exit_code = sky.tail_logs(cluster_name=cluster_name, job_id=job_id, follow=True)
print(f"Job {job_id} finished with exit code {exit_code}.")
