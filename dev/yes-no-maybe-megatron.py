import asyncio
import os

from dotenv import load_dotenv

import art
from art.megatron import MegatronBackend
from dev.yes_no_maybe.prompts import build_prompt_variants, slice_prompts
from dev.yes_no_maybe.runtime_config import resolve_engine_args, resolve_run_config
from dev.yes_no_maybe.trainer import run_training


async def main():
    load_dotenv()

    backend = MegatronBackend()
    try:
        engine_args = resolve_engine_args()
        # base_model = os.environ.get("BASE_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
        base_model = "Qwen/Qwen3.5-35B-A3B"
        model = art.TrainableModel(
            name=os.environ.get("MODEL_NAME", "megatron-001"),
            project="yes-no-maybe-megatron",
            base_model=base_model,
            _internal_config=art.dev.InternalModelConfig(
                engine_args=engine_args,
            ),
        )
        await model.register(backend)

        run_config = resolve_run_config()
        prompts = slice_prompts(
            build_prompt_variants(),
            offset=run_config.prompt_offset,
            limit=run_config.prompt_limit,
        )
        await run_training(model, prompts, run_config)
    finally:
        await backend.close()


if __name__ == "__main__":
    asyncio.run(main())
