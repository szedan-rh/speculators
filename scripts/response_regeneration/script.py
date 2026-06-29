#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp
from datasets import load_dataset
from tqdm import tqdm

DATASET_CONFIGS = {
    "magpie": {
        "id": "Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered",
        "prompt_field": "instruction",
        "default_split": "train",
    },
    "ultrachat": {
        "id": "HuggingFaceH4/ultrachat_200k",
        "prompt_field": "prompt",
        "default_split": "train_sft",
    },
    "gsm8k": {
        "id": "openai/gsm8k",
        "prompt_field": "question",
        "default_split": "train",
        "subset": "main",
    },
    "sharegpt4v_coco": {
        "id": "Lin-Chen/ShareGPT4V",
        "default_split": "train",
        "subset": "ShareGPT4V",
        "multimodal": True,
    },
}


def _get_coco_dir():
    return os.getenv("COCO_DIR") or "coco/"


def _filter_sharegpt4v_coco(row):
    return row.get("image", "").startswith("coco/")


def _extract_sharegpt4v_coco_prompt(row):
    """Extract multimodal prompt from a ShareGPT4V COCO row."""
    coco_dir = _get_coco_dir()
    image_path = os.path.join(coco_dir, row["image"].removeprefix("coco/"))
    if not os.path.exists(image_path):
        return None, None

    convs = row.get("conversations", [])
    user_turn = next((t for t in convs if t.get("from") in ("human", "user")), None)
    if not user_turn:
        return None, None

    text = user_turn["value"].replace("<image>", "").strip()
    image_url = f"file://{Path(image_path).absolute()}"

    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": image_url}},
        {"type": "text", "text": text},
    ]}]

    user_conv = {"from": "human", "value": [
        {"type": "image", "path": str(Path(image_path).absolute())},
        {"type": "text", "text": text},
    ]}

    return messages, user_conv


def parse_args():
    """Parse command-line arguments for the script."""
    parser = argparse.ArgumentParser(
        description="Regenerate responses from Magpie instructions via vLLM Chat API."
    )
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8000/v1/chat/completions",
        help="vLLM OpenAI-compatible Chat Completions endpoint",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name exposed by vLLM (auto-detected if not specified)",
    )
    parser.add_argument(
        "--dataset",
        default="ultrachat",
        choices=list(DATASET_CONFIGS.keys()),
        help="Dataset to process",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split (defaults to dataset-specific split)",
    )
    parser.add_argument(
        "--subset",
        default=None,
        help=(
            "Dataset subset/config name "
            "(auto-detected from dataset config if not specified)"
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="Stop after N rows")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=64,
        help="Max concurrent requests",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="max_tokens for generation",
    )
    parser.add_argument(
        "--outfile",
        default=None,
        help="Output JSONL path (auto-generated if not specified)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip rows already in outfile (by uuid or idx)",
    )
    parser.add_argument(
        "--language-filter",
        default=None,
        help="Only process rows where language==this (e.g., EN)",
    )
    return parser.parse_args()


def sanitize_filename(name: str) -> str:
    """Sanitize a string to be safe for use in filenames."""
    name = re.sub(r'[/\\:*?"<>|]', "_", name)
    name = name.replace(" ", "_")
    return name.strip("._")


def load_seen(path: str):
    """Load previously processed record IDs from output file."""
    seen = set()
    if not os.path.isfile(path):
        return seen

    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = obj.get("uuid") or obj.get("idx")
            if key is not None:
                seen.add(str(key))
    return seen


async def detect_model(endpoint: str) -> str:
    """Automatically detect the model name from the vLLM server."""
    models_endpoint = endpoint.replace("/v1/chat/completions", "/v1/models")

    timeout = aiohttp.ClientTimeout(total=10)
    try:
        async with (
            aiohttp.ClientSession(timeout=timeout) as session,
            session.get(models_endpoint) as response,
        ):
            data = await response.json()
            models = data.get("data", [])
            if models:
                model_name = models[0]["id"]
                print(f"Auto-detected model: {model_name}")
                return model_name
            raise ValueError("No models found at endpoint")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(
            f"Failed to auto-detect model from {models_endpoint}: {e}\n"
            f"Please specify model with --model argument"
        ) from e


async def worker(
    sem: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    queue: "asyncio.Queue[dict[str, Any]]",
    args,
    out_fh,
    endpoint: str,
    progress,
    stats: dict[str, int],
):
    """Worker that pulls items from queue and sends them to the vLLM endpoint."""
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return

        idx = item["idx"]
        messages = item.get("messages") or [
            {"role": "user", "content": item["prompt"]}
        ]
        payload = {
            "model": args.model,
            "messages": messages,
            "max_tokens": args.max_tokens,
        }

        start_time = time.time()
        try:
            async with sem, session.post(endpoint, json=payload) as response:
                data = await response.json()

            choice = data["choices"][0]
            message = choice["message"]
            generated_text = message["content"]
            reasoning_content = message.get("reasoning_content")
            if reasoning_content is None:
                reasoning_content = message.get("reasoning")
            finish_reason = choice.get("finish_reason")
            latency = time.time() - start_time

            # Format output in conversations structure
            metadata = {
                "idx": idx,
                "finish_reason": finish_reason,
                "latency_s": round(latency, 3),
                "usage": data.get("usage"),
                "endpoint": endpoint,
            }

            # Only include reasoning_content if it exists
            if reasoning_content is not None:
                metadata["reasoning_content"] = reasoning_content

            user_conv = item.get("user_conv") or {
                "from": "human", "value": item["prompt"]
            }
            output = {
                "id": item.get("uuid") or f"sample_{idx}",
                "conversations": [
                    user_conv,
                    {"from": "gpt", "value": generated_text},
                ],
                "metadata": metadata,
            }
            out_fh.write(json.dumps(output, ensure_ascii=False) + "\n")
            out_fh.flush()
            stats["ok"] += 1
        except Exception as e:  # noqa: BLE001
            user_conv = item.get("user_conv") or {
                "from": "human", "value": item["prompt"]
            }
            error_output = {
                "id": item.get("uuid") or f"sample_{idx}",
                "conversations": [user_conv],
                "metadata": {
                    "idx": idx,
                    "error": repr(e),
                    "endpoint": endpoint,
                },
            }
            out_fh.write(json.dumps(error_output, ensure_ascii=False) + "\n")
            out_fh.flush()
            stats["errors"] += 1
        finally:
            progress.set_postfix(
                ok=stats["ok"],
                errors=stats["errors"],
                refresh=False,
            )
            progress.update(1)
            queue.task_done()


async def main():
    """Main async function to process dataset through vLLM endpoints."""
    args = parse_args()

    endpoint = args.endpoint
    print(f"Using endpoint: {endpoint}")

    # Auto-detect model if not specified
    if args.model is None:
        args.model = await detect_model(endpoint)

    print(f"Using model: {args.model}")

    # Get dataset configuration
    dataset_config = DATASET_CONFIGS[args.dataset]
    dataset_id = dataset_config["id"]
    is_multimodal = dataset_config.get("multimodal", False)
    prompt_field = dataset_config.get("prompt_field")

    # Use dataset-specific defaults if not provided
    split = args.split if args.split is not None else dataset_config["default_split"]
    subset = args.subset if args.subset is not None else dataset_config.get("subset")

    # Generate output filename if not specified
    if args.outfile is None:
        # Extract simple model name from full path
        model_name = args.model.split("/")[-1] if "/" in args.model else args.model
        model_name = sanitize_filename(model_name)
        args.outfile = f"{args.dataset}_{model_name}.jsonl"

    print(f"Using dataset: {dataset_id}")
    print(f"Split: {split}")
    print(f"Prompt field: {prompt_field}")
    print(f"Output file: {args.outfile}")
    print()

    seen_ids = load_seen(args.outfile) if args.resume else set()
    dataset = load_dataset(dataset_id, name=subset, split=split, streaming=True)

    queue: asyncio.Queue = asyncio.Queue(maxsize=args.concurrency * 4)
    semaphore = asyncio.Semaphore(args.concurrency)

    timeout = aiohttp.ClientTimeout(total=None, sock_connect=90, sock_read=None)
    connector = aiohttp.TCPConnector(
        limit=None, force_close=False, enable_cleanup_closed=True
    )
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession(
        timeout=timeout, connector=connector, headers=headers
    ) as session:
        with (
            open(args.outfile, "a", encoding="utf-8") as output_file,  # noqa: ASYNC230
            tqdm(
                total=args.limit,
                desc="Generating responses",
                unit="sample",
                dynamic_ncols=True,
            ) as progress,
        ):
            stats = {"ok": 0, "errors": 0}
            workers = [
                asyncio.create_task(
                    worker(
                        semaphore,
                        session,
                        queue,
                        args,
                        output_file,
                        endpoint,
                        progress,
                        stats,
                    )
                )
                for _ in range(args.concurrency)
            ]

            processed_count = 0
            for index, row in enumerate(dataset):
                if args.limit is not None and processed_count >= args.limit:
                    break

                if args.language_filter and row.get("language") != args.language_filter:
                    continue

                uuid = row.get("uuid")
                key = str(uuid or index)
                if key in seen_ids:
                    continue

                if is_multimodal and args.dataset == "sharegpt4v_coco":
                    if not _filter_sharegpt4v_coco(row):
                        continue
                    api_messages, user_conv = _extract_sharegpt4v_coco_prompt(row)
                    if api_messages is None:
                        continue
                    await queue.put({
                        "idx": index,
                        "uuid": uuid,
                        "messages": api_messages,
                        "user_conv": user_conv,
                    })
                else:
                    prompt = row.get(prompt_field)
                    if not prompt:
                        continue
                    await queue.put({
                        "idx": index,
                        "uuid": uuid,
                        "prompt": prompt,
                    })
                processed_count += 1

            # Signal workers to stop
            for _ in range(len(workers)):
                await queue.put(None)
            await asyncio.gather(*workers)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
