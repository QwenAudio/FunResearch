import argparse
import json
import os
import sys
import base64
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm


def resolve_audio_path(audio_path: str, repo_root: Path | None) -> str:
    """Resolve repo-relative audio paths to absolute paths."""
    if os.path.isabs(audio_path) and os.path.exists(audio_path):
        return audio_path
    if repo_root is None:
        return audio_path
    candidates = [
        repo_root / audio_path,
        repo_root / "data" / audio_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return audio_path


def extract_label(item: dict) -> str | None:
    """Support ms-swift (solution) and chat (assistant) formats."""
    if "solution" in item and item["solution"]:
        return item["solution"]
    messages = item.get("messages", [])
    if messages and messages[-1].get("role") in ("assistant", "model"):
        return messages[-1].get("content")
    return None


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Call a vLLM multimodal model with audio for inference on a JSON dataset.")

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="The host of the vLLM server."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=12355,
        help="The port of the vLLM server."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen2.5-Omni-7B",
        help="The served-model-name used in the vLLM deployment command."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the LLaMA-Factory format input JSON file."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="inference_results.jsonl",
        help="Path to the output JSONL file to save results."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for sampling. 0.0 means deterministic output."
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help="Repository root for resolving relative audio paths.",
    )

    return parser.parse_args()


def encode_audio_to_base64(audio_path: str) -> str:
    """Read an audio file and encode it as a Base64 string."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")
    with open(audio_path, "rb") as audio_file:
        binary_data = audio_file.read()
        return base64.b64encode(binary_data).decode('utf-8')


def query_vllm_multimodal_api(
    client: OpenAI,
    model_name: str,
    messages_to_send: list,
    max_tokens: int,
    temperature: float,
):
    """Send a multimodal request to the vLLM OpenAI-compatible API."""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages_to_send,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # Log the error and return a sentinel string on failure
        print(f"\nAPI request failed: {e}", file=sys.stderr)
        return f"API_CALL_FAILED: {e}"


def main():
    """Main entry point."""
    args = parse_arguments()
    repo_root = Path(args.repo_root).resolve() if args.repo_root else None

    # 1. Initialize OpenAI client
    client = OpenAI(
        base_url=f"http://{args.host}:{args.port}/v1",
        api_key="NOT_USED"
    )

    print(f"Starting inference using model '{args.model_name}' via API at '{client.base_url}'")
    print(f"Input data: {args.input_file}")
    print(f"Output will be saved to: {args.output_file}")

    # 2. Load input file
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f_in:
            dataset = json.load(f_in)
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_file}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Failed to parse JSON from {args.input_file}", file=sys.stderr)
        sys.exit(1)

    # 3. Ensure output directory exists
    try:
        output_dir = os.path.dirname(args.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {output_dir}: {e}", file=sys.stderr)
        sys.exit(1)

    # 4. Open output file and iterate over samples
    with open(args.output_file, 'w', encoding='utf-8') as f_out:
        for item in tqdm(dataset, desc="Processing items"):
            if not isinstance(item, dict) or 'messages' not in item or 'audios' not in item:
                print(f"\nSkipping invalid item: {item}", file=sys.stderr)
                continue

            label = extract_label(item)
            if label is None:
                print("\nSkipping item without label/solution.", file=sys.stderr)
                continue

            num_audios = len(item['audios'])
            messages_to_send = []

            # Use messages[0] text as the system prompt (functions + environment context)
            system_prompt_content = item['messages'][0]['content']
            messages_to_send.append({"role": "system", "content": system_prompt_content})

            # Build full dialogue history from all audio turns
            # Assumption: audios[i] corresponds to messages[2*i] user turns
            all_audios_encoded = True
            for i in range(num_audios):
                audio_path = resolve_audio_path(item['audios'][i], repo_root)
                user_msg_index = 2 * i
                assistant_msg_index = 2 * i + 1

                # A. Encode current audio file
                try:
                    audio_base64 = encode_audio_to_base64(audio_path)
                except FileNotFoundError as e:
                    print(f"\n{e}", file=sys.stderr)
                    all_audios_encoded = False
                    break  # Skip this item; context is incomplete

                # B. Build multimodal user message
                user_content_parts = []

                # For i > 0, include user text (i=0 text is already in the system prompt)
                if i > 0:
                    user_text = item['messages'][user_msg_index]['content']
                    user_content_parts.append({"type": "text", "text": user_text})

                # Every user turn includes audio
                user_content_parts.append({
                    "type": "audio_url",
                    "audio_url": {"url": f"data:audio/wav;base64,{audio_base64}"}
                })

                messages_to_send.append({
                    "role": "user",
                    "content": user_content_parts
                })

                # C. Add assistant history (exclude the final label turn)
                if assistant_msg_index < len(item['messages']) - 1:
                    assistant_text = item['messages'][assistant_msg_index]['content']
                    messages_to_send.append({
                        "role": "assistant",
                        "content": assistant_text
                    })

            if not all_audios_encoded:
                continue

            # 5. Call vLLM multimodal API
            prediction = query_vllm_multimodal_api(
                client=client,
                model_name=args.model_name,
                messages_to_send=messages_to_send,
                max_tokens=args.max_tokens,
                temperature=args.temperature
            )

            # 6. Write result record (last audio path kept for traceability)
            result_data = {
                "prediction": prediction,
                "label": label,
                "audio_path": item['audios'][-1]
            }

            # 7. Append to output JSONL
            f_out.write(json.dumps(result_data, ensure_ascii=False) + '\n')

    print(f"\nInference complete. Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
