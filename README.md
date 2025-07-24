# LLM Jailbreak Prompt Testing

This is our scratch repo for testing LLM jailbreak techniques and prompt formatting strategies. We use this to evaluate refusal consistency, logit-level answer probabilities, and adversarial response behavior.

## Focus Areas

- Test adversarial prompts (AdvBench, SORRYBench)
- Modify formatting to affect refusal likelihood (colon hacks, bracketed answers, etc.)
- Analyze model outputs + log probabilities (token-level)

## Layout

```
llm-jailbreak-tests/
├── experiments/        # Per-user experiments (Alec, Sarvesh, Andrew)
├── prompts/            # Prompt variants, hacking strategies
├── scripts/            # Quick launchers
├── models/             # Model wrappers for HF API/local runs
└── logs/               # Raw results
```

## Example Usage

Run test with Qwen 7B:

```bash
python scripts/run_qwen7b.py --dataset advbench
```

## References
- [SORRYBench](https://huggingface.co/sorry-bench)
- [AdvBench](https://huggingface.co/datasets/walledai/AdvBench)

## Base Eval Script: `scripts/run_eval.py`

```python
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import json


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    return pipeline("text-generation", model=model, tokenizer=tokenizer)


def run_eval(dataset_name, model_name):
    gen = load_model(model_name)
    dataset = load_dataset(dataset_name, split="test")
    
    results = []
    for ex in dataset:
        prompt = ex["prompt"]
        res = gen(prompt, max_new_tokens=50, return_full_text=False)
        results.append({"prompt": prompt, "response": res[0]["generated_text"]})

    with open(f"results/{model_name.replace('/', '_')}_{dataset_name}.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["walledai/AdvBench", "sorry-bench"], required=True)
    parser.add_argument("--model", default="Qwen/Qwen1.5-7B")
    args = parser.parse_args()

    run_eval(args.dataset, args.model)
```
