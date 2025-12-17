import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import json


DATASET_NAME = "HuggingFaceH4/MATH-500"
SPLIT = "test"

MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"

SECTIONS = [
    "Prealgebra",
    "Algebra",
    "Number Theory",
    "Combinatorics",
    "Geometry",
    "Intermediate Algebra",
    "Calculus"
]

NUM_PER_SECTION = 10
BATCH_SIZE = 8
MAX_NEW_TOKENS = 512

# Temperatures to test
TEMPERATURES = [0.3, 0.7, 1.0]

print("Loading MATH-500 dataset from Hugging Face...")
dataset = load_dataset(DATASET_NAME, split=SPLIT)

problems = [{
    "problem": ex["problem"].strip(),
    "section": ex["subject"].strip()
} for ex in dataset]

print(f"Loaded {len(problems)} problems.")



grouped = {s: [] for s in SECTIONS}
for p in problems:
    if p["section"] in grouped:
        grouped[p["section"]].append(p)

selected = []
for s in SECTIONS:
    available = len(grouped[s])
    if available == 0:
        print(f"Warning: no problems in {s}")
        continue

    k = min(NUM_PER_SECTION, available)
    sampled = random.sample(grouped[s], k)

    print(f"{s}: sampled {k}/{available}")
    selected.extend(sampled)

print(f"Total problems selected: {len(selected)}")



tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True
)

model.eval()



def make_prompt(problem):
    return f"""Problem:
{problem}

Answer:"""



def generate_batch(prompts, temperature):
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,           # Enable sampling
            temperature=temperature,
            top_p=0.9,                # Optional: add top_p for better control
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True
        )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


results = []

for temp in TEMPERATURES:
    print(f"\n--- Running at temperature = {temp} ---")

    for i in tqdm(range(0, len(selected), BATCH_SIZE)):
        batch = selected[i:i + BATCH_SIZE]
        prompts = [make_prompt(p["problem"]) for p in batch]
        outputs = generate_batch(prompts, temperature=temp)

        for p, out in zip(batch, outputs):
            results.append({
                "section": p["section"],
                "problem": p["problem"],
                "temperature": temp,
                "model_output": out
            })


with open("qwen25_math_outputs.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nSaved outputs to qwen25_math_outputs.json")
print(f"Total generations: {len(results)}")
print(f"Problems per section: {NUM_PER_SECTION}")
print(f"Temperatures tested: {TEMPERATURES}")
