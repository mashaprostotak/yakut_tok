import random
import re

INPUT_FILE = "../../data/processed/monolingual/yakut_clean.txt"
OUTPUT_FILE = "mlm_eval.tsv"
NUM_EXAMPLES = 1000  # Number of masked examples to generate
PUNCTUATION = set(".,!?;:\"()[]{}—-") # Common punctuation marks to exclude from masking

def is_maskable(token):
    return token and token not in PUNCTUATION and len(token) > 2

def tokenize(sentence):
    # Basic whitespace + punctuation-aware tokenizer
    return re.findall(r"\b\w+\b|[^\w\s]", sentence, re.UNICODE)

def mask_sentence(tokens):
    maskable_indices = [i for i, tok in enumerate(tokens) if is_maskable(tok)]
    if not maskable_indices:
        return None

    idx = random.choice(maskable_indices)
    original_token = tokens[idx]
    tokens[idx] = "[MASK]"
    masked_sentence = " ".join(tokens)
    return masked_sentence, original_token

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    output = []

    while len(output) < NUM_EXAMPLES:
        sentence = random.choice(lines)
        tokens = tokenize(sentence)
        result = mask_sentence(tokens.copy())
        if result:
            masked_sentence, original_token = result
            output.append((masked_sentence, original_token))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for masked, answer in output:
            f.write(f"{masked}\t{answer}\n")

    print(f"✅ Generated {len(output)} examples in {OUTPUT_FILE}")

if __name__ == "__main__":
    main()