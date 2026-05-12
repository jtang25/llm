# prepare_tinystories.py
from pathlib import Path
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

DATA_DIR = Path("data/tinystories")
DATA_DIR.mkdir(parents=True, exist_ok=True)

VOCAB_SIZE = 8192
TOKENIZER_PATH = DATA_DIR / "tokenizer.json"
TRAIN_BIN = DATA_DIR / "train.bin"
VAL_BIN = DATA_DIR / "val.bin"

# 1. Load dataset. HF caches it in ~/.cache/huggingface/datasets after first call.
print("Loading TinyStories...")
ds = load_dataset("roneneldan/TinyStories")
print(f"  train: {len(ds['train']):,} stories")
print(f"  val:   {len(ds['validation']):,} stories")

# 2. Train a byte-level BPE tokenizer on the train split (one-time).
if not TOKENIZER_PATH.exists():
    print(f"Training BPE tokenizer (vocab={VOCAB_SIZE})...")
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["<|endoftext|>", "<|unk|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    def text_iter():
        for ex in ds["train"]:
            yield ex["text"]

    tokenizer.train_from_iterator(text_iter(), trainer=trainer, length=len(ds["train"]))
    tokenizer.save(str(TOKENIZER_PATH))
    print(f"  saved -> {TOKENIZER_PATH}")
else:
    print(f"Tokenizer already exists at {TOKENIZER_PATH}, skipping training.")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

eot_id = tokenizer.token_to_id("<|endoftext|>")
print(f"  <|endoftext|> id = {eot_id}")

# 3. Tokenize each split and write a flat uint16 array to disk.
#    uint16 fits any vocab up to 65,535, so 8k is fine.
def tokenize_split(split_name, out_path):
    if out_path.exists():
        print(f"{out_path} already exists, skipping.")
        return
    print(f"Tokenizing {split_name}...")
    chunks = []
    total = 0
    for ex in ds[split_name]:
        ids = tokenizer.encode(ex["text"]).ids
        ids.append(eot_id)  # story separator
        chunks.append(np.asarray(ids, dtype=np.uint16))
        total += len(ids)
    arr = np.concatenate(chunks)
    arr.tofile(out_path)
    print(f"  {split_name}: {total:,} tokens -> {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

tokenize_split("train", TRAIN_BIN)
tokenize_split("validation", VAL_BIN)

print("Done. You can now train fully offline using only files in data/tinystories/.")