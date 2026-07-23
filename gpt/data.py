from pathlib import Path
from array import array

import numpy as np
import torch
from tokenizers import Tokenizer


def project_root() -> Path:
    anchor = Path(__file__).resolve().parent
    for parent in [anchor, *anchor.parents]:
        if (parent / "data").exists():
            return parent
    return anchor


ROOT = project_root()
DATA_DIR = ROOT / "data" / "fineweb_edu_small"
TOKENIZER_PATH = DATA_DIR / "gpt2_tokenizer" / "tokenizer.json"
TRAIN_PATH = DATA_DIR / "train.bin"
VAL_PATH = DATA_DIR / "val.bin"


def load_tokenizer() -> Tokenizer:
    if TOKENIZER_PATH.exists():
        return Tokenizer.from_file(str(TOKENIZER_PATH))
    TOKENIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_pretrained("gpt2")
    tok.save(str(TOKENIZER_PATH))
    return tok


def prepare_data(max_examples=1_000_000):
    """Tokenize fineweb-edu to train.bin/val.bin (90/10). No-op if they exist."""
    if TRAIN_PATH.exists() and VAL_PATH.exists():
        return
    from datasets import load_dataset

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer()
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)

    temp = DATA_DIR / "all_tokens_temp.bin"
    buf, total = array("H"), 0
    with open(temp, "wb") as f:
        for i, ex in enumerate(ds):
            if i >= max_examples:
                break
            ids = tok.encode(ex["text"]).ids
            ids.append(50256)                    # EOS
            buf.extend(ids)
            total += len(ids)
            if len(buf) >= 1_000_000:
                buf.tofile(f); buf = array("H")
        if buf:
            buf.tofile(f)

    split = int(0.9 * total)
    mm = np.memmap(temp, dtype=np.uint16, mode="r")
    mm[:split].tofile(TRAIN_PATH)
    mm[split:].tofile(VAL_PATH)
    del mm
    temp.unlink()


class Dataset:
    def __init__(self):
        self.train = np.memmap(TRAIN_PATH, dtype=np.uint16, mode="r")
        self.val = np.memmap(VAL_PATH, dtype=np.uint16, mode="r")

    def get_batch(self, split, batch_size, block_size, device):
        data = self.train if split == "train" else self.val
        ix = torch.randint(len(data) - block_size - 1, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
        if device != "cpu":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y
