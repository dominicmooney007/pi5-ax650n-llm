# Running Qwen2.5-VL-3B on the M5Stack LLM-8850

A step-by-step guide to running AXERA's Qwen2.5-VL-3B-Instruct (a vision-language model) on a Raspberry Pi 5 with an M5Stack LLM-8850 (AX650N) M.2 accelerator.

By the end of this guide you'll be able to pass the model an image plus a prompt and have it describe/analyse the image, either interactively or over an OpenAI-compatible HTTP API.

> This assumes you've already done the baseline setup from [`../qwen3-0.6b/RUN_QWEN3.md`](../qwen3-0.6b/RUN_QWEN3.md) — i.e. the `axclhost` driver is installed, `axcl-smi` sees the card, and `axllm` is on your `PATH`. If not, do that first.

---

## Why this guide exists

AXERA's HuggingFace repo ships a **0-byte `config.json`** for this model, so `axllm run <dir>` does not work out of the box the way Qwen3-0.6B did. This guide documents the hand-written `config.json` that makes it work, plus a non-obvious `tokenizer_type` gotcha.

The [M5Stack docs page](https://docs.m5stack.com/en/guide/ai_accelerator/llm-8850/m5_llm_8850_qwen2.5_vl) describes an older workflow using `main_axcl_aarch64` plus a Python tokenizer service on port 12345. That works but leaves you with a two-process setup and no OpenAI API — we skip it here in favour of `axllm`.

Takes about **30–40 minutes**, most of which is the 6 GB model download.

---

## Step 1 — Clone the model

```bash
mkdir -p ~/qwen2.5-vl-3b
cd ~/qwen2.5-vl-3b

git lfs install
git clone https://huggingface.co/AXERA-TECH/Qwen2.5-VL-3B-Instruct
```

Expect roughly **6 GB** of download over LFS. When it's done you should see this layout:

```
Qwen2.5-VL-3B-Instruct/
├── Qwen2.5-VL-3B-Instruct-AX650-chunk_prefill_512/   ← the weights live here
│   ├── qwen2_5_vl_p128_l0_together.axmodel           (×36 layer files, ~80 MB each)
│   ├── …
│   ├── qwen2_5_vl_p128_l35_together.axmodel
│   ├── qwen2_5_vl_post.axmodel                       (340 MB output projection)
│   ├── Qwen2.5-VL-3B-Instruct_vision_nchw448.axmodel (921 MB vision encoder)
│   └── model.embed_tokens.weight.bfloat16.bin        (622 MB token embeddings)
├── qwen2.5_tokenizer.txt                             (1.6 MB vocab)
├── qwen2_5-vl-tokenizer/                             (HF-style tokenizer — not used by axllm)
├── image/ssd_car.jpg, ssd_horse.jpg                  (sample test images)
├── config.json                                       (0 bytes — we replace this)
├── post_config.json                                  (absent — we add one)
└── run_qwen2_5_vl_image_axcl_aarch64.sh              (reference script for the old workflow)
```

Total on disk: ~12 GB (the `.git/lfs/objects/` cache doubles it — safe to delete after).

---

## Step 2 — Write the axllm config

Create **two** files inside the weight subdirectory:

**`Qwen2.5-VL-3B-Instruct-AX650-chunk_prefill_512/config.json`:**

```json
{
    "system_prompt": "You are a helpful assistant.",
    "model_name": "AXERA-TECH/Qwen2.5-VL-3B-Instruct",
    "url_tokenizer_model": "../qwen2.5_tokenizer.txt",
    "tokenizer_type": "Qwen3VL",
    "vlm_type": "Qwen2_5VL",
    "post_config_path": "post_config.json",
    "template_filename_axmodel": "qwen2_5_vl_p128_l%d_together.axmodel",
    "axmodel_num": 36,
    "filename_post_axmodel": "qwen2_5_vl_post.axmodel",
    "filename_image_encoder_axmodel": "Qwen2.5-VL-3B-Instruct_vision_nchw448.axmodel",
    "filename_tokens_embed": "model.embed_tokens.weight.bfloat16.bin",
    "tokens_embed_num": 151936,
    "tokens_embed_size": 2048,
    "vision_width": 448,
    "vision_height": 448,
    "vision_patch_size": 14,
    "vision_spatial_merge_size": 2,
    "vision_temporal_patch_size": 2,
    "use_mmap_load_embed": true,
    "use_mmap_load_layer": true,
    "devices": [0]
}
```

**`Qwen2.5-VL-3B-Instruct-AX650-chunk_prefill_512/post_config.json`:**

```json
{
    "enable_temperature": false,
    "temperature": 0.9,
    "enable_repetition_penalty": false,
    "repetition_penalty": 1.2,
    "penalty_window": 20,
    "enable_top_p_sampling": false,
    "top_p": 0.8,
    "enable_top_k_sampling": true,
    "top_k": 1
}
```

### The two gotchas

1. **`tokenizer_type` must be `"Qwen3VL"`, not `"Qwen2_5VL"`.** Both enum names are valid to `axllm`, but `Qwen2_5VL` fails at tokenizer-instantiation time (no matching C++ class in the binary — Qwen2.5 and Qwen3 share the same tokenizer vocab so `Qwen3VL` works fine). If you get this wrong you'll see an **empty response** from the server, and the log will contain `unsupport content type: 2` + `placeholder token count mismatch: placeholder=0 vision_tokens=256`.

2. **`vlm_type` IS `"Qwen2_5VL"`.** Different code path from the tokenizer — this one has a real implementation and drives the vision encoder + mRoPE position IDs.

The other numeric fields (`axmodel_num: 36`, `template_filename_axmodel`, `tokens_embed_size: 2048`, `vision_*` dimensions) all come from the shipped `run_qwen2_5_vl_image_axcl_aarch64.sh` script and the `qwen2_5-vl-tokenizer/preprocessor_config.json` file — if AXERA ever updates the model, re-derive them from those sources.

---

## Step 3 — Run interactively

```bash
axllm run ~/qwen2.5-vl-3b/Qwen2.5-VL-3B-Instruct/Qwen2.5-VL-3B-Instruct-AX650-chunk_prefill_512/
```

Expect ~80 seconds on first launch while it mmaps the 36 layer files onto the accelerator (watch `axcl-smi` in another terminal — CMM usage should climb from ~20 MiB to ~1.5 GiB).

When you see `prompt >>`, type your question:

```
prompt >> What is in this image?
image >> /home/dom/qwen2.5-vl-3b/Qwen2.5-VL-3B-Instruct/image/ssd_car.jpg
```

You should get a one-paragraph description at around **6 tokens/sec** with ~1.5 s time-to-first-token after the ~780 ms image encode.

Slash commands work the same as the text model: `/q`, `/exit`, `/reset`, `/dd`, `/pp`.

---

## Step 4 — Serve as an OpenAI-compatible HTTP API

```bash
axllm serve ~/qwen2.5-vl-3b/Qwen2.5-VL-3B-Instruct/Qwen2.5-VL-3B-Instruct-AX650-chunk_prefill_512/ --port 8000
```

Test it with standard OpenAI multimodal content (text + base64 image URL):

```python
import base64, json, urllib.request

img_b64 = base64.b64encode(open("/home/dom/qwen2.5-vl-3b/Qwen2.5-VL-3B-Instruct/image/ssd_car.jpg", "rb").read()).decode()

body = json.dumps({
    "model": "AXERA-TECH/Qwen2.5-VL-3B-Instruct",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image in one sentence."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        ],
    }],
    "max_tokens": 120,
}).encode()

req = urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    data=body,
    headers={"Content-Type": "application/json"},
)
print(urllib.request.urlopen(req, timeout=120).read().decode())
```

Or using the official OpenAI client:

```python
from openai import OpenAI
import base64

img = base64.b64encode(open("image/ssd_car.jpg", "rb").read()).decode()
client = OpenAI(api_key="not-needed", base_url="http://127.0.0.1:8000/v1")
reply = client.chat.completions.create(
    model="AXERA-TECH/Qwen2.5-VL-3B-Instruct",
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "What do you see?"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
    ]}],
)
print(reply.choices[0].message.content)
```

Both forms work. `axllm` also accepts `http://`, `https://`, and `file://` URLs for `image_url.url`; data URLs are the most portable. Intermediate decoded images are cached to `/tmp/axllm_images/`.

---

## Troubleshooting

**Empty response (`content: ""`), log shows `unsupport content type: 2` + `placeholder token count mismatch`.**
Your `tokenizer_type` is wrong. Must be `"Qwen3VL"`. See Step 2.

**Server fails to start with `create_tokenizer(Qwen2_5VL) failed`.**
Same root cause — you used `"Qwen2_5VL"` as `tokenizer_type`. Swap to `"Qwen3VL"`.

**`LLM init` hangs at ~10% then crashes.**
The `Qwen2.5-VL-3B-Instruct_vision_nchw448.axmodel` file (921 MB) is larger than the LFS download probably streamed in one go. Check `ls -la` on the weight dir — any file under ~1 KB that should be large means LFS didn't pull it. Re-run `git lfs pull` inside the repo.

**`Vision preprocess backend: SimpleCV (OpenCV not found at build time)` warning.**
Cosmetic. Vision preprocessing falls back to a built-in lightweight resize/normalize path. Output is slightly different from OpenCV's bilinear resize but accuracy is fine. If you want to eliminate the warning, rebuild `axllm` from source with OpenCV headers present (`sudo apt install libopencv-dev`).

**Out of accelerator memory after running a different model.**
`axcl-smi` will show CMM usage stuck high. Kill any stale `axllm` processes: `pkill -f axllm`, then re-launch.

---

## References

- Model repo: <https://huggingface.co/AXERA-TECH/Qwen2.5-VL-3B-Instruct>
- `axllm` source: <https://github.com/AXERA-TECH/ax-llm/tree/axllm>
- Vision-encoder architecture notes: <https://github.com/AXERA-TECH/ax-llm/blob/axllm/docs/vision_encoder_patterns.md>
- Baseline setup (driver + `axllm` install): [`../qwen3-0.6b/RUN_QWEN3.md`](../qwen3-0.6b/RUN_QWEN3.md)
