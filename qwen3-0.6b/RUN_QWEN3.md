# Running Qwen3-0.6B on the M5Stack LLM-8850

A step-by-step guide to running the Qwen3-0.6B language model on a Raspberry Pi 5 with an M5Stack LLM-8850 (AX650N) M.2 accelerator card.

By the end of this guide you will have an interactive chat prompt backed by a small LLM running entirely on-device — no cloud, no internet.

---

## What you need

**Hardware**
- Raspberry Pi 5 (Model B recommended)
- M5Stack LLM-8850 M.2 card, seated on either:
  - The official **Raspberry Pi M.2 HAT+** (needs a 5 V @ 3 A non-PD supply), or
  - The M5Stack **LLM-8850 PiHat** (needs a PD supply capable of 9 V @ 3 A / 27 W)
- A reasonable SD card or NVMe boot drive (the model is ~1.2 GB and the build pulls down more)

**Software (already on a fresh Raspberry Pi OS Bookworm install unless noted)**
- `git`, `git-lfs`, `python3`, `wget`, `make`, `gcc` — all stock
- `cmake` — needs to be installed (step 3)
- The `axclhost` driver from M5Stack's apt repo (step 2)

**Takes about 30 minutes**, most of which is downloading the model and compiling `axllm`.

---

## Step 1 — Confirm the card is seen by the Pi

With the card seated and the Pi powered on, open a terminal and run:

```bash
lspci | grep -i axera
```

You should see something like:

```
0001:01:00.0 Multimedia video controller: Axera Semiconductor Co., Ltd Device 0650 (rev 01)
```

**If nothing prints:**
- Re-seat the M.2 card and check the HAT+ ribbon cable is the right way up.
- Make sure your EEPROM bootloader is current. `sudo rpi-eeprom-update` — if the date is before December 2023, run `sudo raspi-config` → *Advanced Options* → *Bootloader Version* → *Latest*, then `sudo rpi-eeprom-update -a && sudo reboot`.
- Check power. A weak USB-C supply is the most common cause of a missing PCIe device.

---

## Step 2 — Install the AXCL host driver

The AX650N chip on the card is a full SoC; the Pi talks to it through a driver + userspace library called **AXCL**. M5Stack publishes these as an apt package.

Add the repo:

```bash
sudo wget -qO /etc/apt/keyrings/StackFlow.gpg \
  https://repo.llm.m5stack.com/m5stack-apt-repo/key/StackFlow.gpg

echo 'deb [signed-by=/etc/apt/keyrings/StackFlow.gpg] https://repo.llm.m5stack.com/m5stack-apt-repo axclhost main' \
  | sudo tee /etc/apt/sources.list.d/axclhost.list
```

Install the driver:

```bash
sudo apt update
sudo apt install dkms axclhost
```

Open a fresh shell (so the new env vars load), then verify the card responds:

```bash
axcl-smi
```

You should see a table with the card name (**AX650N**), firmware version, PCI bus ID, memory usage, and temperature. If this works, the host-side stack is complete.

**If `axcl-smi` hangs or errors:** reboot once. DKMS needs a reboot to load its kernel module on some systems.

---

## Step 3 — Install build tools

We're going to compile the inference runtime from source. `cmake` isn't in the default image:

```bash
sudo apt install -y cmake
```

Everything else the build needs (`make`, `gcc`, `git`, `wget`, `unzip`) is already present on Raspberry Pi OS.

---

## Step 4 — Install `axllm`

`axllm` is AXERA's unified CLI for running LLMs on their chips. One command installs it:

```bash
curl -fsSL https://raw.githubusercontent.com/AXERA-TECH/ax-llm/axllm/install.sh | bash
```

What the installer does:
1. Clones the `axllm` branch of [AXERA-TECH/ax-llm](https://github.com/AXERA-TECH/ax-llm)
2. Detects your setup — sees `axcl-smi` and `/usr/lib/axcl`, picks the **AXCL backend**
3. Builds the binary with cmake/make (takes a few minutes on a Pi 5)
4. Installs it to `/usr/bin/axllm` (uses `sudo`, expect a password prompt)

Verify:

```bash
axllm --help
```

You should see `Usage: axllm run <model_path>` etc.

> **Why not follow the M5Stack docs page?** The [official LLM-8850 Qwen3-0.6B page](https://docs.m5stack.com/en/guide/ai_accelerator/llm-8850/m5_llm_8850_qwen3_0.6b) describes an older workflow that uses a `main_axcl_aarch64` binary and a separate Python tokenizer service on port 12345. Those files are no longer in AXERA's Hugging Face repo — the current README points at `axllm` instead. The `axllm` path is simpler and is what we use here.

---

## Step 5 — Download the Qwen3-0.6B model

The model weights — pre-quantized to w8a16 for the AX650N NPU — are on Hugging Face:

```bash
mkdir -p ~/qwen3-0.6b
cd ~/qwen3-0.6b

git lfs install
git clone https://huggingface.co/AXERA-TECH/Qwen3-0.6B
```

Expect roughly **1.2 GB** of download. When it's done you should see a directory of `.axmodel` files (one per transformer layer), an embedding weights file, and a tokenizer text file:

```bash
ls Qwen3-0.6B/
```

You'll see `config.json`, `post_config.json`, `qwen3_tokenizer.txt`, `qwen3_post.axmodel`, `model.embed_tokens.weight.bfloat16.bin`, and 28 `qwen3_p128_l*_together.axmodel` files.

---

## Step 6 — Run the model

Interactive chat:

```bash
axllm run ~/qwen3-0.6b/Qwen3-0.6B/
```

The first time, it will load 31 model files onto the accelerator's memory (~2 seconds) and then drop you at a `prompt >>` line. Try:

```
prompt >> who are you
```

Expect a response at around **12 tokens/sec** with ~250 ms time-to-first-token. The model will think briefly inside `<think>...</think>` tags (Qwen3's reasoning mode) before giving its answer.

Interactive commands while chatting:

| Command | What it does                       |
|---------|-------------------------------------|
| `/q`    | Quit                                |
| `/exit` | Quit                                |
| `/reset`| Clear the conversation / KV cache   |
| `/dd`   | Delete the last turn                |
| `/pp`   | Print the conversation so far       |

---

## Bonus — Serve an OpenAI-compatible API

If you want to call the model from other programs (Python scripts, local web apps, etc.), run it as a server instead:

```bash
axllm serve ~/qwen3-0.6b/Qwen3-0.6B/
```

It listens on `http://0.0.0.0:8000` and speaks the OpenAI chat-completions protocol, so any OpenAI client library works. Example:

```python
from openai import OpenAI

client = OpenAI(api_key="not-needed", base_url="http://127.0.0.1:8000/v1")
reply = client.chat.completions.create(
    model="AXERA-TECH/Qwen3-0.6B",
    messages=[{"role": "user", "content": "Say hi in one word."}],
)
print(reply.choices[0].message.content)
```

---

## Troubleshooting

**`axcl-smi` shows the card but `axllm run` fails with "device not found".**
Open a new terminal. The axclhost package sets env vars in `/etc/profile` that only load on login.

**Model loads but generation is extremely slow (<1 tok/s).**
Check `axcl-smi` in another terminal while running. The NPU column should jump to 50–100% during generation. If it stays at 0%, the runtime isn't actually hitting the card — rebuild `axllm` to make sure it picked the AXCL backend (look for `ax_model_runner_axcl` in the build output).

**"Out of memory" or CMM errors on load.**
The AX650N has 8 GB of its own memory. Qwen3-0.6B uses ~1.3 GB, so you should have plenty — but if you previously ran a larger model (`axllm serve` as a background process, or Qwen3-1.7B) it may still hold memory. Kill any stale axllm processes and try again.

**The build of `axllm` fails partway through.**
Make sure `cmake` is installed (step 3) and that `/usr/include/axcl` and `/usr/lib/axcl` both exist (they come from the `axclhost` package). If either is missing, reinstall: `sudo apt install --reinstall axclhost`.

---

## What to try next

- Swap in **Qwen3-1.7B** (same pattern — just `git clone https://huggingface.co/AXERA-TECH/Qwen3-1.7B` and point `axllm run` at it).
- Explore other AXERA models: [huggingface.co/AXERA-TECH](https://huggingface.co/AXERA-TECH) — vision-language models (Qwen2.5-VL, Qwen3-VL) also run here.
- Wire the OpenAI-compatible server into your favourite tool (`aider`, `open-webui`, an Obsidian plugin, a voice assistant, etc.).

---

## References

- M5Stack LLM-8850 docs: <https://docs.m5stack.com/en/guide/ai_accelerator/llm-8850/llm-8850>
- Driver install page: <https://docs.m5stack.com/en/guide/ai_accelerator/llm-8850/m5_llm_8850_software_install>
- `axllm` source: <https://github.com/AXERA-TECH/ax-llm>
- Model repo: <https://huggingface.co/AXERA-TECH/Qwen3-0.6B>
