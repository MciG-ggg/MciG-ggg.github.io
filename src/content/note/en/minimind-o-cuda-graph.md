---
title: "Pushing MiniMind-O another 7.5×: CUDA Graph practice on the thinker decode stage + a cross-family feasibility verdict"
timestamp: 2026-09-03T22:10:00+08:00
tags:
  - AI
  - LLM
  - Performance Optimization
  - CUDA Graph
  - PyTorch
series: nanovllm-omni Dev Notes
description: After the last post cut MiniMind-O from 846ms to 320ms, the remaining ~510ms turned out to be pure launch overhead — no kernel work. CUDA Graph was the only lever left. This post records how all 5 prerequisites fell into place one by one, the final 7.51× thinker-decode speedup (not full-pipeline E2E — see the correction note at the end), and why the other three model families either weren't worth it or didn't need the same treatment.
toc: true
---
## Where the last post left off

The previous post, [《Cutting MiniMind-O from 846ms to 320ms》](/en/note/minimind-o-from-846ms-to-320ms), ended after 25 rounds of automated experiments got end-to-end generation from 846ms down to 320ms — at which point I had no cards left to play. The last round, wrapping RoPE in `torch.compile(reduce-overhead)`, crashed outright, and block-level compile cost me an extra 91ms. My conclusion back then:

> "Optimizing kernels has saturated at this point; the remaining few hundred milliseconds are probably launch overhead, but chasing that thread to the end needs ncu evidence."

And so came the next session: **take the evidence with Nsight Compute**, and let it decide the direction.

## The truth I hit after 320ms

The answer ncu + Kineto gave me was surprisingly clean.

Kineto produced the same picture on both the RTX 3050 (local target hardware) and a Colab T4 (cross-check):

- a single `Omni.generate` triggers **~17,000 `cudaLaunchKernel`** calls and **~344 `cudaStreamSynchronize`** calls;
- cutlass GEMMs cumulatively account for only **~28ms**;
- in other words, **~510ms is host queueing + GPU idling** — the kernels' actual "working time" is only about 6% of the total wall time.

ncu broke down the top-4 kernels: all below 100% SOL, high hit rates — but every one runs extremely briefly. This is the classic **launch-bound** shape: **did the kernels finish their work? Yes. But they're waiting for the host to queue up the next launch.**

At this point the question flipped: **it's not "which kernel is slow" — it's "how do I pack 17,000 launches into a handful."** CUDA Graph is the standard answer for this shape.

But —

## First capture attempt: failed thoroughly

I started with PyTorch's minimal documentation example: wrap `model(input)` in `torch.cuda.graph(g)`, replacing the `input`'s `copy_` with the "real input at replay time". It failed immediately:

```
RuntimeError: CUDA Graph does not support capture on the default stream.
```

Fixed the stream issue, and hit:

```
RuntimeError: During capture: encountered an autograd node.
```

Got around autograd, and hit:

```
torch.cuda.CudaError: operation not allowed when stream is capturing
```

Behind each error sits a hard constraint of capture. I catalogued the four concrete blockers these errors correspond to — **they are the entire "why not" checklist of CUDA Graph engineering**:

1. **Two host reads in `forward()`**: `if self.thinker.freqs_cos[0, 0] == 0:` and `if self.talker.freqs_cos[0, 0] == 0:`. These branch values live on the host, and CUDA Graph captures a **single execution path** — a host read re-read each time makes the capture path's choice non-deterministic.
2. **`torch.cat` writes into the KV cache every step**: each step of the decode loop, `torch.cat([past_kv, k], dim=-2])` allocates a new tensor. At capture time CUDA Graph burns in "where to write"; at replay that address may already be reused — silent misalignment.
3. **Shape drift every step**: in decode, Q's length is always 1, but K/V past length grows +1 per step. If you try to write the entire decode loop into one graph, the shape isn't fixed; fold it into "one graph per step" and you need to capture `n_steps` graphs.
4. **Re-capture side effects**: if you re-capture every run, the RNG state during capture (even just dropout or the like) gets baked into every run's results, breaking determinism.

These four are the **meta-obstacles of all CUDA Graph engineering** — any model that trips on one of them will have naive capture fail or misbehave. I reorganized them into a 5-condition decision method.

## The 5 prerequisites: why CUDA Graph isn't plug-and-play

For CUDA Graph to pay off on a model, **all 5** conditions must hold simultaneously. If any one fails, don't do it — or go through the upstream library's own optimization channel (vLLM, diffusers, lerobot all have their own serve-loop optimizations).

### Condition 1: launch-bound per step

You can't use "I think it's launch-bound" — you need hard ncu + Kineto evidence. The criterion: Kineto reports "tens of thousands of launches + hundreds of syncs + cumulative GEMM time << total wall time". **Without that evidence, don't reach for CUDA Graph first** — the direction may be wrong.

### Condition 2: shape fixed per step

In the decode loop, every step has the same Q length and same K/V past length. If the shape drifts on every invocation (new noise for diffusion, different resolutions for image generation), the captured graph can't line up with the replay input.

### Condition 3: memory addresses stable

Any per-step `cat` / `stack` / "allocate a new tensor" op invalidates the address baked in at capture time. **Fix: preallocate buffers + a `_kv_pos` cursor** — allocate the buffer once, and each write only moves the cursor, never reallocates.

### Condition 4: control flow branches in the captured region stay stable

Host-side data-driven ifs make the capture path non-deterministic. **Fix: `_patched_forward` rewrites known-dead branches into `if False:` at compile time via `exec(compile(...))`** — provided you've verified those branches really are dead (a warmup run confirms it).

### Condition 5: re-capture cost amortizes

Capture has RNG side effects, so "capture once / replay many" beats capturing every run. Re-capture only fires for cases that can't be amortized (e.g. prompt-length changes), and must be verified to **not contaminate the same-prompt determinism**.

Apply these 5 to MiniMind-O: **all 5 pass**.

## MiniMind-O: 5/5 = 7.51×

What followed was the process of knocking out all 4 obstacles one by one. This section is the session's 50 iterations — each one corresponding to a `tools/bench_*.py` probe.

**Knocking out obstacle 1 (host reads)**: in `nanovllm_omni/optim/cuda_graph.py:48-58` I wrote `_patched_forward` that runs `exec(compile(...))` over the `forward()` source, replacing the two `if self.*.freqs_cos[0, 0] == 0:` with `if False:`. **The original class is not rebound** — a copy is made and invoked. A warmup run confirmed those two branches were never taken, so the compile-time rewrite is safe. Call this "neutralizing host-reads".

**Knocking out obstacle 2 (KV cat)**: in `nanovllm_omni/models/minimind_omni/attention.py` §5 I added `enable_fixed_kv_buffer`: preallocate a `[max_len, n_heads, head_dim]` KV buffer, write the current K/V to the `_kv_pos` slot each step, cursor += 1. A CPU bit-exact test (`tests/test_fixed_kv_buffer_forward.py`) proves this path is mathematically equivalent to the cat path, and it's further verified bit-exact on GPU.

**Knocking out obstacle 3 (shape drift)**: instead of trying to write the whole decode loop into one graph, it's **one graph per decode step** — capture `n_steps - 1` sub-graphs (prefill produces the first token eagerly, then each of the remaining `n_steps - 1` decode steps gets its own captured graph with input shape `[1, 9, 1]`). The shape is fixed inside each sub-graph. Each graph is **position-dependent**: `_kv_pos` is a Python int used for tensor slicing at capture time, so the KV write slot and read range are baked into each graph. Making graphs position-independent would require converting `_kv_pos` to a device-side tensor — a significant attention rewrite for marginal VRAM savings; the per-position approach is standard practice (vLLM, TGI).

**Knocking out obstacle 4 (re-capture side effects)**: use capture-once / replay-many — **capture only on the first call**, pure replay afterward. Each of the `n_steps - 1` graphs gets one eager warmup forward (to settle cuBLAS workspace) + one `torch.cuda.graph(g)` capture on a side stream. The randomness during capture only affects the first call; all subsequent replays are deterministic.

**An extra defect discovered (defect #5)**: there was a problem I didn't foresee with this capture scheme — the per-step graph's KV offset is baked to the **`prefill_len` of the first prompt**. If the next prompt has a different length, reusing the old graph misaligns the KV. Fix: `_needs_recapture()` discards the old graph + zeroes the buffer + re-runs prefill + rebuilds graphs whenever `_prefill_len` changes. Same-prompt reuses stay untouched (preserving repeat determinism). This fix was verified bit-exact on the RTX 3050 across 3 length cycles + 2 round-trips using `tools/bench_defect5_3cycle.py`.

Final numbers (RTX 3050 4GB, torch 2.14, `HF_HUB_OFFLINE=1`, `max_tokens=16`, `PYTHONPATH=.`, seed 42; `docs/perf/aligned/cold-hot-v3.json` + `parity-v2.json`):

- **Single CUDA Graph step** (prior report): eager 36.11 ms/step → graphed 3.49 ms/step, **10.36× per-step speedup**.
- **16-step thinker decode cold/hot split** (`tools/bench_cold_hot_cuda_graph.py`): eager median 1248 ms → graph **hot p50 196 ms / p95 218 ms** (30 requests reusing the same graph), **~6.36× hot speedup**; cold path (incl. capture) 1153 ms, capture cost ≈ 956 ms, **amortises after 1 request**. VRAM peak 490 MB.
- **Graph × Eager token/audio parity** (`tools/bench_graph_eager_parity.py`, 3 prompts × 16 tokens):
  - `Hi` (plen=2): eager 2466 ms → graph 1249 ms (**1.97×**, first-run GPU noise)
  - `Hello there` (plen=2): eager 918 ms → graph 178 ms (**5.14×**)
  - `Tell me a short story` (plen=6): eager 906 ms → graph 1104 ms (**0.82×**, new prompt length triggers recapture, single request)
  - Token counts: eager / graph all 16; audio non-pad rows: 8 / 8 — identical.
  - Average **2.65×** (incl. first-run GPU noise).
- **Public `Omni.generate` API e2e**: 3 different-length prompts in a hetero-prompt cycle + same-prompt repeats — **all produce byte-identical WAV** (matching MD5s).

7.51× isn't the peak theoretical number (the per-step 10.36× is), but the thinker-stage measurement subtracts the host-side sampling, buffer resets, and Mimi codec decode that sit outside the graph. **For the thinker decode path starting at 320ms, zero new dependencies, pure monkey-patch + CUDA Graph capture, this leverage ratio beats all 25 rounds of monkey-patching combined.**

## Cross-family scan: why only one got done

The `nanovllm_omni` repo registers 4 model families:

| Family | Load path |
|----|----------|
| `minimind_omni` | own safetensors + 5 monkey-patches + CUDA Graph |
| `smolvlm` | `transformers.AutoModelForImageTextToText` + `AutoProcessor` |
| `sd_turbo` | `diffusers.StableDiffusionPipeline.from_pretrained` |
| `smolvla` | `lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy` |

After running the 5 conditions over them:

**`smolvlm`**: conditions 1 (launch-bound), 2 (fixed shape), 3–5 all pass — but **conditions 3–5 are being satisfied by transformers' own engine** — we're going through `model.generate(...)`, and its KV cache / capture / replay would collide with our project-level capture. Doing it ourselves would mean abandoning transformers' `generate` and hand-implementing KV cache management + capture hooks — a bigger engineering job than the MiniMind-O run, and vLLM walks the same path. So **we don't do it ourselves**; let transformers / vLLM's own CUDA Graph do the work.

**`sd_turbo`**: **condition 1 fails outright**. Default `num_inference_steps=1` (locked by `deploy/sd_turbo.yaml`) — the whole generation is a single forward pass; the concept of "launch-bound per step" doesn't exist. And each call swaps latent + prompt, so shape drifts — condition 2 fails too. The UNet body is GEMM; the bottleneck is the kernel workload itself, not launches. **CUDA Graph isn't a lever for it** — if you really want to optimize, the direction is `torch.compile` on the UNet / quantization / attention slicing, all on the diffusers side.

**`smolvla`**: LeRobot's `SmolVLAPolicy` has a 50-step chunk queue inside, and `predict_action_chunk` uses its own VLA inference. Condition 5 passes (the chunk queue amortizes capture cost), but **it's not currently wired into the engine** — `nanovllm_omni/models/__init__.py` only exports the minimind_omni trio; the other three families register via the `_load_builtin_pipelines()` side-effect. To do CUDA Graph here, you'd first need to replicate the `enable_fixed_kv_buffer` infrastructure into the lerobot policy — an effort equal to redoing MiniMind-O. **A candidate for later** — not this round's problem.

## Two orthogonal strategies: going deep into forward vs a thin shell

This outcome is a deliberate engineering choice, not a "finished one, might as well do the other".

The docstring at the top of `nanovllm_omni/models/__init__.py` is blunt:

> Each loader is a thin function that returns a handle the rest of the engine can call. We do not reimplement model forward passes here — we delegate to vllm (for AR), diffusers (for image diffusion), and direct safetensors loaders (for MiniMind-Omni, which is small enough to bring up ourselves).

**MiniMind-O is a small self-trained model (< 4GB)**, and the project layer decided to "bring up ourselves" — so it can go deep into forward, monkey-patch 5 optimization points, add buffer-ization, add CUDA Graph capture. The pipeline is 3 stages (thinker LLM_AR + talker LLM_AR + code2wav CODEC).

**SmolVLM / SD-Turbo / SmolVLA are all trained by HuggingFace / Stability / LeRobot**, and the project layer only does a "thin shell" — single-stage closures, where the stage factory calls the upstream library directly. `models/__init__.py` only exports the minimind_omni trio; the other three families go through side-effect registration and users don't import them directly.

Why is this symmetric? Because **the upstream libraries have already done their own serve-loop optimization**. transformers' KV cache + capture, diffusers' UNet optimization paths, lerobot's chunk queue — adding CUDA Graph at our layer on any of them would collide with their graph state (typical symptom: upstream attention reallocates its buffer during capture → our baked-in address goes stale). **Going deep into forward and thin-shell wrapping are two orthogonal strategies** — the former fits "small self-trained models", the latter fits "upstream library + we only chain stages".

If you ever want to do CUDA Graph for SmolVLA too, the first thing to do isn't capture — it's replicating the `enable_fixed_kv_buffer` infrastructure into the lerobot policy. That's another session.

## Ecosystem positioning: patch-only is the minority

A horizontal comparison against the mainstream ML inference projects of 2026-09 ([survey report](https://mcig-ggg.github.io)):

- **sglang**'s `hf_transformers_patches.py` also centralizes patches, but heavily uses subclassing + AutoModel wrapping alongside;
- **vllm-omni**'s `vllm_omni/patch.py` (27.9 KB / 10 sites) keeps patch files too, but its model layer goes subclass + sub-module fork-rewrite (`voxcpm2_talker.py` 139 KB, `siglip2.py` 15 KB, etc.), and it still keeps 1 `nemo_vendored/`;
- **vLLM core / HF optimum / accelerate / TGI / DeepSpeed-MII** all go subclass + adapter — that's the de facto standard;
- **TensorRT-LLM / llama.cpp / MLC-LLM** (performance-sensitive) go fork-and-rewrite into their own runtimes.

nanovllm-omni's real differentiation **isn't "we're the odd one out"** — sglang is similar too — but **vendor = 0 + fork = 0 + subclass = 0**: the 5 monkey-patches in this post and the last are all `setattr`, without going through vLLM's layer of indirection (most of vllm-omni's patches are written for vLLM — cross-cutting version-compat wrappers).

Ollama just de-forked its own GGML in v0.30 and directly depends on llama.cpp — it took them years to learn the "long-term cost of maintaining a fork" lesson; our project going patch-only from day one is standing on their shoulders. The full comparison table lives in `docs/perf/ncu-generate-kernels-2026-09-01.md` §51.3 in the nanovllm-omni repo.

## Wrap-up

This session delivered 7.51× on the thinker decode stage + all 5 GPU gates green (serve-path / cross-prompt / longrun / determinism / Omni() e2e). But the more valuable byproduct is **those 5 conditions** — it turned "can CUDA Graph be used here" from a luck-based voodoo question into an engineering check you can tick off 5 lines at a time.

> **Correction note (2026-09)**: The 7.51× (320→42 ms) figure measures the **thinker decode loop only** — not the full MiniMind-O E2E pipeline. The complete pipeline includes thinker + talker + code2wav; real E2E with thinker graph engaged is ~640–850 ms depending on prompt length, with the talker and Mimi codec stages dominating. The code now captures `n_steps - 1` graphs (not `n_steps`): prefill produces the first token eagerly, then `n_steps - 1` decode steps each replay one captured graph. This saves one wasted warmup + capture + VRAM. Graphs remain position-dependent by design. Re-measured on WSL RTX 3050 after the off-by-one fix and KV-zero-on-prefill fix: thinker decode **hot p50 ≈ 196 ms** vs eager median 1248 ms (~**6.36×**); capture cost amortises after a single request. See `tools/bench_cold_hot_cuda_graph.py` for the cold/hot split benchmark and `tools/bench_graph_eager_parity.py` for the token/audio parity check.

If your model also wants CUDA Graph optimization, the suggested flow:

1. **Take evidence with ncu + Kineto first** — don't start without hard "launch-bound" proof;
2. **Tick off all 5 conditions** — if any fails, don't do it, or let the upstream library optimize itself;
3. **All 5 pass → enter the implementation phase**, following the "knock out the 4 obstacles" order: KV buffer-ization first (condition 3) → neutralize host-reads (condition 4) → capture-once/replay-many (condition 5) → finally handle the re-capture edge case (defect #5 and the like).

The full evidence chain is in `docs/perf/ncu-generate-kernels-2026-09-01.md` (74KB, 51 sections) in the nanovllm-omni repo; §51 is the cross-family feasibility scan material written for this post. For per-step GPU probes: all 24 `tools/bench_*.py` files, each corresponding to one decision point.

The next post plans to cover SmolVLA integration — the same 5 checks ticked off and 4 obstacles knocked out, but the scenario shifts from AR text+audio to VLA vision+action. We'll see then.