---
title: "Cutting MiniMind-O from 846ms to 320ms: a 25-round performance-optimization field report"
timestamp: 2026-08-31T19:34:00+08:00
tags:
  - AI
  - LLM
  - Performance Optimization
  - CUDA
  - PyTorch
series: nanovllm-omni Dev Notes
description: Under the constraints of monkey-patch only, zero new dependencies, and no touching upstream model code, one person used 25 rounds of automated experiments to cut the end-to-end latency of MiniMind-O on an RTX 3050 from 846ms to 320ms — including the biggest real bug, which lurked for 22 rounds.
toc: true
---
## Why nickel-and-dime these few hundred milliseconds on a 4GB card

While building `nanovllm-omni`, I ran MiniMind-O (a three-stage omni model: thinker → talker → mimi speech decode) on a local **RTX 3050 4GB Laptop** for end-to-end audio generation. A full `Omni.generate` pipeline looks like:

```
prompt → 16 tokens of token-by-token autoregression → mimi.decode → WAV encoding
```

Originally this chain sat around **928ms**, which feels noticeably laggy for a voice-demo. I set the project a goal: get the end-to-end wall-clock down to **≤ 500ms**.

But the real constraints weren't the goal — they were the boundaries. Three hard rules:

- No **new dependencies** (only built-in torch ops)
- No copying or modifying **upstream model code** (HF remote model, loaded via `AutoModelForCausalLM(trust_remote_code=True)`)
- Keep the **public API unchanged** (`Omni / AsyncOmni / SamplingParams / OmniRequestOutput`)

In other words: I could only **patch from the outside** (monkey-patch), never touch the screws inside. That constraint is exactly what turned this optimization from "editing code" into "stitching on the outside" — and ultimately forged a very interesting bug.

## Build a trustworthy ruler first: measure first

Before optimizing, I had to answer "how do I know a change actually helped". This part is one of the biggest things I took away from the whole experience.

On a small GPU, the random variation of a single bare `generate` measurement dwarfs the gains of any optimization. The benchmark scheme we finally settled on:

- 6 prompts × 5 runs × `max_tokens=16`
- Take the **median** (not skewed by occasional spikes)
- Run 20+ back-to-back measurements around every change to build a confidence interval

One deeply counterintuitive finding: **GPU temperature has a devastating effect on measurements**:

- Cold GPU (60°C, 1725 MHz): stdev only 7–11ms, very clean measurements
- Hot GPU (>80°C, 1590 MHz): absolute wall-clock **doubles to 1200ms**

The same change can differ by several times depending on cold vs hot state. This drilled in the habit of "check temperature/frequency before running bench" — otherwise you optimize into the void and think you hit a regression.

The lesson became a principle: **any change under 20ms is not credible inside a 7ms noise floor**. Later, E13, E22 and friends — "sounds plausible but the gain is inside the noise" — were all decisively discarded.

## The full 25-round experiment panorama

The whole optimization advanced in a loop of "propose hypothesis → run experiment → measure → decide keep/discard" (autoresearch, 25 rounds). The complete trajectory:

| # | Description | Status | Median (ms) | Δ vs prev |
|---|---|---|---:|---:|
| baseline | KV buffer + SDPA | — | **846.4** | — |
| E1 | Removed KV buffer, kept SDPA | keep | 815.4 | −31 (−3.7%) |
| E3 | Deleted `.clone()` to save a copy | discard | 1050 | +235 (broke correctness) |
| E4 | SDPA decode `is_causal=True` | keep | 803.1 | −12 (−5.1%) |
| E5 | **Fused RMSNorm** (`aten._fused_rms_norm`) | keep | **625.6** | **−178 (−26.1%)** |
| E6 | Fused QKV + gate-up projection | keep* | 467.8 | −158 (−44.7%) |
| E7 | Fused RoPE (eliminated cat) | keep | ~478 | marginal |
| E8 | view/split cleanup | keep | ~478 | — |
| E10 | skip `rp==1.0` repetition penalty | keep | ~408 | −70 |
| E11 | single contiguous `copy_` for model_input | keep | ~397 | −11 |
| E12 | `torch.compile(dynamic=True)` on RoPE | keep | ~386 | −11 |
| E13 | RMSNorm shape cache | keep (later reverted) | ~382 | −4 |
| E14 | `torch.compile(reduce-overhead)` RoPE | crash | — | CUDA Graph error |
| E15 | removed torch.compile on RoPE | discard | ~395 | +13 (confirms E12 works) |
| E16 | `torch.compile` whole MiniMindBlock | keep (off) | ~473 | default off |
| E17–E23 | stability / determinism / cross-prompt validation | keep | 379–422 | — |
| **E24** | **fixed the qkv_fusion dedupe bug** | **keep** | **320** | **−62 (−16%)** |
| E25 | retried KV buffer after E24 | discard | 452 | +132 (regression) |

> `*`: E6's nominal −44.7% was actually contributed by the RMSNorm / RoPE changes merged into the same baseline. **The real gain from QKV fusion only showed up at E24** — because until then it had just been broken.

From 846ms to 320ms, a total **−62.2% wall-clock**. And the largest single "real improvement" in that number isn't some clever operator fusion — it's a bug fix.

## How the 9 monkey-patches were stitched on

All optimizations landed in `nanovllm_omni/models/minimind_omni/attention.py` + `bundle.py` + `generation.py`, all done via **class/instance-level `forward = bound_method.__get__(self, cls)`**, without touching a byte of the remote model. The core ones:

| Patch | How it helps |
|---|---|
| Preallocated `text_buffer` / `audio_buffer` | removes the repeated `torch.cat` growth inside the generation loop |
| SDPA decode `is_causal=False` | at decode time Q has length 1 + a full KV past; `is_causal=True` makes the mask only attend to K[0], producing noise audio |
| Fused RMSNorm (`aten._fused_rms_norm`) | merges `pow+mean+rsqrt+mul×2+float+type_as` into a single kernel (E5, −178ms) |
| Fused QKV / gate-up projection | 3 separate matmuls → 1 `qkv_proj` (the star of E6/E24) |
| fixed the `qkv_fusion` dedupe bug | class-level `seen` dedupe meant 11/12 layers never actually fused; switching to instance-level gave −62ms (E24) |
| Fused RoPE (removed cat) | exploits the repetitive nature of `cos/sin` along dim to avoid `rotate_half`'s cat |
| skip `rp==1.0` repetition penalty | when `rp=1.0` (default), the whole unique+divide pass is a no-op |
| single contiguous `copy_` for model_input | 9 strided writes → 1 view+`copy_` |
| `torch.compile(dynamic=True)` on RoPE | Inductor fuses RoPE's ~6 elementwise kernels into ~2 |

Each of these micro-gains is small on its own (ms-scale), but stacked together they're a qualitative leap — and at zero cost: no new dependencies, no new model weights, just a few `forward`s bound to classes/instances.

## E24: the real culprit that hid for 22 rounds

This is the dramatic peak of the whole process, and the biggest engineering lesson.

`enable_fused_projections` was supposed to apply one QKV / gate-up fusion patch per Attention/MLP. The dedupe logic was written like this:

```python
seen_attn.add(cls)   # dedupe by “class”
seen_mlp.add(cls)
```

But `qkv_proj` / `gate_up_proj` are **instance attributes**, not class attributes. So, disastrously:

> **Only the 1st of the 12 Attentions (thinker.0) was ever patched; the other 11 layers kept running 3 separate matmuls.**

From E6 on, I assumed "Fused QKV" had been in effect for 22 rounds. Only when profiling revealed 11/12 layers were never fused did I fix the dedupe bug — and the **median dropped straight from 382ms to 320ms, −62ms, a 5.4× stdev improvement**.

This lesson I carved into memory:

> **class-level `seen` dedupe is a bug in per-instance patch scenarios. The moment you see `seen.add(...)`, ask first: is this patch applied on the class or on the instance?**

## Wall-hitting log: the edges of CUDA Graph

Not every path works. This section records a few "counterintuitive" failures:

- wrapping RoPE in `torch.compile(reduce-overhead)` → **crash**: conflicts with the subsequent `torch.cat` on KV cache ("accessing tensor output of CUDAGraphs that has been overwritten").
- `torch.compile` on the whole MiniMindBlock.forward → **+91ms regression**: Inductor's launch overhead × 16 layers ≈ 700ms swallows any gain.
- Full-stack CUDA Graph capture: theoretical gain 50–150ms, but requires staticizing the KV cache + monkey-patching the `start_pos` calculation of `model.model.forward` — too big a scope, painful to revert, and it affects every downstream user. **Not implemented** — the gain didn't cover the risk.

On small models + small batches, `torch.compile`'s fixed overhead often eats all the benefit; hand-written operator fusion is the reliable win.

## Final result and guardrails

The final state locked in at commit `11c3c13`:

```text
benchmark (6 prompt × 5 runs × max_tokens=16):
  median:    320 ms
  mean:      323 ms
  stdev:     11.6 ms
  min:       305 ms
  max:       343 ms
  <500ms:    20/20 (100%)
  
VRAM:        493 MB
```

And thoroughly validated:

- **Determinism**: 5× same (prompt, seed=42) → identical WAV MD5, **bit-exact**
- **Cross-prompt**: 12/12 heterogeneous prompts generate full 8 frames, no truncation
- **Public API**: `Omni / AsyncOmni / SamplingParams / OmniRequestOutput` all importable, 49 non-smoke tests pass
- **Wide-prompt**: median 422ms (bench prompts are on the short side; still well under the 500ms target on cross-domain prompts)

Project-side guardrails are all standing too — 49 non-smoke tests passing, bit-exact determinism, 12/12 cross-prompt validation.

## Retrospective: three things this experience taught me

1. **Build a trustworthy ruler before talking about optimization.** GPU temperature can nearly double measurement drift (1200ms vs 846ms); without knowing your noise floor, any sub-20ms "optimization" may be self-comfort. Median + multiple measurements + temperature monitoring is a habit forged by several fake regressions.

2. **The biggest gains often aren't a new trick — they're actually making an existing trick work.** A `seen.add(cls)` vs `seen.add(self)` difference equals 22 rounds of experiments, 11 layers, −62ms. Don't rush to learn new moves; first check whether the old move actually applies across your whole object set.

3. **Constraints forced, more robust.** "Monkey-patch only + built-in torch ops only" pushed me to fuse by operator semantics instead of stuffing CUDA kernels or copying vendor models into the project. The final optimization stack has zero dependencies, zero vendor code, bit-level side-effect free.

The full optimization stack and repro commands live in nanovllm-omni's `docs/perf/minimind-omni-under-500ms.md`; run `python -m nanovllm_omni.optim.bench time` locally to reproduce.

## Follow-up: when the talker actually got wired in

This report was written in August 2026, when the "end-to-end" it dissected actually went only as far as the **thinker stage**: `prompt → 16-token autoregression → mimi.decode`. In September I upgraded the talker/code2wav from their identity placeholders to a real **three-stage** pipeline (thinker → talker → MTP → code2wav) — the nanovllm-omni take on vllm-omni PR #3796. Looking back, one number needs a correction:

- **The 320ms in the old post only reached the thinker stage.** It is thinker-stage decode only (the default `bench time` path). Once all three stages landed, I added `bench time --pipeline full` to measure real end-to-end.
- **The collapsed path is retired.** The single-thinker shortcut was removed (`8f6d522`); `pipeline_kind` defaults to `full` and all three stages really run.
- **Actual full-E2E on RTX 3050 4GB (real weights, torch 2.14):**

```text
6 prompts × 20 runs × max_tokens=16 (--pipeline full)
  medium_01 -> 747 ms (p95 844)
  medium_02 -> 767 ms (p95 850)
  short_01  -> 645 ms (p95 709)
  short_02  -> 628 ms (p95 661)
  short_03  -> 634 ms (p95 669)
  system_01 -> 1038 ms (p95 1109)
  VRAM peak: ~1881 MiB
```

So the honest headline is: **real end-to-end MiniMind-O on RTX 3050 is 0.63–1.04 s** (median). 320ms is just one stage — it used to be quoted as "end-to-end," which was a reporting error; README and blog now use the full-E2E figure.

The time in the three-stage path is dominated by the talker and MTP (the delayed codebooks from multi-token prediction); the thinker CUDA-Graph fast path is still the fastest single stage, but it is not the whole pipeline. On 4GB VRAM, full peaks around 1881 MiB — tight but within budget.

New repro:

```bash
python -m nanovllm_omni.optim.bench time --pipeline full --max-tokens 16 --runs 20 --warmup 1
```

Raw CSV: `docs/perf/full-e2e-rtx3050.csv`; validation detail: `docs/perf/tk005-rtx3050.md`.

(The previous paragraph ended with "next: EOS and audio delay schedule" — and that EOS work did happen in TICKET-05: a post-EOS padding state machine + talker watchdog, see `5da15f7`. The wall-hitting log's "full-stack CUDA Graph: not implemented" call is also worth revisiting — the talker MTP now has its own CUDA-Graph path, `ac49cf9`.)

Step-by-step commits live in nanovllm-omni's `git log perf/cuda-graph-default-on`, from `972224c` (talker skeleton) through `e91ca41` (--pipeline full bench).

The price of getting full working is memory: 1881 MiB peak already edges against the ceiling on a 4GB card. Making the talker capture a full CUDA-Graph span, or quantizing MTP's delayed codebooks, will have to free up VRAM first.

> **Update (2026-09)**: The thinker CUDA Graph now captures `n_steps - 1` graphs (not `n_steps`): prefill produces the first token eagerly, saving one wasted warmup + capture + VRAM. Graph decode also executes the EOS + audio-stop state machine, matching eager frame counts (9/10/15 instead of budget-forced 16). Re-measured on WSL RTX 3050 (cold-hot-v3, `docs/perf/aligned/cold-hot-v3.json`): **thinker decode hot p50 ≈ 196 ms** vs eager median 1248 ms (~**6.36×**); capture cost amortises after a single request. Parity check (`tools/bench_graph_eager_parity.py`): graph and eager produce identical token counts and identical non-pad audio rows across three different prompts. See `tools/bench_cold_hot_cuda_graph.py` for the cold/hot split benchmark that separates capture cost from replay throughput.

That's it for now.