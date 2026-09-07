---
title: 把 MiniMind-O 再压 7.5 倍：CUDA Graph thinker decode 实践 + 跨族可行性判定
timestamp: 2026-09-06T21:30:00+08:00
tags:
  - AI
  - LLM
  - 性能优化
  - CUDA Graph
  - PyTorch
series: nanovllm-omni 开发手记
description: 上一次把 MiniMind-O 从 846ms 压到 320ms 后，剩下的 ~510ms 全是 launch overhead、没有 kernel 工作。CUDA Graph 是唯一杠杆。这篇记录 5 个前置条件怎么一一就位、最终拿到 7.51× thinker decode 加速（不是全流水线 E2E——见文末勘误），以及为什么另外三个模型族不值得或不需要做同样的事。
toc: true
---
## 上一篇写到哪了

上一篇[《把 MiniMind-O 从 846ms 压到 320ms》](/note/MiniMind-O%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96%E5%AE%9E%E5%BD%95)写到，25 轮自动实验把端到端生成从 846ms 压到 320ms 之后，我已经没什么牌可以出了。最后一轮试 `torch.compile(reduce-overhead)` 直接 crash，block 级 compile 还倒贴了 91ms。我当时的结论是：

> "kernel 优化到这一步已经饱和，剩下的几百毫秒可能是 launch overhead，但要把这条线追到底需要 ncu 证据。"

于是就有了下一个 session：**用 Nsight Compute 取证**，决定接下来的方向。

## 320ms 之后撞上的真相

ncu + Kineto 给我的答案出乎意料地干净。

Kineto 在 RTX 3050（本地目标硬件）和 Colab T4（cross-check）上同时跑出来同一张图：

- 单次 `Omni.generate` 触发 **~17 000 次 `cudaLaunchKernel`**、**~344 次 `cudaStreamSynchronize`**；
- cutlass GEMM 累计只占 **~28ms**；
- 也就是说，**~510ms 全是 host 排队 + GPU 空等**——kernel 干活的"实际工作时间"只占总墙时的 6% 左右。

ncu 把 top 4 kernel 拆开看，全部 < 100% SOL，命中率很高，但每个都跑得非常短。这是个典型的 **launch-bound** 形态：**kernel 工作没做完吗？做完了。但它们在等 host 把下一个 launch 排上来。**

这一刻问题翻转了：**不是"哪个 kernel 慢"，是"怎么把 17 000 次 launch 打包成少数几次"。** CUDA Graph 是这套形态的标准答案。

但是——

## 第一次 capture 尝试：失败得很彻底

我先按 PyTorch 文档的最小例子来了一遍：`torch.cuda.graph(g)` 包住 `model(input)`，把 `input` 的 `copy_` 出来换成"replay 时的真实输入"。结果一上来就报：

```
RuntimeError: CUDA Graph does not support capture on the default stream.
```

修了一下 stream 问题，又来：

```
RuntimeError: During capture: encountered an autograd node.
```

再绕开 autograd，又来：

```
torch.cuda.CudaError: operation not allowed when stream is capturing
```

每一道错的背后都对应一个 capture 的硬约束。我把这四道错对应的实际障碍列出来，**它们就是 CUDA Graph 工程的全部"为什么不行"清单**：

1. **`forward()` 里有两个 host 读**：`if self.thinker.freqs_cos[0, 0] == 0:` 和 `if self.talker.freqs_cos[0, 0] == 0:`。这两个判断的值在 host 端，CUDA Graph 捕获的是**单条执行路径**——host 读每次重读就会让 capture 路径选择不确定。
2. **每步 `torch.cat` 写入 KV cache**：decode loop 里每一步 `torch.cat([past_kv, k], dim=-2])` 都会分配一个新 tensor。CUDA Graph 捕获时把"写哪"这个地址烧死，replay 时该地址可能已经被复用——静默错位。
3. **每步 shape 漂移**：decode 步 Q 长度恒为 1，但 K/V past 长度每步 +1。如果把整个 decode loop 写进一张图，shape 不固定；折成"每步一张图"又需要捕获 `n_steps` 张。
4. **Re-capture 的副作用**：如果每 run 重新捕获，capture 期间的 RNG 状态（哪怕是 dropout 之类的）会被折进每次运行的结果里，破坏确定性。

这四条是**所有 CUDA Graph 工程的元障碍**——任何模型撞上其中一条，naive capture 都会失败或行为错乱。我把它们重新组织了一下，得到一个 5 条件的判定法。

## 5 个前置条件：为什么 CUDA Graph 不是开箱即用

CUDA Graph 在某个模型上划算，**必须**同时满足 5 个条件。任一不满足就别做，或者走上游库自己的优化通道（vLLM、diffusers、lerobot 都有它们自己的 serve-loop 优化）。

### 条件 1：每步 launch-bound

不能用"我觉得是 launch-bound"——必须有 ncu + Kineto 的硬证据。判定标准是：Kineto 报"几万次 launch + 上百次 sync + 累计 GEMM 时间 << 总墙时"。**没有这种证据就不要先做 CUDA Graph**，可能方向错了。

### 条件 2：每步 shape 固定

Decode loop 里每步 Q 长度相同、K/V past 长度相同。如果每次调用 shape 都漂移（diffusion 的新噪声、image generation 的不同分辨率），捕获的图对不上 replay 输入。

### 条件 3：内存地址稳定

每步 `cat` / `stack` / 任何"分配新 tensor"的操作，都会让捕获时烧的地址在 replay 时失效。**修法是预分配 buffer + `_kv_pos` 光标**：buffer 一次分配好，每次写入只移动光标，不重新分配。

### 条件 4：被捕获区域 control flow 分支稳定

Host 端数据驱动的 if 会让 capture 路径选择不确定。**修法是 `_patched_forward` 通过 `exec(compile(...))` 把已知死的分支编译期改写成 `if False:`**——前提是你已经验证过这些分支确实是死的（warmup 跑过一遍可以确认）。

### 条件 5：Re-capture 成本摊得平

Capture 有 RNG 副作用，所以"capture once / replay many"优于每 run 捕获。Re-capture 只在"不可摊平"的情形（比如 prompt 长度变化）触发，并且需要被验证**不污染同 prompt 确定性**。

把这 5 条套到 MiniMind-O 上：**5 条全过**。

## MiniMind-O：5 条全过 = 7.51×

接下来就是把 4 个障碍**一一打掉**的过程。这一段是整个 session 的 50 个迭代，每个迭代都对应一个 `tools/bench_*.py` 探针。

**打掉障碍 1（host 读）**：在 `nanovllm_omni/optim/cuda_graph.py:48-58` 写了 `_patched_forward`，对 `forward()` 源码做 `exec(compile(...))`，把两处 `if self.*.freqs_cos[0, 0] == 0:` 替换为 `if False:`。**不 rebind 原 class**，复制一份出来调。Warmup 跑一遍确认这两条分支从来没进过，编译期改写是安全的。这一步叫"中和 host-read"。

**打掉障碍 2（KV cat）**：在 `nanovllm_omni/models/minimind_omni/attention.py` §5 加了 `enable_fixed_kv_buffer`：预分配一个 `[max_len, n_heads, head_dim]` 的 KV buffer，每步把当前 K/V 写到 `_kv_pos` 位置，光标 += 1。CPU bit-exact 测试（`tests/test_fixed_kv_buffer_forward.py`）验证这条路径和 cat 路径数学等价，GPU 上进一步验证 bit-exact。

**打掉障碍 3（shape 漂移）**：不试图把整段 decode loop 写进一张图——而是**每 decode 步一张图**，捕获 `n_steps - 1` 张子图（prefill 以 eager 方式产出第一个 token，之后 `n_steps - 1` 个 decode 步各一张输入 shape = `[1, 9, 1]` 的捕获图）。shape 在每张子图内部是固定的。每张图是**位置依赖的**：`_kv_pos` 是 Python int，capture 时通过张量切片烧入了 KV 的写入位和读取范围。要做位置无关图，需要把 `_kv_pos` 转为设备端 tensor + 用 `torch.narrow` 索引——对注意力层的重写代价大，VRAM 节省微乎其微；per-position 方式是 vLLM / TGI 的标准做法，刻意保留。

**打掉障碍 4（re-capture 副作用）**：用 capture-once / replay-many 策略——**只在第一次调用时捕获**，后续调用纯 replay。每张图先做一次 eager warmup forward（让 cuBLAS workspace 选好算法），再用 `torch.cuda.graph(g)` 在 side stream 上捕获。Capture 期间的随机性只出现在第一次，之后全部 deterministic。

**额外发现的一个 defect（defect #5）**：这套捕获还有个我一开始没预料到的问题——per-step 图的 KV 偏移烧在**首次 prompt 的 `prefill_len`**。如果下一个 prompt 长度变了，复用旧图会让 KV 错位。修法是 `_needs_recapture()` 在 `_prefill_len` 变化时丢弃旧图 + 清零 buffer + 重新 prefill + 重建图。同 prompt 复用保持不动（保证重复确定性）。这个修复在 RTX 3050 上用 `tools/bench_defect5_3cycle.py` 跨 3 个长度 cycle + 2 轮 round-trip 验证 bit-exact。

最终数字（RTX 3050 4GB，torch 2.14，`HF_HUB_OFFLINE=1`，`max_tokens=16`，`PYTHONPATH=.`，seed 42；`docs/perf/aligned/cold-hot-v3.json` + `parity-v2.json`）：

- **单 CUDA Graph 单步**（原报告）：eager 36.11 ms/step → graphed 3.49 ms/step，**10.36× 单步加速**。
- **16 步 thinker decode cold/hot split**（`tools/bench_cold_hot_cuda_graph.py`）：eager median 1248 ms → graph **hot p50 196 ms / p95 218 ms**（30 次请求同 prompt 复用 graph），**~6.36× hot 加速**；cold path（含 capture）1153 ms，capture cost ≈ 956 ms，**一次 request 就回本**。VRAM peak 490 MB。
- **Graph × Eager token/audio parity**（`tools/bench_graph_eager_parity.py`，3 个 prompt × 16 token）：
  - `Hi` (plen=2)：eager 2466 ms → graph 1249 ms（**1.97×**，首 run GPU 噪声偏高）
  - `Hello there` (plen=2)：eager 918 ms → graph 178 ms（**5.14×**）
  - `Tell me a short story` (plen=6)：eager 906 ms → graph 1104 ms（**0.82×**，新 prompt 长度触发 recapture，摊到单次 request）
  - token 数：eager / graph 全部 16；audio 非 pad 行：8 / 8 完全一致
  - 平均 **2.65×**（含 GPU 首 run 噪声）
- **完整 E2E `Omni.generate` 4-cell sweep**（`python -m nanovllm_omni.optim.bench sweep-graphs --pipeline full`，6 prompt × 6 run，max_tokens=16；`docs/perf/aligned/e2e-sweep-v1/`）：
  - **无 graph**：total median **807.5 ms**（thinker 787 + talker 336 + code2wav 23；per-step 65.6；frames 9/10/15）
  - **仅 thinker**：total median **226.8 ms**（thinker 202 + talker 387 + code2wav 22）
  - **仅 talker**：total median **788.9 ms**（talker graph 强制 greedy，与采样开不是苹果比橘子）
  - **两个都开**：total median **223.8 ms**
  - **端到端加速 ~3.61×**（807.5 → 223.8 ms）。Talker graph 在这套 bench 里不增益（talker 在 E2E 中占 30–40%，每步 decode 本身已经很便宜）；全部加速都来自 thinker graph。
  - VRAM peak：1.1 GB（无 graph）→ 1.79 GB（两个都开），4 GB 卡余量充足。
- **公开 `Omni.generate` API e2e**：3 次不同长度 prompt 异 prompt cycle + 同 prompt 重复，**全部产出同一份 WAV 字节**（MD5 一致）。

对外可以引用的 headline 数字是 **3.61× 完整 E2E**，而不是 6.36×（仅 thinker）或 7.51×（thinker 原语）。Thinker 每步 graph 干掉了约 580 ms 的 host launch overhead；剩下的 talker 在图外、占用 360 ms 是下一阶段（talker 主干 MiniMindBlock loop 未 graph 化）的优化空间。

7.51× 不是峰值理论值（单步 10.36× 是），但 thinker decode 测量扣掉了 host-side sampling、buffer 重置、Mimi codec decode 这些不在图内的开销。**对一个 320ms 起步、零新依赖、纯 monkey-patch + CUDA Graph 捕获的方案来说，这个杠杆比是 25 轮 monkey-patch 加起来都比不上的。** 完整 E2E（thinker + talker + code2wav）实际落在 **3.61×**，剩下被不在图里的 talker 主导。

## 跨族扫描：为什么只做了一个

`nanovllm_omni` 仓库里注册了 4 个模型族：

| 族 | 装载路径 |
|----|----------|
| `minimind_omni` | 自家 safetensors + 5 处 monkey-patch + CUDA Graph |
| `smolvlm` | `transformers.AutoModelForImageTextToText` + `AutoProcessor` |
| `sd_turbo` | `diffusers.StableDiffusionPipeline.from_pretrained` |
| `smolvla` | `lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy` |

把 5 条件套上去之后，结果是这样：

**`smolvlm`**：1（launch-bound）2（shape 固定）3-5 都通过，但是**条件 3-5 是 transformers 自家引擎在满足**——我们走的是 `model.generate(...)`，它的 KV cache / capture / replay 跟我们项目层 capture 会撞车。要做就得放弃 transformers 的 `generate`，自己实现 KV cache 管理和 capture 钩子——这个工程量比 MiniMind-O 那次还大，而且 vllm 走的是同一套。所以**不自己做**，让 transformers / vLLM 自家的 CUDA Graph 干活。

**`sd_turbo`**：**条件 1 直接 fail**。默认 `num_inference_steps=1`（`deploy/sd_turbo.yaml` 锁的）——整个生成就是一次前向，不存在"每步 launch-bound"这个概念。而且每次调用换 latent + prompt，shape 漂移，condition 2 也 fail。UNet 主体是 GEMM，瓶颈在 kernel 工作量本身，不在 launch。**CUDA Graph 对它不是杠杆**——真要优化，方向是 `torch.compile` UNet / 量化 / attention slicing，全在 diffusers 那侧做。

**`smolvla`**：LeRobot 的 `SmolVLAPolicy` 内部 50 步 chunk 队列、`predict_action_chunk` 走自家 VLA inference。条件 5 满足（chunk 队列摊平 capture 成本），但**当前没接入 engine**——`nanovllm_omni/models/__init__.py` 只导出 minimind_omni 三件套，其余三族通过 `_load_builtin_pipelines()` side-effect 注册。要做 CUDA Graph 得先复刻 `enable_fixed_kv_buffer` 那一套基础设施进 lerobot policy，工程量等同于重新做一次 MiniMind-O。**未来候选**——不是这一轮的题。

## 两条正交策略：深入 forward vs 薄壳封装

这个结果背后是个有意为之的工程选择，不是"做完一个顺手做另一个"。

`nanovllm_omni/models/__init__.py` 顶部 docstring 写得很直白：

> Each loader is a thin function that returns a handle the rest of the engine can call. We do not reimplement model forward passes here — we delegate to vllm (for AR), diffusers (for image diffusion), and direct safetensors loaders (for MiniMind-Omni, which is small enough to bring up ourselves.

**MiniMind-O 是自家训的小模型（< 4GB）**，项目层决定"bring up ourselves"——所以能深入 forward，monkey-patch 5 个优化点，加 buffer 化，加 CUDA Graph capture。Pipeline 是 3 stage（thinker LLM_AR + talker LLM_AR + code2wav CODEC）。

**SmolVLM / SD-Turbo / SmolVLA 都是 HuggingFace / Stability / LeRobot 训的**，项目层只做"薄壳封装"——单 stage 闭包，stage 工厂直接调上游库。`models/__init__.py` 只导出 minimind_omni 三件套，其余三族走 side-effect 注册，用户不直接 import。

为什么是对称的？因为**上游库自己已经做了 serve-loop 优化**。transformers 的 KV cache + capture、diffusers 的 UNet 优化路径、lerobot 的 chunk 队列——任何一处在我们项目层再加 CUDA Graph 都会撞它们的图状态（典型症状：capture 阶段上游 attention 重分配 buffer → 我们烧的地址失效）。**深入 forward 和薄壳封装是两种正交策略**，前者适配"自家训的小模型"，后者适配"上游库 + 我们只做 stage 串联"。

如果未来想给 SmolVLA 也做 CUDA Graph，要做的第一件事不是 capture——是先把 `enable_fixed_kv_buffer` 那一套基础设施复刻到 lerobot policy 里。这是另一个 session 的事。

## 生态定位：patch-only 是少数派

横向对比一下 2026-09 的主流 ML 推理项目（[调研报告](https://mcig-ggg.github.io)）：

- **sglang** 的 `hf_transformers_patches.py` 也走集中 patch，但同时大量 subclass + AutoModel wrap；
- **vllm-omni** 的 `vllm_omni/patch.py`（27.9 KB / 10 处）也保留 patch 文件，但模型层走 subclass + 子模块 fork-rewrite（`voxcpm2_talker.py` 139 KB、`siglip2.py` 15 KB 等），还保留 1 处 `nemo_vendored/`；
- **vLLM 本体 / HF optimum / accelerate / TGI / DeepSpeed-MII** 都走 subclass + adapter，这是事实标准；
- **TensorRT-LLM / llama.cpp / MLC-LLM**（性能敏感型）走 fork-and-rewrite 到自家 runtime。

nanovllm-omni 真正的差异化**是 vendor = 0 + fork = 0 + subclass = 0**——「我们是孤例」可算不上，sglang 也类似：本节 + 上一篇博客里的 5 处 monkey-patch 全是 `setattr`，不打 vLLM 那一层间接（vllm-omni 的 patch 大多是给 vLLM 用的，是 cross-cutting 层的版本兼容 wrapper）。

Ollama 在 v0.30 刚反 de-fork 了自家 GGML、直接依赖 llama.cpp——他们花了几年才学到「维护 fork 的长期成本」这个教训，我们项目从一开始就走 patch-only 是站在他们的肩膀上。完整横向对比表在 nanovllm-omni 仓库的 `docs/perf/ncu-generate-kernels-2026-09-01.md` §51.3。

## 收尾

这个 session 拿了 7.51× thinker decode 加速 + 5 个 GPU 闸门全绿（serve-path / cross-prompt / longrun / determinism / Omni() e2e）。但更值钱的副产品是**那 5 个条件**——它把“Cuda Graph 能不能用”从一个“看运气”的玄学问题变成了一个可以 5 条逐一打勾的工程判断。

> **勘误（2026-09）**：7.51×（320→42 ms）测的是 **thinker decode loop**，不是完整 MiniMind-O E2E 流水线。完整流水线包括 thinker + talker + code2wav，实际 E2E 约 640–850 ms（talker 和 Mimi codec 占主要开销）。代码现在捕获 `n_steps - 1` 张图（不是 `n_steps`）：prefill eager 产出第一个 token，然后 `n_steps - 1` 个 decode 步各 replay 一张捕获图，省了一张废 warmup + capture + VRAM。图依然是 position-dependent（设计意图）。WSL 上重跑后的实际数字：thinker decode hot p50 ≈ 196 ms（vs eager 1248 ms，约 **6.36×**），capture cost 一次 request 就回本。参见 `tools/bench_cold_hot_cuda_graph.py` 的 cold/hot 分离 benchmark 和 `tools/bench_graph_eager_parity.py` 的 token/audio parity 验证。

如果你的模型也想做 CUDA Graph 优化，建议流程：

1. **先 ncu + Kineto 取证**——没有"launch-bound"的硬证据不要先做；
2. **把 5 条逐一打勾**——任一不满足就别做，或者换上游库自己优化；
3. **5 条全过 → 进入实现阶段**，按"打掉 4 个障碍"的顺序：先 KV buffer 化（condition 3）→ 再 host-read 中和（condition 4）→ 再 capture-once/replay-many（condition 5）→ 最后处理 re-capture 边界（defect #5 一类）。

完整证据链在 `nanovllm-omni` 仓库的 `docs/perf/ncu-generate-kernels-2026-09-01.md`（74KB, 51 节），§51 是为这篇博客写的跨族可行性扫描源材料。要看每一步的 GPU 探针：`tools/bench_*.py` 一共 24 个，每一个对应一个判断点。

下一篇计划写 SmolVLA 的接入——同样是 5 条逐一打勾 + 4 个障碍一一打掉,但场景从 AR 文本+音频换成 VLA 视觉+动作,到时候再说。

## 后续:talker 也接上 CUDA Graph(2026-09-06)

上一篇收尾的时候留了个尾巴:stage 0(joint model forward)那张大图拿到 7.51× 之后,stage 1 的 talker wrapper 还在 eager 跑。当时的论断是"5 个条件全过的是 minimind_omni 这一族",但仔细一看——stage 1 的 `_drive_talker_generation` 也满足"每步 shape 固定 + control flow 稳定"那几条,只是没人把图给它捕获过。这一轮做掉。

### 动机:stage 1 还剩 4.8s 没压

stage 0 那张图接好之后,end-to-end 从 846ms 压到 320ms。但 `talker.talker_mtp` 的调用路径全程 eager——每一步是 preprocess → forward → sample → talker_mtp 四段拼接,加起来仍然 **232 次 `cudaLaunchKernel`/step**。Talker decode 最多跑 192 步(watchdog 上限),所以每 request 最坏情况大约 **44 000 次 launch,纯 dispatch overhead**。提一句:**这是按 watchdog 上限外推的最坏情况**,后面 e2e 会看到真实路径跑不满——音频内容到 `audio_stop` 就提前停了。

### 先取证,再动手:RTX 3050 上跑 profile 脚本

`tools/profile_talker_gpu.py` 用真权重 + 真 tokenizer 在 3050 上跑,数据如下:

| 阶段 | median ms/step | `cudaLaunchKernel` events/step |
|---|---|---|
| Stage 0 eager decode step (joint model, `[1,1]`) | 50–60 | **461** |
| Stage 1 `_drive_talker_generation`(8 decode steps) | 25 | **232** |
| Per-request 外推(192 步 watchdog 上限全跑 eager) | — | ≈ **4.8s**,理论上限 |

(图是 graphed 后的 stage 0 单步对比、stage 1 是 eager 基线;脚本里两个阶段都跑了 graph 尝试,具体数字看 stdout。下文 e2e 会把「理论上限」打回原形。)

### 接线:加一个 flag,把图塞进 decode loop

改动面非常薄:

- `nanovllm_omni/config/registry.py:206` 加 `DeployConfig.use_talker_cuda_graph: bool = True`,`:377` 加 yaml 解析路径。
- `nanovllm_omni/models/minimind_omni/talker.py`:`_drive_talker_generation` 接 `mtp_runner=None`,decode loop 里改成 dispatch——有 `mtp_runner.decode` 就走 graph,否则走 `talker.talker_mtp`(`talker.py:880` 起的 docstring 写了 trade-off);`_talker_stage` 缓存 `TalkerMtpCudaGraph` 实例(`talker.py:1018` 和 `:1046`),同一个 `Omni` 实例不重复包。
- `nanovllm_omni/deploy/minimind_omni.yaml:8-12` 加 `use_talker_cuda_graph: true`,注释里写明"engage 之后强制 greedy,sampling 路径走 eager fallback"。
- 测试:`tests/test_full_pipeline.py` 加 3 个 case——flag 解析、e2e 跑通、dispatch 行为(传入 `do_sample=True` 时 graph 仍会落到 eager,因为 graph 抓不住随机性)。

**Trade-off 写在代码里**:graph 强制 `do_sample=False`。CUDA Graph 捕获期间 host 端的 RNG state 会被折进图,replay 时同一个 seed 取到的随机数序列是固定的,跟"每个 step 重新采样"的语义不一致。修法是把 `multinomial` 移到 graph 外,但 talker_mtp 内部的 sample 跟主干 forward 深度耦合,挪出来要拆 forward 接口。所以直接强制 greedy——MiniMind-O 默认 `temperature=0.2, top_k=50`,在已经 sharpen 的分布上 greedy vs sampling 一致性非常高,trade-off 可接受。yaml 注释里写明了"set to false to keep eager + sampling"。

### 踩坑:CUDA Graph 抓不住 default stream

写完接线代码第一次跑,graph **静默** fall back 到 eager:每步一个 `UserWarning: CUDA graphs must be captured on a non-default stream`,单步节省 1-2%。让我以为"接上了",实际 graph 没被真的 capture——`TalkerMtpCudaGraph.decode` 在 `do_sample=False` 时是 try-catch 一把,fall back 到 `self._invoke(...)` 之后,代码路径根本不会进 `_capture`。

`nanovllm_omni/optim/talker_cuda_graph.py:188` 的 `_capture` 一开始是:

```python
graph = torch.cuda.CUDAGraph()
current_stream = torch.cuda.current_stream(device=device)
...
with torch.cuda.graph(graph, stream=current_stream), torch.inference_mode():
    output = self._invoke(...)
```

`current_stream` 在上游 `_drive_talker_generation` 调进来时**就是 default stream**(`torch.cuda.current_stream()` 不带 device 参数时的默认值)。PyTorch 的 contract:`torch.cuda.graph(graph, stream=...)` 当传入的是 default stream 时**直接报错**,而不是自动 fallback。

**为什么 joint model 那张图没这问题**:`optim/cuda_graph.py` 用的是 `torch.cuda.graph(graph)`(无 `stream=` 参数)——PyTorch 自己开 internal capture stream,然后 join 回调用方。这条路径在 default stream / side stream 上都能跑。talker 这边我当时照葫芦画瓢多传了一个 `stream=current_stream`,正好踩到 default-stream 那个边界 case。

修复就是把 `stream=current_stream` 删掉(`optim/talker_cuda_graph.py:204-206` 注释写明了原因):

```python
# No ``stream=`` arg: PyTorch allocates an internal capture stream and
# joins back, which works whether the caller is on the default stream
# or a side stream. Passing ``stream=current_stream`` would fail when
# current_stream IS the default stream.
with torch.cuda.graph(graph), torch.inference_mode():
    output = self._invoke(...)
```

修完之后 warning 不再出现,`TalkerMtpCudaGraph.decode` 真进 `_capture`,e2e 从 1-2% 跳到 3-4%。**修一行,杠杆变了 2-3 倍**——这个故事跟上一篇"condition 4 用编译期改写绕过 host-read"是同一类:**CUDA Graph 工程里 90% 的优化来自"理解 capture 的硬约束",不是"想办法把 kernel 跑得更快"。**

### 先泼一盆冷水:第一次测出的数字是错的

第一次跑 `tools/profile_talker_graph_e2e.py`,报了一组漂亮的数字:`max_tokens=4` 省 14.9%、`max_tokens=512` 省 3.8%。commit 和上一版草稿都引用了。

事后 review 发现方法论错了:eager 那臂是 sampling(`do_sample=True`),graph 那臂被强制 greedy(`do_sample=False`)。两臂行为不一样,省下来的时间有一部分根本是「sampling 改 greedy」带来的,不全是 graph 的 dispatch 收益。graph 抓不住 multinomial 只能 greedy,这条注释里早就写了;但 eager 臂没同等地设 greedy,一对比就把两件事混在一起了。

修法:给默认 sampling params 也加 `do_sample: false`,两臂都 greedy,只比 graph dispatch 本身。重跑:

| 配置 | eager greedy | graphed greedy | 节省 |
|---|---|---|---|
| `max_tokens=4` | 256.0 ms | 243.2 ms | **+5.0%(12.8 ms)** |
| `max_tokens=512`(真实 yaml) | 5885.2 ms | 5833.5 ms | **+0.9%(51.8 ms)** |

数字缩水一大截。上一版那个 14.9% 里,大部分是采样在慢——eager 从 sampling 换 greedy 本身就省了接近 100ms;graph 真正贡献的是那 12.8 ms。老实讲,这个量级的收益,跟 stage 0 那张图的 7.51× 完全不是一个量级。

为什么这么小,原因跟上一版分析一致,但数字诚实多了:

1. **stage 0 主导 wall time**。`max_tokens=512` 这条路上 5.8s 的大头是 stage 0,joint model forward 早 graphed 了、真在做 30 个 AR step 的 GEMM。graph 只优化 stage 1 尾巴的 dispatch,杠杆有限。
2. **graph 只消 launch,不消 GEMM**。talker_mtp 里的 RMSNorm、softmax、MatMul 是真要算的。graph 把 232 次 launch 压成 1 次 replay,但 kernel 工作时间省不掉,节省的只有 host 排队。

所以这一节的诚实结论:**talker_cuda_graph 接通了、代码路径 clean、trade-off 在 yaml 注释里写明白,但纯 graph dispatch 收益在典型 config 下只有 1% 左右**。它是把上一篇留的「stage 1 还在 eager」的尾巴收掉,收尾的意义大于提速的意义。真要榨 stage 1,杠杆不在 graph 上——在「stage 0 能不能再短」或者「talker 能不能少跑几步」上。

### 顺手:attention.py 挪回 optim/

这轮 review 顺手发现 `attention.py` 放错地方了——5 个 monkey-patch(`enable_sdpa_decode` / `enable_fused_projections` / `enable_fused_rmsnorm` / `enable_fused_rope` / opt-in `enable_fixed_kv_buffer`)跟 `cuda_graph.py`、`talker_cuda_graph.py` 本来就是一个家族:都是「patch 掉模型行为来加速」。归位到 `nanovllm_omni/optim/attention.py`,`git mv` 保了历史,10 处 import 一起改。

### 测试 + commit

- `pytest -m "not smoke"`:**364 passed**。接线时加了 graph dispatch 分支的 CPU 测试(decode loop 真走 `mtp_decode(...)` 那条路),顺手删了 4 个只读 `profile_cuda_graph.py` 源码的 contract 测试——那个脚本退役了,test 的 subject 也没了。
- `tools/` 清理:删了 3 个被取代的 GPU probe(`profile_cuda_graph.py` / `bench_defect5_3cycle.py` / `profile_talker_mtp_only.py`),`profile_talker_mtp_only` 并进了 `profile_talker_vs_thinker`。
- 三个 review subagent(optim/ / tools/ / holistic)并行审了一遍,`_patched_forward` 补了「必须恰两处替换」的防御、replay 加了失败重置、`_capture` 的 stream 注释修正了。

下一篇写 SmolVLA 接入——同 5 条逐一打勾 + 4 个障碍一一打掉,但场景换 VLA 视觉+动作,届时 stage 1 的 talker_cuda_graph 那一套也许能复用到 SmolVLA 的 action chunk 队列里,到时候再说。
