---
title: 把 MiniMind-O 从 846ms 压到 320ms：25 轮实验的性能优化实录
timestamp: 2026-08-31T19:34:00+08:00
tags:
  - AI
  - LLM
  - 性能优化
  - CUDA
  - PyTorch
series: nanovllm-omni 开发手记
description: 在只允许 monkey-patch、零新依赖、不改上游模型代码的约束下，一个人用 25 轮自动实验把 RTX 3050 上 MiniMind-O 的端到端延迟从 846ms 压到 320ms 的全过程——包括一个潜伏了 22 轮的最大真实 bug。
toc: true
---
## 缘起：为什么要在 4GB 显卡上抠这几百毫秒

`nanovllm-omni` 是一个用 monkey-patch 把多种异构模型（**MiniMind-O / SmolVLM / SD-Turbo / SmolVLA**）缝到同一套 `Omni(...)` runtime 上的实验性推理引擎，本系列手记按模型拆开写。这篇挑其中一个跑得最频、瓶颈也最有意思的 case —— **MiniMind-O**（三阶段 omni：thinker → talker → mimi 语音解码）——来讲为什么「只允许打补丁」的优化反而最难；其他三个模型各自的接入实录见系列里的对应文章，不在本文展开。

我在本地一台 **RTX 3050 4GB Laptop** 上跑这个 case 做端到端音频生成。一次完整的 `Omni.generate` 流程是：

```
prompt → 最多 16 个 audio step 自回归 → mimi.decode → WAV 编码
```

最初这条链路要 **928ms** 上下，对于一个语音对话 demo 来说体感明显卡顿。我给项目定了一个目标：把端到端 wall-clock 压到 **≤ 500ms**。

但真正的约束在边界，不在目标。三条硬规则：

- 不引入任何**新依赖**（只能用 torch 内置算子）
- 不复制、不修改**上游模型代码**（HF 远程模型，`AutoModelForCausalLM(trust_remote_code=True)` 加载）
- 保持**公开 API 不变**（`Omni / AsyncOmni / SamplingParams / OmniRequestOutput`）

也就是说：只能在外面**打补丁**（monkey-patch），不能动里面的螺丝。这个约束恰恰把这次优化的性质从"改代码"变成了"在外层缝合"——也最终铸成了一个很有意思的 bug。

## 先搭一把可信的尺子：测量优先

优化之前，先解决"我怎么知道改动有没有用"的问题。这一步是我在这次经历里收获最大的地方之一。

在一小块 GPU 上，裸测一次 `generate` 的偶然波动远大于优化本身的收益。我们最终采用的 benchmark 方案是：

- 6 个 prompt × 5 runs × `max_tokens=16`
- 取 **median**（不被个别偶发尖峰带偏）
- 每次改动前后跑 20+ 次连续测量建立置信区间
- 把它定位成 fast regression gate，记录实际 frame 数和可读 WAV，不把 30 个样本当成可靠的 p95/p99
- 长度矩阵另外跑 `8,16,32,64,120`，用来抓 EOS、decode 和显存问题

一个非常反直觉的发现是 **GPU 温度对测量结果的影响是毁灭性的**：

- 冷态 GPU（60°C，1725 MHz）：stdev 只有 7–11ms，测量很干净
- 热态 GPU（>80°C，1590 MHz）：absolute wall-clock 直接**翻倍到 1200ms**

同一次改动，冷态热态测出来能差好几倍。这让我养成了"跑 bench 前先确认温度/频率"的习惯——不然你优化了个寂寞，还以为是 regression。

教训写成一个原则：**任何 < 20ms 的改动在 7ms 的噪声地板里都不可信**。后面 E13、E22 这类"听起来有道理但收益在噪声内"的改动，全部果断 discard。

### 重新跑一遍：原来的尺子还缺一半

这次把 benchmark 本身补了一轮。改动很小：`RunResult` 现在记录请求的 `max_tokens`，CSV 能区分不同长度；`run_one` 会用 stdlib `wave` 检查空 WAV、错误采样率、错误声道、空 PCM 和全静音；CLI 多了一个 `matrix` 子命令。

```bash
python -m nanovllm_omni.optim.bench matrix \\
  --lengths 8,16,32,64,120 --runs 3 --warmup 2
```

在 RTX 3050 上重新跑的结果有点刺眼：

| 路径 | 配置 | rows | median total | 实际 frames |
|---|---:|---:|---:|---:|
| eager | 6 prompts × 5 × 16 | 30 | **628.14 ms** | 9 |
| CUDA Graph | 6 prompts × 5 × 16 | 30 | **197.91 ms** | 16 |
| CUDA Graph length matrix | 3 prompts × 5 lengths × 3 | 45 | **103.62–134.53 ms/档** | 8 |

最后一行才是值得继续追的地方：`max_tokens=120` 并没有得到 120 个 frame，五个长度档位的实际输出都停在 8 frame。短 gate 测到了 Graph replay 的固定步数，长度矩阵则把“上限”和“真实 EOS 行为”区分开了。

这也解释了为什么 benchmark 不能只看 wall-clock。eager 路径这次是 628ms 左右，Graph 路径约 198ms；旧文里的 320ms 来自另一套执行路径和当时的基线环境，不能和这次结果直接横比。温度、clock、是否启用 Graph、实际 frame 数，都得跟着 CSV 一起记。

## 25 轮实验的全景

整个优化过程是在一轮轮"提出假设 → 做实验 → 测量 → 决定 keep/discard"的循环里推进的（autoresearch，25 轮）。完整轨迹如下：

| # | 描述 | 状态 | Median (ms) | Δ vs prev |
|---|---|---|---:|---:|
| baseline | KV buffer + SDPA | — | **846.4** | — |
| E1 | 移除 KV buffer，保留 SDPA | keep | 815.4 | −31 (−3.7%) |
| E3 | 删 `.clone()` 省拷贝 | discard | 1050 | +235（破坏正确性）|
| E4 | SDPA decode `is_causal=True` | keep | 803.1 | −12 (−5.1%) |
| E5 | **Fused RMSNorm** (`aten._fused_rms_norm`) | keep | **625.6** | **−178 (−26.1%)** |
| E6 | Fused QKV + gate-up 投影 | keep* | 467.8 | −158 (−44.7%) |
| E7 | Fused RoPE（消除 cat） | keep | ~478 | marginal |
| E8 | view/split 清理 | keep | ~478 | — |
| E10 | skip `rp==1.0` repetition penalty | keep | ~408 | −70 |
| E11 | model_input 单次 contiguous `copy_` | keep | ~397 | −11 |
| E12 | `torch.compile(dynamic=True)` on RoPE | keep | ~386 | −11 |
| E13 | RMSNorm shape 缓存 | keep（后 revert）| ~382 | −4 |
| E14 | `torch.compile(reduce-overhead)` RoPE | crash | — | CUDA Graph 错误 |
| E15 | 去掉 torch.compile RoPE | discard | ~395 | +13（确认 E12 有效）|
| E16 | `torch.compile` 整块 MiniMindBlock | keep (off) | ~473 | 默认关 |
| E17–E23 | 稳定性 / 确定性 / 跨 prompt 验证 | keep | 379–422 | — |
| **E24** | **修复 qkv_fusion dedupe bug** | **keep** | **320** | **−62 (−16%)** |
| E25 | E24 后重试 KV buffer | discard | 452 | +132（回归）|

> `*`：E6 标称 −44.7% 其实是被同一批合并进 baseline 的 RMSNorm / RoPE 改动贡献的。**QKV 融合的真实收益直到 E24 才显现**——因为在那之前它一直是坏的。

从 846ms 到 320ms，总计 **−62.2% wall-clock**。而这个数字里，最大的一笔"真实改进"其实是个 bug 的修复，跟聪明的算子融合没关系。

## 那 9 处 monkey-patch 是怎么缝上去的

所有优化都落在 `nanovllm_omni/models/minimind_omni/attention.py` + `bundle.py` + `generation.py`，全部通过**类 / 实例级别的 `forward = bound_method.__get__(self, cls)`** 完成，不碰远程模型一个字节。核心几处：

| Patch | 收益机理 |
|---|---|
| 预分配 `text_buffer` / `audio_buffer` | 消除生成循环内 `torch.cat` 的反复增长 |
| SDPA decode `is_causal=False` | decode 时 Q 长度 1 + 全量 KV past，`is_causal=True` 会让 mask 只 attend 到 K[0]，产出噪音音频 |
| Fused RMSNorm (`aten._fused_rms_norm`) | 把 `pow+mean+rsqrt+mul×2+float+type_as` 合并成单 kernel（E5，−178ms）|
| Fused QKV / gate-up 投影 | 3 个独立 matmul → 1 个 `qkv_proj`（E6/E24 的主角）|
| 修 `qkv_fusion` dedupe bug | 类级 `seen` 去重让 11/12 layer 一直没真融合；改成实例级后 −62ms（E24） |
| Fused RoPE（去 cat） | 利用 `cos/sin` 在 dim 上的重复性质，绕开 `rotate_half` 的 cat |
| skip `rp==1.0` repetition penalty | 默认 `rp=1.0` 时 entire unique+divide 是 no-op |
| model_input 单次 contiguous `copy_` | 9 个 strided write → 1 个 view+`copy_` |
| `torch.compile(dynamic=True)` on RoPE | Inductor 把 RoPE 的 ~6 个 elementwise kernel 融合成 ~2 |

这些缝合的微观收益单个都不大（ms 级），但叠起来就是质变——而且零成本：没有新依赖，没有新模型权重，只有几段绑定到类/实例的 `forward`。

## E24：那个潜伏了 22 轮的真凶

这是全过程的戏剧顶点，也是最大的工程教训。

`enable_fused_projections` 的本意是对每个 Attention/MLP 做一次 QKV / gate-up 融合 patch。去重逻辑是这样写的：

```python
seen_attn.add(cls)   # 按“类”去重
seen_mlp.add(cls)
```

但 `qkv_proj` / `gate_up_proj` 是**实例属性**，不是类属性。于是灾难性地：

> **12 个 Attention 里只有第 1 个（thinker.0）被 patch 了，其余 11 个 layer 继续跑 3 个独立 matmul。**

从 E6 开始，我一度以为"Fused QKV"已经生效了 22 轮。直到某次 profiling 发现 11/12 的层压根没融合，修掉这个去重 bug 之后——**median 直接从 382ms 掉到 320ms，−62ms，是 5.4× stdev 的显著改进**。

这一课我刻在脑子里：

> **class-level 的 `seen` 去重在 per-instance patch 场景下就是 bug。看到 `seen.add(...)`，先问一句：这个 patch 是打在类上还是实例上？**

## 撞墙记录：CUDA Graph 的边界

不是每条路都走得通，这块记录了几个"反直觉"的失败：

- `torch.compile(reduce-overhead)` 包 RoPE → **crash**：与 KV cache 后续的 `torch.cat` 冲突（"accessing tensor output of CUDAGraphs that has been overwritten"）。
- `torch.compile` 整块 MiniMindBlock.forward → **+91ms regression**：Inductor 的 launch overhead × 16 层 ≈ 700ms 远超收益。
- 全栈 CUDA Graph capture：理论收益 50–150ms，但需要把 KV cache 静态化 + monkey-patch `model.model.forward` 的 `start_pos` 计算，范围太大、revert 复杂、影响所有下游用户。**未实施**——收益不够 cover 风险。

在小模型 + 小批量上，`torch.compile` 的固定开销往往吃光收益，手写 operator fusion 反而稳赚。

## 最终结果与护拦

历史基线在 `11c3c13` 上是 320ms 左右。这次在同一类 RTX 3050 环境重新跑出两条结果：

```text
6 prompts × 5 runs × max_tokens=16
  eager median: 628.14 ms
  graph median: 197.91 ms
  eager frames median: 9
  graph frames median: 16
  WAV validation: 30/30 passed on both paths
```

> **勘误（2026-09）**：上面的对比存在 frame 数量不对称：egress 产出 9 frame（EOS 提前停止），graph 产出 16 frame（旧版不检查 EOS，跑满预算）。628 ms / 9 frame vs 197 ms / 16 frame 是苹果比橘子。现在代码已修复：graph decode 也执行 EOS + audio-stop 檢查，产出与 eager 相同的 frame 数（如 9/10/15）。同时捕获数量从 `n_steps` 修正为 `n_steps - 1`。WSL RTX 3050 上重跑（cold-hot-v3）：**thinker decode hot p50 ≈ 196 ms**（vs eager median 1248 ms，约 **6.36×**），capture cost 一次 request 就回本。参见 `tools/bench_cold_hot_cuda_graph.py` 和 `tools/bench_graph_eager_parity.py`。

length matrix: 3 prompts × 5 lengths × 3 runs
  8   frames budget -> median 134.53 ms, actual frames 8
  16  frames budget -> median 106.87 ms, actual frames 8
  32  frames budget -> median 103.62 ms, actual frames 8
  64  frames budget -> median 109.76 ms, actual frames 8
  120 frames budget -> median 111.32 ms, actual frames 8
```

并且验证了三件事：

- **Determinism**：6/6 prompt 的同 seed WAV MD5 一致，§5.4 PASS
- **Long-run stability**：20 次混合调用 control frames 全一致，VRAM tail-10 增长 68.2MB，PASS
- **Audio artifact**：每条有效结果都能通过 24kHz、mono、PCM16、非空 WAV 检查

但长度矩阵也留下了一个明确问题：CUDA Graph 下 `max_tokens=120` 仍然只返回 8 frame。这个结果足以说明短 benchmark 的覆盖边界，暂时不能把 16-step Graph 的速度数字写成“长音频性能”。

项目侧护栏继续保留——公开 API、非 smoke 测试和原来的 bit-exact determinism 都不改。下一步要查的是 EOS 和 audio delay schedule，不急着再造一个更复杂的 benchmark。

## 复盘：这次经历教会我的三件事

1. **先造一把可信的尺子，再谈优化。** GPU 温度能把测量漂移放大近一倍（1200ms vs 846ms）；在没搞清噪声地板之前，任何 <20ms 的"优化"都可能是自我安慰。中位数 + 多次测量 + 温度监控，是被几次假 regression 教训出来的习惯。

2. **最大的收益往往来自把已有的技巧真正做对——新技巧反而在其次。** 一个 `seen.add(cls)` 还是 `seen.add(self)` 的差别，等于 22 轮实验、11 个 layer、−62ms。别急着学新招，先检查老招有没有真的在整个对象集合上生效。

3. **约束逼出来的反而更稳。** "只能 monkey-patch + 只用 torch 内置算子"这条约束让我按算子语义去融合，而不是往项目里塞 CUDA kernel 或者复制 vendor 模型。最终的优化栈零依赖、零 vendor、位级无副作用。

完整优化栈与复现命令在 nanovllm-omni 的 `docs/perf/minimind-omni-under-500ms.md`，本地 `python -m nanovllm_omni.optim.bench time` 即可复现。

## 后续：当 talker 真的被装上之后

这篇实录写于 2026-08，当时剖析的"端到端"其实只到 **thinker 单段**：`prompt → 16 token 自回归 → mimi.decode`。9 月我把 vllm-omni PR #3796 对应的 talker/code2wav 从 identity 占位升级成了真正的**三阶段管线**（thinker → talker → MTP → code2wav），回头看，这里有一个必须修正的口径：

- **旧文里的 320ms 只到 thinker 单段。** 那是 thinker 单段解码（`bench time` 默认路径）的数字。三阶段全落地后，我加了 `bench time --pipeline full` 来量真正的端到端。
- **collapsed 路径退役。** 单 thinker 一把梭的管线删掉了（`8f6d522`），`pipeline_kind` 默认 `full`，三个阶段真跑。
- **真正的 full E2E 实测（RTX 3050 4GB，真权重，torch 2.14）：**

```text
6 prompts × 20 runs × max_tokens=16（--pipeline full）
  medium_01 -> 747 ms（p95 844）
  medium_02 -> 767 ms（p95 850）
  short_01  -> 645 ms（p95 709）
  short_02  -> 628 ms（p95 661）
  short_03  -> 634 ms（p95 669）
  system_01 -> 1038 ms（p95 1109）
  VRAM peak: ~1881 MiB
```

所以诚实的结论是：**MiniMind-O 真正的端到端在 RTX 3050 上是 0.63–1.04 秒**（中位）。320ms 只是其中一个阶段——这个数字以前被引用成"端到端"，是口径错误，现在 README 和博客都以 full E2E 为准。

三阶段里真正吃时间的是 talker 和 MTP（多 token 预测那些延迟 codebook），thinker 的 CUDA Graph 快路径仍然是单段最快的部分，但构成不了全量。4GB 显存上 full 峰值约 1881 MiB，勉强在预算内。

新增的复现：

```bash
python -m nanovllm_omni.optim.bench time --pipeline full --max-tokens 16 --runs 20 --warmup 1
```

数据原始 CSV 在 `docs/perf/full-e2e-rtx3050.csv`，验证细节在 `docs/perf/tk005-rtx3050.md`。

（这节的上一句"下一步要查 EOS 和 audio delay schedule"——结果 EOS 确实在 TICKET-05 里做了：post-EOS padding 状态机 + talker watchdog，见 `5da15f7`。撞墙记录里"全栈 CUDA Graph 未实施"的判断也值得重新审视：talker MTP 已经单独吃到 CUDA Graph 路径，`ac49cf9`。）

一步一步的 commit 顺序在 nanovllm-omni 的 `git log perf/cuda-graph-default-on`，从 `972224c`（talker 骨架）一直追到 `e91ca41`（--pipeline full bench）。

full 跑通的代价是显存：1881 MiB 峰值在 4GB 卡上已经贴着天花板，下一步想让 talker 也吃上 CUDA Graph 的整段 capture，或者把 MTP 的延迟 codebook 量化掉，都得先给显存腾地方。先这样。

