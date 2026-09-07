---
title: CUDA Graph 实战入门：把模型推理的 dispatch 开销砍掉 90%
timestamp: 2026-09-06T20:35:00+08:00
tags:
  - AI
  - CUDA Graph
  - PyTorch
  - 性能优化
series: nanovllm-omni 开发手记
description: PyTorch 官方文档里 torch.cuda.CUDAGraph 那一段描述写得像法条，看完仍然不会写。这篇从一张最小可运行代码开始，把 4 步标准写法、5 条 gotcha、多 shape cache 和怎么验证图真的接通了，一一拆成可以照抄的形态。例子全部来自 nanovllm-omni 仓库的真实代码（optim/cuda_graph.py、optim/talker_cuda_graph.py、tools/profile_cuda_graph.py），不是伪代码。
toc: true
---

## 这篇要回答的问题

[上一篇 torch.profiler 入门](/note/MiniMind-O%20torch.profiler%20%E5%85%A5%E9%97%A8) 在 MiniMind-O 上抓出来 **17 000 次 `cudaLaunchKernel`** 堆在 320ms 的时间线上。[MiniMind-O ncu 入门](/note/MiniMind-O%20ncu%20%E5%85%A5%E9%97%A8) 又把这些 kernel 拆开看了一遍，发现：

> 320ms 里 kernel 真正在跑的时间大约 28ms。剩下 292ms 不是 kernel 慢，是 host 在排队、GPU 在空等。

这个形态叫 **launch-bound**。它的标准答案只有一个名字：**CUDA Graph**——把一段 kernel 序列录成一张图，replay 一次只要 3–5µs，跟图里有多少个 kernel 无关。

但 PyTorch 官方文档对 `torch.cuda.CUDAGraph` 的描述只有几行：

> CUDA graphs are a CUDA feature that allow a sequence of CUDA kernels to be captured and replayed with significantly reduced CPU overhead...

看完你还是不会写。这篇就是补这个缝：**从一个最小可运行的例子开始，把所有用得上的细节拆成 4 步 + 5 条 gotcha + 多 shape cache + 验证手段**。所有例子都从 nanovllm-omni 仓库里抠出来，不是空中楼阁。

## 4 步标准写法

对一个固定输入形状的 `nn.Linear`，录一张 graph 然后 replay 的完整代码长这样（这段是 nanovllm-omni 仓库 `tools/profile_cuda_graph.py` 的精简版）：

```python
import torch
import torch.nn as nn

model = nn.Linear(128, 64).cuda().eval()

# 第 1 步：准备静态 buffer（地址不能变，replay 时往里 copy_）
static_input = torch.randn(1, 128, device="cuda")

# 第 2 步：side stream warmup（让 cuBLAS / cuDNN 选好算法、分配好 workspace）
side = torch.cuda.Stream()
side.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(side):
    with torch.no_grad():
        for _ in range(3):                      # 至少 3 次
            _ = model(static_input)
torch.cuda.current_stream().wait_stream(side)
torch.cuda.synchronize()

# 第 3 步：capture
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph), torch.no_grad():
    static_output = model(static_input)        # 烧进 graph 的就是这个 forward

# 第 4 步：replay（每次换数据，地址不变）
new_input = torch.randn(1, 128, device="cuda")
static_input.copy_(new_input)                   # 必须 copy_，不能 = new_input
graph.replay()                                  # 跑出来的结果在 static_output 里
```

跑完你应该看到 `static_output` 已经装好了 `new_input` 经过 Linear 后的值。**这段代码 4 步缺一不可**：少 static buffer → 形状变了 replay 报错；少 side warmup → capture 失败（allocator 没准备好）；少 `with torch.no_grad()` → autograd 节点冲突；`new_input` 直接赋值给 `static_input` → graph 没收到新数据。

完整可跑的更长版本（带 eager vs replay 对照 + 中位数）长这样：

```python
import time, statistics
import torch, torch.nn as nn

model = nn.Linear(128, 64).cuda().eval()
x = torch.randn(1, 128, device="cuda")

# warmup on side stream
side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(side):
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)
torch.cuda.current_stream().wait_stream(side); torch.cuda.synchronize()

# capture
static_in = x.clone()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph), torch.no_grad():
    static_out = model(static_in)

def bench(fn, n=50):
    times = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e6)   # µs
    return statistics.median(times)

with torch.no_grad():
    eager_us = bench(lambda: _ = model(x))
    replay_us = bench(lambda: graph.replay())

print(f"eager:  {eager_us:7.1f} µs")
print(f"replay: {replay_us:7.1f} µs")
print(f"speedup: {eager_us / replay_us:.2f}x")
```

对一个 `nn.Linear(128→64)` 这种**单 kernel** 的小算子，eager 和 replay 几乎一样快（eager 也是一次 launch）。**CUDA Graph 的杠杆只在 kernel 多的时候才显出来**：你下一步应该把这个 `nn.Linear` 换成你的真实模型，再跑一遍这个 benchmark。

## 5 条 gotcha（按触发频率排序）

下面是 nanovllm-omni 仓库 `nanovllm_omni/optim/cuda_graph.py` 和 `optim/talker_cuda_graph.py` 在真实模型上踩过的所有坑。每条都配实际报错信息和修法。

### Gotcha 1：forward 里分配新 tensor

最常见的 capture 失败原因。PyTorch 的 `torch.zeros(...)`、`torch.cat(...)`、`torch.empty(...)` 在 forward 里被调时，每次都从 allocator 里拿一个新地址。capture 时这些地址被烧进 graph；replay 时 allocator 可能复用同一段地址给别的对象，graph 静默写错地方。

典型报错：

```
torch.cuda.CudaError: an illegal memory access was encountered
```

或者更隐晦：跑出来的数值对不上，但程序不崩。

修法：**预分配 buffer，写入用光标（cursor）**。nanovllm-omni 的做法是在 `nanovllm_omni/models/minimind_omni/attention.py` 里加一个 `enable_fixed_kv_buffer`：

```python
def enable_fixed_kv_buffer(model, max_len):
    for m in model.modules():
        if getattr(m, "_nanovllm_kv_buffer", False):
            m._kv_past_key   = torch.empty(max_len, m.num_heads, m.head_dim, device="cuda")
            m._kv_past_value = torch.empty(max_len, m.num_heads, m.head_dim, device="cuda")
            m._kv_pos = 0
            # 每步写入: m._kv_past_key[m._kv_pos] = k; m._kv_pos += 1
```

buffer 一次分配好，每步只往 `_kv_pos` 位置写入、然后 `_kv_pos += 1`。**地址从 capture 到 replay 一直不变**。

### Gotcha 2：host-device sync（`.item()` / `.cpu()` / `print(tensor)`）

capture 期间任何 host ↔ device 同步都会让 capture 路径不确定：

```
RuntimeError: During capture: encountered a synchronization operation
```

`tensor.item()`、`tensor.cpu()`、`print(tensor)`、`if tensor.sum() > 0:` 这类**把 device 值搬到 host 的操作**都会触发同步。

但**有些 host 同步是 capture 之前就要执行的**——比如判断要不要走某个分支。如果这个判断依赖一个 device tensor 的值，要么提前 `.item()` 到一个 Python 标量（capture 外），要么**改写 forward** 让分支条件变成 host 端的纯 Python 常量。

### Gotcha 3：host 端 read（控制流分支条件）

MiniMind-O 的 `forward()` 里有两行：

```python
if self.thinker.freqs_cos[0, 0] == 0:    # host 端读 freqs tensor
    ...
if self.talker.freqs_cos[0, 0] == 0:
    ...
```

这两行每次 capture 时都会把 device 端 `freqs_cos[0, 0]` 搬到 host 读一下——值在 host 端读一次是合法的，但 capture 阶段 CUDA 不允许 host read（路径选择不确定）。

报错：

```
RuntimeError: During capture: encountered a host-side read
```

nanovllm-omni 的修法是 `_patched_forward`：**不改原 class，而是把 forward 源码拷贝出来编译期改写**：

```python
def _patched_forward(cls, src):
    patched = src.replace(
        "if self.thinker.freqs_cos[0, 0] == 0:",
        "if False:  # CUDA-Graph: warmup precomputed"
    ).replace(
        "if self.talker.freqs_cos[0, 0] == 0:",
        "if False:  # CUDA-Graph: warmup precomputed"
    )
    namespace = dict(cls.forward.__globals__)
    namespace["__qualname__"] = cls.__qualname__ + ".forward_graph"
    exec(compile(textwrap.dedent(patched), "<enable_cuda_graph>", "exec"), namespace)
    return namespace["forward"]
```

关键点：**前提是这两条分支确实在正常 forward 里从来没进过**——这一点由 warmup 跑一遍来保证。Warmup 进过的分支不能这么改，改了会改变 forward 行为。

### Gotcha 4：随机数（`torch.multinomial` / dropout）

CUDA Graph 是**确定性 replay**——capture 时生成的随机数会被"录下来"，replay 多少次都是同一组数。这对 autoregressive sampling 是致命的：每步采同一个 token 就退化了。

报错**没有**——它会安静地输出重复 token。

修法：**采样留在 graph 外面**。nanovllm-omni 的 `optim/cuda_graph.py` 是这么做的：

```python
# graph 内只跑 forward
g, inp, out = self.steps[k]
inp.copy_(next_token)
g.replay()
# 采样在 graph 外，用 caller 的 generator
tok = sample_text_token(out.logits[0, -1], ..., gen=gen)
```

Graph 录的是 deterministic 的 forward（logits 计算），采样单独用 `torch.Generator` 跑——这样 replay 出 logits 永远是同一份，但采样随机性保留。

### Gotcha 5：不同 shape 不命中

CUDA Graph 是**按录制时的 shape 烧死的**。如果输入 batch 从 1 变 2、或者 sequence_len 从 16 变 17，replay 会直接报错：

```
RuntimeError: Cannot copy into a tensor of different shape
```

或者更糟：输出形状对不上但程序不崩，结果错位。

修法见下一节：**多 shape cache**。

### 速查表

| # | 症状 | 报错 / 现象 | 修法 |
|---|---|---|---|
| 1 | forward 里分配新 tensor | `illegal memory access` / 静默错位 | 预分配 buffer + cursor 写入 |
| 2 | host-device sync | `encountered a synchronization operation` | sync 提到 capture 外，或改 forward |
| 3 | host 端 read | `encountered a host-side read` | `_patched_forward` 编译期改写（前提：分支已证死） |
| 4 | `torch.multinomial` / dropout | 没报错，结果重复 | 采样留在 graph 外，forward 留在 graph 内 |
| 5 | shape 变了 | `Cannot copy into different shape` | 多 shape cache（下一节） |

## 多 shape cache：让 graph 适应任意输入

CUDA Graph 单图只能服务一个 shape。但生产里 batch / seq_len 会变。修法是**用 shape 作 key 缓存多张图**，每张图独立 capture、独立 replay。

`nanovllm_omni/optim/talker_cuda_graph.py` 里 `TalkerMtpCudaGraph` 是这个模式的完整实现，核心代码长这样：

```python
@dataclass(frozen=True)
class TalkerMtpGraphKey:
    """Fixed-shape identity for one captured graph."""
    batch_size: int
    input_ids_shape: tuple[int, ...]
    input_embeds_shape: tuple[int, ...]
    last_talker_hidden_shape: tuple[int, ...]
    text_step_shape: tuple[int, ...]
    active_mask_shape: tuple[int, ...] | None
    device: str
    input_ids_dtype: torch.dtype
    input_embeds_dtype: torch.dtype
    last_talker_hidden_dtype: torch.dtype
    text_step_dtype: torch.dtype
    active_mask_dtype: torch.dtype | None


@dataclass
class _TalkerMtpGraph:
    graph: Any
    buffers: dict[str, torch.Tensor]      # 这一 shape 的静态输入 buffer
    output: torch.Tensor                  # capture 时返回的 output


class TalkerMtpCudaGraph:
    _BUFFER_NAMES = ("input_ids", "input_embeds", "last_talker_hidden",
                     "text_step", "active_mask")

    def __init__(self, talker, *, batch_sizes=(1,), ...):
        self._cache: dict[TalkerMtpGraphKey, _TalkerMtpGraph] = {}

    def decode(self, input_ids, input_embeds, last_talker_hidden,
               text_step, *, active_mask=None, do_sample=True, generator=None):
        # 1. 用所有相关 shape/dtype/device 拼一个 key
        key = self._cache_key(input_ids, input_embeds, last_talker_hidden,
                              text_step, active_mask)

        # 2. 不支持的 shape / 采样路径 / CPU 输入：直接 fallback 到 eager
        if do_sample or active_mask is None or not self._graph_safe(key, ...):
            return self._invoke(...)

        # 3. cache 命中 → copy_ 数据 → replay
        entry = self._cache.get(key)
        if entry is None:
            entry = self._capture(key, {...})     # cache miss → capture
            self._cache[key] = entry
        for name in self._BUFFER_NAMES:
            if entry.buffers[name] is not None:
                entry.buffers[name].copy_(...)    # 把 caller 的 tensor 拷进 buffer
        entry.graph.replay()
        return entry.output.clone()

        # 4. capture 失败 → invalidate + fallback 到 eager
        except Exception as exc:
            self.invalidate(key)
            _log.warning("Talker MTP CUDA Graph failed; using eager: %s", exc)
            return self._invoke(...)
```

**这个模式有 4 个要点**：

1. **key 必须包含所有会影响 capture 的维度**——shape、`dtype`、`device`。少一个 key，两个不该复用的 graph 会被合并，replay 出错位。
2. **cache miss 走 capture，cache hit 走 replay**——同 shape 第二次调用直接命中，零额外开销。
3. **每次 replay 前 `copy_` 新数据进 buffer**——buffer 地址不变，只换内容。
4. **capture 失败时 `invalidate(key)`**——这张 shape 的图废了，下次同 shape 重新 capture。如果某种 shape 一直失败，warning 日志会暴露它，但 eager path 仍然能跑（fallback 不是 panic）。

完整的 `TalkerMtpGraphKey` 把 5 个输入的 shape + dtype + device 都写进去了。如果你的模型只用一两个输入（比如只有 `input_ids` 和 `past_kv`），key 可以简化：

```python
key = (input_ids.shape, past_kv[0][0].shape, past_kv[0][0].dtype, str(input_ids.device))
```

注意 `past_kv[0][0].shape` 也要进 key——KV cache 长度变化（prefill 长度不同）也是不同的 shape。

## 跟 torch.compile / dynamic shape 的关系

CUDA Graph、torch.compile、dynamic shapes 是三件经常被混淆的事。一句话分清：

| 机制 | 干什么 | 跟 CUDA Graph 的关系 |
|---|---|---|
| **CUDA Graph** | 把 N 次 launch 打包成 1 次 graph replay | —— |
| **torch.compile** | 把 Python 代码编译成 fused kernel 序列 | 互补：fused kernel 数少 → 单 graph 里 launch 更少 |
| **dynamic shape** | 让同一个 op 处理不同 shape 的输入 | 互补：减少 cache key 数（虽然实现复杂） |

CUDA Graph 是**调度层**优化——它不改 kernel，只改 kernel 的发射方式。torch.compile 是**算子层**优化——它把多个小 op 合成一个 kernel。两者不冲突，可以叠用：

```python
# 先 torch.compile 融合算子，再 CUDA Graph 录发射序列
compiled = torch.compile(model, mode="reduce-overhead")   # 内部已带 CUDA Graph
# 或者手动叠：
graphed = torch.cuda.CUDAGraph()
with torch.cuda.graph(graphed), torch.no_grad():
    out = compiled(static_input)                          # 内部已经 fuse 完
```

但要注意：`torch.compile(mode="reduce-overhead")` **自带 CUDA Graph**——你再手动包一层是冲突的。要么用 `torch.compile` 自带的 graph（最省事），要么自己 capture（最可控）。我们的项目选择自己 capture，是因为 `mode="reduce-overhead"` 在 MiniMind-O 上 crash 了，而且我们需要确定性 RNG 控制（采样留图外）。

## 怎么验证 graph 真的接通了

代码写完 capture 也"成功"了，不代表它真的在跑。nanovllm-omni 仓库的验证手段有 3 条：

### 验证 1：capture 失败的 warning 日志

`optim/cuda_graph.py:_capture` 包了一层 try/except，capture 失败会打 warning 并保留 eager fallback：

```python
try:
    with torch.cuda.graph(graph), torch.no_grad():
        out = self.fwd(self.model, input_ids=inp, past_key_values=None, use_cache=True)
except Exception as exc:
    self.invalidate(key)
    _log.warning("Talker MTP CUDA Graph failed; using eager: %s", exc)
    return self._invoke(...)
```

如果你跑完之后看到这条 warning，说明 capture 失败——图没接通，下次调用走的是 eager。

### 验证 2：用 `cudaLaunchKernel` 事件数对照

这条最硬。`torch.profiler` / Kineto 能给出每次 `model.generate(...)` 触发的 kernel 数。Eager 17 000 次、graphed 应该降到两位数。

简易写法：

```python
from torch.profiler import profile, ProfilerActivity

# 跑同一个 generate 一遍（eager），一遍（graphed），分别 profile
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(3):
        model.generate(prompt, sp)
# 看 Kernel 那一栏的 Total: 应该是几千而不是几万
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))
```

如果 graphed 的 kernel 数没掉一个数量级，图没真正生效。

### 验证 3：确定性（MD5）

CUDA Graph 是确定性的——同 prompt 跑 N 次应该出同一份输出。nanovllm-omni 仓库对 MiniMind-O 的接受标准是：

> 3 个不同长度 prompt 异 prompt cycle + 同 prompt 重复，全部产出同一份 WAV 字节（MD5 一致）。

伪代码：

```python
import hashlib

def md5_wav(wav_bytes): return hashlib.md5(wav_bytes).hexdigest()

# 同 prompt 跑 3 次
outputs = [model.generate(prompt, sp).audio for _ in range(3)]
assert len({md5_wav(o) for o in outputs}) == 1, "determinism broken!"

# 异 prompt 跑 N 次再回到第一个 prompt
out_first = model.generate(prompt_a, sp).audio
out_return = model.generate(prompt_a, sp).audio
assert md5_wav(out_first) == md5_wav(out_return), "cross-prompt determinism broken!"
```

如果 MD5 对不上，说明 `_needs_recapture()` 没正确丢弃旧图（defect #5 一类问题），或者 KV buffer 在不同 prompt 间没清零。仓库里 `optim/cuda_graph.py` 的 `_needs_recapture()` 方法专门处理这个：

```python
def _needs_recapture(self):
    return not (
        self._captured
        and self._captured_len == self._prefill_len
        and self._captured_n_steps == self.n_steps
    )
```

**`prefill_len` 变了必须重建图**——这是跨 prompt 复用 graph 的隐性陷阱。

## 一段能直接抄的最小例子

把上面所有点揉成一个能跑、能验证的最小例子。把 `MyModel` 换成你自己的 `nn.Module` 即可：

```python
import time, statistics, hashlib
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 64),
        )
    def forward(self, x):
        return self.net(x)

model = MyModel().cuda().eval()
x = torch.randn(1, 128, device="cuda")

# warmup
side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(side):
    with torch.no_grad():
        for _ in range(3): _ = model(x)
torch.cuda.current_stream().wait_stream(side); torch.cuda.synchronize()

# capture
static_in = x.clone()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph), torch.no_grad():
    static_out = model(static_in)

# bench
def bench(fn, n=50):
    times = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(times)

with torch.no_grad():
    eager_us  = bench(lambda: _ = model(x))
    replay_us = bench(lambda: graph.replay())

print(f"eager:  {eager_us:7.1f} µs")
print(f"replay: {replay_us:7.1f} µs")

# 验证确定性
new_x = torch.randn(1, 128, device="cuda")
static_in.copy_(new_x); graph.replay()
eager_out = model(new_x)
assert torch.allclose(static_out, eager_out, atol=1e-4), "graph output diverged from eager"
print("determinism OK")
```

跑完之后你会看到：

- 3 层 `nn.Linear` 的 eager 调用大约几十 µs；
- replay 几 µs（具体数字跟 GPU 有关）；
- `determinism OK` 说明 replay 输出和 eager 一致。

这个例子在你自己的模型上跑一遍，**如果 speedup > 1.5x，说明 CUDA Graph 对你这个 workload 有杠杆**；如果 < 1.2x，多半是你的模型单次 forward 的 launch 数还不够多，或者有 gotcha 命中了 eager fallback（看 warning 日志）。

## 收尾

CUDA Graph 的入门门槛主要不在 API 难调，而在**那 5 条 gotcha 都会让你的"成功 capture"静默退化成 eager**。完整流程应该是：

1. 先 `torch.profiler` + `ncu` 取证——确认你的 workload 是 launch-bound（参考[上一篇 torch.profiler 入门](/note/MiniMind-O%20torch.profiler%20%E5%85%A5%E9%97%A8) 和[MiniMind-O ncu 入门](/note/MiniMind-O%20ncu%20%E5%85%A5%E9%97%A8)）；
2. 按 4 步标准写法跑一个最小例子——验证你这边能 capture + replay + 确定性；
3. 把 5 条 gotcha 一条一条排查——每条配一个验证手段（host-read 用 `_patched_forward` + warmup 证明分支已死；KV 分配用 `enable_fixed_kv_buffer` + cursor）；
4. 多 shape 用 key-based cache + eager fallback——cache miss capture，capture 失败 invalidate；
5. 用 `cudaLaunchKernel` 事件数 + MD5 双验证确认图真的接通了。

如果这 5 步走完你的模型仍然不划算，多半是它本来就不 launch-bound（比如 kernel 工作量大、shape 不固定、batch 太小）——这种情况 CUDA Graph 不是杠杆，省下时间去做别的优化。完整的项目级实操记录在[《把 MiniMind-O 再压 7.5 倍：CUDA Graph thinker decode 实践 + 跨族可行性判定》](/note/MiniMind-O%20CUDA%20Graph%E4%BC%98%E5%8C%96%E5%AE%9E%E5%BD%95)，里面是 MiniMind-O 上把 17 000 次 launch 压到 7 次、拿到 7.51× **thinker decode 加速**（注意不是全流水线 E2E）的完整过程——读完这篇入门之后看那篇会顺畅很多。

> **注意**：实际代码捕获 `n_steps - 1` 张图（prefill eager 产出第一个 token，之后每个 decode 步一张），不是 `n_steps` 张。这避免了捕获一张永远不会 replay 的图的浪费。另外，每张图是 position-dependent 的：`_kv_pos` 是 Python int，capture 时通过张量切片烧入；要做到位置无关需要把 `_kv_pos` 转为设备端 tensor，代价大而收益微，vLLM/TGI 也是 per-position 做法。WSL RTX 3050 上的实际数字（cold-hot-v3）：**thinker decode hot p50 ≈ 196 ms**（vs eager 1248 ms，约 **6.36×**），capture cost 一次 request 就回本。参见 `tools/bench_cold_hot_cuda_graph.py` 和 `tools/bench_graph_eager_parity.py`。

下一篇计划写 torch.compile 跟 CUDA Graph 的叠用边界——同样 5 条逐一打勾，但场景从"自家训的小模型"换到"上游库 + 我们只做 stage 串联"，到时候再说。
