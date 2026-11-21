# AMD GPUs

All benchmarks in the README were run on an NVIDIA RTX 3060. AMD GPUs run the Vulkan backend
(`--backend vulkan`), and everything known about them comes from user reports (RX 9070 XT on Linux with RADV, and on
Windows with AMD's driver). Expect problems; this page lists the known ones and what to send when you hit a new one.

Needs a Vulkan 1.3 driver and subgroup size 32 or a fixed subgroup size 64. The wave64 path has been tested on
Radeon VII (Vega 20) with RADV Mesa 26.0.8. It runs the same int8 kernels as wave32: each 64-lane subgroup is split
into two logical 32-lane warps. The optimized batched-prefill path can differ from token-by-token inference at near
ties; `--chunk 0` uses token-by-token prefill when closer agreement with the CPU reference matters more than prompt
processing speed.

## First line of the output

Every run on Vulkan starts with a line like:

```
vulkan: AMD Radeon RX 9070 XT (radv Mesa 26.2.5), device-coherent memory yes, cooperative matrix f16 yes int8 no, conditional rendering yes
```

- `device-coherent memory` must be `yes` for models whose experts do not all fit in VRAM (see below).
- `cooperative matrix` speeds up prefill. `int8 no` is normal on RDNA. Without it, prefill attention uses a
  shared-memory tiled fallback; the `_dp` GEMM kernels still run without tensor cores.
- `conditional rendering` lets decode recover when the CPU expert worker is slow. With `no`, a slow worker is an error.

## Known issues

**Slow decode with experts on the CPU, and "GPU waits for the CPU expert worker that timed out and resumed: N".**
While the CPU computes the experts that are not in VRAM, the GPU polls a small mailbox buffer for the answer. On AMD the
GPU caches that buffer unless it comes from device-coherent memory (`VK_AMD_device_coherent_memory`), and then it misses
the answer and waits until it gives up. RADV hides the extension on RDNA4 (RX 9000) unless the driconf option
`radv_device_coherent_memory` is set. RustingStrata sets it before it opens Vulkan (unless you set it yourself) and prints a
warning when an AMD GPU still has no device-coherent memory. If the first line says `device-coherent memory no`, send
the output of `vulkaninfo | grep -i coherent`. A few resumed waits on a busy CPU are normal; hundreds are not.

**The GPU resets during a long prefill.** amdgpu resets a submission that runs for more than 10 s. Prefill submits
each layer separately, and attention within a chunk is split into smaller submits once its query-key work passes a
budget (`STRATA_VK_ATTN_BUDGET`, query-key pairs per slice, 0 turns slicing off). The default is 268435456 with
cooperative matrices and 33554432 without them (RDNA2 and older: the startup line says `prefill attention: tiled`),
where a pair costs about 8 times more. If it still happens, set a smaller budget (halve it) or try a smaller `--chunk`.

**"the forward pass produced NaN or infinite logits".** Seen once on Qwen3-Next 80B with a 15K prompt. Rerun with
`STRATA_NAN_TRACE=1`, which prints per-layer magnitudes and the first layer with a non-finite value, and attach the
output.

**A model split over two cards (`--device 0,1`) is slower than expected.** Each card waits while the other works, and
amdgpu clocks it down in those gaps. `echo high > /sys/class/drm/cardN/device/power_dpm_force_performance_level` (as
root, for each card; `auto` restores the default) keeps the clocks up: 44 to 57 tok/s decode in one two-card report.
The per-device times that `generate` prints on a split show whether a card is slow.

**Load fails for lack of VRAM.** The desktop on the same card takes memory, and the driver's free-memory estimate can
be more than it will allocate. Lower `--ctx`, use `--kv q8`, or a smaller `--chunk`.

**An integrated GPU refuses a command submission part way through loading a big MoE.** Two things are scarce on an APU
and only one of them shows up in `free`, which is why this took two wrong fixes to find.

*Bytes.* The device heap is a carve-out of system RAM, so a weight on the card is a second copy of bytes the model
already holds in page cache, and sizing that cache from "80% of what the card reports free" over-commits the machine. The
engine now sizes a shared-memory device from RAM instead (`plan` in `src/deepseek/mat.rs`): an eighth of the machine's
size, or what the OS reports free right now less 2 GB, whichever is smaller. Whatever it settled on is printed at load
(`GPU 0: shares system RAM with the CPU; 15.6 GB of weights on it (…)`), a matrix that no longer fits runs on the CPU, and
once budget and mapped files no longer fit together every byte the card *has read* gives its file page back
(`weights::upload_then_cold`; dropping the pages *before* the upload only lets the copy fault them straight back in).
`STRATA_DS_GPU_EXPERTS=<GB>` raises the budget up to what RAM could back, free space included; `STRATA_DS_GPU_NO_CAP=1`
removes the ceiling too.

*Objects.* This is what the Hy4-flash Q4_K_M reports actually hit. Its tensor table has 555 non-expert matrices and 1.0 MB
expert matrices, so a 14 GB expert cache is ~14000 separate device buffers and a 40 GB one ~40000 — and both runs died at
`__radv/amdgpu: Not enough memory for command submission` with ~110 GB of the machine's RAM untouched, which no byte
accounting explains. That message is ENOMEM from the submit ioctl, and the driver returns it when it runs out of its
per-process table of buffer objects just as much as when it runs out of memory. So expert slots and unconverted weights
now come out of 1 GB arenas instead of one buffer each (`Arena` in `src/deepseek/mat.rs`): a whole cache costs one driver
object per gigabyte, and the load says so — `gpu buffers: weights come from 1024 MB arenas, ~15 buffers for a full cache`. `STRATA_DS_PROF=1` prints the live count per token, which is the number to quote if a run still fails.

Neither ceiling was *measured* here until 2026-11-21, when the arena went in: the same binary then loaded Hy4-flash
Q4_K_M and generated greedily on a Strix Halo box (125 GB RAM, Radeon 8060S, Mesa 26.2.6, another engine already holding
38 GB of the card) at budgets of 1, 4, 8, 14 and 16 GB (and 16.8 GB left at the default, an eighth of RAM), and 64 tokens
in 29 s wall at 16 GB — where runs on this same box and file, at 14 GB and at 40 GB, had died at the submission. What
changed is the object count: 14 GB used to be ~14000 buffers, and became 15. So the fix is confirmed as sufficient at
those budgets; *which* limit the old code hit is still a hypothesis, because pinning it down means driving the driver
deliberately past a limit on a box where that costs a reboot. Ramp rather than guess big — one fresh process per step,
stopping at the first failure, because that failure is the answer and not something to retry:

```sh
for gb in 1 2 4 8 16; do
  STRATA_PRELOAD=0 STRATA_DS_GPU_EXPERTS=$gb STRATA_DS_PROF=1 \\
    rustingstrata generate --model $M --backend vulkan --gpu --ctx 512 -n 8 \\
    --prompt 'The capital of France is' 2>&1 | tail -20
done
```

A run that fails at 1 GB is not a memory problem at all; say so in the report with the `GPU 0:`, `gpu buffers:` and `PROF`
lines and the kernel's own words (`sudo dmesg | grep -iE 'amdgpu|gtt|oom' | tail -40` — the kernel, not Mesa, is the only
one that knows which allocator said no). amdgpu cannot fail these calls cleanly and the card can stay unusable until
reboot, so stop at the first such message, and close whatever else holds the card first
(`grep -H . /sys/class/drm/card*/device/mem_info_gtt_used` says what is in use). If every step passes, bytes were the
problem and the last one that passed is where to start. `STRATA_DS_GPU=0` needs no device allocations at all, reads
everything straight from the mapped file, and is a working answer for this model today: the CPU forward returns the same
greedy tokens.

**"The experts never all load, and decode is slow."** They load lazily: an expert is copied to the card the first time
the router picks it, so the cache grows a few megabytes per token and a short session never fills it. Two switches say
what you actually want, and the load prints the arithmetic for both:

```sh
experts: 89.5 GB of routed experts in 41 MoE layers, 22.2 GB of GPU budget free: 24% of the model fits; ...
warm: 15637 expert matrices, 22.2 of 89.5 GB resident in 15 s on 28 gpu buffers
```

* `STRATA_DS_GPU_EXPERTS=all` asks for as much as the machine can back instead of a number — on a shared-memory card
  that is the RAM free when the model loads, less 2 GB for the run — and `STRATA_DS_GPU_WARM=1` (implied by `all`)
  uploads the experts at load, expert 0 of every layer first, so the card is full before the first token instead of
  after the hundredth. Expert-major on purpose: a budget that covers a quarter of the model then holds a quarter of
  *every* layer, because a token routes across all of them.
* Hy4-flash Q4_K_M needs **89.5 GB of experts and 7.4 GB of everything else**, so "all of them on the card" is ~97 GB of
  pinned system RAM. On a 125 GB box that is only possible while nothing else big is running: the measurement above was
  taken with another engine holding 74 GB, which left 29 GB, and the load line says exactly that
  (`29.1 GB of weights on it (all this machine can back: 29 GB free of 134 RAM, …)`). Stop the other model first.
* Warmth is worth having: at 22 GB resident, an expert-cache lookup hit 700 times per token instead of 13. But the same
  run still decoded at 0.65-1.0 s per token, because ~550 experts a token were *not* resident and the CPU read their
  1.5 GB from the mapped file. That is a bandwidth wall, not a knob: the only fix is more of the 89.5 GB on the card, or
  the file in the page cache (`STRATA_PRELOAD=1`, so the CPU side reads it from RAM) with a smaller card budget. The two
  compete for the same RAM, and the `memory:` line at load says which of them you are in.

## Switches for testing

Environment variables; set any of them to turn the feature off:

- `STRATA_VK_NO_COOP`: no cooperative matrices.
- `STRATA_VK_ATTN_BUDGET=N`: query-key pairs of prefill attention per submit (default 268435456 with cooperative matrices, 33554432 without; 0 disables slicing).
- `STRATA_VK_DP_OLD`: use the old per-query prefill attention fallback instead of the tiled one (A/B and rollback).
- `STRATA_VK_NO_COND`: no conditional rendering.
- `STRATA_VK_NO_TQ`: no separate transfer queue for expert uploads.
- `STRATA_VK_SMEM_LIMIT=49152`: treat the device as allowing this much workgroup shared memory. The AMD proprietary driver reports 32768 on an RX 9070 XT although the card has 64 KB; RustingStrata raises that to 49152 itself on RDNA3/4 cards with this driver (no kernel needs more). For any other card that reports too little, set it by hand. Do not set it above what the card really has (issue #20).
- `radv_device_coherent_memory=false`: undo the RADV option RustingStrata sets.

If one of these makes a problem go away, say which one in the report.

## What to send in an issue

- GPU model, OS, driver and version (Mesa version on Linux).
- The output of `vulkaninfo --summary`.
- The full output of the failing command, including the first `vulkan:` line.
- If decode is slow: `rustingstrata bench --model <file> --gpu --backend vulkan --pp 1024 --tg 128` with the same model.
