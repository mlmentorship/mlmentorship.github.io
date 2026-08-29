# JAX Scaling Book content audit

**Prepared:** August 28, 2026

**Source reviewed:** [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) and its public source repository at commit `c6dd456`

**Scope:** Content that helps candidates reason about large-model training, inference, accelerators, and distributed systems interviews

## Bottom line

The book is a strong source for quantitative systems reasoning. Its most useful lesson is not JAX syntax. It is a method:

1. write tensor shapes;
2. count parameters, FLOPs, and bytes;
3. map tensor layouts to communication;
4. compare compute, memory, and network limits;
5. choose a parallel layout;
6. verify the estimate with a trace.

mlmentorship already covered many individual tools, including GPU memory, collectives, tensor parallelism, FSDP, pipeline parallelism, KV cache, prefill and decode, continuous batching, and activation checkpointing.

The main gap was the reasoning that connects these tools. Six concept pages and one worked interview question now fill that gap:

1. [Transformer compute and memory accounting](/concepts/transformer-compute-memory-accounting/)
2. [Sharded matrix multiplication](/concepts/sharded-matrix-multiplication/)
3. [Accelerator network topology](/concepts/accelerator-network-topology/)
4. [Strong scaling, MFU, and parallelism selection](/concepts/strong-scaling-and-parallelism-selection/)
5. [Context parallelism and ring attention](/concepts/context-parallelism-and-ring-attention/)
6. [Profiling distributed ML workloads](/concepts/profiling-distributed-ml-workloads/)
7. [Plan and cost a 70B transformer training run](/questions/plan-70b-training-run/)

A broad JAX tutorial is not recommended. JAX is important for some research and accelerator teams, but it is not a universal ML interview requirement. Candidates should add JAX practice only when the role or recruiter confirms it.

## Decision rules

A book topic belongs in the core library when it:

- applies across frameworks;
- supports a common research-engineering or ML-systems decision;
- can appear in an interview as an estimate, design choice, or debugging task;
- remains useful after a hardware generation changes;
- connects to an existing question, lab, or role path.

A topic should stay external or optional when it is mainly:

- JAX API syntax;
- a current accelerator specification table;
- a current cloud price table;
- one vendor's profiler controls;
- low-level hardware detail with no common interview decision attached.

## Chapter-by-chapter comparison

| Book chapter | Main skill | Coverage before this audit | Interview value | Action |
| --- | --- | --- | --- | --- |
| 1. Roofline analysis | Compare compute time with memory and communication time | Strong introduction in GPU memory hierarchy and accelerator trace practice | High for ML systems, training, inference, and kernels | Revise existing pages. Do not add another roofline page. |
| 2. TPU architecture and networking | Understand compute units, HBM, local memory, and chip topology | Generic GPU hierarchy existed. TPU and cross-platform topology were weak | Medium in general, high for TPU roles | Add a durable GPU and TPU topology model. Omit generation tables. |
| 3. Sharded matrices | Derive local shapes and collectives from tensor layouts | Collectives and tensor parallelism existed, but layout reasoning was missing | High | Add sharded matrix multiplication. |
| 4. Transformer math | Count parameters, training FLOPs, attention FLOPs, state memory, and KV memory | Facts were spread across several pages | High | Add one compact accounting page. |
| 5. Training parallelism | Select and combine data, tensor, pipeline, expert, and fully sharded methods | Individual methods were strong. Selection, MFU, and scaling efficiency were weak | High | Add strong scaling and long-context sharding. Revise existing method pages. |
| 6. Applied LLaMA training | Turn model and hardware dimensions into time, memory, and cost | The 100B answer named the right tools but had limited worked arithmetic | High | Add a worked 70B planning question. |
| 7. Transformer inference | Separate prefill and decode, size KV state, and reason about latency | Strong coverage across prefill, decode, KV cache, paging, batching, quantization, and speculative decoding | High | Keep existing pages. Add disaggregated serving to the production design answer. |
| 8. Applied LLaMA serving | Balance latency, throughput, memory, and serving capacity | Production inference question and scheduler lab cover the main decisions | High for inference roles | No new page now. Add another arithmetic exercise only if candidates use this path heavily. |
| 9. Profiling TPU programs | Compare traces with expected compute, memory, and communication | Accelerator trace question existed, but distributed-program profiling was not a findable concept | High | Add a framework-neutral profiling page. |
| 10. Programming TPUs in JAX | Express sharding and inspect compiled execution in JAX | Intentionally absent | Role-specific | Keep optional and external. Do not put it in every AI-lab path. |
| 11. Conclusion and reading list | Continue into specialist material | Not a curriculum topic | Low | Use as a source list only. |
| 12. GPU architecture and scaling | Understand GPU memory, nodes, scale-out networks, and collectives | Strong local memory overview. Cluster topology and collective details needed correction | High | Add topology page and revise collective cost claims. |

## Gaps that were real

### 1. Sharded matrix reasoning

The site described tensor parallelism and named its collectives. It did not teach the general rule behind them.

The missing skill was to inspect:

- input axes;
- contracting axes;
- output axes;
- global and local shapes;
- device-mesh axes;
- the layout required by the next operation.

This skill transfers across JAX, PyTorch, XLA, and custom distributed systems.

### 2. Transformer accounting in one place

The site had separate pages for architecture, KV cache, training state, and inference phases. It lacked a short worksheet that starts from $L$, $D$, $F$, head counts, vocabulary, batch, and sequence length.

The new page connects:

- gated feed-forward parameters;
- grouped-query attention parameters;
- the $6P$ training baseline;
- the quadratic attention term;
- 12 to 16 bytes of persistent training state per parameter under common storage choices;
- KV bytes per token and request.

### 3. Strong scaling and MFU

The existing pages explained how each method works. They did not provide one selection procedure.

The new procedure is:

1. identify what does not fit;
2. use the lightest sharding stage that resolves it;
3. preserve the intended global batch;
4. place frequent communication on fast links;
5. estimate exposed communication and bubbles;
6. compare MFU, memory, and accelerator-hours;
7. stop adding devices when scaling efficiency is poor.

### 4. Topology-aware communication

The old material often used a simple within-node versus across-node rule. That rule is useful as a starting point, but it is too absolute.

The new content compares:

- local accelerator memory;
- a fast accelerator domain;
- node or slice egress;
- scale-out network links;
- bisection bandwidth;
- measured collective bandwidth;
- large-message bandwidth and small-message latency.

### 5. Context parallelism

Sequence parallelism had a short mention. Context parallelism and exact ring attention did not have a dedicated explanation.

This matters for long-context roles because candidates should know:

- why local queries need remote keys and values;
- how online softmax combines blocks exactly;
- how causal masks work across device boundaries;
- why the method reduces local memory without removing total quadratic attention work;
- how grouped-query attention reduces communication bytes.

### 6. Distributed profiling

The accelerator challenge taught a good experiment loop for one simulated kernel. It did not teach how to read a full distributed training trace.

The new page adds:

- stable step selection;
- compute, memory, and communication lower bounds;
- exposed versus overlapped time;
- rank imbalance;
- unexpected reshard operations;
- pipeline bubbles;
- one-change experiments with correctness checks.

### 7. A complete planning exercise

The 100B answer was a good verbal overview. The new 70B question requires actual arithmetic from a model configuration and cluster description.

It covers parameter count, state memory, long-context attention FLOPs, global batch, tensor and data-parallel degrees, communication, MFU, elapsed time, accelerator-hours, restart, and convergence checks.

## Existing material that was already strong

No new concept page was needed for:

- basic roofline analysis;
- GPU HBM and on-chip memory;
- all-reduce, all-gather, and reduce-scatter definitions;
- FSDP and ZeRO stages;
- tensor and pipeline parallelism introductions;
- activation checkpointing;
- KV-cache sizing;
- prefill versus decode;
- PagedAttention and prefix sharing;
- continuous batching and chunked prefill;
- speculative decoding;
- mixed precision;
- fault-tolerant distributed training.

Adding duplicates would make navigation worse.

## Technical corrections made during the audit

The book comparison also exposed statements that were too broad or used inconsistent units.

### Collective communication

The revised collective page now:

- uses one definition of full tensor size;
- reports ring traffic per rank;
- includes fixed latency and effective bandwidth;
- identifies all-reduce as reduce-scatter plus all-gather;
- avoids calling all-to-all universally the most expensive collective;
- treats algorithm choice as dependent on message size and topology.

### Tensor and pipeline parallelism

The revised pages now:

- size tensor-parallel communication from activation bytes, not parameter count;
- explain all-reduce versus reduce-scatter and all-gather layouts;
- avoid saying tensor parallelism cannot cross nodes;
- describe within-node placement as a common result of bandwidth, not a law;
- remove a fixed example layout that looked universal.

### Training memory

The revised FSDP page now states a range of 12 to 16 bytes per parameter. The difference is whether separate FP32 master weights are stored. It also includes transient gathered parameters, communication buffers, and activations.

The revised activation-checkpointing page uses the correct simple-chain memory model:

$$
O(L/K + K),
$$

where $K$ is the number of layers in a recomputed segment. It no longer claims that checkpointing every block reduces all activation memory by the number of layers.

### Inference bottlenecks

The revised inference pages now state:

- BF16 batch-one decode does about 1 FLOP per weight byte, before other traffic;
- large, efficient prefills are often compute-bound, not every prefill;
- small decode batches are often memory-bound, not every decode batch;
- batching helps until compute, KV traffic, or scheduler work becomes limiting;
- disaggregated serving must repay KV transfer and queueing costs.

## Content intentionally not added

### Broad JAX API tutorial

The book's JAX examples are valuable implementation practice for JAX teams. They should not be required for every Applied Scientist, ML Engineer, Research Scientist, or Research Engineer path.

Use the book's JAX chapter when a role requires:

- JAX or XLA work;
- TPU programming;
- explicit mesh and sharding APIs;
- compiler-level performance debugging.

Otherwise, the framework-neutral concepts are enough.

### Detailed accelerator specification tables

Capacity and bandwidth tables become stale. The core site should teach candidates how to use a supplied specification or benchmark result.

A specialist exercise can provide a dated hardware sheet as input. The concept library should not depend on those exact values.

### Vendor prices

Cloud and reserved-capacity prices change by region, contract, and date. The worked exercise reports accelerator-hours. A candidate can multiply by the current effective hourly price when it is supplied.

### Low-level chip programming

Warp scheduling, TPU vector units, Pallas, CUDA, and Triton remain specialist material. Add them only when a kernel or accelerator path has enough demand and a current specialist reviewer.

## Remaining lower-priority opportunities

These do not block the current curriculum:

1. Deepen the MoE page with hierarchical expert-parallel placement and measured all-to-all balance.
2. Add a separate serving-capacity arithmetic question if inference candidates need more practice after the current scheduler lab.
3. Add an optional JAX sharding exercise to a specialist overlay, not the core role path.
4. Add a current GPU kernel lab only with executable tests and a maintained hardware or simulator target.

## Recommended path placement

### Research Engineer and training systems

Use this order:

1. Transformer compute and memory accounting.
2. Sharded matrix multiplication.
3. Collectives.
4. Accelerator topology.
5. FSDP, tensor, and pipeline parallelism.
6. Strong scaling and parallelism selection.
7. Context parallelism for long-context roles.
8. Plan the 70B run.
9. Profile a distributed workload.

### Inference systems

Use this order:

1. GPU memory hierarchy.
2. Transformer accounting.
3. KV cache.
4. Prefill versus decode.
5. PagedAttention and continuous batching.
6. Production inference service design.
7. Inference scheduler lab.

### General Applied Scientist or ML Engineer

Do not require the full sequence. Keep transformer accounting and the main training or serving cost model. Add sharding, topology, and profiling only when the target role owns large-model infrastructure.

## Primary sources

- [Book overview](https://jax-ml.github.io/scaling-book/)
- [Roofline analysis](https://jax-ml.github.io/scaling-book/roofline)
- [TPU systems](https://jax-ml.github.io/scaling-book/tpus)
- [Sharded matrices](https://jax-ml.github.io/scaling-book/sharding)
- [Transformer math](https://jax-ml.github.io/scaling-book/transformers)
- [Training parallelism](https://jax-ml.github.io/scaling-book/training)
- [Applied training](https://jax-ml.github.io/scaling-book/applied-training)
- [Transformer inference](https://jax-ml.github.io/scaling-book/inference)
- [Applied inference](https://jax-ml.github.io/scaling-book/applied-inference)
- [Profiling](https://jax-ml.github.io/scaling-book/profiling)
- [JAX and TPU programming](https://jax-ml.github.io/scaling-book/jax-stuff)
- [GPU systems and scaling](https://jax-ml.github.io/scaling-book/gpus)
