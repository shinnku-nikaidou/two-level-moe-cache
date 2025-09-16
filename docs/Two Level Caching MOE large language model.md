# Two Level Caching MOE large language model

> @author shinnkunikaidou

## 0. Abstract

pass.

## 1. Introduction

**Mobile edge computing is converging with large language models (LLMs) to deliver interactive intelligence close to users.** Emerging deployments place LLMs at access and metro edge sites to cut backhaul latency, reduce jitter, and improve data locality and privacy for applications such as conversational assistance, on-device agents, AR/XR overlays, and low-latency texts analytics. Unlike cloud data centers, edge nodes are severely resource-constrained—typically a single consumer-class GPU, limited VRAM, modest host↔device bandwidth over PCIe, and commodity NVMe. Under these constraints, end-to-end latency is often dominated not by compute but by moving parameters across the GPU↔CPU↔NVMe hierarchy, a challenge that is especially acute for sparse Mixture-of-Experts (MoE) models.

**Mixture-of-Experts (MoE) LLMs are attractive** because only a small subset of experts (**TOP-K**, e.g., 2–4) is activated per token, substantially reducing compute. On edge nodes with limited VRAM and RAM, however, the bottleneck shifts to **moving expert weights across the memory hierarchy**; naïve cache policies (e.g., FIFO or LRU) can severely stall decoding and dominate end-to-end latency.

**So we propose Two-Tier MoE Expert Caching**, a two-level cache tailored to resource-constrained edge inference with heterogeneous expert sizes and asymmetric promotion costs. It **exploits gating-induced temporal and spatial locality** in expert activations and **combines classical online caching with lightweight prediction**, while remaining model-agnostic and compatible with **mainstream MoE LLMs**..

**We design a Lagrangian-dual, watermark-based online policy**: each tier (VRAM and RAM) maintains a dynamic <u>credit-per-expert <b>watermark</b></u>. An expert’s near-term utility is predicted by blending short-horizon gating signals with recency; we normalize this utility by bytes to obtain a **value credit**. Experts are **admitted or retained only when their credit exceeds the tier’s watermark**. When capacity tightens, the watermark rises; when slack appears, it decays—so evictions occur naturally as low-density items fall below the watermark.

**We orchestrate density-ordered movement across tiers at runtime:** candidates are ranked by value density; **near-term** ones are promoted RAM→GPU, while **medium-horizon** ones are staged NVMe→RAM. Tier **watermarks** adapt to occupancy so evictions occur when pressure rises—without hand-tuned thresholds. This concentrates the GPU hot set on high-value experts, stabilizes latency under bursty gating, and remains model-agnostic for mainstream MoE LLMs.

## 2. Related work

[EdgeMoE: Empowering Sparse Large Language Models on Mobile Devices](https://arxiv.org/abs/2308.14352)

[\_SlimCaching: Edge Caching of Mixture-of-Experts\_\_for Distributed Inference](https://arxiv.org/abs/2507.06567)

[Large Language Models (LLMs) Inference Offloading and Resource Allocation in Cloud-Edge Computing: An Active Inference Approach](https://ieeexplore.ieee.org/document/10591707/)

[Two-level Graph Caching for Expediting Distributed GNN Training](https://ieeexplore.ieee.org/document/10228911)

[Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)

A. Mixture of Expert about Large Language Model Inference

B. Cloud and Edge for Large Language Models

C. Moe Caching

D: Moe Predicting & Selection

## 3. Model (Two-Tier MoE Expert Caching)

This section specifies the system model-entities, resources, timing, costs, constraints, and objectives for serving a **Mixture-of-Experts** ( #MoE ) LLM on a resource-constrained edge telecom gateway inference.

### 3.1 Scope and time

- Discrete time steps $t=1,2,\dots$ correspond to inference micro-steps (e.g., token decode steps or layer boundaries).
- At each step, exactly one Transformer layer $\ell(t)\in\{1,\dots,L\}$ is executed for one request.

inference:

$$
\ell(t)=((t-1) \bmod L)+1 \quad \text { for } t=1,2, \ldots
$$

### 3.2 Entities and Notation

#### 3.2.1 Layers and experts

- $L$ : number of Transformer (decoder) layers.
- For layer $\ell \in\{1, \ldots, L\}$, the expert set is $\mathcal{E}_{\ell}$ with $\left|\mathcal{E}_{\ell}\right|=E_{\ell}$ , Also $\mathcal{E}_{\ell} := \{1, \ldots, E_{\ell}\}$. Some time we treat $\ell$ as a function, $\ell(t) = \text{the on-going Transformer layer}$
- The experts are **layer-local objects**, we index then as $(e,\ell)$ with $e \in \mathcal{E}_{\ell}$ .
- At decoding step $t$, the router selects $\text{top-}k$ experts

$$
\mathcal{A}_{t} \subseteq \mathcal{E}_{\ell(t)}, \quad\left|\mathcal{A}_{t}\right|=k .
$$

#### 3.2.2 Storage Tiers and Capacities Constrains

- Tier $G$ (GPU VRAM) with byte capacity $K_G$ (available for experts after reserving non-expert weights, KV cache, etc.).
- Tier $R$ (host RAM) with byte capacity $K_R$ (can be relatively large).
- NVMe is considered the backing store (effectively unbounded for modeling).

**Residency state variables** at time $t$ :

$$
x_{e, \ell}^G(t) \in\{0,1\}, \quad x_{e, \ell}^R(t) \in\{0,1\}
$$

with the containment constraint

$$
x_{e, \ell}^G(t) \leq x_{e, \ell}^R(t) \leq 1
$$

Whether a given expert is currently resident in a particular cache tier.

$x_{e, \ell}^G(t) \in\{0,1\}$ : indicator for GPU memory ($\mathbf{G} = \mathbf{VRAM}$).

- $1:=\operatorname{expert}(e, \ell)$ is in VRAM at time $t$
- 0 = not in VRAM

$x_{e, \ell}^R(t) \in\{0,1\}$ : indicator for CPU memory ( $\mathbf{R} \boldsymbol{=} \mathbf{R A M}$ ).

- 1 = in RAM at time $t$
- $0=$ not in RAM (i.e., still on NVMe)

With the containment constraint $x_{e, \ell}^G(t) \leq x_{e, \ell}^R(t)$.

> Note: if an expert is in VRAM, it must also be in RAM (VRAM is a higher tier "subset"). Therefore the state $\mathrm{VRAM}=1$ and $\mathrm{RAM}=0$ is not allowed.

and capacity constraints

$$
\sum_{(e, \ell)} S_{e, \ell} x_{e, \ell}^G(t) \leq K_G, \quad \sum_{(e, \ell)} S_{e, \ell} x_{e, \ell}^R(t) \leq K_R
$$

where $S_{e, \ell}>0$ is the **object size (bytes)** for expert $(e, \ell)$ in its chosen precision/format.

### 3.3 Cost Model

Per-expert, per-layer load costs:

- $C_{e, \ell}^G \geq 0$ : cost to bring $(e, \ell)$ from RAM to VRAM (e.g., H2D + optional decompress + warmup).
- $C_{e, \ell}^R \geq 0$ : cost to bring $(e, \ell)$ from NVMe to RAM (e.g., SSD I/O + parse).

let $t^-:= t-1$ means the time **just before** the step’s compute.

An access (use) event for $(e, \ell) \in \mathcal{A}_t$ incurs stall cost

$$
c_t(e, \ell)=\left(1-x_{e, \ell}^G\left(t^{-}\right)\right) C_{e, \ell}^G+\left(1-x_{e, \ell}^R\left(t^{-}\right)\right) C_{e, \ell}^R
$$

to model PCle/NVMe throughput limits.

> If the object is resident in VRAM, cost is 0 ; if in RAM only, pay $C^G$; if on NVMe, pay $C^R+C^G$.

Now we use an abstraction. At each step $t$, the controller may choose admission, eviction actions that change the residency states:

- $u_{e, \ell}^{G,+}(t) \in\{0,1\}$ : promotion $(e, \ell)$ into VRAM (RAM $\rightarrow$ VRAM).
- $u_{e, \ell}^{R,+}(t) \in\{0,1\}$ : load $(e, \ell)$ into RAM (NVMe $\rightarrow$ RAM).
- $u_{e, \ell}^{G,-}(t) \in\{0,1\}$ : evict from VRAM to RAM.
- $u_{e, \ell}^{R,-}(t) \in\{0,1\}$ : evict from RAM to NVMe.

> Note that both load and evict can simultaneously happened in one step.

State transitions:

$$
\begin{aligned}
& x_{e, \ell}^G(t)=\min \left\{1, x_{e, \ell}^G\left(t^{-}\right)+u_{e, \ell}^{G,+}(t)-u_{e, \ell}^{G,-}(t)\right\}, \\
& x_{e, \ell}^R(t)=\min \left\{1, x_{e, \ell}^R\left(t^{-}\right)+u_{e, \ell}^{R,+}(t)-u_{e, \ell}^{R,-}(t)\right\},
\end{aligned}
$$

subject to $x_{e, \ell}^G(t) \leq x_{e, \ell}^R(t)$ and the capacity/bandwidth constraints above.

### 3.4 Objective

Primary objective: minimize cumulative (or average) stall cost/latency due to cache misses:

$$
\min \mathbb{E}\left[\sum_t \sum_{(e, \ell) \in \mathcal{A}_t} c_t(e, \ell)\right]
$$

Some other common multi-objective extensions (weights $\alpha, \beta, \cdots \geq 0$ ):

Define per-step bytes moved into each tier:

$$
\begin{gathered}
L_G(t)=\sum_{e, \ell} S_{e, \ell} u_{e, \ell}^{G,+}(t) \quad(\text { RAM } \rightarrow \text { VRAM promotions }), \\
\quad L_R(t)=\sum_{e, \ell} S_{e, \ell} u_{e, \ell}^{R,+}(t) \quad(\text { NVMe } \rightarrow \text { RAM loads }) .
\end{gathered}
$$

## 4. Algorithm

### 4.1 Probability Fusion

**Inputs (per expert–layer $(e,\ell)$)**

- $\widehat p^{\mathrm{EWMA}}_{e,\ell}(t)$: an **EWMA** (exponential weighted moving average) of recent activations on layer $\ell$ (fast, history-based).
- $\widehat p^{\mathrm{SG}}_{e,\ell}(t)$: **ScoutGate**’s single current prediction (no “cur/next” split) of activation probability.

**Base in-layer fusion:**

$$
p^{\mathrm{base}}_{e,\ell}(t)
 := (1-\eta)\,\widehat p^{\mathrm{EWMA}}_{e,\ell}(t)
+\eta\,\widehat p^{\mathrm{SG}}_{e,\ell}(t),\qquad \eta\in[0,1].
$$

#### Reuse distance and forward-causal weights

Let the executing layer at time $t$ be $\ell(t)$, and total layers $L$. Define the **layer-step reuse distance** until layer $\ell$ is needed again:

$$
D(\ell\mid \ell(t)):=
\begin{cases}
\ell-\ell(t), & \ell\ge \ell(t)\quad\text{(future layers in the current token)}\\[4pt]
\big(L-\ell(t)\big)+\ell, & \ell<\ell(t)\quad\text{(finish current token to }L\text{, then next token from }1\text{ to }\ell)
\end{cases}
$$

Define a **two-sided, non-zero** weighting kernel (forward-causal; past layers are not zeroed):

$$
W(\ell\mid \ell(t)) := \mathrm{e}^{- \gamma \cdot D(\ell\mid \ell(t))}\qquad  \ell \ge \ell(t)
$$

with **decay rates** $\gamma>0$ .

#### Final per-object score

$$
\boxed{\,p^{\mathrm{fuse}}_{e,\ell}(t) := p^{\mathrm{base}}_{e,\ell}(t)\cdot W(\ell\mid \ell(t))\,}
$$

> For the current layer $\ell=\ell(t)$, $D=0\Rightarrow W=1$.Future layers decay with forward distance; due to the definition of $D$, the immediately previous layer is **farthest** in reuse distance within the current+next-token schedule.

### 4.1 EWMA - Exponentially Weighted Moving Average

Define the **visit counter** for layer $\ell$ up to and including time $t$ as

$$
v_{\ell}(t):=\sum_{\tau=1}^t \mathbf{1}\{\ell(\tau)=\ell\} .
$$

So for any time $t$ with $\ell(t)=\ell$, the **visit count (layer-local clock)** for layer at time $t$ is

$$
k=v_{\ell}(t)=\left\lfloor\frac{t-\ell}{L}\right\rfloor+1
$$

The binary activation for expert ( $e, \ell$ ) on that visit count $k$ is

$$
\hat{p}_{e, \ell}^{\mathrm{HIT}}[k]:=\mathbf{1}\left\{(e, \ell) \in \mathcal{A}_{t}\right\} \in\{0,1\} . \qquad t=kL+\ell
$$

The EWMA on the layer-local clock $k$ is

$$
\hat{p}_{e, \ell}^{\mathrm{EWMA}}[k]=(1-\alpha) \hat{p}_{e, \ell}^{\mathrm{EWMA}}[k-1]+\alpha\, \hat{p}_{e, \ell}^{\mathrm{HIT}}[k], \quad \alpha \in(0,1], k=1,2, \ldots,
$$

with $\hat{p}_{e, \ell}^{\mathrm{EWMA}}[0]=p_0$.

Let

$$
\hat{p}_{e, \ell}^{\mathrm{HIT}}(t)= \begin{cases}1 & \text { if } (e, \ell) \in \mathcal{A}_t, \\ 0 & \text {if } (e, \ell) \notin \mathcal{A}_t,\end{cases}
$$

Equivalent global-clock form (updates only when llm is inferring at $\ell$ layer):

$$
\hat{p}_{e, \ell}^{\mathrm{EWMA}}(t)= \begin{cases}(1-\alpha) \hat{p}_{e, \ell}^{\mathrm{EWMA}}\left(t^{-}\right)+\alpha \, \hat{p}_{e, \ell}^{\mathrm{HIT}}(t), & \text { if } \ell(t)=\ell, \\ \hat{p}_{e, \ell}^{\mathrm{EWMA}}\left(t^{-}\right), & \text {if } \ell(t) \neq \ell,\end{cases}
$$

EWMA assigns geometrically decaying weights to past observations:

$$
\hat{p}_{e, \ell}^{\mathrm{EWMA}}[k]=\alpha \sum_{j=0}^{\infty}(1-\alpha)^j x_{k-j}^{(e, \ell)}
$$

The EWMA's variance reduction equals that of a simple moving average over an effective window size

$$
N_{\mathrm{eff}} \approx \frac{2-\alpha}{\alpha} \approx \frac{2}{\alpha}-1 \quad(\alpha \ll 1) .
$$

If $x_k^{(e, \ell)} \sim \operatorname{Bernoulli}(p)$ are i.i.d., then the EWMA is a stable, bounded estimator of $p$. Its steady-state variance satisfies

$$
\operatorname{Var}\left[\hat{p}_{e, \ell}^{\mathrm{EWMA}}\right]=p(1-p) \frac{\alpha}{2-\alpha},
$$

matching the $N_{\text {eff }}$ equivalence above. In non-stationary regimes the EWMA tracks changes with a time constant set by $\alpha$.

> Why layer-local time? Updating on a layer-local clock avoids diluting the signal with global steps where layer $\ell$ is not executed. Concretely, we update $\hat{p}_{e, \ell}^{\text {EWMA }}$ only when $\ell$ runs; other global steps leave it unchanged.

### 4.2 ScoutGate

ScoutGate provides a **single, current-time** probabilistic forecast of which experts will be activated, for every layer of the MoE model, using only information available up to global time $t$.

At global time $t$ (before executing any layer), for every layer $\ell=\{1, \cdots, L\}$ and every expert $e \in \mathcal{E}_{\ell}$ ,  ScoutGate produce independent activation probabilities by using semantic information

$$
\widehat{p}_{e, \ell}^{\mathrm{SG}}(t) \in[0,1],
$$

Each layer is $k$-hot. We therefore use sigmoid (independent Bernoulli) outputs, and add a cardinality regularizer to target $\mathbb{E}\left[\sum_e \hat{p}\right] \approx k$.

#### ScoutGate Algorithm

#####  1. Fetch Token

Take the most recent $m$ tokens: $(\mathrm{tok}_{t-m+1}, \ldots \mathrm{tok}_t)$,   ( with $m = 8$ ).
##### 2. Embedding

Use the main model's token embedding (or a frozen copy) to get $\mathbf{e}_i \in \mathbb{R}^{d_{\text {emb }}}$.

##### 3. Projection:
$$\mathbf{z}_i=\operatorname{LN}\left(\mathbf{e}_i W_{\text {proj }}+b\right) \in \mathbb{R}^{d_{\text {proj }}} ( d_{\text {proj }}= 128 )
$$

$\operatorname{LN}$ is Layer Normalization 
Layer Normalization Definition (for a single vector) Given $\mathbf{x} \in \mathbb{R}^d$,
 $$
\begin{gathered}
\mu=\frac{1}{d} \sum_{j=1}^d x_j, \quad \sigma^2=\frac{1}{d} \sum_{j=1}^d\left(x_j-\mu\right)^2 \\ \\
\operatorname{LN}(\mathbf{x})=\gamma \odot \frac{\mathbf{x}-\mu}{\sqrt{\sigma^2+\varepsilon}}+\beta
\end{gathered}
$$
- $\gamma, \beta \in \mathbb{R}^d$ are learnable per-dimension scale and bias;
- $\odot$ is element-wise multiplication;
- $\varepsilon$ is a small constant (e.g., $10^{-5}$ ) for numerical stability.

##### 4. Concatenate: 

Concatenate with a layer embedding. Let $\mathbf{z}_\ell=\mathrm{Emb}_{\text{layer}}(\ell)\in\mathbb{R}^{d_\ell}$.

$$
\mathbf{z}_{\text{ctx}}=\left[\mathbf{z}_{t-m+1}\ \|\ \cdots\ \|\ \mathbf{z}_t\ \|\ \mathbf{z}_\ell\right]\in\mathbb{R}^{m\cdot d_{\text{proj}}+d_\ell}.
$$

For a minimal change, set $\mathbf{h}_\ell=\mathbf{z}_{\text{ctx}}$. (Optionally add a linear to compress $\mathbf{h}_\ell$ to $\mathbb{R}^{d_h}$.)

##### 5. Two-tower scoring head

Assign each expert an embedding $\mathbf{v}_{e,\ell}\in\mathbb{R}^{d_e}$ per-layer.

With low-rank mappings $W_h\in\mathbb{R}^{d'\times d_h},\ W_e\in\mathbb{R}^{d'\times d_e}$:

$$
s_{e,\ell}=\big(W_h\,\mathbf{h}_\ell\big)^\top \big(W_e\,\mathbf{v}_{e,\ell}\big)+b_{e,\ell},\qquad
\widehat p^{\mathrm{SG}}_{e,\ell}=\sigma\big(s_{e,\ell}\big).
$$

##### 6. Output

$$
\widehat{\mathbf p}^{\mathrm{SG}}_\ell=\big(\widehat p^{\mathrm{SG}}_{1,\ell},\ldots,\widehat p^{\mathrm{SG}}_{E_\ell,\ell}\big)\in[0,1]^{E_\ell}.
$$

If all layers have the same expert count $E$, this reduces to a fixed-width output; otherwise, the two-tower head naturally supports variable $E_\ell$ without training separate models per layer.


### 4.3 Watermarks - two-tier Dual/Lagrange thresholds

We use $p_{e, \ell}^{\text {fuse }}(t) \in[0,1]$ directly in the two-level cache controller:

We design a Lagrangian-dual, watermark-based online policy for two tiers: GPU VRAM $G$ and host RAM $R$, with NVMe as the backing store. Objects are layer-local experts $(e, \ell)$.

#### Signals and "benefit density"

Benefit density (expected miss-cost saved per byte if the object is resident in a tier):

$$
b_{e, \ell}^G(t) := \frac{p_{e, \ell}^{\text {fuse }}(t) C_{e, \ell}^G}{S_{e, \ell}}, \quad b_{e, \ell}^R(t) := \frac{p_{e, \ell}^{\text {fuse }}(t) C_{e, \ell}^R}{S_{e, \ell}}
$$

These are in "cost per byte"; larger means "more value per byte of capacity".

#### Watermarks (dual variables)

Let $\lambda_G(t) \geq 0$ and $\lambda_R(t) \geq 0$ be the VRAM and RAM watermarks, respectively. They have the same unit ("cost per byte") and quantify how scarce each capacity currently is.

We denotes $[z]_{+}$ as **non-negative projection** (projection onto $\mathbb{R}_{\geq 0}$ ), equivalently a **ReLU**:

$$
[z]_{+} \triangleq \max (z, 0)
$$

They evolve by standard subgradient (**watermark**) updates against byte capacities $K_G, K_R$ :

$$
\begin{aligned}
& \lambda_G \leftarrow\left[\lambda_G+\eta_G\left(\sum_{e, \ell} S_{e, \ell} x_{e, \ell}^G(t)-K_G\right)\right]_{+}, \\
& \lambda_R \leftarrow\left[\lambda_R+\eta_R\left(\sum_{e, \ell} S_{e, \ell} x_{e, \ell}^R(t)-K_R\right)\right]_{+} .
\end{aligned}
$$

If a tier is over-occupied in bytes, its watermark rises (harder to keep/promote); if under-occupied, it falls.

##### Keep / Evict rules

We employ **hysteresis** (separate admit/evict thresholds) to suppress thrashing.

At each step $t$, for any expert $(e, \ell)$ not strictly required this very step (required ones must be made resident to execute):

VRAM decision (keep or demote to RAM):

Keep in VRAM iff

$$
b_{e, \ell}^G(t) \geq \lambda_G(t)
$$

Otherwise, can be demote to RAM.

RAM decision (keep or evict to NVMe) Keep in RAM iff

$$
b_{e, \ell}^R(t) \geq \lambda_R(t)
$$

Otherwise, can be evict to NVMe.

This is the natural greedy rule induced by the dual (watermark) objective with heterogeneous costs/sizes.

> Interpretation: $p^{\text {fuse }} \cdot C$ is the expected miss-cost avoided by caching $(e, \ell)$; dividing by size $S$ yields value per byte. Compare that to the tier's watermark (the byte-wise opportunity cost). If value/byte $\geq$ watermark, the object deserves capacity in that tier.

When bytes must be freed in a tier, evict any expert that conform to rule.

## 5 Experimental Result

## 6. Appendix

## 6.1

---

### GPT OSS 120b Per-Layer Attention (exact matrix sizes)

The model card states **64 query heads** with **64 dimensions each** (total Q dimension **4096**), while the **residual width is 2880**. Therefore:

- $W_q:2880→4096$
- $W_k:2880→512$: (GQA with **8 KV heads**, so total KV dim $8\times64=512$ )
- $W_v: 2880 \rightarrow 512$
- $W_o: 4096 \rightarrow 2880$

**Per layer parameter counts:**

- $W_q: 2880 \times 4096 = 11{,}796{,}480$
- $W_k: 2880 \times 512 = 1{,}474{,}560$
- $W_v: 2880 \times 512 = 1{,}474{,}560$
- $W_o: 4096 \times 2880 = 11{,}796{,}480$
- **Total per layer:** $26{,}542{,}080$ ≈ **26.54 M**
- **Across 36 layers:** ≈ **0.956 B**, matching the official ~0.96 B (small deltas come from biases/rounding).

---
