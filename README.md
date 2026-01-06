# TWICE-Tracking-With-Implicit-Complexity-Estimation
## Topic: DDIM-based Motion Predictor for Multi-Object Tracking

### A Deterministic Approach to Non-Linear Motion Prediction

### 1. Introduction

**1.1 Motivation**

Multi-Object Tracking (MOT) in scenarios with non-linear motion remains a significant challenge in computer vision. Traditional Kalman Filter-based approaches assume constant velocity, failing in dance performances, sports events, and dynamic camera movements. Recent work DiffMOT Lv et al. (2024) introduced a Decoupled Diffusion-based Motion Predictor (D²MP) that models motion as a generative task, achieving 62.3% HOTA on DanceTrack. However, D²MP employs **single-step stochastic sampling**, introducing randomness that leads to: 

1. Trajectory instability: Same input produces different predictions 

2. ID switches: fluctuation in consecutive frames 

3. Non-smooth motion: Jerky tracklets in high-speed scenarios

Meanwhile, **Denoising Diffusion Implicit Models (DDIM)** have demonstrated superior deterministic sampling in generative tasks, yet remain unexplored for motion prediction in MOT. This proposal introduces the **first DDIM-based Motion Predictor** that reformulates D²MP's stochastic sampling as a deterministic ODE process, enabling:

1. Deterministic tracking: Identical inputs to identical outputs
2. Multi-step refinement: Flexible multi-step denoising for quality-speed trade-offs
3. Adaptive sampling: Dynamic step allocation based on motion complexity

**1.2 Target Applications**

1. Fancam Tracking: K-pop fancams require stable individual tracking amid rapid choreography and frequent occlusions. Our deterministic predictor reduces ID switches when idols cross paths or perform synchronized moves.
2. Highlight Editing: Automated sports highlight generation demands consistent player tracking across acceleration/deceleration. Adaptive sampling allocates more refinement steps to critical moments (goals, fast breaks) while maintaining efficiency in static scenes.

### 2. Method

**2.1 Problem Formulation**

Given consecutive frames **(X_{t-1}, X_t)** and detection boxes **B_t = {B^i_t}**, the objective is to predict motion state **M_{f,0} = (Δx, Δy, Δw, Δh)** for each object.

**DiffMOT's D²MP**: Models motion prediction as:

```latex
Forward: M_{f,t} = M_{f,0} + tc + √t·z, z ~ N(0,I)
Reverse: M_{f,0} = -c_θ(M_{f,t}, t, C_f)  # Single-step, stochastic
```

**Limitations**:

- Random variable *z* introduces unpredictability
- Single-step sampling lacks refinement capability
- No mechanism for complexity-aware allocation

**2.2 Proposed Model: DDIM Motion Predictor**

Reformulate decoupled diffusion as a deterministic ODE and solve via DDIM.

Mathematical Derivation

**Step 1**: Rewrite decoupled diffusion as ODE

```latex
Standard DDPM: dM_t = f(M_t, t)dt + g(t)dω  (SDE with noise dω)
DDIM Form:     dM_t = f(M_t, t)dt            (ODE, deterministic)
```

**Step 2**: DDIM reverse process

```latex
p_θ(M_{t-Δt}|M_t) = δ(M_{t-Δt} - μ_θ)  # Dirac delta (deterministic)

μ_θ = √(α_{t-Δt}) · [M_t - √(1-α_t)·ε_θ(M_t, t)] / √α_t
    + √(1-α_{t-Δt})·ε_θ(M_t, t)
```

where ε_θ replaces D²MP's c_θ, and α_t = ∏(1-β_s).
**Step 3**: Unified with Decoupled Diffusion

```latex
Original D²MP: M_{f,0} = -c_θ
DDIM-enhanced: M_{t-Δt} = (t-Δt)/t · M_t - Δt/t · c_θ  (Δt=1 → same)
```

Crucially: DDIM removes the √(Δt(t-Δt)/t) · z noise term from DDPM, ensuring determinism.

**2.3 Architecture**

```markdown
┌─────────────────────────────────────────────────────┐
│  Input: (X_{t-1}, X_t) + Historical Motion I_f      │
└──────────────────┬──────────────────────────────────┘
                   │
         ┌─────────▼─────────┐
         │  Frozen Backbone  │ (YOLOX from DiffMOT)
         │   Feature: F_t    │
         └─────────┬─────────┘
                   │
    ┌──────────────▼──────────────┐
    │  HMINet (Condition Encoder) │
    │  • 6-layer MHSA             │
    │  • Class token E_ce         │
    └──────────────┬──────────────┘
                   │
    ┌──────────────▼───────────────────┐
    │   DDIM Denoising Module (NEW)    │
    │  ┌────────────────────────────┐  │
    │  │ Multi-Step Sampling:       │  │
    │  │ for t in [T, T-1, ..., 0]: │  │
    │  │   ε_θ = MFL(M_t, E_ce)     │  │
    │  │   M_{t-1} = μ_θ (no noise) │  │
    │  └────────────────────────────┘  │
    │  ┌────────────────────────────┐  │
    │  │ Complexity Estimator:      │  │
    │  │ • Accel variance σ²_a      │  │
    │  │ • Direction change Δθ      │  │
    │  │ • Size change Δs           │  │
    │  │ → Steps = f(σ²_a, Δθ, Δs)  │  │
    │  └────────────────────────────┘  │
    └──────────────┬───────────────────┘
                   │
          ┌────────▼────────┐
          │   Output M_0    │
          │ (Final Motion)  │
          └─────────────────┘
```

Modifications from D²MP:

1. Replace single-step sampling with iterative DDIM loop
2. Add complexity estimator: C = 0.5·σ²_a + 0.3·Δθ + 0.2·Δs
3. Adaptive steps:
    - C < 0.3 → 1 step (linear motion)
    - 0.3 ≤ C < 0.7 → 2 steps (moderate)
    - C ≥ 0.7 → 3 steps (highly non-linear)

**2.4. Dataset**

**DanceTrack** (Sun et al., 2022):

- Train: 40 videos
- Validation: 25 videos
- Test: 35 videos
- Challenge: Uniform appearance + diverse non-linear motion

**2.5. Evaluation Metrics**

- HOTA (balanced detection + association)
- IDF1 (ID consistency)
- ID Switch (critical for fancam/highlight)

### 3. Expected Contributions

While prior works DiffusionTrack applies DDIM to detection and DiffusionMOT uses decoupled diffusion with single-step stochastic sampling for motion prediction, we propose the first DDIM-based motion predictor with three key innovations:

1. **Deterministic Sampling**: Unlike DiffMOT's stochastic single-step sampling, our DDIM formulation removes randomness through ODE-based denoising, reducing ID switches.
2. **Adaptive Multi-Step Refinement**: While DiffMOT's parallel sampling accelerates inference at fixed steps, our method dynamically adjusts sampling steps based on motion complexity, achieving quality-speed balance without retraining.
3. **Theoretical Unification**: We derive the ODE form of decoupled diffusion and integrate it with DDIM's deterministic framework, providing the first theoretical bridge between these two paradigms for trajectory prediction.

Our approach expect to achieve quality improvements on DanceTrack, while maintaining real-time performance through adaptive sampling.

### 4. Reference

[1] Weiyi Lv, Yuhang Huang, Ning Zhang, Ruei-Sung Lin, Mei Han, and Dan Zeng. Diffmot: A real-time diffusion-based multiple object tracker with non-linear prediction. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 19321–19330, 2024.

[2] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In *ICLR*, 2021.

[3] Peize Sun, Jinkun Cao, Yi Jiang, Zehuan Yuan, Song Bai, Kris Kitani, and Ping Luo. Dancetrack: Multi-object tracking in uniform appearance and diverse motion. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 20993–21002, 2022.

[4] Run Luo, Zikai Song, Lintao Ma, Jinlin Wei, Wei Yang, and Min Yang. Diffusiontrack: Diffusion model for multi-object tracking. In *Proceedings of the AAAI Conference on Artificial Intelligence*, pages 3991–3999, 2024.

[5] Y. Hu, J. Hua, Z. Han, H. Zou, G. Wu and Z. Wang, "DiffusionMOT: A Diffusion-Based Multiple Object Tracker," in *IEEE Transactions on Neural Networks and Learning Systems*, vol. 36, no. 10, pp. 18203-18217, Oct. 2025.
