# Autonomous Driving Navigation with Deep Q-Networks

## Final Project Report

## 1. Project Goal and Problem Setup

The goal of this project is to train an autonomous driving agent in `highway-v0` using deep reinforcement learning, with a focus on both performance and reliability. The target behavior is to drive efficiently, perform safe overtakes, maintain lane discipline, and avoid collisions.

The required environment and control formulation are:

- Environment: `highway-env` (`highway-v0`)
- Observation: kinematic features (`x`, `y`, `vx`, `vy`) for ego and nearby vehicles
- Action space: 5 discrete meta-actions
  - `LANE_LEFT`
  - `LANE_RIGHT`
  - `FASTER`
  - `SLOWER`
  - `IDLE`
- Primary algorithm: custom PyTorch DQN
- Evaluation target: outperform random baseline and survive full episode length consistently

This project was developed iteratively from a basic DQN baseline to a reliability-focused final model.

## 2. Environment Design

### 2.1 Observation and Action Representation

We configured the environment to use a compact, structured state representation based on kinematics. The observation has shape `(8, 4)`, corresponding to 8 vehicles and 4 features each (`x`, `y`, `vx`, `vy`), which is flattened to a 32-dimensional vector before entering the Q-network.

The action space is `Discrete(5)`, matching the project specification and making value-based learning (DQN) a natural choice.

### 2.2 Reward Structure

Initial experiments used a default-like reward balance (speed + lane incentives, collision penalty). This produced acceptable reward improvement but poor safety reliability (frequent crashes under greedy policy).

To align optimization with safety, we introduced configurable reward shaping in `train.py`:

- `collision_reward`
- `right_lane_reward`
- `high_speed_reward`
- `lane_change_reward`
- `reward_speed_range`

Final safety-tuned configuration:

- `collision_reward = -20.0`
- `right_lane_reward = 0.15`
- `high_speed_reward = 0.08`
- `lane_change_reward = -0.10`
- `reward_speed_range = [18, 28]`

This shift penalizes unsafe behavior more strongly and discourages aggressive lane oscillation.

## 3. Algorithm: Custom DQN in PyTorch

### 3.1 Core Components

The custom implementation includes all standard DQN building blocks:

- **QNetwork**: MLP with two hidden layers (256, 256), ReLU activations
- **ReplayBuffer**: preallocated NumPy arrays for efficient transition storage and sampling
- **Target network**: soft-updated from online network
- **Epsilon-greedy exploration**: decayed over time
- **Loss**: `SmoothL1Loss` (Huber)
- **Optimizer**: Adam
- **Gradient clipping**: `max_norm=10.0` for stability

### 3.2 Stability Upgrade: Double DQN Targeting

To reduce overestimation bias and improve training stability, we updated the target computation to a Double DQN style:

1. Online network selects next action: `argmax_a Q_online(s', a)`
2. Target network evaluates that action: `Q_target(s', a_selected)`

This improves target quality versus plain max-over-target and helped produce more stable learning curves.

### 3.3 Training Loop

Per step:

1. Select action via epsilon-greedy
2. Step environment
3. Store transition in replay buffer
4. Sample mini-batch (when available)
5. Compute TD target and optimize online network
6. Soft-update target network
7. Decay epsilon with floor

Metrics tracked each episode:

- episode reward
- episode length
- mean episode loss
- epsilon value

Plotting at the end of training:

- reward per episode + moving average
- episode length + moving average

## 4. Experimental Process and Tuning History

We ran several iterative experiments to diagnose and improve policy behavior.

### 4.1 Early Baselines

- Random policy baseline showed high crash rates and low survival consistency.
- Early DQN versions improved reward, but greedy evaluation still produced 100% collision rates.

Representative early results:

- Random baseline: mean reward around mid-30s, collision rate 100%
- Early DQN runs (100-200 episodes): eval mean reward around 54-63, collision rate still 100%

### 4.2 Reliability-Oriented Tuning

We progressively introduced:

1. Double DQN target update
2. More conservative defaults (smaller LR, slower target update)
3. Extended training horizon
4. Safety-focused reward shaping
5. Slower exploration decay with higher epsilon floor

The final high-reliability run used:

- 400 episodes
- `lr = 3e-4`
- `buffer_size = 150000`
- `epsilon_start = 1.0`
- `epsilon_end = 0.20`
- `epsilon_decay = 0.998`
- `tau = 0.002`
- safety reward configuration listed in Section 2.2

## 5. Final Results

### 5.1 Training Outcome (Final Safety-Tuned Run)

- Episodes trained: 400
- Mean train reward: **124.141**
- Mean train episode length: **125.50**
- Training curves show substantial improvement in both reward and survival length relative to earlier runs.

### 5.2 Greedy Evaluation Outcome (Final Safety-Tuned Model)

Using `evaluate.py` with greedy policy (`epsilon = 0`):

- Evaluation episodes: 15
- Mean reward: **195.390**
- Reward std: **0.474**
- Mean episode length: **200.00 / 200**
- Collision rate: **0.0%**

This indicates reliable policy behavior with consistent full-episode survival.

### 5.3 Comparison Snapshot

| Variant | Episodes | Eval Mean Reward | Eval Mean Length | Collision Rate |
|---|---:|---:|---:|---:|
| Random baseline (representative) | 10 | ~35 | ~45 | 100% |
| Early DQN (100-200 ep range) | 100-200 | 54-63 | 59-73 | 100% |
| Stability-tuned DQN (pre-safety) | 150-400 | 60-63 | 63-69 | 100% |
| **Safety-tuned DQN (final)** | **400** | **195.39** | **200.00** | **0%** |

## 6. Evaluation Pipeline and Reproducibility

### 6.1 Training/Evaluation Integration

`train.py` saves model checkpoints and metadata (including reward config) to `.pt` files. `evaluate.py` loads the saved `q_network_state_dict`, reconstructs the network architecture, and runs greedy inference.

The evaluator supports:

- `render_mode="human"` for live visualization/screen recording
- `render_mode="rgb_array"` for automatic video recording

### 6.2 Practical Compatibility Fixes

During development, we resolved two integration issues:

1. Added `moviepy` dependency for Gymnasium video recording
2. Updated checkpoint loading for newer PyTorch behavior (`weights_only` default changes)

These fixes were committed and are reflected in the final repository setup.

## 7. Challenges and Lessons Learned

### 7.1 What Went Wrong Initially

- Reward and survival improved before safety did.
- Greedy policies looked numerically better but still crashed every episode.
- This exposed a reward alignment issue: speed incentives were too dominant relative to safety constraints.

### 7.2 What Solved It

- Stronger collision penalty and moderated speed reward
- Lane-change penalty to reduce unnecessary risky maneuvers
- Conservative exploration and longer training
- Double DQN target selection to reduce instability

The major lesson is that in safety-critical RL tasks, reward design is often more important than network size.

## 8. Limitations and Future Work

Although final reliability is strong under the tested setup, additional validation can strengthen scientific rigor:

1. Multi-seed evaluation and confidence intervals
2. Stress tests on denser traffic or altered traffic behavior
3. Dueling DQN and prioritized replay variants
4. Comparison against SB3 PPO/DQN under identical reward settings
5. Ablation study on reward terms to quantify each contribution

## 9. Conclusion

This project delivered a full, custom deep RL pipeline for autonomous highway navigation and achieved the target reliability outcome after systematic tuning.

The final safety-tuned DQN demonstrates:

- strong performance,
- stable long-horizon behavior,
- and zero collisions in greedy evaluation.

Beyond implementation, the project highlights a core practical insight: robust autonomous behavior in RL emerges from careful alignment of reward design, exploration strategy, and training stability mechanisms.
