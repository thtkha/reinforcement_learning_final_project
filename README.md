# Autonomous Driving Navigation with Deep Q-Networks

A complete reinforcement learning project implementing a custom PyTorch Deep Q-Network (DQN) for autonomous highway navigation in the `highway-v0` environment.

## Project Overview

This project trains an RL agent to learn safe, efficient driving behavior through batched replay, target networks, epsilon-greedy exploration, and **safety-focused reward shaping**. The final policy achieves **zero collisions** and **195+ mean reward** over greedy evaluation episodes, compared to a 35–40 baseline.

## Key Results

| Model Variant | Episodes | Eval Reward | Eval Length | Collision Rate |
|---|---|---|---|---|
| Random Baseline | — | 35.46 | 44.5 | 100% |
| Vanilla DQN v1 | 100 | 54.33 | 58.67 | 100% |
| Stable DQN (Double target) | 150 | 62.93 | 69.0 | 100% |
| **Safety-Tuned DQN** | **400** | **195.39** | **200.0** | **0%** |

## Installation

### Requirements
- Python 3.11+
- CUDA (for GPU acceleration, optional but recommended)

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd RL_final_proj

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

Train the DQN agent with safety-focused reward shaping:

```bash
python train.py \
  --episodes 400 \
  --max-steps 200 \
  --collision-reward -20.0 \
  --high-speed-reward 0.08 \
  --lane-change-reward -0.10 \
  --model-path artifacts/dqn_model_safety.pt \
  --plot-path artifacts/training_metrics_safety.png
```

**Key hyperparameters:**
- `--episodes`: Number of training episodes (default: 300)
- `--batch-size`: Replay buffer batch size (default: 64)
- `--gamma`: Discount factor (default: 0.99)
- `--lr`: Learning rate (default: 5e-4)
- `--epsilon-start` / `--epsilon-end`: Exploration range (default: 1.0 → 0.10)
- `--collision-reward`: Penalty for crashes (default: -10.0, use -20.0 for safer training)
- `--run-baseline`: Include random-action baseline (flag)

### Evaluation

Evaluate a trained model greedily with optional video recording:

```bash
# Human rendering (for screen recording)
python evaluate.py \
  --model-path artifacts/dqn_model_safety.pt \
  --episodes 5 \
  --render-mode human

# Video recording
python evaluate.py \
  --model-path artifacts/dqn_model_safety.pt \
  --episodes 5 \
  --render-mode rgb_array \
  --video-folder artifacts/eval_videos
```

## Project Structure

```
.
├── train.py                      # Main training script
├── evaluate.py                   # Evaluation and video recording
├── requirements.txt              # Python dependencies
├── report_draft.md               # Project report (1-2 pages)
├── README.md                     # This file
├── .gitignore                    # Git ignore rules
└── artifacts/
    ├── dqn_model_safety.pt       # Trained model (zero collisions)
    ├── training_metrics_safety.png   # Training curves
    └── eval_videos_safety/       # Evaluation video clips
```

## Algorithm Details

### Deep Q-Network (DQN)
- **Architecture**: 2-layer MLP (32 input → 256 → 256 → 5 output)
- **Target Network**: Soft updates with Polyak averaging (tau=0.002)
- **Replay Buffer**: 150,000 capacity, preallocated NumPy arrays
- **Loss**: Smooth L1 (Huber) loss
- **Optimizer**: Adam (lr=3e-4)

### Safety Innovations
1. **Double DQN Target Selection**: Online network selects actions; target network scores them to reduce overestimation.
2. **Reward Shaping**: Increased collision penalty (−20 vs −10) and reduced speed reward (0.08 vs 0.15).
3. **Conservative Exploration**: Epsilon floor at 0.20 and slow decay (0.998) to avoid premature convergence.
4. **Extended Training**: 400 episodes for robust convergence.

## Results & Analysis

### Training Performance
- **Convergence**: Moving average reward stabilizes around 150+ by episode 200
- **Episode Length**: Consistent 200-step completion by episode 225
- **Stability**: Smooth learning curve with low variance in final episodes

### Evaluation (15 greedy episodes)
- **Mean Reward**: 195.39 ± 0.47
- **Mean Episode Length**: 200.0 steps (full maximum)
- **Collision Rate**: 0% (zero crashes across all episodes)
- **Behavior**: Stable lane selection, smooth speed control, reliable overtaking

## Challenges & Lessons

1. **Reward Misalignment**: Initial policies crashed due to speed-heavy rewards. Rebalancing resolved this.
2. **Training Instability**: Double DQN targets and conservative learning rates were critical.
3. **Generalization**: All results use fixed seed (123). Multi-seed evaluation recommended for robustness testing.

## Future Improvements

- **Dueling DQN**: Separate value and advantage streams for faster learning
- **Prioritized Experience Replay**: Focus updates on high-TD-error transitions
- **Multi-Seed Evaluation**: Measure robustness across random seeds and traffic patterns
- **Stress Testing**: Edge cases (dense traffic, high-speed vehicles, varying road configurations)
- **SB3 Comparison**: Benchmark against Stable-Baselines3 PPO/DQN

## References

- Mnih et al., "Human-level control through deep reinforcement learning" (Nature, 2015)
- Van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning" (AAAI, 2016)
- Brockman et al., "OpenAI Gym" (arXiv:1606.01540)
- Raffin et al., "Stable-Baselines3: Reliable Reinforcement Learning Implementations" (JMLR, 2021)

## License

This project is provided for educational purposes.

## Author

Final RL Course Project — Autonomous Driving with Custom DQN  
April 2026
