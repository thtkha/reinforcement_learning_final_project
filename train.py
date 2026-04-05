import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import gymnasium as gym
import highway_env  # noqa: F401 - required to register highway environments
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class BaselineStats:
    mean_reward: float
    std_reward: float
    mean_steps: float
    collision_rate: float


class QNetwork(nn.Module):
    """MLP Q-network for flattened kinematics observations."""

    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        hidden_sizes: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_actions))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() > 2:
            x = x.flatten(start_dim=1)
        return self.network(x)


class ReplayBuffer:
    """Preallocated replay buffer for efficient off-policy sampling."""

    def __init__(self, capacity: int, obs_shape: tuple[int, ...], device: torch.device) -> None:
        self.capacity = capacity
        self.device = device

        self.states = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_states = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

        self.pos = 0
        self.size = 0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = float(done)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = np.random.randint(0, self.size, size=batch_size)

        states = torch.as_tensor(self.states[indices], device=self.device).flatten(start_dim=1)
        actions = torch.as_tensor(self.actions[indices], device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(self.rewards[indices], device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(self.next_states[indices], device=self.device).flatten(start_dim=1)
        dones = torch.as_tensor(self.dones[indices], device=self.device).unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return self.size


@dataclass
class DQNComponents:
    q_network: QNetwork
    target_network: QNetwork
    optimizer: optim.Optimizer
    loss_fn: nn.Module
    replay_buffer: ReplayBuffer
    device: torch.device
    obs_dim: int
    n_actions: int


@dataclass
class TrainingHistory:
    episode_rewards: list[float]
    episode_lengths: list[int]
    episode_losses: list[float]
    epsilons: list[float]


def initialize_dqn_components(
    env: gym.Env,
    buffer_capacity: int = 100_000,
    learning_rate: float = 1e-3,
    hidden_sizes: Sequence[int] = (256, 256),
) -> DQNComponents:
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise ValueError("DQN requires a discrete action space.")

    obs_shape = env.observation_space.shape
    if obs_shape is None:
        raise ValueError("Observation space must define a shape for DQN initialization.")

    obs_dim = int(np.prod(obs_shape))
    n_actions = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    q_network = QNetwork(
        input_dim=obs_dim,
        num_actions=n_actions,
        hidden_sizes=hidden_sizes,
    ).to(device)
    target_network = QNetwork(
        input_dim=obs_dim,
        num_actions=n_actions,
        hidden_sizes=hidden_sizes,
    ).to(device)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    loss_fn = nn.SmoothL1Loss()
    replay_buffer = ReplayBuffer(capacity=buffer_capacity, obs_shape=obs_shape, device=device)

    return DQNComponents(
        q_network=q_network,
        target_network=target_network,
        optimizer=optimizer,
        loss_fn=loss_fn,
        replay_buffer=replay_buffer,
        device=device,
        obs_dim=obs_dim,
        n_actions=n_actions,
    )


def select_action(
    state: np.ndarray,
    dqn: DQNComponents,
    epsilon: float,
    env: gym.Env,
) -> int:
    if random.random() < epsilon:
        return int(env.action_space.sample())

    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=dqn.device).unsqueeze(0)
    with torch.no_grad():
        q_values = dqn.q_network(state_tensor)
    return int(torch.argmax(q_values, dim=1).item())


def optimize_dqn_step(
    dqn: DQNComponents,
    batch_size: int,
    gamma: float,
) -> float | None:
    if len(dqn.replay_buffer) < batch_size:
        return None

    states, actions, rewards, next_states, dones = dqn.replay_buffer.sample(batch_size)

    q_values = dqn.q_network(states).gather(1, actions)
    with torch.no_grad():
        next_q_values = dqn.target_network(next_states).max(dim=1, keepdim=True)[0]
        targets = rewards + gamma * (1.0 - dones) * next_q_values

    loss = dqn.loss_fn(q_values, targets)
    dqn.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(dqn.q_network.parameters(), max_norm=10.0)
    dqn.optimizer.step()

    return float(loss.item())


def soft_update_target_network(
    q_network: QNetwork,
    target_network: QNetwork,
    tau: float,
) -> None:
    for target_param, source_param in zip(target_network.parameters(), q_network.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


def train_dqn(
    env: gym.Env,
    dqn: DQNComponents,
    train_episodes: int = 300,
    max_steps: int = 200,
    batch_size: int = 64,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
    tau: float = 0.005,
    seed: int = 42,
    log_interval: int = 10,
) -> TrainingHistory:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    epsilon = epsilon_start
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    episode_losses: list[float] = []
    epsilons: list[float] = []

    for episode in range(train_episodes):
        state, _ = env.reset(seed=seed + episode)
        total_reward = 0.0
        total_steps = 0
        losses_this_episode: list[float] = []

        done = False
        truncated = False

        while not (done or truncated) and total_steps < max_steps:
            action = select_action(state, dqn, epsilon, env)
            next_state, reward, done, truncated, _ = env.step(action)
            terminal = done or truncated

            dqn.replay_buffer.add(state, action, reward, next_state, terminal)
            loss = optimize_dqn_step(dqn, batch_size=batch_size, gamma=gamma)
            if loss is not None:
                losses_this_episode.append(loss)
                soft_update_target_network(dqn.q_network, dqn.target_network, tau=tau)

            state = next_state
            total_reward += reward
            total_steps += 1
            epsilon = max(epsilon_end, epsilon * epsilon_decay)

        episode_rewards.append(float(total_reward))
        episode_lengths.append(total_steps)
        episode_losses.append(float(np.mean(losses_this_episode)) if losses_this_episode else 0.0)
        epsilons.append(float(epsilon))

        if (episode + 1) % log_interval == 0:
            recent = episode_rewards[-log_interval:]
            print(
                f"Episode {episode + 1:04d}/{train_episodes} | "
                f"AvgReward({log_interval}): {np.mean(recent):8.3f} | "
                f"Len: {episode_lengths[-1]:3d} | Epsilon: {epsilon:.4f}"
            )

    return TrainingHistory(
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        episode_losses=episode_losses,
        epsilons=epsilons,
    )


def rolling_mean(values: list[float], window: int = 20) -> np.ndarray:
    if len(values) == 0:
        return np.array([])
    arr = np.array(values, dtype=np.float32)
    if len(arr) < window:
        return np.full_like(arr, arr.mean())
    kernel = np.ones(window, dtype=np.float32) / window
    valid = np.convolve(arr, kernel, mode="valid")
    pad = np.full(window - 1, valid[0], dtype=np.float32)
    return np.concatenate([pad, valid])


def plot_training_metrics(
    history: TrainingHistory,
    output_path: str = "artifacts/training_metrics.png",
    smoothing_window: int = 20,
) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    episodes = np.arange(1, len(history.episode_rewards) + 1)

    reward_smooth = rolling_mean(history.episode_rewards, window=smoothing_window)
    length_smooth = rolling_mean([float(v) for v in history.episode_lengths], window=smoothing_window)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(episodes, history.episode_rewards, alpha=0.35, label="Episode reward")
    axes[0].plot(episodes, reward_smooth, linewidth=2, label=f"Reward MA({smoothing_window})")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("DQN Training Metrics")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.3)

    axes[1].plot(episodes, history.episode_lengths, alpha=0.35, label="Episode length")
    axes[1].plot(episodes, length_smooth, linewidth=2, label=f"Length MA({smoothing_window})")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Length")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close(fig)

    print(f"Saved training plots to: {output_file}")


def save_model_checkpoint(
    dqn: DQNComponents,
    model_path: str,
    history: TrainingHistory,
    gamma: float,
) -> None:
    output_file = Path(model_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "q_network_state_dict": dqn.q_network.state_dict(),
        "target_network_state_dict": dqn.target_network.state_dict(),
        "optimizer_state_dict": dqn.optimizer.state_dict(),
        "obs_dim": dqn.obs_dim,
        "n_actions": dqn.n_actions,
        "gamma": gamma,
        "episode_rewards": history.episode_rewards,
        "episode_lengths": history.episode_lengths,
        "episode_losses": history.episode_losses,
        "epsilons": history.epsilons,
    }
    torch.save(checkpoint, output_file)
    print(f"Saved model checkpoint to: {output_file}")


def configure_highway_env(render_mode: str | None = None) -> gym.Env:
    """Create and configure the highway-v0 environment for kinematics observations."""
    env = gym.make("highway-v0", render_mode=render_mode)

    # Keep this aligned with project requirements: kinematics + 5 discrete meta-actions.
    env.unwrapped.configure(
        {
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 8,
                "features": ["x", "y", "vx", "vy"],
                "normalize": True,
                "absolute": False,
                "order": "sorted",
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "policy_frequency": 5,
            "duration": 40,
        }
    )
    env.reset(seed=42)
    return env


def print_spaces(env: gym.Env) -> None:
    print("=== Environment Spaces ===")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    action_type = getattr(env.unwrapped, "action_type", None)
    if action_type is not None and hasattr(action_type, "actions"):
        print("Available discrete meta-actions:")
        for idx, action_name in action_type.actions.items():
            print(f"  {idx}: {action_name}")


def run_random_baseline(
    env: gym.Env,
    episodes: int = 10,
    max_steps: int = 200,
    seed: int = 42,
) -> BaselineStats:
    """Run a random-action policy to establish a baseline."""
    random.seed(seed)
    np.random.seed(seed)

    episode_rewards = []
    episode_steps = []
    collisions = 0

    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode)
        del obs, info

        total_reward = 0.0
        done = False
        truncated = False
        step_count = 0

        while not (done or truncated) and step_count < max_steps:
            action = env.action_space.sample()
            _, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            if info.get("crashed", False):
                collisions += 1
                break

        episode_rewards.append(total_reward)
        episode_steps.append(step_count)
        print(
            f"Episode {episode + 1:02d} | Reward: {total_reward:8.3f} | "
            f"Steps: {step_count:3d} | Crashed: {info.get('crashed', False)}"
        )

    rewards = np.array(episode_rewards, dtype=np.float32)
    steps = np.array(episode_steps, dtype=np.float32)

    return BaselineStats(
        mean_reward=float(rewards.mean()),
        std_reward=float(rewards.std()),
        mean_steps=float(steps.mean()),
        collision_rate=float(collisions / max(episodes, 1)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DQN training for highway-v0 with epsilon-greedy and soft target updates"
    )
    parser.add_argument("--episodes", type=int, default=300, help="Training episodes")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Max steps per episode",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for replay sampling")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--buffer-size", type=int, default=100_000, help="Replay buffer capacity")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon-end", type=float, default=0.05, help="Final epsilon floor")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="Per-step epsilon decay")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update factor")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N episodes")
    parser.add_argument(
        "--plot-path",
        type=str,
        default="artifacts/training_metrics.png",
        help="Output path for training plot image",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="artifacts/dqn_model.pt",
        help="Output path for saved model checkpoint",
    )
    parser.add_argument(
        "--run-baseline",
        action="store_true",
        help="Run random-action baseline before training",
    )
    parser.add_argument(
        "--baseline-episodes",
        type=int,
        default=10,
        help="Episodes for random baseline if enabled",
    )
    args = parser.parse_args()

    env = configure_highway_env(render_mode=None)
    try:
        print_spaces(env)
        dqn = initialize_dqn_components(
            env,
            buffer_capacity=args.buffer_size,
            learning_rate=args.lr,
        )
        print("\n=== DQN Initialization ===")
        print(f"Flattened observation dim: {dqn.obs_dim}")
        print(f"Number of actions        : {dqn.n_actions}")
        print("Q-network hidden layers  : [256, 256]")
        print(f"Optimizer                : Adam (lr={args.lr})")
        print("Loss function            : SmoothL1Loss")
        print(f"Device                   : {dqn.device}")

        if args.run_baseline:
            stats = run_random_baseline(
                env,
                episodes=args.baseline_episodes,
                max_steps=args.max_steps,
                seed=args.seed,
            )

            print("\n=== Random Baseline Summary ===")
            print(f"Mean reward      : {stats.mean_reward:.3f}")
            print(f"Reward std       : {stats.std_reward:.3f}")
            print(f"Mean episode step: {stats.mean_steps:.2f}")
            print(f"Collision rate   : {stats.collision_rate * 100:.1f}%")

        print("\n=== Starting DQN Training ===")
        history = train_dqn(
            env,
            dqn,
            train_episodes=args.episodes,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            tau=args.tau,
            seed=args.seed,
            log_interval=args.log_interval,
        )

        plot_training_metrics(history, output_path=args.plot_path)
        save_model_checkpoint(dqn, model_path=args.model_path, history=history, gamma=args.gamma)

        print("\n=== Training Summary ===")
        print(f"Episodes trained : {len(history.episode_rewards)}")
        print(f"Final epsilon    : {history.epsilons[-1]:.4f}")
        print(f"Mean reward (all): {np.mean(history.episode_rewards):.3f}")
        print(f"Mean length (all): {np.mean(history.episode_lengths):.2f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
