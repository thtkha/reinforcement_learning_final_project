import argparse
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import highway_env  # noqa: F401 - required to register highway environments
import numpy as np
import torch

from train import QNetwork, configure_highway_env


def load_q_network(model_path: Path, obs_dim: int, n_actions: int, device: torch.device) -> QNetwork:
    """Load a trained Q-network checkpoint from disk."""
    q_network = QNetwork(input_dim=obs_dim, num_actions=n_actions).to(device)

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "q_network_state_dict" in checkpoint:
        state_dict = checkpoint["q_network_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError("Unsupported checkpoint format. Expected a state_dict or dict with q_network_state_dict.")

    q_network.load_state_dict(state_dict)
    q_network.eval()
    return q_network


def greedy_action(state: np.ndarray, q_network: QNetwork, device: torch.device) -> int:
    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q_values = q_network(state_tensor)
    return int(torch.argmax(q_values, dim=1).item())


def evaluate(
    model_path: Path,
    episodes: int,
    max_steps: int,
    render_mode: str,
    seed: int,
    video_folder: Path,
) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    env = configure_highway_env(render_mode=render_mode)

    if render_mode == "rgb_array":
        video_folder.mkdir(parents=True, exist_ok=True)
        env = RecordVideo(
            env,
            video_folder=str(video_folder),
            episode_trigger=lambda episode_id: True,
            name_prefix="highway_dqn_eval",
        )

    if env.observation_space.shape is None:
        raise ValueError("Observation space shape is undefined.")

    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_network = load_q_network(model_path, obs_dim=obs_dim, n_actions=n_actions, device=device)

    print("=== Evaluation Setup ===")
    print(f"Model path      : {model_path}")
    print(f"Render mode     : {render_mode}")
    print(f"Episodes        : {episodes}")
    print(f"Max steps       : {max_steps}")
    print(f"Device          : {device}")
    print("Policy          : Greedy (epsilon = 0.0)")

    rewards = []
    lengths = []
    crash_count = 0

    try:
        for ep in range(episodes):
            state, _ = env.reset(seed=seed + ep)
            done = False
            truncated = False
            total_reward = 0.0
            steps = 0
            crashed = False

            while not (done or truncated) and steps < max_steps:
                action = greedy_action(state, q_network, device)
                next_state, reward, done, truncated, info = env.step(action)
                state = next_state
                total_reward += reward
                steps += 1
                crashed = bool(info.get("crashed", False))

            rewards.append(total_reward)
            lengths.append(steps)
            crash_count += int(crashed)

            print(
                f"Episode {ep + 1:02d} | Reward: {total_reward:8.3f} | "
                f"Steps: {steps:3d} | Crashed: {crashed}"
            )

    finally:
        env.close()

    print("\n=== Evaluation Summary ===")
    print(f"Mean reward      : {float(np.mean(rewards)):.3f}")
    print(f"Std reward       : {float(np.std(rewards)):.3f}")
    print(f"Mean episode step: {float(np.mean(lengths)):.2f}")
    print(f"Collision rate   : {100.0 * crash_count / max(episodes, 1):.1f}%")

    if render_mode == "rgb_array":
        print(f"Saved evaluation videos to: {video_folder}")
    else:
        print("Human rendering completed. You can use screen recording for your submission video.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained DQN on highway-v0 with greedy policy")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts/dqn_model.pt"),
        help="Path to saved model checkpoint",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=200, help="Maximum steps per episode")
    parser.add_argument(
        "--render-mode",
        type=str,
        default="human",
        choices=["human", "rgb_array"],
        help="Render mode for evaluation",
    )
    parser.add_argument(
        "--video-folder",
        type=Path,
        default=Path("artifacts/eval_videos"),
        help="Folder to save videos when using rgb_array render mode",
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        episodes=args.episodes,
        max_steps=args.max_steps,
        render_mode=args.render_mode,
        seed=args.seed,
        video_folder=args.video_folder,
    )


if __name__ == "__main__":
    main()
