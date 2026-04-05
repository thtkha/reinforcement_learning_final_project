import argparse
import random
from dataclasses import dataclass

import gymnasium as gym
import highway_env  # noqa: F401 - required to register highway environments
import numpy as np


@dataclass
class BaselineStats:
    mean_reward: float
    std_reward: float
    mean_steps: float
    collision_rate: float


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
        description="Step 1 setup: inspect highway-v0 spaces and run random baseline"
    )
    parser.add_argument("--episodes", type=int, default=10, help="Random baseline episodes")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Max steps per episode for random baseline",
    )
    args = parser.parse_args()

    env = configure_highway_env(render_mode=None)
    try:
        print_spaces(env)
        stats = run_random_baseline(
            env,
            episodes=args.episodes,
            max_steps=args.max_steps,
            seed=42,
        )

        print("\n=== Random Baseline Summary ===")
        print(f"Mean reward      : {stats.mean_reward:.3f}")
        print(f"Reward std       : {stats.std_reward:.3f}")
        print(f"Mean episode step: {stats.mean_steps:.2f}")
        print(f"Collision rate   : {stats.collision_rate * 100:.1f}%")
    finally:
        env.close()


if __name__ == "__main__":
    main()
