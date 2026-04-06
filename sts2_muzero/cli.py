import argparse
import time
from pathlib import Path

from .action_space import ACTION_SPACE_SIZE
from .bridge import STS2Bridge, STS2BridgeError
from .env import OBSERVATION_SIZE, STS2MuZeroEnv, extract_observation_features
from .muzero import MuZeroAgent, Transition


def _timestamp() -> str:
    return time.strftime("%H:%M:%S")


def _wait_for_run(env: STS2MuZeroEnv, sleep_seconds: float) -> dict[str, object]:
    last_message = ""
    while True:
        try:
            state = env.fetch_state()
        except STS2BridgeError as exc:
            message = f"[{_timestamp()}] waiting for STS2MCP bridge: {exc}"
            if message != last_message:
                print(message, flush=True)
                last_message = message
            time.sleep(sleep_seconds)
            continue
        if str(state.get("state_type", "unknown")) != "menu":
            return state
        message = f"[{_timestamp()}] no active run in STS2; start or continue a singleplayer run."
        if message != last_message:
            print(message, flush=True)
            last_message = message
        time.sleep(sleep_seconds)


def _format_metrics(metrics: object) -> str:
    if metrics is None:
        return "loss=warmup"
    return (
        f"loss={metrics.total_loss:.4f} "
        f"value={metrics.value_loss:.4f} "
        f"reward={metrics.reward_loss:.4f} "
        f"policy={metrics.policy_loss:.4f} "
        f"consistency={metrics.consistency_loss:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Shell-first MuZero prototype for Slay the Spire 2")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=15526)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means run forever")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--discount", type=float, default=0.997)
    parser.add_argument("--simulations", type=int, default=24)
    parser.add_argument("--updates-per-step", type=int, default=2)
    parser.add_argument("--replay-capacity", type=int, default=4096)
    parser.add_argument("--warmup-samples", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--poll-interval", type=float, default=0.15)
    parser.add_argument("--menu-sleep", type=float, default=2.0)
    parser.add_argument("--checkpoint-path", default="checkpoints/latest.json")
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-root-noise", action="store_true")
    args = parser.parse_args()

    env = STS2MuZeroEnv(STS2Bridge(args.host, args.port, args.timeout), poll_interval=args.poll_interval)
    agent = MuZeroAgent(
        observation_size=OBSERVATION_SIZE,
        action_size=ACTION_SPACE_SIZE,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        discount=args.discount,
        simulations=args.simulations,
        replay_capacity=args.replay_capacity,
        warmup_samples=args.warmup_samples,
        seed=args.seed,
    )
    checkpoint_path = Path(args.checkpoint_path)
    if args.resume and checkpoint_path.exists():
        agent.load(checkpoint_path)
        print(f"[{_timestamp()}] loaded checkpoint from {checkpoint_path}", flush=True)

    print(
        f"[{_timestamp()}] obs={OBSERVATION_SIZE} actions={ACTION_SPACE_SIZE} "
        f"hidden={args.hidden_size} simulations={args.simulations}",
        flush=True,
    )

    state = _wait_for_run(env, args.menu_sleep)
    episode_index = 1
    episode_steps = 0
    episode_reward = 0.0
    total_steps = 0

    while args.max_steps <= 0 or total_steps < args.max_steps:
        legal_action_map = env.build_legal_action_map(state)
        legal_actions = sorted(legal_action_map)
        if not legal_actions:
            print(f"[{_timestamp()}] step={total_steps} state={state.get('state_type')} no legal actions; polling", flush=True)
            time.sleep(args.menu_sleep)
            state = _wait_for_run(env, args.menu_sleep) if str(state.get("state_type")) == "menu" else env.fetch_state()
            continue

        observation = extract_observation_features(state)
        search = agent.plan(observation, legal_actions, use_exploration_noise=not args.no_root_noise)
        action_id = agent.select_action(search, legal_actions, temperature=args.temperature)
        next_state, reward, done, info = env.step(action_id, state)
        transition = Transition(
            observation=observation,
            action_index=action_id,
            reward=reward,
            next_observation=extract_observation_features(next_state),
            done=done,
            policy_target=search.action_probabilities,
        )
        agent.remember(transition)
        metrics = agent.learn(args.updates_per_step)

        total_steps += 1
        episode_steps += 1
        episode_reward += reward

        player = next_state.get("player") if isinstance(next_state.get("player"), dict) else {}
        run = next_state.get("run") if isinstance(next_state.get("run"), dict) else {}
        print(
            f"[{_timestamp()}] step={total_steps} ep={episode_index} act={run.get('act', '?')} floor={run.get('floor', '?')} "
            f"hp={player.get('hp', '?')}/{player.get('max_hp', '?')} state={next_state.get('state_type')} "
            f"action={info['description']} reward={reward:+.3f} ep_reward={episode_reward:+.3f} "
            f"root_value={search.root_value:+.3f} {_format_metrics(metrics)}",
            flush=True,
        )

        if args.checkpoint_interval > 0 and total_steps % args.checkpoint_interval == 0:
            agent.save(checkpoint_path)
            print(f"[{_timestamp()}] checkpoint saved to {checkpoint_path}", flush=True)

        if done:
            print(
                f"[{_timestamp()}] episode={episode_index} finished after {episode_steps} steps "
                f"with shaped_return={episode_reward:+.3f}",
                flush=True,
            )
            agent.save(checkpoint_path)
            episode_index += 1
            episode_steps = 0
            episode_reward = 0.0
            state = _wait_for_run(env, args.menu_sleep)
            continue

        state = next_state

    agent.save(checkpoint_path)
    print(f"[{_timestamp()}] final checkpoint saved to {checkpoint_path}", flush=True)
