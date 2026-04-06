import argparse
import time
from dataclasses import dataclass
from pathlib import Path

from .action_space import ACTION_SPACE_SIZE
from .bridge import STS2Bridge, STS2BridgeError
from .env import COMBAT_STATE_TYPES, OBSERVATION_SIZE, STS2MuZeroEnv, extract_observation_features
from .muzero import MuZeroAgent, Transition


def _timestamp() -> str:
    return time.strftime("%H:%M:%S")


def _read_navigation(state: dict[str, object]) -> tuple[bool, str | None]:
    state_type = str(state.get("state_type", "unknown"))
    if state_type == "menu":
        navigation = state.get("navigation") if isinstance(state.get("navigation"), dict) else {}
        return bool(navigation), str(navigation.get("status")) if navigation else None
    if state_type == "game_over":
        game_over = state.get("game_over") if isinstance(state.get("game_over"), dict) else {}
        can_start = bool(game_over.get("can_start_run", True))
        return not can_start, str(game_over.get("navigation_status")) if game_over else None
    return False, None


def _wait_for_run(
    env: STS2MuZeroEnv,
    sleep_seconds: float,
    auto_character: str,
    ascension: int,
    manual_start: bool,
) -> dict[str, object]:
    last_message = ""
    last_start_attempt = 0.0
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
        state_type = str(state.get("state_type", "unknown"))
        if state_type not in {"menu", "game_over"}:
            return state

        if manual_start:
            message = f"[{_timestamp()}] state={state_type} waiting for manual run start."
            if message != last_message:
                print(message, flush=True)
                last_message = message
            time.sleep(sleep_seconds)
            continue

        navigating, navigation_status = _read_navigation(state)
        now = time.monotonic()
        if not navigating and now - last_start_attempt >= max(2.0, sleep_seconds):
            try:
                response = env.bridge.start_run(
                    character=auto_character,
                    ascension=None if ascension < 0 else ascension,
                )
                message = (
                    f"[{_timestamp()}] auto-start requested: character={auto_character} "
                    f"ascension={'keep' if ascension < 0 else ascension} "
                    f"status={response.get('status', 'unknown')} message={response.get('message', '')}"
                )
                last_start_attempt = now
            except STS2BridgeError as exc:
                message = f"[{_timestamp()}] auto-start request failed: {exc}"
                last_start_attempt = now
        else:
            message = f"[{_timestamp()}] waiting for auto-start: state={state_type} status={navigation_status or 'pending'}"

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


@dataclass
class CombatWindow:
    act: int
    floor: int
    start_step: int
    steps: int = 0
    reward: float = 0.0


@dataclass
class ActWindow:
    act: int
    start_floor: int
    start_step: int
    steps: int = 0
    reward: float = 0.0
    combats: int = 0


def _run_position(state: dict[str, object]) -> tuple[int, int]:
    run = state.get("run") if isinstance(state.get("run"), dict) else {}
    return int(run.get("act", 0) or 0), int(run.get("floor", 0) or 0)


def _is_combat_state(state: dict[str, object]) -> bool:
    return str(state.get("state_type", "unknown")) in COMBAT_STATE_TYPES


def _open_combat_window(state: dict[str, object], next_step: int) -> CombatWindow | None:
    if not _is_combat_state(state):
        return None
    act, floor = _run_position(state)
    return CombatWindow(act=act, floor=floor, start_step=next_step)


def _open_act_window(state: dict[str, object], next_step: int) -> ActWindow:
    act, floor = _run_position(state)
    return ActWindow(act=act, start_floor=floor, start_step=next_step)


def _format_boundary_suffix(transition: Transition) -> str:
    boundaries: list[str] = []
    if transition.combat_end:
        boundaries.append("combat_end")
    if transition.act_end:
        boundaries.append("act_end")
    if transition.run_end:
        boundaries.append("run_end")
    return f" boundary={','.join(boundaries)}" if boundaries else ""


def _format_info_suffix(info: dict[str, object]) -> str:
    response = info.get("response")
    if not isinstance(response, dict) or response.get("status") != "error":
        return ""
    details: list[str] = []
    workaround = response.get("compatibility_workaround")
    if isinstance(workaround, str):
        details.append(f"workaround={workaround}")
    error_text = str(response.get("error", "")).replace("\n", " ").strip()
    if error_text:
        details.append(f"error={error_text}")
    return f" {' '.join(details)}" if details else ""


def _combat_result(next_state: dict[str, object], run_end: bool) -> str:
    player = next_state.get("player") if isinstance(next_state.get("player"), dict) else {}
    hp = float(player.get("hp", 0) or 0)
    if run_end:
        return "victory" if hp > 0 else "defeat"
    return "victory"


def _format_replay_counts(counts: dict[str, int]) -> str:
    return (
        f"replay={counts.get('total', 0)} "
        f"combat={counts.get('combat_end', 0)} "
        f"act={counts.get('act_end', 0)} "
        f"run={counts.get('run_end', 0)}"
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
    parser.add_argument("--character", default="Ironclad", help="Character to auto-start after menu/game_over")
    parser.add_argument("--ascension", type=int, default=-1, help="-1 keeps the game's current ascension selection")
    parser.add_argument("--manual-start", action="store_true", help="Disable automatic run start/restart")
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
        migrations = agent.load(checkpoint_path)
        migration_suffix = f" migrated={' ; '.join(migrations)}" if migrations else ""
        print(f"[{_timestamp()}] loaded checkpoint from {checkpoint_path}{migration_suffix}", flush=True)

    print(
        f"[{_timestamp()}] obs={OBSERVATION_SIZE} actions={ACTION_SPACE_SIZE} "
        f"hidden={args.hidden_size} simulations={args.simulations} "
        f"character={args.character} ascension={'keep' if args.ascension < 0 else args.ascension} "
        f"replay_bonus=+{agent.combat_end_sample_bonus:.1f}/+{agent.act_end_sample_bonus:.1f}/+{agent.run_end_sample_bonus:.1f}",
        flush=True,
    )

    state = _wait_for_run(env, args.menu_sleep, args.character, args.ascension, args.manual_start)
    episode_index = 1
    episode_steps = 0
    episode_reward = 0.0
    episode_combats = 0
    total_steps = 0
    act_window = _open_act_window(state, total_steps + 1)
    combat_window = _open_combat_window(state, total_steps + 1)

    while args.max_steps <= 0 or total_steps < args.max_steps:
        if combat_window is None and _is_combat_state(state):
            combat_window = _open_combat_window(state, total_steps + 1)
        legal_action_map = env.build_legal_action_map(state)
        legal_actions = sorted(legal_action_map)
        if not legal_actions:
            print(f"[{_timestamp()}] step={total_steps} state={state.get('state_type')} no legal actions; polling", flush=True)
            time.sleep(args.menu_sleep)
            state = (
                _wait_for_run(env, args.menu_sleep, args.character, args.ascension, args.manual_start)
                if str(state.get("state_type")) in {"menu", "game_over"}
                else env.fetch_state()
            )
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
            combat_end=bool(info.get("combat_end")),
            act_end=bool(info.get("act_end")),
            run_end=bool(info.get("run_end")),
        )
        agent.remember(transition)
        metrics = agent.learn(args.updates_per_step)

        total_steps += 1
        episode_steps += 1
        episode_reward += reward
        act_window.steps += 1
        act_window.reward += reward
        if combat_window is not None:
            combat_window.steps += 1
            combat_window.reward += reward

        player = next_state.get("player") if isinstance(next_state.get("player"), dict) else {}
        run = next_state.get("run") if isinstance(next_state.get("run"), dict) else {}
        print(
            f"[{_timestamp()}] step={total_steps} ep={episode_index} act={run.get('act', '?')} floor={run.get('floor', '?')} "
            f"hp={player.get('hp', '?')}/{player.get('max_hp', '?')} state={next_state.get('state_type')} "
            f"action={info['description']} reward={reward:+.3f} ep_reward={episode_reward:+.3f} "
            f"root_value={search.root_value:+.3f} {_format_metrics(metrics)}"
            f"{_format_boundary_suffix(transition)}{_format_info_suffix(info)}",
            flush=True,
        )

        if args.checkpoint_interval > 0 and total_steps % args.checkpoint_interval == 0:
            agent.save(checkpoint_path)
            print(f"[{_timestamp()}] checkpoint saved to {checkpoint_path}", flush=True)

        if transition.combat_end and combat_window is not None:
            episode_combats += 1
            act_window.combats += 1
            print(
                f"[{_timestamp()}] combat_end ep={episode_index} act={combat_window.act} floor={combat_window.floor} "
                f"steps={combat_window.steps} reward={combat_window.reward:+.3f} "
                f"result={_combat_result(next_state, transition.run_end)}",
                flush=True,
            )
            combat_window = None

        if transition.act_end:
            print(
                f"[{_timestamp()}] act_summary ep={episode_index} act={act_window.act} reason=act_end "
                f"steps={act_window.steps} floors={act_window.start_floor}->{info.get('next_floor', run.get('floor', '?'))} "
                f"combats={act_window.combats} reward={act_window.reward:+.3f}",
                flush=True,
            )
            act_window = _open_act_window(next_state, total_steps + 1) if not transition.run_end else None

        if done:
            if act_window is not None:
                print(
                    f"[{_timestamp()}] act_summary ep={episode_index} act={act_window.act} reason=run_end "
                    f"steps={act_window.steps} floors={act_window.start_floor}->{run.get('floor', '?')} "
                    f"combats={act_window.combats} reward={act_window.reward:+.3f}",
                    flush=True,
                )
            replay_counts = agent.replay_boundary_counts()
            result = "victory" if float(player.get("hp", 0) or 0) > 0 else "defeat"
            print(
                f"[{_timestamp()}] run_end ep={episode_index} steps={episode_steps} combats={episode_combats} "
                f"shaped_return={episode_reward:+.3f} result={result} {_format_replay_counts(replay_counts)}",
                flush=True,
            )
            agent.save(checkpoint_path)
            episode_index += 1
            episode_steps = 0
            episode_reward = 0.0
            episode_combats = 0
            state = _wait_for_run(env, args.menu_sleep, args.character, args.ascension, args.manual_start)
            act_window = _open_act_window(state, total_steps + 1)
            combat_window = _open_combat_window(state, total_steps + 1)
            continue

        state = next_state

    agent.save(checkpoint_path)
    print(f"[{_timestamp()}] final checkpoint saved to {checkpoint_path}", flush=True)
