import argparse
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

from .action_space import ACTION_SPACE_SIZE
from .bridge import STS2Bridge, STS2BridgeError
from .env import (
    BASE_OBSERVATION_SIZE,
    COMBAT_STATE_TYPES,
    OBSERVATION_SIZE,
    RewardWeights,
    SEMANTIC_CONCEPT_SIZE,
    SEMANTIC_HISTORY_SIZE,
    SEMANTIC_OBSERVATION_SIZE,
    SEMANTIC_RELATION_SIZE,
    SEMANTIC_SCALAR_SIZE,
    STS2MuZeroEnv,
)
from .muzero import MuZeroAgent, Transition


def _timestamp() -> str:
    return time.strftime("%H:%M:%S")


def _clip_value(value: float, bound: float) -> float:
    if bound <= 0.0:
        return value
    if value > bound:
        return bound
    if value < -bound:
        return -bound
    return value


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
    navigation_sleep = min(sleep_seconds, 0.5)
    retry_interval = max(0.75, navigation_sleep)
    while True:
        try:
            state = env.fetch_state()
        except STS2BridgeError as exc:
            message = f"[{_timestamp()}] waiting for STS2MCP bridge: {exc}"
            if message != last_message:
                print(message, flush=True)
                last_message = message
            time.sleep(navigation_sleep)
            continue
        state_type = str(state.get("state_type", "unknown"))
        if state_type not in {"menu", "game_over"}:
            return state

        if manual_start:
            message = f"[{_timestamp()}] state={state_type} waiting for manual run start."
            if message != last_message:
                print(message, flush=True)
                last_message = message
            time.sleep(navigation_sleep)
            continue

        navigating, navigation_status = _read_navigation(state)
        now = time.monotonic()
        if not navigating and now - last_start_attempt >= retry_interval:
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
        time.sleep(navigation_sleep)


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
class SettlementConfig:
    signal_mode: str = "mean"
    normalization: str = "sum"
    clip: float = 10.0
    turn_weight: float = 1.0
    turn_clean_cap: float = 0.15
    turn_skip_penalty: float = 0.0
    combat_weight: float = 1.5
    act_weight: float = 0.5
    run_weight: float = 0.25
    turn_decay: float = 0.90
    combat_decay: float = 0.97
    act_decay: float = 0.985
    run_decay: float = 0.995


@dataclass
class TurnWindow:
    act: int
    floor: int
    round: int
    start_step: int
    steps: int = 0
    reward: float = 0.0
    settlement_reward: float = 0.0
    skip_unspent_count: int = 0
    skip_unspent_energy: int = 0
    history: list[Transition] = field(default_factory=list)


@dataclass
class CombatWindow:
    act: int
    floor: int
    start_step: int
    steps: int = 0
    reward: float = 0.0
    history: list[Transition] = field(default_factory=list)


@dataclass
class ActWindow:
    act: int
    start_floor: int
    start_step: int
    steps: int = 0
    reward: float = 0.0
    combats: int = 0
    history: list[Transition] = field(default_factory=list)


def _run_position(state: dict[str, object]) -> tuple[int, int]:
    run = state.get("run") if isinstance(state.get("run"), dict) else {}
    return int(run.get("act", 0) or 0), int(run.get("floor", 0) or 0)


def _is_combat_state(state: dict[str, object]) -> bool:
    return str(state.get("state_type", "unknown")) in COMBAT_STATE_TYPES


def _is_player_combat_turn(state: dict[str, object]) -> bool:
    if not _is_combat_state(state):
        return False
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    return str(battle.get("turn", "")) == "player"


def _combat_round(state: dict[str, object]) -> int:
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    return int(battle.get("round", 0) or 0)


def _open_turn_window(state: dict[str, object], next_step: int) -> TurnWindow | None:
    if not _is_player_combat_turn(state):
        return None
    act, floor = _run_position(state)
    return TurnWindow(act=act, floor=floor, round=_combat_round(state), start_step=next_step)


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
    if transition.turn_end:
        boundaries.append("turn_end")
    if transition.combat_end:
        boundaries.append("combat_end")
    if transition.act_end:
        boundaries.append("act_end")
    if transition.run_end:
        boundaries.append("run_end")
    return f" boundary={','.join(boundaries)}" if boundaries else ""


def _format_info_suffix(info: dict[str, object]) -> str:
    details: list[str] = []
    reward_breakdown = info.get("reward_breakdown")
    if isinstance(reward_breakdown, dict):
        lost_energy = int(float(reward_breakdown.get("turn_skip_unspent_energy", 0.0) or 0.0))
        if lost_energy > 0:
            details.append(f"lost_energy={lost_energy}")
    response = info.get("response")
    if isinstance(response, dict) and response.get("status") == "error":
        workaround = response.get("compatibility_workaround")
        if isinstance(workaround, str):
            details.append(f"workaround={workaround}")
        error_text = str(response.get("error", "")).replace("\n", " ").strip()
        if error_text:
            details.append(f"error={error_text}")
    return f" {' '.join(details)}" if details else ""


def _format_replay_counts(counts: dict[str, int]) -> str:
    return (
        f"replay={counts.get('total', 0)} "
        f"turn={counts.get('turn_end', 0)} "
        f"combat={counts.get('combat_end', 0)} "
        f"act={counts.get('act_end', 0)} "
        f"run={counts.get('run_end', 0)}"
    )


def _combat_result(next_state: dict[str, object], run_end: bool) -> str:
    player = next_state.get("player") if isinstance(next_state.get("player"), dict) else {}
    hp = float(player.get("hp", 0) or 0)
    if run_end:
        return "victory" if hp > 0 else "defeat"
    return "victory"


def _settlement_signal(total_reward: float, steps: int, mode: str) -> float:
    bounded_steps = max(1, steps)
    if mode == "sum":
        return total_reward
    if mode == "sqrt":
        return total_reward / math.sqrt(bounded_steps)
    return total_reward / bounded_steps


def _apply_credit_settlement(history: list[Transition], total_bonus: float, decay: float, normalization: str) -> float:
    if not history or total_bonus == 0.0:
        return 0.0
    bounded_decay = min(max(decay, 0.0), 1.0)
    weights = [bounded_decay ** (len(history) - 1 - index) for index in range(len(history))]
    normalizer = sum(weights) if normalization == "sum" else 1.0
    if normalizer <= 0.0:
        normalizer = 1.0
    applied = 0.0
    for transition, weight in zip(history, weights):
        delta = total_bonus * (weight / normalizer)
        transition.reward += delta
        transition.credit_adjustment += delta
        applied += delta
    return applied


def _apply_window_settlement(
    history: list[Transition],
    total_reward: float,
    steps: int,
    weight: float,
    decay: float,
    config: SettlementConfig,
) -> float:
    if not history or weight == 0.0 or total_reward == 0.0:
        return 0.0
    signal = _settlement_signal(total_reward, steps, config.signal_mode)
    total_bonus = _clip_value(weight * signal, config.clip)
    return _apply_credit_settlement(history, total_bonus, decay, config.normalization)


def _apply_turn_settlement(
    window: TurnWindow,
    config: SettlementConfig,
) -> float:
    if not window.history or config.turn_weight == 0.0:
        return 0.0
    signal = _settlement_signal(window.settlement_reward, window.steps, config.signal_mode)
    signal = min(signal, config.turn_clean_cap)
    total_bonus = _clip_value(config.turn_weight * signal, config.clip)
    return _apply_credit_settlement(window.history, total_bonus, config.turn_decay, config.normalization)


def _reward_display(transition: Transition) -> str:
    if abs(transition.credit_adjustment) <= 1e-9:
        return f"reward={transition.reward:+.3f}"
    return (
        f"reward={transition.reward:+.3f} "
        f"raw={transition.raw_reward:+.3f} "
        f"settle={transition.credit_adjustment:+.3f}"
    )


def _global_positive_reward_reference(args: argparse.Namespace, settlement_config: SettlementConfig) -> float:
    candidates = [
        max(0.0, args.floor_advance_weight),
        max(0.0, args.act_advance_weight),
        max(0.0, args.hp_delta_weight),
        max(0.0, args.gold_delta_weight),
        max(0.0, args.enemy_hp_delta_weight),
        max(0.0, args.turn_end_weight),
        max(0.0, args.combat_end_weight),
        max(0.0, args.act_end_weight),
        max(0.0, args.run_victory_weight),
        max(0.0, args.combat_tactical_shaping),
        max(0.0, settlement_config.turn_weight * settlement_config.turn_clean_cap),
    ]
    if settlement_config.combat_weight > 0.0:
        candidates.append(settlement_config.clip)
    if settlement_config.act_weight > 0.0:
        candidates.append(settlement_config.clip)
    if settlement_config.run_weight > 0.0:
        candidates.append(settlement_config.clip)
    return max(1.0, max(candidates))


def _resolve_turn_skip_penalty(args: argparse.Namespace, settlement_config: SettlementConfig) -> tuple[float, float]:
    if args.turn_skip_unspent_penalty > 0.0:
        return args.turn_skip_unspent_penalty, 0.0
    positive_reference = _global_positive_reward_reference(args, settlement_config)
    return positive_reference * max(0.0, args.turn_skip_unspent_multiplier), positive_reference


def main() -> None:
    parser = argparse.ArgumentParser(description="Shell-first MuZero prototype for Slay the Spire 2")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=15526)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means run forever")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--discount", type=float, default=0.997)
    parser.add_argument("--simulations", type=int, default=40)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--replay-capacity", type=int, default=12000)
    parser.add_argument("--warmup-samples", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--poll-interval", type=float, default=0.15)
    parser.add_argument("--menu-sleep", type=float, default=2.0)
    parser.add_argument("--error-penalty-weight", type=float, default=1.0)
    parser.add_argument("--floor-advance-weight", type=float, default=5.0)
    parser.add_argument("--act-advance-weight", type=float, default=20.0)
    parser.add_argument("--hp-delta-weight", type=float, default=5.0, help="Weight applied to normalized player HP delta (hp_delta / max_hp)")
    parser.add_argument("--gold-delta-weight", type=float, default=0.5, help="Weight applied to normalized gold delta (gold_delta / gold_delta_scale)")
    parser.add_argument("--gold-delta-scale", type=float, default=25.0, help="Reference scale used to normalize gold delta rewards")
    parser.add_argument("--enemy-hp-delta-weight", type=float, default=8.0, help="Weight applied to normalized enemy HP delta (enemy_hp_delta / active_enemy_max_hp)")
    parser.add_argument("--turn-end-weight", type=float, default=0.0, help="Immediate reward bonus applied when a player turn resolves")
    parser.add_argument("--turn-skip-unspent-penalty", type=float, default=0.0, help="Per-energy penalty override for ending turn with playable cards and unspent energy; <=0 derives from global max positive reward")
    parser.add_argument("--turn-skip-unspent-multiplier", type=float, default=100.0, help="When turn-skip-unspent-penalty <= 0, per-energy skip penalty becomes global_max_positive_reward * this multiplier")
    parser.add_argument("--combat-end-weight", type=float, default=8.0)
    parser.add_argument("--combat-defeat-weight", type=float, default=-8.0, help="Immediate reward applied when combat ends because the player dies")
    parser.add_argument("--act-end-weight", type=float, default=0.0, help="Immediate reward bonus applied on act clear")
    parser.add_argument("--run-victory-weight", type=float, default=10.0)
    parser.add_argument("--run-defeat-weight", type=float, default=-10.0)
    parser.add_argument("--combat-tactical-shaping", type=float, default=0.35, help="Combat-local potential shaping weight")
    parser.add_argument("--end-turn-slack-penalty", type=float, default=0.5, help="Penalty scale for ending turn with playable cards and spare energy")
    parser.add_argument("--settlement-signal-mode", choices=["mean", "sum", "sqrt"], default="mean", help="How to convert a turn/combat/act/run raw return into a settlement signal")
    parser.add_argument("--settlement-normalization", choices=["sum", "none"], default="sum", help="Whether to normalize historical settlement weights to a fixed total bonus")
    parser.add_argument("--settlement-clip", type=float, default=10.0, help="Absolute clip applied to each hierarchy settlement total bonus")
    parser.add_argument("--turn-settlement-weight", type=float, default=1.0, help="Retroactive credit weight for the current turn history")
    parser.add_argument("--turn-settlement-clean-cap", type=float, default=0.15, help="Cap applied to clean turn-end settlement signal so ordinary turn ends only get a minimal bonus")
    parser.add_argument("--turn-settlement-skip-penalty", type=float, default=0.0, help="Deprecated compatibility knob; skip penalty is now assigned only to the end_turn transition")
    parser.add_argument("--combat-settlement-weight", type=float, default=1.5, help="Retroactive credit weight for the current combat history")
    parser.add_argument("--act-settlement-weight", type=float, default=0.5, help="Retroactive credit weight for the current act history")
    parser.add_argument("--run-settlement-weight", type=float, default=0.25, help="Retroactive credit weight for the current run history")
    parser.add_argument("--turn-settlement-decay", type=float, default=0.90)
    parser.add_argument("--combat-settlement-decay", type=float, default=0.97)
    parser.add_argument("--act-settlement-decay", type=float, default=0.985)
    parser.add_argument("--run-settlement-decay", type=float, default=0.995)
    parser.add_argument("--turn-end-sample-bonus", type=float, default=0.0, help="Replay sampling bonus for turn_end transitions")
    parser.add_argument("--combat-end-sample-bonus", type=float, default=1.0, help="Replay sampling bonus for combat_end transitions")
    parser.add_argument("--act-end-sample-bonus", type=float, default=3.0, help="Replay sampling bonus for act_end transitions")
    parser.add_argument("--run-end-sample-bonus", type=float, default=7.0, help="Replay sampling bonus for run_end transitions")
    parser.add_argument("--card-select-candidate-limit", type=int, default=6, help="Top-k shortlist for upgrade/enchant card overlays; 0 disables pruning")
    parser.add_argument("--allow-premature-end-turn", action="store_true", help="Pure training mode: always expose combat_end_turn even when playable combat cards exist")
    parser.add_argument("--end-turn-guard-stall-limit", type=int, default=3, help="Compatibility fallback only: used only when end-turn-guard-timeout-seconds <= 0")
    parser.add_argument("--end-turn-guard-timeout-seconds", type=float, default=3.0, help="When premature end-turn blocking is enabled, keep end turn hidden for this many seconds on the same combat search space before releasing it")
    parser.add_argument("--checkpoint-path", default="checkpoints/latest.json")
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--character", default="Ironclad", help="Character to auto-start after menu/game_over")
    parser.add_argument("--ascension", type=int, default=0, help="-1 keeps the game's current ascension selection")
    parser.add_argument("--manual-start", action="store_true", help="Disable automatic run start/restart")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-root-noise", action="store_true")
    args = parser.parse_args()

    settlement_config = SettlementConfig(
        signal_mode=args.settlement_signal_mode,
        normalization=args.settlement_normalization,
        clip=args.settlement_clip,
        turn_weight=args.turn_settlement_weight,
        turn_clean_cap=args.turn_settlement_clean_cap,
        turn_skip_penalty=args.turn_settlement_skip_penalty,
        combat_weight=args.combat_settlement_weight,
        act_weight=args.act_settlement_weight,
        run_weight=args.run_settlement_weight,
        turn_decay=args.turn_settlement_decay,
        combat_decay=args.combat_settlement_decay,
        act_decay=args.act_settlement_decay,
        run_decay=args.run_settlement_decay,
    )
    resolved_turn_skip_penalty, turn_skip_reference = _resolve_turn_skip_penalty(args, settlement_config)
    reward_weights = RewardWeights(
        error_penalty=args.error_penalty_weight,
        floor_advance=args.floor_advance_weight,
        act_advance=args.act_advance_weight,
        hp_delta=args.hp_delta_weight,
        gold_delta=args.gold_delta_weight,
        gold_delta_scale=args.gold_delta_scale,
        enemy_hp_delta=args.enemy_hp_delta_weight,
        turn_end=args.turn_end_weight,
        turn_skip_unspent_penalty=resolved_turn_skip_penalty,
        combat_end=args.combat_end_weight,
        combat_defeat=args.combat_defeat_weight,
        act_end=args.act_end_weight,
        run_victory=args.run_victory_weight,
        run_defeat=args.run_defeat_weight,
        combat_tactical_shaping=args.combat_tactical_shaping,
        end_turn_slack_penalty=args.end_turn_slack_penalty,
    )

    env = STS2MuZeroEnv(
        STS2Bridge(args.host, args.port, args.timeout),
        poll_interval=args.poll_interval,
        reward_discount=args.discount,
        reward_weights=reward_weights,
        card_select_candidate_limit=args.card_select_candidate_limit,
        block_premature_end_turn=not args.allow_premature_end_turn,
        end_turn_guard_stall_limit=args.end_turn_guard_stall_limit,
        end_turn_guard_timeout_seconds=args.end_turn_guard_timeout_seconds,
    )
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
        turn_end_sample_bonus=args.turn_end_sample_bonus,
        combat_end_sample_bonus=args.combat_end_sample_bonus,
        act_end_sample_bonus=args.act_end_sample_bonus,
        run_end_sample_bonus=args.run_end_sample_bonus,
    )
    checkpoint_path = Path(args.checkpoint_path)
    if args.resume and checkpoint_path.exists():
        migrations = agent.load(checkpoint_path)
        migration_suffix = f" migrated={' ; '.join(migrations)}" if migrations else ""
        print(f"[{_timestamp()}] loaded checkpoint from {checkpoint_path}{migration_suffix}", flush=True)

    print(
        f"[{_timestamp()}] obs={OBSERVATION_SIZE} base_obs={BASE_OBSERVATION_SIZE} "
        f"semantic_obs={SEMANTIC_OBSERVATION_SIZE} "
        f"(concepts={SEMANTIC_CONCEPT_SIZE} scalars={SEMANTIC_SCALAR_SIZE} "
        f"relations={SEMANTIC_RELATION_SIZE} history={SEMANTIC_HISTORY_SIZE}) "
        f"actions={ACTION_SPACE_SIZE} "
        f"hidden={args.hidden_size} lr={args.learning_rate:.5g} simulations={args.simulations} "
        f"updates={args.updates_per_step} replay={args.replay_capacity} warmup={args.warmup_samples} temp={args.temperature:.2f} "
        f"character={args.character} ascension={'keep' if args.ascension < 0 else args.ascension}",
        flush=True,
    )
    print(
        f"[{_timestamp()}] reward_w="
        f"err:{args.error_penalty_weight:.2f} floor:{args.floor_advance_weight:.2f} act:{args.act_advance_weight:.2f} "
        f"hp:{args.hp_delta_weight:.2f}/maxhp gold:{args.gold_delta_weight:.2f}@{args.gold_delta_scale:.0f} "
        f"enemy:{args.enemy_hp_delta_weight:.2f}/enemymax "
        f"turn:{args.turn_end_weight:.2f} turn_skip:{resolved_turn_skip_penalty:.2f}/energy"
        f"{'(manual)' if args.turn_skip_unspent_penalty > 0.0 else f'@max{turn_skip_reference:.2f}x{args.turn_skip_unspent_multiplier:.1f}'} "
        f"combat+:{args.combat_end_weight:.2f} combat-:{args.combat_defeat_weight:.2f} act_end:{args.act_end_weight:.2f} "
        f"run+:{args.run_victory_weight:.2f} run-:{args.run_defeat_weight:.2f} "
        f"shape:{args.combat_tactical_shaping:.2f} slack:{args.end_turn_slack_penalty:.2f} "
        f"end_turn_guard={'off' if args.allow_premature_end_turn else (f'on@timeout{args.end_turn_guard_timeout_seconds:.1f}s' if args.end_turn_guard_timeout_seconds > 0.0 else f'on@stall{args.end_turn_guard_stall_limit}')} "
        f"card_select_topk={args.card_select_candidate_limit}",
        flush=True,
    )
    print(
        f"[{_timestamp()}] settlement="
        f"{args.settlement_signal_mode}/{args.settlement_normalization} clip={args.settlement_clip:.2f} "
        f"turn:{args.turn_settlement_weight:.2f}@{args.turn_settlement_decay:.3f}/cap{args.turn_settlement_clean_cap:.2f}/skip_local_only "
        f"combat:{args.combat_settlement_weight:.2f}@{args.combat_settlement_decay:.3f} "
        f"act:{args.act_settlement_weight:.2f}@{args.act_settlement_decay:.3f} "
        f"run:{args.run_settlement_weight:.2f}@{args.run_settlement_decay:.3f} "
        f"replay_bonus={agent.turn_end_sample_bonus:.1f}/{agent.combat_end_sample_bonus:.1f}/"
        f"{agent.act_end_sample_bonus:.1f}/{agent.run_end_sample_bonus:.1f}",
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
    turn_window = _open_turn_window(state, total_steps + 1)
    run_history: list[Transition] = []

    while args.max_steps <= 0 or total_steps < args.max_steps:
        if combat_window is None and _is_combat_state(state):
            combat_window = _open_combat_window(state, total_steps + 1)
        if turn_window is None and _is_player_combat_turn(state):
            turn_window = _open_turn_window(state, total_steps + 1)

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

        observation = env.extract_observation_features(state)
        search = agent.plan(observation, legal_actions, use_exploration_noise=not args.no_root_noise)
        action_id = agent.select_action(search, legal_actions, temperature=args.temperature)
        next_state, raw_reward, done, info = env.step(action_id, state)
        transition = Transition(
            observation=observation,
            action_index=action_id,
            reward=raw_reward,
            next_observation=env.extract_observation_features(next_state),
            done=done,
            policy_target=search.action_probabilities,
            raw_reward=raw_reward,
            turn_end=bool(info.get("turn_end")),
            combat_end=bool(info.get("combat_end")),
            act_end=bool(info.get("act_end")),
            run_end=bool(info.get("run_end")),
        )

        run_history.append(transition)
        episode_reward += raw_reward
        if act_window is not None:
            act_window.steps += 1
            act_window.reward += raw_reward
            act_window.history.append(transition)
        if combat_window is not None:
            combat_window.steps += 1
            combat_window.reward += raw_reward
            combat_window.history.append(transition)
        if turn_window is not None:
            turn_window.steps += 1
            turn_window.reward += raw_reward
            turn_window.history.append(transition)
            reward_breakdown = info.get("reward_breakdown")
            settlement_delta = raw_reward
            if isinstance(reward_breakdown, dict):
                skip_penalty = float(reward_breakdown.get("turn_skip_unspent", 0.0) or 0.0)
                settlement_delta -= skip_penalty
                if skip_penalty < 0.0:
                    turn_window.skip_unspent_count += 1
                    turn_window.skip_unspent_energy += int(float(reward_breakdown.get("turn_skip_unspent_energy", 0.0) or 0.0))
            turn_window.settlement_reward += settlement_delta

        turn_settlement = 0.0
        combat_settlement = 0.0
        act_settlement = 0.0
        run_settlement = 0.0
        if transition.turn_end and turn_window is not None:
            turn_settlement = _apply_turn_settlement(turn_window, settlement_config)
        if transition.combat_end and combat_window is not None:
            combat_settlement = _apply_window_settlement(
                combat_window.history,
                combat_window.reward,
                combat_window.steps,
                settlement_config.combat_weight,
                settlement_config.combat_decay,
                settlement_config,
            )
        if transition.act_end and act_window is not None:
            act_settlement = _apply_window_settlement(
                act_window.history,
                act_window.reward,
                act_window.steps,
                settlement_config.act_weight,
                settlement_config.act_decay,
                settlement_config,
            )
        if done:
            run_settlement = _apply_window_settlement(
                run_history,
                episode_reward,
                max(1, episode_steps + 1),
                settlement_config.run_weight,
                settlement_config.run_decay,
                settlement_config,
            )

        agent.remember(transition)
        metrics = agent.learn(args.updates_per_step)

        total_steps += 1
        episode_steps += 1

        player = next_state.get("player") if isinstance(next_state.get("player"), dict) else {}
        run = next_state.get("run") if isinstance(next_state.get("run"), dict) else {}
        print(
            f"[{_timestamp()}] step={total_steps} ep={episode_index} act={run.get('act', '?')} floor={run.get('floor', '?')} "
            f"hp={player.get('hp', '?')}/{player.get('max_hp', '?')} state={next_state.get('state_type')} "
            f"action={info['description']} {_reward_display(transition)} ep_raw_reward={episode_reward:+.3f} "
            f"root_value={search.root_value:+.3f} {_format_metrics(metrics)}"
            f"{_format_boundary_suffix(transition)}{_format_info_suffix(info)}",
            flush=True,
        )

        if args.checkpoint_interval > 0 and total_steps % args.checkpoint_interval == 0:
            agent.save(checkpoint_path)
            print(f"[{_timestamp()}] checkpoint saved to {checkpoint_path}", flush=True)

        if transition.turn_end and turn_window is not None:
            print(
                f"[{_timestamp()}] turn_end ep={episode_index} act={turn_window.act} floor={turn_window.floor} "
                f"round={turn_window.round} steps={turn_window.steps} raw_reward={turn_window.reward:+.3f} "
                f"skip_unspent={turn_window.skip_unspent_count} lost_energy={turn_window.skip_unspent_energy} "
                f"settlement={turn_settlement:+.3f}",
                flush=True,
            )
            turn_window = _open_turn_window(next_state, total_steps + 1)

        if transition.combat_end and combat_window is not None:
            episode_combats += 1
            if act_window is not None:
                act_window.combats += 1
            print(
                f"[{_timestamp()}] combat_end ep={episode_index} act={combat_window.act} floor={combat_window.floor} "
                f"steps={combat_window.steps} raw_reward={combat_window.reward:+.3f} "
                f"settlement={combat_settlement:+.3f} result={_combat_result(next_state, transition.run_end)}",
                flush=True,
            )
            combat_window = None
            if not transition.turn_end:
                turn_window = _open_turn_window(next_state, total_steps + 1)

        if transition.act_end and act_window is not None:
            print(
                f"[{_timestamp()}] act_summary ep={episode_index} act={act_window.act} reason=act_end "
                f"steps={act_window.steps} floors={act_window.start_floor}->{info.get('next_floor', run.get('floor', '?'))} "
                f"combats={act_window.combats} raw_reward={act_window.reward:+.3f} "
                f"settlement={act_settlement:+.3f}",
                flush=True,
            )
            act_window = _open_act_window(next_state, total_steps + 1) if not transition.run_end else None

        if done:
            if act_window is not None and not transition.act_end:
                print(
                    f"[{_timestamp()}] act_summary ep={episode_index} act={act_window.act} reason=run_end "
                    f"steps={act_window.steps} floors={act_window.start_floor}->{run.get('floor', '?')} "
                    f"combats={act_window.combats} raw_reward={act_window.reward:+.3f} "
                    f"settlement={act_settlement:+.3f}",
                    flush=True,
                )
            replay_counts = agent.replay_boundary_counts()
            result = "victory" if float(player.get("hp", 0) or 0) > 0 else "defeat"
            print(
                f"[{_timestamp()}] run_end ep={episode_index} steps={episode_steps} combats={episode_combats} "
                f"raw_return={episode_reward:+.3f} run_settlement={run_settlement:+.3f} "
                f"result={result} {_format_replay_counts(replay_counts)}",
                flush=True,
            )
            agent.save(checkpoint_path)
            episode_index += 1
            episode_steps = 0
            episode_reward = 0.0
            episode_combats = 0
            run_history = []
            state = _wait_for_run(env, args.menu_sleep, args.character, args.ascension, args.manual_start)
            act_window = _open_act_window(state, total_steps + 1)
            combat_window = _open_combat_window(state, total_steps + 1)
            turn_window = _open_turn_window(state, total_steps + 1)
            continue

        state = next_state
        if turn_window is None and _is_player_combat_turn(state):
            turn_window = _open_turn_window(state, total_steps + 1)

    agent.save(checkpoint_path)
    print(f"[{_timestamp()}] final checkpoint saved to {checkpoint_path}", flush=True)
