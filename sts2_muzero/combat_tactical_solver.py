from __future__ import annotations

import re
from dataclasses import dataclass

COMBAT_STATE_TYPES = {"monster", "elite", "boss"}
ALL_ENEMY_TARGET_TYPES = {"ALLENEMIES"}
_UNSUPPORTED_COST = {"X", "?", ""}
_DAMAGE_CAP_ONE_TOKENS = ("INTANGIBLE",)
_THORNS_TOKENS = ("THORNS",)
_VULNERABLE_TOKENS = ("VULNERABLE",)

_CHINESE_MULTI_DAMAGE_PATTERNS = (
    re.compile(r"\u9020\u6210\s*(\d+)\s*\u70b9\u4f24\u5bb3\s*(\d+)\s*\u6b21"),
    re.compile(r"\u9020\u6210\s*(\d+)\s*\u70b9\u4f24\u5bb3.*?(\d+)\s*\u6b21"),
)
_ENGLISH_MULTI_DAMAGE_PATTERNS = (
    re.compile(r"deal\s*(\d+)\s*damage\s*(\d+)\s*times"),
    re.compile(r"deal\s*(\d+)\s*damage.*?(\d+)\s*times"),
)
_CHINESE_SIMPLE_DAMAGE = re.compile(r"\u9020\u6210\s*(\d+)\s*\u70b9\u4f24\u5bb3")
_ENGLISH_SIMPLE_DAMAGE = re.compile(r"deal\s*(\d+)\s*damage")
_CHINESE_VULNERABLE = re.compile(r"(?:\u7ed9\u4e88|\u65bd\u52a0)\s*(\d+)\s*\u5c42\u6613\u4f24")
_ENGLISH_VULNERABLE = re.compile(r"apply\s*(\d+)\s*vulnerable")


@dataclass(frozen=True)
class TacticalAction:
    card_index: int
    target_entity_id: str = ""


@dataclass(frozen=True)
class CombatTacticalAnalysis:
    available: bool = False
    lethal_exists: bool = False
    min_required_block_after_best_kill: int = 0
    best_sequence: tuple[TacticalAction, ...] = ()
    best_root_actions: tuple[TacticalAction, ...] = ()
    lethal_required_block: int = 0
    lethal_sequence: tuple[TacticalAction, ...] = ()
    lethal_root_actions: tuple[TacticalAction, ...] = ()


@dataclass(frozen=True)
class _ParsedAttackCard:
    card_index: int
    cost: int
    damage_per_hit: int
    hit_count: int
    vulnerable_amount: int
    targets_all_enemies: bool


@dataclass(frozen=True)
class _EnemyState:
    entity_id: str
    hp: int
    block: int
    incoming_damage: int
    vulnerable: int
    thorns: int
    damage_cap_per_hit: int


@dataclass(frozen=True)
class _Plan:
    lethal: bool = False
    remaining_incoming: int = 0
    remaining_enemy_count: int = 0
    remaining_enemy_total_hp: int = 0
    additional_retaliation: int = 0
    sequence: tuple[TacticalAction, ...] = ()

    @property
    def required_block(self) -> int:
        return self.remaining_incoming + self.additional_retaliation


@dataclass(frozen=True)
class _SolveResult:
    best_plan: _Plan
    best_lethal_plan: _Plan | None


def analyze_combat_turn(state: dict[str, object] | None) -> CombatTacticalAnalysis:
    if not isinstance(state, dict):
        return CombatTacticalAnalysis()
    if not _combat_action_window_ready(state):
        return CombatTacticalAnalysis()
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    player = state.get("player") if isinstance(state.get("player"), dict) else {}
    enemies = tuple(_normalize_enemy(enemy) for enemy in battle.get("enemies", []) if isinstance(enemy, dict))
    living_enemies = tuple(enemy for enemy in enemies if enemy.hp > 0)
    if not living_enemies:
        return CombatTacticalAnalysis(available=True, lethal_exists=True)
    current_energy = _player_energy_units(player)
    attack_cards = tuple(
        parsed
        for parsed in (_parse_attack_card(card, current_energy) for card in player.get("hand", []) if isinstance(card, dict))
        if parsed is not None
    )
    initial_plan = _evaluate_leaf_plan(living_enemies)
    if not attack_cards:
        return CombatTacticalAnalysis(
            available=True,
            lethal_exists=False,
            min_required_block_after_best_kill=initial_plan.required_block,
        )
    memo: dict[tuple[int, int, tuple[tuple[object, ...], ...]], _SolveResult] = {}
    initial_mask = (1 << len(attack_cards)) - 1
    solved = _solve(initial_mask, current_energy, living_enemies, attack_cards, memo)
    best_plan = solved.best_plan
    lethal_plan = solved.best_lethal_plan
    return CombatTacticalAnalysis(
        available=True,
        lethal_exists=lethal_plan is not None,
        min_required_block_after_best_kill=best_plan.required_block,
        best_sequence=best_plan.sequence,
        best_root_actions=best_plan.sequence[:1],
        lethal_required_block=(lethal_plan.required_block if lethal_plan is not None else best_plan.required_block),
        lethal_sequence=(lethal_plan.sequence if lethal_plan is not None else ()),
        lethal_root_actions=(lethal_plan.sequence[:1] if lethal_plan is not None else ()),
    )


def _combat_action_window_ready(state: dict[str, object]) -> bool:
    if str(state.get("state_type", "unknown")) not in COMBAT_STATE_TYPES:
        return False
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    hand_mode = str(battle.get("hand_mode", "play") or "play").lower()
    return (
        str(battle.get("turn", "")) == "player"
        and bool(battle.get("is_play_phase"))
        and not bool(battle.get("player_actions_disabled"))
        and not bool(battle.get("hand_in_card_play"))
        and hand_mode == "play"
    )


def _player_energy_units(player: dict[str, object]) -> int:
    try:
        return max(0, int(float(player.get("energy", 0) or 0)))
    except (TypeError, ValueError):
        return 0


def _normalize_enemy(enemy: dict[str, object]) -> _EnemyState:
    statuses = enemy.get("status") if isinstance(enemy.get("status"), list) else []
    return _EnemyState(
        entity_id=str(enemy.get("entity_id", "") or ""),
        hp=max(0, _to_int(enemy.get("hp"))),
        block=max(0, _to_int(enemy.get("block"))),
        incoming_damage=_enemy_intent_damage(enemy),
        vulnerable=max(0, _status_amount(statuses, _VULNERABLE_TOKENS)),
        thorns=max(0, _status_amount(statuses, _THORNS_TOKENS)),
        damage_cap_per_hit=(1 if _has_status(statuses, _DAMAGE_CAP_ONE_TOKENS) else 0),
    )


def _enemy_intent_damage(enemy: dict[str, object]) -> int:
    intents = enemy.get("intents") if isinstance(enemy.get("intents"), list) else []
    total = 0
    for intent in intents:
        if not isinstance(intent, dict):
            continue
        digits = "".join(character for character in str(intent.get("label", "")) if character.isdigit())
        if digits:
            total += int(digits)
    return total


def _status_amount(statuses: list[object], token_groups: tuple[str, ...]) -> int:
    total = 0
    for status in statuses:
        if not isinstance(status, dict):
            continue
        token = str(status.get("id", "") or "").upper()
        if not any(group in token for group in token_groups):
            continue
        total += max(0, _to_int(status.get("amount")))
    return total


def _has_status(statuses: list[object], token_groups: tuple[str, ...]) -> bool:
    return any(
        isinstance(status, dict) and any(group in str(status.get("id", "") or "").upper() for group in token_groups)
        for status in statuses
    )


def _parse_attack_card(card: dict[str, object], current_energy: int) -> _ParsedAttackCard | None:
    if str(card.get("type", "") or "") != "Attack":
        return None
    if not bool(card.get("can_play")):
        return None
    card_index = card.get("index")
    if not isinstance(card_index, int) or card_index < 0:
        return None
    cost_token = str(card.get("cost", "") or "").strip().upper()
    if cost_token in _UNSUPPORTED_COST:
        return None
    try:
        cost = max(0, int(float(cost_token)))
    except (TypeError, ValueError):
        return None
    if cost > current_energy:
        return None
    damage_per_hit, hit_count = _parse_damage_profile(card)
    if damage_per_hit <= 0 or hit_count <= 0:
        return None
    target_token = str(card.get("target_type", "") or "").strip().upper()
    description = str(card.get("description", "") or "").strip().lower()
    targets_all_enemies = (
        target_token in ALL_ENEMY_TARGET_TYPES
        or "\u6240\u6709\u654c\u4eba" in description
        or "all enemies" in description
    )
    vulnerable_amount = _parse_vulnerable_amount(card)
    return _ParsedAttackCard(
        card_index=card_index,
        cost=cost,
        damage_per_hit=damage_per_hit,
        hit_count=hit_count,
        vulnerable_amount=vulnerable_amount,
        targets_all_enemies=targets_all_enemies,
    )


def _parse_damage_profile(card: dict[str, object]) -> tuple[int, int]:
    text = str(card.get("description", "") or "").strip().lower()
    for pattern in _CHINESE_MULTI_DAMAGE_PATTERNS:
        match = pattern.search(text)
        if match is not None:
            return int(match.group(1)), int(match.group(2))
    for pattern in _ENGLISH_MULTI_DAMAGE_PATTERNS:
        match = pattern.search(text)
        if match is not None:
            return int(match.group(1)), int(match.group(2))
    match = _CHINESE_SIMPLE_DAMAGE.search(text)
    if match is not None:
        return int(match.group(1)), 1
    match = _ENGLISH_SIMPLE_DAMAGE.search(text)
    if match is not None:
        return int(match.group(1)), 1
    return 0, 0


def _parse_vulnerable_amount(card: dict[str, object]) -> int:
    text = str(card.get("description", "") or "").strip().lower()
    match = _CHINESE_VULNERABLE.search(text)
    if match is not None:
        return int(match.group(1))
    match = _ENGLISH_VULNERABLE.search(text)
    if match is not None:
        return int(match.group(1))
    return 0


def _solve(
    available_mask: int,
    energy_left: int,
    enemies: tuple[_EnemyState, ...],
    cards: tuple[_ParsedAttackCard, ...],
    memo: dict[tuple[int, int, tuple[tuple[object, ...], ...]], _SolveResult],
) -> _SolveResult:
    key = (available_mask, energy_left, _enemy_signature(enemies))
    cached = memo.get(key)
    if cached is not None:
        return cached
    best_plan = _evaluate_leaf_plan(enemies)
    best_lethal_plan = best_plan if best_plan.lethal else None
    for card_position, card in enumerate(cards):
        card_bit = 1 << card_position
        if available_mask & card_bit == 0 or card.cost > energy_left:
            continue
        if card.targets_all_enemies:
            next_enemies, retaliation = _apply_attack(enemies, card, None)
            child = _solve(available_mask ^ card_bit, energy_left - card.cost, next_enemies, cards, memo)
            best_plan = _better_general(_prepend_step(child.best_plan, card, retaliation, ""), best_plan)
            if child.best_lethal_plan is not None:
                best_lethal_plan = _better_lethal(
                    _prepend_step(child.best_lethal_plan, card, retaliation, ""),
                    best_lethal_plan,
                )
            continue
        for target_index, enemy in enumerate(enemies):
            if enemy.hp <= 0:
                continue
            next_enemies, retaliation = _apply_attack(enemies, card, target_index)
            child = _solve(available_mask ^ card_bit, energy_left - card.cost, next_enemies, cards, memo)
            best_plan = _better_general(_prepend_step(child.best_plan, card, retaliation, enemy.entity_id), best_plan)
            if child.best_lethal_plan is not None:
                best_lethal_plan = _better_lethal(
                    _prepend_step(child.best_lethal_plan, card, retaliation, enemy.entity_id),
                    best_lethal_plan,
                )
    result = _SolveResult(best_plan=best_plan, best_lethal_plan=best_lethal_plan)
    memo[key] = result
    return result


def _enemy_signature(enemies: tuple[_EnemyState, ...]) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            enemy.entity_id,
            enemy.hp,
            enemy.block,
            enemy.incoming_damage,
            enemy.vulnerable,
            enemy.thorns,
            enemy.damage_cap_per_hit,
        )
        for enemy in enemies
    )


def _evaluate_leaf_plan(enemies: tuple[_EnemyState, ...]) -> _Plan:
    living = tuple(enemy for enemy in enemies if enemy.hp > 0)
    return _Plan(
        lethal=not living,
        remaining_incoming=sum(enemy.incoming_damage for enemy in living),
        remaining_enemy_count=len(living),
        remaining_enemy_total_hp=sum(max(0, enemy.hp) + max(0, enemy.block) for enemy in living),
        additional_retaliation=0,
        sequence=(),
    )


def _prepend_step(plan: _Plan, card: _ParsedAttackCard, retaliation: int, target_entity_id: str) -> _Plan:
    return _Plan(
        lethal=plan.lethal,
        remaining_incoming=plan.remaining_incoming,
        remaining_enemy_count=plan.remaining_enemy_count,
        remaining_enemy_total_hp=plan.remaining_enemy_total_hp,
        additional_retaliation=plan.additional_retaliation + retaliation,
        sequence=(TacticalAction(card_index=card.card_index, target_entity_id=target_entity_id),) + plan.sequence,
    )


def _better_general(left: _Plan, right: _Plan) -> _Plan:
    if _general_plan_key(left) < _general_plan_key(right):
        return left
    return right


def _better_lethal(left: _Plan | None, right: _Plan | None) -> _Plan | None:
    if left is None:
        return right
    if right is None:
        return left
    if _lethal_plan_key(left) < _lethal_plan_key(right):
        return left
    return right


def _general_plan_key(plan: _Plan) -> tuple[int, int, int, int, int, int, int]:
    return (
        plan.required_block,
        0 if plan.lethal else 1,
        plan.remaining_incoming,
        plan.remaining_enemy_count,
        plan.remaining_enemy_total_hp,
        plan.additional_retaliation,
        len(plan.sequence),
    )


def _lethal_plan_key(plan: _Plan) -> tuple[int, int, int]:
    return (
        plan.required_block,
        plan.additional_retaliation,
        len(plan.sequence),
    )


def _apply_attack(
    enemies: tuple[_EnemyState, ...],
    card: _ParsedAttackCard,
    target_index: int | None,
) -> tuple[tuple[_EnemyState, ...], int]:
    next_enemies = list(enemies)
    retaliation = 0
    if card.targets_all_enemies:
        for enemy_index, enemy in enumerate(next_enemies):
            if enemy.hp <= 0:
                continue
            updated, enemy_retaliation = _apply_attack_to_enemy(enemy, card)
            next_enemies[enemy_index] = updated
            retaliation += enemy_retaliation
        return tuple(next_enemies), retaliation
    if target_index is None or target_index < 0 or target_index >= len(next_enemies):
        return enemies, 0
    target_enemy = next_enemies[target_index]
    if target_enemy.hp <= 0:
        return enemies, 0
    updated_enemy, retaliation = _apply_attack_to_enemy(target_enemy, card)
    next_enemies[target_index] = updated_enemy
    return tuple(next_enemies), retaliation


def _apply_attack_to_enemy(enemy: _EnemyState, card: _ParsedAttackCard) -> tuple[_EnemyState, int]:
    hp = enemy.hp
    block = enemy.block
    retaliation = 0
    for _ in range(card.hit_count):
        if hp <= 0:
            break
        retaliation += enemy.thorns
        hit_damage = card.damage_per_hit
        if enemy.vulnerable > 0:
            hit_damage += hit_damage // 2
        if enemy.damage_cap_per_hit > 0:
            hit_damage = min(hit_damage, enemy.damage_cap_per_hit)
        block_absorb = min(block, hit_damage)
        block -= block_absorb
        hp -= max(0, hit_damage - block_absorb)
    vulnerable = enemy.vulnerable + card.vulnerable_amount if hp > 0 else 0
    return (
        _EnemyState(
            entity_id=enemy.entity_id,
            hp=max(0, hp),
            block=max(0, block),
            incoming_damage=enemy.incoming_damage if hp > 0 else 0,
            vulnerable=vulnerable,
            thorns=enemy.thorns if hp > 0 else 0,
            damage_cap_per_hit=enemy.damage_cap_per_hit if hp > 0 else 0,
        ),
        retaliation,
    )


def _to_int(value: object) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0
