from __future__ import annotations

import re
from dataclasses import dataclass

from .monster_strategy import (
    analyze_encounter_profile,
    card_strategy_vector_from_snapshot,
    strategy_alignment_score,
    strategy_value,
)


CAPABILITY_DIMENSIONS = (
    "frontload",
    "aoe",
    "block",
    "draw",
    "energy",
    "scaling",
    "consistency",
    "utility",
    "early",
    "mid",
    "late",
)
_DIMENSION_INDEX = {name: index for index, name in enumerate(CAPABILITY_DIMENSIONS)}
_NODE_TYPE_KEYS = ("Monster", "Elite", "Boss")
_ALL_ENEMY_TARGET_TYPES = {"ALLENEMIES"}
_ENEMY_TARGET_TYPES = {"ANYENEMY", "ENEMY", "SINGLEENEMY"}

_CARD_RARITY_SCORE = {
    "Basic": -0.15,
    "Starter": -0.15,
    "Common": 0.25,
    "Uncommon": 0.55,
    "Rare": 0.90,
    "Special": 0.15,
    "Ancient": 1.05,
}
_CARD_TYPE_SCORE = {
    "Attack": 0.35,
    "Skill": 0.25,
    "Power": 0.70,
    "Status": -1.50,
    "Curse": -2.20,
}
_DIMENSION_WEIGHTS = {
    "frontload": 1.20,
    "aoe": 1.05,
    "block": 1.15,
    "draw": 0.90,
    "energy": 0.95,
    "scaling": 1.00,
    "consistency": 1.05,
    "utility": 0.75,
    "early": 1.10,
    "mid": 0.85,
    "late": 0.95,
}
_OVERSUPPLY_WEIGHTS = {
    "frontload": 0.35,
    "aoe": 0.30,
    "block": 0.40,
    "draw": 0.28,
    "energy": 0.25,
    "scaling": 0.35,
    "consistency": 0.35,
    "utility": 0.20,
    "early": 0.38,
    "mid": 0.20,
    "late": 0.28,
}
_DEFAULT_NODE_DEMANDS = {
    "Monster": {
        "frontload": 0.78,
        "aoe": 0.20,
        "block": 0.55,
        "draw": 0.20,
        "energy": 0.12,
        "scaling": 0.22,
        "consistency": 0.35,
        "utility": 0.20,
        "early": 0.78,
        "mid": 0.26,
        "late": 0.12,
    },
    "Elite": {
        "frontload": 0.92,
        "aoe": 0.30,
        "block": 0.78,
        "draw": 0.28,
        "energy": 0.20,
        "scaling": 0.52,
        "consistency": 0.60,
        "utility": 0.24,
        "early": 0.86,
        "mid": 0.54,
        "late": 0.30,
    },
    "Boss": {
        "frontload": 0.58,
        "aoe": 0.24,
        "block": 0.84,
        "draw": 0.58,
        "energy": 0.46,
        "scaling": 0.98,
        "consistency": 0.88,
        "utility": 0.28,
        "early": 0.48,
        "mid": 0.82,
        "late": 1.12,
    },
}
_ACT_DEMAND_ADJUSTMENTS = {
    "UNDERDOCKS": {
        "Monster": {"frontload": 0.14, "block": 0.12, "early": 0.12},
        "Elite": {"frontload": 0.12, "block": 0.10, "consistency": 0.05},
        "Boss": {"block": 0.12, "consistency": 0.12, "frontload": 0.08},
    },
    "OVERGROWTH": {
        "Monster": {"aoe": 0.22, "utility": 0.12, "block": 0.06},
        "Elite": {"aoe": 0.26, "scaling": 0.12, "utility": 0.10},
        "Boss": {"aoe": 0.18, "scaling": 0.18, "utility": 0.12, "late": 0.10},
    },
    "HIVE": {
        "Monster": {"aoe": 0.26, "frontload": 0.10, "consistency": 0.06},
        "Elite": {"aoe": 0.22, "block": 0.12, "frontload": 0.08},
        "Boss": {"aoe": 0.14, "block": 0.12, "scaling": 0.16, "consistency": 0.16},
    },
}

_CHINESE_DAMAGE_PATTERNS = (
    re.compile(r"\u9020\u6210\s*(\d+)\s*\u70b9\u4f24\u5bb3\s*(\d+)\s*\u6b21"),
    re.compile(r"\u5bf9\u6240\u6709\u654c\u4eba\u9020\u6210\s*(\d+)\s*\u70b9\u4f24\u5bb3"),
    re.compile(r"\u9020\u6210\s*(\d+)\s*\u70b9\u4f24\u5bb3"),
)
_ENGLISH_DAMAGE_PATTERNS = (
    re.compile(r"deal\s*(\d+)\s*damage\s*(\d+)\s*times"),
    re.compile(r"deal\s*(\d+)\s*damage\s*to\s*all\s*enemies"),
    re.compile(r"deal\s*(\d+)\s*damage"),
)
_CHINESE_BLOCK = re.compile(r"\u83b7\u5f97\s*(\d+)\s*\u70b9\u683c\u6321")
_ENGLISH_BLOCK = re.compile(r"gain\s*(\d+)\s*block")
_CHINESE_DRAW = re.compile(r"\u62bd\s*(\d+)\s*\u5f20\u724c")
_ENGLISH_DRAW = re.compile(r"draw\s*(\d+)\s*cards?")
_CHINESE_ENERGY = re.compile(r"\u83b7\u5f97\s*(\d+)\s*\u70b9\u80fd\u91cf")
_ENGLISH_ENERGY = re.compile(r"gain\s*(\d+)\s*energy")
_CHINESE_VULNERABLE = re.compile(r"(?:\u7ed9\u4e88|\u65bd\u52a0)\s*(\d+)\s*\u5c42\u6613\u4f24")
_ENGLISH_VULNERABLE = re.compile(r"apply\s*(\d+)\s*vulnerable")
_CHINESE_WEAK = re.compile(r"(?:\u7ed9\u4e88|\u65bd\u52a0)\s*(\d+)\s*\u5c42\u865a\u5f31")
_ENGLISH_WEAK = re.compile(r"apply\s*(\d+)\s*weak")
_CHINESE_FRAIL = re.compile(r"(?:\u7ed9\u4e88|\u65bd\u52a0)\s*(\d+)\s*\u5c42\u6613\u788e")
_ENGLISH_FRAIL = re.compile(r"apply\s*(\d+)\s*frail")

_STRENGTH_TOKENS = ("strength", "\u529b\u91cf")
_DEXTERITY_TOKENS = ("dexterity", "\u654f\u6377")
_RETAIN_TOKENS = ("retain", "\u4fdd\u7559")
_EXHAUST_TOKENS = ("exhaust", "consume", "\u6d88\u8017", "\u8017\u5c3d", "\u67af\u7aed")
_FREE_TOKENS = ("cost 0", "\u8017\u80fd\u53d8\u4e3a0", "\u514d\u8d39", "0[")
_SEARCH_TOKENS = ("choose", "select", "\u9009\u62e9", "\u68c0\u7d22")
_REMOVE_TOKENS = ("remove", "purge", "cleanse", "\u79fb\u9664", "\u6e05\u9664")
_PER_TURN_TOKENS = ("each turn", "\u6bcf\u56de\u5408")


def _empty_vector() -> list[float]:
    return [0.0 for _ in CAPABILITY_DIMENSIONS]


def _freeze_vector(values: list[float]) -> tuple[float, ...]:
    return tuple(max(0.0, min(3.0, float(value))) for value in values)


def _vector_value(vector: tuple[float, ...], name: str) -> float:
    return float(vector[_DIMENSION_INDEX[name]])


def _vector_add(target: list[float], source: tuple[float, ...], scale: float = 1.0) -> None:
    for index, value in enumerate(source):
        target[index] += float(value) * scale


def _vector_from_mapping(mapping: dict[str, float] | None = None) -> tuple[float, ...]:
    values = _empty_vector()
    for name, value in (mapping or {}).items():
        index = _DIMENSION_INDEX.get(name)
        if index is None:
            continue
        values[index] = float(value)
    return _freeze_vector(values)


def _blend_vectors(left: tuple[float, ...], right: tuple[float, ...], left_weight: float, right_weight: float) -> tuple[float, ...]:
    total = max(1e-9, left_weight + right_weight)
    values = _empty_vector()
    for index in range(len(CAPABILITY_DIMENSIONS)):
        values[index] = ((left[index] * left_weight) + (right[index] * right_weight)) / total
    return _freeze_vector(values)


def _weighted_vector_sum(vector: tuple[float, ...]) -> float:
    total = 0.0
    for name in CAPABILITY_DIMENSIONS:
        total += float(vector[_DIMENSION_INDEX[name]]) * _DIMENSION_WEIGHTS[name]
    return total


def _weighted_vector_norm(vector: tuple[float, ...], scale: float = 7.5, clamp: float = 3.0) -> float:
    if scale <= 0.0:
        return 0.0
    return max(0.0, min(clamp, _weighted_vector_sum(vector) / scale))


def _weighted_alignment(capability: tuple[float, ...], demand: tuple[float, ...]) -> float:
    demand_total = _weighted_vector_sum(demand)
    if demand_total <= 1e-9:
        return 1.0
    matched = 0.0
    for name in CAPABILITY_DIMENSIONS:
        index = _DIMENSION_INDEX[name]
        matched += min(float(capability[index]), float(demand[index])) * _DIMENSION_WEIGHTS[name]
    return max(0.0, min(1.5, matched / demand_total))


def _gap_pressure(vector: tuple[float, ...]) -> float:
    return max(0.0, min(1.5, _weighted_vector_sum(vector) / 3.2))


def _max_positive_gap(demand: tuple[float, ...], capability: tuple[float, ...]) -> tuple[float, ...]:
    values = _empty_vector()
    for index in range(len(CAPABILITY_DIMENSIONS)):
        values[index] = max(0.0, float(demand[index]) - float(capability[index]))
    return tuple(values)


def _signed_gap(demand: tuple[float, ...], capability: tuple[float, ...]) -> tuple[float, ...]:
    values = _empty_vector()
    for index in range(len(CAPABILITY_DIMENSIONS)):
        values[index] = float(demand[index]) - float(capability[index])
    return tuple(values)


def _top_dimensions(vector: tuple[float, ...], limit: int = 3) -> tuple[str, ...]:
    ranked = sorted(
        CAPABILITY_DIMENSIONS,
        key=lambda name: (float(vector[_DIMENSION_INDEX[name]]), -_DIMENSION_INDEX[name]),
        reverse=True,
    )
    return tuple(name for name in ranked[:limit] if vector[_DIMENSION_INDEX[name]] > 0.05)


def _safe_cost_value(cost_token: object) -> float:
    text = str(cost_token or "").strip().upper()
    if text == "X":
        return 1.5
    try:
        return max(0.0, float(text))
    except (TypeError, ValueError):
        return 0.0


def _text_has_any(text: str, tokens: tuple[str, ...] | list[str]) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in tokens)


def _intents_total_damage(enemy: dict[str, object]) -> float:
    intents = enemy.get("intents") if isinstance(enemy.get("intents"), list) else []
    total = 0.0
    for intent in intents:
        if not isinstance(intent, dict):
            continue
        digits = "".join(character for character in str(intent.get("label", "")) if character.isdigit())
        if digits:
            total += float(digits)
    return total


def _power_amount(statuses: list[object], token_groups: tuple[str, ...]) -> float:
    total = 0.0
    for status in statuses:
        if not isinstance(status, dict):
            continue
        status_id = str(status.get("id", "") or "").upper()
        if not any(token in status_id for token in token_groups):
            continue
        try:
            total += float(status.get("amount", 0) or 0.0)
        except (TypeError, ValueError):
            continue
    return total


@dataclass(frozen=True)
class CardSnapshot:
    token: str
    card_id: str
    name: str
    card_type: str
    rarity: str
    cost: str
    target_type: str
    description: str
    is_upgraded: bool
    consumes_on_play: bool
    keywords: tuple[str, ...]

    @property
    def text_blob(self) -> str:
        return " ".join(
            part
            for part in (
                self.card_id,
                self.name,
                self.card_type,
                self.rarity,
                self.description,
                *self.keywords,
            )
            if part
        ).lower()


@dataclass(frozen=True)
class CardModel:
    snapshot: CardSnapshot
    vector: tuple[float, ...]
    base_quality: float


@dataclass(frozen=True)
class DeckBuildProfile:
    cards: tuple[CardSnapshot, ...]
    capability: tuple[float, ...]
    total_score: float
    velocity: float
    cohesion: float
    quality: float
    win_condition: float
    attack_density: float
    skill_density: float
    power_density: float
    draw_engine: float
    energy_engine: float
    exhaust_density: float
    low_cost_density: float
    high_cost_density: float
    clutter_penalty: float


@dataclass(frozen=True)
class BuildStrategyContext:
    act_id: str
    floor: int
    player_hp: float
    player_max_hp: float
    hp_ratio: float
    player_gold: float
    deck_profile: DeckBuildProfile
    current_combat_demand: tuple[float, ...]
    future_demand: tuple[float, ...]
    weighted_demand: tuple[float, ...]
    gap: tuple[float, ...]
    signed_gap: tuple[float, ...]
    primary_shortfalls: tuple[str, ...]
    current_mechanics: tuple[float, ...]
    current_strategy_demand: tuple[float, ...]
    boss_demand: tuple[float, ...]
    boss_gap: tuple[float, ...]
    boss_shortfalls: tuple[str, ...]
    boss_strength: float
    boss_threshold: float
    boss_readiness: float
    boss_strength_gap: float


@dataclass(frozen=True)
class StrengthEconomy:
    current_strength: float
    boss_threshold: float
    boss_readiness: float
    boss_gap: float
    hp_strength_value: float
    gold_strength_ratio: float
    monster_expected_gain: float
    monster_expected_hp_loss: float
    elite_expected_gain: float
    elite_expected_hp_loss: float
    question_expected_gain: float
    question_expected_hp_loss: float
    rest_upgrade_gain: float
    rest_heal_gain: float
    rest_base_heal_ratio: float
    event_expected_gain: float
    event_expected_hp_loss: float
    shop_expected_gain: float
    treasure_expected_gain: float


@dataclass(frozen=True)
class RouteStrengthEvaluation:
    deck_strength: float
    deck_hp_strength: float
    next_risk_strength: float
    projected_deck_hp_strength: float
    boss_margin: float
    next_step_feasible: float
    boss_feasible: float
    path_feasible: float
    total_score: float
    constraint_score: float
    growth_score: float


@dataclass(frozen=True)
class CandidateEvaluation:
    label: str
    total_score: float
    gap_closure: float
    deck_total_delta: float
    multiplier_score: float
    oversupply_penalty: float
    dilution_penalty: float
    immediate_bonus: float
    primary_gap: str


def card_snapshot_from_card_dict(card: dict[str, object]) -> CardSnapshot | None:
    card_id = str(card.get("id", "") or "").strip().upper()
    name = str(card.get("name", "") or "").strip()
    card_type = str(card.get("type", "") or "").strip()
    rarity = str(card.get("rarity", "") or "").strip()
    target_type = str(card.get("target_type", "") or "").strip()
    description = str(card.get("description", "") or "").strip()
    cost = str(card.get("cost", "") or "").strip()
    keywords = card.get("keywords") if isinstance(card.get("keywords"), list) else []
    keyword_tokens = tuple(
        " ".join(
            str(keyword.get(key, "") or "").strip()
            for key in ("name", "description")
            if str(keyword.get(key, "") or "").strip()
        )
        for keyword in keywords
        if isinstance(keyword, dict)
    )
    token = f"{card_id or name}:{cost}:{'u' if card.get('is_upgraded') else 'n'}"
    if not card_id and not name and not description:
        return None
    return CardSnapshot(
        token=token,
        card_id=card_id,
        name=name,
        card_type=card_type,
        rarity=rarity,
        cost=cost,
        target_type=target_type,
        description=description,
        is_upgraded=bool(card.get("is_upgraded")),
        consumes_on_play=bool(card.get("consume") or card.get("consumes") or card.get("exhausts") or card.get("exhaust")),
        keywords=tuple(token for token in keyword_tokens if token),
    )


def card_snapshot_from_shop_item(item: dict[str, object]) -> CardSnapshot | None:
    if str(item.get("category", "") or "") != "card":
        return None
    card_id = str(item.get("card_id", "") or "").strip().upper()
    name = str(item.get("card_name", "") or "").strip()
    if not card_id and not name:
        return None
    keywords = item.get("keywords") if isinstance(item.get("keywords"), list) else []
    keyword_tokens = tuple(
        " ".join(
            str(keyword.get(key, "") or "").strip()
            for key in ("name", "description")
            if str(keyword.get(key, "") or "").strip()
        )
        for keyword in keywords
        if isinstance(keyword, dict)
    )
    return CardSnapshot(
        token=f"{card_id or name}:{item.get('index', '?')}",
        card_id=card_id,
        name=name,
        card_type=str(item.get("card_type", "") or "").strip(),
        rarity=str(item.get("card_rarity", "") or "").strip(),
        cost=str(item.get("card_cost", "") or "").strip(),
        target_type=str(item.get("card_target_type", "") or "").strip(),
        description=str(item.get("card_description", "") or "").strip(),
        is_upgraded=bool(item.get("card_is_upgraded")),
        consumes_on_play=False,
        keywords=tuple(token for token in keyword_tokens if token),
    )


def collect_visible_deck_snapshots(state: dict[str, object]) -> tuple[CardSnapshot, ...]:
    player = state.get("player") if isinstance(state.get("player"), dict) else {}
    visible_cards: list[CardSnapshot] = []
    for key in ("hand", "draw_pile", "discard_pile", "exhaust_pile"):
        cards = player.get(key)
        if not isinstance(cards, list):
            continue
        for raw_card in cards:
            if not isinstance(raw_card, dict):
                continue
            snapshot = card_snapshot_from_card_dict(raw_card)
            if snapshot is not None:
                visible_cards.append(snapshot)
    if len(visible_cards) >= 5:
        return tuple(visible_cards)
    for container_key in ("card_select", "hand_select"):
        container = state.get(container_key) if isinstance(state.get(container_key), dict) else {}
        cards = container.get("cards") if isinstance(container.get("cards"), list) else []
        snapshots = tuple(
            snapshot
            for snapshot in (card_snapshot_from_card_dict(card) for card in cards if isinstance(card, dict))
            if snapshot is not None
        )
        if len(snapshots) >= 5:
            return snapshots
    return tuple(visible_cards)


def _parse_damage(snapshot: CardSnapshot) -> tuple[float, float]:
    text = snapshot.text_blob
    for pattern in _CHINESE_DAMAGE_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        if len(match.groups()) >= 2:
            return float(match.group(1)) * float(match.group(2)), float(match.group(1))
        return float(match.group(1)), float(match.group(1))
    for pattern in _ENGLISH_DAMAGE_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        if len(match.groups()) >= 2:
            return float(match.group(1)) * float(match.group(2)), float(match.group(1))
        return float(match.group(1)), float(match.group(1))
    return 0.0, 0.0


def _parse_block(snapshot: CardSnapshot) -> float:
    text = snapshot.text_blob
    match = _CHINESE_BLOCK.search(text)
    if match is not None:
        return float(match.group(1))
    match = _ENGLISH_BLOCK.search(text)
    if match is not None:
        return float(match.group(1))
    return 0.0


def _parse_draw(snapshot: CardSnapshot) -> float:
    text = snapshot.text_blob
    match = _CHINESE_DRAW.search(text)
    if match is not None:
        return float(match.group(1))
    match = _ENGLISH_DRAW.search(text)
    if match is not None:
        return float(match.group(1))
    return 0.0


def _parse_energy(snapshot: CardSnapshot) -> float:
    text = snapshot.text_blob
    match = _CHINESE_ENERGY.search(text)
    if match is not None:
        return float(match.group(1))
    match = _ENGLISH_ENERGY.search(text)
    if match is not None:
        return float(match.group(1))
    return 0.0


def _parse_status_amount(snapshot: CardSnapshot, chinese_pattern: re.Pattern[str], english_pattern: re.Pattern[str]) -> float:
    text = snapshot.text_blob
    match = chinese_pattern.search(text)
    if match is not None:
        return float(match.group(1))
    match = english_pattern.search(text)
    if match is not None:
        return float(match.group(1))
    return 0.0


def _card_targets_all_enemies(snapshot: CardSnapshot) -> bool:
    return (
        str(snapshot.target_type or "").strip().upper() in _ALL_ENEMY_TARGET_TYPES
        or _text_has_any(snapshot.text_blob, ("\u6240\u6709\u654c\u4eba", "all enemies"))
    )


def _card_targets_enemy(snapshot: CardSnapshot) -> bool:
    if _card_targets_all_enemies(snapshot):
        return True
    return str(snapshot.target_type or "").strip().upper() in _ENEMY_TARGET_TYPES


def _analyze_card(snapshot: CardSnapshot) -> CardModel:
    values = _empty_vector()
    cost_value = _safe_cost_value(snapshot.cost)
    damage_total, damage_per_hit = _parse_damage(snapshot)
    block_amount = _parse_block(snapshot)
    draw_amount = _parse_draw(snapshot)
    energy_gain = _parse_energy(snapshot)
    vulnerable = _parse_status_amount(snapshot, _CHINESE_VULNERABLE, _ENGLISH_VULNERABLE)
    weak = _parse_status_amount(snapshot, _CHINESE_WEAK, _ENGLISH_WEAK)
    frail = _parse_status_amount(snapshot, _CHINESE_FRAIL, _ENGLISH_FRAIL)
    all_enemies = _card_targets_all_enemies(snapshot)
    text = snapshot.text_blob

    if _card_targets_enemy(snapshot):
        frontload = damage_total / max(6.0, 5.0 + (cost_value * 2.5))
        values[_DIMENSION_INDEX["frontload"]] += min(1.8, frontload)
    if all_enemies:
        aoe_value = (damage_total / 10.0) + (0.20 if damage_per_hit > 0.0 else 0.0)
        values[_DIMENSION_INDEX["aoe"]] += min(1.9, aoe_value)
        values[_DIMENSION_INDEX["frontload"]] += min(0.55, damage_total / 18.0)

    if block_amount > 0.0:
        values[_DIMENSION_INDEX["block"]] += min(1.8, block_amount / max(6.0, 5.0 + (cost_value * 2.0)))
    if draw_amount > 0.0:
        values[_DIMENSION_INDEX["draw"]] += min(1.8, draw_amount * 0.65)
    if energy_gain > 0.0:
        values[_DIMENSION_INDEX["energy"]] += min(1.8, energy_gain * 0.75)
    if _text_has_any(text, _FREE_TOKENS):
        values[_DIMENSION_INDEX["energy"]] += 0.22

    power_bonus = 0.0
    if snapshot.card_type == "Power":
        power_bonus += 0.55
    if _text_has_any(text, _STRENGTH_TOKENS):
        power_bonus += 0.28
    if _text_has_any(text, _DEXTERITY_TOKENS):
        power_bonus += 0.22
    if _text_has_any(text, _PER_TURN_TOKENS):
        power_bonus += 0.18
    values[_DIMENSION_INDEX["scaling"]] += min(1.9, power_bonus + (0.18 if snapshot.card_type == "Power" else 0.0))

    utility = 0.0
    utility += min(0.45, vulnerable * 0.18)
    utility += min(0.35, weak * 0.15)
    utility += min(0.25, frail * 0.12)
    if _text_has_any(text, _REMOVE_TOKENS):
        utility += 0.18
    if _text_has_any(text, _EXHAUST_TOKENS):
        utility += 0.12
    if _text_has_any(text, _SEARCH_TOKENS):
        utility += 0.10
    values[_DIMENSION_INDEX["utility"]] += min(1.6, utility)

    consistency = 0.0
    consistency += min(0.85, values[_DIMENSION_INDEX["draw"]] * 0.60)
    consistency += min(0.35, values[_DIMENSION_INDEX["energy"]] * 0.30)
    if _text_has_any(text, _RETAIN_TOKENS):
        consistency += 0.24
    if _text_has_any(text, _SEARCH_TOKENS):
        consistency += 0.18
    if snapshot.card_type == "Power" and cost_value <= 1.0:
        consistency += 0.08
    if cost_value == 0.0:
        consistency += 0.14
    values[_DIMENSION_INDEX["consistency"]] += min(1.8, consistency)

    early = (
        values[_DIMENSION_INDEX["frontload"]] * 0.72
        + values[_DIMENSION_INDEX["block"]] * 0.60
        + values[_DIMENSION_INDEX["draw"]] * 0.20
        + (0.14 if cost_value <= 1.0 else 0.0)
        - (0.10 if snapshot.card_type == "Power" and cost_value >= 2.0 else 0.0)
    )
    mid = (
        values[_DIMENSION_INDEX["frontload"]] * 0.24
        + values[_DIMENSION_INDEX["block"]] * 0.26
        + values[_DIMENSION_INDEX["draw"]] * 0.18
        + values[_DIMENSION_INDEX["energy"]] * 0.20
        + values[_DIMENSION_INDEX["utility"]] * 0.18
        + values[_DIMENSION_INDEX["scaling"]] * 0.16
    )
    late = (
        values[_DIMENSION_INDEX["scaling"]] * 0.82
        + values[_DIMENSION_INDEX["draw"]] * 0.16
        + values[_DIMENSION_INDEX["energy"]] * 0.22
        + (0.12 if snapshot.card_type == "Power" else 0.0)
    )
    values[_DIMENSION_INDEX["early"]] += min(2.0, max(0.0, early))
    values[_DIMENSION_INDEX["mid"]] += min(2.0, max(0.0, mid))
    values[_DIMENSION_INDEX["late"]] += min(2.0, max(0.0, late))

    rarity_score = _CARD_RARITY_SCORE.get(snapshot.rarity, 0.0)
    type_score = _CARD_TYPE_SCORE.get(snapshot.card_type, 0.0)
    upgraded_bonus = 0.18 if snapshot.is_upgraded else 0.0
    basic_penalty = 0.12 if snapshot.rarity in {"Basic", "Starter"} else 0.0
    exhaust_penalty = 0.10 if snapshot.consumes_on_play and snapshot.card_type != "Power" else 0.0
    slow_power_penalty = 0.14 if snapshot.card_type == "Power" and cost_value >= 2.0 else 0.0
    base_quality = rarity_score + type_score + upgraded_bonus - basic_penalty - exhaust_penalty - slow_power_penalty
    return CardModel(snapshot=snapshot, vector=_freeze_vector(values), base_quality=base_quality)


def build_deck_profile(cards: tuple[CardSnapshot, ...]) -> DeckBuildProfile:
    if not cards:
        return DeckBuildProfile(
            cards=(),
            capability=tuple(0.0 for _ in CAPABILITY_DIMENSIONS),
            total_score=0.0,
            velocity=0.0,
            cohesion=0.0,
            quality=0.0,
            win_condition=0.0,
            attack_density=0.0,
            skill_density=0.0,
            power_density=0.0,
            draw_engine=0.0,
            energy_engine=0.0,
            exhaust_density=0.0,
            low_cost_density=0.0,
            high_cost_density=0.0,
            clutter_penalty=0.0,
        )
    models = tuple(_analyze_card(card) for card in cards)
    size = len(models)
    values = _empty_vector()
    quality_sum = 0.0
    attack_count = 0
    skill_count = 0
    power_count = 0
    exhaust_count = 0
    low_cost_count = 0
    high_cost_count = 0
    clutter_count = 0
    for model in models:
        _vector_add(values, model.vector)
        quality_sum += model.base_quality
        if model.snapshot.card_type == "Attack":
            attack_count += 1
        elif model.snapshot.card_type == "Skill":
            skill_count += 1
        elif model.snapshot.card_type == "Power":
            power_count += 1
        if model.snapshot.consumes_on_play:
            exhaust_count += 1
        cost_value = _safe_cost_value(model.snapshot.cost)
        if cost_value <= 1.0:
            low_cost_count += 1
        elif cost_value >= 2.0:
            high_cost_count += 1
        if model.snapshot.card_type in {"Curse", "Status"} or model.snapshot.rarity in {"Basic", "Starter"}:
            clutter_count += 1
    denominator = max(5.0, 3.2 + (size * 0.42) + (max(0, size - 18) * 0.24))
    capability = _freeze_vector([(value * 1.42) / denominator for value in values])
    attack_density = attack_count / max(1, size)
    skill_density = skill_count / max(1, size)
    power_density = power_count / max(1, size)
    exhaust_density = exhaust_count / max(1, size)
    low_cost_density = low_cost_count / max(1, size)
    high_cost_density = high_cost_count / max(1, size)
    clutter_penalty = min(1.35, clutter_count / max(6.0, size * 0.75))
    velocity = max(0.0, min(3.0, (0.55 * _vector_value(capability, "draw")) + (0.65 * _vector_value(capability, "energy")) + (0.38 * low_cost_density) - (0.35 * high_cost_density) - (0.28 * clutter_penalty)))
    cohesion = max(0.0, min(3.0, 0.35 + (0.42 * min(_vector_value(capability, "frontload"), _vector_value(capability, "consistency"))) + (0.34 * min(_vector_value(capability, "scaling"), _vector_value(capability, "draw") + _vector_value(capability, "energy"))) + (0.22 * min(_vector_value(capability, "block"), _vector_value(capability, "utility") + _vector_value(capability, "consistency"))) - (0.25 * clutter_penalty)))
    quality = max(0.0, min(3.0, (quality_sum * 1.15 / max(4.2, size * 0.58)) + min(0.45, power_count * 0.08) - (0.22 * clutter_penalty)))
    win_condition = max(0.0, min(3.0, max(_vector_value(capability, "frontload") + (0.18 * _vector_value(capability, "utility")), _vector_value(capability, "scaling") + (0.24 * _vector_value(capability, "consistency")) + (0.18 * _vector_value(capability, "energy")), _vector_value(capability, "aoe") + (0.34 * _vector_value(capability, "frontload")))))
    total_score = max(0.0, min(3.0, (0.30 * quality) + (0.22 * velocity) + (0.22 * cohesion) + (0.20 * win_condition) + (0.06 * (_vector_value(capability, "consistency") + _vector_value(capability, "scaling")))))
    return DeckBuildProfile(cards=cards, capability=capability, total_score=total_score, velocity=velocity, cohesion=cohesion, quality=quality, win_condition=win_condition, attack_density=attack_density, skill_density=skill_density, power_density=power_density, draw_engine=_vector_value(capability, "draw"), energy_engine=_vector_value(capability, "energy"), exhaust_density=exhaust_density, low_cost_density=low_cost_density, high_cost_density=high_cost_density, clutter_penalty=clutter_penalty)


def estimate_current_combat_demand(state: dict[str, object]) -> tuple[float, ...]:
    state_type = str(state.get("state_type", "unknown") or "")
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    enemies = [enemy for enemy in battle.get("enemies", []) if isinstance(enemy, dict) and float(enemy.get("hp", 0) or 0.0) > 0.0]
    if not enemies:
        return tuple(0.0 for _ in CAPABILITY_DIMENSIONS)
    values = _empty_vector()
    enemy_count = len(enemies)
    total_hp = sum(max(0.0, float(enemy.get("hp", 0) or 0.0) + float(enemy.get("block", 0) or 0.0)) for enemy in enemies)
    incoming = sum(_intents_total_damage(enemy) for enemy in enemies)
    elite_bonus = 0.18 if state_type == "elite" else 0.0
    boss_bonus = 0.35 if state_type == "boss" else 0.0
    values[_DIMENSION_INDEX["aoe"]] += min(2.2, max(0.0, (enemy_count - 1) * 0.42) + boss_bonus * 0.10)
    values[_DIMENSION_INDEX["block"]] += min(2.4, (incoming / 14.0) + elite_bonus + boss_bonus)
    values[_DIMENSION_INDEX["frontload"]] += min(2.0, (total_hp / max(20.0, 18.0 * enemy_count)) + (0.12 * enemy_count) + elite_bonus)
    values[_DIMENSION_INDEX["scaling"]] += min(2.3, (total_hp / 70.0) + (0.30 if state_type == "elite" else 0.0) + (0.65 if state_type == "boss" else 0.0))
    values[_DIMENSION_INDEX["consistency"]] += min(2.0, 0.30 + (0.12 * enemy_count) + elite_bonus + boss_bonus)
    values[_DIMENSION_INDEX["utility"]] += min(1.3, (0.15 if any(float(enemy.get("block", 0) or 0.0) > 0.0 for enemy in enemies) else 0.0) + (0.12 if any(_power_amount(enemy.get("status") if isinstance(enemy.get("status"), list) else [], ("\u6613\u4f24", "WEAK", "FRAIL")) > 0.0 for enemy in enemies) else 0.0))
    values[_DIMENSION_INDEX["early"]] += min(2.2, (0.65 * values[_DIMENSION_INDEX["frontload"]]) + (0.55 * values[_DIMENSION_INDEX["block"]]))
    values[_DIMENSION_INDEX["mid"]] += min(2.0, (0.28 * incoming / 10.0) + (0.18 * enemy_count) + elite_bonus)
    values[_DIMENSION_INDEX["late"]] += min(2.4, (0.75 * values[_DIMENSION_INDEX["scaling"]]) + boss_bonus + (0.12 * elite_bonus))
    encounter_profile = analyze_encounter_profile(state)
    strategy = encounter_profile.strategy
    values[_DIMENSION_INDEX["frontload"]] += (
        (0.28 * strategy_value(strategy, "burst_frontload"))
        + (0.10 * strategy_value(strategy, "focus_scaler"))
        + (0.10 * strategy_value(strategy, "focus_stealer"))
    )
    values[_DIMENSION_INDEX["aoe"]] += 0.42 * strategy_value(strategy, "aoe_clear")
    values[_DIMENSION_INDEX["block"]] += (
        (0.46 * strategy_value(strategy, "prioritize_block"))
        + (0.24 * strategy_value(strategy, "preserve_block"))
    )
    values[_DIMENSION_INDEX["draw"]] += (
        (0.18 * strategy_value(strategy, "status_cleanup"))
        + (0.12 * strategy_value(strategy, "attack_windowing"))
        + (0.08 * strategy_value(strategy, "resource_conservation"))
    )
    values[_DIMENSION_INDEX["energy"]] += 0.34 * strategy_value(strategy, "energy_efficiency")
    values[_DIMENSION_INDEX["scaling"]] += 0.20 * strategy_value(strategy, "resource_conservation")
    values[_DIMENSION_INDEX["consistency"]] += (
        (0.24 * strategy_value(strategy, "status_cleanup"))
        + (0.18 * strategy_value(strategy, "attack_windowing"))
    )
    values[_DIMENSION_INDEX["utility"]] += (
        (0.28 * strategy_value(strategy, "multi_hit"))
        + (0.24 * strategy_value(strategy, "status_cleanup"))
        + (0.12 * strategy_value(strategy, "focus_support"))
    )
    values[_DIMENSION_INDEX["early"]] += (
        (0.30 * strategy_value(strategy, "burst_frontload"))
        + (0.18 * strategy_value(strategy, "aoe_clear"))
    )
    values[_DIMENSION_INDEX["mid"]] += (
        (0.18 * strategy_value(strategy, "attack_windowing"))
        + (0.12 * strategy_value(strategy, "prioritize_block"))
    )
    values[_DIMENSION_INDEX["late"]] += (
        (0.22 * strategy_value(strategy, "resource_conservation"))
        + (0.10 * strategy_value(strategy, "focus_scaler"))
    )
    return _freeze_vector(values)


def update_observed_monster_history(history: tuple[float, ...] | None, current_demand: tuple[float, ...], decay: float = 0.90) -> tuple[float, ...]:
    if history is None or len(history) != len(CAPABILITY_DIMENSIONS):
        history = tuple(0.0 for _ in CAPABILITY_DIMENSIONS)
    values = _empty_vector()
    for index in range(len(CAPABILITY_DIMENSIONS)):
        values[index] = (float(history[index]) * decay) + (float(current_demand[index]) * (1.0 - decay))
    return _freeze_vector(values)


def _node_demand_vector(act_id: str, node_type: str) -> tuple[float, ...]:
    base = dict(_DEFAULT_NODE_DEMANDS.get(node_type, _DEFAULT_NODE_DEMANDS["Monster"]))
    adjustments = _ACT_DEMAND_ADJUSTMENTS.get(act_id, {}).get(node_type, {})
    for name, value in adjustments.items():
        base[name] = base.get(name, 0.0) + float(value)
    return _vector_from_mapping(base)


def estimate_future_demand(*, act_id: str, floor: int, hp_ratio: float, route_profile: dict[str, float] | None = None, observed_history: tuple[float, ...] | None = None) -> tuple[float, ...]:
    values = _empty_vector()
    if route_profile:
        weights = {"Monster": 0.58 + (0.10 * max(0.0, 0.75 - hp_ratio)), "Elite": 0.92 + (0.20 * max(0.0, 0.85 - hp_ratio)), "Boss": 1.28}
        for node_type in _NODE_TYPE_KEYS:
            node_vector = _node_demand_vector(act_id, node_type)
            if node_type == "Monster":
                total = float(route_profile.get("monster_count", 0.0) or 0.0)
            elif node_type == "Elite":
                total = float(route_profile.get("elite_count", 0.0) or 0.0)
            else:
                total = max(1.0, float(route_profile.get("boss_count", 1.0) or 1.0))
            _vector_add(values, node_vector, weights[node_type] * max(0.0, total))
        early_monsters = float(route_profile.get("early_monsters", 0.0) or 0.0)
        early_elites = float(route_profile.get("early_elites", 0.0) or 0.0)
        late_monsters = float(route_profile.get("late_monsters", 0.0) or 0.0)
        late_elites = float(route_profile.get("late_elites", 0.0) or 0.0)
        future_easy_monsters = float(route_profile.get("future_easy_monsters", 0.0) or 0.0)
        future_hard_monsters = float(route_profile.get("future_hard_monsters", 0.0) or 0.0)
        future_question_hard_nodes = float(route_profile.get("future_question_hard_nodes", 0.0) or 0.0)
        difficulty_load = float(route_profile.get("difficulty_load", 0.0) or 0.0)
        values[_DIMENSION_INDEX["early"]] += (early_monsters * 0.12) + (early_elites * 0.18)
        values[_DIMENSION_INDEX["late"]] += (late_monsters * 0.05) + (late_elites * 0.10) + 0.18
        values[_DIMENSION_INDEX["frontload"]] += early_monsters * 0.10
        values[_DIMENSION_INDEX["block"]] += early_elites * 0.12
        values[_DIMENSION_INDEX["scaling"]] += late_elites * 0.08
        values[_DIMENSION_INDEX["frontload"]] += (future_easy_monsters * 0.06) + (future_hard_monsters * 0.14)
        values[_DIMENSION_INDEX["block"]] += (future_hard_monsters * 0.18) + (future_question_hard_nodes * 0.10)
        values[_DIMENSION_INDEX["consistency"]] += (future_question_hard_nodes * 0.08) + (difficulty_load * 0.01)
        values[_DIMENSION_INDEX["mid"]] += (future_hard_monsters * 0.12) + (future_question_hard_nodes * 0.08)
        values[_DIMENSION_INDEX["late"]] += future_hard_monsters * 0.04
    else:
        progress = max(0.0, min(1.0, (max(1, int(floor)) - 1.0) / 16.0))
        _vector_add(values, _node_demand_vector(act_id, "Monster"), 1.15 - (0.20 * progress))
        _vector_add(values, _node_demand_vector(act_id, "Elite"), 0.72 + (0.05 * progress))
        _vector_add(values, _node_demand_vector(act_id, "Boss"), 0.92 + (0.58 * progress))
        values[_DIMENSION_INDEX["early"]] += 0.18 * (1.0 - progress)
        values[_DIMENSION_INDEX["late"]] += 0.22 * progress
    frozen = _freeze_vector(values)
    if observed_history is not None:
        return _blend_vectors(frozen, observed_history, 0.84, 0.16)
    return frozen


def estimate_boss_threshold(act_id: str, floor: int, hp_ratio: float) -> float:
    boss_demand = _node_demand_vector(act_id, "Boss")
    progress = max(0.0, min(1.0, (max(1, int(floor)) - 1.0) / 16.0))
    demand_pressure = _weighted_vector_norm(boss_demand, scale=7.2, clamp=1.5)
    threshold = 1.34 + (0.56 * demand_pressure) + (0.34 * progress) + (0.10 * max(0.0, 0.72 - hp_ratio))
    return max(1.2, min(3.4, threshold))


def estimate_boss_strength(deck_profile: DeckBuildProfile, boss_demand: tuple[float, ...]) -> float:
    alignment = _weighted_alignment(deck_profile.capability, boss_demand)
    capability_score = _weighted_vector_norm(deck_profile.capability, scale=7.2, clamp=1.5)
    strength = (
        (0.72 * deck_profile.total_score)
        + (0.74 * alignment)
        + (0.18 * deck_profile.win_condition)
        + (0.10 * deck_profile.quality)
        + (0.08 * deck_profile.velocity)
        + (0.06 * capability_score)
    )
    return max(0.0, min(3.5, strength))


def _build_strategy_context_from_profile(
    *,
    deck_profile: DeckBuildProfile,
    act_id: str,
    floor: int,
    player_hp: float,
    player_max_hp: float,
    player_gold: float,
    current_combat_demand: tuple[float, ...],
    current_mechanics: tuple[float, ...],
    current_strategy_demand: tuple[float, ...],
    observed_monster_history: tuple[float, ...] | None = None,
    route_profile: dict[str, float] | None = None,
) -> BuildStrategyContext:
    hp_ratio = player_hp / max(player_max_hp, 1.0)
    future_demand = estimate_future_demand(
        act_id=act_id,
        floor=floor,
        hp_ratio=hp_ratio,
        route_profile=route_profile,
        observed_history=observed_monster_history,
    )
    weighted_demand = _blend_vectors(
        future_demand,
        current_combat_demand,
        0.82,
        0.18 if any(current_combat_demand) else 0.0,
    )
    gap = _max_positive_gap(weighted_demand, deck_profile.capability)
    signed_gap = _signed_gap(weighted_demand, deck_profile.capability)
    boss_demand = _node_demand_vector(act_id, "Boss")
    boss_gap = _max_positive_gap(boss_demand, deck_profile.capability)
    boss_strength = estimate_boss_strength(deck_profile, boss_demand)
    boss_threshold = estimate_boss_threshold(act_id, floor, hp_ratio)
    boss_strength_gap = max(0.0, boss_threshold - boss_strength)
    boss_readiness = boss_strength / max(0.8, boss_threshold)
    return BuildStrategyContext(
        act_id=act_id,
        floor=floor,
        player_hp=player_hp,
        player_max_hp=player_max_hp,
        hp_ratio=hp_ratio,
        player_gold=player_gold,
        deck_profile=deck_profile,
        current_combat_demand=current_combat_demand,
        future_demand=future_demand,
        weighted_demand=weighted_demand,
        gap=gap,
        signed_gap=signed_gap,
        primary_shortfalls=_top_dimensions(gap),
        current_mechanics=current_mechanics,
        current_strategy_demand=current_strategy_demand,
        boss_demand=boss_demand,
        boss_gap=boss_gap,
        boss_shortfalls=_top_dimensions(boss_gap),
        boss_strength=boss_strength,
        boss_threshold=boss_threshold,
        boss_readiness=boss_readiness,
        boss_strength_gap=boss_strength_gap,
    )


def build_strategy_context(*, state: dict[str, object], observed_cards: tuple[CardSnapshot, ...] = (), observed_monster_history: tuple[float, ...] | None = None, route_profile: dict[str, float] | None = None) -> BuildStrategyContext:
    deck_cards = observed_cards or collect_visible_deck_snapshots(state)
    deck_profile = build_deck_profile(deck_cards)
    run = state.get("run") if isinstance(state.get("run"), dict) else {}
    player = state.get("player") if isinstance(state.get("player"), dict) else {}
    try:
        player_hp = float(player.get("hp", 0) or 0.0)
    except (TypeError, ValueError):
        player_hp = 0.0
    try:
        player_max_hp = max(1.0, float(player.get("max_hp", 0) or 0.0))
    except (TypeError, ValueError):
        player_max_hp = 1.0
    try:
        player_gold = float(player.get("gold", run.get("gold", 0)) or 0.0)
    except (TypeError, ValueError):
        player_gold = 0.0
    encounter_profile = analyze_encounter_profile(state)
    current_combat_demand = estimate_current_combat_demand(state)
    return _build_strategy_context_from_profile(
        deck_profile=deck_profile,
        act_id=str(run.get("act_id", "") or ""),
        floor=int(run.get("floor", 0) or 0),
        player_hp=player_hp,
        player_max_hp=player_max_hp,
        player_gold=player_gold,
        current_combat_demand=current_combat_demand,
        current_mechanics=encounter_profile.mechanics,
        current_strategy_demand=encounter_profile.strategy,
        observed_monster_history=observed_monster_history,
        route_profile=route_profile,
    )


def score_route_fit(context: BuildStrategyContext, route_profile: dict[str, float], act_id: str, floor: int, hp_ratio: float) -> float:
    route_demand = estimate_future_demand(act_id=act_id, floor=floor, hp_ratio=hp_ratio, route_profile=route_profile, observed_history=context.current_combat_demand if any(context.current_combat_demand) else None)
    shortfall = 0.0
    oversupply = 0.0
    for name in CAPABILITY_DIMENSIONS:
        demand_value = _vector_value(route_demand, name)
        capability_value = _vector_value(context.deck_profile.capability, name)
        if demand_value > capability_value:
            shortfall += (demand_value - capability_value) * _DIMENSION_WEIGHTS[name]
        else:
            oversupply += (capability_value - demand_value) * _OVERSUPPLY_WEIGHTS[name]
    return max(-3.0, min(3.0, (1.15 - shortfall) + (oversupply * 0.08)))


def estimate_strength_economy(
    context: BuildStrategyContext,
    *,
    relic_strength: float = 0.0,
    resource_strength: float = 0.0,
) -> StrengthEconomy:
    current_strength = max(
        0.0,
        min(
            4.0,
            context.boss_strength + (0.28 * max(0.0, relic_strength)) + (0.22 * max(0.0, resource_strength)),
        ),
    )
    boss_gap = max(0.0, context.boss_threshold - current_strength)
    boss_readiness = current_strength / max(0.8, context.boss_threshold)
    gap_pressure = max(_gap_pressure(context.gap), _gap_pressure(context.boss_gap))
    need_pressure = max(0.0, min(1.5, boss_gap / max(0.85, context.boss_threshold)))
    low_hp_pressure = max(0.0, 0.72 - context.hp_ratio)
    hp_strength_value = max(
        0.75,
        min(
            2.5,
            0.88 + (0.62 * need_pressure) + (0.48 * low_hp_pressure) + (0.16 * gap_pressure),
        ),
    )
    gold_strength_ratio = max(
        0.0016,
        min(
            0.0060,
            0.0022
            + (0.0012 * need_pressure)
            + (0.0006 * gap_pressure)
            + (0.00025 * max(0.0, 0.55 - min(1.0, context.deck_profile.quality / 2.0))),
        ),
    )
    monster_expected_gain = max(
        0.05,
        min(
            0.24,
            0.08
            + (0.06 * need_pressure)
            + (0.025 * _vector_value(context.gap, "frontload"))
            + (0.022 * _vector_value(context.gap, "consistency"))
            + (0.018 * _vector_value(context.gap, "aoe"))
            - (0.022 * context.deck_profile.clutter_penalty),
        ),
    )
    monster_expected_hp_loss = max(
        0.035,
        min(
            0.24,
            0.050
            + (0.040 * max(0.0, 1.0 - boss_readiness))
            + (0.022 * _vector_value(context.gap, "block"))
            + (0.016 * _vector_value(context.gap, "frontload")),
        ),
    )
    elite_expected_gain = max(
        0.34,
        min(
            0.92,
            0.46
            + (0.14 * need_pressure)
            + (0.08 * _vector_value(context.boss_gap, "scaling"))
            + (0.06 * _vector_value(context.boss_gap, "consistency"))
            + (0.05 * _vector_value(context.boss_gap, "frontload")),
        ),
    )
    elite_expected_hp_loss = max(
        0.10,
        min(
            0.38,
            0.120
            + (0.070 * max(0.0, 1.0 - boss_readiness))
            + (0.036 * _vector_value(context.gap, "block"))
            + (0.022 * _vector_value(context.gap, "frontload")),
        ),
    )
    question_expected_gain = max(
        0.26,
        min(
            0.76,
            0.34
            + (0.08 * need_pressure)
            + (0.05 * _vector_value(context.gap, "consistency"))
            + (0.03 * _vector_value(context.gap, "utility")),
        ),
    )
    question_expected_hp_loss = max(
        0.015,
        min(
            0.12,
            0.025
            + (0.014 * need_pressure)
            + (0.010 * _vector_value(context.gap, "block")),
        ),
    )
    rest_upgrade_gain = max(
        0.18,
        min(
            0.44,
            0.22
            + (0.10 * need_pressure)
            + (0.06 * context.deck_profile.win_condition)
            + (0.03 * context.deck_profile.quality)
            - (0.03 * context.deck_profile.clutter_penalty),
        ),
    )
    rest_base_heal_ratio = 0.30
    rest_heal_gain = min(rest_base_heal_ratio, max(0.0, (1.0 - context.hp_ratio) + 0.04)) * hp_strength_value
    shop_expected_gain = max(
        0.05,
        min(
            1.0,
            0.06 + min(0.85, context.player_gold * gold_strength_ratio),
        ),
    )
    treasure_expected_gain = max(
        0.52,
        min(
            1.00,
            0.60
            + (0.12 * need_pressure)
            + (0.08 * _vector_value(context.boss_gap, "consistency"))
            + (0.04 * max(0.0, 1.0 - boss_readiness)),
        ),
    )
    event_expected_gain = max(
        0.24,
        min(
            0.68,
            0.26
            + (0.06 * need_pressure)
            + (0.10 * question_expected_gain)
            + (0.08 * treasure_expected_gain)
            + (0.06 * shop_expected_gain)
            + (0.04 * rest_upgrade_gain),
        ),
    )
    event_expected_hp_loss = max(
        0.01,
        min(
            0.12,
            0.022 + (0.012 * need_pressure) + (0.011 * _vector_value(context.gap, "block")),
        ),
    )
    return StrengthEconomy(
        current_strength=current_strength,
        boss_threshold=context.boss_threshold,
        boss_readiness=boss_readiness,
        boss_gap=boss_gap,
        hp_strength_value=hp_strength_value,
        gold_strength_ratio=gold_strength_ratio,
        monster_expected_gain=monster_expected_gain,
        monster_expected_hp_loss=monster_expected_hp_loss,
        elite_expected_gain=elite_expected_gain,
        elite_expected_hp_loss=elite_expected_hp_loss,
        question_expected_gain=question_expected_gain,
        question_expected_hp_loss=question_expected_hp_loss,
        rest_upgrade_gain=rest_upgrade_gain,
        rest_heal_gain=rest_heal_gain,
        rest_base_heal_ratio=rest_base_heal_ratio,
        event_expected_gain=event_expected_gain,
        event_expected_hp_loss=event_expected_hp_loss,
        shop_expected_gain=shop_expected_gain,
        treasure_expected_gain=treasure_expected_gain,
    )


def estimate_hp_adjusted_strength(current_strength: float, hp_ratio: float) -> float:
    bounded_strength = max(0.0, min(4.5, float(current_strength)))
    bounded_hp_ratio = max(0.0, min(1.0, float(hp_ratio)))
    hp_multiplier = 0.45 + (0.55 * bounded_hp_ratio)
    return max(0.0, min(4.5, bounded_strength * hp_multiplier))


def _clamped_margin(value: float, bound: float = 2.0) -> float:
    return max(-bound, min(bound, float(value)))


def evaluate_route_strength(
    context: BuildStrategyContext,
    route_profile: dict[str, float],
    *,
    act_id: str,
    floor: int,
    hp_ratio: float,
    relic_strength: float = 0.0,
    resource_strength: float = 0.0,
) -> RouteStrengthEvaluation:
    fit_score = score_route_fit(context, route_profile, act_id, floor, hp_ratio)
    economy = estimate_strength_economy(
        context,
        relic_strength=relic_strength,
        resource_strength=resource_strength,
    )
    early_monsters = max(0.0, float(route_profile.get("early_monsters", 0.0) or 0.0))
    late_monsters = max(0.0, float(route_profile.get("late_monsters", 0.0) or 0.0))
    early_elites = max(0.0, float(route_profile.get("early_elites", 0.0) or 0.0))
    late_elites = max(0.0, float(route_profile.get("late_elites", 0.0) or 0.0))
    question_count = max(0.0, float(route_profile.get("question_count", 0.0) or 0.0))
    future_easy_monsters = max(0.0, float(route_profile.get("future_easy_monsters", 0.0) or 0.0))
    future_hard_monsters = max(0.0, float(route_profile.get("future_hard_monsters", 0.0) or 0.0))
    future_question_easy_nodes = max(0.0, float(route_profile.get("future_question_easy_nodes", 0.0) or 0.0))
    future_question_hard_nodes = max(0.0, float(route_profile.get("future_question_hard_nodes", 0.0) or 0.0))
    difficulty_load = max(0.0, float(route_profile.get("difficulty_load", 0.0) or 0.0))
    event_count = max(0.0, float(route_profile.get("event_count", 0.0) or 0.0))
    rest_count = max(0.0, float(route_profile.get("rest_count", 0.0) or 0.0))
    shop_count = max(0.0, float(route_profile.get("shop_count", 0.0) or 0.0))
    treasure_count = max(0.0, float(route_profile.get("treasure_count", 0.0) or 0.0))
    early_combats = max(0.0, float(route_profile.get("early_combats", 0.0) or 0.0))
    total_monsters = early_monsters + late_monsters
    total_elites = early_elites + late_elites

    elite_value_factor = 0.62 + (0.26 * min(1.0, economy.boss_readiness)) + (0.12 * min(1.0, hp_ratio))
    monster_gain = economy.monster_expected_gain * (
        (future_easy_monsters * 0.92)
        + (future_hard_monsters * 0.20)
    )
    elite_gain = economy.elite_expected_gain * elite_value_factor * ((early_elites * 0.74) + (late_elites * 0.92))
    question_gain = economy.question_expected_gain * (
        (future_question_easy_nodes * 0.88)
        + (future_question_hard_nodes * 0.62)
    )
    event_gain = event_count * economy.event_expected_gain
    treasure_gain = treasure_count * economy.treasure_expected_gain

    projected_gold = (
        context.player_gold
        + (11.0 * future_easy_monsters)
        + (8.0 * future_hard_monsters)
        + (24.0 * total_elites)
        + (6.0 * future_question_easy_nodes)
        + (5.0 * future_question_hard_nodes)
        + (5.0 * event_count)
        + (10.0 * treasure_count)
    )
    shop_spend_cap = shop_count * (110.0 + (24.0 * min(1.5, economy.boss_gap)))
    shop_spend = min(projected_gold, shop_spend_cap)
    shop_gain = (shop_count * 0.05) + (shop_spend * economy.gold_strength_ratio)

    hp_loss_ratio = (
        economy.monster_expected_hp_loss * ((future_easy_monsters * 0.92) + (future_hard_monsters * 1.95))
        + economy.elite_expected_hp_loss * ((early_elites * 1.12) + late_elites)
        + economy.question_expected_hp_loss * ((future_question_easy_nodes * 0.95) + (future_question_hard_nodes * 1.45))
        + economy.event_expected_hp_loss * event_count
    )
    hp_loss_ratio += max(0.0, early_combats - 1.75) * 0.020 * max(0.0, 1.08 - economy.boss_readiness)
    hp_loss_ratio += future_hard_monsters * 0.012 * max(0.0, 1.0 - hp_ratio)
    hp_after_loss = max(0.05, hp_ratio - hp_loss_ratio)
    heal_preference = 1.0 if hp_ratio < 0.70 else max(0.0, min(1.0, (0.62 - hp_after_loss) / 0.20))
    per_rest_heal_ratio = min(economy.rest_base_heal_ratio, max(0.0, 1.0 - hp_after_loss))
    per_rest_heal_gain = per_rest_heal_ratio * economy.hp_strength_value
    rest_gain = rest_count * (
        (heal_preference * per_rest_heal_gain)
        + ((1.0 - heal_preference) * economy.rest_upgrade_gain)
    )
    projected_hp_ratio = min(1.0, hp_after_loss + (rest_count * heal_preference * per_rest_heal_ratio))
    deck_strength = economy.current_strength
    deck_hp_strength = estimate_hp_adjusted_strength(deck_strength, hp_ratio)
    next_risk_strength = max(0.0, float(route_profile.get("risk_1_strength", 0.0) or 0.0))
    next_risk_margin = deck_hp_strength - next_risk_strength
    next_step_feasible = 1.0 if next_risk_margin >= 0.0 else 0.0

    node_gain = monster_gain + elite_gain + question_gain + event_gain + treasure_gain + shop_gain + rest_gain
    node_loss = hp_loss_ratio * economy.hp_strength_value
    net_strength_delta = node_gain - node_loss
    projected_strength = deck_strength + net_strength_delta
    projected_deck_hp_strength = estimate_hp_adjusted_strength(projected_strength, projected_hp_ratio)
    projected_gap = max(0.0, economy.boss_threshold - projected_strength)
    gap_closure = economy.boss_gap - projected_gap
    boss_margin = projected_deck_hp_strength - economy.boss_threshold
    boss_feasible = 1.0 if boss_margin >= 0.0 else 0.0
    path_feasible = 1.0 if min(next_risk_margin, boss_margin) >= 0.0 else 0.0
    useful_net_delta = min(net_strength_delta, economy.boss_gap + 0.65) if net_strength_delta > 0.0 else net_strength_delta
    survival_penalty = max(0.0, 0.42 - projected_hp_ratio) * (2.10 + economy.boss_gap)
    excess_early_penalty = max(0.0, early_combats - 1.75) * 0.16 * (
        0.70 + max(0.0, 0.86 - hp_ratio) + max(0.0, 0.92 - economy.boss_readiness)
    )
    monster_overflow_penalty = max(0.0, total_monsters - 2.0) * (
        0.12
        + (0.08 * max(0.0, 1.0 - economy.boss_readiness))
        + (0.06 * max(0.0, 0.78 - hp_ratio))
    )
    hard_monster_penalty = (future_hard_monsters * 0.22) * (
        0.55 + max(0.0, 0.88 - hp_ratio) + max(0.0, 0.92 - economy.boss_readiness)
    )
    hard_question_penalty = (future_question_hard_nodes * 0.12) * (
        0.45 + max(0.0, 0.84 - hp_ratio) + max(0.0, 0.86 - economy.boss_readiness)
    )
    difficulty_load_penalty = difficulty_load * 0.018 * (
        0.40 + max(0.0, 0.80 - hp_ratio) + max(0.0, 0.85 - economy.boss_readiness)
    )
    elite_risk_penalty = ((early_elites * 0.18) + (late_elites * 0.12)) * max(
        0.0,
        0.72 - max(hp_ratio, economy.boss_readiness),
    )
    next_risk_margin_bonus = 0.22 * max(0.0, next_risk_margin)
    next_risk_margin_penalty = 0.96 * max(0.0, -next_risk_margin)
    surplus_penalty = max(0.0, projected_strength - (economy.boss_threshold + 0.40)) * (
        0.30 + (0.03 * min(6.0, late_monsters + late_elites))
    )
    growth_score = (
        (0.26 * fit_score)
        + (1.08 * gap_closure)
        + (0.22 * useful_net_delta)
        + next_risk_margin_bonus
        - survival_penalty
        - excess_early_penalty
        - monster_overflow_penalty
        - hard_monster_penalty
        - hard_question_penalty
        - difficulty_load_penalty
        - elite_risk_penalty
        - surplus_penalty
    )
    constraint_score = (
        (4.60 * next_step_feasible)
        + (2.80 * boss_feasible)
        + (1.20 * path_feasible)
        + (0.72 * _clamped_margin(next_risk_margin))
        + (0.66 * _clamped_margin(boss_margin))
        - (1.80 * max(0.0, -next_risk_margin))
        - (1.35 * max(0.0, -boss_margin))
        - (0.96 * projected_gap)
        - next_risk_margin_penalty
    )
    total_score = constraint_score + growth_score
    return RouteStrengthEvaluation(
        deck_strength=deck_strength,
        deck_hp_strength=deck_hp_strength,
        next_risk_strength=next_risk_strength,
        projected_deck_hp_strength=projected_deck_hp_strength,
        boss_margin=boss_margin,
        next_step_feasible=next_step_feasible,
        boss_feasible=boss_feasible,
        path_feasible=path_feasible,
        total_score=total_score,
        constraint_score=constraint_score,
        growth_score=growth_score,
    )


def _context_with_projected_deck(
    context: BuildStrategyContext,
    deck_profile: DeckBuildProfile,
) -> BuildStrategyContext:
    gap = _max_positive_gap(context.weighted_demand, deck_profile.capability)
    boss_gap = _max_positive_gap(context.boss_demand, deck_profile.capability)
    boss_strength = estimate_boss_strength(deck_profile, context.boss_demand)
    boss_strength_gap = max(0.0, context.boss_threshold - boss_strength)
    boss_readiness = boss_strength / max(0.8, context.boss_threshold)
    return BuildStrategyContext(
        act_id=context.act_id,
        floor=context.floor,
        player_hp=context.player_hp,
        player_max_hp=context.player_max_hp,
        hp_ratio=context.hp_ratio,
        player_gold=context.player_gold,
        deck_profile=deck_profile,
        current_combat_demand=context.current_combat_demand,
        future_demand=context.future_demand,
        weighted_demand=context.weighted_demand,
        gap=gap,
        signed_gap=_signed_gap(context.weighted_demand, deck_profile.capability),
        primary_shortfalls=_top_dimensions(gap),
        current_mechanics=context.current_mechanics,
        current_strategy_demand=context.current_strategy_demand,
        boss_demand=context.boss_demand,
        boss_gap=boss_gap,
        boss_shortfalls=_top_dimensions(boss_gap),
        boss_strength=boss_strength,
        boss_threshold=context.boss_threshold,
        boss_readiness=boss_readiness,
        boss_strength_gap=boss_strength_gap,
    )


def _selection_target_demand(context: BuildStrategyContext, temporary: bool) -> tuple[float, ...]:
    if not temporary:
        return context.weighted_demand
    if any(context.current_combat_demand):
        return _blend_vectors(context.current_combat_demand, context.future_demand, 0.80, 0.20)
    return context.future_demand


def _raw_self_gain(model: CardModel) -> float:
    return (
        model.base_quality * 0.55
        + (_vector_value(model.vector, "frontload") * 0.24)
        + (_vector_value(model.vector, "aoe") * 0.24)
        + (_vector_value(model.vector, "block") * 0.26)
        + (_vector_value(model.vector, "draw") * 0.18)
        + (_vector_value(model.vector, "energy") * 0.20)
        + (_vector_value(model.vector, "scaling") * 0.22)
        + (_vector_value(model.vector, "consistency") * 0.22)
    )


def _candidate_immediate_bonus(model: CardModel, context: BuildStrategyContext, temporary: bool) -> float:
    bonus = 0.0
    if temporary:
        for name in CAPABILITY_DIMENSIONS:
            bonus += min(_vector_value(model.vector, name), _vector_value(context.current_combat_demand, name)) * _DIMENSION_WEIGHTS[name] * 0.12
    strategy_vector = card_strategy_vector_from_snapshot(model.snapshot)
    strategy_scale = 0.20 if temporary else 0.05
    bonus += strategy_alignment_score(strategy_vector, context.current_strategy_demand) * strategy_scale
    if model.snapshot.card_type == "Skill":
        avoid_skill_spam = strategy_value(context.current_strategy_demand, "avoid_skill_spam")
        if avoid_skill_spam > 0.0 and strategy_value(strategy_vector, "prioritize_block") < 0.25 and strategy_value(strategy_vector, "status_cleanup") < 0.25:
            bonus -= avoid_skill_spam * (0.16 if temporary else 0.08)
    if model.snapshot.card_type == "Power":
        avoid_power_spam = strategy_value(context.current_strategy_demand, "avoid_power_spam")
        if avoid_power_spam > 0.0:
            bonus -= avoid_power_spam * (0.18 if temporary else 0.10)
    return bonus


def _candidate_oversupply_penalty(model: CardModel, context: BuildStrategyContext, target_demand: tuple[float, ...]) -> float:
    penalty = 0.0
    for name in CAPABILITY_DIMENSIONS:
        capability_value = _vector_value(context.deck_profile.capability, name)
        demand_value = _vector_value(target_demand, name)
        surplus = max(0.0, capability_value - demand_value)
        if surplus <= 0.0:
            continue
        penalty += surplus * _vector_value(model.vector, name) * _OVERSUPPLY_WEIGHTS[name]
    return penalty * 0.22


def _candidate_dilution_penalty(model: CardModel, context: BuildStrategyContext, temporary: bool) -> float:
    if temporary:
        return 0.0
    size = len(context.deck_profile.cards)
    cost_value = _safe_cost_value(model.snapshot.cost)
    slow_penalty = 0.0
    if cost_value >= 2.0 and _vector_value(context.gap, "early") > 0.22:
        slow_penalty += 0.14 + (0.06 * cost_value)
    if model.snapshot.card_type == "Power" and _vector_value(context.gap, "early") > 0.30 and _vector_value(model.vector, "early") < 0.20:
        slow_penalty += 0.18
    dilution = max(0.0, size - 18) * 0.018
    return slow_penalty + dilution + (0.06 * context.deck_profile.clutter_penalty)


def _candidate_gap_closure(model: CardModel, context: BuildStrategyContext, target_demand: tuple[float, ...]) -> tuple[float, str]:
    gap = _max_positive_gap(target_demand, context.deck_profile.capability)
    score = 0.0
    primary_gap = ""
    primary_value = 0.0
    for name in CAPABILITY_DIMENSIONS:
        gain = min(_vector_value(model.vector, name), _vector_value(gap, name))
        weighted = gain * _DIMENSION_WEIGHTS[name]
        score += weighted
        if weighted > primary_value:
            primary_value = weighted
            primary_gap = name
    return score, primary_gap


def evaluate_candidate_cards(snapshots: tuple[CardSnapshot, ...], context: BuildStrategyContext, *, temporary: bool = False) -> tuple[CandidateEvaluation, ...]:
    target_demand = _selection_target_demand(context, temporary)
    results: list[CandidateEvaluation] = []
    for snapshot in snapshots:
        model = _analyze_card(snapshot)
        gap_closure, primary_gap = _candidate_gap_closure(model, context, target_demand)
        projected_profile = build_deck_profile(context.deck_profile.cards + (snapshot,))
        deck_total_delta = projected_profile.total_score - context.deck_profile.total_score
        multiplier_score = deck_total_delta - (_raw_self_gain(model) * 0.12)
        oversupply_penalty = _candidate_oversupply_penalty(model, context, target_demand)
        dilution_penalty = _candidate_dilution_penalty(model, context, temporary)
        immediate_bonus = _candidate_immediate_bonus(model, context, temporary)
        total_score = gap_closure + (deck_total_delta * 1.15) + (multiplier_score * 0.55) + immediate_bonus + (model.base_quality * 0.22) - oversupply_penalty - dilution_penalty
        results.append(CandidateEvaluation(label=snapshot.name or snapshot.card_id or snapshot.token, total_score=total_score, gap_closure=gap_closure, deck_total_delta=deck_total_delta, multiplier_score=multiplier_score, oversupply_penalty=oversupply_penalty, dilution_penalty=dilution_penalty, immediate_bonus=immediate_bonus, primary_gap=primary_gap))
    return tuple(results)


def evaluate_bundle_candidates(bundles: tuple[tuple[CardSnapshot, ...], ...], context: BuildStrategyContext, *, temporary: bool = False) -> tuple[CandidateEvaluation, ...]:
    results: list[CandidateEvaluation] = []
    for bundle_index, bundle in enumerate(bundles):
        if not bundle:
            results.append(CandidateEvaluation(label=f"bundle_{bundle_index}", total_score=-1.0, gap_closure=0.0, deck_total_delta=0.0, multiplier_score=0.0, oversupply_penalty=0.0, dilution_penalty=0.0, immediate_bonus=0.0, primary_gap=""))
            continue
        temporary_cards = context.deck_profile.cards
        working_context = context
        total_score = 0.0
        total_gap = 0.0
        total_immediate = 0.0
        total_oversupply = 0.0
        total_dilution = 0.0
        primary_gap = ""
        primary_gap_value = 0.0
        for position, snapshot in enumerate(bundle):
            evaluation = evaluate_candidate_cards((snapshot,), working_context, temporary=temporary)[0]
            decay = 1.0 if position == 0 else 0.82
            total_score += evaluation.total_score * decay
            total_gap += evaluation.gap_closure * decay
            total_immediate += evaluation.immediate_bonus * decay
            total_oversupply += evaluation.oversupply_penalty * decay
            total_dilution += evaluation.dilution_penalty * decay
            if evaluation.gap_closure > primary_gap_value:
                primary_gap = evaluation.primary_gap
                primary_gap_value = evaluation.gap_closure
            if not temporary:
                temporary_cards = temporary_cards + (snapshot,)
                projected_profile = build_deck_profile(temporary_cards)
                working_context = _context_with_projected_deck(context, projected_profile)
        projected_profile = build_deck_profile(context.deck_profile.cards + bundle) if not temporary else context.deck_profile
        deck_total_delta = projected_profile.total_score - context.deck_profile.total_score if not temporary else 0.0
        multiplier_score = deck_total_delta - (sum(_raw_self_gain(_analyze_card(snapshot)) for snapshot in bundle) * 0.10)
        results.append(CandidateEvaluation(label=f"bundle_{bundle_index}", total_score=total_score + (deck_total_delta * 0.55) + (multiplier_score * 0.30), gap_closure=total_gap, deck_total_delta=deck_total_delta, multiplier_score=multiplier_score, oversupply_penalty=total_oversupply, dilution_penalty=total_dilution, immediate_bonus=total_immediate, primary_gap=primary_gap))
    return tuple(results)


def score_to_bias_map(scores: dict[int, float], scale: float = 1.85) -> dict[int, float]:
    if not scores:
        return {}
    values = tuple(float(value) for value in scores.values())
    best = max(values)
    worst = min(values)
    mean = sum(values) / max(1, len(values))
    spread = max(0.35, best - worst, max(abs(best), abs(worst)) * 0.35)
    result: dict[int, float] = {}
    for action_id, score in scores.items():
        bias = ((float(score) - mean) / spread) * scale
        result[action_id] = max(-2.5, min(2.5, bias))
    return result


def skip_card_reward_score(best_candidate_score: float) -> float:
    if best_candidate_score <= 0.10:
        return 0.65 + (0.25 * max(0.0, 0.10 - best_candidate_score))
    return -min(2.0, best_candidate_score * 0.85)
