import re
import time
from dataclasses import dataclass

from .action_space import action_index
from .bridge import STS2Bridge, STS2BridgeError
from .combat_tactical_solver import analyze_combat_turn
from .deck_building import (
    CAPABILITY_DIMENSIONS,
    BuildStrategyContext,
    CardSnapshot,
    build_strategy_context,
    card_snapshot_from_card_dict,
    card_snapshot_from_shop_item,
    collect_visible_deck_snapshots,
    estimate_current_combat_demand,
    estimate_hp_adjusted_strength,
    estimate_strength_economy,
    evaluate_route_strength,
    evaluate_bundle_candidates,
    evaluate_candidate_cards,
    score_to_bias_map,
    skip_card_reward_score,
    update_observed_monster_history,
)
from .monster_strategy import (
    MECHANIC_DIMENSIONS,
    STRATEGY_DIMENSIONS,
    analyze_encounter_profile,
    card_strategy_vector_from_raw_card,
    mechanic_value,
    strategy_alignment_score,
    strategy_value,
)
from .semantics import (
    CARD_SELECT_SLOT_COUNT,
    COMBAT_ROUND_BUCKETS,
    CONCEPT_VOCAB_SIZE,
    EVENT_OPTION_COUNT,
    HAND_ORDER_PRIORITY_SCALE,
    HAND_SLOT_COUNT,
    HAND_SELECT_SLOT_COUNT,
    MAP_NODE_TYPE_INDEX,
    MAP_ROUTE_DEPTH_LABELS,
    MAP_ROUTE_OPTION_COUNT,
    MAP_ROUTE_TRACKED_TYPES,
    SEMANTIC_HISTORY_SIZE,
    SEMANTIC_OBSERVATION_SIZE as SEMANTIC_VECTOR_SIZE,
    SEMANTIC_RELATION_SIZE,
    SEMANTIC_SCALAR_SIZE,
    SHOP_SLOT_COUNT,
    build_hand_order_action_biases,
    build_hand_order_profile,
    SemanticHistoryTracker,
    combat_round_bucket,
    encode_semantic_observation,
    ensure_semantic_state,
    event_option_kind,
    map_depth_type_ratio,
    map_type_ratio,
)

COMBAT_STATE_TYPES = {"monster", "elite", "boss"}
STATE_TYPES = [
    "menu",
    "game_over",
    "unknown",
    "monster",
    "elite",
    "boss",
    "hand_select",
    "rewards",
    "card_reward",
    "map",
    "event",
    "rest_site",
    "shop",
    "fake_merchant",
    "treasure",
    "card_select",
    "bundle_select",
    "relic_select",
    "crystal_sphere",
    "overlay",
]
ENEMY_TARGET_TYPES = {"AnyEnemy", "Enemy", "SingleEnemy"}
ALL_ENEMY_TARGET_TYPES = {"AllEnemies"}
SELF_TARGET_TYPES = {"Self", "AnyAlly", "AnyPlayer"}
TERMINAL_STATE_TYPES = {"menu", "game_over"}
REST_SITE_FORCE_HEAL_THRESHOLD = 0.70
REST_SITE_AVOID_HEAL_THRESHOLD = 0.90
AUXILIARY_STRENGTH_TARGET_NAMES = (
    "deck_strength_norm",
    "deck_hp_strength_norm",
    "next_risk_strength_norm",
    "projected_deck_hp_strength_norm",
    "boss_margin_norm",
    "next_step_feasible",
    "boss_feasible",
    "path_feasible",
    "constraint_score_norm",
    "growth_score_norm",
    "total_score_norm",
)
AUXILIARY_STRENGTH_TARGET_SIZE = len(AUXILIARY_STRENGTH_TARGET_NAMES)
_CARD_RARITY_STRENGTH = {
    "None": 0.0,
    "Basic": -0.25,
    "Common": 0.25,
    "Uncommon": 0.75,
    "Rare": 1.15,
    "Ancient": 1.40,
    "Special": 0.35,
}
_RELIC_RARITY_STRENGTH = {
    "None": 0.0,
    "Common": 0.35,
    "Uncommon": 0.55,
    "Rare": 0.85,
    "Shop": 0.65,
    "Ancient": 0.95,
    "Starter": 0.45,
    "Event": 0.60,
}
_UNIVERSAL_RELIC_STRENGTH_BONUS = {
    "BURNING_BLOOD": 0.35,
    "BLACK_BLOOD": 0.65,
    "BAG_OF_PREPARATION": 0.55,
    "ANCHOR": 0.40,
    "LANTERN": 0.30,
    "FOSSILIZED_HELIX": 0.75,
    "TOXIC_EGG": 0.45,
    "MOLTEN_EGG": 0.45,
    "FROZEN_EGG": 0.45,
    "QUESTION_CARD": 0.40,
    "OMAMORI": 0.30,
    "PAPER_PHROG": 0.45,
    "CHARON_S_ASHES": 0.50,
}


@dataclass
class BoundAction:
    action_index: int
    tool_name: str
    kwargs: dict[str, object]
    description: str


@dataclass
class RewardWeights:
    error_penalty: float = 1.0
    floor_advance: float = 5.0
    act_advance: float = 20.0
    map_route_choice: float = 1.5
    hp_delta: float = 5.0
    non_combat_hp_loss: float = 6.0
    rest_missed_heal: float = 10.0
    rest_low_hp_heal_bonus: float = 5.0
    rest_high_hp_heal_penalty: float = 8.0
    gold_delta: float = 0.5
    gold_delta_scale: float = 25.0
    enemy_hp_delta: float = 8.0
    turn_end: float = 0.0
    turn_skip_unspent_penalty: float = 0.0
    combat_end: float = 8.0
    combat_defeat: float = -8.0
    act_end: float = 0.0
    run_victory: float = 10.0
    run_defeat: float = -10.0
    combat_tactical_shaping: float = 0.35
    end_turn_slack_penalty: float = 0.5


@dataclass
class _ObservationFeatureBuilder:
    names: list[str]
    values: list[float]

    def __init__(self) -> None:
        self.names = []
        self.values = []

    def add(self, name: str, value: object) -> None:
        self.names.append(name)
        self.values.append(float(value))


def _reward_item_is_card(item: dict[str, object]) -> bool:
    reward_type = str(item.get("type", "")).lower()
    return reward_type in {"card", "special_card"}


def _bool(value: object) -> float:
    return 1.0 if value else 0.0


def _ratio(value: object, scale: float, clamp: float = 2.0) -> float:
    try:
        result = float(value) / scale
    except (TypeError, ValueError):
        return 0.0
    if result < 0.0:
        return 0.0
    if result > clamp:
        return clamp
    return result


def _signed_ratio(value: object, scale: float, clamp: float = 2.0) -> float:
    try:
        result = float(value) / scale
    except (TypeError, ValueError):
        return 0.0
    if result < -clamp:
        return -clamp
    if result > clamp:
        return clamp
    return result


def _strategy_vector_norm(vector: tuple[float, ...], name: str, scale: float = 1.6, clamp: float = 2.0) -> float:
    index = CAPABILITY_DIMENSIONS.index(name)
    return _ratio(float(vector[index]), scale, clamp)


def _monster_mechanic_norm(vector: tuple[float, ...], name: str, scale: float = 1.5, clamp: float = 2.0) -> float:
    index = MECHANIC_DIMENSIONS.index(name)
    return _ratio(float(vector[index]), scale, clamp)


def _monster_strategy_norm(vector: tuple[float, ...], name: str, scale: float = 1.5, clamp: float = 2.0) -> float:
    index = STRATEGY_DIMENSIONS.index(name)
    return _ratio(float(vector[index]), scale, clamp)


def _safe_div(numerator: object, denominator: object) -> float:
    try:
        bottom = float(denominator)
        top = float(numerator)
    except (TypeError, ValueError):
        return 0.0
    if bottom == 0:
        return 0.0
    return top / bottom


def _cost_to_float(cost: object) -> float:
    if cost == "X":
        return 1.0
    return _ratio(cost, 5.0)


def _is_enemy_target(target_type: object) -> bool:
    return str(target_type) in ENEMY_TARGET_TYPES


def _intent_damage(enemy: dict[str, object]) -> float:
    intents = enemy.get("intents")
    if not isinstance(intents, list):
        return 0.0
    for intent in intents:
        if not isinstance(intent, dict):
            continue
        digits = "".join(character for character in str(intent.get("label", "")) if character.isdigit())
        if digits:
            return float(digits)
    return 0.0


def _incoming_enemy_damage(state: dict[str, object]) -> float:
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    enemies = battle.get("enemies") if isinstance(battle.get("enemies"), list) else []
    return sum(_intent_damage(enemy) for enemy in enemies if isinstance(enemy, dict) and float(enemy.get("hp", 0) or 0) > 0.0)


def _int_prefix(text: object) -> int | None:
    digits = []
    for character in str(text):
        if character.isdigit():
            digits.append(character)
        elif digits:
            break
    if not digits:
        return None
    return int("".join(digits))


def _push_state_type(features: list[float], state_type: str) -> None:
    for candidate in STATE_TYPES:
        features.append(1.0 if candidate == state_type else 0.0)


def _push_state_type_features(builder: _ObservationFeatureBuilder, state_type: str) -> None:
    for candidate in STATE_TYPES:
        builder.add(f"state_type.{candidate}", 1.0 if candidate == state_type else 0.0)


def _total_enemy_hp(state: dict[str, object]) -> float:
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    return sum(float(enemy.get("hp", 0) or 0) for enemy in battle.get("enemies", []) if isinstance(enemy, dict))


def _total_enemy_max_hp(state: dict[str, object]) -> float:
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    total = 0.0
    for enemy in battle.get("enemies", []):
        if not isinstance(enemy, dict):
            continue
        hp = float(enemy.get("hp", 0) or 0)
        if hp <= 0.0:
            continue
        total += max(float(enemy.get("max_hp", 0) or 0), hp, 1.0)
    return total


def _player_hp(state: dict[str, object]) -> float:
    player = state.get("player") if isinstance(state.get("player"), dict) else {}
    return float(player.get("hp", 0) or 0)


def _combat_player(state: dict[str, object]) -> dict[str, object]:
    player = state.get("player")
    return player if isinstance(player, dict) else {}


def _count_playable_cards(state: dict[str, object]) -> int:
    player = _combat_player(state)
    hand_cards = player.get("hand", []) if isinstance(player.get("hand"), list) else []
    return sum(
        1
        for card in hand_cards
        if isinstance(card, dict) and card.get("can_play") and isinstance(card.get("index"), int) and int(card["index"]) <= 9
    )


def _unspent_energy_units(state: dict[str, object]) -> int:
    player = _combat_player(state)
    energy = max(0.0, float(player.get("energy", 0) or 0))
    return max(0, int(energy))


def _potion_capacity(player: dict[str, object]) -> int:
    for key in ("max_potions", "potion_slots", "max_potion_slots", "max_potion_capacity"):
        value = player.get(key)
        if isinstance(value, (int, float)) and float(value) > 0:
            return max(1, int(value))
    potions = player.get("potions", []) if isinstance(player.get("potions"), list) else []
    highest_slot = max((int(potion.get("slot", -1)) for potion in potions if isinstance(potion, dict) and isinstance(potion.get("slot"), int)), default=-1)
    return max(3, len(potions), highest_slot + 1)


def _potion_slots_filled(player: dict[str, object]) -> bool:
    potions = player.get("potions", []) if isinstance(player.get("potions"), list) else []
    occupied_slots = {
        int(potion["slot"])
        for potion in potions
        if isinstance(potion, dict) and isinstance(potion.get("slot"), int)
    }
    occupied_count = len(occupied_slots) if occupied_slots else len(potions)
    return occupied_count >= _potion_capacity(player)


def _reward_item_is_potion(item: dict[str, object]) -> bool:
    if isinstance(item.get("potion"), dict):
        return True
    for key in ("type", "category", "reward_type", "item_type", "kind", "name", "label"):
        token = str(item.get(key, "")).lower()
        if "potion" in token:
            return True
    return False


def _has_pending_potion_reward(state: dict[str, object]) -> bool:
    rewards = state.get("rewards") if isinstance(state.get("rewards"), dict) else {}
    items = rewards.get("items") if isinstance(rewards.get("items"), list) else []
    return any(isinstance(item, dict) and _reward_item_is_potion(item) for item in items)


def detect_transition_boundaries(previous_state: dict[str, object], next_state: dict[str, object]) -> dict[str, object]:
    previous_state_type = str(previous_state.get("state_type", "unknown"))
    next_state_type = str(next_state.get("state_type", "unknown"))
    previous_run = previous_state.get("run") if isinstance(previous_state.get("run"), dict) else {}
    next_run = next_state.get("run") if isinstance(next_state.get("run"), dict) else {}
    previous_battle = previous_state.get("battle") if isinstance(previous_state.get("battle"), dict) else {}
    next_battle = next_state.get("battle") if isinstance(next_state.get("battle"), dict) else {}
    previous_act = int(previous_run.get("act", 0) or 0)
    next_act = int(next_run.get("act", 0) or 0)
    previous_floor = int(previous_run.get("floor", 0) or 0)
    next_floor = int(next_run.get("floor", 0) or 0)
    previous_enemy_hp = _total_enemy_hp(previous_state)
    next_enemy_hp = _total_enemy_hp(next_state)
    previous_round = int(previous_battle.get("round", 0) or 0)
    next_round = int(next_battle.get("round", 0) or 0)
    previous_turn = str(previous_battle.get("turn", ""))
    next_turn = str(next_battle.get("turn", ""))
    turn_end = (
        previous_state_type in COMBAT_STATE_TYPES
        and previous_turn == "player"
        and (
            next_state_type not in COMBAT_STATE_TYPES
            or (next_state_type in COMBAT_STATE_TYPES and next_turn == "player" and next_round > previous_round)
        )
    )
    combat_end = previous_state_type in COMBAT_STATE_TYPES and next_state_type not in COMBAT_STATE_TYPES and previous_enemy_hp > 0.0
    act_end = next_act > previous_act
    run_end = previous_state_type not in TERMINAL_STATE_TYPES and next_state_type in TERMINAL_STATE_TYPES
    return {
        "previous_state_type": previous_state_type,
        "next_state_type": next_state_type,
        "previous_act": previous_act,
        "next_act": next_act,
        "previous_floor": previous_floor,
        "next_floor": next_floor,
        "previous_enemy_hp": previous_enemy_hp,
        "next_enemy_hp": next_enemy_hp,
        "turn_end": turn_end,
        "combat_end": combat_end,
        "act_end": act_end,
        "run_end": run_end,
    }


def _cost_units(cost: object, fallback_energy: float = 1.0) -> float:
    if cost == "X":
        return max(0.0, fallback_energy)
    try:
        return max(0.0, float(cost))
    except (TypeError, ValueError):
        return 0.0


def _count_cards_by_type(cards: tuple[object, ...], card_type: str, playable_only: bool = False) -> int:
    return sum(
        1
        for card in cards
        if getattr(card, "card_type", "") == card_type and (not playable_only or bool(getattr(card, "can_play", False)))
    )


def _count_cards_by_target(cards: tuple[object, ...], target_types: set[str], playable_only: bool = False) -> int:
    return sum(
        1
        for card in cards
        if getattr(card, "target_type", "") in target_types and (not playable_only or bool(getattr(card, "can_play", False)))
    )


def _sum_card_status(cards: tuple[object, ...]) -> tuple[int, float, int, float]:
    enchantment_count = 0
    enchantment_amount = 0.0
    affliction_count = 0
    affliction_amount = 0.0
    for card in cards:
        enchantments = tuple(getattr(card, "enchantments", ()))
        afflictions = tuple(getattr(card, "afflictions", ()))
        enchantment_count += len(enchantments)
        affliction_count += len(afflictions)
        enchantment_amount += sum(abs(float(getattr(enchantment, "amount", 0.0) or 0.0)) for enchantment in enchantments)
        affliction_amount += sum(abs(float(getattr(affliction, "amount", 0.0) or 0.0)) for affliction in afflictions)
    return enchantment_count, enchantment_amount, affliction_count, affliction_amount


def _count_relic_status(relics: tuple[object, ...], status: str) -> int:
    return sum(1 for relic in relics if getattr(relic, "status", "") == status)


def _sum_power_amounts(powers: tuple[object, ...]) -> tuple[float, float]:
    positive = 0.0
    negative = 0.0
    for power in powers:
        amount = float(getattr(power, "amount", 0.0) or 0.0)
        if amount >= 0.0:
            positive += amount
        else:
            negative += abs(amount)
    return positive, negative


def _count_reward_kind(items: tuple[object, ...], *tokens: str) -> int:
    expected = tuple(token.lower() for token in tokens)
    count = 0
    for item in items:
        kind = str(getattr(item, "kind", "")).lower()
        if any(token and token in kind for token in expected):
            count += 1
    return count


def _count_option_type(options: tuple[object, ...], option_type: str) -> int:
    return sum(1 for option in options if getattr(option, "option_type", "") == option_type)


def _count_shop_category(items: tuple[object, ...], category: str, stocked_only: bool = False) -> int:
    return sum(
        1
        for item in items
        if getattr(item, "category", "") == category and (not stocked_only or bool(getattr(item, "is_stocked", True)))
    )


def _card_id_token(card: dict[str, object]) -> str:
    for key in ("id", "card_id", "model_id", "entry"):
        raw = str(card.get(key, "")).strip().upper()
        if not raw:
            continue
        return raw.split(".", 1)[1] if raw.startswith("CARD.") else raw
    return ""


def _raw_card_instance_token(card: dict[str, object]) -> str:
    for key in ("instance_id", "uuid", "guid", "entity_id", "instance_uuid", "instance_guid", "card_instance_id", "uid"):
        token = str(card.get(key, "") or "").strip()
        if token:
            return token
    return ""


def _raw_card_signature(card: dict[str, object]) -> tuple[object, ...]:
    enchantments = card.get("enchantments") if isinstance(card.get("enchantments"), list) else []
    afflictions = card.get("afflictions") if isinstance(card.get("afflictions"), list) else []
    return (
        int(card.get("index", -1)),
        _raw_card_instance_token(card),
        _card_id_token(card),
        str(card.get("name", "")),
        str(card.get("type", "")),
        str(card.get("rarity", "")),
        bool(card.get("can_play")),
        bool(card.get("is_upgraded")),
        bool(
            card.get("exhausts")
            or card.get("exhaust")
            or card.get("consume")
            or card.get("consumes")
            or card.get("purge_on_use")
            or card.get("exhausts_on_use")
        ),
        str(card.get("cost", "")),
        str(card.get("target_type", "")),
        len(enchantments),
        len(afflictions),
    )


def _card_equivalence_signature(card: dict[str, object]) -> tuple[object, ...]:
    signature = _raw_card_signature(card)
    return signature[2:]


def _is_basic_strike_or_defend(card: dict[str, object]) -> bool:
    card_id = _card_id_token(card)
    if card_id.startswith("STRIKE_") or card_id.startswith("DEFEND_"):
        return True
    rarity = str(card.get("rarity", "")).strip().lower()
    name = str(card.get("name", "")).strip().lower()
    return rarity in {"basic", "starter"} and name in {"strike", "defend", "打击", "防御"}


def _text_has_any(text: str, tokens: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in tokens)


def _card_description_blob(card: dict[str, object]) -> str:
    return " ".join(
        str(card.get(key, "") or "").strip()
        for key in ("name", "description", "text", "tooltip", "rules_text")
    ).lower()


def _card_grants_block(card: dict[str, object]) -> bool:
    card_id = _card_id_token(card)
    if card_id.startswith("DEFEND_"):
        return True
    if str(card.get("type", "")) != "Skill" or str(card.get("target_type", "")) not in SELF_TARGET_TYPES:
        return False
    description = _card_description_blob(card)
    if _text_has_any(description, ("格挡", "阻挡", "block")):
        return True
    keywords = card.get("keywords") if isinstance(card.get("keywords"), list) else []
    for keyword in keywords:
        if not isinstance(keyword, dict):
            continue
        keyword_text = " ".join(str(keyword.get(key, "") or "").strip() for key in ("name", "description")).lower()
        if _text_has_any(keyword_text, ("格挡", "阻挡", "block")):
            return True
    return False


def _is_pure_defense_card(card: dict[str, object]) -> bool:
    if str(card.get("type", "")) != "Skill" or str(card.get("target_type", "")) not in SELF_TARGET_TYPES:
        return False
    if not _card_grants_block(card):
        return False
    card_id = _card_id_token(card)
    if card_id.startswith("DEFEND_"):
        return True
    description = _card_description_blob(card)
    if _text_has_any(
        description,
        (
            "draw",
            "抽",
            "damage",
            "伤害",
            "攻击",
            "attack",
            "vulnerable",
            "易伤",
            "weak",
            "虚弱",
            "frail",
            "易碎",
            "strength",
            "力量",
            "dexterity",
            "敏捷",
            "energy",
            "能量",
            "heal",
            "回复",
            "恢复",
            "治疗",
            "retain",
            "保留",
            "upgrade",
            "升级",
            "free",
            "免费",
            "copy",
            "复制",
            "double",
            "双倍",
            "thorns",
            "荆棘",
        ),
    ):
        return False
    return True


def _clean_label_token(value: object) -> str:
    text = str(value or "").strip()
    return text if text and text.lower() != "none" else ""


def _player_hp_ratio_from_dict(player: dict[str, object]) -> float:
    return _safe_div(player.get("hp"), max(player.get("max_hp", 0) or 0, 1))


def _low_hp_urgency(hp_ratio: float, threshold: float = 0.65) -> float:
    normalized_ratio = max(0.0, min(1.0, float(hp_ratio)))
    if normalized_ratio >= threshold:
        return 0.0
    return (threshold - normalized_ratio) / max(threshold, 1e-9)


def _extract_int_values(text: object) -> list[int]:
    digits: list[str] = []
    values: list[int] = []
    for character in str(text or ""):
        if character.isdigit():
            digits.append(character)
            continue
        if digits:
            values.append(int("".join(digits)))
            digits = []
    if digits:
        values.append(int("".join(digits)))
    return values


_SELECTION_COUNT_WORD_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\bchoose\s+a\s+card\b"), 1),
    (re.compile(r"\bselect\s+a\s+card\b"), 1),
    (re.compile(r"\ba\s+card\b"), 1),
    (re.compile(r"\bone\s+card\b"), 1),
    (re.compile(r"\bsingle\s+card\b"), 1),
    (re.compile(r"\btwo\s+cards?\b"), 2),
    (re.compile(r"\bthree\s+cards?\b"), 3),
)
_SELECTION_COUNT_TOKEN_VALUES: tuple[tuple[str, int], ...] = (
    ("\u4e00\u5f20", 1),
    ("\u4e8c\u5f20", 2),
    ("\u4e24\u5f20", 2),
    ("\u5169\u5f20", 2),
    ("\u4e09\u5f20", 3),
    ("\u56db\u5f20", 4),
    ("\u4e94\u5f20", 5),
)
_ANY_NUMBER_SELECTION_HINTS: tuple[str, ...] = (
    "any number",
    "choose any",
    "select any",
    "\u4efb\u610f",
)


def _contains_any_number_selection_hint(text: object) -> bool:
    normalized = str(text or "").strip().lower()
    return any(token in normalized for token in _ANY_NUMBER_SELECTION_HINTS)


def _extract_selection_count_hint(text: object) -> int | None:
    normalized = str(text or "").strip().lower()
    values = _extract_int_values(normalized)
    if values:
        return max(1, values[0])
    for pattern, value in _SELECTION_COUNT_WORD_PATTERNS:
        if pattern.search(normalized):
            return value
    for token, value in _SELECTION_COUNT_TOKEN_VALUES:
        if token in normalized:
            return value
    return None


def _floor_reward_multiplier(floor: int) -> float:
    clamped_floor = max(0, int(floor))
    if clamped_floor <= 0:
        return 0.0
    return 0.5 * clamped_floor * (1.1 ** clamped_floor)


def _event_option_guard_key(option: dict[str, object]) -> int:
    try:
        return int(option.get("index", -1))
    except (TypeError, ValueError):
        return -1


def _unlocked_event_options(event_state: dict[str, object]) -> list[dict[str, object]]:
    options = event_state.get("options")
    if not isinstance(options, list):
        return []
    return [option for option in options if isinstance(option, dict) and not option.get("is_locked")]


def _event_option_from_unlocked_index(event_state: dict[str, object], option_index: object) -> dict[str, object] | None:
    if not isinstance(option_index, int):
        return None
    unlocked_options = _unlocked_event_options(event_state)
    if 0 <= option_index < len(unlocked_options):
        return unlocked_options[option_index]
    return None


def _event_option_from_state_index(event_state: dict[str, object], option_index: object) -> dict[str, object] | None:
    if not isinstance(option_index, int):
        return None
    for option in _unlocked_event_options(event_state):
        try:
            state_index = int(option.get("index", -1))
        except (TypeError, ValueError):
            continue
        if state_index == option_index:
            return option
    return None


def _rest_option_from_state_index(rest_state: dict[str, object], option_index: object) -> dict[str, object] | None:
    if not isinstance(option_index, int):
        return None
    options = rest_state.get("options")
    if not isinstance(options, list):
        return None
    for option in options:
        if not isinstance(option, dict):
            continue
        try:
            state_index = int(option.get("index", -1))
        except (TypeError, ValueError):
            continue
        if state_index == option_index:
            return option
    return None


def _describe_indexed_item(label_prefix: str, item: dict[str, object], item_index: int) -> str:
    name = ""
    for key in ("name", "title", "label"):
        name = _clean_label_token(item.get(key))
        if name:
            break
    identifier = ""
    for key in ("id", "card_id", "relic_id", "potion_id", "monster_id", "event_id", "move_id", "model_id", "entity_id"):
        identifier = _clean_label_token(item.get(key))
        if identifier:
            break
    kind = ""
    for key in ("type", "category", "option_type", "screen_type", "rarity"):
        kind = _clean_label_token(item.get(key))
        if kind:
            break
    if identifier and kind and identifier.lower() != kind.lower():
        detail = f"{identifier}<{kind}>"
    elif identifier:
        detail = identifier
    elif name and kind and name.lower() != kind.lower():
        detail = f"{name}<{kind}>"
    elif name:
        detail = name
    else:
        detail = kind or str(item_index)
    return f"{label_prefix} {detail}#{item_index}"


def _describe_event_option_item(event_state: dict[str, object], item: dict[str, object], item_index: int) -> str:
    description = _describe_indexed_item("event option", item, item_index)
    event_id = _clean_label_token(event_state.get("event_id"))
    event_name = _clean_label_token(event_state.get("event_name"))
    if event_id and event_name and event_id.lower() != event_name.lower():
        event_label = f"{event_id}<{event_name}>"
    else:
        event_label = event_id or event_name
    if not event_label:
        return description
    return f"event {event_label} {description}"


def _describe_map_route_item(
    item: dict[str, object],
    item_index: int,
    route: object | None,
) -> str:
    description = _describe_indexed_item("map node", item, item_index)
    if route is None:
        return description
    sequence = " > ".join(
        node_type
        for node_type in tuple(getattr(route, "depth_dominant_types", ()))[:3]
        if isinstance(node_type, str) and node_type and node_type != "Unknown"
    )
    mix_tokens: list[str] = []
    for node_type, short_label in (
        ("Monster", "M"),
        ("Elite", "E"),
        ("RestSite", "R"),
        ("Shop", "S"),
        ("Event", "V"),
        ("Treasure", "T"),
        ("Boss", "B"),
    ):
        ratio = map_type_ratio(getattr(route, "type_counts", ()), node_type)
        if ratio > 0.0:
            mix_tokens.append(f"{short_label}{ratio:.2f}")
        if len(mix_tokens) >= 3:
            break
    boss_distance = getattr(route, "boss_distance", -1)
    return (
        f"{description} score={float(getattr(route, 'route_score', 0.0)):.2f} "
        f"boss={'-' if boss_distance is None or int(boss_distance) < 0 else int(boss_distance)} "
        f"path={sequence or str(getattr(route, 'option_type', 'Unknown') or 'Unknown')} "
        f"mix={','.join(mix_tokens) if mix_tokens else 'none'}"
    )


def _semantic_card_is_basic(card: object) -> bool:
    card_id = str(getattr(card, "id", "") or "").upper()
    if card_id.startswith("STRIKE_") or card_id.startswith("DEFEND_"):
        return True
    return str(getattr(card, "rarity", "") or "") in {"Basic", "Starter"}


def _semantic_card_strength(card: object) -> float:
    card_type = str(getattr(card, "card_type", "") or "None")
    rarity = str(getattr(card, "rarity", "") or "None")
    target_type = str(getattr(card, "target_type", "") or "None")
    value = _CARD_RARITY_STRENGTH.get(rarity, 0.0)
    if card_type == "Attack":
        value += 0.20
    elif card_type == "Skill":
        value += 0.10
    elif card_type == "Power":
        value += 0.45
    elif card_type == "Status":
        value -= 1.60
    elif card_type == "Curse":
        value -= 2.30
    if target_type in ALL_ENEMY_TARGET_TYPES:
        value += 0.35
    elif target_type in ENEMY_TARGET_TYPES:
        value += 0.10
    elif target_type in SELF_TARGET_TYPES and card_type == "Skill":
        value += 0.05
    cost_token = str(getattr(card, "cost", "") or "")
    if cost_token == "0":
        value += 0.10
    elif cost_token == "X":
        value += 0.15
    else:
        cost_units = _cost_units(cost_token, 1.0)
        if cost_units >= 2.0:
            value += 0.15
    if bool(getattr(card, "is_upgraded", False)):
        value += 0.35
    if bool(getattr(card, "consumes_on_play", False)):
        value -= 0.10
    value += min(0.45, len(getattr(card, "enchantments", ())) * 0.15)
    value -= min(0.60, len(getattr(card, "afflictions", ())) * 0.20)
    if _semantic_card_is_basic(card):
        value -= 0.35
    return value


def _estimate_deck_strength(cards: tuple[object, ...]) -> float:
    if not cards:
        return 0.0
    size = len(cards)
    total_value = sum(_semantic_card_strength(card) for card in cards)
    upgraded = sum(1 for card in cards if bool(getattr(card, "is_upgraded", False)))
    powers = sum(1 for card in cards if str(getattr(card, "card_type", "") or "") == "Power")
    aoe = sum(1 for card in cards if str(getattr(card, "target_type", "") or "") in ALL_ENEMY_TARGET_TYPES)
    non_basic = sum(1 for card in cards if not _semantic_card_is_basic(card))
    curses = sum(1 for card in cards if str(getattr(card, "card_type", "") or "") == "Curse")
    statuses = sum(1 for card in cards if str(getattr(card, "card_type", "") or "") == "Status")
    basics = sum(1 for card in cards if _semantic_card_is_basic(card))
    density = max(0.0, total_value) / max(8.0, size * 0.70)
    quality_ratio = non_basic / max(size, 1)
    upgrade_ratio = upgraded / max(size, 1)
    size_penalty = max(0.0, size - 18) * 0.05 + max(0.0, size - 24) * 0.06
    bad_draw_penalty = (curses * 0.25) + (statuses * 0.12) + (max(0, basics - 8) * 0.04)
    strength = (
        density
        + (0.55 * quality_ratio)
        + (0.35 * upgrade_ratio)
        + min(0.35, powers * 0.12)
        + min(0.30, aoe * 0.12)
        - size_penalty
        - bad_draw_penalty
    )
    return max(0.0, min(3.0, strength))


def _semantic_relic_strength(relic: object) -> float:
    relic_id = str(getattr(relic, "id", "") or "").upper()
    rarity = str(getattr(relic, "rarity", "") or "None")
    status = str(getattr(relic, "status", "") or "None")
    value = _RELIC_RARITY_STRENGTH.get(rarity, 0.0)
    value += _UNIVERSAL_RELIC_STRENGTH_BONUS.get(relic_id, 0.0)
    if status == "Active":
        value += 0.10
    elif status == "Disabled":
        value -= 0.25
    if bool(getattr(relic, "is_used_up", False)):
        value -= 0.35
    if bool(getattr(relic, "is_wax", False)) or bool(getattr(relic, "is_melted", False)):
        value -= 0.15
    counter = max(0.0, float(getattr(relic, "counter", 0.0) or 0.0))
    value += min(0.20, counter / 25.0)
    return max(-0.50, value)


def _estimate_relic_strengths(relics: tuple[object, ...]) -> tuple[tuple[float, ...], float]:
    if not relics:
        return (), 0.0
    item_scores = tuple(_semantic_relic_strength(relic) for relic in relics)
    unique_ids = {
        str(getattr(relic, "id", "") or "")
        for relic in relics
        if str(getattr(relic, "id", "") or "")
    }
    overall = (sum(item_scores) / max(1.4, len(item_scores) * 0.75)) + min(0.25, len(unique_ids) * 0.02)
    return item_scores, max(0.0, min(3.0, overall))


def _estimate_resource_strength(semantic_state: object) -> float:
    max_hp = max(float(getattr(semantic_state, "player_max_hp", 0.0) or 0.0), 1.0)
    hp_ratio = float(getattr(semantic_state, "player_hp", 0.0) or 0.0) / max_hp
    potion_bonus = min(0.35, len(getattr(semantic_state, "potions", ())) * 0.12)
    gold_bonus = min(0.25, float(getattr(semantic_state, "player_gold", 0.0) or 0.0) / 220.0)
    max_hp_bonus = min(0.25, max(0.0, max_hp - 72.0) / 48.0)
    strength = (0.75 * hp_ratio) + potion_bonus + gold_bonus + max_hp_bonus
    return max(0.0, min(2.0, strength))


def _estimate_run_readiness(deck_strength: float, relic_strength: float, resource_strength: float) -> float:
    readiness = (0.55 * deck_strength) + (0.25 * relic_strength) + (0.20 * resource_strength)
    return max(0.0, min(3.0, readiness))


def _collect_visible_deck_cards(semantic_state: object) -> tuple[object, ...]:
    combat_cards = tuple(getattr(semantic_state, "hand", ())) + tuple(getattr(semantic_state, "draw_pile", ())) + tuple(getattr(semantic_state, "discard_pile", ())) + tuple(getattr(semantic_state, "exhaust_pile", ()))
    if len(combat_cards) >= 5:
        return combat_cards
    card_select_cards = tuple(getattr(semantic_state, "card_select_cards", ()))
    if len(card_select_cards) >= 5:
        return card_select_cards
    return ()


def _route_depth_count(route: object, node_type: str, depth: int) -> int:
    depth_counts = getattr(route, "depth_type_counts", ())
    if depth < 0 or depth >= len(depth_counts):
        return 0
    type_index = MAP_NODE_TYPE_INDEX.get(node_type)
    if type_index is None or type_index >= len(depth_counts[depth]):
        return 0
    return int(depth_counts[depth][type_index])


def _route_total_count(route: object, node_type: str) -> int:
    type_counts = getattr(route, "type_counts", ())
    type_index = MAP_NODE_TYPE_INDEX.get(node_type)
    if type_index is None or type_index >= len(type_counts):
        return 0
    return int(type_counts[type_index])


def _route_expected_span(route: object) -> float:
    boss_distance = int(getattr(route, "boss_distance", -1) or -1)
    if boss_distance > 0:
        return float(boss_distance)
    max_depth = int(getattr(route, "max_depth", -1) or -1)
    if max_depth >= 0:
        return float(max_depth + 1)
    reachable_count = int(getattr(route, "reachable_count", 0) or 0)
    return float(max(1, min(12, reachable_count)))


def _route_expected_total(route: object, node_type: str) -> float:
    if node_type == "Boss":
        return 1.0 if _route_total_count(route, "Boss") > 0 else 0.0
    return _route_expected_span(route) * map_type_ratio(getattr(route, "type_counts", ()), node_type)


def _route_expected_depth(route: object, node_type: str, depth: int) -> float:
    return map_depth_type_ratio(getattr(route, "depth_type_counts", ()), depth, node_type)


def _route_risk_profile(route: object, *, normal_monster_count: float = 0.0) -> dict[str, float]:
    easy_budget = max(0.0, 3.0 - max(0.0, normal_monster_count))
    depth_limit = max(len(getattr(route, "depth_type_counts", ())), len(MAP_ROUTE_DEPTH_LABELS))
    for depth_index in range(max(0, depth_limit)):
        monster_mass = max(0.0, _route_expected_depth(route, "Monster", depth_index))
        elite_mass = max(0.0, _route_expected_depth(route, "Elite", depth_index))
        unknown_mass = max(0.0, _route_expected_depth(route, "Unknown", depth_index))
        if monster_mass <= 1e-9 and elite_mass <= 1e-9 and unknown_mass <= 1e-9:
            continue
        easy_monster_mass = min(monster_mass, easy_budget)
        hard_monster_mass = max(0.0, monster_mass - easy_monster_mass)
        easy_budget = max(0.0, easy_budget - easy_monster_mass)
        easy_unknown_mass = min(unknown_mass, easy_budget)
        hard_unknown_mass = max(0.0, unknown_mass - easy_unknown_mass)
        easy_budget = max(0.0, easy_budget - easy_unknown_mass)
        easy_mass = easy_monster_mass + easy_unknown_mass
        hard_mass = hard_monster_mass + hard_unknown_mass
        risk_presence = easy_mass + hard_mass + elite_mass
        if risk_presence <= 0.05:
            continue
        return {"risk_1_strength": float(easy_mass + (2.0 * hard_mass) + (4.0 * elite_mass))}
    return {"risk_1_strength": 0.0}


def _split_future_normal_difficulty(
    *,
    normal_monster_count: float,
    monster_count: float,
    question_count: float,
) -> tuple[float, float, float, float, float, float]:
    easy_remaining = max(0.0, 3.0 - max(0.0, normal_monster_count))
    next_normal_monster_difficulty = 1.0 if easy_remaining > 1e-9 else 2.0
    future_easy_monsters = min(monster_count, easy_remaining)
    future_hard_monsters = max(0.0, monster_count - future_easy_monsters)
    question_easy_nodes = min(question_count, max(0.0, easy_remaining - future_easy_monsters))
    question_hard_nodes = max(0.0, question_count - question_easy_nodes)
    question_difficulty_upper = 0.0
    if question_count > 1e-9:
        question_difficulty_upper = 2.0 if question_hard_nodes > 1e-9 else 1.0
    return (
        easy_remaining,
        next_normal_monster_difficulty,
        future_easy_monsters,
        future_hard_monsters,
        question_easy_nodes,
        question_hard_nodes,
    )


def _route_profile(route: object, *, normal_monster_count: float = 0.0) -> dict[str, float]:
    monster_count = _route_expected_total(route, "Monster")
    elite_count = _route_expected_total(route, "Elite")
    rest_count = _route_expected_total(route, "RestSite")
    shop_count = _route_expected_total(route, "Shop")
    event_count = _route_expected_total(route, "Event")
    treasure_count = _route_expected_total(route, "Treasure")
    question_count = _route_expected_total(route, "Unknown")
    boss_count = _route_expected_total(route, "Boss")
    early_monsters = _route_expected_depth(route, "Monster", 0) + _route_expected_depth(route, "Monster", 1)
    late_monsters = max(0.0, monster_count - early_monsters)
    early_elites = _route_expected_depth(route, "Elite", 0) + _route_expected_depth(route, "Elite", 1)
    late_elites = max(0.0, elite_count - early_elites)
    (
        easy_remaining,
        next_normal_monster_difficulty,
        future_easy_monsters,
        future_hard_monsters,
        future_question_easy_nodes,
        future_question_hard_nodes,
    ) = _split_future_normal_difficulty(
        normal_monster_count=normal_monster_count,
        monster_count=monster_count,
        question_count=question_count,
    )
    question_difficulty_upper = 2.0 if future_question_hard_nodes > 1e-9 else (1.0 if question_count > 1e-9 else 0.0)
    difficulty_load = (
        future_easy_monsters
        + (2.0 * future_hard_monsters)
        + (4.0 * elite_count)
        + future_question_easy_nodes
        + (1.6 * future_question_hard_nodes)
    )
    early_combats = early_monsters + early_elites
    profile = {
        "normal_monster_count": float(normal_monster_count),
        "easy_remaining": float(easy_remaining),
        "next_normal_monster_difficulty": float(next_normal_monster_difficulty),
        "monster_count": float(monster_count),
        "elite_count": float(elite_count),
        "rest_count": float(rest_count),
        "shop_count": float(shop_count),
        "event_count": float(event_count),
        "treasure_count": float(treasure_count),
        "question_count": float(question_count),
        "future_easy_monsters": float(future_easy_monsters),
        "future_hard_monsters": float(future_hard_monsters),
        "future_question_easy_nodes": float(future_question_easy_nodes),
        "future_question_hard_nodes": float(future_question_hard_nodes),
        "question_difficulty_upper": float(question_difficulty_upper),
        "difficulty_load": float(difficulty_load),
        "boss_count": float(boss_count),
        "early_monsters": float(early_monsters),
        "late_monsters": float(late_monsters),
        "early_elites": float(early_elites),
        "late_elites": float(late_elites),
        "early_combats": float(early_combats),
        "path_count": float(getattr(route, "path_count", 0) or 0),
        "boss_distance": float(getattr(route, "boss_distance", -1) or -1),
    }
    profile.update(_route_risk_profile(route, normal_monster_count=normal_monster_count))
    return profile


def _strength_target_vector(
    *,
    strategy_context: BuildStrategyContext,
    strength_economy: object,
    route_eval: object | None = None,
) -> tuple[list[float], list[float]]:
    deck_hp_strength = estimate_hp_adjusted_strength(
        float(getattr(strength_economy, "current_strength", 0.0) or 0.0),
        float(getattr(strategy_context, "hp_ratio", 0.0) or 0.0),
    )
    targets = [0.0 for _ in AUXILIARY_STRENGTH_TARGET_NAMES]
    mask = [0.0 for _ in AUXILIARY_STRENGTH_TARGET_NAMES]

    def _set(name: str, value: float, enabled: bool = True) -> None:
        index = AUXILIARY_STRENGTH_TARGET_NAMES.index(name)
        targets[index] = float(value)
        mask[index] = 1.0 if enabled else 0.0

    current_strength = float(getattr(strength_economy, "current_strength", 0.0) or 0.0)
    _set("deck_strength_norm", _ratio(current_strength, 4.0, 2.0))
    _set("deck_hp_strength_norm", _ratio(deck_hp_strength, 4.0, 2.0))
    if route_eval is None:
        _set("boss_margin_norm", 0.0, enabled=False)
        _set("next_risk_strength_norm", 0.0, enabled=False)
        _set("projected_deck_hp_strength_norm", 0.0, enabled=False)
        _set("next_step_feasible", 0.0, enabled=False)
        _set("boss_feasible", 0.0, enabled=False)
        _set("path_feasible", 0.0, enabled=False)
        _set("constraint_score_norm", 0.0, enabled=False)
        _set("growth_score_norm", 0.0, enabled=False)
        _set("total_score_norm", 0.0, enabled=False)
        return targets, mask

    _set("next_risk_strength_norm", _ratio(getattr(route_eval, "next_risk_strength", 0.0), 4.0, 2.0))
    _set("projected_deck_hp_strength_norm", _ratio(getattr(route_eval, "projected_deck_hp_strength", 0.0), 4.0, 2.0))
    _set("boss_margin_norm", _signed_ratio(getattr(route_eval, "boss_margin", 0.0), 2.0, 2.0))
    _set("next_step_feasible", max(0.0, min(1.0, float(getattr(route_eval, "next_step_feasible", 0.0) or 0.0))))
    _set("boss_feasible", max(0.0, min(1.0, float(getattr(route_eval, "boss_feasible", 0.0) or 0.0))))
    _set("path_feasible", max(0.0, min(1.0, float(getattr(route_eval, "path_feasible", 0.0) or 0.0))))
    _set("constraint_score_norm", _signed_ratio(getattr(route_eval, "constraint_score", 0.0), 8.0, 2.0))
    _set("growth_score_norm", _signed_ratio(getattr(route_eval, "growth_score", 0.0), 4.0, 2.0))
    _set("total_score_norm", _signed_ratio(getattr(route_eval, "total_score", 0.0), 10.0, 2.0))
    return targets, mask


def _build_state_space_features(
    state: dict[str, object] | object,
    strategy_context: BuildStrategyContext | None = None,
    normal_monster_count: float = 0.0,
) -> _ObservationFeatureBuilder:
    semantic_state = ensure_semantic_state(state)
    tactical_analysis = analyze_combat_turn(state if isinstance(state, dict) else None)
    encounter_profile = analyze_encounter_profile(state if isinstance(state, dict) else {})
    if strategy_context is None:
        raw_state = state if isinstance(state, dict) else {"state_type": str(getattr(semantic_state, "state_type", "unknown"))}
        strategy_context = build_strategy_context(state=raw_state)
    hand_order_profile = build_hand_order_profile(semantic_state)
    builder = _ObservationFeatureBuilder()
    _push_state_type_features(builder, semantic_state.state_type)

    playable_cards = semantic_state.playable_cards
    playable_costs = [_cost_units(card.cost, semantic_state.player_energy) for card in playable_cards]
    all_cards = (
        semantic_state.hand
        + semantic_state.draw_pile
        + semantic_state.discard_pile
        + semantic_state.exhaust_pile
        + semantic_state.card_reward_cards
        + semantic_state.card_select_cards
        + semantic_state.card_select_preview_cards
        + semantic_state.bundle_cards
        + semantic_state.bundle_preview_cards
        + semantic_state.hand_select_cards
        + semantic_state.hand_select_selected_cards
    )
    hand_enchantment_count, hand_enchantment_amount, hand_affliction_count, hand_affliction_amount = _sum_card_status(semantic_state.hand)
    all_enchantment_count, all_enchantment_amount, all_affliction_count, all_affliction_amount = _sum_card_status(all_cards)
    player_positive_power, player_negative_power = _sum_power_amounts(semantic_state.player_powers)
    enemy_powers = tuple(power for enemy in semantic_state.living_enemies for power in enemy.powers)
    enemy_positive_power, enemy_negative_power = _sum_power_amounts(enemy_powers)
    enemy_block_total = sum(enemy.block for enemy in semantic_state.living_enemies)
    enemy_power_count = len(enemy_powers)
    enemy_attack_intents = sum(1 for enemy in semantic_state.living_enemies for intent in enemy.intent_types if intent.startswith("ATTACK"))
    enemy_buff_intents = sum(1 for enemy in semantic_state.living_enemies for intent in enemy.intent_types if intent == "BUFF")
    enemy_debuff_intents = sum(1 for enemy in semantic_state.living_enemies for intent in enemy.intent_types if "DEBUFF" in intent)
    enemy_max_intent_damage = max((enemy.intent_damage for enemy in semantic_state.living_enemies), default=0.0)
    block_gap = max(0.0, semantic_state.incoming_damage - semantic_state.player_block)
    overblock = max(0.0, semantic_state.player_block - semantic_state.incoming_damage)
    can_take_combat_actions = semantic_state.is_player_turn and semantic_state.is_play_phase and not semantic_state.player_actions_disabled and not semantic_state.hand_in_card_play
    skip_unspent_risk = can_take_combat_actions and len(semantic_state.living_enemies) > 0 and len(playable_cards) > 0 and semantic_state.player_energy > 0.0
    selection_ready = any(
        (
            semantic_state.card_select_can_confirm,
            semantic_state.card_select_can_cancel,
            semantic_state.bundle_can_confirm,
            semantic_state.bundle_can_cancel,
            semantic_state.hand_select_can_confirm,
            semantic_state.relic_select_can_skip,
            semantic_state.card_reward_can_skip,
        )
    )
    proceed_ready = any(
        (
            semantic_state.rewards_can_proceed,
            semantic_state.rest_can_proceed,
            semantic_state.treasure_can_proceed,
            semantic_state.shop_can_proceed,
            semantic_state.fake_merchant_can_proceed,
            semantic_state.crystal_can_proceed,
        )
    )
    potion_enemy_target_count = sum(1 for potion in semantic_state.potions if potion.target_type in ENEMY_TARGET_TYPES | ALL_ENEMY_TARGET_TYPES)
    potion_self_target_count = sum(1 for potion in semantic_state.potions if potion.target_type in SELF_TARGET_TYPES)
    potion_combat_only_count = sum(1 for potion in semantic_state.potions if potion.usage == "CombatOnly")
    potion_any_time_count = sum(1 for potion in semantic_state.potions if potion.usage == "AnyTime")
    potion_combat_usable_count = sum(1 for potion in semantic_state.potions if potion.can_use_in_combat)
    affordable_shop_count = sum(1 for item in semantic_state.shop_items if item.can_afford and item.is_stocked)
    affordable_fake_shop_count = sum(1 for item in semantic_state.fake_merchant_shop_items if item.can_afford and item.is_stocked)
    stocked_shop_count = sum(1 for item in semantic_state.shop_items if item.is_stocked)
    stocked_fake_shop_count = sum(1 for item in semantic_state.fake_merchant_shop_items if item.is_stocked)
    visible_shop_items = semantic_state.fake_merchant_shop_items if semantic_state.state_type == "fake_merchant" else semantic_state.shop_items
    reward_potion_count = _count_reward_kind(semantic_state.reward_items, "potion")
    reward_card_count = _count_reward_kind(semantic_state.reward_items, "card")
    reward_relic_count = sum(1 for item in semantic_state.reward_items if bool(item.relic_id) or "relic" in str(item.kind).lower())
    reward_gold_count = _count_reward_kind(semantic_state.reward_items, "gold")
    reward_other_count = max(0, len(semantic_state.reward_items) - reward_card_count - reward_relic_count - reward_potion_count - reward_gold_count)
    full_potion_slots = len(semantic_state.potions) >= 3
    round_bucket = combat_round_bucket(semantic_state.battle_round)
    _, observed_relic_strength = _estimate_relic_strengths(tuple(semantic_state.relics))
    observed_resource_strength = _estimate_resource_strength(semantic_state)
    strength_economy = estimate_strength_economy(
        strategy_context,
        relic_strength=observed_relic_strength,
        resource_strength=observed_resource_strength,
    )
    deck_hp_strength = estimate_hp_adjusted_strength(strength_economy.current_strength, strategy_context.hp_ratio)
    best_route_eval = None
    route_eval_by_option: dict[int, object] = {}
    if semantic_state.state_type == "map":
        for route in getattr(semantic_state, "map_route_summaries", ()):
            option_index = getattr(route, "option_index", None)
            if not isinstance(option_index, int) or option_index < 0:
                continue
            route_eval_by_option[option_index] = evaluate_route_strength(
                strategy_context,
                _route_profile(route, normal_monster_count=normal_monster_count),
                act_id=str(getattr(semantic_state, "act_id", "") or ""),
                floor=int(getattr(semantic_state, "floor", 0) or 0),
                hp_ratio=_safe_div(semantic_state.player_hp, max(semantic_state.player_max_hp, 1.0)),
                relic_strength=observed_relic_strength,
                resource_strength=observed_resource_strength,
            )
        if route_eval_by_option:
            best_route_eval = max(route_eval_by_option.values(), key=lambda item: item.total_score)

    for name, value in (
        ("act_norm", _ratio(semantic_state.act, 4.0)),
        ("floor_norm", _ratio(semantic_state.floor, 60.0)),
        ("ascension_norm", _ratio(semantic_state.ascension, 20.0)),
        ("battle_round_norm", _ratio(semantic_state.battle_round, 20.0, 2.0)),
        ("in_combat", _bool(semantic_state.in_combat)),
        ("is_player_turn", _bool(semantic_state.is_player_turn)),
        ("is_play_phase", _bool(semantic_state.is_play_phase)),
        ("player_actions_disabled", _bool(semantic_state.player_actions_disabled)),
        ("hand_in_card_play", _bool(semantic_state.hand_in_card_play)),
        ("combat_action_window_ready", _bool(can_take_combat_actions)),
        ("combat_victory_ready", _bool(semantic_state.in_combat and len(semantic_state.living_enemies) == 0)),
        ("skip_unspent_risk", _bool(skip_unspent_risk)),
        ("selection_ready", _bool(selection_ready)),
        ("proceed_ready", _bool(proceed_ready)),
        ("player_hp_norm", _ratio(semantic_state.player_hp, 120.0, 2.0)),
        ("player_hp_ratio", _safe_div(semantic_state.player_hp, max(semantic_state.player_max_hp, 1.0))),
        ("player_max_hp_norm", _ratio(semantic_state.player_max_hp, 120.0, 2.0)),
        ("player_block_norm", _ratio(semantic_state.player_block, 100.0, 2.0)),
        ("player_energy_norm", _ratio(semantic_state.player_energy, 5.0, 2.0)),
        ("player_energy_ratio", _safe_div(semantic_state.player_energy, max(semantic_state.player_max_energy, 1.0))),
        ("player_max_energy_norm", _ratio(semantic_state.player_max_energy, 10.0, 2.0)),
        ("player_gold_norm", _ratio(semantic_state.player_gold, 500.0, 2.0)),
        ("player_power_count_norm", _ratio(len(semantic_state.player_powers), 16.0, 2.0)),
        ("player_positive_power_amount_norm", _ratio(player_positive_power, 30.0, 2.0)),
        ("player_negative_power_amount_norm", _ratio(player_negative_power, 30.0, 2.0)),
        ("relic_count_norm", _ratio(len(semantic_state.relics), 40.0, 2.0)),
        ("relic_counter_sum_norm", _ratio(sum(max(0.0, relic.counter) for relic in semantic_state.relics), 30.0, 2.0)),
        ("relic_active_count_norm", _ratio(_count_relic_status(semantic_state.relics, "Active"), 10.0, 2.0)),
        ("relic_disabled_count_norm", _ratio(_count_relic_status(semantic_state.relics, "Disabled"), 10.0, 2.0)),
        ("relic_used_up_count_norm", _ratio(sum(1 for relic in semantic_state.relics if relic.is_used_up), 10.0, 2.0)),
        ("relic_wax_count_norm", _ratio(sum(1 for relic in semantic_state.relics if relic.is_wax), 10.0, 2.0)),
        ("relic_melted_count_norm", _ratio(sum(1 for relic in semantic_state.relics if relic.is_melted), 10.0, 2.0)),
        ("potion_count_norm", _ratio(len(semantic_state.potions), 5.0, 2.0)),
        ("potion_combat_usable_count_norm", _ratio(potion_combat_usable_count, 5.0, 2.0)),
        ("potion_combat_only_count_norm", _ratio(potion_combat_only_count, 5.0, 2.0)),
        ("potion_any_time_count_norm", _ratio(potion_any_time_count, 5.0, 2.0)),
        ("potion_enemy_target_count_norm", _ratio(potion_enemy_target_count, 5.0, 2.0)),
        ("potion_self_target_count_norm", _ratio(potion_self_target_count, 5.0, 2.0)),
        ("orb_count_norm", _ratio(len(semantic_state.orb_ids), 10.0, 2.0)),
        ("hand_count_norm", _ratio(len(semantic_state.hand), 10.0, 2.0)),
        ("playable_card_count_norm", _ratio(len(playable_cards), 10.0, 2.0)),
        ("attack_hand_count_norm", _ratio(_count_cards_by_type(semantic_state.hand, "Attack"), 10.0, 2.0)),
        ("skill_hand_count_norm", _ratio(_count_cards_by_type(semantic_state.hand, "Skill"), 10.0, 2.0)),
        ("power_hand_count_norm", _ratio(_count_cards_by_type(semantic_state.hand, "Power"), 10.0, 2.0)),
        ("status_hand_count_norm", _ratio(_count_cards_by_type(semantic_state.hand, "Status"), 10.0, 2.0)),
        ("curse_hand_count_norm", _ratio(_count_cards_by_type(semantic_state.hand, "Curse"), 10.0, 2.0)),
        ("upgraded_hand_count_norm", _ratio(sum(1 for card in semantic_state.hand if card.is_upgraded), 10.0, 2.0)),
        ("hand_enchantment_count_norm", _ratio(hand_enchantment_count, 20.0, 2.0)),
        ("hand_affliction_count_norm", _ratio(hand_affliction_count, 20.0, 2.0)),
        ("hand_enchantment_amount_norm", _ratio(hand_enchantment_amount, 30.0, 2.0)),
        ("hand_affliction_amount_norm", _ratio(hand_affliction_amount, 30.0, 2.0)),
        ("visible_card_enchantment_count_norm", _ratio(all_enchantment_count, 60.0, 2.0)),
        ("visible_card_affliction_count_norm", _ratio(all_affliction_count, 60.0, 2.0)),
        ("visible_card_enchantment_amount_norm", _ratio(all_enchantment_amount, 80.0, 2.0)),
        ("visible_card_affliction_amount_norm", _ratio(all_affliction_amount, 80.0, 2.0)),
        ("playable_zero_cost_norm", _ratio(sum(1 for cost in playable_costs if cost == 0.0), 10.0, 2.0)),
        ("playable_one_cost_norm", _ratio(sum(1 for cost in playable_costs if 0.0 < cost <= 1.0), 10.0, 2.0)),
        ("playable_high_cost_norm", _ratio(sum(1 for cost in playable_costs if cost >= 2.0), 10.0, 2.0)),
        ("playable_enemy_target_norm", _ratio(_count_cards_by_target(playable_cards, ENEMY_TARGET_TYPES), 10.0, 2.0)),
        ("playable_all_enemy_norm", _ratio(_count_cards_by_target(playable_cards, ALL_ENEMY_TARGET_TYPES), 10.0, 2.0)),
        ("playable_self_target_norm", _ratio(_count_cards_by_target(playable_cards, SELF_TARGET_TYPES), 10.0, 2.0)),
        ("playable_cost_sum_norm", _ratio(sum(playable_costs), 20.0, 2.0)),
        ("draw_pile_count_norm", _ratio(semantic_state.draw_pile_count, 50.0, 2.0)),
        ("discard_pile_count_norm", _ratio(semantic_state.discard_pile_count, 50.0, 2.0)),
        ("exhaust_pile_count_norm", _ratio(semantic_state.exhaust_pile_count, 50.0, 2.0)),
    ):
        builder.add(name, value)

    for name, value in (
        ("enemy_count_norm", _ratio(len(semantic_state.living_enemies), 4.0, 2.0)),
        ("enemy_hp_total_norm", _ratio(semantic_state.total_enemy_hp, 300.0, 2.0)),
        ("enemy_hp_ratio_sum_norm", _ratio(sum(_safe_div(enemy.hp, max(enemy.max_hp, 1.0)) for enemy in semantic_state.living_enemies), 3.0, 2.0)),
        ("enemy_hp_pressure_ratio", _safe_div(semantic_state.total_enemy_hp, max(semantic_state.total_enemy_max_hp, 1.0))),
        ("enemy_block_sum_norm", _ratio(enemy_block_total, 100.0, 2.0)),
        ("enemy_incoming_damage_norm", _ratio(semantic_state.incoming_damage, 50.0, 2.0)),
        ("enemy_attack_intent_count_norm", _ratio(enemy_attack_intents, 3.0, 2.0)),
        ("enemy_buff_intent_count_norm", _ratio(enemy_buff_intents, 3.0, 2.0)),
        ("enemy_debuff_intent_count_norm", _ratio(enemy_debuff_intents, 3.0, 2.0)),
        ("enemy_power_count_norm", _ratio(enemy_power_count, 20.0, 2.0)),
        ("enemy_positive_power_amount_norm", _ratio(enemy_positive_power, 40.0, 2.0)),
        ("enemy_negative_power_amount_norm", _ratio(enemy_negative_power, 40.0, 2.0)),
        ("enemy_max_intent_damage_norm", _ratio(enemy_max_intent_damage, 40.0, 2.0)),
        ("block_gap_norm", _ratio(block_gap, max(semantic_state.player_max_hp, 1.0), 2.0)),
        ("overblock_norm", _ratio(overblock, max(semantic_state.player_max_hp, 1.0), 2.0)),
        ("block_coverage_ratio", 1.0 if semantic_state.incoming_damage <= 0.0 else min(1.0, semantic_state.player_block / max(semantic_state.incoming_damage, 1.0))),
        ("tactical_lethal_exists", _bool(tactical_analysis.lethal_exists)),
        (
            "tactical_required_block_after_best_kill_norm",
            _ratio(tactical_analysis.min_required_block_after_best_kill, max(semantic_state.player_max_hp, 1.0), 2.0),
        ),
        ("build_deck_total_score_norm", _ratio(strategy_context.deck_profile.total_score, 3.0, 2.0)),
        ("build_deck_velocity_norm", _ratio(strategy_context.deck_profile.velocity, 3.0, 2.0)),
        ("build_deck_cohesion_norm", _ratio(strategy_context.deck_profile.cohesion, 3.0, 2.0)),
        ("build_deck_quality_norm", _ratio(strategy_context.deck_profile.quality, 3.0, 2.0)),
        ("build_deck_win_condition_norm", _ratio(strategy_context.deck_profile.win_condition, 3.0, 2.0)),
        ("build_gap_frontload_norm", _strategy_vector_norm(strategy_context.gap, "frontload")),
        ("build_gap_aoe_norm", _strategy_vector_norm(strategy_context.gap, "aoe")),
        ("build_gap_block_norm", _strategy_vector_norm(strategy_context.gap, "block")),
        ("build_gap_draw_norm", _strategy_vector_norm(strategy_context.gap, "draw")),
        ("build_gap_energy_norm", _strategy_vector_norm(strategy_context.gap, "energy")),
        ("build_gap_scaling_norm", _strategy_vector_norm(strategy_context.gap, "scaling")),
        ("build_gap_consistency_norm", _strategy_vector_norm(strategy_context.gap, "consistency")),
        ("build_gap_utility_norm", _strategy_vector_norm(strategy_context.gap, "utility")),
        ("build_gap_early_norm", _strategy_vector_norm(strategy_context.gap, "early")),
        ("build_gap_late_norm", _strategy_vector_norm(strategy_context.gap, "late")),
        ("build_deck_strength_norm", _ratio(strength_economy.current_strength, 4.0, 2.0)),
        ("build_deck_hp_strength_norm", _ratio(deck_hp_strength, 4.0, 2.0)),
        ("act_normal_monster_count_norm", _ratio(normal_monster_count, 6.0, 2.0)),
        ("act_easy_monsters_remaining_norm", _ratio(max(0.0, 3.0 - normal_monster_count), 3.0, 2.0)),
        ("act_hard_monster_pool_active", _bool(normal_monster_count >= 3.0)),
        (
            "build_best_route_projected_deck_hp_strength_norm",
            _ratio(best_route_eval.projected_deck_hp_strength if best_route_eval is not None else 0.0, 4.0, 2.0),
        ),
        (
            "build_best_route_next_risk_strength_norm",
            _ratio(best_route_eval.next_risk_strength if best_route_eval is not None else 0.0, 4.0, 2.0),
        ),
        (
            "build_best_route_boss_margin_norm",
            _signed_ratio(best_route_eval.boss_margin if best_route_eval is not None else 0.0, 2.0, 2.0),
        ),
        (
            "build_best_route_constraint_score_norm",
            _signed_ratio(best_route_eval.constraint_score if best_route_eval is not None else 0.0, 8.0, 2.0),
        ),
        (
            "build_best_route_growth_score_norm",
            _signed_ratio(best_route_eval.growth_score if best_route_eval is not None else 0.0, 4.0, 2.0),
        ),
        (
            "build_best_route_total_score_norm",
            _signed_ratio(best_route_eval.total_score if best_route_eval is not None else 0.0, 10.0, 2.0),
        ),
        ("build_best_route_next_step_feasible", _bool(best_route_eval is not None and best_route_eval.next_step_feasible >= 0.5)),
        ("build_best_route_boss_feasible", _bool(best_route_eval is not None and best_route_eval.boss_feasible >= 0.5)),
        ("build_best_route_path_feasible", _bool(best_route_eval is not None and best_route_eval.path_feasible >= 0.5)),
        ("reward_item_count_norm", _ratio(len(semantic_state.reward_items), 5.0, 2.0)),
        ("reward_card_count_norm", _ratio(reward_card_count, 5.0, 2.0)),
        ("reward_relic_count_norm", _ratio(reward_relic_count, 5.0, 2.0)),
        ("reward_potion_count_norm", _ratio(reward_potion_count, 5.0, 2.0)),
        ("reward_gold_count_norm", _ratio(reward_gold_count, 5.0, 2.0)),
        ("reward_other_count_norm", _ratio(reward_other_count, 5.0, 2.0)),
        ("rewards_can_proceed", _bool(semantic_state.rewards_can_proceed)),
        ("card_reward_count_norm", _ratio(len(semantic_state.card_reward_cards), 3.0, 2.0)),
        ("card_reward_attack_count_norm", _ratio(_count_cards_by_type(semantic_state.card_reward_cards, "Attack"), 3.0, 2.0)),
        ("card_reward_skill_count_norm", _ratio(_count_cards_by_type(semantic_state.card_reward_cards, "Skill"), 3.0, 2.0)),
        ("card_reward_power_count_norm", _ratio(_count_cards_by_type(semantic_state.card_reward_cards, "Power"), 3.0, 2.0)),
        ("card_reward_can_skip", _bool(semantic_state.card_reward_can_skip)),
        ("map_remaining_node_count_norm", _ratio(semantic_state.map_remaining_node_count, 40.0, 2.0)),
        ("map_remaining_boss_distance_norm", _ratio(max(0, semantic_state.map_remaining_boss_distance), 15.0, 2.0)),
        ("map_option_count_norm", _ratio(len(semantic_state.map_options), 6.0, 2.0)),
        ("map_elite_count_norm", _ratio(_count_option_type(semantic_state.map_options, "Elite"), 6.0, 2.0)),
        ("map_rest_count_norm", _ratio(_count_option_type(semantic_state.map_options, "RestSite"), 6.0, 2.0)),
        ("map_shop_count_norm", _ratio(_count_option_type(semantic_state.map_options, "Shop"), 6.0, 2.0)),
        ("map_event_count_norm", _ratio(_count_option_type(semantic_state.map_options, "Event"), 6.0, 2.0)),
        ("map_treasure_count_norm", _ratio(_count_option_type(semantic_state.map_options, "Treasure"), 6.0, 2.0)),
        ("map_boss_count_norm", _ratio(_count_option_type(semantic_state.map_options, "Boss"), 6.0, 2.0)),
        ("event_option_count_norm", _ratio(len(semantic_state.event_options), 5.0, 2.0)),
        ("event_unlocked_option_count_norm", _ratio(sum(1 for option in semantic_state.event_options if not option.is_locked), 5.0, 2.0)),
        ("event_proceed_option_count_norm", _ratio(sum(1 for option in semantic_state.event_options if option.is_proceed), 5.0, 2.0)),
        ("event_relic_option_count_norm", _ratio(sum(1 for option in semantic_state.event_options if option.relic_id), 5.0, 2.0)),
        ("event_in_dialogue", _bool(semantic_state.event_in_dialogue)),
        ("rest_option_count_norm", _ratio(len(semantic_state.rest_options), 5.0, 2.0)),
        ("rest_enabled_option_count_norm", _ratio(sum(1 for option in semantic_state.rest_options if option.is_enabled), 5.0, 2.0)),
        ("rest_proceed_option_count_norm", _ratio(sum(1 for option in semantic_state.rest_options if option.is_proceed), 5.0, 2.0)),
        ("rest_can_proceed", _bool(semantic_state.rest_can_proceed)),
        ("card_select_count_norm", _ratio(len(semantic_state.card_select_cards), 20.0, 2.0)),
        ("card_select_preview_count_norm", _ratio(len(semantic_state.card_select_preview_cards), 20.0, 2.0)),
        ("card_select_upgraded_count_norm", _ratio(sum(1 for card in semantic_state.card_select_cards if card.is_upgraded), 20.0, 2.0)),
        ("card_select_attack_count_norm", _ratio(_count_cards_by_type(semantic_state.card_select_cards, "Attack"), 20.0, 2.0)),
        ("card_select_skill_count_norm", _ratio(_count_cards_by_type(semantic_state.card_select_cards, "Skill"), 20.0, 2.0)),
        ("card_select_power_count_norm", _ratio(_count_cards_by_type(semantic_state.card_select_cards, "Power"), 20.0, 2.0)),
        ("card_select_is_consume", _bool(semantic_state.card_select_is_consume)),
        ("card_select_is_upgrade", _bool(semantic_state.card_select_is_upgrade)),
        ("card_select_is_remove", _bool(semantic_state.card_select_is_remove)),
        ("card_select_is_enchant", _bool(semantic_state.card_select_is_enchant)),
        ("card_select_can_confirm", _bool(semantic_state.card_select_can_confirm)),
        ("card_select_can_cancel", _bool(semantic_state.card_select_can_cancel)),
        ("bundle_count_norm", _ratio(semantic_state.bundle_count, 3.0, 2.0)),
        ("bundle_card_count_norm", _ratio(len(semantic_state.bundle_cards), 9.0, 2.0)),
        ("bundle_preview_count_norm", _ratio(len(semantic_state.bundle_preview_cards), 9.0, 2.0)),
        ("bundle_upgraded_count_norm", _ratio(sum(1 for card in semantic_state.bundle_cards if card.is_upgraded), 9.0, 2.0)),
        ("bundle_can_confirm", _bool(semantic_state.bundle_can_confirm)),
        ("bundle_can_cancel", _bool(semantic_state.bundle_can_cancel)),
        ("hand_select_count_norm", _ratio(len(semantic_state.hand_select_cards), 10.0, 2.0)),
        ("hand_select_selected_count_norm", _ratio(len(semantic_state.hand_select_selected_cards), 10.0, 2.0)),
        ("hand_select_playable_count_norm", _ratio(sum(1 for card in semantic_state.hand_select_cards if card.can_play), 10.0, 2.0)),
        ("hand_select_is_consume", _bool(semantic_state.hand_select_is_consume)),
        ("hand_select_is_upgrade", _bool(semantic_state.hand_select_is_upgrade)),
        ("hand_select_is_remove", _bool(semantic_state.hand_select_is_remove)),
        ("hand_select_is_enchant", _bool(semantic_state.hand_select_is_enchant)),
        ("hand_select_can_confirm", _bool(semantic_state.hand_select_can_confirm)),
        ("relic_select_count_norm", _ratio(len(semantic_state.relic_select_relics), 3.0, 2.0)),
        ("relic_select_can_skip", _bool(semantic_state.relic_select_can_skip)),
        ("treasure_relic_count_norm", _ratio(len(semantic_state.treasure_relics), 3.0, 2.0)),
        ("treasure_can_proceed", _bool(semantic_state.treasure_can_proceed)),
        ("shop_item_count_norm", _ratio(len(semantic_state.shop_items), 12.0, 2.0)),
        ("shop_stocked_count_norm", _ratio(stocked_shop_count, 12.0, 2.0)),
        ("shop_affordable_count_norm", _ratio(affordable_shop_count, 12.0, 2.0)),
        ("shop_card_count_norm", _ratio(_count_shop_category(semantic_state.shop_items, "card", stocked_only=True), 12.0, 2.0)),
        ("shop_relic_count_norm", _ratio(_count_shop_category(semantic_state.shop_items, "relic", stocked_only=True), 12.0, 2.0)),
        ("shop_potion_count_norm", _ratio(_count_shop_category(semantic_state.shop_items, "potion", stocked_only=True), 12.0, 2.0)),
        ("shop_can_proceed", _bool(semantic_state.shop_can_proceed)),
        ("fake_merchant_item_count_norm", _ratio(len(semantic_state.fake_merchant_shop_items), 12.0, 2.0)),
        ("fake_merchant_stocked_count_norm", _ratio(stocked_fake_shop_count, 12.0, 2.0)),
        ("fake_merchant_affordable_count_norm", _ratio(affordable_fake_shop_count, 12.0, 2.0)),
        ("fake_merchant_card_count_norm", _ratio(_count_shop_category(semantic_state.fake_merchant_shop_items, "card", stocked_only=True), 12.0, 2.0)),
        ("fake_merchant_relic_count_norm", _ratio(_count_shop_category(semantic_state.fake_merchant_shop_items, "relic", stocked_only=True), 12.0, 2.0)),
        ("fake_merchant_potion_count_norm", _ratio(_count_shop_category(semantic_state.fake_merchant_shop_items, "potion", stocked_only=True), 12.0, 2.0)),
        ("fake_merchant_can_proceed", _bool(semantic_state.fake_merchant_can_proceed)),
        ("crystal_clickable_count_norm", _ratio(semantic_state.crystal_clickable_count, 8.0, 2.0)),
        ("crystal_can_proceed", _bool(semantic_state.crystal_can_proceed)),
        ("potion_overflow_risk", _bool(full_potion_slots and reward_potion_count > 0)),
    ):
        builder.add(name, value)

    for name in MECHANIC_DIMENSIONS:
        builder.add(
            f"monster_mechanic_{name}_norm",
            _monster_mechanic_norm(encounter_profile.mechanics, name),
        )
    for name in STRATEGY_DIMENSIONS:
        builder.add(
            f"monster_strategy_{name}_norm",
            _monster_strategy_norm(encounter_profile.strategy, name),
        )
    focus_priorities = sorted(
        (profile.focus_priority for profile in encounter_profile.enemy_profiles),
        reverse=True,
    )
    builder.add("monster_focus_target_count_norm", _ratio(sum(1 for value in focus_priorities if value >= 0.70), 3.0, 2.0))
    builder.add("monster_primary_focus_priority_norm", _ratio(focus_priorities[0] if len(focus_priorities) >= 1 else 0.0, 2.5, 2.0))
    builder.add("monster_secondary_focus_priority_norm", _ratio(focus_priorities[1] if len(focus_priorities) >= 2 else 0.0, 2.5, 2.0))
    builder.add("monster_tertiary_focus_priority_norm", _ratio(focus_priorities[2] if len(focus_priorities) >= 3 else 0.0, 2.5, 2.0))

    for node_type in MAP_ROUTE_TRACKED_TYPES:
        builder.add(
            f"map_remaining_ratio.{node_type}",
            map_type_ratio(semantic_state.map_remaining_type_counts, node_type),
        )
    for depth_index, depth_label in enumerate(MAP_ROUTE_DEPTH_LABELS):
        for node_type in MAP_ROUTE_TRACKED_TYPES:
            builder.add(
                f"map_remaining_depth_ratio.{depth_label}.{node_type}",
                map_depth_type_ratio(semantic_state.map_remaining_depth_type_counts, depth_index, node_type),
            )

    route_by_index = {route.option_index: route for route in semantic_state.map_route_summaries}
    for option_index in range(MAP_ROUTE_OPTION_COUNT):
        route = route_by_index.get(option_index)
        prefix = f"map_route.{option_index}"
        builder.add(f"{prefix}.available", _bool(route is not None))
        builder.add(f"{prefix}.reachable_count_norm", _ratio(route.reachable_count if route else 0, 20.0, 2.0))
        builder.add(f"{prefix}.path_count_norm", _ratio(route.path_count if route else 0, 12.0, 2.0))
        builder.add(f"{prefix}.boss_distance_norm", _ratio(max(0, route.boss_distance) if route else 0, 15.0, 2.0))
        builder.add(f"{prefix}.route_score_norm", _ratio(route.route_score if route else 0.0, 12.0, 2.0))
        route_profile = _route_profile(route, normal_monster_count=normal_monster_count) if route else {}
        route_eval = route_eval_by_option.get(option_index)
        builder.add(f"{prefix}.future_easy_monsters_norm", _ratio(route_profile.get("future_easy_monsters", 0.0), 4.0, 2.0))
        builder.add(f"{prefix}.future_hard_monsters_norm", _ratio(route_profile.get("future_hard_monsters", 0.0), 4.0, 2.0))
        builder.add(f"{prefix}.future_question_hard_nodes_norm", _ratio(route_profile.get("future_question_hard_nodes", 0.0), 4.0, 2.0))
        builder.add(f"{prefix}.difficulty_load_norm", _ratio(route_profile.get("difficulty_load", 0.0), 12.0, 2.0))
        builder.add(f"{prefix}.question_difficulty_upper_norm", _ratio(route_profile.get("question_difficulty_upper", 0.0), 2.0, 2.0))
        builder.add(f"{prefix}.next_risk_strength_norm", _ratio(route_profile.get("risk_1_strength", 0.0), 4.0, 2.0))
        builder.add(
            f"{prefix}.projected_deck_hp_strength_norm",
            _ratio(route_eval.projected_deck_hp_strength if route_eval is not None else 0.0, 4.0, 2.0),
        )
        builder.add(
            f"{prefix}.boss_margin_norm",
            _signed_ratio(route_eval.boss_margin if route_eval is not None else 0.0, 2.0, 2.0),
        )
        builder.add(
            f"{prefix}.constraint_score_norm",
            _signed_ratio(route_eval.constraint_score if route_eval is not None else 0.0, 8.0, 2.0),
        )
        builder.add(
            f"{prefix}.growth_score_norm",
            _signed_ratio(route_eval.growth_score if route_eval is not None else 0.0, 4.0, 2.0),
        )
        builder.add(
            f"{prefix}.total_score_norm",
            _signed_ratio(route_eval.total_score if route_eval is not None else 0.0, 10.0, 2.0),
        )
        builder.add(f"{prefix}.next_step_feasible", _bool(route_eval is not None and route_eval.next_step_feasible >= 0.5))
        builder.add(f"{prefix}.boss_feasible", _bool(route_eval is not None and route_eval.boss_feasible >= 0.5))
        builder.add(f"{prefix}.path_feasible", _bool(route_eval is not None and route_eval.path_feasible >= 0.5))
        for node_type in MAP_ROUTE_TRACKED_TYPES:
            builder.add(
                f"{prefix}.ratio.{node_type}",
                map_type_ratio(route.type_counts, node_type) if route else 0.0,
            )
        for depth_index, depth_label in enumerate(MAP_ROUTE_DEPTH_LABELS):
            for node_type in MAP_ROUTE_TRACKED_TYPES:
                builder.add(
                    f"{prefix}.depth_ratio.{depth_label}.{node_type}",
                    map_depth_type_ratio(route.depth_type_counts, depth_index, node_type) if route else 0.0,
                )

    hand_by_index = {
        card.index: card
        for card in semantic_state.hand
        if 0 <= card.index < HAND_SLOT_COUNT
    }
    for slot_index in range(HAND_SLOT_COUNT):
        card = hand_by_index.get(slot_index)
        prefix = f"hand_slot.{slot_index}"
        enchantment_count = len(card.enchantments) if card else 0
        affliction_count = len(card.afflictions) if card else 0
        builder.add(f"{prefix}.present", _bool(card is not None))
        builder.add(f"{prefix}.playable", _bool(card is not None and card.can_play))
        builder.add(f"{prefix}.cost_norm", _ratio(_cost_units(card.cost, semantic_state.player_energy) if card else 0.0, 5.0, 2.0))
        builder.add(f"{prefix}.is_attack", _bool(card is not None and card.card_type == "Attack"))
        builder.add(f"{prefix}.is_skill", _bool(card is not None and card.card_type == "Skill"))
        builder.add(f"{prefix}.is_power", _bool(card is not None and card.card_type == "Power"))
        builder.add(f"{prefix}.is_status", _bool(card is not None and card.card_type == "Status"))
        builder.add(f"{prefix}.is_curse", _bool(card is not None and card.card_type == "Curse"))
        builder.add(f"{prefix}.targets_enemy", _bool(card is not None and card.target_type in ENEMY_TARGET_TYPES))
        builder.add(f"{prefix}.targets_all_enemy", _bool(card is not None and card.target_type in ALL_ENEMY_TARGET_TYPES))
        builder.add(f"{prefix}.targets_self", _bool(card is not None and card.target_type in SELF_TARGET_TYPES))
        builder.add(f"{prefix}.consumes_on_play", _bool(card is not None and card.consumes_on_play))
        builder.add(f"{prefix}.upgraded", _bool(card is not None and card.is_upgraded))
        builder.add(f"{prefix}.enchantment_count_norm", _ratio(enchantment_count, 4.0, 2.0))
        builder.add(f"{prefix}.affliction_count_norm", _ratio(affliction_count, 4.0, 2.0))
        builder.add(f"{prefix}.op_consume", _bool(card is not None and card.consumes_on_play))
        builder.add(f"{prefix}.op_upgrade", _bool(card is not None and card.is_upgraded))
        builder.add(f"{prefix}.op_remove", 0.0)
        builder.add(f"{prefix}.op_enchant", _bool(card is not None and enchantment_count > 0))
        builder.add(
            f"hand_order_slot.{slot_index}.priority_norm",
            _signed_ratio(hand_order_profile.slot_priorities[slot_index], HAND_ORDER_PRIORITY_SCALE, 1.0),
        )
        builder.add(
            f"hand_order_slot.{slot_index}.before_count_norm",
            _ratio(hand_order_profile.slot_before_counts[slot_index], HAND_SLOT_COUNT - 1, 1.0),
        )
        builder.add(
            f"hand_order_slot.{slot_index}.after_count_norm",
            _ratio(hand_order_profile.slot_after_counts[slot_index], HAND_SLOT_COUNT - 1, 1.0),
        )
        builder.add(
            f"hand_order_slot.{slot_index}.prefer_early",
            _bool(hand_order_profile.slot_before_counts[slot_index] > hand_order_profile.slot_after_counts[slot_index]),
        )
        builder.add(
            f"hand_order_slot.{slot_index}.prefer_late",
            _bool(hand_order_profile.slot_after_counts[slot_index] > hand_order_profile.slot_before_counts[slot_index]),
        )

    card_select_by_index = {
        card.index: card
        for card in semantic_state.card_select_cards
        if 0 <= card.index < CARD_SELECT_SLOT_COUNT
    }
    for slot_index in range(CARD_SELECT_SLOT_COUNT):
        card = card_select_by_index.get(slot_index)
        prefix = f"card_select_slot.{slot_index}"
        enchantment_count = len(card.enchantments) if card else 0
        affliction_count = len(card.afflictions) if card else 0
        builder.add(f"{prefix}.present", _bool(card is not None))
        builder.add(f"{prefix}.playable", _bool(card is not None and card.can_play))
        builder.add(f"{prefix}.cost_norm", _ratio(_cost_units(card.cost, semantic_state.player_energy) if card else 0.0, 5.0, 2.0))
        builder.add(f"{prefix}.is_attack", _bool(card is not None and card.card_type == "Attack"))
        builder.add(f"{prefix}.is_skill", _bool(card is not None and card.card_type == "Skill"))
        builder.add(f"{prefix}.is_power", _bool(card is not None and card.card_type == "Power"))
        builder.add(f"{prefix}.is_status", _bool(card is not None and card.card_type == "Status"))
        builder.add(f"{prefix}.is_curse", _bool(card is not None and card.card_type == "Curse"))
        builder.add(f"{prefix}.consumes_on_play", _bool(card is not None and card.consumes_on_play))
        builder.add(f"{prefix}.upgraded", _bool(card is not None and card.is_upgraded))
        builder.add(f"{prefix}.enchantment_count_norm", _ratio(enchantment_count, 4.0, 2.0))
        builder.add(f"{prefix}.affliction_count_norm", _ratio(affliction_count, 4.0, 2.0))
        builder.add(f"{prefix}.op_consume", _bool(card is not None and semantic_state.card_select_is_consume))
        builder.add(f"{prefix}.op_upgrade", _bool(card is not None and semantic_state.card_select_is_upgrade))
        builder.add(f"{prefix}.op_remove", _bool(card is not None and semantic_state.card_select_is_remove))
        builder.add(f"{prefix}.op_enchant", _bool(card is not None and semantic_state.card_select_is_enchant))

    shop_item_by_index = {item.index: item for item in visible_shop_items if 0 <= item.index < SHOP_SLOT_COUNT}
    for slot_index in range(SHOP_SLOT_COUNT):
        item = shop_item_by_index.get(slot_index)
        prefix = f"shop_slot.{slot_index}"
        builder.add(f"{prefix}.present", _bool(item is not None))
        builder.add(f"{prefix}.stocked", _bool(item.is_stocked if item else False))
        builder.add(f"{prefix}.affordable", _bool(item.can_afford if item else False))
        builder.add(f"{prefix}.price_norm", _ratio(item.price if item else 0.0, 250.0, 2.0))
        builder.add(f"{prefix}.is_card", _bool(item is not None and item.category == "card"))
        builder.add(f"{prefix}.is_relic", _bool(item is not None and item.category == "relic"))
        builder.add(f"{prefix}.is_potion", _bool(item is not None and item.category == "potion"))
        builder.add(f"{prefix}.is_card_removal", _bool(item is not None and item.category == "card_removal"))
    event_option_by_index = {
        option.index: option
        for option in semantic_state.event_options
        if 0 <= option.index < EVENT_OPTION_COUNT
    }
    for option_index in range(EVENT_OPTION_COUNT):
        option = event_option_by_index.get(option_index)
        prefix = f"event_slot.{option_index}"
        kind = event_option_kind(option)
        text_length = len(f"{option.title} {option.description}".strip()) if option else 0
        builder.add(f"{prefix}.present", _bool(option is not None))
        builder.add(f"{prefix}.available", _bool(option is not None and not option.is_locked))
        builder.add(f"{prefix}.locked", _bool(option is not None and option.is_locked))
        builder.add(f"{prefix}.proceed", _bool(option is not None and option.is_proceed))
        builder.add(f"{prefix}.chosen", _bool(option is not None and option.was_chosen))
        builder.add(f"{prefix}.has_relic", _bool(option is not None and bool(option.relic_id)))
        builder.add(f"{prefix}.has_card", _bool(option is not None and bool(option.card_id)))
        builder.add(f"{prefix}.has_potion", _bool(option is not None and bool(option.potion_id)))
        builder.add(f"{prefix}.has_keywords", _bool(option is not None and bool(option.keyword_tokens)))
        builder.add(f"{prefix}.has_text", _bool(option is not None and bool(option.title or option.description)))
        builder.add(f"{prefix}.kind_proceed", _bool(kind == "PROCEED"))
        builder.add(f"{prefix}.kind_relic", _bool(kind == "RELIC"))
        builder.add(f"{prefix}.kind_card", _bool(kind == "CARD"))
        builder.add(f"{prefix}.kind_potion", _bool(kind == "POTION"))
        builder.add(f"{prefix}.kind_keyword", _bool(kind == "KEYWORD"))
        builder.add(f"{prefix}.kind_text", _bool(kind == "TEXT"))
        builder.add(f"{prefix}.keyword_count_norm", _ratio(len(option.keyword_tokens) if option else 0, 4.0, 2.0))
        builder.add(f"{prefix}.text_length_norm", _ratio(text_length, 200.0, 2.0))

    hand_select_by_index = {
        card.index: card
        for card in semantic_state.hand_select_cards
        if 0 <= card.index < HAND_SELECT_SLOT_COUNT
    }
    selected_hand_indices = {
        card.index
        for card in semantic_state.hand_select_selected_cards
        if card.index >= 0
    }
    selected_hand_ids = {
        card.id
        for card in semantic_state.hand_select_selected_cards
        if card.id
    }
    for slot_index in range(HAND_SELECT_SLOT_COUNT):
        card = hand_select_by_index.get(slot_index)
        prefix = f"hand_select_slot.{slot_index}"
        is_selected = bool(card is not None and (card.index in selected_hand_indices or card.id in selected_hand_ids))
        enchantment_count = len(card.enchantments) if card else 0
        affliction_count = len(card.afflictions) if card else 0
        builder.add(f"{prefix}.present", _bool(card is not None))
        builder.add(f"{prefix}.selected", _bool(is_selected))
        builder.add(f"{prefix}.playable", _bool(card is not None and card.can_play))
        builder.add(f"{prefix}.cost_norm", _ratio(_cost_units(card.cost, semantic_state.player_energy) if card else 0.0, 5.0, 2.0))
        builder.add(f"{prefix}.is_attack", _bool(card is not None and card.card_type == "Attack"))
        builder.add(f"{prefix}.is_skill", _bool(card is not None and card.card_type == "Skill"))
        builder.add(f"{prefix}.is_power", _bool(card is not None and card.card_type == "Power"))
        builder.add(f"{prefix}.is_status", _bool(card is not None and card.card_type == "Status"))
        builder.add(f"{prefix}.is_curse", _bool(card is not None and card.card_type == "Curse"))
        builder.add(f"{prefix}.targets_enemy", _bool(card is not None and card.target_type in ENEMY_TARGET_TYPES))
        builder.add(f"{prefix}.targets_all_enemy", _bool(card is not None and card.target_type in ALL_ENEMY_TARGET_TYPES))
        builder.add(f"{prefix}.targets_self", _bool(card is not None and card.target_type in SELF_TARGET_TYPES))
        builder.add(f"{prefix}.consumes_on_play", _bool(card is not None and card.consumes_on_play))
        builder.add(f"{prefix}.upgraded", _bool(card is not None and card.is_upgraded))
        builder.add(f"{prefix}.enchantment_count_norm", _ratio(enchantment_count, 4.0, 2.0))
        builder.add(f"{prefix}.affliction_count_norm", _ratio(affliction_count, 4.0, 2.0))
        builder.add(f"{prefix}.op_consume", _bool(card is not None and semantic_state.hand_select_is_consume))
        builder.add(f"{prefix}.op_upgrade", _bool(card is not None and semantic_state.hand_select_is_upgrade))
        builder.add(f"{prefix}.op_remove", _bool(card is not None and semantic_state.hand_select_is_remove))
        builder.add(f"{prefix}.op_enchant", _bool(card is not None and semantic_state.hand_select_is_enchant))

    for bucket in COMBAT_ROUND_BUCKETS:
        builder.add(f"combat_round_bucket.{bucket}", 1.0 if bucket == round_bucket else 0.0)

    return builder


def _extract_base_observation_features(
    state: dict[str, object],
    strategy_context: BuildStrategyContext | None = None,
    normal_monster_count: float = 0.0,
) -> list[float]:
    return _build_state_space_features(state, strategy_context, normal_monster_count=normal_monster_count).values


def extract_observation_features(state: dict[str, object]) -> list[float]:
    semantic_observation = encode_semantic_observation(state)
    features = _extract_base_observation_features(state, build_strategy_context(state=state), normal_monster_count=0.0)
    features.extend(semantic_observation.vector)
    return features


STATE_SPACE_FEATURE_NAMES = tuple(_build_state_space_features({"state_type": "menu"}).names)
STATE_SPACE_OBSERVATION_SIZE = len(STATE_SPACE_FEATURE_NAMES)
BASE_OBSERVATION_SIZE = STATE_SPACE_OBSERVATION_SIZE
SEMANTIC_CONCEPT_SIZE = CONCEPT_VOCAB_SIZE
SEMANTIC_OBSERVATION_SIZE = SEMANTIC_VECTOR_SIZE
OBSERVATION_SIZE = BASE_OBSERVATION_SIZE + SEMANTIC_OBSERVATION_SIZE


class STS2MuZeroEnv:
    def __init__(
        self,
        bridge: STS2Bridge,
        poll_interval: float = 0.15,
        max_poll_attempts: int = 40,
        reward_discount: float = 0.997,
        reward_weights: RewardWeights | None = None,
        card_select_candidate_limit: int = 6,
        block_premature_end_turn: bool = True,
        end_turn_guard_stall_limit: int = 3,
        end_turn_guard_timeout_seconds: float = 3.0,
        require_card_reward_preview_before_proceed: bool = True,
        card_reward_preview_guard_timeout_seconds: float = 3.0,
    ) -> None:
        self.bridge = bridge
        self.poll_interval = poll_interval
        self.max_poll_attempts = max_poll_attempts
        self.reward_discount = reward_discount
        self.reward_weights = reward_weights or RewardWeights()
        self.card_select_candidate_limit = max(0, int(card_select_candidate_limit))
        self.block_premature_end_turn = block_premature_end_turn
        self.end_turn_guard_stall_limit = max(1, int(end_turn_guard_stall_limit))
        self.end_turn_guard_timeout_seconds = max(0.0, float(end_turn_guard_timeout_seconds))
        self.require_card_reward_preview_before_proceed = require_card_reward_preview_before_proceed
        self.card_reward_preview_guard_timeout_seconds = float(card_reward_preview_guard_timeout_seconds)
        self._disable_play_card_above_ten = False
        self._combat_stall_counts: dict[tuple[object, ...], int] = {}
        self._combat_end_turn_guard_started_at: dict[tuple[object, ...], float] = {}
        self._card_reward_guard_anchor: tuple[int, int] | None = None
        self._card_reward_guard_started_at: float | None = None
        self._card_reward_preview_seen = False
        self._campfire_upgrade_guard_anchor: tuple[int, int] | None = None
        self._potion_overflow_guard_anchor: tuple[object, ...] | None = None
        self._potion_overflow_discard_used = False
        self._active_hand_select_signature: tuple[object, ...] | None = None
        self._hand_select_attempted_indices: dict[tuple[object, ...], set[int]] = {}
        self._active_hand_select_anchor: tuple[object, ...] | None = None
        self._hand_select_seen_without_confirm: dict[tuple[object, ...], bool] = {}
        self._active_card_select_signature: tuple[object, ...] | None = None
        self._card_select_attempted_indices: dict[tuple[object, ...], set[int]] = {}
        self._active_bundle_select_signature: tuple[object, ...] | None = None
        self._bundle_select_attempted_indices: dict[tuple[object, ...], set[int]] = {}
        self._active_event_repeat_signature: tuple[object, ...] | None = None
        self._event_repeat_blocked_options: dict[tuple[object, ...], set[int]] = {}
        self.semantic_history = SemanticHistoryTracker()
        self._route_strength_progress: tuple[str, int, int] | None = None
        self._observed_deck_cards: tuple[CardSnapshot, ...] = ()
        self._observed_monster_history: tuple[float, ...] = tuple(0.0 for _ in CAPABILITY_DIMENSIONS)
        self._observed_normal_monster_counts: dict[int, int] = {}
        self._counted_normal_monster_nodes: set[tuple[int, int, str]] = set()
        self._observed_build_context: BuildStrategyContext | None = None
        self._observed_deck_strength = 0.0
        self._observed_relic_item_strengths: tuple[float, ...] = ()
        self._observed_relic_strength = 0.0
        self._observed_resource_strength = 0.0
        self._observed_run_readiness = 0.0

    def fetch_state(self) -> dict[str, object]:
        return self.bridge.get_game_state(format="json")

    def extract_observation_features(self, state: dict[str, object]) -> list[float]:
        semantic_state = self._sync_route_strength_state(state)
        semantic_observation = encode_semantic_observation(state, history=self.semantic_history)
        features = _extract_base_observation_features(
            state,
            self._observed_build_context,
            normal_monster_count=float(self._current_normal_monster_count(semantic_state)),
        )
        features.extend(semantic_observation.vector)
        return features

    def extract_strength_targets(self, state: dict[str, object]) -> tuple[list[float], list[float]]:
        semantic_state = self._sync_route_strength_state(state)
        if self._observed_build_context is None:
            return [0.0 for _ in AUXILIARY_STRENGTH_TARGET_NAMES], [0.0 for _ in AUXILIARY_STRENGTH_TARGET_NAMES]
        strength_economy = estimate_strength_economy(
            self._observed_build_context,
            relic_strength=self._observed_relic_strength,
            resource_strength=self._observed_resource_strength,
        )
        best_route_eval = None
        if str(getattr(semantic_state, "state_type", "unknown")) == "map":
            route_evals = [
                evaluate_route_strength(
                    self._observed_build_context,
                    _route_profile(route, normal_monster_count=float(self._current_normal_monster_count(semantic_state))),
                    act_id=str(getattr(semantic_state, "act_id", "") or ""),
                    floor=int(getattr(semantic_state, "floor", 0) or 0),
                    hp_ratio=_safe_div(getattr(semantic_state, "player_hp", 0.0), max(getattr(semantic_state, "player_max_hp", 0.0), 1.0)),
                    relic_strength=self._observed_relic_strength,
                    resource_strength=self._observed_resource_strength,
                )
                for route in getattr(semantic_state, "map_route_summaries", ())
            ]
            if route_evals:
                best_route_eval = max(route_evals, key=lambda item: item.total_score)
        return _strength_target_vector(
            strategy_context=self._observed_build_context,
            strength_economy=strength_economy,
            route_eval=best_route_eval,
        )

    def _reset_route_strength_state(self) -> None:
        self._route_strength_progress = None
        self._observed_deck_cards = ()
        self._observed_monster_history = tuple(0.0 for _ in CAPABILITY_DIMENSIONS)
        self._observed_normal_monster_counts = {}
        self._counted_normal_monster_nodes = set()
        self._observed_build_context = None
        self._observed_deck_strength = 0.0
        self._observed_relic_item_strengths = ()
        self._observed_relic_strength = 0.0
        self._observed_resource_strength = 0.0
        self._observed_run_readiness = 0.0

    def _sync_normal_monster_progress(self, semantic_state: object) -> None:
        act = int(getattr(semantic_state, "act", 0) or 0)
        if act > 0 and act not in self._observed_normal_monster_counts:
            self._observed_normal_monster_counts[act] = 0
        if str(getattr(semantic_state, "state_type", "unknown")) != "monster" or act <= 0:
            return
        floor = int(getattr(semantic_state, "floor", 0) or 0)
        encounter_id = str(getattr(semantic_state, "encounter_id", "") or "")
        signature = (act, floor, encounter_id or f"FLOOR_{floor}")
        if signature in self._counted_normal_monster_nodes:
            return
        self._counted_normal_monster_nodes.add(signature)
        self._observed_normal_monster_counts[act] = self._observed_normal_monster_counts.get(act, 0) + 1

    def _current_normal_monster_count(self, semantic_state: object) -> int:
        act = int(getattr(semantic_state, "act", 0) or 0)
        if act <= 0:
            return 0
        return int(self._observed_normal_monster_counts.get(act, 0))

    def _sync_route_strength_state(self, state: dict[str, object] | object) -> object:
        semantic_state = ensure_semantic_state(state)
        if semantic_state.state_type in TERMINAL_STATE_TYPES:
            self._reset_route_strength_state()
            return semantic_state
        progress = (
            str(getattr(semantic_state, "character_id", "") or ""),
            int(getattr(semantic_state, "act", 0) or 0),
            int(getattr(semantic_state, "floor", 0) or 0),
        )
        previous_progress = self._route_strength_progress
        if previous_progress is not None:
            _, previous_act, previous_floor = previous_progress
            if progress[1] < previous_act or progress[2] < previous_floor:
                self._reset_route_strength_state()
        self._route_strength_progress = progress
        self._sync_normal_monster_progress(semantic_state)
        raw_state = state if isinstance(state, dict) else {}
        visible_deck_cards = collect_visible_deck_snapshots(raw_state) if raw_state else ()
        if visible_deck_cards and (
            len(visible_deck_cards) >= len(self._observed_deck_cards)
            or semantic_state.state_type in {"card_select", "hand_select"}
        ):
            self._observed_deck_cards = visible_deck_cards
        if raw_state:
            current_monster_demand = estimate_current_combat_demand(raw_state)
            if any(current_monster_demand):
                self._observed_monster_history = update_observed_monster_history(
                    self._observed_monster_history,
                    current_monster_demand,
                )
            self._observed_build_context = build_strategy_context(
                state=raw_state,
                observed_cards=self._observed_deck_cards,
                observed_monster_history=self._observed_monster_history,
            )
            self._observed_deck_strength = self._observed_build_context.deck_profile.total_score
        else:
            self._observed_build_context = None
            self._observed_deck_strength = 0.0
        self._observed_relic_item_strengths, self._observed_relic_strength = _estimate_relic_strengths(tuple(getattr(semantic_state, "relics", ())))
        self._observed_resource_strength = _estimate_resource_strength(semantic_state)
        self._observed_run_readiness = _estimate_run_readiness(
            self._observed_deck_strength,
            self._observed_relic_strength,
            self._observed_resource_strength,
        )
        return semantic_state

    def _evaluate_map_route_strength(self, semantic_state: object, route: object) -> object | None:
        hp_ratio = _safe_div(getattr(semantic_state, "player_hp", 0.0), max(getattr(semantic_state, "player_max_hp", 0.0), 1.0))
        profile = _route_profile(route, normal_monster_count=float(self._current_normal_monster_count(semantic_state)))
        if self._observed_build_context is None:
            return None
        return evaluate_route_strength(
            self._observed_build_context,
            profile,
            act_id=str(getattr(semantic_state, "act_id", "") or ""),
            floor=int(getattr(semantic_state, "floor", 0) or 0),
            hp_ratio=hp_ratio,
            relic_strength=self._observed_relic_strength,
            resource_strength=self._observed_resource_strength,
        )

    def _evaluate_map_route_value(self, semantic_state: object, route: object) -> float:
        profile = _route_profile(route, normal_monster_count=float(self._current_normal_monster_count(semantic_state)))
        score = (0.02 * float(getattr(route, "route_score", 0.0) or 0.0)) + min(0.05, max(0.0, min(profile["path_count"], 6.0) - 1.0) * 0.01)
        route_strength = self._evaluate_map_route_strength(semantic_state, route)
        if route_strength is not None:
            score += float(getattr(route_strength, "total_score", 0.0) or 0.0)
        return score

    def _map_route_evaluations(self, state: dict[str, object] | object) -> tuple[object, dict[int, object]]:
        semantic_state = self._sync_route_strength_state(state)
        if str(getattr(semantic_state, "state_type", "unknown")) != "map":
            return semantic_state, {}
        evaluations: dict[int, object] = {}
        for route in getattr(semantic_state, "map_route_summaries", ()):
            option_index = getattr(route, "option_index", None)
            if not isinstance(option_index, int) or option_index < 0:
                continue
            route_strength = self._evaluate_map_route_strength(semantic_state, route)
            if route_strength is not None:
                evaluations[option_index] = route_strength
        return semantic_state, evaluations

    def _map_route_scores(self, state: dict[str, object] | object) -> tuple[object, dict[int, float]]:
        semantic_state, route_evaluations = self._map_route_evaluations(state)
        if str(getattr(semantic_state, "state_type", "unknown")) != "map":
            return semantic_state, {}
        scores: dict[int, float] = {}
        for route in getattr(semantic_state, "map_route_summaries", ()):
            option_index = getattr(route, "option_index", None)
            if not isinstance(option_index, int) or option_index < 0:
                continue
            profile = _route_profile(route, normal_monster_count=float(self._current_normal_monster_count(semantic_state)))
            base_score = (0.02 * float(getattr(route, "route_score", 0.0) or 0.0)) + min(0.05, max(0.0, min(profile["path_count"], 6.0) - 1.0) * 0.01)
            route_strength = route_evaluations.get(option_index)
            scores[option_index] = base_score + (
                float(getattr(route_strength, "total_score", 0.0) or 0.0)
                if route_strength is not None
                else self._evaluate_map_route_value(semantic_state, route)
            )
        return semantic_state, scores

    def _map_route_action_biases(
        self,
        state: dict[str, object] | object,
        actions: dict[int, BoundAction] | None = None,
    ) -> dict[int, float]:
        semantic_state, route_scores = self._map_route_scores(state)
        _, route_evaluations = self._map_route_evaluations(state)
        if str(getattr(semantic_state, "state_type", "unknown")) != "map" or len(route_scores) <= 1:
            return {}
        legal_actions = actions or {}
        values = tuple(route_scores.values())
        best = max(values)
        worst = min(values)
        mean = sum(values) / max(1, len(values))
        spread = max(0.40, best - worst)
        any_path_feasible = any(float(getattr(item, "path_feasible", 0.0) or 0.0) >= 0.5 for item in route_evaluations.values())
        result: dict[int, float] = {}
        for action_id, bound in legal_actions.items():
            if bound.tool_name != "map_choose_node":
                continue
            node_index = bound.kwargs.get("node_index")
            if not isinstance(node_index, int):
                continue
            chosen_score = route_scores.get(node_index)
            if chosen_score is None:
                continue
            bias = ((chosen_score - mean) / spread) * 2.10
            best_gap = max(0.0, (best - chosen_score) / spread)
            route_eval = route_evaluations.get(node_index)
            if any_path_feasible and route_eval is not None:
                if float(getattr(route_eval, "path_feasible", 0.0) or 0.0) >= 0.5:
                    bias += 0.65
                else:
                    bias -= 0.85
            if best_gap > 0.60:
                bias -= min(0.70, (best_gap - 0.60) * 0.50)
            bias = max(-2.5, min(2.5, bias))
            if abs(bias) >= 1e-6:
                result[action_id] = bias
        return result

    def _is_combat_action_window_ready(self, state: dict[str, object]) -> bool:
        if str(state.get("state_type", "unknown")) not in COMBAT_STATE_TYPES:
            return False
        battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
        hand_mode = str(battle.get("hand_mode", "play") or "play").lower()
        return (
            battle.get("turn") == "player"
            and bool(battle.get("is_play_phase"))
            and not bool(battle.get("player_actions_disabled"))
            and not bool(battle.get("hand_in_card_play"))
            and hand_mode == "play"
        )

    def _combat_progress_signature(self, state: dict[str, object]) -> tuple[object, ...]:
        state_type = str(state.get("state_type", "unknown"))
        run = state.get("run") if isinstance(state.get("run"), dict) else {}
        player = state.get("player") if isinstance(state.get("player"), dict) else {}
        battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
        hand_cards = player.get("hand", []) if isinstance(player.get("hand"), list) else []
        enemies = battle.get("enemies", []) if isinstance(battle.get("enemies"), list) else []
        hand_select = state.get("hand_select") if isinstance(state.get("hand_select"), dict) else {}
        hand_signature = tuple(
            _raw_card_signature(card)
            for card in hand_cards
            if isinstance(card, dict)
        )
        enemy_signature = tuple(
            (
                str(enemy.get("entity_id", "")),
                float(enemy.get("hp", 0) or 0),
                float(enemy.get("block", 0) or 0),
            )
            for enemy in enemies
            if isinstance(enemy, dict)
        )
        selectable_cards = hand_select.get("cards", []) if isinstance(hand_select.get("cards"), list) else []
        selected_cards = hand_select.get("selected_cards", []) if isinstance(hand_select.get("selected_cards"), list) else []
        selectable_signature = tuple(
            (
                *_raw_card_signature(card),
                bool(card.get("is_selected")),
            )
            for card in selectable_cards
            if isinstance(card, dict)
        )
        selected_signature = tuple(
            _raw_card_signature(card)
            for card in selected_cards
            if isinstance(card, dict)
        )
        return (
            state_type,
            int(run.get("act", 0) or 0),
            int(run.get("floor", 0) or 0),
            battle.get("round"),
            str(battle.get("turn", "")),
            bool(battle.get("is_play_phase")),
            bool(battle.get("player_actions_disabled")),
            bool(battle.get("hand_in_card_play")),
            str(battle.get("hand_mode", "")),
            float(player.get("hp", 0) or 0),
            float(player.get("block", 0) or 0),
            float(player.get("energy", 0) or 0),
            int(player.get("draw_pile_count", 0) or 0),
            int(player.get("discard_pile_count", 0) or 0),
            int(player.get("exhaust_pile_count", 0) or 0),
            hand_signature,
            enemy_signature,
            hand_select.get("mode"),
            selectable_signature,
            selected_signature,
            bool(hand_select.get("can_confirm")),
        )

    def _combat_has_living_enemies(self, state: dict[str, object]) -> bool:
        battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
        return any(isinstance(enemy, dict) and float(enemy.get("hp", 0) or 0) > 0.0 for enemy in battle.get("enemies", []))

    def _card_select_progress_signature(self, state: dict[str, object]) -> tuple[object, ...]:
        state_type = str(state.get("state_type", "unknown"))
        if state_type != "card_select":
            return (state_type,)
        run = state.get("run") if isinstance(state.get("run"), dict) else {}
        card_select = state.get("card_select") if isinstance(state.get("card_select"), dict) else {}
        cards = card_select.get("cards") if isinstance(card_select.get("cards"), list) else []
        preview_cards = card_select.get("preview_cards") if isinstance(card_select.get("preview_cards"), list) else []
        card_signature = tuple(_raw_card_signature(card) for card in cards if isinstance(card, dict))
        preview_signature = tuple(_raw_card_signature(card) for card in preview_cards if isinstance(card, dict))
        return (
            state_type,
            int(run.get("act", 0) or 0),
            int(run.get("floor", 0) or 0),
            str(card_select.get("screen_type", "") or ""),
            str(card_select.get("prompt", "") or ""),
            bool(card_select.get("preview_showing")),
            bool(card_select.get("can_confirm")),
            bool(card_select.get("can_cancel") or card_select.get("can_skip")),
            card_signature,
            preview_signature,
        )

    def _hand_select_progress_signature(self, state: dict[str, object]) -> tuple[object, ...]:
        hand_select = state.get("hand_select") if isinstance(state.get("hand_select"), dict) else {}
        has_hand_select = bool(
            isinstance(hand_select.get("cards"), list)
            or isinstance(hand_select.get("selected_cards"), list)
            or hand_select.get("can_confirm")
        )
        state_type = str(state.get("state_type", "unknown"))
        if not has_hand_select:
            return (state_type,)
        run = state.get("run") if isinstance(state.get("run"), dict) else {}
        cards = hand_select.get("cards") if isinstance(hand_select.get("cards"), list) else []
        selected_cards = hand_select.get("selected_cards") if isinstance(hand_select.get("selected_cards"), list) else []
        selectable_signature = tuple(
            (
                *_raw_card_signature(card),
                bool(card.get("is_selected")),
            )
            for card in cards
            if isinstance(card, dict)
        )
        selected_signature = tuple(_raw_card_signature(card) for card in selected_cards if isinstance(card, dict))
        return (
            state_type,
            int(run.get("act", 0) or 0),
            int(run.get("floor", 0) or 0),
            str(hand_select.get("mode", "") or ""),
            str(hand_select.get("prompt", "") or ""),
            bool(hand_select.get("can_confirm")),
            selectable_signature,
            selected_signature,
        )

    def _hand_select_anchor(self, state: dict[str, object]) -> tuple[object, ...] | None:
        hand_select = state.get("hand_select") if isinstance(state.get("hand_select"), dict) else {}
        has_hand_select = bool(
            isinstance(hand_select.get("cards"), list)
            or isinstance(hand_select.get("selected_cards"), list)
            or hand_select.get("can_confirm")
        )
        if not has_hand_select:
            return None
        run = state.get("run") if isinstance(state.get("run"), dict) else {}
        cards = hand_select.get("cards") if isinstance(hand_select.get("cards"), list) else []
        selected_cards = hand_select.get("selected_cards") if isinstance(hand_select.get("selected_cards"), list) else []
        card_pool_signature = tuple(
            sorted(
                _card_equivalence_signature(card)
                for card in [*cards, *selected_cards]
                if isinstance(card, dict)
            )
        )
        return (
            int(run.get("act", 0) or 0),
            int(run.get("floor", 0) or 0),
            str(hand_select.get("mode", "") or ""),
            str(hand_select.get("prompt", "") or ""),
            card_pool_signature,
        )

    def _bundle_select_progress_signature(self, state: dict[str, object]) -> tuple[object, ...]:
        bundle_select = state.get("bundle_select") if isinstance(state.get("bundle_select"), dict) else {}
        bundles = bundle_select.get("bundles") if isinstance(bundle_select.get("bundles"), list) else []
        has_bundle_select = bool(bundles or bundle_select.get("can_confirm") or bundle_select.get("preview_showing"))
        state_type = str(state.get("state_type", "unknown"))
        if not has_bundle_select:
            return (state_type,)
        run = state.get("run") if isinstance(state.get("run"), dict) else {}
        bundle_signature = tuple(
            (
                int(bundle.get("index", -1)) if isinstance(bundle.get("index"), int) else -1,
                int(bundle.get("card_count", 0) or 0),
                tuple(
                    _raw_card_signature(card)
                    for card in bundle.get("cards", [])
                    if isinstance(bundle.get("cards"), list) and isinstance(card, dict)
                ),
            )
            for bundle in bundles
            if isinstance(bundle, dict)
        )
        preview_cards = bundle_select.get("preview_cards") if isinstance(bundle_select.get("preview_cards"), list) else []
        preview_signature = tuple(_raw_card_signature(card) for card in preview_cards if isinstance(card, dict))
        return (
            state_type,
            int(run.get("act", 0) or 0),
            int(run.get("floor", 0) or 0),
            str(bundle_select.get("screen_type", "") or ""),
            str(bundle_select.get("prompt", "") or ""),
            bool(bundle_select.get("preview_showing")),
            bool(bundle_select.get("can_confirm")),
            bool(bundle_select.get("can_cancel")),
            bundle_signature,
            preview_signature,
        )

    def _event_repeat_signature(self, state: dict[str, object]) -> tuple[object, ...] | None:
        if str(state.get("state_type", "unknown")) != "event":
            return None
        run = state.get("run") if isinstance(state.get("run"), dict) else {}
        event_state = state.get("event") if isinstance(state.get("event"), dict) else {}
        return (
            int(run.get("act", 0) or 0),
            int(run.get("floor", 0) or 0),
            str(event_state.get("event_id", "") or ""),
            str(event_state.get("event_name", "") or ""),
        )

    def _stall_count(self, state: dict[str, object]) -> int:
        return self._combat_stall_counts.get(self._combat_progress_signature(state), 0)

    def _clear_combat_guard_state(self, signature: tuple[object, ...]) -> None:
        self._combat_stall_counts.pop(signature, None)
        self._combat_end_turn_guard_started_at.pop(signature, None)

    def _clear_hand_select_guard_state(self, signature: tuple[object, ...]) -> None:
        self._hand_select_attempted_indices.pop(signature, None)
        if self._active_hand_select_signature == signature:
            self._active_hand_select_signature = None

    def _clear_hand_select_anchor_state(self, anchor: tuple[object, ...]) -> None:
        self._hand_select_seen_without_confirm.pop(anchor, None)
        if self._active_hand_select_anchor == anchor:
            self._active_hand_select_anchor = None

    def _sync_hand_select_guard_state(self, state: dict[str, object]) -> None:
        signature = self._hand_select_progress_signature(state)
        anchor = self._hand_select_anchor(state)
        hand_select = state.get("hand_select") if isinstance(state.get("hand_select"), dict) else {}
        if len(signature) == 1 or anchor is None:
            if self._active_hand_select_signature is not None:
                self._clear_hand_select_guard_state(self._active_hand_select_signature)
            if self._active_hand_select_anchor is not None:
                self._clear_hand_select_anchor_state(self._active_hand_select_anchor)
            return
        if self._active_hand_select_anchor != anchor:
            if self._active_hand_select_anchor is not None:
                self._clear_hand_select_anchor_state(self._active_hand_select_anchor)
            self._active_hand_select_anchor = anchor
        self._hand_select_seen_without_confirm.setdefault(anchor, False)
        if not bool(hand_select.get("can_confirm")):
            self._hand_select_seen_without_confirm[anchor] = True
        if self._active_hand_select_signature == signature:
            return
        if self._active_hand_select_signature is not None:
            self._clear_hand_select_guard_state(self._active_hand_select_signature)
        self._active_hand_select_signature = signature

    def _attempted_hand_select_indices(self, state: dict[str, object]) -> set[int]:
        self._sync_hand_select_guard_state(state)
        signature = self._hand_select_progress_signature(state)
        return set(self._hand_select_attempted_indices.get(signature, set()))

    def _hand_select_started_without_confirm(self, state: dict[str, object]) -> bool:
        self._sync_hand_select_guard_state(state)
        anchor = self._hand_select_anchor(state)
        if anchor is None:
            return False
        return bool(self._hand_select_seen_without_confirm.get(anchor))

    def _clear_card_select_guard_state(self, signature: tuple[object, ...]) -> None:
        self._card_select_attempted_indices.pop(signature, None)
        if self._active_card_select_signature == signature:
            self._active_card_select_signature = None

    def _sync_card_select_guard_state(self, state: dict[str, object]) -> None:
        if str(state.get("state_type", "unknown")) != "card_select":
            if self._active_card_select_signature is not None:
                self._clear_card_select_guard_state(self._active_card_select_signature)
            return
        signature = self._card_select_progress_signature(state)
        if self._active_card_select_signature == signature:
            return
        if self._active_card_select_signature is not None:
            self._clear_card_select_guard_state(self._active_card_select_signature)
        self._active_card_select_signature = signature

    def _attempted_card_select_indices(self, state: dict[str, object]) -> set[int]:
        self._sync_card_select_guard_state(state)
        signature = self._card_select_progress_signature(state)
        return set(self._card_select_attempted_indices.get(signature, set()))

    def _clear_bundle_select_guard_state(self, signature: tuple[object, ...]) -> None:
        self._bundle_select_attempted_indices.pop(signature, None)
        if self._active_bundle_select_signature == signature:
            self._active_bundle_select_signature = None

    def _sync_bundle_select_guard_state(self, state: dict[str, object]) -> None:
        signature = self._bundle_select_progress_signature(state)
        if len(signature) == 1:
            if self._active_bundle_select_signature is not None:
                self._clear_bundle_select_guard_state(self._active_bundle_select_signature)
            return
        if self._active_bundle_select_signature == signature:
            return
        if self._active_bundle_select_signature is not None:
            self._clear_bundle_select_guard_state(self._active_bundle_select_signature)
        self._active_bundle_select_signature = signature

    def _attempted_bundle_select_indices(self, state: dict[str, object]) -> set[int]:
        self._sync_bundle_select_guard_state(state)
        signature = self._bundle_select_progress_signature(state)
        return set(self._bundle_select_attempted_indices.get(signature, set()))

    def _clear_event_repeat_guard_state(self, signature: tuple[object, ...]) -> None:
        self._event_repeat_blocked_options.pop(signature, None)
        if self._active_event_repeat_signature == signature:
            self._active_event_repeat_signature = None

    def _sync_event_repeat_guard_state(self, state: dict[str, object]) -> None:
        signature = self._event_repeat_signature(state)
        if signature is None:
            if self._active_event_repeat_signature is not None:
                self._clear_event_repeat_guard_state(self._active_event_repeat_signature)
            return
        if self._active_event_repeat_signature == signature:
            return
        if self._active_event_repeat_signature is not None:
            self._clear_event_repeat_guard_state(self._active_event_repeat_signature)
        self._active_event_repeat_signature = signature

    def _event_has_alternative_option(
        self,
        state: dict[str, object],
        blocked_key: int | None = None,
    ) -> bool:
        event_state = state.get("event") if isinstance(state.get("event"), dict) else {}
        for option in _unlocked_event_options(event_state):
            option_key = _event_option_guard_key(option)
            if blocked_key is not None and option_key == blocked_key:
                continue
            return True
        return False

    def _should_block_repeated_harmful_event_option(self, state: dict[str, object], option: dict[str, object]) -> bool:
        self._sync_event_repeat_guard_state(state)
        signature = self._event_repeat_signature(state)
        if signature is None:
            return False
        option_key = _event_option_guard_key(option)
        blocked = self._event_repeat_blocked_options.get(signature, set())
        if option_key not in blocked:
            return False
        return self._event_has_alternative_option(state, blocked_key=option_key)

    def _reset_card_reward_guard(self) -> None:
        self._card_reward_guard_anchor = None
        self._card_reward_guard_started_at = None
        self._card_reward_preview_seen = False

    def _card_reward_guard_anchor_for_state(self, state: dict[str, object]) -> tuple[int, int]:
        run = state.get("run") if isinstance(state.get("run"), dict) else {}
        return int(run.get("act", 0) or 0), int(run.get("floor", 0) or 0)

    def _potion_overflow_guard_anchor_for_state(self, state: dict[str, object]) -> tuple[object, ...]:
        run = state.get("run") if isinstance(state.get("run"), dict) else {}
        rewards = state.get("rewards") if isinstance(state.get("rewards"), dict) else {}
        items = rewards.get("items") if isinstance(rewards.get("items"), list) else []
        potion_signature = tuple(
            (
                int(item.get("index", -1)),
                str(item.get("type", "") or item.get("category", "") or item.get("kind", "")),
                str(item.get("potion_id", "") or item.get("name", "")),
            )
            for item in items
            if isinstance(item, dict) and _reward_item_is_potion(item)
        )
        return (
            int(run.get("act", 0) or 0),
            int(run.get("floor", 0) or 0),
            potion_signature,
        )

    def _has_pending_card_reward_claim(self, state: dict[str, object]) -> bool:
        rewards = state.get("rewards") if isinstance(state.get("rewards"), dict) else {}
        items = rewards.get("items") if isinstance(rewards.get("items"), list) else []
        return any(isinstance(item, dict) and _reward_item_is_card(item) for item in items)

    def _reset_potion_overflow_guard(self) -> None:
        self._potion_overflow_guard_anchor = None
        self._potion_overflow_discard_used = False

    def _sync_potion_overflow_guard_state(self, state: dict[str, object]) -> None:
        if str(state.get("state_type", "unknown")) != "rewards":
            self._reset_potion_overflow_guard()
            return
        if not _has_pending_potion_reward(state):
            self._reset_potion_overflow_guard()
            return
        anchor = self._potion_overflow_guard_anchor_for_state(state)
        if self._potion_overflow_guard_anchor != anchor:
            self._potion_overflow_guard_anchor = anchor
            self._potion_overflow_discard_used = False

    def _sync_card_reward_guard_state(self, state: dict[str, object]) -> None:
        state_type = str(state.get("state_type", "unknown"))
        if state_type not in {"rewards", "card_reward"}:
            self._reset_card_reward_guard()
            return
        anchor = self._card_reward_guard_anchor_for_state(state)
        if self._card_reward_guard_anchor != anchor:
            self._card_reward_guard_anchor = anchor
            self._card_reward_guard_started_at = None
            self._card_reward_preview_seen = False
        if state_type == "card_reward":
            self._card_reward_preview_seen = True
            return
        if not self._has_pending_card_reward_claim(state):
            self._reset_card_reward_guard()
            return
        if self._card_reward_guard_started_at is None:
            self._card_reward_guard_started_at = time.monotonic()

    def _reward_claim_was_card(self, state: dict[str, object], reward_index: object) -> bool:
        if not isinstance(reward_index, int):
            return False
        rewards = state.get("rewards") if isinstance(state.get("rewards"), dict) else {}
        items = rewards.get("items") if isinstance(rewards.get("items"), list) else []
        for item in items:
            if not isinstance(item, dict):
                continue
            if int(item.get("index", -1)) != reward_index:
                continue
            return _reward_item_is_card(item)
        return False

    def _reset_campfire_upgrade_guard(self) -> None:
        self._campfire_upgrade_guard_anchor = None

    def _sync_campfire_upgrade_guard_state(self, state: dict[str, object]) -> None:
        if self._campfire_upgrade_guard_anchor is None:
            return
        if str(state.get("state_type", "unknown")) != "card_select":
            self._reset_campfire_upgrade_guard()
            return
        if self._card_reward_guard_anchor_for_state(state) != self._campfire_upgrade_guard_anchor:
            self._reset_campfire_upgrade_guard()
            return
        card_select_state = state.get("card_select")
        if not isinstance(card_select_state, dict) or not self._is_upgrade_like_card_select(card_select_state):
            self._reset_campfire_upgrade_guard()

    def _mark_campfire_upgrade_transition(
        self,
        previous_state: dict[str, object],
        next_state: dict[str, object],
        tool_name: str,
        response: dict[str, object],
    ) -> None:
        self._sync_campfire_upgrade_guard_state(previous_state)
        if (
            tool_name == "rest_choose_option"
            and response.get("status") != "error"
            and str(previous_state.get("state_type", "unknown")) == "rest_site"
            and str(next_state.get("state_type", "unknown")) == "card_select"
        ):
            card_select_state = next_state.get("card_select")
            if isinstance(card_select_state, dict) and self._is_upgrade_like_card_select(card_select_state):
                self._campfire_upgrade_guard_anchor = self._card_reward_guard_anchor_for_state(next_state)
        self._sync_campfire_upgrade_guard_state(next_state)

    def _should_filter_campfire_basic_cards(self, card_select_state: dict[str, object]) -> bool:
        return self._campfire_upgrade_guard_anchor is not None and self._is_upgrade_like_card_select(card_select_state)

    def _should_filter_enchant_basic_cards(self, card_select_state: dict[str, object]) -> bool:
        return self._is_enchant_like_card_select(card_select_state)

    def _should_skip_basic_only_enchant(self, card_select_state: dict[str, object]) -> bool:
        if not self._is_enchant_like_card_select(card_select_state):
            return False
        if not bool(card_select_state.get("can_cancel") or card_select_state.get("can_skip")):
            return False
        cards = card_select_state.get("cards")
        if not isinstance(cards, list) or not cards:
            return False
        selectable_cards = [
            card
            for card in cards
            if isinstance(card, dict) and isinstance(card.get("index"), int) and int(card.get("index", -1)) < 20
        ]
        if not selectable_cards:
            return False
        return all(_is_basic_strike_or_defend(card) for card in selectable_cards)

    def _should_filter_basic_strike_defend_cards(self, card_select_state: dict[str, object]) -> bool:
        return (
            self._is_upgrade_like_card_select(card_select_state)
            or self._should_filter_campfire_basic_cards(card_select_state)
            or self._should_filter_enchant_basic_cards(card_select_state)
        )

    def _should_prioritize_basic_strike_defend_for_removal(self, card_select_state: dict[str, object]) -> bool:
        return self._is_remove_like_card_select(card_select_state)

    def _mark_card_reward_preview_attempt(
        self,
        previous_state: dict[str, object],
        next_state: dict[str, object],
        tool_name: str,
        response: dict[str, object],
        action_kwargs: dict[str, object],
    ) -> None:
        self._sync_card_reward_guard_state(previous_state)
        if (
            tool_name == "rewards_claim"
            and response.get("status") != "error"
            and self._reward_claim_was_card(previous_state, action_kwargs.get("reward_index"))
        ):
            self._card_reward_preview_seen = True
        self._sync_card_reward_guard_state(next_state)

    def _should_block_rewards_proceed(self, state: dict[str, object]) -> bool:
        if not self.require_card_reward_preview_before_proceed:
            return False
        self._sync_card_reward_guard_state(state)
        if str(state.get("state_type", "unknown")) != "rewards":
            return False
        if not self._has_pending_card_reward_claim(state):
            return False
        if self._card_reward_preview_seen:
            return False
        if self.card_reward_preview_guard_timeout_seconds <= 0.0:
            return True
        started_at = self._card_reward_guard_started_at
        if started_at is None:
            self._card_reward_guard_started_at = time.monotonic()
            return True
        return (time.monotonic() - started_at) < self.card_reward_preview_guard_timeout_seconds

    def _record_combat_action_outcome(
        self,
        previous_state: dict[str, object],
        next_state: dict[str, object],
        tool_name: str,
        response: dict[str, object],
    ) -> None:
        if str(previous_state.get("state_type", "unknown")) not in COMBAT_STATE_TYPES:
            return
        previous_signature = self._combat_progress_signature(previous_state)
        if str(next_state.get("state_type", "unknown")) not in COMBAT_STATE_TYPES:
            self._clear_combat_guard_state(previous_signature)
            return
        next_signature = self._combat_progress_signature(next_state)
        if tool_name == "combat_end_turn" or previous_signature != next_signature:
            self._clear_combat_guard_state(previous_signature)
            return
        if tool_name not in {"combat_play_card", "use_potion", "discard_potion", "combat_select_card", "combat_confirm_selection"}:
            return
        if response.get("status") == "error" or previous_signature == next_signature:
            self._combat_stall_counts[previous_signature] = self._combat_stall_counts.get(previous_signature, 0) + 1
        else:
            self._clear_combat_guard_state(previous_signature)

    def _should_block_combat_end_turn(self, state: dict[str, object], actions: dict[int, BoundAction]) -> bool:
        if not self.block_premature_end_turn:
            return False
        if not self._is_combat_action_window_ready(state):
            return False
        if not self._combat_has_living_enemies(state):
            return False
        if not any(bound.tool_name == "combat_play_card" for bound in actions.values()):
            return False
        signature = self._combat_progress_signature(state)
        if self.end_turn_guard_timeout_seconds > 0.0:
            started_at = self._combat_end_turn_guard_started_at.get(signature)
            if started_at is None:
                self._combat_end_turn_guard_started_at[signature] = time.monotonic()
                return True
            return (time.monotonic() - started_at) < self.end_turn_guard_timeout_seconds
        if self._stall_count(state) >= self.end_turn_guard_stall_limit:
            return False
        return True

    def _redundant_defense_card_indexes(self, state: dict[str, object], hand_cards: list[dict[str, object]]) -> set[int]:
        if not self._is_combat_action_window_ready(state):
            return set()
        player = state.get("player") if isinstance(state.get("player"), dict) else {}
        current_block = float(player.get("block", 0) or 0)
        incoming_damage = _incoming_enemy_damage(state)
        required_block = incoming_damage
        if current_block < incoming_damage:
            analysis = analyze_combat_turn(state)
            if analysis.available and analysis.best_root_actions:
                required_block = min(required_block, float(analysis.min_required_block_after_best_kill))
        if required_block > 0.0 and current_block < required_block:
            return set()
        return {
            int(card["index"])
            for card in hand_cards
            if isinstance(card, dict)
            and card.get("can_play")
            and isinstance(card.get("index"), int)
            and int(card["index"]) <= 9
            and _is_pure_defense_card(card)
        }

    def _can_use_potions_in_combat(self, state: dict[str, object], player: dict[str, object]) -> bool:
        if not self._is_combat_action_window_ready(state):
            return False
        state_type = str(state.get("state_type", "unknown"))
        if state_type in {"elite", "boss"}:
            return True
        if state_type != "monster":
            return False
        hp_ratio = _safe_div(player.get("hp"), max(player.get("max_hp", 0) or 0, 1))
        return hp_ratio < 0.4

    def _can_discard_potion(self, state: dict[str, object], player: dict[str, object]) -> bool:
        self._sync_potion_overflow_guard_state(state)
        return (
            str(state.get("state_type", "unknown")) == "rewards"
            and _potion_slots_filled(player)
            and _has_pending_potion_reward(state)
            and not self._potion_overflow_discard_used
        )

    def _mark_potion_overflow_transition(
        self,
        previous_state: dict[str, object],
        next_state: dict[str, object],
        tool_name: str,
        response: dict[str, object],
    ) -> None:
        self._sync_potion_overflow_guard_state(previous_state)
        if (
            tool_name == "discard_potion"
            and response.get("status") != "error"
            and self._potion_overflow_guard_anchor is not None
        ):
            self._potion_overflow_discard_used = True
        self._sync_potion_overflow_guard_state(next_state)

    def _record_hand_select_action_outcome(
        self,
        previous_state: dict[str, object],
        next_state: dict[str, object],
        tool_name: str,
        response: dict[str, object],
        action_kwargs: dict[str, object],
    ) -> None:
        self._sync_hand_select_guard_state(previous_state)
        previous_signature = self._hand_select_progress_signature(previous_state)
        if len(previous_signature) == 1:
            self._sync_hand_select_guard_state(next_state)
            return
        next_signature = self._hand_select_progress_signature(next_state)
        if tool_name == "combat_select_card":
            card_index = action_kwargs.get("card_index")
            if isinstance(card_index, int) and previous_signature == next_signature:
                attempted = self._hand_select_attempted_indices.setdefault(previous_signature, set())
                attempted.add(card_index)
        if (
            tool_name == "combat_confirm_selection"
            or response.get("status") == "error"
            or len(next_signature) == 1
            or previous_signature != next_signature
        ):
            self._clear_hand_select_guard_state(previous_signature)
        self._sync_hand_select_guard_state(next_state)

    def _record_card_select_action_outcome(
        self,
        previous_state: dict[str, object],
        next_state: dict[str, object],
        tool_name: str,
        response: dict[str, object],
        action_kwargs: dict[str, object],
    ) -> None:
        self._sync_card_select_guard_state(previous_state)
        if str(previous_state.get("state_type", "unknown")) != "card_select":
            self._sync_card_select_guard_state(next_state)
            return
        previous_signature = self._card_select_progress_signature(previous_state)
        next_signature = self._card_select_progress_signature(next_state)
        if tool_name == "deck_select_card":
            card_index = action_kwargs.get("card_index")
            if isinstance(card_index, int) and previous_signature == next_signature:
                attempted = self._card_select_attempted_indices.setdefault(previous_signature, set())
                attempted.add(card_index)
        if (
            tool_name in {"deck_confirm_selection", "deck_cancel_selection"}
            or str(next_state.get("state_type", "unknown")) != "card_select"
            or previous_signature != next_signature
        ):
            self._clear_card_select_guard_state(previous_signature)
        self._sync_card_select_guard_state(next_state)

    def _record_bundle_select_action_outcome(
        self,
        previous_state: dict[str, object],
        next_state: dict[str, object],
        tool_name: str,
        response: dict[str, object],
        action_kwargs: dict[str, object],
    ) -> None:
        self._sync_bundle_select_guard_state(previous_state)
        previous_signature = self._bundle_select_progress_signature(previous_state)
        if len(previous_signature) == 1:
            self._sync_bundle_select_guard_state(next_state)
            return
        next_signature = self._bundle_select_progress_signature(next_state)
        if tool_name == "bundle_select":
            bundle_index = action_kwargs.get("bundle_index")
            if isinstance(bundle_index, int) and previous_signature == next_signature:
                attempted = self._bundle_select_attempted_indices.setdefault(previous_signature, set())
                attempted.add(bundle_index)
        if (
            tool_name in {"bundle_confirm_selection", "bundle_cancel_selection"}
            or response.get("status") == "error"
            or len(next_signature) == 1
            or previous_signature != next_signature
        ):
            self._clear_bundle_select_guard_state(previous_signature)
        self._sync_bundle_select_guard_state(next_state)

    def _record_event_action_outcome(
        self,
        previous_state: dict[str, object],
        next_state: dict[str, object],
        tool_name: str,
        response: dict[str, object],
        action_kwargs: dict[str, object],
    ) -> None:
        self._sync_event_repeat_guard_state(previous_state)
        previous_signature = self._event_repeat_signature(previous_state)
        if (
            previous_signature is None
            or tool_name != "event_choose_option"
            or response.get("status") == "error"
        ):
            self._sync_event_repeat_guard_state(next_state)
            return
        next_signature = self._event_repeat_signature(next_state)
        if next_signature != previous_signature:
            self._clear_event_repeat_guard_state(previous_signature)
            self._sync_event_repeat_guard_state(next_state)
            return
        previous_event = previous_state.get("event") if isinstance(previous_state.get("event"), dict) else {}
        next_event = next_state.get("event") if isinstance(next_state.get("event"), dict) else {}
        chosen_option = _event_option_from_unlocked_index(previous_event, action_kwargs.get("option_index"))
        previous_player = previous_state.get("player") if isinstance(previous_state.get("player"), dict) else {}
        next_player = next_state.get("player") if isinstance(next_state.get("player"), dict) else {}
        hp_delta = float(next_player.get("hp", 0) or 0.0) - float(previous_player.get("hp", 0) or 0.0)
        if chosen_option is not None and hp_delta < 0.0:
            repeated_option = _event_option_from_state_index(next_event, chosen_option.get("index"))
            if repeated_option is not None:
                option_key = _event_option_guard_key(repeated_option)
            else:
                option_key = None
            if option_key is not None and self._event_has_alternative_option(next_state, blocked_key=option_key):
                blocked = self._event_repeat_blocked_options.setdefault(previous_signature, set())
                blocked.add(option_key)
        self._sync_event_repeat_guard_state(next_state)

    def _combat_play_order_biases(self, state: dict[str, object]) -> dict[int, float]:
        semantic_state = ensure_semantic_state(state)
        if not semantic_state.in_combat:
            return {}
        return build_hand_order_action_biases(semantic_state, build_hand_order_profile(semantic_state))

    def _combat_tactical_action_biases(
        self,
        state: dict[str, object],
        actions: dict[int, BoundAction] | None = None,
        analysis: object | None = None,
    ) -> tuple[dict[int, float], float]:
        if analysis is None:
            analysis = analyze_combat_turn(state)
        if not analysis.available:
            return {}, 1.0
        player = state.get("player") if isinstance(state.get("player"), dict) else {}
        hand_cards = player.get("hand", []) if isinstance(player.get("hand"), list) else []
        hand_by_index = {
            int(card["index"]): card
            for card in hand_cards
            if isinstance(card, dict) and isinstance(card.get("index"), int)
        }
        current_block = max(0.0, float(player.get("block", 0) or 0.0))
        preferred_actions = analysis.best_root_actions
        attack_boost = 0.0
        defense_penalty = 0.0
        other_attack_penalty = 0.0
        hand_order_scale = 1.0
        if analysis.lethal_exists and analysis.lethal_root_actions and current_block >= float(analysis.lethal_required_block):
            preferred_actions = analysis.lethal_root_actions
            attack_boost = 1.85
            defense_penalty = 1.25
            other_attack_penalty = 0.75
            hand_order_scale = 0.25
        elif preferred_actions and current_block >= float(analysis.min_required_block_after_best_kill):
            attack_boost = 1.55
            defense_penalty = 1.10
            other_attack_penalty = 0.55
            hand_order_scale = 0.35
        elif preferred_actions:
            attack_boost = 0.45
            hand_order_scale = 0.85
        preferred_by_card: dict[int, set[str]] = {}
        preferred_by_signature: dict[tuple[object, ...], set[str]] = {}
        for action in preferred_actions:
            card_index = int(action.card_index)
            target_id = str(action.target_entity_id or "")
            preferred_by_card.setdefault(card_index, set()).add(target_id)
            preferred_card = hand_by_index.get(card_index)
            if isinstance(preferred_card, dict):
                preferred_by_signature.setdefault(_card_equivalence_signature(preferred_card), set()).add(target_id)
        if not preferred_by_card and defense_penalty <= 0.0:
            return {}, hand_order_scale
        legal_actions = actions or {}
        result: dict[int, float] = {}
        for action_id, bound in legal_actions.items():
            if bound.tool_name != "combat_play_card":
                continue
            card_index = bound.kwargs.get("card_index")
            if not isinstance(card_index, int):
                continue
            target_id = str(bound.kwargs.get("target", "") or "")
            bias = 0.0
            preferred_targets = preferred_by_card.get(card_index)
            card = hand_by_index.get(card_index)
            if preferred_targets is None and isinstance(card, dict):
                preferred_targets = preferred_by_signature.get(_card_equivalence_signature(card))
            if preferred_targets is not None and (target_id in preferred_targets or (not target_id and "" in preferred_targets)):
                bias += attack_boost
            else:
                if defense_penalty > 0.0 and isinstance(card, dict) and _is_pure_defense_card(card):
                    bias -= defense_penalty
                elif other_attack_penalty > 0.0 and isinstance(card, dict) and str(card.get("type", "")) == "Attack":
                    bias -= other_attack_penalty
            if abs(bias) >= 1e-6:
                result[action_id] = bias
        return result, hand_order_scale

    def _combat_strategy_action_biases(
        self,
        state: dict[str, object],
        actions: dict[int, BoundAction] | None = None,
    ) -> dict[int, float]:
        if str(state.get("state_type", "unknown")) not in COMBAT_STATE_TYPES:
            return {}
        profile = analyze_encounter_profile(state)
        if not profile.enemy_profiles:
            return {}
        legal_actions = actions or {}
        player = state.get("player") if isinstance(state.get("player"), dict) else {}
        hand_cards = player.get("hand", []) if isinstance(player.get("hand"), list) else []
        hand_by_index = {
            int(card["index"]): card
            for card in hand_cards
            if isinstance(card, dict) and isinstance(card.get("index"), int)
        }
        battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
        enemies = battle.get("enemies", []) if isinstance(battle.get("enemies"), list) else []
        enemy_by_id = {
            str(enemy.get("entity_id", "") or ""): enemy
            for enemy in enemies
            if isinstance(enemy, dict)
        }
        priority_by_entity = {
            str(enemy_profile.entity_id or ""): float(enemy_profile.focus_priority)
            for enemy_profile in profile.enemy_profiles
            if enemy_profile.entity_id
        }
        max_priority = max(priority_by_entity.values(), default=0.0)
        current_block = max(0.0, float(player.get("block", 0) or 0.0))
        incoming_damage = _incoming_enemy_damage(state)
        result: dict[int, float] = {}
        for action_id, bound in legal_actions.items():
            if bound.tool_name != "combat_play_card":
                continue
            card_index = bound.kwargs.get("card_index")
            if not isinstance(card_index, int):
                continue
            card = hand_by_index.get(card_index)
            if not isinstance(card, dict):
                continue
            card_type = str(card.get("type", "") or "")
            target_id = str(bound.kwargs.get("target", "") or "")
            card_strategy = card_strategy_vector_from_raw_card(card)
            bias = strategy_alignment_score(card_strategy, profile.strategy) * 0.18
            if card_type == "Attack":
                if target_id and target_id in priority_by_entity:
                    bias += 0.62 * (priority_by_entity[target_id] / max(0.8, max_priority))
                elif strategy_value(card_strategy, "aoe_clear") > 0.0:
                    bias += 0.18 * strategy_value(profile.strategy, "aoe_clear")
                enemy = enemy_by_id.get(target_id) if target_id else None
                enemy_statuses = enemy.get("status", []) if isinstance(enemy, dict) and isinstance(enemy.get("status"), list) else []
                target_has_thorns = any(isinstance(status, dict) and "THORNS" in str(status.get("id", "") or "").upper() for status in enemy_statuses)
                all_target_has_thorns = any(
                    isinstance(enemy_candidate, dict)
                    and any(
                        isinstance(status, dict) and "THORNS" in str(status.get("id", "") or "").upper()
                        for status in enemy_candidate.get("status", [])
                    )
                    for enemy_candidate in enemies
                )
                if strategy_value(profile.strategy, "respect_thorns") > 0.0 and strategy_value(card_strategy, "multi_hit") > 0.20 and (target_has_thorns or (not target_id and all_target_has_thorns)):
                    bias -= 0.36 * strategy_value(profile.strategy, "respect_thorns")
            elif _is_pure_defense_card(card):
                if incoming_damage > current_block:
                    bias += 0.40 * strategy_value(profile.strategy, "prioritize_block")
                    bias += 0.20 * strategy_value(profile.strategy, "preserve_block")
                else:
                    bias -= 0.24 * max(
                        strategy_value(profile.strategy, "burst_frontload"),
                        strategy_value(profile.strategy, "attack_windowing"),
                    )
            elif card_type == "Skill":
                avoid_skill_spam = strategy_value(profile.strategy, "avoid_skill_spam")
                if (
                    avoid_skill_spam > 0.0
                    and strategy_value(card_strategy, "prioritize_block") < 0.25
                    and strategy_value(card_strategy, "status_cleanup") < 0.25
                    and strategy_value(card_strategy, "energy_efficiency") < 0.25
                ):
                    bias -= 0.34 * avoid_skill_spam
            elif card_type == "Power":
                avoid_power_spam = strategy_value(profile.strategy, "avoid_power_spam")
                if avoid_power_spam > 0.0:
                    bias -= 0.38 * avoid_power_spam
            if abs(bias) >= 1e-6:
                result[action_id] = max(-2.0, min(2.0, bias))
        return result

    def _combat_play_order_suffix(self, card_index: int, card_biases: dict[int, float]) -> str:
        bias = float(card_biases.get(card_index, 0.0))
        if abs(bias) < 0.05:
            return ""
        return f" [ord {bias:+.2f}]"

    def _card_building_action_biases(
        self,
        state: dict[str, object],
        actions: dict[int, BoundAction] | None = None,
    ) -> dict[int, float]:
        legal_actions = actions or {}
        if not legal_actions:
            return {}
        self._sync_route_strength_state(state)
        context = self._observed_build_context or build_strategy_context(state=state)
        scores: dict[int, float] = {}

        reward_state = state.get("card_reward") if isinstance(state.get("card_reward"), dict) else {}
        reward_cards = reward_state.get("cards") if isinstance(reward_state.get("cards"), list) else []
        reward_snapshots: list[tuple[int, CardSnapshot]] = []
        for raw_card in reward_cards:
            if not isinstance(raw_card, dict) or not isinstance(raw_card.get("index"), int):
                continue
            snapshot = card_snapshot_from_card_dict(raw_card)
            if snapshot is None:
                continue
            reward_snapshots.append((int(raw_card["index"]), snapshot))
        if reward_snapshots:
            reward_evaluations = {
                index: evaluation
                for (index, _), evaluation in zip(
                    reward_snapshots,
                    evaluate_candidate_cards(tuple(snapshot for _, snapshot in reward_snapshots), context, temporary=False),
                )
            }
            best_reward_score = max((evaluation.total_score for evaluation in reward_evaluations.values()), default=0.0)
            for action_id, bound in legal_actions.items():
                if bound.tool_name == "rewards_pick_card":
                    card_index = bound.kwargs.get("card_index")
                    if isinstance(card_index, int) and card_index in reward_evaluations:
                        scores[action_id] = reward_evaluations[card_index].total_score
                elif bound.tool_name == "rewards_skip_card":
                    scores[action_id] = skip_card_reward_score(best_reward_score)

        shop_state = self._extract_shop_state(state)
        shop_items = shop_state.get("items") if isinstance(shop_state.get("items"), list) else []
        shop_snapshots: list[tuple[int, CardSnapshot, float]] = []
        for item in shop_items:
            if not isinstance(item, dict) or not isinstance(item.get("index"), int):
                continue
            snapshot = card_snapshot_from_shop_item(item)
            if snapshot is None:
                continue
            shop_snapshots.append((int(item["index"]), snapshot, float(item.get("price", 0.0) or 0.0)))
        if shop_snapshots:
            shop_evaluations = {
                index: (evaluation, price)
                for (index, _, price), evaluation in zip(
                    shop_snapshots,
                    evaluate_candidate_cards(tuple(snapshot for _, snapshot, _ in shop_snapshots), context, temporary=False),
                )
            }
            for action_id, bound in legal_actions.items():
                if bound.tool_name != "shop_purchase":
                    continue
                item_index = bound.kwargs.get("item_index")
                if not isinstance(item_index, int) or item_index not in shop_evaluations:
                    continue
                evaluation, price = shop_evaluations[item_index]
                scores[action_id] = evaluation.total_score - min(0.45, price / 180.0)
            for action_id, bound in legal_actions.items():
                if bound.tool_name == "proceed_to_map":
                    scores.setdefault(action_id, 0.0)

        bundle_state = state.get("bundle_select") if isinstance(state.get("bundle_select"), dict) else {}
        bundles = bundle_state.get("bundles") if isinstance(bundle_state.get("bundles"), list) else []
        bundle_snapshots: list[tuple[int, tuple[CardSnapshot, ...]]] = []
        for bundle in bundles:
            if not isinstance(bundle, dict) or not isinstance(bundle.get("index"), int):
                continue
            cards = bundle.get("cards") if isinstance(bundle.get("cards"), list) else []
            snapshots = tuple(
                snapshot
                for snapshot in (card_snapshot_from_card_dict(card) for card in cards if isinstance(card, dict))
                if snapshot is not None
            )
            if snapshots:
                bundle_snapshots.append((int(bundle["index"]), snapshots))
        if bundle_snapshots:
            temporary = self._combat_has_living_enemies(state)
            bundle_evaluations = {
                index: evaluation
                for (index, _), evaluation in zip(
                    bundle_snapshots,
                    evaluate_bundle_candidates(tuple(snapshots for _, snapshots in bundle_snapshots), context, temporary=temporary),
                )
            }
            for action_id, bound in legal_actions.items():
                if bound.tool_name != "bundle_select":
                    continue
                bundle_index = bound.kwargs.get("bundle_index")
                if isinstance(bundle_index, int) and bundle_index in bundle_evaluations:
                    scores[action_id] = bundle_evaluations[bundle_index].total_score
            for action_id, bound in legal_actions.items():
                if bound.tool_name == "bundle_cancel_selection":
                    scores.setdefault(action_id, 0.0)

        card_select_state = state.get("card_select") if isinstance(state.get("card_select"), dict) else {}
        plain_card_select = not any(
            (
                self._is_upgrade_like_card_select(card_select_state),
                self._is_remove_like_card_select(card_select_state),
                self._is_enchant_like_card_select(card_select_state),
                self._is_consume_like_card_select(card_select_state),
            )
        )
        if plain_card_select:
            select_cards = card_select_state.get("cards") if isinstance(card_select_state.get("cards"), list) else []
            select_snapshots: list[tuple[int, CardSnapshot]] = []
            for raw_card in select_cards:
                if not isinstance(raw_card, dict) or not isinstance(raw_card.get("index"), int):
                    continue
                snapshot = card_snapshot_from_card_dict(raw_card)
                if snapshot is None:
                    continue
                select_snapshots.append((int(raw_card["index"]), snapshot))
            if select_snapshots:
                temporary = self._combat_has_living_enemies(state)
                select_evaluations = {
                    index: evaluation
                    for (index, _), evaluation in zip(
                        select_snapshots,
                        evaluate_candidate_cards(tuple(snapshot for _, snapshot in select_snapshots), context, temporary=temporary),
                    )
                }
                for action_id, bound in legal_actions.items():
                    if bound.tool_name != "deck_select_card":
                        continue
                    card_index = bound.kwargs.get("card_index")
                    if isinstance(card_index, int) and card_index in select_evaluations:
                        scores[action_id] = select_evaluations[card_index].total_score
                for action_id, bound in legal_actions.items():
                    if bound.tool_name == "deck_cancel_selection":
                        scores.setdefault(action_id, 0.0)

        return score_to_bias_map(scores)

    def build_action_prior_biases(
        self,
        state: dict[str, object],
        actions: dict[int, BoundAction] | None = None,
    ) -> dict[int, float]:
        legal_actions = actions or self.build_legal_action_map(state)
        tactical_analysis = analyze_combat_turn(state)
        tactical_biases, hand_order_scale = self._combat_tactical_action_biases(
            state,
            legal_actions,
            tactical_analysis,
        )
        result: dict[int, float] = {}
        card_biases = self._combat_play_order_biases(state)
        player = state.get("player") if isinstance(state.get("player"), dict) else {}
        hand_cards = player.get("hand", []) if isinstance(player.get("hand"), list) else []
        hand_by_index = {
            int(card["index"]): card
            for card in hand_cards
            if isinstance(card, dict) and isinstance(card.get("index"), int)
        }
        if card_biases:
            for action_id, bound in legal_actions.items():
                if bound.tool_name != "combat_play_card":
                    continue
                card_index = bound.kwargs.get("card_index")
                if not isinstance(card_index, int):
                    continue
                bias = float(card_biases.get(card_index, 0.0)) * hand_order_scale
                card = hand_by_index.get(card_index)
                if (
                    tactical_analysis.available
                    and tactical_analysis.best_root_actions
                    and isinstance(card, dict)
                    and _is_pure_defense_card(card)
                    and hand_order_scale < 1.0
                    and bias > 0.0
                ):
                    bias *= 0.25
                if abs(bias) >= 1e-6:
                    result[action_id] = bias
        for action_id, bias in tactical_biases.items():
            result[action_id] = float(result.get(action_id, 0.0)) + float(bias)
        for action_id, bias in self._combat_strategy_action_biases(state, legal_actions).items():
            result[action_id] = float(result.get(action_id, 0.0)) + float(bias)
        for action_id, bias in self._card_building_action_biases(state, legal_actions).items():
            result[action_id] = float(result.get(action_id, 0.0)) + float(bias)
        result.update(self._map_route_action_biases(state, legal_actions))
        return result

    def build_legal_action_map(self, state: dict[str, object]) -> dict[int, BoundAction]:
        actions: dict[int, BoundAction] = {}
        state_type = str(state.get("state_type", "unknown"))
        semantic_state = self._sync_route_strength_state(state)
        self._sync_campfire_upgrade_guard_state(state)
        self._sync_hand_select_guard_state(state)
        self._sync_card_select_guard_state(state)
        self._sync_bundle_select_guard_state(state)
        self._sync_event_repeat_guard_state(state)
        combat_card_order_biases = self._combat_play_order_biases(state)
        map_route_scores: dict[int, float] = {}
        if str(getattr(semantic_state, "state_type", "unknown")) == "map":
            for route in getattr(semantic_state, "map_route_summaries", ()):
                option_index = getattr(route, "option_index", None)
                if not isinstance(option_index, int) or option_index < 0:
                    continue
                map_route_scores[option_index] = self._evaluate_map_route_value(semantic_state, route)
        player = state.get("player") if isinstance(state.get("player"), dict) else {}
        battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
        hand_cards = player.get("hand", []) if isinstance(player.get("hand"), list) else []
        oversized_hand = len(hand_cards) > 10
        if not oversized_hand:
            self._disable_play_card_above_ten = False

        if self._is_combat_action_window_ready(state):
            enemies = [
                enemy
                for enemy in battle.get("enemies", [])
                if isinstance(enemy, dict) and float(enemy.get("hp", 0) or 0) > 0
            ]
            allow_card_play = not (self._disable_play_card_above_ten and oversized_hand)
            redundant_defense_indexes = self._redundant_defense_card_indexes(state, hand_cards) if allow_card_play else set()
            if allow_card_play:
                for card in hand_cards:
                    if not isinstance(card, dict) or not card.get("can_play"):
                        continue
                    card_index = card.get("index")
                    if not isinstance(card_index, int) or card_index > 9:
                        continue
                    if card_index in redundant_defense_indexes:
                        continue
                    card_name = str(card.get("name", f"card_{card_index}"))
                    card_id = str(card.get("card_id", "") or "")
                    if _is_enemy_target(card.get("target_type")) and enemies:
                        for target_slot, enemy in enumerate(enemies[:3]):
                            index = action_index("combat_play_card", card_index, target_slot)
                            kwargs: dict[str, object] = {"card_index": card_index, "target": str(enemy.get("entity_id", ""))}
                            if card_id:
                                kwargs["card_id"] = card_id
                            actions[index] = BoundAction(
                                index,
                                "combat_play_card",
                                kwargs,
                                f"play {card_name} -> {enemy.get('name', target_slot)}{self._combat_play_order_suffix(card_index, combat_card_order_biases)}",
                            )
                    elif not _is_enemy_target(card.get("target_type")):
                        index = action_index("combat_play_card", card_index, -1)
                        kwargs = {"card_index": card_index}
                        if card_id:
                            kwargs["card_id"] = card_id
                        actions[index] = BoundAction(
                            index,
                            "combat_play_card",
                            kwargs,
                            f"play {card_name}{self._combat_play_order_suffix(card_index, combat_card_order_biases)}",
                        )
            if not self._should_block_combat_end_turn(state, actions):
                end_turn_index = action_index("combat_end_turn")
                actions[end_turn_index] = BoundAction(end_turn_index, "combat_end_turn", {}, "end turn")

        potions = player.get("potions", []) if isinstance(player.get("potions"), list) else []
        can_discard_potion = self._can_discard_potion(state, player)

        if self._can_use_potions_in_combat(state, player):
            enemies = [
                enemy
                for enemy in battle.get("enemies", [])
                if isinstance(enemy, dict) and float(enemy.get("hp", 0) or 0) > 0
            ]
            for potion in potions:
                if not isinstance(potion, dict):
                    continue
                slot = potion.get("slot")
                if not isinstance(slot, int) or slot > 2:
                    continue
                potion_name = str(potion.get("name", f"potion_{slot}"))
                if _is_enemy_target(potion.get("target_type")) and enemies:
                    for target_slot, enemy in enumerate(enemies[:3]):
                        index = action_index("use_potion", slot, target_slot)
                        actions[index] = BoundAction(
                            index,
                            "use_potion",
                            {"slot": slot, "target": str(enemy.get("entity_id", ""))},
                            f"use {potion_name} -> {enemy.get('name', target_slot)}",
                        )
                else:
                    index = action_index("use_potion", slot, -1)
                    actions[index] = BoundAction(index, "use_potion", {"slot": slot}, f"use {potion_name}")

        if can_discard_potion:
            for potion in potions:
                if not isinstance(potion, dict):
                    continue
                slot = potion.get("slot")
                if not isinstance(slot, int) or slot > 2:
                    continue
                potion_name = str(potion.get("name", f"potion_{slot}"))
                discard_index = action_index("discard_potion", slot, None)
                actions[discard_index] = BoundAction(discard_index, "discard_potion", {"slot": slot}, f"discard {potion_name}")

        self._append_hand_select_actions(actions, state, state.get("hand_select", {}))
        if isinstance(state.get("hand_select"), dict) and state["hand_select"].get("can_confirm"):
            confirm_index = action_index("combat_confirm_selection")
            actions[confirm_index] = BoundAction(confirm_index, "combat_confirm_selection", {}, "combat confirm")

        self._append_indexed_actions(actions, state.get("rewards", {}), "items", 5, "rewards_claim", "rewards_claim", "reward_index", "claim reward")
        if (
            isinstance(state.get("rewards"), dict)
            and state["rewards"].get("can_proceed")
            and not self._should_block_rewards_proceed(state)
        ):
            proceed_index = action_index("proceed_to_map")
            actions[proceed_index] = BoundAction(proceed_index, "proceed_to_map", {}, "proceed to map")

        self._append_indexed_actions(actions, state.get("card_reward", {}), "cards", 3, "rewards_pick_card", "rewards_pick_card", "card_index", "pick reward card")
        if isinstance(state.get("card_reward"), dict) and state["card_reward"].get("can_skip"):
            skip_index = action_index("rewards_skip_card")
            actions[skip_index] = BoundAction(skip_index, "rewards_skip_card", {}, "skip card reward")

        self._append_map_actions(actions, state.get("map", {}), semantic_state, map_route_scores)

        if isinstance(state.get("event"), dict):
            event_state = state["event"]
            if event_state.get("in_dialogue"):
                advance_index = action_index("event_advance_dialogue")
                event_id = _clean_label_token(event_state.get("event_id"))
                dialogue_label = f"advance dialogue {event_id}" if event_id else "advance dialogue"
                actions[advance_index] = BoundAction(advance_index, "event_advance_dialogue", {}, dialogue_label)
            else:
                unlocked_option_index = 0
                for option in event_state.get("options", []) if isinstance(event_state.get("options"), list) else []:
                    if not isinstance(option, dict) or option.get("is_locked"):
                        continue
                    if self._should_block_repeated_harmful_event_option(state, option):
                        continue
                    state_option_index = option.get("index")
                    if not isinstance(state_option_index, int) or state_option_index > 4:
                        continue
                    index = action_index("event_choose_option", state_option_index, None)
                    actions[index] = BoundAction(
                        index,
                        "event_choose_option",
                        {"option_index": unlocked_option_index},
                        _describe_event_option_item(event_state, option, state_option_index),
                    )
                    unlocked_option_index += 1

        if isinstance(state.get("rest_site"), dict):
            rest_state = state["rest_site"]
            forced_heal_option = self._forced_rest_heal_option(rest_state, player)
            forced_heal_index = forced_heal_option.get("index") if isinstance(forced_heal_option, dict) else None
            forbid_high_hp_heal = self._should_forbid_rest_heal_option(rest_state, player)
            heal_option = self._rest_heal_option(rest_state)
            heal_option_index = heal_option.get("index") if isinstance(heal_option, dict) else None
            for option in rest_state.get("options", []) if isinstance(rest_state.get("options"), list) else []:
                if not isinstance(option, dict) or not option.get("is_enabled"):
                    continue
                option_index = option.get("index")
                if not isinstance(option_index, int) or option_index > 4:
                    continue
                if isinstance(forced_heal_index, int) and option_index != forced_heal_index:
                    continue
                if forbid_high_hp_heal and option_index == heal_option_index:
                    continue
                index = action_index("rest_choose_option", option_index, None)
                actions[index] = BoundAction(
                    index,
                    "rest_choose_option",
                    {"option_index": option_index},
                    _describe_indexed_item("rest option", option, option_index),
                )
            if rest_state.get("can_proceed") and not isinstance(forced_heal_index, int):
                proceed_index = action_index("proceed_to_map")
                actions[proceed_index] = BoundAction(proceed_index, "proceed_to_map", {}, "leave rest site")

        shop_state = self._extract_shop_state(state)
        if isinstance(shop_state, dict):
            for item in shop_state.get("items", []) if isinstance(shop_state.get("items"), list) else []:
                if not isinstance(item, dict) or not item.get("can_afford") or not item.get("is_stocked", True):
                    continue
                item_index = item.get("index")
                if not isinstance(item_index, int) or item_index > 11:
                    continue
                index = action_index("shop_purchase", item_index, None)
                actions[index] = BoundAction(
                    index,
                    "shop_purchase",
                    {"item_index": item_index},
                    _describe_indexed_item("shop purchase", item, item_index),
                )
            if state_type in {"shop", "fake_merchant"} and not shop_state.get("error"):
                proceed_index = action_index("proceed_to_map")
                actions[proceed_index] = BoundAction(proceed_index, "proceed_to_map", {}, "leave shop")

        self._append_indexed_actions(actions, state.get("treasure", {}), "relics", 3, "treasure_claim_relic", "treasure_claim_relic", "relic_index", "treasure relic")
        if isinstance(state.get("treasure"), dict) and state["treasure"].get("can_proceed"):
            proceed_index = action_index("proceed_to_map")
            actions[proceed_index] = BoundAction(proceed_index, "proceed_to_map", {}, "leave treasure")

        card_select_state = state.get("card_select", {}) if isinstance(state.get("card_select"), dict) else {}
        card_select_preview = bool(card_select_state.get("preview_showing")) if isinstance(card_select_state, dict) else False
        if not card_select_preview and not bool(card_select_state.get("can_confirm")):
            self._append_card_select_actions(actions, state, card_select_state)
        if isinstance(card_select_state, dict) and card_select_state.get("can_confirm"):
            confirm_index = action_index("deck_confirm_selection")
            actions[confirm_index] = BoundAction(confirm_index, "deck_confirm_selection", {}, "deck confirm")
        if isinstance(card_select_state, dict) and (card_select_state.get("can_cancel") or card_select_state.get("can_skip")):
            cancel_index = action_index("deck_cancel_selection")
            actions[cancel_index] = BoundAction(cancel_index, "deck_cancel_selection", {}, "deck cancel")

        bundle_select_state = state.get("bundle_select", {}) if isinstance(state.get("bundle_select"), dict) else {}
        self._append_bundle_select_actions(actions, state, bundle_select_state)
        if isinstance(bundle_select_state, dict) and bundle_select_state.get("can_confirm"):
            confirm_index = action_index("bundle_confirm_selection")
            actions[confirm_index] = BoundAction(confirm_index, "bundle_confirm_selection", {}, "bundle confirm")
        if isinstance(bundle_select_state, dict) and bundle_select_state.get("can_cancel"):
            cancel_index = action_index("bundle_cancel_selection")
            actions[cancel_index] = BoundAction(cancel_index, "bundle_cancel_selection", {}, "bundle cancel")

        self._append_indexed_actions(actions, state.get("relic_select", {}), "relics", 3, "relic_select", "relic_select", "relic_index", "relic select")
        if isinstance(state.get("relic_select"), dict) and state["relic_select"].get("can_skip"):
            skip_index = action_index("relic_skip")
            actions[skip_index] = BoundAction(skip_index, "relic_skip", {}, "skip relic")

        if isinstance(state.get("crystal_sphere"), dict):
            crystal = state["crystal_sphere"]
            if crystal.get("can_use_big_tool"):
                index = action_index("crystal_sphere_set_tool", 0, None)
                actions[index] = BoundAction(index, "crystal_sphere_set_tool", {"tool": "big"}, "crystal big tool")
            if crystal.get("can_use_small_tool"):
                index = action_index("crystal_sphere_set_tool", 1, None)
                actions[index] = BoundAction(index, "crystal_sphere_set_tool", {"tool": "small"}, "crystal small tool")
            for cell_index, cell in enumerate(crystal.get("clickable_cells", [])[:8] if isinstance(crystal.get("clickable_cells"), list) else []):
                if not isinstance(cell, dict):
                    continue
                index = action_index("crystal_sphere_click_cell", cell_index, None)
                actions[index] = BoundAction(index, "crystal_sphere_click_cell", {"x": int(cell.get("x", 0)), "y": int(cell.get("y", 0))}, f"crystal cell {cell_index}")
            if crystal.get("can_proceed"):
                proceed_index = action_index("crystal_sphere_proceed")
                actions[proceed_index] = BoundAction(proceed_index, "crystal_sphere_proceed", {}, "crystal proceed")
        return actions

    def step(self, action_id: int, state: dict[str, object]) -> tuple[dict[str, object], float, bool, dict[str, object]]:
        legal_actions = self.build_legal_action_map(state)
        if action_id not in legal_actions:
            raise ValueError(f"Action {action_id} is not legal in state {state.get('state_type')}")
        bound = legal_actions[action_id]
        try:
            response = self.bridge.call_tool(bound.tool_name, **bound.kwargs)
            next_state = self._poll_state(bound.tool_name, state)
        except STS2BridgeError as exc:
            error_text = str(exc)
            compatibility_workaround = None
            hand_cards = state.get("player", {}).get("hand", []) if isinstance(state.get("player"), dict) else []
            if bound.tool_name == "combat_play_card" and "Hand size" in error_text and isinstance(hand_cards, list) and len(hand_cards) > 10:
                self._disable_play_card_above_ten = True
                compatibility_workaround = "disable_play_card_above_ten"
            response = {"status": "error", "error": error_text}
            if compatibility_workaround is not None:
                response["compatibility_workaround"] = compatibility_workaround
            next_state = self.fetch_state()
        self._mark_card_reward_preview_attempt(state, next_state, bound.tool_name, response, bound.kwargs)
        self._mark_campfire_upgrade_transition(state, next_state, bound.tool_name, response)
        self._mark_potion_overflow_transition(state, next_state, bound.tool_name, response)
        self._record_hand_select_action_outcome(state, next_state, bound.tool_name, response, bound.kwargs)
        self._record_card_select_action_outcome(state, next_state, bound.tool_name, response, bound.kwargs)
        self._record_bundle_select_action_outcome(state, next_state, bound.tool_name, response, bound.kwargs)
        self._record_event_action_outcome(state, next_state, bound.tool_name, response, bound.kwargs)
        self._record_combat_action_outcome(state, next_state, bound.tool_name, response)
        boundaries = detect_transition_boundaries(state, next_state)
        reward, reward_breakdown = self._compute_reward(
            state,
            next_state,
            response,
            bound.tool_name,
            bound.kwargs,
            boundaries,
        )
        self.semantic_history.update_from_transition(
            state,
            next_state,
            bound.tool_name,
            response,
            boundaries,
            action_kwargs=bound.kwargs,
        )
        done = bool(boundaries["run_end"])
        return next_state, reward, done, {
            "tool_name": bound.tool_name,
            "description": bound.description,
            "response": response,
            "reward_breakdown": reward_breakdown,
            **boundaries,
        }

    def _append_indexed_actions(
        self,
        actions: dict[int, BoundAction],
        container: object,
        list_key: str,
        max_index: int,
        lookup_kind: str,
        tool_name: str,
        argument_name: str,
        label_prefix: str,
    ) -> None:
        if not isinstance(container, dict):
            return
        items = container.get(list_key)
        if not isinstance(items, list):
            return
        for item in items:
            if not isinstance(item, dict):
                continue
            item_index = item.get("index")
            if not isinstance(item_index, int) or item_index >= max_index:
                continue
            index = action_index(lookup_kind, item_index, None)
            actions[index] = BoundAction(
                index,
                tool_name,
                {argument_name: item_index},
                _describe_indexed_item(label_prefix, item, item_index),
            )

    def _append_map_actions(
        self,
        actions: dict[int, BoundAction],
        map_state: object,
        semantic_state: object,
        route_scores: dict[int, float] | None = None,
    ) -> None:
        if not isinstance(map_state, dict):
            return
        items = map_state.get("next_options")
        if not isinstance(items, list):
            return
        route_scores = route_scores or {}
        route_by_index = {
            int(getattr(route, "option_index", -1)): route
            for route in getattr(semantic_state, "map_route_summaries", ())
            if isinstance(getattr(route, "option_index", None), int)
        }
        for item in items:
            if not isinstance(item, dict):
                continue
            item_index = item.get("index")
            if not isinstance(item_index, int) or item_index >= 6:
                continue
            index = action_index("map_choose_node", item_index, None)
            description = _describe_map_route_item(item, item_index, route_by_index.get(item_index))
            if item_index in route_scores:
                description = (
                    f"{description} fit={route_scores[item_index]:+.2f} "
                    f"ready={self._observed_run_readiness:.2f} "
                    f"deck={self._observed_deck_strength:.2f} "
                    f"relic={self._observed_relic_strength:.2f} "
                    f"res={self._observed_resource_strength:.2f}"
                )
            actions[index] = BoundAction(
                index,
                "map_choose_node",
                {"node_index": item_index},
                description,
            )

    def _hand_select_allows_any_number(self, hand_select_state: dict[str, object]) -> bool:
        text = self._card_select_text_blob(hand_select_state)
        return _contains_any_number_selection_hint(text)

    def _required_hand_select_count(self, hand_select_state: dict[str, object]) -> int | None:
        if self._hand_select_allows_any_number(hand_select_state):
            return None
        mode = str(hand_select_state.get("mode", "") or "").strip().lower()
        text = self._card_select_text_blob(hand_select_state)
        if mode == "upgrade_select":
            return 1
        return _extract_selection_count_hint(text)

    def _should_lock_hand_select_to_confirm(
        self,
        state: dict[str, object],
        hand_select_state: dict[str, object],
    ) -> bool:
        if not bool(hand_select_state.get("can_confirm")):
            return False
        required_count = self._required_hand_select_count(hand_select_state)
        if required_count is not None:
            if str(hand_select_state.get("mode", "") or "").strip().lower() == "upgrade_select":
                return True
            selected_cards = hand_select_state.get("selected_cards")
            selected_count = len(selected_cards) if isinstance(selected_cards, list) else 0
            return selected_count >= required_count
        selected_cards = hand_select_state.get("selected_cards")
        selected_count = len(selected_cards) if isinstance(selected_cards, list) else 0
        return (
            str(hand_select_state.get("mode", "") or "").strip().lower() == "simple_select"
            and selected_count > 0
            and self._hand_select_started_without_confirm(state)
        )

    def _append_hand_select_actions(
        self,
        actions: dict[int, BoundAction],
        state: dict[str, object],
        hand_select_state: object,
    ) -> None:
        if not isinstance(hand_select_state, dict):
            return
        if self._should_lock_hand_select_to_confirm(state, hand_select_state):
            return
        label_prefix = self._hand_select_label_prefix(hand_select_state)
        items = hand_select_state.get("cards")
        if not isinstance(items, list):
            return
        attempted_indices = self._attempted_hand_select_indices(state)
        emitted_equivalence_signatures: set[tuple[object, ...]] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            item_index = item.get("index")
            if not isinstance(item_index, int) or item_index >= 10 or item_index in attempted_indices:
                continue
            equivalence_signature = _card_equivalence_signature(item)
            if equivalence_signature in emitted_equivalence_signatures:
                continue
            emitted_equivalence_signatures.add(equivalence_signature)
            index = action_index("combat_select_card", item_index, None)
            actions[index] = BoundAction(
                index,
                "combat_select_card",
                {"card_index": item_index},
                _describe_indexed_item(label_prefix, item, item_index),
            )

    def _append_bundle_select_actions(
        self,
        actions: dict[int, BoundAction],
        state: dict[str, object],
        bundle_select_state: object,
    ) -> None:
        if not isinstance(bundle_select_state, dict):
            return
        if bool(bundle_select_state.get("preview_showing")) or bool(bundle_select_state.get("can_confirm")):
            return
        bundles = bundle_select_state.get("bundles")
        if not isinstance(bundles, list):
            return
        attempted_indices = self._attempted_bundle_select_indices(state)
        for bundle in bundles:
            if not isinstance(bundle, dict):
                continue
            bundle_index = bundle.get("index")
            if not isinstance(bundle_index, int) or bundle_index >= 3 or bundle_index in attempted_indices:
                continue
            index = action_index("bundle_select", bundle_index, None)
            actions[index] = BoundAction(
                index,
                "bundle_select",
                {"bundle_index": bundle_index},
                _describe_indexed_item("bundle select", bundle, bundle_index),
            )

    def _hand_select_label_prefix(self, hand_select_state: dict[str, object]) -> str:
        text = self._card_select_text_blob(hand_select_state)
        if any(token in text for token in ("exhaust", "consume", "burn", "sacrifice", "消耗", "耗尽", "枯竭", "焚烧", "献祭")):
            return "combat consume"
        if any(token in text for token in ("enchant", "imbue", "附魔", "灌注")):
            return "combat enchant"
        if any(token in text for token in ("remove", "purge", "delete", "forget", "移除", "删除", "遗忘", "删牌", "变换", "变化", "替换", "变形")):
            return "combat remove"
        if any(token in text for token in ("upgrade", "强化", "升级")):
            return "combat upgrade"
        return "combat select"

    def _append_card_select_actions(
        self,
        actions: dict[int, BoundAction],
        state: dict[str, object],
        card_select_state: object,
    ) -> None:
        if not isinstance(card_select_state, dict):
            return
        items = card_select_state.get("cards")
        if not isinstance(items, list):
            return
        if self._should_skip_basic_only_enchant(card_select_state):
            return
        allowed_indices = self._filtered_card_select_indices(card_select_state)
        attempted_indices = self._attempted_card_select_indices(state)
        label_prefix = self._card_select_label_prefix(card_select_state)
        emitted_equivalence_signatures: set[tuple[object, ...]] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            item_index = item.get("index")
            if (
                not isinstance(item_index, int)
                or item_index >= 20
                or item_index not in allowed_indices
                or item_index in attempted_indices
            ):
                continue
            equivalence_signature = _card_equivalence_signature(item)
            if equivalence_signature in emitted_equivalence_signatures:
                continue
            emitted_equivalence_signatures.add(equivalence_signature)
            index = action_index("deck_select_card", item_index, None)
            actions[index] = BoundAction(
                index,
                "deck_select_card",
                {"card_index": item_index},
                _describe_indexed_item(label_prefix, item, item_index),
            )

    def _card_select_label_prefix(self, card_select_state: dict[str, object]) -> str:
        if self._is_consume_like_card_select(card_select_state):
            return "deck consume"
        if self._is_enchant_like_card_select(card_select_state):
            return "deck enchant"
        if self._is_remove_like_card_select(card_select_state):
            return "deck remove"
        if self._is_upgrade_like_card_select(card_select_state):
            return "deck upgrade"
        return "deck select"

    def _filtered_card_select_indices(self, card_select_state: dict[str, object]) -> set[int]:
        cards = card_select_state.get("cards")
        if not isinstance(cards, list):
            return set()
        available_indices = {
            int(card["index"])
            for card in cards
            if isinstance(card, dict) and isinstance(card.get("index"), int) and int(card["index"]) < 20
        }
        if not available_indices:
            return set()
        if self._should_prioritize_basic_strike_defend_for_removal(card_select_state):
            basic_indices = {
                int(card["index"])
                for card in cards
                if (
                    isinstance(card, dict)
                    and isinstance(card.get("index"), int)
                    and int(card["index"]) in available_indices
                    and _is_basic_strike_or_defend(card)
                )
            }
            if basic_indices:
                available_indices = basic_indices
        if self._should_filter_basic_strike_defend_cards(card_select_state):
            non_basic_indices = {
                int(card["index"])
                for card in cards
                if (
                    isinstance(card, dict)
                    and isinstance(card.get("index"), int)
                    and int(card["index"]) in available_indices
                    and not _is_basic_strike_or_defend(card)
                )
            }
            if non_basic_indices:
                available_indices = non_basic_indices
        if not self._is_upgrade_like_card_select(card_select_state) or self.card_select_candidate_limit <= 0:
            return available_indices
        candidate_count = max(1, self._required_card_select_count(card_select_state))
        shortlist_size = max(candidate_count, self.card_select_candidate_limit)
        ranked_cards = sorted(
            (
                card
                for card in cards
                if isinstance(card, dict) and isinstance(card.get("index"), int) and int(card["index"]) in available_indices
            ),
            key=self._score_card_select_candidate,
            reverse=True,
        )
        return {int(card["index"]) for card in ranked_cards[:shortlist_size]}

    def _card_select_text_blob(self, card_select_state: dict[str, object]) -> str:
        return " ".join(
            str(card_select_state.get(key, "") or "").strip().lower()
            for key in ("screen_type", "prompt", "title", "header", "body", "message", "reason")
        )

    def _is_upgrade_like_card_select(self, card_select_state: dict[str, object]) -> bool:
        text = self._card_select_text_blob(card_select_state)
        return any(token in text for token in ("upgrade", "enchant", "imbue", "升级", "强化", "附魔", "灌注"))

    def _is_consume_like_card_select(self, card_select_state: dict[str, object]) -> bool:
        text = self._card_select_text_blob(card_select_state)
        return any(token in text for token in ("exhaust", "consume", "burn", "sacrifice", "消耗", "耗尽", "枯竭", "焚烧", "献祭"))

    def _is_enchant_like_card_select(self, card_select_state: dict[str, object]) -> bool:
        text = self._card_select_text_blob(card_select_state)
        return any(token in text for token in ("enchant", "imbue", "附魔", "灌注"))

    def _is_remove_like_card_select(self, card_select_state: dict[str, object]) -> bool:
        text = self._card_select_text_blob(card_select_state)
        remove_tokens = ("remove", "purge", "delete", "forget", "移除", "删除", "遗忘", "删牌")
        return any(token in text for token in remove_tokens)

    def _required_card_select_count(self, card_select_state: dict[str, object]) -> int:
        prompt = str(card_select_state.get("prompt", ""))
        value = _int_prefix(prompt)
        return value if value is not None and value > 0 else 1

    def _score_card_select_candidate(self, card: dict[str, object]) -> tuple[float, float, float, int]:
        card_type = str(card.get("type", ""))
        rarity = str(card.get("rarity", ""))
        is_upgraded = bool(card.get("is_upgraded"))
        raw_cost = card.get("cost")
        cost_value = 0.0
        if raw_cost == "X":
            cost_value = 1.5
        else:
            try:
                cost_value = float(raw_cost)
            except (TypeError, ValueError):
                cost_value = 0.0
        type_score = {
            "Power": 2.4,
            "Attack": 2.0,
            "Skill": 1.8,
            "Status": -3.0,
            "Curse": -4.0,
        }.get(card_type, 0.0)
        rarity_score = {
            "Rare": 1.3,
            "Uncommon": 0.8,
            "Common": 0.4,
            "Starter": 0.1,
            "Basic": 0.1,
        }.get(rarity, 0.0)
        upgrade_score = -5.0 if is_upgraded else 3.5
        index = int(card.get("index", 0) or 0)
        return (upgrade_score + type_score + rarity_score + min(2.0, cost_value * 0.35), rarity_score, cost_value, -index)

    def _extract_shop_state(self, state: dict[str, object]) -> dict[str, object]:
        state_type = str(state.get("state_type", "unknown"))
        if state_type == "fake_merchant":
            fake_merchant = state.get("fake_merchant")
            if isinstance(fake_merchant, dict):
                nested_shop = fake_merchant.get("shop")
                if isinstance(nested_shop, dict):
                    return nested_shop
            return {}
        shop_state = state.get("shop")
        return shop_state if isinstance(shop_state, dict) else {}

    def _poll_state(self, tool_name: str, previous_state: dict[str, object]) -> dict[str, object]:
        state = self.fetch_state()
        used_post_selection_hand_refresh = False
        pending_ready_combat_signature: tuple[object, ...] | None = None
        for _ in range(self.max_poll_attempts):
            if not self._needs_extra_poll(tool_name, previous_state, state):
                if (
                    not used_post_selection_hand_refresh
                    and self._should_force_post_selection_hand_refresh(tool_name, previous_state, state)
                ):
                    # Some in-combat selection confirms (notably card-generating potions)
                    # can return to combat one snapshot before the chosen card is visible in hand.
                    used_post_selection_hand_refresh = True
                    time.sleep(self.poll_interval)
                    state = self.fetch_state()
                    continue
                if self._should_force_post_combat_ready_refresh(tool_name, previous_state, state):
                    ready_signature = self._combat_progress_signature(state)
                    if pending_ready_combat_signature != ready_signature:
                        # Combat snapshots can briefly claim the action window is ready while the
                        # UI still has a card in-flight or an end-turn transition finishing.
                        pending_ready_combat_signature = ready_signature
                        time.sleep(self.poll_interval)
                        state = self.fetch_state()
                        continue
                return state
            pending_ready_combat_signature = None
            time.sleep(self.poll_interval)
            state = self.fetch_state()
        return state

    def _should_force_post_selection_hand_refresh(
        self,
        tool_name: str,
        previous_state: dict[str, object],
        state: dict[str, object],
    ) -> bool:
        if tool_name not in {"combat_confirm_selection", "bundle_confirm_selection", "deck_confirm_selection"}:
            return False
        if str(previous_state.get("state_type", "unknown")) not in {"hand_select", "bundle_select", "card_select"}:
            return False
        return str(state.get("state_type", "unknown")) in COMBAT_STATE_TYPES

    def _should_force_post_combat_ready_refresh(
        self,
        tool_name: str,
        previous_state: dict[str, object],
        state: dict[str, object],
    ) -> bool:
        if tool_name not in {"combat_play_card", "combat_end_turn", "use_potion", "discard_potion"}:
            return False
        if str(previous_state.get("state_type", "unknown")) not in COMBAT_STATE_TYPES | {"hand_select"}:
            return False
        if str(state.get("state_type", "unknown")) not in COMBAT_STATE_TYPES:
            return False
        return self._is_combat_action_window_ready(state)

    def _needs_extra_poll(self, tool_name: str, previous_state: dict[str, object], state: dict[str, object]) -> bool:
        state_type = str(state.get("state_type", "unknown"))
        previous_signature = self._combat_progress_signature(previous_state)
        current_signature = self._combat_progress_signature(state)
        if tool_name in {"combat_play_card", "combat_end_turn", "use_potion", "discard_potion", "combat_select_card", "combat_confirm_selection"}:
            if current_signature == previous_signature:
                return True
        if tool_name == "combat_end_turn" and state_type in COMBAT_STATE_TYPES:
            return not self._is_combat_action_window_ready(state)
        if tool_name in {"combat_play_card", "use_potion", "discard_potion", "combat_select_card", "combat_confirm_selection"} and state_type in COMBAT_STATE_TYPES:
            battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
            hand_mode = str(battle.get("hand_mode", "play") or "play").lower()
            return bool(battle.get("player_actions_disabled")) or bool(battle.get("hand_in_card_play")) or hand_mode != "play"
        if tool_name in {"combat_select_card", "combat_confirm_selection"}:
            previous_hand_select_signature = self._hand_select_progress_signature(previous_state)
            current_hand_select_signature = self._hand_select_progress_signature(state)
            if current_hand_select_signature == previous_hand_select_signature:
                return True
        if tool_name in {"deck_select_card", "deck_confirm_selection", "deck_cancel_selection"}:
            previous_card_select_signature = self._card_select_progress_signature(previous_state)
            current_card_select_signature = self._card_select_progress_signature(state)
            if current_card_select_signature == previous_card_select_signature:
                return True
        if tool_name in {"bundle_select", "bundle_confirm_selection", "bundle_cancel_selection"}:
            previous_bundle_signature = self._bundle_select_progress_signature(previous_state)
            current_bundle_signature = self._bundle_select_progress_signature(state)
            if current_bundle_signature == previous_bundle_signature:
                return True
        if state_type == "treasure" and isinstance(state.get("treasure"), dict):
            treasure = state["treasure"]
            return bool(treasure.get("message")) and not treasure.get("relics")
        if state_type in {"shop", "fake_merchant"}:
            return bool(self._extract_shop_state(state).get("error"))
        return False

    def _floor_advance_reward(self, previous_floor: int, next_floor: int) -> float:
        if next_floor <= previous_floor:
            return 0.0
        total = 0.0
        for floor in range(previous_floor + 1, next_floor + 1):
            total += self.reward_weights.floor_advance * _floor_reward_multiplier(floor)
        return total

    def _rest_heal_option(self, rest_state: dict[str, object]) -> dict[str, object] | None:
        options = rest_state.get("options")
        if not isinstance(options, list):
            return None
        for option in options:
            if not isinstance(option, dict):
                continue
            option_id = str(option.get("id", "") or "").strip().upper()
            option_name = str(option.get("name", "") or "").strip().lower()
            if option_id == "HEAL" or option_name in {"rest", "heal", "休息", "回复"}:
                return option
        return None

    def _rest_heal_amount(self, rest_option: dict[str, object], player: dict[str, object]) -> float:
        integers = _extract_int_values(rest_option.get("description"))
        if integers:
            return max(0.0, float(integers[-1]))
        max_hp = float(player.get("max_hp", 0) or 0.0)
        if max_hp <= 0.0:
            return 0.0
        return max(0.0, round(max_hp * 0.3))

    def _forced_rest_heal_option(
        self,
        rest_state: dict[str, object],
        player: dict[str, object],
        threshold: float = REST_SITE_FORCE_HEAL_THRESHOLD,
    ) -> dict[str, object] | None:
        if _player_hp_ratio_from_dict(player) >= threshold:
            return None
        heal_option = self._rest_heal_option(rest_state)
        if heal_option is None or not heal_option.get("is_enabled"):
            return None
        return heal_option

    def _should_forbid_rest_heal_option(
        self,
        rest_state: dict[str, object],
        player: dict[str, object],
        threshold: float = REST_SITE_AVOID_HEAL_THRESHOLD,
    ) -> bool:
        if _player_hp_ratio_from_dict(player) <= threshold:
            return False
        heal_option = self._rest_heal_option(rest_state)
        if heal_option is None or not heal_option.get("is_enabled"):
            return False
        heal_index = heal_option.get("index")
        options = rest_state.get("options")
        if not isinstance(options, list):
            return False
        for option in options:
            if not isinstance(option, dict) or not option.get("is_enabled"):
                continue
            if option.get("index") == heal_index or option.get("is_proceed"):
                continue
            return True
        return False

    def _rest_site_heal_shaping(
        self,
        previous_state: dict[str, object],
        next_state: dict[str, object],
        tool_name: str,
        action_kwargs: dict[str, object],
    ) -> dict[str, float]:
        if tool_name != "rest_choose_option" or str(previous_state.get("state_type", "unknown")) != "rest_site":
            return {}
        rest_state = previous_state.get("rest_site") if isinstance(previous_state.get("rest_site"), dict) else {}
        heal_option = self._rest_heal_option(rest_state)
        if heal_option is None:
            return {}
        previous_player = previous_state.get("player") if isinstance(previous_state.get("player"), dict) else {}
        next_player = next_state.get("player") if isinstance(next_state.get("player"), dict) else {}
        current_hp_ratio = _player_hp_ratio_from_dict(previous_player)
        chosen_option = _rest_option_from_state_index(rest_state, action_kwargs.get("option_index"))
        chosen_id = str(chosen_option.get("id", "") or "").strip().upper() if isinstance(chosen_option, dict) else ""
        if chosen_id == "HEAL" and current_hp_ratio > REST_SITE_AVOID_HEAL_THRESHOLD:
            penalty_scale = max(
                0.0,
                min(
                    1.0,
                    (current_hp_ratio - REST_SITE_AVOID_HEAL_THRESHOLD)
                    / max(1e-9, 1.0 - REST_SITE_AVOID_HEAL_THRESHOLD),
                ),
            )
            penalty = self.reward_weights.rest_high_hp_heal_penalty * (0.6 + (0.4 * penalty_scale))
            return {"rest_high_hp_heal": -penalty} if penalty != 0.0 else {}
        max_hp = max(float(previous_player.get("max_hp", 0) or next_player.get("max_hp", 0) or 0.0), 1.0)
        current_hp = float(previous_player.get("hp", 0) or 0.0)
        missing_hp = max(0.0, max_hp - current_hp)
        if missing_hp <= 0.0:
            return {}
        urgency = _low_hp_urgency(_player_hp_ratio_from_dict(previous_player))
        if urgency <= 0.0:
            return {}
        heal_amount = min(missing_hp, self._rest_heal_amount(heal_option, previous_player))
        heal_ratio = min(1.0, heal_amount / max_hp)
        if heal_ratio <= 0.0:
            return {}
        if chosen_id == "HEAL":
            bonus = self.reward_weights.rest_low_hp_heal_bonus * heal_ratio * urgency
            return {"rest_low_hp_heal": bonus} if bonus != 0.0 else {}
        penalty = self.reward_weights.rest_missed_heal * heal_ratio * urgency
        return {"rest_missed_heal": -penalty} if penalty != 0.0 else {}

    def _map_route_choice_shaping(
        self,
        previous_state: dict[str, object],
        tool_name: str,
        action_kwargs: dict[str, object],
    ) -> float:
        if tool_name != "map_choose_node" or self.reward_weights.map_route_choice <= 0.0:
            return 0.0
        chosen_index = action_kwargs.get("node_index")
        if not isinstance(chosen_index, int):
            return 0.0
        _, route_scores = self._map_route_scores(previous_state)
        _, route_evaluations = self._map_route_evaluations(previous_state)
        if len(route_scores) <= 1:
            return 0.0
        chosen_score = route_scores.get(chosen_index)
        if chosen_score is None:
            return 0.0
        values = tuple(route_scores.values())
        best = max(values)
        worst = min(values)
        mean = sum(values) / max(1, len(values))
        spread = max(0.40, best - worst)
        normalized = (chosen_score - mean) / spread
        regret = max(0.0, (best - chosen_score) / spread)
        shaping = self.reward_weights.map_route_choice * (1.40 * normalized)
        any_path_feasible = any(float(getattr(item, "path_feasible", 0.0) or 0.0) >= 0.5 for item in route_evaluations.values())
        chosen_eval = route_evaluations.get(chosen_index)
        if any_path_feasible and chosen_eval is not None:
            if float(getattr(chosen_eval, "path_feasible", 0.0) or 0.0) >= 0.5:
                shaping += self.reward_weights.map_route_choice * 0.45
            else:
                shaping -= self.reward_weights.map_route_choice * 0.80
        if regret > 0.55:
            shaping -= self.reward_weights.map_route_choice * min(0.80, (regret - 0.55) * 0.70)
        limit = self.reward_weights.map_route_choice * 1.75
        return max(-limit, min(limit, shaping))

    def _non_combat_hp_loss_shaping(
        self,
        previous_state: dict[str, object],
        next_state: dict[str, object],
        hp_delta: float,
        player_max_hp: float,
    ) -> float:
        if hp_delta >= 0.0:
            return 0.0
        previous_state_type = str(previous_state.get("state_type", "unknown"))
        if previous_state_type in COMBAT_STATE_TYPES | {"hand_select"}:
            return 0.0
        next_player = next_state.get("player") if isinstance(next_state.get("player"), dict) else {}
        urgency = 1.0 + _low_hp_urgency(_player_hp_ratio_from_dict(next_player))
        return self.reward_weights.non_combat_hp_loss * (hp_delta / max(player_max_hp, 1.0)) * urgency

    def _compute_reward(
        self,
        previous_state: dict[str, object],
        next_state: dict[str, object],
        response: dict[str, object],
        tool_name: str,
        action_kwargs: dict[str, object] | None = None,
        boundaries: dict[str, object] | None = None,
    ) -> tuple[float, dict[str, float]]:
        if boundaries is None:
            boundaries = detect_transition_boundaries(previous_state, next_state)
        action_kwargs = action_kwargs or {}
        breakdown: dict[str, float] = {}
        reward = 0.0

        if response.get("status") == "error":
            breakdown["error_penalty"] = -abs(self.reward_weights.error_penalty)
            reward += breakdown["error_penalty"]

        previous_run = previous_state.get("run") if isinstance(previous_state.get("run"), dict) else {}
        next_run = next_state.get("run") if isinstance(next_state.get("run"), dict) else {}
        floor_delta = max(0, int(next_run.get("floor", 0) or 0) - int(previous_run.get("floor", 0) or 0))
        act_delta = max(0, int(next_run.get("act", 0) or 0) - int(previous_run.get("act", 0) or 0))
        if floor_delta > 0:
            breakdown["floor_advance"] = self._floor_advance_reward(
                int(previous_run.get("floor", 0) or 0),
                int(next_run.get("floor", 0) or 0),
            )
            reward += breakdown["floor_advance"]
        if act_delta > 0:
            breakdown["act_advance"] = self.reward_weights.act_advance * act_delta
            reward += breakdown["act_advance"]

        previous_player = previous_state.get("player") if isinstance(previous_state.get("player"), dict) else {}
        next_player = next_state.get("player") if isinstance(next_state.get("player"), dict) else {}
        hp_delta = float(next_player.get("hp", 0) or 0) - float(previous_player.get("hp", 0) or 0)
        gold_delta = float(next_player.get("gold", 0) or 0) - float(previous_player.get("gold", 0) or 0)
        player_max_hp = max(float(previous_player.get("max_hp", 0) or next_player.get("max_hp", 0) or 0), 1.0)
        if hp_delta != 0.0:
            breakdown["hp_delta"] = self.reward_weights.hp_delta * (hp_delta / player_max_hp)
            reward += breakdown["hp_delta"]
        non_combat_hp_loss = self._non_combat_hp_loss_shaping(previous_state, next_state, hp_delta, player_max_hp)
        if non_combat_hp_loss != 0.0:
            breakdown["non_combat_hp_loss"] = non_combat_hp_loss
            reward += non_combat_hp_loss
        if gold_delta != 0.0:
            gold_scale = max(self.reward_weights.gold_delta_scale, 1.0)
            breakdown["gold_delta"] = self.reward_weights.gold_delta * (gold_delta / gold_scale)
            reward += breakdown["gold_delta"]
        rest_heal_shaping = self._rest_site_heal_shaping(previous_state, next_state, tool_name, action_kwargs)
        for label, value in rest_heal_shaping.items():
            breakdown[label] = value
            reward += value
        map_route_choice = self._map_route_choice_shaping(previous_state, tool_name, action_kwargs)
        if map_route_choice != 0.0:
            breakdown["map_route_choice"] = map_route_choice
            reward += map_route_choice

        player_survived = self._combat_player_survived(previous_state, next_state)
        previous_enemy_hp = float(boundaries["previous_enemy_hp"])
        next_enemy_hp = float(boundaries["next_enemy_hp"]) if player_survived or not boundaries["combat_end"] else previous_enemy_hp
        enemy_hp_delta = previous_enemy_hp - next_enemy_hp
        previous_enemy_max_hp = max(_total_enemy_max_hp(previous_state), previous_enemy_hp, 1.0)
        if enemy_hp_delta != 0.0:
            breakdown["enemy_hp_delta"] = self.reward_weights.enemy_hp_delta * (enemy_hp_delta / previous_enemy_max_hp)
            reward += breakdown["enemy_hp_delta"]

        tactical_delta = self._combat_tactical_delta(previous_state, next_state)
        if tactical_delta != 0.0:
            breakdown["combat_tactical"] = tactical_delta
            reward += tactical_delta

        turn_skip_unspent_penalty, turn_skip_unspent_energy = self._turn_skip_unspent_penalty(
            previous_state,
            tool_name,
            boundaries,
            player_survived,
        )
        end_turn_slack = 0.0 if turn_skip_unspent_penalty > 0.0 else self._combat_end_turn_slack(previous_state, next_state, tool_name, boundaries)
        if end_turn_slack != 0.0:
            breakdown["end_turn_slack"] = -end_turn_slack
            reward -= end_turn_slack

        if boundaries["turn_end"] and self.reward_weights.turn_end != 0.0 and turn_skip_unspent_penalty <= 0.0:
            breakdown["turn_end"] = self.reward_weights.turn_end
            reward += breakdown["turn_end"]
        if turn_skip_unspent_penalty != 0.0:
            breakdown["turn_skip_unspent"] = -turn_skip_unspent_penalty
            breakdown["turn_skip_unspent_energy"] = float(turn_skip_unspent_energy)
            reward -= turn_skip_unspent_penalty
        if boundaries["combat_end"]:
            combat_weight = self.reward_weights.combat_end if player_survived else self.reward_weights.combat_defeat
            breakdown["combat_end"] = combat_weight
            reward += combat_weight
        if boundaries["act_end"] and self.reward_weights.act_end != 0.0:
            breakdown["act_end"] = self.reward_weights.act_end
            reward += breakdown["act_end"]
        if boundaries["run_end"]:
            run_weight = self.reward_weights.run_victory if player_survived else self.reward_weights.run_defeat
            breakdown["run_end"] = run_weight
            reward += run_weight
        breakdown["total"] = reward
        return reward, breakdown

    def _combat_player_survived(self, previous_state: dict[str, object], next_state: dict[str, object]) -> bool:
        next_hp = _player_hp(next_state)
        if next_hp > 0.0:
            return True
        if str(next_state.get("state_type", "unknown")) == "menu":
            return _player_hp(previous_state) > 0.0
        return False

    def _turn_skip_unspent_penalty(
        self,
        previous_state: dict[str, object],
        tool_name: str,
        boundaries: dict[str, object],
        player_survived: bool,
    ) -> tuple[float, int]:
        if tool_name != "combat_end_turn" or not boundaries["turn_end"]:
            return 0.0, 0
        if str(previous_state.get("state_type", "unknown")) not in COMBAT_STATE_TYPES:
            return 0.0, 0
        if boundaries["combat_end"] and player_survived:
            return 0.0, 0
        lost_energy = _unspent_energy_units(previous_state)
        if lost_energy <= 0:
            return 0.0, 0
        if _count_playable_cards(previous_state) <= 0:
            return 0.0, 0
        configured = self.reward_weights.turn_skip_unspent_penalty
        if configured > 0.0:
            return configured * lost_energy, lost_energy
        fallback = max(0.0, self.reward_weights.turn_end)
        return fallback * lost_energy, lost_energy

    def _combat_tactical_delta(self, previous_state: dict[str, object], next_state: dict[str, object]) -> float:
        if self.reward_weights.combat_tactical_shaping <= 0.0 or str(previous_state.get("state_type", "unknown")) not in COMBAT_STATE_TYPES:
            return 0.0
        previous_value = self._combat_tactical_potential(previous_state)
        next_value = self._combat_tactical_potential(next_state) if str(next_state.get("state_type", "unknown")) in COMBAT_STATE_TYPES else 0.0
        return self.reward_weights.combat_tactical_shaping * ((self.reward_discount * next_value) - previous_value)

    def _combat_tactical_potential(self, state: dict[str, object]) -> float:
        if str(state.get("state_type", "unknown")) not in COMBAT_STATE_TYPES:
            return 0.0
        player = state.get("player") if isinstance(state.get("player"), dict) else {}
        battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
        enemies = battle.get("enemies") if isinstance(battle.get("enemies"), list) else []
        max_hp = max(float(player.get("max_hp", 0) or 0), 1.0)
        hp_ratio = _safe_div(player.get("hp"), max_hp)
        block = float(player.get("block", 0) or 0)
        incoming_damage = _incoming_enemy_damage(state)
        block_coverage = min(block, incoming_damage) / max(incoming_damage, 1.0) if incoming_damage > 0.0 else 0.0
        unblocked_pressure = min(1.0, max(0.0, incoming_damage - block) / max_hp)
        overblock_pressure = min(1.0, max(0.0, block - incoming_damage) / max_hp)
        normalized_enemy_hp = 0.0
        for enemy in enemies:
            if not isinstance(enemy, dict):
                continue
            enemy_hp = float(enemy.get("hp", 0) or 0)
            enemy_max_hp = max(float(enemy.get("max_hp", 0) or 0), 1.0)
            if enemy_hp <= 0.0:
                continue
            normalized_enemy_hp += min(1.0, enemy_hp / enemy_max_hp)
        normalized_enemy_hp = min(1.0, normalized_enemy_hp / 3.0)
        return (
            (0.80 * hp_ratio)
            + (0.35 * block_coverage)
            - (0.65 * normalized_enemy_hp)
            - (0.45 * unblocked_pressure)
            - (0.20 * overblock_pressure)
        )

    def _combat_end_turn_slack(
        self,
        previous_state: dict[str, object],
        next_state: dict[str, object],
        tool_name: str,
        boundaries: dict[str, object],
    ) -> float:
        if self.reward_weights.end_turn_slack_penalty <= 0.0 or tool_name != "combat_end_turn" or boundaries["combat_end"]:
            return 0.0
        if str(previous_state.get("state_type", "unknown")) not in COMBAT_STATE_TYPES:
            return 0.0
        player = previous_state.get("player") if isinstance(previous_state.get("player"), dict) else {}
        hand_cards = player.get("hand", []) if isinstance(player.get("hand"), list) else []
        playable_cards = _count_playable_cards(previous_state)
        if playable_cards <= 0:
            return 0.0
        energy = max(0.0, float(player.get("energy", 0) or 0))
        max_energy = max(float(player.get("max_energy", 0) or 0), 1.0)
        if energy <= 0.0:
            return 0.0
        hand_size = max(1, len(hand_cards))
        incoming_damage = _incoming_enemy_damage(previous_state)
        block = float(player.get("block", 0) or 0)
        max_hp = max(float(player.get("max_hp", 0) or 0), 1.0)
        danger_ratio = min(1.0, max(0.0, incoming_damage - block) / max_hp)
        playable_ratio = min(1.0, playable_cards / hand_size)
        energy_ratio = min(1.0, energy / max_energy)
        enemy_progress = max(0.0, _total_enemy_hp(previous_state) - _total_enemy_hp(next_state))
        progress_relief = min(0.5, enemy_progress / 20.0)
        slack = (0.55 * energy_ratio) + (0.25 * playable_ratio) + (0.60 * danger_ratio) - progress_relief
        return self.reward_weights.end_turn_slack_penalty * max(0.0, slack)
