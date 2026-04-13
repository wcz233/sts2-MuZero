from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

CATALOG_GLOB = "sts2*.json"
COMBAT_STATE_TYPES = {"monster", "elite", "boss"}
COMBAT_RELATED_STATE_TYPES = COMBAT_STATE_TYPES | {"hand_select"}
TERMINAL_STATE_TYPES = {"menu", "game_over"}
ENEMY_TARGET_TYPES = {"ANYENEMY", "ENEMY", "SINGLEENEMY"}
ALL_ENEMY_TARGET_TYPES = {"ALLENEMIES"}
SELF_TARGET_TYPES = {"SELF", "ANYALLY", "ANYPLAYER"}
ACTION_HISTORY_BUCKETS = (
    "none",
    "combat_attack_card",
    "combat_skill_card",
    "combat_power_card",
    "combat_other_card",
    "combat_end_turn",
    "use_potion",
    "discard_potion",
    "combat_select",
    "rewards",
    "map",
    "event",
    "rest",
    "shop",
    "proceed",
    "deck_select",
    "bundle_select",
    "relic_select",
    "treasure",
    "crystal",
    "other",
)
COMBAT_ROUND_BUCKETS = (
    "ROUND_0",
    "ROUND_1",
    "ROUND_2",
    "ROUND_3",
    "ROUND_4",
    "ROUND_5",
    "ROUND_6",
    "ROUND_7",
    "ROUND_8",
    "ROUND_9",
    "ROUND_10",
    "ROUND_11_PLUS",
)
MAP_NODE_TYPES = (
    "Start",
    "Monster",
    "Elite",
    "RestSite",
    "Shop",
    "Event",
    "Treasure",
    "Boss",
    "Unknown",
)
MAP_ROUTE_TRACKED_TYPES = (
    "Monster",
    "Elite",
    "RestSite",
    "Shop",
    "Event",
    "Treasure",
    "Boss",
)
MAP_ROUTE_OPTION_COUNT = 6
MAP_ROUTE_DEPTH_LABELS = ("DEPTH_0", "DEPTH_1", "DEPTH_2", "DEPTH_3")
MAP_ROUTE_DEPTH_LIMIT = len(MAP_ROUTE_DEPTH_LABELS) - 1
MAP_NODE_TYPE_INDEX = {node_type: index for index, node_type in enumerate(MAP_NODE_TYPES)}
SHOP_SLOT_COUNT = 12
SHOP_SLOT_CATEGORIES = ("CARD", "RELIC", "POTION", "CARD_REMOVAL", "SPECIAL", "UNKNOWN")
EVENT_OPTION_COUNT = 5
EVENT_OPTION_KINDS = ("NONE", "PROCEED", "RELIC", "CARD", "POTION", "KEYWORD", "TEXT", "UNKNOWN")
HAND_SLOT_COUNT = 10
HAND_SELECT_SLOT_COUNT = HAND_SLOT_COUNT
CARD_SELECT_SLOT_COUNT = 20
HAND_ORDER_PRIORITY_SCALE = 8.0
CARD_OPERATION_KINDS = ("CONSUME", "UPGRADE", "REMOVE", "ENCHANT")
MAP_ROUTE_SCORE_WEIGHTS = {
    "Start": 0.0,
    "Monster": 0.55,
    "Elite": 0.80,
    "RestSite": 1.10,
    "Shop": 1.00,
    "Event": 1.00,
    "Treasure": 1.20,
    "Boss": 2.80,
    "Unknown": 0.15,
}
RUNTIME_CONCEPTS: dict[str, tuple[str, ...]] = {
    "STATE_TYPE": (
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
    ),
    "CARD_TYPE": ("None", "Attack", "Skill", "Power", "Status", "Curse", "Quest"),
    "CARD_RARITY": ("None", "Basic", "Common", "Uncommon", "Rare", "Ancient", "Special"),
    "POWER_TYPE": ("None", "Buff", "Debuff", "Special", "Neutral"),
    "RELIC_RARITY": ("None", "Common", "Uncommon", "Rare", "Shop", "Ancient", "Starter", "Event"),
    "RELIC_STATUS": ("None", "Normal", "Active", "Disabled"),
    "POTION_RARITY": ("None", "Common", "Uncommon", "Rare", "Starter", "Event"),
    "POTION_USAGE": ("None", "CombatOnly", "AnyTime"),
    "TARGET_TYPE": ("None", "Self", "AnyEnemy", "Enemy", "SingleEnemy", "AllEnemies", "AnyAlly", "AnyPlayer"),
    "INTENT_TYPE": (
        "None",
        "Attack",
        "AttackBuff",
        "AttackDebuff",
        "AttackDefend",
        "Buff",
        "Debuff",
        "Defend",
        "DefendBuff",
        "Escape",
        "Sleep",
        "Stun",
    ),
    "ENCHANTMENT_STATUS": ("None", "Normal", "Disabled"),
    "COMBAT_ROUND_BUCKET": COMBAT_ROUND_BUCKETS,
}
ENTITY_GROUPS = (
    "acts",
    "characters",
    "cards",
    "relics",
    "potions",
    "powers",
    "afflictions",
    "enchantments",
    "events",
    "ancients",
    "encounters",
    "monsters",
    "orbs",
)


def _slugify(text: object) -> str:
    raw = str(text).strip()
    if not raw:
        return ""
    raw = re.sub(r"(?<!^)(?=[A-Z])", "_", raw)
    raw = re.sub(r"[\s/\-]+", "_", raw.upper())
    return re.sub(r"[^A-Z0-9_]", "", raw)


def _normalize_model_token(value: object) -> str:
    text = str(value).strip()
    if not text:
        return ""
    if "." not in text:
        return _slugify(text)
    parts = [_slugify(part) for part in text.split(".")]
    return ".".join(part for part in parts if part)


def _clean_text(value: object) -> str:
    text = str(value or "").strip()
    return text if text and text.lower() != "none" else ""


def _text_has_any(text: str, tokens: tuple[str, ...]) -> bool:
    return any(token in text for token in tokens)


def _selection_text_blob(container: object) -> str:
    if not isinstance(container, dict):
        return ""
    return " ".join(
        str(container.get(key, "") or "").strip().lower()
        for key in ("screen_type", "prompt", "title", "header", "body", "message", "reason", "mode")
    )


def _selection_operation_flags(container: object) -> dict[str, bool]:
    text = _selection_text_blob(container)
    return {
        "consume": _text_has_any(text, ("exhaust", "consume", "burn", "sacrifice", "消耗", "耗尽", "枯竭", "焚烧", "献祭")),
        "upgrade": _text_has_any(text, ("upgrade", "强化", "升级")),
        "remove": _text_has_any(text, ("remove", "purge", "delete", "forget", "transform", "change", "replace", "mutate", "移除", "删除", "遗忘", "删牌", "变换", "变化", "替换", "变形")),
        "enchant": _text_has_any(text, ("enchant", "imbue", "附魔", "灌注")),
    }


def _raw_card_consumes_on_play(raw: dict[str, object]) -> bool:
    for key in (
        "exhausts",
        "exhaust",
        "consumes",
        "consume",
        "purge_on_use",
        "remove_on_use",
        "exhausts_on_use",
        "is_exhaust",
        "is_exhaust_on_play",
        "is_consumed_on_play",
    ):
        value = raw.get(key)
        if isinstance(value, bool):
            if value:
                return True
            continue
        token = str(value or "").strip().lower()
        if token in {"true", "yes", "1", "exhaust", "consume", "consumed"}:
            return True
    description = " ".join(
        str(raw.get(key, "") or "").strip().lower()
        for key in ("rules_text", "description", "text", "tooltip")
    )
    return _text_has_any(description, ("exhaust", "consume", "消耗", "耗尽", "枯竭"))


def _normalize_keyword_tokens(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    tokens: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        raw = (
            item.get("glossary_id")
            or item.get("display_text")
            or item.get("name")
            or item.get("title")
            or item.get("id")
        )
        token = _slugify(raw)
        if not token or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tuple(tokens)


def _iter_dicts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _dict_at(parent: object, key: str) -> dict[str, object]:
    if isinstance(parent, dict):
        child = parent.get(key)
        if isinstance(child, dict):
            return child
    return {}


def _list_at(parent: object, key: str) -> list[dict[str, object]]:
    if isinstance(parent, dict):
        return _iter_dicts(parent.get(key))
    return []


def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _ratio(value: object, scale: float, clamp: float = 1.0) -> float:
    if scale <= 0.0:
        return 0.0
    ratio = _to_float(value) / scale
    if ratio < 0.0:
        return 0.0
    if ratio > clamp:
        return clamp
    return ratio


def _signed_ratio(value: object, scale: float, clamp: float = 1.0) -> float:
    if scale <= 0.0:
        return 0.0
    ratio = _to_float(value) / scale
    if ratio < -clamp:
        return -clamp
    if ratio > clamp:
        return clamp
    return ratio


def _normalize_map_node_type(value: object) -> str:
    token = _slugify(value)
    return {
        "START": "Start",
        "MONSTER": "Monster",
        "ELITE": "Elite",
        "REST_SITE": "RestSite",
        "RESTSITE": "RestSite",
        "SHOP": "Shop",
        "EVENT": "Event",
        "TREASURE": "Treasure",
        "BOSS": "Boss",
        "UNKNOWN": "Unknown",
    }.get(token, "Unknown")


def _normalize_option_type(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    node_type = _normalize_map_node_type(text)
    if node_type != "Unknown" or _slugify(text) == "UNKNOWN":
        return node_type
    return text


def _coordinate_pair(value: object) -> tuple[int, int] | None:
    if isinstance(value, dict):
        raw_col = value.get("col")
        raw_row = value.get("row")
    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        raw_col, raw_row = value[0], value[1]
    else:
        return None
    try:
        return int(raw_col), int(raw_row)
    except (TypeError, ValueError):
        return None


def _empty_map_type_counts() -> tuple[int, ...]:
    return tuple(0 for _ in MAP_NODE_TYPES)


def _empty_map_depth_type_counts() -> tuple[tuple[int, ...], ...]:
    return tuple(_empty_map_type_counts() for _ in MAP_ROUTE_DEPTH_LABELS)


def _dominant_map_node_type(type_counts: tuple[int, ...]) -> str:
    best_type = "Unknown"
    best_count = 0
    for node_type in MAP_ROUTE_TRACKED_TYPES:
        count = type_counts[MAP_NODE_TYPE_INDEX[node_type]]
        if count > best_count:
            best_type = node_type
            best_count = count
    return best_type if best_count > 0 else "Unknown"


def map_type_count(type_counts: tuple[int, ...], node_type: str) -> int:
    if not type_counts:
        return 0
    return type_counts[MAP_NODE_TYPE_INDEX.get(node_type, MAP_NODE_TYPE_INDEX["Unknown"])]


def map_type_ratio(type_counts: tuple[int, ...], node_type: str) -> float:
    total = sum(type_counts)
    if total <= 0:
        return 0.0
    return map_type_count(type_counts, node_type) / total


def map_depth_type_ratio(depth_type_counts: tuple[tuple[int, ...], ...], depth: int, node_type: str) -> float:
    if depth < 0 or depth >= len(depth_type_counts):
        return 0.0
    return map_type_ratio(depth_type_counts[depth], node_type)


def map_depth_dominant_type(depth_type_counts: tuple[tuple[int, ...], ...], depth: int) -> str:
    if depth < 0 or depth >= len(depth_type_counts):
        return "Unknown"
    return _dominant_map_node_type(depth_type_counts[depth])


def _safe_div(numerator: object, denominator: object) -> float:
    bottom = _to_float(denominator)
    if bottom == 0.0:
        return 0.0
    return _to_float(numerator) / bottom


def _bool(value: object) -> float:
    return 1.0 if value else 0.0


def _cost_units(cost: object, fallback_energy: float = 1.0) -> float:
    if cost == "X":
        return max(0.0, fallback_energy)
    return max(0.0, _to_float(cost))


def _match_token(value: object, allowed: set[str]) -> bool:
    return _slugify(value) in allowed


def _hand_card_type_rank(card_type: str) -> int:
    return {
        "Power": 5,
        "Attack": 4,
        "Skill": 3,
        "Status": 2,
        "Curse": 1,
    }.get(card_type, 0)


def _hand_card_target_rank(card: SemanticCard) -> int:
    if _match_token(card.target_type, ALL_ENEMY_TARGET_TYPES):
        return 3
    if _match_token(card.target_type, ENEMY_TARGET_TYPES):
        return 2
    if _match_token(card.target_type, SELF_TARGET_TYPES) and card.card_type != "Attack":
        return 1
    return 0


def _hand_card_priority(card: SemanticCard, state: SemanticState) -> float:
    score = 0.0
    living_enemy_count = len(state.living_enemies)
    incoming_damage = max(0.0, state.incoming_damage)
    block_gap = max(0.0, state.incoming_damage - state.player_block)
    block_surplus = max(0.0, state.player_block - state.incoming_damage)
    cost_units = _cost_units(card.cost, state.player_energy)
    hp_ratio = state.player_hp / max(state.player_max_hp, 1.0)
    is_self_non_attack = _match_token(card.target_type, SELF_TARGET_TYPES) and card.card_type != "Attack"
    is_attack = card.card_type == "Attack"
    if card.card_type == "Power":
        score += 4.25
    elif card.card_type == "Skill":
        score += 0.75
    elif card.card_type == "Attack":
        score += 1.25
    elif card.card_type == "Status":
        score -= 3.5
    elif card.card_type == "Curse":
        score -= 4.0
    if is_self_non_attack:
        score += 0.25
    if _match_token(card.target_type, ALL_ENEMY_TARGET_TYPES) and living_enemy_count > 1:
        score += 0.75
    if block_gap > 0.0:
        if is_self_non_attack:
            defensive_urgency = block_gap / max(10.0, state.player_max_hp * 0.18)
            if hp_ratio < 0.35:
                defensive_urgency += 0.15
            score += min(1.35, 0.35 + defensive_urgency)
        elif is_attack:
            score -= 0.20
    elif is_self_non_attack:
        if incoming_damage <= 0.0:
            score -= 1.00
        elif block_surplus > 0.0:
            score -= min(0.90, 0.20 + (block_surplus / max(12.0, state.player_max_hp * 0.20)))
    if incoming_damage <= 0.0 and is_attack:
        score += 0.35
    if cost_units >= 2.0:
        score += 0.50
    elif cost_units == 0.0 and card.can_play:
        score += 0.25
    if card.is_upgraded:
        score += 0.35
    score += min(1.2, len(card.enchantments) * 0.35)
    score -= min(1.8, len(card.afflictions) * 0.6)
    if not card.can_play:
        score -= 6.0
    return score


def _hand_card_sort_key(card: SemanticCard, state: SemanticState) -> tuple[float, int, int, int, float, int]:
    priority = _hand_card_priority(card, state)
    return (
        priority,
        _hand_card_type_rank(card.card_type),
        _hand_card_target_rank(card),
        1 if card.is_upgraded else 0,
        _cost_units(card.cost, state.player_energy),
        -card.index,
    )


def build_hand_order_profile(state: dict[str, object] | SemanticState) -> HandOrderProfile:
    semantic_state = ensure_semantic_state(state)
    hand_by_index = {
        card.index: card
        for card in semantic_state.hand
        if 0 <= card.index < HAND_SLOT_COUNT
    }
    slot_present = [False] * HAND_SLOT_COUNT
    slot_priorities = [0.0] * HAND_SLOT_COUNT
    slot_before_counts = [0] * HAND_SLOT_COUNT
    slot_after_counts = [0] * HAND_SLOT_COUNT
    pair_prefer_before = [[False] * HAND_SLOT_COUNT for _ in range(HAND_SLOT_COUNT)]
    for slot_index, card in hand_by_index.items():
        slot_present[slot_index] = True
        slot_priorities[slot_index] = _hand_card_priority(card, semantic_state)
    ordered_slots = sorted(hand_by_index)
    for left_index in ordered_slots:
        left_card = hand_by_index[left_index]
        left_key = _hand_card_sort_key(left_card, semantic_state)
        for right_index in ordered_slots:
            if left_index >= right_index:
                continue
            right_card = hand_by_index[right_index]
            right_key = _hand_card_sort_key(right_card, semantic_state)
            if left_key > right_key:
                earlier_index, later_index = left_index, right_index
            else:
                earlier_index, later_index = right_index, left_index
            pair_prefer_before[earlier_index][later_index] = True
            slot_before_counts[earlier_index] += 1
            slot_after_counts[later_index] += 1
    return HandOrderProfile(
        slot_present=tuple(slot_present),
        slot_priorities=tuple(slot_priorities),
        slot_before_counts=tuple(slot_before_counts),
        slot_after_counts=tuple(slot_after_counts),
        pair_prefer_before=tuple(tuple(row) for row in pair_prefer_before),
    )


def build_hand_order_action_biases(
    state: dict[str, object] | SemanticState,
    profile: HandOrderProfile | None = None,
) -> dict[int, float]:
    semantic_state = ensure_semantic_state(state)
    hand_profile = profile or build_hand_order_profile(semantic_state)
    available_slots = hand_profile.available_slots
    if not available_slots:
        return {}
    denominator = max(len(available_slots) - 1, 1)
    biases: dict[int, float] = {}
    for slot_index in available_slots:
        relation_balance = (hand_profile.slot_before_counts[slot_index] - hand_profile.slot_after_counts[slot_index]) / denominator
        priority_component = _signed_ratio(hand_profile.slot_priorities[slot_index], HAND_ORDER_PRIORITY_SCALE, 1.0)
        biases[slot_index] = max(-2.0, min(2.0, relation_balance * 1.25 + priority_component * 0.75))
    return biases


def _intent_damage_from_label(label: object) -> float:
    digits = "".join(character for character in str(label) if character.isdigit())
    return float(digits) if digits else 0.0


def combat_round_bucket(value: object) -> str:
    round_value = max(0, _to_int(value))
    if round_value >= 11:
        return COMBAT_ROUND_BUCKETS[-1]
    return COMBAT_ROUND_BUCKETS[min(round_value, len(COMBAT_ROUND_BUCKETS) - 2)]


def _default_catalog_candidates() -> list[Path]:
    env_path = os.getenv("STS2_SEMANTIC_CATALOG_PATH")
    here = Path(__file__).resolve()
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    for root in (here.parents[1], here.parents[2], Path.cwd()):
        doc_dir = root / "doc"
        if not doc_dir.exists():
            continue
        candidates.extend(sorted(doc_dir.glob(CATALOG_GLOB)))
    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


@dataclass(frozen=True)
class SemanticCatalog:
    source_path: Path | None
    concept_keys: tuple[str, ...]
    concept_index: dict[str, int]

    @property
    def size(self) -> int:
        return len(self.concept_keys)

    @classmethod
    def empty(cls) -> "SemanticCatalog":
        return cls(source_path=None, concept_keys=(), concept_index={})


@dataclass(frozen=True)
class ConceptActivation:
    active_keys: tuple[str, ...]
    vector: list[float]


@dataclass(frozen=True)
class SemanticPower:
    id: str = ""
    power_type: str = ""
    amount: float = 0.0


@dataclass(frozen=True)
class SemanticEnchantment:
    id: str = ""
    status: str = ""
    amount: float = 0.0


@dataclass(frozen=True)
class SemanticAffliction:
    id: str = ""
    amount: float = 0.0


@dataclass(frozen=True)
class SemanticCard:
    index: int = -1
    id: str = ""
    name: str = ""
    card_type: str = ""
    rarity: str = ""
    cost: str = ""
    target_type: str = ""
    can_play: bool = False
    consumes_on_play: bool = False
    is_upgraded: bool = False
    enchantments: tuple[SemanticEnchantment, ...] = ()
    afflictions: tuple[SemanticAffliction, ...] = ()


@dataclass(frozen=True)
class SemanticRelic:
    id: str = ""
    rarity: str = ""
    status: str = ""
    counter: float = 0.0
    is_used_up: bool = False
    is_wax: bool = False
    is_melted: bool = False


@dataclass(frozen=True)
class SemanticPotion:
    slot: int = -1
    id: str = ""
    rarity: str = ""
    usage: str = ""
    target_type: str = ""
    can_use_in_combat: bool = False


@dataclass(frozen=True)
class SemanticEnemy:
    entity_id: str = ""
    monster_id: str = ""
    move_id: str = ""
    hp: float = 0.0
    max_hp: float = 0.0
    block: float = 0.0
    powers: tuple[SemanticPower, ...] = ()
    intent_types: tuple[str, ...] = ()
    intent_damage: float = 0.0


@dataclass(frozen=True)
class SemanticRewardItem:
    kind: str = ""
    relic_id: str = ""
    potion_id: str = ""
    relic_rarity: str = ""
    potion_rarity: str = ""
    potion_usage: str = ""


@dataclass(frozen=True)
class SemanticShopItem:
    index: int = -1
    category: str = ""
    can_afford: bool = False
    is_stocked: bool = True
    price: float = 0.0
    item_token: str = ""
    card_id: str = ""
    relic_id: str = ""
    potion_id: str = ""
    card_type: str = ""
    card_rarity: str = ""
    relic_rarity: str = ""
    potion_rarity: str = ""
    potion_usage: str = ""


@dataclass(frozen=True)
class SemanticOption:
    index: int = -1
    is_locked: bool = False
    is_proceed: bool = False
    is_enabled: bool = False
    was_chosen: bool = False
    relic_id: str = ""
    card_id: str = ""
    potion_id: str = ""
    option_type: str = ""
    title: str = ""
    description: str = ""
    keyword_tokens: tuple[str, ...] = ()


@dataclass(frozen=True)
class SemanticMapRoute:
    option_index: int = -1
    col: int = 0
    row: int = 0
    option_type: str = "Unknown"
    reachable_count: int = 0
    path_count: int = 0
    max_depth: int = 0
    boss_distance: int = -1
    route_score: float = 0.0
    dominant_type: str = "Unknown"
    depth_dominant_types: tuple[str, ...] = ()
    type_counts: tuple[int, ...] = ()
    depth_type_counts: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True)
class SemanticState:
    state_type: str = "unknown"
    act: int = 0
    act_id: str = ""
    floor: int = 0
    ascension: int = 0
    character_id: str = ""
    player_hp: float = 0.0
    player_max_hp: float = 0.0
    player_block: float = 0.0
    player_energy: float = 0.0
    player_max_energy: float = 0.0
    player_gold: float = 0.0
    player_powers: tuple[SemanticPower, ...] = ()
    relics: tuple[SemanticRelic, ...] = ()
    potions: tuple[SemanticPotion, ...] = ()
    orb_ids: tuple[str, ...] = ()
    hand: tuple[SemanticCard, ...] = ()
    draw_pile: tuple[SemanticCard, ...] = ()
    discard_pile: tuple[SemanticCard, ...] = ()
    exhaust_pile: tuple[SemanticCard, ...] = ()
    draw_pile_count: int = 0
    discard_pile_count: int = 0
    exhaust_pile_count: int = 0
    encounter_id: str = ""
    battle_round: int = 0
    battle_turn: str = ""
    is_play_phase: bool = False
    player_actions_disabled: bool = False
    hand_in_card_play: bool = False
    enemies: tuple[SemanticEnemy, ...] = ()
    reward_items: tuple[SemanticRewardItem, ...] = ()
    rewards_can_proceed: bool = False
    card_reward_cards: tuple[SemanticCard, ...] = ()
    card_reward_can_skip: bool = False
    map_options: tuple[SemanticOption, ...] = ()
    map_current_row: int = 0
    map_remaining_node_count: int = 0
    map_remaining_boss_distance: int = -1
    map_remaining_dominant_type: str = "Unknown"
    map_remaining_type_counts: tuple[int, ...] = ()
    map_remaining_depth_type_counts: tuple[tuple[int, ...], ...] = ()
    map_route_summaries: tuple[SemanticMapRoute, ...] = ()
    event_id: str = ""
    event_name: str = ""
    event_in_dialogue: bool = False
    event_options: tuple[SemanticOption, ...] = ()
    rest_options: tuple[SemanticOption, ...] = ()
    rest_can_proceed: bool = False
    card_select_cards: tuple[SemanticCard, ...] = ()
    card_select_preview_cards: tuple[SemanticCard, ...] = ()
    card_select_is_consume: bool = False
    card_select_is_upgrade: bool = False
    card_select_is_remove: bool = False
    card_select_is_enchant: bool = False
    card_select_can_confirm: bool = False
    card_select_can_cancel: bool = False
    bundle_cards: tuple[SemanticCard, ...] = ()
    bundle_preview_cards: tuple[SemanticCard, ...] = ()
    bundle_count: int = 0
    bundle_can_confirm: bool = False
    bundle_can_cancel: bool = False
    hand_select_cards: tuple[SemanticCard, ...] = ()
    hand_select_selected_cards: tuple[SemanticCard, ...] = ()
    hand_select_is_consume: bool = False
    hand_select_is_upgrade: bool = False
    hand_select_is_remove: bool = False
    hand_select_is_enchant: bool = False
    hand_select_can_confirm: bool = False
    relic_select_relics: tuple[SemanticRelic, ...] = ()
    relic_select_can_skip: bool = False
    treasure_relics: tuple[SemanticRelic, ...] = ()
    treasure_can_proceed: bool = False
    shop_items: tuple[SemanticShopItem, ...] = ()
    shop_can_proceed: bool = False
    fake_merchant_shop_items: tuple[SemanticShopItem, ...] = ()
    fake_merchant_can_proceed: bool = False
    crystal_clickable_count: int = 0
    crystal_can_proceed: bool = False

    @property
    def in_combat(self) -> bool:
        return self.state_type in COMBAT_RELATED_STATE_TYPES

    @property
    def is_player_turn(self) -> bool:
        return self.in_combat and self.battle_turn == "player"

    @property
    def living_enemies(self) -> tuple[SemanticEnemy, ...]:
        return tuple(enemy for enemy in self.enemies if enemy.hp > 0.0)

    @property
    def total_enemy_hp(self) -> float:
        return sum(enemy.hp for enemy in self.living_enemies)

    @property
    def total_enemy_max_hp(self) -> float:
        return sum(max(enemy.max_hp, enemy.hp, 1.0) for enemy in self.living_enemies)

    @property
    def incoming_damage(self) -> float:
        return sum(enemy.intent_damage for enemy in self.living_enemies)

    @property
    def playable_cards(self) -> tuple[SemanticCard, ...]:
        return tuple(card for card in self.hand if card.can_play)


@dataclass(frozen=True)
class SemanticObservation:
    semantic_state: SemanticState
    concept_activation: ConceptActivation
    scalar_vector: list[float]
    relation_vector: list[float]
    history_vector: list[float]
    vector: list[float]


EMPTY_HAND_SLOT_PRESENT = tuple(False for _ in range(HAND_SLOT_COUNT))
EMPTY_HAND_SLOT_FLOATS = tuple(0.0 for _ in range(HAND_SLOT_COUNT))
EMPTY_HAND_SLOT_INTS = tuple(0 for _ in range(HAND_SLOT_COUNT))
EMPTY_HAND_SLOT_MATRIX = tuple(tuple(False for _ in range(HAND_SLOT_COUNT)) for _ in range(HAND_SLOT_COUNT))


@dataclass(frozen=True)
class HandOrderProfile:
    slot_present: tuple[bool, ...] = EMPTY_HAND_SLOT_PRESENT
    slot_priorities: tuple[float, ...] = EMPTY_HAND_SLOT_FLOATS
    slot_before_counts: tuple[int, ...] = EMPTY_HAND_SLOT_INTS
    slot_after_counts: tuple[int, ...] = EMPTY_HAND_SLOT_INTS
    pair_prefer_before: tuple[tuple[bool, ...], ...] = EMPTY_HAND_SLOT_MATRIX

    @property
    def available_slots(self) -> tuple[int, ...]:
        return tuple(index for index, present in enumerate(self.slot_present) if present)


class _FeatureBuilder:
    def __init__(self) -> None:
        self.names: list[str] = []
        self.values: list[float] = []

    def add(self, name: str, value: object) -> None:
        self.names.append(name)
        self.values.append(float(value))


class SemanticHistoryTracker:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.last_action_bucket = ACTION_HISTORY_BUCKETS[0]
        self.last_action_error = False
        self.last_action_changed_state_type = False
        self.last_turn_end = False
        self.last_combat_end = False
        self.last_act_end = False
        self.last_run_end = False
        self.last_turn_skipped_unspent = False
        self.last_enemy_hp_progress = 0.0
        self.last_player_hp_loss = 0.0
        self.last_hand_order_refresh = False
        self.last_hand_order_refresh_turn_start = False
        self.last_hand_order_refresh_hand_changed = False
        self.turn_step_count = 0
        self.turn_cards_played = 0
        self.turn_attack_cards = 0
        self.turn_skill_cards = 0
        self.turn_power_cards = 0
        self.turn_potions_used = 0
        self.turn_failed_actions = 0
        self.turn_energy_spent = 0.0
        self.turn_enemy_damage = 0.0
        self.turn_player_damage = 0.0
        self.turn_hand_order_refreshes = 0
        self.combat_step_count = 0
        self.combat_cards_played = 0
        self.combat_potions_used = 0
        self.combat_failed_actions = 0
        self.combat_turn_ends = 0
        self.combat_skip_unspent = 0
        self.combat_enemy_damage = 0.0
        self.combat_player_damage = 0.0
        self._run_progress: tuple[int, int] | None = None
        self._combat_anchor: tuple[int, int, str] | None = None
        self._current_turn_round: int | None = None
        self._hand_order_signature: tuple[object, ...] | None = None
        self._hand_order_profile = HandOrderProfile()
        self._awaiting_run_reset = False

    def _clear_transients(self) -> None:
        self.last_action_error = False
        self.last_action_changed_state_type = False
        self.last_turn_end = False
        self.last_combat_end = False
        self.last_act_end = False
        self.last_run_end = False
        self.last_turn_skipped_unspent = False
        self.last_enemy_hp_progress = 0.0
        self.last_player_hp_loss = 0.0
        self.last_hand_order_refresh = False
        self.last_hand_order_refresh_turn_start = False
        self.last_hand_order_refresh_hand_changed = False

    def _reset_turn_progress(self) -> None:
        self.turn_step_count = 0
        self.turn_cards_played = 0
        self.turn_attack_cards = 0
        self.turn_skill_cards = 0
        self.turn_power_cards = 0
        self.turn_potions_used = 0
        self.turn_failed_actions = 0
        self.turn_energy_spent = 0.0
        self.turn_enemy_damage = 0.0
        self.turn_player_damage = 0.0
        self.turn_hand_order_refreshes = 0

    def _reset_combat_progress(self) -> None:
        self._reset_turn_progress()
        self.combat_step_count = 0
        self.combat_cards_played = 0
        self.combat_potions_used = 0
        self.combat_failed_actions = 0
        self.combat_turn_ends = 0
        self.combat_skip_unspent = 0
        self.combat_enemy_damage = 0.0
        self.combat_player_damage = 0.0
        self._combat_anchor = None
        self._current_turn_round = None
        self._hand_order_signature = None
        self._hand_order_profile = HandOrderProfile()

    @property
    def hand_order_profile(self) -> HandOrderProfile:
        return self._hand_order_profile

    def _hand_order_state_signature(self, state: SemanticState) -> tuple[object, ...] | None:
        if not state.in_combat or not state.is_player_turn:
            return None
        return (
            state.act,
            state.floor,
            state.encounter_id,
            state.battle_round,
            tuple(
                (
                    card.index,
                    card.id,
                    card.card_type,
                    card.cost,
                    card.target_type,
                    card.can_play,
                    card.consumes_on_play,
                    card.is_upgraded,
                    len(card.enchantments),
                    len(card.afflictions),
                )
                for card in state.hand
                if 0 <= card.index < HAND_SLOT_COUNT
            ),
        )

    def _sync_hand_order_profile(self, state: SemanticState, *, turn_started: bool) -> None:
        signature = self._hand_order_state_signature(state)
        if signature is None:
            self._hand_order_signature = None
            self._hand_order_profile = HandOrderProfile()
            return
        if self._hand_order_signature == signature:
            return
        self._hand_order_signature = signature
        self._hand_order_profile = build_hand_order_profile(state)
        self.last_hand_order_refresh = True
        self.last_hand_order_refresh_turn_start = turn_started
        self.last_hand_order_refresh_hand_changed = not turn_started
        self.turn_hand_order_refreshes += 1

    def sync_state(self, state: dict[str, object] | SemanticState) -> SemanticState:
        semantic_state = ensure_semantic_state(state)
        turn_started = False
        if semantic_state.state_type not in TERMINAL_STATE_TYPES:
            if self._awaiting_run_reset:
                self.reset()
                self._awaiting_run_reset = False
            if self._run_progress is not None:
                previous_act, previous_floor = self._run_progress
                if semantic_state.act < previous_act or semantic_state.floor < previous_floor:
                    self.reset()
            self._run_progress = (semantic_state.act, semantic_state.floor)
        combat_anchor = (
            semantic_state.act,
            semantic_state.floor,
            semantic_state.encounter_id,
        )
        if semantic_state.in_combat:
            if self._combat_anchor is None:
                self._combat_anchor = combat_anchor
                self._current_turn_round = semantic_state.battle_round
                turn_started = semantic_state.is_player_turn
            elif self._combat_anchor != combat_anchor:
                self._reset_combat_progress()
                self._combat_anchor = combat_anchor
                self._current_turn_round = semantic_state.battle_round
                turn_started = semantic_state.is_player_turn
            elif semantic_state.is_player_turn and self._current_turn_round != semantic_state.battle_round:
                self._reset_turn_progress()
                self._current_turn_round = semantic_state.battle_round
                turn_started = True
        else:
            self._hand_order_signature = None
            self._hand_order_profile = HandOrderProfile()
        self._sync_hand_order_profile(semantic_state, turn_started=turn_started)
        return semantic_state

    def update_from_transition(
        self,
        previous_state: dict[str, object] | SemanticState,
        next_state: dict[str, object] | SemanticState,
        tool_name: str,
        response: dict[str, object],
        boundaries: dict[str, object],
        action_kwargs: dict[str, object] | None = None,
    ) -> None:
        previous = self.sync_state(previous_state)
        next_semantic = ensure_semantic_state(next_state)
        self._clear_transients()
        self.last_action_bucket = _action_bucket(tool_name, previous, action_kwargs)
        self.last_action_error = response.get("status") == "error"
        self.last_action_changed_state_type = previous.state_type != next_semantic.state_type
        self.last_turn_end = bool(boundaries.get("turn_end"))
        self.last_combat_end = bool(boundaries.get("combat_end"))
        self.last_act_end = bool(boundaries.get("act_end"))
        self.last_run_end = bool(boundaries.get("run_end"))
        self.last_enemy_hp_progress = max(0.0, previous.total_enemy_hp - next_semantic.total_enemy_hp)
        self.last_player_hp_loss = max(0.0, previous.player_hp - next_semantic.player_hp)

        if previous.in_combat:
            self.turn_step_count += 1
            self.combat_step_count += 1
            self.turn_enemy_damage += self.last_enemy_hp_progress
            self.turn_player_damage += self.last_player_hp_loss
            self.combat_enemy_damage += self.last_enemy_hp_progress
            self.combat_player_damage += self.last_player_hp_loss

        if self.last_action_error and previous.in_combat:
            self.turn_failed_actions += 1
            self.combat_failed_actions += 1

        if tool_name == "combat_play_card" and previous.in_combat and not self.last_action_error:
            card_index = _to_int((action_kwargs or {}).get("card_index", -1))
            played_card = next(
                (card for card in previous.hand if card.index == card_index),
                SemanticCard(index=card_index),
            )
            self.turn_cards_played += 1
            self.combat_cards_played += 1
            if played_card.card_type == "Attack":
                self.turn_attack_cards += 1
            elif played_card.card_type == "Skill":
                self.turn_skill_cards += 1
            elif played_card.card_type == "Power":
                self.turn_power_cards += 1
            self.turn_energy_spent += _cost_units(played_card.cost, previous.player_energy)

        if tool_name == "use_potion" and previous.in_combat and not self.last_action_error:
            self.turn_potions_used += 1
            self.combat_potions_used += 1

        if self.last_turn_end:
            self.combat_turn_ends += 1
            self.last_turn_skipped_unspent = _turn_skipped_unspent(previous, next_semantic, tool_name)
            if self.last_turn_skipped_unspent:
                self.combat_skip_unspent += 1
            self._reset_turn_progress()
            self._current_turn_round = next_semantic.battle_round if next_semantic.is_player_turn else None

        if self.last_combat_end:
            self._reset_combat_progress()

        if self.last_run_end:
            self._reset_combat_progress()
            self._awaiting_run_reset = True
            self._run_progress = None

        self.sync_state(next_semantic)


def _load_catalog_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_semantic_catalog(path: str | Path | None = None) -> SemanticCatalog:
    candidates = [Path(path)] if path is not None else _default_catalog_candidates()
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            payload = _load_catalog_json(candidate)
        except (OSError, json.JSONDecodeError):
            continue
        entities = payload.get("entities") if isinstance(payload.get("entities"), dict) else {}
        keys: list[str] = []
        seen: set[str] = set()

        def add(key: str) -> None:
            normalized = _normalize_model_token(key)
            if not normalized or normalized in seen:
                return
            seen.add(normalized)
            keys.append(normalized)

        for group in ENTITY_GROUPS:
            for item in _iter_dicts(entities.get(group)):
                add(str(item.get("model_id", "")))
                if group == "encounters":
                    encounter_model = str(item.get("model_id", "") or item.get("entry", ""))
                    encounter_token = encounter_model.split(".")[-1]
                    for round_bucket in COMBAT_ROUND_BUCKETS:
                        if encounter_token:
                            add(f"ENCOUNTER_ROUND.{encounter_token}.{round_bucket}")
                if group == "monsters":
                    monster_model = str(item.get("entry") or item.get("model_id", ""))
                    monster_token = monster_model.split(".")[-1]
                    for round_bucket in COMBAT_ROUND_BUCKETS:
                        if monster_token:
                            add(f"MONSTER_ROUND.{monster_token}.{round_bucket}")
                    for move_id in item.get("move_ids", []):
                        move_token = str(move_id)
                        if monster_token and move_token:
                            add(f"MONSTER_MOVE.{monster_token}.{move_token}")
                            for round_bucket in COMBAT_ROUND_BUCKETS:
                                add(f"MONSTER_MOVE_ROUND.{monster_token}.{move_token}.{round_bucket}")

        for category, values in RUNTIME_CONCEPTS.items():
            for value in values:
                add(f"{category}.{_slugify(value)}")

        for slot_index in range(SHOP_SLOT_COUNT):
            slot_token = f"SLOT_{slot_index}"
            for category in SHOP_SLOT_CATEGORIES:
                add(f"SHOP_SLOT_CATEGORY.{slot_token}.{category}")
            add(f"SHOP_SLOT_ITEM.{slot_token}.SPECIAL.CARD_REMOVAL")
            add(f"SHOP_SLOT_ITEM.{slot_token}.SPECIAL.UNKNOWN")
            for item in _iter_dicts(entities.get("cards")):
                model_id = str(item.get("model_id", ""))
                if model_id:
                    add(f"SHOP_SLOT_ITEM.{slot_token}.{model_id}")
            for item in _iter_dicts(entities.get("relics")):
                model_id = str(item.get("model_id", ""))
                if model_id:
                    add(f"SHOP_SLOT_ITEM.{slot_token}.{model_id}")
            for item in _iter_dicts(entities.get("potions")):
                model_id = str(item.get("model_id", ""))
                if model_id:
                    add(f"SHOP_SLOT_ITEM.{slot_token}.{model_id}")

        for group in ("events", "ancients"):
            for item in _iter_dicts(entities.get(group)):
                event_token = _normalize_model_token(item.get("model_id"))
                if not event_token:
                    continue
                event_token = event_token.split(".")[-1]
                for option_index in range(EVENT_OPTION_COUNT):
                    add(f"EVENT_OPTION_SLOT.{event_token}.OPTION_{option_index}")
        for option_index in range(EVENT_OPTION_COUNT):
            option_token = f"OPTION_{option_index}"
            for kind in EVENT_OPTION_KINDS:
                add(f"EVENT_OPTION_KIND.{option_token}.{kind}")
            add(f"EVENT_OPTION_RELIC.{option_token}.SPECIAL.UNKNOWN")
            add(f"EVENT_OPTION_CARD.{option_token}.SPECIAL.UNKNOWN")
            add(f"EVENT_OPTION_POTION.{option_token}.SPECIAL.UNKNOWN")
            for item in _iter_dicts(entities.get("cards")):
                model_id = _normalize_model_token(item.get("model_id"))
                if model_id:
                    add(f"EVENT_OPTION_CARD.{option_token}.{model_id.split('.')[-1]}")
            for item in _iter_dicts(entities.get("relics")):
                model_id = _normalize_model_token(item.get("model_id"))
                if model_id:
                    add(f"EVENT_OPTION_RELIC.{option_token}.{model_id.split('.')[-1]}")
            for item in _iter_dicts(entities.get("potions")):
                model_id = _normalize_model_token(item.get("model_id"))
                if model_id:
                    add(f"EVENT_OPTION_POTION.{option_token}.{model_id.split('.')[-1]}")

        for slot_index in range(HAND_SLOT_COUNT):
            slot_token = f"SLOT_{slot_index}"
            add(f"HAND_SLOT_CARD.{slot_token}.SPECIAL.UNKNOWN")
            for item in _iter_dicts(entities.get("cards")):
                model_id = _normalize_model_token(item.get("model_id"))
                if model_id:
                    add(f"HAND_SLOT_CARD.{slot_token}.{model_id.split('.')[-1]}")

        for slot_index in range(CARD_SELECT_SLOT_COUNT):
            slot_token = f"SLOT_{slot_index}"
            add(f"CARD_SELECT_SLOT_CARD.{slot_token}.SPECIAL.UNKNOWN")
            for operation in CARD_OPERATION_KINDS:
                add(f"CARD_SELECT_SLOT_OPERATION.{slot_token}.{operation}")
            for item in _iter_dicts(entities.get("cards")):
                model_id = _normalize_model_token(item.get("model_id"))
                if model_id:
                    add(f"CARD_SELECT_SLOT_CARD.{slot_token}.{model_id.split('.')[-1]}")

        for slot_index in range(HAND_SELECT_SLOT_COUNT):
            slot_token = f"SLOT_{slot_index}"
            add(f"HAND_SELECT_SELECTED.{slot_token}")
            add(f"HAND_SELECT_SLOT_CARD.{slot_token}.SPECIAL.UNKNOWN")
            for operation in CARD_OPERATION_KINDS:
                add(f"HAND_SELECT_SLOT_OPERATION.{slot_token}.{operation}")
            for item in _iter_dicts(entities.get("cards")):
                model_id = _normalize_model_token(item.get("model_id"))
                if model_id:
                    add(f"HAND_SELECT_SLOT_CARD.{slot_token}.{model_id.split('.')[-1]}")

        for node_type in MAP_NODE_TYPES:
            add(f"MAP_REMAINING_DOMINANT.{node_type}")
            add(f"MAP_REMAINING_HAS.{node_type}")
            for depth_label in MAP_ROUTE_DEPTH_LABELS:
                add(f"MAP_REMAINING_DEPTH.{depth_label}.{node_type}")
        for option_index in range(MAP_ROUTE_OPTION_COUNT):
            option_token = f"OPTION_{option_index}"
            for node_type in MAP_NODE_TYPES:
                add(f"MAP_OPTION_TYPE.{option_token}.{node_type}")
                add(f"MAP_ROUTE_DOMINANT.{option_token}.{node_type}")
                add(f"MAP_ROUTE_HAS.{option_token}.{node_type}")
                for depth_label in MAP_ROUTE_DEPTH_LABELS:
                    add(f"MAP_ROUTE_DEPTH.{option_token}.{depth_label}.{node_type}")

        concept_keys = tuple(keys)
        concept_index = {key: index for index, key in enumerate(concept_keys)}
        return SemanticCatalog(source_path=candidate, concept_keys=concept_keys, concept_index=concept_index)
    return SemanticCatalog.empty()


DEFAULT_SEMANTIC_CATALOG = load_semantic_catalog()
CONCEPT_VOCAB_SIZE = DEFAULT_SEMANTIC_CATALOG.size


def _normalize_power(raw: dict[str, object]) -> SemanticPower:
    power_type = str(raw.get("type", "") or "None")
    return SemanticPower(
        id=_normalize_model_token(raw.get("id")),
        power_type=power_type if power_type else "None",
        amount=_to_float(raw.get("amount")),
    )


def _normalize_enchantment(raw: dict[str, object]) -> SemanticEnchantment:
    status = str(raw.get("status", "") or "None")
    return SemanticEnchantment(
        id=_normalize_model_token(raw.get("id")),
        status=status if status else "None",
        amount=_to_float(raw.get("amount")),
    )


def _normalize_affliction(raw: dict[str, object]) -> SemanticAffliction:
    return SemanticAffliction(
        id=_normalize_model_token(raw.get("id")),
        amount=_to_float(raw.get("amount")),
    )


def _normalize_card(raw: dict[str, object]) -> SemanticCard:
    return SemanticCard(
        index=_to_int(raw.get("index", -1)),
        id=_normalize_model_token(raw.get("id")),
        name=str(raw.get("name", "")),
        card_type=str(raw.get("type", "") or "None"),
        rarity=str(raw.get("rarity", "") or "None"),
        cost=str(raw.get("cost", "")),
        target_type=str(raw.get("target_type", "") or "None"),
        can_play=bool(raw.get("can_play")),
        consumes_on_play=_raw_card_consumes_on_play(raw),
        is_upgraded=bool(raw.get("is_upgraded")),
        enchantments=tuple(_normalize_enchantment(item) for item in _list_at(raw, "enchantments")),
        afflictions=tuple(_normalize_affliction(item) for item in _list_at(raw, "afflictions")),
    )


def _normalize_relic(raw: dict[str, object]) -> SemanticRelic:
    return SemanticRelic(
        id=_normalize_model_token(raw.get("id")),
        rarity=str(raw.get("rarity", "") or "None"),
        status=str(raw.get("status", "") or "None"),
        counter=_to_float(raw.get("counter")),
        is_used_up=bool(raw.get("is_used_up")),
        is_wax=bool(raw.get("is_wax")),
        is_melted=bool(raw.get("is_melted")),
    )


def _normalize_potion(raw: dict[str, object]) -> SemanticPotion:
    return SemanticPotion(
        slot=_to_int(raw.get("slot", -1)),
        id=_normalize_model_token(raw.get("id")),
        rarity=str(raw.get("rarity", "") or "None"),
        usage=str(raw.get("usage", "") or "None"),
        target_type=str(raw.get("target_type", "") or "None"),
        can_use_in_combat=bool(raw.get("can_use_in_combat")),
    )


def _monster_entry_from_raw_enemy(raw: dict[str, object]) -> str:
    monster_id = _normalize_model_token(raw.get("monster_id"))
    if monster_id:
        return monster_id
    return _normalize_model_token(raw.get("name"))


def _normalize_enemy(raw: dict[str, object]) -> SemanticEnemy:
    intents = _list_at(raw, "intents")
    return SemanticEnemy(
        entity_id=str(raw.get("entity_id", "")),
        monster_id=_monster_entry_from_raw_enemy(raw),
        move_id=_normalize_model_token(raw.get("move_id")),
        hp=_to_float(raw.get("hp")),
        max_hp=_to_float(raw.get("max_hp")),
        block=_to_float(raw.get("block")),
        powers=tuple(_normalize_power(item) for item in _list_at(raw, "status")),
        intent_types=tuple(_slugify(intent.get("type")) or "NONE" for intent in intents),
        intent_damage=sum(_intent_damage_from_label(intent.get("label")) for intent in intents),
    )


def _normalize_reward_item(raw: dict[str, object]) -> SemanticRewardItem:
    return SemanticRewardItem(
        kind=str(raw.get("type", raw.get("kind", ""))),
        relic_id=_normalize_model_token(raw.get("relic_id")),
        potion_id=_normalize_model_token(raw.get("potion_id")),
        relic_rarity=str(raw.get("relic_rarity", "") or "None"),
        potion_rarity=str(raw.get("potion_rarity", "") or "None"),
        potion_usage=str(raw.get("potion_usage", "") or "None"),
    )


def _shop_item_token(raw: dict[str, object]) -> str:
    category = _slugify(raw.get("category"))
    if category == "CARD":
        token = _normalize_model_token(raw.get("card_id"))
        return f"CARD.{token}" if token else "CARD.UNKNOWN"
    if category == "RELIC":
        token = _normalize_model_token(raw.get("relic_id"))
        return f"RELIC.{token}" if token else "RELIC.UNKNOWN"
    if category == "POTION":
        token = _normalize_model_token(raw.get("potion_id"))
        return f"POTION.{token}" if token else "POTION.UNKNOWN"
    if category == "CARD_REMOVAL":
        return "SPECIAL.CARD_REMOVAL"
    if category:
        return f"SPECIAL.{category}"
    return "SPECIAL.UNKNOWN"


def _normalize_shop_item(raw: dict[str, object]) -> SemanticShopItem:
    return SemanticShopItem(
        index=_to_int(raw.get("index", -1)),
        category=str(raw.get("category", "")),
        can_afford=bool(raw.get("can_afford")),
        is_stocked=bool(raw.get("is_stocked", True)),
        price=_to_float(raw.get("price")),
        item_token=_shop_item_token(raw),
        card_id=_normalize_model_token(raw.get("card_id")),
        relic_id=_normalize_model_token(raw.get("relic_id")),
        potion_id=_normalize_model_token(raw.get("potion_id")),
        card_type=str(raw.get("card_type", "") or "None"),
        card_rarity=str(raw.get("card_rarity", "") or "None"),
        relic_rarity=str(raw.get("relic_rarity", "") or "None"),
        potion_rarity=str(raw.get("potion_rarity", "") or "None"),
        potion_usage=str(raw.get("potion_usage", "") or "None"),
    )


def _normalize_option(raw: dict[str, object]) -> SemanticOption:
    return SemanticOption(
        index=_to_int(raw.get("index", -1)),
        is_locked=bool(raw.get("is_locked")),
        is_proceed=bool(raw.get("is_proceed")),
        is_enabled=bool(raw.get("is_enabled", True)),
        was_chosen=bool(raw.get("was_chosen")),
        relic_id=_normalize_model_token(raw.get("relic_id")),
        card_id=_normalize_model_token(raw.get("card_id")),
        potion_id=_normalize_model_token(raw.get("potion_id")),
        option_type=_normalize_option_type(raw.get("type", raw.get("option_type", ""))),
        title=_clean_text(raw.get("title") or raw.get("name") or raw.get("label")),
        description=_clean_text(raw.get("description") or raw.get("body")),
        keyword_tokens=_normalize_keyword_tokens(raw.get("keywords")),
    )


def event_option_kind(option: SemanticOption | None) -> str:
    if option is None or option.index < 0:
        return "NONE"
    if option.is_proceed:
        return "PROCEED"
    if option.relic_id:
        return "RELIC"
    if option.card_id:
        return "CARD"
    if option.potion_id:
        return "POTION"
    if option.keyword_tokens:
        return "KEYWORD"
    if option.title or option.description:
        return "TEXT"
    return "UNKNOWN"


@dataclass(frozen=True)
class _MapNodeRecord:
    col: int
    row: int
    node_type: str
    children: tuple[tuple[int, int], ...]


def _map_current_row(map_state: dict[str, object]) -> int:
    current_position = _dict_at(map_state, "current_position")
    if "row" in current_position:
        return _to_int(current_position.get("row"))
    visited = _list_at(map_state, "visited")
    if visited:
        return max(_to_int(item.get("row")) for item in visited)
    return 0


def _build_map_node_lookup(map_state: dict[str, object]) -> tuple[dict[tuple[int, int], _MapNodeRecord], tuple[int, int] | None]:
    node_lookup: dict[tuple[int, int], _MapNodeRecord] = {}
    boss_key = _coordinate_pair(_dict_at(map_state, "boss"))

    for raw_node in _list_at(map_state, "nodes"):
        key = _coordinate_pair(raw_node)
        if key is None:
            continue
        children = tuple(
            child
            for child in (_coordinate_pair(item) for item in raw_node.get("children", []) if isinstance(raw_node.get("children"), list))
            if child is not None
        )
        node_type = _normalize_map_node_type(raw_node.get("type"))
        if boss_key is not None and key == boss_key:
            node_type = "Boss"
        node_lookup[key] = _MapNodeRecord(key[0], key[1], node_type, children)

    for raw_option in _list_at(map_state, "next_options"):
        key = _coordinate_pair(raw_option)
        if key is None:
            continue
        option_children = tuple(
            child
            for child in (
                _coordinate_pair(item)
                for item in raw_option.get("leads_to", [])
                if isinstance(raw_option.get("leads_to"), list)
            )
            if child is not None
        )
        option_type = _normalize_map_node_type(raw_option.get("type"))
        existing = node_lookup.get(key)
        if existing is None:
            node_lookup[key] = _MapNodeRecord(key[0], key[1], option_type, option_children)
        else:
            merged_type = existing.node_type if existing.node_type != "Unknown" else option_type
            merged_children = existing.children or option_children
            node_lookup[key] = _MapNodeRecord(key[0], key[1], merged_type, merged_children)

    if boss_key is not None and boss_key not in node_lookup:
        node_lookup[boss_key] = _MapNodeRecord(boss_key[0], boss_key[1], "Boss", ())

    return node_lookup, boss_key


def _count_map_paths(
    start_key: tuple[int, int],
    node_lookup: dict[tuple[int, int], _MapNodeRecord],
    memo: dict[tuple[int, int], int],
) -> int:
    cached = memo.get(start_key)
    if cached is not None:
        return cached
    record = node_lookup.get(start_key)
    if record is None:
        memo[start_key] = 0
        return 0
    children = tuple(child for child in record.children if child in node_lookup)
    if not children:
        memo[start_key] = 1
        return 1
    total = 0
    for child in children:
        total += _count_map_paths(child, node_lookup, memo)
        if total >= 999:
            total = 999
            break
    memo[start_key] = total
    return total


def _map_shortest_depths(
    start_keys: tuple[tuple[int, int], ...],
    node_lookup: dict[tuple[int, int], _MapNodeRecord],
) -> dict[tuple[int, int], int]:
    pending = [(key, 0) for key in start_keys if key in node_lookup]
    depths: dict[tuple[int, int], int] = {}
    while pending:
        key, depth = pending.pop(0)
        if key in depths and depth >= depths[key]:
            continue
        depths[key] = depth
        record = node_lookup.get(key)
        if record is None:
            continue
        for child in record.children:
            if child not in node_lookup:
                continue
            pending.append((child, depth + 1))
    return depths


def _depth_dominant_types(depth_type_counts: tuple[tuple[int, ...], ...]) -> tuple[str, ...]:
    return tuple(_dominant_map_node_type(type_counts) for type_counts in depth_type_counts)


def _route_node_weight(node_type: str, depth: int) -> float:
    weight = MAP_ROUTE_SCORE_WEIGHTS.get(node_type, 0.0)
    if node_type == "Monster":
        if depth >= 3:
            weight -= 0.45
        elif depth >= 2:
            weight -= 0.25
        elif depth == 1:
            weight -= 0.10
    elif node_type == "Elite":
        if depth == 0:
            weight -= 0.40
        elif depth == 1:
            weight -= 0.20
    elif node_type == "RestSite" and depth <= 1:
        weight += 0.10
    elif node_type == "Shop" and depth <= 1:
        weight += 0.05
    elif node_type == "Event" and depth <= 2:
        weight += 0.05
    return weight


def _route_score_from_depths(
    depths: dict[tuple[int, int], int],
    node_lookup: dict[tuple[int, int], _MapNodeRecord],
    path_count: int,
    boss_distance: int,
) -> float:
    score = 0.0
    for key, depth in depths.items():
        node_type = node_lookup.get(key, _MapNodeRecord(0, 0, "Unknown", ())).node_type
        score += _route_node_weight(node_type, depth) / (1.0 + (depth * 0.45))
    score += min(1.0, max(0, path_count - 1) * 0.04)
    if boss_distance >= 0:
        score += 0.60 / (1.0 + boss_distance)
    return score


def _summarize_map_route(
    raw_option: dict[str, object],
    node_lookup: dict[tuple[int, int], _MapNodeRecord],
    boss_key: tuple[int, int] | None,
) -> SemanticMapRoute:
    option_index = _to_int(raw_option.get("index", -1))
    start_key = _coordinate_pair(raw_option)
    option_type = _normalize_map_node_type(raw_option.get("type"))
    if start_key is None or start_key not in node_lookup:
        return SemanticMapRoute(
            option_index=option_index,
            option_type=option_type,
            type_counts=_empty_map_type_counts(),
            depth_type_counts=_empty_map_depth_type_counts(),
            depth_dominant_types=tuple("Unknown" for _ in MAP_ROUTE_DEPTH_LABELS),
        )
    depths = _map_shortest_depths((start_key,), node_lookup)
    type_counts = [0 for _ in MAP_NODE_TYPES]
    depth_type_counts = [[0 for _ in MAP_NODE_TYPES] for _ in MAP_ROUTE_DEPTH_LABELS]
    max_depth = 0
    for key, depth in depths.items():
        record = node_lookup[key]
        type_index = MAP_NODE_TYPE_INDEX.get(record.node_type, MAP_NODE_TYPE_INDEX["Unknown"])
        type_counts[type_index] += 1
        if depth <= MAP_ROUTE_DEPTH_LIMIT:
            depth_type_counts[depth][type_index] += 1
        if depth > max_depth:
            max_depth = depth
    boss_distance = depths.get(
        boss_key,
        min((depth for key, depth in depths.items() if node_lookup[key].node_type == "Boss"), default=-1),
    )
    path_count = _count_map_paths(start_key, node_lookup, {})
    frozen_type_counts = tuple(type_counts)
    frozen_depth_counts = tuple(tuple(counts) for counts in depth_type_counts)
    return SemanticMapRoute(
        option_index=option_index,
        col=start_key[0],
        row=start_key[1],
        option_type=node_lookup[start_key].node_type if node_lookup[start_key].node_type != "Unknown" else option_type,
        reachable_count=len(depths),
        path_count=path_count,
        max_depth=max_depth,
        boss_distance=boss_distance,
        route_score=_route_score_from_depths(depths, node_lookup, path_count, boss_distance),
        dominant_type=_dominant_map_node_type(frozen_type_counts),
        depth_dominant_types=_depth_dominant_types(frozen_depth_counts),
        type_counts=frozen_type_counts,
        depth_type_counts=frozen_depth_counts,
    )


def _summarize_map_overview(
    next_options: list[dict[str, object]],
    node_lookup: dict[tuple[int, int], _MapNodeRecord],
    boss_key: tuple[int, int] | None,
) -> tuple[int, int, str, tuple[int, ...], tuple[tuple[int, ...], ...]]:
    start_keys = tuple(
        key
        for key in (_coordinate_pair(option) for option in next_options)
        if key is not None and key in node_lookup
    )
    if not start_keys:
        return 0, -1, "Unknown", _empty_map_type_counts(), _empty_map_depth_type_counts()
    depths = _map_shortest_depths(start_keys, node_lookup)
    type_counts = [0 for _ in MAP_NODE_TYPES]
    depth_type_counts = [[0 for _ in MAP_NODE_TYPES] for _ in MAP_ROUTE_DEPTH_LABELS]
    for key, depth in depths.items():
        record = node_lookup[key]
        type_index = MAP_NODE_TYPE_INDEX.get(record.node_type, MAP_NODE_TYPE_INDEX["Unknown"])
        type_counts[type_index] += 1
        if depth <= MAP_ROUTE_DEPTH_LIMIT:
            depth_type_counts[depth][type_index] += 1
    frozen_type_counts = tuple(type_counts)
    frozen_depth_counts = tuple(tuple(counts) for counts in depth_type_counts)
    boss_distance = depths.get(
        boss_key,
        min((depth for key, depth in depths.items() if node_lookup[key].node_type == "Boss"), default=-1),
    )
    return len(depths), boss_distance, _dominant_map_node_type(frozen_type_counts), frozen_type_counts, frozen_depth_counts


def _flatten_bundle_cards(bundle_select: dict[str, object]) -> tuple[SemanticCard, ...]:
    cards: list[SemanticCard] = []
    for bundle in _list_at(bundle_select, "bundles"):
        cards.extend(_normalize_card(card) for card in _list_at(bundle, "cards"))
    return tuple(cards)


def normalize_semantic_state(state: dict[str, object]) -> SemanticState:
    state_type = str(state.get("state_type", "unknown"))
    run = _dict_at(state, "run")
    player = _dict_at(state, "player")
    battle = _dict_at(state, "battle")
    rewards = _dict_at(state, "rewards")
    card_reward = _dict_at(state, "card_reward")
    map_state = _dict_at(state, "map")
    event_state = _dict_at(state, "event")
    rest_state = _dict_at(state, "rest_site")
    card_select = _dict_at(state, "card_select")
    bundle_select = _dict_at(state, "bundle_select")
    hand_select = _dict_at(state, "hand_select")
    relic_select = _dict_at(state, "relic_select")
    treasure = _dict_at(state, "treasure")
    shop_state = _dict_at(state, "shop")
    fake_merchant = _dict_at(state, "fake_merchant")
    fake_shop_state = _dict_at(fake_merchant, "shop")
    crystal = _dict_at(state, "crystal_sphere")

    draw_pile = tuple(_normalize_card(card) for card in _list_at(player, "draw_pile"))
    discard_pile = tuple(_normalize_card(card) for card in _list_at(player, "discard_pile"))
    exhaust_pile = tuple(_normalize_card(card) for card in _list_at(player, "exhaust_pile"))
    event_id = _normalize_model_token(event_state.get("event_id") or fake_merchant.get("event_id"))
    event_name = _clean_text(event_state.get("event_name") or fake_merchant.get("event_name"))
    raw_map_options = _list_at(map_state, "next_options")
    map_node_lookup, boss_key = _build_map_node_lookup(map_state)
    map_route_summaries = tuple(_summarize_map_route(option, map_node_lookup, boss_key) for option in raw_map_options)
    card_select_ops = _selection_operation_flags(card_select)
    hand_select_ops = _selection_operation_flags(hand_select)
    (
        map_remaining_node_count,
        map_remaining_boss_distance,
        map_remaining_dominant_type,
        map_remaining_type_counts,
        map_remaining_depth_type_counts,
    ) = _summarize_map_overview(raw_map_options, map_node_lookup, boss_key)

    return SemanticState(
        state_type=state_type,
        act=_to_int(run.get("act")),
        act_id=_normalize_model_token(run.get("act_id")),
        floor=_to_int(run.get("floor")),
        ascension=_to_int(run.get("ascension")),
        character_id=_normalize_model_token(player.get("character_id")),
        player_hp=_to_float(player.get("hp")),
        player_max_hp=_to_float(player.get("max_hp")),
        player_block=_to_float(player.get("block")),
        player_energy=_to_float(player.get("energy")),
        player_max_energy=_to_float(player.get("max_energy")),
        player_gold=_to_float(player.get("gold")),
        player_powers=tuple(_normalize_power(item) for item in _list_at(player, "status")),
        relics=tuple(_normalize_relic(item) for item in _list_at(player, "relics")),
        potions=tuple(_normalize_potion(item) for item in _list_at(player, "potions")),
        orb_ids=tuple(_normalize_model_token(orb.get("id")) for orb in _list_at(player, "orbs") if _normalize_model_token(orb.get("id"))),
        hand=tuple(_normalize_card(card) for card in _list_at(player, "hand")),
        draw_pile=draw_pile,
        discard_pile=discard_pile,
        exhaust_pile=exhaust_pile,
        draw_pile_count=_to_int(player.get("draw_pile_count", len(draw_pile))),
        discard_pile_count=_to_int(player.get("discard_pile_count", len(discard_pile))),
        exhaust_pile_count=_to_int(player.get("exhaust_pile_count", len(exhaust_pile))),
        encounter_id=_normalize_model_token(battle.get("encounter_id")),
        battle_round=_to_int(battle.get("round")),
        battle_turn=str(battle.get("turn", "")),
        is_play_phase=bool(battle.get("is_play_phase")),
        player_actions_disabled=bool(battle.get("player_actions_disabled")),
        hand_in_card_play=bool(battle.get("hand_in_card_play")),
        enemies=tuple(_normalize_enemy(enemy) for enemy in _list_at(battle, "enemies")),
        reward_items=tuple(_normalize_reward_item(item) for item in _list_at(rewards, "items")),
        rewards_can_proceed=bool(rewards.get("can_proceed")),
        card_reward_cards=tuple(_normalize_card(card) for card in _list_at(card_reward, "cards")),
        card_reward_can_skip=bool(card_reward.get("can_skip")),
        map_options=tuple(_normalize_option(option) for option in raw_map_options),
        map_current_row=_map_current_row(map_state),
        map_remaining_node_count=map_remaining_node_count,
        map_remaining_boss_distance=map_remaining_boss_distance,
        map_remaining_dominant_type=map_remaining_dominant_type,
        map_remaining_type_counts=map_remaining_type_counts,
        map_remaining_depth_type_counts=map_remaining_depth_type_counts,
        map_route_summaries=map_route_summaries,
        event_id=event_id,
        event_name=event_name,
        event_in_dialogue=bool(event_state.get("in_dialogue")),
        event_options=tuple(_normalize_option(option) for option in _list_at(event_state, "options")),
        rest_options=tuple(_normalize_option(option) for option in _list_at(rest_state, "options")),
        rest_can_proceed=bool(rest_state.get("can_proceed")),
        card_select_cards=tuple(_normalize_card(card) for card in _list_at(card_select, "cards")),
        card_select_preview_cards=tuple(_normalize_card(card) for card in _list_at(card_select, "preview_cards")),
        card_select_is_consume=card_select_ops["consume"],
        card_select_is_upgrade=card_select_ops["upgrade"],
        card_select_is_remove=card_select_ops["remove"],
        card_select_is_enchant=card_select_ops["enchant"],
        card_select_can_confirm=bool(card_select.get("can_confirm")),
        card_select_can_cancel=bool(card_select.get("can_cancel") or card_select.get("can_skip")),
        bundle_cards=_flatten_bundle_cards(bundle_select),
        bundle_preview_cards=tuple(_normalize_card(card) for card in _list_at(bundle_select, "preview_cards")),
        bundle_count=len(_list_at(bundle_select, "bundles")),
        bundle_can_confirm=bool(bundle_select.get("can_confirm")),
        bundle_can_cancel=bool(bundle_select.get("can_cancel")),
        hand_select_cards=tuple(_normalize_card(card) for card in _list_at(hand_select, "cards")),
        hand_select_selected_cards=tuple(_normalize_card(card) for card in _list_at(hand_select, "selected_cards")),
        hand_select_is_consume=hand_select_ops["consume"],
        hand_select_is_upgrade=hand_select_ops["upgrade"],
        hand_select_is_remove=hand_select_ops["remove"],
        hand_select_is_enchant=hand_select_ops["enchant"],
        hand_select_can_confirm=bool(hand_select.get("can_confirm")),
        relic_select_relics=tuple(_normalize_relic(relic) for relic in _list_at(relic_select, "relics")),
        relic_select_can_skip=bool(relic_select.get("can_skip")),
        treasure_relics=tuple(_normalize_relic(relic) for relic in _list_at(treasure, "relics")),
        treasure_can_proceed=bool(treasure.get("can_proceed")),
        shop_items=tuple(_normalize_shop_item(item) for item in _list_at(shop_state, "items")),
        shop_can_proceed=bool(shop_state.get("can_proceed")),
        fake_merchant_shop_items=tuple(_normalize_shop_item(item) for item in _list_at(fake_shop_state, "items")),
        fake_merchant_can_proceed=bool(fake_shop_state.get("can_proceed")),
        crystal_clickable_count=len(_list_at(crystal, "clickable_cells")),
        crystal_can_proceed=bool(crystal.get("can_proceed")),
    )


def ensure_semantic_state(state: dict[str, object] | SemanticState) -> SemanticState:
    if isinstance(state, SemanticState):
        return state
    return normalize_semantic_state(state)


def _candidate_concept_keys(value: object, prefixes: tuple[str, ...]) -> list[str]:
    token = _normalize_model_token(value)
    if not token:
        return []
    if "." in token:
        candidates = [token]
        tail = token.split(".")[-1]
        candidates.extend(f"{prefix}.{tail}" for prefix in prefixes if prefix)
        return candidates
    return [f"{prefix}.{token}" for prefix in prefixes if prefix]


def _activate_key(active_keys: set[str], catalog: SemanticCatalog, key: str) -> None:
    normalized = _normalize_model_token(key)
    if normalized in catalog.concept_index:
        active_keys.add(normalized)


def _activate_prefixed(active_keys: set[str], catalog: SemanticCatalog, value: object, *prefixes: str) -> None:
    for candidate in _candidate_concept_keys(value, tuple(prefixes)):
        _activate_key(active_keys, catalog, candidate)


def _activate_runtime(active_keys: set[str], catalog: SemanticCatalog, category: str, value: object) -> None:
    token = _slugify(value) or "NONE"
    _activate_key(active_keys, catalog, f"{category}.{token}")


def activate_state_concepts(
    state: dict[str, object] | SemanticState,
    catalog: SemanticCatalog | None = None,
) -> ConceptActivation:
    semantic_state = ensure_semantic_state(state)
    active_keys: set[str] = set()
    catalog = catalog or DEFAULT_SEMANTIC_CATALOG
    if catalog.size == 0:
        return ConceptActivation(active_keys=(), vector=[])

    _activate_runtime(active_keys, catalog, "STATE_TYPE", semantic_state.state_type)
    _activate_runtime(active_keys, catalog, "COMBAT_ROUND_BUCKET", combat_round_bucket(semantic_state.battle_round))
    _activate_prefixed(active_keys, catalog, semantic_state.act_id, "ACT")
    _activate_prefixed(active_keys, catalog, semantic_state.character_id, "CHARACTER")
    _activate_prefixed(active_keys, catalog, semantic_state.encounter_id, "ENCOUNTER")
    _activate_prefixed(active_keys, catalog, semantic_state.event_id, "EVENT", "ANCIENT")
    encounter_token = semantic_state.encounter_id.split(".")[-1] if semantic_state.encounter_id else ""
    round_bucket = combat_round_bucket(semantic_state.battle_round)
    if encounter_token:
        _activate_key(active_keys, catalog, f"ENCOUNTER_ROUND.{encounter_token}.{round_bucket}")

    for power in semantic_state.player_powers:
        _activate_prefixed(active_keys, catalog, power.id, "POWER")
        _activate_runtime(active_keys, catalog, "POWER_TYPE", power.power_type)
    for relic in semantic_state.relics:
        _activate_prefixed(active_keys, catalog, relic.id, "RELIC")
        _activate_runtime(active_keys, catalog, "RELIC_RARITY", relic.rarity)
        _activate_runtime(active_keys, catalog, "RELIC_STATUS", relic.status)
    for potion in semantic_state.potions:
        _activate_prefixed(active_keys, catalog, potion.id, "POTION")
        _activate_runtime(active_keys, catalog, "POTION_RARITY", potion.rarity)
        _activate_runtime(active_keys, catalog, "POTION_USAGE", potion.usage)
        _activate_runtime(active_keys, catalog, "TARGET_TYPE", potion.target_type)
    for orb_id in semantic_state.orb_ids:
        _activate_prefixed(active_keys, catalog, orb_id, "ORB")

    for card in semantic_state.hand:
        if card.index < 0 or card.index >= HAND_SLOT_COUNT:
            continue
        slot_token = f"SLOT_{card.index}"
        _activate_prefixed(active_keys, catalog, card.id, f"HAND_SLOT_CARD.{slot_token}")

    card_select_operations = {
        "CONSUME": semantic_state.card_select_is_consume,
        "UPGRADE": semantic_state.card_select_is_upgrade,
        "REMOVE": semantic_state.card_select_is_remove,
        "ENCHANT": semantic_state.card_select_is_enchant,
    }
    for card in semantic_state.card_select_cards:
        if card.index < 0 or card.index >= CARD_SELECT_SLOT_COUNT:
            continue
        slot_token = f"SLOT_{card.index}"
        _activate_prefixed(active_keys, catalog, card.id, f"CARD_SELECT_SLOT_CARD.{slot_token}")
        for operation, enabled in card_select_operations.items():
            if enabled:
                _activate_key(active_keys, catalog, f"CARD_SELECT_SLOT_OPERATION.{slot_token}.{operation}")

    for card_group in (
        semantic_state.hand,
        semantic_state.draw_pile,
        semantic_state.discard_pile,
        semantic_state.exhaust_pile,
        semantic_state.card_reward_cards,
        semantic_state.card_select_cards,
        semantic_state.card_select_preview_cards,
        semantic_state.bundle_cards,
        semantic_state.bundle_preview_cards,
        semantic_state.hand_select_cards,
        semantic_state.hand_select_selected_cards,
    ):
        for card in card_group:
            _activate_prefixed(active_keys, catalog, card.id, "CARD")
            _activate_runtime(active_keys, catalog, "CARD_TYPE", card.card_type)
            _activate_runtime(active_keys, catalog, "CARD_RARITY", card.rarity)
            _activate_runtime(active_keys, catalog, "TARGET_TYPE", card.target_type)
            for enchantment in card.enchantments:
                _activate_prefixed(active_keys, catalog, enchantment.id, "ENCHANTMENT")
                _activate_runtime(active_keys, catalog, "ENCHANTMENT_STATUS", enchantment.status)
            for affliction in card.afflictions:
                _activate_prefixed(active_keys, catalog, affliction.id, "AFFLICTION")

    for enemy in semantic_state.enemies:
        monster_token = enemy.monster_id.split(".")[-1] if enemy.monster_id else ""
        _activate_prefixed(active_keys, catalog, enemy.monster_id, "MONSTER")
        _activate_prefixed(active_keys, catalog, f"MONSTER_MOVE.{monster_token}.{enemy.move_id}", "MONSTER_MOVE")
        if monster_token:
            _activate_key(active_keys, catalog, f"MONSTER_ROUND.{monster_token}.{round_bucket}")
            if enemy.move_id:
                _activate_key(active_keys, catalog, f"MONSTER_MOVE_ROUND.{monster_token}.{enemy.move_id}.{round_bucket}")
        for intent_type in enemy.intent_types:
            _activate_runtime(active_keys, catalog, "INTENT_TYPE", intent_type)
        for power in enemy.powers:
            _activate_prefixed(active_keys, catalog, power.id, "POWER")
            _activate_runtime(active_keys, catalog, "POWER_TYPE", power.power_type)

    for reward_item in semantic_state.reward_items:
        _activate_prefixed(active_keys, catalog, reward_item.relic_id, "RELIC")
        _activate_prefixed(active_keys, catalog, reward_item.potion_id, "POTION")
        _activate_runtime(active_keys, catalog, "RELIC_RARITY", reward_item.relic_rarity)
        _activate_runtime(active_keys, catalog, "POTION_RARITY", reward_item.potion_rarity)
        _activate_runtime(active_keys, catalog, "POTION_USAGE", reward_item.potion_usage)

    for option in semantic_state.event_options:
        if option.index < 0 or option.index >= EVENT_OPTION_COUNT:
            continue
        option_token = f"OPTION_{option.index}"
        if semantic_state.event_id:
            event_token = semantic_state.event_id.split(".")[-1]
            _activate_key(active_keys, catalog, f"EVENT_OPTION_SLOT.{event_token}.{option_token}")
        _activate_key(active_keys, catalog, f"EVENT_OPTION_KIND.{option_token}.{event_option_kind(option)}")
        _activate_prefixed(active_keys, catalog, option.relic_id, f"EVENT_OPTION_RELIC.{option_token}")
        _activate_prefixed(active_keys, catalog, option.card_id, f"EVENT_OPTION_CARD.{option_token}")
        _activate_prefixed(active_keys, catalog, option.potion_id, f"EVENT_OPTION_POTION.{option_token}")

    selected_hand_indices = {card.index for card in semantic_state.hand_select_selected_cards if card.index >= 0}
    selected_hand_ids = {card.id for card in semantic_state.hand_select_selected_cards if card.id}
    hand_select_operations = {
        "CONSUME": semantic_state.hand_select_is_consume,
        "UPGRADE": semantic_state.hand_select_is_upgrade,
        "REMOVE": semantic_state.hand_select_is_remove,
        "ENCHANT": semantic_state.hand_select_is_enchant,
    }
    for card in semantic_state.hand_select_cards:
        if card.index < 0 or card.index >= HAND_SELECT_SLOT_COUNT:
            continue
        slot_token = f"SLOT_{card.index}"
        _activate_prefixed(active_keys, catalog, card.id, f"HAND_SELECT_SLOT_CARD.{slot_token}")
        for operation, enabled in hand_select_operations.items():
            if enabled:
                _activate_key(active_keys, catalog, f"HAND_SELECT_SLOT_OPERATION.{slot_token}.{operation}")
        if card.index in selected_hand_indices or card.id in selected_hand_ids:
            _activate_key(active_keys, catalog, f"HAND_SELECT_SELECTED.{slot_token}")

    for shop_group in (semantic_state.shop_items, semantic_state.fake_merchant_shop_items):
        for item in shop_group:
            _activate_prefixed(active_keys, catalog, item.card_id, "CARD")
            _activate_prefixed(active_keys, catalog, item.relic_id, "RELIC")
            _activate_prefixed(active_keys, catalog, item.potion_id, "POTION")
            _activate_runtime(active_keys, catalog, "CARD_TYPE", item.card_type)
            _activate_runtime(active_keys, catalog, "CARD_RARITY", item.card_rarity)
            _activate_runtime(active_keys, catalog, "RELIC_RARITY", item.relic_rarity)
            _activate_runtime(active_keys, catalog, "POTION_RARITY", item.potion_rarity)
            _activate_runtime(active_keys, catalog, "POTION_USAGE", item.potion_usage)
            if 0 <= item.index < SHOP_SLOT_COUNT:
                slot_token = f"SLOT_{item.index}"
                _activate_key(active_keys, catalog, f"SHOP_SLOT_CATEGORY.{slot_token}.{_slugify(item.category) or 'UNKNOWN'}")
                _activate_key(active_keys, catalog, f"SHOP_SLOT_ITEM.{slot_token}.{item.item_token or 'SPECIAL.UNKNOWN'}")

    if semantic_state.map_remaining_dominant_type:
        _activate_key(active_keys, catalog, f"MAP_REMAINING_DOMINANT.{semantic_state.map_remaining_dominant_type}")
    for node_type in MAP_NODE_TYPES:
        if map_type_count(semantic_state.map_remaining_type_counts, node_type) > 0:
            _activate_key(active_keys, catalog, f"MAP_REMAINING_HAS.{node_type}")
    for depth_index, depth_label in enumerate(MAP_ROUTE_DEPTH_LABELS):
        dominant_type = map_depth_dominant_type(semantic_state.map_remaining_depth_type_counts, depth_index)
        _activate_key(active_keys, catalog, f"MAP_REMAINING_DEPTH.{depth_label}.{dominant_type}")

    for route in semantic_state.map_route_summaries:
        if route.option_index < 0 or route.option_index >= MAP_ROUTE_OPTION_COUNT:
            continue
        option_token = f"OPTION_{route.option_index}"
        _activate_key(active_keys, catalog, f"MAP_OPTION_TYPE.{option_token}.{route.option_type}")
        _activate_key(active_keys, catalog, f"MAP_ROUTE_DOMINANT.{option_token}.{route.dominant_type}")
        for node_type in MAP_NODE_TYPES:
            if map_type_count(route.type_counts, node_type) > 0:
                _activate_key(active_keys, catalog, f"MAP_ROUTE_HAS.{option_token}.{node_type}")
        for depth_label, dominant_type in zip(MAP_ROUTE_DEPTH_LABELS, route.depth_dominant_types):
            _activate_key(active_keys, catalog, f"MAP_ROUTE_DEPTH.{option_token}.{depth_label}.{dominant_type}")

    for option_group in (semantic_state.map_options, semantic_state.event_options, semantic_state.rest_options):
        for option in option_group:
            _activate_prefixed(active_keys, catalog, option.relic_id, "RELIC")

    for relic in semantic_state.relic_select_relics + semantic_state.treasure_relics:
        _activate_prefixed(active_keys, catalog, relic.id, "RELIC")
        _activate_runtime(active_keys, catalog, "RELIC_RARITY", relic.rarity)

    ordered_keys = tuple(sorted(active_keys, key=catalog.concept_index.get))
    vector = [0.0] * catalog.size
    for key in ordered_keys:
        vector[catalog.concept_index[key]] = 1.0
    return ConceptActivation(active_keys=ordered_keys, vector=vector)


def _count_cards_by_type(cards: tuple[SemanticCard, ...], card_type: str, playable_only: bool = False) -> int:
    return sum(1 for card in cards if card.card_type == card_type and (card.can_play or not playable_only))


def _count_cards_by_target(cards: tuple[SemanticCard, ...], predicate: set[str], playable_only: bool = False) -> int:
    return sum(
        1
        for card in cards
        if _match_token(card.target_type, predicate) and (card.can_play or not playable_only)
    )


def _visible_shop_items(state: SemanticState) -> tuple[SemanticShopItem, ...]:
    if state.state_type == "fake_merchant":
        return state.fake_merchant_shop_items
    return state.shop_items


def _build_scalar_features(state: SemanticState) -> _FeatureBuilder:
    builder = _FeatureBuilder()
    playable_cards = state.playable_cards
    playable_costs = [_cost_units(card.cost, state.player_energy) for card in playable_cards]
    living_enemy_count = len(state.living_enemies)
    affordable_shop_count = sum(1 for item in state.shop_items if item.can_afford and item.is_stocked)
    affordable_fake_shop_count = sum(1 for item in state.fake_merchant_shop_items if item.can_afford and item.is_stocked)
    positive_player_power = sum(max(0.0, power.amount) for power in state.player_powers)
    negative_player_power = sum(max(0.0, -power.amount) for power in state.player_powers)
    enemy_power_count = sum(len(enemy.powers) for enemy in state.living_enemies)
    enemy_block_total = sum(enemy.block for enemy in state.living_enemies)
    enemy_attack_intents = sum(1 for enemy in state.living_enemies for intent in enemy.intent_types if intent.startswith("ATTACK"))
    enemy_buff_intents = sum(1 for enemy in state.living_enemies for intent in enemy.intent_types if intent == "BUFF")
    enemy_debuff_intents = sum(1 for enemy in state.living_enemies for intent in enemy.intent_types if "DEBUFF" in intent)
    map_special_count = sum(1 for option in state.map_options if option.option_type in {"RestSite", "Shop", "Event", "Treasure", "Boss"})
    map_route_scores = [route.route_score for route in state.map_route_summaries]
    map_best_route_score = max(map_route_scores, default=0.0)
    map_mean_route_score = sum(map_route_scores) / max(len(map_route_scores), 1)
    map_path_count_total = sum(route.path_count for route in state.map_route_summaries)
    visible_shop_items = _visible_shop_items(state)
    event_unlocked_count = sum(1 for option in state.event_options if not option.is_locked)
    rest_enabled_count = sum(1 for option in state.rest_options if option.is_enabled)

    builder.add("player_hp_ratio", _safe_div(state.player_hp, max(state.player_max_hp, 1.0)))
    builder.add("player_block_norm", _ratio(state.player_block, 100.0, 2.0))
    builder.add("player_energy_norm", _ratio(state.player_energy, 5.0, 2.0))
    builder.add("player_energy_ratio", _safe_div(state.player_energy, max(state.player_max_energy, 1.0)))
    builder.add("player_gold_norm", _ratio(state.player_gold, 500.0, 2.0))
    builder.add("player_power_count_norm", _ratio(len(state.player_powers), 16.0, 2.0))
    builder.add("player_positive_power_amount_norm", _ratio(positive_player_power, 30.0, 2.0))
    builder.add("player_negative_power_amount_norm", _ratio(negative_player_power, 30.0, 2.0))
    builder.add("relic_count_norm", _ratio(len(state.relics), 40.0, 2.0))
    builder.add("relic_counter_sum_norm", _ratio(sum(max(0.0, relic.counter) for relic in state.relics), 30.0, 2.0))
    builder.add("potion_count_norm", _ratio(len(state.potions), 5.0, 2.0))
    builder.add("orb_count_norm", _ratio(len(state.orb_ids), 10.0, 2.0))
    builder.add("hand_count_norm", _ratio(len(state.hand), 10.0, 2.0))
    builder.add("playable_card_count_norm", _ratio(len(playable_cards), 10.0, 2.0))
    builder.add("attack_hand_count_norm", _ratio(_count_cards_by_type(state.hand, "Attack"), 10.0, 2.0))
    builder.add("skill_hand_count_norm", _ratio(_count_cards_by_type(state.hand, "Skill"), 10.0, 2.0))
    builder.add("power_hand_count_norm", _ratio(_count_cards_by_type(state.hand, "Power"), 10.0, 2.0))
    builder.add("upgraded_hand_count_norm", _ratio(sum(1 for card in state.hand if card.is_upgraded), 10.0, 2.0))
    builder.add("playable_zero_cost_norm", _ratio(sum(1 for cost in playable_costs if cost == 0.0), 10.0, 2.0))
    builder.add("playable_one_cost_norm", _ratio(sum(1 for cost in playable_costs if 0.0 < cost <= 1.0), 10.0, 2.0))
    builder.add("playable_high_cost_norm", _ratio(sum(1 for cost in playable_costs if cost >= 2.0), 10.0, 2.0))
    builder.add("playable_enemy_target_norm", _ratio(_count_cards_by_target(playable_cards, ENEMY_TARGET_TYPES), 10.0, 2.0))
    builder.add("playable_all_enemy_norm", _ratio(_count_cards_by_target(playable_cards, ALL_ENEMY_TARGET_TYPES), 10.0, 2.0))
    builder.add("playable_self_target_norm", _ratio(_count_cards_by_target(playable_cards, SELF_TARGET_TYPES), 10.0, 2.0))
    builder.add("playable_cost_sum_norm", _ratio(sum(playable_costs), 20.0, 2.0))
    builder.add("draw_pile_count_norm", _ratio(state.draw_pile_count, 50.0, 2.0))
    builder.add("discard_pile_count_norm", _ratio(state.discard_pile_count, 50.0, 2.0))
    builder.add("exhaust_pile_count_norm", _ratio(state.exhaust_pile_count, 50.0, 2.0))
    builder.add("enemy_count_norm", _ratio(living_enemy_count, 4.0, 2.0))
    builder.add("enemy_hp_ratio_sum_norm", _ratio(sum(_safe_div(enemy.hp, max(enemy.max_hp, 1.0)) for enemy in state.living_enemies), 3.0, 2.0))
    builder.add("enemy_block_sum_norm", _ratio(enemy_block_total, 100.0, 2.0))
    builder.add("enemy_incoming_damage_norm", _ratio(state.incoming_damage, 50.0, 2.0))
    builder.add("enemy_attack_intent_count_norm", _ratio(enemy_attack_intents, 3.0, 2.0))
    builder.add("enemy_buff_intent_count_norm", _ratio(enemy_buff_intents, 3.0, 2.0))
    builder.add("enemy_debuff_intent_count_norm", _ratio(enemy_debuff_intents, 3.0, 2.0))
    builder.add("enemy_power_count_norm", _ratio(enemy_power_count, 20.0, 2.0))
    builder.add("reward_item_count_norm", _ratio(len(state.reward_items), 5.0, 2.0))
    builder.add("card_reward_count_norm", _ratio(len(state.card_reward_cards), 3.0, 2.0))
    builder.add("map_remaining_node_count_norm", _ratio(state.map_remaining_node_count, 40.0, 2.0))
    builder.add("map_remaining_boss_distance_norm", _ratio(max(0, state.map_remaining_boss_distance), 15.0, 2.0))
    builder.add("map_best_route_score_norm", _ratio(map_best_route_score, 12.0, 2.0))
    builder.add("map_mean_route_score_norm", _ratio(map_mean_route_score, 12.0, 2.0))
    builder.add("map_path_count_total_norm", _ratio(map_path_count_total, 24.0, 2.0))
    builder.add("map_option_count_norm", _ratio(len(state.map_options), 6.0, 2.0))
    builder.add("map_special_option_count_norm", _ratio(map_special_count, 6.0, 2.0))
    for node_type in MAP_ROUTE_TRACKED_TYPES:
        builder.add(
            f"map_remaining_ratio.{node_type}",
            map_type_ratio(state.map_remaining_type_counts, node_type),
        )
    for depth_index, depth_label in enumerate(MAP_ROUTE_DEPTH_LABELS):
        for node_type in MAP_ROUTE_TRACKED_TYPES:
            builder.add(
                f"map_remaining_depth_ratio.{depth_label}.{node_type}",
                map_depth_type_ratio(state.map_remaining_depth_type_counts, depth_index, node_type),
            )
    route_by_index = {route.option_index: route for route in state.map_route_summaries}
    for option_index in range(MAP_ROUTE_OPTION_COUNT):
        route = route_by_index.get(option_index)
        prefix = f"map_route.{option_index}"
        builder.add(f"{prefix}.available", _bool(route is not None))
        builder.add(f"{prefix}.reachable_count_norm", _ratio(route.reachable_count if route else 0, 20.0, 2.0))
        builder.add(f"{prefix}.path_count_norm", _ratio(route.path_count if route else 0, 12.0, 2.0))
        builder.add(f"{prefix}.boss_distance_norm", _ratio(max(0, route.boss_distance) if route else 0, 15.0, 2.0))
        builder.add(f"{prefix}.route_score_norm", _ratio(route.route_score if route else 0.0, 12.0, 2.0))
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

    hand_by_index = {card.index: card for card in state.hand if 0 <= card.index < HAND_SLOT_COUNT}
    for slot_index in range(HAND_SLOT_COUNT):
        card = hand_by_index.get(slot_index)
        prefix = f"hand_slot.{slot_index}"
        enchantment_count = len(card.enchantments) if card else 0
        affliction_count = len(card.afflictions) if card else 0
        builder.add(f"{prefix}.present", _bool(card is not None))
        builder.add(f"{prefix}.playable", _bool(card is not None and card.can_play))
        builder.add(f"{prefix}.cost_norm", _ratio(_cost_units(card.cost, state.player_energy) if card else 0.0, 5.0, 2.0))
        builder.add(f"{prefix}.is_attack", _bool(card is not None and card.card_type == "Attack"))
        builder.add(f"{prefix}.is_skill", _bool(card is not None and card.card_type == "Skill"))
        builder.add(f"{prefix}.is_power", _bool(card is not None and card.card_type == "Power"))
        builder.add(f"{prefix}.is_status", _bool(card is not None and card.card_type == "Status"))
        builder.add(f"{prefix}.is_curse", _bool(card is not None and card.card_type == "Curse"))
        builder.add(f"{prefix}.targets_enemy", _bool(card is not None and _match_token(card.target_type, ENEMY_TARGET_TYPES)))
        builder.add(f"{prefix}.targets_all_enemy", _bool(card is not None and _match_token(card.target_type, ALL_ENEMY_TARGET_TYPES)))
        builder.add(f"{prefix}.targets_self", _bool(card is not None and _match_token(card.target_type, SELF_TARGET_TYPES)))
        builder.add(f"{prefix}.consumes_on_play", _bool(card is not None and card.consumes_on_play))
        builder.add(f"{prefix}.upgraded", _bool(card is not None and card.is_upgraded))
        builder.add(f"{prefix}.enchantment_count_norm", _ratio(enchantment_count, 4.0, 2.0))
        builder.add(f"{prefix}.affliction_count_norm", _ratio(affliction_count, 4.0, 2.0))
        builder.add(f"{prefix}.op_consume", _bool(card is not None and card.consumes_on_play))
        builder.add(f"{prefix}.op_upgrade", _bool(card is not None and card.is_upgraded))
        builder.add(f"{prefix}.op_remove", 0.0)
        builder.add(f"{prefix}.op_enchant", _bool(card is not None and enchantment_count > 0))

    card_select_by_index = {card.index: card for card in state.card_select_cards if 0 <= card.index < CARD_SELECT_SLOT_COUNT}
    for slot_index in range(CARD_SELECT_SLOT_COUNT):
        card = card_select_by_index.get(slot_index)
        prefix = f"card_select_slot.{slot_index}"
        enchantment_count = len(card.enchantments) if card else 0
        affliction_count = len(card.afflictions) if card else 0
        builder.add(f"{prefix}.present", _bool(card is not None))
        builder.add(f"{prefix}.playable", _bool(card is not None and card.can_play))
        builder.add(f"{prefix}.cost_norm", _ratio(_cost_units(card.cost, state.player_energy) if card else 0.0, 5.0, 2.0))
        builder.add(f"{prefix}.is_attack", _bool(card is not None and card.card_type == "Attack"))
        builder.add(f"{prefix}.is_skill", _bool(card is not None and card.card_type == "Skill"))
        builder.add(f"{prefix}.is_power", _bool(card is not None and card.card_type == "Power"))
        builder.add(f"{prefix}.is_status", _bool(card is not None and card.card_type == "Status"))
        builder.add(f"{prefix}.is_curse", _bool(card is not None and card.card_type == "Curse"))
        builder.add(f"{prefix}.consumes_on_play", _bool(card is not None and card.consumes_on_play))
        builder.add(f"{prefix}.upgraded", _bool(card is not None and card.is_upgraded))
        builder.add(f"{prefix}.enchantment_count_norm", _ratio(enchantment_count, 4.0, 2.0))
        builder.add(f"{prefix}.affliction_count_norm", _ratio(affliction_count, 4.0, 2.0))
        builder.add(f"{prefix}.op_consume", _bool(card is not None and state.card_select_is_consume))
        builder.add(f"{prefix}.op_upgrade", _bool(card is not None and state.card_select_is_upgrade))
        builder.add(f"{prefix}.op_remove", _bool(card is not None and state.card_select_is_remove))
        builder.add(f"{prefix}.op_enchant", _bool(card is not None and state.card_select_is_enchant))

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
    event_option_by_index = {option.index: option for option in state.event_options if 0 <= option.index < EVENT_OPTION_COUNT}
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
    hand_select_by_index = {card.index: card for card in state.hand_select_cards if 0 <= card.index < HAND_SELECT_SLOT_COUNT}
    selected_hand_indices = {card.index for card in state.hand_select_selected_cards if card.index >= 0}
    selected_hand_ids = {card.id for card in state.hand_select_selected_cards if card.id}
    for slot_index in range(HAND_SELECT_SLOT_COUNT):
        card = hand_select_by_index.get(slot_index)
        prefix = f"hand_select_slot.{slot_index}"
        is_selected = bool(card is not None and (card.index in selected_hand_indices or card.id in selected_hand_ids))
        enchantment_count = len(card.enchantments) if card else 0
        affliction_count = len(card.afflictions) if card else 0
        builder.add(f"{prefix}.present", _bool(card is not None))
        builder.add(f"{prefix}.selected", _bool(is_selected))
        builder.add(f"{prefix}.playable", _bool(card is not None and card.can_play))
        builder.add(f"{prefix}.cost_norm", _ratio(_cost_units(card.cost, state.player_energy) if card else 0.0, 5.0, 2.0))
        builder.add(f"{prefix}.is_attack", _bool(card is not None and card.card_type == "Attack"))
        builder.add(f"{prefix}.is_skill", _bool(card is not None and card.card_type == "Skill"))
        builder.add(f"{prefix}.is_power", _bool(card is not None and card.card_type == "Power"))
        builder.add(f"{prefix}.is_status", _bool(card is not None and card.card_type == "Status"))
        builder.add(f"{prefix}.is_curse", _bool(card is not None and card.card_type == "Curse"))
        builder.add(f"{prefix}.targets_enemy", _bool(card is not None and _match_token(card.target_type, ENEMY_TARGET_TYPES)))
        builder.add(f"{prefix}.targets_all_enemy", _bool(card is not None and _match_token(card.target_type, ALL_ENEMY_TARGET_TYPES)))
        builder.add(f"{prefix}.targets_self", _bool(card is not None and _match_token(card.target_type, SELF_TARGET_TYPES)))
        builder.add(f"{prefix}.consumes_on_play", _bool(card is not None and card.consumes_on_play))
        builder.add(f"{prefix}.upgraded", _bool(card is not None and card.is_upgraded))
        builder.add(f"{prefix}.enchantment_count_norm", _ratio(enchantment_count, 4.0, 2.0))
        builder.add(f"{prefix}.affliction_count_norm", _ratio(affliction_count, 4.0, 2.0))
        builder.add(f"{prefix}.op_consume", _bool(card is not None and state.hand_select_is_consume))
        builder.add(f"{prefix}.op_upgrade", _bool(card is not None and state.hand_select_is_upgrade))
        builder.add(f"{prefix}.op_remove", _bool(card is not None and state.hand_select_is_remove))
        builder.add(f"{prefix}.op_enchant", _bool(card is not None and state.hand_select_is_enchant))
    builder.add("event_option_count_norm", _ratio(len(state.event_options), 5.0, 2.0))
    builder.add("event_unlocked_option_count_norm", _ratio(event_unlocked_count, 5.0, 2.0))
    builder.add("rest_option_count_norm", _ratio(len(state.rest_options), 5.0, 2.0))
    builder.add("rest_enabled_option_count_norm", _ratio(rest_enabled_count, 5.0, 2.0))
    builder.add("card_select_count_norm", _ratio(len(state.card_select_cards), 20.0, 2.0))
    builder.add("card_select_preview_count_norm", _ratio(len(state.card_select_preview_cards), 20.0, 2.0))
    builder.add("card_select_is_consume", _bool(state.card_select_is_consume))
    builder.add("card_select_is_upgrade", _bool(state.card_select_is_upgrade))
    builder.add("card_select_is_remove", _bool(state.card_select_is_remove))
    builder.add("card_select_is_enchant", _bool(state.card_select_is_enchant))
    builder.add("bundle_count_norm", _ratio(state.bundle_count, 3.0, 2.0))
    builder.add("bundle_card_count_norm", _ratio(len(state.bundle_cards), 9.0, 2.0))
    builder.add("bundle_preview_count_norm", _ratio(len(state.bundle_preview_cards), 9.0, 2.0))
    builder.add("hand_select_count_norm", _ratio(len(state.hand_select_cards), 10.0, 2.0))
    builder.add("hand_select_selected_count_norm", _ratio(len(state.hand_select_selected_cards), 10.0, 2.0))
    builder.add("hand_select_is_consume", _bool(state.hand_select_is_consume))
    builder.add("hand_select_is_upgrade", _bool(state.hand_select_is_upgrade))
    builder.add("hand_select_is_remove", _bool(state.hand_select_is_remove))
    builder.add("hand_select_is_enchant", _bool(state.hand_select_is_enchant))
    builder.add("relic_select_count_norm", _ratio(len(state.relic_select_relics), 3.0, 2.0))
    builder.add("treasure_relic_count_norm", _ratio(len(state.treasure_relics), 3.0, 2.0))
    builder.add("shop_item_count_norm", _ratio(len(state.shop_items), 12.0, 2.0))
    builder.add("shop_affordable_count_norm", _ratio(affordable_shop_count, 12.0, 2.0))
    builder.add("fake_merchant_item_count_norm", _ratio(len(state.fake_merchant_shop_items), 12.0, 2.0))
    builder.add("fake_merchant_affordable_count_norm", _ratio(affordable_fake_shop_count, 12.0, 2.0))
    builder.add("crystal_clickable_count_norm", _ratio(state.crystal_clickable_count, 8.0, 2.0))
    builder.add("rewards_can_proceed", _bool(state.rewards_can_proceed))
    builder.add("card_reward_can_skip", _bool(state.card_reward_can_skip))
    builder.add("event_in_dialogue", _bool(state.event_in_dialogue))
    builder.add("rest_can_proceed", _bool(state.rest_can_proceed))
    builder.add("card_select_can_confirm", _bool(state.card_select_can_confirm))
    builder.add("card_select_can_cancel", _bool(state.card_select_can_cancel))
    builder.add("bundle_can_confirm", _bool(state.bundle_can_confirm))
    builder.add("bundle_can_cancel", _bool(state.bundle_can_cancel))
    builder.add("hand_select_can_confirm", _bool(state.hand_select_can_confirm))
    builder.add("relic_select_can_skip", _bool(state.relic_select_can_skip))
    builder.add("treasure_can_proceed", _bool(state.treasure_can_proceed))
    builder.add("shop_can_proceed", _bool(state.shop_can_proceed))
    builder.add("fake_merchant_can_proceed", _bool(state.fake_merchant_can_proceed))
    builder.add("crystal_can_proceed", _bool(state.crystal_can_proceed))
    return builder


def encode_state_semantic_scalars(state: dict[str, object] | SemanticState) -> list[float]:
    return _build_scalar_features(ensure_semantic_state(state)).values


def _build_relation_features(state: SemanticState, history: SemanticHistoryTracker | None = None) -> _FeatureBuilder:
    builder = _FeatureBuilder()
    playable_count = len(state.playable_cards)
    living_enemy_count = len(state.living_enemies)
    hand_order_profile = history.hand_order_profile if history is not None else build_hand_order_profile(state)
    hand_order_slots = hand_order_profile.available_slots
    incoming_damage = state.incoming_damage
    max_hp = max(state.player_max_hp, 1.0)
    block_gap = max(0.0, incoming_damage - state.player_block)
    overblock = max(0.0, state.player_block - incoming_damage)
    can_take_combat_actions = state.is_player_turn and state.is_play_phase and not state.player_actions_disabled and not state.hand_in_card_play
    skip_risk = can_take_combat_actions and living_enemy_count > 0 and playable_count > 0 and state.player_energy > 0.0
    attack_playables = _count_cards_by_type(state.playable_cards, "Attack", playable_only=False)
    skill_playables = _count_cards_by_type(state.playable_cards, "Skill", playable_only=False)
    power_playables = _count_cards_by_type(state.playable_cards, "Power", playable_only=False)
    enemy_target_playables = _count_cards_by_target(state.playable_cards, ENEMY_TARGET_TYPES)
    aoe_playables = _count_cards_by_target(state.playable_cards, ALL_ENEMY_TARGET_TYPES)
    affordable_shop_count = sum(1 for item in state.shop_items if item.can_afford and item.is_stocked)
    full_potion_slots = len(state.potions) >= 3
    reward_has_potion = any(item.potion_id for item in state.reward_items)
    proceed_ready = any(
        (
            state.rewards_can_proceed,
            state.rest_can_proceed,
            state.treasure_can_proceed,
            state.shop_can_proceed,
            state.fake_merchant_can_proceed,
            state.crystal_can_proceed,
        )
    )
    selection_ready = any(
        (
            state.card_select_can_confirm,
            state.card_select_can_cancel,
            state.bundle_can_confirm,
            state.bundle_can_cancel,
            state.hand_select_can_confirm,
            state.relic_select_can_skip,
            state.card_reward_can_skip,
        )
    )
    reward_pressure = len(state.reward_items) + len(state.card_reward_cards) + len(state.relic_select_relics) + len(state.treasure_relics)
    selection_pressure = len(state.card_select_cards) + len(state.bundle_cards) + len(state.hand_select_cards)
    map_route_scores = [route.route_score for route in state.map_route_summaries]
    map_best_route_score = max(map_route_scores, default=0.0)
    map_worst_route_score = min(map_route_scores, default=0.0)

    builder.add("in_combat", _bool(state.in_combat))
    builder.add("is_player_turn", _bool(state.is_player_turn))
    builder.add("combat_action_window_ready", _bool(can_take_combat_actions))
    builder.add("combat_victory_ready", _bool(state.in_combat and living_enemy_count == 0))
    builder.add("skip_unspent_risk", _bool(skip_risk))
    builder.add("unspent_energy_norm", _ratio(state.player_energy, 5.0, 2.0))
    builder.add("block_gap_norm", _ratio(block_gap, max_hp, 2.0))
    builder.add("overblock_norm", _ratio(overblock, max_hp, 2.0))
    builder.add("block_coverage_ratio", 1.0 if incoming_damage <= 0.0 else min(1.0, state.player_block / max(incoming_damage, 1.0)))
    builder.add("enemy_hp_pressure_ratio", _safe_div(state.total_enemy_hp, max(state.total_enemy_max_hp, 1.0)))
    builder.add("attack_playable_ratio", _safe_div(attack_playables, max(playable_count, 1)))
    builder.add("skill_playable_ratio", _safe_div(skill_playables, max(playable_count, 1)))
    builder.add("power_playable_ratio", _safe_div(power_playables, max(playable_count, 1)))
    builder.add("enemy_target_playable_ratio", _safe_div(enemy_target_playables, max(playable_count, 1)))
    builder.add("aoe_playable_ratio", _safe_div(aoe_playables, max(playable_count, 1)))
    builder.add("enemy_target_coverage_ratio", _safe_div(enemy_target_playables + aoe_playables, max(living_enemy_count, 1)))
    builder.add("action_window_blocked", _bool(state.player_actions_disabled or state.hand_in_card_play))
    builder.add("shop_afford_ratio", _safe_div(affordable_shop_count, max(len(state.shop_items), 1)))
    builder.add("reward_pressure_norm", _ratio(reward_pressure, 6.0, 2.0))
    builder.add("selection_pressure_norm", _ratio(selection_pressure, 20.0, 2.0))
    builder.add("selection_ready", _bool(selection_ready))
    builder.add("proceed_ready", _bool(proceed_ready))
    builder.add("map_branching_norm", _ratio(len(state.map_options), 6.0, 2.0))
    builder.add("map_remaining_boss_distance_norm", _ratio(max(0, state.map_remaining_boss_distance), 15.0, 2.0))
    builder.add("map_best_route_score_norm", _ratio(map_best_route_score, 12.0, 2.0))
    builder.add("map_route_score_gap_norm", _ratio(max(0.0, map_best_route_score - map_worst_route_score), 8.0, 2.0))
    builder.add("map_elite_remaining_ratio", map_type_ratio(state.map_remaining_type_counts, "Elite"))
    builder.add("map_rest_remaining_ratio", map_type_ratio(state.map_remaining_type_counts, "RestSite"))
    builder.add("map_shop_remaining_ratio", map_type_ratio(state.map_remaining_type_counts, "Shop"))
    builder.add("map_event_remaining_ratio", map_type_ratio(state.map_remaining_type_counts, "Event"))
    builder.add("map_treasure_remaining_ratio", map_type_ratio(state.map_remaining_type_counts, "Treasure"))
    builder.add("event_branching_norm", _safe_div(sum(1 for option in state.event_options if not option.is_locked), max(len(state.event_options), 1)))
    builder.add("rest_choice_ratio", _safe_div(sum(1 for option in state.rest_options if option.is_enabled), max(len(state.rest_options), 1)))
    builder.add("potion_overflow_risk", _bool(full_potion_slots and reward_has_potion))
    builder.add("card_select_preview_ratio", _safe_div(len(state.card_select_preview_cards), max(len(state.card_select_cards), 1)))
    builder.add("bundle_preview_ratio", _safe_div(len(state.bundle_preview_cards), max(len(state.bundle_cards), 1)))
    builder.add("crystal_work_remaining_norm", _ratio(state.crystal_clickable_count, 8.0, 2.0))
    builder.add("hand_order_slot_count_norm", _ratio(len(hand_order_slots), HAND_SLOT_COUNT, 2.0))
    builder.add(
        "hand_order_edge_density",
        _safe_div(sum(hand_order_profile.slot_before_counts), max(len(hand_order_slots) * max(len(hand_order_slots) - 1, 0), 1)),
    )
    for slot_index in range(HAND_SLOT_COUNT):
        before_count = hand_order_profile.slot_before_counts[slot_index]
        after_count = hand_order_profile.slot_after_counts[slot_index]
        builder.add(f"hand_order_slot.{slot_index}.priority_norm", _signed_ratio(hand_order_profile.slot_priorities[slot_index], HAND_ORDER_PRIORITY_SCALE, 1.0))
        builder.add(f"hand_order_slot.{slot_index}.before_count_norm", _ratio(before_count, HAND_SLOT_COUNT - 1, 1.0))
        builder.add(f"hand_order_slot.{slot_index}.after_count_norm", _ratio(after_count, HAND_SLOT_COUNT - 1, 1.0))
        builder.add(f"hand_order_slot.{slot_index}.prefer_early", _bool(before_count > after_count))
        builder.add(f"hand_order_slot.{slot_index}.prefer_late", _bool(after_count > before_count))
    for left_index in range(HAND_SLOT_COUNT):
        for right_index in range(left_index + 1, HAND_SLOT_COUNT):
            builder.add(
                f"hand_order_pair.{left_index}.{right_index}.left_before_right",
                _bool(hand_order_profile.pair_prefer_before[left_index][right_index]),
            )
            builder.add(
                f"hand_order_pair.{left_index}.{right_index}.right_before_left",
                _bool(hand_order_profile.pair_prefer_before[right_index][left_index]),
            )
    return builder


def encode_state_relation_semantics(
    state: dict[str, object] | SemanticState,
    history: SemanticHistoryTracker | None = None,
) -> list[float]:
    return _build_relation_features(ensure_semantic_state(state), history).values


def _action_bucket(
    tool_name: str,
    previous_state: SemanticState,
    action_kwargs: dict[str, object] | None,
) -> str:
    kwargs = action_kwargs or {}
    if tool_name == "combat_play_card":
        card_index = _to_int(kwargs.get("card_index", -1))
        card = next((item for item in previous_state.hand if item.index == card_index), None)
        if card is None:
            return "combat_other_card"
        if card.card_type == "Attack":
            return "combat_attack_card"
        if card.card_type == "Skill":
            return "combat_skill_card"
        if card.card_type == "Power":
            return "combat_power_card"
        return "combat_other_card"
    if tool_name == "combat_end_turn":
        return "combat_end_turn"
    if tool_name == "use_potion":
        return "use_potion"
    if tool_name == "discard_potion":
        return "discard_potion"
    if tool_name in {"combat_select_card", "combat_confirm_selection"}:
        return "combat_select"
    if tool_name in {"rewards_claim", "rewards_pick_card", "rewards_skip_card"}:
        return "rewards"
    if tool_name == "map_choose_node":
        return "map"
    if tool_name in {"event_choose_option", "event_advance_dialogue"}:
        return "event"
    if tool_name == "rest_choose_option":
        return "rest"
    if tool_name == "shop_purchase":
        return "shop"
    if tool_name == "proceed_to_map":
        return "proceed"
    if tool_name in {"deck_select_card", "deck_confirm_selection", "deck_cancel_selection"}:
        return "deck_select"
    if tool_name in {"bundle_select", "bundle_confirm_selection", "bundle_cancel_selection"}:
        return "bundle_select"
    if tool_name in {"relic_select", "relic_skip"}:
        return "relic_select"
    if tool_name == "treasure_claim_relic":
        return "treasure"
    if tool_name.startswith("crystal_sphere"):
        return "crystal"
    return "other"


def _turn_skipped_unspent(previous: SemanticState, next_state: SemanticState, tool_name: str) -> bool:
    if tool_name != "combat_end_turn" or not previous.in_combat:
        return False
    if previous.player_energy <= 0.0:
        return False
    if len(previous.playable_cards) == 0:
        return False
    if len(previous.living_enemies) == 0:
        return False
    if next_state.in_combat and len(next_state.living_enemies) == 0:
        return False
    if not previous.is_player_turn:
        return False
    return True


def _build_history_features(state: SemanticState, history: SemanticHistoryTracker | None) -> _FeatureBuilder:
    tracker = history or SemanticHistoryTracker()
    tracker.sync_state(state)
    builder = _FeatureBuilder()
    for bucket in ACTION_HISTORY_BUCKETS:
        builder.add(f"last_action_bucket.{bucket}", _bool(tracker.last_action_bucket == bucket))
    builder.add("last_action_error", _bool(tracker.last_action_error))
    builder.add("last_action_changed_state_type", _bool(tracker.last_action_changed_state_type))
    builder.add("last_turn_end", _bool(tracker.last_turn_end))
    builder.add("last_combat_end", _bool(tracker.last_combat_end))
    builder.add("last_act_end", _bool(tracker.last_act_end))
    builder.add("last_run_end", _bool(tracker.last_run_end))
    builder.add("last_turn_skipped_unspent", _bool(tracker.last_turn_skipped_unspent))
    builder.add("last_hand_order_refresh", _bool(tracker.last_hand_order_refresh))
    builder.add("last_hand_order_refresh_turn_start", _bool(tracker.last_hand_order_refresh_turn_start))
    builder.add("last_hand_order_refresh_hand_changed", _bool(tracker.last_hand_order_refresh_hand_changed))
    builder.add("last_enemy_hp_progress_norm", _ratio(tracker.last_enemy_hp_progress, 50.0, 2.0))
    builder.add("last_player_hp_loss_norm", _ratio(tracker.last_player_hp_loss, 50.0, 2.0))
    builder.add("turn_step_count_norm", _ratio(tracker.turn_step_count, 10.0, 2.0))
    builder.add("turn_cards_played_norm", _ratio(tracker.turn_cards_played, 10.0, 2.0))
    builder.add("turn_attack_cards_norm", _ratio(tracker.turn_attack_cards, 10.0, 2.0))
    builder.add("turn_skill_cards_norm", _ratio(tracker.turn_skill_cards, 10.0, 2.0))
    builder.add("turn_power_cards_norm", _ratio(tracker.turn_power_cards, 10.0, 2.0))
    builder.add("turn_potions_used_norm", _ratio(tracker.turn_potions_used, 3.0, 2.0))
    builder.add("turn_failed_actions_norm", _ratio(tracker.turn_failed_actions, 5.0, 2.0))
    builder.add("turn_energy_spent_norm", _ratio(tracker.turn_energy_spent, 10.0, 2.0))
    builder.add("turn_enemy_damage_norm", _ratio(tracker.turn_enemy_damage, 60.0, 2.0))
    builder.add("turn_player_damage_norm", _ratio(tracker.turn_player_damage, 60.0, 2.0))
    builder.add("turn_hand_order_refreshes_norm", _ratio(tracker.turn_hand_order_refreshes, 6.0, 2.0))
    builder.add("combat_step_count_norm", _ratio(tracker.combat_step_count, 60.0, 2.0))
    builder.add("combat_cards_played_norm", _ratio(tracker.combat_cards_played, 40.0, 2.0))
    builder.add("combat_potions_used_norm", _ratio(tracker.combat_potions_used, 6.0, 2.0))
    builder.add("combat_failed_actions_norm", _ratio(tracker.combat_failed_actions, 10.0, 2.0))
    builder.add("combat_turn_ends_norm", _ratio(tracker.combat_turn_ends, 20.0, 2.0))
    builder.add("combat_skip_unspent_norm", _ratio(tracker.combat_skip_unspent, 10.0, 2.0))
    builder.add("combat_enemy_damage_norm", _ratio(tracker.combat_enemy_damage, 200.0, 2.0))
    builder.add("combat_player_damage_norm", _ratio(tracker.combat_player_damage, 200.0, 2.0))
    builder.add("current_round_norm", _ratio(state.battle_round, 20.0, 2.0))
    return builder


def encode_state_history_semantics(
    state: dict[str, object] | SemanticState,
    history: SemanticHistoryTracker | None = None,
) -> list[float]:
    return _build_history_features(ensure_semantic_state(state), history).values


def encode_state_concepts(
    state: dict[str, object] | SemanticState,
    catalog: SemanticCatalog | None = None,
) -> list[float]:
    return activate_state_concepts(state, catalog).vector


def encode_semantic_observation(
    state: dict[str, object] | SemanticState,
    *,
    catalog: SemanticCatalog | None = None,
    history: SemanticHistoryTracker | None = None,
) -> SemanticObservation:
    semantic_state = ensure_semantic_state(state)
    if history is not None:
        semantic_state = history.sync_state(semantic_state)
    concept_activation = activate_state_concepts(semantic_state, catalog or DEFAULT_SEMANTIC_CATALOG)
    scalar_vector = encode_state_semantic_scalars(semantic_state)
    relation_vector = encode_state_relation_semantics(semantic_state, history)
    history_vector = encode_state_history_semantics(semantic_state, history)
    vector = concept_activation.vector + scalar_vector + relation_vector + history_vector
    return SemanticObservation(
        semantic_state=semantic_state,
        concept_activation=concept_activation,
        scalar_vector=scalar_vector,
        relation_vector=relation_vector,
        history_vector=history_vector,
        vector=vector,
    )


SEMANTIC_SCALAR_FEATURE_NAMES = tuple(_build_scalar_features(SemanticState()).names)
SEMANTIC_SCALAR_SIZE = len(SEMANTIC_SCALAR_FEATURE_NAMES)
SEMANTIC_RELATION_FEATURE_NAMES = tuple(_build_relation_features(SemanticState()).names)
SEMANTIC_RELATION_SIZE = len(SEMANTIC_RELATION_FEATURE_NAMES)
SEMANTIC_HISTORY_FEATURE_NAMES = tuple(_build_history_features(SemanticState(), SemanticHistoryTracker()).names)
SEMANTIC_HISTORY_SIZE = len(SEMANTIC_HISTORY_FEATURE_NAMES)
SEMANTIC_OBSERVATION_SIZE = CONCEPT_VOCAB_SIZE + SEMANTIC_SCALAR_SIZE + SEMANTIC_RELATION_SIZE + SEMANTIC_HISTORY_SIZE
