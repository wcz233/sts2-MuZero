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


def _intent_damage_from_label(label: object) -> float:
    digits = "".join(character for character in str(label) if character.isdigit())
    return float(digits) if digits else 0.0


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
    category: str = ""
    can_afford: bool = False
    is_stocked: bool = True
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
    relic_id: str = ""
    option_type: str = ""


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
    event_id: str = ""
    event_in_dialogue: bool = False
    event_options: tuple[SemanticOption, ...] = ()
    rest_options: tuple[SemanticOption, ...] = ()
    rest_can_proceed: bool = False
    card_select_cards: tuple[SemanticCard, ...] = ()
    card_select_preview_cards: tuple[SemanticCard, ...] = ()
    card_select_can_confirm: bool = False
    card_select_can_cancel: bool = False
    bundle_cards: tuple[SemanticCard, ...] = ()
    bundle_preview_cards: tuple[SemanticCard, ...] = ()
    bundle_count: int = 0
    bundle_can_confirm: bool = False
    bundle_can_cancel: bool = False
    hand_select_cards: tuple[SemanticCard, ...] = ()
    hand_select_selected_cards: tuple[SemanticCard, ...] = ()
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

    def sync_state(self, state: dict[str, object] | SemanticState) -> SemanticState:
        semantic_state = ensure_semantic_state(state)
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
            elif self._combat_anchor != combat_anchor:
                self._reset_combat_progress()
                self._combat_anchor = combat_anchor
                self._current_turn_round = semantic_state.battle_round
            elif semantic_state.is_player_turn and self._current_turn_round != semantic_state.battle_round:
                self._reset_turn_progress()
                self._current_turn_round = semantic_state.battle_round
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
                if group == "monsters":
                    monster_token = _normalize_model_token(item.get("entry") or item.get("model_id", "")).split(".")[-1]
                    for move_id in item.get("move_ids", []):
                        move_token = _normalize_model_token(move_id)
                        if monster_token and move_token:
                            add(f"MONSTER_MOVE.{monster_token}.{move_token}")

        for category, values in RUNTIME_CONCEPTS.items():
            for value in values:
                add(f"{category}.{_slugify(value)}")

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


def _normalize_shop_item(raw: dict[str, object]) -> SemanticShopItem:
    return SemanticShopItem(
        category=str(raw.get("category", "")),
        can_afford=bool(raw.get("can_afford")),
        is_stocked=bool(raw.get("is_stocked", True)),
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
        relic_id=_normalize_model_token(raw.get("relic_id")),
        option_type=str(raw.get("type", raw.get("option_type", ""))),
    )


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
        map_options=tuple(_normalize_option(option) for option in _list_at(map_state, "next_options")),
        event_id=event_id,
        event_in_dialogue=bool(event_state.get("in_dialogue")),
        event_options=tuple(_normalize_option(option) for option in _list_at(event_state, "options")),
        rest_options=tuple(_normalize_option(option) for option in _list_at(rest_state, "options")),
        rest_can_proceed=bool(rest_state.get("can_proceed")),
        card_select_cards=tuple(_normalize_card(card) for card in _list_at(card_select, "cards")),
        card_select_preview_cards=tuple(_normalize_card(card) for card in _list_at(card_select, "preview_cards")),
        card_select_can_confirm=bool(card_select.get("can_confirm")),
        card_select_can_cancel=bool(card_select.get("can_cancel") or card_select.get("can_skip")),
        bundle_cards=_flatten_bundle_cards(bundle_select),
        bundle_preview_cards=tuple(_normalize_card(card) for card in _list_at(bundle_select, "preview_cards")),
        bundle_count=len(_list_at(bundle_select, "bundles")),
        bundle_can_confirm=bool(bundle_select.get("can_confirm")),
        bundle_can_cancel=bool(bundle_select.get("can_cancel")),
        hand_select_cards=tuple(_normalize_card(card) for card in _list_at(hand_select, "cards")),
        hand_select_selected_cards=tuple(_normalize_card(card) for card in _list_at(hand_select, "selected_cards")),
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
    if key in catalog.concept_index:
        active_keys.add(key)


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
    _activate_prefixed(active_keys, catalog, semantic_state.act_id, "ACT")
    _activate_prefixed(active_keys, catalog, semantic_state.character_id, "CHARACTER")
    _activate_prefixed(active_keys, catalog, semantic_state.encounter_id, "ENCOUNTER")
    _activate_prefixed(active_keys, catalog, semantic_state.event_id, "EVENT", "ANCIENT")

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
    builder.add("map_option_count_norm", _ratio(len(state.map_options), 6.0, 2.0))
    builder.add("map_special_option_count_norm", _ratio(map_special_count, 6.0, 2.0))
    builder.add("event_option_count_norm", _ratio(len(state.event_options), 5.0, 2.0))
    builder.add("event_unlocked_option_count_norm", _ratio(event_unlocked_count, 5.0, 2.0))
    builder.add("rest_option_count_norm", _ratio(len(state.rest_options), 5.0, 2.0))
    builder.add("rest_enabled_option_count_norm", _ratio(rest_enabled_count, 5.0, 2.0))
    builder.add("card_select_count_norm", _ratio(len(state.card_select_cards), 20.0, 2.0))
    builder.add("card_select_preview_count_norm", _ratio(len(state.card_select_preview_cards), 20.0, 2.0))
    builder.add("bundle_count_norm", _ratio(state.bundle_count, 3.0, 2.0))
    builder.add("bundle_card_count_norm", _ratio(len(state.bundle_cards), 9.0, 2.0))
    builder.add("bundle_preview_count_norm", _ratio(len(state.bundle_preview_cards), 9.0, 2.0))
    builder.add("hand_select_count_norm", _ratio(len(state.hand_select_cards), 10.0, 2.0))
    builder.add("hand_select_selected_count_norm", _ratio(len(state.hand_select_selected_cards), 10.0, 2.0))
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


def _build_relation_features(state: SemanticState) -> _FeatureBuilder:
    builder = _FeatureBuilder()
    playable_count = len(state.playable_cards)
    living_enemy_count = len(state.living_enemies)
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
    builder.add("event_branching_norm", _safe_div(sum(1 for option in state.event_options if not option.is_locked), max(len(state.event_options), 1)))
    builder.add("rest_choice_ratio", _safe_div(sum(1 for option in state.rest_options if option.is_enabled), max(len(state.rest_options), 1)))
    builder.add("potion_overflow_risk", _bool(full_potion_slots and reward_has_potion))
    builder.add("card_select_preview_ratio", _safe_div(len(state.card_select_preview_cards), max(len(state.card_select_cards), 1)))
    builder.add("bundle_preview_ratio", _safe_div(len(state.bundle_preview_cards), max(len(state.bundle_cards), 1)))
    builder.add("crystal_work_remaining_norm", _ratio(state.crystal_clickable_count, 8.0, 2.0))
    return builder


def encode_state_relation_semantics(state: dict[str, object] | SemanticState) -> list[float]:
    return _build_relation_features(ensure_semantic_state(state)).values


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
    relation_vector = encode_state_relation_semantics(semantic_state)
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
