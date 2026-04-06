import time
from dataclasses import dataclass

from .action_space import action_index
from .bridge import STS2Bridge, STS2BridgeError
from .semantics import (
    CONCEPT_VOCAB_SIZE,
    SEMANTIC_HISTORY_SIZE,
    SEMANTIC_OBSERVATION_SIZE as SEMANTIC_VECTOR_SIZE,
    SEMANTIC_RELATION_SIZE,
    SEMANTIC_SCALAR_SIZE,
    SemanticHistoryTracker,
    encode_semantic_observation,
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
TERMINAL_STATE_TYPES = {"menu", "game_over"}


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
    hp_delta: float = 5.0
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


def _extract_base_observation_features(state: dict[str, object]) -> list[float]:
    state_type = str(state.get("state_type", "unknown"))
    run = state.get("run") if isinstance(state.get("run"), dict) else {}
    player = state.get("player") if isinstance(state.get("player"), dict) else {}
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    features: list[float] = []

    _push_state_type(features, state_type)
    features.extend(
        [
            _ratio(run.get("act"), 4.0),
            _ratio(run.get("floor"), 60.0),
            _ratio(run.get("ascension"), 20.0),
            _safe_div(player.get("hp"), max(player.get("max_hp", 0) or 0, 1)),
            _ratio(player.get("gold"), 500.0),
            _ratio(player.get("block"), 100.0),
            _safe_div(player.get("energy"), max(player.get("max_energy", 0) or 0, 1)),
            _ratio(len(player.get("relics", [])), 40.0),
            _ratio(len(player.get("potions", [])), 5.0),
            _ratio(len(player.get("hand", [])), 10.0),
            _ratio(player.get("draw_pile_count"), 50.0),
            _ratio(player.get("discard_pile_count"), 50.0),
        ]
    )

    hand = player.get("hand") if isinstance(player.get("hand"), list) else []
    hand_by_index = {card.get("index"): card for card in hand if isinstance(card, dict)}
    for slot in range(10):
        card = hand_by_index.get(slot)
        if not isinstance(card, dict):
            features.extend([0.0, 0.0, 0.0, 0.0])
            continue
        features.extend(
            [
                1.0,
                _bool(card.get("can_play")),
                _cost_to_float(card.get("cost")),
                _bool(_is_enemy_target(card.get("target_type"))),
            ]
        )

    enemies = battle.get("enemies") if isinstance(battle.get("enemies"), list) else []
    for slot in range(3):
        enemy = enemies[slot] if slot < len(enemies) and isinstance(enemies[slot], dict) else None
        if enemy is None:
            features.extend([0.0, 0.0, 0.0])
            continue
        features.extend(
            [
                _bool(float(enemy.get("hp", 0) or 0) > 0),
                _safe_div(enemy.get("hp"), max(enemy.get("max_hp", 0) or 0, 1)),
                _ratio(_intent_damage(enemy), 50.0),
            ]
        )

    map_state = state.get("map") if isinstance(state.get("map"), dict) else {}
    options = map_state.get("next_options") if isinstance(map_state.get("next_options"), list) else []
    map_by_index = {option.get("index"): option for option in options if isinstance(option, dict)}
    for slot in range(6):
        option = map_by_index.get(slot)
        option_type = str(option.get("type", "")) if isinstance(option, dict) else ""
        features.extend(
            [
                _bool(isinstance(option, dict)),
                _bool(option_type == "Elite"),
                _bool(option_type in {"RestSite", "Shop", "Event", "Treasure", "Boss"}),
            ]
        )

    rewards = state.get("rewards") if isinstance(state.get("rewards"), dict) else {}
    reward_items = rewards.get("items") if isinstance(rewards.get("items"), list) else []
    reward_by_index = {item.get("index"): item for item in reward_items if isinstance(item, dict)}
    for slot in range(5):
        item = reward_by_index.get(slot)
        reward_type = str(item.get("type", "")) if isinstance(item, dict) else ""
        features.extend(
            [
                _bool(isinstance(item, dict)),
                _bool(reward_type in {"card", "special_card"}),
            ]
        )

    event_state = state.get("event") if isinstance(state.get("event"), dict) else {}
    event_options = event_state.get("options") if isinstance(event_state.get("options"), list) else []
    event_by_index = {option.get("index"): option for option in event_options if isinstance(option, dict)}
    for slot in range(5):
        option = event_by_index.get(slot)
        features.extend(
            [
                _bool(isinstance(option, dict)),
                _bool(isinstance(option, dict) and not option.get("is_locked")),
                _bool(isinstance(option, dict) and option.get("is_proceed")),
            ]
        )

    rest_state = state.get("rest_site") if isinstance(state.get("rest_site"), dict) else {}
    rest_options = rest_state.get("options") if isinstance(rest_state.get("options"), list) else []
    rest_by_index = {option.get("index"): option for option in rest_options if isinstance(option, dict)}
    for slot in range(5):
        option = rest_by_index.get(slot)
        features.extend([_bool(isinstance(option, dict)), _bool(isinstance(option, dict) and option.get("is_enabled"))])

    card_reward = state.get("card_reward") if isinstance(state.get("card_reward"), dict) else {}
    reward_cards = card_reward.get("cards") if isinstance(card_reward.get("cards"), list) else []
    reward_card_by_index = {card.get("index"): card for card in reward_cards if isinstance(card, dict)}
    for slot in range(3):
        card = reward_card_by_index.get(slot)
        card_type = str(card.get("type", "")) if isinstance(card, dict) else ""
        features.extend([_bool(isinstance(card, dict)), _bool(card_type == "Attack"), _bool(card_type == "Skill")])

    shop_state = state.get("shop") if isinstance(state.get("shop"), dict) else {}
    fake_merchant = state.get("fake_merchant") if isinstance(state.get("fake_merchant"), dict) else {}
    if state_type == "fake_merchant" and isinstance(fake_merchant.get("shop"), dict):
        shop_state = fake_merchant["shop"]
    shop_items = shop_state.get("items") if isinstance(shop_state.get("items"), list) else []
    shop_by_index = {item.get("index"): item for item in shop_items if isinstance(item, dict)}
    for slot in range(12):
        item = shop_by_index.get(slot)
        category = str(item.get("category", "")) if isinstance(item, dict) else ""
        features.extend(
            [
                _bool(isinstance(item, dict)),
                _bool(isinstance(item, dict) and item.get("can_afford")),
                _bool(category == "relic"),
            ]
        )

    features.extend(
        [
            _ratio(len(state.get("card_select", {}).get("cards", [])) if isinstance(state.get("card_select"), dict) else 0, 20.0),
            _bool(isinstance(state.get("card_select"), dict) and state["card_select"].get("can_confirm")),
            _bool(isinstance(state.get("card_select"), dict) and (state["card_select"].get("can_cancel") or state["card_select"].get("can_skip"))),
            _ratio(len(state.get("bundle_select", {}).get("bundles", [])) if isinstance(state.get("bundle_select"), dict) else 0, 3.0),
            _bool(isinstance(state.get("bundle_select"), dict) and state["bundle_select"].get("can_confirm")),
            _bool(isinstance(state.get("bundle_select"), dict) and state["bundle_select"].get("can_cancel")),
            _ratio(len(state.get("relic_select", {}).get("relics", [])) if isinstance(state.get("relic_select"), dict) else 0, 3.0),
            _bool(isinstance(state.get("relic_select"), dict) and state["relic_select"].get("can_skip")),
            _ratio(len(state.get("crystal_sphere", {}).get("clickable_cells", [])) if isinstance(state.get("crystal_sphere"), dict) else 0, 8.0),
            _bool(isinstance(event_state, dict) and event_state.get("in_dialogue")),
        ]
    )
    return features


def extract_observation_features(state: dict[str, object]) -> list[float]:
    features = _extract_base_observation_features(state)
    features.extend(encode_semantic_observation(state).vector)
    return features


BASE_OBSERVATION_SIZE = len(_extract_base_observation_features({"state_type": "menu"}))
SEMANTIC_CONCEPT_SIZE = CONCEPT_VOCAB_SIZE
SEMANTIC_OBSERVATION_SIZE = SEMANTIC_VECTOR_SIZE
OBSERVATION_SIZE = BASE_OBSERVATION_SIZE + SEMANTIC_OBSERVATION_SIZE


class STS2MuZeroEnv:
    def __init__(
        self,
        bridge: STS2Bridge,
        poll_interval: float = 0.15,
        max_poll_attempts: int = 20,
        reward_discount: float = 0.997,
        reward_weights: RewardWeights | None = None,
        card_select_candidate_limit: int = 6,
        block_premature_end_turn: bool = True,
        end_turn_guard_stall_limit: int = 3,
        end_turn_guard_timeout_seconds: float = 3.0,
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
        self._disable_play_card_above_ten = False
        self._combat_stall_counts: dict[tuple[object, ...], int] = {}
        self._combat_end_turn_guard_started_at: dict[tuple[object, ...], float] = {}
        self.semantic_history = SemanticHistoryTracker()

    def fetch_state(self) -> dict[str, object]:
        return self.bridge.get_game_state(format="json")

    def extract_observation_features(self, state: dict[str, object]) -> list[float]:
        features = _extract_base_observation_features(state)
        features.extend(encode_semantic_observation(state, history=self.semantic_history).vector)
        return features

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
            (
                int(card.get("index", -1)),
                str(card.get("name", "")),
                bool(card.get("can_play")),
                str(card.get("cost", "")),
                str(card.get("target_type", "")),
            )
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
            hand_signature,
            enemy_signature,
            hand_select.get("mode"),
            len(selectable_cards),
            len(selected_cards),
            bool(hand_select.get("can_confirm")),
        )

    def _combat_has_living_enemies(self, state: dict[str, object]) -> bool:
        battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
        return any(isinstance(enemy, dict) and float(enemy.get("hp", 0) or 0) > 0.0 for enemy in battle.get("enemies", []))

    def _stall_count(self, state: dict[str, object]) -> int:
        return self._combat_stall_counts.get(self._combat_progress_signature(state), 0)

    def _clear_combat_guard_state(self, signature: tuple[object, ...]) -> None:
        self._combat_stall_counts.pop(signature, None)
        self._combat_end_turn_guard_started_at.pop(signature, None)

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

    def _can_discard_potion(self, state: dict[str, object], player: dict[str, object]) -> bool:
        return _potion_slots_filled(player) and _has_pending_potion_reward(state)

    def build_legal_action_map(self, state: dict[str, object]) -> dict[int, BoundAction]:
        actions: dict[int, BoundAction] = {}
        state_type = str(state.get("state_type", "unknown"))
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
            if allow_card_play:
                for card in hand_cards:
                    if not isinstance(card, dict) or not card.get("can_play"):
                        continue
                    card_index = card.get("index")
                    if not isinstance(card_index, int) or card_index > 9:
                        continue
                    card_name = str(card.get("name", f"card_{card_index}"))
                    if _is_enemy_target(card.get("target_type")) and enemies:
                        for target_slot, enemy in enumerate(enemies[:3]):
                            index = action_index("combat_play_card", card_index, target_slot)
                            actions[index] = BoundAction(
                                index,
                                "combat_play_card",
                                {"card_index": card_index, "target": str(enemy.get("entity_id", ""))},
                                f"play {card_name} -> {enemy.get('name', target_slot)}",
                            )
                    elif not _is_enemy_target(card.get("target_type")):
                        index = action_index("combat_play_card", card_index, -1)
                        actions[index] = BoundAction(index, "combat_play_card", {"card_index": card_index}, f"play {card_name}")
            if not self._should_block_combat_end_turn(state, actions):
                end_turn_index = action_index("combat_end_turn")
                actions[end_turn_index] = BoundAction(end_turn_index, "combat_end_turn", {}, "end turn")

        potions = player.get("potions", []) if isinstance(player.get("potions"), list) else []
        can_discard_potion = self._can_discard_potion(state, player)

        if self._is_combat_action_window_ready(state):
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

        self._append_indexed_actions(
            actions,
            state.get("hand_select", {}),
            "cards",
            10,
            "combat_select_card",
            "combat_select_card",
            "card_index",
            "combat select",
        )
        if isinstance(state.get("hand_select"), dict) and state["hand_select"].get("can_confirm"):
            confirm_index = action_index("combat_confirm_selection")
            actions[confirm_index] = BoundAction(confirm_index, "combat_confirm_selection", {}, "combat confirm")

        self._append_indexed_actions(actions, state.get("rewards", {}), "items", 5, "rewards_claim", "rewards_claim", "reward_index", "claim reward")
        if isinstance(state.get("rewards"), dict) and state["rewards"].get("can_proceed"):
            proceed_index = action_index("proceed_to_map")
            actions[proceed_index] = BoundAction(proceed_index, "proceed_to_map", {}, "proceed to map")

        self._append_indexed_actions(actions, state.get("card_reward", {}), "cards", 3, "rewards_pick_card", "rewards_pick_card", "card_index", "pick reward card")
        if isinstance(state.get("card_reward"), dict) and state["card_reward"].get("can_skip"):
            skip_index = action_index("rewards_skip_card")
            actions[skip_index] = BoundAction(skip_index, "rewards_skip_card", {}, "skip card reward")

        self._append_indexed_actions(actions, state.get("map", {}), "next_options", 6, "map_choose_node", "map_choose_node", "node_index", "map node")

        if isinstance(state.get("event"), dict):
            event_state = state["event"]
            if event_state.get("in_dialogue"):
                advance_index = action_index("event_advance_dialogue")
                actions[advance_index] = BoundAction(advance_index, "event_advance_dialogue", {}, "advance dialogue")
            else:
                for option in event_state.get("options", []) if isinstance(event_state.get("options"), list) else []:
                    if not isinstance(option, dict) or option.get("is_locked"):
                        continue
                    option_index = option.get("index")
                    if not isinstance(option_index, int) or option_index > 4:
                        continue
                    index = action_index("event_choose_option", option_index, None)
                    actions[index] = BoundAction(index, "event_choose_option", {"option_index": option_index}, f"event option {option_index}")

        if isinstance(state.get("rest_site"), dict):
            rest_state = state["rest_site"]
            for option in rest_state.get("options", []) if isinstance(rest_state.get("options"), list) else []:
                if not isinstance(option, dict) or not option.get("is_enabled"):
                    continue
                option_index = option.get("index")
                if not isinstance(option_index, int) or option_index > 4:
                    continue
                index = action_index("rest_choose_option", option_index, None)
                actions[index] = BoundAction(index, "rest_choose_option", {"option_index": option_index}, f"rest option {option_index}")
            if rest_state.get("can_proceed"):
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
                actions[index] = BoundAction(index, "shop_purchase", {"item_index": item_index}, f"shop purchase {item_index}")
            if state_type in {"shop", "fake_merchant"} and not shop_state.get("error"):
                proceed_index = action_index("proceed_to_map")
                actions[proceed_index] = BoundAction(proceed_index, "proceed_to_map", {}, "leave shop")

        self._append_indexed_actions(actions, state.get("treasure", {}), "relics", 3, "treasure_claim_relic", "treasure_claim_relic", "relic_index", "treasure relic")
        if isinstance(state.get("treasure"), dict) and state["treasure"].get("can_proceed"):
            proceed_index = action_index("proceed_to_map")
            actions[proceed_index] = BoundAction(proceed_index, "proceed_to_map", {}, "leave treasure")

        card_select_state = state.get("card_select", {}) if isinstance(state.get("card_select"), dict) else {}
        card_select_preview = bool(card_select_state.get("preview_showing")) if isinstance(card_select_state, dict) else False
        if not card_select_preview:
            self._append_card_select_actions(actions, card_select_state)
        if isinstance(card_select_state, dict) and card_select_state.get("can_confirm"):
            confirm_index = action_index("deck_confirm_selection")
            actions[confirm_index] = BoundAction(confirm_index, "deck_confirm_selection", {}, "deck confirm")
        if isinstance(card_select_state, dict) and (card_select_state.get("can_cancel") or card_select_state.get("can_skip")):
            cancel_index = action_index("deck_cancel_selection")
            actions[cancel_index] = BoundAction(cancel_index, "deck_cancel_selection", {}, "deck cancel")

        bundle_select_state = state.get("bundle_select", {}) if isinstance(state.get("bundle_select"), dict) else {}
        if not (isinstance(bundle_select_state, dict) and bundle_select_state.get("preview_showing")):
            self._append_indexed_actions(actions, bundle_select_state, "bundles", 3, "bundle_select", "bundle_select", "bundle_index", "bundle select")
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
        self._record_combat_action_outcome(state, next_state, bound.tool_name, response)
        boundaries = detect_transition_boundaries(state, next_state)
        reward, reward_breakdown = self._compute_reward(state, next_state, response, bound.tool_name, boundaries)
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
            actions[index] = BoundAction(index, tool_name, {argument_name: item_index}, f"{label_prefix} {item_index}")

    def _append_card_select_actions(self, actions: dict[int, BoundAction], card_select_state: object) -> None:
        if not isinstance(card_select_state, dict):
            return
        items = card_select_state.get("cards")
        if not isinstance(items, list):
            return
        allowed_indices = self._filtered_card_select_indices(card_select_state)
        for item in items:
            if not isinstance(item, dict):
                continue
            item_index = item.get("index")
            if not isinstance(item_index, int) or item_index >= 20 or item_index not in allowed_indices:
                continue
            index = action_index("deck_select_card", item_index, None)
            label = str(item.get("name", item_index))
            actions[index] = BoundAction(index, "deck_select_card", {"card_index": item_index}, f"deck select {label}")

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

    def _is_upgrade_like_card_select(self, card_select_state: dict[str, object]) -> bool:
        screen_type = str(card_select_state.get("screen_type", "")).lower()
        prompt = str(card_select_state.get("prompt", "")).lower()
        return any(token in screen_type for token in ("upgrade", "enchant")) or any(token in prompt for token in ("upgrade", "enchant", "imbue"))

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
        for _ in range(self.max_poll_attempts):
            if not self._needs_extra_poll(tool_name, previous_state, state):
                return state
            time.sleep(self.poll_interval)
            state = self.fetch_state()
        return state

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
        if state_type == "treasure" and isinstance(state.get("treasure"), dict):
            treasure = state["treasure"]
            return bool(treasure.get("message")) and not treasure.get("relics")
        if state_type in {"shop", "fake_merchant"}:
            return bool(self._extract_shop_state(state).get("error"))
        return False

    def _compute_reward(
        self,
        previous_state: dict[str, object],
        next_state: dict[str, object],
        response: dict[str, object],
        tool_name: str,
        boundaries: dict[str, object] | None = None,
    ) -> tuple[float, dict[str, float]]:
        if boundaries is None:
            boundaries = detect_transition_boundaries(previous_state, next_state)
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
            breakdown["floor_advance"] = self.reward_weights.floor_advance * floor_delta
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
        if gold_delta != 0.0:
            gold_scale = max(self.reward_weights.gold_delta_scale, 1.0)
            breakdown["gold_delta"] = self.reward_weights.gold_delta * (gold_delta / gold_scale)
            reward += breakdown["gold_delta"]

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
        return (0.80 * hp_ratio) + (0.35 * block_coverage) - (0.65 * normalized_enemy_hp) - (0.45 * unblocked_pressure)

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
