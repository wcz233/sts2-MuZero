import time
from dataclasses import dataclass

from .action_space import action_index
from .bridge import STS2Bridge

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


def _push_state_type(features: list[float], state_type: str) -> None:
    for candidate in STATE_TYPES:
        features.append(1.0 if candidate == state_type else 0.0)


def extract_observation_features(state: dict[str, object]) -> list[float]:
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


OBSERVATION_SIZE = len(extract_observation_features({"state_type": "menu"}))


class STS2MuZeroEnv:
    def __init__(self, bridge: STS2Bridge, poll_interval: float = 0.15, max_poll_attempts: int = 6) -> None:
        self.bridge = bridge
        self.poll_interval = poll_interval
        self.max_poll_attempts = max_poll_attempts

    def fetch_state(self) -> dict[str, object]:
        return self.bridge.get_game_state(format="json")

    def build_legal_action_map(self, state: dict[str, object]) -> dict[int, BoundAction]:
        actions: dict[int, BoundAction] = {}
        state_type = str(state.get("state_type", "unknown"))
        player = state.get("player") if isinstance(state.get("player"), dict) else {}
        battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}

        if state_type in COMBAT_STATE_TYPES and battle.get("turn") == "player" and battle.get("is_play_phase"):
            enemies = [
                enemy
                for enemy in battle.get("enemies", [])
                if isinstance(enemy, dict) and float(enemy.get("hp", 0) or 0) > 0
            ]
            for card in player.get("hand", []) if isinstance(player.get("hand"), list) else []:
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
            end_turn_index = action_index("combat_end_turn")
            actions[end_turn_index] = BoundAction(end_turn_index, "combat_end_turn", {}, "end turn")

        if state_type not in {"menu", "unknown", "overlay"}:
            enemies = [
                enemy
                for enemy in battle.get("enemies", [])
                if isinstance(enemy, dict) and float(enemy.get("hp", 0) or 0) > 0
            ]
            potions = player.get("potions", []) if isinstance(player.get("potions"), list) else []
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

        self._append_indexed_actions(actions, state.get("card_select", {}), "cards", 20, "deck_select_card", "deck_select_card", "card_index", "deck select")
        if isinstance(state.get("card_select"), dict) and state["card_select"].get("can_confirm"):
            confirm_index = action_index("deck_confirm_selection")
            actions[confirm_index] = BoundAction(confirm_index, "deck_confirm_selection", {}, "deck confirm")
        if isinstance(state.get("card_select"), dict) and (state["card_select"].get("can_cancel") or state["card_select"].get("can_skip")):
            cancel_index = action_index("deck_cancel_selection")
            actions[cancel_index] = BoundAction(cancel_index, "deck_cancel_selection", {}, "deck cancel")

        self._append_indexed_actions(actions, state.get("bundle_select", {}), "bundles", 3, "bundle_select", "bundle_select", "bundle_index", "bundle select")
        if isinstance(state.get("bundle_select"), dict) and state["bundle_select"].get("can_confirm"):
            confirm_index = action_index("bundle_confirm_selection")
            actions[confirm_index] = BoundAction(confirm_index, "bundle_confirm_selection", {}, "bundle confirm")
        if isinstance(state.get("bundle_select"), dict) and state["bundle_select"].get("can_cancel"):
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
        response = self.bridge.call_tool(bound.tool_name, **bound.kwargs)
        next_state = self._poll_state(bound.tool_name)
        reward = self._compute_reward(state, next_state, response)
        previous_state_type = str(state.get("state_type", "unknown"))
        next_state_type = str(next_state.get("state_type", "unknown"))
        done = previous_state_type not in TERMINAL_STATE_TYPES and next_state_type in TERMINAL_STATE_TYPES
        return next_state, reward, done, {"tool_name": bound.tool_name, "description": bound.description, "response": response}

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

    def _poll_state(self, tool_name: str) -> dict[str, object]:
        state = self.fetch_state()
        for _ in range(self.max_poll_attempts):
            if not self._needs_extra_poll(tool_name, state):
                return state
            time.sleep(self.poll_interval)
            state = self.fetch_state()
        return state

    def _needs_extra_poll(self, tool_name: str, state: dict[str, object]) -> bool:
        state_type = str(state.get("state_type", "unknown"))
        if tool_name == "combat_end_turn" and state_type in COMBAT_STATE_TYPES:
            battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
            return battle.get("turn") != "player" or not battle.get("is_play_phase")
        if state_type == "treasure" and isinstance(state.get("treasure"), dict):
            treasure = state["treasure"]
            return bool(treasure.get("message")) and not treasure.get("relics")
        if state_type in {"shop", "fake_merchant"}:
            return bool(self._extract_shop_state(state).get("error"))
        return False

    def _compute_reward(self, previous_state: dict[str, object], next_state: dict[str, object], response: dict[str, object]) -> float:
        reward = -1.0 if response.get("status") == "error" else 0.0
        previous_run = previous_state.get("run") if isinstance(previous_state.get("run"), dict) else {}
        next_run = next_state.get("run") if isinstance(next_state.get("run"), dict) else {}
        reward += 5.0 * max(0, int(next_run.get("floor", 0) or 0) - int(previous_run.get("floor", 0) or 0))
        reward += 20.0 * max(0, int(next_run.get("act", 0) or 0) - int(previous_run.get("act", 0) or 0))
        previous_player = previous_state.get("player") if isinstance(previous_state.get("player"), dict) else {}
        next_player = next_state.get("player") if isinstance(next_state.get("player"), dict) else {}
        reward += 0.20 * (float(next_player.get("hp", 0) or 0) - float(previous_player.get("hp", 0) or 0))
        reward += 0.02 * (float(next_player.get("gold", 0) or 0) - float(previous_player.get("gold", 0) or 0))
        previous_battle = previous_state.get("battle") if isinstance(previous_state.get("battle"), dict) else {}
        next_battle = next_state.get("battle") if isinstance(next_state.get("battle"), dict) else {}
        previous_enemy_hp = sum(float(enemy.get("hp", 0) or 0) for enemy in previous_battle.get("enemies", []) if isinstance(enemy, dict))
        next_enemy_hp = sum(float(enemy.get("hp", 0) or 0) for enemy in next_battle.get("enemies", []) if isinstance(enemy, dict))
        reward += 0.10 * (previous_enemy_hp - next_enemy_hp)
        if str(previous_state.get("state_type", "")) in COMBAT_STATE_TYPES and str(next_state.get("state_type", "")) not in COMBAT_STATE_TYPES and previous_enemy_hp > 0:
            reward += 8.0
        if str(previous_state.get("state_type", "unknown")) not in TERMINAL_STATE_TYPES and str(next_state.get("state_type", "unknown")) in TERMINAL_STATE_TYPES:
            reward += 10.0 if float(previous_player.get("hp", 0) or 0) > 0 else -10.0
        return reward
