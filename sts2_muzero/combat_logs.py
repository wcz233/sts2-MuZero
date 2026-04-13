import copy
import json
import lzma
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .action_space import action_spec
from .env import BoundAction, COMBAT_STATE_TYPES
from .muzero import Transition

LOG_SCHEMA = "sts2_muzero_episode_log/v2"
LOG_COMPRESSION = "json-delta+xz"
TERMINAL_STATE_TYPES = {"menu", "game_over"}
MAP_STATE_TYPES = {"map"}
COMBAT_NODE_STATE_TYPES = COMBAT_STATE_TYPES | {"hand_select", "rewards", "card_reward"}
QUESTION_MARK_ROUTE_TYPES = {"event", "unknown", "question_mark", "question", "questionmark"}
LIST_TOKEN_PREFIX = {
    "enemies": "enemy",
    "hand": "hand_card",
    "draw_pile": "draw_card",
    "discard_pile": "discard_card",
    "exhaust_pile": "exhaust_card",
    "relics": "relic",
    "potions": "potion",
    "items": "item",
    "options": "option",
    "cards": "card",
    "next_options": "route",
    "selected_cards": "selected_card",
}


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _timestamp_parts() -> tuple[str, str]:
    current = datetime.now().astimezone()
    return current.strftime("%Y%m%d"), current.strftime("%H%M%S")


def _sanitize_filename_part(value: object, fallback: str) -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    safe = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in text)
    while "__" in safe:
        safe = safe.replace("__", "_")
    safe = safe.strip("._-")
    return safe or fallback


def _normalize_text_token(value: object) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    if text == "?":
        return "question_mark"
    normalized = "".join(character if character.isalnum() else "_" for character in text)
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True, ensure_ascii=True).encode("utf-8")


def _normalize_json_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {
            str(key): _normalize_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, list):
        return [_normalize_json_value(item) for item in value]
    return str(value)


def _capture_global_state(state: dict[str, object]) -> dict[str, object]:
    return _normalize_json_value(state) if isinstance(state, dict) else {"state_type": "unknown"}


def _state_type(state: dict[str, object]) -> str:
    return str(state.get("state_type", "unknown"))


def _run_position(state: dict[str, object]) -> tuple[int, int]:
    run = state.get("run") if isinstance(state.get("run"), dict) else {}
    return int(run.get("act", 0) or 0), int(run.get("floor", 0) or 0)


def _character_name(state: dict[str, object], fallback: str) -> str:
    run = state.get("run") if isinstance(state.get("run"), dict) else {}
    player = state.get("player") if isinstance(state.get("player"), dict) else {}
    for candidate in (
        run.get("character"),
        run.get("character_name"),
        player.get("character"),
        player.get("name"),
        fallback,
    ):
        if candidate:
            return str(candidate)
    return fallback


def _stable_identity(value: object, index: int) -> str:
    if isinstance(value, dict):
        for key in (
            "entity_id",
            "card_id",
            "relic_id",
            "potion_id",
            "event_id",
            "monster_id",
            "combat_id",
            "slot",
            "index",
            "id",
            "name",
            "label",
            "type",
        ):
            item = value.get(key)
            if item not in (None, ""):
                return _sanitize_filename_part(item, f"idx{index}")
    return f"idx{index}"


def _list_token(parent_name: str, value: object, index: int) -> str:
    prefix = LIST_TOKEN_PREFIX.get(parent_name, parent_name[:-1] if parent_name.endswith("s") else parent_name or "item")
    return f"{prefix}[{_stable_identity(value, index)}@{index}]"


def _flatten_global_state(value: object, path: tuple[str, ...], output: dict[tuple[str, ...], object]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten_global_state(item, path + (str(key),), output)
        return
    if isinstance(value, list):
        output[path + ("__count__",)] = len(value)
        parent_name = path[-1] if path else "items"
        for index, item in enumerate(value):
            _flatten_global_state(item, path + (_list_token(parent_name, item, index),), output)
        return
    output[path] = value


def _format_subject(path: tuple[str, ...]) -> tuple[str, str]:
    if not path:
        return "state", "value"
    if path[-1] == "__count__":
        return ".".join(path[:-1]) or "state", "count"
    if len(path) == 1:
        return path[0], "value"
    return ".".join(path[:-1]), path[-1]


def _numeric_delta(before: object, after: object) -> float | None:
    if isinstance(before, bool) or isinstance(after, bool):
        return None
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        return float(after) - float(before)
    return None


def _build_state_change_list(previous_state: dict[str, object], next_state: dict[str, object]) -> list[dict[str, object]]:
    previous_flat: dict[tuple[str, ...], object] = {}
    next_flat: dict[tuple[str, ...], object] = {}
    _flatten_global_state(previous_state, (), previous_flat)
    _flatten_global_state(next_state, (), next_flat)
    changes: list[dict[str, object]] = []
    for path in sorted(set(previous_flat) | set(next_flat)):
        has_before = path in previous_flat
        has_after = path in next_flat
        before = previous_flat.get(path)
        after = next_flat.get(path)
        if has_before and has_after and before == after:
            continue
        subject, field = _format_subject(path)
        change: dict[str, object] = {"subject": subject, "field": field}
        if not has_before:
            change["kind"] = "added"
            change["after"] = after
        elif not has_after:
            change["kind"] = "removed"
            change["before"] = before
        else:
            change["kind"] = "updated"
            change["before"] = before
            change["after"] = after
            delta = _numeric_delta(before, after)
            if delta is not None:
                change["delta"] = delta
        changes.append(change)
    changes.sort(key=lambda item: (str(item.get("subject", "")), str(item.get("field", "")), str(item.get("kind", ""))))
    return changes


def _make_patch(previous: object, current: object) -> dict[str, object] | None:
    if previous == current:
        return None
    if type(previous) is not type(current):
        return {"op": "replace", "value": copy.deepcopy(current)}
    if isinstance(current, dict):
        changed: dict[str, object] = {}
        removed: list[str] = []
        previous_keys = set(previous)
        current_keys = set(current)
        for key in sorted(previous_keys - current_keys):
            removed.append(key)
        for key in sorted(current_keys):
            if key not in previous:
                changed[key] = {"op": "replace", "value": copy.deepcopy(current[key])}
                continue
            child = _make_patch(previous[key], current[key])
            if child is not None:
                changed[key] = child
        if not changed and not removed:
            return None
        patch: dict[str, object] = {"op": "dict"}
        if changed:
            patch["set"] = changed
        if removed:
            patch["del"] = removed
        return patch
    if isinstance(current, list):
        shared = min(len(previous), len(current))
        changed_items: dict[str, object] = {}
        for index in range(shared):
            child = _make_patch(previous[index], current[index])
            if child is not None:
                changed_items[str(index)] = child
        append_items = copy.deepcopy(current[shared:]) if len(current) > shared else []
        patch = {"op": "list", "size": len(current)}
        if changed_items:
            patch["set"] = changed_items
        if append_items:
            patch["append"] = append_items
        if len(current) == len(previous) and not changed_items:
            return None
        return patch
    return {"op": "replace", "value": copy.deepcopy(current)}


def _apply_patch(previous: object, patch: dict[str, object]) -> object:
    operation = str(patch.get("op", "replace"))
    if operation == "replace":
        return copy.deepcopy(patch.get("value"))
    if operation == "dict":
        result = copy.deepcopy(previous if isinstance(previous, dict) else {})
        for key in patch.get("del", []):
            if isinstance(key, str):
                result.pop(key, None)
        changed = patch.get("set", {})
        if isinstance(changed, dict):
            for key, child_patch in changed.items():
                if not isinstance(key, str) or not isinstance(child_patch, dict):
                    continue
                result[key] = _apply_patch(result.get(key), child_patch)
        return result
    if operation == "list":
        result = copy.deepcopy(previous if isinstance(previous, list) else [])
        target_size = int(patch.get("size", len(result)) or 0)
        if target_size < len(result):
            result = result[:target_size]
        changed = patch.get("set", {})
        if isinstance(changed, dict):
            for key, child_patch in changed.items():
                if not isinstance(key, str) or not isinstance(child_patch, dict):
                    continue
                index = int(key)
                if 0 <= index < len(result):
                    result[index] = _apply_patch(result[index], child_patch)
        append_items = patch.get("append", [])
        if isinstance(append_items, list):
            result.extend(copy.deepcopy(append_items))
        return result
    raise ValueError(f"Unknown patch operation: {operation}")


class StateStore:
    def __init__(self) -> None:
        self._entries: list[dict[str, object]] = []
        self._states: list[object] = []
        self._canonical: list[bytes] = []

    def add(self, state: object) -> int:
        canonical = _canonical_json_bytes(state)
        if self._canonical and canonical == self._canonical[-1]:
            return len(self._canonical) - 1
        state_copy = copy.deepcopy(state)
        if not self._states:
            entry = {"kind": "full", "value": state_copy}
        else:
            patch = _make_patch(self._states[-1], state_copy)
            if patch is None:
                return len(self._states) - 1
            patch_size = len(_canonical_json_bytes(patch))
            if patch_size < len(canonical):
                entry = {"kind": "patch", "base": len(self._states) - 1, "value": patch}
            else:
                entry = {"kind": "full", "value": state_copy}
        self._entries.append(entry)
        self._states.append(state_copy)
        self._canonical.append(canonical)
        return len(self._states) - 1

    def encoded_entries(self) -> list[dict[str, object]]:
        return copy.deepcopy(self._entries)

    @staticmethod
    def decode_entries(entries: list[dict[str, object]]) -> list[object]:
        states: list[object] = []
        for entry in entries:
            kind = str(entry.get("kind", "full"))
            if kind == "full":
                states.append(copy.deepcopy(entry.get("value")))
                continue
            if kind != "patch":
                raise ValueError(f"Unknown encoded state kind: {kind}")
            base_index = int(entry.get("base", len(states) - 1))
            if base_index < 0 or base_index >= len(states):
                raise ValueError(f"Invalid state base index: {base_index}")
            patch = entry.get("value")
            if not isinstance(patch, dict):
                raise ValueError("Invalid patch payload")
            states.append(_apply_patch(states[base_index], patch))
        return states


def _copy_json_dict(value: object) -> dict[str, object]:
    return copy.deepcopy(value) if isinstance(value, dict) else {}


def _extract_shop_state(state: dict[str, object]) -> dict[str, object]:
    state_type = _state_type(state)
    if state_type == "fake_merchant":
        fake_merchant = state.get("fake_merchant")
        if isinstance(fake_merchant, dict):
            nested_shop = fake_merchant.get("shop")
            if isinstance(nested_shop, dict):
                return nested_shop
    shop_state = state.get("shop")
    return shop_state if isinstance(shop_state, dict) else {}


def _extract_map_choice(state: dict[str, object], node_index: int | None) -> dict[str, object] | None:
    if node_index is None:
        return None
    map_state = state.get("map")
    if not isinstance(map_state, dict):
        return None
    options = map_state.get("next_options")
    if not isinstance(options, list):
        return None
    for option in options:
        if not isinstance(option, dict):
            continue
        if option.get("index") == node_index:
            return copy.deepcopy(option)
    return None


def _map_route_type(route_choice: dict[str, object] | None) -> str:
    if not isinstance(route_choice, dict):
        return ""
    for key in ("type", "node_type", "kind", "label"):
        value = route_choice.get(key)
        if value:
            return str(value)
    return ""


def _normalized_route_type(route_choice: dict[str, object] | None) -> str:
    return _normalize_text_token(_map_route_type(route_choice))


def _is_question_mark_route(route_choice: dict[str, object] | None) -> bool:
    route_type = _normalized_route_type(route_choice)
    return route_type in QUESTION_MARK_ROUTE_TYPES


def _infer_resolved_type(state: dict[str, object], route_choice: dict[str, object] | None) -> str:
    state_type = _state_type(state)
    route_type = _normalized_route_type(route_choice)
    if state_type in COMBAT_STATE_TYPES | {"hand_select"}:
        return "combat"
    if state_type in {"shop", "fake_merchant"}:
        return "shop"
    if state_type == "rest_site":
        return "rest_site"
    if state_type in {"treasure", "relic_select"}:
        return "treasure"
    if state_type == "event":
        return "event"
    if state_type in {"rewards", "card_reward"}:
        if route_type in {"monster", "elite", "boss"} or _is_question_mark_route(route_choice):
            return "combat"
        return "reward"
    if state_type == "card_select":
        if route_type in {"restsite", "rest_site"}:
            return "rest_site"
        return "selection"
    if state_type in {"bundle_select", "crystal_sphere"}:
        return "selection"
    if state_type in TERMINAL_STATE_TYPES | MAP_STATE_TYPES:
        return "none"
    return state_type


def _snapshot_player(state: dict[str, object]) -> dict[str, object]:
    player = state.get("player")
    return _normalize_json_value(player) if isinstance(player, dict) else {}


def _snapshot_battle(state: dict[str, object]) -> dict[str, object]:
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    return {"battle": _normalize_json_value(battle), "player": _snapshot_player(state)}


def _snapshot_rewards(state: dict[str, object]) -> dict[str, object]:
    return {
        "rewards": _normalize_json_value(state.get("rewards")) if isinstance(state.get("rewards"), dict) else {},
        "card_reward": _normalize_json_value(state.get("card_reward")) if isinstance(state.get("card_reward"), dict) else {},
    }


def _snapshot_shop(state: dict[str, object]) -> dict[str, object]:
    return {"shop": _normalize_json_value(_extract_shop_state(state)), "player": _snapshot_player(state)}


def _snapshot_rest_site(state: dict[str, object]) -> dict[str, object]:
    return {
        "rest_site": _normalize_json_value(state.get("rest_site")) if isinstance(state.get("rest_site"), dict) else {},
        "player": _snapshot_player(state),
    }


def _snapshot_event(state: dict[str, object]) -> dict[str, object]:
    return {
        "event": _normalize_json_value(state.get("event")) if isinstance(state.get("event"), dict) else {},
        "player": _snapshot_player(state),
    }


def _snapshot_treasure(state: dict[str, object]) -> dict[str, object]:
    return {
        "treasure": _normalize_json_value(state.get("treasure")) if isinstance(state.get("treasure"), dict) else {},
        "player": _snapshot_player(state),
    }


def _snapshot_selection(state: dict[str, object]) -> dict[str, object]:
    snapshot: dict[str, object] = {"player": _snapshot_player(state)}
    for key in ("card_select", "bundle_select", "relic_select", "crystal_sphere", "hand_select"):
        value = state.get(key)
        if isinstance(value, dict):
            snapshot[key] = _normalize_json_value(value)
    return snapshot


def _build_node_details(kind: str, state: dict[str, object]) -> dict[str, object]:
    if kind == "combat":
        return {"entry_snapshot": _snapshot_battle(state), "turns": [], "reward_actions": [], "result": None}
    if kind == "shop":
        return {"entry_snapshot": _snapshot_shop(state), "purchases": [], "leave_actions": []}
    if kind == "rest_site":
        return {"entry_snapshot": _snapshot_rest_site(state), "choices": []}
    if kind == "event":
        entry_event = state.get("event") if isinstance(state.get("event"), dict) else {}
        return {
            "entry_snapshot": _snapshot_event(state),
            "event_id": entry_event.get("event_id"),
            "event_name": entry_event.get("event_name"),
            "choices": [],
            "dialogue_steps": [],
        }
    if kind == "treasure":
        return {"entry_snapshot": _snapshot_treasure(state), "claimed_items": []}
    if kind == "selection":
        return {"entry_snapshot": _snapshot_selection(state), "choices": []}
    if kind == "reward":
        return {"entry_snapshot": _snapshot_rewards(state), "choices": []}
    return {"entry_snapshot": _capture_global_state(state)}


def _combat_turn_marker(state: dict[str, object]) -> tuple[int, str] | None:
    battle = state.get("battle")
    if not isinstance(battle, dict):
        return None
    round_index = int(battle.get("round", 0) or 0)
    side = str(battle.get("turn", "unknown") or "unknown")
    return round_index, side


def _append_turn_step(details: dict[str, object], state: dict[str, object], action_record: dict[str, object]) -> None:
    marker = _combat_turn_marker(state)
    turns = details.get("turns")
    if not isinstance(turns, list):
        return
    global_index = int(action_record.get("global_index", 0))
    state_before_id = int(action_record.get("state_before_id", -1))
    state_after_id = int(action_record.get("state_after_id", -1))
    entry_state_type = str(action_record.get("state_type", "unknown"))
    final_state_type = str(action_record.get("next_state_type", "unknown"))
    if marker is None:
        if turns and turns[-1].get("phase") == "post_combat":
            turns[-1]["step_indices"].append(global_index)
            turns[-1]["state_after_id"] = state_after_id
            turns[-1]["final_state_type"] = final_state_type
            return
        turns.append(
            {
                "phase": "post_combat",
                "step_indices": [global_index],
                "state_before_id": state_before_id,
                "state_after_id": state_after_id,
                "entry_state_type": entry_state_type,
                "final_state_type": final_state_type,
            }
        )
        return
    round_index, side = marker
    if turns and turns[-1].get("round") == round_index and turns[-1].get("side") == side:
        turns[-1]["step_indices"].append(global_index)
        turns[-1]["state_after_id"] = state_after_id
        turns[-1]["final_state_type"] = final_state_type
        return
    turns.append(
        {
            "round": round_index,
            "side": side,
            "step_indices": [global_index],
            "state_before_id": state_before_id,
            "state_after_id": state_after_id,
            "entry_state_type": entry_state_type,
            "final_state_type": final_state_type,
        }
    )


def _selected_item_snapshot(container: dict[str, object], list_key: str, index_value: object) -> dict[str, object] | None:
    items = container.get(list_key)
    if not isinstance(items, list):
        return None
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("index") == index_value:
            return copy.deepcopy(item)
    return None


def _selection_choice_snapshot(
    previous_state: dict[str, object],
    action_kind: str,
    action: dict[str, object],
    action_kwargs: dict[str, object],
) -> dict[str, object] | None:
    container: dict[str, object] | None = None
    list_key = ""
    index_key = ""
    if action_kind == "rewards_claim":
        container = previous_state.get("rewards") if isinstance(previous_state.get("rewards"), dict) else None
        list_key = "items"
        index_key = "reward_index"
    elif action_kind == "rewards_pick_card":
        container = previous_state.get("card_reward") if isinstance(previous_state.get("card_reward"), dict) else None
        list_key = "cards"
        index_key = "card_index"
    elif action_kind == "deck_select_card":
        container = previous_state.get("card_select") if isinstance(previous_state.get("card_select"), dict) else None
        list_key = "cards"
        index_key = "card_index"
    elif action_kind == "bundle_select":
        container = previous_state.get("bundle_select") if isinstance(previous_state.get("bundle_select"), dict) else None
        list_key = "bundles"
        index_key = "bundle_index"
    elif action_kind == "relic_select":
        container = previous_state.get("relic_select") if isinstance(previous_state.get("relic_select"), dict) else None
        list_key = "relics"
        index_key = "relic_index"
    if container is None:
        return None
    return _selected_item_snapshot(container, list_key, action_kwargs.get(index_key))


def _event_choice_snapshot(previous_state: dict[str, object], action: dict[str, object]) -> dict[str, object] | None:
    event_state = previous_state.get("event")
    if not isinstance(event_state, dict):
        return None
    raw_index = action.get("primary")
    if isinstance(raw_index, int):
        selected = _selected_item_snapshot(event_state, "options", raw_index)
        if selected is not None:
            return selected
    options = event_state.get("options")
    if not isinstance(options, list):
        return None
    unlocked_options = [option for option in options if isinstance(option, dict) and not option.get("is_locked")]
    unlocked_index = action.get("kwargs", {}).get("option_index") if isinstance(action.get("kwargs"), dict) else None
    if isinstance(unlocked_index, int) and 0 <= unlocked_index < len(unlocked_options):
        return copy.deepcopy(unlocked_options[unlocked_index])
    return None


def _update_node_details(
    node: dict[str, object],
    previous_state: dict[str, object],
    next_state: dict[str, object],
    action_record: dict[str, object],
) -> None:
    details = node.get("details")
    if not isinstance(details, dict):
        return
    kind = str(node.get("details_kind", "unknown"))
    action = action_record.get("action") if isinstance(action_record.get("action"), dict) else {}
    action_kind = str(action.get("kind", ""))
    action_kwargs = action.get("kwargs") if isinstance(action.get("kwargs"), dict) else {}
    global_index = int(action_record.get("global_index", 0))
    previous_type = _state_type(previous_state)
    if kind == "combat":
        if previous_type in COMBAT_NODE_STATE_TYPES:
            _append_turn_step(details, previous_state, action_record)
        if action_kind in {"rewards_claim", "rewards_pick_card", "rewards_skip_card", "proceed_to_map"}:
            reward_actions = details.get("reward_actions")
            if isinstance(reward_actions, list):
                reward_actions.append({"step_index": global_index, "action": copy.deepcopy(action)})
        boundaries = action_record.get("boundaries") if isinstance(action_record.get("boundaries"), dict) else {}
        if boundaries.get("combat_end"):
            player = next_state.get("player") if isinstance(next_state.get("player"), dict) else {}
            details["result"] = "victory" if float(player.get("hp", 0) or 0.0) > 0.0 else "defeat"
            details["post_combat_snapshot"] = _snapshot_rewards(next_state)
        return
    if kind == "shop":
        if action_kind == "shop_purchase":
            selected = _selected_item_snapshot(_extract_shop_state(previous_state), "items", action_kwargs.get("item_index"))
            purchases = details.get("purchases")
            if isinstance(purchases, list):
                purchases.append({"step_index": global_index, "item": selected, "action": copy.deepcopy(action)})
        if action_kind == "proceed_to_map":
            leave_actions = details.get("leave_actions")
            if isinstance(leave_actions, list):
                leave_actions.append({"step_index": global_index, "action": copy.deepcopy(action)})
        return
    if kind == "rest_site":
        if action_kind == "rest_choose_option":
            rest_state = previous_state.get("rest_site") if isinstance(previous_state.get("rest_site"), dict) else {}
            selected = _selected_item_snapshot(rest_state, "options", action_kwargs.get("option_index"))
            choices = details.get("choices")
            if isinstance(choices, list):
                choices.append({"step_index": global_index, "choice": selected, "action": copy.deepcopy(action)})
        return
    if kind == "event":
        if action_kind == "event_choose_option":
            selected = _event_choice_snapshot(previous_state, action)
            choices = details.get("choices")
            if isinstance(choices, list):
                choices.append({"step_index": global_index, "choice": selected, "action": copy.deepcopy(action)})
        if action_kind == "event_advance_dialogue":
            dialogue_steps = details.get("dialogue_steps")
            if isinstance(dialogue_steps, list):
                dialogue_steps.append(global_index)
        return
    if kind == "treasure":
        if action_kind == "treasure_claim_relic":
            treasure_state = previous_state.get("treasure") if isinstance(previous_state.get("treasure"), dict) else {}
            selected = _selected_item_snapshot(treasure_state, "relics", action_kwargs.get("relic_index"))
            claimed_items = details.get("claimed_items")
            if isinstance(claimed_items, list):
                claimed_items.append({"step_index": global_index, "item": selected, "action": copy.deepcopy(action)})
        return
    if kind in {"selection", "reward"} and action_kind:
        choices = details.get("choices")
        if isinstance(choices, list):
            choices.append(
                {
                    "step_index": global_index,
                    "choice": _selection_choice_snapshot(previous_state, action_kind, action, action_kwargs),
                    "action": copy.deepcopy(action),
                }
            )


@dataclass
class StepBundle:
    total_step: int
    episode_step: int
    previous_state: dict[str, object]
    next_state: dict[str, object]
    action_index: int
    action_kwargs: dict[str, object]
    description: str
    tool_name: str
    reward: float
    raw_reward: float
    credit_adjustment: float
    response: dict[str, object]
    reward_breakdown: dict[str, object]
    boundaries: dict[str, object]
    changes: list[dict[str, object]]


@dataclass
class ActiveEpisodeArchive:
    episode_index: int
    character: str
    model_name: str
    started_at: str
    start_total_step: int
    initial_state_id: int
    state_store: StateStore = field(default_factory=StateStore)
    nodes: list[dict[str, object]] = field(default_factory=list)
    current_node_index: int | None = None
    pending_map_choice: dict[str, object] | None = None
    highest_floor: int = 0
    max_act: int = 0


class EpisodeArchiveRecorder:
    def __init__(
        self,
        log_dir: Path | None,
        model_name: str,
        checkpoint_path: Path,
        fallback_character: str = "Unknown",
    ) -> None:
        self.log_dir = log_dir
        self.model_name = _sanitize_filename_part(model_name, "model")
        self.checkpoint_path = checkpoint_path
        self.fallback_character = fallback_character
        self.active: ActiveEpisodeArchive | None = None
        self.last_written_path: Path | None = None
        self.next_episode_index = 1

    @property
    def enabled(self) -> bool:
        return self.log_dir is not None

    def start_episode(self, state: dict[str, object], total_step: int, episode_index: int) -> None:
        if not self.enabled or self.active is not None:
            return
        self.next_episode_index = max(1, int(episode_index))
        initial_state = _capture_global_state(state)
        state_store = StateStore()
        initial_state_id = state_store.add(initial_state)
        act, floor = _run_position(state)
        self.active = ActiveEpisodeArchive(
            episode_index=episode_index,
            character=_character_name(state, self.fallback_character),
            model_name=self.model_name,
            started_at=_now_iso(),
            start_total_step=total_step,
            initial_state_id=initial_state_id,
            state_store=state_store,
            highest_floor=max(0, floor),
            max_act=max(0, act),
        )

    def record_transition(
        self,
        previous_state: dict[str, object],
        next_state: dict[str, object],
        transition: Transition,
        bound_action: BoundAction,
        info: dict[str, object],
        total_step: int,
        episode_step: int,
        episode_transitions: list[Transition] | None = None,
    ) -> Path | None:
        if not self.enabled:
            return None
        if self.active is None:
            self.start_episode(previous_state, total_step, self.next_episode_index)
        if self.active is None:
            return None
        active = self.active
        self._update_episode_position(active, previous_state, next_state)
        self._capture_map_choice(active, previous_state, bound_action)
        self._ensure_node_for_transition(active, previous_state, next_state, bound_action)
        bundle = self._build_step_bundle(previous_state, next_state, transition, bound_action, info, total_step, episode_step)
        self._append_step(active, bundle)
        self._maybe_close_node(active, next_state)
        if transition.done:
            if episode_transitions is not None:
                self._synchronize_transition_rewards(active, episode_transitions)
            return self._finalize_episode(next_state, total_step, episode_step)
        return None

    def reset(self) -> None:
        self.active = None

    def _update_episode_position(self, active: ActiveEpisodeArchive, previous_state: dict[str, object], next_state: dict[str, object]) -> None:
        for state in (previous_state, next_state):
            act, floor = _run_position(state)
            active.highest_floor = max(active.highest_floor, max(0, floor))
            active.max_act = max(active.max_act, max(0, act))

    def _capture_map_choice(self, active: ActiveEpisodeArchive, previous_state: dict[str, object], bound_action: BoundAction) -> None:
        if bound_action.tool_name != "map_choose_node":
            return
        node_index = bound_action.kwargs.get("node_index")
        active.pending_map_choice = _extract_map_choice(previous_state, int(node_index)) if isinstance(node_index, int) else None

    def _ensure_node_for_transition(
        self,
        active: ActiveEpisodeArchive,
        previous_state: dict[str, object],
        next_state: dict[str, object],
        bound_action: BoundAction,
    ) -> None:
        if active.current_node_index is not None:
            return
        candidate_state: dict[str, object] | None = None
        previous_type = _state_type(previous_state)
        next_type = _state_type(next_state)
        if previous_type not in TERMINAL_STATE_TYPES | MAP_STATE_TYPES:
            candidate_state = previous_state
        elif bound_action.tool_name == "map_choose_node" and next_type not in TERMINAL_STATE_TYPES | MAP_STATE_TYPES:
            candidate_state = next_state
        elif next_type not in TERMINAL_STATE_TYPES | MAP_STATE_TYPES and _run_position(next_state)[1] > 0:
            candidate_state = next_state
        if candidate_state is None:
            return
        act, floor = _run_position(candidate_state)
        route_choice = copy.deepcopy(active.pending_map_choice)
        resolved_type = _infer_resolved_type(candidate_state, route_choice)
        node_type = "question_mark" if _is_question_mark_route(route_choice) else resolved_type
        details_kind = resolved_type if node_type == "question_mark" else node_type
        node = {
            "node_index": len(active.nodes),
            "act": act,
            "floor": floor,
            "type": node_type,
            "resolved_type": resolved_type,
            "details_kind": details_kind,
            "entry_state_type": _state_type(candidate_state),
            "entry_global_state_id": None,
            "final_global_state_id": None,
            "route_choice": route_choice,
            "started_at": _now_iso(),
            "ended_at": None,
            "actions": [],
            "details": _build_node_details(details_kind, candidate_state),
            "summary": {},
        }
        active.nodes.append(node)
        active.current_node_index = len(active.nodes) - 1
        active.pending_map_choice = None

    def _build_step_bundle(
        self,
        previous_state: dict[str, object],
        next_state: dict[str, object],
        transition: Transition,
        bound_action: BoundAction,
        info: dict[str, object],
        total_step: int,
        episode_step: int,
    ) -> StepBundle:
        previous_global_state = _capture_global_state(previous_state)
        next_global_state = _capture_global_state(next_state)
        boundaries = {
            key: copy.deepcopy(value)
            for key, value in info.items()
            if key not in {"tool_name", "description", "response", "reward_breakdown"}
        }
        return StepBundle(
            total_step=total_step,
            episode_step=episode_step,
            previous_state=previous_global_state,
            next_state=next_global_state,
            action_index=transition.action_index,
            action_kwargs=copy.deepcopy(bound_action.kwargs),
            description=str(info.get("description", bound_action.description)),
            tool_name=str(info.get("tool_name", bound_action.tool_name)),
            reward=transition.reward,
            raw_reward=transition.raw_reward,
            credit_adjustment=transition.credit_adjustment,
            response=_copy_json_dict(info.get("response")),
            reward_breakdown=_copy_json_dict(info.get("reward_breakdown")),
            boundaries=boundaries,
            changes=_build_state_change_list(previous_global_state, next_global_state),
        )

    def _append_step(self, active: ActiveEpisodeArchive, bundle: StepBundle) -> None:
        if active.current_node_index is None:
            return
        node = active.nodes[active.current_node_index]
        previous_state_id = active.state_store.add(bundle.previous_state)
        next_state_id = active.state_store.add(bundle.next_state)
        spec = action_spec(bundle.action_index)
        actions = node.get("actions")
        if not isinstance(actions, list):
            return
        if node.get("entry_global_state_id") is None:
            node["entry_global_state_id"] = previous_state_id
        node["final_global_state_id"] = next_state_id
        action_record = {
            "global_index": sum(len(item.get("actions", [])) for item in active.nodes),
            "total_step": bundle.total_step,
            "episode_step": bundle.episode_step,
            "state_before_id": previous_state_id,
            "state_after_id": next_state_id,
            "state_type": _state_type(bundle.previous_state) if isinstance(bundle.previous_state, dict) else "unknown",
            "next_state_type": _state_type(bundle.next_state) if isinstance(bundle.next_state, dict) else "unknown",
            "action": {
                "action_index": bundle.action_index,
                "kind": spec.kind,
                "label": spec.label,
                "primary": spec.primary,
                "secondary": spec.secondary,
                "tool_name": bundle.tool_name,
                "description": bundle.description,
                "kwargs": copy.deepcopy(bundle.action_kwargs),
            },
            "reward": {
                "final": bundle.reward,
                "raw": bundle.raw_reward,
                "credit_adjustment": bundle.credit_adjustment,
                "breakdown": copy.deepcopy(bundle.reward_breakdown),
            },
            "response": copy.deepcopy(bundle.response),
            "boundaries": copy.deepcopy(bundle.boundaries),
            "changes": copy.deepcopy(bundle.changes),
        }
        actions.append(action_record)
        _update_node_details(node, bundle.previous_state, bundle.next_state, action_record)

    def _maybe_close_node(self, active: ActiveEpisodeArchive, next_state: dict[str, object]) -> None:
        if active.current_node_index is None:
            return
        next_type = _state_type(next_state)
        if next_type not in TERMINAL_STATE_TYPES | MAP_STATE_TYPES:
            return
        node = active.nodes[active.current_node_index]
        node["ended_at"] = _now_iso()
        self._populate_node_summary(node)
        active.current_node_index = None

    def _populate_node_summary(self, node: dict[str, object]) -> None:
        actions = node.get("actions") if isinstance(node.get("actions"), list) else []
        raw_reward = 0.0
        final_reward = 0.0
        for action_record in actions:
            reward = action_record.get("reward") if isinstance(action_record.get("reward"), dict) else {}
            raw_reward += float(reward.get("raw", 0.0) or 0.0)
            final_reward += float(reward.get("final", 0.0) or 0.0)
        summary = node.get("summary") if isinstance(node.get("summary"), dict) else {}
        summary["action_count"] = len(actions)
        summary["raw_reward"] = raw_reward
        summary["final_reward"] = final_reward
        details_kind = str(node.get("details_kind", ""))
        details = node.get("details") if isinstance(node.get("details"), dict) else {}
        if details_kind == "combat":
            summary["result"] = details.get("result")
            turns = details.get("turns") if isinstance(details.get("turns"), list) else []
            summary["turn_count"] = len(turns)
        elif details_kind == "shop":
            purchases = details.get("purchases") if isinstance(details.get("purchases"), list) else []
            summary["purchase_count"] = len(purchases)
        elif details_kind == "rest_site":
            choices = details.get("choices") if isinstance(details.get("choices"), list) else []
            summary["choice_count"] = len(choices)
        elif details_kind == "event":
            choices = details.get("choices") if isinstance(details.get("choices"), list) else []
            dialogue_steps = details.get("dialogue_steps") if isinstance(details.get("dialogue_steps"), list) else []
            summary["choice_count"] = len(choices)
            summary["dialogue_step_count"] = len(dialogue_steps)
        elif details_kind == "treasure":
            claimed_items = details.get("claimed_items") if isinstance(details.get("claimed_items"), list) else []
            summary["claimed_item_count"] = len(claimed_items)
        elif details_kind in {"selection", "reward"}:
            choices = details.get("choices") if isinstance(details.get("choices"), list) else []
            summary["choice_count"] = len(choices)
        node["summary"] = summary

    def _synchronize_transition_rewards(self, active: ActiveEpisodeArchive, transitions: list[Transition]) -> None:
        if not transitions:
            return
        action_records: list[dict[str, object]] = []
        for node in active.nodes:
            actions = node.get("actions")
            if not isinstance(actions, list):
                continue
            for action_record in actions:
                if isinstance(action_record, dict):
                    action_records.append(action_record)
        for action_record, transition in zip(action_records, transitions):
            reward = action_record.get("reward") if isinstance(action_record.get("reward"), dict) else {}
            reward["final"] = transition.reward
            reward["raw"] = transition.raw_reward
            reward["credit_adjustment"] = transition.credit_adjustment
            action_record["reward"] = reward
        for node in active.nodes:
            self._populate_node_summary(node)

    def _finalize_episode(self, final_state: dict[str, object], total_step: int, episode_step: int) -> Path | None:
        if not self.enabled or self.active is None:
            return None
        active = self.active
        if active.current_node_index is not None:
            node = active.nodes[active.current_node_index]
            node["ended_at"] = _now_iso()
            self._populate_node_summary(node)
            active.current_node_index = None
        final_state_id = active.state_store.add(_capture_global_state(final_state))
        player = final_state.get("player") if isinstance(final_state.get("player"), dict) else {}
        result = "victory" if float(player.get("hp", 0) or 0.0) > 0.0 else "defeat"
        node_counts: dict[str, int] = {}
        total_actions = 0
        raw_reward = 0.0
        final_reward = 0.0
        for node in active.nodes:
            node_type = str(node.get("type", "unknown"))
            node_counts[node_type] = node_counts.get(node_type, 0) + 1
            total_actions += len(node.get("actions", [])) if isinstance(node.get("actions"), list) else 0
            summary = node.get("summary") if isinstance(node.get("summary"), dict) else {}
            raw_reward += float(summary.get("raw_reward", 0.0) or 0.0)
            final_reward += float(summary.get("final_reward", 0.0) or 0.0)
        payload = {
            "schema": LOG_SCHEMA,
            "compression": LOG_COMPRESSION,
            "metadata": {
                "episode_index": active.episode_index,
                "character": active.character,
                "model_name": active.model_name,
                "checkpoint_path": str(self.checkpoint_path),
                "started_at": active.started_at,
                "ended_at": _now_iso(),
                "start_total_step": active.start_total_step,
                "end_total_step": total_step,
                "end_episode_step": episode_step,
                "highest_floor": active.highest_floor,
                "max_act": active.max_act,
                "initial_global_state_id": active.initial_state_id,
                "final_global_state_id": final_state_id,
                "result": result,
                "final_state_type": _state_type(final_state),
            },
            "summary": {
                "total_nodes": len(active.nodes),
                "total_steps": total_actions,
                "highest_floor": active.highest_floor,
                "raw_reward": raw_reward,
                "final_reward": final_reward,
                "node_counts": node_counts,
                "result": result,
            },
            "global_states": active.state_store.encoded_entries(),
            "nodes": active.nodes,
        }
        path = self._build_log_path(active.character, active.highest_floor, active.episode_index, active.model_name)
        self._write_archive(path, payload)
        self.last_written_path = path
        self.active = None
        self.next_episode_index = active.episode_index + 1
        return path

    def _build_log_path(self, character: str, highest_floor: int, episode_index: int, model_name: str) -> Path:
        if self.log_dir is None:
            raise ValueError("Log directory is disabled")
        directory = self.log_dir
        directory.mkdir(parents=True, exist_ok=True)
        date_slug, time_slug = _timestamp_parts()
        stem = (
            f"{_sanitize_filename_part(character, self.fallback_character)}-"
            f"{max(0, int(highest_floor))}-"
            f"{max(1, int(episode_index))}-"
            f"{date_slug}-"
            f"{time_slug}-"
            f"{_sanitize_filename_part(model_name, 'model')}"
        )
        path = directory / f"{stem}.logs"
        counter = 1
        while path.exists():
            path = directory / f"{stem}-{counter:02d}.logs"
            counter += 1
        return path

    def _write_archive(self, path: Path, payload: dict[str, object]) -> None:
        raw = json.dumps(payload, separators=(",", ":"), sort_keys=True, ensure_ascii=True).encode("utf-8")
        compressed = lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME, format=lzma.FORMAT_XZ)
        path.write_bytes(compressed)


def load_episode_archive(path: str | Path) -> dict[str, object]:
    archive_path = Path(path)
    raw = lzma.decompress(archive_path.read_bytes(), format=lzma.FORMAT_XZ)
    payload = json.loads(raw.decode("utf-8"))
    if str(payload.get("schema")) != LOG_SCHEMA:
        raise ValueError(f"Unsupported archive schema: {payload.get('schema')}")
    return payload


def decode_episode_states(payload: dict[str, object]) -> list[object]:
    entries = payload.get("global_states")
    if not isinstance(entries, list):
        raise ValueError("Archive does not contain encoded global states")
    return StateStore.decode_entries(entries)


def expand_episode_archive(payload: dict[str, object]) -> dict[str, object]:
    expanded = copy.deepcopy(payload)
    expanded["decoded_global_states"] = decode_episode_states(payload)
    return expanded


load_battle_archive = load_episode_archive
decode_battle_states = decode_episode_states
expand_battle_archive = expand_episode_archive
