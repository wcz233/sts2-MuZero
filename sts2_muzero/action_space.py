from dataclasses import dataclass


@dataclass(frozen=True)
class ActionSpec:
    index: int
    kind: str
    primary: int | None
    secondary: int | None
    label: str


ACTION_SPECS: list[ActionSpec] = []
ACTION_INDEX: dict[tuple[str, int | None, int | None], int] = {}


def _add(kind: str, primary: int | None = None, secondary: int | None = None, label: str | None = None) -> None:
    index = len(ACTION_SPECS)
    spec = ActionSpec(
        index=index,
        kind=kind,
        primary=primary,
        secondary=secondary,
        label=label or kind,
    )
    ACTION_SPECS.append(spec)
    ACTION_INDEX[(kind, primary, secondary)] = index


_add("combat_end_turn", label="end_turn")
_add("proceed_to_map", label="proceed")
_add("event_advance_dialogue", label="advance_dialogue")

for card_index in range(10):
    _add("combat_play_card", card_index, -1, f"play_card:{card_index}")
    for target_slot in range(3):
        _add("combat_play_card", card_index, target_slot, f"play_card:{card_index}:{target_slot}")

for slot in range(3):
    _add("use_potion", slot, -1, f"use_potion:{slot}")
    for target_slot in range(3):
        _add("use_potion", slot, target_slot, f"use_potion:{slot}:{target_slot}")
    _add("discard_potion", slot, None, f"discard_potion:{slot}")

for reward_index in range(5):
    _add("rewards_claim", reward_index, None, f"claim_reward:{reward_index}")

for card_index in range(3):
    _add("rewards_pick_card", card_index, None, f"pick_card_reward:{card_index}")

_add("rewards_skip_card", label="skip_card_reward")

for node_index in range(6):
    _add("map_choose_node", node_index, None, f"choose_map_node:{node_index}")

for option_index in range(5):
    _add("event_choose_option", option_index, None, f"event_option:{option_index}")
    _add("rest_choose_option", option_index, None, f"rest_option:{option_index}")

for item_index in range(12):
    _add("shop_purchase", item_index, None, f"shop_purchase:{item_index}")

for card_index in range(10):
    _add("combat_select_card", card_index, None, f"combat_select_card:{card_index}")

_add("combat_confirm_selection", label="combat_confirm_selection")

for card_index in range(20):
    _add("deck_select_card", card_index, None, f"deck_select_card:{card_index}")

_add("deck_confirm_selection", label="deck_confirm_selection")
_add("deck_cancel_selection", label="deck_cancel_selection")

for bundle_index in range(3):
    _add("bundle_select", bundle_index, None, f"bundle_select:{bundle_index}")

_add("bundle_confirm_selection", label="bundle_confirm_selection")
_add("bundle_cancel_selection", label="bundle_cancel_selection")

for relic_index in range(3):
    _add("relic_select", relic_index, None, f"relic_select:{relic_index}")
    _add("treasure_claim_relic", relic_index, None, f"treasure_relic:{relic_index}")

_add("relic_skip", label="relic_skip")
_add("crystal_sphere_set_tool", 0, None, "crystal_tool_big")
_add("crystal_sphere_set_tool", 1, None, "crystal_tool_small")

for cell_index in range(8):
    _add("crystal_sphere_click_cell", cell_index, None, f"crystal_cell:{cell_index}")

_add("crystal_sphere_proceed", label="crystal_proceed")

ACTION_SPACE_SIZE = len(ACTION_SPECS)


def action_spec(index: int) -> ActionSpec:
    return ACTION_SPECS[index]


def action_index(kind: str, primary: int | None = None, secondary: int | None = None) -> int:
    return ACTION_INDEX[(kind, primary, secondary)]

