import json
import urllib.error
import urllib.parse
import urllib.request


class STS2BridgeError(RuntimeError):
    pass


class STS2Bridge:
    def __init__(self, host: str = "localhost", port: int = 15526, timeout: float = 10.0) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout

    @property
    def _base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def _singleplayer_path(self) -> str:
        return "/api/v1/singleplayer"

    def _request_json(
        self,
        method: str,
        path: str,
        query: dict[str, object] | None = None,
        payload: dict[str, object] | None = None,
    ) -> dict[str, object]:
        url = self._base_url + path
        if query:
            url += "?" + urllib.parse.urlencode(query)

        body = None
        headers: dict[str, str] = {}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(url=url, data=body, headers=headers, method=method.upper())
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            raise STS2BridgeError(f"HTTP {exc.code}: {raw}") from exc
        except urllib.error.URLError as exc:
            raise STS2BridgeError(
                "Cannot connect to STS2_MCP mod. Check that Slay the Spire 2 is running with the mod enabled."
            ) from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise STS2BridgeError(f"Invalid JSON response from STS2 bridge: {raw}") from exc

        if not isinstance(data, dict):
            raise STS2BridgeError(f"Unexpected response payload: {data!r}")
        return data

    def _post_action(self, action: str, **kwargs: object) -> dict[str, object]:
        payload = {"action": action}
        payload.update(kwargs)
        return self._request_json("POST", self._singleplayer_path, payload=payload)

    def call_tool(self, tool_name: str, **kwargs: object) -> dict[str, object]:
        method = getattr(self, tool_name, None)
        if method is None:
            raise STS2BridgeError(f"Unknown tool: {tool_name}")
        result = method(**kwargs)
        if not isinstance(result, dict):
            raise STS2BridgeError(f"Unexpected tool result from {tool_name}: {result!r}")
        return result

    def get_game_state(self, format: str = "json") -> dict[str, object]:
        return self._request_json("GET", self._singleplayer_path, query={"format": format})

    def start_run(self, character: str = "Ironclad", ascension: int | None = None) -> dict[str, object]:
        payload: dict[str, object] = {"character": character}
        if ascension is not None:
            payload["ascension"] = ascension
        return self._post_action("start_run", **payload)

    def use_potion(self, slot: int, target: str | None = None) -> dict[str, object]:
        payload: dict[str, object] = {"slot": slot}
        if target is not None:
            payload["target"] = target
        return self._post_action("use_potion", **payload)

    def discard_potion(self, slot: int) -> dict[str, object]:
        return self._post_action("discard_potion", slot=slot)

    def proceed_to_map(self) -> dict[str, object]:
        return self._post_action("proceed")

    def combat_play_card(self, card_index: int, target: str | None = None, card_id: str | None = None) -> dict[str, object]:
        payload: dict[str, object] = {"card_index": card_index}
        if card_id is not None:
            payload["card_id"] = card_id
        if target is not None:
            payload["target"] = target
        return self._post_action("play_card", **payload)

    def combat_end_turn(self) -> dict[str, object]:
        return self._post_action("end_turn")

    def combat_select_card(self, card_index: int) -> dict[str, object]:
        return self._post_action("combat_select_card", card_index=card_index)

    def combat_confirm_selection(self) -> dict[str, object]:
        return self._post_action("combat_confirm_selection")

    def rewards_claim(self, reward_index: int) -> dict[str, object]:
        return self._post_action("claim_reward", index=reward_index)

    def rewards_pick_card(self, card_index: int) -> dict[str, object]:
        return self._post_action("select_card_reward", card_index=card_index)

    def rewards_skip_card(self) -> dict[str, object]:
        return self._post_action("skip_card_reward")

    def map_choose_node(self, node_index: int) -> dict[str, object]:
        return self._post_action("choose_map_node", index=node_index)

    def rest_choose_option(self, option_index: int) -> dict[str, object]:
        return self._post_action("choose_rest_option", index=option_index)

    def shop_purchase(self, item_index: int) -> dict[str, object]:
        return self._post_action("shop_purchase", index=item_index)

    def event_choose_option(self, option_index: int) -> dict[str, object]:
        return self._post_action("choose_event_option", index=option_index)

    def event_advance_dialogue(self) -> dict[str, object]:
        return self._post_action("advance_dialogue")

    def deck_select_card(self, card_index: int) -> dict[str, object]:
        return self._post_action("select_card", index=card_index)

    def deck_confirm_selection(self) -> dict[str, object]:
        return self._post_action("confirm_selection")

    def deck_cancel_selection(self) -> dict[str, object]:
        return self._post_action("cancel_selection")

    def bundle_select(self, bundle_index: int) -> dict[str, object]:
        return self._post_action("select_bundle", index=bundle_index)

    def bundle_confirm_selection(self) -> dict[str, object]:
        return self._post_action("confirm_bundle_selection")

    def bundle_cancel_selection(self) -> dict[str, object]:
        return self._post_action("cancel_bundle_selection")

    def relic_select(self, relic_index: int) -> dict[str, object]:
        return self._post_action("select_relic", index=relic_index)

    def relic_skip(self) -> dict[str, object]:
        return self._post_action("skip_relic_selection")

    def treasure_claim_relic(self, relic_index: int) -> dict[str, object]:
        return self._post_action("claim_treasure_relic", index=relic_index)

    def crystal_sphere_set_tool(self, tool: str) -> dict[str, object]:
        return self._post_action("crystal_sphere_set_tool", tool=tool)

    def crystal_sphere_click_cell(self, x: int, y: int) -> dict[str, object]:
        return self._post_action("crystal_sphere_click_cell", x=x, y=y)

    def crystal_sphere_proceed(self) -> dict[str, object]:
        return self._post_action("crystal_sphere_proceed")
