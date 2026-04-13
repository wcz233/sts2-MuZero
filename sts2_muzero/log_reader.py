import argparse
import json
from pathlib import Path

from .combat_logs import decode_episode_states, expand_episode_archive, load_episode_archive


def _reward_text(action_record: dict[str, object]) -> str:
    reward = action_record.get("reward") if isinstance(action_record.get("reward"), dict) else {}
    final_reward = float(reward.get("final", 0.0) or 0.0)
    raw_reward = float(reward.get("raw", 0.0) or 0.0)
    credit = float(reward.get("credit_adjustment", 0.0) or 0.0)
    if abs(credit) <= 1e-12:
        return f"{final_reward:+.3f}"
    return f"{final_reward:+.3f} raw={raw_reward:+.3f} settle={credit:+.3f}"


def _boundary_text(action_record: dict[str, object]) -> str:
    boundaries = action_record.get("boundaries") if isinstance(action_record.get("boundaries"), dict) else {}
    labels = [name for name in ("turn_end", "combat_end", "act_end", "run_end") if boundaries.get(name)]
    return ",".join(labels)


def _iter_actions(payload: dict[str, object]) -> list[tuple[dict[str, object], dict[str, object]]]:
    nodes = payload.get("nodes") if isinstance(payload.get("nodes"), list) else []
    pairs: list[tuple[dict[str, object], dict[str, object]]] = []
    for node in nodes:
        actions = node.get("actions") if isinstance(node.get("actions"), list) else []
        for action_record in actions:
            pairs.append((node, action_record))
    return pairs


def _print_summary(path: Path, payload: dict[str, object]) -> None:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    print(f"path: {path}")
    print(f"schema: {payload.get('schema')}")
    print(f"compression: {payload.get('compression')}")
    print(
        "episode: "
        f"{metadata.get('character', '?')} "
        f"highest_floor={metadata.get('highest_floor', '?')} "
        f"model={metadata.get('model_name', '?')} "
        f"result={metadata.get('result', 'unknown')}"
    )
    print(
        "stats: "
        f"nodes={summary.get('total_nodes', 0)} "
        f"actions={summary.get('total_steps', 0)} "
        f"raw_reward={float(summary.get('raw_reward', 0.0) or 0.0):+.3f} "
        f"final_reward={float(summary.get('final_reward', 0.0) or 0.0):+.3f}"
    )
    node_counts = summary.get("node_counts")
    if isinstance(node_counts, dict):
        print(f"node_counts: {json.dumps(node_counts, ensure_ascii=False, sort_keys=True)}")
    print(
        "time: "
        f"started={metadata.get('started_at', '?')} "
        f"ended={metadata.get('ended_at', '?')}"
    )


def _node_detail_text(node: dict[str, object]) -> str:
    details_kind = str(node.get("details_kind", ""))
    summary = node.get("summary") if isinstance(node.get("summary"), dict) else {}
    if details_kind == "combat":
        return f"turns={summary.get('turn_count', 0)} result={summary.get('result', 'unknown')}"
    if details_kind == "shop":
        return f"purchases={summary.get('purchase_count', 0)}"
    if details_kind == "rest_site":
        return f"choices={summary.get('choice_count', 0)}"
    if details_kind == "event":
        return (
            f"choices={summary.get('choice_count', 0)} "
            f"dialogue={summary.get('dialogue_step_count', 0)}"
        )
    if details_kind == "treasure":
        return f"claimed={summary.get('claimed_item_count', 0)}"
    if details_kind in {"selection", "reward"}:
        return f"choices={summary.get('choice_count', 0)}"
    return ""


def _print_nodes(payload: dict[str, object]) -> None:
    nodes = payload.get("nodes") if isinstance(payload.get("nodes"), list) else []
    for node in nodes:
        summary = node.get("summary") if isinstance(node.get("summary"), dict) else {}
        route_choice = node.get("route_choice") if isinstance(node.get("route_choice"), dict) else {}
        route_type = route_choice.get("type") or route_choice.get("node_type") or route_choice.get("kind") or "-"
        detail_text = _node_detail_text(node)
        detail_suffix = f" {detail_text}" if detail_text else ""
        print(
            f"[{node.get('node_index', '?'):02d}] "
            f"act={node.get('act', '?')} floor={node.get('floor', '?')} "
            f"type={node.get('type', '?')} resolved={node.get('resolved_type', '?')} route={route_type} "
            f"actions={summary.get('action_count', 0)} reward={float(summary.get('final_reward', 0.0) or 0.0):+.3f}{detail_suffix}"
        )


def _print_timeline(payload: dict[str, object], node_index: int | None, limit: int) -> None:
    count = 0
    for node, action_record in _iter_actions(payload):
        if node_index is not None and int(node.get("node_index", -1)) != node_index:
            continue
        if limit > 0 and count >= limit:
            break
        action = action_record.get("action") if isinstance(action_record.get("action"), dict) else {}
        boundary = _boundary_text(action_record)
        boundary_suffix = f" boundary={boundary}" if boundary else ""
        print(
            f"[n{node.get('node_index', '?'):02d}/a{action_record.get('global_index', '?'):03d}] "
            f"floor={node.get('floor', '?')} {action_record.get('state_type', '?')} -> "
            f"{action.get('description', '?')} -> {action_record.get('next_state_type', '?')} "
            f"reward={_reward_text(action_record)} changes={len(action_record.get('changes', []))}{boundary_suffix}"
        )
        count += 1


def _print_node(payload: dict[str, object], node_index: int, state_view: str) -> None:
    nodes = payload.get("nodes") if isinstance(payload.get("nodes"), list) else []
    if node_index < 0 or node_index >= len(nodes):
        raise IndexError(f"node index out of range: {node_index}")
    node = nodes[node_index]
    print(json.dumps(node, indent=2, ensure_ascii=False, sort_keys=True))
    if state_view == "none":
        return
    states = decode_episode_states(payload)
    actions = node.get("actions") if isinstance(node.get("actions"), list) else []
    if not actions:
        return
    first_before_id = int(node.get("entry_global_state_id", actions[0].get("state_before_id", -1)))
    last_after_id = int(node.get("final_global_state_id", actions[-1].get("state_after_id", -1)))
    if state_view in {"entry", "both"}:
        print("\n[entry_global_state]")
        print(json.dumps(states[first_before_id], indent=2, ensure_ascii=False, sort_keys=True))
    if state_view in {"final", "both"}:
        print("\n[final_global_state]")
        print(json.dumps(states[last_after_id], indent=2, ensure_ascii=False, sort_keys=True))


def _print_step(payload: dict[str, object], node_index: int, action_index: int, state_view: str) -> None:
    nodes = payload.get("nodes") if isinstance(payload.get("nodes"), list) else []
    if node_index < 0 or node_index >= len(nodes):
        raise IndexError(f"node index out of range: {node_index}")
    node = nodes[node_index]
    actions = node.get("actions") if isinstance(node.get("actions"), list) else []
    if action_index < 0 or action_index >= len(actions):
        raise IndexError(f"action index out of range: {action_index}")
    action_record = actions[action_index]
    print(json.dumps(action_record, indent=2, ensure_ascii=False, sort_keys=True))
    if state_view == "none":
        return
    states = decode_episode_states(payload)
    before_id = int(action_record.get("state_before_id", -1))
    after_id = int(action_record.get("state_after_id", -1))
    if state_view in {"before", "both"}:
        print("\n[state_before]")
        print(json.dumps(states[before_id], indent=2, ensure_ascii=False, sort_keys=True))
    if state_view in {"after", "both"}:
        print("\n[state_after]")
        print(json.dumps(states[after_id], indent=2, ensure_ascii=False, sort_keys=True))


def _export_json(path: Path, payload: dict[str, object], output: Path | None) -> None:
    expanded = expand_episode_archive(payload)
    text = json.dumps(expanded, indent=2, ensure_ascii=False, sort_keys=True)
    if output is None:
        print(text)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(f"exported: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Read compressed sts2-MuZero episode logs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary_parser = subparsers.add_parser("summary", help="Print episode summary")
    summary_parser.add_argument("path")

    nodes_parser = subparsers.add_parser("nodes", help="List all floor nodes")
    nodes_parser.add_argument("path")

    timeline_parser = subparsers.add_parser("timeline", help="Print node/action timeline")
    timeline_parser.add_argument("path")
    timeline_parser.add_argument("--node-index", type=int, default=None)
    timeline_parser.add_argument("--limit", type=int, default=0, help="0 means no limit")

    show_node_parser = subparsers.add_parser("show-node", help="Show one node")
    show_node_parser.add_argument("path")
    show_node_parser.add_argument("node_index", type=int)
    show_node_parser.add_argument("--states", choices=["entry", "final", "both", "none"], default="none")

    show_step_parser = subparsers.add_parser("show-step", help="Show one action inside a node")
    show_step_parser.add_argument("path")
    show_step_parser.add_argument("node_index", type=int)
    show_step_parser.add_argument("action_index", type=int)
    show_step_parser.add_argument("--states", choices=["before", "after", "both", "none"], default="both")

    export_parser = subparsers.add_parser("export-json", help="Export expanded JSON with decoded global states")
    export_parser.add_argument("path")
    export_parser.add_argument("--output", default="")

    args = parser.parse_args()
    path = Path(args.path)
    payload = load_episode_archive(path)

    if args.command == "summary":
        _print_summary(path, payload)
        return
    if args.command == "nodes":
        _print_nodes(payload)
        return
    if args.command == "timeline":
        _print_timeline(payload, args.node_index, args.limit)
        return
    if args.command == "show-node":
        _print_node(payload, args.node_index, args.states)
        return
    if args.command == "show-step":
        _print_step(payload, args.node_index, args.action_index, args.states)
        return
    if args.command == "export-json":
        output = Path(args.output) if args.output else None
        _export_json(path, payload, output)
        return
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
