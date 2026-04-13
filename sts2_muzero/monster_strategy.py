from __future__ import annotations

import re
from dataclasses import dataclass

COMBAT_STATE_TYPES = {"monster", "elite", "boss"}
ALL_ENEMY_TARGET_TYPES = {"ALLENEMIES"}

MECHANIC_DIMENSIONS = (
    "summon",
    "self_destruct",
    "damage_cap",
    "multi_enemy",
    "high_damage",
    "scaling",
    "burrow",
    "debuff",
    "boss",
    "shell_counter",
    "intangible",
    "stun_window",
    "cost_tax",
    "deck_pollution",
    "attack_cycle",
    "steal",
    "card_disruption",
    "support_protect",
    "punish_skills",
    "punish_powers",
    "stat_drain",
    "hp_cap_loss",
    "thorns",
)
MECHANIC_INDEX = {name: index for index, name in enumerate(MECHANIC_DIMENSIONS)}

STRATEGY_DIMENSIONS = (
    "focus_summoner",
    "focus_exploder",
    "focus_support",
    "focus_scaler",
    "focus_stealer",
    "burst_frontload",
    "aoe_clear",
    "multi_hit",
    "prioritize_block",
    "attack_windowing",
    "preserve_block",
    "energy_efficiency",
    "status_cleanup",
    "avoid_skill_spam",
    "avoid_power_spam",
    "respect_thorns",
    "resource_conservation",
)
STRATEGY_INDEX = {name: index for index, name in enumerate(STRATEGY_DIMENSIONS)}

_STRATEGY_WEIGHTS = {
    "focus_summoner": 0.80,
    "focus_exploder": 0.85,
    "focus_support": 0.80,
    "focus_scaler": 0.82,
    "focus_stealer": 0.88,
    "burst_frontload": 1.15,
    "aoe_clear": 1.18,
    "multi_hit": 1.08,
    "prioritize_block": 1.12,
    "attack_windowing": 0.92,
    "preserve_block": 1.00,
    "energy_efficiency": 0.88,
    "status_cleanup": 0.95,
    "avoid_skill_spam": 0.72,
    "avoid_power_spam": 0.72,
    "respect_thorns": 0.86,
    "resource_conservation": 0.78,
}

_MOVE_HINTS = {
    "CALL_FOR_BACKUP_MOVE": {"summon": 1.0, "scaling": 0.25},
    "EXPLODE_MOVE": {"self_destruct": 1.0, "high_damage": 0.95},
    "GIMME_MOVE": {"steal": 1.0},
    "FLEE_MOVE": {"steal": 0.55},
    "CONSTRICT": {"debuff": 0.85, "high_damage": 0.35},
    "SHRINKER_MOVE": {"debuff": 0.95, "stat_drain": 0.80},
    "FRAIL_SPORES_MOVE": {"debuff": 0.85},
    "VULNERABLE_SPORES_MOVE": {"debuff": 0.85},
    "DISEASE_BITE_MOVE": {"debuff": 0.70, "deck_pollution": 0.35},
    "OIL_SPRAY_MOVE": {"deck_pollution": 0.95, "debuff": 0.35},
    "GOOP_MOVE": {"deck_pollution": 0.65},
    "STICKY_SHOT": {"deck_pollution": 0.80, "debuff": 0.20},
    "STICKY_SHOT_MOVE": {"deck_pollution": 0.80, "debuff": 0.20},
    "CLUMP_SHOT": {"deck_pollution": 0.55},
    "CLUMP_SHOT_MOVE": {"deck_pollution": 0.55},
    "INCANTATION_MOVE": {"scaling": 1.0},
    "ENLARGE_MOVE": {"scaling": 0.85},
    "INHALE": {"scaling": 0.70, "attack_cycle": 0.35},
    "ROAR_MOVE": {"scaling": 0.68},
    "PRESSURIZE_MOVE": {"shell_counter": 0.65, "support_protect": 0.20},
    "ILLUSION_MOVE": {"card_disruption": 0.70, "support_protect": 0.20},
    "READY_MOVE": {"attack_cycle": 0.90, "high_damage": 0.25},
    "WAKE_MOVE": {"attack_cycle": 0.65},
    "STUNNED": {"stun_window": 1.0},
    "DISTRACT_MOVE": {"debuff": 0.70, "card_disruption": 0.45},
    "PIERCING_GAZE_MOVE": {"debuff": 0.85, "card_disruption": 0.40},
    "INFECT_MOVE": {"deck_pollution": 0.82, "debuff": 0.42},
    "GRASPING_VINES_MOVE": {"debuff": 0.70, "support_protect": 0.25},
    "ENERGY_ORB_MOVE": {"support_protect": 0.55},
    "BLOAT_MOVE": {"debuff": 0.75, "deck_pollution": 0.25},
    "HAUNT_MOVE": {"debuff": 0.55, "card_disruption": 0.45},
    "RELOAD_MOVE": {"attack_cycle": 0.58},
    "SCREECH_MOVE": {"debuff": 0.90},
    "SWOOP_MOVE": {"attack_cycle": 0.40, "high_damage": 0.35},
    "SPAWNED_MOVE": {"summon": 0.18},
}

_MONSTER_HINTS = {
    "GAS_BOMB": {"self_destruct": 1.0, "high_damage": 0.85},
    "TWO_TAILED_RAT": {"summon": 1.0},
    "FAT_GREMLIN": {"steal": 0.95},
    "GREMLIN_MERC": {"steal": 1.0, "debuff": 0.42},
    "SEWER_CLAM": {"shell_counter": 0.95, "support_protect": 0.18},
    "SKULKING_COLONY": {"shell_counter": 1.0},
    "BYGONE_EFFIGY": {"stun_window": 0.80, "attack_cycle": 0.35},
    "PHANTASMAL_GARDENER": {"debuff": 0.52, "stun_window": 0.45},
    "PHROG_PARASITE": {"deck_pollution": 0.82, "debuff": 0.35},
    "PUNCH_CONSTRUCT": {"stun_window": 0.52},
    "VINE_SHAMBLER": {"support_protect": 0.32, "debuff": 0.25},
    "CALCIFIED_CULTIST": {"scaling": 1.0},
    "DAMP_CULTIST": {"scaling": 1.0},
    "TOADPOLE": {"thorns": 0.85},
    "CORPSE_SLUG": {"high_damage": 0.52, "stun_window": 0.25},
    "BYRDONIS": {"attack_cycle": 0.48, "high_damage": 0.55},
    "SLITHERING_STRANGLER": {"debuff": 0.42},
    "BRUTE_RUBY_RAIDER": {"high_damage": 0.92, "scaling": 0.25},
    "AXE_RUBY_RAIDER": {"high_damage": 0.78},
    "CROSSBOW_RUBY_RAIDER": {"high_damage": 0.65},
    "LIVING_FOG": {"debuff": 0.72, "card_disruption": 0.32},
}

_POWER_HINTS = {
    "THORNS_POWER": {"thorns": 1.0},
    "HEIST_POWER": {"steal": 1.0},
    "THIEVERY_POWER": {"steal": 1.0},
    "RITUAL_POWER": {"scaling": 1.0},
    "HARDENED_SHELL_POWER": {"shell_counter": 1.0},
    "PLATING_POWER": {"shell_counter": 0.85},
    "MINION_POWER": {"summon": 0.55},
    "ILLUSION_POWER": {"card_disruption": 0.60},
    "SUCK_POWER": {"stat_drain": 1.0},
    "SHRIEK_POWER": {"debuff": 0.70},
    "RAVENOUS_POWER": {"scaling": 0.55},
    "ARTIFACT_POWER": {"support_protect": 0.22},
    "SKITTISH_POWER": {"stun_window": 0.42},
    "SLOW_POWER": {"attack_cycle": 0.35},
}

_INTENT_HINTS = {
    "STATUSCARD": {"deck_pollution": 0.85, "debuff": 0.20},
    "DEBUFF": {"debuff": 0.65},
    "DEBUFFSTRONG": {"debuff": 0.85},
    "ATTACKDEBUFF": {"debuff": 0.62, "high_damage": 0.20},
    "ATTACKBUFF": {"scaling": 0.45, "high_damage": 0.22},
    "BUFF": {"attack_cycle": 0.35},
    "STUN": {"stun_window": 1.0},
    "SLEEP": {"attack_cycle": 0.55},
}

_CHINESE_MULTI_DAMAGE_PATTERNS = (
    re.compile(r"\u9020\u6210\s*(\d+)\s*\u70b9\u4f24\u5bb3\s*(\d+)\s*\u6b21"),
    re.compile(r"\u9020\u6210\s*(\d+)\s*\u70b9\u4f24\u5bb3.*?(\d+)\s*\u6b21"),
)
_ENGLISH_MULTI_DAMAGE_PATTERNS = (
    re.compile(r"deal\s*(\d+)\s*damage\s*(\d+)\s*times"),
    re.compile(r"deal\s*(\d+)\s*damage.*?(\d+)\s*times"),
)
_CHINESE_SIMPLE_DAMAGE = re.compile(r"\u9020\u6210\s*(\d+)\s*\u70b9\u4f24\u5bb3")
_ENGLISH_SIMPLE_DAMAGE = re.compile(r"deal\s*(\d+)\s*damage")
_CHINESE_BLOCK = re.compile(r"\u83b7\u5f97\s*(\d+)\s*\u70b9\u683c\u6321")
_ENGLISH_BLOCK = re.compile(r"gain\s*(\d+)\s*block")
_CHINESE_DRAW = re.compile(r"\u62bd\s*(\d+)\s*\u5f20\u724c")
_ENGLISH_DRAW = re.compile(r"draw\s*(\d+)\s*cards?")
_CHINESE_ENERGY = re.compile(r"\u83b7\u5f97\s*(\d+)\s*\u70b9\u80fd\u91cf")
_ENGLISH_ENERGY = re.compile(r"gain\s*(\d+)\s*energy")
_STATUS_CLEANUP_TOKENS = ("exhaust", "purge", "remove", "consume", "cleanse", "retain no affliction", "\u6d88\u8017", "\u79fb\u9664", "\u6e05\u9664")


@dataclass(frozen=True)
class EnemyMechanicProfile:
    entity_id: str
    monster_id: str
    move_id: str
    mechanics: tuple[float, ...]
    focus_priority: float
    immediate_threat: float
    top_mechanics: tuple[str, ...]


@dataclass(frozen=True)
class EncounterStrategyProfile:
    mechanics: tuple[float, ...]
    strategy: tuple[float, ...]
    enemy_profiles: tuple[EnemyMechanicProfile, ...]
    incoming_damage: float
    enemy_count: int
    attack_enemy_count: int
    top_mechanics: tuple[str, ...]
    top_strategies: tuple[str, ...]


def mechanic_value(vector: tuple[float, ...], name: str) -> float:
    return float(vector[MECHANIC_INDEX[name]])


def strategy_value(vector: tuple[float, ...], name: str) -> float:
    return float(vector[STRATEGY_INDEX[name]])


def top_mechanic_dimensions(vector: tuple[float, ...], limit: int = 4) -> tuple[str, ...]:
    ranked = sorted(
        MECHANIC_DIMENSIONS,
        key=lambda name: (float(vector[MECHANIC_INDEX[name]]), -MECHANIC_INDEX[name]),
        reverse=True,
    )
    return tuple(name for name in ranked[:limit] if vector[MECHANIC_INDEX[name]] > 0.20)


def top_strategy_dimensions(vector: tuple[float, ...], limit: int = 4) -> tuple[str, ...]:
    ranked = sorted(
        STRATEGY_DIMENSIONS,
        key=lambda name: (float(vector[STRATEGY_INDEX[name]]), -STRATEGY_INDEX[name]),
        reverse=True,
    )
    return tuple(name for name in ranked[:limit] if vector[STRATEGY_INDEX[name]] > 0.20)


def strategy_alignment_score(card_vector: tuple[float, ...], demand_vector: tuple[float, ...]) -> float:
    score = 0.0
    for name in STRATEGY_DIMENSIONS:
        gain = min(strategy_value(card_vector, name), strategy_value(demand_vector, name))
        score += gain * _STRATEGY_WEIGHTS[name]
    return score


def analyze_encounter_profile(state: dict[str, object] | object) -> EncounterStrategyProfile:
    if not isinstance(state, dict):
        return EncounterStrategyProfile(
            mechanics=tuple(0.0 for _ in MECHANIC_DIMENSIONS),
            strategy=tuple(0.0 for _ in STRATEGY_DIMENSIONS),
            enemy_profiles=(),
            incoming_damage=0.0,
            enemy_count=0,
            attack_enemy_count=0,
            top_mechanics=(),
            top_strategies=(),
        )
    state_type = str(state.get("state_type", "unknown") or "")
    if state_type not in COMBAT_STATE_TYPES:
        return EncounterStrategyProfile(
            mechanics=tuple(0.0 for _ in MECHANIC_DIMENSIONS),
            strategy=tuple(0.0 for _ in STRATEGY_DIMENSIONS),
            enemy_profiles=(),
            incoming_damage=0.0,
            enemy_count=0,
            attack_enemy_count=0,
            top_mechanics=(),
            top_strategies=(),
        )
    player = state.get("player") if isinstance(state.get("player"), dict) else {}
    battle = state.get("battle") if isinstance(state.get("battle"), dict) else {}
    enemies = tuple(enemy for enemy in battle.get("enemies", []) if isinstance(enemy, dict) and _to_float(enemy.get("hp")) > 0.0)
    if not enemies:
        return EncounterStrategyProfile(
            mechanics=tuple(0.0 for _ in MECHANIC_DIMENSIONS),
            strategy=tuple(0.0 for _ in STRATEGY_DIMENSIONS),
            enemy_profiles=(),
            incoming_damage=0.0,
            enemy_count=0,
            attack_enemy_count=0,
            top_mechanics=(),
            top_strategies=(),
        )
    max_hp = max(1.0, _to_float(player.get("max_hp")))
    incoming_damage = 0.0
    attack_enemy_count = 0
    aggregate = [0.0 for _ in MECHANIC_DIMENSIONS]
    enemy_profiles: list[EnemyMechanicProfile] = []
    encounter_id = _normalize_token(battle.get("encounter_id"))
    for enemy in enemies:
        profile = _build_enemy_profile(
            enemy=enemy,
            encounter_id=encounter_id,
            state_type=state_type,
            player_max_hp=max_hp,
            enemy_count=len(enemies),
        )
        enemy_profiles.append(profile)
        incoming_damage += _enemy_intent_damage(enemy)
        if any(_normalize_token(intent.get("type")) in {"ATTACK", "ATTACKBUFF", "ATTACKDEBUFF", "ATTACKDEFEND"} for intent in enemy.get("intents", []) if isinstance(intent, dict)):
            attack_enemy_count += 1
        for index, value in enumerate(profile.mechanics):
            aggregate[index] += float(value)
    if len(enemies) >= 2:
        aggregate[MECHANIC_INDEX["multi_enemy"]] += 0.42 + (0.18 * max(0, len(enemies) - 2))
    if state_type == "boss":
        aggregate[MECHANIC_INDEX["boss"]] += 1.0
    mechanic_vector = _freeze_vector(aggregate, clamp=2.5)
    strategy_vector = _derive_strategy_vector(
        mechanic_vector,
        enemy_profiles=tuple(enemy_profiles),
        incoming_damage=incoming_damage,
        player_hp=_to_float(player.get("hp")),
        player_max_hp=max_hp,
        player_block=_to_float(player.get("block")),
        attack_enemy_count=attack_enemy_count,
        enemy_count=len(enemies),
    )
    return EncounterStrategyProfile(
        mechanics=mechanic_vector,
        strategy=strategy_vector,
        enemy_profiles=tuple(enemy_profiles),
        incoming_damage=incoming_damage,
        enemy_count=len(enemies),
        attack_enemy_count=attack_enemy_count,
        top_mechanics=top_mechanic_dimensions(mechanic_vector),
        top_strategies=top_strategy_dimensions(strategy_vector),
    )


def card_strategy_vector_from_raw_card(card: dict[str, object]) -> tuple[float, ...]:
    card_type = str(card.get("type", "") or "")
    description = str(card.get("description", "") or "").strip().lower()
    target_type = str(card.get("target_type", "") or "").strip().upper()
    cost = _safe_cost_value(card.get("cost"))
    return _build_card_strategy_vector(
        card_type=card_type,
        description=description,
        target_type=target_type,
        cost=cost,
    )


def card_strategy_vector_from_snapshot(snapshot: object) -> tuple[float, ...]:
    return _build_card_strategy_vector(
        card_type=str(getattr(snapshot, "card_type", "") or ""),
        description=str(getattr(snapshot, "description", "") or "").strip().lower(),
        target_type=str(getattr(snapshot, "target_type", "") or "").strip().upper(),
        cost=_safe_cost_value(getattr(snapshot, "cost", "")),
    )


def _build_enemy_profile(
    *,
    enemy: dict[str, object],
    encounter_id: str,
    state_type: str,
    player_max_hp: float,
    enemy_count: int,
) -> EnemyMechanicProfile:
    mechanics = [0.0 for _ in MECHANIC_DIMENSIONS]
    monster_id = _normalize_token(enemy.get("monster_id") or enemy.get("name"))
    move_id = _normalize_token(enemy.get("move_id"))
    intent_types = tuple(_normalize_token(intent.get("type")) for intent in enemy.get("intents", []) if isinstance(intent, dict))
    power_ids = tuple(_normalize_token(status.get("id")) for status in enemy.get("status", []) if isinstance(status, dict))
    _apply_hint_mapping(mechanics, _MONSTER_HINTS.get(monster_id))
    _apply_hint_mapping(mechanics, _MOVE_HINTS.get(move_id))
    for intent_type in intent_types:
        _apply_hint_mapping(mechanics, _INTENT_HINTS.get(intent_type))
    for power_id in power_ids:
        _apply_hint_mapping(mechanics, _POWER_HINTS.get(power_id))

    intent_damage = _enemy_intent_damage(enemy)
    if intent_damage >= max(12.0, player_max_hp * 0.18):
        mechanics[MECHANIC_INDEX["high_damage"]] += min(1.2, intent_damage / max(12.0, player_max_hp * 0.12))
    if enemy_count >= 2:
        mechanics[MECHANIC_INDEX["multi_enemy"]] += 0.22
    if state_type == "boss":
        mechanics[MECHANIC_INDEX["boss"]] += 1.0
    if any(intent_type in {"BUFF", "DEBUFF", "DEBUFFSTRONG", "STATUSCARD"} for intent_type in intent_types) and not any(intent_type.startswith("ATTACK") or intent_type == "ATTACK" for intent_type in intent_types):
        mechanics[MECHANIC_INDEX["attack_cycle"]] += 0.35
    if "INTANGIBLE" in power_ids:
        mechanics[MECHANIC_INDEX["intangible"]] += 1.0
        mechanics[MECHANIC_INDEX["damage_cap"]] += 0.55
    if "THORNS_POWER" in power_ids:
        mechanics[MECHANIC_INDEX["thorns"]] += 1.0
    if "HEIST_POWER" in power_ids or "THIEVERY_POWER" in power_ids:
        mechanics[MECHANIC_INDEX["steal"]] += 0.85
    if "SUCK_POWER" in power_ids:
        mechanics[MECHANIC_INDEX["stat_drain"]] += 1.0
    if any(token in move_id for token in ("BURROW", "DIG", "UNDERGROUND", "SUBMERGE")):
        mechanics[MECHANIC_INDEX["burrow"]] += 1.0
    if any(token in move_id for token in ("SHELL", "CARAPACE", "PLATE", "PLATING")):
        mechanics[MECHANIC_INDEX["shell_counter"]] += 0.75
    if any(token in move_id for token in ("TAX", "COST", "DRAIN_ENERGY", "PIERCING_GAZE")):
        mechanics[MECHANIC_INDEX["cost_tax"]] += 0.65
    if any(token in move_id for token in ("STUN", "DAZE")) or "STUNNED" in move_id:
        mechanics[MECHANIC_INDEX["stun_window"]] += 1.0
    if any(token in move_id for token in ("SEAL", "NULLIFY", "DISPEL")):
        mechanics[MECHANIC_INDEX["card_disruption"]] += 0.70
    if any(token in move_id for token in ("BOOK", "DEMON", "MAX_HP", "SOUL_TAX")):
        mechanics[MECHANIC_INDEX["hp_cap_loss"]] += 1.0
    if any(token in move_id for token in ("REBUKE_SKILL", "PUNISH_SKILL", "SPIKE_SKILL")):
        mechanics[MECHANIC_INDEX["punish_skills"]] += 1.0
    if any(token in move_id for token in ("PUNISH_POWER", "POWER_EATER")):
        mechanics[MECHANIC_INDEX["punish_powers"]] += 1.0
    if any(token in encounter_id for token in ("TRIPLE", "SWARM", "SLIMES", "GREMLINS")) and enemy_count >= 3:
        mechanics[MECHANIC_INDEX["multi_enemy"]] += 0.35

    vector = _freeze_vector(mechanics, clamp=2.0)
    immediate_threat = max(0.0, min(2.0, (intent_damage / max(6.0, player_max_hp * 0.10)) + (0.22 * len(intent_types))))
    focus_priority = (
        (0.18 * immediate_threat)
        + (0.26 * mechanic_value(vector, "summon"))
        + (0.28 * mechanic_value(vector, "self_destruct"))
        + (0.22 * mechanic_value(vector, "support_protect"))
        + (0.20 * mechanic_value(vector, "scaling"))
        + (0.26 * mechanic_value(vector, "steal"))
        + (0.18 * mechanic_value(vector, "debuff"))
        + (0.22 * mechanic_value(vector, "stat_drain"))
        + (0.28 * mechanic_value(vector, "hp_cap_loss"))
    )
    hp = max(0.0, _to_float(enemy.get("hp")))
    if hp <= max(12.0, player_max_hp * 0.16) and (
        mechanic_value(vector, "self_destruct") > 0.0
        or mechanic_value(vector, "summon") > 0.0
        or mechanic_value(vector, "support_protect") > 0.0
    ):
        focus_priority += 0.18
    return EnemyMechanicProfile(
        entity_id=str(enemy.get("entity_id", "") or ""),
        monster_id=monster_id,
        move_id=move_id,
        mechanics=vector,
        focus_priority=max(0.0, min(2.5, focus_priority)),
        immediate_threat=immediate_threat,
        top_mechanics=top_mechanic_dimensions(vector, limit=3),
    )


def _derive_strategy_vector(
    mechanic_vector: tuple[float, ...],
    *,
    enemy_profiles: tuple[EnemyMechanicProfile, ...],
    incoming_damage: float,
    player_hp: float,
    player_max_hp: float,
    player_block: float,
    attack_enemy_count: int,
    enemy_count: int,
) -> tuple[float, ...]:
    values = [0.0 for _ in STRATEGY_DIMENSIONS]
    hp_ratio = player_hp / max(player_max_hp, 1.0)
    block_gap_ratio = max(0.0, incoming_damage - player_block) / max(player_max_hp, 1.0)
    no_attack_window = max(0.0, enemy_count - attack_enemy_count) / max(1.0, float(enemy_count))

    values[STRATEGY_INDEX["focus_summoner"]] += mechanic_value(mechanic_vector, "summon")
    values[STRATEGY_INDEX["focus_exploder"]] += mechanic_value(mechanic_vector, "self_destruct")
    values[STRATEGY_INDEX["focus_support"]] += mechanic_value(mechanic_vector, "support_protect") + (0.35 * mechanic_value(mechanic_vector, "card_disruption"))
    values[STRATEGY_INDEX["focus_scaler"]] += mechanic_value(mechanic_vector, "scaling")
    values[STRATEGY_INDEX["focus_stealer"]] += mechanic_value(mechanic_vector, "steal") + (0.45 * mechanic_value(mechanic_vector, "stat_drain"))
    values[STRATEGY_INDEX["burst_frontload"]] += (
        (0.55 * mechanic_value(mechanic_vector, "high_damage"))
        + (0.70 * mechanic_value(mechanic_vector, "scaling"))
        + (0.42 * mechanic_value(mechanic_vector, "steal"))
        + (0.55 * mechanic_value(mechanic_vector, "stat_drain"))
        + (0.48 * mechanic_value(mechanic_vector, "card_disruption"))
    )
    values[STRATEGY_INDEX["aoe_clear"]] += (
        (0.78 * mechanic_value(mechanic_vector, "multi_enemy"))
        + (0.92 * mechanic_value(mechanic_vector, "summon"))
        + (0.22 * mechanic_value(mechanic_vector, "support_protect"))
    )
    values[STRATEGY_INDEX["multi_hit"]] += (
        (0.95 * mechanic_value(mechanic_vector, "shell_counter"))
        + (0.88 * mechanic_value(mechanic_vector, "intangible"))
        + (0.72 * mechanic_value(mechanic_vector, "damage_cap"))
    )
    values[STRATEGY_INDEX["prioritize_block"]] += (
        (1.25 * block_gap_ratio)
        + (0.62 * mechanic_value(mechanic_vector, "high_damage"))
        + (0.46 * mechanic_value(mechanic_vector, "self_destruct"))
        + (0.42 * mechanic_value(mechanic_vector, "boss"))
        + (0.90 * mechanic_value(mechanic_vector, "hp_cap_loss"))
    )
    values[STRATEGY_INDEX["attack_windowing"]] += (
        (0.82 * mechanic_value(mechanic_vector, "attack_cycle"))
        + (0.84 * mechanic_value(mechanic_vector, "burrow"))
        + (0.38 * no_attack_window)
        + (0.36 * sum(1.0 for profile in enemy_profiles if "stun_window" in profile.top_mechanics))
    )
    values[STRATEGY_INDEX["preserve_block"]] += (
        (0.95 * mechanic_value(mechanic_vector, "stun_window"))
        + (0.55 * mechanic_value(mechanic_vector, "hp_cap_loss"))
        + (0.35 * max(0.0, 0.72 - hp_ratio))
    )
    values[STRATEGY_INDEX["energy_efficiency"]] += (
        (0.92 * mechanic_value(mechanic_vector, "cost_tax"))
        + (0.24 * mechanic_value(mechanic_vector, "boss"))
        + (0.20 * mechanic_value(mechanic_vector, "high_damage"))
    )
    values[STRATEGY_INDEX["status_cleanup"]] += (
        (0.98 * mechanic_value(mechanic_vector, "deck_pollution"))
        + (0.46 * mechanic_value(mechanic_vector, "debuff"))
        + (0.24 * mechanic_value(mechanic_vector, "card_disruption"))
    )
    values[STRATEGY_INDEX["avoid_skill_spam"]] += mechanic_value(mechanic_vector, "punish_skills")
    values[STRATEGY_INDEX["avoid_power_spam"]] += mechanic_value(mechanic_vector, "punish_powers")
    values[STRATEGY_INDEX["respect_thorns"]] += mechanic_value(mechanic_vector, "thorns")
    values[STRATEGY_INDEX["resource_conservation"]] += (
        (0.62 * mechanic_value(mechanic_vector, "boss"))
        + (0.44 * mechanic_value(mechanic_vector, "attack_cycle"))
        + (0.28 * mechanic_value(mechanic_vector, "burrow"))
        + (0.20 * no_attack_window)
    )
    return _freeze_vector(values, clamp=2.5)


def _build_card_strategy_vector(
    *,
    card_type: str,
    description: str,
    target_type: str,
    cost: float,
) -> tuple[float, ...]:
    values = [0.0 for _ in STRATEGY_DIMENSIONS]
    damage_per_hit, hit_count = _parse_damage_profile(description)
    damage_total = damage_per_hit * hit_count
    block_amount = _parse_value(description, _CHINESE_BLOCK, _ENGLISH_BLOCK)
    draw_amount = _parse_value(description, _CHINESE_DRAW, _ENGLISH_DRAW)
    energy_gain = _parse_value(description, _CHINESE_ENERGY, _ENGLISH_ENERGY)
    is_aoe = target_type in ALL_ENEMY_TARGET_TYPES or "\u6240\u6709\u654c\u4eba" in description or "all enemies" in description
    if card_type == "Attack":
        burst = damage_total / max(6.0, 5.0 + (cost * 2.2))
        values[STRATEGY_INDEX["burst_frontload"]] += min(1.8, burst)
        values[STRATEGY_INDEX["attack_windowing"]] += min(1.0, burst * 0.35) + (0.16 if cost <= 1.0 else 0.0)
        if is_aoe:
            values[STRATEGY_INDEX["aoe_clear"]] += min(1.8, (damage_total / 10.0) + 0.24)
        if hit_count > 1:
            values[STRATEGY_INDEX["multi_hit"]] += min(1.8, (hit_count * 0.34) + min(0.6, damage_total / 18.0))
    if block_amount > 0.0:
        block_value = min(1.8, block_amount / max(6.0, 4.5 + (cost * 1.8)))
        values[STRATEGY_INDEX["prioritize_block"]] += block_value
        values[STRATEGY_INDEX["preserve_block"]] += min(1.4, block_value * 0.82)
        values[STRATEGY_INDEX["resource_conservation"]] += min(0.8, block_value * 0.30)
    if draw_amount > 0.0 or energy_gain > 0.0 or cost == 0.0:
        values[STRATEGY_INDEX["energy_efficiency"]] += min(1.7, (draw_amount * 0.45) + (energy_gain * 0.75) + (0.28 if cost == 0.0 else 0.0))
        values[STRATEGY_INDEX["resource_conservation"]] += min(1.0, (draw_amount * 0.22) + (energy_gain * 0.28))
    if any(token in description for token in _STATUS_CLEANUP_TOKENS):
        values[STRATEGY_INDEX["status_cleanup"]] += 0.85
    if card_type == "Power":
        values[STRATEGY_INDEX["resource_conservation"]] += 0.45
    return _freeze_vector(values, clamp=2.0)


def _apply_hint_mapping(target: list[float], mapping: dict[str, float] | None) -> None:
    for name, value in (mapping or {}).items():
        index = MECHANIC_INDEX.get(name)
        if index is None:
            continue
        target[index] += float(value)


def _freeze_vector(values: list[float], clamp: float) -> tuple[float, ...]:
    return tuple(max(0.0, min(clamp, float(value))) for value in values)


def _normalize_token(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    # Split only actual camel-case boundaries. Existing uppercase snake-case ids
    # like GAS_BOMB must remain stable so hint tables keep matching.
    raw = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", raw)
    raw = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", raw)
    raw = re.sub(r"[\s/\-]+", "_", raw.upper())
    raw = re.sub(r"[^A-Z0-9_]", "", raw)
    raw = re.sub(r"_+", "_", raw)
    return raw.strip("_")


def _to_float(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _safe_cost_value(value: object) -> float:
    token = str(value or "").strip().upper()
    if token in {"", "?", "X"}:
        return 1.5 if token == "X" else 0.0
    try:
        return max(0.0, float(token))
    except (TypeError, ValueError):
        return 0.0


def _enemy_intent_damage(enemy: dict[str, object]) -> float:
    intents = enemy.get("intents") if isinstance(enemy.get("intents"), list) else []
    total = 0.0
    for intent in intents:
        if not isinstance(intent, dict):
            continue
        digits = "".join(character for character in str(intent.get("label", "")) if character.isdigit())
        if digits:
            total += float(digits)
    return total


def _parse_damage_profile(text: str) -> tuple[int, int]:
    for pattern in _CHINESE_MULTI_DAMAGE_PATTERNS:
        match = pattern.search(text)
        if match is not None:
            return int(match.group(1)), int(match.group(2))
    for pattern in _ENGLISH_MULTI_DAMAGE_PATTERNS:
        match = pattern.search(text)
        if match is not None:
            return int(match.group(1)), int(match.group(2))
    match = _CHINESE_SIMPLE_DAMAGE.search(text)
    if match is not None:
        return int(match.group(1)), 1
    match = _ENGLISH_SIMPLE_DAMAGE.search(text)
    if match is not None:
        return int(match.group(1)), 1
    return 0, 0


def _parse_value(text: str, chinese_pattern: re.Pattern[str], english_pattern: re.Pattern[str]) -> float:
    match = chinese_pattern.search(text)
    if match is not None:
        return float(match.group(1))
    match = english_pattern.search(text)
    if match is not None:
        return float(match.group(1))
    return 0.0
