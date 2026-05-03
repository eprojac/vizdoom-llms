#!/usr/bin/env python3
"""
Doom VLM duel: RedHatAI/Qwen3.6-35B-A3B-NVFP4 vs RedHatAI/gemma-4-26B-A4B-it-NVFP4

Single-script controller:
  - launches two vLLM OpenAI-compatible servers, one per model
  - runs a 2-player ViZDoom deathmatch on one map
  - optionally adds in-game bots
  - sends each player POV frame as a 640x360 JPEG image to its assigned model
  - requires a single quoted action word, enforced with vLLM structured choice when available
  - holds the last valid button command until the next valid model decision arrives
  - logs decisions, per-frame scoreboard data, and inferred head-to-head kills
  - records continuous 360p POV MP4 videos for both model-controlled players
    with debug overlays, independent of whether a new model decision arrived
  - records a third tactical map/radar MP4 showing sector/linedef geometry,
    both model players, visible enemies/bots, and frag counters

Example:
  python doom_vlm_duel_dask_vllm.py \
    --bots 0 \
    --duration-s 300 \
    --qwen-gpu-mem 0.40 \
    --gemma-gpu-mem 0.40 \
    --record-fps 35

If you already launched vLLM yourself:
  python doom_vlm_duel_dask_vllm.py --no-launch-vllm \
    --qwen-url http://127.0.0.1:8001/v1 \
    --gemma-url http://127.0.0.1:8002/v1
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from dask.distributed import Client, LocalCluster
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
except ImportError:  # Recording is optional; fail only when recording is requested.
    cv2 = None

try:
    import vizdoom as vzd
except ImportError as exc:
    raise SystemExit(
        "ViZDoom is not installed. Try: pip install vizdoom pillow dask distributed openai requests opencv-python-headless"
    ) from exc

try:
    from openai import OpenAI
except ImportError as exc:
    raise SystemExit("OpenAI Python client is not installed. Try: pip install openai") from exc


QWEN_MODEL = "RedHatAI/Qwen3.6-35B-A3B-NVFP4"
GEMMA_MODEL = "RedHatAI/gemma-4-26B-A4B-it-NVFP4"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SCENARIO_CONFIG = SCRIPT_DIR / "scenarios" / "basic.cfg"

# Single quoted words. Underscores keep combos as one word.
ACTION_CHOICES = [
    '"NOOP"',
    '"FORWARD"',
    '"BACK"',
    '"TURN_LEFT"',
    '"TURN_RIGHT"',
    '"STRAFE_LEFT"',
    '"STRAFE_RIGHT"',
    '"ATTACK"',
    '"FORWARD_ATTACK"',
    '"TURN_LEFT_ATTACK"',
    '"TURN_RIGHT_ATTACK"',
    '"USE"',
]
MODEL_ACTION_CHOICES = [choice for choice in ACTION_CHOICES if choice != '"NOOP"']

BUTTON_NAMES = [
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "TURN_LEFT",
    "TURN_RIGHT",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "ATTACK",
    "USE",
]

# Button vector order follows BUTTON_NAMES.
ACTION_TO_BUTTONS: Dict[str, List[int]] = {
    '"NOOP"': [0, 0, 0, 0, 0, 0, 0, 0],
    '"FORWARD"': [1, 0, 0, 0, 0, 0, 0, 0],
    '"BACK"': [0, 1, 0, 0, 0, 0, 0, 0],
    '"TURN_LEFT"': [0, 0, 1, 0, 0, 0, 0, 0],
    '"TURN_RIGHT"': [0, 0, 0, 1, 0, 0, 0, 0],
    '"STRAFE_LEFT"': [0, 0, 0, 0, 1, 0, 0, 0],
    '"STRAFE_RIGHT"': [0, 0, 0, 0, 0, 1, 0, 0],
    '"ATTACK"': [0, 0, 0, 0, 0, 0, 1, 0],
    '"FORWARD_ATTACK"': [1, 0, 0, 0, 0, 0, 1, 0],
    '"TURN_LEFT_ATTACK"': [0, 0, 1, 0, 0, 0, 1, 0],
    '"TURN_RIGHT_ATTACK"': [0, 0, 0, 1, 0, 0, 1, 0],
    '"USE"': [0, 0, 0, 0, 0, 0, 0, 1],
}

SYSTEM_PROMPT = """You are the live controller for one Doom deathmatch player.
You receive one first-person 360p frame plus short tactical text.
Your job is to survive, keep moving, hunt enemies, and fire when a target is visible.
Never idle during live play. Do not choose NOOP.
Return exactly one quoted action word and nothing else.
No explanation. No reasoning. No extra words.
Example valid reply: "FORWARD_ATTACK""".strip()

STRATEGY_TEXT = """Strategy:
1. If an enemy/player is visible near the crosshair, choose "FORWARD_ATTACK".
2. If an enemy is visible left or right of center, choose "TURN_LEFT_ATTACK" or "TURN_RIGHT_ATTACK".
3. If no enemy is visible, explore: "FORWARD", with occasional turning to scan.
4. If a wall, corner, door, or obstacle blocks you, choose "BACK", "TURN_LEFT", "TURN_RIGHT", "STRAFE_LEFT", "STRAFE_RIGHT", or "USE".
5. Avoid repeating the same action when stuck. Never answer "NOOP".""".strip()

USER_TEXT_PROMPT = (
    "Allowed actions: " + ", ".join(MODEL_ACTION_CHOICES) +
    ". Pick exactly one action for the next frame. Reply with exactly one quoted action word."
)

QUOTED_ONE_WORD = re.compile(r'^\s*"([A-Z0-9_]+)"\s*$')


@dataclass
class AgentRuntime:
    name: str
    model_id: str
    base_url: str
    last_action: str = '"FORWARD"'
    applied_action: str = '"FORWARD"'
    pending: Any = None
    decisions: int = 0
    invalid: int = 0
    errors: int = 0
    last_latency_ms: float = 0.0
    last_raw: str = ""
    same_action_streak: int = 0
    noop_substitutions: int = 0
    motionless_frames: int = 0
    escape_frames: int = 0
    last_pose: Optional[Tuple[float, float, float]] = None


@dataclass
class MatchScore:
    qwen_killed_gemma: int = 0
    gemma_killed_qwen: int = 0
    last_qwen_frags: Optional[int] = None
    last_gemma_frags: Optional[int] = None
    last_qwen_deaths: Optional[int] = None
    last_gemma_deaths: Optional[int] = None

    def update(
        self,
        *,
        qwen_frags: Optional[int],
        gemma_frags: Optional[int],
        qwen_deaths: Optional[int],
        gemma_deaths: Optional[int],
    ) -> None:
        if (
            self.last_qwen_frags is not None
            and self.last_gemma_frags is not None
            and self.last_qwen_deaths is not None
            and self.last_gemma_deaths is not None
            and qwen_frags is not None
            and gemma_frags is not None
            and qwen_deaths is not None
            and gemma_deaths is not None
        ):
            qwen_frag_gain = max(0, qwen_frags - self.last_qwen_frags)
            gemma_frag_gain = max(0, gemma_frags - self.last_gemma_frags)
            qwen_death_gain = max(0, qwen_deaths - self.last_qwen_deaths)
            gemma_death_gain = max(0, gemma_deaths - self.last_gemma_deaths)
            self.qwen_killed_gemma += min(qwen_frag_gain, gemma_death_gain)
            self.gemma_killed_qwen += min(gemma_frag_gain, qwen_death_gain)

        if qwen_frags is not None:
            self.last_qwen_frags = qwen_frags
        if gemma_frags is not None:
            self.last_gemma_frags = gemma_frags
        if qwen_deaths is not None:
            self.last_qwen_deaths = qwen_deaths
        if gemma_deaths is not None:
            self.last_gemma_deaths = gemma_deaths


def parse_action(raw: Optional[str]) -> Tuple[Optional[str], str]:
    if raw is None:
        return None, "empty"
    cleaned = raw.strip()
    match = QUOTED_ONE_WORD.fullmatch(cleaned)
    if match:
        action = f'"{match.group(1)}"'
        if action not in ACTION_TO_BUTTONS:
            return None, "unknown_action"
        return action, "ok"

    matches: List[str] = []
    for action_choice in ACTION_CHOICES:
        bare = action_choice.strip('"')
        if action_choice in cleaned or re.search(rf"\b{re.escape(bare)}\b", cleaned):
            matches.append(action_choice)
    unique_matches = sorted(set(matches))
    if len(unique_matches) == 1:
        return unique_matches[0], "embedded_action"
    if len(unique_matches) > 1:
        return None, "multiple_actions"
    return None, "not_exactly_one_quoted_word"


def screen_buffer_to_rgb_image(screen_buffer: np.ndarray) -> Image.Image:
    arr = np.asarray(screen_buffer)

    # ViZDoom may return CHW or HWC depending on screen format/build.
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim == 2:
        img = Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")
    else:
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        img = Image.fromarray(arr.astype(np.uint8), mode="RGB")

    return img.resize((640, 360), Image.Resampling.BILINEAR)


def encode_frame_to_jpeg_b64(screen_buffer: np.ndarray, quality: int = 55) -> str:
    # The model input is always 360p, even if ViZDoom rendered another size.
    img = screen_buffer_to_rgb_image(screen_buffer)
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=quality, optimize=False)
    return base64.b64encode(out.getvalue()).decode("ascii")


def model_short_name(model_id: str) -> str:
    leaf = model_id.split("/")[-1]
    if leaf.lower().startswith("qwen"):
        return "Qwen3.6-35B"
    if leaf.lower().startswith("gemma"):
        return "Gemma-4-26B"
    return leaf[:22]


def game_var_number(game: "vzd.DoomGame", variable_name: str) -> Optional[float]:
    try:
        return float(game.get_game_variable(getattr(vzd.GameVariable, variable_name)))
    except Exception:  # noqa: BLE001
        return None


def game_var(game: "vzd.DoomGame", variable_name: str) -> Optional[int]:
    value = game_var_number(game, variable_name)
    return int(value) if value is not None else None


def get_player_pose(game: "vzd.DoomGame") -> Dict[str, Optional[float]]:
    return {
        "x": game_var_number(game, "POSITION_X"),
        "y": game_var_number(game, "POSITION_Y"),
        "z": game_var_number(game, "POSITION_Z"),
        "angle": game_var_number(game, "ANGLE"),
    }


def get_player_frag_table(game: "vzd.DoomGame") -> Dict[str, int]:
    count = game_var(game, "PLAYER_COUNT") or 0
    table: Dict[str, int] = {}
    for idx in range(1, min(max(count, 2), 16) + 1):
        frags = game_var(game, f"PLAYER{idx}_FRAGCOUNT")
        if frags is not None:
            table[f"P{idx}"] = frags
    return table


def normalize_degrees(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def update_motion_state(agent: AgentRuntime, pose: Dict[str, Optional[float]]) -> None:
    x = pose.get("x")
    y = pose.get("y")
    angle = pose.get("angle")
    if x is None or y is None or angle is None:
        return

    current = (float(x), float(y), float(angle))
    if agent.last_pose is not None:
        prev_x, prev_y, prev_angle = agent.last_pose
        moved = math.hypot(current[0] - prev_x, current[1] - prev_y)
        turned = abs(normalize_degrees(current[2] - prev_angle))
        if moved < 1.0 and turned < 2.0:
            agent.motionless_frames += 1
        else:
            agent.motionless_frames = 0
    agent.last_pose = current


def describe_pose_relation(
    pose: Dict[str, Optional[float]],
    opponent_pose: Dict[str, Optional[float]],
) -> str:
    x = pose.get("x")
    y = pose.get("y")
    angle = pose.get("angle")
    ox = opponent_pose.get("x")
    oy = opponent_pose.get("y")
    if x is None or y is None or angle is None or ox is None or oy is None:
        return "opponent radar: unknown"

    dx = float(ox) - float(x)
    dy = float(oy) - float(y)
    distance = math.hypot(dx, dy)
    bearing = math.degrees(math.atan2(dy, dx))
    relative = normalize_degrees(bearing - float(angle))
    if abs(relative) <= 20.0:
        direction = "ahead"
    elif relative > 0.0:
        direction = "left"
    else:
        direction = "right"
    return f"opponent radar: {distance:.0f} units away, roughly {abs(relative):.0f} deg {direction}"


def format_optional(label: str, value: Optional[int]) -> str:
    return f"{label}={value if value is not None else '?'}"


def build_user_text_prompt(
    *,
    agent: AgentRuntime,
    frame_id: int,
    pose: Dict[str, Optional[float]],
    opponent_pose: Dict[str, Optional[float]],
    health: Optional[int],
    armor: Optional[int],
    ammo: Optional[int],
    frags: Optional[int],
    deaths: Optional[int],
) -> str:
    status_parts = [
        f"agent={agent.name}",
        f"frame={frame_id}",
        f"last_action={agent.last_action}",
        f"same_action_streak={agent.same_action_streak}",
        f"motionless_frames={agent.motionless_frames}",
        format_optional("health", health),
        format_optional("armor", armor),
        format_optional("ammo", ammo),
        format_optional("frags", frags),
        format_optional("deaths", deaths),
        describe_pose_relation(pose, opponent_pose),
    ]
    if agent.motionless_frames >= 12 or agent.last_action == '"NOOP"':
        status_parts.append(
            'stuck_warning=escape now with "BACK", a turn, a strafe, or "USE"; do not idle'
        )
    if agent.same_action_streak >= 4:
        status_parts.append("repeat_warning=change action unless you are actively shooting a visible target")

    return "\n".join([USER_TEXT_PROMPT, STRATEGY_TEXT, "Status: " + "; ".join(status_parts)])


def substitute_noop_action(agent: AgentRuntime, frame_id: int) -> str:
    if "gemma" in agent.name.lower():
        sequence = ['"BACK"', '"TURN_RIGHT"', '"STRAFE_LEFT"', '"FORWARD"']
    else:
        sequence = ['"BACK"', '"TURN_LEFT"', '"STRAFE_RIGHT"', '"FORWARD"']
    return sequence[(frame_id // 8) % len(sequence)]


def apply_decision_safety(agent: AgentRuntime, result: Dict[str, Any], frame_id: int) -> Dict[str, Any]:
    if result.get("valid") and result.get("action") == '"NOOP"':
        replacement = substitute_noop_action(agent, frame_id)
        result = dict(result)
        result["action"] = replacement
        result["reason"] = "noop_replaced_with_" + replacement.strip('"')
        result["valid"] = True
        agent.noop_substitutions += 1
    return result


def choose_applied_action(agent: AgentRuntime, frame_id: int, stuck_escape_frames: int) -> str:
    if agent.last_action == '"NOOP"':
        agent.escape_frames += 1
        return substitute_noop_action(agent, frame_id)

    stuck_prone_actions = {'"FORWARD"', '"BACK"', '"STRAFE_LEFT"', '"STRAFE_RIGHT"', '"USE"'}
    if stuck_escape_frames > 0 and agent.motionless_frames >= stuck_escape_frames and agent.last_action in stuck_prone_actions:
        agent.escape_frames += 1
        return substitute_noop_action(agent, frame_id)

    return agent.last_action


def overlay_player_frame(
    *,
    screen_buffer: np.ndarray,
    agent: AgentRuntime,
    frame_id: int,
    fragcount: Optional[int],
    deathcount: Optional[int],
) -> Image.Image:
    img = screen_buffer_to_rgb_image(screen_buffer)
    draw = ImageDraw.Draw(img, "RGBA")
    font = ImageFont.load_default()

    lines = [
        f"{model_short_name(agent.model_id)} acting",
        f"model action: {agent.last_action} | applied: {agent.applied_action}",
        f"valid actions: {agent.decisions} | invalid: {agent.invalid} | errors: {agent.errors}",
        f"repeat: {agent.same_action_streak} | motionless: {agent.motionless_frames} | escapes: {agent.escape_frames}",
        f"last latency: {agent.last_latency_ms:.1f} ms | frame: {frame_id}",
        f"frags: {fragcount if fragcount is not None else '?'} | deaths: {deathcount if deathcount is not None else '?'}",
    ]

    pad = 7
    line_h = 14
    box_w = 430
    box_h = pad * 2 + line_h * len(lines)
    draw.rectangle((6, 6, 6 + box_w, 6 + box_h), fill=(0, 0, 0, 170))
    for i, text in enumerate(lines):
        draw.text((6 + pad, 6 + pad + i * line_h), text, font=font, fill=(255, 255, 255, 255))
    return img


HOSTILE_NAME_HINTS = (
    "player",
    "doomplayer",
    "marine",
    "bot",
    "zombieman",
    "shotgun",
    "chaingun",
    "imp",
    "demon",
    "spectre",
    "cacodemon",
    "baron",
    "knight",
    "revenant",
    "mancubus",
    "arachnotron",
    "cyberdemon",
    "spider",
)


def _first_attr(obj: Any, names: Tuple[str, ...]) -> Any:
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return None


def _looks_like_enemy(name: str) -> bool:
    lowered = name.lower()
    return any(hint in lowered for hint in HOSTILE_NAME_HINTS)


def pose_xy(pose: Dict[str, Optional[float]]) -> Optional[Tuple[float, float]]:
    if pose.get("x") is None or pose.get("y") is None:
        return None
    return float(pose["x"]), float(pose["y"])


def close_to_any_pose(x: float, y: float, poses: Tuple[Dict[str, Optional[float]], ...], radius: float) -> bool:
    for pose in poses:
        xy = pose_xy(pose)
        if xy is not None and math.hypot(x - xy[0], y - xy[1]) <= radius:
            return True
    return False


def extract_map_lines(*states: Any) -> List[Dict[str, Any]]:
    lines: List[Dict[str, Any]] = []
    seen = set()
    for state in states:
        if state is None:
            continue
        for sector in getattr(state, "sectors", None) or ():
            for line in getattr(sector, "lines", None) or ():
                try:
                    x1 = float(line.x1)
                    y1 = float(line.y1)
                    x2 = float(line.x2)
                    y2 = float(line.y2)
                except (AttributeError, TypeError, ValueError):
                    continue
                p1 = (round(x1, 1), round(y1, 1))
                p2 = (round(x2, 1), round(y2, 1))
                key = tuple(sorted((p1, p2)))
                if key in seen:
                    continue
                seen.add(key)
                lines.append(
                    {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "blocking": bool(getattr(line, "is_blocking", False)),
                    }
                )
    return lines


def extract_visible_entities(
    *states: Any,
    model_poses: Tuple[Dict[str, Optional[float]], ...] = (),
    max_entities: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Extract visible enemies/bots from ViZDoom metadata when available.

    Object/label fields vary by ViZDoom version and scenario. This function is
    deliberately defensive: if object-position metadata is unavailable, the
    tactical map still records both model players and reports zero visible
    enemies instead of failing the match.
    """
    candidates: List[Dict[str, Any]] = []
    seen_ids = set()

    for state in states:
        if state is None:
            continue
        for seq_name in ("objects", "labels"):
            seq = getattr(state, seq_name, None)
            if not seq:
                continue
            for obj in seq:
                name = str(
                    _first_attr(obj, ("name", "object_name", "label", "type", "value")) or "object"
                )
                if not _looks_like_enemy(name):
                    continue
                x = _first_attr(obj, ("position_x", "object_position_x", "x"))
                y = _first_attr(obj, ("position_y", "object_position_y", "y"))
                angle = _first_attr(obj, ("angle", "object_angle"))
                object_id = _first_attr(obj, ("id", "object_id"))
                try:
                    xf = float(x)
                    yf = float(y)
                except (TypeError, ValueError):
                    # Some labels only expose screen-space boxes. Those are useful for POV
                    # overlays, but not for a world map/radar.
                    continue
                if close_to_any_pose(xf, yf, model_poses, radius=72.0):
                    continue
                try:
                    id_key = int(object_id)
                except (TypeError, ValueError):
                    id_key = None
                if id_key is not None and id_key > 0:
                    if id_key in seen_ids:
                        continue
                    seen_ids.add(id_key)
                candidates.append(
                    {
                        "name": name[:18],
                        "x": xf,
                        "y": yf,
                        "angle": float(angle) if angle is not None else None,
                        "id": id_key,
                        "source": seq_name,
                    }
                )

    entities: List[Dict[str, Any]] = []
    for candidate in candidates:
        if any(math.hypot(candidate["x"] - ent["x"], candidate["y"] - ent["y"]) <= 96.0 for ent in entities):
            continue
        entities.append(candidate)

    if model_poses:
        def nearest_model_distance(ent: Dict[str, Any]) -> float:
            distances = []
            for pose in model_poses:
                xy = pose_xy(pose)
                if xy is not None:
                    distances.append(math.hypot(ent["x"] - xy[0], ent["y"] - xy[1]))
            return min(distances) if distances else 0.0

        entities.sort(key=nearest_model_distance)

    if max_entities is not None:
        entities = entities[: max(0, max_entities)]
    return entities


def draw_tactical_map_frame(
    *,
    qwen_pose: Dict[str, Optional[float]],
    gemma_pose: Dict[str, Optional[float]],
    entities: List[Dict[str, Any]],
    map_lines: List[Dict[str, Any]],
    qwen: AgentRuntime,
    gemma: AgentRuntime,
    score: MatchScore,
    qwen_frags: Optional[int],
    gemma_frags: Optional[int],
    qwen_deaths: Optional[int],
    gemma_deaths: Optional[int],
    player_frags: Dict[str, int],
    expected_bots: int,
    frame_id: int,
    units_per_px: float,
    width: int = 640,
    height: int = 360,
) -> Image.Image:
    img = Image.new("RGB", (width, height), (10, 12, 16))
    draw = ImageDraw.Draw(img, "RGBA")
    font = ImageFont.load_default()

    known_points: List[Tuple[float, float]] = []
    for pose in (qwen_pose, gemma_pose):
        if pose.get("x") is not None and pose.get("y") is not None:
            known_points.append((float(pose["x"]), float(pose["y"])))
    for ent in entities:
        known_points.append((float(ent["x"]), float(ent["y"])))
    for line in map_lines:
        known_points.append((float(line["x1"]), float(line["y1"])))
        known_points.append((float(line["x2"]), float(line["y2"])))

    if map_lines:
        xs = [float(line[xkey]) for line in map_lines for xkey in ("x1", "x2")]
        ys = [float(line[ykey]) for line in map_lines for ykey in ("y1", "y2")]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0
        map_scale = max((max_x - min_x) / max(width - 72, 1), (max_y - min_y) / max(height - 72, 1), 1.0)
        scale = max(units_per_px, map_scale)
    elif known_points:
        cx = sum(p[0] for p in known_points) / len(known_points)
        cy = sum(p[1] for p in known_points) / len(known_points)
        scale = max(units_per_px, 1.0)
    else:
        cx = cy = 0.0
        scale = max(units_per_px, 1.0)

    # Auto-zoom out if visible actors would otherwise sit outside the frame.
    if len(known_points) >= 2 and not map_lines:
        max_dx = max(abs(p[0] - cx) for p in known_points)
        max_dy = max(abs(p[1] - cy) for p in known_points)
        scale = max(scale, (max_dx * 2.4) / max(width, 1), (max_dy * 2.4) / max(height, 1))

    def to_screen(x: float, y: float) -> Tuple[int, int]:
        return int(width / 2 + (x - cx) / scale), int(height / 2 - (y - cy) / scale)

    # Editor-like map background: dim grid plus sector/linedef geometry.
    grid_px = max(24, int(256 / scale))
    for gx in range(0, width, grid_px):
        draw.line((gx, 0, gx, height), fill=(255, 255, 255, 24))
    for gy in range(0, height, grid_px):
        draw.line((0, gy, width, gy), fill=(255, 255, 255, 24))
    for line in map_lines:
        x1, y1 = to_screen(float(line["x1"]), float(line["y1"]))
        x2, y2 = to_screen(float(line["x2"]), float(line["y2"]))
        if line.get("blocking"):
            fill = (120, 170, 210, 120)
            width_px = 2
        else:
            fill = (90, 100, 115, 85)
            width_px = 1
        draw.line((x1, y1, x2, y2), fill=fill, width=width_px)

    def draw_actor(
        pose: Dict[str, Optional[float]],
        label: str,
        fill: Tuple[int, int, int, int],
        radius: int = 8,
    ) -> None:
        if pose.get("x") is None or pose.get("y") is None:
            return
        sx, sy = to_screen(float(pose["x"]), float(pose["y"]))
        draw.ellipse((sx - radius, sy - radius, sx + radius, sy + radius), fill=fill, outline=(255, 255, 255, 220))
        angle = pose.get("angle")
        if angle is not None:
            # Doom angles are degrees. Draw a small heading ray.
            rad = math.radians(float(angle))
            hx = sx + int(math.cos(rad) * (radius + 14))
            hy = sy - int(math.sin(rad) * (radius + 14))
            draw.line((sx, sy, hx, hy), fill=(255, 255, 255, 230), width=2)
        draw.text((sx + radius + 3, sy - radius), label, font=font, fill=(255, 255, 255, 255))

    for ent in entities:
        sx, sy = to_screen(float(ent["x"]), float(ent["y"]))
        r = 5
        draw.rectangle((sx - r, sy - r, sx + r, sy + r), fill=(255, 120, 40, 230), outline=(255, 255, 255, 180))
        draw.text((sx + 7, sy - 7), ent["name"], font=font, fill=(255, 210, 170, 255))

    draw_actor(qwen_pose, "Qwen", (0, 190, 255, 235))
    draw_actor(gemma_pose, "Gemma", (255, 80, 220, 235))

    player_frags_text = " ".join(f"{name}:{frags}" for name, frags in player_frags.items()) or "unavailable"
    lines = [
        "Tactical map / radar",
        f"frame: {frame_id} | map lines: {len(map_lines)} | visible bots: {len(entities)}/{expected_bots} | scale: {scale:.1f}",
        f"Qwen frags/deaths: {qwen_frags if qwen_frags is not None else '?'}/{qwen_deaths if qwen_deaths is not None else '?'} | action: {qwen.applied_action}",
        f"Gemma frags/deaths: {gemma_frags if gemma_frags is not None else '?'}/{gemma_deaths if gemma_deaths is not None else '?'} | action: {gemma.applied_action}",
        f"head-to-head kills: Qwen->Gemma {score.qwen_killed_gemma} | Gemma->Qwen {score.gemma_killed_qwen}",
        f"server player frags: {player_frags_text}",
        "markers: Qwen=blue, Gemma=magenta, bots/enemies=orange",
    ]
    pad = 7
    line_h = 14
    box_w = 610
    box_h = pad * 2 + line_h * len(lines)
    draw.rectangle((6, 6, 6 + box_w, 6 + box_h), fill=(0, 0, 0, 178))
    for i, text in enumerate(lines):
        draw.text((6 + pad, 6 + pad + i * line_h), text, font=font, fill=(255, 255, 255, 255))
    return img


class VideoRecorder:
    def __init__(self, path: Path, fps: float, width: int = 640, height: int = 360) -> None:
        if cv2 is None:
            raise RuntimeError(
                "OpenCV is required for video recording. Install it with: pip install opencv-python-headless"
            )
        self.path = path
        self.width = width
        self.height = height
        path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {path}")
        self.frames = 0

    def write_pil_rgb(self, img: Image.Image) -> None:
        if img.size != (self.width, self.height):
            img = img.resize((self.width, self.height), Image.Resampling.BILINEAR)
        rgb = np.asarray(img.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self.writer.write(bgr)
        self.frames += 1

    def close(self) -> None:
        self.writer.release()


def call_vllm_action(
    *,
    model_id: str,
    base_url: str,
    frame_id: int,
    image_b64: str,
    prompt_text: str,
    timeout_s: float,
    use_guided_choice: bool,
) -> Dict[str, Any]:
    started = time.perf_counter()
    client = OpenAI(api_key="EMPTY", base_url=base_url, timeout=timeout_s)
    data_uri = f"data:image/jpeg;base64,{image_b64}"

    kwargs: Dict[str, Any] = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            },
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 16,
    }

    extra_body: Dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": False}}

    # vLLM-specific: exact-choice structured decoding. If a model/server build does not support it,
    # run with --no-guided-choice and let parse_action discard invalid outputs.
    if use_guided_choice:
        extra_body["structured_outputs"] = {"choice": MODEL_ACTION_CHOICES}
    kwargs["extra_body"] = extra_body

    response = client.chat.completions.create(**kwargs)
    message = response.choices[0].message
    raw = message.content or ""
    if not raw:
        raw = str(getattr(message, "reasoning_content", "") or "")
    latency_ms = (time.perf_counter() - started) * 1000.0
    parsed, reason = parse_action(raw)
    return {
        "frame_id": frame_id,
        "raw": raw,
        "action": parsed,
        "valid": parsed is not None,
        "reason": reason,
        "latency_ms": latency_ms,
    }


def make_warmup_image_b64() -> str:
    img = Image.new("RGB", (640, 360), (20, 20, 20))
    draw = ImageDraw.Draw(img)
    draw.line((300, 180, 340, 180), fill=(180, 180, 180), width=2)
    draw.line((320, 160, 320, 200), fill=(180, 180, 180), width=2)
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=55, optimize=False)
    return base64.b64encode(out.getvalue()).decode("ascii")


def warmup_vllm_action(
    *,
    name: str,
    model_id: str,
    base_url: str,
    timeout_s: float,
    use_guided_choice: bool,
) -> None:
    prompt_text = "\n".join(
        [
            USER_TEXT_PROMPT,
            STRATEGY_TEXT,
            'Status: warmup_request=true; choose "FORWARD" unless the image clearly suggests another active move',
        ]
    )
    result = call_vllm_action(
        model_id=model_id,
        base_url=base_url,
        frame_id=-1,
        image_b64=make_warmup_image_b64(),
        prompt_text=prompt_text,
        timeout_s=timeout_s,
        use_guided_choice=use_guided_choice,
    )
    if not result.get("valid"):
        raise RuntimeError(f"{name} action warmup returned invalid output: {result}")
    print(
        f"[doom-vlm] {name} action warmup returned {result['action']} "
        f"in {float(result.get('latency_ms', 0.0)):.1f}ms",
        flush=True,
    )


def wait_for_vllm(
    base_url: str,
    timeout_s: int = 900,
    process: Optional[subprocess.Popen] = None,
    name: str = "vLLM",
) -> None:
    models_url = base_url.rstrip("/") + "/models"
    deadline = time.time() + timeout_s
    last_error = ""
    last_status_print = 0.0
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"{name} exited before becoming ready with code {process.returncode}. "
                f"Check its streamed output above and its log file under runs/*/."
            )
        try:
            r = requests.get(models_url, timeout=5)
            if r.status_code == 200:
                print(f"[doom-vlm] {name} is ready at {models_url}", flush=True)
                return
            last_error = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)

        now = time.time()
        if now - last_status_print >= 30:
            remaining = max(0, int(deadline - now))
            print(f"[doom-vlm] Waiting for {name} at {models_url}; {remaining}s left. Last: {last_error}", flush=True)
            last_status_print = now
        time.sleep(2)
    raise TimeoutError(f"{name} did not become ready at {models_url}. Last error: {last_error}")


def wait_for_vllm_servers(
    servers: List[Tuple[str, str, subprocess.Popen]],
    timeout_s: int,
) -> None:
    """Wait for all vLLM servers and fail fast if any child exits."""
    deadline = None if timeout_s <= 0 else time.time() + timeout_s
    ready: set[str] = set()
    last_errors: Dict[str, str] = {name: "" for name, _, _ in servers}
    last_status_print = 0.0

    while deadline is None or time.time() < deadline:
        for name, base_url, process in servers:
            if name in ready:
                continue

            if process.poll() is not None:
                raise RuntimeError(
                    f"{name} exited before becoming ready with code {process.returncode}. "
                    f"Check its streamed output above and its log file under runs/*/."
                )

            models_url = base_url.rstrip("/") + "/models"
            try:
                r = requests.get(models_url, timeout=5)
                if r.status_code == 200:
                    print(f"[doom-vlm] {name} is ready at {models_url}", flush=True)
                    ready.add(name)
                    continue
                last_errors[name] = f"HTTP {r.status_code}: {r.text[:200]}"
            except Exception as exc:  # noqa: BLE001
                last_errors[name] = repr(exc)

        if len(ready) == len(servers):
            return

        now = time.time()
        if now - last_status_print >= 30:
            remaining = "no startup timeout" if deadline is None else f"{max(0, int(deadline - now))}s left"
            waiting = ", ".join(name for name, _, _ in servers if name not in ready)
            details = "; ".join(
                f"{name}: {last_errors[name]}" for name, _, _ in servers if name not in ready and last_errors[name]
            )
            print(f"[doom-vlm] Waiting for vLLM servers ({waiting}); {remaining}. Last: {details}", flush=True)
            last_status_print = now

        time.sleep(2)

    waiting = ", ".join(name for name, _, _ in servers if name not in ready)
    raise TimeoutError(f"vLLM servers did not become ready in time: {waiting}. Last errors: {last_errors}")


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{num_bytes} B"


def hf_model_cache_dir(model_id: str) -> Path:
    hub = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub:
        hub_dir = Path(hub)
    else:
        hf_home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
        hub_dir = hf_home / "hub"
    return hub_dir / f"models--{model_id.replace('/', '--')}"


def scan_hf_model_cache(model_id: str) -> Dict[str, Any]:
    cache_dir = hf_model_cache_dir(model_id)
    stats: Dict[str, Any] = {
        "path": str(cache_dir),
        "bytes": 0,
        "files": 0,
        "incomplete_bytes": 0,
        "incomplete_files": 0,
        "snapshot_links": 0,
    }
    if not cache_dir.exists():
        return stats

    for path in cache_dir.rglob("*"):
        try:
            if path.is_symlink():
                stats["snapshot_links"] += 1
                continue
            if not path.is_file():
                continue
            size = path.stat().st_size
        except OSError:
            continue
        stats["bytes"] += size
        stats["files"] += 1
        if path.name.endswith(".incomplete"):
            stats["incomplete_bytes"] += size
            stats["incomplete_files"] += 1
    return stats


def find_local_hf_snapshot(model_id: str) -> Optional[Path]:
    cache_dir = hf_model_cache_dir(model_id)
    snapshots_dir = cache_dir / "snapshots"
    if not snapshots_dir.exists():
        return None

    ref_path = cache_dir / "refs" / "main"
    candidates: List[Path] = []
    if ref_path.exists():
        revision = ref_path.read_text(encoding="utf-8").strip()
        if revision:
            candidates.append(snapshots_dir / revision)
    candidates.extend(sorted((p for p in snapshots_dir.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True))

    for snapshot in candidates:
        if not snapshot.is_dir():
            continue
        if any(cache_dir.rglob("*.incomplete")):
            continue
        required = ("config.json", "tokenizer.json")
        if not all((snapshot / name).exists() for name in required):
            continue
        if not any((snapshot / name).exists() for name in ("model.safetensors", "model.safetensors.index.json")):
            continue
        broken_link = False
        for path in snapshot.rglob("*"):
            if path.is_symlink() and not path.exists():
                broken_link = True
                break
        if not broken_link:
            return snapshot
    return None


def resolve_vllm_model_arg(model_id: str, *, allow_hf_download: bool) -> Tuple[str, List[str]]:
    model_path = Path(model_id).expanduser()
    if model_path.exists():
        return str(model_path), []

    snapshot = find_local_hf_snapshot(model_id)
    if snapshot is not None:
        print(f"[doom-vlm] Using local HF snapshot for {model_id}: {snapshot}", flush=True)
        return str(snapshot), ["--served-model-name", model_id]

    if allow_hf_download:
        print(f"[doom-vlm] Local HF snapshot for {model_id} was not found; allowing Hugging Face download.", flush=True)
        return model_id, []

    raise FileNotFoundError(
        f"Local HF snapshot for {model_id} was not found or is incomplete under {hf_model_cache_dir(model_id)}. "
        "Run once with --allow-hf-download if you intentionally want to download it."
    )


def start_hf_cache_progress_monitor(
    models: List[Tuple[str, str]],
    interval_s: float,
) -> Tuple[threading.Event, threading.Thread]:
    stop_event = threading.Event()

    def summarize() -> str:
        parts = []
        total = 0
        for label, model_id in models:
            stats = scan_hf_model_cache(model_id)
            total += int(stats["bytes"])
            detail = (
                f"{label}={format_bytes(int(stats['bytes']))} "
                f"files={stats['files']} links={stats['snapshot_links']}"
            )
            if stats["incomplete_files"]:
                detail += (
                    f" incomplete={format_bytes(int(stats['incomplete_bytes']))}"
                    f"/{stats['incomplete_files']}"
                )
            parts.append(detail)
        return f"[doom-vlm] HF cache status: total={format_bytes(total)}; " + "; ".join(parts)

    def monitor() -> None:
        while not stop_event.is_set():
            print(summarize(), flush=True)
            stop_event.wait(interval_s)

    thread = threading.Thread(target=monitor, name="hf-cache-progress", daemon=True)
    thread.start()
    return stop_event, thread


def tee_process_output(process: subprocess.Popen, log_path: Path, prefix: str) -> threading.Thread:
    """Mirror a child process stream to both a file and docker-compose stdout.

    vLLM/Hugging Face download progress normally goes to stderr/stdout. Previous
    versions wrote that only to runs/latest/vllm_*.log, which made compose output
    look idle while downloads were actually in progress. This tee keeps the log
    file and also makes progress visible in `docker compose up`.
    """
    if process.stdout is None:
        raise RuntimeError("tee_process_output requires stdout=PIPE")

    def pump() -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        at_line_start = True
        try:
            with open(log_path, "w", encoding="utf-8", buffering=1) as log_file:
                while True:
                    chunk = process.stdout.read(1)
                    if chunk == "":
                        if process.poll() is not None:
                            break
                        time.sleep(0.05)
                        continue
                    log_file.write(chunk)
                    log_file.flush()
                    if at_line_start:
                        sys.stdout.write(f"[{prefix}] ")
                        at_line_start = False
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
                    if chunk in ("\n", "\r"):
                        at_line_start = True
            rc = process.poll()
            if rc is not None:
                print(f"[{prefix}] process exited with code {rc}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[{prefix}] log tee failed: {exc!r}", flush=True)

    thread = threading.Thread(target=pump, name=f"tee-{prefix}", daemon=True)
    thread.start()
    return thread


def launch_vllm_server(
    *,
    model_id: str,
    model_arg: str,
    served_model_args: List[str],
    port: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    kv_cache_memory_bytes: str,
    gpu_memory_utilization: float,
    extra_args: List[str],
    log_path: Path,
    log_prefix: str,
    cuda_visible_devices: str = "0",
) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    env.setdefault("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", "0")
    for noisy_vllm_build_var in ("VLLM_BUILD_URL", "VLLM_IMAGE_TAG", "VLLM_BUILD_PIPELINE", "VLLM_BUILD_COMMIT"):
        env.pop(noisy_vllm_build_var, None)

    cmd = [
        "vllm",
        "serve",
        model_arg,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dtype",
        "auto",
        "--trust-remote-code",
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        str(max_num_batched_tokens),
        "--limit-mm-per-prompt",
        '{"image": 1}',
    ] + served_model_args + extra_args
    if kv_cache_memory_bytes:
        cmd.extend(["--kv-cache-memory-bytes", kv_cache_memory_bytes])

    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("Launching:", " ".join(cmd), "| log:", log_path, flush=True)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        bufsize=1,
    )
    tee_process_output(proc, log_path, log_prefix)
    return proc


def configure_game(
    *,
    is_host: bool,
    port: int,
    name: str,
    color: int,
    scenario_config: Path,
    map_name: Optional[str],
    timelimit_min: float,
    render_window: bool,
) -> "vzd.DoomGame":
    game = vzd.DoomGame()
    game.load_config(str(scenario_config))
    if map_name:
        game.set_doom_map(map_name)

    # Prefer native 640x360 if the installed ViZDoom build exposes it.
    resolution = getattr(vzd.ScreenResolution, "RES_640X360", vzd.ScreenResolution.RES_640X480)
    game.set_screen_resolution(resolution)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_window_visible(render_window)
    game.set_sound_enabled(False)
    game.set_render_hud(True)
    game.set_render_weapon(True)
    game.set_render_messages(False)
    game.set_render_corpses(False)

    # Best-effort metadata for the third tactical map stream. Availability varies
    # by ViZDoom version/build, so every call is guarded.
    for method_name in ("set_labels_buffer_enabled", "set_objects_info_enabled", "set_sectors_info_enabled"):
        method = getattr(game, method_name, None)
        if callable(method):
            try:
                method(True)
            except Exception:
                pass

    try:
        game.clear_available_buttons()
    except AttributeError:
        pass
    for button_name in BUTTON_NAMES:
        game.add_available_button(getattr(vzd.Button, button_name))

    if is_host:
        game.add_game_args(
            f"-host 2 "
            f"-port {port} "
            f"+viz_connect_timeout 60 "
            f"-deathmatch "
            f"+timelimit {timelimit_min} "
            f"+sv_forcerespawn 1 "
            f"+sv_noautoaim 1 "
            f"+sv_respawnprotect 1 "
            f"+sv_spawnfarthest 1 "
            f"+sv_nocrouch 1 "
            f"+viz_respawn_delay 1 "
            f"+viz_nocheat 0 "
        )
    else:
        game.add_game_args(f"-join 127.0.0.1 -port {port} +viz_connect_timeout 60 ")

    game.add_game_args(f"+name {name} +colorset {color}")
    game.set_mode(vzd.Mode.ASYNC_PLAYER)
    return game


def init_multiplayer(host: "vzd.DoomGame", guest: "vzd.DoomGame") -> None:
    # host.init() blocks until the guest joins, so initialize them concurrently.
    with ThreadPoolExecutor(max_workers=2) as ex:
        host_future = ex.submit(host.init)
        time.sleep(0.5)
        guest_future = ex.submit(guest.init)
        host_future.result()
        guest_future.result()


def add_bots(host: "vzd.DoomGame", bot_count: int) -> None:
    for _ in range(bot_count):
        try:
            host.send_game_command("addbot")
            time.sleep(0.1)
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: could not add bot via send_game_command('addbot'): {exc}")
            print("Continuing without adding the remaining bots.")
            break


def write_decision_row(writer: csv.DictWriter, frame_id: int, agent: AgentRuntime, result: Dict[str, Any]) -> None:
    writer.writerow(
        {
            "t": time.time(),
            "frame_id": frame_id,
            "agent": agent.name,
            "model": agent.model_id,
            "valid": result.get("valid"),
            "action": result.get("action") or "",
            "held_action_after": agent.last_action,
            "latency_ms": f"{float(result.get('latency_ms', 0.0)):.2f}",
            "reason": result.get("reason", ""),
            "raw": str(result.get("raw", "")).replace("\n", "\\n")[:200],
        }
    )


def write_scoreboard_row(
    writer: csv.DictWriter,
    *,
    frame_id: int,
    qwen_frags: Optional[int],
    gemma_frags: Optional[int],
    qwen_deaths: Optional[int],
    gemma_deaths: Optional[int],
    score: MatchScore,
    player_frags: Dict[str, int],
) -> None:
    writer.writerow(
        {
            "t": time.time(),
            "frame_id": frame_id,
            "qwen_frags": qwen_frags if qwen_frags is not None else "",
            "gemma_frags": gemma_frags if gemma_frags is not None else "",
            "qwen_deaths": qwen_deaths if qwen_deaths is not None else "",
            "gemma_deaths": gemma_deaths if gemma_deaths is not None else "",
            "qwen_killed_gemma": score.qwen_killed_gemma,
            "gemma_killed_qwen": score.gemma_killed_qwen,
            "server_player_frags": json.dumps(player_frags, sort_keys=True),
        }
    )


def terminate_process(proc: Optional[subprocess.Popen], name: str) -> None:
    if proc is None or proc.poll() is not None:
        return
    print(f"Stopping {name}...")
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()


def timestamped_latest_dir(path: Path) -> Path:
    stamp = time.strftime("%y-%m-%d-%H-%M", time.localtime())
    candidate = path.with_name(f"{path.name}-{stamp}")
    if not candidate.exists():
        return candidate

    second_stamp = time.strftime("%y-%m-%d-%H-%M-%S", time.localtime())
    candidate = path.with_name(f"{path.name}-{second_stamp}")
    if not candidate.exists():
        return candidate

    for idx in range(2, 1000):
        numbered = path.with_name(f"{path.name}-{second_stamp}-{idx}")
        if not numbered.exists():
            return numbered
    raise RuntimeError(f"Could not find an unused archive name for {path}")


def prepare_output_dir(out_dir: Path) -> Path:
    if out_dir.name == "latest" and out_dir.exists():
        archived = timestamped_latest_dir(out_dir)
        print(f"[doom-vlm] Archiving previous latest run: {out_dir} -> {archived}", flush=True)
        out_dir.rename(archived)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def resolve_existing_path(value: str, *, default_base: Path = SCRIPT_DIR) -> Path:
    raw = Path(value).expanduser()
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend(
            [
                Path.cwd() / raw,
                default_base / raw,
                default_base / "scenarios" / raw,
                Path(vzd.scenarios_path) / raw,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find scenario config {value!r}. Searched: {searched}")


def resolve_map_argument(map_arg: str) -> Tuple[Path, Optional[str]]:
    """Return (scenario_config_path, doom_map_override).

    For backwards compatibility, bare values such as "map01" still select a map
    inside the default scenario config. Values ending in .cfg load that config
    directly, letting the config choose its own doom_map unless the user passes a
    separate bare map value.
    """
    if not map_arg:
        return DEFAULT_SCENARIO_CONFIG.resolve(), None

    if map_arg.lower().endswith(".cfg"):
        return resolve_existing_path(map_arg), None

    return DEFAULT_SCENARIO_CONFIG.resolve(), map_arg


def run_match(args: argparse.Namespace) -> None:
    out_dir = prepare_output_dir(Path(args.out_dir))
    scenario_config, doom_map_override = resolve_map_argument(args.map)
    map_label = doom_map_override or "(from config)"
    print(f"[doom-vlm] ViZDoom config: {scenario_config} | map: {map_label}", flush=True)

    qwen_proc: Optional[subprocess.Popen] = None
    gemma_proc: Optional[subprocess.Popen] = None
    cache_monitor_stop: Optional[threading.Event] = None
    cache_monitor_thread: Optional[threading.Thread] = None

    if not args.no_launch_vllm:
        qwen_extra = ["--moe-backend", "flashinfer_cutlass"]
        if args.qwen_extra:
            qwen_extra.extend(args.qwen_extra.split())

        gemma_extra: List[str] = []
        if args.gemma_extra:
            gemma_extra.extend(args.gemma_extra.split())

        qwen_model_arg, qwen_served_model_args = resolve_vllm_model_arg(
            args.qwen_model,
            allow_hf_download=args.allow_hf_download,
        )
        gemma_model_arg, gemma_served_model_args = resolve_vllm_model_arg(
            args.gemma_model,
            allow_hf_download=args.allow_hf_download,
        )

        cache_monitor_stop, cache_monitor_thread = start_hf_cache_progress_monitor(
            [("qwen", args.qwen_model), ("gemma", args.gemma_model)],
            interval_s=args.hf_cache_status_interval_s,
        )
        try:
            qwen_proc = launch_vllm_server(
                model_id=args.qwen_model,
                model_arg=qwen_model_arg,
                served_model_args=qwen_served_model_args,
                port=args.qwen_port,
                max_model_len=args.max_model_len,
                max_num_batched_tokens=args.max_num_batched_tokens,
                kv_cache_memory_bytes=args.qwen_kv_cache_memory_bytes,
                gpu_memory_utilization=args.qwen_gpu_mem,
                extra_args=qwen_extra,
                log_path=out_dir / "vllm_qwen.log",
                log_prefix="vllm-qwen",
                cuda_visible_devices=args.cuda_visible_devices,
            )
            gemma_proc = launch_vllm_server(
                model_id=args.gemma_model,
                model_arg=gemma_model_arg,
                served_model_args=gemma_served_model_args,
                port=args.gemma_port,
                max_model_len=args.max_model_len,
                max_num_batched_tokens=args.max_num_batched_tokens,
                kv_cache_memory_bytes=args.gemma_kv_cache_memory_bytes,
                gpu_memory_utilization=args.gemma_gpu_mem,
                extra_args=gemma_extra,
                log_path=out_dir / "vllm_gemma.log",
                log_prefix="vllm-gemma",
                cuda_visible_devices=args.cuda_visible_devices,
            )

            print("Waiting for vLLM servers...", flush=True)
            wait_for_vllm_servers(
                [
                    ("Qwen vLLM", args.qwen_url, qwen_proc),
                    ("Gemma vLLM", args.gemma_url, gemma_proc),
                ],
                timeout_s=args.vllm_startup_timeout_s,
            )
        except Exception:
            if not args.keep_vllm:
                terminate_process(qwen_proc, "qwen vLLM")
                terminate_process(gemma_proc, "gemma vLLM")
            raise
        finally:
            if cache_monitor_stop is not None:
                cache_monitor_stop.set()
            if cache_monitor_thread is not None:
                cache_monitor_thread.join(timeout=2)

    if not args.no_launch_vllm and not args.skip_vllm_action_warmup:
        try:
            warmup_timeout_s = max(args.request_timeout_s, 120.0)
            warmup_vllm_action(
                name="Qwen vLLM",
                model_id=args.qwen_model,
                base_url=args.qwen_url,
                timeout_s=warmup_timeout_s,
                use_guided_choice=not args.no_guided_choice,
            )
            warmup_vllm_action(
                name="Gemma vLLM",
                model_id=args.gemma_model,
                base_url=args.gemma_url,
                timeout_s=warmup_timeout_s,
                use_guided_choice=not args.no_guided_choice,
            )
        except Exception:
            if not args.keep_vllm:
                terminate_process(qwen_proc, "qwen vLLM")
                terminate_process(gemma_proc, "gemma vLLM")
            raise

    cluster = LocalCluster(
        n_workers=args.dask_workers,
        threads_per_worker=1,
        processes=True,
        dashboard_address=None,
    )
    client = Client(cluster)

    host = None
    guest = None
    qwen_video: Optional[VideoRecorder] = None
    gemma_video: Optional[VideoRecorder] = None
    map_video: Optional[VideoRecorder] = None
    decision_file = out_dir / "decisions.csv"
    scoreboard_file = out_dir / "scoreboard.csv"
    qwen_video_file = out_dir / "qwen_player_pov.mp4"
    gemma_video_file = out_dir / "gemma_player_pov.mp4"
    map_video_file = out_dir / "tactical_map.mp4"

    qwen = AgentRuntime(name="qwen_player", model_id=args.qwen_model, base_url=args.qwen_url)
    gemma = AgentRuntime(name="gemma_player", model_id=args.gemma_model, base_url=args.gemma_url)
    score = MatchScore()

    try:
        print("Starting ViZDoom multiplayer match...")
        host = configure_game(
            is_host=True,
            port=args.doom_port,
            name="QwenVLM",
            color=0,
            scenario_config=scenario_config,
            map_name=doom_map_override,
            timelimit_min=max(args.duration_s / 60.0, 1.0),
            render_window=args.render_window,
        )
        guest = configure_game(
            is_host=False,
            port=args.doom_port,
            name="GemmaVLM",
            color=3,
            scenario_config=scenario_config,
            map_name=doom_map_override,
            timelimit_min=max(args.duration_s / 60.0, 1.0),
            render_window=args.render_window,
        )
        init_multiplayer(host, guest)

        if args.bots > 0:
            print(f"Adding {args.bots} bots...")
            add_bots(host, args.bots)

        if not args.no_record_video:
            qwen_video = VideoRecorder(qwen_video_file, fps=args.record_fps)
            gemma_video = VideoRecorder(gemma_video_file, fps=args.record_fps)
            if not args.no_record_map_video:
                map_video = VideoRecorder(map_video_file, fps=args.record_fps)
            print(f"Recording POV videos: {qwen_video_file} and {gemma_video_file}")
            if map_video is not None:
                print(f"Recording tactical map video: {map_video_file}")

        with open(decision_file, "w", newline="", encoding="utf-8") as f, open(
            scoreboard_file, "w", newline="", encoding="utf-8"
        ) as score_f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "t",
                    "frame_id",
                    "agent",
                    "model",
                    "valid",
                    "action",
                    "held_action_after",
                    "latency_ms",
                    "reason",
                    "raw",
                ],
            )
            writer.writeheader()
            score_writer = csv.DictWriter(
                score_f,
                fieldnames=[
                    "t",
                    "frame_id",
                    "qwen_frags",
                    "gemma_frags",
                    "qwen_deaths",
                    "gemma_deaths",
                    "qwen_killed_gemma",
                    "gemma_killed_qwen",
                    "server_player_frags",
                ],
            )
            score_writer.writeheader()

            start_time = time.perf_counter()
            frame_id = 0
            last_print = time.perf_counter()

            while True:
                if args.duration_s > 0 and (time.perf_counter() - start_time) >= args.duration_s:
                    break
                if host.is_episode_finished() or guest.is_episode_finished():
                    break

                h_state = host.get_state()
                g_state = guest.get_state()
                if h_state is None or g_state is None:
                    time.sleep(0.01)
                    continue

                qwen_pose = get_player_pose(host)
                gemma_pose = get_player_pose(guest)
                update_motion_state(qwen, qwen_pose)
                update_motion_state(gemma, gemma_pose)
                qwen_frags = game_var(host, "FRAGCOUNT")
                gemma_frags = game_var(guest, "FRAGCOUNT")
                qwen_deaths = game_var(host, "DEATHCOUNT")
                gemma_deaths = game_var(guest, "DEATHCOUNT")
                player_frags = get_player_frag_table(host)
                score.update(
                    qwen_frags=qwen_frags,
                    gemma_frags=gemma_frags,
                    qwen_deaths=qwen_deaths,
                    gemma_deaths=gemma_deaths,
                )
                write_scoreboard_row(
                    score_writer,
                    frame_id=frame_id,
                    qwen_frags=qwen_frags,
                    gemma_frags=gemma_frags,
                    qwen_deaths=qwen_deaths,
                    gemma_deaths=gemma_deaths,
                    score=score,
                    player_frags=player_frags,
                )

                h_img = encode_frame_to_jpeg_b64(h_state.screen_buffer, quality=args.jpeg_quality)
                g_img = encode_frame_to_jpeg_b64(g_state.screen_buffer, quality=args.jpeg_quality)

                qwen_prompt = build_user_text_prompt(
                    agent=qwen,
                    frame_id=frame_id,
                    pose=qwen_pose,
                    opponent_pose=gemma_pose,
                    health=game_var(host, "HEALTH"),
                    armor=game_var(host, "ARMOR"),
                    ammo=game_var(host, "SELECTED_WEAPON_AMMO"),
                    frags=qwen_frags,
                    deaths=qwen_deaths,
                )
                gemma_prompt = build_user_text_prompt(
                    agent=gemma,
                    frame_id=frame_id,
                    pose=gemma_pose,
                    opponent_pose=qwen_pose,
                    health=game_var(guest, "HEALTH"),
                    armor=game_var(guest, "ARMOR"),
                    ammo=game_var(guest, "SELECTED_WEAPON_AMMO"),
                    frags=gemma_frags,
                    deaths=gemma_deaths,
                )

                for agent, img, prompt_text in ((qwen, h_img, qwen_prompt), (gemma, g_img, gemma_prompt)):
                    if agent.pending is not None and agent.pending.done():
                        try:
                            result = agent.pending.result()
                            result = apply_decision_safety(agent, result, frame_id)
                            agent.last_raw = str(result.get("raw", ""))
                            agent.last_latency_ms = float(result.get("latency_ms", 0.0))
                            if result.get("valid") and result.get("action") in ACTION_TO_BUTTONS:
                                next_action = result["action"]
                                if next_action == agent.last_action:
                                    agent.same_action_streak += 1
                                else:
                                    agent.same_action_streak = 1
                                agent.last_action = next_action
                                agent.decisions += 1
                            else:
                                agent.invalid += 1
                            write_decision_row(writer, frame_id, agent, result)
                        except Exception as exc:  # noqa: BLE001
                            agent.errors += 1
                            write_decision_row(
                                writer,
                                frame_id,
                                agent,
                                {
                                    "valid": False,
                                    "action": None,
                                    "latency_ms": 0.0,
                                    "reason": "exception",
                                    "raw": repr(exc),
                                },
                            )
                        finally:
                            agent.pending = None

                    if agent.pending is None:
                        agent.pending = client.submit(
                            call_vllm_action,
                            model_id=agent.model_id,
                            base_url=agent.base_url,
                            frame_id=frame_id,
                            image_b64=img,
                            prompt_text=prompt_text,
                            timeout_s=args.request_timeout_s,
                            use_guided_choice=not args.no_guided_choice,
                            pure=False,
                        )

                qwen.applied_action = choose_applied_action(qwen, frame_id, args.stuck_escape_frames)
                gemma.applied_action = choose_applied_action(gemma, frame_id, args.stuck_escape_frames)

                if qwen_video is not None:
                    qwen_video.write_pil_rgb(
                        overlay_player_frame(
                            screen_buffer=h_state.screen_buffer,
                            agent=qwen,
                            frame_id=frame_id,
                            fragcount=qwen_frags,
                            deathcount=qwen_deaths,
                        )
                    )
                if gemma_video is not None:
                    gemma_video.write_pil_rgb(
                        overlay_player_frame(
                            screen_buffer=g_state.screen_buffer,
                            agent=gemma,
                            frame_id=frame_id,
                            fragcount=gemma_frags,
                            deathcount=gemma_deaths,
                        )
                    )
                if map_video is not None:
                    map_lines = extract_map_lines(h_state, g_state)
                    visible_entities = extract_visible_entities(
                        h_state,
                        g_state,
                        model_poses=(qwen_pose, gemma_pose),
                        max_entities=args.bots,
                    )
                    map_video.write_pil_rgb(
                        draw_tactical_map_frame(
                            qwen_pose=qwen_pose,
                            gemma_pose=gemma_pose,
                            entities=visible_entities,
                            map_lines=map_lines,
                            qwen=qwen,
                            gemma=gemma,
                            score=score,
                            qwen_frags=qwen_frags,
                            gemma_frags=gemma_frags,
                            qwen_deaths=qwen_deaths,
                            gemma_deaths=gemma_deaths,
                            player_frags=player_frags,
                            expected_bots=args.bots,
                            frame_id=frame_id,
                            units_per_px=args.map_video_scale,
                        )
                    )

                host.make_action(ACTION_TO_BUTTONS[qwen.applied_action], args.tics_per_step)
                guest.make_action(ACTION_TO_BUTTONS[gemma.applied_action], args.tics_per_step)

                if host.is_player_dead():
                    host.respawn_player()
                if guest.is_player_dead():
                    guest.respawn_player()

                frame_id += 1

                now = time.perf_counter()
                if now - last_print >= args.status_interval_s:
                    elapsed = now - start_time
                    print(
                        f"t={elapsed:7.1f}s frame={frame_id:6d} "
                        f"qwen={qwen.applied_action:>20s} {qwen.last_latency_ms:7.1f}ms "
                        f"gemma={gemma.applied_action:>20s} {gemma.last_latency_ms:7.1f}ms "
                        f"frags=({qwen_frags},{gemma_frags}) "
                        f"h2h=({score.qwen_killed_gemma},{score.gemma_killed_qwen}) "
                        f"valid=({qwen.decisions},{gemma.decisions}) invalid=({qwen.invalid},{gemma.invalid}) "
                        f"stuck=({qwen.motionless_frames},{gemma.motionless_frames})"
                    )
                    last_print = now
                    f.flush()
                    score_f.flush()

        summary = {
            "qwen": {
                "model": qwen.model_id,
                "valid_decisions": qwen.decisions,
                "invalid_decisions": qwen.invalid,
                "errors": qwen.errors,
                "noop_substitutions": qwen.noop_substitutions,
                "escape_frames": qwen.escape_frames,
                "fragcount": game_var(host, "FRAGCOUNT"),
                "deathcount": game_var(host, "DEATHCOUNT"),
                "player_number": game_var(host, "PLAYER_NUMBER"),
            },
            "gemma": {
                "model": gemma.model_id,
                "valid_decisions": gemma.decisions,
                "invalid_decisions": gemma.invalid,
                "errors": gemma.errors,
                "noop_substitutions": gemma.noop_substitutions,
                "escape_frames": gemma.escape_frames,
                "fragcount": game_var(guest, "FRAGCOUNT"),
                "deathcount": game_var(guest, "DEATHCOUNT"),
                "player_number": game_var(guest, "PLAYER_NUMBER"),
            },
            "scoreboard": {
                "csv": str(scoreboard_file),
                "server_player_frags": get_player_frag_table(host),
                "qwen_killed_gemma": score.qwen_killed_gemma,
                "gemma_killed_qwen": score.gemma_killed_qwen,
            },
            "scenario": {
                "config": str(scenario_config),
                "map": doom_map_override,
            },
            "decisions_csv": str(decision_file),
            "videos": {
                "qwen_player_pov": str(qwen_video_file) if qwen_video is not None else None,
                "gemma_player_pov": str(gemma_video_file) if gemma_video is not None else None,
                "tactical_map": str(map_video_file) if map_video is not None else None,
                "recorded_frames": {
                    "qwen": qwen_video.frames if qwen_video is not None else 0,
                    "gemma": gemma_video.frames if gemma_video is not None else 0,
                    "tactical_map": map_video.frames if map_video is not None else 0,
                },
            },
            "args": vars(args),
        }
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))

    finally:
        for agent in (qwen, gemma):
            if agent.pending is not None:
                agent.pending.cancel()
        if qwen_video is not None:
            qwen_video.close()
        if gemma_video is not None:
            gemma_video.close()
        if map_video is not None:
            map_video.close()
        if host is not None:
            host.close()
        if guest is not None:
            guest.close()
        client.close()
        cluster.close()
        if not args.keep_vllm:
            terminate_process(qwen_proc, "qwen vLLM")
            terminate_process(gemma_proc, "gemma vLLM")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--qwen-model", default=QWEN_MODEL)
    p.add_argument("--gemma-model", default=GEMMA_MODEL)
    p.add_argument("--qwen-port", type=int, default=8001)
    p.add_argument("--gemma-port", type=int, default=8002)
    p.add_argument("--qwen-url", default="http://127.0.0.1:8001/v1")
    p.add_argument("--gemma-url", default="http://127.0.0.1:8002/v1")
    p.add_argument("--no-launch-vllm", action="store_true")
    p.add_argument("--keep-vllm", action="store_true")
    p.add_argument("--no-guided-choice", action="store_true")
    p.add_argument("--allow-hf-download", action="store_true")
    p.add_argument("--skip-vllm-action-warmup", action="store_true")
    p.add_argument("--qwen-extra", default="")
    p.add_argument("--gemma-extra", default="")
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-num-batched-tokens", type=int, default=3072)
    p.add_argument("--qwen-kv-cache-memory-bytes", default="2G")
    p.add_argument("--gemma-kv-cache-memory-bytes", default="2G")
    p.add_argument("--hf-cache-status-interval-s", type=float, default=10.0)
    p.add_argument("--qwen-gpu-mem", type=float, default=0.34)
    p.add_argument("--gemma-gpu-mem", type=float, default=0.62)
    p.add_argument("--cuda-visible-devices", default="0")
    p.add_argument(
        "--vllm-startup-timeout-s",
        type=int,
        default=0,
        help="Seconds to wait for launched vLLM servers. Use 0 to wait indefinitely during first-run model downloads.",
    )
    p.add_argument("--request-timeout-s", type=float, default=30.0)
    p.add_argument("--doom-port", type=int, default=5029)
    p.add_argument(
        "--map",
        default=DEFAULT_SCENARIO_CONFIG.name,
        help="Doom map name such as map01, or a path to a ViZDoom .cfg scenario file.",
    )
    p.add_argument("--bots", type=int, default=0)
    p.add_argument("--duration-s", type=int, default=300)
    p.add_argument("--tics-per-step", type=int, default=1)
    p.add_argument("--stuck-escape-frames", type=int, default=24)
    p.add_argument("--jpeg-quality", type=int, default=55)
    p.add_argument("--render-window", action="store_true")
    p.add_argument("--no-record-video", action="store_true")
    p.add_argument("--no-record-map-video", action="store_true")
    p.add_argument("--record-fps", type=float, default=35.0)
    p.add_argument("--map-video-scale", type=float, default=20.0, help="World units per pixel for tactical map; auto-zooms out when needed.")
    p.add_argument("--dask-workers", type=int, default=4)
    p.add_argument("--status-interval-s", type=float, default=5.0)
    p.add_argument("--out-dir", default="runs/latest")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    run_match(args)


if __name__ == "__main__":
    main()
