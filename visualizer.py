from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from PIL import Image, ImageDraw, ImageFont

from environment import GRID_SIZE, Action


class MazeVisualizer:
    """Small PIL visualizer used by main.py for GIFs and PNG snapshots."""

    def __init__(self, output_dir: str | Path, cell_px: int = 10, header_px: int = 56) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cell_px = cell_px
        self.header_px = header_px
        self.width = GRID_SIZE * cell_px
        self.height = GRID_SIZE * cell_px + header_px
        self.font = ImageFont.load_default()

    def render_frame(
        self,
        env,
        agent_memory: Optional[dict] = None,
        title: str = "Maze Run",
        episode: int = 0,
        turn: int = 0,
        show_true_hazards: bool = False,
        show_agent_memory: bool = True,
    ) -> Image.Image:
        img = Image.new("RGB", (self.width, self.height), "white")
        draw = ImageDraw.Draw(img)

        draw.rectangle([0, 0, self.width, self.header_px], fill=(245, 245, 245))
        draw.text((8, 8), title, fill=(0, 0, 0), font=self.font)
        draw.text(
            (8, 28),
            f"episode={episode} turn={turn} pos={getattr(env, 'current_position', None)} "
            f"fire_phase={getattr(env, 'fire_phase', 0)} deaths={getattr(env, 'deaths', 0)}",
            fill=(0, 0, 0),
            font=self.font,
        )

        # Base grid.
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                self._cell(draw, (x, y), fill=(255, 255, 255), outline=(235, 235, 235))

        # Agent memory overlay first, then true hazards/path/agent on top.
        if agent_memory and show_agent_memory:
            for pos in agent_memory.get("safe_cells", set()):
                self._cell(draw, pos, fill=(220, 238, 255))
            for pos in agent_memory.get("episode_path", []):
                self._cell(draw, pos, fill=(180, 220, 255))
            for cells in agent_memory.get("death_cells_by_phase", {}).values():
                for pos in cells:
                    self._cell(draw, pos, fill=(255, 210, 210))
            for pos in agent_memory.get("confusion_cells", set()):
                self._cell(draw, pos, fill=(255, 255, 185))
            for src, dst in agent_memory.get("teleports", {}).items():
                self._cell(draw, src, fill=(218, 205, 255))
                self._cell(draw, dst, fill=(235, 225, 255))

        if show_true_hazards:
            for pos in getattr(env, "fire_cells", set()):
                self._cell(draw, pos, fill=(255, 115, 65))
            for pos in getattr(env, "confusion_cells", set()):
                self._cell(draw, pos, fill=(255, 235, 70))
            for src, dst in getattr(env, "teleport_map", {}).items():
                self._cell(draw, src, fill=(145, 85, 220))
                self._cell(draw, dst, fill=(190, 165, 230))
            for pos, action in getattr(env, "arrow_tiles", {}).items():
                self._cell(draw, pos, fill=(90, 160, 255))
                self._draw_arrow(draw, pos, action)

        # Draw walls from blocked edges.
        for a, b in getattr(env, "blocked_edges", set()):
            if a > b:
                continue
            self._wall(draw, a, b)

        # Start / goal / current position.
        if hasattr(env, "start"):
            self._cell(draw, env.start, fill=(70, 200, 100))
            draw.text(self._text_pos(env.start), "S", fill=(0, 0, 0), font=self.font)
        if hasattr(env, "goal"):
            self._cell(draw, env.goal, fill=(255, 215, 70))
            draw.text(self._text_pos(env.goal), "G", fill=(0, 0, 0), font=self.font)
        if hasattr(env, "current_position"):
            self._cell(draw, env.current_position, fill=(30, 50, 220))

        return img

    def save_gif(self, frames: Iterable[Image.Image], filename: str, duration: int = 180) -> Path:
        path = self.output_dir / filename
        frame_list = list(frames)
        if not frame_list:
            # Create a placeholder so callers can still rely on the returned path.
            frame_list = [Image.new("RGB", (self.width, self.height), "white")]
        frame_list[0].save(path, save_all=True, append_images=frame_list[1:], duration=duration, loop=0)
        return path

    def save_explored_map(self, env, agent_memory: Optional[dict], filename: str) -> Path:
        img = self.render_frame(
            env=env,
            agent_memory=agent_memory,
            title="Explored / Learned Map",
            episode=0,
            turn=getattr(env, "turns_taken", 0),
            show_true_hazards=False,
            show_agent_memory=True,
        )
        path = self.output_dir / filename
        img.save(path)
        return path

    def _cell(self, draw: ImageDraw.ImageDraw, pos, fill, outline: Optional[tuple[int, int, int]] = None) -> None:
        x, y = pos
        px = x * self.cell_px
        py = self.header_px + y * self.cell_px
        draw.rectangle([px, py, px + self.cell_px - 1, py + self.cell_px - 1], fill=fill, outline=outline)

    def _wall(self, draw: ImageDraw.ImageDraw, a, b) -> None:
        ax, ay = a
        bx, by = b
        p = self.cell_px
        top = self.header_px
        if ax == bx and abs(ay - by) == 1:
            y = max(ay, by) * p + top
            x0 = ax * p
            draw.line([x0, y, x0 + p, y], fill=(0, 0, 0), width=2)
        elif ay == by and abs(ax - bx) == 1:
            x = max(ax, bx) * p
            y0 = ay * p + top
            draw.line([x, y0, x, y0 + p], fill=(0, 0, 0), width=2)

    def _text_pos(self, pos) -> tuple[int, int]:
        x, y = pos
        return x * self.cell_px + 2, self.header_px + y * self.cell_px

    def _draw_arrow(self, draw: ImageDraw.ImageDraw, pos, action: Action) -> None:
        x, y = pos
        cx = x * self.cell_px + self.cell_px // 2
        cy = self.header_px + y * self.cell_px + self.cell_px // 2
        d = max(3, self.cell_px // 3)
        if action == Action.MOVE_UP:
            pts = [(cx, cy - d), (cx - d, cy + d), (cx + d, cy + d)]
        elif action == Action.MOVE_DOWN:
            pts = [(cx, cy + d), (cx - d, cy - d), (cx + d, cy - d)]
        elif action == Action.MOVE_LEFT:
            pts = [(cx - d, cy), (cx + d, cy - d), (cx + d, cy + d)]
        else:
            pts = [(cx + d, cy), (cx - d, cy - d), (cx - d, cy + d)]
        draw.polygon(pts, fill=(0, 0, 0))
