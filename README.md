# Silent Cartographer Maze Solver - Third Check-in

This package is set up for the workflow you requested:

1. **Train/practice on maze-alpha using HONEST mode**.
   - The agent does not use the image-derived map for planning during this phase.
   - It learns from `TurnResult` feedback: wall hits, deaths, teleports, confusion, pushes, and current position.

2. **Solve/evaluate using HYBRID mode**.
   - After honest training, the same agent switches to hybrid solving.
   - Hybrid mode loads the parsed maze model and uses hazard-aware time-expanded A* over `(position, time mod fire cycle)`.
   - This handles the rotating fire/death pits because a cell's safety depends on time, not just location.

Maze-beta and maze-gamma are **not trained on**. They are evaluated directly after alpha training.

## Files

- `main.py` - command-line runner for alpha/beta/gamma.
- `environment.py` - environment API, hazard mechanics, image parsing.
- `agent.py` - hazard-aware RL/planning agent.
- `visualizer.py` - saves learned maps, final path images, and GIFs.
- `maze_alpha/maze_1.png` - alpha maze.
- `maze_beta/maze_1.png` - beta maze.
- `maze_gamma/maze_1.png` - gamma maze.

## Install

```bash
pip install -r requirements.txt
```

## Default run: honest training, hybrid solving

You do **not** need to pass `--mode` anymore. The default is now `--mode auto`, which means:

- training mode = `honest`
- solving/evaluation mode = `hybrid`

```bash
python main.py \
  --train-maze maze_alpha/maze_1.png \
  --test-maze maze_beta/maze_1.png \
  --gamma-maze maze_gamma/maze_1.png \
  --train-episodes 5 \
  --eval-episodes 5 \
  --max-turns 10000 \
  --frame-stride 1 \
  --gif-duration 100 \
  --output-dir results
```

## Optional mode overrides

Force hybrid for both training and solving:

```bash
python main.py --train-maze maze_alpha/maze_1.png --test-maze maze_beta/maze_1.png --mode hybrid
```

Force honest for both training and solving:

```bash
python main.py --train-maze maze_alpha/maze_1.png --test-maze maze_beta/maze_1.png --mode honest
```

Mix modes manually:

```bash
python main.py \
  --train-maze maze_alpha/maze_1.png \
  --test-maze maze_beta/maze_1.png \
  --train-mode honest \
  --solve-mode hybrid
```

## GIF settings

For smoother GIFs, use:

```bash
--frame-stride 1 --gif-duration 100 --max-gif-frames 1200
```

- `--frame-stride 1` saves every turn.
- `--gif-duration 100` gives a smooth presentation speed. The default is now 100 ms/frame.
- Increase `--gif-duration` to slow it down.


## Teleport behavior fix

Teleports are paired/two-way: stepping on either endpoint sends the agent to the other endpoint. The environment applies exactly one teleport per cell entry, so landing on the paired pad does not instantly bounce the agent back in the same action.


## Latest fixes in this remake

- Default workflow is `auto`: honest training on alpha, hybrid solving/evaluation.
- Teleports are two-way pairs.
- Teleport execution is one-hop per cell entry, so the agent does not instantly bounce back.
- GIF defaults are smoother: `--frame-stride 1`, `--gif-duration 100`, `--max-gif-frames 1200`.
- Maze-beta and maze-gamma are evaluated without a training loop on those mazes.
