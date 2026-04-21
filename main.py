from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

from environment import MazeEnvironment, TurnResult
from agent import HazardAwareRLAgent
from visualizer import MazeVisualizer


METHOD_JUSTIFICATION = (
    "Chosen method: hybrid reinforcement learning plus model-based planning. "
    "Training/practice uses HONEST mode, so the agent updates Q-values and transition "
    "memory only from TurnResult feedback. Solving/evaluation then uses HYBRID mode: "
    "the image-derived environment model is loaded into the same agent and the planner "
    "runs hazard-aware time-expanded A* over (position, time mod fire-cycle). This is "
    "necessary because rotating fire makes safety depend on both location and time."
)


def make_output_dir(base_output_dir: Optional[str], maze_path: str, label: str = "") -> Path:
    """
    Default rule:
        --train-maze maze_alpha/maze_1.png  -> outputs go in maze_alpha/
        --test-maze  maze_beta/maze_1.png   -> outputs go in maze_beta/

    If --output-dir is provided, each maze gets a subfolder under that folder.
    """
    maze_file = Path(maze_path).expanduser()
    if base_output_dir is None or str(base_output_dir).strip() == "":
        output_dir = maze_file.resolve().parent
    else:
        safe_label = label or maze_file.stem
        output_dir = Path(base_output_dir).expanduser().resolve() / safe_label
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def output_name(maze_path: str, suffix: str) -> str:
    return f"{Path(maze_path).stem}_{suffix}"


def print_metrics(metrics: dict, prefix: str = "") -> None:
    title = f"=== {prefix} Metrics ===" if prefix else "=== Metrics ==="
    print(f"\n{title}")
    print(f"Success Rate: {metrics['success_rate']:.2f}%")
    print(f"Average Path Length: {metrics['average_path_length']}")
    print(f"Average Turns to Solution: {metrics['average_turns_to_solution']}")
    print(f"Death Rate: {metrics['death_rate']:.4f}")
    print(f"Exploration Efficiency: {metrics.get('exploration_efficiency', 0.0):.4f}")
    print(f"Map Completeness: {metrics.get('map_completeness', 0.0):.4f}")
    print(f"Average Confusions: {metrics.get('average_confusions', 0.0):.2f}")
    print(f"Average Arrow Pushes: {metrics.get('average_arrow_pushes', 0.0):.2f}")


def configure_agent_for_maze(
    agent: HazardAwareRLAgent,
    env: MazeEnvironment,
    solver_mode: str,
    *,
    reset_spatial_memory: bool,
) -> None:
    if reset_spatial_memory:
        # Keep Q/policy, remove maze-specific walls/hazards from the previous maze.
        agent.prepare_for_new_maze(env.start, env.goal)

    agent.start_pos = env.start
    agent.goal_pos = env.goal
    agent.fire_rotation_interval = env.fire_rotation_interval
    agent.fire_cycle = env.fire_rotation_interval * 4

    if solver_mode == "hybrid":
        agent.load_detected_hazards_from_environment(
            env,
            enable_image_guidance=True,
            reveal_to_agent_memory=False,
        )
    else:
        agent.use_image_guidance = False
        agent.image_guided_actions.clear()


def run_training(
    maze_path: str,
    episodes: int,
    max_turns: int,
    frame_stride: int,
    max_gif_frames: int,
    gif_duration: int,
    output_dir: Path,
    solver_mode: str,
) -> Tuple[HazardAwareRLAgent, List[dict]]:
    env = MazeEnvironment(maze_path)
    agent = HazardAwareRLAgent(
        start_pos=env.start,
        goal_pos=env.goal,
        fire_rotation_interval=env.fire_rotation_interval,
    )
    configure_agent_for_maze(agent, env, solver_mode, reset_spatial_memory=True)

    visualizer = MazeVisualizer(output_dir=output_dir)
    training_stats: List[dict] = []
    learning_frames = []

    print(f"=== Training on MAZE-ALPHA only ({solver_mode.upper()} mode) ===")
    print("Maze path:", maze_path)
    print("Start:", env.start)
    print("Goal:", env.goal)
    print("Hazard summary:", env.get_hazard_summary())
    print(METHOD_JUSTIFICATION)
    print()

    for episode in range(1, episodes + 1):
        env.reset()
        agent.reset_episode(env.start, env.goal, keep_learning_memory=True)

        last_result: Optional[TurnResult] = None
        for turn in range(max_turns):
            actions = agent.plan_turn(last_result)
            last_result = env.step(actions)

            capture_frame = (
                len(learning_frames) < max_gif_frames
                and turn % max(1, frame_stride) == 0
                and (episode == 1 or episode == episodes or episode % max(1, episodes // 5) == 0)
            )
            if capture_frame:
                learning_frames.append(
                    visualizer.render_frame(
                        env=env,
                        agent_memory=agent.get_known_map(),
                        title="Training / Learning Process",
                        episode=episode,
                        turn=turn,
                        show_true_hazards=False,
                        show_agent_memory=True,
                    )
                )
            if last_result.is_goal_reached:
                break

        stats = env.get_episode_stats()
        training_stats.append(stats)
        agent.end_episode(goal_reached=stats["goal_reached"])
        print(
            f"train_episode={episode:03d} goal={stats['goal_reached']} "
            f"turns={stats['turns_taken']} deaths={stats['deaths']} "
            f"path_length={stats['path_length']} explored={stats['cells_explored']} "
            f"epsilon={agent.epsilon:.3f}"
        )

    if learning_frames:
        gif_path = visualizer.save_gif(
            learning_frames,
            filename=output_name(maze_path, "training_process.gif"),
            duration=gif_duration,
        )
        print("Saved training GIF:", gif_path)

    explored_path = visualizer.save_explored_map(
        env=env,
        agent_memory=agent.get_known_map(),
        filename=output_name(maze_path, "learned_map.png"),
    )
    print("Saved learned map:", explored_path)
    return agent, training_stats


def run_evaluation(
    maze_path: str,
    agent: HazardAwareRLAgent,
    episodes: int,
    max_turns: int,
    frame_stride: int,
    max_gif_frames: int,
    gif_duration: int,
    output_dir: Path,
    solver_mode: str,
    label: str,
    *,
    reset_spatial_memory: bool,
) -> List[dict]:
    env = MazeEnvironment(maze_path)
    configure_agent_for_maze(agent, env, solver_mode, reset_spatial_memory=reset_spatial_memory)
    visualizer = MazeVisualizer(output_dir=output_dir)

    eval_stats: List[dict] = []
    old_epsilon = agent.epsilon
    agent.epsilon = max(agent.min_epsilon, 0.04)

    best_success_stats = None
    best_success_frames = []
    best_success_image = None
    best_attempt_stats = None
    best_attempt_frames = []
    best_attempt_image = None

    print(f"\n=== Evaluation on {label.upper()} ({solver_mode.upper()} mode) ===")
    if label.lower().startswith("beta") or label.lower().startswith("gamma"):
        print("No training loop is run on this maze. The alpha-trained agent is solved/evaluated directly.")
    print("Maze path:", maze_path)
    print("Start:", env.start, "Goal:", env.goal, "Hazards:", env.get_hazard_summary())

    for episode in range(1, episodes + 1):
        env.reset()
        # Keep the same model. For beta/gamma, spatial memory was cleared once before episode 1;
        # inside the five official attempts, the agent may still use normal TurnResult feedback.
        agent.reset_episode(env.start, env.goal, keep_learning_memory=True)

        last_result: Optional[TurnResult] = None
        current_frames = []

        for turn in range(max_turns):
            actions = agent.plan_turn(last_result)
            last_result = env.step(actions)

            if len(current_frames) < max_gif_frames and turn % max(1, frame_stride) == 0:
                current_frames.append(
                    visualizer.render_frame(
                        env=env,
                        agent_memory=agent.get_known_map(),
                        title=f"{label} Evaluation",
                        episode=episode,
                        turn=turn,
                        show_true_hazards=True,
                        show_agent_memory=True,
                    )
                )

            if last_result.is_goal_reached:
                current_frames.append(
                    visualizer.render_frame(
                        env=env,
                        agent_memory=agent.get_known_map(),
                        title=f"{label} Solved",
                        episode=episode,
                        turn=turn,
                        show_true_hazards=True,
                        show_agent_memory=True,
                    )
                )
                break

        stats = env.get_episode_stats()
        eval_stats.append(stats)
        print(
            f"{label}_episode={episode:02d} goal={stats['goal_reached']} "
            f"turns={stats['turns_taken']} deaths={stats['deaths']} "
            f"path_length={stats['path_length']} explored={stats['cells_explored']} "
            f"eff={stats['exploration_efficiency']:.3f}"
        )

        final_image = visualizer.render_frame(
            env=env,
            agent_memory=agent.get_known_map(),
            title=f"{label} Final Snapshot",
            episode=episode,
            turn=stats["turns_taken"],
            show_true_hazards=True,
            show_agent_memory=True,
        )

        if stats["goal_reached"]:
            if best_success_stats is None or stats["turns_taken"] < best_success_stats["turns_taken"]:
                best_success_stats = stats
                best_success_frames = list(current_frames)
                best_success_image = final_image
        if best_attempt_stats is None or stats["cells_explored"] > best_attempt_stats["cells_explored"]:
            best_attempt_stats = stats
            best_attempt_frames = list(current_frames)
            best_attempt_image = final_image

    if best_success_stats is not None:
        gif_path = visualizer.save_gif(
            best_success_frames,
            filename=output_name(maze_path, f"{label}_solution.gif"),
            duration=gif_duration,
        )
        final_path_path = output_dir / output_name(maze_path, f"{label}_final_path.png")
        best_success_image.save(final_path_path)
        print("Saved final solution GIF:", gif_path)
        print("Saved final path image:", final_path_path)
    else:
        gif_path = visualizer.save_gif(
            best_attempt_frames,
            filename=output_name(maze_path, f"{label}_best_attempt.gif"),
            duration=gif_duration,
        )
        attempt_path = output_dir / output_name(maze_path, f"{label}_best_attempt_path.png")
        best_attempt_image.save(attempt_path)
        print("No episode reached the goal.")
        print("Saved best attempt GIF:", gif_path)
        print("Saved best attempt image:", attempt_path)

    agent.epsilon = old_epsilon
    return eval_stats


def save_report_log(
    output_dir: Path,
    label: str,
    maze_path: str,
    training_stats: Optional[List[dict]],
    eval_stats: List[dict],
    solver_mode: str,
) -> Path:
    metrics = MazeEnvironment.calculate_metrics(eval_stats)
    output_path = output_dir / output_name(maze_path, f"{label}_metrics_report.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Silent Cartographer Third Check-in Metrics\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Maze Label: {label}\n")
        f.write(f"Maze Path: {maze_path}\n")
        f.write(f"Workflow: {solver_mode}\n")
        f.write(METHOD_JUSTIFICATION + "\n\n")
        if training_stats is not None:
            f.write("Training was run on this maze.\n")
            for i, stats in enumerate(training_stats, start=1):
                f.write(
                    f"Train {i:03d}: goal={stats['goal_reached']}, turns={stats['turns_taken']}, "
                    f"deaths={stats['deaths']}, path_length={stats['path_length']}, "
                    f"explored={stats['cells_explored']}\n"
                )
        else:
            f.write("No training loop was run on this maze. This is direct transfer/evaluation.\n")

        f.write("\nEvaluation Episodes\n")
        f.write("-" * 22 + "\n")
        for i, stats in enumerate(eval_stats, start=1):
            f.write(
                f"Eval {i:02d}: goal={stats['goal_reached']}, turns={stats['turns_taken']}, "
                f"deaths={stats['deaths']}, path_length={stats['path_length']}, "
                f"explored={stats['cells_explored']}, "
                f"exploration_efficiency={stats.get('exploration_efficiency', 0):.4f}, "
                f"map_completeness={stats.get('map_completeness', 0):.4f}\n"
            )

        f.write("\nRequired Metrics\n")
        f.write("-" * 16 + "\n")
        f.write(f"Success Rate: {metrics['success_rate']:.2f}%\n")
        f.write(f"Average Path Length: {metrics['average_path_length']}\n")
        f.write(f"Average Turns to Solution: {metrics['average_turns_to_solution']}\n")
        f.write(f"Death Rate: {metrics['death_rate']:.4f}\n")

        f.write("\nBonus Metrics\n")
        f.write("-" * 13 + "\n")
        f.write(f"Exploration Efficiency: {metrics.get('exploration_efficiency', 0):.4f}\n")
        f.write(f"Map Completeness: {metrics.get('map_completeness', 0):.4f}\n")
        f.write(f"Average Confusions: {metrics.get('average_confusions', 0):.2f}\n")
        f.write(f"Average Arrow Pushes: {metrics.get('average_arrow_pushes', 0):.2f}\n")
    return output_path



def main() -> None:
    parser = argparse.ArgumentParser(description="Third Check-in Maze Solver")
    parser.add_argument("--train-maze", type=str, help="Maze-alpha image used for training.")
    parser.add_argument("--test-maze", type=str, default=None, help="Maze-beta image used for no-training evaluation.")
    parser.add_argument("--gamma-maze", type=str, default=None, help="Optional maze-gamma image for extra credit.")
    parser.add_argument("--maze", type=str, default=None, help="Backward-compatible single-maze mode.")
    parser.add_argument("--train-episodes", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--max-turns", type=int, default=10000)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-gif-frames", type=int, default=1200)
    parser.add_argument(
        "--gif-duration",
        type=int,
        default=100,
        help="Milliseconds per GIF frame. Increase this to slow down the GIF.",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--mode",
        choices=("auto", "hybrid", "honest"),
        default="auto",
        help=(
            "auto = train/practice in honest mode, then solve/evaluate in hybrid mode; "
            "hybrid = use hybrid for both; honest = use honest for both."
        ),
    )
    parser.add_argument(
        "--train-mode",
        choices=("honest", "hybrid"),
        default=None,
        help="Override only the training mode. Default in auto mode is honest.",
    )
    parser.add_argument(
        "--solve-mode",
        choices=("honest", "hybrid"),
        default=None,
        help="Override only the solving/evaluation mode. Default in auto mode is hybrid.",
    )
    args = parser.parse_args()

    train_maze = args.train_maze or args.maze
    if train_maze is None:
        raise SystemExit("Provide --train-maze maze_alpha/maze_1.png or use --maze for single-maze mode.")

    # Default requested workflow:
    #   1) Train/practice honestly from TurnResult only on maze-alpha.
    #   2) Solve/evaluate with the hybrid time-expanded planner.
    # Passing --mode hybrid or --mode honest still lets you force one mode for both phases.
    if args.mode == "auto":
        train_mode = args.train_mode or "honest"
        solve_mode = args.solve_mode or "hybrid"
    else:
        train_mode = args.train_mode or args.mode
        solve_mode = args.solve_mode or args.mode

    alpha_output = make_output_dir(args.output_dir, train_maze, "maze_alpha")
    print("Alpha output folder:", alpha_output)
    print("Training mode:", train_mode.upper())
    print("Solving/evaluation mode:", solve_mode.upper())

    agent, training_stats = run_training(
        maze_path=train_maze,
        episodes=args.train_episodes,
        max_turns=args.max_turns,
        frame_stride=args.frame_stride,
        max_gif_frames=args.max_gif_frames,
        gif_duration=args.gif_duration,
        output_dir=alpha_output,
        solver_mode=train_mode,
    )

    # Evaluate maze-alpha after honest training, using the chosen solve mode.
    alpha_eval_stats = run_evaluation(
        maze_path=train_maze,
        agent=agent,
        episodes=args.eval_episodes,
        max_turns=args.max_turns,
        frame_stride=args.frame_stride,
        max_gif_frames=args.max_gif_frames,
        gif_duration=args.gif_duration,
        output_dir=alpha_output,
        solver_mode=solve_mode,
        label="alpha",
        reset_spatial_memory=(solve_mode == "hybrid"),
    )
    alpha_metrics = MazeEnvironment.calculate_metrics(alpha_eval_stats)
    print_metrics(alpha_metrics, "MAZE-ALPHA")
    alpha_log = save_report_log(
        alpha_output,
        "alpha",
        train_maze,
        training_stats,
        alpha_eval_stats,
        f"train={train_mode}, solve={solve_mode}",
    )
    print("Saved alpha metrics report:", alpha_log)

    # Evaluate maze-beta with NO training loop on beta.
    if args.test_maze is not None:
        beta_output = make_output_dir(args.output_dir, args.test_maze, "maze_beta")
        beta_stats = run_evaluation(
            maze_path=args.test_maze,
            agent=agent,
            episodes=args.eval_episodes,
            max_turns=args.max_turns,
            frame_stride=args.frame_stride,
            max_gif_frames=args.max_gif_frames,
            gif_duration=args.gif_duration,
            output_dir=beta_output,
            solver_mode=solve_mode,
            label="beta",
            reset_spatial_memory=True,
        )
        beta_metrics = MazeEnvironment.calculate_metrics(beta_stats)
        print_metrics(beta_metrics, "MAZE-BETA")
        beta_log = save_report_log(
            beta_output,
            "beta",
            args.test_maze,
            None,
            beta_stats,
            f"train={train_mode}, solve={solve_mode}",
        )
        print("Saved beta metrics report:", beta_log)

    # Optional extra credit: gamma with arrows/teleports.
    if args.gamma_maze is not None:
        gamma_output = make_output_dir(args.output_dir, args.gamma_maze, "maze_gamma")
        gamma_stats = run_evaluation(
            maze_path=args.gamma_maze,
            agent=agent,
            episodes=args.eval_episodes,
            max_turns=args.max_turns,
            frame_stride=args.frame_stride,
            max_gif_frames=args.max_gif_frames,
            gif_duration=args.gif_duration,
            output_dir=gamma_output,
            solver_mode=solve_mode,
            label="gamma",
            reset_spatial_memory=True,
        )
        gamma_metrics = MazeEnvironment.calculate_metrics(gamma_stats)
        print_metrics(gamma_metrics, "MAZE-GAMMA")
        gamma_log = save_report_log(
            gamma_output,
            "gamma",
            args.gamma_maze,
            None,
            gamma_stats,
            f"train={train_mode}, solve={solve_mode}",
        )
        print("Saved gamma metrics report:", gamma_log)


if __name__ == "__main__":
    main()
