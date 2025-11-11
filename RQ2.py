# RQ2.py - Experiment results analysis
# Load curves from logs, compute metrics and export CSV
from pathlib import Path
import json
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# === Configuration ===
ROOT = Path(__file__).parent / "logdir" / "dreamer"
SCORES_FILE = "scores.jsonl"  # Read from scores.jsonl
LAST_N = 20  # Final reward: mean of last N points
CONFIGS = ["base_training", "shift_only", "shift_rotate_jitter_noise"]  # Experiment configs
OUT_CSV = Path(__file__).parent / "rq2_metrics.csv"
CURVE_DIR = Path(__file__).parent / "rq2_curves"  # Export curve data
PLOT_DIR = Path(__file__).parent / "rq2_plots"  # Save plots
CURVE_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)


def load_series_from_jsonl(run_dir: Path, seed: str = None) -> tuple[np.ndarray, np.ndarray]:
    """Load data from scores.jsonl. Returns (steps, scores) arrays."""
    scores_path = run_dir / SCORES_FILE
    if not scores_path.is_file():
        raise FileNotFoundError(f"Missing file: {scores_path}")
    
    steps = []
    scores = []
    with open(scores_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if 'step' in data and 'episode/score' in data:
                    step = data['step']
                    score = data['episode/score']
                    if np.isfinite(score):  # Filter NaN/Inf
                        steps.append(step)
                        scores.append(score)
            except json.JSONDecodeError:
                continue
    
    if len(steps) == 0:
        raise ValueError(f"No valid data found: {scores_path}")
    
    return np.array(steps, dtype=np.float64), np.array(scores, dtype=np.float64)


def find_all_runs(root: Path, configs: list) -> list[tuple[str, str, Path]]:
    """Find all runs (config×seed). Returns [(config, seed, run_dir), ...]"""
    runs = []
    for config in configs:
        config_dir = root / config
        if not config_dir.is_dir():
            print(f"Warning: Config dir not found: {config_dir}")
            continue
        
        # Check for subdirs (multiple seeds)
        subdirs = [d for d in config_dir.iterdir() if d.is_dir() and (d / SCORES_FILE).is_file()]
        
        if len(subdirs) > 0:
            # Subdirs exist, each is a seed
            for subdir in subdirs:
                seed = subdir.name
                runs.append((config, seed, subdir))
        else:
            # No subdirs, check current dir
            if (config_dir / SCORES_FILE).is_file():
                runs.append((config, "seed0", config_dir))
            else:
                print(f"Warning: {SCORES_FILE} not found in {config}")
    
    return runs


def final_reward(scores: np.ndarray, last_n: int = LAST_N) -> float:
    """Compute mean reward of last N episodes"""
    if scores.size == 0:
        return float("nan")
    n = min(last_n, scores.size)
    return float(scores[-n:].mean())


def auc_normalized(steps: np.ndarray, scores: np.ndarray, common_step_range: tuple = None) -> float:
    """
    Compute AUC of reward-steps curve, normalized to common step budget.
    Only compute AUC in common range to avoid bias.
    
    Args:
        steps: Step array
        scores: Reward array
        common_step_range: (min_step, max_step) for normalization
    
    Returns:
        Normalized AUC (mean reward = AUC / step_range)
    """
    if scores.size <= 1:
        return float("nan")
    
    if common_step_range is not None:
        # Compute AUC only in common range
        min_step, max_step = common_step_range
        
        # Extract data in common range
        mask = (steps >= min_step) & (steps <= max_step)
        if np.sum(mask) < 2:
            return float("nan")
        
        steps_common = steps[mask]
        scores_common = scores[mask]
        
        # Interpolate at boundaries if needed
        if steps_common[0] > min_step:
            score_at_min = np.interp(min_step, steps, scores)
            steps_common = np.concatenate([[min_step], steps_common])
            scores_common = np.concatenate([[score_at_min], scores_common])
        
        if steps_common[-1] < max_step:
            score_at_max = np.interp(max_step, steps, scores)
            steps_common = np.concatenate([steps_common, [max_step]])
            scores_common = np.concatenate([scores_common, [score_at_max]])
        
        # Compute AUC in common range
        area = float(np.trapezoid(scores_common, steps_common))
        step_range = max_step - min_step
    else:
        # No common range, use full curve
        area = float(np.trapezoid(scores, steps))
        step_range = steps[-1] - steps[0]
    
    if step_range > 0:
        normalized_auc = area / step_range
    else:
        normalized_auc = float("nan")
    
    return normalized_auc


def steps_to_target(steps: np.ndarray, scores: np.ndarray, target: float) -> float:
    """Compute steps to reach target (sample efficiency). Returns NaN if never reached."""
    if scores.size == 0 or not np.isfinite(target):
        return float("nan")
    
    # Find first point reaching target
    mask = scores >= target
    if not np.any(mask):
        return float("nan")  # Never reached
    
    first_idx = np.argmax(mask)
    return float(steps[first_idx])


def write_curve_csv(config: str, seed: str, steps: np.ndarray, scores: np.ndarray):
    """Export curve data to CSV"""
    path = CURVE_DIR / f"{config}_{seed}_episode_score_curve.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "episode_score"])
        for s, sc in zip(steps, scores):
            w.writerow([float(s), float(sc)])


def plot_curves(all_data: dict, output_dir: Path):
    """Plot learning curves comparison. all_data: {(config, seed): (steps, scores), ...}"""
    # Group by config
    configs = sorted(set(config for config, _ in all_data.keys()))
    colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))
    config_color_map = {config: colors[i] for i, config in enumerate(configs)}
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: all curves
    ax1 = axes[0]
    for (config, seed), (steps, scores) in all_data.items():
        color = config_color_map[config]
        alpha = 0.7 if len([s for c, s in all_data.keys() if c == config]) > 1 else 1.0
        label = f"{config}" if seed == "seed0" else f"{config}_{seed}"
        ax1.plot(steps, scores, label=label, color=color, alpha=alpha, linewidth=1.5)
    
    ax1.set_xlabel("Steps", fontsize=12)
    ax1.set_ylabel("Episode Score", fontsize=12)
    ax1.set_title("Learning Curves (All Runs)", fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Right: mean by config (if multiple seeds)
    ax2 = axes[1]
    for config in configs:
        config_runs = [(c, s) for c, s in all_data.keys() if c == config]
        if len(config_runs) == 0:
            continue
        
        # Collect all steps and scores
        all_steps_list = []
        all_scores_list = []
        for c, s in config_runs:
            steps, scores = all_data[(c, s)]
            all_steps_list.append(steps)
            all_scores_list.append(scores)
        
        # Find common step range
        min_step = max(s[0] for s in all_steps_list)
        max_step = min(s[-1] for s in all_steps_list)
        
        # Interpolate to common step grid
        if max_step > min_step:
            common_steps = np.linspace(min_step, max_step, 200)
            interpolated_scores = []
            for steps, scores in zip(all_steps_list, all_scores_list):
                interp_scores = np.interp(common_steps, steps, scores)
                interpolated_scores.append(interp_scores)
            
            mean_scores = np.mean(interpolated_scores, axis=0)
            std_scores = np.std(interpolated_scores, axis=0)
            
            color = config_color_map[config]
            ax2.plot(common_steps, mean_scores, label=config, color=color, linewidth=2)
            ax2.fill_between(common_steps, mean_scores - std_scores, mean_scores + std_scores,
                           color=color, alpha=0.2)
        else:
            # Single run, plot directly
            steps, scores = all_data[config_runs[0]]
            color = config_color_map[config]
            ax2.plot(steps, scores, label=config, color=color, linewidth=2)
    
    ax2.set_xlabel("Steps", fontsize=12)
    ax2.set_ylabel("Episode Score", fontsize=12)
    ax2.set_title("Learning Curves (Mean ± Std)", fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "learning_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")


def plot_comprehensive_comparison(rows: list, all_data: dict, output_dir: Path):
    """Plot comparison: curves + bar charts + Δ% table"""
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    
    # Prepare data: group by config
    configs = sorted(set(r["config"] for r in rows))
    baseline_config = "base_training"
    
    # Compute mean per config (if multiple seeds)
    config_metrics = {}
    for config in configs:
        config_rows = [r for r in rows if r["config"] == config]
        if len(config_rows) == 0:
            continue
        
        config_metrics[config] = {
            "final_reward": np.nanmean([r["final_reward"] for r in config_rows]),
            "auc_normalized": np.nanmean([r["auc_normalized"] for r in config_rows]),
            "steps_to_target": np.nanmean([r["steps_to_target"] for r in config_rows if np.isfinite(r["steps_to_target"])]),
            "final_reward_std": np.nanstd([r["final_reward"] for r in config_rows]) if len(config_rows) > 1 else 0,
            "auc_std": np.nanstd([r["auc_normalized"] for r in config_rows]) if len(config_rows) > 1 else 0,
            "steps_std": np.nanstd([r["steps_to_target"] for r in config_rows if np.isfinite(r["steps_to_target"])]) if len(config_rows) > 1 else 0,
        }
    
    # Compute relative change vs baseline
    if baseline_config in config_metrics:
        baseline_metrics = config_metrics[baseline_config]
        delta_percent = {}
        for config in configs:
            if config == baseline_config:
                delta_percent[config] = {"final_reward": 0, "auc_normalized": 0, "steps_to_target": 0}
            else:
                metrics = config_metrics[config]
                delta_percent[config] = {
                    "final_reward": ((metrics["final_reward"] - baseline_metrics["final_reward"]) / baseline_metrics["final_reward"]) * 100,
                    "auc_normalized": ((metrics["auc_normalized"] - baseline_metrics["auc_normalized"]) / baseline_metrics["auc_normalized"]) * 100,
                    "steps_to_target": ((metrics["steps_to_target"] - baseline_metrics["steps_to_target"]) / baseline_metrics["steps_to_target"]) * 100 if np.isfinite(metrics["steps_to_target"]) and np.isfinite(baseline_metrics["steps_to_target"]) else float("nan"),
                }
    else:
        delta_percent = {config: {"final_reward": 0, "auc_normalized": 0, "steps_to_target": 0} for config in configs}
    
    # Create figure
    fig = plt.figure(figsize=(18, 11))
    gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.4, 
                  height_ratios=[2, 1.5, 1.0], width_ratios=[1, 1, 1],
                  left=0.06, right=0.98, top=0.95, bottom=0.08)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))
    config_color_map = {config: colors[i] for i, config in enumerate(configs)}
    
    # 1. Learning curves (top-left, 2 cols)
    ax_curve = fig.add_subplot(gs[0, :2])
    
    # Plot learning curves
    for config in configs:
        config_runs = [(c, s) for c, s in all_data.keys() if c == config]
        if len(config_runs) == 0:
            continue
        
        # Collect all steps and scores
        all_steps_list = []
        all_scores_list = []
        for c, s in config_runs:
            steps, scores = all_data[(c, s)]
            all_steps_list.append(steps)
            all_scores_list.append(scores)
        
        # Find common step range
        min_step = max(s[0] for s in all_steps_list)
        max_step = min(s[-1] for s in all_steps_list)
        
        # Interpolate to common step grid
        if max_step > min_step:
            common_steps = np.linspace(min_step, max_step, 200)
            interpolated_scores = []
            for steps, scores in zip(all_steps_list, all_scores_list):
                interp_scores = np.interp(common_steps, steps, scores)
                interpolated_scores.append(interp_scores)
            
            mean_scores = np.mean(interpolated_scores, axis=0)
            std_scores = np.std(interpolated_scores, axis=0)
            
            color = config_color_map[config]
            ax_curve.plot(common_steps, mean_scores, label=config.replace('_', ' '), 
                         color=color, linewidth=2.5)
            if len(config_runs) > 1:
                ax_curve.fill_between(common_steps, mean_scores - std_scores, 
                                     mean_scores + std_scores, color=color, alpha=0.2)
        else:
            # Single run, plot directly
            steps, scores = all_data[config_runs[0]]
            color = config_color_map[config]
            ax_curve.plot(steps, scores, label=config.replace('_', ' '), 
                         color=color, linewidth=2.5)
    
    ax_curve.set_xlabel("Steps", fontsize=11)
    ax_curve.set_ylabel("Episode Score", fontsize=11)
    ax_curve.set_title("Learning Curves", fontsize=13, fontweight='bold')
    ax_curve.legend(fontsize=9, loc='best')
    ax_curve.grid(True, alpha=0.3)
    
    # 2. Bar chart: Final Reward (top-right)
    ax_fr = fig.add_subplot(gs[0, 2])
    config_names = [c.replace('_', '\n') for c in configs]
    fr_values = [config_metrics[c]["final_reward"] for c in configs]
    fr_stds = [config_metrics[c]["final_reward_std"] for c in configs]
    bars_fr = ax_fr.bar(config_names, fr_values, yerr=fr_stds, 
                        color=[config_color_map[c] for c in configs], 
                        capsize=5, alpha=0.7, edgecolor='black', linewidth=1.5, width=0.6)
    ax_fr.set_ylabel("Final Reward", fontsize=11, fontweight='bold')
    ax_fr.set_title("Final Reward\n(Last 20 Episodes)", fontsize=11, fontweight='bold', pad=10)
    ax_fr.grid(True, alpha=0.3, axis='y')
    ax_fr.tick_params(axis='x', labelsize=8, rotation=0)
    
    # Add value labels
    for i, (val, std) in enumerate(zip(fr_values, fr_stds)):
        ax_fr.text(i, val + std + (max(fr_values) * 0.03), f'{val:.1f}', 
                  ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 3. Bar chart: AUC Normalized (middle-left)
    ax_auc = fig.add_subplot(gs[1, 0])
    auc_values = [config_metrics[c]["auc_normalized"] for c in configs]
    auc_stds = [config_metrics[c]["auc_std"] for c in configs]
    bars_auc = ax_auc.bar(config_names, auc_values, yerr=auc_stds,
                          color=[config_color_map[c] for c in configs],
                          capsize=5, alpha=0.7, edgecolor='black', linewidth=1.5, width=0.6)
    ax_auc.set_ylabel("AUC (Normalized)", fontsize=11, fontweight='bold')
    ax_auc.set_title("AUC Normalized\n(Reward-Steps Area)", fontsize=11, fontweight='bold', pad=10)
    ax_auc.grid(True, alpha=0.3, axis='y')
    ax_auc.tick_params(axis='x', labelsize=8, rotation=0)
    
    for i, (val, std) in enumerate(zip(auc_values, auc_stds)):
        ax_auc.text(i, val + std + (max(auc_values) * 0.03), f'{val:.1f}',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 4. Bar chart: Steps to Target (middle-center)
    ax_st = fig.add_subplot(gs[1, 1])
    st_values = [config_metrics[c]["steps_to_target"] if np.isfinite(config_metrics[c]["steps_to_target"]) else 0 for c in configs]
    st_stds = [config_metrics[c]["steps_std"] if np.isfinite(config_metrics[c]["steps_to_target"]) else 0 for c in configs]
    bars_st = ax_st.bar(config_names, st_values, yerr=st_stds,
                        color=[config_color_map[c] for c in configs],
                        capsize=5, alpha=0.7, edgecolor='black', linewidth=1.5, width=0.6)
    ax_st.set_ylabel("Steps to Target", fontsize=11, fontweight='bold')
    ax_st.set_title("Steps to Target\n(Sample Efficiency)", fontsize=11, fontweight='bold', pad=10)
    ax_st.grid(True, alpha=0.3, axis='y')
    ax_st.tick_params(axis='x', labelsize=8, rotation=0)
    
    for i, (val, std) in enumerate(zip(st_values, st_stds)):
        if val > 0:
            ax_st.text(i, val + std + (max(st_values) * 0.03), f'{val/1000:.0f}K',
                      ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 5. Δ% table (middle-right)
    ax_table = fig.add_subplot(gs[1, 2])
    ax_table.axis('off')
    
    # Prepare table data
    table_data = []
    table_data.append(['Config', 'Final Reward\nΔ%', 'AUC\nΔ%', 'Steps to Target\nΔ%'])
    
    for config in configs:
        delta = delta_percent[config]
        row = [
            config.replace('_', ' '),
            f"{delta['final_reward']:+.2f}%",
            f"{delta['auc_normalized']:+.2f}%",
            f"{delta['steps_to_target']:+.2f}%" if np.isfinite(delta['steps_to_target']) else "N/A"
        ]
        table_data.append(row)
    
    # Create table
    table = ax_table.table(cellText=table_data[1:], colLabels=table_data[0],
                           cellLoc='center', loc='center',
                           colWidths=[0.32, 0.22, 0.22, 0.24])
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 2.2)
    
    # Set header style
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Set row colors by change direction
    for i, config in enumerate(configs):
        row_idx = i + 1
        delta = delta_percent[config]
        
        # Color by change direction
        for col_idx in [1, 2]:  # Final Reward and AUC (higher is better)
            if col_idx == 1:
                val = delta['final_reward']
            else:
                val = delta['auc_normalized']
            
            if val > 0:
                table[(row_idx, col_idx)].set_facecolor('#C5E0B4')  # Green: better
            elif val < 0:
                table[(row_idx, col_idx)].set_facecolor('#FFC7CE')  # Red: worse
            else:
                table[(row_idx, col_idx)].set_facecolor('#F2F2F2')  # Gray: baseline
        
        # Steps to Target (lower is better)
        if np.isfinite(delta['steps_to_target']):
            val = delta['steps_to_target']
            if val < 0:
                table[(row_idx, 3)].set_facecolor('#C5E0B4')  # Green: better (fewer steps)
            elif val > 0:
                table[(row_idx, 3)].set_facecolor('#FFC7CE')  # Red: worse (more steps)
            else:
                table[(row_idx, 3)].set_facecolor('#F2F2F2')
        else:
            table[(row_idx, 3)].set_facecolor('#F2F2F2')
    
    ax_table.set_title("Relative Performance\n(Δ% vs Baseline)", fontsize=11, fontweight='bold', pad=15)
    
    # 6. Detailed metrics table (bottom, 3 cols)
    ax_detail = fig.add_subplot(gs[2, :])
    ax_detail.axis('off')
    
    detail_data = []
    detail_data.append(['Config', 'Final Reward', 'AUC (Norm)', 'Steps to Target', 
                       'Final Reward Δ%', 'AUC Δ%', 'Steps Δ%'])
    
    for config in configs:
        metrics = config_metrics[config]
        delta = delta_percent[config]
        row = [
            config.replace('_', ' '),
            f"{metrics['final_reward']:.2f} ± {metrics['final_reward_std']:.2f}" if metrics['final_reward_std'] > 0 else f"{metrics['final_reward']:.2f}",
            f"{metrics['auc_normalized']:.2f} ± {metrics['auc_std']:.2f}" if metrics['auc_std'] > 0 else f"{metrics['auc_normalized']:.2f}",
            f"{metrics['steps_to_target']:.0f} ± {metrics['steps_std']:.0f}" if np.isfinite(metrics['steps_to_target']) and metrics['steps_std'] > 0 else (f"{metrics['steps_to_target']:.0f}" if np.isfinite(metrics['steps_to_target']) else "N/A"),
            f"{delta['final_reward']:+.2f}%",
            f"{delta['auc_normalized']:+.2f}%",
            f"{delta['steps_to_target']:+.2f}%" if np.isfinite(delta['steps_to_target']) else "N/A"
        ]
        detail_data.append(row)
    
    detail_table = ax_detail.table(cellText=detail_data[1:], colLabels=detail_data[0],
                                   cellLoc='center', loc='center')
    detail_table.auto_set_font_size(False)
    detail_table.set_fontsize(7.5)
    detail_table.scale(1, 2.0)
    
    # Adjust column widths to avoid overlap
    col_widths = [0.15, 0.14, 0.12, 0.14, 0.13, 0.12, 0.12]
    for i, width in enumerate(col_widths):
        for j in range(len(detail_data)):
            detail_table[(j, i)].set_width(width)
    
    # Set header style
    for i in range(len(detail_data[0])):
        detail_table[(0, i)].set_facecolor('#4472C4')
        detail_table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax_detail.set_title("Detailed Metrics Summary", fontsize=11, fontweight='bold', pad=8)
    
    plt.suptitle("RQ2: Results Comparison", fontsize=15, fontweight='bold', y=0.985)
    
    output_path = output_dir / "results_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot: {output_path}")


def main():
    print("=" * 60)
    print("RQ2 Experiment Results Analysis")
    print("=" * 60)
    
    # 1) Find all runs
    runs = find_all_runs(ROOT, CONFIGS)
    print(f"\nFound {len(runs)} runs:")
    for config, seed, _ in runs:
        print(f"  - {config} (seed: {seed})")
    
    if len(runs) == 0:
        print("Error: No run data found!")
        return
    
    # 2) Load all curve data
    print("\nLoading data...")
    all_data = {}
    for config, seed, run_dir in runs:
        try:
            steps, scores = load_series_from_jsonl(run_dir, seed)
            all_data[(config, seed)] = (steps, scores)
            write_curve_csv(config, seed, steps, scores)
            print(f"  ✓ {config}_{seed}: {len(scores)} points, steps=[{steps[0]:.0f}, {steps[-1]:.0f}]")
        except Exception as e:
            print(f"  ✗ {config}_{seed}: Error - {e}")
    
    if len(all_data) == 0:
        print("Error: Failed to load any data!")
        return
    
    # 3) Compute baseline final reward (for Steps-to-Target)
    baseline_config = "base_training"
    baseline_runs = [(c, s) for c, s in all_data.keys() if c == baseline_config]
    
    if len(baseline_runs) > 0:
        # If multiple baseline runs, take mean
        baseline_finals = []
        for c, s in baseline_runs:
            steps, scores = all_data[(c, s)]
            fr = final_reward(scores, LAST_N)
            if np.isfinite(fr):
                baseline_finals.append(fr)
        
        if len(baseline_finals) > 0:
            baseline_final = np.mean(baseline_finals)
            target = 0.9 * baseline_final
            print(f"\nBaseline final reward: {baseline_final:.2f}")
            print(f"Target threshold (90%): {target:.2f}")
        else:
            target = float("nan")
            print("\nWarning: Cannot compute baseline final reward")
    else:
        target = float("nan")
        print("\nWarning: Baseline config (base_training) not found")
    
    # 4) Compute all metrics
    print("\nComputing metrics...")
    
    # Find common step range (for AUC normalization)
    all_min_steps = [steps[0] for steps, _ in all_data.values()]
    all_max_steps = [steps[-1] for steps, _ in all_data.values()]
    common_min_step = max(all_min_steps)  # Max start step
    common_max_step = min(all_max_steps)  # Min end step
    common_step_range = (common_min_step, common_max_step)
    
    print(f"Common step range: [{common_min_step:.0f}, {common_max_step:.0f}]")
    
    # Compute all metrics first (without relative baseline)
    rows = []
    for (config, seed), (steps, scores) in all_data.items():
        fr = final_reward(scores, LAST_N)
        auc = auc_normalized(steps, scores, common_step_range)
        st = steps_to_target(steps, scores, target)
        
        row = {
            "config": config,
            "seed": seed,
            "final_reward": fr,
            "auc_normalized": auc,
            "steps_to_target": st,
            "target_threshold": target,
            "max_step": float(steps[-1]),
            "n_episodes": int(len(scores)),
        }
        rows.append(row)
    
    # Compute relative change vs baseline (after all data collected)
    baseline_config = "base_training"
    baseline_rows = [r for r in rows if r["config"] == baseline_config]
    
    if len(baseline_rows) > 0:
        baseline_fr = np.nanmean([r["final_reward"] for r in baseline_rows])
        baseline_auc = np.nanmean([r["auc_normalized"] for r in baseline_rows])
        baseline_st = np.nanmean([r["steps_to_target"] for r in baseline_rows if np.isfinite(r["steps_to_target"])])
        
        # Add relative baseline columns to all rows
        for row in rows:
            if row["config"] == baseline_config:
                row["final_reward_delta_pct"] = 0.0
                row["auc_delta_pct"] = 0.0
                row["steps_delta_pct"] = 0.0
            else:
                fr = row["final_reward"]
                auc = row["auc_normalized"]
                st = row["steps_to_target"]
                
                row["final_reward_delta_pct"] = ((fr - baseline_fr) / baseline_fr) * 100 if np.isfinite(baseline_fr) and baseline_fr != 0 else float("nan")
                row["auc_delta_pct"] = ((auc - baseline_auc) / baseline_auc) * 100 if np.isfinite(baseline_auc) and baseline_auc != 0 else float("nan")
                row["steps_delta_pct"] = ((st - baseline_st) / baseline_st) * 100 if np.isfinite(st) and np.isfinite(baseline_st) and baseline_st != 0 else float("nan")
    else:
        # No baseline, set all relative values to NaN
        for row in rows:
            row["final_reward_delta_pct"] = float("nan")
            row["auc_delta_pct"] = float("nan")
            row["steps_delta_pct"] = float("nan")
    
    # Print metrics for each run
    for row in rows:
        config = row["config"]
        seed = row["seed"]
        fr = row["final_reward"]
        auc = row["auc_normalized"]
        st = row["steps_to_target"]
        print(f"  {config}_{seed}:")
        print(f"    Final Reward: {fr:.2f}")
        print(f"    AUC (normalized): {auc:.2f}")
        print(f"    Steps to Target: {st:.0f}" if np.isfinite(st) else f"    Steps to Target: N/A")
    
    # 5) Export CSV
    print(f"\nExporting CSV: {OUT_CSV}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, float_format='%.6f')
    print(f"  ✓ Saved {len(rows)} rows")
    
    # 6) Plot curves
    print(f"\nPlotting learning curves...")
    plot_curves(all_data, PLOT_DIR)
    
    # 7) Plot comparison
    print(f"Plotting comparison...")
    plot_comprehensive_comparison(rows, all_data, PLOT_DIR)
    
    # 8) Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    for config in CONFIGS:
        config_rows = [r for r in rows if r["config"] == config]
        if len(config_rows) == 0:
            continue
        
        print(f"\n{config}:")
        if len(config_rows) > 1:
            # Multiple seeds, show mean ± std
            fr_mean = np.nanmean([r["final_reward"] for r in config_rows])
            fr_std = np.nanstd([r["final_reward"] for r in config_rows])
            auc_mean = np.nanmean([r["auc_normalized"] for r in config_rows])
            auc_std = np.nanstd([r["auc_normalized"] for r in config_rows])
            st_mean = np.nanmean([r["steps_to_target"] for r in config_rows if np.isfinite(r["steps_to_target"])])
            st_std = np.nanstd([r["steps_to_target"] for r in config_rows if np.isfinite(r["steps_to_target"])])
            
            print(f"  Final Reward: {fr_mean:.2f} ± {fr_std:.2f} (n={len(config_rows)})")
            print(f"  AUC (normalized): {auc_mean:.2f} ± {auc_std:.2f}")
            if np.isfinite(st_mean):
                print(f"  Steps to Target: {st_mean:.0f} ± {st_std:.0f}")
            else:
                print(f"  Steps to Target: N/A")
        else:
            # Single seed
            r = config_rows[0]
            print(f"  Final Reward: {r['final_reward']:.2f}")
            print(f"  AUC (normalized): {r['auc_normalized']:.2f}")
            if np.isfinite(r['steps_to_target']):
                print(f"  Steps to Target: {r['steps_to_target']:.0f}")
            else:
                print(f"  Steps to Target: N/A")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"  - CSV file: {OUT_CSV}")
    print(f"  - Curve data: {CURVE_DIR}")
    print(f"  - Plots: {PLOT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
