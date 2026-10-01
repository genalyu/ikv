"""Full-resolution numerical audit and bounded-size, offline visualizations."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from .data import episodes, causal_indices, aligned_timeline, quality_exclusion_reason
from .config import paths, fingerprint


def analyze(task):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    out = paths(task)["reports"]
    out.mkdir(parents=True, exist_ok=True)
    rows, tracks, all_xyz, details = [], [], [], []
    fps = float(task["robot"].get("fps", 30))
    for ep in episodes(task):
        t = ep.timestamps
        if not np.isfinite(t).all() or (np.diff(t) < 0).any():
            raise ValueError(f"{ep.source_id}: invalid action timestamps")
        row = dict(
            episode=ep.source_id,
            frames=len(t),
            duration_s=float(t[-1] - t[0] + 1 / fps),
            max_timestamp_gap_s=float(np.diff(t).max()),
            duplicate_timestamps=int((np.diff(t) == 0).sum()),
            estimated_missing_ticks=int(
                np.maximum(np.rint(np.diff(t) * fps) - 1, 0).sum()
            ),
            original_task=ep.metadata.get(
                "task_key", ep.metadata.get("memory_task", "")
            ),
            state_alignment_max_s=ep.metadata.get("state_alignment_max_s", 0),
        )
        labels = ep.metadata.get("quality", {}).get("labels", [])
        quality_reason = quality_exclusion_reason(ep, task)
        row["quality_labels"] = "|".join(labels)
        row["training_quality_valid"] = quality_reason is None
        row["training_exclusion_reason"] = quality_reason or ""
        alignment = {}
        common_start = max([t[0]] + [ct[0] for ct in ep.camera_times.values()])
        row["common_start_offset_s"] = float(common_start - t[0])
        try:
            aligned_timeline(ep, task["robot"])
            row["training_alignment_valid"] = True
        except ValueError as error:
            row["training_alignment_valid"] = False
            reason = str(error)
            row["training_exclusion_reason"] = (
                f"{row['training_exclusion_reason']}; {reason}"
                if row["training_exclusion_reason"] else reason
            )
        row["training_usable"] = (
            row["training_quality_valid"] and row["training_alignment_valid"]
        )
        for cam, cts in ep.camera_times.items():
            try:
                ids, age = causal_indices(
                    cts,
                    t[t >= common_start],
                    task["robot"].get("max_alignment_age_s", 0.1),
                )
                alignment[cam] = dict(
                    max_age_s=float(age.max()),
                    mean_age_s=float(age.mean()),
                    repeated_mapping_frames=int((np.diff(ids) == 0).sum()),
                    unused_source_frames=int(len(cts) - len(np.unique(ids))),
                )
            except ValueError as e:
                alignment[cam] = {"error": str(e)}
        for arm, off in enumerate(range(0, ep.state.shape[1], 10)):
            xyz = ep.state[:, off : off + 3]
            dt = np.diff(t)
            speed = np.linalg.norm(np.diff(xyz, axis=0), axis=1)[dt > 0] / dt[dt > 0]
            delta = np.linalg.norm(ep.action[:, off : off + 3] - xyz, axis=1)
            prefix = f"arm{arm}_"
            row.update(
                {
                    prefix + "path_length_m": float(
                        np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()
                    ),
                    prefix + "speed_mean_m_s": float(speed.mean()),
                    prefix + "speed_p95_m_s": float(np.quantile(speed, 0.95)),
                    prefix + "speed_max_m_s": float(speed.max()),
                    prefix + "idle_fraction": float((speed < 0.005).mean()),
                    prefix + "tracking_mean_m": float(delta.mean())
                    if np.isfinite(delta).all()
                    else None,
                    prefix + "tracking_p95_m": float(np.quantile(delta, 0.95))
                    if np.isfinite(delta).all()
                    else None,
                }
            )
            stride = max(1, int(np.ceil(len(t) / 600)))
            take = np.unique(np.r_[np.arange(0, len(t), stride), len(t) - 1])
            tracks.append(
                dict(
                    episode=ep.source_id,
                    arm=arm,
                    t=t[take],
                    xyz=xyz[take],
                    grip=ep.state[take, off + 9],
                    action_grip=ep.action[take, off + 9],
                    error=delta[take],
                    stride=stride,
                )
            )
            all_xyz.append((arm, xyz))
            for axis, name in enumerate("xyz"):
                row[prefix + name + "_min_m"] = float(xyz[:, axis].min())
                row[prefix + name + "_max_m"] = float(xyz[:, axis].max())
                row[prefix + name + "_start_m"] = float(xyz[0, axis])
                row[prefix + name + "_end_m"] = float(xyz[-1, axis])
        rows.append(row)
        details.append(
            dict(episode=ep.source_id, source_metadata=ep.metadata, alignment=alignment)
        )
    if not rows:
        raise ValueError("Empty dataset")
    df = pd.DataFrame(rows)
    df.to_csv(out / "episodes.csv", index=False)
    durations = df.duration_s.to_numpy()
    summary = dict(
        episodes=len(rows),
        frames=int(df.frames.sum()),
        duration_mean_s=float(durations.mean()),
        duration_median_s=float(np.median(durations)),
        duration_min_s=float(durations.min()),
        duration_max_s=float(durations.max()),
        duration_p05_s=float(np.quantile(durations, 0.05)),
        duration_p95_s=float(np.quantile(durations, 0.95)),
        total_minutes=float(durations.sum() / 60),
        prompt=task["prompt"],
        source=task["source"],
        task_fingerprint=fingerprint(task),
        robot=task["robot"]["name"],
        coordinate_frame=task["robot"]["coordinate_frame"],
        position_unit="m",
        idle_threshold_m_s=0.005,
        stats_use_all_points=True,
        plotted_max_points_per_trajectory=601,
        original_task_labels=sorted(set(df.original_task)),
        duplicate_timestamps=int(df.duplicate_timestamps.sum()),
        alignment_valid_episodes=int(df.training_alignment_valid.sum()),
        alignment_excluded_episodes=int((~df.training_alignment_valid).sum()),
        quality_excluded_episodes=int((~df.training_quality_valid).sum()),
        training_usable_episodes=int(df.training_usable.sum()),
        training_excluded_episodes=int((~df.training_usable).sum()),
        action_label_policy=task.get("action_labels", "recorded_command"),
        scope=task.get("dataset_scope", "all_local_episodes"),
        note="Integrity labels are not task success measurements. Oracle labels are analysis-only.",
    )
    for name, value in (("summary", summary), ("audit", details)):
        (out / (name + ".json")).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, default=str)
        )
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    axs[0, 0].hist(durations, bins=20)
    axs[0, 0].set(xlabel="Duration (s)", ylabel="Episodes")
    xyz = np.concatenate([x for arm, x in all_xyz if arm == 0])
    for ax, (i, j) in zip([axs[0, 1], axs[0, 2], axs[1, 0]], [(0, 1), (0, 2), (1, 2)]):
        ax.hist2d(xyz[:, i], xyz[:, j], bins=60)
        ax.set(xlabel="XYZ"[i] + " (m)", ylabel="XYZ"[j] + " (m)", aspect="equal")
    axs[1, 1].hist(df.arm0_path_length_m, bins=20)
    axs[1, 1].set_xlabel("Path length (m)")
    axs[1, 2].hist(pd.to_numeric(df.arm0_tracking_p95_m).dropna() * 1000, bins=20)
    axs[1, 2].set_xlabel("Target–state distance p95 (mm)")
    fig.suptitle(
        f"{task['name']}: {len(rows)} episodes (arm 0); full-resolution statistics"
    )
    fig.tight_layout()
    fig.savefig(out / "overview.png", dpi=160)
    plt.close(fig)
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(projection="3d")
    for tr in tracks:
        if tr["arm"] == 0:
            x = tr["xyz"]
            ax.plot(*x.T, alpha=0.22, lw=0.7)
            ax.scatter(*x[0], c="green", s=5)
            ax.scatter(*x[-1], c="red", s=5)
    ax.set(
        xlabel="X (m)",
        ylabel="Y (m)",
        zlabel="Z (m)",
        title="Trajectories: green=start, red=end",
    )
    fig.savefig(out / "trajectories.png", dpi=160)
    plt.close(fig)
    view = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=[
            "Trajectory (m)",
            "Duration (s)",
            "Gripper open ratio",
            "Target–state distance (m)",
        ],
    )
    for idx, tr in enumerate(tracks):
        x = tr["xyz"]
        name = f"{tr['episode']} arm{tr['arm']}"
        visible = True if idx < 12 else "legendonly"
        view.add_trace(
            go.Scatter3d(
                x=x[:, 0],
                y=x[:, 1],
                z=x[:, 2],
                mode="lines",
                name=name,
                legendgroup=name,
                visible=visible,
            ),
            row=1,
            col=1,
        )
        view.add_trace(
            go.Scatter(
                x=tr["t"],
                y=tr["grip"],
                name=name + " measured",
                legendgroup=name,
                visible=visible,
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        view.add_trace(
            go.Scatter(
                x=tr["t"],
                y=tr["action_grip"],
                name=name + " commanded",
                legendgroup=name,
                visible=visible,
                showlegend=False,
                line=dict(dash="dot"),
            ),
            row=2,
            col=1,
        )
        view.add_trace(
            go.Scatter(
                x=tr["t"],
                y=tr["error"],
                name=name,
                legendgroup=name,
                visible=visible,
                showlegend=False,
            ),
            row=2,
            col=2,
        )
    view.add_trace(go.Histogram(x=durations, name="Duration"), row=1, col=2)
    view.update_layout(
        height=1000,
        title=f"{task['name']} — {len(rows)} episodes; {summary['scope']}",
        legend=dict(groupclick="togglegroup"),
        scene=dict(aspectmode="data"),
    )
    intro = (
        "<h1>Dataset audit</h1><pre>"
        + __import__("html").escape(json.dumps(summary, ensure_ascii=False, indent=2))
        + "</pre>"
    )
    invalid = df.loc[~df.training_usable]
    if not invalid.empty:
        intro += "<h2>Quality and alignment exclusions for training</h2>" + invalid[
            ["episode", "training_exclusion_reason"]
        ].to_html(index=False, escape=True)
    html = view.to_html(include_plotlyjs=True, full_html=True)
    html = html.replace(
        "<body>",
        "<body>"
        + intro
        + '<p>Numerical statistics use all points. Lines are decimated; click legend to select episodes.</p><img src="overview.png" width="100%"><img src="trajectories.png" width="70%">',
    )
    (out / "report.html").write_text(html)
    # Cue/target labels stay confined to audit reports, never training observations.
    memory = [
        d["source_metadata"]
        for d in details
        if "cue_start_step" in d["source_metadata"]
    ]
    if memory:
        pd.DataFrame(memory).to_json(
            out / "memory_metadata.json", orient="records", indent=2
        )
    if task["format"] == "neosim_hdf5":
        memory_rows = [
            json.loads(f.read_text())
            for f in sorted((Path(task["source"]) / "metadata").glob("*.json"))
        ]
        if memory_rows:
            md = pd.DataFrame(memory_rows)
            cols = [
                k
                for k in (
                    "seed",
                    "target_slot_idx",
                    "memory_delay_seconds",
                    "cue_start_step",
                    "cue_end_step",
                    "query_start_step",
                    "observation_pose_group",
                    "result",
                )
                if k in md
            ]
            md[cols].to_csv(out / "memory_episodes.csv", index=False)
            hz = task["robot"]["simulation_hz"]
            md["actual_delay_s"] = (md.query_start_step - md.cue_end_step) / hz
            memory_summary = dict(
                metadata_episodes=len(md),
                local_hdf5_episodes=len(rows),
                target_counts={
                    str(k): int(v) for k, v in md.target_slot_idx.value_counts().items()
                },
                actual_delay_counts={
                    str(k): int(v) for k, v in md.actual_delay_s.value_counts().items()
                },
                oracle_use="analysis only; never observations, actions or prompt",
            )
            (out / "memory_summary.json").write_text(
                json.dumps(memory_summary, indent=2)
            )
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            md.target_slot_idx.value_counts().sort_index().plot.bar(
                ax=axes[0], title="Target slots (metadata only)"
            )
            md.actual_delay_s.value_counts().sort_index().plot.bar(
                ax=axes[1], title="Actual cue–query delay (s)"
            )
            fig.tight_layout()
            fig.savefig(out / "memory_distribution.png", dpi=160)
            plt.close(fig)
            html = html.replace(
                "</body>",
                "<h2>Full remote metadata coverage</h2><pre>"
                + __import__("html").escape(json.dumps(memory_summary, indent=2))
                + '</pre><img src="memory_distribution.png" width="90%"></body>',
            )
            (out / "report.html").write_text(html)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary
