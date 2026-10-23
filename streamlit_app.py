"""Streamlit UI for pair previewing, cancellable coach stitching and test data."""

from __future__ import annotations

import json
import queue
import os
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import ConstructCoachImage as CCI
from stitcher_tools import (Cancelled, analyze_coach, build_panorama, discover_coaches,
                            generate_synthetic_dataset, process_coach)

st.set_page_config(page_title="Coach Stitcher", page_icon="🚆", layout="wide")

# ── Recent-path persistence ─────────────────────────────────────────────────
_RECENT_FILE = Path(__file__).parent / ".streamlit_recent_paths.json"
_MAX_RECENT = 12


def _load_recent_paths() -> dict:
    """Return {"input": [...], "output": [...]} from the JSON file on disk."""
    try:
        if _RECENT_FILE.is_file():
            data = json.loads(_RECENT_FILE.read_text(encoding="utf-8"))
            return {
                "input":  [p for p in data.get("input",  []) if isinstance(p, str)],
                "output": [p for p in data.get("output", []) if isinstance(p, str)],
            }
    except Exception:
        pass
    return {"input": [], "output": []}


def _save_recent_paths() -> None:
    """Persist the in-session recent-path lists to disk."""
    try:
        data = {
            "input":  st.session_state.get("_recent_input",  []),
            "output": st.session_state.get("_recent_output", []),
        }
        _RECENT_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def _push_recent(kind: str, path: str) -> None:
    """Move *path* to the front of the recent-<kind> list and persist."""
    key = f"_recent_{kind}"
    paths: list[str] = list(st.session_state.get(key, []))
    if path in paths:
        paths.remove(path)
    paths.insert(0, path)
    st.session_state[key] = paths[:_MAX_RECENT]
    _save_recent_paths()


def _browse_folder(initial: str = "") -> str:
    """Open a native OS folder-picker dialog. Returns the chosen path or ''."""
    try:
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", True)
        start = initial if initial and Path(initial).is_dir() else str(Path.home())
        folder = filedialog.askdirectory(title="Select folder", initialdir=start, parent=root)
        root.destroy()
        return folder or ""
    except Exception:
        return ""


def _folder_input(label: str, state_key: str, recent_kind: str, widget_suffix: str) -> None:
    """Render a folder-path text-input with a 📁 Browse button and a recents dropdown."""
    pending_key = f"_pending_browse_{widget_suffix}"

    # ── Consume any pending browse request BEFORE the text_input is rendered ─
    # (Streamlit forbids setting session_state[key] after the widget that owns
    #  that key has already been instantiated, so we must act here, at the top.)
    if st.session_state.get(pending_key, False):
        st.session_state[pending_key] = False
        chosen = _browse_folder(st.session_state.get(state_key, ""))
        if chosen:
            st.session_state[state_key] = chosen   # safe: widget not rendered yet
            _push_recent(recent_kind, chosen)

    recent_paths: list[str] = st.session_state.get(f"_recent_{recent_kind}", [])
    placeholder = "─  select a recent folder  ─"

    # ── text input + browse button ───────────────────────────────────────────
    col_txt, col_btn = st.columns([5, 1])
    with col_txt:
        st.text_input(label, key=state_key,
                      help="Paste a path, pick one from the dropdown, or click 📁 to browse.")
    with col_btn:
        st.markdown("<div style='margin-top:1.75rem'></div>", unsafe_allow_html=True)

        def _request_browse(pk: str = pending_key) -> None:
            """on_click: set a flag; the dialog opens on the next rerun."""
            st.session_state[pk] = True

        st.button("📁", key=f"_browse_{widget_suffix}", help="Open folder browser",
                  on_click=_request_browse, use_container_width=True)

    # ── recents dropdown ─────────────────────────────────────────────────────
    if recent_paths:
        options = [placeholder] + recent_paths

        def _apply_recent(sk: str = state_key, rk: str = recent_kind,
                          ws: str = widget_suffix) -> None:
            val = st.session_state.get(f"_rs_{ws}", placeholder)
            if val != placeholder:
                st.session_state[sk] = val
                _push_recent(rk, val)   # promote to top

        st.selectbox(
            f"Recent {label.lower()}s",
            options,
            key=f"_rs_{widget_suffix}",
            on_change=_apply_recent,
            label_visibility="collapsed",
            help=f"Pick a recently used {label.lower()} to populate the field above.",
        )


def _default_values() -> dict:
    return {
        "input_dir": str(CCI.BASE_INPUT_DIR),
        "output_dir": str(CCI.BASE_OUTPUT_DIR),
        "coach_depth": int(CCI.COACH_DEPTH),
        "top_crop": int(CCI.TOP_CROP),
        "clip_frac": float(CCI.CLIP_FRAC),
        "dx_min": int(CCI.DX_MIN),
        "dx_max": int(CCI.DX_MAX_SINGLE),
        "dx_reject": int(CCI.DX_REJECT),
        "resp_min": float(CCI.RESP_MIN),
        "fisheye": bool(CCI.APPLY_FISHEYE_CORRECTION),
        "camera_matrix": ", ".join(f"{v:g}" for v in CCI.CAMERA_MATRIX.flatten()),
        "dist_coeffs": ", ".join(f"{v:g}" for v in CCI.DIST_COEFFS.flatten()),
    }


def _init_state() -> None:
    for key, value in _default_values().items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("coach_results", {})
    st.session_state.setdefault("batch_log", [])
    st.session_state.setdefault("batch_progress", 0.0)
    st.session_state.setdefault("batch_current", "Idle")
    st.session_state.setdefault("batch_state", "idle")
    st.session_state.setdefault("preview_result", None)
    st.session_state.setdefault("preview_folder", "")
    st.session_state.setdefault("preview_stitch", False)
    st.session_state.setdefault("selected_output", "")
    st.session_state.setdefault("synthetic_root", str(CCI.BASE_INPUT_DIR))
    # Load persisted recent paths once per browser session
    if "_recents_loaded" not in st.session_state:
        recent = _load_recent_paths()
        st.session_state["_recent_input"]  = recent["input"]
        st.session_state["_recent_output"] = recent["output"]
        st.session_state["_recents_loaded"] = True


def _reset_defaults() -> None:
    for key, value in _default_values().items():
        st.session_state[key] = value
    st.session_state.preview_result = None
    st.session_state.preview_folder = ""
    st.session_state.preview_stitch = False


def _config() -> dict:
    input_text = st.session_state.input_dir.strip()
    output_text = st.session_state.output_dir.strip()
    if not input_text or not output_text:
        raise ValueError("Choose both an input root and an output folder.")
    config = {
        "input_dir": Path(input_text).expanduser(),
        "output_dir": Path(output_text).expanduser(),
        "depth": int(st.session_state.coach_depth),
        "top_crop": int(st.session_state.top_crop),
        "clip_frac": float(st.session_state.clip_frac),
        "dx_min": int(st.session_state.dx_min),
        "dx_max": int(st.session_state.dx_max),
        "dx_reject": int(st.session_state.dx_reject),
        "resp_min": float(st.session_state.resp_min),
        "fisheye": bool(st.session_state.fisheye),
    }
    try:
        matrix = [float(v.strip()) for v in st.session_state.camera_matrix.split(",")]
        coeffs = [float(v.strip()) for v in st.session_state.dist_coeffs.split(",")]
    except ValueError as exc:
        raise ValueError("Camera calibration fields must contain comma-separated numbers.") from exc
    if len(matrix) != 9 or not np.isfinite(matrix).all():
        raise ValueError("Camera matrix must contain exactly 9 finite values.")
    if len(coeffs) != 4 or not np.isfinite(coeffs).all():
        raise ValueError("Distortion coefficients must contain exactly 4 finite values.")
    if config["depth"] not in (1, 2):
        raise ValueError("Coach depth must be 1 or 2.")
    if config["top_crop"] < 0:
        raise ValueError("Top crop must be zero or greater.")
    if not 0 <= config["clip_frac"] < 0.5:
        raise ValueError("Clip fraction must be at least 0 and less than 0.5.")
    if not (0 < config["dx_min"] <= config["dx_max"] < config["dx_reject"]):
        raise ValueError("DX values must satisfy 0 < DX min ≤ DX max single < DX reject.")
    if not 0 <= config["resp_min"] <= 1:
        raise ValueError("Response minimum must be between 0 and 1.")
    config["camera_matrix"] = np.asarray(matrix, dtype=np.float32).reshape(3, 3)
    config["dist_coeffs"] = np.asarray(coeffs, dtype=np.float32)
    return config


@st.cache_data(ttl=4, show_spinner=False)
def _cached_discover(path: str, depth: int) -> list[str]:
    try:
        return [str(p) for p in discover_coaches(Path(path), depth)]
    except (OSError, PermissionError):
        return []


def _pair_figure(result: dict):
    records = result.get("records", [])
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.12,
                        subplot_titles=("Raw horizontal shift by frame pair", "Phase-correlation response"))
    colors = {"train-present": "#21a179", "rejected": "#e05d5d",
              "gap-filled": "#e6a23c"}
    labels = {"train-present": "Train-present", "rejected": "Rejected",
              "gap-filled": "Gap-filled"}
    for category in ("train-present", "rejected", "gap-filled"):
        rows = [r for r in records if r["category"] == category]
        if not rows:
            continue
        fig.add_trace(go.Scatter(
            x=[r["pair"] for r in rows], y=[r["raw_dx"] for r in rows], mode="markers",
            name=labels[category], showlegend=False,
            marker={"color": colors[category], "size": 9},
            customdata=[[r["response"], r["corrected_step"], r["frame_a"], r["frame_b"]] for r in rows],
            hovertemplate="Pair %{x}<br>dx %{y:.2f}px<br>response %{customdata[0]:.4f}<br>step %{customdata[1]:.2f}px<br>%{customdata[2]} → %{customdata[3]}<extra>%{fullData.name}</extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[r["pair"] for r in rows], y=[r["response"] for r in rows], mode="markers",
            name=labels[category], legendgroup=category, showlegend=False,
            marker={"color": colors[category], "size": 8},
            hovertemplate="Pair %{x}<br>response %{y:.4f}<extra>" + labels[category] + "</extra>",
        ), row=2, col=1)
    if result.get("records"):
        fig.add_hline(y=result["records"][0]["true_dx_est"], line_dash="dot", line_color="#8d77c7",
                      annotation_text="Estimated step", row=1, col=1)
        fig.add_hline(y=st.session_state.dx_min, line_dash="dash", line_color="#999999",
                      annotation_text="DX min", row=1, col=1)
        fig.add_hline(y=st.session_state.resp_min, line_dash="dash", line_color="#999999",
                      annotation_text="Response min", row=2, col=1)
    fig.update_xaxes(title_text="Consecutive frame pair", row=2, col=1)
    fig.update_yaxes(title_text="dx (pixels)", row=1, col=1, zeroline=True)
    fig.update_yaxes(title_text="response", row=2, col=1, rangemode="tozero")
    fig.update_layout(height=520, margin={"l": 18, "r": 18, "t": 55, "b": 35},
                      showlegend=False, hovermode="closest")
    return fig


def _worker(input_dir: Path, output_dir: Path, depth: int, config: dict,
            folders: list[Path], events: queue.Queue, cancel: threading.Event) -> None:
    started = time.monotonic()
    completed = 0
    cancelled = False
    events.put(("log", f"Starting {len(folders)} coach folder(s). Output: {output_dir}"))
    for index, folder in enumerate(folders, start=1):
        if cancel.is_set():
            cancelled = True
            break
        events.put(("current", f"Coach {index}/{len(folders)} · {folder.name}"))
        events.put(("log", f"[{index}/{len(folders)}] Processing {folder}"))
        output_parent = output_dir / folder.relative_to(input_dir).parent
        try:
            outcome = process_coach(folder, output_parent, config, cancel,
                                    log=lambda line: events.put(("log", line)))
        except Cancelled:
            cancelled = True
            events.put(("log", f"Cancellation reached a safe checkpoint in {folder.name}."))
            break
        except Exception as exc:
            outcome = {"status": "error", "folder": str(folder), "reason": f"{type(exc).__name__}: {exc}"}
            events.put(("log", f"[ERROR] {folder.name}: {outcome['reason']}"))
        events.put(("coach", outcome))
        completed += 1
        events.put(("progress", completed / max(1, len(folders)), completed, len(folders)))
    cancelled = cancelled or cancel.is_set()
    state = "cancelled" if cancelled else "complete"
    events.put(("done", state, completed, len(folders), time.monotonic() - started))


def _drain_events() -> None:
    job = st.session_state.get("run_job")
    if not job:
        return
    events = job["events"]
    while True:
        try:
            event = events.get_nowait()
        except queue.Empty:
            break
        kind, *payload = event
        if kind == "log":
            st.session_state.batch_log = (st.session_state.batch_log + [payload[0]])[-250:]
        elif kind == "current":
            st.session_state.batch_current = payload[0]
        elif kind == "progress":
            st.session_state.batch_progress = payload[0]
        elif kind == "coach":
            result = payload[0]
            st.session_state.coach_results[result["folder"]] = result
        elif kind == "done":
            state, completed, total, seconds = payload
            st.session_state.batch_state = state
            st.session_state.batch_current = f"{state.title()} · {completed}/{total} coaches · {seconds:.1f}s"
            st.session_state.batch_log = (st.session_state.batch_log + [st.session_state.batch_current])[-250:]
            st.session_state.run_job = None


def _batch_panel() -> None:
    @st.fragment(run_every="500ms")
    def panel():
        _drain_events()
        state = st.session_state.batch_state
        st.progress(float(st.session_state.batch_progress), text=st.session_state.batch_current)
        if state == "running":
            job = st.session_state.get("run_job")
            if job and not job["cancel"].is_set():
                if st.button("Cancel run", key="cancel_batch", type="secondary"):
                    job["cancel"].set()
                    st.session_state.batch_current = "Cancelling after the current frame pair…"
                    st.session_state.batch_log = (st.session_state.batch_log + ["Cancel requested; stopping at the next safe checkpoint."])[-250:]
        elif state == "complete":
            st.success(st.session_state.batch_current)
        elif state == "cancelled":
            st.warning(st.session_state.batch_current)
        with st.expander("Run log", expanded=state == "running"):
            st.code("\n".join(st.session_state.batch_log[-120:]) or "No run log yet.", language=None, height=240)
    panel()


def _render_stitcher() -> None:
    left, right = st.columns([.95, 1.8], gap="large")
    with left:
        st.subheader("Folders and parameters")
        _folder_input("Input root", "input_dir", "input", "input_dir")
        _folder_input("Output folder", "output_dir", "output", "output_dir")
        st.selectbox("Coach depth", [1, 2], key="coach_depth",
                     help="1: coach folders directly under root. 2: one parent folder above coaches.")
        input_path = Path(st.session_state.input_dir).expanduser()
        coach_paths = [Path(p) for p in _cached_discover(str(input_path), int(st.session_state.coach_depth))]
        st.metric("Coach folders found", len(coach_paths))
        if not input_path.is_dir():
            st.caption("Input root does not exist yet. Paste a valid folder path or generate sample data.")
        elif not coach_paths:
            st.caption("No folders were found at this depth.")
        else:
            with st.expander("Found coaches", expanded=False):
                st.caption("\n".join(p.name if int(st.session_state.coach_depth) == 1 else f"{p.parent.name}/{p.name}" for p in coach_paths[:100]))

        st.divider()
        st.markdown("**Preprocessing**")
        st.number_input("Top crop (rows)", min_value=0, step=1, key="top_crop")
        st.number_input("Clip fraction", min_value=0.0, max_value=.49, step=.01, format="%.3f", key="clip_frac")
        st.markdown("**Phase correlation**")
        st.number_input("DX minimum (px)", min_value=1, step=1, key="dx_min")
        st.number_input("DX maximum single step (px)", min_value=1, step=1, key="dx_max")
        st.number_input("DX reject threshold (px)", min_value=1, step=1, key="dx_reject")
        st.number_input("Response minimum", min_value=0.0, max_value=1.0, step=.01, format="%.3f", key="resp_min")
        with st.expander("Fisheye correction and calibration", expanded=False):
            st.checkbox("Enable fisheye correction", key="fisheye")
            st.caption("The defaults below are copied from ConstructCoachImage.py. Replace the example calibration values with your camera's calibration when available.")
            st.text_input("Camera matrix (9 comma-separated values)", key="camera_matrix")
            st.text_input("Distortion coefficients (4 values)", key="dist_coeffs")
        st.button("Reset to defaults", on_click=_reset_defaults, use_container_width=True)
        if st.button("Open output folder", use_container_width=True):
            output_folder = Path(st.session_state.output_dir).expanduser()
            output_folder.mkdir(parents=True, exist_ok=True)
            if hasattr(os, "startfile"):
                os.startfile(str(output_folder.resolve()))
            else:
                st.info(f"Output folder: {output_folder.resolve()}")

        st.divider()
        st.markdown("**Pair-only preview**")
        choices = [str(p) for p in coach_paths]
        selected_preview = st.selectbox("Coach to preview", choices, format_func=lambda p: Path(p).name,
                                        key="preview_choice", disabled=not choices)
        st.checkbox("Also build a stitched panorama (slower)", key="preview_stitch",
                    help="Creates the panorama in memory using the same strip-stitch algorithm. It does not save files.")
        if st.button("Preview dx and response", use_container_width=True, disabled=not selected_preview):
            try:
                config = _config()
                with st.spinner("Loading frames and measuring pairs…"):
                    result = analyze_coach(Path(selected_preview), config,
                                           include_images=bool(st.session_state.preview_stitch))
                    if result["status"] == "ok" and st.session_state.preview_stitch:
                        images = result.pop("_images")
                        panorama = build_panorama(images, result, config)
                        scale = min(1.0, 1400 / max(panorama.shape))
                        if scale < 1.0:
                            panorama = cv2.resize(panorama, None, fx=scale, fy=scale,
                                                  interpolation=cv2.INTER_AREA)
                        result["preview_panorama"] = panorama
                st.session_state.preview_result = result
                st.session_state.preview_folder = selected_preview
                if result["status"] == "skipped":
                    st.warning(result["reason"])
            except Exception as exc:
                st.error(f"Preview failed: {exc}")

        running = st.session_state.batch_state == "running"
        if st.button("Run all coaches", type="primary", use_container_width=True,
                     disabled=running or not coach_paths):
            try:
                config = _config()
                if not input_path.is_dir():
                    raise ValueError(f"Input root does not exist: {input_path}")
                output_path = config["output_dir"]
                output_path.mkdir(parents=True, exist_ok=True)
                events: queue.Queue = queue.Queue()
                cancel = threading.Event()
                st.session_state.batch_log = []
                st.session_state.batch_progress = 0.0
                st.session_state.batch_current = "Preparing background worker…"
                st.session_state.batch_state = "running"
                thread = threading.Thread(target=_worker,
                                          args=(input_path, output_path, config["depth"], config,
                                                coach_paths, events, cancel), daemon=True,
                                          name="coach-stitcher-worker")
                st.session_state.run_job = {"events": events, "cancel": cancel, "thread": thread}
                thread.start()
            except Exception as exc:
                st.error(f"Could not start stitcher: {exc}")

    with right:
        st.subheader("Coach results and viewer")
        results = st.session_state.coach_results
        rows = []
        for folder in coach_paths:
            outcome = results.get(str(folder))
            status = outcome["status"] if outcome else "ready"
            icon = {"ok": "✅ OK", "skipped": "⏭ Skipped",
                    "error": "❌ Error", "ready": "○ Ready"}.get(status, status)
            rows.append({"Coach": folder.name, "Status": icon,
                         "Details": (outcome or {}).get("reason", "")})
        selection = None
        if rows:
            event = st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                                 on_select="rerun", selection_mode="single-row", key="coach_status_table")
            selected_rows = event.selection.rows if event and event.selection else []
            if selected_rows:
                selected_folder = str(coach_paths[selected_rows[0]])
                st.session_state.selected_output = selected_folder
                selection = selected_folder
        else:
            st.info("Coach folders will appear here when the input root and depth point to a valid dataset.")
        if selection is None and st.session_state.selected_output in {str(p) for p in coach_paths}:
            selection = st.session_state.selected_output

        preview = st.session_state.preview_result
        if preview and st.session_state.preview_folder in {str(p) for p in coach_paths}:
            st.markdown(f"**Preview: {Path(st.session_state.preview_folder).name}**")
            preview_path = Path(st.session_state.preview_folder)
            preview_root = Path(st.session_state.input_dir).expanduser()
            preview_output = (Path(st.session_state.output_dir).expanduser()
                              / preview_path.relative_to(preview_root).parent
                              / f"{preview_path.name}.png")
            preview_panorama = preview.get("preview_panorama")
            panorama_caption = "In-memory stitched preview · not saved"
            if preview_panorama is None and preview_output.is_file():
                existing_panorama = cv2.imread(str(preview_output), cv2.IMREAD_COLOR)
                if existing_panorama is not None:
                    scale = min(1.0, 1200 / max(existing_panorama.shape[:2]))
                    if scale < 1.0:
                        existing_panorama = cv2.resize(existing_panorama, None, fx=scale, fy=scale,
                                                       interpolation=cv2.INTER_AREA)
                    preview_panorama = cv2.cvtColor(existing_panorama, cv2.COLOR_BGR2GRAY)
                    panorama_caption = "Existing downscaled panorama · not recomputed by preview"

            if preview_panorama is not None:
                plot_col, image_col = st.columns([1.35, 1], gap="medium")
                with plot_col:
                    if preview.get("records"):
                        st.caption("🟢 Train-present · 🔴 Rejected · 🟠 Gap-filled")
                        st.plotly_chart(_pair_figure(preview), use_container_width=True,
                                        key="pair_preview_plot")
                with image_col:
                    st.image(preview_panorama, caption=panorama_caption, use_container_width=True)
            elif preview.get("records"):
                st.caption("🟢 Train-present · 🔴 Rejected · 🟠 Gap-filled")
                st.plotly_chart(_pair_figure(preview), use_container_width=True, key="pair_preview_plot")
            if preview.get("records"):
                st.caption(f"{preview['frame_count']} frames · estimated step {preview['true_dx']:.1f}px · preview writes no files")
            if preview_panorama is None and preview.get("preview_frame") is not None:
                st.image(preview["preview_frame"], caption="Representative input frame · no panorama exists yet",
                         use_container_width=True)

        if selection:
            outcome = results.get(selection)
            if outcome and outcome.get("status") == "ok" and Path(outcome.get("png", "")).is_file():
                st.markdown(f"**Panorama: {Path(selection).name}**")
                zoom = st.slider("Viewer zoom", min_value=25, max_value=250, value=100, step=25,
                                 format="%d%%", key="panorama_zoom")
                image = cv2.imread(outcome["png"], cv2.IMREAD_COLOR)
                if image is not None:
                    if zoom != 100:
                        factor = zoom / 100
                        image = cv2.resize(image, None, fx=factor, fy=factor,
                                           interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR)
                    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=f"{image.shape[1]} × {image.shape[0]} px",
                             use_container_width=False)
                    st.caption("Use the page scrollbars to pan when the zoomed panorama extends beyond the viewer.")
                c1, c2 = st.columns(2)
                with c1:
                    st.download_button("Download panorama PNG", Path(outcome["png"]).read_bytes(),
                                       file_name=Path(outcome["png"]).name, mime="image/png")
                with c2:
                    st.download_button("Download speed CSV", Path(outcome["speed_csv"]).read_bytes(),
                                       file_name=Path(outcome["speed_csv"]).name, mime="text/csv")
                if Path(outcome.get("speed_csv", "")).is_file():
                    st.markdown("**Speed CSV**")
                    st.dataframe(pd.read_csv(outcome["speed_csv"]), hide_index=True,
                                 use_container_width=True, height=250)
                if Path(outcome.get("all_pairs_csv", "")).is_file():
                    with st.expander("All-pairs measurements"):
                        st.dataframe(pd.read_csv(outcome["all_pairs_csv"]), hide_index=True,
                                     use_container_width=True, height=250)
            elif outcome:
                st.info(outcome.get("reason", "This coach has no panorama to display."))

    st.divider()
    st.subheader("Batch progress and log")
    _batch_panel()
    st.caption(f"Output folder: `{Path(st.session_state.output_dir).expanduser()}`")


def _render_generator() -> None:
    st.subheader("Generate synthetic coach data")
    st.write("Create ordinary grayscale frame folders that the stitcher can discover and process. Generated files stay under the selected root.")
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        _folder_input("Synthetic input root", "synthetic_root", "input", "synthetic_root")
        root = st.session_state.get("synthetic_root", str(CCI.BASE_INPUT_DIR))
        depth = st.selectbox("Folder depth", [1, 2], index=0, key="synthetic_depth")
        coaches = st.number_input("Number of coaches", min_value=1, max_value=100, value=2, step=1, key="synthetic_coaches")
        frames = st.number_input("Frames per coach", min_value=3, max_value=500, value=30, step=1, key="synthetic_frames")
    with col2:
        width = st.number_input("Frame width", min_value=320, max_value=2560, value=1280, step=80, key="synthetic_width")
        height = st.number_input("Frame height", min_value=160, max_value=1440, value=720, step=40, key="synthetic_height")
        step = st.number_input("Train shift per pair (px)", min_value=1, max_value=500, value=42, step=1, key="synthetic_step")
        gap = st.number_input("Insert a gap every N pairs (0 disables)", min_value=0, max_value=100, value=8, step=1, key="synthetic_gap")
    st.info("Generated motion is set to the requested pixel shift. Periodic zero-motion pairs are optional and are useful for checking gap filling.")
    if st.button("Generate coach data", type="primary", use_container_width=False):
        try:
            target = Path(root).expanduser()
            made = generate_synthetic_dataset(target, int(depth), int(coaches), int(frames),
                                              int(width), int(height), int(step), int(gap))
            _cached_discover.clear()
            st.success(f"Created {len(made)} coach folder(s) under {target}.")
            st.code("\n".join(str(p) for p in made), language=None)
            st.caption("Set Stitcher → Input root to this folder and choose the same coach depth to use the generated data.")
        except Exception as exc:
            st.error(f"Data generation failed: {exc}")


def main() -> None:
    _init_state()
    st.title("🚆 Coach Stitcher")
    st.caption("Tune pair detection, stitch coach panoramas, and inspect per-pair speed measurements.")
    stitcher_tab, generator_tab = st.tabs(["Stitcher", "Generate sample data"])
    with stitcher_tab:
        _render_stitcher()
    with generator_tab:
        _render_generator()


if __name__ == "__main__":
    main()
