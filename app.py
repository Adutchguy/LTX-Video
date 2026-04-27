"""
LTX-Video Gradio UI
Run with: python app.py
Then open http://localhost:7860 in your browser.
"""

import os
import sys
import gradio as gr
from pathlib import Path
from datetime import datetime

# Ensure the LTX-Video package is importable from this directory
sys.path.insert(0, str(Path(__file__).parent))

from ltx_video.inference import InferenceConfig, infer

# ---------------------------------------------------------------------------
# Model config registry
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "13B Distilled ⚡ (Recommended)":   "configs/ltxv-13b-0.9.8-distilled.yaml",
    "13B Dev 🎨 (Highest Quality)":     "configs/ltxv-13b-0.9.8-dev.yaml",
    "13B Distilled FP8 ⚡ (Less VRAM)": "configs/ltxv-13b-0.9.8-distilled-fp8.yaml",
    "13B Dev FP8 🎨 (Less VRAM)":       "configs/ltxv-13b-0.9.8-dev-fp8.yaml",
    "2B Distilled 🪶 (Low VRAM)":       "configs/ltxv-2b-0.9.8-distilled.yaml",
    "2B Distilled FP8 🪶 (Lowest VRAM)":"configs/ltxv-2b-0.9.8-distilled-fp8.yaml",
}

RESOLUTION_PRESETS = {
    "1216×704  (16:9 Landscape)": (1216, 704),
    "704×1216  (9:16 Portrait)":  (704, 1216),
    "960×960   (1:1 Square)":     (960, 960),
    "768×512   (3:2)":            (768, 512),
    "Custom":                     None,
}

DEFAULT_NEGATIVE = (
    "worst quality, inconsistent motion, blurry, jittery, distorted"
)

OUTPUT_DIR = Path("outputs") / "gradio"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_output_dir() -> Path:
    out = OUTPUT_DIR / datetime.today().strftime("%Y-%m-%d")
    out.mkdir(parents=True, exist_ok=True)
    return out


def latest_mp4(directory: Path):
    files = sorted(directory.glob("*.mp4"), key=os.path.getmtime, reverse=True)
    return str(files[0]) if files else None


def apply_resolution_preset(preset_name, current_width, current_height):
    """Return (width, height) based on preset selection."""
    entry = RESOLUTION_PRESETS.get(preset_name)
    if entry is None:
        return current_width, current_height
    w, h = entry
    return w, h


# ---------------------------------------------------------------------------
# Inference wrappers
# ---------------------------------------------------------------------------

def run_text_to_video(
    prompt,
    negative_prompt,
    model_config_name,
    width,
    height,
    num_frames,
    frame_rate,
    seed,
    offload_to_cpu,
    progress=gr.Progress(track_tqdm=True),
):
    if not prompt.strip():
        return None, "⚠️ Please enter a prompt."

    output_dir = get_output_dir()

    config = InferenceConfig(
        prompt=prompt,
        negative_prompt=negative_prompt,
        pipeline_config=MODEL_CONFIGS[model_config_name],
        height=int(height),
        width=int(width),
        num_frames=int(num_frames),
        frame_rate=int(frame_rate),
        seed=int(seed),
        offload_to_cpu=offload_to_cpu,
        output_path=str(output_dir),
    )

    try:
        infer(config=config)
    except Exception as e:
        return None, f"❌ Generation failed:\n{e}"

    video_path = latest_mp4(output_dir)
    if video_path:
        return video_path, f"✅ Saved to: {video_path}"
    return None, "❌ No output file found — check terminal for errors."


def run_image_to_video(
    image_path,
    prompt,
    negative_prompt,
    model_config_name,
    width,
    height,
    num_frames,
    frame_rate,
    seed,
    offload_to_cpu,
    progress=gr.Progress(track_tqdm=True),
):
    if not prompt.strip():
        return None, "⚠️ Please enter a prompt."
    if image_path is None:
        return None, "⚠️ Please upload a conditioning image."

    output_dir = get_output_dir()

    config = InferenceConfig(
        prompt=prompt,
        negative_prompt=negative_prompt,
        pipeline_config=MODEL_CONFIGS[model_config_name],
        height=int(height),
        width=int(width),
        num_frames=int(num_frames),
        frame_rate=int(frame_rate),
        seed=int(seed),
        offload_to_cpu=offload_to_cpu,
        output_path=str(output_dir),
        conditioning_media_paths=[image_path],
        conditioning_start_frames=[0],
    )

    try:
        infer(config=config)
    except Exception as e:
        return None, f"❌ Generation failed:\n{e}"

    video_path = latest_mp4(output_dir)
    if video_path:
        return video_path, f"✅ Saved to: {video_path}"
    return None, "❌ No output file found — check terminal for errors."


# ---------------------------------------------------------------------------
# Shared parameter block builder
# ---------------------------------------------------------------------------

def build_shared_params(tab_prefix: str):
    """Returns a dict of Gradio components used in both tabs."""
    with gr.Row():
        model_dd = gr.Dropdown(
            choices=list(MODEL_CONFIGS.keys()),
            value=list(MODEL_CONFIGS.keys())[0],
            label="Model Config",
            scale=2,
        )

    with gr.Row():
        preset_dd = gr.Dropdown(
            choices=list(RESOLUTION_PRESETS.keys()),
            value="1216×704  (16:9 Landscape)",
            label="Resolution Preset",
            scale=2,
        )

    with gr.Row():
        width_sl = gr.Slider(
            minimum=256, maximum=1920, step=32,
            value=1216, label="Width (px)"
        )
        height_sl = gr.Slider(
            minimum=256, maximum=1080, step=32,
            value=704, label="Height (px)"
        )

    with gr.Row():
        frames_sl = gr.Slider(
            minimum=9, maximum=257, step=8,
            value=97, label="Number of Frames",
            info="Must be N×8 + 1 (e.g. 25, 33, 65, 97, 121, 161 …)"
        )
        fps_sl = gr.Slider(
            minimum=8, maximum=60, step=1,
            value=30, label="Frame Rate (fps)"
        )

    with gr.Accordion("Advanced Options", open=False):
        with gr.Row():
            seed_nb = gr.Number(value=171198, label="Seed", precision=0)
            offload_cb = gr.Checkbox(
                value=False,
                label="Offload to CPU (saves VRAM, slower)",
            )
        neg_prompt = gr.Textbox(
            value=DEFAULT_NEGATIVE,
            label="Negative Prompt",
            lines=2,
        )

    # Wire preset → width/height
    def _update_res(preset, w, h):
        entry = RESOLUTION_PRESETS.get(preset)
        if entry is None:
            return w, h
        return entry[0], entry[1]

    preset_dd.change(
        _update_res,
        inputs=[preset_dd, width_sl, height_sl],
        outputs=[width_sl, height_sl],
    )

    return dict(
        model_dd=model_dd,
        width_sl=width_sl,
        height_sl=height_sl,
        frames_sl=frames_sl,
        fps_sl=fps_sl,
        seed_nb=seed_nb,
        offload_cb=offload_cb,
        neg_prompt=neg_prompt,
    )


# ---------------------------------------------------------------------------
# Build the Gradio app
# ---------------------------------------------------------------------------

def build_app():
    with gr.Blocks(title="LTX-Video") as demo:

        gr.Markdown(
            """
            # 🎬 LTX-Video
            **Lightricks LTX-Video** — local video generation.
            Select a tab, fill in your prompt, and hit **Generate**.
            *Model weights are downloaded from HuggingFace on first run.*
            """
        )

        with gr.Tabs():

            # ── TEXT-TO-VIDEO ────────────────────────────────────────────
            with gr.Tab("📝 Text-to-Video"):
                t2v_prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="A serene mountain lake at sunrise, mist rising off the water…",
                    lines=3,
                )
                t2v_params = build_shared_params("t2v")

                with gr.Row():
                    t2v_btn = gr.Button("🎬 Generate", variant="primary", scale=2)
                    t2v_clear = gr.Button("🗑️ Clear", scale=1)

                t2v_status = gr.Textbox(label="Status", interactive=False, lines=1)
                t2v_out = gr.Video(label="Output Video", interactive=False)

                t2v_btn.click(
                    fn=run_text_to_video,
                    inputs=[
                        t2v_prompt,
                        t2v_params["neg_prompt"],
                        t2v_params["model_dd"],
                        t2v_params["width_sl"],
                        t2v_params["height_sl"],
                        t2v_params["frames_sl"],
                        t2v_params["fps_sl"],
                        t2v_params["seed_nb"],
                        t2v_params["offload_cb"],
                    ],
                    outputs=[t2v_out, t2v_status],
                )
                t2v_clear.click(
                    fn=lambda: (None, ""),
                    outputs=[t2v_out, t2v_status],
                )

            # ── IMAGE-TO-VIDEO ───────────────────────────────────────────
            with gr.Tab("🖼️ Image-to-Video"):
                with gr.Row():
                    i2v_image = gr.Image(
                        label="Conditioning Image",
                        type="filepath",
                        scale=1,
                    )
                    with gr.Column(scale=2):
                        i2v_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="Describe the motion you want to see…",
                            lines=4,
                        )

                i2v_params = build_shared_params("i2v")

                with gr.Row():
                    i2v_btn = gr.Button("🎬 Generate", variant="primary", scale=2)
                    i2v_clear = gr.Button("🗑️ Clear", scale=1)

                i2v_status = gr.Textbox(label="Status", interactive=False, lines=1)
                i2v_out = gr.Video(label="Output Video", interactive=False)

                i2v_btn.click(
                    fn=run_image_to_video,
                    inputs=[
                        i2v_image,
                        i2v_prompt,
                        i2v_params["neg_prompt"],
                        i2v_params["model_dd"],
                        i2v_params["width_sl"],
                        i2v_params["height_sl"],
                        i2v_params["frames_sl"],
                        i2v_params["fps_sl"],
                        i2v_params["seed_nb"],
                        i2v_params["offload_cb"],
                    ],
                    outputs=[i2v_out, i2v_status],
                )
                i2v_clear.click(
                    fn=lambda: (None, None, ""),
                    outputs=[i2v_image, i2v_out, i2v_status],
                )

        gr.Markdown(
            """
            ---
            **Tips:**
            - Frame count must be **N×8 + 1** (e.g. 25, 33, 65, 97, 121, 161, 241 …)
            - The **Distilled** models are much faster and need fewer VRAM.  Use **FP8** variants if you have < 16 GB VRAM.
            - Videos are saved to `outputs/gradio/YYYY-MM-DD/` in your LTX-Video folder.
            - If generation is slow, enable **Offload to CPU** in Advanced Options.
            """
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        inbrowser=True,
        share=False,
        theme=gr.themes.Soft(primary_hue="violet", neutral_hue="slate"),
    )
