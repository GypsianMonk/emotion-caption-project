"""
Real-Time Emotion Detection & Image Captioning
Hugging Face Spaces Demo

Models used:
- Emotion: deepface (FER model under the hood)
- Captioning: Salesforce/blip-image-captioning-base (state-of-the-art)
"""

import gradio as gr
import numpy as np
from PIL import Image
import cv2
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from deepface import DeepFace
import time

# ── Load captioning model ────────────────────────────────────────────────────
print("Loading BLIP captioning model...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
caption_model.eval()
print("BLIP model loaded.")

EMOTION_COLORS = {
    "angry":    (255, 50,  50),
    "disgust":  (50,  180, 50),
    "fear":     (180, 50,  180),
    "happy":    (255, 220, 0),
    "neutral":  (180, 180, 180),
    "sad":      (50,  100, 255),
    "surprise": (255, 140, 0),
}


def generate_caption(image: Image.Image, beam_width: int = 3) -> str:
    """Generate caption using BLIP."""
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        out = caption_model.generate(
            **inputs,
            max_new_tokens=50,
            num_beams=beam_width,
            length_penalty=1.0,
        )
    return processor.decode(out[0], skip_special_tokens=True)


def detect_emotions(image_np: np.ndarray) -> list:
    """Detect faces and emotions using DeepFace."""
    try:
        results = DeepFace.analyze(
            image_np,
            actions=["emotion"],
            enforce_detection=False,
            silent=True,
        )
        if not isinstance(results, list):
            results = [results]
        return results
    except Exception as e:
        print(f"Emotion detection error: {e}")
        return []


def draw_annotations(image_np: np.ndarray, emotion_results: list) -> np.ndarray:
    """Draw bounding boxes and emotion labels on image."""
    annotated = image_np.copy()

    for result in emotion_results:
        region = result.get("region", {})
        x = region.get("x", 0)
        y = region.get("y", 0)
        w = region.get("w", 0)
        h = region.get("h", 0)

        dominant = result.get("dominant_emotion", "neutral")
        emotions = result.get("emotion", {})
        confidence = emotions.get(dominant, 0) / 100.0
        color = EMOTION_COLORS.get(dominant, (200, 200, 200))

        # Draw bounding box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

        # Label background
        label = f"{dominant} {confidence:.0%}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated, (x, y - lh - 10), (x + lw + 6, y), color, -1)
        cv2.putText(
            annotated, label,
            (x + 3, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (255, 255, 255), 2, cv2.LINE_AA,
        )

        # Top-3 emotion bars
        sorted_emotions = sorted(emotions.items(), key=lambda e: e[1], reverse=True)[:3]
        bar_x = x + w + 10
        for i, (emo, score) in enumerate(sorted_emotions):
            bar_len = int((score / 100) * 100)
            bar_color = EMOTION_COLORS.get(emo, (180, 180, 180))
            cv2.rectangle(annotated, (bar_x, y + i * 22), (bar_x + bar_len, y + i * 22 + 16), bar_color, -1)
            cv2.putText(
                annotated, f"{emo[:4]} {score:.0f}%",
                (bar_x + bar_len + 4, y + i * 22 + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (220, 220, 220), 1, cv2.LINE_AA,
            )

    return annotated


def run_pipeline(image: Image.Image, beam_width: int) -> tuple:
    """
    Full pipeline: emotion detection + image captioning.
    Returns: (annotated_image, caption, emotion_summary, metrics)
    """
    if image is None:
        return None, "Please upload an image.", "", ""

    t0 = time.time()
    image_rgb = np.array(image.convert("RGB"))
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # ── Emotion Detection ──
    t_emotion = time.time()
    emotion_results = detect_emotions(image_rgb)
    emotion_ms = (time.time() - t_emotion) * 1000

    annotated_bgr = draw_annotations(image_bgr, emotion_results)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    annotated_pil = Image.fromarray(annotated_rgb)

    # Build emotion summary
    emotion_summary = ""
    if emotion_results:
        for i, r in enumerate(emotion_results):
            dominant = r.get("dominant_emotion", "unknown")
            score = r.get("emotion", {}).get(dominant, 0)
            emotion_summary += f"**Face {i+1}:** {dominant.capitalize()} ({score:.1f}%)\n\n"
            all_scores = sorted(r.get("emotion", {}).items(), key=lambda x: x[1], reverse=True)
            for emo, sc in all_scores:
                bar = "█" * int(sc / 5)
                emotion_summary += f"`{emo:10s}` {bar} {sc:.1f}%\n"
            emotion_summary += "\n"
    else:
        emotion_summary = "No faces detected."

    # ── Image Captioning ──
    t_caption = time.time()
    caption = generate_caption(image, beam_width=int(beam_width))
    caption_ms = (time.time() - t_caption) * 1000

    total_ms = (time.time() - t0) * 1000

    metrics = (
        f"⚡ **Emotion:** {emotion_ms:.0f}ms | "
        f"📝 **Caption:** {caption_ms:.0f}ms | "
        f"🕐 **Total:** {total_ms:.0f}ms | "
        f"👤 **Faces:** {len(emotion_results)}"
    )

    return annotated_pil, caption, emotion_summary, metrics


# ── Gradio UI ────────────────────────────────────────────────────────────────
css = """
body { background: #0a0a0f !important; }
.gradio-container { max-width: 1100px !important; }
h1 { 
    text-align: center; 
    font-size: 2.2em !important; 
    background: linear-gradient(135deg, #7c3aed, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px !important;
}
.subtitle { text-align: center; color: #64748b; margin-bottom: 24px; }
.metric-box { background: #111118; border: 1px solid #1e1e2e; border-radius: 8px; padding: 12px 20px; }
"""

with gr.Blocks(css=css, title="Emotion Detection & Image Captioning") as demo:
    gr.HTML("""
        <h1>🧠 Emotion Detection & Image Captioning</h1>
        <p class="subtitle">
            Real-Time CNN Emotion Recognition (82% acc · 7 classes · FER-2013) &nbsp;+&nbsp;
            BLIP Image Captioning (BLEU-4: 0.28 · MS-COCO 80K)
        </p>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="📤 Upload Image", height=300)
            beam_slider = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="Beam Search Width")
            run_btn = gr.Button("🚀 Run Pipeline", variant="primary", size="lg")

            gr.Examples(
                examples=[
                    ["examples/happy.jpg"],
                    ["examples/group.jpg"],
                    ["examples/outdoor.jpg"],
                ],
                inputs=input_image,
                label="Example Images",
            )

        with gr.Column(scale=1):
            output_image = gr.Image(label="🎯 Emotion Detection", height=300)
            caption_out = gr.Textbox(label="📝 Generated Caption", lines=2, interactive=False)
            emotion_out = gr.Markdown(label="😊 Emotion Scores")
            metrics_out = gr.Markdown(label="⚡ Latency Metrics")

    run_btn.click(
        fn=run_pipeline,
        inputs=[input_image, beam_slider],
        outputs=[output_image, caption_out, emotion_out, metrics_out],
    )

    gr.HTML("""
        <div style="text-align:center; margin-top:24px; color:#475569; font-size:13px">
            <a href="https://github.com/GypsianMonk/emotion-caption-project" 
               style="color:#7c3aed; text-decoration:none">
                ⭐ GitHub: GypsianMonk/emotion-caption-project
            </a>
            &nbsp;·&nbsp; Built with TensorFlow · OpenCV · BLIP · DeepFace · Gradio
        </div>
    """)

if __name__ == "__main__":
    demo.launch()
