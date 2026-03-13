import gradio as gr
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
emotion_extractor = AutoFeatureExtractor.from_pretrained("dima806/facial_emotions_image_detection")
emotion_model = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection").to(device)

def analyze(image, num_beams):
    if image is None:
        return "No image provided.", "No image provided."
    pil_image = Image.fromarray(image).convert("RGB")
    inputs = blip_processor(pil_image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = blip_model.generate(**inputs, num_beams=int(num_beams), max_length=50)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    emotion_inputs = emotion_extractor(images=pil_image, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = emotion_model(**emotion_inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    labels = emotion_model.config.id2label
    results = sorted([(labels[i], probs[i].item()) for i in range(len(probs))], key=lambda x: x[1], reverse=True)
    return caption, "\n".join([f"{label}: {score:.1%}" for label, score in results])

with gr.Blocks(title="Emotion and Caption AI") as demo:
    gr.Markdown("# Emotion Detection and Image Captioning")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Image", type="numpy")
            beam_slider = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="Beam Width")
            run_btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            caption_output = gr.Textbox(label="Caption", lines=3)
            emotion_output = gr.Textbox(label="Emotions", lines=10)
    run_btn.click(fn=analyze, inputs=[image_input, beam_slider], outputs=[caption_output, emotion_output], api_name="analyze")

demo.launch(server_name="0.0.0.0", server_port=7860)
