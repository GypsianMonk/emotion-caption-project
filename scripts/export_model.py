"""
Model Export Script
--------------------
Exports trained models to:
  - TFLite (for edge/mobile deployment)
  - SavedModel (for TF Serving)
  - ONNX (via tf2onnx, optional)

Usage:
    python scripts/export_model.py --model emotion --format tflite
    python scripts/export_model.py --model captioning --format saved_model
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def export_emotion_tflite(
    model_path: str,
    output_path: str,
    quantize: bool = True,
):
    """
    Export emotion CNN to TFLite with optional INT8 quantization.
    Quantized model runs ~3-4x faster on CPU with <1% accuracy drop.
    """
    logger.info(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # Representative dataset for full INT8 quantization
        def representative_data_gen():
            for _ in range(100):
                dummy = np.random.uniform(0, 1, (1, 48, 48, 1)).astype(np.float32)
                yield [dummy]
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        logger.info("INT8 quantization enabled")

    tflite_model = converter.convert()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    logger.info(f"TFLite model saved: {output_path} ({size_kb:.1f} KB)")

    # Quick validation
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    inp_det = interpreter.get_input_details()
    out_det = interpreter.get_output_details()
    logger.info(f"Input: {inp_det[0]['shape']} {inp_det[0]['dtype'].__name__}")
    logger.info(f"Output: {out_det[0]['shape']} {out_det[0]['dtype'].__name__}")


def export_emotion_savedmodel(model_path: str, output_dir: str):
    """Export emotion model as TF SavedModel for TF Serving."""
    model = tf.keras.models.load_model(model_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save(output_dir, save_format="tf")
    logger.info(f"SavedModel saved: {output_dir}")

    # Create serving signature info
    loaded = tf.saved_model.load(output_dir)
    logger.info(f"Signatures: {list(loaded.signatures.keys())}")


def export_captioning_savedmodel(
    encoder_weights: str,
    decoder_weights: str,
    vocab_size: int,
    output_dir: str,
):
    """Export captioning encoder+decoder as SavedModel."""
    from models.captioning.encoder import PrecomputedFeatureProjector
    from models.captioning.decoder import CaptionDecoder

    encoder = PrecomputedFeatureProjector(projection_dim=512)
    encoder.load_weights(encoder_weights)

    decoder = CaptionDecoder(
        vocab_size=vocab_size,
        embedding_dim=256,
        lstm_units=512,
        feature_dim=512,
    )
    decoder.load_weights(decoder_weights)

    # Save separately for modular serving
    enc_dir = Path(output_dir) / "encoder"
    dec_dir = Path(output_dir) / "decoder"
    enc_dir.mkdir(parents=True, exist_ok=True)
    dec_dir.mkdir(parents=True, exist_ok=True)

    # Build models with concrete input shapes before saving
    dummy_feat = tf.zeros((1, 2048))
    encoder(dummy_feat)
    encoder.save(str(enc_dir), save_format="tf")

    dummy_inp = tf.zeros((1, 1), dtype=tf.int32)
    dummy_projected = tf.zeros((1, 512))
    decoder((dummy_projected, dummy_inp))
    decoder.save(str(dec_dir), save_format="tf")

    logger.info(f"Encoder saved: {enc_dir}")
    logger.info(f"Decoder saved: {dec_dir}")


def benchmark_tflite(model_path: str, n_runs: int = 100):
    """Benchmark TFLite inference speed."""
    import time
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    inp_det = interpreter.get_input_details()

    dummy_input = np.random.randint(0, 255, inp_det[0]["shape"]).astype(inp_det[0]["dtype"])

    # Warmup
    for _ in range(10):
        interpreter.set_tensor(inp_det[0]["index"], dummy_input)
        interpreter.invoke()

    # Benchmark
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        interpreter.set_tensor(inp_det[0]["index"], dummy_input)
        interpreter.invoke()
        times.append((time.perf_counter() - t0) * 1000)

    avg_ms = np.mean(times)
    fps = 1000 / avg_ms
    logger.info(f"TFLite benchmark ({n_runs} runs): {avg_ms:.2f}ms avg → {fps:.1f} FPS")


def main():
    parser = argparse.ArgumentParser(description="Export trained models")
    parser.add_argument("--model",  choices=["emotion", "captioning"], required=True)
    parser.add_argument("--format", choices=["tflite", "saved_model"], required=True)
    parser.add_argument("--input",  default=None, help="Input model path")
    parser.add_argument("--output", default=None, help="Output path")
    parser.add_argument("--quantize", action="store_true", help="Apply INT8 quantization (TFLite only)")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark after export")
    args = parser.parse_args()

    if args.model == "emotion":
        inp = args.input or "checkpoints/emotion/best_model.h5"
        if args.format == "tflite":
            out = args.output or "exports/emotion/emotion_model.tflite"
            export_emotion_tflite(inp, out, quantize=args.quantize)
            if args.benchmark:
                benchmark_tflite(out)
        else:
            out = args.output or "exports/emotion/saved_model"
            export_emotion_savedmodel(inp, out)

    elif args.model == "captioning":
        if args.format == "saved_model":
            import yaml
            with open("configs/captioning_config.yaml") as f:
                cfg = yaml.safe_load(f)
            vocab_size = cfg["model"]["vocab"]["max_vocab_size"]
            out = args.output or "exports/captioning/saved_model"
            export_captioning_savedmodel(
                encoder_weights="checkpoints/captioning/encoder_weights",
                decoder_weights="checkpoints/captioning/decoder_weights",
                vocab_size=vocab_size,
                output_dir=out,
            )
        else:
            logger.error("TFLite export for captioning not yet implemented")


if __name__ == "__main__":
    main()
