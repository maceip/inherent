from pathlib import Path

from inherent import HEAD_ORDER
from inherent.export.registry import list_backends


ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "docs" / "browser-demo"


def test_browser_demo_documents_supported_backend_matrix():
    readme = (DEMO_DIR / "README.md").read_text()

    assert "only ONNX is" in readme
    assert "`onnx`" in readme
    assert "`tflite` / `litert`" in readme
    assert "`litertlm`" in readme
    assert "`mlx`" in readme
    assert set(list_backends()) == {"litert", "litertlm", "mlx", "onnx", "tflite"}


def test_browser_demo_loads_onnx_runtime_web_and_default_assets():
    index = (DEMO_DIR / "index.html").read_text()
    app = (DEMO_DIR / "app.js").read_text()

    assert "onnxruntime-web" in index
    assert "./assets/inherent.onnx" in index
    assert "./assets/inherent.onnx.metadata.json" in index
    assert "./assets/theme-reference.svg" in index
    assert 'id="scope"' in index
    assert 'id="flowHeads"' in index
    assert 'id="startRecording" type="button">Start recording</button>' in index
    assert "Start recording to preview the mic oscilloscope" in index
    assert "webgpu" in app
    assert "wasm" in app
    assert "mel_spectrogram" in app
    assert "intent_output" in app
    assert "drawScope" in app
    assert "renderFlowHeads" in app
    assert "Recording mic preview" in app


def test_browser_demo_embeds_runtime_head_order():
    app = (DEMO_DIR / "app.js").read_text()

    for head in HEAD_ORDER:
        assert head in app
