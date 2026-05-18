from pathlib import Path

from inherent import HEAD_ORDER
from inherent.export.registry import list_backends


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
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
    root_index = (DOCS_DIR / "index.html").read_text()
    legacy_index = (DEMO_DIR / "index.html").read_text()
    app = (DEMO_DIR / "app.js").read_text()

    assert 'url=../' in legacy_index
    assert 'href="../"' in legacy_index
    assert "onnxruntime-web" in root_index
    assert "assets/inherent.onnx" in root_index
    assert "./browser-demo/assets/theme-reference.svg" in root_index
    assert 'id="scope"' in root_index
    assert 'id="flowHeads"' in root_index
    assert 'id="startRecording" type="button" disabled>Record</button>' in root_index
    assert "Loading bundled model" in root_index
    assert "Or local ONNX model" not in root_index
    assert "Load ONNX model" not in root_index
    assert "webgpu" in app
    assert "wasm" in app
    assert "mel_spectrogram" in app
    assert "intent_output" in app
    assert "DEFAULT_MODEL_URL" in app
    assert './assets/inherent.onnx' in app
    assert './assets/inherent.onnx.metadata.json' in app
    assert "recentAudioWindow" in app
    assert "browserPaddingValue" in app
    assert "drawScope" in app
    assert "renderFlowHeads" in app
    assert (DOCS_DIR / "assets" / "inherent.onnx").is_file()
    assert (DOCS_DIR / "assets" / "inherent.onnx.metadata.json").is_file()


def test_browser_demo_embeds_runtime_head_order():
    app = (DEMO_DIR / "app.js").read_text()

    for head in HEAD_ORDER:
        assert head in app
