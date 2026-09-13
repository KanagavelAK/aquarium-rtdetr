"""Hugging Face Spaces entry point (Gradio SDK, ZeroGPU / CPU free tier).

One process serves both the demo page and the FastAPI app from app/main.py.

    On Spaces (SPACE_ID set): Gradio owns the server, the API is mounted under
        /api  ->  /api/detect, /api/ask, /api/health, /api/docs
    Locally (python space_app.py -> http://localhost:7860): the API is at the
        root as usual, /detect, /ask, /docs, and the page is at /.

The two layouts exist because the Spaces runner only marks a Gradio Space
live once `demo.launch()` has run, and ZeroGPU additionally requires at least
one `@spaces.GPU` function; a plain uvicorn process is shut down.
"""
import io
import os

ON_SPACES = bool(os.getenv("SPACE_ID"))
if ON_SPACES:
    # ZeroGPU emulates torch.cuda outside @spaces.GPU functions and raises if it is
    # touched; the API routes are not GPU functions, so keep inference on CPU there.
    os.environ.setdefault("DETECTOR_DEVICE", "cpu")

try:
    import spaces  # ZeroGPU runtime; must be imported before torch on Spaces
except ImportError:  # local run: make @spaces.GPU a no-op
    class spaces:  # noqa: N801
        @staticmethod
        def GPU(fn=None, **_):
            return fn if fn is not None else (lambda f: f)

import gradio as gr
import uvicorn
from PIL import Image, ImageDraw, ImageFont
from starlette.routing import Mount

from app import reasoning
from app.detector import CLASSES, DEFAULT_CONF, detector
from app.main import app as api
from app.scene import build_scene, plural

API_PREFIX = "/api" if ON_SPACES else ""
REPO = "https://github.com/KanagavelAK/aquarium-rtdetr"

# One colour per class, chosen to stay apart against blue-green water.
COLOURS = {
    "fish": (255, 196, 0), "jellyfish": (255, 105, 180), "penguin": (236, 240, 241),
    "puffin": (255, 140, 0), "shark": (46, 204, 113), "starfish": (231, 76, 60),
    "stingray": (0, 200, 255),
}
HEX = {k: "#%02x%02x%02x" % v for k, v in COLOURS.items()}
SAMPLES = [p for p in ("samples/tank_01.jpg", "samples/tank_02.jpg", "samples/tank_01.png", "samples/tank_02.png")
           if os.path.exists(p)]
QUESTIONS = ["How many fish are in this tank?", "What is the most common animal here?",
             "Are there any sharks?", "Are there more fish than jellyfish?",
             "How many different kinds of animal are there?",
             "What species of fish is that?", "What is the capital of France?"]


def _font(size):
    for name in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, "PNG")
    return buf.getvalue()


def _draw(image: Image.Image, detections) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    scale = max(out.width, out.height) / 640
    width = max(2, round(2.5 * scale))
    font = _font(max(12, round(13 * scale)))
    for d in detections:
        x1, y1, x2, y2 = d.box_xyxy
        colour = COLOURS.get(d.label, (255, 255, 255))
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=width)
        text = f"{d.label} {d.confidence:.2f}"
        tw = draw.textlength(text, font=font)
        th = font.size + 4
        ty = y1 - th if y1 - th >= 0 else y1
        draw.rectangle([x1, ty, x1 + tw + 8, ty + th], fill=colour)
        draw.text((x1 + 4, ty + 1), text, fill=(0, 0, 0), font=font)
    return out


def _pill(text, kind):
    return f'<span class="pill pill-{kind}">{text}</span>'


def _summary_html(facts):
    parts = [_pill(f"{facts.total} {plural('animal', facts.total)}", "info"),
             _pill(f"{facts.kinds} kind{'s' if facts.kinds != 1 else ''}", "muted")]
    for label, n in facts.ranking:
        parts.append(f'<span class="pill pill-class" style="--c:{HEX.get(label, "#999")}">{n} {plural(label, n)}</span>')
    if facts.duplicates_suppressed:
        parts.append(_pill(f"{facts.duplicates_suppressed} duplicate box{'es' if facts.duplicates_suppressed != 1 else ''} collapsed", "muted"))
    if facts.crowding > reasoning.CROWDING_MAX:
        parts.append(_pill(f"crowding {facts.crowding:.0%}", "warn"))
    if facts.conflicts:
        parts.append(_pill(f"{len(facts.conflicts)} label conflict{'s' if len(facts.conflicts) != 1 else ''}", "bad"))
    return '<div class="pills">' + "".join(parts) + "</div>"


def _verdict_html(decision, guard, source):
    if decision and not decision.needs_detection:
        return _pill("detector not called · " + decision.kind.replace("_", " "), "info")
    if guard is None:
        return ""
    if guard.sufficient:
        return _pill("answered · evidence sufficient", "ok") + _pill(f"phrased by {source}", "muted")
    return _pill("refused · insufficient information", "warn")


def _class_rows(facts):
    return [[cf.label, cf.count, cf.confident, cf.tentative, f"{cf.max_confidence:.2f}",
             f"{cf.crowding:.0%}", ", ".join(cf.conflicts_with) or "—"]
            for _, cf in sorted(facts.classes.items(), key=lambda kv: -kv[1].count)]


@spaces.GPU(duration=5)
def _zero_gpu_placeholder():
    """Never called. ZeroGPU refuses to start a Space with no @spaces.GPU function,
    but reserving a GPU for a CPU-pinned model only burns the daily quota, so the
    real handlers below run undecorated on CPU (~2 s/image) and cost nothing."""
    return "ok"


def run_detect(image, confidence):
    if image is None:
        return None, "", _pill("upload an image first", "warn"), [], {}
    detections, w, h, ms = detector.predict(_png_bytes(image), conf=confidence)
    facts = build_scene(detections, w, h, confidence)
    payload = {
        "image": {"width": w, "height": h},
        "confidence_threshold": confidence,
        "count": len(detections),
        "counts_by_class": {k: v for k, v in sorted(
            ((d.label, sum(1 for x in detections if x.label == d.label)) for d in detections))},
        "detections": [d.to_dict() for d in detections],
        "inference_ms": ms,
    }
    verdict = _pill(f"{len(detections)} detections · {ms:.0f} ms", "info")
    return _draw(image, detections), _summary_html(facts), verdict, _class_rows(facts), payload


def run_ask(image, question, confidence):
    """Same four steps as POST /ask, in the same order."""
    decision = reasoning.route(question or "")
    routing = {"needs_detection": decision.needs_detection, "question_kind": decision.kind,
               "targets": decision.targets, "rationale": decision.rationale, "decided_by": decision.decided_by}
    if not decision.needs_detection:
        answer = reasoning.insufficient_message(decision.kind, [])
        return (gr.update(), answer, "", _verdict_html(decision, None, None), [],
                {"answer": answer, "sufficient_information": False, "routing": routing,
                 "detector_called": False, "answer_source": "template"})
    if image is None:
        return (gr.update(), "This question needs an image, but none was uploaded.", "",
                _pill("upload an image first", "warn"), [], {"routing": routing})

    detections, w, h, ms = detector.predict(_png_bytes(image), conf=confidence)
    facts = build_scene(detections, w, h, confidence)
    guard = reasoning.guardrail(decision, facts)
    if guard.sufficient:
        answer, source = reasoning.compose(question, decision, facts)
    else:
        answer, source = reasoning.insufficient_message(decision.kind, guard.reasons), "template"
    payload = {
        "answer": answer,
        "sufficient_information": guard.sufficient,
        "routing": routing,
        "detector_called": True,
        "guardrail_reasons": guard.reasons,
        "answer_source": source,
        "evidence": {k: v for k, v in facts.to_dict().items() if k not in ("detections", "animals")},
        "inference_ms": ms,
    }
    return (_draw(image, detections), answer, _summary_html(facts),
            _verdict_html(decision, guard, source), _class_rows(facts), payload)


PILL_CSS = """
.pills { display: flex; flex-wrap: wrap; gap: 6px; padding: 4px 0; }
.pill { display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 0.82rem;
        font-weight: 600; line-height: 1.5; border: 1px solid transparent; margin-right: 6px;
        font-family: var(--font, inherit); }
.pill-ok    { background: #e8f7ee; color: #14663b; border-color: #b9e6cb; }
.pill-bad   { background: #fdecea; color: #a12a1f; border-color: #f5c2bc; }
.pill-warn  { background: #fff4d6; color: #7a5200; border-color: #f2d78a; }
.pill-info  { background: #e6f4f8; color: #145a73; border-color: #b7dfe9; }
.pill-muted { background: #eef0f3; color: #4b5563; border-color: #d5d9e0; }
.pill-class { background: color-mix(in srgb, var(--c) 18%, white); color: #1f2933;
              border-color: color-mix(in srgb, var(--c) 55%, white); }
@media (prefers-color-scheme: dark) {
  .pill-ok    { background: #123322; color: #7ddba3; border-color: #1f5236; }
  .pill-bad   { background: #3a1b18; color: #f39c8f; border-color: #5c2a24; }
  .pill-warn  { background: #3a2f10; color: #f2cd6a; border-color: #5c4a1a; }
  .pill-info  { background: #0f2a35; color: #8fd0e8; border-color: #1c4a5c; }
  .pill-muted { background: #262b33; color: #aab2bf; border-color: #3a414c; }
  .pill-class { background: color-mix(in srgb, var(--c) 22%, #1a1f26); color: #e6eaf0;
                border-color: color-mix(in srgb, var(--c) 45%, #1a1f26); }
}
"""

LEGEND_CSS = """
.legend { font-size: 0.85rem; opacity: 0.85; display: flex; flex-wrap: wrap; gap: 12px; align-items: center; padding: 2px 0 6px; }
.sw { display: inline-block; width: 11px; height: 11px; border-radius: 3px; margin-right: 5px; vertical-align: -1px; }
"""

HERO_CSS = """
.hero { padding: 4px 0 6px; }
.hero h1 { font-size: 1.7rem; margin: 0 0 4px; letter-spacing: -0.01em; }
.hero p { margin: 0; opacity: 0.8; line-height: 1.5; }
.hero a { color: #0e7c9c; text-decoration: none; font-weight: 600; }
"""

CSS = """
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; }
#answer textarea { font-size: 1.05rem; line-height: 1.5; }
footer { display: none !important; }
"""

THEME = gr.themes.Soft(primary_hue="cyan", neutral_hue="slate", radius_size="md")
LEGEND = '<div class="legend">' + "".join(
    f'<span><i class="sw" style="background:{HEX[c]}"></i>{c}</span>' for c in CLASSES) + "</div>"

with gr.Blocks(title="Aquarium Animal Census", theme=THEME, css=CSS) as demo:
    gr.HTML(css_template=HERO_CSS, value=f"""
    <div class="hero">
      <h1>Aquarium Animal Census</h1>
      <p>RT-DETR fine-tuned for <b>fish, jellyfish, penguin, puffin, shark, starfish</b> and <b>stingray</b>,
         with a hand-written reasoning layer that counts, ranks and compares from the detections and
         <b>refuses when schooling, label conflicts or weak boxes make the answer unreliable</b>.
         &nbsp;·&nbsp; <a href="{API_PREFIX}/docs" target="_blank">API docs</a>
         &nbsp;·&nbsp; <a href="{REPO}" target="_blank">source</a></p>
    </div>""")

    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            image = gr.Image(type="pil", label="Aquarium image", height=360)
            if SAMPLES:
                gr.Examples(SAMPLES, inputs=image, label="Held-out test images (never trained on)",
                            examples_per_page=4)
            confidence = gr.Slider(0.05, 0.9, value=DEFAULT_CONF, step=0.05,
                                   label="Confidence threshold",
                                   info="boxes below this are dropped; the guardrail's assertion floor is 0.45")
            detect_btn = gr.Button("Detect animals", variant="secondary")
            question = gr.Textbox(label="Ask a question about the image", value=QUESTIONS[0], lines=2,
                                  max_lines=3, placeholder="e.g. How many sharks are in this tank?")
            gr.Examples([[q] for q in QUESTIONS], inputs=question, label="Try these",
                        examples_per_page=7)
            ask_btn = gr.Button("Ask", variant="primary")

        with gr.Column(scale=6):
            annotated = gr.Image(label="Detections", height=360, interactive=False)
            gr.HTML(css_template=LEGEND_CSS, value=LEGEND)
            verdict = gr.HTML(css_template=PILL_CSS)
            answer = gr.Textbox(label="Answer", lines=3, max_lines=5, interactive=False, elem_id="answer")
            summary = gr.HTML(css_template=PILL_CSS)
            with gr.Accordion("Per-class evidence", open=False):
                classes_df = gr.Dataframe(
                    headers=["class", "count", "confident", "tentative", "max conf", "crowding", "conflicts with"],
                    datatype=["str", "number", "number", "number", "str", "str", "str"],
                    interactive=False, wrap=True)
            with gr.Accordion("Raw response (same fields as the API)", open=False):
                details = gr.JSON()

    gr.HTML(css_template=".foot { opacity: 0.7; font-size: 0.85rem; text-align: center; padding-top: 4px; }",
            value=f'<div class="foot">Same model and code as <code>POST {API_PREFIX}/detect</code> and '
                  f'<code>POST {API_PREFIX}/ask</code>. Everything here runs on CPU, about 2 s per image.</div>')

    detect_btn.click(run_detect, [image, confidence], [annotated, summary, verdict, classes_df, details])
    ask_btn.click(run_ask, [image, question, confidence], [annotated, answer, summary, verdict, classes_df, details])
    question.submit(run_ask, [image, question, confidence], [annotated, answer, summary, verdict, classes_df, details])


if ON_SPACES:
    from gradio.routes import App

    detector.load()   # the mounted sub-app's startup hook does not run under Gradio's server

    _create_app = App.create_app

    def _create_app_with_api(blocks, *args, **kwargs):
        gradio_app = _create_app(blocks, *args, **kwargs)
        # Gradio ends its route table with a catch-all, so the API mount must go first.
        gradio_app.router.routes.insert(0, Mount("/api", app=api))
        return gradio_app

    App.create_app = staticmethod(_create_app_with_api)
    demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)   # blocks, as the runner expects
else:
    app = gr.mount_gradio_app(api, demo, path="/", ssr_mode=False)
    if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
