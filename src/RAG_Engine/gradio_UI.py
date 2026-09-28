import sys
import os
import json
import base64
import time
from collections import Counter
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))

import gradio as gr
from rag_pipeline import (
    route_documents, group_by_doc_id, documents_from_routed,
    build_index, load_model, query
)
from llama_index.core.schema import Document

_IMG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "Tom&Jerry.jpg"
)
with open(_IMG_PATH, "rb") as _f:
    _TJ_B64 = base64.b64encode(_f.read()).decode()

_QJ_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "question_jerry.jpg")
with open(_QJ_PATH, "rb") as _f:
    _QJ_B64 = base64.b64encode(_f.read()).decode()

# T&J palette: sky blue bg, warm dark panels, amber accent, Jerry-brown user bubble
CSS = """
body { margin: 0; background: #5AA4CF; }

.gradio-container {
    background: #5AA4CF !important;
    max-width: 100% !important;
    padding: 0 24px 32px !important;
    font-family: system-ui, -apple-system, sans-serif !important;
}

.contain, .gap { background: transparent !important; }
.block, .form {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* ── LEFT PANEL ── */
#upload-col > .block,
#upload-col > div > .block {
    background: #1C1008 !important;
    border-radius: 16px !important;
    padding: 18px !important;
    margin-bottom: 10px !important;
}
#upload-col label > span,
#upload-col .label-wrap > span {
    color: #D4892A !important;
    font-weight: 700 !important;
    font-size: 11px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
#file-upload .wrap,
#file-upload .upload-container {
    background: #2A1A0C !important;
    border: 2px dashed #D4892A !important;
    border-radius: 12px !important;
    color: #D4892A !important;
}
#file-upload svg { stroke: #D4892A !important; }
#file-upload span { color: #aaa !important; }

#mode-drop .wrap, #mode-drop input {
    background: #2A1A0C !important;
    border: 2px solid #3A2810 !important;
    border-radius: 10px !important;
    color: #F5F2EC !important;
}

#status-html {
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* ── RIGHT PANEL ── */
#chat-col > .block,
#chat-col > div > .block {
    background: #1C1008 !important;
    border-radius: 16px !important;
    padding: 16px !important;
    margin-bottom: 10px !important;
}
#chatbot-box {
    background: #261508 !important;
    border-radius: 12px !important;
    border: none !important;
}

/* User bubble — Jerry brown */
.message.user > div, .user > .message-bubble-border > div {
    background: #9B6535 !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    border-radius: 16px 16px 4px 16px !important;
    border: none !important;
}
/* Bot bubble — warm dark */
.message.bot > div, .bot > .message-bubble-border > div {
    background: #1C1008 !important;
    color: #F5F2EC !important;
    border-radius: 16px 16px 16px 4px !important;
    border: 1px solid #3A2810 !important;
}

#user-input textarea {
    background: #F5F2EC !important;
    border: 3px solid #1C1008 !important;
    border-radius: 12px !important;
    font-size: 15px !important;
    color: #1C1008 !important;
    resize: none !important;
    padding: 12px !important;
}
#user-input textarea::placeholder { color: #999 !important; }
#user-input label > span { display: none !important; }

/* ── BUTTONS ── */
#send-btn {
    background: #9B6535 !important;
    color: #FFFFFF !important;
    border: 3px solid #1C1008 !important;
    border-radius: 12px !important;
    font-family: 'Arial Black', sans-serif !important;
    font-weight: 900 !important;
    font-size: 15px !important;
    transition: background 0.15s, color 0.15s !important;
}
#send-btn:hover {
    background: #D4892A !important;
    color: #1C1008 !important;
}
#clear-btn {
    background: #2A1A0C !important;
    color: #F5F2EC !important;
    border: 2px solid #3A2810 !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
}
#clear-btn:hover { border-color: #D4892A !important; }

#save-btn {
    background: #2A1A0C !important;
    color: #D4892A !important;
    border: 2px solid #3A2810 !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    width: 100% !important;
}
#save-btn:hover { border-color: #D4892A !important; }

#download-file {
    background: #2A1A0C !important;
    border-radius: 10px !important;
}
#download-file label > span { color: #D4892A !important; }

/* Settings accordion */
#settings-panel > .label-wrap {
    background: #2A1A0C !important;
    border-radius: 10px !important;
    color: #D4892A !important;
    font-weight: 700 !important;
    padding: 8px 12px !important;
}
#settings-panel > .label-wrap span { color: #D4892A !important; }
#settings-panel .block {
    background: #1C1008 !important;
    border-radius: 0 0 10px 10px !important;
    padding: 12px !important;
}
#type-filter label > span {
    color: #D4892A !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
#type-filter .wrap, #type-filter input {
    background: #2A1A0C !important;
    border: 2px solid #3A2810 !important;
    border-radius: 8px !important;
    color: #F5F2EC !important;
}
#whimsy-slider label > span {
    color: #D4892A !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
#whimsy-slider input[type=range] {
    accent-color: #D4892A !important;
}
#whimsy-slider .output-number input {
    background: #2A1A0C !important;
    border: 2px solid #3A2810 !important;
    border-radius: 6px !important;
    color: #F5F2EC !important;
}
"""

HEADER_HTML = f"""
<div style="
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 28px 16px 16px;
    background: #5AA4CF;
">
  <img
    src="data:image/jpeg;base64,{_TJ_B64}"
    style="height:150px; border-radius:20px; border:4px solid #1C1008; object-fit:cover;"
    alt="Tom and Jerry"
  />
  <div style="text-align:center; flex:1;">
    <div style="
      font-family: 'Arial Black', Impact, sans-serif;
      font-size: 68px;
      font-weight: 900;
      color: #1C1008;
      letter-spacing: -3px;
      line-height: 1;
      text-transform: uppercase;
    ">DOC Q&amp;A</div>
    <div style="
      font-size: 13px;
      color: #1C1008;
      opacity: 0.6;
      margin-top: 6px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    ">Upload any document &mdash; ask anything</div>
  </div>
  <img
    src="data:image/jpeg;base64,{_QJ_B64}"
    style="height:150px; border-radius:20px; border:4px solid #1C1008; object-fit:cover;"
    alt="Question Jerry"
  />
</div>
"""

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PDF_EXTENSIONS   = {".pdf"}


def _process_image(file_path: str) -> list[dict]:
    """OCR a standalone image and return a single synthetic routed page."""
    import numpy as np
    from PIL import Image
    from Data_Extract.OCR_comparisons import PaddleOCRExtractor

    ocr = PaddleOCRExtractor()
    img = Image.open(file_path).convert("RGB")  # strip alpha channel (RGBA crashes PaddleOCR)
    spans = ocr.extract(np.array(img))
    text = " ".join(s["text"] for s in spans if s.get("text"))
    if not text.strip():
        raise ValueError("OCR found no text in image — image may be blank or too low resolution")
    return [{
        "page_num":    0,
        "source_file": os.path.basename(file_path),
        "text":        text,
        "page_type":   "image",
        "doc_id":      0,
    }]


def _build_status_html(success_names, total_pages, n_docs, total_chunks,
                       doc_types, elapsed, grouped, chunk_counts, errors):
    s = '<div style="background:#1C1008;border-radius:12px;padding:16px;color:#F5F2EC;font-family:system-ui;font-size:13px;line-height:1.8;">'
    s += '<div style="font-size:15px;font-weight:700;color:#D4892A;margin-bottom:10px;">📋 Document Info</div>'

    if success_names:
        s += '<div style="color:#7BC67E;font-weight:700;margin-bottom:6px;">✅ Successfully Processed:</div>'
        for name in success_names:
            s += f'<div style="margin-left:12px;">○ 📄 File: {name}</div>'
        s += f'<div style="margin-left:12px;">○ 📑 Pages: {total_pages}</div>'
        s += f'<div style="margin-left:12px;">○ 📚 Documents Found: {n_docs}</div>'
        s += f'<div style="margin-left:12px;">○ ✂️ Chunks Created: {total_chunks}</div>'
        type_labels = ", ".join(t.replace("_", " ").title() for t in sorted(doc_types))
        s += f'<div style="margin-left:12px;">○ 🏷️ Types: {type_labels}</div>'
        s += f'<div style="margin-left:12px;">○ ⏱️ Time: {elapsed:.1f}s</div>'

        if grouped:
            s += '<hr style="border:none;border-top:1px solid #3A2810;margin:10px 0;"/>'
            parts = []
            for group in grouped:
                doc_id  = group["doc_id"]
                chunks  = chunk_counts.get(doc_id, 0)
                label   = group["page_type"].replace("_", " ").title()
                p_start = group.get("page_start", 0) + 1
                p_end   = group.get("page_end",   0) + 1
                parts.append(f'· <b>{label}</b> (Pages {p_start}–{p_end}): {chunks} chunk{"s" if chunks != 1 else ""}')
            s += '<div style="color:#aaa;font-size:12px;">' + " ".join(parts) + "</div>"

    if errors:
        s += '<div style="color:#FF6B6B;font-weight:700;margin-top:10px;">❌ Failed:</div>'
        for err in errors:
            s += f'<div style="margin-left:12px;color:#FF9999;">○ {err}</div>'

    s += "</div>"
    return s


def process_files(files, mode):
    """Process each file independently. Returns rich HTML status + pipeline state + doc types."""
    _WAITING = '<div style="background:#1C1008;border-radius:12px;padding:14px;color:#aaa;font-size:13px;">Waiting for upload...</div>'
    if not files:
        return _WAITING, None, None, []

    t_start    = time.time()
    all_routed = []
    errors     = []

    for file in files:
        ext  = os.path.splitext(file.name)[1].lower()
        name = os.path.basename(file.name)
        try:
            if ext in PDF_EXTENSIONS:
                use_ocr = (mode == "Img Mode")
                all_routed.extend(route_documents(file.name, use_ocr=use_ocr))
            elif ext in IMAGE_EXTENSIONS:
                all_routed.extend(_process_image(file.name))
            else:
                errors.append(f"{name}: unsupported type ({ext})")
        except Exception as e:
            errors.append(f"{name}: {e}")

    if not all_routed:
        html = _build_status_html([], 0, 0, 0, [], 0, [], {}, errors)
        return html, None, None, []

    grouped = group_by_doc_id(all_routed)
    docs    = documents_from_routed(grouped)

    try:
        index = build_index(docs)
    except ValueError as e:
        html = _build_status_html([], 0, 0, 0, [], 0, [], {}, [str(e)])
        return html, None, None, []

    elapsed       = time.time() - t_start
    doc_types     = list({p["page_type"] for p in all_routed})
    success_names = list(dict.fromkeys(p["source_file"] for p in all_routed))
    total_pages   = len(all_routed)
    n_docs        = len(grouped)
    chunk_counts  = Counter(
        node.metadata.get("doc_id")
        for node in index.docstore.docs.values()
    )
    total_chunks  = sum(chunk_counts.values())

    html = _build_status_html(
        success_names, total_pages, n_docs, total_chunks,
        doc_types, elapsed, grouped, chunk_counts, errors
    )
    return html, index, all_routed, doc_types


def chat(message, history, index_state, routed_state, doc_type_filter, whimsy):
    if not message.strip():
        return history, ""
    if index_state is None:
        history = history + [
            {"role": "user",      "content": message},
            {"role": "assistant", "content": "Upload a document first — I'll process it automatically."},
        ]
        return history, ""
    try:
        forced = doc_type_filter if doc_type_filter and doc_type_filter != "Auto (predict)" else None
        answer = query(index_state, message, routed_pages=routed_state,
                       forced_doc_type=forced, temperature=float(whimsy))
    except Exception as e:
        answer = f"Error: {e}"

    history = history + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": answer},
    ]
    return history, ""


def save_chat(history):
    if not history:
        return gr.update(visible=False)
    save_path = "/tmp/chat_history.json"
    with open(save_path, "w") as f:
        json.dump(history, f, indent=2)
    return gr.update(value=save_path, visible=True)


with gr.Blocks(title="Doc Q&A", css=CSS) as demo:

    index_state      = gr.State(None)
    routed_state     = gr.State(None)
    doc_types_state  = gr.State([])

    gr.HTML(HEADER_HTML)

    with gr.Row(equal_height=False):

        with gr.Column(scale=1, elem_id="upload-col"):
            file_input = gr.File(
                label="Upload Documents",
                file_count="multiple",
                file_types=[".pdf", ".png", ".jpg", ".jpeg"],
                elem_id="file-upload",
            )
            mode_dropdown = gr.Dropdown(
                choices=["Standard Mode", "Img Mode"],
                value="Standard Mode",
                label="Upload Type",
                elem_id="mode-drop",
            )
            status_html = gr.HTML(
                value='<div style="background:#1C1008;border-radius:12px;padding:14px;color:#aaa;font-size:13px;">Waiting for upload...</div>',
                elem_id="status-html",
            )
            with gr.Accordion("⚙️ Settings", open=False, elem_id="settings-panel"):
                doc_type_filter = gr.Dropdown(
                    choices=["Auto (predict)"],
                    value="Auto (predict)",
                    label="Filter by Document Type",
                    elem_id="type-filter",
                )
                whimsy_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.7,
                    step=0.05,
                    label="Whimsy",
                    elem_id="whimsy-slider",
                )
            save_btn = gr.Button("Save Chat History", elem_id="save-btn")
            download = gr.File(label="Download", visible=False, elem_id="download-file")

        with gr.Column(scale=2, elem_id="chat-col"):
            chatbot = gr.Chatbot(label="", height=440, elem_id="chatbot-box")
            user_input = gr.Textbox(
                placeholder="Ask anything about your document...",
                label="",
                lines=2,
                elem_id="user-input",
            )
            with gr.Row():
                send_btn  = gr.Button("→ Send", variant="primary", elem_id="send-btn")
                clear_btn = gr.Button("Clear", elem_id="clear-btn")

    def _update_filter(doc_types):
        choices = ["Auto (predict)"] + sorted(doc_types)
        return gr.update(choices=choices, value="Auto (predict)")

    file_input.change(
        fn=process_files,
        inputs=[file_input, mode_dropdown],
        outputs=[status_html, index_state, routed_state, doc_types_state],
    ).then(
        fn=_update_filter,
        inputs=[doc_types_state],
        outputs=[doc_type_filter],
    )
    mode_dropdown.change(
        fn=process_files,
        inputs=[file_input, mode_dropdown],
        outputs=[status_html, index_state, routed_state, doc_types_state],
    ).then(
        fn=_update_filter,
        inputs=[doc_types_state],
        outputs=[doc_type_filter],
    )
    send_btn.click(
        fn=chat,
        inputs=[user_input, chatbot, index_state, routed_state, doc_type_filter, whimsy_slider],
        outputs=[chatbot, user_input],
    )
    user_input.submit(
        fn=chat,
        inputs=[user_input, chatbot, index_state, routed_state, doc_type_filter, whimsy_slider],
        outputs=[chatbot, user_input],
    )
    clear_btn.click(fn=lambda: [], outputs=[chatbot])
    save_btn.click(
        fn=save_chat,
        inputs=[chatbot],
        outputs=[download],
    )


if __name__ == "__main__":
    demo.launch()
