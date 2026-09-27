import sys
import os
import json
import base64
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

#status-box textarea {
    background: #F5F2EC !important;
    border: 2px solid #1C1008 !important;
    border-radius: 10px !important;
    color: #1C1008 !important;
    font-weight: 700 !important;
    font-size: 12px !important;
    min-height: 36px !important;
}
#status-box label > span { display: none !important; }

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
    img = np.array(Image.open(file_path))
    spans = ocr.extract(img)
    text = " ".join(s["text"] for s in spans)
    return [{
        "page_num":    0,
        "source_file": os.path.basename(file_path),
        "text":        text,
        "page_type":   "image",
        "doc_id":      0,
    }]


def process_files(files, mode):
    """Process each file independently — PDFs via pipeline, images via OCR.
    Failures on individual files are reported without stopping other files."""
    if not files:
        return "Waiting for upload...", None, None

    all_routed = []
    errors     = []

    for file in files:
        ext  = os.path.splitext(file.name)[1].lower()
        name = os.path.basename(file.name)
        try:
            if ext in PDF_EXTENSIONS:
                use_ocr = mode == "Img Mode"
                routed  = route_documents(file.name)
                all_routed.extend(routed)
            elif ext in IMAGE_EXTENSIONS:
                routed = _process_image(file.name)
                all_routed.extend(routed)
            else:
                errors.append(f"{name}: unsupported type ({ext})")
        except Exception as e:
            errors.append(f"{name}: {e}")

    if not all_routed:
        error_msg = " | ".join(errors) if errors else "No processable files."
        return error_msg, None, None

    grouped = group_by_doc_id(all_routed)
    docs    = documents_from_routed(grouped)

    try:
        index = build_index(docs)
    except ValueError as e:
        return str(e), None, None

    doc_types  = list({p["page_type"] for p in all_routed})
    status     = f"Ready — {len(docs)} doc(s) | {', '.join(doc_types)}"
    if errors:
        status += f" | Skipped: {'; '.join(errors)}"
    return status, index, all_routed


def chat(message, history, index_state, routed_state):
    if not message.strip():
        return history, ""
    if index_state is None:
        history = history + [
            {"role": "user",      "content": message},
            {"role": "assistant", "content": "Upload a document first — I'll process it automatically."},
        ]
        return history, ""
    try:
        answer = query(index_state, message, routed_pages=routed_state)
    except Exception as e:
        answer = f"Error: {e}"

    history = history + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": answer},
    ]
    return history, ""


def save_chat(history):
    if not history:
        return None
    save_path = "/tmp/chat_history.json"
    with open(save_path, "w") as f:
        json.dump(history, f, indent=2)
    return save_path


with gr.Blocks(title="Doc Q&A", css=CSS) as demo:

    index_state  = gr.State(None)
    routed_state = gr.State(None)

    gr.HTML(HEADER_HTML)

    with gr.Row(equal_height=False):

        with gr.Column(scale=1, elem_id="upload-col"):
            file_input = gr.File(
                label="Upload Documents",
                file_count="multiple",
                file_types=[".pdf", ".png", ".jpg", ".jpeg", ".mp3", ".wav"],
                elem_id="file-upload",
            )
            mode_dropdown = gr.Dropdown(
                choices=["Standard Mode", "Img Mode"],
                value="Standard Mode",
                label="Upload Type",
                elem_id="mode-drop",
            )
            status_box = gr.Textbox(
                value="Waiting for upload...",
                interactive=False,
                label="Status",
                elem_id="status-box",
                lines=1,
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

    file_input.change(
        fn=process_files,
        inputs=[file_input, mode_dropdown],
        outputs=[status_box, index_state, routed_state],
    )
    mode_dropdown.change(
        fn=process_files,
        inputs=[file_input, mode_dropdown],
        outputs=[status_box, index_state, routed_state],
    )
    send_btn.click(
        fn=chat,
        inputs=[user_input, chatbot, index_state, routed_state],
        outputs=[chatbot, user_input],
    )
    user_input.submit(
        fn=chat,
        inputs=[user_input, chatbot, index_state, routed_state],
        outputs=[chatbot, user_input],
    )
    clear_btn.click(fn=lambda: [], outputs=[chatbot])
    save_btn.click(
        fn=save_chat,
        inputs=[chatbot],
        outputs=[download],
    ).then(fn=lambda: gr.update(visible=True), outputs=[download])


if __name__ == "__main__":
    demo.launch()
