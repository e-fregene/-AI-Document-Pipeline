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

_IMG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "Tom&Jerry.jpg"
)
with open(_IMG_PATH, "rb") as _f:
    _TJ_B64 = base64.b64encode(_f.read()).decode()

HEADER_HTML = f"""
<div style="
    text-align: center;
    padding: 28px 0 16px;
    background: #CAFF00;
">
  <img
    src="data:image/jpeg;base64,{_TJ_B64}"
    style="height:150px; border-radius:20px; border:4px solid #0A0A0A; object-fit:cover;"
    alt="mascot"
  />
  <div style="
    font-family: 'Arial Black', Impact, sans-serif;
    font-size: 68px;
    font-weight: 900;
    color: #0A0A0A;
    letter-spacing: -3px;
    line-height: 1;
    margin-top: 10px;
    text-transform: uppercase;
  ">DOC Q&amp;A</div>
  <div style="
    font-size: 13px;
    color: #0A0A0A;
    opacity: 0.55;
    margin-top: 6px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  ">Upload any document &mdash; ask anything</div>
</div>
"""

CSS = """
/* ── GLOBAL ── */
body { margin: 0; background: #CAFF00; }

.gradio-container {
    background: #CAFF00 !important;
    max-width: 100% !important;
    padding: 0 24px 32px !important;
    font-family: system-ui, -apple-system, sans-serif !important;
}

/* Strip Gradio's default white cards from layout wrappers */
.contain, .gap { background: transparent !important; }

.block, .form {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* ── LEFT PANEL ── */
#upload-col > .block,
#upload-col > div > .block {
    background: #0A0A0A !important;
    border-radius: 16px !important;
    padding: 18px !important;
    margin-bottom: 10px !important;
}

#upload-col label > span,
#upload-col .label-wrap > span {
    color: #CAFF00 !important;
    font-weight: 700 !important;
    font-size: 11px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}

/* File drop zone */
#file-upload .wrap,
#file-upload .upload-container {
    background: #181818 !important;
    border: 2px dashed #CAFF00 !important;
    border-radius: 12px !important;
    color: #CAFF00 !important;
}
#file-upload svg { stroke: #CAFF00 !important; }
#file-upload span { color: #aaa !important; }

/* Mode dropdown */
#mode-drop .wrap,
#mode-drop input {
    background: #181818 !important;
    border: 2px solid #333 !important;
    border-radius: 10px !important;
    color: #fff !important;
}

/* Status */
#status-box textarea {
    background: #CAFF00 !important;
    border: 2px solid #0A0A0A !important;
    border-radius: 10px !important;
    color: #0A0A0A !important;
    font-weight: 700 !important;
    font-size: 12px !important;
    min-height: 36px !important;
}
#status-box label > span { display: none !important; }

/* ── RIGHT PANEL ── */
#chat-col > .block,
#chat-col > div > .block {
    background: #0A0A0A !important;
    border-radius: 16px !important;
    padding: 16px !important;
    margin-bottom: 10px !important;
}

/* Chatbot scroll area */
#chatbot-box {
    background: #141414 !important;
    border-radius: 12px !important;
    border: none !important;
}
#chatbot-box .bubble-wrap { padding: 12px !important; }

/* User bubble — lime */
.message.user > div, .user > .message-bubble-border > div {
    background: #CAFF00 !important;
    color: #0A0A0A !important;
    font-weight: 600 !important;
    border-radius: 16px 16px 4px 16px !important;
    border: none !important;
}
/* Bot bubble — dark */
.message.bot > div, .bot > .message-bubble-border > div {
    background: #232323 !important;
    color: #f0f0f0 !important;
    border-radius: 16px 16px 16px 4px !important;
    border: none !important;
}

/* Question input */
#user-input textarea {
    background: #fff !important;
    border: 3px solid #0A0A0A !important;
    border-radius: 12px !important;
    font-size: 15px !important;
    color: #0A0A0A !important;
    resize: none !important;
    padding: 12px !important;
}
#user-input textarea::placeholder { color: #999 !important; }
#user-input label > span { display: none !important; }

/* ── BUTTONS ── */
#send-btn {
    background: #CAFF00 !important;
    color: #0A0A0A !important;
    border: 3px solid #0A0A0A !important;
    border-radius: 12px !important;
    font-family: 'Arial Black', sans-serif !important;
    font-weight: 900 !important;
    font-size: 15px !important;
    letter-spacing: 0.04em !important;
    transition: background 0.15s, color 0.15s !important;
}
#send-btn:hover {
    background: #0A0A0A !important;
    color: #CAFF00 !important;
}

#clear-btn {
    background: #222 !important;
    color: #CAFF00 !important;
    border: 2px solid #333 !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
}
#clear-btn:hover { border-color: #CAFF00 !important; }

#save-btn {
    background: #181818 !important;
    color: #CAFF00 !important;
    border: 2px solid #333 !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    width: 100% !important;
}
#save-btn:hover { border-color: #CAFF00 !important; }

#download-file {
    background: #181818 !important;
    border-radius: 10px !important;
    color: #CAFF00 !important;
}
#download-file label > span { color: #CAFF00 !important; }
"""


# ── BACKEND ──────────────────────────────────────────────────────────────────

def process_files(files, mode):
    if not files:
        return "Waiting for upload...", None, None

    all_routed = []
    for file in files:
        try:
            routed = route_documents(file.name)
            all_routed.extend(routed)
        except Exception as e:
            return f"Error: {os.path.basename(file.name)} — {e}", None, None

    grouped = group_by_doc_id(all_routed)
    docs    = documents_from_routed(grouped)

    try:
        index = build_index(docs)
    except ValueError as e:
        return str(e), None, None

    doc_types = list({p["page_type"] for p in all_routed})
    status = f"Ready — {len(docs)} doc(s) | {', '.join(doc_types)}"
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


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Doc Q&A", css=CSS) as demo:

    index_state  = gr.State(None)
    routed_state = gr.State(None)

    gr.HTML(HEADER_HTML)

    with gr.Row(equal_height=False):

        # Left — upload controls
        with gr.Column(scale=1, elem_id="upload-col"):
            file_input = gr.File(
                label="Upload Documents",
                file_count="multiple",
                file_types=[".pdf", ".png", ".jpg", ".jpeg", ".mp3", ".wav"],
                elem_id="file-upload",
            )
            mode_dropdown = gr.Dropdown(
                choices=["Standard Mode", "OCR Mode"],
                value="Standard Mode",
                label="Mode",
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

        # Right — chat
        with gr.Column(scale=2, elem_id="chat-col"):
            chatbot = gr.Chatbot(
                label="",
                height=440,
                elem_id="chatbot-box",
            )
            user_input = gr.Textbox(
                placeholder="Ask anything about your document...",
                label="",
                lines=2,
                elem_id="user-input",
            )
            with gr.Row():
                send_btn  = gr.Button("→ Send", variant="primary", elem_id="send-btn")
                clear_btn = gr.Button("Clear", elem_id="clear-btn")

    # ── Event wiring ──
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
