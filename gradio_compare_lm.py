import argparse
import gc
import os

import gradio as gr
import sentencepiece as spm
import torch

from inference_demo import decode_tokens, load_model


MODELS = {
    "lm_10m": "checkpoints/lm_10m/best.pt",
    "lm_20m": "checkpoints/lm_20m/best.pt",
    "lm_30m": "checkpoints/lm_30m/best.pt",
    "lm_40m": "checkpoints/lm_40m/best.pt",
}

EXAMPLES = [
    ["lm_40m", "The camera app can", 40],
    ["lm_40m", "To save battery on a smartphone", 50],
    ["lm_30m", "A maps app uses location", 50],
    ["lm_20m", "The best way to organize photos is", 50],
    ["lm_10m", "A mobile payment app should", 50],
]


def filter_logits(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    logits = logits / max(float(temperature), 1e-4)
    if int(top_k) > 0:
        keep = min(int(top_k), logits.numel())
        threshold = torch.topk(logits, keep).values[-1]
        logits = logits.masked_fill(logits < threshold, -float("inf"))
    if float(top_p) < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = cumulative > float(top_p)
        remove[0] = False
        logits = logits.scatter(0, sorted_idx[remove], -float("inf"))
    return logits


def build_app(tokenizer_path: str):
    state = {"name": None, "device": None, "sp": None, "model": None}

    def get_runtime(model_name: str):
        if state["model"] is not None and state["name"] != model_name:
            del state["model"]
            state["model"] = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if state["model"] is None:
            path = MODELS[model_name]
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            sp = state["sp"] or spm.SentencePieceProcessor(model_file=tokenizer_path)
            model = load_model(path, device)
            state.update({"name": model_name, "device": device, "sp": sp, "model": model})
        return state["device"], state["sp"], state["model"]

    @torch.no_grad()
    def continue_lm(
        model_name: str,
        text: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
    ) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        device, sp, model = get_runtime(model_name)
        ids = [2] + sp.encode(text, out_type=int)
        generated = []
        for _ in range(int(max_new_tokens)):
            inp = torch.tensor([ids[-256:]], dtype=torch.long, device=device)
            logits, _ = model(inp)
            next_logits = logits[0, -1].float()
            if float(repetition_penalty) > 1.0:
                for token_id in set(ids[-128:] + generated):
                    if next_logits[token_id] < 0:
                        next_logits[token_id] *= float(repetition_penalty)
                    else:
                        next_logits[token_id] /= float(repetition_penalty)
            next_logits = filter_logits(next_logits, temperature, top_k, top_p)
            probs = torch.softmax(next_logits, dim=-1)
            if not torch.isfinite(probs).all() or float(probs.sum()) <= 0:
                probs = torch.softmax(logits[0, -1].float(), dim=-1)
            nxt = int(torch.multinomial(probs, num_samples=1).item())
            if nxt == 3:
                break
            ids.append(nxt)
            generated.append(nxt)
        continuation = decode_tokens(sp, generated)
        full_text = (text + " " + continuation).strip() if continuation else text
        return f"Continuation:\n{continuation}\n\nFull text:\n{full_text}"

    with gr.Blocks(title="Tiny Decoder LM Compare") as demo:
        gr.Markdown("# Tiny Decoder LM Compare")
        with gr.Row():
            model_name = gr.Dropdown(list(MODELS), value="lm_40m", label="Model")
            max_new = gr.Slider(1, 200, value=50, step=1, label="Max tokens")
        with gr.Row():
            temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="Temperature")
            top_k = gr.Slider(0, 100, value=40, step=1, label="Top-k")
            top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
            repetition_penalty = gr.Slider(1.0, 2.0, value=1.15, step=0.05, label="Repeat penalty")
        text = gr.Textbox(label="Start words", lines=3)
        run = gr.Button("Run", variant="primary")
        output = gr.Textbox(label="Prediction", lines=10)
        gr.Examples(EXAMPLES, inputs=[model_name, text, max_new])
        run.click(
            continue_lm,
            inputs=[model_name, text, max_new, temperature, top_k, top_p, repetition_penalty],
            outputs=output,
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", default="tokenizer/tokenizer.model")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7864)
    args = parser.parse_args()
    build_app(args.tokenizer_path).launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
