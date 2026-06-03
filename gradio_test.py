import argparse
import os

import gradio as gr
import sentencepiece as spm
import torch

from inference_demo import decode_tokens, load_model


EXAMPLES = [
    ["how many", 20],
    ["best places to visit in", 20],
    ["beach sunset with", 20],
    ["what is the meaning of", 30],
    ["show me photos from", 30],
    ["the capital city of", 20],
]


def build_app(model_path: str, tokenizer_path: str):
    state = {"device": None, "sp": None, "model": None}

    def get_runtime():
        if state["model"] is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
            model = load_model(model_path, device)
            state.update({"device": device, "sp": sp, "model": model})
        return state["device"], state["sp"], state["model"]

    @torch.no_grad()
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

    @torch.no_grad()
    def continue_lm(
        text: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        avoid_repeats: bool,
    ) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        device, sp, model = get_runtime()
        ids = [2] + sp.encode(text, out_type=int)
        generated = []
        for _ in range(int(max_new_tokens)):
            inp = torch.tensor([ids], dtype=torch.long, device=device)
            logits, _ = model(inp)
            next_logits = logits[0, -1].float()
            if float(repetition_penalty) > 1.0:
                for token_id in set(ids[-128:] + generated):
                    if next_logits[token_id] < 0:
                        next_logits[token_id] *= float(repetition_penalty)
                    else:
                        next_logits[token_id] /= float(repetition_penalty)
            if avoid_repeats and generated:
                for token_id in set(generated):
                    next_logits[token_id] = -float("inf")
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

    with gr.Blocks(title="Tiny Decoder LM Test") as demo:
        gr.Markdown("# Tiny Decoder LM Test")
        gr.Markdown(f"Model: `{model_path}`")
        with gr.Row():
            max_new = gr.Slider(1, 200, value=30, step=1, label="Max tokens")
            temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="Temperature")
        with gr.Row():
            top_k = gr.Slider(0, 100, value=40, step=1, label="Top-k")
            top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
            repetition_penalty = gr.Slider(1.0, 2.0, value=1.25, step=0.05, label="Repeat penalty")
            avoid_repeats = gr.Checkbox(value=True, label="Avoid repeats")
        text = gr.Textbox(label="Start words", lines=3)
        run = gr.Button("Run", variant="primary")
        output = gr.Textbox(label="Prediction", lines=8)
        gr.Examples(EXAMPLES, inputs=[text, max_new])
        run.click(
            continue_lm,
            inputs=[text, max_new, temperature, top_k, top_p, repetition_penalty, avoid_repeats],
            outputs=output,
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="checkpoints/stage3/step_99000.pt")
    parser.add_argument("--tokenizer_path", default="tokenizer/tokenizer.model")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(args.model_path)
    if not os.path.exists(args.tokenizer_path):
        raise FileNotFoundError(args.tokenizer_path)
    build_app(args.model_path, args.tokenizer_path).launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
