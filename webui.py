

import gradio as gr
import os, gc, copy, torch
from datetime import datetime
from huggingface_hub import hf_hub_download
from pynvml import *

# Flag to check if GPU is present
HAS_GPU = True

# Model title and context size limit
ctx_limit = 2000
title = "RWKV-5-H-World-7B"
model_file = "rwkv-5-h-world-7B"

# Get the GPU count
try:
    nvmlInit()
    GPU_COUNT = nvmlDeviceGetCount()
    if GPU_COUNT > 0:
        HAS_GPU = True
        gpu_h = nvmlDeviceGetHandleByIndex(0)
except NVMLError as error:
    print(error)


os.environ["RWKV_CUDA_ON"] = '1'
MODEL_STRAT = "cuda fp16i8"

# Load the model accordingly
from rwkv.model import RWKV
model_path = hf_hub_download(repo_id="a686d380/rwkv-5-h-world", filename=f"{model_file}.pth")
model = RWKV(model=model_path, strategy=MODEL_STRAT)
from rwkv.utils import PIPELINE, PIPELINE_ARGS
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

# Prompt generation
def generate_prompt(instruction, input=""):
    instruction = instruction.strip().replace('\r\n','\n').replace('\n\n','\n')
    input = input.strip().replace('\r\n','\n').replace('\n\n','\n')
    if input:
        return f"""Instruction: {instruction}
Input: {input}
Response:"""
    else:
        return f"""User: hi
Assistant: Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.
User: {instruction}
Assistant:"""

# Evaluation logic
def evaluate(
    ctx,
    token_count=200,
    temperature=1.0,
    top_p=0.7,
    presencePenalty = 0.1,
    countPenalty = 0.1,
):
    print(ctx)
    args = PIPELINE_ARGS(temperature = max(0.2, float(temperature)), top_p = float(top_p),
                     alpha_frequency = countPenalty,
                     alpha_presence = presencePenalty,
                     token_ban = [], # ban the generation of some tokens
                     token_stop = [0]) # stop generation whenever you see any token here
    ctx = ctx.strip()
    all_tokens = []
    out_last = 0
    out_str = ''
    occurrence = {}
    state = None
    for i in range(int(token_count)):
        out, state = model.forward(pipeline.encode(ctx)[-ctx_limit:] if i == 0 else [token], state)
        for n in occurrence:
            out[n] -= (args.alpha_presence + occurrence[n] * args.alpha_frequency)

        token = pipeline.sample_logits(out, temperature=args.temperature, top_p=args.top_p)
        if token in args.token_stop:
            break
        all_tokens += [token]
        for xxx in occurrence:
            occurrence[xxx] *= 0.996
        if token not in occurrence:
            occurrence[token] = 1
        else:
            occurrence[token] += 1

        tmp = pipeline.decode(all_tokens[out_last:])
        if '\ufffd' not in tmp:
            out_str += tmp
            yield out_str.strip()
            out_last = i + 1

    if HAS_GPU == True :
        gpu_info = nvmlDeviceGetMemoryInfo(gpu_h)
        print(f'vram {gpu_info.total} used {gpu_info.used} free {gpu_info.free}')

    del out
    del state
    gc.collect()

    if HAS_GPU == True :
        torch.cuda.empty_cache()

    yield out_str.strip()

# Gradio blocks
with gr.Blocks(title=title) as demo:
    gr.HTML(f"<div style=\"text-align: center;\">\n<h1>RWKV-5 World v2 - {title}</h1>\n</div>")
    with gr.Tab("Raw Generation"):
        gr.Markdown(f"This is RWKV-5-h ")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(lines=2, label="Prompt", value="""边儿上还有两条腿，修长、结实，光滑得出奇，潜伏着
媚人的活力。他紧张得脊梁都皱了起来。但他不动声色""")
                token_count = gr.Slider(10, 500, label="Max Tokens", step=10, value=200)
                temperature = gr.Slider(0.2, 2.0, label="Temperature", step=0.1, value=1.0)
                top_p = gr.Slider(0.0, 1.0, label="Top P", step=0.05, value=0.3)
                presence_penalty = gr.Slider(0.0, 1.0, label="Presence Penalty", step=0.1, value=1)
                count_penalty = gr.Slider(0.0, 1.0, label="Count Penalty", step=0.1, value=1)
            with gr.Column():
                with gr.Row():
                    submit = gr.Button("Submit", variant="primary")
                    clear = gr.Button("Clear", variant="secondary")
                output = gr.Textbox(label="Output", lines=5)
        data = gr.Dataset(components=[prompt, token_count, temperature, top_p, presence_penalty, count_penalty], label="Example Instructions", headers=["Prompt", "Max Tokens", "Temperature", "Top P", "Presence Penalty", "Count Penalty"])
        submit.click(evaluate, [prompt, token_count, temperature, top_p, presence_penalty, count_penalty], [output])
        clear.click(lambda: None, [], [output])
        data.click(lambda x: x, [data], [prompt, token_count, temperature, top_p, presence_penalty, count_penalty])

demo.queue().launch(
        server_name="0.0.0.0",
        inbrowser=True,
        share=True,
        server_port=9872,
        quiet=True,
    )
