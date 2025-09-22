# app.py - launch Gradio dashboard
import gradio as gr
from pathlib import Path
from src.configs import OUTPUT_DIR
from src.features import make_features
from src.llm import local_llm_answer

OUT = Path(OUTPUT_DIR)

def ui_deploy():
    p = OUT / 'segments.csv'
    return p.read_text() if p.exists() else 'No segments.csv found. Run src.features.'

def ui_lap():
    p = OUT / 'lap_summary.csv'
    return p.read_text() if p.exists() else 'No lap_summary.csv found. Run src.features.'

def ui_ask(q):
    return local_llm_answer(q or 'Summarize deploy events', use_local=False)

def ui_report():
    p = OUT / 'f1_ers_report.txt'
    return p.read_text() if p.exists() else 'No report found.'

with gr.Blocks(title='F1 ERS — Final Dashboard') as demo:
    gr.Markdown('# 🏎️ F1 ERS Deployment — Demo')
    with gr.Tab('ERS Analysis'):
        out_deploy = gr.Textbox(lines=12, label='Deployment Events')
        out_lap = gr.Textbox(lines=8, label='Lap Summary')
        gr.Button('Show deployments').click(lambda: ui_deploy(), outputs=out_deploy)
        gr.Button('Show lap summary').click(lambda: ui_lap(), outputs=out_lap)
    with gr.Tab('RL / Agent'):
        if (OUT/'ppo_real_rewards.png').exists():
            gr.Image(value=str(OUT/'ppo_real_rewards.png'), label='PPO Episode Rewards (per lap)')
        else:
            gr.Markdown('⚠️ PPO training plot not found. Run `python -m src.train`.')
    with gr.Tab('Assistant'):
        q = gr.Textbox(label='Your question')
        ans = gr.Textbox(lines=6, label='Answer')
        gr.Button('Ask').click(lambda x: ui_ask(x), inputs=q, outputs=ans)
    with gr.Tab('📑 Report'):
        gr.Textbox(label='Report Preview', lines=15, value=ui_report())
        if (OUT/'f1_ers_report.txt').exists():
            gr.File(value=str(OUT/'f1_ers_report.txt'), label='Download Report')

demo.launch(server_name='127.0.0.1', share=False)
