"""
app.py: Aplicação Gradio para Hospedagem Permanente da Macro LLM
"""
import os
import torch
import gradio as gr
from tokenizers import ByteLevelBPETokenizer
from model import GPT, GPTConfig

# 1. Configuração de Hardware e Caminhos
device = 'cuda' if torch.cuda.is_available() else 'cpu'
base_dir = os.path.dirname(os.path.abspath(__file__))

checkpoint_path = os.path.join(base_dir, 'minha_llm_best.pt')
tokenizer_dir = os.path.join(base_dir, 'tokenizer')
vocab_path = os.path.join(tokenizer_dir, 'vocab.json')
merges_path = os.path.join(tokenizer_dir, 'merges.txt')

# 2. Inicialização do Modelo e Tokenizador
assert os.path.exists(checkpoint_path), f"Checkpoint não encontrado em: {checkpoint_path}"
assert os.path.exists(vocab_path) and os.path.exists(merges_path), "Arquivos do tokenizador ausentes."

tokenizer = ByteLevelBPETokenizer(vocab_path, merges_path)

config = GPTConfig()
model = GPT(config).to(device)

checkpoint = torch.load(checkpoint_path, map_location=device)
state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
model.load_state_dict(state_dict)
model.eval()

# 3. Motor de Prompt Wrapping (Adaptação para Modelo Base Causal)
def converter_pergunta_para_prefixo(pergunta: str) -> str:
    q = pergunta.strip().lower()

    if any(k in q for k in ['crédito', 'credito', 'inadimplência', 'inadimplencia', 'spread', 'bancos']):
        return "Em relação às condições de crédito no Sistema Financeiro Nacional, as taxas cobradas e o spread bancário refletem o risco de crédito na medida em que"

    elif any(k in q for k in ['selic', 'copom', 'juros', 'política monetária', 'politica monetaria']):
        return "O Copom avalia que as decisões de política monetária sobre a taxa Selic impactam a inflação e a economia porque"

    elif any(k in q for k in ['inflação', 'inflacao', 'preço', 'precos', 'ipca', 'igp']):
        return "No cenário de referência para a inflação, o comportamento dos preços livres e dos preços administrados por contrato indica que"

    elif any(k in q for k in ['provisão', 'provisao', 'provisões', 'provisoes', 'solvência', 'solvencia', 'susep', 'seguro', 'sinistro']):
        return "No âmbito atuarial e regulatório da SUSEP, a constituição de provisões técnicas e os limites de solvência são essenciais porque"

    else:
        q_limpa = pergunta.replace('?', '').replace('Como', '').replace('como', '').replace('Qual', '').replace('qual', '').strip()
        return f"No contexto econômico e financeiro brasileiro, {q_limpa} ocorre porque"

# 4. Pipeline de Inferência
def responder_usuario(pergunta_usuario, modo_livre, tamanho_resposta, temperatura):
    if not pergunta_usuario.strip():
        return "Por favor, digite uma consulta ou tema econômico/atuarial.", ""

    prefixo = pergunta_usuario.strip() if modo_livre else converter_pergunta_para_prefixo(pergunta_usuario)
    ids = torch.tensor(tokenizer.encode(prefixo).ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        saida_ids = model.generate(
            ids,
            max_new_tokens=int(tamanho_resposta),
            temperature=float(temperatura),
            top_k=25,
            repetition_penalty=1.15
        )[0].tolist()

    texto_gerado = tokenizer.decode(saida_ids)
    return texto_gerado, prefixo

# 5. Interface Visual
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate"
)

with gr.Blocks(title="Macro LLM - BCB & Atuária", theme=theme) as demo:
    gr.Markdown("# 🏛️ Macro LLM: Sistema Financeiro, Política Monetária e Atuária")
    gr.Markdown(
        "Modelo de linguagem causal de **12,3M de parâmetros** pré-treinado sobre decisões do Copom, "
        "Relatórios de Política Monetária (RPM), Relatórios de Estabilidade Financeira (REF) e normas da SUSEP."
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_texto = gr.Textbox(
                label="Consulta ou Tópico:",
                placeholder="Ex.: Como a taxa Selic impacta as provisões técnicas e a inflação?",
                lines=4
            )
            check_modo_livre = gr.Checkbox(
                label="Modo Livre (Geração estrita a partir do texto digitado, sem prompt wrapper)",
                value=False
            )
            with gr.Accordion("Parâmetros de Amostragem", open=False):
                slider_tokens = gr.Slider(minimum=40, maximum=180, value=90, step=10, label="Tokens a Gerar")
                slider_temp = gr.Slider(minimum=0.2, maximum=1.0, value=0.6, step=0.05, label="Temperatura")

            btn_enviar = gr.Button("Gerar Análise", variant="primary")

        with gr.Column(scale=1):
            output_resposta = gr.Textbox(label="Texto Produzido pelo Modelo:", lines=9)
            output_prefixo = gr.Textbox(label="Prefixo Técnico Injetado (Prompt Wrapper):", lines=2)

    btn_enviar.click(
        fn=responder_usuario,
        inputs=[input_texto, check_modo_livre, slider_tokens, slider_temp],
        outputs=[output_resposta, output_prefixo]
    )

if __name__ == "__main__":
    demo.launch()