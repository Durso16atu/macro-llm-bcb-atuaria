#!/usr/bin/env python3
"""
export_onnx.py: Script para exportação da Macro LLM (12,3M de parâmetros) para ONNX WebAssembly.
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Importa a arquitetura e configuração da Macro LLM
from model import GPT, GPTConfig


class ClassicCausalSelfAttention(nn.Module):
    """
    Fallback clássico para atenção causal matricial:
    (q @ k.T / sqrt(d) + causal_mask)
    Garante compatibilidade total com runtimes ONNX e WebAssembly.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Atenção causal matricial clássica:
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        causal_mask = torch.tril(torch.ones((T, T), dtype=torch.bool, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(~causal_mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


def apply_classic_attention_fallback(model: GPT, config: GPTConfig):
    """Substitui os módulos de atenção causal pelo fallback matricial clássico."""
    print("Aplicando fallback clássico para atenção causal matricial nos blocos do Transformer...")
    for block in model.transformer.h:
        classic_attn = ClassicCausalSelfAttention(config)
        classic_attn.load_state_dict(block.attn.state_dict())
        block.attn = classic_attn


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(base_dir, "minha_llm_best.pt")
    onnx_output_path = os.path.join(base_dir, "macro_llm.onnx")

    # 1. Carregamento do checkpoint em modo CPU
    print(f"[1/4] Carregando checkpoint '{checkpoint_path}' em modo CPU...")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint não encontrado em {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

    # 2. Instanciação da arquitetura GPT
    print("[2/4] Instanciando a arquitetura GPT (12,3M parâmetros)...")
    config = GPTConfig(
        vocab_size=4096,
        block_size=256,
        n_embd=384,
        n_head=6,
        n_layer=6,
        dropout=0.0
    )
    model = GPT(config)
    model.load_state_dict(state_dict)
    model.eval()

    # Contagem de parâmetros
    total_params = sum(p.numel() for p in model.parameters())
    print(f"       Parâmetros totais: {total_params:,} (~12.3M)")

    # 3. Exportação para ONNX
    print(f"[3/4] Exportando modelo para ONNX: '{onnx_output_path}'...")
    dummy_input = torch.zeros((1, 8), dtype=torch.long)
    input_names = ["idx"]
    output_names = ["logits"]
    dynamic_axes = {"idx": {1: "seq_len"}}
    opset_version = 14

    export_args = dict(
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )

    def run_export(target_model):
        try:
            torch.onnx.export(
                target_model,
                (dummy_input,),
                onnx_output_path,
                dynamo=False,
                **export_args
            )
        except TypeError:
            torch.onnx.export(
                target_model,
                (dummy_input,),
                onnx_output_path,
                **export_args
            )

    try:
        run_export(model)
        print("       Exportação ONNX concluída com sucesso!")
    except Exception as e:
        print(f"       Conflito ou erro na exportação padrão ({e}).")
        apply_classic_attention_fallback(model, config)
        run_export(model)
        print("       Exportação ONNX concluída com fallback clássico!")

    # 4. Validação da saída
    print("\n[4/4] Validando artefato gerado e executando teste de inferência...")
    if not os.path.exists(onnx_output_path):
        raise FileNotFoundError(f"Erro: O arquivo {onnx_output_path} não foi gerado.")

    tamanho_bytes = os.path.getsize(onnx_output_path)
    tamanho_mb = tamanho_bytes / (1024 * 1024)
    print(f"       Arquivo gerado: {onnx_output_path}")
    print(f"       Tamanho do arquivo: {tamanho_bytes:,} bytes (~{tamanho_mb:.2f} MB)")

    # Teste rápido de inferência com onnxruntime
    try:
        import onnxruntime as ort
        print("\n       Inicializando sessão de inferência no onnxruntime...")
        session = ort.InferenceSession(onnx_output_path)
        tensor_teste = np.array([[1, 2, 3]], dtype=np.int64)

        saida = session.run(None, {"idx": tensor_teste})
        logits = saida[0]

        print(f"       Entrada de teste: idx = {tensor_teste.tolist()} (shape: {tensor_teste.shape})")
        print(f"       Dimensão de saída (logits): {list(logits.shape)}")

        assert logits.shape == (1, 1, 4096), f"Esperado (1, 1, 4096), obtido {logits.shape}"
        print("       Validação de inferência ONNX CONCLUÍDA COM SUCESSO! Dimensão [1, 1, 4096] confirmada.")
    except Exception as e:
        print(f"       Erro durante a validação no onnxruntime: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

