/**
 * app.js: Motor de Inferência ONNX WebAssembly 100% Client-Side para a Macro LLM.
 * Executa a rede neural autoregressiva de 12,3M de parâmetros diretamente no navegador.
 */

// Estado da Aplicação
const state = {
  session: null,
  tokenizer: null,
  isModelLoading: false,
  isModelReady: false,
  isGenerating: false,
  abortController: null,
  modelUrl: './macro_llm.onnx',
  vocabUrl: './tokenizer/vocab.json',
  mergesUrl: './tokenizer/merges.txt',
};

// Exemplos Temáticos dos 4 Botões Rápidos
const ATALHOS_TEMATICOS = {
  copom: {
    titulo: '🏛️ Copom & Selic',
    prompt: 'Como as decisões do Copom sobre a taxa Selic impactam a inflação e a economia?',
  },
  credito: {
    titulo: '💳 Risco de Crédito',
    prompt: 'De que forma as taxas de juros e o spread bancário refletem o risco de crédito e a inadimplência no SFN?',
  },
  inflacao: {
    titulo: '📈 Inflação & IPCA',
    prompt: 'Qual é o comportamento dos preços livres e monitorados na trajetória recente do IPCA?',
  },
  susep: {
    titulo: '🛡️ Provisões SUSEP',
    prompt: 'Como a regulação da SUSEP sobre provisões técnicas e margens de solvência protege o mercado segurador?',
  },
};

// Elementos da UI
let dom = {};

document.addEventListener('DOMContentLoaded', () => {
  initDomReferences();
  setupEventListeners();
  initModelAndTokenizer();
});

function initDomReferences() {
  dom = {
    // Status e Progresso
    statusBadge: document.getElementById('statusBadge'),
    statusText: document.getElementById('statusText'),
    progressContainer: document.getElementById('progressContainer'),
    progressBar: document.getElementById('progressBar'),
    progressPercent: document.getElementById('progressPercent'),
    progressDetails: document.getElementById('progressDetails'),

    // Inputs e Controles
    inputPrompt: document.getElementById('inputPrompt'),
    toggleModoLivre: document.getElementById('toggleModoLivre'),
    btnGerar: document.getElementById('btnGerar'),
    btnParar: document.getElementById('btnParar'),

    // Configurações
    toggleConfigBtn: document.getElementById('toggleConfigBtn'),
    configPanel: document.getElementById('configPanel'),
    sliderTokens: document.getElementById('sliderTokens'),
    valTokens: document.getElementById('valTokens'),
    sliderTemp: document.getElementById('sliderTemp'),
    valTemp: document.getElementById('valTemp'),
    sliderPenalty: document.getElementById('sliderPenalty'),
    valPenalty: document.getElementById('valPenalty'),

    // Saída e Streaming
    outputCard: document.getElementById('outputCard'),
    outputPrefixCard: document.getElementById('outputPrefixCard'),
    outputPrefix: document.getElementById('outputPrefix'),
    outputStreaming: document.getElementById('outputStreaming'),
    streamingCursor: document.getElementById('streamingCursor'),

    // Métricas
    metricsContainer: document.getElementById('metricsContainer'),
    metricTokens: document.getElementById('metricTokens'),
    metricSpeed: document.getElementById('metricSpeed'),
    metricTime: document.getElementById('metricTime'),

    // Botões de Atalhos
    btnAtalhoCopom: document.getElementById('btnAtalhoCopom'),
    btnAtalhoCredito: document.getElementById('btnAtalhoCredito'),
    btnAtalhoInflacao: document.getElementById('btnAtalhoInflacao'),
    btnAtalhoSusep: document.getElementById('btnAtalhoSusep'),
  };
}

function setupEventListeners() {
  // Atalhos Rápidos
  if (dom.btnAtalhoCopom) {
    dom.btnAtalhoCopom.addEventListener('click', () => carregarAtalho('copom'));
  }
  if (dom.btnAtalhoCredito) {
    dom.btnAtalhoCredito.addEventListener('click', () => carregarAtalho('credito'));
  }
  if (dom.btnAtalhoInflacao) {
    dom.btnAtalhoInflacao.addEventListener('click', () => carregarAtalho('inflacao'));
  }
  if (dom.btnAtalhoSusep) {
    dom.btnAtalhoSusep.addEventListener('click', () => carregarAtalho('susep'));
  }

  // Sliders com feedback numérico em tempo real
  dom.sliderTokens.addEventListener('input', (e) => {
    dom.valTokens.textContent = e.target.value;
  });
  dom.sliderTemp.addEventListener('input', (e) => {
    dom.valTemp.textContent = parseFloat(e.target.value).toFixed(2);
  });
  dom.sliderPenalty.addEventListener('input', (e) => {
    dom.valPenalty.textContent = parseFloat(e.target.value).toFixed(2);
  });

  // Alternador de Configurações
  dom.toggleConfigBtn.addEventListener('click', () => {
    dom.configPanel.classList.toggle('hidden');
    const isExpanded = !dom.configPanel.classList.contains('hidden');
    dom.toggleConfigBtn.querySelector('.chevron-icon').style.transform = isExpanded ? 'rotate(180deg)' : 'rotate(0deg)';
  });

  // Ações de Geração
  dom.btnGerar.addEventListener('click', iniciarGeracao);
  dom.btnParar.addEventListener('click', pararGeracao);

  // Permitir Ctrl+Enter / Cmd+Enter para gerar
  dom.inputPrompt.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      iniciarGeracao();
    }
  });
}

function carregarAtalho(chave) {
  const atalho = ATALHOS_TEMATICOS[chave];
  if (!atalho) return;
  dom.inputPrompt.value = atalho.prompt;
  dom.inputPrompt.focus();
}

/**
 * Inicializa o modelo ONNX (com streaming de download e progresso) e o tokenizador.
 */
async function initModelAndTokenizer() {
  if (state.isModelReady || state.isModelLoading) return;
  state.isModelLoading = true;

  atualizarStatus('Carregando Tokenizador...', 'loading');
  dom.progressContainer.classList.remove('hidden');

  try {
    // 1. Carrega o tokenizador
    console.log('[Tokenizador] Carregando arquivos vocab.json e merges.txt...');
    state.tokenizer = await ByteLevelBPETokenizer.fromUrl(state.vocabUrl, state.mergesUrl);
    console.log('[Tokenizador] Pronto com 4096 tokens.');

    // 2. Carrega o modelo ONNX com streaming de progresso
    atualizarStatus('Baixando Modelo Macro LLM (~53 MB)...', 'loading');

    // Configura threads para o WebAssembly
    if (window.ort && ort.env && ort.env.wasm) {
      ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);
    }

    const modelBuffer = await fetchModelWithProgress(state.modelUrl, (pct, loaded, total) => {
      dom.progressBar.style.width = `${pct}%`;
      dom.progressPercent.textContent = `${pct}%`;
      const loadedMb = (loaded / (1024 * 1024)).toFixed(1);
      const totalMb = (total / (1024 * 1024)).toFixed(1);
      dom.progressDetails.textContent = `${loadedMb} MB / ${totalMb} MB transferidos`;
    });

    atualizarStatus('Compilando WebAssembly e carregando pesos na memória...', 'loading');
    dom.progressDetails.textContent = 'Inicializando sessão no runtime WASM...';

    state.session = await ort.InferenceSession.create(modelBuffer, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });

    state.isModelReady = true;
    state.isModelLoading = false;

    // Atualiza a interface para estado Pronto
    atualizarStatus('Pronto para Análise (WASM Local)', 'ready');
    dom.progressContainer.classList.add('hidden');
    dom.btnGerar.disabled = false;
    dom.btnGerar.classList.remove('opacity-50', 'cursor-not-allowed');

    console.log('[ONNX Runtime] Sessão criada com sucesso!');
  } catch (error) {
    console.error('[Erro na Inicialização]', error);
    atualizarStatus(`Erro no carregamento: ${error.message}`, 'error');
    dom.progressDetails.textContent = 'Falha ao baixar ou inicializar o modelo WebAssembly.';
    state.isModelLoading = false;
  }
}

/**
 * Faz download do arquivo .onnx monitorando o progresso da requisição HTTP.
 */
async function fetchModelWithProgress(url, onProgress) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Falha no download HTTP (${response.status} ${response.statusText})`);
  }

  // Se o servidor fornecer Content-Length, usamos; caso contrário, estimamos 55.5 MB
  const contentLength = response.headers.get('content-length');
  const total = contentLength ? parseInt(contentLength, 10) : 55564853;

  if (!response.body) {
    const buf = await response.arrayBuffer();
    onProgress(100, total, total);
    return buf;
  }

  const reader = response.body.getReader();
  let loaded = 0;
  const chunks = [];

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    const pct = Math.min(100, Math.round((loaded / total) * 100));
    onProgress(pct, loaded, total);
  }

  const completeBuffer = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    completeBuffer.set(chunk, offset);
    offset += chunk.length;
  }

  return completeBuffer.buffer;
}

/**
 * Inicia o loop autorregressivo de geração de texto em streaming.
 */
async function iniciarGeracao() {
  if (!state.isModelReady || state.isGenerating) return;

  const textoEntrada = dom.inputPrompt.value.trim();
  if (!textoEntrada) {
    alert('Por favor, digite uma pergunta ou selecione um dos atalhos temáticos.');
    dom.inputPrompt.focus();
    return;
  }

  const modoLivre = dom.toggleModoLivre.checked;
  const maxTokens = parseInt(dom.sliderTokens.value, 10) || 80;
  const temperature = parseFloat(dom.sliderTemp.value) || 0.6;
  const repetitionPenalty = parseFloat(dom.sliderPenalty.value) || 1.15;

  // Aplicação do Prompt Wrapper ou Modo Livre
  let prefixoFinal = textoEntrada;
  if (!modoLivre && typeof converterPerguntaParaPrefixo === 'function') {
    prefixoFinal = converterPerguntaParaPrefixo(textoEntrada);
    dom.outputPrefixCard.classList.remove('hidden');
    dom.outputPrefix.textContent = prefixoFinal;
  } else {
    dom.outputPrefixCard.classList.add('hidden');
    dom.outputPrefix.textContent = '';
  }

  // Preparação de Estado de Geração
  state.isGenerating = true;
  state.abortController = new AbortController();

  dom.btnGerar.classList.add('hidden');
  dom.btnParar.classList.remove('hidden');
  dom.outputStreaming.textContent = '';
  dom.streamingCursor.classList.remove('hidden');
  dom.metricsContainer.classList.remove('hidden');

  atualizarStatus('Gerando análise com WebAssembly...', 'generating');

  const inicioTempo = performance.now();
  let tokensGeradosCount = 0;

  try {
    // Tokeniza o prefixo
    let tokenIds = state.tokenizer.encode(prefixoFinal);
    if (!tokenIds || tokenIds.length === 0) {
      tokenIds = [2]; // Fallback unk
    }

    const blockSize = 256;
    let tokensHistorico = [...tokenIds];

    for (let step = 0; step < maxTokens; step++) {
      if (state.abortController.signal.aborted) {
        console.log('[Geração] Interrompida pelo usuário.');
        break;
      }

      // Janela de contexto: garante no máximo 256 tokens
      const contexto = tokensHistorico.length <= blockSize
        ? tokensHistorico
        : tokensHistorico.slice(-blockSize);

      // Criação do Tensor de Entrada ONNX: shape [1, seq_len]
      const tensorInput = new ort.Tensor(
        'int64',
        BigInt64Array.from(contexto.map(BigInt)),
        [1, contexto.length]
      );

      // Execução da inferência no modelo ONNX
      const resultado = await state.session.run({ idx: tensorInput });
      const logitsData = resultado.logits.data; // Float32Array(4096)
      const logits = new Float32Array(logitsData);

      // 1. Penalidade de Repetição (sobre o histórico gerado)
      const uniqueHistorico = new Set(tokensHistorico);
      for (const tId of uniqueHistorico) {
        if (tId >= 0 && tId < logits.length) {
          if (logits[tId] > 0) {
            logits[tId] /= repetitionPenalty;
          } else {
            logits[tId] *= repetitionPenalty;
          }
        }
      }

      // 2. Aplicação de Temperatura
      const tempEfetiva = Math.max(temperature, 1e-4);
      for (let i = 0; i < logits.length; i++) {
        logits[i] /= tempEfetiva;
      }

      // 3. Amostragem Top-K (k=25) Ponderada por Softmax
      const nextTokenId = sampleTopK(logits, 25);

      // Adiciona o novo token ao histórico
      tokensHistorico.push(nextTokenId);
      tokensGeradosCount++;

      // Decodifica o token gerado e atualiza a interface em streaming
      const pedacoTexto = state.tokenizer.decode([nextTokenId]);
      dom.outputStreaming.textContent += pedacoTexto;

      // Métricas de velocidade
      const agora = performance.now();
      const segundosDecorridos = (agora - inicioTempo) / 1000;
      const velocidade = segundosDecorridos > 0 ? (tokensGeradosCount / segundosDecorridos).toFixed(1) : '0.0';

      dom.metricTokens.textContent = `${tokensGeradosCount} tokens`;
      dom.metricSpeed.textContent = `${velocidade} t/s`;
      dom.metricTime.textContent = `${segundosDecorridos.toFixed(2)}s`;

      // Pausa microscópica para o navegador pintar a tela (yield streaming)
      await new Promise((r) => setTimeout(r, 0));
    }
  } catch (err) {
    console.error('[Erro na geração]', err);
    dom.outputStreaming.textContent += `\n[Erro durante a inferência: ${err.message}]`;
  } finally {
    finalizarGeracao(inicioTempo, tokensGeradosCount);
  }
}

/**
 * Interrompe a geração sob demanda do usuário.
 */
function pararGeracao() {
  if (state.abortController) {
    state.abortController.abort();
  }
}

/**
 * Conclui a geração e restaura a interface.
 */
function finalizarGeracao(inicioTempo, totalTokens) {
  state.isGenerating = false;
  state.abortController = null;

  dom.btnGerar.classList.remove('hidden');
  dom.btnParar.classList.add('hidden');
  dom.streamingCursor.classList.add('hidden');

  const duracao = ((performance.now() - inicioTempo) / 1000).toFixed(2);
  const vel = duracao > 0 ? (totalTokens / duracao).toFixed(1) : '0.0';

  dom.metricTokens.textContent = `${totalTokens} tokens`;
  dom.metricSpeed.textContent = `${vel} t/s`;
  dom.metricTime.textContent = `${duracao}s`;

  atualizarStatus('Pronto para Análise (WASM Local)', 'ready');
}

/**
 * Amostragem Multinomial com filtro Top-K.
 * @param {Float32Array} logits - Logits de saída (4096 valores).
 * @param {number} topK - Número de maiores candidatos a considerar.
 * @returns {number} ID do token sorteado.
 */
function sampleTopK(logits, topK = 25) {
  const n = logits.length;
  const indices = new Int32Array(n);
  for (let i = 0; i < n; i++) indices[i] = i;

  // Ordena os índices em ordem decrescente de valor de logit
  indices.sort((a, b) => logits[b] - logits[a]);

  const k = Math.min(topK, n);
  const maxVal = logits[indices[0]];

  // Softmax numericamente estável para os k maiores
  const probs = new Float32Array(k);
  let sumExp = 0;
  for (let i = 0; i < k; i++) {
    const p = Math.exp(logits[indices[i]] - maxVal);
    probs[i] = p;
    sumExp += p;
  }

  // Sorteio multinomial cumulativo
  let r = Math.random() * sumExp;
  for (let i = 0; i < k; i++) {
    r -= probs[i];
    if (r <= 0) return indices[i];
  }
  return indices[0];
}

/**
 * Atualiza o badge e o texto de status na barra principal.
 */
function atualizarStatus(texto, tipo) {
  if (!dom.statusText || !dom.statusBadge) return;
  dom.statusText.textContent = texto;

  dom.statusBadge.className = 'w-2.5 h-2.5 rounded-full inline-block mr-2';
  if (tipo === 'ready') {
    dom.statusBadge.classList.add('bg-emerald-400', 'shadow-[0_0_8px_rgba(52,211,153,0.8)]');
  } else if (tipo === 'loading' || tipo === 'generating') {
    dom.statusBadge.classList.add('bg-amber-400', 'animate-pulse');
  } else if (tipo === 'error') {
    dom.statusBadge.classList.add('bg-rose-500');
  } else {
    dom.statusBadge.classList.add('bg-slate-500');
  }
}
