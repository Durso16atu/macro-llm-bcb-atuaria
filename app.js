/**
 * app.js: Motor de Inferência ONNX WebAssembly 100% Client-Side para a Macro LLM.
 * Executa a rede neural autoregressiva de 12,3M de parâmetros diretamente no navegador.
 * 
 * Estabilização de Memória & Prevenção de Crash 61696 (Out-of-Memory / Threading):
 * 1. Força estritamente ort.env.wasm.numThreads = 1 e proxy = false (sem SharedArrayBuffer/COOP).
 * 2. Libera buffers intermediários (chunks = null, modelBuffer = null) após carregar o binário.
 * 3. Garante descarte de tensores e ciclos de Garbage Collection a cada token gerado.
 */

// 1. Configuração Estrita de Ambiente WASM (Single-Thread / Sem Proxy Workers)
// Impede o erro fatal 61696 e crashes por falta de SharedArrayBuffer / COOP no GitHub Pages
if (typeof ort !== 'undefined' && ort.env && ort.env.wasm) {
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
}

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
  // Reafirma a configuração estrita do WASM no carregamento do DOM
  if (typeof ort !== 'undefined' && ort.env && ort.env.wasm) {
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
  }

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

    // Alerta de Erro Visual
    errorAlert: document.getElementById('errorAlert'),
    errorMessage: document.getElementById('errorMessage'),

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
    sliderTopP: document.getElementById('sliderTopP'),
    valTopP: document.getElementById('valTopP'),
    sliderTopK: document.getElementById('sliderTopK'),
    valTopK: document.getElementById('valTopK'),

    // Saída Unificada e Streaming
    outputCard: document.getElementById('outputCard'),
    placeholderText: document.getElementById('placeholderText'),
    outputPrefix: document.getElementById('outputPrefix'),
    outputStreaming: document.getElementById('outputStreaming'),
    streamingCursor: document.getElementById('streamingCursor'),
    btnCopiar: document.getElementById('btnCopiar'),

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
  if (dom.sliderTokens) {
    dom.sliderTokens.addEventListener('input', (e) => {
      dom.valTokens.textContent = e.target.value;
    });
  }
  if (dom.sliderTemp) {
    dom.sliderTemp.addEventListener('input', (e) => {
      dom.valTemp.textContent = parseFloat(e.target.value).toFixed(2);
    });
  }
  if (dom.sliderPenalty) {
    dom.sliderPenalty.addEventListener('input', (e) => {
      dom.valPenalty.textContent = parseFloat(e.target.value).toFixed(2);
    });
  }
  if (dom.sliderTopP) {
    dom.sliderTopP.addEventListener('input', (e) => {
      dom.valTopP.textContent = parseFloat(e.target.value).toFixed(2);
    });
  }
  if (dom.sliderTopK) {
    dom.sliderTopK.addEventListener('input', (e) => {
      dom.valTopK.textContent = e.target.value;
    });
  }

  // Alternador de Configurações
  if (dom.toggleConfigBtn) {
    dom.toggleConfigBtn.addEventListener('click', () => {
      dom.configPanel.classList.toggle('hidden');
      const isExpanded = !dom.configPanel.classList.contains('hidden');
      const icon = dom.toggleConfigBtn.querySelector('.chevron-icon');
      if (icon) {
        icon.style.transform = isExpanded ? 'rotate(180deg)' : 'rotate(0deg)';
      }
    });
  }

  // Botão Copiar Análise Unificada
  if (dom.btnCopiar) {
    dom.btnCopiar.addEventListener('click', copiarAnaliseCompleta);
  }

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
 * Exibe um alerta de erro amigável na tela.
 */
function exibirAlertaErro(mensagem) {
  if (dom.errorAlert && dom.errorMessage) {
    dom.errorMessage.textContent = mensagem;
    dom.errorAlert.classList.remove('hidden');
  }
}

function ocultarAlertaErro() {
  if (dom.errorAlert) {
    dom.errorAlert.classList.add('hidden');
  }
}

/**
 * Inicializa o modelo ONNX (com streaming de download e progresso) e o tokenizador.
 * Libera explicitamente todos os buffers intermediários para evitar OOM no V8.
 */
async function initModelAndTokenizer() {
  if (state.isModelReady || state.isModelLoading) return;
  state.isModelLoading = true;
  ocultarAlertaErro();

  atualizarStatus('Carregando Tokenizador...', 'loading');
  dom.progressContainer.classList.remove('hidden');

  try {
    // 1. Carrega o tokenizador
    console.log('[Tokenizador] Carregando arquivos vocab.json e merges.txt...');
    state.tokenizer = await ByteLevelBPETokenizer.fromUrl(state.vocabUrl, state.mergesUrl);
    console.log('[Tokenizador] Pronto com 4096 tokens.');

    // 2. Garante configuração estrita single-thread do WASM
    if (typeof ort !== 'undefined' && ort.env && ort.env.wasm) {
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.proxy = false;
    }

    // 3. Carrega o modelo ONNX com streaming de progresso e descarte de buffers
    atualizarStatus('Baixando Modelo Macro LLM (~53 MB)...', 'loading');

    let modelBuffer = await fetchModelWithProgress(state.modelUrl, (pct, loaded, total) => {
      dom.progressBar.style.width = `${pct}%`;
      dom.progressPercent.textContent = `${pct}%`;
      const loadedMb = (loaded / (1024 * 1024)).toFixed(1);
      const totalMb = (total / (1024 * 1024)).toFixed(1);
      dom.progressDetails.textContent = `${loadedMb} MB / ${totalMb} MB transferidos`;
    });

    atualizarStatus('Compilando WebAssembly (Single-Thread)...', 'loading');
    dom.progressDetails.textContent = 'Alocando runtime WebAssembly de baixa memória...';

    // Cria a sessão com provedor WASM single-thread
    state.session = await ort.InferenceSession.create(modelBuffer, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });

    // Libera imediatamente a referência ao ArrayBuffer de 53 MB no JS Heap
    modelBuffer = null;

    state.isModelReady = true;
    state.isModelLoading = false;

    // Atualiza a interface para estado Pronto
    atualizarStatus('Pronto para Análise (WASM Local)', 'ready');
    dom.progressContainer.classList.add('hidden');
    dom.btnGerar.disabled = false;
    dom.btnGerar.classList.remove('opacity-50', 'cursor-not-allowed');

    console.log('[ONNX Runtime] Sessão criada com sucesso (WASM Single-Thread)!');
  } catch (error) {
    console.error('[Erro na Inicialização]', error);
    let msgAmigavel = `Falha na inicialização do WebAssembly: ${error.message}`;
    if (error.message && (error.message.includes('61696') || error.message.includes('out of memory') || error.message.includes('OOM'))) {
      msgAmigavel = 'O navegador atingiu o limite de memória para WebAssembly. Tente fechar outras abas e recarregar a página.';
    }
    atualizarStatus('Erro ao carregar modelo', 'error');
    dom.progressDetails.textContent = msgAmigavel;
    exibirAlertaErro(msgAmigavel);
    state.isModelLoading = false;
  }
}

/**
 * Faz download do arquivo .onnx com monitoramento de progresso e
 * descarte progressivo dos chunks para minimizar a pressão de memória.
 */
async function fetchModelWithProgress(url, onProgress) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Falha no download HTTP (${response.status} ${response.statusText})`);
  }

  const contentLength = response.headers.get('content-length');
  const total = contentLength ? parseInt(contentLength, 10) : 55564853;

  if (!response.body) {
    const buf = await response.arrayBuffer();
    onProgress(100, total, total);
    return buf;
  }

  const reader = response.body.getReader();
  let loaded = 0;
  let chunks = [];

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.length;
      const pct = Math.min(100, Math.round((loaded / total) * 100));
      onProgress(pct, loaded, total);
    }
  } catch (streamErr) {
    chunks = null;
    throw streamErr;
  }

  // Concatenação em buffer único contíguo liberando cada chunk progressivamente
  const completeBuffer = new Uint8Array(loaded);
  let offset = 0;
  while (chunks.length > 0) {
    const chunk = chunks.shift();
    completeBuffer.set(chunk, offset);
    offset += chunk.length;
  }
  chunks = null; // Anula explicitamente o array de chunks do heap do V8

  return completeBuffer.buffer;
}

/**
 * Sanitiza o primeiro token gerado para garantir continuidade limpa e sem espaços espúrios.
 * @param {string} pedacoTexto - Texto bruto decodificado do 1º token gerado.
 * @param {string} prefixo - Prefixo condicional anterior.
 * @returns {string} Token com espaçamento calibrado.
 */
function sanitizarPrimeiroToken(pedacoTexto, prefixo) {
  if (!pedacoTexto) return '';

  // Remove caracteres \u00a0 e quebras de linha espúrias
  let tokenLimpo = pedacoTexto.replace(/\u00a0/g, ' ').replace(/[\r\n]+/g, ' ');

  // Remove espaços repetidos e espaços iniciais brutos
  tokenLimpo = tokenLimpo.replace(/^[ \t]+/, '');
  if (tokenLimpo.length === 0) return '';

  const prefixoTrim = prefixo ? prefixo.trimEnd() : '';
  const ultimoChar = prefixoTrim.length > 0 ? prefixoTrim[prefixoTrim.length - 1] : '';

  const ehPontuacao = /^[.,;:!?]/.test(tokenLimpo);

  if (ehPontuacao) {
    // Se o token gerado for pontuação, anexa colado sem espaço
    return tokenLimpo;
  } else {
    // Se for palavra alfanumérica e o prefixo não termina com espaço ou parênteses aberto
    if (ultimoChar && !/\s|[(\[{]/.test(ultimoChar)) {
      return ' ' + tokenLimpo;
    }
    return tokenLimpo;
  }
}

/**
 * Pós-processamento semântico contra truncamento abrupto no meio de orações.
 * Retrocede o texto gerado até a última pontuação forte (. ! ?).
 * @param {string} textoContinuacao - Texto gerado acumulado.
 * @returns {string} Texto ajustado com terminação gramatical legítima.
 */
function truncarSemanticamente(textoContinuacao) {
  if (!textoContinuacao) return '';
  // Expressão regular que captura até a última pontuação forte final legítima
  const regex = /([\s\S]*[.!?]["')\]]?)/;
  const match = textoContinuacao.match(regex);
  if (match && match[1].trim().length > 0) {
    return match[1].trimEnd();
  }
  // Se não houver pontuação intermediária, mantém o texto gerado
  return textoContinuacao.trimEnd();
}

/**
 * Inicia o loop autorregressivo de geração de texto em streaming com alívio de memória.
 */
async function iniciarGeracao() {
  if (!state.isModelReady || state.isGenerating) return;
  ocultarAlertaErro();

  const textoEntrada = dom.inputPrompt.value.trim();
  if (!textoEntrada) {
    alert('Por favor, digite uma pergunta ou selecione um dos atalhos temáticos.');
    dom.inputPrompt.focus();
    return;
  }

  const modoLivre = dom.toggleModoLivre ? dom.toggleModoLivre.checked : false;
  const maxTokens = parseInt(dom.sliderTokens?.value, 10) || 80;
  const temperature = parseFloat(dom.sliderTemp?.value) || 0.65;
  const repetitionPenalty = parseFloat(dom.sliderPenalty?.value) || 1.18;
  const topP = parseFloat(dom.sliderTopP?.value) || 0.88;
  const topK = parseInt(dom.sliderTopK?.value, 10) || 40;

  // Aplicação do Prompt Wrapper ou Modo Livre
  let prefixoFinal = textoEntrada;
  if (!modoLivre && typeof converterPerguntaParaPrefixo === 'function') {
    prefixoFinal = converterPerguntaParaPrefixo(textoEntrada);
  }

  // Preparação de Estado de Geração
  state.isGenerating = true;
  state.abortController = new AbortController();

  dom.btnGerar.classList.add('hidden');
  dom.btnParar.classList.remove('hidden');

  // Terminal Unificado: renderiza o prefixo em destaque atenuado e limpa a saída de continuação
  if (dom.placeholderText) dom.placeholderText.classList.add('hidden');
  dom.outputPrefix.textContent = modoLivre ? '' : prefixoFinal;
  dom.outputStreaming.textContent = '';
  dom.streamingCursor.classList.remove('hidden');
  dom.metricsContainer.classList.remove('hidden');

  atualizarStatus('Gerando análise com WebAssembly...', 'generating');

  const inicioTempo = performance.now();
  let tokensGeradosCount = 0;
  let parouPorEarlyStop = false;
  let textoGeradoAcumulado = '';

  try {
    // Tokeniza o prefixo
    let tokenIds = state.tokenizer.encode(prefixoFinal);
    if (!tokenIds || tokenIds.length === 0) {
      tokenIds = [2]; // Fallback unk
    }

    const blockSize = 256;
    let tokensHistorico = [...tokenIds];
    const minTokensSeguranca = 35; // Patamar mínimo antes do Early Stop

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

      // Descarte explícito de referências internas do runtime WASM
      if (typeof resultado.logits.dispose === 'function') {
        resultado.logits.dispose();
      }
      if (typeof tensorInput.dispose === 'function') {
        tensorInput.dispose();
      }

      // 1. Penalidade de Repetição em janela móvel de 64 tokens recentes
      const janelaRecente = tokensHistorico.slice(-64);
      const uniqueRecente = new Set(janelaRecente);
      for (const tId of uniqueRecente) {
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

      // 3. Amostragem Calibrada: Top-K (40) combinada com Top-P (0.88)
      const nextTokenId = sampleTopKTopP(logits, topK, topP);

      // Adiciona o novo token ao histórico
      tokensHistorico.push(nextTokenId);
      tokensGeradosCount++;

      // Decodifica o token gerado
      let pedacoTexto = state.tokenizer.decode([nextTokenId]);

      // Correção de Fronteira no 1º Token (step === 0)
      if (step === 0) {
        pedacoTexto = sanitizarPrimeiroToken(pedacoTexto, prefixoFinal);
      }

      textoGeradoAcumulado += pedacoTexto;
      dom.outputStreaming.textContent = textoGeradoAcumulado;

      // Métricas de velocidade em tempo real
      const agora = performance.now();
      const segundosDecorridos = (agora - inicioTempo) / 1000;
      const velocidade = segundosDecorridos > 0 ? (tokensGeradosCount / segundosDecorridos).toFixed(1) : '0.0';

      dom.metricTokens.textContent = `${tokensGeradosCount} tokens`;
      dom.metricSpeed.textContent = `${velocidade} t/s`;
      dom.metricTime.textContent = `${segundosDecorridos.toFixed(2)}s`;

      // Critério de Parada Natural (Early Stop):
      // Após o patamar mínimo (35 tokens), se encontrar pontuação forte (. ! ?), encerra a oração
      if (tokensGeradosCount >= minTokensSeguranca && /[.!?]/.test(pedacoTexto)) {
        console.log(`[Early Stop] Conclusão de período atingida no token ${tokensGeradosCount}.`);
        parouPorEarlyStop = true;
        break;
      }

      // Pausa essencial: libera a thread principal para renderizar e ativa o Garbage Collector do V8
      await new Promise((r) => setTimeout(r, 0));
    }

    // Pós-Processamento Semântico: se atingiu maxTokens sem parar naturalmente por pontuação
    if (!parouPorEarlyStop && !state.abortController?.signal?.aborted) {
      const textoSemantico = truncarSemanticamente(textoGeradoAcumulado);
      if (textoSemantico && textoSemantico !== textoGeradoAcumulado) {
        dom.outputStreaming.textContent = textoSemantico;
        console.log('[Pós-Processamento] Truncamento semântico aplicado na última oração completa.');
      }
    }
  } catch (err) {
    console.error('[Erro na geração]', err);
    let msgErro = err.message || 'Erro desconhecido';
    if (msgErro.includes('61696') || msgErro.includes('out of memory') || msgErro.includes('OOM') || msgErro.includes('unreachable')) {
      msgErro = 'O navegador atingiu o limite de memória WebAssembly (OOM). Sugestão: feche abas pesadas e tente gerar menos tokens.';
    }
    dom.outputStreaming.textContent += `\n\n[⚠️ ${msgErro}]`;
    exibirAlertaErro(msgErro);
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
 * Amostragem Combinada: Top-K filtrada por Top-P (Nucleus Sampling).
 * @param {Float32Array} logits - Logits de saída (4096 valores).
 * @param {number} topK - Limite máximo de candidatos (default 40).
 * @param {number} topP - Massa de probabilidade acumulada (default 0.88).
 * @returns {number} ID do token sorteado.
 */
function sampleTopKTopP(logits, topK = 40, topP = 0.88) {
  const n = logits.length;
  const indices = new Int32Array(n);
  for (let i = 0; i < n; i++) indices[i] = i;

  // Ordena os índices em ordem decrescente de logit
  indices.sort((a, b) => logits[b] - logits[a]);

  const k = Math.min(topK, n);
  const maxVal = logits[indices[0]];

  // 1. Softmax numericamente estável sobre os top-k candidatos
  const expProbs = new Float32Array(k);
  let sumExp = 0;
  for (let i = 0; i < k; i++) {
    const p = Math.exp(logits[indices[i]] - maxVal);
    expProbs[i] = p;
    sumExp += p;
  }

  for (let i = 0; i < k; i++) {
    expProbs[i] /= sumExp;
  }

  // 2. Corte por Top-P (Nucleus)
  let cumProb = 0;
  let cutoffIndex = k;
  for (let i = 0; i < k; i++) {
    cumProb += expProbs[i];
    if (cumProb >= topP) {
      cutoffIndex = i + 1;
      break;
    }
  }

  // 3. Sorteio multinomial sobre o núcleo filtrado
  let sumNucleus = 0;
  for (let i = 0; i < cutoffIndex; i++) {
    sumNucleus += expProbs[i];
  }

  let r = Math.random() * sumNucleus;
  for (let i = 0; i < cutoffIndex; i++) {
    r -= expProbs[i];
    if (r <= 0) return indices[i];
  }
  return indices[0];
}

/**
 * Copia o texto consolidado (prefixo + continuação gerada) para a área de transferência.
 */
async function copiarAnaliseCompleta() {
  const prefixo = dom.outputPrefix ? dom.outputPrefix.textContent : '';
  const continuacao = dom.outputStreaming ? dom.outputStreaming.textContent : '';
  const textoCompleto = (prefixo + continuacao).trim();

  if (!textoCompleto) {
    alert('Nenhum texto gerado para copiar.');
    return;
  }

  try {
    await navigator.clipboard.writeText(textoCompleto);
    const spanOriginal = dom.btnCopiar.innerHTML;
    dom.btnCopiar.innerHTML = `
      <svg class="w-3.5 h-3.5 text-emerald-400 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M5 13l4 4L19 7"></path>
      </svg>
      <span class="text-emerald-400 font-semibold">Copiado! ✓</span>
    `;
    dom.btnCopiar.classList.add('border-emerald-500/50', 'bg-emerald-500/10');

    setTimeout(() => {
      dom.btnCopiar.innerHTML = spanOriginal;
      dom.btnCopiar.classList.remove('border-emerald-500/50', 'bg-emerald-500/10');
    }, 2000);
  } catch (err) {
    console.error('Falha ao copiar:', err);
    // Fallback tradicional
    const textarea = document.createElement('textarea');
    textarea.value = textoCompleto;
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand('copy');
    document.body.removeChild(textarea);
    alert('Análise copiada com sucesso!');
  }
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
