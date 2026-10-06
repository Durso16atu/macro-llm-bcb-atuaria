/**
 * prompt_wrapper.js: Motor de Prompt Wrapping para a Macro LLM (BCB & Atuária).
 * Transforma perguntas abertas dos usuários em prefixos analíticos assertivos
 * adaptados para modelos autoregressivos decoder-only pré-treinados do zero.
 */

/**
 * Converte uma pergunta ou consulta do usuário em um prefixo temático estruturado.
 * @param {string} pergunta - Texto da pergunta inserida pelo usuário.
 * @returns {string} Prefixo técnico padronizado para geração causal.
 */
function converterPerguntaParaPrefixo(pergunta) {
  if (!pergunta || typeof pergunta !== 'string') {
    return "No contexto econômico e financeiro brasileiro, este cenário ocorre porque";
  }

  const q = pergunta.trim().toLowerCase();

  // 1. Crédito / Inadimplência / Spread / Bancos
  if (['crédito', 'credito', 'inadimplência', 'inadimplencia', 'spread', 'bancos'].some(k => q.includes(k))) {
    return "Em relação às condições de crédito no Sistema Financeiro Nacional, as taxas cobradas e o spread bancário refletem o risco de crédito na medida em que";
  }

  // 2. Copom / Selic / Juros / Política Monetária
  if (['selic', 'copom', 'juros', 'política monetária', 'politica monetaria'].some(k => q.includes(k))) {
    return "O Copom avalia que as decisões de política monetária sobre a taxa Selic impactam a inflação e a economia porque";
  }

  // 3. Inflação / Preços / IPCA / IGP
  if (['inflação', 'inflacao', 'preço', 'precos', 'ipca', 'igp'].some(k => q.includes(k))) {
    return "No cenário de referência para a inflação, o comportamento dos preços livres e dos preços administrados por contrato indica que";
  }

  // 4. Provisões / Solvência / SUSEP / Seguro / Sinistro
  if (['provisão', 'provisao', 'provisões', 'provisoes', 'solvência', 'solvencia', 'susep', 'seguro', 'sinistro'].some(k => q.includes(k))) {
    return "No âmbito atuarial e regulatório da SUSEP, a constituição de provisões técnicas e os limites de solvência são essenciais porque";
  }

  // 5. Fallback Afirmativo
  const q_limpa = pergunta
    .split('?').join('')
    .split('Como').join('')
    .split('como').join('')
    .split('Qual').join('')
    .split('qual').join('')
    .trim();

  return `No contexto econômico e financeiro brasileiro, ${q_limpa} ocorre porque`;
}

// Exportação universal (compatível com Node.js e Browser)
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { converterPerguntaParaPrefixo };
}
if (typeof window !== 'undefined') {
  window.converterPerguntaParaPrefixo = converterPerguntaParaPrefixo;
}

