/**
 * test_tokenizer.js: Validação do Tokenizador Byte-Level BPE e do Motor de Prompt Wrapping.
 */

const assert = require('assert');
const path = require('path');
const { ByteLevelBPETokenizer } = require('./tokenizer.js');
const { converterPerguntaParaPrefixo } = require('./prompt_wrapper.js');

console.log('='.repeat(70));
console.log('  TESTE DE VALIDAÇÃO: TOKENIZADOR BPE & PROMPT WRAPPER (ETAPA 2)');
console.log('='.repeat(70));

// ============================================================================
// 1. Inicialização do Tokenizador
// ============================================================================
console.log('\n[1/3] Inicializando ByteLevelBPETokenizer a partir dos arquivos locais...');
const vocabPath = path.join(__dirname, 'tokenizer/vocab.json');
const mergesPath = path.join(__dirname, 'tokenizer/merges.txt');

const tokenizer = new ByteLevelBPETokenizer(vocabPath, mergesPath);

const vocabSize = Object.keys(tokenizer.vocab).length;
const mergesCount = tokenizer.bpeRanks.size;
console.log(`      Vocabulário carregado: ${vocabSize} tokens`);
console.log(`      Merges BPE carregados: ${mergesCount} pares`);
assert.strictEqual(vocabSize, 4096, 'O vocabulário deve conter exatamente 4096 tokens');

// ============================================================================
// 2. Validação das Frases-Chave e Teste Roundtrip (encode -> decode)
// ============================================================================
console.log('\n[2/3] Validando frases-chave e roundtrip de integridade UTF-8...');

const testCases = [
  {
    nome: 'Frase-Chave 1 (Copom / Selic)',
    texto: 'O Copom decidiu manter a taxa Selic',
    esperadoIds: [49, 622, 2884, 2252, 261, 519, 1197]
  },
  {
    nome: 'Frase-Chave 2 (Provisões / SUSEP)',
    texto: 'As provisões técnicas da SUSEP',
    esperadoIds: [1544, 352, 2019, 3676, 315, 459, 1852, 39, 50]
  },
  {
    nome: 'Frase Adicional (Sistema Financeiro Nacional)',
    texto: 'Em relação às condições de crédito no Sistema Financeiro Nacional',
    esperadoIds: [1687, 563, 968, 1199, 262, 647, 320, 2788, 2004, 483, 1057]
  },
  {
    nome: 'Frase Numérica / Pontuação',
    texto: 'No contexto econômico e financeiro brasileiro, 12,3% ao ano.',
    esperadoIds: [1740, 1747, 2044, 263, 1686, 1239, 483, 14, 980, 14, 21, 7, 411, 504, 16]
  }
];

for (const tc of testCases) {
  console.log(`\n  ▸ ${tc.nome}:`);
  console.log(`    Texto Original: "${tc.texto}"`);

  const ids = tokenizer.encode(tc.texto);
  console.log(`    Token IDs: [${ids.join(', ')}]`);

  if (tc.esperadoIds) {
    assert.deepStrictEqual(Array.from(ids), tc.esperadoIds, `IDs gerados não coincidem com o esperado para: ${tc.texto}`);
    console.log(`    ✓ IDs coincidem exatamente com o padrão oficial de referência!`);
  }

  // Validação BigInt64Array para ONNX WebAssembly
  const bigIntIds = tokenizer.encode(tc.texto, { returnBigInt: true });
  assert(bigIntIds instanceof BigInt64Array, 'Deveria retornar BigInt64Array com a opção returnBigInt: true');
  assert.strictEqual(bigIntIds.length, ids.length, 'O comprimento de BigInt64Array deve ser igual aos IDs normais');

  // Validação Roundtrip
  const decodificado = tokenizer.decode(ids);
  console.log(`    Decodificado:   "${decodificado}"`);
  assert.strictEqual(decodificado, tc.texto, `Falha no roundtrip: "${decodificado}" !== "${tc.texto}"`);
  console.log(`    ✓ Roundtrip perfeito (texto original 100% preservado)`);
}

// ============================================================================
// 3. Validação das Ramificações do Prompt Wrapper
// ============================================================================
console.log('\n[3/3] Validando ramificações do Motor de Prompt Wrapping...');

const wrapperTests = [
  {
    ramo: '1. Crédito / Inadimplência / Spread / Bancos',
    pergunta: 'Como o spread bancário e a inadimplência afetam as taxas de crédito?',
    esperado: 'Em relação às condições de crédito no Sistema Financeiro Nacional, as taxas cobradas e o spread bancário refletem o risco de crédito na medida em que'
  },
  {
    ramo: '2. Copom / Selic / Juros / Política Monetária',
    pergunta: 'Qual a decisão do Copom sobre a taxa Selic e política monetária?',
    esperado: 'O Copom avalia que as decisões de política monetária sobre a taxa Selic impactam a inflação e a economia porque'
  },
  {
    ramo: '3. Inflação / Preços / IPCA / IGP',
    pergunta: 'Qual é o impacto do aumento de preços e da inflação no IPCA?',
    esperado: 'No cenário de referência para a inflação, o comportamento dos preços livres e dos preços administrados por contrato indica que'
  },
  {
    ramo: '4. Provisões / Solvência / SUSEP / Seguro / Sinistro',
    pergunta: 'Como as seguradoras calculam as provisões técnicas e solvência na SUSEP?',
    esperado: 'No âmbito atuarial e regulatório da SUSEP, a constituição de provisões técnicas e os limites de solvência são essenciais porque'
  },
  {
    ramo: '5. Fallback Afirmativo (remoção de interrogativos)',
    pergunta: 'Como o agronegócio impacta o PIB brasileiro?',
    esperado: 'No contexto econômico e financeiro brasileiro, o agronegócio impacta o PIB brasileiro ocorre porque'
  }
];

for (const wt of wrapperTests) {
  console.log(`\n  ▸ Ramo: ${wt.ramo}`);
  console.log(`    Pergunta de Entrada: "${wt.pergunta}"`);
  const resultado = converterPerguntaParaPrefixo(wt.pergunta);
  console.log(`    Prefixo Produzido:   "${resultado}"`);

  assert.strictEqual(resultado, wt.esperado, `Divergência no prefixo para: "${wt.pergunta}"`);

  // Validação adicional: testar se o prefixo gerado tokeniza e decodifica sem erros
  const prefixIds = tokenizer.encode(resultado);
  const prefixDecoded = tokenizer.decode(prefixIds);
  assert.strictEqual(prefixDecoded, resultado, 'Prefixo deve ter roundtrip perfeito no tokenizador');
  console.log(`    ✓ Prefixo validado com sucesso (Token count: ${prefixIds.length})`);
}

console.log('\n' + '='.repeat(70));
console.log('  TODOS OS TESTES FORAM EXECUTADOS E VALIDADOS COM 100% DE SUCESSO! 🎉');
console.log('='.repeat(70));

