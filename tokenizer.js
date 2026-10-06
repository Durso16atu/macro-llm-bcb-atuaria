/**
 * tokenizer.js: Tokenizador BPE Byte-Level para a Macro LLM (4096 tokens).
 * Implementação pura em JavaScript compatível tanto com Node.js quanto com Browser (WebAssembly/ONNX Runtime Web).
 */

class ByteLevelBPETokenizer {
  /**
   * Construtor do Tokenizador Byte-Level BPE.
   * @param {Object|string} [vocab] - Objeto do vocabulário { token: id } ou caminho para vocab.json (Node.js).
   * @param {string} [merges] - Conteúdo do merges.txt ou caminho para merges.txt (Node.js).
   */
  constructor(vocab, merges) {
    this.vocab = {};
    this.idToToken = {};
    this.bpeRanks = new Map();
    this.bpeCache = new Map();

    const { byteToChar, charToByte } = ByteLevelBPETokenizer.bytesToUnicode();
    this.byteToChar = byteToChar;
    this.charToByte = charToByte;

    this.encoder = new TextEncoder();
    this.decoder = new TextDecoder('utf-8');

    // Expressão regular oficial GPT-2 / Hugging Face para segmentação de tokens
    this.pat = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

    if (vocab !== undefined || merges !== undefined) {
      this.init(vocab, merges);
    } else if (typeof process !== 'undefined' && process.versions && process.versions.node) {
      // Se executado em Node.js sem parâmetros, carrega os arquivos padrão locais
      this.loadFromLocalFiles();
    }
  }

  /**
   * Inicializa o vocabulário e os merges fornecidos.
   * @param {Object|string} vocab
   * @param {string} merges
   */
  init(vocab, merges) {
    // Processamento do Vocabulário
    if (typeof vocab === 'string') {
      if (typeof require !== 'undefined') {
        const fs = require('fs');
        const path = require('path');
        const resolvedVocab = path.isAbsolute(vocab) ? vocab : path.resolve(vocab);
        this.vocab = JSON.parse(fs.readFileSync(resolvedVocab, 'utf-8'));
      } else {
        this.vocab = JSON.parse(vocab);
      }
    } else if (typeof vocab === 'object' && vocab !== null) {
      this.vocab = vocab;
    }

    this.idToToken = {};
    for (const [token, id] of Object.entries(this.vocab)) {
      this.idToToken[id] = token;
    }

    // Processamento dos Merges
    let mergesText = merges;
    if (typeof merges === 'string' && typeof require !== 'undefined') {
      const fs = require('fs');
      const path = require('path');
      if (fs.existsSync(merges)) {
        const resolvedMerges = path.isAbsolute(merges) ? merges : path.resolve(merges);
        mergesText = fs.readFileSync(resolvedMerges, 'utf-8');
      }
    }

    this.bpeRanks = new Map();
    if (typeof mergesText === 'string') {
      const lines = mergesText.split(/\r?\n/);
      let rank = 0;
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith('#')) continue;
        this.bpeRanks.set(trimmed, rank++);
      }
    }

    this.bpeCache.clear();
  }

  /**
   * Carrega os arquivos locais padrão quando executado no Node.js.
   */
  loadFromLocalFiles(vocabPath = 'tokenizer/vocab.json', mergesPath = 'tokenizer/merges.txt') {
    if (typeof require === 'undefined') {
      throw new Error('loadFromLocalFiles está disponível apenas no ambiente Node.js.');
    }
    const fs = require('fs');
    const path = require('path');

    let resolvedVocab = path.resolve(vocabPath);
    let resolvedMerges = path.resolve(mergesPath);

    if (!fs.existsSync(resolvedVocab) && typeof __dirname !== 'undefined') {
      resolvedVocab = path.join(__dirname, vocabPath);
      resolvedMerges = path.join(__dirname, mergesPath);
    }

    this.init(resolvedVocab, resolvedMerges);
  }

  /**
   * Carregamento assíncrono para navegadores web (Fetch API).
   * @param {string} vocabUrl
   * @param {string} mergesUrl
   * @returns {Promise<ByteLevelBPETokenizer>}
   */
  static async fromUrl(vocabUrl, mergesUrl) {
    const [vocabRes, mergesRes] = await Promise.all([
      fetch(vocabUrl),
      fetch(mergesUrl)
    ]);
    const vocab = await vocabRes.json();
    const merges = await mergesRes.text();
    const tokenizer = new ByteLevelBPETokenizer();
    tokenizer.init(vocab, merges);
    return tokenizer;
  }

  /**
   * Mapeamento fiel de bytes para caracteres Unicode (bytes_to_unicode do GPT-2).
   * @returns {{ byteToChar: Object, charToByte: Object }}
   */
  static bytesToUnicode() {
    const bs = [];
    const cs = [];

    for (let b = '!'.charCodeAt(0); b <= '~'.charCodeAt(0); b++) bs.push(b);
    for (let b = '¡'.charCodeAt(0); b <= '¬'.charCodeAt(0); b++) bs.push(b);
    for (let b = '®'.charCodeAt(0); b <= 'ÿ'.charCodeAt(0); b++) bs.push(b);

    for (let i = 0; i < bs.length; i++) cs.push(bs[i]);

    let n = 0;
    for (let b = 0; b < 256; b++) {
      if (!bs.includes(b)) {
        bs.push(b);
        cs.push(256 + n);
        n++;
      }
    }

    const byteToChar = {};
    const charToByte = {};
    for (let i = 0; i < bs.length; i++) {
      const ch = String.fromCharCode(cs[i]);
      byteToChar[bs[i]] = ch;
      charToByte[ch] = bs[i];
    }
    return { byteToChar, charToByte };
  }

  /**
   * Aplicação do algoritmo de merges BPE para um token em caracteres-byte.
   * @param {string} token
   * @returns {string[]}
   */
  _bpe(token) {
    if (this.bpeCache.has(token)) {
      return this.bpeCache.get(token);
    }

    let word = Array.from(token);
    if (word.length <= 1) {
      this.bpeCache.set(token, word);
      return word;
    }

    while (true) {
      let minRank = Infinity;
      let bestPair = null;

      for (let i = 0; i < word.length - 1; i++) {
        const pair = word[i] + ' ' + word[i + 1];
        if (this.bpeRanks.has(pair)) {
          const rank = this.bpeRanks.get(pair);
          if (rank < minRank) {
            minRank = rank;
            bestPair = [word[i], word[i + 1]];
          }
        }
      }

      if (!bestPair) break;

      const [first, second] = bestPair;
      const newWord = [];
      let i = 0;
      while (i < word.length) {
        if (i < word.length - 1 && word[i] === first && word[i + 1] === second) {
          newWord.push(first + second);
          i += 2;
        } else {
          newWord.push(word[i]);
          i += 1;
        }
      }
      word = newWord;
      if (word.length <= 1) break;
    }

    this.bpeCache.set(token, word);
    return word;
  }

  /**
   * Converte texto em IDs numéricos de tokens.
   * @param {string} text - Texto de entrada.
   * @param {Object} [options]
   * @param {boolean} [options.returnBigInt=false] - Se true, retorna BigInt64Array.
   * @returns {number[]|BigInt64Array}
   */
  encode(text, options = {}) {
    if (typeof text !== 'string' || text.length === 0) {
      const empty = [];
      empty.toBigInt64Array = () => new BigInt64Array(0);
      empty.ids = empty;
      return options.returnBigInt ? new BigInt64Array(0) : empty;
    }

    const matches = text.match(this.pat) || [];
    const tokenIds = [];
    const unkId = this.vocab['<|unk|>'] ?? 2;

    for (const chunk of matches) {
      const bytes = this.encoder.encode(chunk);
      let byteChars = '';
      for (let i = 0; i < bytes.length; i++) {
        byteChars += this.byteToChar[bytes[i]];
      }

      const bpePieces = this._bpe(byteChars);
      for (const piece of bpePieces) {
        if (this.vocab[piece] !== undefined) {
          tokenIds.push(this.vocab[piece]);
        } else {
          tokenIds.push(unkId);
        }
      }
    }

    // Helper method para compatibilidade com ONNX Runtime WebAssembly
    tokenIds.toBigInt64Array = function() {
      return BigInt64Array.from(tokenIds.map(BigInt));
    };
    tokenIds.ids = tokenIds;

    if (options.returnBigInt) {
      return tokenIds.toBigInt64Array();
    }
    return tokenIds;
  }

  /**
   * Decodifica uma sequência de IDs de tokens de volta para string UTF-8.
   * @param {number[]|BigInt64Array|Uint32Array|Int32Array|Object} tokenIds
   * @returns {string}
   */
  decode(tokenIds) {
    if (!tokenIds) return '';

    const ids = Array.isArray(tokenIds)
      ? tokenIds
      : tokenIds.ids && Array.isArray(tokenIds.ids)
      ? tokenIds.ids
      : Array.from(tokenIds);

    let byteChars = '';
    for (let i = 0; i < ids.length; i++) {
      const id = Number(ids[i]);
      if (this.idToToken[id] !== undefined) {
        byteChars += this.idToToken[id];
      }
    }

    const bytes = [];
    for (let i = 0; i < byteChars.length; i++) {
      const ch = byteChars[i];
      if (this.charToByte[ch] !== undefined) {
        bytes.push(this.charToByte[ch]);
      }
    }

    return this.decoder.decode(new Uint8Array(bytes));
  }
}

// Exportação universal (compatível com Node.js e Browser)
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { ByteLevelBPETokenizer };
}
if (typeof window !== 'undefined') {
  window.ByteLevelBPETokenizer = ByteLevelBPETokenizer;
}

