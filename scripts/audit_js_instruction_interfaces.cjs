/* Parse frozen source strings; never evaluate source programs or alter data. */
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const readline = require('node:readline');
const root = path.resolve(__dirname, '..');
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const toolsFile = path.join(root, '.tools/javascript-toolchain-v1.json');
const tools = JSON.parse(fs.readFileSync(toolsFile));
if (hash(process.execPath) !== tools.node_sha256) throw Error('Pinned Node binary changed');
const parserRoot = path.join(tools.toolchain_root, 'node_modules/@babel/parser');
for (const [file, digest] of Object.entries(tools.dependency_files_sha256)) {
  if (file.startsWith('node_modules/@babel/parser/') && hash(path.join(tools.toolchain_root, file)) !== digest)
    throw Error('Pinned parser source changed');
}
const parser = require(parserRoot);
const parserVersion = JSON.parse(fs.readFileSync(path.join(parserRoot, 'package.json'))).version;
function candidate(node) {
  if (!node) return null;
  if (['FunctionDeclaration', 'FunctionExpression', 'ArrowFunctionExpression', 'ClassMethod', 'ObjectMethod'].includes(node.type)) return node;
  if (node.type === 'VariableDeclaration' && node.declarations.length === 1) return candidate(node.declarations[0].init);
  if (node.type === 'ExpressionStatement') return candidate(node.expression);
  if (node.type === 'AssignmentExpression') return candidate(node.right);
  return null;
}
function interfaceOf(code) {
  const contexts = [
    ['', '', ast => candidate(ast.program.body[0])],
    ['(', ')', ast => candidate(ast.program.body[0])],
    ['class OriginalContext {\n', '\n}', ast => candidate(ast.program.body[0].body.body[0])],
    ['({\n', '\n})', ast => candidate(ast.program.body[0].expression.properties[0])],
  ];
  let diagnostic = null;
  for (const [prefix, suffix, select] of contexts) {
    try {
      const ast = parser.parse(prefix + code + suffix, {sourceType: 'unambiguous', plugins: ['jsx', 'flow']});
      const fn = select(ast);
      if (!fn || !fn.body) continue;
      return {node_type: fn.type, context: prefix ? 'surrounding-syntax-wrapper' : 'original-source',
        body_start: fn.body.start - prefix.length,
        parameter_ends: fn.params.map(p => p.end - prefix.length)};
    } catch (error) { diagnostic = error.message; }
  }
  return {diagnostic: diagnostic || 'No supported outer function/method node'};
}
function measure(code) {
  const original = interfaceOf(code), prefix = [...code.split('{', 1)[0]].slice(0, 500).join('');
  if (original.diagnostic) return {parsed: false, diagnostic: original.diagnostic};
  return {parsed: true, ...original, signature_prefix: prefix,
    parameter_text_cut: original.parameter_ends.some(end => end > prefix.length),
    body_source_in_signature_context: prefix.length > original.body_start,
    original_full_header: code.slice(0, original.body_start)};
}
async function main() {
  if (process.argv.includes('--verify')) {
    const cases = [
      ['normal', 'function sum(a, b) { return a+b; }', false, false],
      ['destructuring', 'function sum({a,b}) { return a+b; }', true, false],
      ['default-object', 'function configure(options = {}) { return options; }', true, false],
      ['expression-arrow', 'const double = n => n * 2;', false, true],
      ['class-method', 'calculate({value}) { return value; }', true, false],
      ['anonymous', 'function (value) { return value; }', false, false],
      ['unicode', 'const 倍 = ({値}) => 値 * 2;', true, false],
      ['astral-default', "function label(value = '😀') { return value; }", false, false],
    ];
    const results = cases.map(([name, code, cut, body]) => {
      const result = measure(code);
      if (!result.parsed || result.parameter_text_cut !== cut || result.body_source_in_signature_context !== body)
        throw Error('Real parser interface probe failed: ' + name);
      return {name, ...result};
    });
    console.log(JSON.stringify({passed: true, parser_version: parserVersion, results})); return;
  }
  const planPath = path.join(root, 'reports/extended-curriculum-v1/plan.json');
  const stage = JSON.parse(fs.readFileSync(planPath)).stages.find(item => item.name === 'multilingual-code-instruction');
  const output = path.join(root, 'reports/javascript-instruction-interface-audit-v1.json');
  if (fs.existsSync(output)) throw Error('Preserve existing completed interface audit');
  // Streaming input identity avoids allocating the entire two-gigabyte file.
  const digest = crypto.createHash('sha256');
  for await (const bytes of fs.createReadStream(stage.data)) digest.update(bytes);
  if (digest.digest('hex') !== stage.data_sha256) throw Error('Frozen instruction input changed');
  const counts = {all_rows_scanned: 0, javascript_rows: 0, parsed_outer_interfaces: 0,
    parser_or_context_unsupported: 0, parameter_text_cut: 0, body_source_in_signature_context: 0};
  const examples = {parameter_text_cut: [], body_source_in_signature_context: [], unsupported: []};
  const implementationSha = hash(__filename), toolsSha = hash(toolsFile);
  const input = fs.createReadStream(stage.data), inputDigest = crypto.createHash('sha256');
  input.on('data', bytes => inputDigest.update(bytes));
  const source = readline.createInterface({input, crlfDelay: Infinity});
  for await (const line of source) {
    const row = JSON.parse(line); counts.all_rows_scanned++;
    if (row.source !== 'CodeSearchNet' || row.language !== 'JavaScript') continue;
    counts.javascript_rows++;
    const match = /^```javascript\n([\s\S]*)\n```$/.exec(row.response);
    if (!match) throw Error('Expected complete original JavaScript fenced source');
    const code = match[1], result = measure(code);
    if (!result.parsed) {
      counts.parser_or_context_unsupported++;
      if (examples.unsupported.length < 10) examples.unsupported.push({id: row.id, source_url: row.source_url, ...result});
      continue;
    }
    if (!row.prompt.includes('Signature/context: ' + result.signature_prefix + '\n\n')) throw Error('Known signature transform differs from actual prompt');
    counts.parsed_outer_interfaces++;
    for (const key of ['parameter_text_cut', 'body_source_in_signature_context']) if (result[key]) {
      counts[key]++;
      if (examples[key].length < 10) examples[key].push({id: row.id, source_url: row.source_url,
        signature_context: result.signature_prefix, original_full_header: result.original_full_header,
        node_type: result.node_type, context: result.context});
    }
  }
  if (inputDigest.digest('hex') !== stage.data_sha256 || hash(__filename) !== implementationSha || hash(toolsFile) !== toolsSha)
    throw Error('Actually scanned data or parser implementation/toolchain changed');
  const report = {completed_at: new Date().toISOString(), data_sha256: stage.data_sha256,
    toolchain_manifest_sha256: toolsSha, parser_version: parserVersion,
    implementation_sha256: implementationSha, counts, examples, data_changed: false, source_programs_executed: false,
    goal_achieved: false, limitations: ['Parser/context failures are not proof that original source is invalid.',
      'Truncated header text or copied body text can coexist with informative documentation; semantic instruction sufficiency is not proven.',
      'This is a source-format audit, not compilation, algorithmic correctness, model learning or independent capability evidence.']};
  fs.writeFileSync(output, JSON.stringify(report, null, 2) + '\n', {flag: 'wx'});
  console.log(JSON.stringify({report: output, counts}));
}
main().catch(error => {console.error(error.stack); process.exitCode = 1;});
