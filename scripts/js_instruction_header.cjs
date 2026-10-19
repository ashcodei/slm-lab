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
module.exports = {measure, parserVersion, toolsFile, tools};
