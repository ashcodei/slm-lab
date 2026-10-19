/* Keep source code labels; replace only a demonstrably incomplete parameter header. */
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const readline = require('node:readline');
const {measure, parserVersion, toolsFile} = require('./js_instruction_header.cjs');
const root = path.resolve(__dirname, '..');
const sha = text => crypto.createHash('sha256').update(text).digest('hex');
async function main() {
  const auditPath = path.join(root, 'reports/javascript-instruction-interface-audit-v1.json');
  const audit = JSON.parse(fs.readFileSync(auditPath));
  const auditorPath = path.join(root, 'scripts/audit_js_instruction_interfaces.cjs');
  if (sha(fs.readFileSync(auditorPath)) !== audit.implementation_sha256) throw Error('Completed original interface auditor changed');
  const stage = JSON.parse(fs.readFileSync(path.join(root, 'reports/extended-curriculum-v1/plan.json'))).stages.find(s => s.name === 'multilingual-code-instruction');
  const directory = path.join(root, 'data/complete-js-parameters-v1');
  if (fs.existsSync(directory)) throw Error('Preserve prior variants and inspect; do not overwrite');
  fs.mkdirSync(directory);
  const output = path.join(directory, 'candidates.jsonl'), fd = fs.openSync(output, 'wx');
  const input = fs.createReadStream(stage.data), digest = crypto.createHash('sha256');
  input.on('data', bytes => digest.update(bytes));
  const source = readline.createInterface({input, crlfDelay: Infinity});
  let rows = 0;
  try {
    for await (const line of source) {
      const row = JSON.parse(line);
      if (row.source !== 'CodeSearchNet' || row.language !== 'JavaScript') continue;
      const match = /^```javascript\n([\s\S]*)\n```$/.exec(row.response);
      if (!match) throw Error('Original complete fenced JavaScript target required');
      const code = match[1], parsed = measure(code);
      if (!parsed.parsed || !parsed.parameter_text_cut) continue;
      if (sha(code) !== row.source_code_sha256) throw Error('Original JavaScript code target identity changed');
      const marker = 'Signature/context: ' + parsed.signature_prefix + '\n\n';
      if (!row.prompt.includes(marker)) throw Error('Original truncated parameter header differs');
      const prompt = row.prompt.replace(marker, 'Signature/context: ' + parsed.original_full_header + '\n\n');
      const variant = {...row, id: sha(row.id + '\n' + prompt + '\n' + row.response), prompt,
        parent_instruction_id: row.id, parent_prompt_sha256: sha(row.prompt),
        parent_response_sha256: sha(row.response), task: 'complete_documented_code_parameters',
        complete_source_header: parsed.original_full_header,
        parser_node_type: parsed.node_type, parser_context: parsed.context,
        parameter_header_origin: 'Exact parsed source header before the outer function/method body; original code response and source metadata unchanged.',
        external_model_teacher_used: false};
      fs.writeSync(fd, JSON.stringify(variant) + '\n'); rows++;
    }
  } finally { fs.closeSync(fd); }
  const actual = digest.digest('hex');
  if (actual !== stage.data_sha256 || actual !== audit.data_sha256 || rows !== audit.counts.parameter_text_cut)
    throw Error('Actually scanned original source or full correction population differs');
  const receipt = {prepared_at: new Date().toISOString(), candidates: rows,
    candidates_sha256: sha(fs.readFileSync(output)), source_data_sha256: actual,
    interface_audit_sha256: sha(fs.readFileSync(auditPath)), parser_version: parserVersion,
    toolchain_manifest_sha256: sha(fs.readFileSync(toolsFile)),
    implementation_sha256: sha(fs.readFileSync(__filename)),
    helper_sha256: sha(fs.readFileSync(path.join(__dirname, 'js_instruction_header.cjs'))),
    original_training_data_changed: false, token_input_validation_pending: true,
    training_admitted: false, trained: false, goal_achieved: false};
  fs.writeFileSync(path.join(directory, 'candidate-receipt.json'), JSON.stringify(receipt, null, 2) + '\n', {flag: 'wx'});
  console.log(JSON.stringify(receipt));
}
main().catch(error => {console.error(error.stack); process.exitCode = 1;});
