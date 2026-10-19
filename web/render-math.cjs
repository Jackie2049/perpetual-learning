// Build-time rendering: invalid formulas must stop publication.
'use strict';
const fs = require('node:fs');
const katex = require('./vendor/katex/katex.min.js');
const formulas = JSON.parse(fs.readFileSync(0, 'utf8'));
const output = formulas.map(({source, display}) => katex.renderToString(source, {
  displayMode: display, throwOnError: true, trust: false,
  output: 'htmlAndMathml', macros: {},
}));
process.stdout.write(JSON.stringify(output));
