// Run with: node tests/test_viewer.cjs
// Exercise the actual viewer script with a minimal DOM, without a browser dependency.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const context = new Proxy({}, { get: () => () => {} });
function element() {
  return {
    value: '', textContent: '', innerHTML: '', width: 900, height: 600,
    classList: { toggle() {} }, appendChild() {}, append() {},
    addEventListener() {}, setCustomValidity() {}, reportValidity() {},
    getContext: () => context,
    getBoundingClientRect: () => ({ width: 900, height: 600, left: 0, top: 0 }),
  };
}
const elements = new Map();
const sandbox = vm.createContext({
  document: {
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, element());
      return elements.get(id);
    },
    createElement: element,
  },
  devicePixelRatio: 1,
  ResizeObserver: class { observe() {} },
});
const nodes = [1, 2, 3].map(id => ({
  gid: String(id), cluster: id === 1 ? 1 : 2, score: .5,
  role: 'peripheral', evidence: 'test', depth: 1, seed: false,
  incoming: 1, outgoing: 1, inKzt: 100, outKzt: 100, seedReach: 1,
}));
const edges = [{ src: '1', dst: '2', amount: 100, count: 1 },
  { src: '2', dst: '3', amount: 100, count: 1 }];
const html = fs.readFileSync(path.join(__dirname, '..', 'viewer.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1]
  .replace('/* GRAPH_DATA */ null', JSON.stringify({ nodes, edges }));
vm.runInContext(script, sandbox);
vm.runInContext("select('1'); setMode('two');", sandbox);
elements.get('cluster').onchange({ target: { value: '2' } });
assert.equal(vm.runInContext('selected', sandbox), null);
assert.equal(vm.runInContext('mode', sandbox), 'overview');
assert.equal(vm.runInContext("visible.map(n => n.gid).join(',')", sandbox), '2,3');
assert.equal(vm.runInContext('shownEdges.length', sandbox), 1);
assert.equal(elements.get('search').value, '');
vm.runInContext("select('2');", sandbox);
assert.equal(vm.runInContext('selected', sandbox), '2');
assert.equal(vm.runInContext('cluster', sandbox), '2');
assert.equal(vm.runInContext('mode', sandbox), 'one');
console.log('Viewer cluster regression passed');
