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
    checked: false,
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
  role: id === 3 ? 'terminal' : 'peripheral', evidence: 'test', depth: 1,
  seed: id === 1, boundary: id === 3,
  incoming: 1, outgoing: 1, inKzt: 100, outKzt: 100, seedReach: 1,
  cycles: 0, syncPayers: 1, pHidden: 0, attention: id === 2 ? 'flag' : 'нет',
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
elements.get('cluster').value = 'all';
vm.runInContext("cluster='all'; selected=null; document.getElementById('roleFilter').value='terminal'; layout();", sandbox);
assert.equal(vm.runInContext("visible.map(n => n.gid).join(',')", sandbox), '3');
vm.runInContext("document.getElementById('roleFilter').value='all'; document.getElementById('flaggedOnly').checked=true; layout();", sandbox);
assert.equal(vm.runInContext("visible.map(n => n.gid).join(',')", sandbox), '2');
vm.runInContext("select('1');", sandbox);
assert.equal(vm.runInContext("document.getElementById('flaggedOnly').checked", sandbox), false);
assert.equal(vm.runInContext('selected', sandbox), '1');
console.log('Viewer cluster regression passed');
