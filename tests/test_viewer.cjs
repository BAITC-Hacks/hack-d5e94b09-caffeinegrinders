// Run with: node tests/test_viewer.cjs
// Exercise the actual viewer script with a minimal DOM, without a browser dependency.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const context = new Proxy({}, { get: () => () => {} });
function element() {
  const listeners = new Map();
  return {
    value: '', textContent: '', innerHTML: '', width: 900, height: 600,
    checked: false,
    classList: { toggle() {} }, appendChild() {}, append() {},
    addEventListener(type, listener) {
      if (!listeners.has(type)) listeners.set(type, []);
      listeners.get(type).push(listener);
    },
    dispatchEvent(event) {
      for (const listener of listeners.get(event.type) || []) {
        listener({ ...event, target: this });
      }
    },
    setCustomValidity() {}, reportValidity() {},
    getContext: () => context,
    getBoundingClientRect: () => ({ width: 900, height: 600, left: 0, top: 0 }),
  };
}
const elements = new Map();
const storage = new Map();
const sandbox = vm.createContext({
  document: {
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, element());
      return elements.get(id);
    },
    createElement: element,
  },
  localStorage: {
    getItem(key) { return storage.has(key) ? storage.get(key) : null; },
    setItem(key, value) { storage.set(key, String(value)); },
  },
  devicePixelRatio: 1,
  ResizeObserver: class { observe() {} },
});
const nodes = [1, 2, 3].map(id => ({
  gid: String(id), cluster: id === 1 ? 1 : 2, score: .5,
  role: id === 3 ? 'terminal' : 'peripheral', evidence: 'test', depth: 1,
  seed: id === 1, boundary: id === 3,
  incoming: 1, outgoing: 1, inKzt: 100, outKzt: 100, seedReach: 1,
  inTx: 1, outTx: 1, nearThresholdIn: 0,
  cycles: 0, syncPayers: 1, pHidden: id === 3 ? .6 : 0, attention: id === 2 ? 'flag' : 'нет',
}));
const edges = [{ src: '1', dst: '2', amount: 100, count: 1 },
  { src: '2', dst: '3', amount: 100, count: 1 }];
const routes = [{ route_id: 1, kind: 'repeated', hops: 2, path: '1 → 2 → 3',
  relay_days: 2, forwarded_kzt: 100, first_date: '2026-07-01', last_date: '2026-07-02', n_seed: 1 }];
const html = fs.readFileSync(path.join(__dirname, '..', 'viewer.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1]
  .replace('/* GRAPH_DATA */ null', JSON.stringify({ nodes, edges, routes }));
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
vm.runInContext("addCaseNode('1'); caseState.notes['1']='watch'; saveCase();", sandbox);
vm.runInContext("addCaseNode('2');", sandbox);
assert.equal(JSON.parse(storage.get('moneyGraphCase:v1')).gids.join(','), '1,2');
assert.equal(JSON.parse(storage.get('moneyGraphCase:v1')).notes['1'], 'watch');
assert.equal(vm.runInContext("sharedCase('downstream').map(x=>x.id).includes('3')", sandbox), true);
const report = vm.runInContext('caseReport()', sandbox);
assert.match(report, /# Рабочий отчёт/);
assert.match(report, /## 1/);
assert.match(report, /Evidence: test/);
assert.match(report, /Заметка аналитика: watch/);
assert.match(vm.runInContext("dataRequestsForNode(byId.get('3')).join(';')", sandbox), /продлить обход/);
assert.equal(vm.runInContext("routesForNode('2')[0].path", sandbox), '1 → 2 → 3');
elements.get('clearCase').onclick();
assert.equal(vm.runInContext('caseState.gids.length', sandbox), 0);
console.log('Viewer cluster regression passed');

// Changing the actual input must clear a now-hidden selection and its edges.
// The inclusive score boundary must restore the nodes without a stale selection.
const minScore = elements.get('minScore');
minScore.value = '0.51';
minScore.dispatchEvent({ type: 'input' });
assert.equal(vm.runInContext('selected', sandbox), null);
assert.equal(vm.runInContext('mode', sandbox), 'overview');
assert.equal(vm.runInContext('visible.length', sandbox), 0);
assert.equal(vm.runInContext('shownEdges.length', sandbox), 0);
assert.equal(elements.get('search').value, '');
assert.equal(elements.get('card').textContent, 'Выберите узел с текущими фильтрами.');
assert.equal(elements.get('connections').textContent, 'Выберите узел.');
minScore.value = '0.5';
minScore.dispatchEvent({ type: 'input' });
assert.equal(vm.runInContext("visible.map(n => n.gid).join(',')", sandbox), '1,2,3');
assert.equal(vm.runInContext('shownEdges.length', sandbox), 2);
assert.equal(vm.runInContext('selected', sandbox), null);
assert.equal(vm.runInContext('mode', sandbox), 'overview');
console.log('Viewer minimum-priority filter regression passed');
