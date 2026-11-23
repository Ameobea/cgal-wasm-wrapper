import fs from "node:fs";
const d = JSON.parse(fs.readFileSync("alpha-wrap.json","utf8"));
const ev = d.traceEvents;

// Reconstruct CPU profiles per (pid,tid). Chrome emits 'Profile' then 'ProfileChunk'.
const profiles = new Map(); // key pid:tid -> {nodes:Map, samples:[], deltas:[]}
function key(e){ return e.pid+":"+e.tid; }
for (const e of ev) {
  if (e.name === "Profile") {
    profiles.set(key(e), {nodes:new Map(), samples:[], deltas:[]});
  } else if (e.name === "ProfileChunk") {
    let p = profiles.get(key(e));
    if (!p) { p = {nodes:new Map(), samples:[], deltas:[]}; profiles.set(key(e), p); }
    const cp = e.args?.data?.cpuProfile;
    if (cp?.nodes) for (const n of cp.nodes) p.nodes.set(n.id, n);
    if (cp?.samples) p.samples.push(...cp.samples);
    if (e.args?.data?.timeDeltas) p.deltas.push(...e.args.data.timeDeltas);
  }
}

// thread names
const tname = new Map();
for (const e of ev) if (e.name==="thread_name") tname.set(key(e), e.args.name);

// Find profile with most total time
const ranked = [];
for (const [k,p] of profiles) {
  let tot = 0;
  for (const dt of p.deltas) tot += Math.max(0,dt);
  ranked.push([k, tot, p.samples.length, tname.get(k)||"?"]);
}
ranked.sort((a,b)=>b[1]-a[1]);
console.log("=== profiles by total sampled time (us) ===");
for (const [k,tot,ns,nm] of ranked.slice(0,8)) console.log(`${k.padEnd(14)} ${(tot/1000).toFixed(1).padStart(9)}ms  samples=${ns}  ${nm}`);

// Pick the hottest worker thread
const [hotKey] = ranked[0];
const p = profiles.get(hotKey);
console.log(`\n=== hottest profile: ${hotKey} (${tname.get(hotKey)}) ===`);

// self time per node
const selfByNode = new Map();
for (let i=0;i<p.samples.length;i++){
  const id = p.samples[i]; const dt = p.deltas[i]||0;
  selfByNode.set(id,(selfByNode.get(id)||0)+dt);
}
// aggregate by function name
const byFn = new Map();
let total=0;
for (const [id,t] of selfByNode){
  const n = p.nodes.get(id);
  const cf = n?.callFrame || {};
  let nm = cf.functionName || "(anonymous)";
  if (nm==="") nm="(program/idle)";
  // tag wasm vs js
  const url = cf.url||"";
  const tag = (cf.codeType==="JS")?"":(url.includes("wasm")||cf.scriptId===0?"":"");
  byFn.set(nm,(byFn.get(nm)||0)+t);
  total+=t;
}
console.log(`total self-time: ${(total/1000).toFixed(1)}ms across ${selfByNode.size} nodes\n`);
const arr=[...byFn.entries()].sort((a,b)=>b[1]-a[1]);
console.log("=== top 45 functions by self-time ===");
for (const [nm,t] of arr.slice(0,45)){
  console.log(`${(t/1000).toFixed(1).padStart(8)}ms ${(100*t/total).toFixed(1).padStart(5)}%  ${nm.slice(0,90)}`);
}
