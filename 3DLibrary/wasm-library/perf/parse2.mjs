import fs from "node:fs";
const d = JSON.parse(fs.readFileSync("alpha-wrap.json","utf8"));
const ev = d.traceEvents;
const profiles = new Map();
const k=e=>e.pid+":"+e.tid;
for (const e of ev){
  if(e.name==="Profile"){ profiles.set(k(e),{nodes:new Map(),samples:[],deltas:[]}); }
  else if(e.name==="ProfileChunk"){
    let p=profiles.get(k(e)); if(!p){p={nodes:new Map(),samples:[],deltas:[]};profiles.set(k(e),p);}
    const cp=e.args?.data?.cpuProfile;
    if(cp?.nodes) for(const n of cp.nodes) p.nodes.set(n.id,n);
    if(cp?.samples) p.samples.push(...cp.samples);
    if(e.args?.data?.timeDeltas) p.deltas.push(...e.args.data.timeDeltas);
  }
}
function dump(key){
  const p=profiles.get(key); if(!p){console.log("no profile",key);return;}
  const self=new Map();
  for(let i=0;i<p.samples.length;i++){ const id=p.samples[i]; self.set(id,(self.get(id)||0)+(p.deltas[i]||0)); }
  const byFn=new Map(); let total=0, idle=0;
  for(const [id,t] of self){
    const cf=p.nodes.get(id)?.callFrame||{};
    let nm=cf.functionName||"(anon)"; if(nm==="")nm="(program/idle)";
    const url=(cf.url||"");
    const isWasm = url.includes(".wasm")||/wasm-function/.test(nm)||cf.codeType==="wasm";
    const label=(isWasm?"[wasm] ":"[js]   ")+nm;
    if(/idle|program/.test(nm)){ idle+=t; }
    byFn.set(label,(byFn.get(label)||0)+t); total+=t;
  }
  console.log(`\n===== profile ${key}  total=${(total/1000).toFixed(0)}ms idle/prog=${(idle/1000).toFixed(0)}ms active=${((total-idle)/1000).toFixed(0)}ms =====`);
  const arr=[...byFn.entries()].filter(([n])=>!/\(program\/idle\)/.test(n)).sort((a,b)=>b[1]-a[1]);
  for(const [nm,t] of arr.slice(0,40)) console.log(`${(t/1000).toFixed(1).padStart(8)}ms ${(100*t/(total-idle)).toFixed(1).padStart(5)}%a  ${nm.slice(0,95)}`);
}
dump("135586:21");
dump("135586:47");
