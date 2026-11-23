import fs from "node:fs";
const d=JSON.parse(fs.readFileSync("alpha-wrap.json","utf8"));
const ev=d.traceEvents; const k=e=>e.pid+":"+e.tid;
const profiles=new Map();
for(const e of ev){
  if(e.name==="Profile")profiles.set(k(e),{nodes:new Map(),samples:[],deltas:[]});
  else if(e.name==="ProfileChunk"){let p=profiles.get(k(e));if(!p){p={nodes:new Map(),samples:[],deltas:[]};profiles.set(k(e),p);}
    const cp=e.args?.data?.cpuProfile;
    if(cp?.nodes)for(const n of cp.nodes)p.nodes.set(n.id,n);
    if(cp?.samples)p.samples.push(...cp.samples);
    if(e.args?.data?.timeDeltas)p.deltas.push(...e.args.data.timeDeltas);}
}
const p=profiles.get("135586:21");
// build children map for inclusive time
const self=new Map();
for(let i=0;i<p.samples.length;i++){const id=p.samples[i];self.set(id,(self.get(id)||0)+(p.deltas[i]||0));}
// inclusive: walk parent chain
function incl(){
  const inc=new Map();
  for(const [id,n] of p.nodes){ inc.set(id,0); }
  for(const [id,t] of self){
    let cur=id;
    const seen=new Set();
    while(cur!=null && p.nodes.has(cur) && !seen.has(cur)){ seen.add(cur); inc.set(cur,(inc.get(cur)||0)+t); cur=p.nodes.get(cur).parent; }
  }
  return inc;
}
const inc=incl();
// categorize self-time into buckets via regex on function name
const buckets=[
 ["Interval arithmetic (directed rounding / nextUp)", /Interval_nt|INTERN_INTERVAL_NT|circumcenterC3|orientationC3|determinant<CGAL::Interval|Construct_vector_3<CGAL::Simple_cartesian<CGAL::Interval/],
 ["Static filters (double, no rounding)", /Static_filters_predicates/],
 ["AABB tree traversal / bbox", /AABB_node|AABB_traits|Construct_bbox|Do_intersect.*Alpha_wrap_AABB|Alpha_wrap_AABB.*Do_intersect/],
 ["Filtered_predicate machinery / Uncertain", /Filtered_predicate|make_certain|Uncertain|Same_uncertainty/],
 ["Alpha_wrap core / heap / triangulation DS", /Alpha_wrap_3|Tetrahedron_with_outside|pairing_heap|CC_iterator|Compact_container|Triangulation|neighbor_search|polygon_soup/],
 ["Geoscript (point gen/filter/eval)", /geoscript|EvalCtx|pratt|pest|parse_/],
 ["Alloc", /malloc|free|operator new|reserve/],
];
const bsum=buckets.map(()=>0); let other=0, idle=0, total=0;
for(const [id,t] of self){
  const nm=p.nodes.get(id)?.callFrame?.functionName||"";
  total+=t;
  if(/^$|idle|program/.test(nm)&&!/geoscript/.test(nm)){ if(nm===""||/idle|program/.test(nm)){idle+=t;continue;} }
  let hit=-1;
  for(let b=0;b<buckets.length;b++){ if(buckets[b][1].test(nm)){hit=b;break;} }
  if(hit>=0) bsum[hit]+=t; else other+=t;
}
const active=total-idle;
console.log(`active compute = ${(active/1000).toFixed(0)}ms (total ${(total/1000).toFixed(0)}ms, idle ${(idle/1000).toFixed(0)}ms)\n`);
console.log("=== self-time by category (% of active) ===");
const rows=buckets.map((b,i)=>[b[0],bsum[i]]).concat([["Other (uncategorized)",other]]).sort((a,b)=>b[1]-a[1]);
for(const [nm,t] of rows) console.log(`${(t/1000).toFixed(1).padStart(7)}ms ${(100*t/active).toFixed(1).padStart(5)}%  ${nm}`);
// inclusive time of key entry points
console.log("\n=== inclusive time of notable frames (max over nodes w/ that name) ===");
const names=["alphaWrapPointCloud(std","CGAL::alpha_wrap","Alpha_wrap_3","geoscript_repl_eval"];
const incByName=new Map();
for(const [id,t] of inc){ const nm=p.nodes.get(id)?.callFrame?.functionName||""; if(!nm)continue; incByName.set(nm,Math.max(incByName.get(nm)||0,t)); }
for(const want of names){ for(const [nm,t] of incByName){ if(nm.includes(want)){ console.log(`${(t/1000).toFixed(0).padStart(5)}ms incl  ${nm.slice(0,80)}`); break; } } }
