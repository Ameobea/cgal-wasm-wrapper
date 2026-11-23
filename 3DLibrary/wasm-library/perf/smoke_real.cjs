const factory = require("../build/index_cjs.cjs");
function torusShell(n){let s=0xc0ffee>>>0;const rnd=()=>{s^=s<<13;s^=s>>>17;s^=s<<5;s>>>=0;return s/4294967296;};
  const R=8,band=2,B=R*1.1,out=[];while(out.length/3<n){const x=(rnd()*2-1)*B,y=(rnd()*2-1)*B,z=(rnd()*2-1)*B;
  const a=Math.sqrt(x*x+z*z)-R;if(Math.sqrt(y*y+a*a)<band)out.push(x,y,z);}return out;}
factory().then((M)=>{
  const pts=torusShell(1550);
  const vec=new M.vector$float$();vec.resize(pts.length,0);
  const ptr=vec.data();M.HEAPF32.subarray(ptr/4,ptr/4+pts.length).set(pts);
  const t0=performance.now();
  const w=M.alphaWrapPointCloud(vec,1/100,1/100);
  const t1=performance.now();
  const nv=w.getVertices().size(),nf=w.getIndices().size()/3;
  console.log(`real-artifact alphaWrapPointCloud: ${(t1-t0).toFixed(0)} ms  verts=${nv/3} faces=${nf}`);
  const ok=nv>0&&nf>100;console.log(ok?"SMOKE OK":"SMOKE FAIL");
  vec.delete();w.delete();process.exit(ok?0:1);
}).catch(e=>{console.error("LOAD/RUN ERROR:",e);process.exit(2);});
