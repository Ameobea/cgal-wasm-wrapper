// Standalone microbenchmark: is a Wasm-SIMD Interval_nt worth porting (H2b)?
// Compares the post-H1 SCALAR interval multiply/add (round-to-nearest + nextUp, no barriers)
// against a SIMD version that stores {-inf, sup} in a v128 and applies one SIMD nextUp.
// Verifies bit-exact parity, then times a long mul/add chain. Build for wasm:
//   emcc -O3 -msimd128 -msse4.2 -DNDEBUG microbench_interval.cpp -o mbi.js && node mbi.js
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <wasm_simd128.h>
#include <chrono>

static inline int64_t as_i64(double x){int64_t i;std::memcpy(&i,&x,8);return i;}
static inline double as_f64(int64_t i){double x;std::memcpy(&x,&i,8);return x;}

// --- scalar nextUp (CGAL's branchless patched version) ---
static inline double nextUp(double x){
  int64_t u=as_i64(x);
  if(((uint64_t)(u<<1))==0) return as_f64(1);
  int64_t sign=u<0; int64_t delta=1-sign*2; return as_f64(u+delta);
}

// ============ SCALAR interval (post-H1: nextUp, no opacify barriers) ============
struct IS { double inf, sup; };
static inline double iup(double d){ return nextUp(d); }
static inline IS mul_scalar(const IS&a,const IS&b){
  // mirrors CGAL Interval_nt scalar operator* sign-case logic
  if(a.inf>=0.0){
    double aa=a.inf,bb=a.sup;
    if(bb<=0.) return {0.,0.};
    if(b.inf<0.0){ aa=bb; if(b.sup<0.0) bb=a.inf; }
    double r=(b.sup==0)?0.:iup(bb*b.sup);
    return { -iup(aa*-b.inf), r };
  } else if(a.sup<=0.0){
    double aa=a.sup,bb=a.inf;
    if(b.inf<0.0){ aa=bb; if(b.sup<=0.0) bb=a.sup; }
    else if(b.sup<=0) return {0.,0.};
    return { -iup(-bb*b.sup), iup(-aa*-b.inf) };
  } else {
    if(b.inf>=0.0){ if(b.sup<=0.0) return {0.,0.};
      return { -iup(-a.inf*b.sup), iup(a.sup*b.sup) }; }
    if(b.sup<=0.0){ return { -iup(a.sup*-b.inf), iup(-a.inf*-b.inf) }; }
    double t1=iup(-a.inf*b.sup),t2=iup(a.sup*-b.inf),t3=iup(-a.inf*-b.inf),t4=iup(a.sup*b.sup);
    return { -std::fmax(t1,t2), std::fmax(t3,t4) };
  }
}
static inline IS add_scalar(const IS&a,const IS&b){
  return { -iup(-a.inf + -b.inf), iup(a.sup + b.sup) };
}

// ============ SIMD interval: store v = {-inf, sup} in f64x2 lane0,lane1 ============
// nextUp on both lanes, branchless, ±0 -> smallest denormal.
static inline v128_t simd_nextUp(v128_t x){
  v128_t i=x; // reinterpret bits
  v128_t shl1 = wasm_i64x2_shl(i,1);
  v128_t zmask = wasm_i64x2_eq(shl1, wasm_i64x2_const(0,0));
  v128_t sign = wasm_i64x2_shr(i,63);                 // all-ones if negative
  v128_t delta = wasm_v128_bitselect(wasm_i64x2_const(-1,-1), wasm_i64x2_const(1,1), sign);
  v128_t normal = wasm_i64x2_add(i, delta);
  v128_t res = wasm_v128_bitselect(wasm_i64x2_const(1,1), normal, zmask);
  return res;
}
struct IV { v128_t v; }; // v = {-inf, sup}
static inline IV to_iv(const IS&s){ return { wasm_f64x2_make(-s.inf, s.sup) }; }
static inline IS from_iv(const IV&x){ return { -wasm_f64x2_extract_lane(x.v,0), wasm_f64x2_extract_lane(x.v,1) }; }
static inline IV add_simd(const IV&a,const IV&b){
  return { simd_nextUp(wasm_f64x2_add(a.v,b.v)) };
}
// brutal multiply (CGAL SSE2 "compute all products" variant) adapted to wasm + nextUp.
static inline IV mul_simd(const IV&A,const IV&B){
  v128_t aa=A.v;                                  // {-ai, as}
  v128_t bb=B.v;                                  // {-bi, bs}
  v128_t negmask0 = wasm_i64x2_make((int64_t)0x8000000000000000LL,0);   // {-0,+0}
  v128_t negmask1 = wasm_i64x2_make((int64_t)0x8000000000000000LL,(int64_t)0x8000000000000000LL); // {-0,-0}
  v128_t ax = wasm_i64x2_shuffle(aa,aa,1,0);      // {as,-ai}
  v128_t ap = wasm_v128_xor(ax, negmask1);        // {-as, ai}
  v128_t bz = wasm_v128_xor(bb, negmask0);        // {bi, bs}
  v128_t c  = wasm_i64x2_shuffle(bz,bz,1,0);      // {bs, bi}
  v128_t big = wasm_f64x2_const(1.7976931348623157e308, 1.7976931348623157e308);
  v128_t x1 = wasm_f64x2_mul(aa,bz);              // {-ai*bi, as*bs}
  v128_t x2 = wasm_f64x2_mul(aa,c);               // {-ai*bs, as*bi}
  x2 = wasm_f64x2_pmin(x2,big);                   // kill NaN (0*inf)
  v128_t x3 = wasm_f64x2_mul(ap,bz);              // {-as*bi, ai*bs}
  v128_t x4 = wasm_f64x2_mul(ap,c);               // {-as*bs, ai*bi}
  x4 = wasm_f64x2_pmin(x4,big);
  v128_t y1 = wasm_f64x2_pmax(x1,x2);
  v128_t y2 = wasm_f64x2_pmax(x3,x4);
  v128_t r  = wasm_f64x2_pmax(y1,y2);
  return { simd_nextUp(r) };
}

int main(int argc,char**argv){
  int N = argc>1?atoi(argv[1]):20000000;
  // deterministic random intervals spanning all sign cases
  uint64_t s=0x12345; auto rnd=[&](){ s^=s<<13;s^=s>>7;s^=s<<17; return (double)((int64_t)(s>>11))*(1.0/(1LL<<52)) - 2.0; };
  std::vector<IS> data(4096);
  for(auto&d:data){ double a=rnd(),b=rnd(); d.inf=std::fmin(a,b); d.sup=std::fmax(a,b); }

  // parity check: every pair, scalar vs simd must be bit-identical
  int mism=0;
  for(size_t i=0;i<data.size();i++) for(size_t j=0;j<32;j++){
    IS sc=mul_scalar(data[i],data[(i+j)&4095]);
    IS sv=from_iv(mul_simd(to_iv(data[i]),to_iv(data[(i+j)&4095])));
    if(as_i64(sc.inf)!=as_i64(sv.inf)||as_i64(sc.sup)!=as_i64(sv.sup)){ if(mism<5) fprintf(stderr,"MISMATCH mul [%g,%g]x[%g,%g]: sc[%g,%g] sv[%g,%g]\n",data[i].inf,data[i].sup,data[(i+j)&4095].inf,data[(i+j)&4095].sup,sc.inf,sc.sup,sv.inf,sv.sup); mism++; }
    IS ac=add_scalar(data[i],data[(i+j)&4095]);
    IS av=from_iv(add_simd(to_iv(data[i]),to_iv(data[(i+j)&4095])));
    if(as_i64(ac.inf)!=as_i64(av.inf)||as_i64(ac.sup)!=as_i64(av.sup)){ if(mism<5) fprintf(stderr,"MISMATCH add\n"); mism++; }
  }
  fprintf(stderr,"parity: %s (%d mismatches)\n", mism?"FAIL":"OK", mism);

  // timing: chained mul+add accumulation
  auto now=[](){ return std::chrono::high_resolution_clock::now(); };
  double bestS=1e30,bestV=1e30;
  for(int rep=0;rep<5;rep++){
    { IS acc={1.0,1.0001}; auto t0=now();
      for(int k=0;k<N;k++){ acc=mul_scalar(acc,data[k&4095]); acc=add_scalar(acc,data[(k*7+1)&4095]);
        if((k&1023)==0) acc={1.0,1.0001}; }
      auto t1=now(); double ms=std::chrono::duration<double,std::milli>(t1-t0).count(); bestS=std::fmin(bestS,ms);
      volatile double sink=acc.sup;(void)sink; }
    { IV acc=to_iv({1.0,1.0001}); auto t0=now();
      for(int k=0;k<N;k++){ acc=mul_simd(acc,to_iv(data[k&4095])); acc=add_simd(acc,to_iv(data[(k*7+1)&4095]));
        if((k&1023)==0) acc=to_iv({1.0,1.0001}); }
      auto t1=now(); double ms=std::chrono::duration<double,std::milli>(t1-t0).count(); bestV=std::fmin(bestV,ms);
      volatile double sink=from_iv(acc).sup;(void)sink; }
  }
  printf("scalar=%.1fms  simd=%.1fms  speedup=%.2fx  (N=%d)\n", bestS, bestV, bestS/bestV, N);
  return mism?1:0;
}
