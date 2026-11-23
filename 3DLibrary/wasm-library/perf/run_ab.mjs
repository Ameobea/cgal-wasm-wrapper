// A/B-run two (or more) emcc-built alpha_wrap bench variants in Node/V8 across a set of
// cases, print best-of-N ms, speedup, and a parity check on the output fingerprint.
// Usage: node run_ab.mjs [reps] [variantA variantB ...]   (variant = nb_<name>.js basename "<name>")
import { execFileSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
const __dirname = path.dirname(fileURLToPath(import.meta.url));

const reps = parseInt(process.argv[2] || "5", 10);
const variants = process.argv.slice(3).length ? process.argv.slice(3) : ["baseline", "nobarriers"];

// case = [label, gen, npoints, alphaInv, offInv]
const CASES = [
  ["torus/1550/a100", "torus_shell", 1550, 100, 100],
  ["torus/4000/a120", "torus_shell", 4000, 120, 120],
  ["sphere/8000/a140", "sphere_shell", 8000, 140, 250],
  ["noise/12000/a160", "noise_shell", 12000, 160, 250],
];

function run(variant, c) {
  const js = path.join(__dirname, `nb_${variant}.js`);
  const args = [js, c[1], String(c[2]), String(c[3]), String(c[4]), String(reps)];
  const out = execFileSync("node", args, { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] });
  const last = out.trim().split("\n").pop().split(/\s+/);
  return { ms: parseFloat(last[0]), fp: last[1], verts: +last[2], faces: +last[3] };
}

console.log(`reps=${reps} variants=${variants.join(",")}\n`);
const header = ["case", ...variants.map((v) => v.slice(0, 10)), "speedup", "parity"].join("\t");
console.log(header);
for (const c of CASES) {
  const res = variants.map((v) => run(v, c));
  const base = res[0].ms;
  const speed = res.map((r) => ((base - r.ms) / base) * 100);
  const fps = new Set(res.map((r) => r.fp));
  const parity = fps.size === 1 ? "OK" : "MISMATCH " + res.map((r) => r.fp.slice(0, 8)).join("/");
  const cells = [
    c[0],
    ...res.map((r) => r.ms.toFixed(0)),
    res.length > 1 ? speed[1].toFixed(1) + "%" : "-",
    parity + ` (${res[0].verts}v/${res[0].faces}f)`,
  ];
  console.log(cells.join("\t"));
}
