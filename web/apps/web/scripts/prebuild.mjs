#!/usr/bin/env node

import { existsSync, mkdirSync, rmSync, cpSync, readdirSync, readFileSync, writeFileSync } from "node:fs"
import { spawnSync } from "node:child_process"
import path from "node:path"
import process from "node:process"
import { fileURLToPath } from "node:url"

const DEFAULT_REPO = "https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace"
const repo = process.env.FIB_DATASET_REPO ?? DEFAULT_REPO
const ref = process.env.FIB_DATASET_REF ?? "origin/main"
const token = process.env.GH_TOKEN

const datasetRootEnv = process.env.FIB_DATASET_PATH ?? "/tmp/flashinfer-trace"
const datasetRoot = path.isAbsolute(datasetRootEnv)
  ? datasetRootEnv
  : path.resolve(process.cwd(), datasetRootEnv)

function withAuth(url) {
  if (!token) return url
  if (!url.startsWith("https://")) return url
  const withoutProtocol = url.slice("https://".length)
  return `https://x-access-token:${token}@${withoutProtocol}`
}

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    stdio: "inherit",
    env: { ...process.env, GIT_LFS_SKIP_SMUDGE: "1" },
    ...options,
  })

  if (result.status !== 0) {
    throw new Error(`Command failed: ${command} ${args.join(" ")}`)
  }
}

function ensureDataset() {
  console.log(`[prebuild] Preparing flashinfer-trace dataset in ${datasetRoot}`)
  const authRepo = withAuth(repo)
  mkdirSync(path.dirname(datasetRoot), { recursive: true })

  if (!existsSync(datasetRoot)) {
    console.log(`[prebuild] ${datasetRoot} does not exist; cloning from ${repo}:${ref}`)
    run("git", ["clone", "--depth=1", authRepo, datasetRoot])
    run("git", ["-C", datasetRoot, "reset", "--hard", ref])
    return
  }

  console.log("[prebuild] Found existing dataset, skipping")
}

/**
 * Sphinx Documentation
 */
function resolvePublicDir(repoRoot) {
  const publicDir = path.join(repoRoot, "web", "apps", "web", "public")
  mkdirSync(publicDir, { recursive: true })
  return publicDir
}

function runCmd(cmd, args, options = {}) {
  const r = spawnSync(cmd, args, {
    encoding: "utf8",
    maxBuffer: 16 * 1024 * 1024,
    ...options,
  })
  if (r.stdout) process.stdout.write(r.stdout)
  if (r.stderr) process.stderr.write(r.stderr)
  if (r.status !== 0) throw new Error(`Command failed: ${cmd} ${args.join(" ")}`)
}

function havePython() {
  const r = spawnSync("python3", ["--version"], { stdio: "ignore" })
  return r.status === 0
}

function findRepoRoot() {
  const __filename = fileURLToPath(import.meta.url)
  const starts = [process.cwd(), path.dirname(__filename)]
  for (const start of starts) {
    let dir = start
    for (let i = 0; i < 10; i++) {
      const cand = path.join(dir, "docs", "api")
      if (existsSync(cand)) return dir
      const parent = path.dirname(dir)
      if (parent === dir) break
      dir = parent
    }
  }
  throw new Error("Cannot locate repo root containing docs/api")
}


function injectIndexBases(rootDir, mountPath) {
  const mount = `/${mountPath.replace(/^\/+|\/+$/g, "")}/`;

  const stack = [rootDir];
  while (stack.length) {
    const dir = stack.pop();
    const ents = readdirSync(dir, { withFileTypes: true });

    for (const e of ents) if (e.isDirectory()) stack.push(path.join(dir, e.name));

    const hasIndex = ents.some(e => e.isFile() && e.name.toLowerCase() === "index.html");
    if (!hasIndex) continue;

    const relDirFs = path.relative(rootDir, dir);                 // e.g. "rst/apply"
    const relDir = relDirFs ? relDirFs.split(path.sep).join("/") + "/" : "";
    const baseHref = mount + relDir;                              // e.g. "/docs/api/python/rst/apply/"

    const p = path.join(dir, "index.html");
    let html = readFileSync(p, "utf8");
    html = html.replace(/<base\b[^>]*>\s*/i, "");
    html = html.replace(/<head([^>]*)>/i, `<head$1>\n  <base href="${baseHref}">`);
    writeFileSync(p, html, "utf8");
  }
}

function buildSphinxDocs() {
  console.log("[prebuild] Building Sphinx docs...")

  const repoRoot = findRepoRoot()
  const SPHINX_SRC = path.join(repoRoot, "docs", "api")
  const REQ_FILE = path.join(SPHINX_SRC, "requirements.txt")
  const SPHINX_DEPS = path.join(SPHINX_SRC, ".sphinx-deps")
  const SPHINX_BUILD = path.join(SPHINX_SRC, "_build", "dirhtml")

  console.log("[prebuild] Repo root:", repoRoot)
  console.log("[prebuild] Sphinx source:", SPHINX_SRC)

  if (!havePython()) {
    console.warn("[prebuild] python3 not found; skipping Sphinx build")
    return
  }
  if (!existsSync(REQ_FILE)) {
    throw new Error(`[prebuild] Missing ${REQ_FILE}. Create it with Sphinx deps.`)
  }

  // 1) Install dependencies to local directory
  mkdirSync(SPHINX_DEPS, { recursive: true })
  runCmd("python3", [
    "-m", "pip", "install",
    "--disable-pip-version-check",
    "--upgrade",
    "-r", "requirements.txt",
    "--target", ".sphinx-deps",
  ], { cwd: SPHINX_SRC })

  // 2) Build dirhtml
  rmSync(SPHINX_BUILD, { recursive: true, force: true })
  runCmd("python3", [
    "-m", "sphinx",
    "-T", "-v",
    "-b", "dirhtml",
    ".", "_build/dirhtml",
  ], {
    cwd: SPHINX_SRC,
    env: {
      ...process.env,
      PYTHONUTF8: "1",
      PYTHONIOENCODING: "utf-8",
      PYTHONPATH: [
        SPHINX_DEPS,
        repoRoot,
        process.env.PYTHONPATH || "",
      ].filter(Boolean).join(path.delimiter),
      BUILD_DOC: "1",
      FLASHINFER_BUILDING_DOCS: "1",
    },
  })

  // 3) Copy to Next public/docs/api/python
  const PUBLIC_DIR = resolvePublicDir(repoRoot)
  const TARGET = path.join(PUBLIC_DIR, "docs", "api", "python")
  rmSync(TARGET, { recursive: true, force: true })
  cpSync(SPHINX_BUILD, TARGET, { recursive: true })
  console.log("[prebuild] Sphinx docs staged at", TARGET)

  // 4) Inject <base href> into HTML
  const baseHref = `/docs/api/python/`;
  injectIndexBases(TARGET, baseHref);
  console.log("[prebuild] Injected <base href=\"%s\"> into HTML", baseHref);
}


try {
  ensureDataset()
  console.log("[prebuild] Dataset ready")

  buildSphinxDocs()
  console.log("[prebuild] Sphinx docs ready")
} catch (error) {
  console.error("[prebuild] Failed to prepare dataset or Sphinx docs")
  if (error instanceof Error) console.error(error.message)
  process.exit(1)
}
