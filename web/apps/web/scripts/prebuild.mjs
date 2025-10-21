#!/usr/bin/env node

import { existsSync, mkdirSync } from "node:fs"
import { spawnSync } from "node:child_process"
import path from "node:path"
import process from "node:process"

const DEFAULT_REPO = "https://github.com/flashinfer-ai/flashinfer-trace"
const repo = process.env.FLASHINFER_TRACE_REPO ?? DEFAULT_REPO
const ref = process.env.FLASHINFER_TRACE_REF ?? "origin/upd-baseline"
const token = process.env.GH_TOKEN

const datasetRootEnv = process.env.FLASHINFER_TRACE_PATH ?? "/tmp/flashinfer-trace"
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
    console.log(`[prebuild] Cloning dataset from ${repo}`)
    run("git", ["clone", "--depth=1", authRepo, datasetRoot])
    return
  }

  console.log("[prebuild] Refreshing existing dataset checkout")
  run("git", ["-C", datasetRoot, "fetch", "--all", "--prune"])
  run("git", ["-C", datasetRoot, "reset", "--hard", ref])
  run("git", ["-C", datasetRoot, "clean", "-fd"])
}

try {
  ensureDataset()
  console.log("[prebuild] Dataset ready")
} catch (error) {
  console.error("[prebuild] Failed to prepare dataset")
  if (error instanceof Error) {
    console.error(error.message)
  }
  process.exit(1)
}
