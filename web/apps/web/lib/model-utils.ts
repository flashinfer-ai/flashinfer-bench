import { Model } from "./schemas"

/**
 * Get children of a module
 */
export function getChildren(model: Model, moduleName: string): string[] {
  return Object.entries(model.modules)
    .filter(([_, module]) => module.parent === moduleName)
    .map(([name]) => name)
}

/**
 * Get root modules (modules with no parent)
 */
export function getRootModules(model: Model): string[] {
  return Object.entries(model.modules)
    .filter(([_, module]) => !module.parent)
    .map(([name]) => name)
}