import { rename, writeFile } from "node:fs/promises";

export async function replaceAtomically(destination: string, contents: string): Promise<void> {
  const staging = `${destination}.next`;
  await writeFile(staging, contents, "utf8");
  await rename(staging, destination);
}

export function stillUsable(savedAt: number, now: number, maxAgeMs: number): boolean {
  return now - savedAt <= maxAgeMs;
}

export function selectNewest<T extends { changedAt: number }>(rows: T[]): T | undefined {
  return rows.slice().sort((left, right) => right.changedAt - left.changedAt)[0];
}

export function decorateLabel(label: string): string {
  return label.trim().replaceAll("_", " ");
}
