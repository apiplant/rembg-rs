import { defineConfig } from "vite";
import solid from "@solidjs/vite-plugin";
import tailwindcss from "@tailwindcss/vite";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

function crateVersion(): string {
  const manifest = readFileSync(fileURLToPath(new URL("../Cargo.toml", import.meta.url)), "utf8");
  const packageSection = manifest.slice(manifest.indexOf("[package]"));
  const match = /^\s*version\s*=\s*"([^"]+)"/m.exec(packageSection);
  if (!match) throw new Error("no [package] version in ../Cargo.toml");
  return match[1];
}

export default defineConfig({
  plugins: [solid(), tailwindcss()],
  define: { __VERSION__: JSON.stringify(crateVersion()) },
  server: { port: 5275 },
  build: {
    target: "es2022",
    outDir: "dist",
    emptyOutDir: true,
  },
});
