import { createSignal } from "solid-js";

export type Theme = "dark" | "light";

const STORAGE_KEY = "rembg-rs-theme";

function systemTheme(): Theme {
  return window.matchMedia?.("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

function storedTheme(): Theme | null {
  try {
    const value = localStorage.getItem(STORAGE_KEY);
    return value === "dark" || value === "light" ? value : null;
  } catch {
    return null;
  }
}

const [theme, setThemeSignal] = createSignal<Theme>(storedTheme() ?? systemTheme());

function applyTheme(next: Theme) {
  document.documentElement.classList.toggle("light", next === "light");
  document.documentElement.style.colorScheme = next;
}

applyTheme(theme());

export function toggleTheme() {
  const next = theme() === "dark" ? "light" : "dark";
  setThemeSignal(next);
  try {
    localStorage.setItem(STORAGE_KEY, next);
  } catch {
    // The theme still applies when storage is unavailable.
  }
  applyTheme(next);
}
