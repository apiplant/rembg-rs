import { createSignal, For, Show } from "solid-js";
import { toggleTheme } from "../lib/theme";

export function Header(props: { docs?: boolean }) {
  const [open, setOpen] = createSignal(false);
  const nav = () => [
    { label: "Features", href: props.docs ? "/#features" : "#features" },
    { label: "Install", href: props.docs ? "/#install" : "#install" },
    { label: "Docs", href: "/docs" },
  ];

  return (
    <header class="site-header">
      <div class="container header-inner">
        <a class="brand" href="/" aria-label="rembg-rs home">
          <span>rembg<span class="brand-suffix">-rs</span></span>
        </a>
        <nav class="main-nav" aria-label="Main navigation">
          <For each={nav()}>{(item) => <a href={item.href} aria-current={item.label === "Docs" && props.docs ? "page" : undefined}>{item.label}</a>}</For>
        </nav>
        <div class="header-actions">
          <button class="icon-button theme-toggle" type="button" aria-label="Switch theme" title="Switch theme" onClick={toggleTheme}>
            <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4" /><path d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.1 5.1l1.4 1.4M17.5 17.5l1.4 1.4M18.9 5.1l-1.4 1.4M6.5 17.5l-1.4-1.4" /></svg>
          </button>
          <a class="icon-button github-link" href="https://github.com/apiplant/rembg-rs" aria-label="rembg-rs on GitHub" target="_blank" rel="noreferrer noopener">
            <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.4 7.4 0 0 1 2-.27c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z" /></svg>
          </a>
          <button class="icon-button menu-toggle" type="button" aria-label="Open navigation" aria-expanded={open() ? "true" : "false"} aria-controls="mobile-nav" onClick={() => setOpen(!open())}>
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d={open() ? "M6 6l12 12M18 6L6 18" : "M4 7h16M4 12h16M4 17h16"} /></svg>
          </button>
        </div>
      </div>
      <Show when={open()}>
        <nav class="mobile-nav" id="mobile-nav" aria-label="Mobile navigation">
          <For each={nav()}>{(item) => <a href={item.href} aria-current={item.label === "Docs" && props.docs ? "page" : undefined} onClick={() => setOpen(false)}>{item.label}</a>}</For>
        </nav>
      </Show>
    </header>
  );
}
