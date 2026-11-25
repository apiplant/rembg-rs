import { createSignal, For } from "solid-js";

const FEATURES = [
  ["01", "Run locally", "Remove backgrounds on your own machine with a native Rust pipeline. Images stay with the server you run."],
  ["02", "Work visually", "Use the built-in workspace to inspect the cutout, add a background, and download a PNG."],
  ["03", "Build on it", "Use the CLI for files, the HTTP API for services, or the Rust library inside your application."],
];

const DOCS = [
  { label: "GETTING STARTED", title: "Install and quick start", text: "Install a prebuilt executable, fetch the model, and remove a background.", href: "/docs#install", link: "OPEN DOCS" },
  { label: "CLI / HTTP", title: "Serve and API", text: "Use the browser UI, readiness endpoint, and raw-byte image API.", href: "/docs#server", link: "OPEN DOCS" },
  { label: "RUST", title: "Library API", text: "Load the native pipeline from a Rust application.", href: "/docs#library", link: "OPEN DOCS" },
  { label: "BENCHMARKS", title: "Performance notes", text: "Warm timings, stage profiles, VRAM use, and ComfyUI comparison.", href: "https://github.com/apiplant/rembg-rs/blob/main/docs/performance.md", link: "BENCHMARKS" },
];

const INSTALL_METHODS = [
  {
    label: "Homebrew",
    platform: "macOS Apple Silicon · Linux x86_64 / arm64",
    commands: "brew tap apiplant/tap\nbrew install apiplant/tap/rembg-rs",
  },
  {
    label: "Debian / Ubuntu",
    platform: "amd64 · arm64",
    commands: [
      "curl -fsSL https://apt.apiplant.com/apiplant-archive-keyring.gpg | sudo tee /usr/share/keyrings/apiplant.gpg >/dev/null",
      "echo 'deb [signed-by=/usr/share/keyrings/apiplant.gpg] https://apt.apiplant.com stable main' | sudo tee /etc/apt/sources.list.d/apiplant.list >/dev/null",
      "sudo apt update",
      "sudo apt install rembg-rs",
    ].join("\n"),
  },
  {
    label: "Arch Linux",
    platform: "x86_64",
    commands: [
      "curl -fsSL https://apiplant.github.io/pacman/apiplant.gpg -o /tmp/apiplant.gpg",
      "keyid=$(gpg --show-keys --with-colons /tmp/apiplant.gpg | awk -F: '/^pub:/ {print $5; exit}')",
      "sudo pacman-key --add /tmp/apiplant.gpg",
      "sudo pacman-key --finger \"$keyid\" && sudo pacman-key --lsign-key \"$keyid\"",
      "printf '\\n[apiplant]\\nSigLevel = Required DatabaseOptional\\nServer = https://apiplant.github.io/pacman/$arch\\n' | sudo tee -a /etc/pacman.conf >/dev/null",
      "sudo pacman -Sy rembg-rs",
    ].join("\n"),
  },
];

export function Home() {
  const [split, setSplit] = createSignal(50);
  const [installMethod, setInstallMethod] = createSignal(0);
  const [installVariant, setInstallVariant] = createSignal<"cpu" | "cuda">("cpu");
  const [copied, setCopied] = createSignal(false);

  const installCommands = () => installVariant() === "cuda" && installMethod() !== 0
    ? INSTALL_METHODS[installMethod()].commands.replace(/rembg-rs$/, "rembg-rs-cuda")
    : INSTALL_METHODS[installMethod()].commands;

  async function copyInstall() {
    await navigator.clipboard.writeText(installCommands());
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1800);
  }

  return (
    <div class="container">
      <section class="hero">
        <div class="hero-copy">
          <div class="badges">
            <span class="badge badge-accent">RMBG-3.0</span>
            <span class="badge">Native Rust</span>
            <span class="badge">Local inference</span>
          </div>
          <h1>Background removal<br /><span class="text-accent">in native Rust.</span></h1>
          <p class="hero-lead">Clean, transparent cutouts with Bria RMBG-3.0. Run it locally from the browser, command line, or your own application.</p>
          <div class="hero-actions">
            <a class="button button-primary" href="#install">Install <span aria-hidden="true">↓</span></a>
            <a class="button button-secondary" href="/docs">Read the docs</a>
          </div>
          <p class="hero-note">Local processing · full-resolution transparent PNG</p>
        </div>

        <figure class="compare-card">
          <div class="window-bar">
            <span class="window-dots" aria-hidden="true"><i></i><i></i><i></i></span>
            <span class="window-title">rembg-rs / preview</span>
            <span class="window-status"><i></i> ready</span>
          </div>
          <div class="compare-stage" style={"--split: " + split() + "%"}>
            <div class="checkerboard" aria-hidden="true"></div>
            <img class="compare-image compare-before" src="/assets/me-before.png" width="512" height="512" alt="Original portrait with its outdoor background" fetchpriority="high" draggable={false} />
            <img class="compare-image compare-after" src="/assets/me-cutout.png" width="512" height="512" alt="The same portrait with the background removed and transparency preserved" fetchpriority="high" draggable={false} />
            <div class="split-marker" aria-hidden="true"><span class="split-handle"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round"><path d="m9 7-5 5 5 5m6-10 5 5-5 5" /></svg></span></div>
            <span class="image-label image-label-before">Original</span>
            <span class="image-label image-label-after">Transparent cutout</span>
            <input class="compare-range" type="range" min="0" max="100" value={split()} onInput={(event) => setSplit(Number(event.currentTarget.value))} aria-label="Drag to compare the original and background-removed image" />
          </div>
          <figcaption class="compare-caption">
            <span><b>Real inference result</b><small>512 × 512 · Bria RMBG-3.0 · RGBA PNG</small></span>
            <span class="caption-hint">Drag to compare</span>
          </figcaption>
        </figure>
      </section>

      <section class="section ui-section" aria-labelledby="ui-title">
        <div class="ui-workspace-layout">
          <div class="ui-workspace-copy">
            <div><p class="eyebrow">BUILT-IN WORKSPACE</p><h2 id="ui-title">The image gets <span class="text-accent">the space.</span></h2></div>
            <div><p>Compare the complete original and cutout side by side. Add a photo or color behind the cutout when you need it, then download a full-resolution PNG.</p><a class="text-link" href="/docs#server">Run the local workspace →</a></div>
          </div>
          <figure class="ui-shot-card">
            <div class="ui-shot-bar"><span class="window-dots" aria-hidden="true"><i></i><i></i><i></i></span><span class="window-title">rembg-rs / local workspace</span><span class="ui-shot-state"><i></i> LOCAL</span></div>
            <a class="ui-shot-link" href="/assets/app-ui.png" target="_blank" rel="noreferrer noopener" aria-label="Open the full-size rembg-rs interface screenshot">
              <img class="ui-shot" src="/assets/app-ui.png" width="1440" height="1003" alt="rembg-rs local workspace showing a complete original portrait beside its transparent cutout" loading="lazy" />
            </a>
            <figcaption><span><b>Runs on your machine.</b><small>Your image stays on your computer.</small></span><a href="/assets/app-ui.png" target="_blank" rel="noreferrer noopener">Open full screenshot</a></figcaption>
          </figure>
        </div>
      </section>

      <section class="section" id="features">
        <div class="section-heading">
          <div><p class="eyebrow">WHAT IT DOES</p><h2>One tool, <span class="text-accent">several ways to use it.</span></h2></div>
          <p>Use the same local model from the browser, terminal, or your code.</p>
        </div>
        <div class="feature-grid">
          <For each={FEATURES}>{(feature) =>
            <article class="feature-card"><span class="card-index">{feature[0]}</span><h3>{feature[1]}</h3><p>{feature[2]}</p></article>
          }</For>
        </div>
        <div class="benchmark-card">
          <div><p class="eyebrow">PERFORMANCE</p><h3>15.967 s <span>RTX 4090 warm median</span></h3><p>Measured on a 1410 × 1080 image with the public INT8 model bundle.</p></div>
          <a class="text-link" href="https://github.com/apiplant/rembg-rs/blob/main/docs/performance.md">See methodology →</a>
        </div>
      </section>

      <section class="section install-section" id="install">
        <div class="section-heading">
          <div><p class="eyebrow">INSTALL</p><h2>Ready from <span class="text-accent">your terminal.</span></h2></div>
          <p>Add the ApiPlant repository once, then install with your package manager. The model downloads automatically on first use.</p>
        </div>
        <div class="install-card">
          <div class="install-card-head"><div class="install-tabs" role="tablist" aria-label="Package manager">
            <For each={INSTALL_METHODS}>{(method, index) => <button type="button" role="tab" aria-selected={installMethod() === index() ? "true" : "false"} aria-controls="install-panel" onClick={() => { setInstallMethod(index()); setInstallVariant("cpu"); setCopied(false); }}>{method.label}</button>}</For>
          </div><button class="copy-button" type="button" onClick={copyInstall}>{copied() ? "Copied" : "Copy commands"}</button></div>
          <div id="install-panel" role="tabpanel" class="install-panel"><div class="install-panel-meta"><p>{INSTALL_METHODS[installMethod()].platform}</p>{installMethod() !== 0 && <div class="variant-switch" role="group" aria-label="Package variant"><button type="button" aria-pressed={installVariant() === "cpu" ? "true" : "false"} onClick={() => setInstallVariant("cpu")}>CPU</button><button type="button" aria-pressed={installVariant() === "cuda" ? "true" : "false"} onClick={() => setInstallVariant("cuda")}>NVIDIA CUDA</button></div>}</div><pre><code>{installCommands()}</code></pre></div>
        </div>
        <div class="quick-start-row">
          <div><p class="eyebrow">FIRST RUN</p><h3>Start with an image.</h3><p>The public model is about 13.6 GB. The Rust binary downloads missing files automatically and resumes interrupted downloads. No Python or Hugging Face token is needed.</p></div>
          <pre><code>{"rembg-rs remove input.jpg cutout.png\nrembg-rs serve"}</code></pre>
        </div>
        <p class="license-warning">CPU packages run on every supported platform. NVIDIA users on Linux x86_64 can install <code>rembg-rs-cuda</code> from apt or pacman after adding the same repository, or download the CUDA archive from <a href="https://github.com/apiplant/rembg-rs/releases/latest">GitHub Releases</a>.</p>
      </section>

      <section class="section docs-section" id="docs">
        <div class="section-heading">
          <div><p class="eyebrow">DOCS</p><h2>Everything you need <span class="text-accent">to run it.</span></h2></div>
          <p>Setup, CLI and server examples, Rust API details, performance measurements, and model licensing.</p>
        </div>
        <div class="docs-grid">
          <For each={DOCS}>{(doc) =>
            <a class="doc-card" href={doc.href}><span>{doc.label}</span><h3>{doc.title}</h3><p>{doc.text}</p><b>{doc.link}</b></a>
          }</For>
        </div>
      </section>
    </div>
  );
}
