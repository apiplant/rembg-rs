import { createSignal, For } from "solid-js";

const FEATURES = [
  ["01", "Native Rust inference", "Candle runs Qwen3-VL, Fibo-2, and the FLUX.2 VAE without Python at inference time."],
  ["02", "Packed INT8 CUDA", "ConvRot weights stay packed; row-quantized activations use CUDA tensor-core GEMMs."],
  ["03", "CLI and library", "Remove a background from a file or load the model components from Rust code."],
  ["04", "Local web UI", "Preview uploads immediately, inspect the full-size cutout, and position it over a resized or cropped background."],
  ["05", "ntex HTTP server", "One model stays loaded. The API accepts raw image bytes and returns an RGBA PNG."],
  ["06", "Automatic VAE placement", "Choose CPU or CUDA. In auto mode, the VAE uses CUDA only when enough VRAM is free."],
];

const DOCS = [
  { label: "GETTING STARTED", title: "Install and quick start", text: "Build, fetch the model, choose a device, and run an image.", href: "https://github.com/apiplant/rembg-rs/blob/main/README.md#quick-start", link: "README →" },
  { label: "CLI / HTTP", title: "Serve and API", text: "Use the browser UI, readiness endpoint, and raw-byte image API.", href: "https://github.com/apiplant/rembg-rs/blob/main/README.md#server-and-api", link: "README →" },
  { label: "RUST", title: "Library API", text: "Load the native pipeline and use it from your Rust application.", href: "https://github.com/apiplant/rembg-rs/blob/main/README.md#library-use", link: "README →" },
  { label: "BENCHMARKS", title: "Performance notes", text: "Warm timings, stage profiles, VRAM use, and ComfyUI comparison.", href: "https://github.com/apiplant/rembg-rs/blob/main/docs/performance.md", link: "PERFORMANCE →" },
];

export function Home() {
  const [split, setSplit] = createSignal(50);

  return (
    <div class="container">
      <section class="hero">
        <div class="hero-copy">
          <div class="badges">
            <span class="badge badge-accent">RMBG-3.0</span>
            <span class="badge">Native Rust</span>
            <span class="badge">Candle</span>
          </div>
          <h1>Background removal<br /><span class="text-accent">in native Rust.</span></h1>
          <p class="hero-lead">Bria RMBG-3.0, running locally. Use the CLI, start the built-in web UI, or send an image to the HTTP API.</p>
          <div class="hero-actions">
            <a class="button button-primary" href="#install">Install <span aria-hidden="true">↓</span></a>
            <a class="button button-secondary" href="#docs">Read the docs</a>
          </div>
          <p class="hero-note">Open weights · local inference · transparent PNG output</p>
        </div>

        <figure class="compare-card">
          <div class="window-bar">
            <span class="window-dots" aria-hidden="true"><i></i><i></i><i></i></span>
            <span class="window-title">rembg-rs / preview</span>
            <span class="window-status"><i></i> ready</span>
          </div>
          <div class="compare-stage" style={"--split: " + split() + "%"}>
            <div class="checkerboard" aria-hidden="true"></div>
            <img class="compare-image compare-before" src="/assets/me-before.png" width="512" height="512" alt="Original portrait with its outdoor background" fetchpriority="high" />
            <img class="compare-image compare-after" src="/assets/me-cutout.png" width="512" height="512" alt="The same portrait with the background removed and transparency preserved" fetchpriority="high" />
            <div class="split-marker" aria-hidden="true"></div>
            <span class="image-label image-label-before">Original</span>
            <span class="image-label image-label-after">Transparent cutout</span>
            <input class="compare-range" type="range" min="5" max="95" value={split()} onInput={(event) => setSplit(Number(event.currentTarget.value))} aria-label="Drag to compare the original and background-removed image" />
          </div>
          <figcaption class="compare-caption">
            <span><b>Real inference result</b><small>512 × 512 · Bria RMBG-3.0 · RGBA PNG</small></span>
            <span class="caption-hint">Drag to compare</span>
          </figcaption>
        </figure>
      </section>

      <section class="stats" aria-label="Project facts">
        <div><strong>15.97<span>s</span></strong><small>RTX 4090 warm median</small></div>
        <div><strong>INT8</strong><small>packed ConvRot CUDA path</small></div>
        <div><strong>4</strong><small>Flow Euler steps</small></div>
        <div><strong>MIT</strong><small>repository code license</small></div>
      </section>

      <section class="section" id="features">
        <div class="section-heading">
          <div><p class="eyebrow">FEATURES</p><h2>One model. <span class="text-accent">Three ways to use it.</span></h2></div>
          <p>Built around the full RMBG-3.0 pipeline, with native inference and a small local server.</p>
        </div>
        <div class="feature-grid">
          <For each={FEATURES}>{(feature) =>
            <article class="feature-card"><span class="card-index">{feature[0]}</span><h3>{feature[1]}</h3><p>{feature[2]}</p></article>
          }</For>
        </div>
        <div class="benchmark-card">
          <div><p class="eyebrow">MEASURED ON RTX 4090</p><h3>15.967 s <span>Rust · warm median</span></h3><p>10.8% lower than the 17.891 s ComfyUI median on the same 1410 × 1080 image and model bundle.</p></div>
          <a class="text-link" href="https://github.com/apiplant/rembg-rs/blob/main/docs/performance.md">Benchmark details <span aria-hidden="true">↗</span></a>
        </div>
      </section>

      <section class="section install-section" id="install">
        <div class="section-heading">
          <div><p class="eyebrow">INSTALL</p><h2>Fetch once.<br /><span class="text-accent">Run locally.</span></h2></div>
          <p>Download the public INT8 model bundle, then remove backgrounds from the terminal or open the local browser UI.</p>
        </div>
        <div class="install-layout">
          <div class="install-info">
            <h3>Requirements</h3>
            <ul><li>Rust stable</li><li>About 13.6 GB for model files</li><li>CUDA GPU for CUDA inference</li></ul>
            <p>Model download uses Python's <code>huggingface_hub</code>. Native inference does not need Python or an HF token for the public INT8 bundle.</p>
            <a class="button button-secondary" href="https://github.com/apiplant/rembg-rs/releases/latest" target="_blank" rel="noreferrer noopener">Download v{__VERSION__} <span aria-hidden="true">↗</span></a>
          </div>
          <div class="terminal">
            <div class="terminal-bar"><span class="window-dots" aria-hidden="true"><i></i><i></i><i></i></span><span class="window-title">Terminal</span><span class="terminal-state">CUDA</span></div>
            <pre><code><span class="code-comment"># Install the model download helper</span>
<span class="code-prompt">$</span> python3 -m pip install huggingface_hub

<span class="code-comment"># Fetch the model files once</span>
<span class="code-prompt">$</span> cargo run --release -- fetch-comfy

<span class="code-comment"># Create a transparent PNG</span>
<span class="code-prompt">$</span> cargo run --release --features cuda-cudnn \
    -- remove input.jpg cutout.png

<span class="code-comment"># Start the web UI and API</span>
<span class="code-prompt">$</span> cargo run --release --features cuda-cudnn \
    -- serve --addr 127.0.0.1:8080</code></pre>
            <div class="terminal-foot"><span><i></i> model cache · ~/.cache/rembg-rs</span><span>ready</span></div>
          </div>
        </div>
        <p class="license-warning"><b>Model terms:</b> RMBG-3.0 weights are CC BY-NC 4.0. Commercial use requires a Bria license. The code in this repository is MIT licensed.</p>
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
