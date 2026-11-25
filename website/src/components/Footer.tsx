export function Footer() {
  return (
    <>
      <section class="license-band">
        <div class="container license-content">
          <span class="license-symbol">i</span>
          <p><b>Check the model terms before use.</b> The weights are CC BY-NC 4.0; commercial use needs a separate Bria license.</p>
          <a href="https://huggingface.co/briaai/RMBG-3.0" target="_blank" rel="noreferrer noopener">Model card</a>
        </div>
      </section>
      <footer class="site-footer">
        <div class="container footer-grid">
          <div class="footer-about">
            <a class="brand" href="/"><span>rembg<span class="brand-suffix">-rs</span></span></a>
            <p>Native Rust inference for Bria RMBG-3.0. Local CLI, web UI, and HTTP API.</p>
          </div>
          <div><h2>Product</h2><a href="/#features">Features</a><a href="/#install">Install</a><a href="/docs">Docs</a></div>
          <div><h2>Docs</h2><a href="/docs#quick-start">Quick start</a><a href="/docs#server">Server &amp; API</a><a href="/docs#performance">Benchmarks</a></div>
          <div><h2>Source</h2><a href="https://github.com/apiplant/rembg-rs" target="_blank" rel="noreferrer noopener">GitHub</a><a href="https://github.com/apiplant/rembg-rs/issues" target="_blank" rel="noreferrer noopener">Issues</a><a href="https://github.com/apiplant/rembg-rs/releases" target="_blank" rel="noreferrer noopener">Releases</a></div>
        </div>
        <div class="container footer-bottom"><span>© 2026 rembg-rs · MIT code</span><a href="https://apiplant.com" target="_blank" rel="noreferrer noopener">API Plant</a></div>
      </footer>
    </>
  );
}
