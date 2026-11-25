const DOCS_NAV = [
  { title: "GETTING STARTED", links: [["Install", "install"], ["Quick start", "quick-start"], ["Model files", "models"]] },
  { title: "USAGE", links: [["CLI", "cli"], ["Web UI and HTTP API", "server"]] },
  { title: "REFERENCE", links: [["Rust library", "library"], ["Performance", "performance"], ["License", "license"]] },
];

function Snippet(props: { text: string }) {
  return <pre class="docs-code"><code>{props.text}</code></pre>;
}

export function Docs() {
  return (
    <div class="docs-layout container">
      <aside class="docs-sidebar">
        <nav aria-label="Documentation sections">
          {DOCS_NAV.map((group) => (
            <div class="docs-nav-group">
              <p>{group.title}</p>
              {group.links.map(([label, id]) => <a href={"#" + id}>{label}</a>)}
            </div>
          ))}
        </nav>
      </aside>

      <article class="docs-content">
        <p class="eyebrow">REMBG-RS DOCUMENTATION</p>
        <h1>Run background removal locally.</h1>
        <p class="docs-lead">Install the prebuilt executable, then remove backgrounds from the command line or the local web interface. Missing model files download automatically on first use.</p>

        <section class="docs-block" id="install">
          <p class="eyebrow">INSTALL</p>
          <h2>Install a prebuilt executable</h2>
          <p>Package-manager binaries do not need a Rust toolchain or Python. Add the signed ApiPlant repository once, then install and update with your package manager. CPU builds work without an NVIDIA GPU.</p>
          <div class="docs-package">
            <h3>Homebrew · Apple Silicon and Linux x86_64</h3>
            <Snippet text={["brew tap apiplant/tap", "brew install apiplant/tap/rembg-rs"].join("\n")} />
          </div>
          <div class="docs-package">
            <h3>Debian and Ubuntu · amd64 or arm64</h3>
            <Snippet text={[
              "curl -fsSL https://apt.apiplant.com/apiplant-archive-keyring.gpg | sudo tee /usr/share/keyrings/apiplant.gpg >/dev/null",
              "echo 'deb [signed-by=/usr/share/keyrings/apiplant.gpg] https://apt.apiplant.com stable main' | sudo tee /etc/apt/sources.list.d/apiplant.list >/dev/null",
              "sudo apt update",
              "sudo apt install rembg-rs",
            ].join("\n")} />
            <p>For NVIDIA CUDA on amd64, use <code>sudo apt install rembg-rs-cuda</code> for the final command.</p>
          </div>
          <div class="docs-package">
            <h3>Arch Linux · x86_64</h3>
            <Snippet text={[
              "curl -fsSL https://apiplant.github.io/pacman/apiplant.gpg -o /tmp/apiplant.gpg",
              "keyid=$(gpg --show-keys --with-colons /tmp/apiplant.gpg | awk -F: '/^pub:/ {print $5; exit}')",
              "sudo pacman-key --add /tmp/apiplant.gpg",
              "sudo pacman-key --finger \"$keyid\" && sudo pacman-key --lsign-key \"$keyid\"",
              "printf '\\n[apiplant]\\nSigLevel = Required DatabaseOptional\\nServer = https://apiplant.github.io/pacman/$arch\\n' | sudo tee -a /etc/pacman.conf >/dev/null",
              "sudo pacman -Sy rembg-rs",
            ].join("\n")} />
            <p>For NVIDIA CUDA on x86_64, use <code>sudo pacman -Sy rembg-rs-cuda</code> for the final command.</p>
          </div>
          <p class="docs-note">The packages above use CPU inference. On Linux x86_64, install <code>rembg-rs-cuda</code> instead for NVIDIA CUDA after adding the same repository and a compatible CUDA runtime. Allow about 13.6 GB of disk space for the model.</p>
        </section>

        <section class="docs-block" id="quick-start">
          <p class="eyebrow">QUICK START</p>
          <h2>Remove an image</h2>
          <p>The Rust executable downloads the public model files on first use and resumes an interrupted download. No Python setup or Hugging Face token is needed.</p>
          <Snippet text={[
            "rembg-rs remove input.jpg cutout.png",
            "rembg-rs serve",
          ].join("\n")} />
          <p>The default model directory is <code>~/.cache/rembg-rs/models/comfyui</code>. Set <code>RMBG_MODELS_DIR</code> or use <code>--models-dir</code> to choose another location.</p>
        </section>

        <section class="docs-block" id="models">
          <p class="eyebrow">MODEL FILES</p>
          <h2>Model download and cache</h2>
          <p><code>fetch-comfy</code> downloads Bria's public INT8 ConvRot weights and Qwen tokenizer metadata ahead of time. The Rust downloader resumes partial files and checks their sizes before loading the model.</p>
          <Snippet text={[
            "rembg-rs fetch-comfy",
            "",
            "# Use a different model directory",
            "rembg-rs --models-dir /data/models/rembg-rs fetch-comfy",
          ].join("\n")} />
          <p>Inference keeps one model loaded for the life of the server process.</p>
        </section>

        <section class="docs-block" id="cli">
          <p class="eyebrow">COMMAND LINE</p>
          <h2>CLI options</h2>
          <p>Remove one image to a transparent PNG or start the local server. Global options can appear before the subcommand.</p>
          <Snippet text={[
            "rembg-rs --device cpu remove input.jpg cutout.png",
            "rembg-rs --device cpu serve --addr 127.0.0.1:8080",
            "",
            "# Select CUDA for a CUDA-enabled build",
            "rembg-rs --device cuda --vae-device cpu remove input.jpg cutout.png",
            "",
            "rembg-rs --help",
            "rembg-rs serve --help",
          ].join("\n")} />
          <p><code>--device</code> selects the denoiser device. <code>--vae-device</code> accepts <code>auto</code>, <code>cpu</code>, <code>same</code>, or a CUDA device such as <code>cuda:0</code>. In auto mode, the VAE uses CPU unless enough GPU memory is free.</p>
          <p>Build CUDA support from source with <code>--features cuda</code>; <code>cuda-cudnn</code> additionally requires the CUDA toolkit, NVCC, and cuDNN.</p>
        </section>

        <section class="docs-block" id="server">
          <p class="eyebrow">WEB UI AND HTTP</p>
          <h2>Run the local server</h2>
          <Snippet text="rembg-rs --device cpu serve --addr 127.0.0.1:8080" />
          <p>The server prints its URL when ready. Open it to use the browser UI. Uploaded images stay on the machine running the server.</p>
          <div class="docs-table-wrap">
            <table class="docs-table">
              <thead><tr><th>Route</th><th>Purpose</th></tr></thead>
              <tbody>
                <tr><td><code>GET /</code></td><td>Browser interface</td></tr>
                <tr><td><code>GET /health</code></td><td>Readiness check</td></tr>
                <tr><td><code>POST /remove</code></td><td>Raw image bytes in; full-size PNG with alpha out</td></tr>
              </tbody>
            </table>
          </div>
          <Snippet text={[
            "curl --data-binary @input.jpg \\",
            "  -H 'Content-Type: image/jpeg' \\",
            "  http://127.0.0.1:8080/remove -o cutout.png",
          ].join("\n")} />
          <p>The default upload limit is 50 MiB and one request can wait in the queue. Configure these with <code>--max-upload-mib</code> and <code>--queue-size</code>.</p>
        </section>

        <section class="docs-block" id="library">
          <p class="eyebrow">RUST LIBRARY</p>
          <h2>Use the pipeline from Rust</h2>
          <p>The inference components are exposed by the <code>rembg_rs</code> crate in this repository. Use the library when building from source.</p>
          <Snippet text={[
            "let pipeline = NativePipeline::load(&default_models_dir(), \"cuda\")?;",
            "let matte_png = pipeline.remove_background(&image_bytes)?;",
          ].join("\n")} />
          <p>See the <a href="https://github.com/apiplant/rembg-rs/blob/main/README.md#library-use">README library example</a> for imports and model setup.</p>
        </section>

        <section class="docs-block" id="performance">
          <p class="eyebrow">PERFORMANCE</p>
          <h2>Measured on an RTX 4090</h2>
          <p>Native Rust CUDA path: 15.967 s warm median. ComfyUI AILab_RMBG3 native INT8 path: 17.891 s on the same 1410 × 1080 image and model bundle.</p>
          <p><a href="https://github.com/apiplant/rembg-rs/blob/main/docs/performance.md">Read the measurements and benchmark method.</a></p>
        </section>

        <section class="docs-block" id="license">
          <p class="eyebrow">LICENSE</p>
          <h2>Code and model terms</h2>
          <p>The repository code is MIT licensed. RMBG-3.0 weights are CC BY-NC 4.0; commercial use requires a separate Bria license. Review the <a href="https://huggingface.co/briaai/RMBG-3.0" target="_blank" rel="noreferrer noopener">model card</a> before downloading or using the weights.</p>
        </section>
      </article>
    </div>
  );
}
