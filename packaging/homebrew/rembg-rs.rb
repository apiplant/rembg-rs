# Rendered from this template by the tagged release workflow.
class RembgRs < Formula
  desc "Native Rust background removal with Bria RMBG-3.0"
  homepage "https://rembg-rs.apiplant.com"
  version "@VERSION@"
  license "MIT"

  on_macos do
    on_arm do
      url "https://github.com/apiplant/rembg-rs/releases/download/v@VERSION@/rembg-rs-v@VERSION@-aarch64-apple-darwin.tar.gz"
      sha256 "@SHA_MACOS_ARM64@"
    end
  end
  on_linux do
    on_intel do
      url "https://github.com/apiplant/rembg-rs/releases/download/v@VERSION@/rembg-rs-v@VERSION@-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "@SHA_LINUX_X86_64@"
    end
    on_arm do
      url "https://github.com/apiplant/rembg-rs/releases/download/v@VERSION@/rembg-rs-v@VERSION@-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "@SHA_LINUX_ARM64@"
    end
  end

  def install
    bin.install "rembg-rs"
    doc.install "README.md"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/rembg-rs --version")
  end
end
