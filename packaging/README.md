# Release packaging

A `v*` tag builds CPU archives for macOS Apple Silicon, Linux x86_64, and Linux
arm64. Separate Linux x86_64 CUDA archives target Debian/Ubuntu and Arch. Each archive contains the
`rembg-rs` executable, README, and license. The model weights are not included;
the executable downloads missing files automatically on first use.

| Platform | Release assets | Package manager |
| --- | --- | --- |
| macOS Apple Silicon | CPU archive | Homebrew |
| Linux x86_64 | CPU and CUDA archives | Homebrew, Debian/Ubuntu, Arch |
| Linux arm64 | CPU archive | Homebrew, Debian/Ubuntu |

The release workflow builds `.deb` and `.pkg.tar.zst` packages from the same
archives. It publishes the archives and packages to GitHub Releases, the
Homebrew formula to `apiplant/homebrew-tap`, Debian packages to the signed
repository at `apt.apiplant.com`, and Arch packages to the signed repository at
`apiplant.github.io/pacman`. Package builds do not compile the project a
second time. The CUDA Debian and Arch packages conflict with the CPU package
because they install the same executable.

The Debian/Ubuntu CUDA archive is built with CUDA 12.1; the Arch CUDA archive
is built against Arch's CUDA package. Both require a compatible NVIDIA
driver and CUDA runtime on the host. The Arch CUDA package depends on Arch's
`cuda` and `nvidia-utils` packages. Debian/Ubuntu users need a compatible CUDA
runtime from NVIDIA's repository in addition to the declared `libcuda1`
driver dependency. The Debian/Ubuntu build targets Turing (compute capability
7.5) and newer; the Arch build targets Ampere (8.0) and newer. The CUDA
archives are also available for manual installation.
The published benchmark uses a local `cuda-cudnn` build; these release archives
use the `cuda` feature.

The workflows use optional `CARGO_REGISTRY_TOKEN`, `HOMEBREW_TAP_TOKEN`, `APT_REPO_TOKEN`,
`APT_GPG_PRIVATE_KEY`, `APT_GPG_PASSPHRASE`, `PACMAN_REPO_TOKEN`,
`PACMAN_GPG_PRIVATE_KEY`, and `PACMAN_GPG_PASSPHRASE`. Publication jobs skip
when their repository token is absent. The release assets still publish.

## Install

```sh
brew tap apiplant/tap
brew install apiplant/tap/rembg-rs
```

For apt or pacman, add the signed ApiPlant repository as described in the
[installation docs](https://rembg-rs.apiplant.com/docs#install), then install
`rembg-rs` or `rembg-rs-cuda` on x86_64 Linux. CPU builds default to CPU
inference, and CUDA builds default to CUDA with automatic VAE placement.

The Rust executable downloads and validates the model bundle directly. It
resumes partial downloads and starts from already complete files offline.
