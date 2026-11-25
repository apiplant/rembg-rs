# rembg-rs site

SolidJS 2 project site for [rembg-rs](https://github.com/apiplant/rembg-rs), published with Cloudflare Pages at [rembg-rs.apiplant.com](https://rembg-rs.apiplant.com/).

The app follows the Vite, Solid, and Tailwind setup used by gliner-rs. Cloudflare Pages should use /website as its root, pnpm install --frozen-lockfile && pnpm build as its build command, and dist as its output directory. Git-connected production deploys from main; pushes to that branch update the site.

The hero comparison uses the actual 512 × 512 input portrait and its RMBG-3.0 output in public/assets/. The runnable image processing UI is served separately by rembg-rs serve.
