# rembg-rs site

Static project site for [rembg-rs](https://github.com/apiplant/rembg-rs), served at [rembg-rs.apiplant.com](https://rembg-rs.apiplant.com/) through Cloudflare Pages.

There is no build step. The `rembg-rs-website` Cloudflare Pages project is Git-connected to this repository. Its production branch is `main`, with root directory `/website`, an empty build command, and `.` as the output directory. Pushes to `main` publish automatically; other branches use Cloudflare preview URLs.

The source is plain HTML, CSS, and SVG. Keep the deployment root pointed at `website/`; the server upload UI lives separately in `../ui/`.
