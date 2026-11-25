# rembg-rs site

Static project site for [rembg-rs](https://github.com/apiplant/rembg-rs), served at [rembg-rs.apiplant.com](https://rembg-rs.apiplant.com/) through Cloudflare Pages.

There is no build step. Configure the Cloudflare Pages project with production branch `main`, root directory `website`, an empty build command, and `.` as the output directory. Cloudflare publishes each push to `main` automatically. Preview branches use Cloudflare's preview deployment URLs.

The source is plain HTML, CSS, and SVG. Keep the deployment root pointed at `website/`; the server upload UI lives separately in `../ui/`.
