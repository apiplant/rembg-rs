use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use ntex::web;
use rembg_rs_bria::{Config, Rmbg, default_models_dir, download::ensure_models};
use std::{path::PathBuf, sync::mpsc};

#[derive(Parser)]
#[command(name = "rembg-rs", version, about = "RMBG-3.0 background removal")]
struct Args {
    #[arg(long, global = true, env = "RMBG_DEVICE", default_value = if cfg!(feature = "cuda") { "cuda" } else { "cpu" })]
    device: String,
    /// Device for the FLUX.2 VAE (`auto`, `cpu`, `same`, `cuda`, or `cuda:N`).
    #[arg(long, global = true, env = "RMBG_VAE_DEVICE", default_value = "auto")]
    vae_device: String,
    #[arg(
        long,
        global = true,
        env = "RMBG_MODELS_DIR",
        default_value_os_t = default_models_dir()
    )]
    models_dir: PathBuf,
    #[command(subcommand)]
    command: Action,
}

#[derive(Subcommand)]
enum Action {
    Remove {
        input: PathBuf,
        output: PathBuf,
    },
    Serve {
        /// Address to listen on.
        #[arg(
            long,
            visible_alias = "bind",
            env = "RMBG_ADDR",
            default_value = "127.0.0.1:8080"
        )]
        addr: String,
        /// Maximum number of uploads waiting behind the active inference.
        #[arg(long, env = "RMBG_QUEUE_SIZE", default_value_t = 1)]
        queue_size: usize,
        /// Maximum upload size in MiB.
        #[arg(long, env = "RMBG_MAX_UPLOAD_MIB", default_value_t = 50)]
        max_upload_mib: usize,
    },
    FetchComfy {
        /// Directory to populate with official INT8 files (default: the user model cache).
        comfy_models_dir: Option<PathBuf>,
    },
}

struct InferenceJob {
    bytes: Vec<u8>,
    reply: futures_channel::oneshot::Sender<anyhow::Result<Vec<u8>>>,
}

#[ntex::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let mut config = Config::default();
    config.device = args.device;
    config.vae_device = args.vae_device;
    config.models_dir = args.models_dir;
    match args.command {
        Action::FetchComfy { comfy_models_dir } => {
            let comfy_models_dir = comfy_models_dir.unwrap_or(config.models_dir);
            ensure_models(&comfy_models_dir)?;
            eprintln!("Model files ready in {}", comfy_models_dir.display());
        }
        Action::Remove { input, output } => {
            let model = Rmbg::new(config)?;
            let bytes =
                std::fs::read(&input).with_context(|| format!("reading {}", input.display()))?;
            std::fs::write(&output, model.remove_background(&bytes)?)?;
        }
        Action::Serve {
            addr,
            queue_size,
            max_upload_mib,
        } => {
            if queue_size == 0 {
                anyhow::bail!("--queue-size must be greater than zero");
            }
            if max_upload_mib == 0 {
                anyhow::bail!("--max-upload-mib must be greater than zero");
            }
            let max_upload_bytes = max_upload_mib
                .checked_mul(1024 * 1024)
                .ok_or_else(|| anyhow::anyhow!("--max-upload-mib is too large"))?;
            eprintln!("loading native on {}", config.device);
            let device = config.device.clone();
            let model = Rmbg::new(config)?;
            let (worker_tx, worker_rx) = mpsc::sync_channel::<InferenceJob>(queue_size);
            std::thread::Builder::new()
                .name("rmbg-inference".into())
                .spawn(move || {
                    for job in worker_rx {
                        let result = model.remove_background(&job.bytes);
                        let _ = job.reply.send(result);
                    }
                })?;
            let server = web::HttpServer::new(async move |_| {
                let handler_worker = worker_tx.clone();
                web::App::new()
                    .service(web::resource("/").route(web::get().to(index)))
                    .service(web::resource("/health").route(web::get().to(async || "ready")))
                    .service(web::resource("/remove").route(web::post().to(
                        move |body: web::types::Payload| {
                            remove(body, handler_worker.clone(), max_upload_bytes)
                        },
                    )))
            })
            .bind(&addr, ntex::SharedCfg::default())?;
            print_server_ready(&addr, &device);
            server.run().await?;
        }
    }
    Ok(())
}

fn print_server_ready(addr: &str, device: &str) {
    use std::io::IsTerminal;
    let tty = std::io::stderr().is_terminal();
    let color = |code: &str, text: &str| {
        if tty {
            format!("\x1b[{code}m{text}\x1b[0m")
        } else {
            text.to_owned()
        }
    };
    let host = addr
        .replace("0.0.0.0", "localhost")
        .replace("[::]", "localhost");
    eprintln!();
    eprintln!(
        "  {}  {}",
        color("1;38;5;215", "▐▌ rembg-rs"),
        color("2", "background removal")
    );
    eprintln!(
        "  {}  {}",
        color("38;5;215", "web UI   "),
        color("4", &format!("http://{host}/"))
    );
    eprintln!(
        "  {}  {} /remove",
        color("38;5;215", "API      "),
        color("2", "POST")
    );
    eprintln!("  {}  native on {}", color("2", "model    "), device);
    eprintln!();
}

async fn index() -> web::HttpResponse {
    web::HttpResponse::Ok()
        .content_type("text/html; charset=utf-8")
        .body(include_str!("../ui/index.html"))
}

async fn remove(
    mut body: web::types::Payload,
    worker: mpsc::SyncSender<InferenceJob>,
    max_upload_bytes: usize,
) -> web::HttpResponse {
    let mut bytes = Vec::new();
    while let Some(chunk) = body.recv().await {
        match chunk {
            Ok(chunk) => bytes.extend_from_slice(&chunk),
            Err(_) => return web::HttpResponse::BadRequest().body("invalid request body"),
        }
        if bytes.len() > max_upload_bytes {
            return web::HttpResponse::PayloadTooLarge()
                .body("uploaded image exceeds --max-upload-mib");
        }
    }
    let (tx, rx) = futures_channel::oneshot::channel();
    if worker.try_send(InferenceJob { bytes, reply: tx }).is_err() {
        let mut response = web::HttpResponse::ServiceUnavailable();
        response.set_header("Retry-After", "2");
        return response.body("inference queue is full; retry shortly");
    }
    match rx.await {
        Ok(Ok(png)) => web::HttpResponse::Ok().content_type("image/png").body(png),
        Ok(Err(error)) => web::HttpResponse::BadRequest().body(error.to_string()),
        Err(_) => web::HttpResponse::InternalServerError().body("inference thread failed"),
    }
}
