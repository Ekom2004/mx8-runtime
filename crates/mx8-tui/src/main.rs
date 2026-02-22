#![forbid(unsafe_code)]
#![cfg_attr(not(test), deny(clippy::expect_used, clippy::unwrap_used))]

use std::io::{self, Stdout};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::Result;
use clap::Parser;
use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::ExecutableCommand;
use mx8_core::types::MANIFEST_SCHEMA_VERSION;
use mx8_proto::v0::coordinator_client::CoordinatorClient;
use mx8_proto::v0::{GetJobSnapshotRequest, GetJobSnapshotResponse, GetManifestRequest};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Terminal;
use tonic::transport::Channel;

const DEFAULT_GRPC_MAX_MESSAGE_BYTES: usize = 64 * 1024 * 1024;
const DEFAULT_STALE_HEARTBEAT_MS: u64 = 5_000;
const DEFAULT_LEASE_STALL_MS: u64 = 10_000;

#[derive(Debug, Parser, Clone)]
#[command(name = "mx8-tui")]
#[command(about = "Read-only local MX8 TUI for manifest + live job state")]
struct Args {
    /// Coordinator URL, e.g. http://127.0.0.1:50051
    #[arg(long, env = "MX8_COORD_URL", default_value = "http://127.0.0.1:50051")]
    coord_url: String,
    /// Job ID in coordinator APIs.
    #[arg(long, env = "MX8_JOB_ID", default_value = "local-job")]
    job_id: String,
    /// Poll interval in milliseconds.
    #[arg(long, env = "MX8_TUI_POLL_MS", default_value_t = 1000)]
    poll_ms: u64,
    /// Headless mode polls. >0 runs without TTY and exits.
    #[arg(long, env = "MX8_TUI_HEADLESS_POLLS", default_value_t = 0)]
    headless_polls: u32,
    /// Optional initial manifest hash override.
    #[arg(long, env = "MX8_TUI_MANIFEST_HASH")]
    manifest_hash: Option<String>,
    /// Optional initial location substring filter.
    #[arg(long, env = "MX8_TUI_SEARCH", default_value = "")]
    search: String,
    /// Rows per manifest page in the panel.
    #[arg(long, env = "MX8_TUI_ROWS_PER_PAGE", default_value_t = 12)]
    rows_per_page: usize,
    /// gRPC max message size (both decode/encode).
    #[arg(
        long,
        env = "MX8_GRPC_MAX_MESSAGE_BYTES",
        default_value_t = DEFAULT_GRPC_MAX_MESSAGE_BYTES
    )]
    grpc_max_message_bytes: usize,
    /// Optional local manifest TSV fallback when coordinator manifest is unavailable.
    #[arg(long, env = "MX8_TUI_MANIFEST_PATH")]
    manifest_path: Option<PathBuf>,
    /// Heartbeat age threshold used to mark a node as stale.
    #[arg(
        long,
        env = "MX8_TUI_STALE_HEARTBEAT_MS",
        default_value_t = DEFAULT_STALE_HEARTBEAT_MS
    )]
    stale_heartbeat_ms: u64,
    /// No-progress threshold with active leases before marking job stalled.
    #[arg(
        long,
        env = "MX8_TUI_LEASE_STALL_MS",
        default_value_t = DEFAULT_LEASE_STALL_MS
    )]
    lease_stall_ms: u64,
}

#[derive(Debug, Clone)]
struct ManifestRow {
    sample_id: u64,
    location: String,
    decode_hint: Option<String>,
    byte_length: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InputMode {
    Normal,
    Search,
    Jump,
}

#[derive(Debug, Default)]
struct PanelHealth {
    lease_non_empty: bool,
    runtime_non_empty: bool,
    manifest_non_empty: bool,
}

#[derive(Debug, Clone)]
struct ClusterSummary {
    health: &'static str,
    stale_nodes: usize,
    total_nodes: usize,
    max_heartbeat_age_ms: u64,
    total_inflight_bytes: u64,
    total_ram_hwm_bytes: u64,
    autotune_nodes: usize,
    lease_stalled: bool,
    progress_stall_ms: u64,
}

#[derive(Debug)]
struct App {
    args: Args,
    snapshot: Option<GetJobSnapshotResponse>,
    manifest_hash: Option<String>,
    manifest_rows: Vec<ManifestRow>,
    filtered_indices: Vec<usize>,
    selected_row: usize,
    search_input: String,
    jump_input: String,
    input_mode: InputMode,
    status: String,
    last_refresh: Instant,
    last_progress_total: Option<u64>,
    last_progress_change_at: Instant,
}

impl App {
    fn new(args: Args) -> Self {
        Self {
            manifest_hash: args.manifest_hash.clone(),
            search_input: args.search.clone(),
            args,
            snapshot: None,
            manifest_rows: Vec::new(),
            filtered_indices: Vec::new(),
            selected_row: 0,
            jump_input: String::new(),
            input_mode: InputMode::Normal,
            status: "starting".to_string(),
            last_refresh: Instant::now() - Duration::from_secs(30),
            last_progress_total: None,
            last_progress_change_at: Instant::now(),
        }
    }

    fn apply_filter(&mut self) {
        let q = self.search_input.to_lowercase();
        self.filtered_indices = self
            .manifest_rows
            .iter()
            .enumerate()
            .filter_map(|(idx, row)| {
                let hit = q.is_empty()
                    || row.location.to_lowercase().contains(&q)
                    || row
                        .decode_hint
                        .as_ref()
                        .is_some_and(|h| h.to_lowercase().contains(&q));
                if hit {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect();
        if self.filtered_indices.is_empty() {
            self.selected_row = 0;
        } else {
            self.selected_row = self.selected_row.min(self.filtered_indices.len() - 1);
        }
    }

    fn move_selection(&mut self, delta: isize) {
        if self.filtered_indices.is_empty() {
            self.selected_row = 0;
            return;
        }
        let len = self.filtered_indices.len() as isize;
        let next = (self.selected_row as isize + delta).clamp(0, len - 1);
        self.selected_row = next as usize;
    }

    fn jump_to_sample_id(&mut self, sample_id: u64) {
        if self.filtered_indices.is_empty() {
            return;
        }
        if let Some((pos, _)) = self
            .filtered_indices
            .iter()
            .enumerate()
            .find(|(_, idx)| self.manifest_rows[**idx].sample_id >= sample_id)
        {
            self.selected_row = pos;
        } else {
            self.selected_row = self.filtered_indices.len() - 1;
        }
    }

    fn current_health(&self) -> PanelHealth {
        let lease_non_empty = self
            .snapshot
            .as_ref()
            .is_some_and(|s| s.registered_nodes > 0 || s.active_leases > 0);
        let runtime_non_empty = self.snapshot.as_ref().is_some_and(|s| {
            s.nodes
                .iter()
                .any(|n| n.stats.is_some() || n.last_heartbeat_unix_time_ms > 0)
        });
        let manifest_non_empty = !self.manifest_rows.is_empty();
        PanelHealth {
            lease_non_empty,
            runtime_non_empty,
            manifest_non_empty,
        }
    }

    fn summary(&self) -> Option<ClusterSummary> {
        let snapshot = self.snapshot.as_ref()?;
        let mut stale_nodes = 0usize;
        let mut max_heartbeat_age_ms = 0u64;
        let mut total_inflight_bytes = 0u64;
        let mut total_ram_hwm_bytes = 0u64;
        let mut autotune_nodes = 0usize;
        for node in &snapshot.nodes {
            let age_ms = snapshot
                .server_unix_time_ms
                .saturating_sub(node.last_heartbeat_unix_time_ms);
            max_heartbeat_age_ms = max_heartbeat_age_ms.max(age_ms);
            if age_ms > self.args.stale_heartbeat_ms {
                stale_nodes += 1;
            }
            if let Some(stats) = &node.stats {
                total_inflight_bytes = total_inflight_bytes.saturating_add(stats.inflight_bytes);
                total_ram_hwm_bytes =
                    total_ram_hwm_bytes.saturating_add(stats.ram_high_water_bytes);
                if stats.autotune_enabled {
                    autotune_nodes += 1;
                }
            }
        }

        let progress_stall_ms = self.last_progress_change_at.elapsed().as_millis() as u64;
        let lease_stalled =
            snapshot.active_leases > 0 && progress_stall_ms >= self.args.lease_stall_ms;
        let health = if lease_stalled {
            "stalled"
        } else if stale_nodes > 0 {
            "degraded"
        } else if !snapshot.job_ready {
            "warming"
        } else if snapshot.job_drained {
            "drained"
        } else {
            "healthy"
        };

        Some(ClusterSummary {
            health,
            stale_nodes,
            total_nodes: snapshot.nodes.len(),
            max_heartbeat_age_ms,
            total_inflight_bytes,
            total_ram_hwm_bytes,
            autotune_nodes,
            lease_stalled,
            progress_stall_ms,
        })
    }
}

fn fmt_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    let b = bytes as f64;
    if b >= GIB {
        format!("{:.2}GiB", b / GIB)
    } else if b >= MIB {
        format!("{:.2}MiB", b / MIB)
    } else if b >= KIB {
        format!("{:.2}KiB", b / KIB)
    } else {
        format!("{bytes}B")
    }
}

fn ratio_pct(numer: u64, denom: u64) -> String {
    if denom == 0 {
        "-".to_string()
    } else {
        format!("{:.1}%", (numer as f64 * 100.0) / denom as f64)
    }
}

fn parse_manifest_tsv(bytes: &[u8]) -> Result<Vec<ManifestRow>> {
    let text = std::str::from_utf8(bytes)?;
    let mut out = Vec::new();
    let mut lines = text.lines();
    let header = lines
        .find(|l| !l.trim().is_empty())
        .ok_or_else(|| anyhow::anyhow!("empty manifest"))?;
    if !header.trim_start().starts_with("schema_version=") {
        anyhow::bail!("manifest header missing schema_version");
    }
    for raw in lines {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 2 {
            continue;
        }
        let sample_id: u64 = match cols[0].trim().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let location = cols[1].trim().to_string();
        let byte_length = if cols.len() >= 4 {
            cols[3].trim().parse::<u64>().ok()
        } else {
            None
        };
        let decode_hint = if cols.len() >= 5 {
            let hint = cols[4].trim();
            if hint.is_empty() {
                None
            } else {
                Some(hint.to_string())
            }
        } else {
            None
        };
        out.push(ManifestRow {
            sample_id,
            location,
            decode_hint,
            byte_length,
        });
    }
    Ok(out)
}

async fn fetch_manifest_rows(
    client: &mut CoordinatorClient<Channel>,
    job_id: &str,
    manifest_hash: &str,
) -> Result<Vec<ManifestRow>> {
    let mut stream = client
        .get_manifest_stream(GetManifestRequest {
            job_id: job_id.to_string(),
            manifest_hash: manifest_hash.to_string(),
        })
        .await?
        .into_inner();
    let mut bytes = Vec::new();
    while let Some(chunk) = stream.message().await? {
        if chunk.schema_version != MANIFEST_SCHEMA_VERSION {
            anyhow::bail!(
                "manifest schema version mismatch: expected {}, got {}",
                MANIFEST_SCHEMA_VERSION,
                chunk.schema_version
            );
        }
        bytes.extend_from_slice(&chunk.data);
    }
    parse_manifest_tsv(&bytes)
}

async fn refresh_once(app: &mut App, client: &mut CoordinatorClient<Channel>) -> Result<()> {
    let snapshot = client
        .get_job_snapshot(GetJobSnapshotRequest {
            job_id: app.args.job_id.clone(),
        })
        .await?
        .into_inner();

    let incoming_hash = snapshot.manifest_hash.clone();
    let manifest_changed = app
        .manifest_hash
        .as_ref()
        .map(|h| h != &incoming_hash)
        .unwrap_or(true);

    let progress_total = snapshot.counters.as_ref().map(|c| c.progress_total);
    if app.last_progress_total != progress_total {
        app.last_progress_total = progress_total;
        app.last_progress_change_at = Instant::now();
    }

    app.snapshot = Some(snapshot);
    app.manifest_hash = Some(incoming_hash.clone());

    if manifest_changed || app.manifest_rows.is_empty() {
        match fetch_manifest_rows(client, &app.args.job_id, &incoming_hash).await {
            Ok(rows) => {
                app.manifest_rows = rows;
                app.apply_filter();
            }
            Err(err) => {
                if let Some(path) = app.args.manifest_path.clone() {
                    let bytes = std::fs::read(&path)?;
                    app.manifest_rows = parse_manifest_tsv(&bytes)?;
                    app.apply_filter();
                    app.status = format!(
                        "manifest fallback used from {} ({err})",
                        path.to_string_lossy()
                    );
                } else {
                    app.status = format!("manifest fetch failed: {err}");
                }
            }
        }
    }
    app.last_refresh = Instant::now();
    Ok(())
}

fn render(stdout: &mut Terminal<CrosstermBackend<Stdout>>, app: &App) -> Result<()> {
    stdout.draw(|f| {
        let root = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(10),
                Constraint::Length(13),
                Constraint::Min(8),
            ])
            .split(f.area());

        let top = {
            let mut lines = Vec::new();
            if let Some(s) = &app.snapshot {
                let counters = s.counters.as_ref();
                let summary = app.summary();
                lines.push(format!(
                    "job={} manifest={} ready={} drained={} nodes={}/{} leases={} ranges={}",
                    app.args.job_id,
                    s.manifest_hash,
                    s.job_ready,
                    s.job_drained,
                    s.registered_nodes,
                    s.world_size,
                    s.active_leases,
                    s.available_ranges
                ));
                if let Some(summary) = summary {
                    lines.push(format!(
                        "health={} stale_nodes={}/{} max_hb_age_ms={} lease_stalled={} stall_ms={}",
                        summary.health,
                        summary.stale_nodes,
                        summary.total_nodes,
                        summary.max_heartbeat_age_ms,
                        summary.lease_stalled,
                        summary.progress_stall_ms
                    ));
                    lines.push(format!(
                        "memory: inflight_total={} ram_hwm_total={} autotune_nodes={}/{}",
                        fmt_bytes(summary.total_inflight_bytes),
                        fmt_bytes(summary.total_ram_hwm_bytes),
                        summary.autotune_nodes,
                        summary.total_nodes
                    ));
                }
                lines.push(format!(
                    "counters: register={} hb={} lease_req={} granted={} expired={} requeued={} progress={}",
                    counters.map(|c| c.register_total).unwrap_or(0),
                    counters.map(|c| c.heartbeat_total).unwrap_or(0),
                    counters.map(|c| c.request_lease_total).unwrap_or(0),
                    counters.map(|c| c.leases_granted_total).unwrap_or(0),
                    counters.map(|c| c.leases_expired_total).unwrap_or(0),
                    counters.map(|c| c.ranges_requeued_total).unwrap_or(0),
                    counters.map(|c| c.progress_total).unwrap_or(0),
                ));
                lines.push(format!(
                    "live_lease_ids: {}",
                    s.live_leases
                        .iter()
                        .take(6)
                        .map(|l| l.lease_id.clone())
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            } else {
                lines.push("no snapshot yet".to_string());
            }
            lines.push(format!(
                "mode={:?} search='{}' jump='{}' status={}",
                app.input_mode, app.search_input, app.jump_input, app.status
            ));
            lines.join("\n")
        };
        let top_widget = Paragraph::new(top).block(
            Block::default()
                .title("Coordinator / Lease Overview")
                .borders(Borders::ALL),
        );
        f.render_widget(top_widget, root[0]);

        let mid = {
            let mut lines = Vec::new();
            if let Some(s) = &app.snapshot {
                lines.push(
                    "node_id rank hb_age_ms state inflight/cap ram_hwm/cap fetch_q decode_q pack_q autotune want pref queue pressure cd jitter breaches".to_string(),
                );
                for node in &s.nodes {
                    let hb_age_ms = s
                        .server_unix_time_ms
                        .saturating_sub(node.last_heartbeat_unix_time_ms);
                    let state = if hb_age_ms > app.args.stale_heartbeat_ms {
                        "stale"
                    } else {
                        "ok"
                    };
                    let (inflight, ram, fq, dq, pq, autotune_enabled, want, prefetch, queue, pressure_milli, cooldown, jitter_milli, jitter_breaches) = node
                        .stats
                        .as_ref()
                        .map(|st| {
                            (
                                st.inflight_bytes,
                                st.ram_high_water_bytes,
                                st.fetch_queue_depth,
                                st.decode_queue_depth,
                                st.pack_queue_depth,
                                st.autotune_enabled,
                                st.effective_want,
                                st.effective_prefetch_batches,
                                st.effective_max_queue_batches,
                                st.autotune_pressure_milli,
                                st.autotune_cooldown_ticks,
                                st.batch_payload_p95_over_p50_milli,
                                st.batch_jitter_slo_breaches_total,
                            )
                        })
                        .unwrap_or((0, 0, 0, 0, 0, false, 0, 0, 0, 0, 0, 0, 0));
                    let (cap_inflight, cap_ram) = node
                        .caps
                        .as_ref()
                        .map(|caps| (caps.max_inflight_bytes, caps.max_ram_bytes))
                        .unwrap_or((0, 0));
                    let inflight_ratio = ratio_pct(inflight, cap_inflight);
                    let ram_ratio = ratio_pct(ram, cap_ram);
                    lines.push(format!(
                        "{} {} {} {} {}/{}({}) {}/{}({}) {} {} {} {} {} {} {} {:.2} {} {:.2} {}",
                        node.node_id,
                        node.assigned_rank,
                        hb_age_ms,
                        state,
                        inflight,
                        cap_inflight,
                        inflight_ratio,
                        ram,
                        cap_ram,
                        ram_ratio,
                        fq,
                        dq,
                        pq,
                        if autotune_enabled { "on" } else { "off" },
                        want,
                        prefetch,
                        queue,
                        pressure_milli as f64 / 1000.0,
                        cooldown,
                        jitter_milli as f64 / 1000.0,
                        jitter_breaches
                    ));
                }
            } else {
                lines.push("no runtime data yet".to_string());
            }
            lines.push("autotune/jitter fields come from heartbeat NodeStats; zeros mean unavailable for that node.".to_string());
            lines.join("\n")
        };
        let mid_widget = Paragraph::new(mid).block(
            Block::default()
                .title("Runtime Panel")
                .borders(Borders::ALL),
        );
        f.render_widget(mid_widget, root[1]);

        let mut lines = Vec::new();
        let total = app.filtered_indices.len();
        let start = app
            .selected_row
            .saturating_sub(app.args.rows_per_page.saturating_sub(1));
        let end = (start + app.args.rows_per_page).min(total);
        lines.push(format!(
            "rows={} filtered={} selected={} keys[q / j k PgUp PgDn g Enter]",
            app.manifest_rows.len(),
            total,
            app.selected_row
        ));
        for i in start..end {
            let row = &app.manifest_rows[app.filtered_indices[i]];
            let marker = if i == app.selected_row { ">" } else { " " };
            lines.push(format!(
                "{} {:>8} {:>8} {}",
                marker,
                row.sample_id,
                row.byte_length.unwrap_or(0),
                row.location
            ));
        }
        let style = if app.input_mode != InputMode::Normal {
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
        } else {
            Style::default()
        };
        let bottom_widget = Paragraph::new(lines.join("\n")).style(style).block(
            Block::default()
                .title("Manifest Explorer")
                .borders(Borders::ALL),
        );
        f.render_widget(bottom_widget, root[2]);
    })?;
    Ok(())
}

fn handle_key(app: &mut App, key: crossterm::event::KeyEvent) -> bool {
    match app.input_mode {
        InputMode::Normal => match key.code {
            KeyCode::Char('q') => return true,
            KeyCode::Down | KeyCode::Char('j') => app.move_selection(1),
            KeyCode::Up | KeyCode::Char('k') => app.move_selection(-1),
            KeyCode::PageDown => app.move_selection(app.args.rows_per_page as isize),
            KeyCode::PageUp => app.move_selection(-(app.args.rows_per_page as isize)),
            KeyCode::Char('/') => app.input_mode = InputMode::Search,
            KeyCode::Char('g') => {
                app.jump_input.clear();
                app.input_mode = InputMode::Jump;
            }
            KeyCode::Char('o') => app.move_selection(0),
            _ => {}
        },
        InputMode::Search => match key.code {
            KeyCode::Esc => app.input_mode = InputMode::Normal,
            KeyCode::Enter => {
                app.apply_filter();
                app.input_mode = InputMode::Normal;
            }
            KeyCode::Backspace => {
                app.search_input.pop();
            }
            KeyCode::Char(c) => {
                if !key.modifiers.contains(KeyModifiers::CONTROL) {
                    app.search_input.push(c);
                }
            }
            _ => {}
        },
        InputMode::Jump => match key.code {
            KeyCode::Esc => app.input_mode = InputMode::Normal,
            KeyCode::Enter => {
                if let Ok(id) = app.jump_input.parse::<u64>() {
                    app.jump_to_sample_id(id);
                }
                app.jump_input.clear();
                app.input_mode = InputMode::Normal;
            }
            KeyCode::Backspace => {
                app.jump_input.pop();
            }
            KeyCode::Char(c) if c.is_ascii_digit() => app.jump_input.push(c),
            _ => {}
        },
    }
    false
}

async fn run_headless(mut app: App, mut client: CoordinatorClient<Channel>) -> Result<()> {
    let mut saw = PanelHealth::default();
    let mut last_refresh_error = String::new();
    let polls = app.args.headless_polls.max(1);
    for _ in 0..polls {
        if let Err(err) = refresh_once(&mut app, &mut client).await {
            last_refresh_error = err.to_string();
            app.status = format!("refresh error: {err}");
        }
        let h = app.current_health();
        saw.lease_non_empty |= h.lease_non_empty;
        saw.runtime_non_empty |= h.runtime_non_empty;
        saw.manifest_non_empty |= h.manifest_non_empty;
        tokio::time::sleep(Duration::from_millis(app.args.poll_ms)).await;
    }
    let snapshot_hint = app
        .snapshot
        .as_ref()
        .map(|s| {
            format!(
                "registered_nodes={} active_leases={} manifest_hash={}",
                s.registered_nodes, s.active_leases, s.manifest_hash
            )
        })
        .unwrap_or_else(|| "snapshot=none".to_string());
    let err_hint = if last_refresh_error.is_empty() {
        "refresh_error=none".to_string()
    } else {
        format!("refresh_error={}", last_refresh_error)
    };
    anyhow::ensure!(
        saw.lease_non_empty,
        "lease panel remained empty ({snapshot_hint}; {err_hint})"
    );
    anyhow::ensure!(
        saw.runtime_non_empty,
        "runtime panel remained empty ({snapshot_hint}; {err_hint})"
    );
    anyhow::ensure!(
        saw.manifest_non_empty,
        "manifest panel remained empty ({snapshot_hint}; {err_hint})"
    );
    println!(
        "mx8-tui headless OK lease={} runtime={} manifest={}",
        saw.lease_non_empty, saw.runtime_non_empty, saw.manifest_non_empty
    );
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let channel = Channel::from_shared(args.coord_url.clone())?
        .connect()
        .await?;
    let mut client = CoordinatorClient::new(channel)
        .max_decoding_message_size(args.grpc_max_message_bytes)
        .max_encoding_message_size(args.grpc_max_message_bytes);
    let mut app = App::new(args.clone());

    if args.headless_polls > 0 {
        return run_headless(app, client).await;
    }

    enable_raw_mode()?;
    let mut out = io::stdout();
    out.execute(EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(out);
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;

    let mut quit = false;
    while !quit {
        if app.last_refresh.elapsed() >= Duration::from_millis(app.args.poll_ms) {
            if let Err(err) = refresh_once(&mut app, &mut client).await {
                app.status = format!("refresh error: {err}");
            } else {
                app.status = "ok".to_string();
            }
            if app.input_mode == InputMode::Search {
                app.apply_filter();
            }
        }

        render(&mut terminal, &app)?;
        if event::poll(Duration::from_millis(100))? {
            let ev = event::read()?;
            if let Event::Key(key) = ev {
                quit = handle_key(&mut app, key);
            }
        }
    }

    disable_raw_mode()?;
    terminal.backend_mut().execute(LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}
