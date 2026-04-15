use crate::acp::connection::AcpConnection;
use crate::config::AgentConfig;
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::path::PathBuf;
use tokio::sync::RwLock;
use tokio::time::Instant;
use tracing::{error, info, warn};

/// Combined state protected by a single lock to prevent deadlocks.
/// Lock ordering: always acquire `state` before any operation on either map.
struct PoolState {
    /// Active connections: thread_key → AcpConnection.
    active: HashMap<String, AcpConnection>,
    /// Suspended sessions: thread_key → ACP sessionId.
    /// Saved on eviction so sessions can be resumed via `session/load`.
    suspended: HashMap<String, String>,
    /// Per-session workspace directories for cleanup on drop.
    /// Only populated when max_sessions > 1 (multi-session mode).
    session_dirs: HashMap<String, PathBuf>,
}

pub struct SessionPool {
    state: RwLock<PoolState>,
    config: AgentConfig,
    max_sessions: usize,
}

impl SessionPool {
    pub fn new(config: AgentConfig, max_sessions: usize) -> Self {
        Self {
            state: RwLock::new(PoolState {
                active: HashMap::new(),
                suspended: HashMap::new(),
                session_dirs: HashMap::new(),
            }),
            config,
            max_sessions,
        }
    }

    /// Create an isolated workspace for a session via `git clone --local`.
    /// Only active when max_sessions > 1 (multi-session mode).
    /// Returns session-specific path, or base working_dir on failure.
    async fn create_session_workspace(&self, state: &mut PoolState, thread_id: &str) -> String {
        let base = &self.config.working_dir;

        if self.max_sessions <= 1 {
            return base.clone();
        }

        let session_dir = format!("{}/sessions/{}", base, thread_id);

        // Reuse existing session dir (reconnect case)
        if tokio::fs::metadata(&session_dir).await.is_ok() {
            info!(%thread_id, dir = %session_dir, "reusing session workspace");
            return session_dir;
        }

        // Create isolated session directory.
        // Strategy: try git clone --local first (if base is a git repo).
        // Fallback: just mkdir (agent will clone repos as needed).
        // Either way, each session gets its own cwd for file isolation.
        let sessions_base = format!("{}/sessions", base);
        if let Err(e) = tokio::fs::create_dir_all(&sessions_base).await {
            warn!(%thread_id, error = %e, "mkdir sessions failed, using default");
            return base.clone();
        }

        // Try git clone first (works when base is a git repo, e.g. local dev)
        let clone_output = tokio::process::Command::new("git")
            .args(["clone", "--local", base, &session_dir])
            .output()
            .await;

        let clone_ok = clone_output.as_ref().map(|o| o.status.success()).unwrap_or(false);

        if clone_ok {
            info!(%thread_id, dir = %session_dir, "created session workspace via git clone");
        } else {
            // Fallback: just create an empty directory (for PV-mounted pods)
            match tokio::fs::create_dir_all(&session_dir).await {
                Ok(()) => info!(%thread_id, dir = %session_dir, "created session workspace (empty dir)"),
                Err(e) => {
                    warn!(%thread_id, error = %e, "session dir creation failed, using default");
                    return base.clone();
                }
            }
        }

        state.session_dirs.insert(thread_id.to_string(), PathBuf::from(&session_dir));
        session_dir
    }

    /// Clean up a session's isolated workspace.
    async fn cleanup_session_workspace(state: &mut PoolState, thread_id: &str) {
        if let Some(dir) = state.session_dirs.remove(thread_id) {
            if dir.exists() {
                match tokio::fs::remove_dir_all(&dir).await {
                    Ok(()) => info!(%thread_id, dir = %dir.display(), "cleaned up session workspace"),
                    Err(e) => error!(%thread_id, dir = %dir.display(), error = %e, "workspace cleanup failed"),
                }
            }
        }
    }

    pub async fn get_or_create(&self, thread_id: &str) -> Result<()> {
        // Check if alive connection exists
        {
            let state = self.state.read().await;
            if let Some(conn) = state.active.get(thread_id) {
                if conn.alive() {
                    return Ok(());
                }
            }
        }

        // Need to create or rebuild
        let mut state = self.state.write().await;

        // Double-check after acquiring write lock
        if let Some(conn) = state.active.get(thread_id) {
            if conn.alive() {
                return Ok(());
            }
            warn!(thread_id, "stale connection, rebuilding");
            suspend_entry(&mut state, thread_id);
        }

        if state.active.len() >= self.max_sessions {
            // LRU evict: suspend the oldest idle session to make room
            let oldest = state.active
                .iter()
                .min_by_key(|(_, c)| c.last_active)
                .map(|(k, _)| k.clone());
            if let Some(key) = oldest {
                info!(evicted = %key, "pool full, suspending oldest idle session");
                suspend_entry(&mut state, &key);
            } else {
                return Err(anyhow!("pool exhausted ({} sessions)", self.max_sessions));
            }
        }

        // Create isolated workspace (multi-session only, no-op if max_sessions=1)
        let session_cwd = self.create_session_workspace(&mut state, thread_id).await;

        let mut conn = AcpConnection::spawn(
            &self.config.command,
            &self.config.args,
            &session_cwd,
            &self.config.env,
        )
        .await?;

        conn.initialize().await?;

        // Try to resume a suspended session via session/load
        let saved_session_id = state.suspended.remove(thread_id);
        let mut resumed = false;
        if let Some(ref sid) = saved_session_id {
            if conn.supports_load_session {
                match conn.session_load(sid, &session_cwd).await {
                    Ok(()) => {
                        info!(thread_id, session_id = %sid, "session resumed via session/load");
                        resumed = true;
                    }
                    Err(e) => {
                        warn!(thread_id, session_id = %sid, error = %e, "session/load failed, creating new session");
                    }
                }
            }
        }

        if !resumed {
            conn.session_new(&session_cwd).await?;
            if saved_session_id.is_some() {
                conn.session_reset = true;
            }
        }

        state.active.insert(thread_id.to_string(), conn);
        Ok(())
    }

    /// Get mutable access to a connection. Caller must have called get_or_create first.
    pub async fn with_connection<F, R>(&self, thread_id: &str, f: F) -> Result<R>
    where
        F: FnOnce(&mut AcpConnection) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<R>> + Send + '_>>,
    {
        let mut state = self.state.write().await;
        let conn = state.active
            .get_mut(thread_id)
            .ok_or_else(|| anyhow!("no connection for thread {thread_id}"))?;
        f(conn).await
    }

    pub async fn cleanup_idle(&self, ttl_secs: u64) {
        let cutoff = Instant::now() - std::time::Duration::from_secs(ttl_secs);
        let mut state = self.state.write().await;
        let stale: Vec<String> = state.active
            .iter()
            .filter(|(_, c)| c.last_active < cutoff || !c.alive())
            .map(|(k, _)| k.clone())
            .collect();
        for key in stale {
            info!(thread_id = %key, "cleaning up idle session");
            suspend_entry(&mut state, &key);
            Self::cleanup_session_workspace(&mut state, &key).await;
        }
    }

    pub async fn shutdown(&self) {
        let mut state = self.state.write().await;
        // Clean up all session workspaces
        let thread_ids: Vec<String> = state.session_dirs.keys().cloned().collect();
        for tid in &thread_ids {
            Self::cleanup_session_workspace(&mut state, tid).await;
        }
        let count = state.active.len();
        state.active.clear(); // Drop impl kills process groups
        info!(count, "pool shutdown complete");
    }
}

/// Suspend a connection: save its sessionId to the suspended map and remove
/// from active. The connection is dropped, triggering process group kill.
fn suspend_entry(state: &mut PoolState, thread_id: &str) {
    if let Some(conn) = state.active.remove(thread_id) {
        if let Some(sid) = &conn.acp_session_id {
            info!(thread_id, session_id = %sid, "suspending session");
            state.suspended.insert(thread_id.to_string(), sid.clone());
        }
        // conn dropped here → Drop impl kills process group
    }
}
