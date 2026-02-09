/// Transaction journal module.
///
/// Implements a simple rollback journal for crash recovery.
/// The journal stores original page content before modification,
/// allowing rollback on failure.
///
/// The actual journal management is integrated into the Pager module
/// for simplicity. This module provides the journal file format
/// and recovery logic.
use std::fs;
use std::path::{Path, PathBuf};

/// Get the journal file path for a database file.
pub fn journal_path(db_path: &Path) -> PathBuf {
    let mut path = db_path.to_path_buf();
    let name = path
        .file_name()
        .map(|n| format!("{}-journal", n.to_string_lossy()))
        .unwrap_or_else(|| "journal".into());
    path.set_file_name(name);
    path
}

/// Check if a hot journal exists (indicates a crash during transaction).
pub fn has_hot_journal(db_path: &Path) -> bool {
    let journal = journal_path(db_path);
    journal.exists() && fs::metadata(&journal).map(|m| m.len() > 0).unwrap_or(false)
}

/// Delete the journal file after successful commit.
pub fn delete_journal(db_path: &Path) -> std::io::Result<()> {
    let journal = journal_path(db_path);
    if journal.exists() {
        fs::remove_file(journal)?;
    }
    Ok(())
}
