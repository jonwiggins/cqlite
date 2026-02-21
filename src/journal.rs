// Transaction rollback journal.
//
// Before modifying any page, the original page content is saved to a journal
// file. On COMMIT the journal is deleted. On ROLLBACK (or crash recovery),
// pages are restored from the journal to their pre-transaction state.
//
// Journal file format (simplified):
//   - 8-byte magic: "RSQLJOURNAL\0" (padded to 8 bytes: b"RSQLJRNL")
//   - 4-byte page count (big-endian u32): number of pages in journal
//   - 4-byte page size (big-endian u32)
//   - For each journaled page:
//       - 4-byte page number (big-endian u32)
//       - page_size bytes of original page data

use std::collections::HashSet;

/// Result of a journal rollback: saved pages and optional original page count.
type RollbackResult = (Vec<(u32, Vec<u8>)>, Option<u32>);
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::error::{Result, RsqliteError};

const JOURNAL_MAGIC: &[u8; 8] = b"RSQLJRNL";
const JOURNAL_HEADER_SIZE: usize = 16; // 8 magic + 4 page_count + 4 page_size

/// A rollback journal that stores original page contents before modification.
pub struct Journal {
    /// Path to the journal file (database_path + "-journal").
    path: Option<PathBuf>,
    /// Open file handle for the journal.
    file: Option<File>,
    /// Page size.
    page_size: usize,
    /// Set of page numbers already journaled in this transaction.
    journaled_pages: HashSet<u32>,
    /// Number of pages written to the journal.
    page_count: u32,
    /// Whether the journal is active (a transaction is in progress).
    active: bool,
    /// In-memory journal pages (used for in-memory databases or as a backup).
    mem_pages: Vec<(u32, Vec<u8>)>,
    /// Saved header page count, for restoring on rollback.
    saved_page_count: Option<u32>,
}

impl Journal {
    /// Create a journal for a file-backed database.
    pub fn new(db_path: Option<&Path>, page_size: usize) -> Self {
        let path = db_path.map(|p| {
            let mut jp = p.to_path_buf();
            let name = jp
                .file_name()
                .map(|n| {
                    let mut s = n.to_os_string();
                    s.push("-journal");
                    s
                })
                .unwrap_or_default();
            jp.set_file_name(name);
            jp
        });

        Self {
            path,
            file: None,
            page_size,
            journaled_pages: HashSet::new(),
            page_count: 0,
            active: false,
            mem_pages: Vec::new(),
            saved_page_count: None,
        }
    }

    /// Create a journal for an in-memory database.
    pub fn in_memory(page_size: usize) -> Self {
        Self {
            path: None,
            file: None,
            page_size,
            journaled_pages: HashSet::new(),
            page_count: 0,
            active: false,
            mem_pages: Vec::new(),
            saved_page_count: None,
        }
    }

    /// Whether the journal is currently active.
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Begin a new transaction by creating the journal file.
    /// `current_page_count` is the database's current page count, saved for rollback.
    pub fn begin_with_page_count(&mut self, current_page_count: u32) -> Result<()> {
        if self.active {
            return Err(RsqliteError::Runtime(
                "cannot start a transaction within a transaction".into(),
            ));
        }

        if let Some(ref path) = self.path {
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(path)?;
            self.file = Some(file);

            // Write journal header.
            self.write_header()?;
        }

        self.journaled_pages.clear();
        self.page_count = 0;
        self.mem_pages.clear();
        self.saved_page_count = Some(current_page_count);
        self.active = true;
        Ok(())
    }

    /// Begin a new transaction (convenience method that saves page count 0).
    pub fn begin(&mut self) -> Result<()> {
        self.begin_with_page_count(0)
    }

    /// Journal a page: save its original content before modification.
    /// Does nothing if the page has already been journaled in this transaction.
    pub fn journal_page(&mut self, page_num: u32, data: &[u8]) -> Result<()> {
        if !self.active {
            return Ok(());
        }

        if self.journaled_pages.contains(&page_num) {
            return Ok(());
        }

        // Always store in memory for rollback support (needed for in-memory dbs).
        self.mem_pages
            .push((page_num, data[..self.page_size].to_vec()));

        if let Some(ref mut file) = self.file {
            // Seek to the end and write the page entry.
            let offset =
                JOURNAL_HEADER_SIZE as u64 + self.page_count as u64 * (4 + self.page_size as u64);
            file.seek(SeekFrom::Start(offset))?;

            // Write page number (big-endian u32).
            file.write_all(&page_num.to_be_bytes())?;
            // Write page data.
            file.write_all(&data[..self.page_size])?;

            // Update the page count in the header.
            self.page_count += 1;
            file.seek(SeekFrom::Start(8))?;
            file.write_all(&self.page_count.to_be_bytes())?;

            file.sync_all()?;
        }

        self.journaled_pages.insert(page_num);
        Ok(())
    }

    /// Commit the transaction: delete the journal file.
    pub fn commit(&mut self) -> Result<()> {
        if !self.active {
            return Ok(());
        }

        // Close the file first.
        self.file = None;

        // Delete the journal file.
        if let Some(ref path) = self.path {
            if path.exists() {
                fs::remove_file(path)?;
            }
        }

        self.journaled_pages.clear();
        self.mem_pages.clear();
        self.page_count = 0;
        self.saved_page_count = None;
        self.active = false;
        Ok(())
    }

    /// Rollback the transaction: restore pages from journal, then delete it.
    /// Returns the list of (page_number, original_data) pairs to restore,
    /// and the saved page count to restore.
    pub fn rollback(&mut self) -> Result<RollbackResult> {
        if !self.active {
            return Ok((vec![], None));
        }

        // Use in-memory pages if available, otherwise read from file.
        let pages = if !self.mem_pages.is_empty() {
            std::mem::take(&mut self.mem_pages)
        } else {
            self.read_journal_pages()?
        };

        let saved_page_count = self.saved_page_count;

        // Close the file.
        self.file = None;

        // Delete the journal file.
        if let Some(ref path) = self.path {
            if path.exists() {
                fs::remove_file(path)?;
            }
        }

        self.journaled_pages.clear();
        self.mem_pages.clear();
        self.page_count = 0;
        self.saved_page_count = None;
        self.active = false;
        Ok((pages, saved_page_count))
    }

    /// Read all journaled pages from the journal file.
    fn read_journal_pages(&mut self) -> Result<Vec<(u32, Vec<u8>)>> {
        let file = match self.file.as_mut() {
            Some(f) => f,
            None => return Ok(vec![]),
        };

        file.seek(SeekFrom::Start(0))?;

        // Read and verify header.
        let mut header = [0u8; JOURNAL_HEADER_SIZE];
        if file.read_exact(&mut header).is_err() {
            return Ok(vec![]);
        }

        if &header[..8] != JOURNAL_MAGIC {
            return Ok(vec![]);
        }

        let page_count = u32::from_be_bytes([header[8], header[9], header[10], header[11]]);
        let page_size =
            u32::from_be_bytes([header[12], header[13], header[14], header[15]]) as usize;

        let mut pages = Vec::with_capacity(page_count as usize);

        for _ in 0..page_count {
            let mut num_buf = [0u8; 4];
            file.read_exact(&mut num_buf)?;
            let page_num = u32::from_be_bytes(num_buf);

            let mut data = vec![0u8; page_size];
            file.read_exact(&mut data)?;

            pages.push((page_num, data));
        }

        Ok(pages)
    }

    /// Write the journal header.
    fn write_header(&mut self) -> Result<()> {
        let file = match self.file.as_mut() {
            Some(f) => f,
            None => return Ok(()),
        };

        file.seek(SeekFrom::Start(0))?;
        file.write_all(JOURNAL_MAGIC)?;
        file.write_all(&self.page_count.to_be_bytes())?;
        file.write_all(&(self.page_size as u32).to_be_bytes())?;
        file.sync_all()?;
        Ok(())
    }

    /// Check if a hot journal exists for a database path and recover from it.
    /// Returns the list of pages to restore, or empty if no recovery needed.
    pub fn hot_journal_recovery(db_path: &Path, page_size: usize) -> Result<Vec<(u32, Vec<u8>)>> {
        let mut jp = db_path.to_path_buf();
        let name = jp
            .file_name()
            .map(|n| {
                let mut s = n.to_os_string();
                s.push("-journal");
                s
            })
            .unwrap_or_default();
        jp.set_file_name(name);

        if !jp.exists() {
            return Ok(vec![]);
        }

        // Read the journal.
        let mut file = match File::open(&jp) {
            Ok(f) => f,
            Err(_) => return Ok(vec![]),
        };

        let mut header = [0u8; JOURNAL_HEADER_SIZE];
        if file.read_exact(&mut header).is_err() {
            // Corrupt or empty journal — just delete it.
            fs::remove_file(&jp).ok();
            return Ok(vec![]);
        }

        if &header[..8] != JOURNAL_MAGIC {
            fs::remove_file(&jp).ok();
            return Ok(vec![]);
        }

        let page_count = u32::from_be_bytes([header[8], header[9], header[10], header[11]]);
        let stored_page_size =
            u32::from_be_bytes([header[12], header[13], header[14], header[15]]) as usize;

        if stored_page_size != page_size {
            fs::remove_file(&jp).ok();
            return Err(RsqliteError::Corrupt("journal page size mismatch".into()));
        }

        let mut pages = Vec::with_capacity(page_count as usize);

        for _ in 0..page_count {
            let mut num_buf = [0u8; 4];
            if file.read_exact(&mut num_buf).is_err() {
                break;
            }
            let page_num = u32::from_be_bytes(num_buf);

            let mut data = vec![0u8; page_size];
            if file.read_exact(&mut data).is_err() {
                break;
            }

            pages.push((page_num, data));
        }

        // Delete the journal after reading.
        drop(file);
        fs::remove_file(&jp).ok();

        Ok(pages)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_in_memory_journal_is_noop() {
        let mut j = Journal::in_memory(4096);
        assert!(!j.is_active());
        j.begin().unwrap();
        assert!(j.is_active());
        j.journal_page(1, &[0u8; 4096]).unwrap();
        j.commit().unwrap();
        assert!(!j.is_active());
    }

    #[test]
    fn test_journal_begin_commit() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        std::fs::write(&db_path, b"dummy").unwrap();

        let mut j = Journal::new(Some(&db_path), 4096);
        j.begin().unwrap();
        assert!(j.is_active());

        // Journal a page.
        let data = vec![0xABu8; 4096];
        j.journal_page(1, &data).unwrap();

        // Journal file should exist.
        let journal_path = dir.path().join("test.db-journal");
        assert!(journal_path.exists());

        // Commit should delete journal.
        j.commit().unwrap();
        assert!(!j.is_active());
        assert!(!journal_path.exists());
    }

    #[test]
    fn test_journal_rollback_restores_pages() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        std::fs::write(&db_path, b"dummy").unwrap();

        let mut j = Journal::new(Some(&db_path), 4096);
        j.begin().unwrap();

        let data1 = vec![0x11u8; 4096];
        let data2 = vec![0x22u8; 4096];
        j.journal_page(1, &data1).unwrap();
        j.journal_page(2, &data2).unwrap();

        // Journaling same page again should be a no-op.
        j.journal_page(1, &[0xFFu8; 4096]).unwrap();

        let (pages, _) = j.rollback().unwrap();
        assert_eq!(pages.len(), 2);
        assert_eq!(pages[0].0, 1);
        assert_eq!(pages[0].1, data1);
        assert_eq!(pages[1].0, 2);
        assert_eq!(pages[1].1, data2);
    }

    #[test]
    fn test_hot_journal_recovery() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        std::fs::write(&db_path, b"dummy").unwrap();

        let page_size = 4096;

        // Create a journal manually.
        {
            let mut j = Journal::new(Some(&db_path), page_size);
            j.begin().unwrap();
            let data = vec![0x42u8; page_size];
            j.journal_page(3, &data).unwrap();
            // Simulate crash: don't commit, just drop.
            // But we need to keep the file, so set active to false without deleting.
            j.active = false;
            j.file = None;
        }

        // Now recover.
        let pages = Journal::hot_journal_recovery(&db_path, page_size).unwrap();
        assert_eq!(pages.len(), 1);
        assert_eq!(pages[0].0, 3);
        assert_eq!(pages[0].1[0], 0x42);

        // Journal should be deleted after recovery.
        let journal_path = dir.path().join("test.db-journal");
        assert!(!journal_path.exists());
    }

    #[test]
    fn test_double_begin_fails() {
        let mut j = Journal::in_memory(4096);
        j.begin().unwrap();
        assert!(j.begin().is_err());
    }

    #[test]
    fn test_no_hot_journal() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("nojournal.db");
        let pages = Journal::hot_journal_recovery(&db_path, 4096).unwrap();
        assert!(pages.is_empty());
    }
}
