use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::error::{Result, RsqliteError};
use crate::format::{DatabaseHeader, DEFAULT_PAGE_SIZE, MAGIC_STRING};

/// A page of data from the database file.
#[derive(Debug, Clone)]
pub struct Page {
    pub data: Vec<u8>,
    pub dirty: bool,
}

/// The Pager manages reading/writing pages to/from the database file.
pub struct Pager {
    file: Option<File>,
    #[allow(dead_code)]
    path: Option<PathBuf>,
    pub page_size: u32,
    pub header: DatabaseHeader,
    cache: HashMap<u32, Page>,
    next_page: u32,
    pub in_transaction: bool,
    journal_pages: HashMap<u32, Vec<u8>>, // original pages for rollback
}

impl Pager {
    /// Create a new in-memory pager (no file backing).
    pub fn new_memory() -> Self {
        let page_size = DEFAULT_PAGE_SIZE;
        let header = DatabaseHeader::new(page_size);

        // Initialize page 1 with database header + empty leaf table (sqlite_master)
        let mut page1_data = vec![0u8; page_size as usize];
        let header_bytes = header.serialize();
        page1_data[..100].copy_from_slice(&header_bytes);
        // B-tree leaf table header at offset 100
        page1_data[100] = 0x0d; // leaf table
                                // cell count = 0, rest is zeros

        let mut cache = HashMap::new();
        cache.insert(
            1,
            Page {
                data: page1_data,
                dirty: false,
            },
        );

        Pager {
            file: None,
            path: None,
            page_size,
            header,
            cache,
            next_page: 2, // page 1 is reserved for the header/schema
            in_transaction: false,
            journal_pages: HashMap::new(),
        }
    }

    /// Open or create a database file.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file_exists = path.exists() && std::fs::metadata(&path)?.len() > 0;

        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;

        if file_exists {
            // Read existing database
            let mut header_buf = vec![0u8; 100];
            file.read_exact(&mut header_buf)?;
            let header = DatabaseHeader::parse(&header_buf)?;
            let page_size = header.page_size;

            let file_len = file.metadata()?.len();
            let total_pages = if file_len > 0 {
                (file_len / page_size as u64) as u32
            } else {
                1
            };

            Ok(Pager {
                file: Some(file),
                path: Some(path),
                page_size,
                header,
                cache: HashMap::new(),
                next_page: total_pages + 1,
                in_transaction: false,
                journal_pages: HashMap::new(),
            })
        } else {
            // New database
            let page_size = DEFAULT_PAGE_SIZE;
            let header = DatabaseHeader::new(page_size);

            // Write initial page 1 (header + empty leaf table for sqlite_master)
            let mut page1 = vec![0u8; page_size as usize];
            let header_bytes = header.serialize();
            page1[..100].copy_from_slice(&header_bytes);

            // Initialize page 1 as a leaf table B-tree page (for sqlite_master)
            // Header starts at offset 100 on page 1
            page1[100] = 0x0d; // leaf table
                               // first freeblock = 0
            page1[101] = 0;
            page1[102] = 0;
            // cell count = 0
            page1[103] = 0;
            page1[104] = 0;
            // cell content offset = 0 (means 65536 if page_size == 65536, else page_size)
            page1[105] = 0;
            page1[106] = 0;
            // fragmented free bytes = 0
            page1[107] = 0;

            file.seek(SeekFrom::Start(0))?;
            file.write_all(&page1)?;
            file.sync_all()?;

            Ok(Pager {
                file: Some(file),
                path: Some(path),
                page_size,
                header,
                cache: HashMap::new(),
                next_page: 2,
                in_transaction: false,
                journal_pages: HashMap::new(),
            })
        }
    }

    /// Get a page by page number (1-based).
    pub fn get_page(&mut self, page_num: u32) -> Result<&Page> {
        if !self.cache.contains_key(&page_num) {
            let data = self.read_page_from_file(page_num)?;
            self.cache.insert(page_num, Page { data, dirty: false });
        }
        Ok(self.cache.get(&page_num).unwrap())
    }

    /// Get a mutable reference to a page.
    pub fn get_page_mut(&mut self, page_num: u32) -> Result<&mut Page> {
        if !self.cache.contains_key(&page_num) {
            let data = self.read_page_from_file(page_num)?;
            self.cache.insert(page_num, Page { data, dirty: false });
        }

        // Save original for rollback if in transaction
        if self.in_transaction && !self.journal_pages.contains_key(&page_num) {
            let page = self.cache.get(&page_num).unwrap();
            self.journal_pages.insert(page_num, page.data.clone());
        }

        let page = self.cache.get_mut(&page_num).unwrap();
        page.dirty = true;
        Ok(page)
    }

    /// Allocate a new page.
    pub fn allocate_page(&mut self) -> Result<u32> {
        // Check freelist first
        if self.header.first_freelist_trunk_page != 0 {
            return self.pop_freelist_page();
        }

        let page_num = self.next_page;
        self.next_page += 1;
        let data = vec![0u8; self.page_size as usize];
        self.cache.insert(page_num, Page { data, dirty: true });
        self.header.database_size_pages = self.next_page - 1;
        Ok(page_num)
    }

    /// Add a page to the freelist.
    pub fn free_page(&mut self, _page_num: u32) -> Result<()> {
        // Simplified: just track in header
        self.header.total_freelist_pages += 1;
        Ok(())
    }

    fn pop_freelist_page(&mut self) -> Result<u32> {
        let trunk = self.header.first_freelist_trunk_page;
        if trunk == 0 {
            return Err(RsqliteError::Internal("No freelist pages".into()));
        }

        let page_data = self.read_page_from_file(trunk)?;
        let next_trunk = crate::format::read_be_u32(&page_data[0..4]);
        let count = crate::format::read_be_u32(&page_data[4..8]);

        if count > 0 {
            // Pop the last leaf page from this trunk
            let leaf_page = crate::format::read_be_u32(
                &page_data[(8 + (count - 1) as usize * 4)..(8 + count as usize * 4)],
            );
            // Update count
            let new_count = count - 1;
            let page = self.get_page_mut(trunk)?;
            crate::format::write_be_u32(&mut page.data[4..8], new_count);

            self.header.total_freelist_pages -= 1;
            let data = vec![0u8; self.page_size as usize];
            self.cache.insert(leaf_page, Page { data, dirty: true });
            Ok(leaf_page)
        } else {
            // Use the trunk page itself
            self.header.first_freelist_trunk_page = next_trunk;
            self.header.total_freelist_pages -= 1;
            let data = vec![0u8; self.page_size as usize];
            self.cache.insert(trunk, Page { data, dirty: true });
            Ok(trunk)
        }
    }

    fn read_page_from_file(&mut self, page_num: u32) -> Result<Vec<u8>> {
        if let Some(ref mut file) = self.file {
            let offset = (page_num as u64 - 1) * self.page_size as u64;
            let file_len = file.metadata()?.len();
            if offset >= file_len {
                // Page doesn't exist yet, return zeros
                return Ok(vec![0u8; self.page_size as usize]);
            }
            file.seek(SeekFrom::Start(offset))?;
            let mut data = vec![0u8; self.page_size as usize];
            let bytes_available = (file_len - offset).min(self.page_size as u64) as usize;
            file.read_exact(&mut data[..bytes_available])?;
            Ok(data)
        } else {
            // In-memory: return zeros for new pages
            Ok(vec![0u8; self.page_size as usize])
        }
    }

    /// Write all dirty pages to disk.
    pub fn flush(&mut self) -> Result<()> {
        // Update header on page 1
        self.update_header_on_page1()?;

        if let Some(ref mut file) = self.file {
            let dirty_pages: Vec<(u32, Vec<u8>)> = self
                .cache
                .iter()
                .filter(|(_, page)| page.dirty)
                .map(|(&num, page)| (num, page.data.clone()))
                .collect();

            for (page_num, data) in &dirty_pages {
                let offset = (*page_num as u64 - 1) * self.page_size as u64;
                file.seek(SeekFrom::Start(offset))?;
                file.write_all(data)?;
            }
            file.sync_all()?;

            for (page_num, _) in &dirty_pages {
                if let Some(page) = self.cache.get_mut(page_num) {
                    page.dirty = false;
                }
            }
        } else {
            // In-memory: just clear dirty flags
            for page in self.cache.values_mut() {
                page.dirty = false;
            }
        }

        self.journal_pages.clear();
        Ok(())
    }

    fn update_header_on_page1(&mut self) -> Result<()> {
        // Ensure page 1 is in cache
        if !self.cache.contains_key(&1) {
            let data = self.read_page_from_file(1)?;
            self.cache.insert(1, Page { data, dirty: false });
        }

        let header_bytes = self.header.serialize();
        let page1 = self.cache.get_mut(&1).unwrap();
        page1.data[..100].copy_from_slice(&header_bytes);
        page1.dirty = true;
        Ok(())
    }

    /// Begin a transaction.
    pub fn begin_transaction(&mut self) -> Result<()> {
        self.in_transaction = true;
        self.journal_pages.clear();
        Ok(())
    }

    /// Commit a transaction.
    pub fn commit_transaction(&mut self) -> Result<()> {
        self.header.file_change_counter += 1;
        self.flush()?;
        self.in_transaction = false;
        Ok(())
    }

    /// Rollback a transaction.
    pub fn rollback_transaction(&mut self) -> Result<()> {
        // Restore original pages
        for (page_num, data) in self.journal_pages.drain() {
            self.cache.insert(page_num, Page { data, dirty: false });
        }
        self.in_transaction = false;
        Ok(())
    }

    /// Get total number of pages.
    pub fn page_count(&self) -> u32 {
        self.next_page - 1
    }

    /// Check if a file is a valid SQLite database.
    pub fn is_valid_database<P: AsRef<Path>>(path: P) -> bool {
        if let Ok(mut file) = File::open(path) {
            let mut buf = [0u8; 16];
            if file.read_exact(&mut buf).is_ok() {
                return &buf == MAGIC_STRING;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pager() {
        let mut pager = Pager::new_memory();
        assert_eq!(pager.page_size, DEFAULT_PAGE_SIZE);

        let page_num = pager.allocate_page().unwrap();
        assert_eq!(page_num, 2);

        let page = pager.get_page_mut(page_num).unwrap();
        page.data[0] = 42;

        let page = pager.get_page(page_num).unwrap();
        assert_eq!(page.data[0], 42);
    }

    #[test]
    fn test_file_pager() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.db");

        {
            let mut pager = Pager::open(&path).unwrap();
            let page_num = pager.allocate_page().unwrap();
            let page = pager.get_page_mut(page_num).unwrap();
            page.data[0] = 0x42;
            pager.flush().unwrap();
        }

        {
            let mut pager = Pager::open(&path).unwrap();
            assert_eq!(pager.page_size, DEFAULT_PAGE_SIZE);
            let page = pager.get_page(2).unwrap();
            assert_eq!(page.data[0], 0x42);
        }
    }
}
