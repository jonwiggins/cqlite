// Page-based I/O layer between B-tree and filesystem.
//
// The pager reads and writes fixed-size pages, manages a page cache with
// LRU eviction, and coordinates page-level access for the B-tree layer.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::error::{Result, RsqliteError};
use crate::format::{self, DatabaseHeader, HEADER_SIZE};

/// A page number. Page 1 is the first page (contains the database header).
/// Page 0 is invalid.
pub type PageNumber = u32;

/// A single database page.
#[derive(Clone)]
pub struct Page {
    /// The page number (1-based).
    pub number: PageNumber,
    /// Raw page data.
    pub data: Vec<u8>,
    /// Whether this page has been modified since it was read.
    pub dirty: bool,
}

impl Page {
    pub fn new(number: PageNumber, page_size: usize) -> Self {
        Self {
            number,
            data: vec![0u8; page_size],
            dirty: false,
        }
    }
}

/// The pager manages reading and writing pages from/to the database file.
pub struct Pager {
    /// Path to the database file (None for in-memory databases).
    #[allow(dead_code)]
    path: Option<PathBuf>,
    /// The open file handle (None for in-memory databases).
    file: Option<File>,
    /// Database page size in bytes.
    page_size: usize,
    /// Parsed database header.
    pub header: DatabaseHeader,
    /// In-memory page cache. Maps page number to cached page.
    cache: HashMap<PageNumber, Page>,
    /// Maximum number of pages to cache.
    cache_limit: usize,
    /// LRU ordering: most recently accessed page numbers.
    lru: Vec<PageNumber>,
}

impl Pager {
    /// Open an existing database file, or create a new one.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        let file_exists = path.exists() && std::fs::metadata(&path).map(|m| m.len() > 0).unwrap_or(false);

        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;

        let header = if file_exists {
            // Read and parse the existing header.
            let mut buf = [0u8; HEADER_SIZE];
            file.seek(SeekFrom::Start(0))?;
            file.read_exact(&mut buf)?;
            DatabaseHeader::parse(&buf)?
        } else {
            // New database: write a fresh header and initialize page 1 as
            // an empty table leaf page for sqlite_master.
            let header = DatabaseHeader::new();
            let mut buf = [0u8; HEADER_SIZE];
            header.write(&mut buf);
            let page_size = header.page_size as usize;
            let mut page = vec![0u8; page_size];
            page[..HEADER_SIZE].copy_from_slice(&buf);

            // B-tree header for sqlite_master at offset 100.
            page[HEADER_SIZE] = 0x0D; // TableLeaf
            format::write_be_u16(&mut page, HEADER_SIZE + 1, 0);
            format::write_be_u16(&mut page, HEADER_SIZE + 3, 0);
            format::write_be_u16(&mut page, HEADER_SIZE + 5, page_size as u16);
            page[HEADER_SIZE + 7] = 0;

            file.seek(SeekFrom::Start(0))?;
            file.write_all(&page)?;
            file.sync_all()?;
            header
        };

        let page_size = header.page_size as usize;

        Ok(Self {
            path: Some(path),
            file: Some(file),
            page_size,
            header,
            cache: HashMap::new(),
            cache_limit: 100,
            lru: Vec::new(),
        })
    }

    /// Create an in-memory pager (no backing file).
    pub fn in_memory() -> Self {
        let header = DatabaseHeader::new();
        let page_size = header.page_size as usize;

        // Create the first page with the header.
        let mut page1 = Page::new(1, page_size);
        let mut hdr_buf = [0u8; HEADER_SIZE];
        header.write(&mut hdr_buf);
        page1.data[..HEADER_SIZE].copy_from_slice(&hdr_buf);

        // Initialize the B-tree header for sqlite_master (empty table leaf page).
        // The B-tree header starts at offset 100 on page 1.
        page1.data[HEADER_SIZE] = 0x0D; // TableLeaf
        format::write_be_u16(&mut page1.data, HEADER_SIZE + 1, 0); // first freeblock
        format::write_be_u16(&mut page1.data, HEADER_SIZE + 3, 0); // cell count
        format::write_be_u16(&mut page1.data, HEADER_SIZE + 5, page_size as u16); // cell content offset
        page1.data[HEADER_SIZE + 7] = 0; // fragmented free bytes

        let mut cache = HashMap::new();
        cache.insert(1, page1);

        Self {
            path: None,
            file: None,
            page_size,
            header,
            cache,
            cache_limit: 100,
            lru: vec![1],
        }
    }

    /// Get the page size.
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Read a page by its page number. Returns a reference to the cached page.
    pub fn get_page(&mut self, page_num: PageNumber) -> Result<&Page> {
        if page_num == 0 {
            return Err(RsqliteError::Corrupt("page number 0 is invalid".into()));
        }

        if !self.cache.contains_key(&page_num) {
            self.load_page(page_num)?;
        }
        self.touch_lru(page_num);

        Ok(self.cache.get(&page_num).unwrap())
    }

    /// Get a mutable reference to a page (marks it dirty).
    pub fn get_page_mut(&mut self, page_num: PageNumber) -> Result<&mut Page> {
        if page_num == 0 {
            return Err(RsqliteError::Corrupt("page number 0 is invalid".into()));
        }

        if !self.cache.contains_key(&page_num) {
            self.load_page(page_num)?;
        }
        self.touch_lru(page_num);

        let page = self.cache.get_mut(&page_num).unwrap();
        page.dirty = true;
        Ok(page)
    }

    /// Allocate a new page, returning its page number.
    pub fn allocate_page(&mut self) -> Result<PageNumber> {
        // TODO: Check freelist first.
        self.header.page_count += 1;
        let page_num = self.header.page_count;

        let page = Page::new(page_num, self.page_size);
        self.cache.insert(page_num, page);
        self.lru.push(page_num);

        Ok(page_num)
    }

    /// Write all dirty pages to disk and update the header.
    pub fn flush(&mut self) -> Result<()> {
        let file = match self.file.as_mut() {
            Some(f) => f,
            None => return Ok(()), // In-memory: nothing to flush.
        };

        // Write dirty pages.
        let dirty_pages: Vec<PageNumber> = self
            .cache
            .iter()
            .filter(|(_, p)| p.dirty)
            .map(|(&n, _)| n)
            .collect();

        for page_num in dirty_pages {
            let page = self.cache.get(&page_num).unwrap();
            let offset = (page_num as u64 - 1) * self.page_size as u64;
            file.seek(SeekFrom::Start(offset))?;
            file.write_all(&page.data)?;
            self.cache.get_mut(&page_num).unwrap().dirty = false;
        }

        // Update header on page 1.
        let mut hdr_buf = [0u8; HEADER_SIZE];
        self.header.write(&mut hdr_buf);
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&hdr_buf)?;

        // Also update cache for page 1.
        if let Some(page1) = self.cache.get_mut(&1) {
            page1.data[..HEADER_SIZE].copy_from_slice(&hdr_buf);
        }

        file.sync_all()?;
        Ok(())
    }

    /// Total number of pages in the database.
    pub fn page_count(&self) -> u32 {
        // If the file exists, compute from file size if header says 0.
        if self.header.page_count > 0 {
            return self.header.page_count;
        }
        if let Some(ref file) = self.file {
            if let Ok(metadata) = file.metadata() {
                let len = metadata.len();
                if len > 0 {
                    return (len / self.page_size as u64) as u32;
                }
            }
        }
        0
    }

    /// The usable size of each page (page size minus reserved space).
    pub fn usable_size(&self) -> usize {
        self.header.usable_size() as usize
    }

    /// Load a page from disk into the cache.
    fn load_page(&mut self, page_num: PageNumber) -> Result<()> {
        self.maybe_evict()?;

        let mut page = Page::new(page_num, self.page_size);

        if let Some(ref mut file) = self.file {
            let offset = (page_num as u64 - 1) * self.page_size as u64;
            let file_len = file.metadata()?.len();

            if offset < file_len {
                file.seek(SeekFrom::Start(offset))?;
                // Read what's available (file might be shorter than expected).
                let available = (file_len - offset).min(self.page_size as u64) as usize;
                file.read_exact(&mut page.data[..available])?;
            }
            // If the page is beyond the file, it stays zeroed (new page).
        }

        self.cache.insert(page_num, page);
        Ok(())
    }

    /// Update LRU tracking.
    fn touch_lru(&mut self, page_num: PageNumber) {
        self.lru.retain(|&p| p != page_num);
        self.lru.push(page_num);
    }

    /// Evict pages if cache is full.
    fn maybe_evict(&mut self) -> Result<()> {
        while self.cache.len() >= self.cache_limit {
            // Find the LRU page that isn't dirty.
            if let Some(pos) = self.lru.iter().position(|&p| {
                self.cache
                    .get(&p)
                    .map(|page| !page.dirty)
                    .unwrap_or(true)
            }) {
                let evict = self.lru.remove(pos);
                self.cache.remove(&evict);
            } else {
                // All pages are dirty — flush first, then evict.
                self.flush()?;
                if let Some(evict) = self.lru.first().copied() {
                    self.lru.remove(0);
                    self.cache.remove(&evict);
                }
            }
        }
        Ok(())
    }
}

/// The offset within a page where the B-tree header starts.
/// For page 1, this is after the 100-byte database header.
/// For all other pages, it's at the start.
pub fn btree_header_offset(page_num: PageNumber) -> usize {
    if page_num == 1 {
        format::HEADER_SIZE
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_in_memory_pager() {
        let mut pager = Pager::in_memory();
        assert_eq!(pager.page_size(), 4096);

        // Page 1 should exist with the header.
        let page = pager.get_page(1).unwrap();
        assert_eq!(page.number, 1);
        assert_eq!(&page.data[0..16], format::MAGIC.as_slice());
    }

    #[test]
    fn test_allocate_page() {
        let mut pager = Pager::in_memory();
        let p2 = pager.allocate_page().unwrap();
        assert_eq!(p2, 2);
        let p3 = pager.allocate_page().unwrap();
        assert_eq!(p3, 3);

        // We should be able to read/write the new pages.
        let page = pager.get_page_mut(p2).unwrap();
        page.data[0] = 0x42;

        let page = pager.get_page(p2).unwrap();
        assert_eq!(page.data[0], 0x42);
    }

    #[test]
    fn test_file_pager_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");

        // Create and write.
        {
            let mut pager = Pager::open(&db_path).unwrap();
            let p2 = pager.allocate_page().unwrap();
            let page = pager.get_page_mut(p2).unwrap();
            page.data[0] = 0xAB;
            page.data[1] = 0xCD;
            pager.flush().unwrap();
        }

        // Reopen and read.
        {
            let mut pager = Pager::open(&db_path).unwrap();
            assert_eq!(pager.header.page_size, 4096);
            let page = pager.get_page(2).unwrap();
            assert_eq!(page.data[0], 0xAB);
            assert_eq!(page.data[1], 0xCD);
        }

        // Clean up.
        fs::remove_file(&db_path).ok();
    }

    #[test]
    fn test_invalid_page_zero() {
        let mut pager = Pager::in_memory();
        assert!(pager.get_page(0).is_err());
    }

    #[test]
    fn test_btree_header_offset() {
        assert_eq!(btree_header_offset(1), 100);
        assert_eq!(btree_header_offset(2), 0);
        assert_eq!(btree_header_offset(100), 0);
    }
}
