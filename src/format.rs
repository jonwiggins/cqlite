// SQLite3 file format constants and header parsing.
// See: https://www.sqlite.org/fileformat2.html

use crate::error::{Result, RsqliteError};

/// The magic string at the start of every SQLite3 database file.
pub const MAGIC: &[u8; 16] = b"SQLite format 3\0";

/// Default page size in bytes.
pub const DEFAULT_PAGE_SIZE: u32 = 4096;

/// Size of the database file header in bytes.
pub const HEADER_SIZE: usize = 100;

/// Maximum supported page size.
pub const MAX_PAGE_SIZE: u32 = 65536;

/// Minimum page size.
pub const MIN_PAGE_SIZE: u32 = 512;

/// The 100-byte database file header.
#[derive(Debug, Clone)]
pub struct DatabaseHeader {
    /// Page size in bytes (must be power of 2 between 512 and 65536).
    pub page_size: u32,
    /// File format write version (1 = legacy, 2 = WAL).
    pub write_version: u8,
    /// File format read version (1 = legacy, 2 = WAL).
    pub read_version: u8,
    /// Bytes of unused space at the end of each page (usually 0).
    pub reserved_space: u8,
    /// Total number of pages in the database file.
    pub page_count: u32,
    /// Page number of first freelist trunk page (0 if none).
    pub first_freelist_page: u32,
    /// Total number of freelist pages.
    pub freelist_count: u32,
    /// Schema cookie (incremented on schema changes).
    pub schema_cookie: u32,
    /// Schema format number (should be 4 for current SQLite).
    pub schema_format: u32,
    /// Default page cache size.
    pub default_cache_size: u32,
    /// Page number of the largest root b-tree page for auto-vacuum/incremental-vacuum modes.
    pub largest_root_page: u32,
    /// Database text encoding: 1=UTF-8, 2=UTF-16le, 3=UTF-16be.
    pub text_encoding: u32,
    /// User version (set by PRAGMA user_version).
    pub user_version: u32,
    /// Non-zero for incremental vacuum mode.
    pub incremental_vacuum: u32,
    /// Application ID (set by PRAGMA application_id).
    pub application_id: u32,
    /// Version-valid-for number.
    pub version_valid_for: u32,
    /// SQLite version number that wrote the database.
    pub sqlite_version: u32,
}

impl DatabaseHeader {
    /// Create a new default header for a fresh database.
    pub fn new() -> Self {
        Self {
            page_size: DEFAULT_PAGE_SIZE,
            write_version: 1,
            read_version: 1,
            reserved_space: 0,
            page_count: 1,
            first_freelist_page: 0,
            freelist_count: 0,
            schema_cookie: 0,
            schema_format: 4,
            default_cache_size: 0,
            largest_root_page: 0,
            text_encoding: 1, // UTF-8
            user_version: 0,
            incremental_vacuum: 0,
            application_id: 0,
            version_valid_for: 0,
            sqlite_version: 0,
        }
    }

    /// Parse the header from a 100-byte buffer.
    pub fn parse(buf: &[u8; HEADER_SIZE]) -> Result<Self> {
        // Validate magic string.
        if &buf[0..16] != MAGIC {
            return Err(RsqliteError::Corrupt(
                "not a SQLite database (invalid magic string)".into(),
            ));
        }

        let raw_page_size = read_be_u16(buf, 16) as u32;
        // Page size of 1 means 65536.
        let page_size = if raw_page_size == 1 {
            65536
        } else {
            raw_page_size
        };

        if page_size < MIN_PAGE_SIZE || page_size > MAX_PAGE_SIZE || !page_size.is_power_of_two() {
            return Err(RsqliteError::Corrupt(format!(
                "invalid page size: {page_size}"
            )));
        }

        let text_encoding = read_be_u32(buf, 56);
        if text_encoding == 0 || text_encoding > 3 {
            return Err(RsqliteError::Corrupt(format!(
                "invalid text encoding: {text_encoding}"
            )));
        }

        Ok(Self {
            page_size,
            write_version: buf[18],
            read_version: buf[19],
            reserved_space: buf[20],
            page_count: read_be_u32(buf, 28),
            first_freelist_page: read_be_u32(buf, 32),
            freelist_count: read_be_u32(buf, 36),
            schema_cookie: read_be_u32(buf, 40),
            schema_format: read_be_u32(buf, 44),
            default_cache_size: read_be_u32(buf, 48),
            largest_root_page: read_be_u32(buf, 52),
            text_encoding,
            user_version: read_be_u32(buf, 60),
            incremental_vacuum: read_be_u32(buf, 64),
            application_id: read_be_u32(buf, 68),
            // Bytes 72-91 are reserved (zero).
            version_valid_for: read_be_u32(buf, 92),
            sqlite_version: read_be_u32(buf, 96),
        })
    }

    /// Serialize the header into a 100-byte buffer.
    pub fn write(&self, buf: &mut [u8; HEADER_SIZE]) {
        buf[0..16].copy_from_slice(MAGIC);

        let raw_page_size = if self.page_size == 65536 {
            1u16
        } else {
            self.page_size as u16
        };
        write_be_u16(buf, 16, raw_page_size);

        buf[18] = self.write_version;
        buf[19] = self.read_version;
        buf[20] = self.reserved_space;

        // Bytes 21-23: max embedded payload fraction (64), min (32), leaf (32).
        buf[21] = 64;
        buf[22] = 32;
        buf[23] = 32;

        // Bytes 24-27: file change counter (use version_valid_for).
        write_be_u32(buf, 24, self.version_valid_for);

        write_be_u32(buf, 28, self.page_count);
        write_be_u32(buf, 32, self.first_freelist_page);
        write_be_u32(buf, 36, self.freelist_count);
        write_be_u32(buf, 40, self.schema_cookie);
        write_be_u32(buf, 44, self.schema_format);
        write_be_u32(buf, 48, self.default_cache_size);
        write_be_u32(buf, 52, self.largest_root_page);
        write_be_u32(buf, 56, self.text_encoding);
        write_be_u32(buf, 60, self.user_version);
        write_be_u32(buf, 64, self.incremental_vacuum);
        write_be_u32(buf, 68, self.application_id);

        // Reserved bytes 72-91: zero.
        buf[72..92].fill(0);

        write_be_u32(buf, 92, self.version_valid_for);
        write_be_u32(buf, 96, self.sqlite_version);
    }

    /// Usable page size (page size minus reserved space).
    pub fn usable_size(&self) -> u32 {
        self.page_size - self.reserved_space as u32
    }
}

impl Default for DatabaseHeader {
    fn default() -> Self {
        Self::new()
    }
}

// Big-endian read/write helpers.

pub fn read_be_u16(buf: &[u8], offset: usize) -> u16 {
    u16::from_be_bytes([buf[offset], buf[offset + 1]])
}

pub fn read_be_u32(buf: &[u8], offset: usize) -> u32 {
    u32::from_be_bytes([buf[offset], buf[offset + 1], buf[offset + 2], buf[offset + 3]])
}

pub fn read_be_i32(buf: &[u8], offset: usize) -> i32 {
    i32::from_be_bytes([buf[offset], buf[offset + 1], buf[offset + 2], buf[offset + 3]])
}

pub fn read_be_u64(buf: &[u8], offset: usize) -> u64 {
    u64::from_be_bytes([
        buf[offset],
        buf[offset + 1],
        buf[offset + 2],
        buf[offset + 3],
        buf[offset + 4],
        buf[offset + 5],
        buf[offset + 6],
        buf[offset + 7],
    ])
}

pub fn read_be_i64(buf: &[u8], offset: usize) -> i64 {
    i64::from_be_bytes([
        buf[offset],
        buf[offset + 1],
        buf[offset + 2],
        buf[offset + 3],
        buf[offset + 4],
        buf[offset + 5],
        buf[offset + 6],
        buf[offset + 7],
    ])
}

pub fn read_be_f64(buf: &[u8], offset: usize) -> f64 {
    f64::from_be_bytes([
        buf[offset],
        buf[offset + 1],
        buf[offset + 2],
        buf[offset + 3],
        buf[offset + 4],
        buf[offset + 5],
        buf[offset + 6],
        buf[offset + 7],
    ])
}

pub fn write_be_u16(buf: &mut [u8], offset: usize, value: u16) {
    buf[offset..offset + 2].copy_from_slice(&value.to_be_bytes());
}

pub fn write_be_u32(buf: &mut [u8], offset: usize, value: u32) {
    buf[offset..offset + 4].copy_from_slice(&value.to_be_bytes());
}

pub fn write_be_u64(buf: &mut [u8], offset: usize, value: u64) {
    buf[offset..offset + 8].copy_from_slice(&value.to_be_bytes());
}

pub fn write_be_i64(buf: &mut [u8], offset: usize, value: i64) {
    buf[offset..offset + 8].copy_from_slice(&value.to_be_bytes());
}

pub fn write_be_f64(buf: &mut [u8], offset: usize, value: f64) {
    buf[offset..offset + 8].copy_from_slice(&value.to_be_bytes());
}

/// B-tree page types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BTreePageType {
    /// Interior page of an index B-tree (type flag 0x02).
    IndexInterior,
    /// Interior page of a table B-tree (type flag 0x05).
    TableInterior,
    /// Leaf page of an index B-tree (type flag 0x0A).
    IndexLeaf,
    /// Leaf page of a table B-tree (type flag 0x0D).
    TableLeaf,
}

impl BTreePageType {
    pub fn from_flag(flag: u8) -> Result<Self> {
        match flag {
            0x02 => Ok(Self::IndexInterior),
            0x05 => Ok(Self::TableInterior),
            0x0A => Ok(Self::IndexLeaf),
            0x0D => Ok(Self::TableLeaf),
            _ => Err(RsqliteError::Corrupt(format!(
                "invalid b-tree page type flag: {flag:#04x}"
            ))),
        }
    }

    pub fn to_flag(self) -> u8 {
        match self {
            Self::IndexInterior => 0x02,
            Self::TableInterior => 0x05,
            Self::IndexLeaf => 0x0A,
            Self::TableLeaf => 0x0D,
        }
    }

    pub fn is_leaf(self) -> bool {
        matches!(self, Self::IndexLeaf | Self::TableLeaf)
    }

    pub fn is_interior(self) -> bool {
        !self.is_leaf()
    }

    pub fn is_table(self) -> bool {
        matches!(self, Self::TableInterior | Self::TableLeaf)
    }

    pub fn is_index(self) -> bool {
        matches!(self, Self::IndexInterior | Self::IndexLeaf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_round_trip() {
        let header = DatabaseHeader::new();
        let mut buf = [0u8; HEADER_SIZE];
        header.write(&mut buf);

        let parsed = DatabaseHeader::parse(&buf).unwrap();
        assert_eq!(parsed.page_size, DEFAULT_PAGE_SIZE);
        assert_eq!(parsed.text_encoding, 1);
        assert_eq!(parsed.schema_format, 4);
        assert_eq!(parsed.write_version, 1);
        assert_eq!(parsed.read_version, 1);
        assert_eq!(parsed.reserved_space, 0);
    }

    #[test]
    fn test_header_65536_page_size() {
        let mut header = DatabaseHeader::new();
        header.page_size = 65536;
        let mut buf = [0u8; HEADER_SIZE];
        header.write(&mut buf);
        // The raw page size field should be 1 for 65536.
        assert_eq!(read_be_u16(&buf, 16), 1);

        let parsed = DatabaseHeader::parse(&buf).unwrap();
        assert_eq!(parsed.page_size, 65536);
    }

    #[test]
    fn test_header_invalid_magic() {
        let buf = [0u8; HEADER_SIZE];
        let err = DatabaseHeader::parse(&buf).unwrap_err();
        assert!(matches!(err, RsqliteError::Corrupt(_)));
    }

    #[test]
    fn test_usable_size() {
        let mut header = DatabaseHeader::new();
        assert_eq!(header.usable_size(), 4096);
        header.reserved_space = 8;
        assert_eq!(header.usable_size(), 4088);
    }

    #[test]
    fn test_btree_page_type() {
        assert_eq!(
            BTreePageType::from_flag(0x0D).unwrap(),
            BTreePageType::TableLeaf
        );
        assert_eq!(
            BTreePageType::from_flag(0x05).unwrap(),
            BTreePageType::TableInterior
        );
        assert!(BTreePageType::from_flag(0xFF).is_err());

        assert!(BTreePageType::TableLeaf.is_leaf());
        assert!(BTreePageType::TableLeaf.is_table());
        assert!(!BTreePageType::TableLeaf.is_interior());
        assert!(!BTreePageType::TableLeaf.is_index());
    }

    #[test]
    fn test_be_helpers() {
        let mut buf = [0u8; 8];
        write_be_u16(&mut buf, 0, 0x1234);
        assert_eq!(read_be_u16(&buf, 0), 0x1234);

        write_be_u32(&mut buf, 0, 0x12345678);
        assert_eq!(read_be_u32(&buf, 0), 0x12345678);

        write_be_u64(&mut buf, 0, 0x123456789ABCDEF0);
        assert_eq!(read_be_u64(&buf, 0), 0x123456789ABCDEF0);
    }
}
