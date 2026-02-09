//! SQLite3 file format constants and helpers.
//! Reference: https://www.sqlite.org/fileformat2.html

/// The SQLite magic string at the beginning of every database file.
pub const MAGIC_STRING: &[u8; 16] = b"SQLite format 3\0";

/// Default page size.
pub const DEFAULT_PAGE_SIZE: u32 = 4096;

/// Database file header (first 100 bytes of page 1).
#[derive(Debug, Clone)]
pub struct DatabaseHeader {
    pub page_size: u32,
    pub write_format: u8,
    pub read_format: u8,
    pub reserved_space: u8,
    pub file_change_counter: u32,
    pub database_size_pages: u32,
    pub first_freelist_trunk_page: u32,
    pub total_freelist_pages: u32,
    pub schema_cookie: u32,
    pub schema_format: u32,
    pub default_cache_size: u32,
    pub largest_root_btree: u32,
    pub text_encoding: u32,
    pub user_version: u32,
    pub incremental_vacuum: u32,
    pub application_id: u32,
    pub version_valid_for: u32,
    pub sqlite_version: u32,
}

impl DatabaseHeader {
    pub fn new(page_size: u32) -> Self {
        DatabaseHeader {
            page_size,
            write_format: 1,
            read_format: 1,
            reserved_space: 0,
            file_change_counter: 0,
            database_size_pages: 1,
            first_freelist_trunk_page: 0,
            total_freelist_pages: 0,
            schema_cookie: 0,
            schema_format: 4,
            default_cache_size: 0,
            largest_root_btree: 0,
            text_encoding: 1, // UTF-8
            user_version: 0,
            incremental_vacuum: 0,
            application_id: 0,
            version_valid_for: 0,
            sqlite_version: 0,
        }
    }

    /// Parse a database header from the first 100 bytes.
    pub fn parse(data: &[u8]) -> crate::error::Result<Self> {
        if data.len() < 100 {
            return Err(crate::error::RsqliteError::NotADatabase);
        }
        if &data[0..16] != MAGIC_STRING {
            return Err(crate::error::RsqliteError::NotADatabase);
        }

        let raw_page_size = read_be_u16(&data[16..18]) as u32;
        let page_size = if raw_page_size == 1 {
            65536
        } else {
            raw_page_size
        };

        Ok(DatabaseHeader {
            page_size,
            write_format: data[18],
            read_format: data[19],
            reserved_space: data[20],
            file_change_counter: read_be_u32(&data[24..28]),
            database_size_pages: read_be_u32(&data[28..32]),
            first_freelist_trunk_page: read_be_u32(&data[32..36]),
            total_freelist_pages: read_be_u32(&data[36..40]),
            schema_cookie: read_be_u32(&data[40..44]),
            schema_format: read_be_u32(&data[44..48]),
            default_cache_size: read_be_u32(&data[48..52]),
            largest_root_btree: read_be_u32(&data[52..56]),
            text_encoding: read_be_u32(&data[56..60]),
            user_version: read_be_u32(&data[60..64]),
            incremental_vacuum: read_be_u32(&data[64..68]),
            application_id: read_be_u32(&data[68..72]),
            version_valid_for: read_be_u32(&data[92..96]),
            sqlite_version: read_be_u32(&data[96..100]),
        })
    }

    /// Serialize the header to 100 bytes.
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = vec![0u8; 100];
        buf[0..16].copy_from_slice(MAGIC_STRING);

        let ps = if self.page_size == 65536 {
            1u16
        } else {
            self.page_size as u16
        };
        write_be_u16(&mut buf[16..18], ps);
        buf[18] = self.write_format;
        buf[19] = self.read_format;
        buf[20] = self.reserved_space;
        // bytes 21-23: max/min embedded payload fraction, leaf payload fraction
        buf[21] = 64;
        buf[22] = 32;
        buf[23] = 32;
        write_be_u32(&mut buf[24..28], self.file_change_counter);
        write_be_u32(&mut buf[28..32], self.database_size_pages);
        write_be_u32(&mut buf[32..36], self.first_freelist_trunk_page);
        write_be_u32(&mut buf[36..40], self.total_freelist_pages);
        write_be_u32(&mut buf[40..44], self.schema_cookie);
        write_be_u32(&mut buf[44..48], self.schema_format);
        write_be_u32(&mut buf[48..52], self.default_cache_size);
        write_be_u32(&mut buf[52..56], self.largest_root_btree);
        write_be_u32(&mut buf[56..60], self.text_encoding);
        write_be_u32(&mut buf[60..64], self.user_version);
        write_be_u32(&mut buf[64..68], self.incremental_vacuum);
        write_be_u32(&mut buf[68..72], self.application_id);
        // bytes 72-91 reserved (zeros)
        write_be_u32(&mut buf[92..96], self.version_valid_for);
        write_be_u32(&mut buf[96..100], self.sqlite_version);
        buf
    }
}

/// B-tree page types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BTreePageType {
    InteriorIndex, // 0x02
    InteriorTable, // 0x05
    LeafIndex,     // 0x0a
    LeafTable,     // 0x0d
}

impl BTreePageType {
    pub fn from_byte(b: u8) -> crate::error::Result<Self> {
        match b {
            0x02 => Ok(BTreePageType::InteriorIndex),
            0x05 => Ok(BTreePageType::InteriorTable),
            0x0a => Ok(BTreePageType::LeafIndex),
            0x0d => Ok(BTreePageType::LeafTable),
            _ => Err(crate::error::RsqliteError::Corruption(format!(
                "Invalid B-tree page type: 0x{:02x}",
                b
            ))),
        }
    }

    pub fn to_byte(self) -> u8 {
        match self {
            BTreePageType::InteriorIndex => 0x02,
            BTreePageType::InteriorTable => 0x05,
            BTreePageType::LeafIndex => 0x0a,
            BTreePageType::LeafTable => 0x0d,
        }
    }

    pub fn is_leaf(self) -> bool {
        matches!(self, BTreePageType::LeafIndex | BTreePageType::LeafTable)
    }

    pub fn is_interior(self) -> bool {
        !self.is_leaf()
    }

    pub fn is_table(self) -> bool {
        matches!(
            self,
            BTreePageType::InteriorTable | BTreePageType::LeafTable
        )
    }

    pub fn is_index(self) -> bool {
        matches!(
            self,
            BTreePageType::InteriorIndex | BTreePageType::LeafIndex
        )
    }
}

/// B-tree page header.
#[derive(Debug, Clone)]
pub struct BTreePageHeader {
    pub page_type: BTreePageType,
    pub first_freeblock: u16,
    pub cell_count: u16,
    pub cell_content_offset: u16,
    pub fragmented_free_bytes: u8,
    pub right_most_pointer: Option<u32>, // only for interior pages
}

impl BTreePageHeader {
    /// Parse a B-tree page header. `data` starts at the beginning of the header.
    pub fn parse(data: &[u8]) -> crate::error::Result<Self> {
        let page_type = BTreePageType::from_byte(data[0])?;
        let first_freeblock = read_be_u16(&data[1..3]);
        let cell_count = read_be_u16(&data[3..5]);
        let cell_content_offset = read_be_u16(&data[5..7]);
        let fragmented_free_bytes = data[7];

        let right_most_pointer = if page_type.is_interior() {
            Some(read_be_u32(&data[8..12]))
        } else {
            None
        };

        Ok(BTreePageHeader {
            page_type,
            first_freeblock,
            cell_count,
            cell_content_offset,
            fragmented_free_bytes,
            right_most_pointer,
        })
    }

    /// Size of this header in bytes.
    pub fn header_size(&self) -> usize {
        if self.page_type.is_interior() {
            12
        } else {
            8
        }
    }

    /// Serialize header to bytes.
    pub fn serialize(&self, buf: &mut [u8]) {
        buf[0] = self.page_type.to_byte();
        write_be_u16(&mut buf[1..3], self.first_freeblock);
        write_be_u16(&mut buf[3..5], self.cell_count);
        write_be_u16(&mut buf[5..7], self.cell_content_offset);
        buf[7] = self.fragmented_free_bytes;
        if let Some(rmp) = self.right_most_pointer {
            write_be_u32(&mut buf[8..12], rmp);
        }
    }
}

/// Read a big-endian u16.
pub fn read_be_u16(buf: &[u8]) -> u16 {
    u16::from_be_bytes([buf[0], buf[1]])
}

/// Read a big-endian u32.
pub fn read_be_u32(buf: &[u8]) -> u32 {
    u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]])
}

/// Read a big-endian u48 (6 bytes) as u64.
pub fn read_be_u48(buf: &[u8]) -> u64 {
    ((buf[0] as u64) << 40)
        | ((buf[1] as u64) << 32)
        | ((buf[2] as u64) << 24)
        | ((buf[3] as u64) << 16)
        | ((buf[4] as u64) << 8)
        | (buf[5] as u64)
}

/// Read a big-endian i64.
pub fn read_be_i64(buf: &[u8]) -> i64 {
    i64::from_be_bytes([
        buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
    ])
}

/// Read a big-endian f64.
pub fn read_be_f64(buf: &[u8]) -> f64 {
    f64::from_be_bytes([
        buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
    ])
}

/// Write a big-endian u16.
pub fn write_be_u16(buf: &mut [u8], val: u16) {
    let bytes = val.to_be_bytes();
    buf[0] = bytes[0];
    buf[1] = bytes[1];
}

/// Write a big-endian u32.
pub fn write_be_u32(buf: &mut [u8], val: u32) {
    let bytes = val.to_be_bytes();
    buf[..4].copy_from_slice(&bytes);
}

/// Write a big-endian i16.
pub fn write_be_i16(buf: &mut [u8], val: i16) {
    let bytes = val.to_be_bytes();
    buf[0] = bytes[0];
    buf[1] = bytes[1];
}

/// Write a big-endian i32.
pub fn write_be_i32(buf: &mut [u8], val: i32) {
    let bytes = val.to_be_bytes();
    buf[..4].copy_from_slice(&bytes);
}

/// Write a big-endian i64.
pub fn write_be_i64(buf: &mut [u8], val: i64) {
    let bytes = val.to_be_bytes();
    buf[..8].copy_from_slice(&bytes);
}

/// Write a big-endian f64.
pub fn write_be_f64(buf: &mut [u8], val: f64) {
    let bytes = val.to_be_bytes();
    buf[..8].copy_from_slice(&bytes);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_roundtrip() {
        let header = DatabaseHeader::new(4096);
        let data = header.serialize();
        assert_eq!(data.len(), 100);
        assert_eq!(&data[0..16], MAGIC_STRING);

        let parsed = DatabaseHeader::parse(&data).unwrap();
        assert_eq!(parsed.page_size, 4096);
        assert_eq!(parsed.text_encoding, 1);
        assert_eq!(parsed.schema_format, 4);
    }

    #[test]
    fn test_page_size_65536() {
        let mut header = DatabaseHeader::new(65536);
        header.page_size = 65536;
        let data = header.serialize();
        let parsed = DatabaseHeader::parse(&data).unwrap();
        assert_eq!(parsed.page_size, 65536);
    }

    #[test]
    fn test_btree_page_types() {
        assert_eq!(
            BTreePageType::from_byte(0x02).unwrap(),
            BTreePageType::InteriorIndex
        );
        assert_eq!(
            BTreePageType::from_byte(0x05).unwrap(),
            BTreePageType::InteriorTable
        );
        assert_eq!(
            BTreePageType::from_byte(0x0a).unwrap(),
            BTreePageType::LeafIndex
        );
        assert_eq!(
            BTreePageType::from_byte(0x0d).unwrap(),
            BTreePageType::LeafTable
        );
        assert!(BTreePageType::from_byte(0xff).is_err());
    }

    #[test]
    fn test_be_roundtrip() {
        let mut buf = [0u8; 8];
        write_be_u32(&mut buf, 0xDEADBEEF);
        assert_eq!(read_be_u32(&buf), 0xDEADBEEF);

        write_be_u16(&mut buf, 0x1234);
        assert_eq!(read_be_u16(&buf), 0x1234);
    }
}
