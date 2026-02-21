// B-tree / B+tree engine for tables and indexes.
//
// Implements the SQLite B-tree on-disk format for reading:
// - B-tree page header parsing (leaf and interior pages)
// - Cell parsing for all four page types
// - Cell pointer array reading
// - Overflow page chain following
// - BTreeCursor for sequential scan and rowid-based seek
//
// See: https://www.sqlite.org/fileformat2.html

use crate::error::{Result, RsqliteError};
use crate::format::{self, BTreePageType};
use crate::pager::{self, PageNumber, Pager};
use crate::varint;

// ---------------------------------------------------------------------------
// B-tree page header
// ---------------------------------------------------------------------------

/// Parsed B-tree page header. Leaf pages have an 8-byte header; interior pages
/// have a 12-byte header (the extra 4 bytes store the right-most child pointer).
#[derive(Debug, Clone)]
pub struct BTreePageHeader {
    /// The type of this B-tree page.
    pub page_type: BTreePageType,
    /// Byte offset of the first freeblock on this page (0 if none).
    pub first_freeblock: u16,
    /// Number of cells on this page.
    pub cell_count: u16,
    /// Byte offset of the first byte of the cell content area.
    /// A value of 0 is interpreted as 65536.
    pub cell_content_offset: u16,
    /// Number of fragmented free bytes in the cell content area.
    pub fragmented_free_bytes: u8,
    /// Right-most child page pointer (interior pages only; 0 for leaf pages).
    pub right_child: PageNumber,
}

impl BTreePageHeader {
    /// Size of the header in bytes: 12 for interior pages, 8 for leaf pages.
    pub fn header_size(&self) -> usize {
        if self.page_type.is_interior() {
            12
        } else {
            8
        }
    }

    /// Parse a B-tree page header from raw page data.
    ///
    /// `data` is the full page buffer. `offset` is the byte offset where the
    /// B-tree header begins (100 for page 1, 0 for all other pages).
    pub fn parse(data: &[u8], offset: usize) -> Result<Self> {
        if data.len() < offset + 8 {
            return Err(RsqliteError::Corrupt(
                "page too small for B-tree header".into(),
            ));
        }

        let page_type = BTreePageType::from_flag(data[offset])?;

        let first_freeblock = format::read_be_u16(data, offset + 1);
        let cell_count = format::read_be_u16(data, offset + 3);
        let cell_content_offset = format::read_be_u16(data, offset + 5);
        let fragmented_free_bytes = data[offset + 7];

        let right_child = if page_type.is_interior() {
            if data.len() < offset + 12 {
                return Err(RsqliteError::Corrupt(
                    "page too small for interior B-tree header".into(),
                ));
            }
            format::read_be_u32(data, offset + 8)
        } else {
            0
        };

        Ok(Self {
            page_type,
            first_freeblock,
            cell_count,
            cell_content_offset,
            fragmented_free_bytes,
            right_child,
        })
    }

    /// The effective cell content offset, interpreting 0 as 65536.
    pub fn content_offset(&self) -> usize {
        if self.cell_content_offset == 0 {
            65536
        } else {
            self.cell_content_offset as usize
        }
    }
}

// ---------------------------------------------------------------------------
// Cell pointer array
// ---------------------------------------------------------------------------

/// Read the cell pointer array from a page. Returns a vector of byte offsets
/// (within the page) where each cell begins.
///
/// The cell pointer array starts immediately after the B-tree page header.
pub fn read_cell_pointers(data: &[u8], header_offset: usize, header: &BTreePageHeader) -> Result<Vec<u16>> {
    let array_start = header_offset + header.header_size();
    let count = header.cell_count as usize;

    // Each pointer is 2 bytes.
    let needed = array_start + count * 2;
    if data.len() < needed {
        return Err(RsqliteError::Corrupt(
            "page too small for cell pointer array".into(),
        ));
    }

    let mut pointers = Vec::with_capacity(count);
    for i in 0..count {
        let ptr = format::read_be_u16(data, array_start + i * 2);
        pointers.push(ptr);
    }

    Ok(pointers)
}

// ---------------------------------------------------------------------------
// Overflow page threshold computation
// ---------------------------------------------------------------------------

/// Compute the maximum payload that can be stored entirely on a single page
/// (no overflow) for a **table** B-tree leaf cell.
///
/// From the SQLite file format spec:
///   Let U = usable_size, P = payload_size
///   For table leaf: max in-page = U - 35
///   If P > max, then: min_local = (U - 12) * 32 / 255 - 23
///                      space = min_local + (P - min_local) % (U - 4)
///                      local = if space <= max { space } else { min_local }
pub fn table_leaf_max_local(usable_size: usize) -> usize {
    usable_size - 35
}

/// Compute the minimum amount of payload stored locally on a table leaf cell
/// when overflow is needed.
pub fn table_leaf_min_local(usable_size: usize) -> usize {
    (usable_size - 12) * 32 / 255 - 23
}

/// Compute the maximum payload that can be stored entirely on a single page
/// for an **index** B-tree cell (both leaf and interior).
///
/// For index pages: max_local = (U - 12) * 64 / 255 - 23
pub fn index_max_local(usable_size: usize) -> usize {
    (usable_size - 12) * 64 / 255 - 23
}

/// Compute the minimum amount of payload stored locally on an index cell
/// when overflow is needed.
pub fn index_min_local(usable_size: usize) -> usize {
    (usable_size - 12) * 32 / 255 - 23
}

/// Compute how many bytes of a payload are stored locally (on the page) and
/// whether there is overflow.
///
/// Returns `(local_size, has_overflow)`.
fn compute_local_payload_size(
    payload_size: usize,
    usable_size: usize,
    page_type: BTreePageType,
) -> (usize, bool) {
    let (max_local, min_local) = match page_type {
        BTreePageType::TableLeaf => (
            table_leaf_max_local(usable_size),
            table_leaf_min_local(usable_size),
        ),
        BTreePageType::TableInterior => {
            // Table interior cells have no payload (just rowid + left child),
            // but we handle the call gracefully.
            (usable_size - 35, (usable_size - 12) * 32 / 255 - 23)
        }
        BTreePageType::IndexLeaf | BTreePageType::IndexInterior => (
            index_max_local(usable_size),
            index_min_local(usable_size),
        ),
    };

    if payload_size <= max_local {
        (payload_size, false)
    } else {
        // Some payload spills to overflow pages.
        let surplus = (payload_size - min_local) % (usable_size - 4);
        let local = min_local + surplus;
        if local <= max_local {
            (local, true)
        } else {
            (min_local, true)
        }
    }
}

// ---------------------------------------------------------------------------
// Parsed cells
// ---------------------------------------------------------------------------

/// A parsed cell from a B-tree page.
#[derive(Debug, Clone)]
pub enum BTreeCell {
    /// Cell from a table B-tree leaf page.
    TableLeaf {
        /// The rowid of this row.
        rowid: i64,
        /// The full payload (possibly assembled from overflow pages).
        payload: Vec<u8>,
    },
    /// Cell from a table B-tree interior page.
    TableInterior {
        /// The left child page number.
        left_child: PageNumber,
        /// The rowid key dividing the left child from the right child/sibling.
        rowid: i64,
    },
    /// Cell from an index B-tree leaf page.
    IndexLeaf {
        /// The full payload (the index record).
        payload: Vec<u8>,
    },
    /// Cell from an index B-tree interior page.
    IndexInterior {
        /// The left child page number.
        left_child: PageNumber,
        /// The full payload (the index record).
        payload: Vec<u8>,
    },
}

impl BTreeCell {
    /// If this cell has a rowid (table leaf or table interior), return it.
    pub fn rowid(&self) -> Option<i64> {
        match self {
            BTreeCell::TableLeaf { rowid, .. } => Some(*rowid),
            BTreeCell::TableInterior { rowid, .. } => Some(*rowid),
            _ => None,
        }
    }

    /// If this cell has payload data, return a reference to it.
    pub fn payload(&self) -> Option<&[u8]> {
        match self {
            BTreeCell::TableLeaf { payload, .. } => Some(payload),
            BTreeCell::IndexLeaf { payload, .. } => Some(payload),
            BTreeCell::IndexInterior { payload, .. } => Some(payload),
            BTreeCell::TableInterior { .. } => None,
        }
    }

    /// If this cell has a left child pointer (interior cells), return it.
    pub fn left_child(&self) -> Option<PageNumber> {
        match self {
            BTreeCell::TableInterior { left_child, .. } => Some(*left_child),
            BTreeCell::IndexInterior { left_child, .. } => Some(*left_child),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Cell parsing (single page, no overflow resolution)
// ---------------------------------------------------------------------------

/// Parse a single cell from page data at the given byte offset.
///
/// This function reads only the on-page portion. If the payload spills to
/// overflow pages, the returned payload will contain only the local portion
/// and a 4-byte overflow page pointer at the end is NOT included in payload.
///
/// Returns `(cell, local_payload_size, first_overflow_page)`.
fn parse_cell_raw(
    data: &[u8],
    offset: usize,
    page_type: BTreePageType,
    usable_size: usize,
) -> Result<(BTreeCell, usize, PageNumber)> {
    let mut pos = offset;

    match page_type {
        BTreePageType::TableLeaf => {
            // payload-length (varint), rowid (varint), payload, [overflow-page (4 bytes)]
            let (payload_len, n) = read_varint_checked(data, pos)?;
            pos += n;
            let payload_size = payload_len as usize;

            let (rowid, n) = read_varint_i64_checked(data, pos)?;
            pos += n;

            let (local_size, has_overflow) =
                compute_local_payload_size(payload_size, usable_size, page_type);

            if pos + local_size > data.len() {
                return Err(RsqliteError::Corrupt(
                    "table leaf cell payload extends beyond page".into(),
                ));
            }

            let local_payload = data[pos..pos + local_size].to_vec();
            pos += local_size;

            let overflow_page = if has_overflow {
                if pos + 4 > data.len() {
                    return Err(RsqliteError::Corrupt(
                        "overflow page pointer extends beyond page".into(),
                    ));
                }
                format::read_be_u32(data, pos)
            } else {
                0
            };

            Ok((
                BTreeCell::TableLeaf {
                    rowid,
                    payload: local_payload,
                },
                payload_size,
                overflow_page,
            ))
        }

        BTreePageType::TableInterior => {
            // left-child (4 bytes BE), rowid (varint)
            if pos + 4 > data.len() {
                return Err(RsqliteError::Corrupt(
                    "table interior cell too short for left child pointer".into(),
                ));
            }
            let left_child = format::read_be_u32(data, pos);
            pos += 4;

            let (rowid, _n) = read_varint_i64_checked(data, pos)?;

            Ok((
                BTreeCell::TableInterior { left_child, rowid },
                0, // no payload
                0, // no overflow
            ))
        }

        BTreePageType::IndexLeaf => {
            // payload-length (varint), payload, [overflow-page (4 bytes)]
            let (payload_len, n) = read_varint_checked(data, pos)?;
            pos += n;
            let payload_size = payload_len as usize;

            let (local_size, has_overflow) =
                compute_local_payload_size(payload_size, usable_size, page_type);

            if pos + local_size > data.len() {
                return Err(RsqliteError::Corrupt(
                    "index leaf cell payload extends beyond page".into(),
                ));
            }

            let local_payload = data[pos..pos + local_size].to_vec();
            pos += local_size;

            let overflow_page = if has_overflow {
                if pos + 4 > data.len() {
                    return Err(RsqliteError::Corrupt(
                        "overflow page pointer extends beyond page".into(),
                    ));
                }
                format::read_be_u32(data, pos)
            } else {
                0
            };

            Ok((
                BTreeCell::IndexLeaf {
                    payload: local_payload,
                },
                payload_size,
                overflow_page,
            ))
        }

        BTreePageType::IndexInterior => {
            // left-child (4 bytes BE), payload-length (varint), payload, [overflow-page (4 bytes)]
            if pos + 4 > data.len() {
                return Err(RsqliteError::Corrupt(
                    "index interior cell too short for left child pointer".into(),
                ));
            }
            let left_child = format::read_be_u32(data, pos);
            pos += 4;

            let (payload_len, n) = read_varint_checked(data, pos)?;
            pos += n;
            let payload_size = payload_len as usize;

            let (local_size, has_overflow) =
                compute_local_payload_size(payload_size, usable_size, page_type);

            if pos + local_size > data.len() {
                return Err(RsqliteError::Corrupt(
                    "index interior cell payload extends beyond page".into(),
                ));
            }

            let local_payload = data[pos..pos + local_size].to_vec();
            pos += local_size;

            let overflow_page = if has_overflow {
                if pos + 4 > data.len() {
                    return Err(RsqliteError::Corrupt(
                        "overflow page pointer extends beyond page".into(),
                    ));
                }
                format::read_be_u32(data, pos)
            } else {
                0
            };

            Ok((
                BTreeCell::IndexInterior {
                    left_child,
                    payload: local_payload,
                },
                payload_size,
                overflow_page,
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// Overflow page chain reading
// ---------------------------------------------------------------------------

/// Read a full payload, following overflow pages if necessary.
///
/// `local_payload` is the payload bytes already read from the primary page.
/// `total_payload_size` is the declared total payload length.
/// `first_overflow_page` is the page number of the first overflow page (0 if none).
fn read_full_payload(
    pager: &mut Pager,
    local_payload: Vec<u8>,
    total_payload_size: usize,
    first_overflow_page: PageNumber,
) -> Result<Vec<u8>> {
    if local_payload.len() >= total_payload_size {
        // No overflow needed; truncate to exact size in case local_payload is bigger.
        let mut payload = local_payload;
        payload.truncate(total_payload_size);
        return Ok(payload);
    }

    let mut payload = local_payload;
    payload.reserve(total_payload_size - payload.len());

    let usable_size = pager.usable_size();
    let mut overflow_page_num = first_overflow_page;
    let remaining_to_read = total_payload_size - payload.len();
    let mut read_so_far = 0usize;

    while read_so_far < remaining_to_read {
        if overflow_page_num == 0 {
            return Err(RsqliteError::Corrupt(
                "overflow chain ended prematurely".into(),
            ));
        }

        let page = pager.get_page(overflow_page_num)?;
        let data = &page.data;

        // First 4 bytes of an overflow page: pointer to next overflow page (0 = last).
        if data.len() < 4 {
            return Err(RsqliteError::Corrupt(
                "overflow page too small".into(),
            ));
        }
        let next_overflow = format::read_be_u32(data, 0);

        // The rest of the usable space holds payload data.
        let available = usable_size - 4;
        let to_copy = available.min(remaining_to_read - read_so_far);

        if 4 + to_copy > data.len() {
            return Err(RsqliteError::Corrupt(
                "overflow page data extends beyond page".into(),
            ));
        }

        payload.extend_from_slice(&data[4..4 + to_copy]);
        read_so_far += to_copy;
        overflow_page_num = next_overflow;
    }

    debug_assert_eq!(payload.len(), total_payload_size);
    Ok(payload)
}

/// Parse a cell and resolve any overflow pages, producing a complete BTreeCell.
pub fn parse_cell(
    pager: &mut Pager,
    page_num: PageNumber,
    cell_offset: usize,
    page_type: BTreePageType,
) -> Result<BTreeCell> {
    let usable_size = pager.usable_size();
    let data = pager.get_page(page_num)?.data.clone();

    let (cell, total_payload_size, first_overflow) =
        parse_cell_raw(&data, cell_offset, page_type, usable_size)?;

    // If there is overflow, read the full payload.
    if first_overflow != 0 {
        match cell {
            BTreeCell::TableLeaf { rowid, payload } => {
                let full = read_full_payload(pager, payload, total_payload_size, first_overflow)?;
                Ok(BTreeCell::TableLeaf {
                    rowid,
                    payload: full,
                })
            }
            BTreeCell::IndexLeaf { payload } => {
                let full = read_full_payload(pager, payload, total_payload_size, first_overflow)?;
                Ok(BTreeCell::IndexLeaf { payload: full })
            }
            BTreeCell::IndexInterior {
                left_child,
                payload,
            } => {
                let full = read_full_payload(pager, payload, total_payload_size, first_overflow)?;
                Ok(BTreeCell::IndexInterior {
                    left_child,
                    payload: full,
                })
            }
            // Table interior cells have no payload, so no overflow possible.
            other => Ok(other),
        }
    } else {
        Ok(cell)
    }
}

// ---------------------------------------------------------------------------
// Convenience: read all cells from a page
// ---------------------------------------------------------------------------

/// Parse the B-tree header and all cells on a single page.
/// Overflow pages are followed to assemble complete payloads.
pub fn read_page_cells(pager: &mut Pager, page_num: PageNumber) -> Result<(BTreePageHeader, Vec<BTreeCell>)> {
    let header_offset = pager::btree_header_offset(page_num);
    let data = pager.get_page(page_num)?.data.clone();
    let header = BTreePageHeader::parse(&data, header_offset)?;
    let pointers = read_cell_pointers(&data, header_offset, &header)?;

    let mut cells = Vec::with_capacity(pointers.len());
    for &ptr in &pointers {
        let cell = parse_cell(pager, page_num, ptr as usize, header.page_type)?;
        cells.push(cell);
    }

    Ok((header, cells))
}

// ---------------------------------------------------------------------------
// BTreeCursor
// ---------------------------------------------------------------------------

/// Position within a B-tree page: which cell index we are at.
#[derive(Debug, Clone)]
struct CursorFrame {
    /// The page number.
    page_num: PageNumber,
    /// The B-tree page type.
    page_type: BTreePageType,
    /// The cell pointer array for this page.
    cell_pointers: Vec<u16>,
    /// Current cell index (0-based). May equal cell_pointers.len() when
    /// exhausted (for interior pages, that means we go to right_child).
    cell_index: usize,
    /// Right-most child pointer (interior pages only).
    right_child: PageNumber,
}

/// A cursor for traversing a table B-tree (B+tree where data is in leaf pages).
///
/// The cursor maintains a stack of frames representing the path from the root
/// to the current leaf. The top of the stack is always a leaf page when the
/// cursor is positioned on a valid cell.
pub struct BTreeCursor {
    /// The root page of this B-tree.
    root_page: PageNumber,
    /// Stack of frames from root to current position. The last entry is the
    /// current (leaf) page.
    stack: Vec<CursorFrame>,
    /// Whether the cursor is positioned on a valid cell.
    valid: bool,
}

impl BTreeCursor {
    /// Create a new cursor for the B-tree rooted at `root_page`.
    /// The cursor starts in an invalid (unpositioned) state.
    pub fn new(root_page: PageNumber) -> Self {
        Self {
            root_page,
            stack: Vec::new(),
            valid: false,
        }
    }

    /// Returns true if the cursor is positioned on a valid cell.
    pub fn is_valid(&self) -> bool {
        self.valid
    }

    /// Move the cursor to the first (leftmost) cell in the B-tree.
    /// After this call, the cursor points at the row with the smallest rowid
    /// (for table B-trees).
    pub fn move_to_first(&mut self, pager: &mut Pager) -> Result<()> {
        self.stack.clear();
        self.valid = false;
        self.descend_to_leftmost(pager, self.root_page)
    }

    /// Move the cursor to the next cell. For a table B-tree this advances
    /// to the next row in rowid order.
    ///
    /// Returns Ok(true) if the cursor now points at a valid cell,
    /// Ok(false) if there are no more cells (end of table).
    pub fn move_to_next(&mut self, pager: &mut Pager) -> Result<bool> {
        if !self.valid {
            return Ok(false);
        }

        // Current frame is the leaf.
        let leaf = self.stack.last_mut().unwrap();
        leaf.cell_index += 1;

        if leaf.cell_index < leaf.cell_pointers.len() {
            // Still more cells in this leaf page.
            return Ok(true);
        }

        // Exhausted the current leaf. Walk back up the stack to find an
        // interior page that has more children to descend into.
        loop {
            // Pop the exhausted page.
            self.stack.pop();

            if self.stack.is_empty() {
                // We have walked off the end of the entire tree.
                self.valid = false;
                return Ok(false);
            }

            let parent = self.stack.last_mut().unwrap();

            // In an interior page, after visiting the subtree pointed to by
            // cell[i].left_child, we visit cell[i] itself (its rowid is the
            // divider), then descend into the next child. But for a table
            // B+tree, interior cells only carry the rowid key and left_child;
            // there is no payload to "visit". The pattern is:
            //
            //   child_0, key_0, child_1, key_1, ..., child_{n-1}, key_{n-1}, right_child
            //
            // We already descended into child at `cell_index`. Advance the
            // index: if cell_index < cell_count, the next child to descend
            // into is at cell[cell_index].left_child (which is actually the
            // child to the RIGHT of the key we just passed). Wait -- let me
            // re-think.
            //
            // Interior page layout (for table B-tree):
            //   cells[0]: left_child=page_A, rowid=K0
            //   cells[1]: left_child=page_B, rowid=K1
            //   ...
            //   cells[n-1]: left_child=page_{n-1}, rowid=K_{n-1}
            //   right_child = page_R
            //
            // The traversal order is:
            //   descend page_A, then descend page_B, ..., descend page_{n-1}, descend page_R
            //
            // cell_index tracks which child we are ABOUT TO descend into.
            // When we first enter an interior page, cell_index=0 and we
            // descend into cells[0].left_child. After coming back up,
            // cell_index is still 0. We now increment it to 1:
            //   - if 1 < n, descend into cells[1].left_child
            //   - if 1 == n, descend into right_child
            //   - if 1 > n, pop (shouldn't happen normally)

            parent.cell_index += 1;

            if parent.cell_index < parent.cell_pointers.len() {
                // Descend into the left_child of cell[cell_index].
                let cell_offset = parent.cell_pointers[parent.cell_index] as usize;
                let page_num = parent.page_num;
                let page_type = parent.page_type;

                let child_page = {
                    let data = &pager.get_page(page_num)?.data;
                    read_left_child_pointer(data, cell_offset, page_type)?
                };

                return self.descend_to_leftmost(pager, child_page).map(|()| true);
            } else if parent.cell_index == parent.cell_pointers.len()
                && parent.right_child != 0
            {
                // Descend into right_child. We only do this when cell_index
                // is exactly len() (meaning we just finished the last cell's
                // subtree). Set cell_index beyond that to mark right_child
                // as consumed.
                let right_child = parent.right_child;
                parent.cell_index = parent.cell_pointers.len() + 1;
                return self.descend_to_leftmost(pager, right_child).map(|()| true);
            }

            // This interior page is fully exhausted. Pop it and try the parent.
        }
    }

    /// Seek to the cell with the given rowid in a table B-tree.
    ///
    /// If the exact rowid is found, the cursor is positioned on that cell and
    /// `Ok(true)` is returned.
    ///
    /// If the rowid is not found, the cursor is positioned on the cell with the
    /// smallest rowid greater than the target (or becomes invalid if the target
    /// is greater than all keys), and `Ok(false)` is returned.
    pub fn seek_rowid(&mut self, pager: &mut Pager, target: i64) -> Result<bool> {
        self.stack.clear();
        self.valid = false;

        self.seek_rowid_from(pager, self.root_page, target)
    }

    /// Internal: seek within the subtree rooted at `page_num`.
    fn seek_rowid_from(
        &mut self,
        pager: &mut Pager,
        page_num: PageNumber,
        target: i64,
    ) -> Result<bool> {
        let header_offset = pager::btree_header_offset(page_num);
        let data = pager.get_page(page_num)?.data.clone();
        let header = BTreePageHeader::parse(&data, header_offset)?;
        let pointers = read_cell_pointers(&data, header_offset, &header)?;
        let usable_size = pager.usable_size();

        match header.page_type {
            BTreePageType::TableInterior => {
                // Binary search for the child to descend into.
                // Interior cells are in ascending rowid order. We want the
                // first cell whose rowid >= target. If found at index i, we
                // descend into cells[i].left_child. If target > all keys, we
                // descend into right_child.
                let mut lo = 0usize;
                let mut hi = pointers.len();

                while lo < hi {
                    let mid = lo + (hi - lo) / 2;
                    let (cell, _, _) = parse_cell_raw(
                        &data,
                        pointers[mid] as usize,
                        header.page_type,
                        usable_size,
                    )?;
                    let key = cell.rowid().unwrap();
                    if key < target {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }

                // `lo` is the index of the first cell with rowid >= target.
                // We descend into the child to the left of that cell.
                // If lo == pointers.len(), descend into right_child.

                let frame = CursorFrame {
                    page_num,
                    page_type: header.page_type,
                    cell_pointers: pointers.clone(),
                    cell_index: lo,
                    right_child: header.right_child,
                };
                self.stack.push(frame);

                let child_page = if lo < pointers.len() {
                    // Descend into cells[lo].left_child.
                    let (cell, _, _) = parse_cell_raw(
                        &data,
                        pointers[lo] as usize,
                        header.page_type,
                        usable_size,
                    )?;
                    cell.left_child().unwrap()
                } else {
                    header.right_child
                };

                self.seek_rowid_from(pager, child_page, target)
            }

            BTreePageType::TableLeaf => {
                // Binary search within the leaf cells.
                let mut lo = 0usize;
                let mut hi = pointers.len();

                while lo < hi {
                    let mid = lo + (hi - lo) / 2;
                    let (cell, _, _) = parse_cell_raw(
                        &data,
                        pointers[mid] as usize,
                        header.page_type,
                        usable_size,
                    )?;
                    let key = cell.rowid().unwrap();
                    if key < target {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }

                let frame = CursorFrame {
                    page_num,
                    page_type: header.page_type,
                    cell_pointers: pointers.clone(),
                    cell_index: lo,
                    right_child: 0,
                };
                self.stack.push(frame);

                if lo < pointers.len() {
                    // Check if exact match.
                    let (cell, _, _) = parse_cell_raw(
                        &data,
                        pointers[lo] as usize,
                        header.page_type,
                        usable_size,
                    )?;
                    let key = cell.rowid().unwrap();
                    self.valid = true;
                    Ok(key == target)
                } else {
                    // Target is greater than all keys in this leaf.
                    // Try to advance to the next leaf via the parent stack.
                    self.valid = false;
                    // Try to move to the next cell across leaves.
                    // We need to walk up the stack and find the next subtree.
                    self.advance_past_exhausted_leaf(pager)?;
                    Ok(false)
                }
            }

            _ => Err(RsqliteError::Corrupt(
                "seek_rowid called on an index B-tree page".into(),
            )),
        }
    }

    /// When a seek lands past the end of a leaf page, walk up and find the
    /// next leaf with data, effectively positioning on the successor.
    fn advance_past_exhausted_leaf(&mut self, pager: &mut Pager) -> Result<()> {
        // Pop the exhausted leaf.
        loop {
            self.stack.pop();
            if self.stack.is_empty() {
                self.valid = false;
                return Ok(());
            }

            let parent = self.stack.last_mut().unwrap();
            parent.cell_index += 1;

            if parent.cell_index < parent.cell_pointers.len() {
                let cell_offset = parent.cell_pointers[parent.cell_index] as usize;
                let page_num = parent.page_num;
                let page_type = parent.page_type;

                let child_page = {
                    let data = &pager.get_page(page_num)?.data;
                    read_left_child_pointer(data, cell_offset, page_type)?
                };

                return self.descend_to_leftmost(pager, child_page);
            } else if parent.right_child != 0
                && parent.cell_index == parent.cell_pointers.len()
            {
                let right_child = parent.right_child;
                parent.cell_index = parent.cell_pointers.len() + 1;
                return self.descend_to_leftmost(pager, right_child);
            }
            // This parent is also exhausted; keep popping.
        }
    }

    /// Read the current cell's rowid. The cursor must be valid and on a table
    /// B-tree leaf page.
    pub fn current_rowid(&self, pager: &mut Pager) -> Result<i64> {
        if !self.valid {
            return Err(RsqliteError::Runtime("cursor is not valid".into()));
        }

        let frame = self.stack.last().unwrap();
        let cell_offset = frame.cell_pointers[frame.cell_index] as usize;
        let usable_size = pager.usable_size();
        let data = pager.get_page(frame.page_num)?.data.clone();

        let (cell, _, _) = parse_cell_raw(&data, cell_offset, frame.page_type, usable_size)?;

        cell.rowid().ok_or_else(|| {
            RsqliteError::Runtime("current cell has no rowid (not a table B-tree?)".into())
        })
    }

    /// Read the current cell's full payload. The cursor must be valid and on a
    /// leaf page with payload data. Overflow pages are followed automatically.
    pub fn current_payload(&self, pager: &mut Pager) -> Result<Vec<u8>> {
        if !self.valid {
            return Err(RsqliteError::Runtime("cursor is not valid".into()));
        }

        let frame = self.stack.last().unwrap();
        let cell = parse_cell(
            pager,
            frame.page_num,
            frame.cell_pointers[frame.cell_index] as usize,
            frame.page_type,
        )?;

        cell.payload()
            .map(|p| p.to_vec())
            .ok_or_else(|| RsqliteError::Runtime("current cell has no payload".into()))
    }

    /// Read the current cell. The cursor must be valid.
    pub fn current_cell(&self, pager: &mut Pager) -> Result<BTreeCell> {
        if !self.valid {
            return Err(RsqliteError::Runtime("cursor is not valid".into()));
        }

        let frame = self.stack.last().unwrap();
        parse_cell(
            pager,
            frame.page_num,
            frame.cell_pointers[frame.cell_index] as usize,
            frame.page_type,
        )
    }

    /// Descend from the given page to the leftmost leaf, pushing frames onto
    /// the stack along the way. Sets `self.valid` to true if a leaf cell exists.
    fn descend_to_leftmost(&mut self, pager: &mut Pager, mut page_num: PageNumber) -> Result<()> {
        loop {
            let header_offset = pager::btree_header_offset(page_num);
            let data = pager.get_page(page_num)?.data.clone();
            let header = BTreePageHeader::parse(&data, header_offset)?;
            let pointers = read_cell_pointers(&data, header_offset, &header)?;

            if header.page_type.is_leaf() {
                self.stack.push(CursorFrame {
                    page_num,
                    page_type: header.page_type,
                    cell_pointers: pointers,
                    cell_index: 0,
                    right_child: 0,
                });
                // Valid only if the leaf has at least one cell.
                self.valid = !self.stack.last().unwrap().cell_pointers.is_empty();
                return Ok(());
            }

            // Interior page: descend into the leftmost child.
            if pointers.is_empty() {
                // Edge case: interior page with no cells. This is unusual but
                // we can try the right_child if present.
                if header.right_child != 0 {
                    self.stack.push(CursorFrame {
                        page_num,
                        page_type: header.page_type,
                        cell_pointers: pointers,
                        cell_index: 0,
                        right_child: header.right_child,
                    });
                    page_num = header.right_child;
                    continue;
                }
                self.valid = false;
                return Ok(());
            }

            let usable_size = pager.usable_size();
            let (cell, _, _) = parse_cell_raw(
                &data,
                pointers[0] as usize,
                header.page_type,
                usable_size,
            )?;

            let child = cell.left_child().ok_or_else(|| {
                RsqliteError::Corrupt("interior cell missing left child pointer".into())
            })?;

            self.stack.push(CursorFrame {
                page_num,
                page_type: header.page_type,
                cell_pointers: pointers,
                cell_index: 0,
                right_child: header.right_child,
            });

            page_num = child;
        }
    }
}

// ---------------------------------------------------------------------------
// B-tree write operations
// ---------------------------------------------------------------------------

/// Initialize a page as an empty table leaf page.
pub fn init_table_leaf_page(pager: &mut Pager, page_num: PageNumber) -> Result<()> {
    let header_offset = pager::btree_header_offset(page_num);
    let page = pager.get_page_mut(page_num)?;
    let page_size = page.data.len();

    page.data[header_offset] = BTreePageType::TableLeaf.to_flag();
    format::write_be_u16(&mut page.data, header_offset + 1, 0); // first freeblock
    format::write_be_u16(&mut page.data, header_offset + 3, 0); // cell count
    // Cell content offset: start at end of page (no cells yet).
    format::write_be_u16(&mut page.data, header_offset + 5, page_size as u16);
    page.data[header_offset + 7] = 0; // fragmented free bytes

    Ok(())
}

/// Build the on-page representation of a table leaf cell.
/// Format: payload_length (varint), rowid (varint), payload.
pub fn build_table_leaf_cell(rowid: i64, payload: &[u8]) -> Vec<u8> {
    let mut cell = Vec::new();
    let mut tmp = [0u8; 9];

    let n = varint::write_varint(&mut tmp, payload.len() as u64);
    cell.extend_from_slice(&tmp[..n]);

    let n = varint::write_varint(&mut tmp, rowid as u64);
    cell.extend_from_slice(&tmp[..n]);

    cell.extend_from_slice(payload);
    cell
}

/// Insert a row into a table B-tree. The payload should be an encoded record.
///
/// Returns the (possibly new) root page number. The root changes if the
/// original root had to be split.
pub fn btree_insert(
    pager: &mut Pager,
    root_page: PageNumber,
    rowid: i64,
    payload: &[u8],
) -> Result<PageNumber> {
    let cell_data = build_table_leaf_cell(rowid, payload);

    // Try to insert into the tree, possibly splitting.
    let split = insert_into_subtree(pager, root_page, rowid, &cell_data)?;

    match split {
        InsertResult::Ok => Ok(root_page),
        InsertResult::Split {
            new_page,
            divider_rowid,
        } => {
            // The root was split. Create a new root.
            let new_root = pager.allocate_page()?;
            let header_offset = pager::btree_header_offset(new_root);
            {
                let page = pager.get_page_mut(new_root)?;
                let page_size = page.data.len();

                // Interior page header.
                page.data[header_offset] = BTreePageType::TableInterior.to_flag();
                format::write_be_u16(&mut page.data, header_offset + 1, 0);
                format::write_be_u16(&mut page.data, header_offset + 3, 1); // 1 cell
                page.data[header_offset + 7] = 0;
                // right_child = new_page
                format::write_be_u32(&mut page.data, header_offset + 8, new_page);

                // Build interior cell: left_child (4 bytes) + rowid (varint)
                let mut cell = Vec::new();
                cell.extend_from_slice(&root_page.to_be_bytes());
                let mut tmp = [0u8; 9];
                let n = varint::write_varint(&mut tmp, divider_rowid as u64);
                cell.extend_from_slice(&tmp[..n]);

                // Place cell at end of page.
                let cell_start = page_size - cell.len();
                page.data[cell_start..page_size].copy_from_slice(&cell);

                // Cell content offset.
                format::write_be_u16(&mut page.data, header_offset + 5, cell_start as u16);

                // Cell pointer.
                let ptr_offset = header_offset + 12; // after 12-byte interior header
                format::write_be_u16(&mut page.data, ptr_offset, cell_start as u16);
            }

            Ok(new_root)
        }
    }
}

/// Find the maximum rowid in a table B-tree. Returns 0 if the tree is empty.
pub fn find_max_rowid(pager: &mut Pager, root_page: PageNumber) -> Result<i64> {
    let mut page_num = root_page;

    loop {
        let header_offset = pager::btree_header_offset(page_num);
        let data = pager.get_page(page_num)?.data.clone();
        let header = BTreePageHeader::parse(&data, header_offset)?;

        if header.page_type.is_leaf() {
            if header.cell_count == 0 {
                return Ok(0);
            }
            // The last cell in a leaf has the highest rowid.
            let pointers = read_cell_pointers(&data, header_offset, &header)?;
            let last_ptr = pointers[pointers.len() - 1] as usize;
            let usable_size = pager.usable_size();
            let (cell, _, _) = parse_cell_raw(&data, last_ptr, header.page_type, usable_size)?;
            return Ok(cell.rowid().unwrap_or(0));
        }

        // Interior page: go to the rightmost child.
        page_num = header.right_child;
        if page_num == 0 {
            return Ok(0);
        }
    }
}

/// Result of inserting into a subtree.
enum InsertResult {
    /// Insertion succeeded without splitting.
    Ok,
    /// The page was split. `new_page` is the new right sibling,
    /// `divider_rowid` is the key to promote to the parent.
    Split {
        new_page: PageNumber,
        divider_rowid: i64,
    },
}

/// Insert a cell into the subtree rooted at `page_num`.
fn insert_into_subtree(
    pager: &mut Pager,
    page_num: PageNumber,
    rowid: i64,
    cell_data: &[u8],
) -> Result<InsertResult> {
    let header_offset = pager::btree_header_offset(page_num);
    let data = pager.get_page(page_num)?.data.clone();
    let header = BTreePageHeader::parse(&data, header_offset)?;

    match header.page_type {
        BTreePageType::TableLeaf => {
            // Try to insert the cell into this leaf page.
            let pointers = read_cell_pointers(&data, header_offset, &header)?;
            let usable_size = pager.usable_size();

            // Find insertion position (maintain rowid order).
            let mut insert_pos = pointers.len();
            for (i, &ptr) in pointers.iter().enumerate() {
                let (cell, _, _) = parse_cell_raw(&data, ptr as usize, header.page_type, usable_size)?;
                if cell.rowid().unwrap() >= rowid {
                    insert_pos = i;
                    break;
                }
            }

            // Check if there's space: need 2 bytes for pointer + cell_data.len() for content.
            let ptr_array_end = header_offset + header.header_size() + (header.cell_count as usize + 1) * 2;
            let content_start = header.content_offset();
            let free_space = if content_start > ptr_array_end {
                content_start - ptr_array_end
            } else {
                0
            };

            if free_space >= 2 + cell_data.len() {
                // Insert in-place.
                insert_cell_into_page(pager, page_num, cell_data, insert_pos)?;
                Ok(InsertResult::Ok)
            } else {
                // Need to split.
                split_leaf_page(pager, page_num, cell_data, rowid, insert_pos)
            }
        }

        BTreePageType::TableInterior => {
            // Find which child to descend into.
            let pointers = read_cell_pointers(&data, header_offset, &header)?;
            let usable_size = pager.usable_size();

            let mut child_page = header.right_child;
            let mut child_index = pointers.len(); // means right_child

            for (i, &ptr) in pointers.iter().enumerate() {
                let (cell, _, _) = parse_cell_raw(&data, ptr as usize, header.page_type, usable_size)?;
                let key = cell.rowid().unwrap();
                if rowid <= key {
                    child_page = cell.left_child().unwrap();
                    child_index = i;
                    break;
                }
            }

            // Recurse.
            let result = insert_into_subtree(pager, child_page, rowid, cell_data)?;

            match result {
                InsertResult::Ok => Ok(InsertResult::Ok),
                InsertResult::Split {
                    new_page,
                    divider_rowid,
                } => {
                    // Insert the divider into this interior page.
                    // Build interior cell: left_child (4 bytes) + rowid (varint)
                    let mut int_cell = Vec::new();
                    int_cell.extend_from_slice(&child_page.to_be_bytes());
                    let mut tmp = [0u8; 9];
                    let n = varint::write_varint(&mut tmp, divider_rowid as u64);
                    int_cell.extend_from_slice(&tmp[..n]);

                    // Re-read page header (may have changed).
                    let data = pager.get_page(page_num)?.data.clone();
                    let header = BTreePageHeader::parse(&data, pager::btree_header_offset(page_num))?;
                    let ptr_array_end = pager::btree_header_offset(page_num) + header.header_size() + (header.cell_count as usize + 1) * 2;
                    let content_start = header.content_offset();
                    let free_space = if content_start > ptr_array_end {
                        content_start - ptr_array_end
                    } else {
                        0
                    };

                    if free_space >= 2 + int_cell.len() {
                        // Insert the interior cell at child_index.
                        insert_cell_into_page(pager, page_num, &int_cell, child_index)?;

                        // Update: the child to the right of the new cell should be new_page.
                        // Actually, we set left_child = child_page in the cell.
                        // The right sibling is handled by the next cell's left_child or right_child.
                        // If child_index was the rightmost (pointing to right_child), update right_child.
                        let header_off = pager::btree_header_offset(page_num);
                        let page = pager.get_page_mut(page_num)?;
                        if child_index >= (header.cell_count as usize) {
                            // The new cell is at the end. new_page becomes right_child.
                            format::write_be_u32(&mut page.data, header_off + 8, new_page);
                        } else {
                            // The new cell was inserted before existing cells.
                            // The old cell at child_index now has new_page as its left_child.
                            // We need to update its left_child pointer.
                            let reread_header = BTreePageHeader::parse(&page.data, header_off)?;
                            let ptrs = read_cell_pointers(&page.data, header_off, &reread_header)?;
                            // The cell now at child_index+1 should point to new_page.
                            let next_cell_offset = ptrs[child_index + 1] as usize;
                            format::write_be_u32(&mut page.data, next_cell_offset, new_page);
                        }

                        Ok(InsertResult::Ok)
                    } else {
                        // Split the interior page (complex; for now, return error).
                        Err(RsqliteError::NotImplemented(
                            "interior page splitting".into(),
                        ))
                    }
                }
            }
        }

        _ => Err(RsqliteError::Corrupt(
            "unexpected page type during insert".into(),
        )),
    }
}

/// Insert a cell into a page at the given position (in the cell pointer array).
/// Assumes there is enough free space.
fn insert_cell_into_page(
    pager: &mut Pager,
    page_num: PageNumber,
    cell_data: &[u8],
    position: usize,
) -> Result<()> {
    let header_offset = pager::btree_header_offset(page_num);
    let page = pager.get_page_mut(page_num)?;

    let header = BTreePageHeader::parse(&page.data, header_offset)?;
    let cell_count = header.cell_count as usize;

    // Place cell content just before the current content area.
    let content_start = header.content_offset();
    let new_content_start = content_start - cell_data.len();
    page.data[new_content_start..content_start].copy_from_slice(cell_data);

    // Update cell content offset.
    format::write_be_u16(&mut page.data, header_offset + 5, new_content_start as u16);

    // Update cell count.
    format::write_be_u16(&mut page.data, header_offset + 3, (cell_count + 1) as u16);

    // Insert cell pointer at the correct position.
    let ptr_array_start = header_offset + header.header_size();
    // Shift existing pointers after `position` to the right by 2 bytes.
    if position < cell_count {
        let src_start = ptr_array_start + position * 2;
        let src_end = ptr_array_start + cell_count * 2;
        // Copy backwards to avoid overlap issues.
        for i in (src_start..src_end).rev() {
            page.data[i + 2] = page.data[i];
        }
    }

    // Write the new pointer.
    format::write_be_u16(
        &mut page.data,
        ptr_array_start + position * 2,
        new_content_start as u16,
    );

    Ok(())
}

/// Split a leaf page and return the new sibling info.
fn split_leaf_page(
    pager: &mut Pager,
    page_num: PageNumber,
    new_cell_data: &[u8],
    new_rowid: i64,
    insert_pos: usize,
) -> Result<InsertResult> {
    // Collect all existing cells + the new cell.
    let header_offset = pager::btree_header_offset(page_num);
    let usable_size = pager.usable_size();

    let data = pager.get_page(page_num)?.data.clone();
    let header = BTreePageHeader::parse(&data, header_offset)?;
    let pointers = read_cell_pointers(&data, header_offset, &header)?;

    // Collect all cells as (rowid, cell_bytes).
    let mut all_cells: Vec<(i64, Vec<u8>)> = Vec::with_capacity(pointers.len() + 1);

    for (i, &ptr) in pointers.iter().enumerate() {
        if i == insert_pos {
            all_cells.push((new_rowid, new_cell_data.to_vec()));
        }
        let (cell, _, _) = parse_cell_raw(&data, ptr as usize, header.page_type, usable_size)?;
        let cell_rowid = cell.rowid().unwrap();
        // Re-read the raw cell data from the page.
        let raw_cell = extract_raw_table_leaf_cell(&data, ptr as usize)?;
        all_cells.push((cell_rowid, raw_cell));
    }
    if insert_pos >= pointers.len() {
        all_cells.push((new_rowid, new_cell_data.to_vec()));
    }

    // Split at the midpoint.
    let mid = all_cells.len() / 2;
    let divider_rowid = all_cells[mid - 1].0;

    // Left page gets cells [0..mid), right page gets cells [mid..].
    let left_cells = &all_cells[..mid];
    let right_cells = &all_cells[mid..];

    // Rewrite the left (original) page.
    rewrite_leaf_page(pager, page_num, left_cells)?;

    // Create and fill the right (new) page.
    let new_page_num = pager.allocate_page()?;
    init_table_leaf_page(pager, new_page_num)?;
    rewrite_leaf_page(pager, new_page_num, right_cells)?;

    Ok(InsertResult::Split {
        new_page: new_page_num,
        divider_rowid,
    })
}

/// Extract raw cell bytes for a table leaf cell at the given offset.
fn extract_raw_table_leaf_cell(data: &[u8], offset: usize) -> Result<Vec<u8>> {
    let mut pos = offset;

    // payload_length varint
    let (payload_len, n) = read_varint_checked(data, pos)?;
    pos += n;
    let payload_size = payload_len as usize;

    // rowid varint
    let (_rowid, n) = read_varint_i64_checked(data, pos)?;
    pos += n;

    // payload bytes (local only, no overflow handling for now)
    let cell_end = pos + payload_size;
    if cell_end > data.len() {
        return Err(RsqliteError::Corrupt("cell extends beyond page".into()));
    }

    Ok(data[offset..cell_end].to_vec())
}

/// Rewrite a leaf page with the given cells.
fn rewrite_leaf_page(
    pager: &mut Pager,
    page_num: PageNumber,
    cells: &[(i64, Vec<u8>)],
) -> Result<()> {
    let header_offset = pager::btree_header_offset(page_num);
    let page = pager.get_page_mut(page_num)?;
    let page_size = page.data.len();

    // Clear the page (preserve the database header on page 1).
    let clear_start = header_offset;
    page.data[clear_start..page_size].fill(0);

    // Write B-tree leaf header.
    page.data[header_offset] = BTreePageType::TableLeaf.to_flag();
    format::write_be_u16(&mut page.data, header_offset + 1, 0); // first freeblock
    format::write_be_u16(&mut page.data, header_offset + 3, cells.len() as u16);
    page.data[header_offset + 7] = 0;

    // Place cells from end of page backward.
    let mut content_end = page_size;
    let ptr_array_start = header_offset + 8; // leaf header = 8 bytes

    for (i, (_rowid, cell_data)) in cells.iter().enumerate() {
        content_end -= cell_data.len();
        page.data[content_end..content_end + cell_data.len()].copy_from_slice(cell_data);
        format::write_be_u16(&mut page.data, ptr_array_start + i * 2, content_end as u16);
    }

    // Cell content offset.
    format::write_be_u16(&mut page.data, header_offset + 5, content_end as u16);

    Ok(())
}

/// Delete a row from a table B-tree by rowid.
/// Returns true if the row was found and deleted.
pub fn btree_delete(
    pager: &mut Pager,
    root_page: PageNumber,
    rowid: i64,
) -> Result<bool> {
    delete_from_subtree(pager, root_page, rowid)
}

/// Delete a cell with the given rowid from the subtree rooted at `page_num`.
fn delete_from_subtree(
    pager: &mut Pager,
    page_num: PageNumber,
    rowid: i64,
) -> Result<bool> {
    let header_offset = pager::btree_header_offset(page_num);
    let data = pager.get_page(page_num)?.data.clone();
    let header = BTreePageHeader::parse(&data, header_offset)?;
    let usable_size = pager.usable_size();

    match header.page_type {
        BTreePageType::TableLeaf => {
            let pointers = read_cell_pointers(&data, header_offset, &header)?;

            // Find the cell with matching rowid.
            for (i, &ptr) in pointers.iter().enumerate() {
                let (cell, _, _) = parse_cell_raw(&data, ptr as usize, header.page_type, usable_size)?;
                if cell.rowid() == Some(rowid) {
                    // Remove this cell by collecting all others and rewriting.
                    let mut remaining_cells: Vec<(i64, Vec<u8>)> = Vec::new();
                    for (j, &p) in pointers.iter().enumerate() {
                        if j == i {
                            continue;
                        }
                        let (c, _, _) = parse_cell_raw(&data, p as usize, header.page_type, usable_size)?;
                        let raw = extract_raw_table_leaf_cell(&data, p as usize)?;
                        remaining_cells.push((c.rowid().unwrap(), raw));
                    }
                    rewrite_leaf_page(pager, page_num, &remaining_cells)?;
                    return Ok(true);
                }
            }
            Ok(false)
        }

        BTreePageType::TableInterior => {
            let pointers = read_cell_pointers(&data, header_offset, &header)?;

            // Find which child to descend into.
            for &ptr in pointers.iter() {
                let (cell, _, _) = parse_cell_raw(&data, ptr as usize, header.page_type, usable_size)?;
                let key = cell.rowid().unwrap();
                if rowid <= key {
                    let child = cell.left_child().unwrap();
                    return delete_from_subtree(pager, child, rowid);
                }
            }
            // rowid > all keys, descend into right_child.
            if header.right_child != 0 {
                delete_from_subtree(pager, header.right_child, rowid)
            } else {
                Ok(false)
            }
        }

        _ => Err(RsqliteError::Corrupt("unexpected page type during delete".into())),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Read a varint from `data[offset..]`, returning an error if the buffer is
/// too short.
fn read_varint_checked(data: &[u8], offset: usize) -> Result<(u64, usize)> {
    if offset >= data.len() {
        return Err(RsqliteError::Corrupt(
            "varint extends beyond page boundary".into(),
        ));
    }
    varint::try_read_varint(&data[offset..]).ok_or_else(|| {
        RsqliteError::Corrupt("truncated varint in cell".into())
    })
}

/// Read a signed varint from `data[offset..]`.
fn read_varint_i64_checked(data: &[u8], offset: usize) -> Result<(i64, usize)> {
    let (v, n) = read_varint_checked(data, offset)?;
    Ok((v as i64, n))
}

/// Read the left-child page pointer from an interior cell at `cell_offset`.
fn read_left_child_pointer(
    data: &[u8],
    cell_offset: usize,
    page_type: BTreePageType,
) -> Result<PageNumber> {
    if !page_type.is_interior() {
        return Err(RsqliteError::Corrupt(
            "cannot read left child from a leaf cell".into(),
        ));
    }
    if cell_offset + 4 > data.len() {
        return Err(RsqliteError::Corrupt(
            "cell too short for left child pointer".into(),
        ));
    }
    Ok(format::read_be_u32(data, cell_offset))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE_SIZE: usize = 4096;
    const USABLE_SIZE: usize = 4096;

    // -----------------------------------------------------------------------
    // Helper: build a table leaf page
    // -----------------------------------------------------------------------

    /// Build a minimal table leaf page with the given cells.
    /// Each cell is (rowid, payload_bytes).
    fn build_table_leaf_page(page_num: PageNumber, cells: &[(i64, &[u8])]) -> Vec<u8> {
        let mut page = vec![0u8; PAGE_SIZE];
        let header_offset = if page_num == 1 { 100 } else { 0 };

        // Build cells from the end of the page backward.
        let mut content_end = PAGE_SIZE;
        let mut cell_offsets = Vec::new();

        for &(rowid, payload) in cells {
            // Cell format: payload_len (varint), rowid (varint), payload
            let mut cell = Vec::new();
            let mut tmp = [0u8; 9];
            let n = varint::write_varint(&mut tmp, payload.len() as u64);
            cell.extend_from_slice(&tmp[..n]);
            let n = varint::write_varint(&mut tmp, rowid as u64);
            cell.extend_from_slice(&tmp[..n]);
            cell.extend_from_slice(payload);

            content_end -= cell.len();
            page[content_end..content_end + cell.len()].copy_from_slice(&cell);
            cell_offsets.push(content_end as u16);
        }

        // Write B-tree page header.
        page[header_offset] = BTreePageType::TableLeaf.to_flag(); // 0x0D
        format::write_be_u16(&mut page, header_offset + 1, 0); // first freeblock
        format::write_be_u16(&mut page, header_offset + 3, cells.len() as u16); // cell count
        format::write_be_u16(&mut page, header_offset + 5, content_end as u16); // content offset
        page[header_offset + 7] = 0; // fragmented free bytes

        // Write cell pointer array (immediately after header).
        let array_start = header_offset + 8; // leaf header = 8 bytes
        for (i, &off) in cell_offsets.iter().enumerate() {
            format::write_be_u16(&mut page, array_start + i * 2, off);
        }

        page
    }

    /// Build a minimal table interior page.
    /// `children` is a list of (left_child_page, rowid) tuples.
    /// `right_child` is the rightmost child page number.
    fn build_table_interior_page(
        page_num: PageNumber,
        children: &[(PageNumber, i64)],
        right_child: PageNumber,
    ) -> Vec<u8> {
        let mut page = vec![0u8; PAGE_SIZE];
        let header_offset = if page_num == 1 { 100 } else { 0 };

        let mut content_end = PAGE_SIZE;
        let mut cell_offsets = Vec::new();

        for &(left_child, rowid) in children {
            // Cell format: left_child (4 bytes BE), rowid (varint)
            let mut cell = Vec::new();
            let mut tmp4 = [0u8; 4];
            tmp4.copy_from_slice(&left_child.to_be_bytes());
            cell.extend_from_slice(&tmp4);
            let mut tmp = [0u8; 9];
            let n = varint::write_varint(&mut tmp, rowid as u64);
            cell.extend_from_slice(&tmp[..n]);

            content_end -= cell.len();
            page[content_end..content_end + cell.len()].copy_from_slice(&cell);
            cell_offsets.push(content_end as u16);
        }

        // Write B-tree page header (12 bytes for interior).
        page[header_offset] = BTreePageType::TableInterior.to_flag(); // 0x05
        format::write_be_u16(&mut page, header_offset + 1, 0);
        format::write_be_u16(&mut page, header_offset + 3, children.len() as u16);
        format::write_be_u16(&mut page, header_offset + 5, content_end as u16);
        page[header_offset + 7] = 0;
        format::write_be_u32(&mut page, header_offset + 8, right_child);

        // Cell pointer array starts after the 12-byte header.
        let array_start = header_offset + 12;
        for (i, &off) in cell_offsets.iter().enumerate() {
            format::write_be_u16(&mut page, array_start + i * 2, off);
        }

        page
    }

    /// Build an index leaf page with the given payloads.
    fn build_index_leaf_page(page_num: PageNumber, payloads: &[&[u8]]) -> Vec<u8> {
        let mut page = vec![0u8; PAGE_SIZE];
        let header_offset = if page_num == 1 { 100 } else { 0 };

        let mut content_end = PAGE_SIZE;
        let mut cell_offsets = Vec::new();

        for payload in payloads {
            // Cell format: payload_len (varint), payload
            let mut cell = Vec::new();
            let mut tmp = [0u8; 9];
            let n = varint::write_varint(&mut tmp, payload.len() as u64);
            cell.extend_from_slice(&tmp[..n]);
            cell.extend_from_slice(payload);

            content_end -= cell.len();
            page[content_end..content_end + cell.len()].copy_from_slice(&cell);
            cell_offsets.push(content_end as u16);
        }

        page[header_offset] = BTreePageType::IndexLeaf.to_flag(); // 0x0A
        format::write_be_u16(&mut page, header_offset + 1, 0);
        format::write_be_u16(&mut page, header_offset + 3, payloads.len() as u16);
        format::write_be_u16(&mut page, header_offset + 5, content_end as u16);
        page[header_offset + 7] = 0;

        let array_start = header_offset + 8;
        for (i, &off) in cell_offsets.iter().enumerate() {
            format::write_be_u16(&mut page, array_start + i * 2, off);
        }

        page
    }

    /// Build an index interior page.
    fn build_index_interior_page(
        page_num: PageNumber,
        children: &[(PageNumber, &[u8])],
        right_child: PageNumber,
    ) -> Vec<u8> {
        let mut page = vec![0u8; PAGE_SIZE];
        let header_offset = if page_num == 1 { 100 } else { 0 };

        let mut content_end = PAGE_SIZE;
        let mut cell_offsets = Vec::new();

        for &(left_child, payload) in children {
            // Cell format: left_child (4 bytes BE), payload_len (varint), payload
            let mut cell = Vec::new();
            cell.extend_from_slice(&left_child.to_be_bytes());
            let mut tmp = [0u8; 9];
            let n = varint::write_varint(&mut tmp, payload.len() as u64);
            cell.extend_from_slice(&tmp[..n]);
            cell.extend_from_slice(payload);

            content_end -= cell.len();
            page[content_end..content_end + cell.len()].copy_from_slice(&cell);
            cell_offsets.push(content_end as u16);
        }

        page[header_offset] = BTreePageType::IndexInterior.to_flag(); // 0x02
        format::write_be_u16(&mut page, header_offset + 1, 0);
        format::write_be_u16(&mut page, header_offset + 3, children.len() as u16);
        format::write_be_u16(&mut page, header_offset + 5, content_end as u16);
        page[header_offset + 7] = 0;
        format::write_be_u32(&mut page, header_offset + 8, right_child);

        let array_start = header_offset + 12;
        for (i, &off) in cell_offsets.iter().enumerate() {
            format::write_be_u16(&mut page, array_start + i * 2, off);
        }

        page
    }

    /// Install a page into an in-memory pager.
    fn install_page(pager: &mut Pager, page_num: PageNumber, data: Vec<u8>) {
        // Make sure pager has enough pages allocated.
        while pager.page_count() < page_num {
            pager.allocate_page().unwrap();
        }
        let page = pager.get_page_mut(page_num).unwrap();
        page.data = data;
    }

    // -----------------------------------------------------------------------
    // B-tree page header tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_table_leaf_header() {
        let page = build_table_leaf_page(2, &[(1, b"hello"), (2, b"world")]);
        let header = BTreePageHeader::parse(&page, 0).unwrap();

        assert_eq!(header.page_type, BTreePageType::TableLeaf);
        assert_eq!(header.cell_count, 2);
        assert_eq!(header.first_freeblock, 0);
        assert_eq!(header.fragmented_free_bytes, 0);
        assert_eq!(header.right_child, 0);
        assert_eq!(header.header_size(), 8);
    }

    #[test]
    fn test_parse_table_interior_header() {
        let page = build_table_interior_page(2, &[(3, 10), (4, 20)], 5);
        let header = BTreePageHeader::parse(&page, 0).unwrap();

        assert_eq!(header.page_type, BTreePageType::TableInterior);
        assert_eq!(header.cell_count, 2);
        assert_eq!(header.right_child, 5);
        assert_eq!(header.header_size(), 12);
    }

    #[test]
    fn test_parse_header_page1_offset() {
        // On page 1, the B-tree header starts at offset 100.
        let page = build_table_leaf_page(1, &[(1, b"test")]);
        let header = BTreePageHeader::parse(&page, 100).unwrap();

        assert_eq!(header.page_type, BTreePageType::TableLeaf);
        assert_eq!(header.cell_count, 1);
    }

    #[test]
    fn test_parse_header_invalid_type() {
        let mut page = vec![0u8; PAGE_SIZE];
        page[0] = 0xFF; // invalid page type
        let result = BTreePageHeader::parse(&page, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_content_offset_zero_means_65536() {
        let mut page = vec![0u8; PAGE_SIZE];
        page[0] = BTreePageType::TableLeaf.to_flag();
        format::write_be_u16(&mut page, 5, 0); // content offset = 0
        let header = BTreePageHeader::parse(&page, 0).unwrap();
        assert_eq!(header.content_offset(), 65536);
    }

    // -----------------------------------------------------------------------
    // Cell pointer array tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_read_cell_pointers() {
        let page = build_table_leaf_page(2, &[(1, b"hello"), (2, b"world")]);
        let header = BTreePageHeader::parse(&page, 0).unwrap();
        let pointers = read_cell_pointers(&page, 0, &header).unwrap();

        assert_eq!(pointers.len(), 2);
        // Pointers should be valid offsets within the page.
        for &ptr in &pointers {
            assert!((ptr as usize) < PAGE_SIZE);
            assert!((ptr as usize) >= 8 + 4); // after header + pointers
        }
    }

    #[test]
    fn test_read_cell_pointers_empty_page() {
        let page = build_table_leaf_page(2, &[]);
        let header = BTreePageHeader::parse(&page, 0).unwrap();
        let pointers = read_cell_pointers(&page, 0, &header).unwrap();
        assert!(pointers.is_empty());
    }

    // -----------------------------------------------------------------------
    // Cell parsing tests (raw, no overflow)
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_table_leaf_cell() {
        let payload = b"hello world";
        let page = build_table_leaf_page(2, &[(42, payload)]);
        let header = BTreePageHeader::parse(&page, 0).unwrap();
        let pointers = read_cell_pointers(&page, 0, &header).unwrap();

        let (cell, total_size, overflow) =
            parse_cell_raw(&page, pointers[0] as usize, header.page_type, USABLE_SIZE)
                .unwrap();

        assert_eq!(total_size, payload.len());
        assert_eq!(overflow, 0);

        match cell {
            BTreeCell::TableLeaf {
                rowid,
                payload: data,
            } => {
                assert_eq!(rowid, 42);
                assert_eq!(data, payload.to_vec());
            }
            _ => panic!("expected TableLeaf cell"),
        }
    }

    #[test]
    fn test_parse_table_interior_cell() {
        let page = build_table_interior_page(2, &[(10, 100)], 20);
        let header = BTreePageHeader::parse(&page, 0).unwrap();
        let pointers = read_cell_pointers(&page, 0, &header).unwrap();

        let (cell, total_size, overflow) =
            parse_cell_raw(&page, pointers[0] as usize, header.page_type, USABLE_SIZE)
                .unwrap();

        assert_eq!(total_size, 0); // no payload
        assert_eq!(overflow, 0);

        match cell {
            BTreeCell::TableInterior { left_child, rowid } => {
                assert_eq!(left_child, 10);
                assert_eq!(rowid, 100);
            }
            _ => panic!("expected TableInterior cell"),
        }
    }

    #[test]
    fn test_parse_index_leaf_cell() {
        let payload = b"index key data";
        let page = build_index_leaf_page(2, &[payload]);
        let header = BTreePageHeader::parse(&page, 0).unwrap();
        let pointers = read_cell_pointers(&page, 0, &header).unwrap();

        let (cell, total_size, overflow) =
            parse_cell_raw(&page, pointers[0] as usize, header.page_type, USABLE_SIZE)
                .unwrap();

        assert_eq!(total_size, payload.len());
        assert_eq!(overflow, 0);

        match cell {
            BTreeCell::IndexLeaf { payload: data } => {
                assert_eq!(data, payload.to_vec());
            }
            _ => panic!("expected IndexLeaf cell"),
        }
    }

    #[test]
    fn test_parse_index_interior_cell() {
        let payload = b"index key";
        let page = build_index_interior_page(2, &[(7, payload)], 8);
        let header = BTreePageHeader::parse(&page, 0).unwrap();
        let pointers = read_cell_pointers(&page, 0, &header).unwrap();

        let (cell, total_size, overflow) =
            parse_cell_raw(&page, pointers[0] as usize, header.page_type, USABLE_SIZE)
                .unwrap();

        assert_eq!(total_size, payload.len());
        assert_eq!(overflow, 0);

        match cell {
            BTreeCell::IndexInterior {
                left_child,
                payload: data,
            } => {
                assert_eq!(left_child, 7);
                assert_eq!(data, payload.to_vec());
            }
            _ => panic!("expected IndexInterior cell"),
        }
    }

    // -----------------------------------------------------------------------
    // Cell accessor method tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cell_rowid() {
        let cell = BTreeCell::TableLeaf {
            rowid: 42,
            payload: vec![],
        };
        assert_eq!(cell.rowid(), Some(42));

        let cell = BTreeCell::TableInterior {
            left_child: 1,
            rowid: 99,
        };
        assert_eq!(cell.rowid(), Some(99));

        let cell = BTreeCell::IndexLeaf {
            payload: vec![1, 2, 3],
        };
        assert_eq!(cell.rowid(), None);
    }

    #[test]
    fn test_cell_payload() {
        let cell = BTreeCell::TableLeaf {
            rowid: 1,
            payload: vec![0xAA, 0xBB],
        };
        assert_eq!(cell.payload(), Some(&[0xAA, 0xBB][..]));

        let cell = BTreeCell::TableInterior {
            left_child: 1,
            rowid: 2,
        };
        assert_eq!(cell.payload(), None);
    }

    #[test]
    fn test_cell_left_child() {
        let cell = BTreeCell::TableInterior {
            left_child: 5,
            rowid: 10,
        };
        assert_eq!(cell.left_child(), Some(5));

        let cell = BTreeCell::TableLeaf {
            rowid: 1,
            payload: vec![],
        };
        assert_eq!(cell.left_child(), None);
    }

    // -----------------------------------------------------------------------
    // Multiple cells on a page
    // -----------------------------------------------------------------------

    #[test]
    fn test_multiple_table_leaf_cells() {
        let cells_data: Vec<(i64, &[u8])> = vec![
            (1, b"alpha"),
            (2, b"beta"),
            (3, b"gamma"),
            (4, b"delta"),
        ];
        let page = build_table_leaf_page(2, &cells_data);
        let header = BTreePageHeader::parse(&page, 0).unwrap();
        let pointers = read_cell_pointers(&page, 0, &header).unwrap();

        assert_eq!(pointers.len(), 4);

        for (i, &ptr) in pointers.iter().enumerate() {
            let (cell, _, _) =
                parse_cell_raw(&page, ptr as usize, header.page_type, USABLE_SIZE).unwrap();
            match cell {
                BTreeCell::TableLeaf { rowid, payload } => {
                    assert_eq!(rowid, cells_data[i].0);
                    assert_eq!(payload, cells_data[i].1);
                }
                _ => panic!("expected TableLeaf"),
            }
        }
    }

    #[test]
    fn test_multiple_interior_cells() {
        let children = vec![(2, 10i64), (3, 20), (4, 30)];
        let page = build_table_interior_page(5, &children, 6);
        let header = BTreePageHeader::parse(&page, 0).unwrap();
        let pointers = read_cell_pointers(&page, 0, &header).unwrap();

        assert_eq!(pointers.len(), 3);
        assert_eq!(header.right_child, 6);

        for (i, &ptr) in pointers.iter().enumerate() {
            let (cell, _, _) =
                parse_cell_raw(&page, ptr as usize, header.page_type, USABLE_SIZE).unwrap();
            match cell {
                BTreeCell::TableInterior { left_child, rowid } => {
                    assert_eq!(left_child, children[i].0);
                    assert_eq!(rowid, children[i].1);
                }
                _ => panic!("expected TableInterior"),
            }
        }
    }

    // -----------------------------------------------------------------------
    // Overflow page threshold computation
    // -----------------------------------------------------------------------

    #[test]
    fn test_table_leaf_max_local() {
        // For 4096-byte pages: max_local = 4096 - 35 = 4061
        assert_eq!(table_leaf_max_local(4096), 4061);
    }

    #[test]
    fn test_table_leaf_min_local() {
        // For 4096-byte pages: min_local = (4096 - 12) * 32 / 255 - 23
        // = 4084 * 32 / 255 - 23 = 512 - 23 = 489 (integer arithmetic)
        let min = table_leaf_min_local(4096);
        assert_eq!(min, (4096 - 12) * 32 / 255 - 23);
    }

    #[test]
    fn test_index_max_local() {
        let max = index_max_local(4096);
        assert_eq!(max, (4096 - 12) * 64 / 255 - 23);
    }

    #[test]
    fn test_compute_local_payload_no_overflow() {
        // Small payload fits entirely.
        let (local, overflow) = compute_local_payload_size(100, 4096, BTreePageType::TableLeaf);
        assert_eq!(local, 100);
        assert!(!overflow);
    }

    #[test]
    fn test_compute_local_payload_with_overflow() {
        // Payload larger than max_local should trigger overflow.
        let payload_size = 5000; // > 4061 (max_local for table leaf, 4096 page)
        let (local, overflow) =
            compute_local_payload_size(payload_size, 4096, BTreePageType::TableLeaf);
        assert!(overflow);
        assert!(local <= table_leaf_max_local(4096));
        assert!(local >= table_leaf_min_local(4096));
    }

    // -----------------------------------------------------------------------
    // Overflow page chain tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_overflow_page_chain() {
        let mut pager = Pager::in_memory();

        // Build a table leaf cell whose payload spills to overflow.
        // Use a large payload that exceeds max_local.
        let total_payload_size = 5000;
        let payload_data: Vec<u8> = (0..total_payload_size).map(|i| (i % 256) as u8).collect();

        let usable_size = pager.usable_size();
        let (local_size, has_overflow) =
            compute_local_payload_size(total_payload_size, usable_size, BTreePageType::TableLeaf);
        assert!(has_overflow);

        // Allocate overflow pages and fill them.
        let overflow_content = &payload_data[local_size..];
        let content_per_overflow = usable_size - 4;
        let num_overflow_pages =
            (overflow_content.len() + content_per_overflow - 1) / content_per_overflow;

        let mut overflow_page_nums = Vec::new();
        for _ in 0..num_overflow_pages {
            overflow_page_nums.push(pager.allocate_page().unwrap());
        }

        // Fill overflow pages.
        let mut remaining = overflow_content;
        for (idx, &pg) in overflow_page_nums.iter().enumerate() {
            let next_page = if idx + 1 < overflow_page_nums.len() {
                overflow_page_nums[idx + 1]
            } else {
                0
            };
            let to_copy = remaining.len().min(content_per_overflow);
            let page = pager.get_page_mut(pg).unwrap();
            format::write_be_u32(&mut page.data, 0, next_page);
            page.data[4..4 + to_copy].copy_from_slice(&remaining[..to_copy]);
            remaining = &remaining[to_copy..];
        }

        // Now build the leaf page with the cell pointing to overflow.
        let first_overflow = overflow_page_nums[0];

        // Build cell manually: payload_len varint, rowid varint, local payload, overflow ptr
        let mut cell_data = Vec::new();
        let mut tmp = [0u8; 9];
        let n = varint::write_varint(&mut tmp, total_payload_size as u64);
        cell_data.extend_from_slice(&tmp[..n]);
        let n = varint::write_varint(&mut tmp, 1u64); // rowid = 1
        cell_data.extend_from_slice(&tmp[..n]);
        cell_data.extend_from_slice(&payload_data[..local_size]);
        cell_data.extend_from_slice(&first_overflow.to_be_bytes());

        // Build a leaf page with this cell.
        let leaf_page_num = pager.allocate_page().unwrap();
        let leaf_page = pager.get_page_mut(leaf_page_num).unwrap();
        let header_offset = 0;

        // Place cell at the end of the page.
        let cell_start = PAGE_SIZE - cell_data.len();
        leaf_page.data[cell_start..].copy_from_slice(&cell_data);

        // Write the B-tree header.
        leaf_page.data[header_offset] = BTreePageType::TableLeaf.to_flag();
        format::write_be_u16(&mut leaf_page.data, header_offset + 1, 0);
        format::write_be_u16(&mut leaf_page.data, header_offset + 3, 1); // 1 cell
        format::write_be_u16(&mut leaf_page.data, header_offset + 5, cell_start as u16);
        leaf_page.data[header_offset + 7] = 0;

        // Cell pointer.
        format::write_be_u16(&mut leaf_page.data, header_offset + 8, cell_start as u16);

        // Now parse the cell with overflow resolution.
        let cell = parse_cell(&mut pager, leaf_page_num, cell_start, BTreePageType::TableLeaf)
            .unwrap();

        match cell {
            BTreeCell::TableLeaf { rowid, payload } => {
                assert_eq!(rowid, 1);
                assert_eq!(payload.len(), total_payload_size);
                assert_eq!(payload, payload_data);
            }
            _ => panic!("expected TableLeaf cell"),
        }
    }

    // -----------------------------------------------------------------------
    // BTreeCursor tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cursor_single_leaf_page() {
        let mut pager = Pager::in_memory();

        let cells: Vec<(i64, &[u8])> = vec![
            (1, b"alpha"),
            (2, b"beta"),
            (3, b"gamma"),
        ];
        let leaf_data = build_table_leaf_page(2, &cells);

        // Allocate page 2 and install data.
        pager.allocate_page().unwrap(); // page 2
        install_page(&mut pager, 2, leaf_data);

        let mut cursor = BTreeCursor::new(2);
        cursor.move_to_first(&mut pager).unwrap();
        assert!(cursor.is_valid());

        // Read all cells sequentially.
        let mut results = Vec::new();
        while cursor.is_valid() {
            let rowid = cursor.current_rowid(&mut pager).unwrap();
            let payload = cursor.current_payload(&mut pager).unwrap();
            results.push((rowid, payload));
            cursor.move_to_next(&mut pager).unwrap();
        }

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, 1);
        assert_eq!(results[0].1, b"alpha");
        assert_eq!(results[1].0, 2);
        assert_eq!(results[1].1, b"beta");
        assert_eq!(results[2].0, 3);
        assert_eq!(results[2].1, b"gamma");
    }

    #[test]
    fn test_cursor_empty_leaf() {
        let mut pager = Pager::in_memory();

        let leaf_data = build_table_leaf_page(2, &[]);
        pager.allocate_page().unwrap();
        install_page(&mut pager, 2, leaf_data);

        let mut cursor = BTreeCursor::new(2);
        cursor.move_to_first(&mut pager).unwrap();
        assert!(!cursor.is_valid());
    }

    #[test]
    fn test_cursor_two_level_tree() {
        let mut pager = Pager::in_memory();

        // Create a two-level tree:
        // Interior page 2 with cells pointing to leaves 3 and 4, right_child=4
        //   Leaf 3: rowids 1, 2, 3
        //   Leaf 4: rowids 4, 5, 6

        let leaf3 = build_table_leaf_page(3, &[
            (1, b"one"),
            (2, b"two"),
            (3, b"three"),
        ]);
        let leaf4 = build_table_leaf_page(4, &[
            (4, b"four"),
            (5, b"five"),
            (6, b"six"),
        ]);
        // Interior page: left_child=3 with rowid=3, right_child=4
        let interior = build_table_interior_page(2, &[(3, 3)], 4);

        pager.allocate_page().unwrap(); // page 2
        pager.allocate_page().unwrap(); // page 3
        pager.allocate_page().unwrap(); // page 4
        install_page(&mut pager, 2, interior);
        install_page(&mut pager, 3, leaf3);
        install_page(&mut pager, 4, leaf4);

        let mut cursor = BTreeCursor::new(2);
        cursor.move_to_first(&mut pager).unwrap();

        let mut results = Vec::new();
        while cursor.is_valid() {
            let rowid = cursor.current_rowid(&mut pager).unwrap();
            let payload = cursor.current_payload(&mut pager).unwrap();
            results.push((rowid, String::from_utf8(payload).unwrap()));
            cursor.move_to_next(&mut pager).unwrap();
        }

        assert_eq!(results.len(), 6);
        assert_eq!(results[0], (1, "one".to_string()));
        assert_eq!(results[1], (2, "two".to_string()));
        assert_eq!(results[2], (3, "three".to_string()));
        assert_eq!(results[3], (4, "four".to_string()));
        assert_eq!(results[4], (5, "five".to_string()));
        assert_eq!(results[5], (6, "six".to_string()));
    }

    #[test]
    fn test_cursor_three_level_tree() {
        let mut pager = Pager::in_memory();

        // Three-level tree:
        //   Root (page 2, interior): cells=(left=3, key=6), right_child=4
        //   Interior page 3: cells=(left=5, key=3), right_child=6
        //   Interior page 4: cells=(left=7, key=9), right_child=8
        //   Leaf 5: rowids 1, 2, 3
        //   Leaf 6: rowids 4, 5, 6
        //   Leaf 7: rowids 7, 8, 9
        //   Leaf 8: rowids 10, 11, 12

        let leaf5 = build_table_leaf_page(5, &[(1, b"a"), (2, b"b"), (3, b"c")]);
        let leaf6 = build_table_leaf_page(6, &[(4, b"d"), (5, b"e"), (6, b"f")]);
        let leaf7 = build_table_leaf_page(7, &[(7, b"g"), (8, b"h"), (9, b"i")]);
        let leaf8 = build_table_leaf_page(8, &[(10, b"j"), (11, b"k"), (12, b"l")]);

        let int3 = build_table_interior_page(3, &[(5, 3)], 6);
        let int4 = build_table_interior_page(4, &[(7, 9)], 8);
        let root = build_table_interior_page(2, &[(3, 6)], 4);

        for _ in 0..7 {
            pager.allocate_page().unwrap(); // pages 2-8
        }
        install_page(&mut pager, 2, root);
        install_page(&mut pager, 3, int3);
        install_page(&mut pager, 4, int4);
        install_page(&mut pager, 5, leaf5);
        install_page(&mut pager, 6, leaf6);
        install_page(&mut pager, 7, leaf7);
        install_page(&mut pager, 8, leaf8);

        let mut cursor = BTreeCursor::new(2);
        cursor.move_to_first(&mut pager).unwrap();

        let mut rowids = Vec::new();
        while cursor.is_valid() {
            rowids.push(cursor.current_rowid(&mut pager).unwrap());
            cursor.move_to_next(&mut pager).unwrap();
        }

        assert_eq!(rowids, (1..=12).collect::<Vec<i64>>());
    }

    // -----------------------------------------------------------------------
    // BTreeCursor seek tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cursor_seek_exact() {
        let mut pager = Pager::in_memory();

        let cells: Vec<(i64, &[u8])> = vec![
            (10, b"ten"),
            (20, b"twenty"),
            (30, b"thirty"),
        ];
        let leaf = build_table_leaf_page(2, &cells);
        pager.allocate_page().unwrap();
        install_page(&mut pager, 2, leaf);

        let mut cursor = BTreeCursor::new(2);

        // Seek to existing rowid.
        let found = cursor.seek_rowid(&mut pager, 20).unwrap();
        assert!(found);
        assert!(cursor.is_valid());
        assert_eq!(cursor.current_rowid(&mut pager).unwrap(), 20);
        assert_eq!(cursor.current_payload(&mut pager).unwrap(), b"twenty");
    }

    #[test]
    fn test_cursor_seek_not_found_positioned_on_successor() {
        let mut pager = Pager::in_memory();

        let cells: Vec<(i64, &[u8])> = vec![
            (10, b"ten"),
            (20, b"twenty"),
            (30, b"thirty"),
        ];
        let leaf = build_table_leaf_page(2, &cells);
        pager.allocate_page().unwrap();
        install_page(&mut pager, 2, leaf);

        let mut cursor = BTreeCursor::new(2);

        // Seek to non-existing rowid 15 -- should land on 20.
        let found = cursor.seek_rowid(&mut pager, 15).unwrap();
        assert!(!found);
        assert!(cursor.is_valid());
        assert_eq!(cursor.current_rowid(&mut pager).unwrap(), 20);
    }

    #[test]
    fn test_cursor_seek_past_end() {
        let mut pager = Pager::in_memory();

        let cells: Vec<(i64, &[u8])> = vec![
            (10, b"ten"),
            (20, b"twenty"),
        ];
        let leaf = build_table_leaf_page(2, &cells);
        pager.allocate_page().unwrap();
        install_page(&mut pager, 2, leaf);

        let mut cursor = BTreeCursor::new(2);

        // Seek to rowid 99 which is past all keys.
        let found = cursor.seek_rowid(&mut pager, 99).unwrap();
        assert!(!found);
        assert!(!cursor.is_valid());
    }

    #[test]
    fn test_cursor_seek_before_start() {
        let mut pager = Pager::in_memory();

        let cells: Vec<(i64, &[u8])> = vec![
            (10, b"ten"),
            (20, b"twenty"),
        ];
        let leaf = build_table_leaf_page(2, &cells);
        pager.allocate_page().unwrap();
        install_page(&mut pager, 2, leaf);

        let mut cursor = BTreeCursor::new(2);

        // Seek to rowid 1 which is before all keys -- should land on 10.
        let found = cursor.seek_rowid(&mut pager, 1).unwrap();
        assert!(!found);
        assert!(cursor.is_valid());
        assert_eq!(cursor.current_rowid(&mut pager).unwrap(), 10);
    }

    #[test]
    fn test_cursor_seek_two_level() {
        let mut pager = Pager::in_memory();

        // Two-level tree with leaves at pages 3 and 4, root at page 2.
        let leaf3 = build_table_leaf_page(3, &[(10, b"ten"), (20, b"twenty")]);
        let leaf4 = build_table_leaf_page(4, &[(30, b"thirty"), (40, b"forty")]);
        let root = build_table_interior_page(2, &[(3, 20)], 4);

        pager.allocate_page().unwrap(); // 2
        pager.allocate_page().unwrap(); // 3
        pager.allocate_page().unwrap(); // 4
        install_page(&mut pager, 2, root);
        install_page(&mut pager, 3, leaf3);
        install_page(&mut pager, 4, leaf4);

        let mut cursor = BTreeCursor::new(2);

        // Seek to 30 (in the second leaf).
        let found = cursor.seek_rowid(&mut pager, 30).unwrap();
        assert!(found);
        assert!(cursor.is_valid());
        assert_eq!(cursor.current_rowid(&mut pager).unwrap(), 30);
        assert_eq!(cursor.current_payload(&mut pager).unwrap(), b"thirty");

        // Seek to 25 (not found, should land on 30).
        let found = cursor.seek_rowid(&mut pager, 25).unwrap();
        assert!(!found);
        assert!(cursor.is_valid());
        assert_eq!(cursor.current_rowid(&mut pager).unwrap(), 30);

        // Seek to 10 (first key in first leaf).
        let found = cursor.seek_rowid(&mut pager, 10).unwrap();
        assert!(found);
        assert_eq!(cursor.current_rowid(&mut pager).unwrap(), 10);

        // Seek to 40 (last key).
        let found = cursor.seek_rowid(&mut pager, 40).unwrap();
        assert!(found);
        assert_eq!(cursor.current_rowid(&mut pager).unwrap(), 40);

        // Seek past end.
        let found = cursor.seek_rowid(&mut pager, 50).unwrap();
        assert!(!found);
        assert!(!cursor.is_valid());
    }

    #[test]
    fn test_cursor_seek_then_scan() {
        let mut pager = Pager::in_memory();

        let leaf3 = build_table_leaf_page(3, &[(1, b"a"), (2, b"b"), (3, b"c")]);
        let leaf4 = build_table_leaf_page(4, &[(4, b"d"), (5, b"e"), (6, b"f")]);
        let root = build_table_interior_page(2, &[(3, 3)], 4);

        pager.allocate_page().unwrap();
        pager.allocate_page().unwrap();
        pager.allocate_page().unwrap();
        install_page(&mut pager, 2, root);
        install_page(&mut pager, 3, leaf3);
        install_page(&mut pager, 4, leaf4);

        let mut cursor = BTreeCursor::new(2);

        // Seek to rowid 3 then scan forward.
        cursor.seek_rowid(&mut pager, 3).unwrap();
        assert!(cursor.is_valid());

        let mut rowids = Vec::new();
        while cursor.is_valid() {
            rowids.push(cursor.current_rowid(&mut pager).unwrap());
            cursor.move_to_next(&mut pager).unwrap();
        }

        assert_eq!(rowids, vec![3, 4, 5, 6]);
    }

    // -----------------------------------------------------------------------
    // read_page_cells convenience function
    // -----------------------------------------------------------------------

    #[test]
    fn test_read_page_cells() {
        let mut pager = Pager::in_memory();

        let leaf = build_table_leaf_page(2, &[(1, b"x"), (2, b"y")]);
        pager.allocate_page().unwrap();
        install_page(&mut pager, 2, leaf);

        let (header, cells) = read_page_cells(&mut pager, 2).unwrap();
        assert_eq!(header.page_type, BTreePageType::TableLeaf);
        assert_eq!(cells.len(), 2);

        assert_eq!(cells[0].rowid(), Some(1));
        assert_eq!(cells[0].payload(), Some(&b"x"[..]));
        assert_eq!(cells[1].rowid(), Some(2));
        assert_eq!(cells[1].payload(), Some(&b"y"[..]));
    }

    // -----------------------------------------------------------------------
    // read_left_child_pointer
    // -----------------------------------------------------------------------

    #[test]
    fn test_read_left_child_pointer() {
        let page = build_table_interior_page(2, &[(42, 100)], 50);
        let header = BTreePageHeader::parse(&page, 0).unwrap();
        let ptrs = read_cell_pointers(&page, 0, &header).unwrap();
        let child = read_left_child_pointer(&page, ptrs[0] as usize, header.page_type).unwrap();
        assert_eq!(child, 42);
    }

    #[test]
    fn test_read_left_child_pointer_leaf_error() {
        let page = build_table_leaf_page(2, &[(1, b"x")]);
        let result = read_left_child_pointer(&page, 0, BTreePageType::TableLeaf);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Varint helper error handling
    // -----------------------------------------------------------------------

    #[test]
    fn test_read_varint_checked_empty() {
        let result = read_varint_checked(&[], 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_varint_checked_out_of_bounds() {
        let buf = [0u8; 5];
        let result = read_varint_checked(&buf, 10);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Edge case: page 1 with B-tree
    // -----------------------------------------------------------------------

    #[test]
    fn test_page1_btree_header() {
        let page = build_table_leaf_page(1, &[(1, b"test")]);
        // On page 1, the B-tree header is at offset 100.
        let header = BTreePageHeader::parse(&page, 100).unwrap();
        assert_eq!(header.page_type, BTreePageType::TableLeaf);
        assert_eq!(header.cell_count, 1);
    }

    // -----------------------------------------------------------------------
    // Cursor on invalid state
    // -----------------------------------------------------------------------

    #[test]
    fn test_cursor_current_rowid_when_invalid() {
        let mut pager = Pager::in_memory();
        let cursor = BTreeCursor::new(2);
        let result = cursor.current_rowid(&mut pager);
        assert!(result.is_err());
    }

    #[test]
    fn test_cursor_current_payload_when_invalid() {
        let mut pager = Pager::in_memory();
        let cursor = BTreeCursor::new(2);
        let result = cursor.current_payload(&mut pager);
        assert!(result.is_err());
    }

    #[test]
    fn test_cursor_move_to_next_when_invalid() {
        let mut pager = Pager::in_memory();
        let mut cursor = BTreeCursor::new(2);
        let result = cursor.move_to_next(&mut pager).unwrap();
        assert!(!result);
    }
}
