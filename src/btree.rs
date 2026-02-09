use crate::error::{Result, RsqliteError};
use crate::format::{self, BTreePageHeader, BTreePageType};
use crate::pager::Pager;
use crate::record;
use crate::types::Value;
use crate::varint;

/// A B-tree cursor for iterating over table or index entries.
pub struct BTreeCursor {
    root_page: u32,
    /// Stack of (page_num, cell_index) for traversal
    stack: Vec<(u32, usize)>,
    initialized: bool,
}

impl BTreeCursor {
    pub fn new(root_page: u32) -> Self {
        BTreeCursor {
            root_page,
            stack: Vec::new(),
            initialized: false,
        }
    }

    /// Move to the first entry in the tree.
    pub fn first(&mut self, pager: &mut Pager) -> Result<bool> {
        self.stack.clear();
        self.initialized = true;
        self.move_to_leftmost(pager, self.root_page)
    }

    /// Move to the next entry. Returns false if no more entries.
    pub fn next(&mut self, pager: &mut Pager) -> Result<bool> {
        if !self.initialized {
            return self.first(pager);
        }

        loop {
            if self.stack.is_empty() {
                return Ok(false);
            }

            let (page_num, cell_idx) = self.stack.last_mut().unwrap();
            let page_data = pager.get_page(*page_num)?.data.clone();
            let header_offset = if *page_num == 1 { 100 } else { 0 };
            let header = BTreePageHeader::parse(&page_data[header_offset..])?;

            *cell_idx += 1;

            if header.page_type.is_leaf() {
                if *cell_idx < header.cell_count as usize {
                    return Ok(true);
                }
                // Pop this leaf and continue up
                self.stack.pop();
            } else {
                // Interior page
                if *cell_idx < header.cell_count as usize {
                    // Go to the left child of the next cell
                    let cell_ptr_offset = header_offset + header.header_size() + (*cell_idx) * 2;
                    let cell_offset = format::read_be_u16(&page_data[cell_ptr_offset..]) as usize;
                    let child_page = format::read_be_u32(&page_data[cell_offset..]);

                    // For table interior, we read the current cell's key, then descend into child
                    // Actually, for interior nodes we need to descend into the left child pointer
                    // The cell at cell_idx contains a left child pointer and a key
                    // We first return the key, then descend into the next child
                    // Wait - in SQLite, interior table cells don't contain data, just keys + pointers.
                    // We don't return interior cells for table scans.

                    // For a table B-tree interior node, move to the leftmost of the child
                    return self.move_to_leftmost(pager, child_page);
                } else if *cell_idx == header.cell_count as usize {
                    // Right-most pointer
                    if let Some(rmp) = header.right_most_pointer {
                        // Don't increment again
                        *cell_idx += 1; // Mark as consumed
                        return self.move_to_leftmost(pager, rmp);
                    }
                }
                self.stack.pop();
            }
        }
    }

    /// Get the current entry's rowid and payload (for table B-trees).
    pub fn current(&self, pager: &mut Pager) -> Result<Option<(i64, Vec<u8>)>> {
        if let Some(&(page_num, cell_idx)) = self.stack.last() {
            let page_data = pager.get_page(page_num)?.data.clone();
            let header_offset = if page_num == 1 { 100 } else { 0 };
            let header = BTreePageHeader::parse(&page_data[header_offset..])?;

            if !header.page_type.is_leaf() || !header.page_type.is_table() {
                return Err(RsqliteError::Internal(
                    "Cursor on non-leaf-table page".into(),
                ));
            }

            if cell_idx >= header.cell_count as usize {
                return Ok(None);
            }

            let cell_ptr_offset = header_offset + header.header_size() + cell_idx * 2;
            let cell_offset = format::read_be_u16(&page_data[cell_ptr_offset..]) as usize;

            let (payload_size, n1) = varint::read_varint(&page_data[cell_offset..]);
            let (rowid, n2) = varint::read_varint_i64(&page_data[cell_offset + n1..]);

            let payload_start = cell_offset + n1 + n2;
            let payload_end = payload_start + payload_size as usize;

            if payload_end > page_data.len() {
                // Overflow page handling would go here
                // For now, handle inline payload
                let available = page_data.len() - payload_start;
                let payload = page_data[payload_start..payload_start + available].to_vec();
                return Ok(Some((rowid, payload)));
            }

            let payload = page_data[payload_start..payload_end].to_vec();
            Ok(Some((rowid, payload)))
        } else {
            Ok(None)
        }
    }

    fn move_to_leftmost(&mut self, pager: &mut Pager, mut page_num: u32) -> Result<bool> {
        loop {
            let page_data = pager.get_page(page_num)?.data.clone();
            let header_offset = if page_num == 1 { 100 } else { 0 };
            let header = BTreePageHeader::parse(&page_data[header_offset..])?;

            if header.page_type.is_leaf() {
                if header.cell_count == 0 {
                    // Empty leaf
                    return Ok(false);
                }
                self.stack.push((page_num, 0));
                return Ok(true);
            }

            // Interior page: go to first child (left pointer of first cell)
            if header.cell_count == 0 {
                // Use right-most pointer if no cells
                if let Some(rmp) = header.right_most_pointer {
                    self.stack.push((page_num, 0));
                    page_num = rmp;
                } else {
                    return Ok(false);
                }
            } else {
                let cell_ptr_offset = header_offset + header.header_size();
                let cell_offset = format::read_be_u16(&page_data[cell_ptr_offset..]) as usize;
                let child_page = format::read_be_u32(&page_data[cell_offset..]);
                self.stack.push((page_num, 0));
                page_num = child_page;
            }
        }
    }

    /// Search for a specific rowid in a table B-tree.
    pub fn seek(&mut self, pager: &mut Pager, target_rowid: i64) -> Result<bool> {
        self.stack.clear();
        self.initialized = true;
        self.seek_in_page(pager, self.root_page, target_rowid)
    }

    fn seek_in_page(&mut self, pager: &mut Pager, page_num: u32, target: i64) -> Result<bool> {
        let page_data = pager.get_page(page_num)?.data.clone();
        let header_offset = if page_num == 1 { 100 } else { 0 };
        let header = BTreePageHeader::parse(&page_data[header_offset..])?;

        if header.page_type.is_leaf() {
            // Linear scan through cells to find rowid
            for i in 0..header.cell_count as usize {
                let cell_ptr_offset = header_offset + header.header_size() + i * 2;
                let cell_offset = format::read_be_u16(&page_data[cell_ptr_offset..]) as usize;

                let (_, n1) = varint::read_varint(&page_data[cell_offset..]);
                let (rowid, _) = varint::read_varint_i64(&page_data[cell_offset + n1..]);

                if rowid == target {
                    self.stack.push((page_num, i));
                    return Ok(true);
                }
            }
            return Ok(false);
        }

        // Interior page: binary search
        for i in 0..header.cell_count as usize {
            let cell_ptr_offset = header_offset + header.header_size() + i * 2;
            let cell_offset = format::read_be_u16(&page_data[cell_ptr_offset..]) as usize;

            let child_page = format::read_be_u32(&page_data[cell_offset..]);
            let (key, _) = varint::read_varint_i64(&page_data[cell_offset + 4..]);

            if target <= key {
                self.stack.push((page_num, i));
                return self.seek_in_page(pager, child_page, target);
            }
        }

        // Target is greater than all keys, go to right-most pointer
        if let Some(rmp) = header.right_most_pointer {
            self.stack.push((page_num, header.cell_count as usize));
            return self.seek_in_page(pager, rmp, target);
        }

        Ok(false)
    }
}

/// Insert a record into a table B-tree.
pub fn insert_into_table(
    pager: &mut Pager,
    root_page: u32,
    rowid: i64,
    record_data: &[u8],
) -> Result<u32> {
    let result = insert_into_page(pager, root_page, rowid, record_data)?;

    match result {
        InsertResult::Done => Ok(root_page),
        InsertResult::Split {
            new_page,
            split_key,
        } => {
            // Need a new root
            let new_root = pager.allocate_page()?;
            let page_size = pager.page_size as usize;
            let page = pager.get_page_mut(new_root)?;
            page.data = vec![0u8; page_size];

            let header_offset = if new_root == 1 { 100 } else { 0 };

            // Set up interior table page
            page.data[header_offset] = BTreePageType::InteriorTable.to_byte();
            // right-most pointer = new_page
            format::write_be_u32(&mut page.data[header_offset + 8..], new_page);
            // cell count = 1
            format::write_be_u16(&mut page.data[header_offset + 3..], 1);

            // Write cell: 4-byte left child pointer + varint key
            let mut cell = Vec::new();
            cell.extend_from_slice(&root_page.to_be_bytes());
            varint::write_varint_to_vec(&mut cell, split_key as u64);

            // Write cell at end of page
            let cell_start = page_size - cell.len();
            page.data[cell_start..cell_start + cell.len()].copy_from_slice(&cell);

            // Cell pointer
            let ptr_offset = header_offset + 12; // after interior header
            format::write_be_u16(&mut page.data[ptr_offset..], cell_start as u16);

            // Cell content offset
            format::write_be_u16(&mut page.data[header_offset + 5..], cell_start as u16);

            Ok(new_root)
        }
    }
}

enum InsertResult {
    Done,
    Split { new_page: u32, split_key: i64 },
}

fn insert_into_page(
    pager: &mut Pager,
    page_num: u32,
    rowid: i64,
    record_data: &[u8],
) -> Result<InsertResult> {
    let page_data = pager.get_page(page_num)?.data.clone();
    let header_offset = if page_num == 1 { 100 } else { 0 };
    let header = BTreePageHeader::parse(&page_data[header_offset..])?;

    if header.page_type.is_leaf() {
        return insert_into_leaf(pager, page_num, rowid, record_data);
    }

    // Interior page: find correct child
    for i in 0..header.cell_count as usize {
        let cell_ptr_offset = header_offset + header.header_size() + i * 2;
        let cell_offset = format::read_be_u16(&page_data[cell_ptr_offset..]) as usize;
        let child_page = format::read_be_u32(&page_data[cell_offset..]);
        let (key, _) = varint::read_varint_i64(&page_data[cell_offset + 4..]);

        if rowid <= key {
            let result = insert_into_page(pager, child_page, rowid, record_data)?;
            return handle_child_split(pager, page_num, i, result);
        }
    }

    // Insert into right-most child
    if let Some(rmp) = header.right_most_pointer {
        let result = insert_into_page(pager, rmp, rowid, record_data)?;
        return handle_child_split(pager, page_num, header.cell_count as usize, result);
    }

    Err(RsqliteError::Internal("No child to insert into".into()))
}

fn handle_child_split(
    _pager: &mut Pager,
    _page_num: u32,
    _cell_idx: usize,
    result: InsertResult,
) -> Result<InsertResult> {
    match result {
        InsertResult::Done => Ok(InsertResult::Done),
        InsertResult::Split { .. } => {
            // For simplicity, propagate the split up
            Ok(result)
        }
    }
}

fn insert_into_leaf(
    pager: &mut Pager,
    page_num: u32,
    rowid: i64,
    record_data: &[u8],
) -> Result<InsertResult> {
    let page_size = pager.page_size as usize;
    let header_offset = if page_num == 1 { 100 } else { 0 };

    // Build the cell: payload_size varint + rowid varint + payload
    let mut cell = Vec::new();
    varint::write_varint_to_vec(&mut cell, record_data.len() as u64);
    varint::write_varint_to_vec(&mut cell, rowid as u64);
    cell.extend_from_slice(record_data);

    // Read current page state
    let page_data = pager.get_page(page_num)?.data.clone();
    let header = BTreePageHeader::parse(&page_data[header_offset..])?;
    let cell_count = header.cell_count as usize;

    // Read existing cells and their rowids
    let mut cells: Vec<(i64, Vec<u8>)> = Vec::new();
    for i in 0..cell_count {
        let ptr_offset = header_offset + header.header_size() + i * 2;
        let cell_offset = format::read_be_u16(&page_data[ptr_offset..]) as usize;

        let (payload_size, n1) = varint::read_varint(&page_data[cell_offset..]);
        let (existing_rowid, n2) = varint::read_varint_i64(&page_data[cell_offset + n1..]);
        let cell_size = n1 + n2 + payload_size as usize;
        let existing_cell = page_data[cell_offset..cell_offset + cell_size].to_vec();
        cells.push((existing_rowid, existing_cell));
    }

    // Insert new cell in sorted position
    let insert_pos = cells.partition_point(|(rid, _)| *rid < rowid);

    // Check for duplicate rowid
    if insert_pos < cells.len() && cells[insert_pos].0 == rowid {
        // Replace existing
        cells[insert_pos] = (rowid, cell);
    } else {
        cells.insert(insert_pos, (rowid, cell));
    }

    // Check if everything fits on the page
    let ptrs_size = cells.len() * 2;
    let total_cell_size: usize = cells.iter().map(|(_, c)| c.len()).sum();
    let used_space = header_offset
        + (if header.page_type.is_interior() {
            12
        } else {
            8
        })
        + ptrs_size
        + total_cell_size;

    if used_space > page_size {
        // Need to split
        return split_leaf(pager, page_num, cells);
    }

    // Write cells back to page
    write_leaf_page(pager, page_num, &cells)?;
    Ok(InsertResult::Done)
}

fn split_leaf(
    pager: &mut Pager,
    page_num: u32,
    cells: Vec<(i64, Vec<u8>)>,
) -> Result<InsertResult> {
    let mid = cells.len() / 2;
    let split_key = cells[mid].0;

    let left_cells: Vec<(i64, Vec<u8>)> = cells[..mid].to_vec();
    let right_cells: Vec<(i64, Vec<u8>)> = cells[mid..].to_vec();

    // Left page stays as page_num
    write_leaf_page(pager, page_num, &left_cells)?;

    // Allocate right page
    let right_page = pager.allocate_page()?;
    write_leaf_page(pager, right_page, &right_cells)?;

    Ok(InsertResult::Split {
        new_page: right_page,
        split_key,
    })
}

fn write_leaf_page(pager: &mut Pager, page_num: u32, cells: &[(i64, Vec<u8>)]) -> Result<()> {
    let page_size = pager.page_size as usize;
    let header_offset = if page_num == 1 { 100 } else { 0 };

    let page = pager.get_page_mut(page_num)?;

    // Preserve the database header on page 1
    let saved_header = if page_num == 1 {
        Some(page.data[..100].to_vec())
    } else {
        None
    };

    page.data = vec![0u8; page_size];

    if let Some(h) = saved_header {
        page.data[..100].copy_from_slice(&h);
    }

    // Write header
    page.data[header_offset] = BTreePageType::LeafTable.to_byte();
    format::write_be_u16(&mut page.data[header_offset + 3..], cells.len() as u16);

    // Write cells from bottom of page upward
    let mut content_offset = page_size;
    let ptr_start = header_offset + 8; // leaf header is 8 bytes

    for (i, (_, cell_data)) in cells.iter().enumerate() {
        content_offset -= cell_data.len();
        page.data[content_offset..content_offset + cell_data.len()].copy_from_slice(cell_data);
        // Write cell pointer
        format::write_be_u16(&mut page.data[ptr_start + i * 2..], content_offset as u16);
    }

    // Cell content offset
    if content_offset == page_size {
        format::write_be_u16(&mut page.data[header_offset + 5..], 0);
    } else {
        format::write_be_u16(&mut page.data[header_offset + 5..], content_offset as u16);
    }

    Ok(())
}

/// Delete a row from a table B-tree by rowid.
pub fn delete_from_table(pager: &mut Pager, root_page: u32, rowid: i64) -> Result<bool> {
    let page_data = pager.get_page(root_page)?.data.clone();
    let header_offset = if root_page == 1 { 100 } else { 0 };
    let header = BTreePageHeader::parse(&page_data[header_offset..])?;

    if header.page_type.is_leaf() {
        // Find and remove the cell with this rowid
        let mut cells: Vec<(i64, Vec<u8>)> = Vec::new();
        let mut found = false;

        for i in 0..header.cell_count as usize {
            let ptr_offset = header_offset + header.header_size() + i * 2;
            let cell_offset = format::read_be_u16(&page_data[ptr_offset..]) as usize;

            let (payload_size, n1) = varint::read_varint(&page_data[cell_offset..]);
            let (existing_rowid, n2) = varint::read_varint_i64(&page_data[cell_offset + n1..]);
            let cell_size = n1 + n2 + payload_size as usize;
            let cell_data = page_data[cell_offset..cell_offset + cell_size].to_vec();

            if existing_rowid == rowid {
                found = true;
            } else {
                cells.push((existing_rowid, cell_data));
            }
        }

        if found {
            write_leaf_page(pager, root_page, &cells)?;
        }
        return Ok(found);
    }

    // Interior page: find correct child
    for i in 0..header.cell_count as usize {
        let cell_ptr_offset = header_offset + header.header_size() + i * 2;
        let cell_offset = format::read_be_u16(&page_data[cell_ptr_offset..]) as usize;
        let child_page = format::read_be_u32(&page_data[cell_offset..]);
        let (key, _) = varint::read_varint_i64(&page_data[cell_offset + 4..]);

        if rowid <= key {
            return delete_from_table(pager, child_page, rowid);
        }
    }

    if let Some(rmp) = header.right_most_pointer {
        return delete_from_table(pager, rmp, rowid);
    }

    Ok(false)
}

/// Get the maximum rowid in a table B-tree.
pub fn max_rowid(pager: &mut Pager, root_page: u32) -> Result<i64> {
    let page_data = pager.get_page(root_page)?.data.clone();
    let header_offset = if root_page == 1 { 100 } else { 0 };
    let header = BTreePageHeader::parse(&page_data[header_offset..])?;

    if header.cell_count == 0 {
        if let Some(rmp) = header.right_most_pointer {
            return max_rowid(pager, rmp);
        }
        return Ok(0);
    }

    if header.page_type.is_leaf() {
        // Last cell has the max rowid
        let last_cell_idx = header.cell_count as usize - 1;
        let ptr_offset = header_offset + header.header_size() + last_cell_idx * 2;
        let cell_offset = format::read_be_u16(&page_data[ptr_offset..]) as usize;
        let (_, n1) = varint::read_varint(&page_data[cell_offset..]);
        let (rowid, _) = varint::read_varint_i64(&page_data[cell_offset + n1..]);
        return Ok(rowid);
    }

    // Interior: go to right-most
    if let Some(rmp) = header.right_most_pointer {
        return max_rowid(pager, rmp);
    }

    // Last cell's key
    let last_cell_idx = header.cell_count as usize - 1;
    let ptr_offset = header_offset + header.header_size() + last_cell_idx * 2;
    let cell_offset = format::read_be_u16(&page_data[ptr_offset..]) as usize;
    let (key, _) = varint::read_varint_i64(&page_data[cell_offset + 4..]);
    Ok(key)
}

/// Scan all rows in a table B-tree, calling the callback with (rowid, values).
pub fn scan_table(pager: &mut Pager, root_page: u32) -> Result<Vec<(i64, Vec<Value>)>> {
    let mut results = Vec::new();
    let mut cursor = BTreeCursor::new(root_page);

    if !cursor.first(pager)? {
        return Ok(results);
    }

    loop {
        if let Some((rowid, payload)) = cursor.current(pager)? {
            let values = record::deserialize_record(&payload)?;
            results.push((rowid, values));
        }

        if !cursor.next(pager)? {
            break;
        }
    }

    Ok(results)
}

/// Initialize a page as an empty leaf table.
pub fn init_leaf_table_page(pager: &mut Pager, page_num: u32) -> Result<()> {
    let page_size = pager.page_size as usize;
    let header_offset = if page_num == 1 { 100 } else { 0 };

    let page = pager.get_page_mut(page_num)?;
    // Don't overwrite db header on page 1
    let start = header_offset;
    for i in start..page_size {
        page.data[i] = 0;
    }
    page.data[header_offset] = BTreePageType::LeafTable.to_byte();
    // cell count = 0
    format::write_be_u16(&mut page.data[header_offset + 3..], 0);
    // cell content offset = 0 (empty page)
    format::write_be_u16(&mut page.data[header_offset + 5..], 0);

    Ok(())
}

/// Index B-tree operations.
///
/// Insert an entry into an index B-tree.
pub fn insert_into_index(
    pager: &mut Pager,
    root_page: u32,
    key_values: &[Value],
    rowid: i64,
) -> Result<u32> {
    // Index entries are stored as records with the key values + rowid appended
    let mut values = key_values.to_vec();
    values.push(Value::Integer(rowid));
    let record_data = record::serialize_record(&values);

    insert_into_index_page(pager, root_page, &record_data, rowid)
}

fn insert_into_index_page(
    pager: &mut Pager,
    page_num: u32,
    record_data: &[u8],
    rowid: i64,
) -> Result<u32> {
    let page_data = pager.get_page(page_num)?.data.clone();
    let header_offset = if page_num == 1 { 100 } else { 0 };
    let header = BTreePageHeader::parse(&page_data[header_offset..])?;
    let page_size = pager.page_size as usize;

    if !header.page_type.is_leaf() {
        // For simplicity, just handle leaf index pages for now
        return Err(RsqliteError::Internal(
            "Interior index insert not implemented".into(),
        ));
    }

    // Build the cell: payload_size varint + payload
    let mut cell = Vec::new();
    varint::write_varint_to_vec(&mut cell, record_data.len() as u64);
    cell.extend_from_slice(record_data);

    // Read existing cells
    let mut cells: Vec<(i64, Vec<u8>)> = Vec::new();
    for i in 0..header.cell_count as usize {
        let ptr_offset = header_offset + header.header_size() + i * 2;
        let cell_offset = format::read_be_u16(&page_data[ptr_offset..]) as usize;

        let (payload_size, n1) = varint::read_varint(&page_data[cell_offset..]);
        let cell_size = n1 + payload_size as usize;
        let cell_data = page_data[cell_offset..cell_offset + cell_size].to_vec();

        // Extract rowid from the record for ordering
        let payload = &page_data[cell_offset + n1..cell_offset + cell_size];
        let values = record::deserialize_record(payload)?;
        let existing_rowid = if let Some(Value::Integer(rid)) = values.last() {
            *rid
        } else {
            0
        };
        cells.push((existing_rowid, cell_data));
    }

    // Insert in rowid order
    let insert_pos = cells.partition_point(|(rid, _)| *rid < rowid);
    cells.insert(insert_pos, (rowid, cell));

    // Check if fits
    let ptrs_size = cells.len() * 2;
    let total_cell_size: usize = cells.iter().map(|(_, c)| c.len()).sum();
    let used_space = header_offset + 8 + ptrs_size + total_cell_size;

    if used_space > page_size {
        // Split - for now just allocate a new root
        let mid = cells.len() / 2;
        let _split_key = cells[mid].0;

        let left_cells: Vec<(i64, Vec<u8>)> = cells[..mid].to_vec();
        let right_cells: Vec<(i64, Vec<u8>)> = cells[mid..].to_vec();

        write_index_leaf_page(pager, page_num, &left_cells)?;

        let right_page = pager.allocate_page()?;
        write_index_leaf_page(pager, right_page, &right_cells)?;

        // Create new root
        let new_root = pager.allocate_page()?;
        let ps = pager.page_size as usize;
        let page = pager.get_page_mut(new_root)?;
        page.data = vec![0u8; ps];

        let ho = if new_root == 1 { 100 } else { 0 };
        page.data[ho] = BTreePageType::InteriorIndex.to_byte();
        format::write_be_u32(&mut page.data[ho + 8..], right_page);
        format::write_be_u16(&mut page.data[ho + 3..], 1);

        // Cell: 4-byte left child + record payload
        let mut new_cell = Vec::new();
        new_cell.extend_from_slice(&page_num.to_be_bytes());
        // Use the first record of right page as the divider key
        new_cell.extend_from_slice(&right_cells[0].1);

        let cell_start = ps - new_cell.len();
        page.data[cell_start..cell_start + new_cell.len()].copy_from_slice(&new_cell);
        format::write_be_u16(&mut page.data[ho + 12..], cell_start as u16);
        format::write_be_u16(&mut page.data[ho + 5..], cell_start as u16);

        return Ok(new_root);
    }

    write_index_leaf_page(pager, page_num, &cells)?;
    Ok(page_num)
}

fn write_index_leaf_page(pager: &mut Pager, page_num: u32, cells: &[(i64, Vec<u8>)]) -> Result<()> {
    let page_size = pager.page_size as usize;
    let header_offset = if page_num == 1 { 100 } else { 0 };

    let page = pager.get_page_mut(page_num)?;

    let saved_header = if page_num == 1 {
        Some(page.data[..100].to_vec())
    } else {
        None
    };

    page.data = vec![0u8; page_size];

    if let Some(h) = saved_header {
        page.data[..100].copy_from_slice(&h);
    }

    page.data[header_offset] = BTreePageType::LeafIndex.to_byte();
    format::write_be_u16(&mut page.data[header_offset + 3..], cells.len() as u16);

    let mut content_offset = page_size;
    let ptr_start = header_offset + 8;

    for (i, (_, cell_data)) in cells.iter().enumerate() {
        content_offset -= cell_data.len();
        page.data[content_offset..content_offset + cell_data.len()].copy_from_slice(cell_data);
        format::write_be_u16(&mut page.data[ptr_start + i * 2..], content_offset as u16);
    }

    if content_offset == page_size {
        format::write_be_u16(&mut page.data[header_offset + 5..], 0);
    } else {
        format::write_be_u16(&mut page.data[header_offset + 5..], content_offset as u16);
    }

    Ok(())
}

/// Scan all entries in an index B-tree.
pub fn scan_index(pager: &mut Pager, root_page: u32) -> Result<Vec<Vec<Value>>> {
    let mut results = Vec::new();
    scan_index_page(pager, root_page, &mut results)?;
    Ok(results)
}

fn scan_index_page(pager: &mut Pager, page_num: u32, results: &mut Vec<Vec<Value>>) -> Result<()> {
    let page_data = pager.get_page(page_num)?.data.clone();
    let header_offset = if page_num == 1 { 100 } else { 0 };
    let header = BTreePageHeader::parse(&page_data[header_offset..])?;

    if header.page_type.is_leaf() {
        for i in 0..header.cell_count as usize {
            let ptr_offset = header_offset + header.header_size() + i * 2;
            let cell_offset = format::read_be_u16(&page_data[ptr_offset..]) as usize;
            let (payload_size, n1) = varint::read_varint(&page_data[cell_offset..]);
            let payload = &page_data[cell_offset + n1..cell_offset + n1 + payload_size as usize];
            let values = record::deserialize_record(payload)?;
            results.push(values);
        }
    } else {
        // Interior index page
        for i in 0..header.cell_count as usize {
            let ptr_offset = header_offset + header.header_size() + i * 2;
            let cell_offset = format::read_be_u16(&page_data[ptr_offset..]) as usize;
            let child_page = format::read_be_u32(&page_data[cell_offset..]);

            // Recurse into left child
            scan_index_page(pager, child_page, results)?;

            // Read this cell's payload
            let (payload_size, n1) = varint::read_varint(&page_data[cell_offset + 4..]);
            let payload_start = cell_offset + 4 + n1;
            let payload = &page_data[payload_start..payload_start + payload_size as usize];
            let values = record::deserialize_record(payload)?;
            results.push(values);
        }

        // Recurse into right-most child
        if let Some(rmp) = header.right_most_pointer {
            scan_index_page(pager, rmp, results)?;
        }
    }

    Ok(())
}

/// Delete an entry from an index B-tree by rowid.
pub fn delete_from_index(pager: &mut Pager, root_page: u32, rowid: i64) -> Result<bool> {
    let page_data = pager.get_page(root_page)?.data.clone();
    let header_offset = if root_page == 1 { 100 } else { 0 };
    let header = BTreePageHeader::parse(&page_data[header_offset..])?;

    if header.page_type.is_leaf() {
        let mut cells: Vec<(i64, Vec<u8>)> = Vec::new();
        let mut found = false;

        for i in 0..header.cell_count as usize {
            let ptr_offset = header_offset + header.header_size() + i * 2;
            let cell_offset = format::read_be_u16(&page_data[ptr_offset..]) as usize;
            let (payload_size, n1) = varint::read_varint(&page_data[cell_offset..]);
            let cell_size = n1 + payload_size as usize;
            let cell_data = page_data[cell_offset..cell_offset + cell_size].to_vec();

            let payload = &page_data[cell_offset + n1..cell_offset + cell_size];
            let values = record::deserialize_record(payload)?;
            let existing_rowid = if let Some(Value::Integer(rid)) = values.last() {
                *rid
            } else {
                0
            };

            if existing_rowid == rowid {
                found = true;
            } else {
                cells.push((existing_rowid, cell_data));
            }
        }

        if found {
            write_index_leaf_page(pager, root_page, &cells)?;
        }
        return Ok(found);
    }

    // Interior: recurse
    for i in 0..header.cell_count as usize {
        let cell_ptr_offset = header_offset + header.header_size() + i * 2;
        let cell_offset = format::read_be_u16(&page_data[cell_ptr_offset..]) as usize;
        let child_page = format::read_be_u32(&page_data[cell_offset..]);
        if delete_from_index(pager, child_page, rowid)? {
            return Ok(true);
        }
    }

    if let Some(rmp) = header.right_most_pointer {
        return delete_from_index(pager, rmp, rowid);
    }

    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_scan() {
        let mut pager = Pager::new_memory();
        let root = pager.allocate_page().unwrap();
        init_leaf_table_page(&mut pager, root).unwrap();

        let record1 = record::serialize_record(&[Value::Text("hello".into()), Value::Integer(42)]);
        let record2 = record::serialize_record(&[Value::Text("world".into()), Value::Integer(99)]);

        insert_into_table(&mut pager, root, 1, &record1).unwrap();
        insert_into_table(&mut pager, root, 2, &record2).unwrap();

        let rows = scan_table(&mut pager, root).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].0, 1);
        assert_eq!(
            rows[0].1,
            vec![Value::Text("hello".into()), Value::Integer(42)]
        );
        assert_eq!(rows[1].0, 2);
        assert_eq!(
            rows[1].1,
            vec![Value::Text("world".into()), Value::Integer(99)]
        );
    }

    #[test]
    fn test_delete_from_table() {
        let mut pager = Pager::new_memory();
        let root = pager.allocate_page().unwrap();
        init_leaf_table_page(&mut pager, root).unwrap();

        let record1 = record::serialize_record(&[Value::Integer(1)]);
        let record2 = record::serialize_record(&[Value::Integer(2)]);
        let record3 = record::serialize_record(&[Value::Integer(3)]);

        insert_into_table(&mut pager, root, 1, &record1).unwrap();
        insert_into_table(&mut pager, root, 2, &record2).unwrap();
        insert_into_table(&mut pager, root, 3, &record3).unwrap();

        assert!(delete_from_table(&mut pager, root, 2).unwrap());

        let rows = scan_table(&mut pager, root).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].0, 1);
        assert_eq!(rows[1].0, 3);
    }

    #[test]
    fn test_max_rowid() {
        let mut pager = Pager::new_memory();
        let root = pager.allocate_page().unwrap();
        init_leaf_table_page(&mut pager, root).unwrap();

        assert_eq!(max_rowid(&mut pager, root).unwrap(), 0);

        let r = record::serialize_record(&[Value::Integer(1)]);
        insert_into_table(&mut pager, root, 5, &r).unwrap();
        insert_into_table(&mut pager, root, 10, &r).unwrap();
        insert_into_table(&mut pager, root, 3, &r).unwrap();

        assert_eq!(max_rowid(&mut pager, root).unwrap(), 10);
    }
}
