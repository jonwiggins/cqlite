// Reader for the sqlite_master (sqlite_schema) table.
//
// The sqlite_master table is a regular table B-tree rooted at page 1.
// Each row has 5 columns:
//   1. type (TEXT)     - "table", "index", "view", or "trigger"
//   2. name (TEXT)     - name of the object
//   3. tbl_name (TEXT) - name of the table the object is associated with
//   4. rootpage (INTEGER) - root page number of the object's B-tree
//   5. sql (TEXT)      - the SQL CREATE statement (NULL for autoindex)

use crate::btree::BTreeCursor;
use crate::error::{Result, RsqliteError};
use crate::pager::Pager;
use crate::record;
use crate::types::Value;

/// A single entry from the sqlite_master table.
#[derive(Debug, Clone, PartialEq)]
pub struct SchemaEntry {
    /// The type of schema object: "table", "index", "view", or "trigger".
    pub entry_type: String,
    /// The name of the schema object.
    pub name: String,
    /// The name of the table this object is associated with.
    pub tbl_name: String,
    /// The root page number of the object's B-tree (0 for views/triggers).
    pub rootpage: i64,
    /// The SQL CREATE statement that defined this object (None for autoindex).
    pub sql: Option<String>,
}

/// Read all entries from the sqlite_master table.
///
/// The sqlite_master table is always a table B-tree rooted at page 1.
/// This function creates a cursor, iterates every row, decodes each record,
/// and returns the collected schema entries.
pub fn read_schema(pager: &mut Pager) -> Result<Vec<SchemaEntry>> {
    let mut cursor = BTreeCursor::new(1);
    cursor.move_to_first(pager)?;

    let mut entries = Vec::new();

    while cursor.is_valid() {
        let payload = cursor.current_payload(pager)?;
        let values = record::decode_record(&payload)?;

        let entry = schema_entry_from_values(&values)?;
        entries.push(entry);

        cursor.move_to_next(pager)?;
    }

    Ok(entries)
}

/// Extract a SchemaEntry from the decoded column values of a sqlite_master row.
///
/// Expects exactly 5 values in the order: type, name, tbl_name, rootpage, sql.
fn schema_entry_from_values(values: &[Value]) -> Result<SchemaEntry> {
    if values.len() < 5 {
        return Err(RsqliteError::Corrupt(format!(
            "sqlite_master row has {} columns, expected 5",
            values.len()
        )));
    }

    let entry_type = value_to_string(&values[0]).ok_or_else(|| {
        RsqliteError::Corrupt("sqlite_master.type is not text".into())
    })?;

    let name = value_to_string(&values[1]).ok_or_else(|| {
        RsqliteError::Corrupt("sqlite_master.name is not text".into())
    })?;

    let tbl_name = value_to_string(&values[2]).ok_or_else(|| {
        RsqliteError::Corrupt("sqlite_master.tbl_name is not text".into())
    })?;

    let rootpage = match &values[3] {
        Value::Integer(i) => *i,
        Value::Null => 0,
        other => {
            return Err(RsqliteError::Corrupt(format!(
                "sqlite_master.rootpage has unexpected type: {}",
                other.type_name()
            )));
        }
    };

    let sql = match &values[4] {
        Value::Text(s) => Some(s.clone()),
        Value::Null => None,
        other => {
            return Err(RsqliteError::Corrupt(format!(
                "sqlite_master.sql has unexpected type: {}",
                other.type_name()
            )));
        }
    };

    Ok(SchemaEntry {
        entry_type,
        name,
        tbl_name,
        rootpage,
        sql,
    })
}

/// Extract a String from a Value, accepting Text directly.
fn value_to_string(value: &Value) -> Option<String> {
    match value {
        Value::Text(s) => Some(s.clone()),
        _ => None,
    }
}

/// Find a table entry by name (case-insensitive) in a list of schema entries.
///
/// Only returns entries where `entry_type` is "table".
pub fn find_table<'a>(schema: &'a [SchemaEntry], name: &str) -> Option<&'a SchemaEntry> {
    let name_lower = name.to_lowercase();
    schema.iter().find(|entry| {
        entry.entry_type == "table" && entry.name.to_lowercase() == name_lower
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{self, BTreePageType};
    use crate::varint;

    const PAGE_SIZE: usize = 4096;

    // -----------------------------------------------------------------------
    // Helper: build a sqlite_master record payload for a table entry
    // -----------------------------------------------------------------------

    fn build_master_record(
        entry_type: &str,
        name: &str,
        tbl_name: &str,
        rootpage: i64,
        sql: Option<&str>,
    ) -> Vec<u8> {
        let values = vec![
            Value::Text(entry_type.to_string()),
            Value::Text(name.to_string()),
            Value::Text(tbl_name.to_string()),
            Value::Integer(rootpage),
            match sql {
                Some(s) => Value::Text(s.to_string()),
                None => Value::Null,
            },
        ];
        record::encode_record(&values)
    }

    /// Build a table leaf page (page 1) with the given cells.
    /// Each cell is (rowid, payload_bytes).
    fn build_page1_leaf(cells: &[(i64, &[u8])]) -> Vec<u8> {
        let mut page = vec![0u8; PAGE_SIZE];

        // Write a valid database header in the first 100 bytes.
        let header = format::DatabaseHeader::new();
        let mut hdr_buf = [0u8; format::HEADER_SIZE];
        header.write(&mut hdr_buf);
        page[..format::HEADER_SIZE].copy_from_slice(&hdr_buf);

        let header_offset = 100; // page 1 B-tree header starts at offset 100

        // Build cells from the end of the page backward.
        let mut content_end = PAGE_SIZE;
        let mut cell_offsets = Vec::new();

        for &(rowid, payload) in cells {
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

        // Write B-tree page header at offset 100.
        page[header_offset] = BTreePageType::TableLeaf.to_flag();
        format::write_be_u16(&mut page, header_offset + 1, 0); // first freeblock
        format::write_be_u16(&mut page, header_offset + 3, cells.len() as u16);
        format::write_be_u16(&mut page, header_offset + 5, content_end as u16);
        page[header_offset + 7] = 0; // fragmented free bytes

        // Write cell pointer array (after 8-byte leaf header).
        let array_start = header_offset + 8;
        for (i, &off) in cell_offsets.iter().enumerate() {
            format::write_be_u16(&mut page, array_start + i * 2, off);
        }

        page
    }

    /// Install page 1 data into an in-memory pager.
    fn install_page1(pager: &mut Pager, data: Vec<u8>) {
        let page = pager.get_page_mut(1).unwrap();
        page.data = data;
    }

    // -----------------------------------------------------------------------
    // Unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_schema_entry_from_values() {
        let values = vec![
            Value::Text("table".into()),
            Value::Text("users".into()),
            Value::Text("users".into()),
            Value::Integer(2),
            Value::Text("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)".into()),
        ];

        let entry = schema_entry_from_values(&values).unwrap();
        assert_eq!(entry.entry_type, "table");
        assert_eq!(entry.name, "users");
        assert_eq!(entry.tbl_name, "users");
        assert_eq!(entry.rootpage, 2);
        assert_eq!(
            entry.sql,
            Some("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)".into())
        );
    }

    #[test]
    fn test_schema_entry_with_null_sql() {
        let values = vec![
            Value::Text("index".into()),
            Value::Text("sqlite_autoindex_users_1".into()),
            Value::Text("users".into()),
            Value::Integer(3),
            Value::Null,
        ];

        let entry = schema_entry_from_values(&values).unwrap();
        assert_eq!(entry.entry_type, "index");
        assert_eq!(entry.name, "sqlite_autoindex_users_1");
        assert_eq!(entry.sql, None);
    }

    #[test]
    fn test_schema_entry_too_few_columns() {
        let values = vec![
            Value::Text("table".into()),
            Value::Text("users".into()),
        ];
        let result = schema_entry_from_values(&values);
        assert!(result.is_err());
    }

    #[test]
    fn test_find_table_case_insensitive() {
        let entries = vec![
            SchemaEntry {
                entry_type: "table".into(),
                name: "Users".into(),
                tbl_name: "Users".into(),
                rootpage: 2,
                sql: Some("CREATE TABLE Users (id INTEGER)".into()),
            },
            SchemaEntry {
                entry_type: "index".into(),
                name: "idx_users_name".into(),
                tbl_name: "Users".into(),
                rootpage: 3,
                sql: Some("CREATE INDEX idx_users_name ON Users (name)".into()),
            },
        ];

        // Find by exact case.
        let found = find_table(&entries, "Users");
        assert!(found.is_some());
        assert_eq!(found.unwrap().name, "Users");

        // Find by different case.
        let found = find_table(&entries, "users");
        assert!(found.is_some());
        assert_eq!(found.unwrap().name, "Users");

        let found = find_table(&entries, "USERS");
        assert!(found.is_some());

        // Should not find an index entry.
        let found = find_table(&entries, "idx_users_name");
        assert!(found.is_none());

        // Should not find non-existent table.
        let found = find_table(&entries, "orders");
        assert!(found.is_none());
    }

    #[test]
    fn test_read_schema_single_table() {
        let mut pager = Pager::in_memory();

        let sql = "CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)";
        let record = build_master_record("table", "test", "test", 2, Some(sql));

        let page_data = build_page1_leaf(&[(1, &record)]);
        install_page1(&mut pager, page_data);

        let schema = read_schema(&mut pager).unwrap();
        assert_eq!(schema.len(), 1);
        assert_eq!(schema[0].entry_type, "table");
        assert_eq!(schema[0].name, "test");
        assert_eq!(schema[0].tbl_name, "test");
        assert_eq!(schema[0].rootpage, 2);
        assert_eq!(schema[0].sql, Some(sql.to_string()));
    }

    #[test]
    fn test_read_schema_multiple_entries() {
        let mut pager = Pager::in_memory();

        let table_sql = "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)";
        let index_sql = "CREATE INDEX idx_email ON users (email)";

        let record1 = build_master_record("table", "users", "users", 2, Some(table_sql));
        let record2 = build_master_record("index", "idx_email", "users", 3, Some(index_sql));

        let page_data = build_page1_leaf(&[
            (1, &record1),
            (2, &record2),
        ]);
        install_page1(&mut pager, page_data);

        let schema = read_schema(&mut pager).unwrap();
        assert_eq!(schema.len(), 2);

        assert_eq!(schema[0].entry_type, "table");
        assert_eq!(schema[0].name, "users");
        assert_eq!(schema[0].rootpage, 2);

        assert_eq!(schema[1].entry_type, "index");
        assert_eq!(schema[1].name, "idx_email");
        assert_eq!(schema[1].tbl_name, "users");
        assert_eq!(schema[1].rootpage, 3);
    }

    #[test]
    fn test_read_schema_empty_database() {
        let mut pager = Pager::in_memory();

        // Build page 1 with no cells (empty schema).
        let page_data = build_page1_leaf(&[]);
        install_page1(&mut pager, page_data);

        let schema = read_schema(&mut pager).unwrap();
        assert!(schema.is_empty());
    }

    #[test]
    fn test_read_schema_with_autoindex() {
        let mut pager = Pager::in_memory();

        let table_sql = "CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT UNIQUE)";
        let record1 = build_master_record("table", "t", "t", 2, Some(table_sql));
        // Autoindex has NULL sql.
        let record2 = build_master_record("index", "sqlite_autoindex_t_1", "t", 3, None);

        let page_data = build_page1_leaf(&[
            (1, &record1),
            (2, &record2),
        ]);
        install_page1(&mut pager, page_data);

        let schema = read_schema(&mut pager).unwrap();
        assert_eq!(schema.len(), 2);

        assert_eq!(schema[1].entry_type, "index");
        assert_eq!(schema[1].name, "sqlite_autoindex_t_1");
        assert_eq!(schema[1].sql, None);
    }

    #[test]
    fn test_read_schema_and_find_table() {
        let mut pager = Pager::in_memory();

        let sql1 = "CREATE TABLE orders (id INTEGER PRIMARY KEY)";
        let sql2 = "CREATE TABLE products (id INTEGER PRIMARY KEY)";
        let record1 = build_master_record("table", "orders", "orders", 2, Some(sql1));
        let record2 = build_master_record("table", "products", "products", 3, Some(sql2));

        let page_data = build_page1_leaf(&[
            (1, &record1),
            (2, &record2),
        ]);
        install_page1(&mut pager, page_data);

        let schema = read_schema(&mut pager).unwrap();
        assert_eq!(schema.len(), 2);

        let found = find_table(&schema, "orders");
        assert!(found.is_some());
        assert_eq!(found.unwrap().rootpage, 2);

        let found = find_table(&schema, "Products");
        assert!(found.is_some());
        assert_eq!(found.unwrap().rootpage, 3);

        let found = find_table(&schema, "nonexistent");
        assert!(found.is_none());
    }

    // -----------------------------------------------------------------------
    // Integration test: read schema from a real SQLite database
    // -----------------------------------------------------------------------

    /// If sqlite3 is available on the system, create a real database, then
    /// read its schema using our implementation and verify correctness.
    #[test]
    fn test_read_schema_from_real_sqlite_db() {
        use std::process::Command;

        // Check if sqlite3 is available.
        let check = Command::new("sqlite3").arg("--version").output();
        if check.is_err() || !check.unwrap().status.success() {
            eprintln!("sqlite3 not found, skipping integration test");
            return;
        }

        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_schema.db");

        // Create a database with two tables and an index using sqlite3.
        let sql = "\
            CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT);\n\
            CREATE TABLE posts (id INTEGER PRIMARY KEY, user_id INTEGER, title TEXT, body TEXT);\n\
            CREATE INDEX idx_posts_user ON posts (user_id);\n\
        ";

        let status = Command::new("sqlite3")
            .arg(db_path.to_str().unwrap())
            .stdin(std::process::Stdio::piped())
            .spawn()
            .and_then(|mut child| {
                use std::io::Write;
                child.stdin.as_mut().unwrap().write_all(sql.as_bytes())?;
                child.wait()
            });

        if status.is_err() || !status.unwrap().success() {
            eprintln!("failed to create test database with sqlite3, skipping");
            return;
        }

        // Open with our pager and read the schema.
        let mut pager = Pager::open(&db_path).unwrap();
        let schema = read_schema(&mut pager).unwrap();

        // We should have at least 3 entries: 2 tables + 1 index.
        assert!(
            schema.len() >= 3,
            "expected at least 3 schema entries, got {}",
            schema.len()
        );

        // Verify the users table.
        let users = find_table(&schema, "users");
        assert!(users.is_some(), "users table not found in schema");
        let users = users.unwrap();
        assert_eq!(users.entry_type, "table");
        assert_eq!(users.name, "users");
        assert!(users.rootpage > 0);
        assert!(users.sql.is_some());
        let users_sql = users.sql.as_ref().unwrap().to_uppercase();
        assert!(
            users_sql.contains("CREATE TABLE"),
            "users sql should contain CREATE TABLE"
        );

        // Verify the posts table.
        let posts = find_table(&schema, "posts");
        assert!(posts.is_some(), "posts table not found in schema");
        let posts = posts.unwrap();
        assert_eq!(posts.entry_type, "table");
        assert_eq!(posts.name, "posts");
        assert!(posts.rootpage > 0);

        // Verify the index.
        let idx = schema.iter().find(|e| e.name == "idx_posts_user");
        assert!(idx.is_some(), "idx_posts_user index not found in schema");
        let idx = idx.unwrap();
        assert_eq!(idx.entry_type, "index");
        assert_eq!(idx.tbl_name, "posts");
        assert!(idx.rootpage > 0);
        assert!(idx.sql.is_some());

        // Verify that find_table does NOT return the index.
        let not_table = find_table(&schema, "idx_posts_user");
        assert!(
            not_table.is_none(),
            "find_table should not return index entries"
        );
    }
}
