// Query planner: translates AST into execution plans.
//
// This module provides the `Database` struct which is the main entry point
// for executing SQL statements. It bridges:
//   - Parser: SQL text → AST
//   - Schema: reads sqlite_master to discover tables/indexes
//   - VM: executes SELECT queries against B-tree storage
//
// Column names are extracted from CREATE TABLE SQL in the schema entries.

use crate::ast::Statement;
use crate::error::{Result, RsqliteError};
use crate::pager::Pager;
use crate::schema::{self, SchemaEntry};
use crate::types::Value;
use crate::vm::{self, ColumnInfo, QueryResult, Row, TableSchema};
use std::path::Path;

/// Top-level database handle. Owns the pager and provides SQL execution.
pub struct Database {
    pub pager: Pager,
}

impl Database {
    /// Open an existing database file (or create a new one).
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let pager = Pager::open(path)?;
        Ok(Self { pager })
    }

    /// Create an in-memory database.
    pub fn in_memory() -> Self {
        Self {
            pager: Pager::in_memory(),
        }
    }

    /// Execute a SQL statement and return the result.
    pub fn execute(&mut self, sql: &str) -> Result<QueryResult> {
        let stmt = crate::parser::parse(sql)?;
        self.execute_statement(&stmt)
    }

    /// Execute an already-parsed statement.
    pub fn execute_statement(&mut self, stmt: &Statement) -> Result<QueryResult> {
        match stmt {
            Statement::Select(select) => {
                let schema_entries = schema::read_schema(&mut self.pager)?;
                let table_schemas = build_table_schemas(&schema_entries)?;
                vm::execute_select(&mut self.pager, select, &table_schemas)
            }
            Statement::Explain(inner) => {
                // Simple EXPLAIN: show the statement type
                let description = format!("{inner:?}");
                Ok(QueryResult {
                    columns: vec![ColumnInfo {
                        name: "detail".to_string(),
                        table: None,
                    }],
                    rows: vec![Row {
                        values: vec![Value::Text(description)],
                    }],
                })
            }
            Statement::Pragma(pragma) => execute_pragma(pragma, &mut self.pager),
            _ => Err(RsqliteError::NotImplemented(format!(
                "statement type: {stmt:?}"
            ))),
        }
    }

    /// Read the database schema (all entries from sqlite_master).
    pub fn schema(&mut self) -> Result<Vec<SchemaEntry>> {
        schema::read_schema(&mut self.pager)
    }

    /// List all table names in the database.
    pub fn table_names(&mut self) -> Result<Vec<String>> {
        let entries = self.schema()?;
        Ok(entries
            .into_iter()
            .filter(|e| e.entry_type == "table")
            .map(|e| e.name)
            .collect())
    }
}

/// Build TableSchema structs from schema entries by parsing column names
/// from CREATE TABLE SQL.
fn build_table_schemas(entries: &[SchemaEntry]) -> Result<Vec<TableSchema>> {
    let mut schemas = Vec::new();

    for entry in entries {
        if entry.entry_type != "table" {
            continue;
        }

        let (columns, rowid_column) = if let Some(ref sql) = entry.sql {
            extract_table_info(sql)
        } else {
            (vec![], None)
        };

        schemas.push(TableSchema {
            name: entry.name.clone(),
            columns,
            root_page: entry.rootpage as u32,
            rowid_column,
        });
    }

    Ok(schemas)
}

/// Extract column names and INTEGER PRIMARY KEY index from a CREATE TABLE SQL.
///
/// Returns (column_names, rowid_column_index).
/// rowid_column_index is Some(i) if column i is declared as INTEGER PRIMARY KEY.
fn extract_table_info(sql: &str) -> (Vec<String>, Option<usize>) {
    // Try the parser first.
    if let Ok(stmt) = crate::parser::parse(sql) {
        if let Statement::CreateTable(ct) = stmt {
            let columns: Vec<String> = ct.columns.iter().map(|c| c.name.clone()).collect();
            let rowid_col = find_integer_primary_key(&ct);
            return (columns, rowid_col);
        }
    }

    // Fallback: simple text-based extraction.
    let columns = extract_column_names_text(sql);
    // In text fallback, also try to detect INTEGER PRIMARY KEY.
    let rowid_col = detect_integer_pk_text(sql, &columns);
    (columns, rowid_col)
}

/// Extract column names from a CREATE TABLE SQL statement (text-based fallback).
fn extract_column_names(sql: &str) -> Vec<String> {
    extract_table_info(sql).0
}

/// Find the INTEGER PRIMARY KEY column from a parsed CREATE TABLE statement.
fn find_integer_primary_key(ct: &crate::ast::CreateTableStatement) -> Option<usize> {
    for (i, col) in ct.columns.iter().enumerate() {
        let has_pk = col.constraints.iter().any(|c| {
            matches!(c, crate::ast::ColumnConstraint::PrimaryKey { .. })
        });
        if has_pk {
            // Check if the type is INTEGER (case-insensitive).
            if let Some(ref type_name) = col.type_name {
                if type_name.eq_ignore_ascii_case("INTEGER") {
                    return Some(i);
                }
            }
        }
    }
    None
}

/// Text-based detection of INTEGER PRIMARY KEY column.
fn detect_integer_pk_text(sql: &str, columns: &[String]) -> Option<usize> {
    let upper = sql.to_uppercase();
    for (i, col) in columns.iter().enumerate() {
        let pattern = format!(
            "{} INTEGER PRIMARY KEY",
            col.to_uppercase()
        );
        if upper.contains(&pattern) {
            return Some(i);
        }
    }
    None
}

/// Simple text-based column name extraction from CREATE TABLE SQL.
/// Finds content between first '(' and matching ')', splits by comma
/// at the top level, and takes the first word of each item.
fn extract_column_names_text(sql: &str) -> Vec<String> {
    let open = match sql.find('(') {
        Some(i) => i,
        None => return vec![],
    };
    let close = match sql.rfind(')') {
        Some(i) => i,
        None => return vec![],
    };

    if open >= close {
        return vec![];
    }

    let body = &sql[open + 1..close];

    // Split by commas at the top level (respecting parentheses).
    let mut columns = Vec::new();
    let mut depth = 0;
    let mut start = 0;

    for (i, ch) in body.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => {
                let part = body[start..i].trim();
                if let Some(name) = extract_first_identifier(part) {
                    // Skip table constraints (PRIMARY KEY, UNIQUE, CHECK, FOREIGN KEY).
                    let upper = name.to_uppercase();
                    if upper != "PRIMARY"
                        && upper != "UNIQUE"
                        && upper != "CHECK"
                        && upper != "FOREIGN"
                        && upper != "CONSTRAINT"
                    {
                        columns.push(name);
                    }
                }
                start = i + 1;
            }
            _ => {}
        }
    }

    // Handle the last segment.
    let part = body[start..].trim();
    if let Some(name) = extract_first_identifier(part) {
        let upper = name.to_uppercase();
        if upper != "PRIMARY"
            && upper != "UNIQUE"
            && upper != "CHECK"
            && upper != "FOREIGN"
            && upper != "CONSTRAINT"
        {
            columns.push(name);
        }
    }

    columns
}

/// Extract the first identifier from a column definition string.
/// Handles quoted identifiers (double-quoted, backtick, bracket).
fn extract_first_identifier(s: &str) -> Option<String> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }

    // Check for quoted identifier.
    if s.starts_with('"') {
        if let Some(end) = s[1..].find('"') {
            return Some(s[1..1 + end].to_string());
        }
    }
    if s.starts_with('`') {
        if let Some(end) = s[1..].find('`') {
            return Some(s[1..1 + end].to_string());
        }
    }
    if s.starts_with('[') {
        if let Some(end) = s[1..].find(']') {
            return Some(s[1..1 + end].to_string());
        }
    }

    // Unquoted: take until whitespace or punctuation.
    let end = s
        .find(|c: char| c.is_whitespace() || c == '(' || c == ',' || c == ')')
        .unwrap_or(s.len());

    if end == 0 {
        None
    } else {
        Some(s[..end].to_string())
    }
}

/// Execute a PRAGMA statement.
fn execute_pragma(
    pragma: &crate::ast::PragmaStatement,
    pager: &mut Pager,
) -> Result<QueryResult> {
    let name = pragma.name.to_lowercase();

    match name.as_str() {
        "table_info" => {
            // PRAGMA table_info(table_name) - return column information.
            let table_name = match &pragma.value {
                Some(crate::ast::PragmaValue::Name(n)) => n.clone(),
                Some(crate::ast::PragmaValue::StringLiteral(s)) => s.clone(),
                _ => {
                    return Err(RsqliteError::Runtime(
                        "PRAGMA table_info requires a table name".into(),
                    ))
                }
            };

            let schema_entries = schema::read_schema(pager)?;
            let entry = schema::find_table(&schema_entries, &table_name).ok_or_else(|| {
                RsqliteError::Runtime(format!("no such table: {table_name}"))
            })?;

            let columns = if let Some(ref sql) = entry.sql {
                extract_column_names(sql)
            } else {
                vec![]
            };

            let col_infos = vec![
                ColumnInfo {
                    name: "cid".into(),
                    table: None,
                },
                ColumnInfo {
                    name: "name".into(),
                    table: None,
                },
                ColumnInfo {
                    name: "type".into(),
                    table: None,
                },
                ColumnInfo {
                    name: "notnull".into(),
                    table: None,
                },
                ColumnInfo {
                    name: "dflt_value".into(),
                    table: None,
                },
                ColumnInfo {
                    name: "pk".into(),
                    table: None,
                },
            ];

            let rows: Vec<Row> = columns
                .iter()
                .enumerate()
                .map(|(i, name)| Row {
                    values: vec![
                        Value::Integer(i as i64),
                        Value::Text(name.clone()),
                        Value::Text(String::new()), // type - would need parsing
                        Value::Integer(0),           // notnull
                        Value::Null,                 // dflt_value
                        Value::Integer(0),           // pk
                    ],
                })
                .collect();

            Ok(QueryResult {
                columns: col_infos,
                rows,
            })
        }
        "page_size" => Ok(QueryResult {
            columns: vec![ColumnInfo {
                name: "page_size".into(),
                table: None,
            }],
            rows: vec![Row {
                values: vec![Value::Integer(pager.page_size() as i64)],
            }],
        }),
        "page_count" => {
            let count = pager.header.page_count;
            Ok(QueryResult {
                columns: vec![ColumnInfo {
                    name: "page_count".into(),
                    table: None,
                }],
                rows: vec![Row {
                    values: vec![Value::Integer(count as i64)],
                }],
            })
        }
        _ => {
            // Unknown pragma: return empty result.
            Ok(QueryResult {
                columns: vec![],
                rows: vec![],
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_column_names_simple() {
        let sql = "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)";
        let cols = extract_column_names_text(sql);
        assert_eq!(cols, vec!["id", "name", "email"]);
    }

    #[test]
    fn test_extract_column_names_with_constraints() {
        let sql = "CREATE TABLE t (a INT, b TEXT, PRIMARY KEY (a))";
        let cols = extract_column_names_text(sql);
        assert_eq!(cols, vec!["a", "b"]);
    }

    #[test]
    fn test_extract_column_names_quoted() {
        let sql = r#"CREATE TABLE t ("col 1" TEXT, `col2` INT, [col 3] REAL)"#;
        let cols = extract_column_names_text(sql);
        assert_eq!(cols, vec!["col 1", "col2", "col 3"]);
    }

    #[test]
    fn test_extract_column_names_with_default() {
        let sql = "CREATE TABLE t (id INTEGER, status TEXT DEFAULT 'active', count INT DEFAULT 0)";
        let cols = extract_column_names_text(sql);
        assert_eq!(cols, vec!["id", "status", "count"]);
    }

    #[test]
    fn test_extract_column_names_no_parens() {
        let cols = extract_column_names_text("CREATE TABLE t");
        assert!(cols.is_empty());
    }

    #[test]
    fn test_extract_first_identifier() {
        assert_eq!(extract_first_identifier("name TEXT"), Some("name".into()));
        assert_eq!(
            extract_first_identifier("  id INTEGER PRIMARY KEY"),
            Some("id".into())
        );
        assert_eq!(
            extract_first_identifier(r#""my col" TEXT"#),
            Some("my col".into())
        );
        assert_eq!(extract_first_identifier(""), None);
    }

    #[test]
    fn test_build_table_schemas_empty() {
        let schemas = build_table_schemas(&[]).unwrap();
        assert!(schemas.is_empty());
    }

    #[test]
    fn test_build_table_schemas_with_index() {
        let entries = vec![
            SchemaEntry {
                entry_type: "table".into(),
                name: "users".into(),
                tbl_name: "users".into(),
                rootpage: 2,
                sql: Some(
                    "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)".into(),
                ),
            },
            SchemaEntry {
                entry_type: "index".into(),
                name: "idx".into(),
                tbl_name: "users".into(),
                rootpage: 3,
                sql: Some("CREATE INDEX idx ON users (name)".into()),
            },
        ];

        let schemas = build_table_schemas(&entries).unwrap();
        assert_eq!(schemas.len(), 1);
        assert_eq!(schemas[0].name, "users");
        assert_eq!(schemas[0].root_page, 2);
        assert_eq!(schemas[0].columns, vec!["id", "name"]);
    }

    #[test]
    fn test_database_in_memory() {
        let _db = Database::in_memory();
        // Just verify construction doesn't panic.
    }

    #[test]
    fn test_extract_column_names_complex() {
        let sql = "CREATE TABLE orders (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL REFERENCES users(id), total REAL CHECK(total >= 0), created_at TEXT DEFAULT CURRENT_TIMESTAMP)";
        let cols = extract_column_names_text(sql);
        assert_eq!(cols, vec!["id", "user_id", "total", "created_at"]);
    }

    #[test]
    fn test_extract_column_names_unique_constraint() {
        let sql = "CREATE TABLE t (a TEXT, b TEXT, UNIQUE(a, b))";
        let cols = extract_column_names_text(sql);
        assert_eq!(cols, vec!["a", "b"]);
    }

    #[test]
    fn test_extract_column_names_foreign_key() {
        let sql =
            "CREATE TABLE t (id INT, ref_id INT, FOREIGN KEY (ref_id) REFERENCES other(id))";
        let cols = extract_column_names_text(sql);
        assert_eq!(cols, vec!["id", "ref_id"]);
    }

    #[test]
    fn test_extract_column_names_check_constraint() {
        let sql = "CREATE TABLE t (val INT, CHECK(val > 0))";
        let cols = extract_column_names_text(sql);
        assert_eq!(cols, vec!["val"]);
    }

    #[test]
    fn test_extract_column_names_constraint_keyword() {
        let sql =
            "CREATE TABLE t (a INT, b INT, CONSTRAINT pk PRIMARY KEY (a))";
        let cols = extract_column_names_text(sql);
        assert_eq!(cols, vec!["a", "b"]);
    }

    // Integration test with real sqlite3.
    #[test]
    fn test_read_table_data_from_real_db() {
        use std::process::Command;

        // Check if sqlite3 is available.
        let check = Command::new("sqlite3").arg("--version").output();
        if check.is_err() || !check.unwrap().status.success() {
            eprintln!("sqlite3 not found, skipping integration test");
            return;
        }

        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_read.db");

        // Create a database with a table and some data.
        let sql = "\
            CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER);\n\
            INSERT INTO users VALUES (1, 'Alice', 30);\n\
            INSERT INTO users VALUES (2, 'Bob', 25);\n\
            INSERT INTO users VALUES (3, 'Charlie', 35);\n\
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

        // Open with our Database and read the schema.
        let mut db = Database::open(&db_path).unwrap();

        // Verify schema reading works.
        let tables = db.table_names().unwrap();
        assert!(
            tables.iter().any(|t| t == "users"),
            "users table not found"
        );

        // Read the schema and build table schemas.
        let entries = db.schema().unwrap();
        let table_schemas = build_table_schemas(&entries).unwrap();

        let users_schema = table_schemas.iter().find(|s| s.name == "users").unwrap();
        assert_eq!(users_schema.columns, vec!["id", "name", "age"]);
        assert_eq!(users_schema.rowid_column, Some(0), "id should be detected as INTEGER PRIMARY KEY");
        assert!(users_schema.root_page > 0);

        // Execute a full table scan using the VM directly.
        let select = crate::ast::SelectStatement {
            distinct: false,
            columns: vec![crate::ast::ResultColumn::AllColumns],
            from: Some(crate::ast::FromClause {
                table: crate::ast::TableRef::Table {
                    name: "users".into(),
                    alias: None,
                },
                joins: vec![],
            }),
            where_clause: None,
            group_by: None,
            having: None,
            order_by: None,
            limit: None,
        };

        let result =
            vm::execute_select(&mut db.pager, &select, &table_schemas).unwrap();

        assert_eq!(result.columns.len(), 3);
        assert_eq!(result.columns[0].name, "id");
        assert_eq!(result.columns[1].name, "name");
        assert_eq!(result.columns[2].name, "age");

        assert_eq!(result.rows.len(), 3);

        // Verify data (rows should be in insertion order since it's a B-tree by rowid).
        assert_eq!(result.rows[0].values[0], Value::Integer(1));
        assert_eq!(result.rows[0].values[1], Value::Text("Alice".into()));
        assert_eq!(result.rows[0].values[2], Value::Integer(30));

        assert_eq!(result.rows[1].values[0], Value::Integer(2));
        assert_eq!(result.rows[1].values[1], Value::Text("Bob".into()));
        assert_eq!(result.rows[1].values[2], Value::Integer(25));

        assert_eq!(result.rows[2].values[0], Value::Integer(3));
        assert_eq!(result.rows[2].values[1], Value::Text("Charlie".into()));
        assert_eq!(result.rows[2].values[2], Value::Integer(35));
    }

    // Test reading with a WHERE clause.
    #[test]
    fn test_select_with_where_from_real_db() {
        use std::process::Command;

        let check = Command::new("sqlite3").arg("--version").output();
        if check.is_err() || !check.unwrap().status.success() {
            eprintln!("sqlite3 not found, skipping integration test");
            return;
        }

        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_where.db");

        let sql = "\
            CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT, price REAL);\n\
            INSERT INTO items VALUES (1, 'Apple', 1.50);\n\
            INSERT INTO items VALUES (2, 'Banana', 0.75);\n\
            INSERT INTO items VALUES (3, 'Cherry', 3.00);\n\
            INSERT INTO items VALUES (4, 'Date', 5.00);\n\
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
            eprintln!("failed to create test database, skipping");
            return;
        }

        let mut db = Database::open(&db_path).unwrap();
        let entries = db.schema().unwrap();
        let table_schemas = build_table_schemas(&entries).unwrap();

        // SELECT * FROM items WHERE price > 2.0
        let select = crate::ast::SelectStatement {
            distinct: false,
            columns: vec![crate::ast::ResultColumn::AllColumns],
            from: Some(crate::ast::FromClause {
                table: crate::ast::TableRef::Table {
                    name: "items".into(),
                    alias: None,
                },
                joins: vec![],
            }),
            where_clause: Some(crate::ast::Expr::BinaryOp {
                left: Box::new(crate::ast::Expr::ColumnRef {
                    table: None,
                    column: "price".into(),
                }),
                op: crate::ast::BinaryOp::Gt,
                right: Box::new(crate::ast::Expr::Literal(
                    crate::ast::LiteralValue::Real(2.0),
                )),
            }),
            group_by: None,
            having: None,
            order_by: None,
            limit: None,
        };

        let result =
            vm::execute_select(&mut db.pager, &select, &table_schemas).unwrap();

        assert_eq!(result.rows.len(), 2);
        // Cherry (3.00) and Date (5.00) should match.
        assert_eq!(result.rows[0].values[1], Value::Text("Cherry".into()));
        assert_eq!(result.rows[1].values[1], Value::Text("Date".into()));
    }

    // End-to-end test: parse SQL → plan → execute against real SQLite DB.
    #[test]
    fn test_end_to_end_sql_execution() {
        use std::process::Command;

        let check = Command::new("sqlite3").arg("--version").output();
        if check.is_err() || !check.unwrap().status.success() {
            eprintln!("sqlite3 not found, skipping integration test");
            return;
        }

        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_e2e.db");

        let sql = "\
            CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price REAL, qty INTEGER);\n\
            INSERT INTO products VALUES (1, 'Widget', 9.99, 100);\n\
            INSERT INTO products VALUES (2, 'Gadget', 24.99, 50);\n\
            INSERT INTO products VALUES (3, 'Doohickey', 4.99, 200);\n\
            INSERT INTO products VALUES (4, 'Thingamajig', 14.99, 75);\n\
            INSERT INTO products VALUES (5, 'Whatsit', 39.99, 10);\n\
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
            eprintln!("failed to create test database, skipping");
            return;
        }

        let mut db = Database::open(&db_path).unwrap();

        // Test 1: SELECT * FROM products
        let result = db.execute("SELECT * FROM products").unwrap();
        assert_eq!(result.rows.len(), 5);
        assert_eq!(result.columns.len(), 4);

        // Test 2: SELECT with WHERE
        let result = db
            .execute("SELECT name, price FROM products WHERE price > 10.0")
            .unwrap();
        assert_eq!(result.rows.len(), 3); // Gadget, Thingamajig, Whatsit
        assert_eq!(result.columns.len(), 2);
        assert_eq!(result.columns[0].name, "name");
        assert_eq!(result.columns[1].name, "price");

        // Test 3: SELECT with ORDER BY
        let result = db
            .execute("SELECT name, price FROM products ORDER BY price DESC")
            .unwrap();
        assert_eq!(result.rows.len(), 5);
        assert_eq!(result.rows[0].values[0], Value::Text("Whatsit".into()));
        assert_eq!(result.rows[4].values[0], Value::Text("Doohickey".into()));

        // Test 4: SELECT with LIMIT
        let result = db
            .execute("SELECT name FROM products ORDER BY price LIMIT 2")
            .unwrap();
        assert_eq!(result.rows.len(), 2);
        assert_eq!(result.rows[0].values[0], Value::Text("Doohickey".into()));
        assert_eq!(result.rows[1].values[0], Value::Text("Widget".into()));

        // Test 5: SELECT with expression
        let result = db
            .execute("SELECT name, price * qty FROM products WHERE id = 1")
            .unwrap();
        assert_eq!(result.rows.len(), 1);
        // 9.99 * 100 = 999.0
        if let Value::Real(v) = &result.rows[0].values[1] {
            assert!((v - 999.0).abs() < 0.01);
        } else if let Value::Integer(v) = &result.rows[0].values[1] {
            assert_eq!(*v, 999);
        }

        // Test 6: SELECT with LIKE
        let result = db
            .execute("SELECT name FROM products WHERE name LIKE '%a%'")
            .unwrap();
        assert!(result.rows.len() >= 2); // Gadget, Thingamajig, Whatsit (all have 'a')

        // Test 7: SELECT with function
        let result = db
            .execute("SELECT upper(name) FROM products WHERE id = 1")
            .unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0].values[0], Value::Text("WIDGET".into()));
    }
}
