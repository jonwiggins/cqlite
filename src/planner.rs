// Query planner: translates AST into execution plans.
//
// This module provides the `Database` struct which is the main entry point
// for executing SQL statements. It bridges:
//   - Parser: SQL text → AST
//   - Schema: reads sqlite_master to discover tables/indexes
//   - VM: executes SELECT queries against B-tree storage
//
// Column names are extracted from CREATE TABLE SQL in the schema entries.

use crate::ast::{Expr, InsertSource, LiteralValue, Statement};
use crate::btree;
use crate::error::{Result, RsqliteError};
use crate::pager::Pager;
use crate::record;
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
            Statement::CreateTable(ct) => self.execute_create_table(ct),
            Statement::Insert(insert) => self.execute_insert(insert),
            Statement::Delete(delete) => self.execute_delete(delete),
            Statement::Update(update) => self.execute_update(update),
            Statement::DropTable(drop) => self.execute_drop_table(drop),
            Statement::Explain(inner) => {
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

    /// Execute a CREATE TABLE statement.
    fn execute_create_table(
        &mut self,
        ct: &crate::ast::CreateTableStatement,
    ) -> Result<QueryResult> {
        // Check if the table already exists.
        let schema_entries = schema::read_schema(&mut self.pager)?;
        if schema::find_table(&schema_entries, &ct.name).is_some() {
            if ct.if_not_exists {
                return Ok(empty_result());
            }
            return Err(RsqliteError::Runtime(format!(
                "table {} already exists",
                ct.name
            )));
        }

        // Allocate a new page for the table's root B-tree.
        let root_page = self.pager.allocate_page()?;
        btree::init_table_leaf_page(&mut self.pager, root_page)?;

        // Reconstruct the CREATE TABLE SQL from the AST.
        let sql = reconstruct_create_table_sql(ct);

        // Insert a row into sqlite_master (the table B-tree on page 1).
        // sqlite_master columns: type, name, tbl_name, rootpage, sql
        let master_values = vec![
            Value::Text("table".into()),
            Value::Text(ct.name.clone()),
            Value::Text(ct.name.clone()),
            Value::Integer(root_page as i64),
            Value::Text(sql),
        ];
        let master_payload = record::encode_record(&master_values);

        // Find the next rowid for sqlite_master.
        let master_rowid = btree::find_max_rowid(&mut self.pager, 1)? + 1;

        // Insert into page 1 (sqlite_master).
        let new_root = btree::btree_insert(&mut self.pager, 1, master_rowid, &master_payload)?;
        if new_root != 1 {
            // sqlite_master's root must stay on page 1. This would be a serious issue.
            return Err(RsqliteError::Runtime(
                "sqlite_master root page split is not yet supported".into(),
            ));
        }

        // Increment schema cookie.
        self.pager.header.schema_cookie += 1;

        // Flush to disk.
        self.pager.flush()?;

        Ok(empty_result())
    }

    /// Execute an INSERT statement.
    fn execute_insert(
        &mut self,
        insert: &crate::ast::InsertStatement,
    ) -> Result<QueryResult> {
        // Look up the table schema.
        let schema_entries = schema::read_schema(&mut self.pager)?;
        let table_schemas = build_table_schemas(&schema_entries)?;
        let schema = table_schemas
            .iter()
            .find(|t| t.name.eq_ignore_ascii_case(&insert.table))
            .ok_or_else(|| {
                RsqliteError::Runtime(format!("no such table: {}", insert.table))
            })?
            .clone();

        let rows = match &insert.source {
            InsertSource::Values(row_list) => {
                let mut result_rows = Vec::new();
                for value_exprs in row_list {
                    let values = eval_insert_values(value_exprs)?;
                    result_rows.push(values);
                }
                result_rows
            }
            InsertSource::DefaultValues => {
                // Single row of all defaults (NULLs).
                let values: Vec<Value> = schema.columns.iter().map(|_| Value::Null).collect();
                vec![values]
            }
            InsertSource::Select(_) => {
                return Err(RsqliteError::NotImplemented(
                    "INSERT ... SELECT".into(),
                ));
            }
        };

        let mut changes = 0i64;
        let mut root_page = schema.root_page;

        for mut values in rows {
            // Map columns if specified.
            if let Some(ref target_cols) = insert.columns {
                let mut full_values = vec![Value::Null; schema.columns.len()];
                for (i, col_name) in target_cols.iter().enumerate() {
                    if let Some(col_idx) = schema
                        .columns
                        .iter()
                        .position(|c| c.eq_ignore_ascii_case(col_name))
                    {
                        if i < values.len() {
                            full_values[col_idx] = values[i].clone();
                        }
                    } else {
                        return Err(RsqliteError::Runtime(format!(
                            "table {} has no column named {}",
                            insert.table, col_name
                        )));
                    }
                }
                values = full_values;
            }

            // Pad or truncate to match column count.
            values.resize(schema.columns.len(), Value::Null);

            // Determine rowid.
            let rowid = if let Some(pk_idx) = schema.rowid_column {
                match &values[pk_idx] {
                    Value::Integer(id) => {
                        // Use the provided INTEGER PRIMARY KEY value as rowid.
                        let id = *id;
                        // Store NULL in the record for the IPK column.
                        values[pk_idx] = Value::Null;
                        id
                    }
                    Value::Null => {
                        // Auto-assign rowid.
                        let id = btree::find_max_rowid(&mut self.pager, root_page)? + 1;
                        // Keep NULL in the record.
                        id
                    }
                    _ => {
                        // Try to convert to integer.
                        let id = btree::find_max_rowid(&mut self.pager, root_page)? + 1;
                        values[pk_idx] = Value::Null;
                        id
                    }
                }
            } else {
                btree::find_max_rowid(&mut self.pager, root_page)? + 1
            };

            let payload = record::encode_record(&values);
            let new_root = btree::btree_insert(&mut self.pager, root_page, rowid, &payload)?;

            // If the root changed due to a split, update the schema.
            if new_root != root_page {
                update_root_page(&mut self.pager, &schema.name, new_root)?;
                root_page = new_root;
            }

            changes += 1;
        }

        self.pager.flush()?;

        Ok(QueryResult {
            columns: vec![],
            rows: vec![Row {
                values: vec![Value::Integer(changes)],
            }],
        })
    }

    /// Execute a DELETE statement.
    fn execute_delete(
        &mut self,
        delete: &crate::ast::DeleteStatement,
    ) -> Result<QueryResult> {
        let schema_entries = schema::read_schema(&mut self.pager)?;
        let table_schemas = build_table_schemas(&schema_entries)?;
        let schema = table_schemas
            .iter()
            .find(|t| t.name.eq_ignore_ascii_case(&delete.table))
            .ok_or_else(|| {
                RsqliteError::Runtime(format!("no such table: {}", delete.table))
            })?
            .clone();

        // Scan to find rows matching the WHERE clause.
        let column_names: Vec<&str> = schema.columns.iter().map(|c| c.as_str()).collect();
        let table_name = Some(schema.name.as_str());

        let mut to_delete: Vec<i64> = Vec::new();
        let mut cursor = btree::BTreeCursor::new(schema.root_page);
        cursor.move_to_first(&mut self.pager)?;

        while cursor.is_valid() {
            let rowid = cursor.current_rowid(&mut self.pager)?;
            let payload = cursor.current_payload(&mut self.pager)?;
            let mut values = record::decode_record(&payload)?;

            if let Some(pk_idx) = schema.rowid_column {
                if pk_idx < values.len() {
                    values[pk_idx] = Value::Integer(rowid);
                }
            }

            let should_delete = if let Some(ref where_expr) = delete.where_clause {
                let result = vm::eval_expr(where_expr, &column_names, &values, rowid, table_name)?;
                result.is_truthy()
            } else {
                true
            };

            if should_delete {
                to_delete.push(rowid);
            }
            cursor.move_to_next(&mut self.pager)?;
        }

        let changes = to_delete.len() as i64;
        for rowid in to_delete {
            btree::btree_delete(&mut self.pager, schema.root_page, rowid)?;
        }

        self.pager.flush()?;

        Ok(QueryResult {
            columns: vec![],
            rows: vec![Row {
                values: vec![Value::Integer(changes)],
            }],
        })
    }

    /// Execute an UPDATE statement.
    fn execute_update(
        &mut self,
        update: &crate::ast::UpdateStatement,
    ) -> Result<QueryResult> {
        let schema_entries = schema::read_schema(&mut self.pager)?;
        let table_schemas = build_table_schemas(&schema_entries)?;
        let schema = table_schemas
            .iter()
            .find(|t| t.name.eq_ignore_ascii_case(&update.table))
            .ok_or_else(|| {
                RsqliteError::Runtime(format!("no such table: {}", update.table))
            })?
            .clone();

        let column_names: Vec<&str> = schema.columns.iter().map(|c| c.as_str()).collect();
        let table_name = Some(schema.name.as_str());

        // Scan all rows and collect those that match.
        let mut updates: Vec<(i64, Vec<Value>)> = Vec::new();
        let mut cursor = btree::BTreeCursor::new(schema.root_page);
        cursor.move_to_first(&mut self.pager)?;

        while cursor.is_valid() {
            let rowid = cursor.current_rowid(&mut self.pager)?;
            let payload = cursor.current_payload(&mut self.pager)?;
            let mut values = record::decode_record(&payload)?;

            if let Some(pk_idx) = schema.rowid_column {
                if pk_idx < values.len() {
                    values[pk_idx] = Value::Integer(rowid);
                }
            }

            let should_update = if let Some(ref where_expr) = update.where_clause {
                let result = vm::eval_expr(where_expr, &column_names, &values, rowid, table_name)?;
                result.is_truthy()
            } else {
                true
            };

            if should_update {
                // Apply assignments.
                for assignment in &update.assignments {
                    if let Some(col_idx) = schema
                        .columns
                        .iter()
                        .position(|c| c.eq_ignore_ascii_case(&assignment.column))
                    {
                        let new_val = vm::eval_expr(
                            &assignment.value,
                            &column_names,
                            &values,
                            rowid,
                            table_name,
                        )?;
                        values[col_idx] = new_val;
                    }
                }
                updates.push((rowid, values));
            }
            cursor.move_to_next(&mut self.pager)?;
        }

        let changes = updates.len() as i64;
        let mut root_page = schema.root_page;

        for (rowid, mut values) in updates {
            // Delete old row.
            btree::btree_delete(&mut self.pager, root_page, rowid)?;

            // For INTEGER PRIMARY KEY columns, store NULL in the record.
            if let Some(pk_idx) = schema.rowid_column {
                if pk_idx < values.len() {
                    values[pk_idx] = Value::Null;
                }
            }

            // Re-insert with same rowid.
            let payload = record::encode_record(&values);
            let new_root = btree::btree_insert(&mut self.pager, root_page, rowid, &payload)?;
            if new_root != root_page {
                update_root_page(&mut self.pager, &schema.name, new_root)?;
                root_page = new_root;
            }
        }

        self.pager.flush()?;

        Ok(QueryResult {
            columns: vec![],
            rows: vec![Row {
                values: vec![Value::Integer(changes)],
            }],
        })
    }

    /// Execute a DROP TABLE statement.
    fn execute_drop_table(
        &mut self,
        drop: &crate::ast::DropTableStatement,
    ) -> Result<QueryResult> {
        let schema_entries = schema::read_schema(&mut self.pager)?;

        if schema::find_table(&schema_entries, &drop.name).is_none() {
            if drop.if_exists {
                return Ok(empty_result());
            }
            return Err(RsqliteError::Runtime(format!(
                "no such table: {}",
                drop.name
            )));
        }

        // Find and delete the sqlite_master entry.
        let mut cursor = btree::BTreeCursor::new(1);
        cursor.move_to_first(&mut self.pager)?;

        let mut rowid_to_delete = None;
        while cursor.is_valid() {
            let rowid = cursor.current_rowid(&mut self.pager)?;
            let payload = cursor.current_payload(&mut self.pager)?;
            let values = record::decode_record(&payload)?;
            if values.len() >= 2 {
                if let Value::Text(ref name) = values[1] {
                    if name.eq_ignore_ascii_case(&drop.name) {
                        rowid_to_delete = Some(rowid);
                        break;
                    }
                }
            }
            cursor.move_to_next(&mut self.pager)?;
        }

        if let Some(rowid) = rowid_to_delete {
            btree::btree_delete(&mut self.pager, 1, rowid)?;
        }

        self.pager.header.schema_cookie += 1;
        self.pager.flush()?;

        Ok(empty_result())
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

/// Create an empty QueryResult (for DDL statements that don't return rows).
fn empty_result() -> QueryResult {
    QueryResult {
        columns: vec![],
        rows: vec![],
    }
}

/// Evaluate a list of expressions into values for INSERT.
fn eval_insert_values(exprs: &[Expr]) -> Result<Vec<Value>> {
    let mut values = Vec::with_capacity(exprs.len());
    for expr in exprs {
        let value = eval_constant_expr(expr)?;
        values.push(value);
    }
    Ok(values)
}

/// Evaluate a constant expression (no column references).
fn eval_constant_expr(expr: &Expr) -> Result<Value> {
    match expr {
        Expr::Literal(lit) => match lit {
            LiteralValue::Integer(i) => Ok(Value::Integer(*i)),
            LiteralValue::Real(f) => Ok(Value::Real(*f)),
            LiteralValue::String(s) => Ok(Value::Text(s.clone())),
            LiteralValue::Blob(b) => Ok(Value::Blob(b.clone())),
            LiteralValue::Null => Ok(Value::Null),
            _ => Ok(Value::Null),
        },
        Expr::UnaryOp {
            op: crate::ast::UnaryOp::Negate,
            operand,
        } => {
            let val = eval_constant_expr(operand)?;
            match val {
                Value::Integer(i) => Ok(Value::Integer(-i)),
                Value::Real(f) => Ok(Value::Real(-f)),
                _ => Ok(Value::Null),
            }
        }
        _ => {
            // For complex expressions, try evaluating with no column context.
            vm::eval_expr(expr, &[], &[], 0, None)
        }
    }
}

/// Reconstruct CREATE TABLE SQL from the AST.
fn reconstruct_create_table_sql(ct: &crate::ast::CreateTableStatement) -> String {
    let mut sql = String::from("CREATE TABLE ");
    sql.push_str(&ct.name);
    sql.push_str(" (");

    for (i, col) in ct.columns.iter().enumerate() {
        if i > 0 {
            sql.push_str(", ");
        }
        sql.push_str(&col.name);
        if let Some(ref type_name) = col.type_name {
            sql.push(' ');
            sql.push_str(type_name);
        }
        for constraint in &col.constraints {
            match constraint {
                crate::ast::ColumnConstraint::PrimaryKey { autoincrement, .. } => {
                    sql.push_str(" PRIMARY KEY");
                    if *autoincrement {
                        sql.push_str(" AUTOINCREMENT");
                    }
                }
                crate::ast::ColumnConstraint::NotNull => sql.push_str(" NOT NULL"),
                crate::ast::ColumnConstraint::Unique => sql.push_str(" UNIQUE"),
                crate::ast::ColumnConstraint::Default(expr) => {
                    sql.push_str(" DEFAULT ");
                    sql.push_str(&format!("{expr:?}"));
                }
                _ => {}
            }
        }
    }

    // Table constraints.
    for constraint in &ct.constraints {
        sql.push_str(", ");
        match constraint {
            crate::ast::TableConstraint::PrimaryKey(cols) => {
                sql.push_str("PRIMARY KEY (");
                for (i, col) in cols.iter().enumerate() {
                    if i > 0 {
                        sql.push_str(", ");
                    }
                    sql.push_str(&col.name);
                }
                sql.push(')');
            }
            crate::ast::TableConstraint::Unique(cols) => {
                sql.push_str("UNIQUE (");
                for (i, col) in cols.iter().enumerate() {
                    if i > 0 {
                        sql.push_str(", ");
                    }
                    sql.push_str(&col.name);
                }
                sql.push(')');
            }
            _ => {}
        }
    }

    sql.push(')');
    sql
}

/// Update the root page number for a table in sqlite_master.
fn update_root_page(pager: &mut Pager, table_name: &str, new_root: u32) -> Result<()> {
    let mut cursor = btree::BTreeCursor::new(1);
    cursor.move_to_first(pager)?;

    while cursor.is_valid() {
        let rowid = cursor.current_rowid(pager)?;
        let payload = cursor.current_payload(pager)?;
        let mut values = record::decode_record(&payload)?;

        if values.len() >= 4 {
            if let Value::Text(ref name) = values[1] {
                if name.eq_ignore_ascii_case(table_name) {
                    // Update rootpage (column 3, 0-indexed).
                    values[3] = Value::Integer(new_root as i64);
                    let new_payload = record::encode_record(&values);

                    // Delete and re-insert.
                    btree::btree_delete(pager, 1, rowid)?;
                    btree::btree_insert(pager, 1, rowid, &new_payload)?;
                    return Ok(());
                }
            }
        }
        cursor.move_to_next(pager)?;
    }

    Ok(())
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

    // -----------------------------------------------------------------------
    // Write path tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_create_table_in_memory() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
            .unwrap();

        let tables = db.table_names().unwrap();
        assert_eq!(tables, vec!["users"]);

        // Table should be empty.
        let result = db.execute("SELECT * FROM users").unwrap();
        assert_eq!(result.rows.len(), 0);
        assert_eq!(result.columns.len(), 3);
    }

    #[test]
    fn test_create_table_if_not_exists() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE t (a INT)").unwrap();
        // Should not error with IF NOT EXISTS.
        db.execute("CREATE TABLE IF NOT EXISTS t (a INT)").unwrap();
        // Should error without IF NOT EXISTS.
        let result = db.execute("CREATE TABLE t (a INT)");
        assert!(result.is_err());
    }

    #[test]
    fn test_insert_and_select() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
            .unwrap();

        db.execute("INSERT INTO users VALUES (1, 'Alice', 30)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (2, 'Bob', 25)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (3, 'Charlie', 35)")
            .unwrap();

        let result = db.execute("SELECT * FROM users").unwrap();
        assert_eq!(result.rows.len(), 3);
        assert_eq!(result.rows[0].values[0], Value::Integer(1));
        assert_eq!(result.rows[0].values[1], Value::Text("Alice".into()));
        assert_eq!(result.rows[0].values[2], Value::Integer(30));
        assert_eq!(result.rows[1].values[0], Value::Integer(2));
        assert_eq!(result.rows[2].values[0], Value::Integer(3));
    }

    #[test]
    fn test_insert_with_column_names() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE t (a TEXT, b INTEGER, c REAL)")
            .unwrap();

        db.execute("INSERT INTO t (b, a) VALUES (42, 'hello')")
            .unwrap();

        let result = db.execute("SELECT a, b, c FROM t").unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0].values[0], Value::Text("hello".into()));
        assert_eq!(result.rows[0].values[1], Value::Integer(42));
        assert_eq!(result.rows[0].values[2], Value::Null);
    }

    #[test]
    fn test_insert_auto_rowid() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE t (name TEXT, value INTEGER)")
            .unwrap();

        db.execute("INSERT INTO t VALUES ('a', 1)").unwrap();
        db.execute("INSERT INTO t VALUES ('b', 2)").unwrap();

        let result = db.execute("SELECT rowid, name FROM t").unwrap();
        assert_eq!(result.rows.len(), 2);
        // Rowids should be 1 and 2.
        assert_eq!(result.rows[0].values[0], Value::Integer(1));
        assert_eq!(result.rows[1].values[0], Value::Integer(2));
    }

    #[test]
    fn test_insert_multi_row() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE t (x INTEGER)").unwrap();

        db.execute("INSERT INTO t VALUES (10), (20), (30)")
            .unwrap();

        let result = db.execute("SELECT x FROM t").unwrap();
        assert_eq!(result.rows.len(), 3);
        assert_eq!(result.rows[0].values[0], Value::Integer(10));
        assert_eq!(result.rows[1].values[0], Value::Integer(20));
        assert_eq!(result.rows[2].values[0], Value::Integer(30));
    }

    #[test]
    fn test_delete_with_where() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)")
            .unwrap();
        db.execute("INSERT INTO t VALUES (1, 'a')").unwrap();
        db.execute("INSERT INTO t VALUES (2, 'b')").unwrap();
        db.execute("INSERT INTO t VALUES (3, 'c')").unwrap();

        db.execute("DELETE FROM t WHERE id = 2").unwrap();

        let result = db.execute("SELECT * FROM t").unwrap();
        assert_eq!(result.rows.len(), 2);
        assert_eq!(result.rows[0].values[0], Value::Integer(1));
        assert_eq!(result.rows[1].values[0], Value::Integer(3));
    }

    #[test]
    fn test_delete_all() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE t (x INTEGER)").unwrap();
        db.execute("INSERT INTO t VALUES (1)").unwrap();
        db.execute("INSERT INTO t VALUES (2)").unwrap();

        db.execute("DELETE FROM t").unwrap();

        let result = db.execute("SELECT * FROM t").unwrap();
        assert_eq!(result.rows.len(), 0);
    }

    #[test]
    fn test_update_with_where() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT, score INTEGER)")
            .unwrap();
        db.execute("INSERT INTO t VALUES (1, 'Alice', 80)")
            .unwrap();
        db.execute("INSERT INTO t VALUES (2, 'Bob', 90)")
            .unwrap();

        db.execute("UPDATE t SET score = 95 WHERE id = 1")
            .unwrap();

        let result = db.execute("SELECT * FROM t ORDER BY id").unwrap();
        assert_eq!(result.rows[0].values[2], Value::Integer(95));
        assert_eq!(result.rows[1].values[2], Value::Integer(90));
    }

    #[test]
    fn test_update_all() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE t (x INTEGER)").unwrap();
        db.execute("INSERT INTO t VALUES (1)").unwrap();
        db.execute("INSERT INTO t VALUES (2)").unwrap();

        db.execute("UPDATE t SET x = x + 10").unwrap();

        let result = db.execute("SELECT x FROM t ORDER BY x").unwrap();
        assert_eq!(result.rows[0].values[0], Value::Integer(11));
        assert_eq!(result.rows[1].values[0], Value::Integer(12));
    }

    #[test]
    fn test_drop_table() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE t1 (a INT)").unwrap();
        db.execute("CREATE TABLE t2 (b INT)").unwrap();

        let tables = db.table_names().unwrap();
        assert_eq!(tables.len(), 2);

        db.execute("DROP TABLE t1").unwrap();

        let tables = db.table_names().unwrap();
        assert_eq!(tables.len(), 1);
        assert_eq!(tables[0], "t2");
    }

    #[test]
    fn test_drop_table_if_exists() {
        let mut db = Database::in_memory();
        // Should not error.
        db.execute("DROP TABLE IF EXISTS nonexistent").unwrap();
        // Should error without IF EXISTS.
        assert!(db.execute("DROP TABLE nonexistent").is_err());
    }

    #[test]
    fn test_insert_select_aggregate() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE scores (name TEXT, score INTEGER)")
            .unwrap();
        db.execute("INSERT INTO scores VALUES ('Alice', 90)")
            .unwrap();
        db.execute("INSERT INTO scores VALUES ('Bob', 85)")
            .unwrap();
        db.execute("INSERT INTO scores VALUES ('Alice', 95)")
            .unwrap();

        let result = db
            .execute("SELECT name, AVG(score) FROM scores GROUP BY name ORDER BY name")
            .unwrap();
        assert_eq!(result.rows.len(), 2);
        assert_eq!(result.rows[0].values[0], Value::Text("Alice".into()));
        assert_eq!(result.rows[1].values[0], Value::Text("Bob".into()));
    }

    #[test]
    fn test_write_and_read_with_sqlite3() {
        use std::process::Command;

        let check = Command::new("sqlite3").arg("--version").output();
        if check.is_err() || !check.unwrap().status.success() {
            eprintln!("sqlite3 not found, skipping cross-compatibility test");
            return;
        }

        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_write_compat.db");

        // Create and populate using our engine.
        {
            let mut db = Database::open(&db_path).unwrap();
            db.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT, price REAL)")
                .unwrap();
            db.execute("INSERT INTO items VALUES (1, 'Widget', 9.99)")
                .unwrap();
            db.execute("INSERT INTO items VALUES (2, 'Gadget', 24.99)")
                .unwrap();
            db.execute("INSERT INTO items VALUES (3, 'Doohickey', 4.99)")
                .unwrap();
        }

        // Read with sqlite3 and verify.
        let output = Command::new("sqlite3")
            .arg(db_path.to_str().unwrap())
            .arg("SELECT id, name, price FROM items ORDER BY id;")
            .output()
            .unwrap();

        let stdout = String::from_utf8(output.stdout).unwrap();
        let lines: Vec<&str> = stdout.trim().lines().collect();
        assert_eq!(lines.len(), 3, "sqlite3 output: {stdout}");
        assert!(lines[0].contains("Widget"), "expected Widget in: {}", lines[0]);
        assert!(lines[1].contains("Gadget"), "expected Gadget in: {}", lines[1]);
        assert!(lines[2].contains("Doohickey"), "expected Doohickey in: {}", lines[2]);
    }

    #[test]
    fn test_many_inserts_causes_split() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE big (id INTEGER PRIMARY KEY, data TEXT)")
            .unwrap();

        // Insert enough rows that the page should split.
        for i in 1..=100 {
            let sql = format!(
                "INSERT INTO big VALUES ({i}, '{}')",
                "x".repeat(20)
            );
            db.execute(&sql).unwrap();
        }

        let result = db.execute("SELECT COUNT(*) FROM big").unwrap();
        assert_eq!(result.rows[0].values[0], Value::Integer(100));

        // Verify ordering.
        let result = db.execute("SELECT id FROM big ORDER BY id").unwrap();
        assert_eq!(result.rows.len(), 100);
        for (i, row) in result.rows.iter().enumerate() {
            assert_eq!(row.values[0], Value::Integer(i as i64 + 1));
        }
    }

    // -- JOIN tests --

    fn setup_join_db() -> Database {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, dept_id INTEGER)")
            .unwrap();
        db.execute("CREATE TABLE departments (id INTEGER PRIMARY KEY, dept_name TEXT)")
            .unwrap();
        db.execute("INSERT INTO departments VALUES (1, 'Engineering')")
            .unwrap();
        db.execute("INSERT INTO departments VALUES (2, 'Sales')")
            .unwrap();
        db.execute("INSERT INTO departments VALUES (3, 'Marketing')")
            .unwrap();
        db.execute("INSERT INTO users VALUES (1, 'Alice', 1)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (2, 'Bob', 2)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (3, 'Charlie', 1)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (4, 'Diana', NULL)")
            .unwrap();
        db
    }

    #[test]
    fn test_inner_join_on() {
        let mut db = setup_join_db();
        let result = db
            .execute(
                "SELECT users.name, departments.dept_name \
                 FROM users INNER JOIN departments ON users.dept_id = departments.id",
            )
            .unwrap();
        // Diana has NULL dept_id, so she should not appear.
        assert_eq!(result.rows.len(), 3);
        let names: Vec<&Value> = result.rows.iter().map(|r| &r.values[0]).collect();
        assert!(names.contains(&&Value::Text("Alice".into())));
        assert!(names.contains(&&Value::Text("Bob".into())));
        assert!(names.contains(&&Value::Text("Charlie".into())));
    }

    #[test]
    fn test_left_join() {
        let mut db = setup_join_db();
        let result = db
            .execute(
                "SELECT users.name, departments.dept_name \
                 FROM users LEFT JOIN departments ON users.dept_id = departments.id \
                 ORDER BY users.id",
            )
            .unwrap();
        // All 4 users should appear; Diana has NULL for dept_name.
        assert_eq!(result.rows.len(), 4);
        // Diana is last (id=4).
        assert_eq!(result.rows[3].values[0], Value::Text("Diana".into()));
        assert_eq!(result.rows[3].values[1], Value::Null);
        // Alice is first (id=1), department = Engineering.
        assert_eq!(result.rows[0].values[0], Value::Text("Alice".into()));
        assert_eq!(
            result.rows[0].values[1],
            Value::Text("Engineering".into())
        );
    }

    #[test]
    fn test_cross_join() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE a (x INTEGER)").unwrap();
        db.execute("CREATE TABLE b (y INTEGER)").unwrap();
        db.execute("INSERT INTO a VALUES (1)").unwrap();
        db.execute("INSERT INTO a VALUES (2)").unwrap();
        db.execute("INSERT INTO b VALUES (10)").unwrap();
        db.execute("INSERT INTO b VALUES (20)").unwrap();
        db.execute("INSERT INTO b VALUES (30)").unwrap();

        let result = db.execute("SELECT x, y FROM a CROSS JOIN b").unwrap();
        // 2 * 3 = 6 rows.
        assert_eq!(result.rows.len(), 6);
    }

    #[test]
    fn test_join_using() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE t1 (id INTEGER PRIMARY KEY, val TEXT)")
            .unwrap();
        db.execute("CREATE TABLE t2 (id INTEGER PRIMARY KEY, info TEXT)")
            .unwrap();
        db.execute("INSERT INTO t1 VALUES (1, 'a')").unwrap();
        db.execute("INSERT INTO t1 VALUES (2, 'b')").unwrap();
        db.execute("INSERT INTO t2 VALUES (1, 'x')").unwrap();
        db.execute("INSERT INTO t2 VALUES (3, 'z')").unwrap();

        let result = db
            .execute("SELECT id, val, info FROM t1 INNER JOIN t2 USING (id)")
            .unwrap();
        // Only id=1 matches.
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0].values[0], Value::Integer(1));
        assert_eq!(result.rows[0].values[1], Value::Text("a".into()));
        assert_eq!(result.rows[0].values[2], Value::Text("x".into()));
    }

    #[test]
    fn test_join_with_where() {
        let mut db = setup_join_db();
        let result = db
            .execute(
                "SELECT users.name, departments.dept_name \
                 FROM users INNER JOIN departments ON users.dept_id = departments.id \
                 WHERE departments.dept_name = 'Engineering'",
            )
            .unwrap();
        assert_eq!(result.rows.len(), 2);
        let names: Vec<&Value> = result.rows.iter().map(|r| &r.values[0]).collect();
        assert!(names.contains(&&Value::Text("Alice".into())));
        assert!(names.contains(&&Value::Text("Charlie".into())));
    }

    #[test]
    fn test_join_with_aggregate() {
        let mut db = setup_join_db();
        let result = db
            .execute(
                "SELECT departments.dept_name, COUNT(*) \
                 FROM users INNER JOIN departments ON users.dept_id = departments.id \
                 GROUP BY departments.dept_name \
                 ORDER BY departments.dept_name",
            )
            .unwrap();
        // Engineering: 2 (Alice, Charlie), Sales: 1 (Bob)
        assert_eq!(result.rows.len(), 2);
        assert_eq!(
            result.rows[0].values[0],
            Value::Text("Engineering".into())
        );
        assert_eq!(result.rows[0].values[1], Value::Integer(2));
        assert_eq!(result.rows[1].values[0], Value::Text("Sales".into()));
        assert_eq!(result.rows[1].values[1], Value::Integer(1));
    }

    #[test]
    fn test_implicit_join() {
        let mut db = setup_join_db();
        // Implicit join via comma in FROM (treated as CROSS JOIN).
        let result = db
            .execute(
                "SELECT users.name, departments.dept_name \
                 FROM users, departments \
                 WHERE users.dept_id = departments.id \
                 ORDER BY users.id",
            )
            .unwrap();
        assert_eq!(result.rows.len(), 3);
        assert_eq!(result.rows[0].values[0], Value::Text("Alice".into()));
        assert_eq!(
            result.rows[0].values[1],
            Value::Text("Engineering".into())
        );
    }

    #[test]
    fn test_left_join_unmatched_right() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount REAL)")
            .unwrap();
        db.execute("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT)")
            .unwrap();
        db.execute("INSERT INTO customers VALUES (1, 'Alice')").unwrap();
        db.execute("INSERT INTO customers VALUES (2, 'Bob')").unwrap();
        db.execute("INSERT INTO orders VALUES (1, 1, 99.99)").unwrap();

        let result = db
            .execute(
                "SELECT customers.name, orders.amount \
                 FROM customers LEFT JOIN orders ON customers.id = orders.customer_id \
                 ORDER BY customers.id",
            )
            .unwrap();
        assert_eq!(result.rows.len(), 2);
        assert_eq!(result.rows[0].values[0], Value::Text("Alice".into()));
        assert_eq!(result.rows[0].values[1], Value::Real(99.99));
        assert_eq!(result.rows[1].values[0], Value::Text("Bob".into()));
        assert_eq!(result.rows[1].values[1], Value::Null);
    }

    #[test]
    fn test_self_join() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, manager_id INTEGER)")
            .unwrap();
        db.execute("INSERT INTO employees VALUES (1, 'Boss', NULL)")
            .unwrap();
        db.execute("INSERT INTO employees VALUES (2, 'Alice', 1)")
            .unwrap();
        db.execute("INSERT INTO employees VALUES (3, 'Bob', 1)")
            .unwrap();

        let result = db
            .execute(
                "SELECT e.name, m.name \
                 FROM employees AS e INNER JOIN employees AS m ON e.manager_id = m.id",
            )
            .unwrap();
        // Boss has no manager (NULL), so only Alice and Bob appear.
        assert_eq!(result.rows.len(), 2);
    }

    // -- Scalar function tests --

    #[test]
    fn test_scalar_functions_basic() {
        let mut db = Database::in_memory();

        // substr
        let r = db.execute("SELECT substr('hello world', 7)").unwrap();
        assert_eq!(r.rows[0].values[0], Value::Text("world".into()));

        let r = db.execute("SELECT substr('hello', 2, 3)").unwrap();
        assert_eq!(r.rows[0].values[0], Value::Text("ell".into()));

        // negative start (from end)
        let r = db.execute("SELECT substr('hello', -3)").unwrap();
        assert_eq!(r.rows[0].values[0], Value::Text("llo".into()));

        // trim
        let r = db.execute("SELECT trim('  hello  ')").unwrap();
        assert_eq!(r.rows[0].values[0], Value::Text("hello".into()));

        let r = db.execute("SELECT ltrim('  hello  ')").unwrap();
        assert_eq!(r.rows[0].values[0], Value::Text("hello  ".into()));

        let r = db.execute("SELECT rtrim('  hello  ')").unwrap();
        assert_eq!(r.rows[0].values[0], Value::Text("  hello".into()));

        // replace
        let r = db
            .execute("SELECT replace('hello world', 'world', 'rust')")
            .unwrap();
        assert_eq!(r.rows[0].values[0], Value::Text("hello rust".into()));

        // instr
        let r = db.execute("SELECT instr('hello world', 'world')").unwrap();
        assert_eq!(r.rows[0].values[0], Value::Integer(7));

        let r = db.execute("SELECT instr('hello', 'xyz')").unwrap();
        assert_eq!(r.rows[0].values[0], Value::Integer(0));
    }

    #[test]
    fn test_scalar_functions_hex_quote_round() {
        let mut db = Database::in_memory();

        // hex
        let r = db.execute("SELECT hex('ABC')").unwrap();
        assert_eq!(r.rows[0].values[0], Value::Text("414243".into()));

        // quote
        let r = db.execute("SELECT quote('hello')").unwrap();
        assert_eq!(r.rows[0].values[0], Value::Text("'hello'".into()));

        let r = db.execute("SELECT quote(NULL)").unwrap();
        assert_eq!(r.rows[0].values[0], Value::Text("NULL".into()));

        let r = db.execute("SELECT quote(42)").unwrap();
        assert_eq!(r.rows[0].values[0], Value::Text("42".into()));

        // round
        let r = db.execute("SELECT round(3.14159, 2)").unwrap();
        assert_eq!(r.rows[0].values[0], Value::Real(3.14));

        let r = db.execute("SELECT round(3.5)").unwrap();
        assert_eq!(r.rows[0].values[0], Value::Real(4.0));

        // unicode
        let r = db.execute("SELECT unicode('A')").unwrap();
        assert_eq!(r.rows[0].values[0], Value::Integer(65));

        // char
        let r = db.execute("SELECT char(65, 66, 67)").unwrap();
        assert_eq!(r.rows[0].values[0], Value::Text("ABC".into()));
    }

    #[test]
    fn test_scalar_functions_with_table_data() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE t (name TEXT, val INTEGER)").unwrap();
        db.execute("INSERT INTO t VALUES ('Hello World', 42)").unwrap();
        db.execute("INSERT INTO t VALUES ('  spaces  ', -7)").unwrap();

        let r = db
            .execute("SELECT upper(name), abs(val) FROM t ORDER BY rowid")
            .unwrap();
        assert_eq!(r.rows[0].values[0], Value::Text("HELLO WORLD".into()));
        assert_eq!(r.rows[0].values[1], Value::Integer(42));
        assert_eq!(r.rows[1].values[0], Value::Text("  SPACES  ".into()));
        assert_eq!(r.rows[1].values[1], Value::Integer(7));

        let r = db
            .execute("SELECT substr(name, 1, 5), trim(name) FROM t ORDER BY rowid")
            .unwrap();
        assert_eq!(r.rows[0].values[0], Value::Text("Hello".into()));
        assert_eq!(r.rows[1].values[1], Value::Text("spaces".into()));

        let r = db
            .execute("SELECT replace(name, ' ', '_') FROM t ORDER BY rowid")
            .unwrap();
        assert_eq!(r.rows[0].values[0], Value::Text("Hello_World".into()));
    }

    #[test]
    fn test_printf() {
        let mut db = Database::in_memory();
        let r = db
            .execute("SELECT printf('Hello %s, you are %d', 'World', 42)")
            .unwrap();
        assert_eq!(
            r.rows[0].values[0],
            Value::Text("Hello World, you are 42".into())
        );
    }

    // -- DISTINCT test --

    #[test]
    fn test_select_distinct() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE t (category TEXT, value INTEGER)")
            .unwrap();
        db.execute("INSERT INTO t VALUES ('a', 1)").unwrap();
        db.execute("INSERT INTO t VALUES ('b', 2)").unwrap();
        db.execute("INSERT INTO t VALUES ('a', 1)").unwrap();
        db.execute("INSERT INTO t VALUES ('b', 3)").unwrap();
        db.execute("INSERT INTO t VALUES ('a', 1)").unwrap();

        // DISTINCT should remove duplicates.
        let r = db
            .execute("SELECT DISTINCT category, value FROM t ORDER BY category, value")
            .unwrap();
        assert_eq!(r.rows.len(), 3); // (a,1), (b,2), (b,3)

        // DISTINCT on single column.
        let r = db
            .execute("SELECT DISTINCT category FROM t ORDER BY category")
            .unwrap();
        assert_eq!(r.rows.len(), 2); // a, b
        assert_eq!(r.rows[0].values[0], Value::Text("a".into()));
        assert_eq!(r.rows[1].values[0], Value::Text("b".into()));
    }

    // -- CAST test --

    #[test]
    fn test_cast_expressions() {
        let mut db = Database::in_memory();

        let r = db.execute("SELECT CAST('42' AS INTEGER)").unwrap();
        assert_eq!(r.rows[0].values[0], Value::Integer(42));

        let r = db.execute("SELECT CAST(42 AS TEXT)").unwrap();
        assert_eq!(r.rows[0].values[0], Value::Text("42".into()));

        let r = db.execute("SELECT CAST(3.14 AS INTEGER)").unwrap();
        assert_eq!(r.rows[0].values[0], Value::Integer(3));

        let r = db.execute("SELECT CAST('3.14' AS REAL)").unwrap();
        assert_eq!(r.rows[0].values[0], Value::Real(3.14));
    }

    // -- Subquery tests --

    #[test]
    fn test_in_subquery() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE t1 (id INTEGER PRIMARY KEY, name TEXT)")
            .unwrap();
        db.execute("CREATE TABLE t2 (id INTEGER PRIMARY KEY, t1_id INTEGER)")
            .unwrap();
        db.execute("INSERT INTO t1 VALUES (1, 'Alice')").unwrap();
        db.execute("INSERT INTO t1 VALUES (2, 'Bob')").unwrap();
        db.execute("INSERT INTO t1 VALUES (3, 'Charlie')").unwrap();
        db.execute("INSERT INTO t2 VALUES (1, 1)").unwrap();
        db.execute("INSERT INTO t2 VALUES (2, 3)").unwrap();

        // Select names where id is in the subquery.
        let r = db
            .execute("SELECT name FROM t1 WHERE id IN (SELECT t1_id FROM t2)")
            .unwrap();
        assert_eq!(r.rows.len(), 2);
        let names: Vec<&Value> = r.rows.iter().map(|r| &r.values[0]).collect();
        assert!(names.contains(&&Value::Text("Alice".into())));
        assert!(names.contains(&&Value::Text("Charlie".into())));
    }

    #[test]
    fn test_not_in_subquery() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE t1 (id INTEGER PRIMARY KEY, name TEXT)")
            .unwrap();
        db.execute("CREATE TABLE t2 (id INTEGER PRIMARY KEY, t1_id INTEGER)")
            .unwrap();
        db.execute("INSERT INTO t1 VALUES (1, 'Alice')").unwrap();
        db.execute("INSERT INTO t1 VALUES (2, 'Bob')").unwrap();
        db.execute("INSERT INTO t1 VALUES (3, 'Charlie')").unwrap();
        db.execute("INSERT INTO t2 VALUES (1, 1)").unwrap();
        db.execute("INSERT INTO t2 VALUES (2, 3)").unwrap();

        let r = db
            .execute("SELECT name FROM t1 WHERE id NOT IN (SELECT t1_id FROM t2)")
            .unwrap();
        assert_eq!(r.rows.len(), 1);
        assert_eq!(r.rows[0].values[0], Value::Text("Bob".into()));
    }

    #[test]
    fn test_exists_subquery() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, customer TEXT, amount REAL)")
            .unwrap();
        db.execute("INSERT INTO orders VALUES (1, 'Alice', 100.0)")
            .unwrap();
        db.execute("INSERT INTO orders VALUES (2, 'Bob', 50.0)")
            .unwrap();

        // EXISTS returns true when subquery has rows.
        let r = db
            .execute("SELECT 1 WHERE EXISTS (SELECT * FROM orders WHERE amount > 75.0)")
            .unwrap();
        assert_eq!(r.rows.len(), 1);

        // NOT EXISTS.
        let r = db
            .execute("SELECT 1 WHERE NOT EXISTS (SELECT * FROM orders WHERE amount > 200.0)")
            .unwrap();
        assert_eq!(r.rows.len(), 1);
    }

    #[test]
    fn test_scalar_subquery() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE t (val INTEGER)").unwrap();
        db.execute("INSERT INTO t VALUES (10)").unwrap();
        db.execute("INSERT INTO t VALUES (20)").unwrap();
        db.execute("INSERT INTO t VALUES (30)").unwrap();

        let r = db
            .execute("SELECT (SELECT MAX(val) FROM t)")
            .unwrap();
        assert_eq!(r.rows[0].values[0], Value::Integer(30));
    }

    // -- PRAGMA tests --

    #[test]
    fn test_pragma_table_info() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
            .unwrap();

        let r = db.execute("PRAGMA table_info(t)").unwrap();
        assert_eq!(r.rows.len(), 3);
        assert_eq!(r.rows[0].values[1], Value::Text("id".into()));
        assert_eq!(r.rows[1].values[1], Value::Text("name".into()));
        assert_eq!(r.rows[2].values[1], Value::Text("age".into()));
    }

    #[test]
    fn test_pragma_page_size() {
        let mut db = Database::in_memory();
        let r = db.execute("PRAGMA page_size").unwrap();
        assert_eq!(r.rows[0].values[0], Value::Integer(4096));
    }
}
