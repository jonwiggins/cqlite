// Query planner: translates AST into execution plans.
//
// This module provides the `Database` struct which is the main entry point
// for executing SQL statements. It bridges:
//   - Parser: SQL text → AST
//   - Schema: reads sqlite_master to discover tables/indexes
//   - VM: executes SELECT queries against B-tree storage
//
// Column names are extracted from CREATE TABLE SQL in the schema entries.

use crate::ast::{BinaryOp, Expr, FunctionArgs, InsertSource, LiteralValue, Statement};
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
    /// Whether we are inside an explicit transaction (BEGIN was called).
    in_transaction: bool,
}

impl Database {
    /// Open an existing database file (or create a new one).
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let pager = Pager::open(path)?;
        Ok(Self {
            pager,
            in_transaction: false,
        })
    }

    /// Create an in-memory database.
    pub fn in_memory() -> Self {
        Self {
            pager: Pager::in_memory(),
            in_transaction: false,
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
            Statement::Begin(_) => self.execute_begin(),
            Statement::Commit => self.execute_commit(),
            Statement::Rollback => self.execute_rollback(),
            Statement::CreateTable(ct) => self.execute_write(|db| db.execute_create_table(ct)),
            Statement::Insert(insert) => self.execute_write(|db| db.execute_insert(insert)),
            Statement::Delete(delete) => self.execute_write(|db| db.execute_delete(delete)),
            Statement::Update(update) => self.execute_write(|db| db.execute_update(update)),
            Statement::DropTable(drop) => self.execute_write(|db| db.execute_drop_table(drop)),
            Statement::CreateIndex(ci) => self.execute_write(|db| db.execute_create_index(ci)),
            Statement::DropIndex(di) => self.execute_write(|db| db.execute_drop_index(di)),
            Statement::AlterTable(alter) => self.execute_write(|db| db.execute_alter_table(alter)),
            Statement::Explain(inner) => explain_statement(inner),
            Statement::ExplainQueryPlan(inner) => {
                let schema_entries = schema::read_schema(&mut self.pager)?;
                let table_schemas = build_table_schemas(&schema_entries)?;
                explain_query_plan(inner, &table_schemas)
            }
            Statement::Pragma(pragma) => execute_pragma(pragma, &mut self.pager),
        }
    }

    /// Execute a write operation with auto-commit semantics.
    /// If not in an explicit transaction, wraps the operation in an implicit one.
    fn execute_write<F>(&mut self, f: F) -> Result<QueryResult>
    where
        F: FnOnce(&mut Self) -> Result<QueryResult>,
    {
        if self.in_transaction {
            // Already in explicit transaction — just execute.
            f(self)
        } else {
            // Auto-commit: begin, execute, commit (or rollback on error).
            let pc = self.pager.header.page_count;
            self.pager.journal.begin_with_page_count(pc)?;
            match f(self) {
                Ok(result) => {
                    self.pager.journal.commit()?;
                    Ok(result)
                }
                Err(e) => {
                    self.pager.rollback()?;
                    Err(e)
                }
            }
        }
    }

    /// Execute BEGIN.
    fn execute_begin(&mut self) -> Result<QueryResult> {
        if self.in_transaction {
            return Err(RsqliteError::Runtime(
                "cannot start a transaction within a transaction".into(),
            ));
        }
        let pc = self.pager.header.page_count;
        self.pager.journal.begin_with_page_count(pc)?;
        self.in_transaction = true;
        Ok(empty_result())
    }

    /// Execute COMMIT.
    fn execute_commit(&mut self) -> Result<QueryResult> {
        if !self.in_transaction {
            return Err(RsqliteError::Runtime(
                "cannot commit - no transaction is active".into(),
            ));
        }
        self.pager.flush()?;
        self.pager.journal.commit()?;
        self.in_transaction = false;
        Ok(empty_result())
    }

    /// Execute ROLLBACK.
    fn execute_rollback(&mut self) -> Result<QueryResult> {
        if !self.in_transaction {
            return Err(RsqliteError::Runtime(
                "cannot rollback - no transaction is active".into(),
            ));
        }
        self.pager.rollback()?;
        self.in_transaction = false;
        Ok(empty_result())
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
    fn execute_insert(&mut self, insert: &crate::ast::InsertStatement) -> Result<QueryResult> {
        // Look up the table schema.
        let schema_entries = schema::read_schema(&mut self.pager)?;
        let table_schemas = build_table_schemas(&schema_entries)?;
        let schema = table_schemas
            .iter()
            .find(|t| t.name.eq_ignore_ascii_case(&insert.table))
            .ok_or_else(|| RsqliteError::Runtime(format!("no such table: {}", insert.table)))?
            .clone();

        // Find indexes on this table.
        let mut indexes = build_index_schemas(&schema_entries, &schema.name, &schema.columns);

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
            InsertSource::Select(select) => {
                let result = vm::execute_select(&mut self.pager, select, &table_schemas)?;
                result.rows.into_iter().map(|row| row.values).collect()
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

            // Apply DEFAULT values for NULL columns and enforce NOT NULL.
            for (i, val) in values.iter_mut().enumerate() {
                if let Some(cc) = schema.column_constraints.get(i) {
                    if matches!(val, Value::Null) {
                        if let Some(ref default_val) = cc.default_value {
                            *val = default_val.clone();
                        }
                    }
                }
            }

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

                        // Keep NULL in the record.
                        btree::find_max_rowid(&mut self.pager, root_page)? + 1
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

            // Enforce NOT NULL constraints (skip the IPK column which stores NULL in record).
            for (i, val) in values.iter().enumerate() {
                if schema.rowid_column == Some(i) {
                    continue; // IPK stores NULL in record, actual value is rowid.
                }
                if let Some(cc) = schema.column_constraints.get(i) {
                    if cc.not_null && matches!(val, Value::Null) {
                        return Err(RsqliteError::Constraint(format!(
                            "NOT NULL constraint failed: {}.{}",
                            schema.name, schema.columns[i]
                        )));
                    }
                }
            }

            // Enforce CHECK constraints.
            if !schema.check_constraints.is_empty() {
                // Build the full row with IPK restored for expression evaluation.
                let mut check_values = values.clone();
                if let Some(pk_idx) = schema.rowid_column {
                    check_values[pk_idx] = Value::Integer(rowid);
                }
                let col_names: Vec<&str> = schema.columns.iter().map(|s| s.as_str()).collect();
                for check_expr in &schema.check_constraints {
                    let result = vm::eval_expr(
                        check_expr,
                        &col_names,
                        &check_values,
                        rowid,
                        Some(&schema.name),
                    )?;
                    if let Value::Integer(0) = result {
                        return Err(RsqliteError::Constraint(
                            "CHECK constraint failed".to_string(),
                        ));
                    }
                }
            }

            // Handle conflict resolution for UNIQUE/PK violations.
            let conflict = insert.or_conflict;

            // Check for rowid (PRIMARY KEY) conflict.
            {
                let mut cursor = btree::BTreeCursor::new(root_page);
                if cursor.seek_rowid(&mut self.pager, rowid)? {
                    // Rowid already exists — conflict!
                    match conflict {
                        Some(crate::ast::ConflictResolution::Replace) => {
                            // Delete old index entries and the old row.
                            if !indexes.is_empty() {
                                delete_from_indexes(&mut self.pager, &indexes, rowid)?;
                            }
                            btree::btree_delete(&mut self.pager, root_page, rowid)?;
                        }
                        Some(crate::ast::ConflictResolution::Ignore) => {
                            continue; // Skip this row.
                        }
                        _ => {
                            // Abort (default) / Fail / Rollback — error out.
                            return Err(RsqliteError::Constraint(format!(
                                "UNIQUE constraint failed: {}.rowid",
                                schema.name
                            )));
                        }
                    }
                }
            }

            // Check for UNIQUE index conflicts.
            if !indexes.is_empty() {
                let mut full_check_values = values.clone();
                if let Some(pk_idx) = schema.rowid_column {
                    if pk_idx < full_check_values.len() {
                        full_check_values[pk_idx] = Value::Integer(rowid);
                    }
                }

                let mut skip_row = false;
                for index in &indexes {
                    if !index.unique {
                        continue;
                    }
                    let key_values: Vec<Value> = index
                        .column_indices
                        .iter()
                        .map(|&i| full_check_values.get(i).cloned().unwrap_or(Value::Null))
                        .collect();

                    // NULLs don't conflict.
                    if key_values.iter().any(|v| matches!(v, Value::Null)) {
                        continue;
                    }

                    if let Some(conflicting_rowid) =
                        find_unique_conflict(&mut self.pager, index, &key_values)?
                    {
                        match conflict {
                            Some(crate::ast::ConflictResolution::Replace) => {
                                // Delete the conflicting row and its index entries.
                                delete_from_indexes(&mut self.pager, &indexes, conflicting_rowid)?;
                                btree::btree_delete(&mut self.pager, root_page, conflicting_rowid)?;
                            }
                            Some(crate::ast::ConflictResolution::Ignore) => {
                                skip_row = true;
                                break;
                            }
                            _ => {
                                return Err(RsqliteError::Constraint(format!(
                                    "UNIQUE constraint failed: {}",
                                    index.name
                                )));
                            }
                        }
                    }
                }
                if skip_row {
                    continue;
                }
            }

            let payload = record::encode_record(&values);
            let new_root = btree::btree_insert(&mut self.pager, root_page, rowid, &payload)?;

            // If the root changed due to a split, update the schema.
            if new_root != root_page {
                update_root_page(&mut self.pager, &schema.name, new_root)?;
                root_page = new_root;
            }

            // Maintain indexes: insert new entries (no UNIQUE check needed — conflicts already handled).
            if !indexes.is_empty() {
                let mut full_values = values;
                if let Some(pk_idx) = schema.rowid_column {
                    if pk_idx < full_values.len() {
                        full_values[pk_idx] = Value::Integer(rowid);
                    }
                }
                insert_into_indexes_no_unique_check(
                    &mut self.pager,
                    &mut indexes,
                    &full_values,
                    rowid,
                )?;
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
    fn execute_delete(&mut self, delete: &crate::ast::DeleteStatement) -> Result<QueryResult> {
        let schema_entries = schema::read_schema(&mut self.pager)?;
        let table_schemas = build_table_schemas(&schema_entries)?;
        let schema = table_schemas
            .iter()
            .find(|t| t.name.eq_ignore_ascii_case(&delete.table))
            .ok_or_else(|| RsqliteError::Runtime(format!("no such table: {}", delete.table)))?
            .clone();

        // Find indexes on this table.
        let indexes = build_index_schemas(&schema_entries, &schema.name, &schema.columns);

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
        for rowid in &to_delete {
            // Remove from indexes first, then from the table.
            if !indexes.is_empty() {
                delete_from_indexes(&mut self.pager, &indexes, *rowid)?;
            }
            btree::btree_delete(&mut self.pager, schema.root_page, *rowid)?;
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
    fn execute_update(&mut self, update: &crate::ast::UpdateStatement) -> Result<QueryResult> {
        let schema_entries = schema::read_schema(&mut self.pager)?;
        let table_schemas = build_table_schemas(&schema_entries)?;
        let schema = table_schemas
            .iter()
            .find(|t| t.name.eq_ignore_ascii_case(&update.table))
            .ok_or_else(|| RsqliteError::Runtime(format!("no such table: {}", update.table)))?
            .clone();

        // Find indexes on this table.
        let mut indexes = build_index_schemas(&schema_entries, &schema.name, &schema.columns);

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
                // Enforce NOT NULL constraints on updated values.
                for (i, val) in values.iter().enumerate() {
                    if schema.rowid_column == Some(i) {
                        continue;
                    }
                    if let Some(cc) = schema.column_constraints.get(i) {
                        if cc.not_null && matches!(val, Value::Null) {
                            return Err(RsqliteError::Constraint(format!(
                                "NOT NULL constraint failed: {}.{}",
                                schema.name, schema.columns[i]
                            )));
                        }
                    }
                }
                // Enforce CHECK constraints on updated values.
                if !schema.check_constraints.is_empty() {
                    let mut check_values = values.clone();
                    if let Some(pk_idx) = schema.rowid_column {
                        check_values[pk_idx] = Value::Integer(rowid);
                    }
                    let col_names: Vec<&str> = schema.columns.iter().map(|s| s.as_str()).collect();
                    for check_expr in &schema.check_constraints {
                        let result = vm::eval_expr(
                            check_expr,
                            &col_names,
                            &check_values,
                            rowid,
                            Some(&schema.name),
                        )?;
                        if let Value::Integer(0) = result {
                            return Err(RsqliteError::Constraint(
                                "CHECK constraint failed".to_string(),
                            ));
                        }
                    }
                }
                updates.push((rowid, values));
            }
            cursor.move_to_next(&mut self.pager)?;
        }

        let changes = updates.len() as i64;
        let mut root_page = schema.root_page;

        for (rowid, mut values) in updates {
            // Remove old index entries.
            if !indexes.is_empty() {
                delete_from_indexes(&mut self.pager, &indexes, rowid)?;
            }

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

            // Insert new index entries.
            if !indexes.is_empty() {
                let mut full_values = values;
                if let Some(pk_idx) = schema.rowid_column {
                    if pk_idx < full_values.len() {
                        full_values[pk_idx] = Value::Integer(rowid);
                    }
                }
                insert_into_indexes(&mut self.pager, &mut indexes, &full_values, rowid)?;
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
    fn execute_drop_table(&mut self, drop: &crate::ast::DropTableStatement) -> Result<QueryResult> {
        let schema_entries = schema::read_schema(&mut self.pager)?;

        let table_entry = schema::find_table(&schema_entries, &drop.name);
        if table_entry.is_none() {
            if drop.if_exists {
                return Ok(empty_result());
            }
            return Err(RsqliteError::Runtime(format!(
                "no such table: {}",
                drop.name
            )));
        }
        let table_root = table_entry.unwrap().rootpage as u32;

        // Collect pages to free: table B-tree + all associated index B-trees.
        let mut pages_to_free: Vec<u32> = btree::collect_btree_pages(&mut self.pager, table_root)?;

        // Find and collect index pages, then delete index schema entries.
        let mut schema_rowids_to_delete = Vec::new();
        let mut cursor = btree::BTreeCursor::new(1);
        cursor.move_to_first(&mut self.pager)?;

        while cursor.is_valid() {
            let rowid = cursor.current_rowid(&mut self.pager)?;
            let payload = cursor.current_payload(&mut self.pager)?;
            let values = record::decode_record(&payload)?;
            if values.len() >= 3 {
                if let (Value::Text(ref entry_type), Value::Text(ref name)) =
                    (&values[0], &values[1])
                {
                    let tbl_name = if let Value::Text(ref t) = values[2] {
                        t.as_str()
                    } else {
                        ""
                    };
                    if name.eq_ignore_ascii_case(&drop.name)
                        || (entry_type == "index" && tbl_name.eq_ignore_ascii_case(&drop.name))
                    {
                        schema_rowids_to_delete.push(rowid);
                        if entry_type == "index" {
                            if let Value::Integer(rp) = &values[3] {
                                let idx_pages =
                                    btree::collect_btree_pages(&mut self.pager, *rp as u32)?;
                                pages_to_free.extend(idx_pages);
                            }
                        }
                    }
                }
            }
            cursor.move_to_next(&mut self.pager)?;
        }

        // Delete schema entries.
        for rowid in schema_rowids_to_delete {
            btree::btree_delete(&mut self.pager, 1, rowid)?;
        }

        // Free all collected pages.
        for page_num in pages_to_free {
            self.pager.free_page(page_num)?;
        }

        self.pager.header.schema_cookie += 1;
        self.pager.flush()?;

        Ok(empty_result())
    }

    /// Execute a CREATE INDEX statement.
    fn execute_create_index(
        &mut self,
        ci: &crate::ast::CreateIndexStatement,
    ) -> Result<QueryResult> {
        let schema_entries = schema::read_schema(&mut self.pager)?;

        // Check if the index already exists.
        if schema_entries
            .iter()
            .any(|e| e.entry_type == "index" && e.name.eq_ignore_ascii_case(&ci.name))
        {
            if ci.if_not_exists {
                return Ok(empty_result());
            }
            return Err(RsqliteError::Runtime(format!(
                "index {} already exists",
                ci.name
            )));
        }

        // Verify the table exists and get its schema.
        let table_schemas = build_table_schemas(&schema_entries)?;
        let table_schema = table_schemas
            .iter()
            .find(|t| t.name.eq_ignore_ascii_case(&ci.table))
            .ok_or_else(|| RsqliteError::Runtime(format!("no such table: {}", ci.table)))?
            .clone();

        // Verify all indexed columns exist.
        let col_indices: Vec<usize> = ci
            .columns
            .iter()
            .map(|ic| {
                table_schema
                    .columns
                    .iter()
                    .position(|c| c.eq_ignore_ascii_case(&ic.name))
                    .ok_or_else(|| {
                        RsqliteError::Runtime(format!(
                            "table {} has no column named {}",
                            ci.table, ic.name
                        ))
                    })
            })
            .collect::<Result<_>>()?;

        // Allocate a new page for the index B-tree.
        let root_page = self.pager.allocate_page()?;
        btree::init_table_leaf_page(&mut self.pager, root_page)?;

        // Build index entries from existing table data.
        let mut idx_root = root_page;
        let mut cursor = btree::BTreeCursor::new(table_schema.root_page);
        cursor.move_to_first(&mut self.pager)?;
        let mut idx_rowid = 0i64;

        while cursor.is_valid() {
            let table_rowid = cursor.current_rowid(&mut self.pager)?;
            let payload = cursor.current_payload(&mut self.pager)?;
            let mut values = record::decode_record(&payload)?;

            // Substitute INTEGER PRIMARY KEY value.
            if let Some(pk_idx) = table_schema.rowid_column {
                if pk_idx < values.len() {
                    values[pk_idx] = Value::Integer(table_rowid);
                }
            }

            // Build the index entry: indexed column values + table rowid.
            let mut index_values: Vec<Value> = col_indices
                .iter()
                .map(|&i| values.get(i).cloned().unwrap_or(Value::Null))
                .collect();
            index_values.push(Value::Integer(table_rowid));

            let index_payload = record::encode_record(&index_values);
            idx_rowid += 1;
            let new_root =
                btree::btree_insert(&mut self.pager, idx_root, idx_rowid, &index_payload)?;
            if new_root != idx_root {
                idx_root = new_root;
            }

            cursor.move_to_next(&mut self.pager)?;
        }

        // If root changed, use the final root for the schema entry.
        let final_root = idx_root;

        // Reconstruct the CREATE INDEX SQL.
        let unique_str = if ci.unique { "UNIQUE " } else { "" };
        let cols_str: Vec<String> = ci.columns.iter().map(|c| c.name.clone()).collect();
        let sql = format!(
            "CREATE {}INDEX {} ON {}({})",
            unique_str,
            ci.name,
            ci.table,
            cols_str.join(", ")
        );

        // Insert into sqlite_master.
        let master_values = vec![
            Value::Text("index".into()),
            Value::Text(ci.name.clone()),
            Value::Text(ci.table.clone()),
            Value::Integer(final_root as i64),
            Value::Text(sql),
        ];
        let master_payload = record::encode_record(&master_values);
        let master_rowid = btree::find_max_rowid(&mut self.pager, 1)? + 1;
        let new_master_root =
            btree::btree_insert(&mut self.pager, 1, master_rowid, &master_payload)?;
        if new_master_root != 1 {
            return Err(RsqliteError::Runtime(
                "sqlite_master root page split is not yet supported".into(),
            ));
        }

        self.pager.header.schema_cookie += 1;
        self.pager.flush()?;

        Ok(empty_result())
    }

    /// Execute a DROP INDEX statement.
    fn execute_drop_index(&mut self, drop: &crate::ast::DropIndexStatement) -> Result<QueryResult> {
        let schema_entries = schema::read_schema(&mut self.pager)?;

        let idx_entry = schema_entries
            .iter()
            .find(|e| e.entry_type == "index" && e.name.eq_ignore_ascii_case(&drop.name));

        if idx_entry.is_none() {
            if drop.if_exists {
                return Ok(empty_result());
            }
            return Err(RsqliteError::Runtime(format!(
                "no such index: {}",
                drop.name
            )));
        }
        let index_root = idx_entry.unwrap().rootpage as u32;

        // Collect all pages of the index B-tree.
        let pages_to_free = btree::collect_btree_pages(&mut self.pager, index_root)?;

        // Find and delete the sqlite_master entry.
        let mut cursor = btree::BTreeCursor::new(1);
        cursor.move_to_first(&mut self.pager)?;

        let mut rowid_to_delete = None;
        while cursor.is_valid() {
            let rowid = cursor.current_rowid(&mut self.pager)?;
            let payload = cursor.current_payload(&mut self.pager)?;
            let values = record::decode_record(&payload)?;
            if values.len() >= 2 {
                if let (Value::Text(ref entry_type), Value::Text(ref name)) =
                    (&values[0], &values[1])
                {
                    if entry_type == "index" && name.eq_ignore_ascii_case(&drop.name) {
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

        // Free the index pages.
        for page_num in pages_to_free {
            self.pager.free_page(page_num)?;
        }

        self.pager.header.schema_cookie += 1;
        self.pager.flush()?;

        Ok(empty_result())
    }

    /// Execute an ALTER TABLE statement.
    fn execute_alter_table(
        &mut self,
        alter: &crate::ast::AlterTableStatement,
    ) -> Result<QueryResult> {
        use crate::ast::AlterTableAction;

        let schema_entries = schema::read_schema(&mut self.pager)?;
        let entry = schema::find_table(&schema_entries, &alter.table)
            .ok_or_else(|| RsqliteError::Runtime(format!("no such table: {}", alter.table)))?
            .clone();

        match &alter.action {
            AlterTableAction::RenameTable(new_name) => {
                // Check that the new name doesn't already exist.
                if schema::find_table(&schema_entries, new_name).is_some() {
                    return Err(RsqliteError::Runtime(format!(
                        "there is already another table or index with this name: {new_name}"
                    )));
                }

                // Update the SQL in sqlite_master: replace the old table name.
                let new_sql = if let Some(ref sql) = entry.sql {
                    // Replace table name in the CREATE TABLE statement.
                    // Parse and reconstruct to be safe.
                    if let Ok(Statement::CreateTable(mut ct)) = crate::parser::parse(sql) {
                        ct.name = new_name.clone();
                        reconstruct_create_table_sql(&ct)
                    } else {
                        sql.replace(&alter.table, new_name)
                    }
                } else {
                    return Err(RsqliteError::Runtime("table has no SQL definition".into()));
                };

                // Update sqlite_master entry.
                self.update_schema_entry(&entry.name, &entry.entry_type, |values| {
                    values[1] = Value::Text(new_name.clone()); // name
                    values[2] = Value::Text(new_name.clone()); // tbl_name
                    values[4] = Value::Text(new_sql.clone()); // sql
                })?;

                // Also update any indexes that reference this table.
                for idx_entry in &schema_entries {
                    if idx_entry.entry_type == "index"
                        && idx_entry.tbl_name.eq_ignore_ascii_case(&alter.table)
                    {
                        let new_idx_sql = idx_entry
                            .sql
                            .as_ref()
                            .map(|s| s.replace(&alter.table, new_name));
                        self.update_schema_entry(&idx_entry.name, "index", |values| {
                            values[2] = Value::Text(new_name.clone()); // tbl_name
                            if let Some(ref sql) = new_idx_sql {
                                values[4] = Value::Text(sql.clone()); // sql
                            }
                        })?;
                    }
                }
            }

            AlterTableAction::AddColumn(col_def) => {
                // Modify the CREATE TABLE SQL to add the new column.
                let new_sql = if let Some(ref sql) = entry.sql {
                    if let Ok(Statement::CreateTable(mut ct)) = crate::parser::parse(sql) {
                        ct.columns.push(col_def.clone());
                        reconstruct_create_table_sql(&ct)
                    } else {
                        return Err(RsqliteError::Runtime(
                            "could not parse table definition".into(),
                        ));
                    }
                } else {
                    return Err(RsqliteError::Runtime("table has no SQL definition".into()));
                };

                self.update_schema_entry(&entry.name, &entry.entry_type, |values| {
                    values[4] = Value::Text(new_sql.clone());
                })?;

                // Note: existing rows will naturally return NULL for the new column
                // since they have fewer values than the new column count.
            }

            AlterTableAction::RenameColumn { old, new } => {
                let new_sql = if let Some(ref sql) = entry.sql {
                    if let Ok(Statement::CreateTable(mut ct)) = crate::parser::parse(sql) {
                        let found = ct
                            .columns
                            .iter_mut()
                            .find(|c| c.name.eq_ignore_ascii_case(old));
                        if let Some(col) = found {
                            col.name = new.clone();
                        } else {
                            return Err(RsqliteError::Runtime(format!("no such column: {old}")));
                        }
                        reconstruct_create_table_sql(&ct)
                    } else {
                        return Err(RsqliteError::Runtime(
                            "could not parse table definition".into(),
                        ));
                    }
                } else {
                    return Err(RsqliteError::Runtime("table has no SQL definition".into()));
                };

                self.update_schema_entry(&entry.name, &entry.entry_type, |values| {
                    values[4] = Value::Text(new_sql.clone());
                })?;
            }

            AlterTableAction::DropColumn(col_name) => {
                // SQLite's DROP COLUMN has restrictions. We implement a simple version
                // that just removes the column from the schema.
                let (new_sql, col_idx) = if let Some(ref sql) = entry.sql {
                    if let Ok(Statement::CreateTable(mut ct)) = crate::parser::parse(sql) {
                        let idx = ct
                            .columns
                            .iter()
                            .position(|c| c.name.eq_ignore_ascii_case(col_name));
                        if let Some(idx) = idx {
                            ct.columns.remove(idx);
                            (reconstruct_create_table_sql(&ct), idx)
                        } else {
                            return Err(RsqliteError::Runtime(format!(
                                "no such column: {col_name}"
                            )));
                        }
                    } else {
                        return Err(RsqliteError::Runtime(
                            "could not parse table definition".into(),
                        ));
                    }
                } else {
                    return Err(RsqliteError::Runtime("table has no SQL definition".into()));
                };

                // Rewrite all rows to remove the column data.
                let table_schemas = build_table_schemas(&schema_entries)?;
                let schema = table_schemas
                    .iter()
                    .find(|t| t.name.eq_ignore_ascii_case(&alter.table))
                    .ok_or_else(|| {
                        RsqliteError::Runtime(format!("no such table: {}", alter.table))
                    })?
                    .clone();

                let mut rows_to_rewrite: Vec<(i64, Vec<Value>)> = Vec::new();
                let mut cursor = btree::BTreeCursor::new(schema.root_page);
                cursor.move_to_first(&mut self.pager)?;
                while cursor.is_valid() {
                    let rowid = cursor.current_rowid(&mut self.pager)?;
                    let payload = cursor.current_payload(&mut self.pager)?;
                    let mut values = record::decode_record(&payload)?;
                    if col_idx < values.len() {
                        values.remove(col_idx);
                    }
                    rows_to_rewrite.push((rowid, values));
                    cursor.move_to_next(&mut self.pager)?;
                }

                // Delete and re-insert all rows.
                for (rowid, _) in &rows_to_rewrite {
                    btree::btree_delete(&mut self.pager, schema.root_page, *rowid)?;
                }
                for (rowid, values) in rows_to_rewrite {
                    let payload = record::encode_record(&values);
                    btree::btree_insert(&mut self.pager, schema.root_page, rowid, &payload)?;
                }

                self.update_schema_entry(&entry.name, &entry.entry_type, |values| {
                    values[4] = Value::Text(new_sql.clone());
                })?;
            }
        }

        self.pager.header.schema_cookie += 1;
        self.pager.flush()?;
        Ok(empty_result())
    }

    /// Helper: update a single sqlite_master entry by name and type.
    fn update_schema_entry<F>(&mut self, name: &str, entry_type: &str, update_fn: F) -> Result<()>
    where
        F: FnOnce(&mut Vec<Value>),
    {
        let mut cursor = btree::BTreeCursor::new(1);
        cursor.move_to_first(&mut self.pager)?;

        while cursor.is_valid() {
            let rowid = cursor.current_rowid(&mut self.pager)?;
            let payload = cursor.current_payload(&mut self.pager)?;
            let mut values = record::decode_record(&payload)?;

            if values.len() >= 5 {
                let matches = match (&values[0], &values[1]) {
                    (Value::Text(et), Value::Text(n)) => {
                        et.eq_ignore_ascii_case(entry_type) && n.eq_ignore_ascii_case(name)
                    }
                    _ => false,
                };
                if matches {
                    update_fn(&mut values);
                    let new_payload = record::encode_record(&values);
                    btree::btree_delete(&mut self.pager, 1, rowid)?;
                    btree::btree_insert(&mut self.pager, 1, rowid, &new_payload)?;
                    return Ok(());
                }
            }
            cursor.move_to_next(&mut self.pager)?;
        }
        Ok(())
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

        let info = if let Some(ref sql) = entry.sql {
            extract_table_info(sql)
        } else {
            FullTableInfo {
                columns: vec![],
                rowid_column: None,
                constraints: vec![],
                check_constraints: Vec::new(),
            }
        };

        // Collect index info for this table from schema entries.
        let mut indexes = Vec::new();
        for idx_entry in entries {
            if idx_entry.entry_type != "index"
                || !idx_entry.tbl_name.eq_ignore_ascii_case(&entry.name)
            {
                continue;
            }
            if let Some(ref sql) = idx_entry.sql {
                if let Ok(Statement::CreateIndex(ci)) = crate::parser::parse(sql) {
                    indexes.push(vm::TableIndexInfo {
                        name: idx_entry.name.clone(),
                        columns: ci.columns.iter().map(|c| c.name.clone()).collect(),
                        root_page: idx_entry.rootpage as u32,
                        unique: ci.unique,
                    });
                }
            }
        }

        schemas.push(TableSchema {
            name: entry.name.clone(),
            columns: info.columns,
            root_page: entry.rootpage as u32,
            rowid_column: info.rowid_column,
            column_constraints: info.constraints,
            check_constraints: info.check_constraints,
            indexes,
        });
    }

    Ok(schemas)
}

/// Schema information for an index, used during write operations.
#[derive(Debug, Clone)]
struct IndexSchema {
    name: String,
    #[allow(dead_code)]
    table_name: String,
    root_page: u32,
    /// Column indices into the parent table's column list.
    column_indices: Vec<usize>,
    #[allow(dead_code)]
    unique: bool,
}

/// Build IndexSchema structs for a given table from schema entries.
fn build_index_schemas(
    entries: &[SchemaEntry],
    table_name: &str,
    table_columns: &[String],
) -> Vec<IndexSchema> {
    let mut indexes = Vec::new();

    for entry in entries {
        if entry.entry_type != "index" {
            continue;
        }
        if !entry.tbl_name.eq_ignore_ascii_case(table_name) {
            continue;
        }
        // Skip autoindexes (those with NULL sql).
        let sql = match &entry.sql {
            Some(s) => s,
            None => continue,
        };

        // Parse the CREATE INDEX SQL to get column names.
        if let Ok(Statement::CreateIndex(ci)) = crate::parser::parse(sql) {
            let col_indices: Vec<usize> = ci
                .columns
                .iter()
                .filter_map(|ic| {
                    table_columns
                        .iter()
                        .position(|c| c.eq_ignore_ascii_case(&ic.name))
                })
                .collect();

            if col_indices.len() == ci.columns.len() {
                indexes.push(IndexSchema {
                    name: entry.name.clone(),
                    table_name: entry.tbl_name.clone(),
                    root_page: entry.rootpage as u32,
                    column_indices: col_indices,
                    unique: ci.unique,
                });
            }
        }
    }

    indexes
}

/// Insert a row into all indexes for a table, enforcing UNIQUE constraints.
fn insert_into_indexes(
    pager: &mut Pager,
    indexes: &mut [IndexSchema],
    row_values: &[Value],
    table_rowid: i64,
) -> Result<()> {
    for index in indexes.iter_mut() {
        let key_values: Vec<Value> = index
            .column_indices
            .iter()
            .map(|&i| row_values.get(i).cloned().unwrap_or(Value::Null))
            .collect();

        // Enforce UNIQUE constraint: check for existing entries with the same key.
        // Per SQL standard, multiple NULLs are allowed in UNIQUE columns.
        if index.unique && !key_values.iter().any(|v| matches!(v, Value::Null)) {
            check_unique_violation(pager, index, &key_values)?;
        }

        let mut index_values = key_values;
        index_values.push(Value::Integer(table_rowid));

        let index_payload = record::encode_record(&index_values);
        let idx_rowid = btree::find_max_rowid(pager, index.root_page)? + 1;
        let new_root = btree::btree_insert(pager, index.root_page, idx_rowid, &index_payload)?;
        if new_root != index.root_page {
            update_root_page(pager, &index.name, new_root)?;
            index.root_page = new_root;
        }
    }
    Ok(())
}

/// Insert a row into all indexes without checking UNIQUE constraints.
/// Used when conflicts have already been resolved (e.g., INSERT OR REPLACE).
fn insert_into_indexes_no_unique_check(
    pager: &mut Pager,
    indexes: &mut [IndexSchema],
    row_values: &[Value],
    table_rowid: i64,
) -> Result<()> {
    for index in indexes.iter_mut() {
        let mut index_values: Vec<Value> = index
            .column_indices
            .iter()
            .map(|&i| row_values.get(i).cloned().unwrap_or(Value::Null))
            .collect();
        index_values.push(Value::Integer(table_rowid));

        let index_payload = record::encode_record(&index_values);
        let idx_rowid = btree::find_max_rowid(pager, index.root_page)? + 1;
        let new_root = btree::btree_insert(pager, index.root_page, idx_rowid, &index_payload)?;
        if new_root != index.root_page {
            update_root_page(pager, &index.name, new_root)?;
            index.root_page = new_root;
        }
    }
    Ok(())
}

/// Find the table rowid that conflicts with the given key values in a UNIQUE index.
/// Returns Some(rowid) if a conflict exists, None otherwise.
fn find_unique_conflict(
    pager: &mut Pager,
    index: &IndexSchema,
    key_values: &[Value],
) -> Result<Option<i64>> {
    let num_cols = index.column_indices.len();
    let mut cursor = btree::BTreeCursor::new(index.root_page);
    cursor.move_to_first(pager)?;

    while cursor.is_valid() {
        let payload = cursor.current_payload(pager)?;
        let existing = record::decode_record(&payload)?;

        let all_equal = key_values
            .iter()
            .take(num_cols)
            .enumerate()
            .all(|(i, new_val)| {
                let existing_val = existing.get(i).unwrap_or(&Value::Null);
                !matches!(existing_val, Value::Null)
                    && crate::types::sqlite_cmp(existing_val, new_val) == std::cmp::Ordering::Equal
            });

        if all_equal {
            // The last value in the index record is the table rowid.
            if let Some(Value::Integer(table_rowid)) = existing.get(num_cols) {
                return Ok(Some(*table_rowid));
            }
        }

        cursor.move_to_next(pager)?;
    }

    Ok(None)
}

/// Check if inserting `key_values` would violate a UNIQUE index constraint.
fn check_unique_violation(
    pager: &mut Pager,
    index: &IndexSchema,
    key_values: &[Value],
) -> Result<()> {
    if find_unique_conflict(pager, index, key_values)?.is_some() {
        return Err(RsqliteError::Constraint(format!(
            "UNIQUE constraint failed: {}",
            index.name
        )));
    }

    Ok(())
}

/// Delete a row from all indexes for a table by scanning for the matching table rowid.
fn delete_from_indexes(pager: &mut Pager, indexes: &[IndexSchema], table_rowid: i64) -> Result<()> {
    for index in indexes {
        // Scan the index to find the entry with this table rowid.
        let mut cursor = btree::BTreeCursor::new(index.root_page);
        cursor.move_to_first(pager)?;

        let num_cols = index.column_indices.len();

        while cursor.is_valid() {
            let idx_rowid = cursor.current_rowid(pager)?;
            let payload = cursor.current_payload(pager)?;
            let values = record::decode_record(&payload)?;

            // The last value in the index record is the table rowid.
            if let Some(Value::Integer(stored_rowid)) = values.get(num_cols) {
                if *stored_rowid == table_rowid {
                    btree::btree_delete(pager, index.root_page, idx_rowid)?;
                    break;
                }
            }
            cursor.move_to_next(pager)?;
        }
    }
    Ok(())
}

/// Full table info extracted from CREATE TABLE SQL.
struct FullTableInfo {
    columns: Vec<String>,
    rowid_column: Option<usize>,
    constraints: Vec<vm::ColumnConstraints>,
    check_constraints: Vec<crate::ast::Expr>,
}

/// Extract column names, INTEGER PRIMARY KEY index, and constraints from a CREATE TABLE SQL.
fn extract_table_info(sql: &str) -> FullTableInfo {
    // Try the parser first.
    if let Ok(Statement::CreateTable(ct)) = crate::parser::parse(sql) {
        let columns: Vec<String> = ct.columns.iter().map(|c| c.name.clone()).collect();
        let rowid_col = find_integer_primary_key(&ct);
        let constraints = extract_constraints(&ct, rowid_col);
        let check_constraints = extract_check_constraints(&ct);
        return FullTableInfo {
            columns,
            rowid_column: rowid_col,
            constraints,
            check_constraints,
        };
    }

    // Fallback: simple text-based extraction.
    let columns = extract_column_names_text(sql);
    let rowid_col = detect_integer_pk_text(sql, &columns);
    let constraints = vec![vm::ColumnConstraints::default(); columns.len()];
    FullTableInfo {
        columns,
        rowid_column: rowid_col,
        constraints,
        check_constraints: Vec::new(),
    }
}

/// Extract per-column constraints from a parsed CREATE TABLE.
fn extract_constraints(
    ct: &crate::ast::CreateTableStatement,
    rowid_col: Option<usize>,
) -> Vec<vm::ColumnConstraints> {
    ct.columns
        .iter()
        .enumerate()
        .map(|(i, col)| {
            let mut cc = vm::ColumnConstraints::default();
            for constraint in &col.constraints {
                match constraint {
                    crate::ast::ColumnConstraint::NotNull => {
                        cc.not_null = true;
                    }
                    crate::ast::ColumnConstraint::PrimaryKey { .. } => {
                        // PRIMARY KEY implies NOT NULL.
                        cc.not_null = true;
                    }
                    crate::ast::ColumnConstraint::Default(expr) => {
                        cc.default_value = eval_constant_expr(expr).ok();
                    }
                    _ => {}
                }
            }
            // INTEGER PRIMARY KEY columns get their rowid auto-assigned, so
            // don't enforce NOT NULL on them (NULL means auto-assign).
            if rowid_col == Some(i) {
                cc.not_null = false;
            }
            cc
        })
        .collect()
}

/// Extract CHECK constraint expressions from both column-level and table-level constraints.
fn extract_check_constraints(ct: &crate::ast::CreateTableStatement) -> Vec<crate::ast::Expr> {
    let mut checks = Vec::new();
    // Column-level CHECK constraints.
    for col in &ct.columns {
        for constraint in &col.constraints {
            if let crate::ast::ColumnConstraint::Check(expr) = constraint {
                checks.push(expr.clone());
            }
        }
    }
    // Table-level CHECK constraints.
    for constraint in &ct.constraints {
        if let crate::ast::TableConstraint::Check(expr) = constraint {
            checks.push(expr.clone());
        }
    }
    checks
}

/// Find the INTEGER PRIMARY KEY column from a parsed CREATE TABLE statement.
fn find_integer_primary_key(ct: &crate::ast::CreateTableStatement) -> Option<usize> {
    for (i, col) in ct.columns.iter().enumerate() {
        let has_pk = col
            .constraints
            .iter()
            .any(|c| matches!(c, crate::ast::ColumnConstraint::PrimaryKey { .. }));
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
        let pattern = format!("{} INTEGER PRIMARY KEY", col.to_uppercase());
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
    if let Some(stripped) = s.strip_prefix('"') {
        if let Some(end) = stripped.find('"') {
            return Some(stripped[..end].to_string());
        }
    }
    if let Some(stripped) = s.strip_prefix('`') {
        if let Some(end) = stripped.find('`') {
            return Some(stripped[..end].to_string());
        }
    }
    if let Some(stripped) = s.strip_prefix('[') {
        if let Some(end) = stripped.find(']') {
            return Some(stripped[..end].to_string());
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
                    sql.push_str(&expr_to_sql(expr));
                }
                crate::ast::ColumnConstraint::Check(expr) => {
                    sql.push_str(" CHECK(");
                    sql.push_str(&expr_to_sql(expr));
                    sql.push(')');
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
            crate::ast::TableConstraint::Check(expr) => {
                sql.push_str("CHECK(");
                sql.push_str(&expr_to_sql(expr));
                sql.push(')');
            }
            _ => {}
        }
    }

    sql.push(')');
    sql
}

/// Convert an expression to SQL text for storage in sqlite_master.
fn expr_to_sql(expr: &Expr) -> String {
    match expr {
        Expr::Literal(lit) => match lit {
            LiteralValue::Integer(i) => i.to_string(),
            LiteralValue::Real(f) => format!("{f}"),
            LiteralValue::String(s) => format!("'{}'", s.replace('\'', "''")),
            LiteralValue::Blob(b) => {
                let hex: String = b.iter().map(|byte| format!("{byte:02X}")).collect();
                format!("X'{hex}'")
            }
            LiteralValue::Null => "NULL".to_string(),
            LiteralValue::CurrentTime => "CURRENT_TIME".to_string(),
            LiteralValue::CurrentDate => "CURRENT_DATE".to_string(),
            LiteralValue::CurrentTimestamp => "CURRENT_TIMESTAMP".to_string(),
        },
        Expr::UnaryOp { op, operand } => {
            let inner = expr_to_sql(operand);
            match op {
                crate::ast::UnaryOp::Negate => format!("-{inner}"),
                crate::ast::UnaryOp::Not => format!("NOT {inner}"),
                crate::ast::UnaryOp::BitwiseNot => format!("~{inner}"),
                crate::ast::UnaryOp::Plus => format!("+{inner}"),
            }
        }
        Expr::BinaryOp { left, op, right } => {
            let l = expr_to_sql(left);
            let r = expr_to_sql(right);
            let op_str = match op {
                BinaryOp::Add => "+",
                BinaryOp::Subtract => "-",
                BinaryOp::Multiply => "*",
                BinaryOp::Divide => "/",
                BinaryOp::Modulo => "%",
                BinaryOp::Eq => "=",
                BinaryOp::NotEq => "!=",
                BinaryOp::Lt => "<",
                BinaryOp::Gt => ">",
                BinaryOp::Le => "<=",
                BinaryOp::Ge => ">=",
                BinaryOp::And => "AND",
                BinaryOp::Or => "OR",
                BinaryOp::Concat => "||",
                BinaryOp::BitAnd => "&",
                BinaryOp::BitOr => "|",
                BinaryOp::ShiftLeft => "<<",
                BinaryOp::ShiftRight => ">>",
                BinaryOp::Is => "IS",
                BinaryOp::IsNot => "IS NOT",
                BinaryOp::Glob => "GLOB",
                BinaryOp::Like => "LIKE",
            };
            format!("{l} {op_str} {r}")
        }
        Expr::Parenthesized(inner) => format!("({})", expr_to_sql(inner)),
        Expr::FunctionCall { name, args } => {
            let args_str = match args {
                FunctionArgs::Wildcard => "*".to_string(),
                FunctionArgs::Exprs { args, .. } => {
                    args.iter().map(expr_to_sql).collect::<Vec<_>>().join(", ")
                }
            };
            format!("{name}({args_str})")
        }
        Expr::ColumnRef { table, column } => {
            if let Some(t) = table {
                format!("{t}.{column}")
            } else {
                column.clone()
            }
        }
        Expr::IsNull { operand, negated } => {
            if *negated {
                format!("{} IS NOT NULL", expr_to_sql(operand))
            } else {
                format!("{} IS NULL", expr_to_sql(operand))
            }
        }
        Expr::Between {
            operand,
            low,
            high,
            negated,
        } => {
            let not = if *negated { " NOT" } else { "" };
            format!(
                "{}{not} BETWEEN {} AND {}",
                expr_to_sql(operand),
                expr_to_sql(low),
                expr_to_sql(high)
            )
        }
        Expr::In {
            operand,
            list,
            negated,
        } => {
            let not = if *negated { " NOT" } else { "" };
            match list {
                crate::ast::InList::Values(vals) => {
                    let vals_str: Vec<String> = vals.iter().map(expr_to_sql).collect();
                    format!("{}{not} IN ({})", expr_to_sql(operand), vals_str.join(", "))
                }
                crate::ast::InList::Subquery(sel) => {
                    format!("{}{not} IN ({sel:?})", expr_to_sql(operand))
                }
            }
        }
        Expr::Cast {
            expr: inner,
            type_name,
        } => {
            format!("CAST({} AS {type_name})", expr_to_sql(inner))
        }
        _ => format!("{expr:?}"), // Fallback for complex expressions
    }
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
fn execute_pragma(pragma: &crate::ast::PragmaStatement, pager: &mut Pager) -> Result<QueryResult> {
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
            let entry = schema::find_table(&schema_entries, &table_name)
                .ok_or_else(|| RsqliteError::Runtime(format!("no such table: {table_name}")))?;

            let info = if let Some(ref sql) = entry.sql {
                extract_table_info(sql)
            } else {
                FullTableInfo {
                    columns: vec![],
                    rowid_column: None,
                    constraints: vec![],
                    check_constraints: Vec::new(),
                }
            };

            // Also parse column types and PK status from the CREATE TABLE AST.
            let parsed = entry.sql.as_ref().and_then(|sql| {
                crate::parser::parse(sql).ok().and_then(|s| {
                    if let Statement::CreateTable(ct) = s {
                        Some(ct)
                    } else {
                        None
                    }
                })
            });

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

            let rows: Vec<Row> = info
                .columns
                .iter()
                .enumerate()
                .map(|(i, name)| {
                    let (type_name, notnull, dflt, pk) = if let Some(ref ct) = parsed {
                        if let Some(col_def) = ct.columns.get(i) {
                            let tn = col_def.type_name.clone().unwrap_or_default();
                            let nn = col_def.constraints.iter().any(|c| {
                                matches!(
                                    c,
                                    crate::ast::ColumnConstraint::NotNull
                                        | crate::ast::ColumnConstraint::PrimaryKey { .. }
                                )
                            });
                            let df = col_def.constraints.iter().find_map(|c| {
                                if let crate::ast::ColumnConstraint::Default(expr) = c {
                                    eval_constant_expr(expr).ok()
                                } else {
                                    None
                                }
                            });
                            let is_pk = col_def.constraints.iter().any(|c| {
                                matches!(c, crate::ast::ColumnConstraint::PrimaryKey { .. })
                            });
                            (tn, nn, df, is_pk)
                        } else {
                            (String::new(), false, None, false)
                        }
                    } else {
                        (String::new(), false, None, false)
                    };
                    Row {
                        values: vec![
                            Value::Integer(i as i64),
                            Value::Text(name.clone()),
                            Value::Text(type_name),
                            Value::Integer(if notnull { 1 } else { 0 }),
                            dflt.unwrap_or(Value::Null),
                            Value::Integer(if pk { 1 } else { 0 }),
                        ],
                    }
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
        "database_list" => Ok(QueryResult {
            columns: vec![
                ColumnInfo {
                    name: "seq".into(),
                    table: None,
                },
                ColumnInfo {
                    name: "name".into(),
                    table: None,
                },
                ColumnInfo {
                    name: "file".into(),
                    table: None,
                },
            ],
            rows: vec![Row {
                values: vec![
                    Value::Integer(0),
                    Value::Text("main".into()),
                    Value::Text(
                        pager
                            .path()
                            .map(|p| p.to_string_lossy().to_string())
                            .unwrap_or_default(),
                    ),
                ],
            }],
        }),
        "journal_mode" => Ok(QueryResult {
            columns: vec![ColumnInfo {
                name: "journal_mode".into(),
                table: None,
            }],
            rows: vec![Row {
                values: vec![Value::Text("delete".into())],
            }],
        }),
        "index_list" => {
            let table_name = match &pragma.value {
                Some(crate::ast::PragmaValue::Name(n)) => n.clone(),
                Some(crate::ast::PragmaValue::StringLiteral(s)) => s.clone(),
                _ => {
                    return Err(RsqliteError::Runtime(
                        "PRAGMA index_list requires a table name".into(),
                    ))
                }
            };
            let schema_entries = schema::read_schema(pager)?;
            let cols = vec![
                ColumnInfo {
                    name: "seq".into(),
                    table: None,
                },
                ColumnInfo {
                    name: "name".into(),
                    table: None,
                },
                ColumnInfo {
                    name: "unique".into(),
                    table: None,
                },
                ColumnInfo {
                    name: "origin".into(),
                    table: None,
                },
                ColumnInfo {
                    name: "partial".into(),
                    table: None,
                },
            ];
            let mut rows = Vec::new();
            let mut seq = 0i64;
            for entry in &schema_entries {
                if entry.entry_type == "index" && entry.tbl_name.eq_ignore_ascii_case(&table_name) {
                    let is_unique = entry
                        .sql
                        .as_ref()
                        .is_some_and(|s| s.to_uppercase().contains("UNIQUE"));
                    rows.push(Row {
                        values: vec![
                            Value::Integer(seq),
                            Value::Text(entry.name.clone()),
                            Value::Integer(if is_unique { 1 } else { 0 }),
                            Value::Text("c".into()),
                            Value::Integer(0),
                        ],
                    });
                    seq += 1;
                }
            }
            Ok(QueryResult {
                columns: cols,
                rows,
            })
        }
        "index_info" => {
            let index_name = match &pragma.value {
                Some(crate::ast::PragmaValue::Name(n)) => n.clone(),
                Some(crate::ast::PragmaValue::StringLiteral(s)) => s.clone(),
                _ => {
                    return Err(RsqliteError::Runtime(
                        "PRAGMA index_info requires an index name".into(),
                    ))
                }
            };
            let schema_entries = schema::read_schema(pager)?;
            let idx_entry = schema_entries
                .iter()
                .find(|e| e.entry_type == "index" && e.name.eq_ignore_ascii_case(&index_name));
            let cols = vec![
                ColumnInfo {
                    name: "seqno".into(),
                    table: None,
                },
                ColumnInfo {
                    name: "cid".into(),
                    table: None,
                },
                ColumnInfo {
                    name: "name".into(),
                    table: None,
                },
            ];
            let mut rows = Vec::new();
            if let Some(entry) = idx_entry {
                if let Some(ref sql) = entry.sql {
                    if let Ok(Statement::CreateIndex(ci)) = crate::parser::parse(sql) {
                        // Get parent table columns.
                        let table_entry = schema::find_table(&schema_entries, &ci.table);
                        let table_cols = table_entry
                            .and_then(|e| e.sql.as_ref())
                            .map(|sql| extract_table_info(sql).columns)
                            .unwrap_or_default();

                        for (seq, ic) in ci.columns.iter().enumerate() {
                            let cid = table_cols
                                .iter()
                                .position(|c| c.eq_ignore_ascii_case(&ic.name));
                            rows.push(Row {
                                values: vec![
                                    Value::Integer(seq as i64),
                                    Value::Integer(cid.map_or(-1, |c| c as i64)),
                                    Value::Text(ic.name.clone()),
                                ],
                            });
                        }
                    }
                }
            }
            Ok(QueryResult {
                columns: cols,
                rows,
            })
        }
        "schema_version" | "user_version" => {
            let val = if name == "schema_version" {
                pager.header.schema_cookie as i64
            } else {
                0 // user_version not tracked yet
            };
            Ok(QueryResult {
                columns: vec![ColumnInfo {
                    name: name.clone(),
                    table: None,
                }],
                rows: vec![Row {
                    values: vec![Value::Integer(val)],
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

/// Produce EXPLAIN output: one row per opcode-style description of what the VM will do.
fn explain_statement(stmt: &Statement) -> Result<QueryResult> {
    let columns = vec![
        ColumnInfo {
            name: "addr".into(),
            table: None,
        },
        ColumnInfo {
            name: "opcode".into(),
            table: None,
        },
        ColumnInfo {
            name: "p1".into(),
            table: None,
        },
        ColumnInfo {
            name: "p2".into(),
            table: None,
        },
        ColumnInfo {
            name: "p3".into(),
            table: None,
        },
        ColumnInfo {
            name: "p4".into(),
            table: None,
        },
        ColumnInfo {
            name: "p5".into(),
            table: None,
        },
        ColumnInfo {
            name: "comment".into(),
            table: None,
        },
    ];

    let mut rows = Vec::new();
    let mut addr = 0i64;

    let mut emit = |opcode: &str, p1: i64, p2: i64, p3: i64, p4: &str, comment: &str| {
        rows.push(Row {
            values: vec![
                Value::Integer(addr),
                Value::Text(opcode.into()),
                Value::Integer(p1),
                Value::Integer(p2),
                Value::Integer(p3),
                Value::Text(p4.into()),
                Value::Integer(0),
                Value::Text(comment.into()),
            ],
        });
        addr += 1;
    };

    match stmt {
        Statement::Select(select) => {
            emit("Init", 0, 0, 0, "", "Start at 0");
            if let Some(ref from) = select.from {
                if let crate::ast::TableRef::Table { ref name, .. } = from.table {
                    emit("OpenRead", 0, 0, 0, name, &format!("table {name}"));
                    emit("Rewind", 0, 0, 0, "", "");
                }
            }
            if select.where_clause.is_some() {
                emit("Filter", 0, 0, 0, "", "WHERE clause");
            }
            emit("ResultRow", 0, 0, 0, "", "output row");
            if select.from.is_some() {
                emit("Next", 0, 0, 0, "", "advance cursor");
            }
            emit("Halt", 0, 0, 0, "", "");
        }
        Statement::Insert(insert) => {
            emit("Init", 0, 0, 0, "", "Start at 0");
            emit(
                "OpenWrite",
                0,
                0,
                0,
                &insert.table,
                &format!("table {}", insert.table),
            );
            if let InsertSource::Values(ref val_rows) = insert.source {
                for _ in val_rows {
                    emit("MakeRecord", 0, 0, 0, "", "");
                    emit("Insert", 0, 0, 0, "", "");
                }
            }
            emit("Halt", 0, 0, 0, "", "");
        }
        Statement::Update(update) => {
            emit("Init", 0, 0, 0, "", "Start at 0");
            emit(
                "OpenWrite",
                0,
                0,
                0,
                &update.table,
                &format!("table {}", update.table),
            );
            emit("Rewind", 0, 0, 0, "", "");
            if update.where_clause.is_some() {
                emit("Filter", 0, 0, 0, "", "WHERE clause");
            }
            emit("MakeRecord", 0, 0, 0, "", "build updated row");
            emit("Insert", 0, 0, 0, "", "replace row");
            emit("Next", 0, 0, 0, "", "advance cursor");
            emit("Halt", 0, 0, 0, "", "");
        }
        Statement::Delete(delete) => {
            emit("Init", 0, 0, 0, "", "Start at 0");
            emit(
                "OpenWrite",
                0,
                0,
                0,
                &delete.table,
                &format!("table {}", delete.table),
            );
            emit("Rewind", 0, 0, 0, "", "");
            if delete.where_clause.is_some() {
                emit("Filter", 0, 0, 0, "", "WHERE clause");
            }
            emit("Delete", 0, 0, 0, "", "delete row");
            emit("Next", 0, 0, 0, "", "advance cursor");
            emit("Halt", 0, 0, 0, "", "");
        }
        _ => {
            emit("Init", 0, 0, 0, "", "");
            emit("Halt", 0, 0, 0, "", "");
        }
    }

    Ok(QueryResult { columns, rows })
}

/// Produce EXPLAIN QUERY PLAN output with id/parent/notused/detail columns.
fn explain_query_plan(stmt: &Statement, tables: &[TableSchema]) -> Result<QueryResult> {
    let columns = vec![
        ColumnInfo {
            name: "id".into(),
            table: None,
        },
        ColumnInfo {
            name: "parent".into(),
            table: None,
        },
        ColumnInfo {
            name: "notused".into(),
            table: None,
        },
        ColumnInfo {
            name: "detail".into(),
            table: None,
        },
    ];

    let mut rows = Vec::new();
    let mut next_id = 2i64; // SQLite starts sub-plan IDs at 2

    match stmt {
        Statement::Select(select) => {
            eqp_select(select, tables, 0, &mut next_id, &mut rows);
        }
        Statement::Insert(insert) => {
            rows.push(eqp_row(1, 0, 0, &format!("SCAN table {}", insert.table)));
        }
        Statement::Update(update) => {
            rows.push(eqp_row(1, 0, 0, &format!("SCAN table {}", update.table)));
        }
        Statement::Delete(delete) => {
            rows.push(eqp_row(1, 0, 0, &format!("SCAN table {}", delete.table)));
        }
        _ => {}
    }

    Ok(QueryResult { columns, rows })
}

fn eqp_select(
    select: &crate::ast::SelectStatement,
    tables: &[TableSchema],
    parent: i64,
    next_id: &mut i64,
    rows: &mut Vec<Row>,
) {
    if let Some(ref from) = select.from {
        // Main table scan.
        let table_name = match &from.table {
            crate::ast::TableRef::Table { name, .. } => name.as_str(),
            crate::ast::TableRef::Subquery { alias, .. } => alias.as_str(),
        };

        let detail = if let crate::ast::TableRef::Subquery { select: sub, .. } = &from.table {
            let id = *next_id;
            *next_id += 1;
            rows.push(eqp_row(
                id,
                parent,
                0,
                &format!("SCAN subquery {table_name}"),
            ));
            eqp_select(sub, tables, id, next_id, rows);
            return; // already handled
        } else {
            // Check if any index could be used for the WHERE clause.
            let index_detail = select
                .where_clause
                .as_ref()
                .and_then(|where_expr| find_usable_index(table_name, where_expr, tables));
            match index_detail {
                Some(idx_name) => format!("SEARCH table {table_name} USING INDEX {idx_name}"),
                None => format!("SCAN table {table_name}"),
            }
        };

        let id = *next_id;
        *next_id += 1;
        rows.push(eqp_row(id, parent, 0, &detail));

        // JOINs.
        for join in &from.joins {
            let right_name = match &join.table {
                crate::ast::TableRef::Table { name, .. } => name.as_str(),
                crate::ast::TableRef::Subquery { alias, .. } => alias.as_str(),
            };
            let join_detail = format!("SCAN table {right_name}");
            let jid = *next_id;
            *next_id += 1;
            rows.push(eqp_row(jid, parent, 0, &join_detail));
        }
    }

    // Compound selects.
    for compound in &select.compound {
        let op_name = match compound.op {
            crate::ast::CompoundOp::Union => "UNION",
            crate::ast::CompoundOp::UnionAll => "UNION ALL",
            crate::ast::CompoundOp::Intersect => "INTERSECT",
            crate::ast::CompoundOp::Except => "EXCEPT",
        };
        let cid = *next_id;
        *next_id += 1;
        rows.push(eqp_row(
            cid,
            parent,
            0,
            &format!("COMPOUND SUBQUERY ({op_name})"),
        ));
        eqp_select(&compound.select, tables, cid, next_id, rows);
    }

    // ORDER BY / GROUP BY use temporary B-trees.
    if select.order_by.is_some() {
        rows.push(eqp_row(0, 0, 0, "USE TEMP B-TREE FOR ORDER BY"));
    }
    if select.group_by.is_some() {
        rows.push(eqp_row(0, 0, 0, "USE TEMP B-TREE FOR GROUP BY"));
    }
    if select.distinct {
        rows.push(eqp_row(0, 0, 0, "USE TEMP B-TREE FOR DISTINCT"));
    }
}

/// Check if there's an index on `table_name` whose first column matches a simple
/// equality or comparison in the WHERE expression.
fn find_usable_index(
    table_name: &str,
    where_expr: &Expr,
    tables: &[TableSchema],
) -> Option<String> {
    // Collect column names referenced in simple comparisons.
    let mut where_columns = Vec::new();
    collect_where_columns(where_expr, &mut where_columns);

    // Look for an index whose first column matches.
    for ts in tables {
        if !ts.name.eq_ignore_ascii_case(table_name) {
            continue;
        }
        for idx in &ts.indexes {
            if let Some(first_col) = idx.columns.first() {
                if where_columns
                    .iter()
                    .any(|c| c.eq_ignore_ascii_case(first_col))
                {
                    return Some(idx.name.clone());
                }
            }
        }
    }
    None
}

/// Recursively collect column names from simple comparison expressions.
fn collect_where_columns(expr: &Expr, cols: &mut Vec<String>) {
    match expr {
        Expr::BinaryOp { left, op, right } => {
            use crate::ast::BinaryOp;
            match op {
                BinaryOp::Eq | BinaryOp::Lt | BinaryOp::Gt | BinaryOp::Le | BinaryOp::Ge => {
                    if let Expr::ColumnRef { column, .. } = left.as_ref() {
                        cols.push(column.clone());
                    }
                    if let Expr::ColumnRef { column, .. } = right.as_ref() {
                        cols.push(column.clone());
                    }
                }
                BinaryOp::And => {
                    collect_where_columns(left, cols);
                    collect_where_columns(right, cols);
                }
                _ => {}
            }
        }
        Expr::IsNull { operand, .. } => {
            if let Expr::ColumnRef { column, .. } = operand.as_ref() {
                cols.push(column.clone());
            }
        }
        _ => {}
    }
}

fn eqp_row(id: i64, parent: i64, notused: i64, detail: &str) -> Row {
    Row {
        values: vec![
            Value::Integer(id),
            Value::Integer(parent),
            Value::Integer(notused),
            Value::Text(detail.into()),
        ],
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
                sql: Some("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)".into()),
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
        let sql = "CREATE TABLE t (id INT, ref_id INT, FOREIGN KEY (ref_id) REFERENCES other(id))";
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
        let sql = "CREATE TABLE t (a INT, b INT, CONSTRAINT pk PRIMARY KEY (a))";
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
        assert!(tables.iter().any(|t| t == "users"), "users table not found");

        // Read the schema and build table schemas.
        let entries = db.schema().unwrap();
        let table_schemas = build_table_schemas(&entries).unwrap();

        let users_schema = table_schemas.iter().find(|s| s.name == "users").unwrap();
        assert_eq!(users_schema.columns, vec!["id", "name", "age"]);
        assert_eq!(
            users_schema.rowid_column,
            Some(0),
            "id should be detected as INTEGER PRIMARY KEY"
        );
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
            compound: vec![],
            ctes: vec![],
        };

        let result = vm::execute_select(&mut db.pager, &select, &table_schemas).unwrap();

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
                right: Box::new(crate::ast::Expr::Literal(crate::ast::LiteralValue::Real(
                    2.0,
                ))),
            }),
            group_by: None,
            having: None,
            order_by: None,
            limit: None,
            compound: vec![],
            ctes: vec![],
        };

        let result = vm::execute_select(&mut db.pager, &select, &table_schemas).unwrap();

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

        db.execute("INSERT INTO t VALUES (10), (20), (30)").unwrap();

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
        db.execute("INSERT INTO t VALUES (1, 'Alice', 80)").unwrap();
        db.execute("INSERT INTO t VALUES (2, 'Bob', 90)").unwrap();

        db.execute("UPDATE t SET score = 95 WHERE id = 1").unwrap();

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
        db.execute("INSERT INTO scores VALUES ('Bob', 85)").unwrap();
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
        assert!(
            lines[0].contains("Widget"),
            "expected Widget in: {}",
            lines[0]
        );
        assert!(
            lines[1].contains("Gadget"),
            "expected Gadget in: {}",
            lines[1]
        );
        assert!(
            lines[2].contains("Doohickey"),
            "expected Doohickey in: {}",
            lines[2]
        );
    }

    #[test]
    fn test_many_inserts_causes_split() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE big (id INTEGER PRIMARY KEY, data TEXT)")
            .unwrap();

        // Insert enough rows that the page should split.
        for i in 1..=100 {
            let sql = format!("INSERT INTO big VALUES ({i}, '{}')", "x".repeat(20));
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
        assert_eq!(result.rows[0].values[1], Value::Text("Engineering".into()));
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
        assert_eq!(result.rows[0].values[0], Value::Text("Engineering".into()));
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
        assert_eq!(result.rows[0].values[1], Value::Text("Engineering".into()));
    }

    #[test]
    fn test_left_join_unmatched_right() {
        let mut db = Database::in_memory();
        db.execute(
            "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount REAL)",
        )
        .unwrap();
        db.execute("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT)")
            .unwrap();
        db.execute("INSERT INTO customers VALUES (1, 'Alice')")
            .unwrap();
        db.execute("INSERT INTO customers VALUES (2, 'Bob')")
            .unwrap();
        db.execute("INSERT INTO orders VALUES (1, 1, 99.99)")
            .unwrap();

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
        db.execute(
            "CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, manager_id INTEGER)",
        )
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
        db.execute("CREATE TABLE t (name TEXT, val INTEGER)")
            .unwrap();
        db.execute("INSERT INTO t VALUES ('Hello World', 42)")
            .unwrap();
        db.execute("INSERT INTO t VALUES ('  spaces  ', -7)")
            .unwrap();

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

        let r = db.execute("SELECT (SELECT MAX(val) FROM t)").unwrap();
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

    #[test]
    fn test_create_index_basic() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT, price REAL)")
            .unwrap();
        db.execute("INSERT INTO items VALUES (1, 'apple', 1.50)")
            .unwrap();
        db.execute("INSERT INTO items VALUES (2, 'banana', 0.75)")
            .unwrap();
        db.execute("INSERT INTO items VALUES (3, 'cherry', 2.00)")
            .unwrap();

        // Create an index.
        db.execute("CREATE INDEX idx_items_name ON items(name)")
            .unwrap();

        // Verify the index appears in the schema.
        let entries = db.schema().unwrap();
        let idx = entries
            .iter()
            .find(|e| e.entry_type == "index" && e.name == "idx_items_name");
        assert!(idx.is_some(), "index should appear in schema");

        // Data should still be queryable.
        let r = db.execute("SELECT name FROM items ORDER BY name").unwrap();
        assert_eq!(r.rows.len(), 3);
        assert_eq!(r.rows[0].values[0], Value::Text("apple".into()));
        assert_eq!(r.rows[1].values[0], Value::Text("banana".into()));
        assert_eq!(r.rows[2].values[0], Value::Text("cherry".into()));
    }

    #[test]
    fn test_create_index_if_not_exists() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE t (a TEXT)").unwrap();
        db.execute("CREATE INDEX idx_t_a ON t(a)").unwrap();

        // Without IF NOT EXISTS, should fail.
        let result = db.execute("CREATE INDEX idx_t_a ON t(a)");
        assert!(result.is_err());

        // With IF NOT EXISTS, should succeed silently.
        db.execute("CREATE INDEX IF NOT EXISTS idx_t_a ON t(a)")
            .unwrap();
    }

    #[test]
    fn test_drop_index() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE t (a TEXT, b INTEGER)").unwrap();
        db.execute("CREATE INDEX idx_t_a ON t(a)").unwrap();

        // Index should exist.
        let entries = db.schema().unwrap();
        assert!(entries.iter().any(|e| e.name == "idx_t_a"));

        // Drop it.
        db.execute("DROP INDEX idx_t_a").unwrap();

        // Index should be gone.
        let entries = db.schema().unwrap();
        assert!(!entries.iter().any(|e| e.name == "idx_t_a"));
    }

    #[test]
    fn test_drop_index_if_exists() {
        let mut db = Database::in_memory();

        // Without IF EXISTS, should fail.
        assert!(db.execute("DROP INDEX no_such_idx").is_err());

        // With IF EXISTS, should succeed silently.
        db.execute("DROP INDEX IF EXISTS no_such_idx").unwrap();
    }

    #[test]
    fn test_index_maintained_on_insert() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
            .unwrap();
        db.execute("CREATE INDEX idx_t_val ON t(val)").unwrap();

        // Insert rows after index creation - index should be maintained.
        db.execute("INSERT INTO t VALUES (1, 'hello')").unwrap();
        db.execute("INSERT INTO t VALUES (2, 'world')").unwrap();

        // Verify the index has entries by scanning it.
        let entries = db.schema().unwrap();
        let idx_entry = entries.iter().find(|e| e.name == "idx_t_val").unwrap();
        let idx_root = idx_entry.rootpage as u32;

        // Read index entries.
        let mut cursor = btree::BTreeCursor::new(idx_root);
        cursor.move_to_first(&mut db.pager).unwrap();
        let mut index_entries = Vec::new();
        while cursor.is_valid() {
            let payload = cursor.current_payload(&mut db.pager).unwrap();
            let vals = record::decode_record(&payload).unwrap();
            index_entries.push(vals);
            cursor.move_to_next(&mut db.pager).unwrap();
        }

        // Should have 2 index entries (one per row).
        assert_eq!(index_entries.len(), 2);
        // Each entry should be (val_text, table_rowid).
        assert!(index_entries
            .iter()
            .any(|v| v[0] == Value::Text("hello".into())));
        assert!(index_entries
            .iter()
            .any(|v| v[0] == Value::Text("world".into())));
    }

    #[test]
    fn test_index_maintained_on_delete() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
            .unwrap();
        db.execute("INSERT INTO t VALUES (1, 'alpha')").unwrap();
        db.execute("INSERT INTO t VALUES (2, 'beta')").unwrap();
        db.execute("INSERT INTO t VALUES (3, 'gamma')").unwrap();
        db.execute("CREATE INDEX idx_t_val ON t(val)").unwrap();

        // Delete a row.
        db.execute("DELETE FROM t WHERE id = 2").unwrap();

        // Verify index no longer contains the deleted entry.
        let entries = db.schema().unwrap();
        let idx_entry = entries.iter().find(|e| e.name == "idx_t_val").unwrap();
        let idx_root = idx_entry.rootpage as u32;

        let mut cursor = btree::BTreeCursor::new(idx_root);
        cursor.move_to_first(&mut db.pager).unwrap();
        let mut index_entries = Vec::new();
        while cursor.is_valid() {
            let payload = cursor.current_payload(&mut db.pager).unwrap();
            let vals = record::decode_record(&payload).unwrap();
            index_entries.push(vals);
            cursor.move_to_next(&mut db.pager).unwrap();
        }

        assert_eq!(index_entries.len(), 2);
        assert!(!index_entries
            .iter()
            .any(|v| v[0] == Value::Text("beta".into())));
        assert!(index_entries
            .iter()
            .any(|v| v[0] == Value::Text("alpha".into())));
        assert!(index_entries
            .iter()
            .any(|v| v[0] == Value::Text("gamma".into())));
    }

    #[test]
    fn test_index_maintained_on_update() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
            .unwrap();
        db.execute("INSERT INTO t VALUES (1, 'old_a')").unwrap();
        db.execute("INSERT INTO t VALUES (2, 'old_b')").unwrap();
        db.execute("CREATE INDEX idx_t_val ON t(val)").unwrap();

        // Update a row.
        db.execute("UPDATE t SET val = 'new_b' WHERE id = 2")
            .unwrap();

        // Verify the index reflects the update.
        let entries = db.schema().unwrap();
        let idx_entry = entries.iter().find(|e| e.name == "idx_t_val").unwrap();
        let idx_root = idx_entry.rootpage as u32;

        let mut cursor = btree::BTreeCursor::new(idx_root);
        cursor.move_to_first(&mut db.pager).unwrap();
        let mut index_vals: Vec<Value> = Vec::new();
        while cursor.is_valid() {
            let payload = cursor.current_payload(&mut db.pager).unwrap();
            let vals = record::decode_record(&payload).unwrap();
            index_vals.push(vals[0].clone());
            cursor.move_to_next(&mut db.pager).unwrap();
        }

        assert_eq!(index_vals.len(), 2);
        assert!(index_vals.contains(&Value::Text("old_a".into())));
        assert!(index_vals.contains(&Value::Text("new_b".into())));
        assert!(!index_vals.contains(&Value::Text("old_b".into())));
    }

    #[test]
    fn test_create_unique_index() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, email TEXT)")
            .unwrap();
        db.execute("INSERT INTO t VALUES (1, 'a@b.com')").unwrap();
        db.execute("INSERT INTO t VALUES (2, 'c@d.com')").unwrap();

        db.execute("CREATE UNIQUE INDEX idx_t_email ON t(email)")
            .unwrap();

        // Schema should show UNIQUE.
        let entries = db.schema().unwrap();
        let idx = entries.iter().find(|e| e.name == "idx_t_email").unwrap();
        assert!(idx.sql.as_ref().unwrap().contains("UNIQUE"));
    }

    #[test]
    fn test_create_multi_column_index() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE t (a TEXT, b INTEGER, c REAL)")
            .unwrap();
        db.execute("INSERT INTO t VALUES ('x', 1, 1.0)").unwrap();
        db.execute("INSERT INTO t VALUES ('y', 2, 2.0)").unwrap();

        db.execute("CREATE INDEX idx_t_ab ON t(a, b)").unwrap();

        // Verify the index has entries.
        let entries = db.schema().unwrap();
        let idx_entry = entries.iter().find(|e| e.name == "idx_t_ab").unwrap();
        let idx_root = idx_entry.rootpage as u32;

        let mut cursor = btree::BTreeCursor::new(idx_root);
        cursor.move_to_first(&mut db.pager).unwrap();
        let mut count = 0;
        while cursor.is_valid() {
            let payload = cursor.current_payload(&mut db.pager).unwrap();
            let vals = record::decode_record(&payload).unwrap();
            // Each entry: (a, b, rowid) = 3 values.
            assert_eq!(vals.len(), 3);
            count += 1;
            cursor.move_to_next(&mut db.pager).unwrap();
        }
        assert_eq!(count, 2);
    }

    #[test]
    fn test_begin_commit() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
            .unwrap();

        db.execute("BEGIN").unwrap();
        db.execute("INSERT INTO t VALUES (1, 'a')").unwrap();
        db.execute("INSERT INTO t VALUES (2, 'b')").unwrap();
        db.execute("COMMIT").unwrap();

        let r = db.execute("SELECT val FROM t ORDER BY id").unwrap();
        assert_eq!(r.rows.len(), 2);
        assert_eq!(r.rows[0].values[0], Value::Text("a".into()));
        assert_eq!(r.rows[1].values[0], Value::Text("b".into()));
    }

    #[test]
    fn test_rollback_discards_changes() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
            .unwrap();
        db.execute("INSERT INTO t VALUES (1, 'original')").unwrap();

        db.execute("BEGIN").unwrap();
        db.execute("INSERT INTO t VALUES (2, 'should_be_gone')")
            .unwrap();
        db.execute("UPDATE t SET val = 'modified' WHERE id = 1")
            .unwrap();
        db.execute("ROLLBACK").unwrap();

        let r = db.execute("SELECT val FROM t ORDER BY id").unwrap();
        assert_eq!(r.rows.len(), 1);
        assert_eq!(r.rows[0].values[0], Value::Text("original".into()));
    }

    #[test]
    fn test_auto_commit_on_error() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
            .unwrap();
        db.execute("INSERT INTO t VALUES (1, 'one')").unwrap();

        // This should auto-rollback because the table already exists.
        let _ = db.execute("CREATE TABLE t (x TEXT)");

        // Original table should still be intact.
        let r = db.execute("SELECT val FROM t").unwrap();
        assert_eq!(r.rows.len(), 1);
        assert_eq!(r.rows[0].values[0], Value::Text("one".into()));
    }

    #[test]
    fn test_double_begin_fails() {
        let mut db = Database::in_memory();
        db.execute("BEGIN").unwrap();
        assert!(db.execute("BEGIN").is_err());
    }

    #[test]
    fn test_commit_without_begin_fails() {
        let mut db = Database::in_memory();
        assert!(db.execute("COMMIT").is_err());
    }

    #[test]
    fn test_rollback_without_begin_fails() {
        let mut db = Database::in_memory();
        assert!(db.execute("ROLLBACK").is_err());
    }

    #[test]
    fn test_transaction_with_multiple_tables() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE a (id INTEGER PRIMARY KEY, val TEXT)")
            .unwrap();
        db.execute("CREATE TABLE b (id INTEGER PRIMARY KEY, ref_id INTEGER)")
            .unwrap();

        db.execute("BEGIN").unwrap();
        db.execute("INSERT INTO a VALUES (1, 'hello')").unwrap();
        db.execute("INSERT INTO b VALUES (1, 1)").unwrap();
        db.execute("COMMIT").unwrap();

        let r = db
            .execute("SELECT a.val FROM a INNER JOIN b ON a.id = b.ref_id")
            .unwrap();
        assert_eq!(r.rows.len(), 1);
        assert_eq!(r.rows[0].values[0], Value::Text("hello".into()));
    }

    #[test]
    fn test_rollback_file_backed() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("txn_test.db");

        {
            let mut db = Database::open(&db_path).unwrap();
            db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
                .unwrap();
            db.execute("INSERT INTO t VALUES (1, 'persisted')").unwrap();

            db.execute("BEGIN").unwrap();
            db.execute("INSERT INTO t VALUES (2, 'rolled_back')")
                .unwrap();
            db.execute("ROLLBACK").unwrap();

            let r = db.execute("SELECT val FROM t").unwrap();
            assert_eq!(r.rows.len(), 1);
            assert_eq!(r.rows[0].values[0], Value::Text("persisted".into()));
        }

        // Reopen and verify.
        {
            let mut db = Database::open(&db_path).unwrap();
            let r = db.execute("SELECT val FROM t").unwrap();
            assert_eq!(r.rows.len(), 1);
            assert_eq!(r.rows[0].values[0], Value::Text("persisted".into()));
        }
    }

    // -- Constraint enforcement tests --

    #[test]
    fn test_not_null_constraint_insert() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
            .unwrap();
        // Valid insert should work.
        db.execute("INSERT INTO t VALUES (1, 'Alice')").unwrap();
        // NULL in NOT NULL column should fail.
        let err = db.execute("INSERT INTO t VALUES (2, NULL)").unwrap_err();
        assert!(err.to_string().contains("NOT NULL constraint failed"));
    }

    #[test]
    fn test_not_null_constraint_update() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
            .unwrap();
        db.execute("INSERT INTO t VALUES (1, 'Alice')").unwrap();
        let err = db
            .execute("UPDATE t SET name = NULL WHERE id = 1")
            .unwrap_err();
        assert!(err.to_string().contains("NOT NULL constraint failed"));
    }

    #[test]
    fn test_default_value_applied() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, status TEXT DEFAULT 'active', count INTEGER DEFAULT 0)")
            .unwrap();
        db.execute("INSERT INTO t (id) VALUES (1)").unwrap();
        let r = db
            .execute("SELECT status, count FROM t WHERE id = 1")
            .unwrap();
        assert_eq!(r.rows[0].values[0], Value::Text("active".into()));
        assert_eq!(r.rows[0].values[1], Value::Integer(0));
    }

    #[test]
    fn test_default_value_not_applied_when_explicit() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, status TEXT DEFAULT 'active')")
            .unwrap();
        db.execute("INSERT INTO t VALUES (1, 'inactive')").unwrap();
        let r = db.execute("SELECT status FROM t WHERE id = 1").unwrap();
        assert_eq!(r.rows[0].values[0], Value::Text("inactive".into()));
    }

    #[test]
    fn test_not_null_with_default() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT NOT NULL DEFAULT 'hello')")
            .unwrap();
        // Omitting the column should use the default, not fail NOT NULL.
        db.execute("INSERT INTO t (id) VALUES (1)").unwrap();
        let r = db.execute("SELECT val FROM t WHERE id = 1").unwrap();
        assert_eq!(r.rows[0].values[0], Value::Text("hello".into()));
    }

    #[test]
    fn test_pragma_table_info_with_types() {
        let mut db = Database::in_memory();
        db.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT NOT NULL, age INTEGER DEFAULT 0)",
        )
        .unwrap();
        let r = db.execute("PRAGMA table_info(t)").unwrap();
        assert_eq!(r.rows.len(), 3);
        // id column: type=INTEGER, notnull=1, pk=1
        assert_eq!(r.rows[0].values[1], Value::Text("id".into()));
        assert_eq!(r.rows[0].values[2], Value::Text("INTEGER".into()));
        assert_eq!(r.rows[0].values[5], Value::Integer(1)); // pk
                                                            // name column: notnull=1
        assert_eq!(r.rows[1].values[3], Value::Integer(1)); // notnull
                                                            // age column: default=0
        assert_eq!(r.rows[2].values[4], Value::Integer(0)); // default
    }

    // -- ALTER TABLE tests --

    #[test]
    fn test_alter_table_rename() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE old_name (id INTEGER PRIMARY KEY, val TEXT)")
            .unwrap();
        db.execute("INSERT INTO old_name VALUES (1, 'hello')")
            .unwrap();
        db.execute("ALTER TABLE old_name RENAME TO new_name")
            .unwrap();
        // Old name should not work.
        assert!(db.execute("SELECT * FROM old_name").is_err());
        // New name should work and preserve data.
        let r = db.execute("SELECT val FROM new_name").unwrap();
        assert_eq!(r.rows[0].values[0], Value::Text("hello".into()));
    }

    #[test]
    fn test_alter_table_add_column() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)")
            .unwrap();
        db.execute("INSERT INTO t VALUES (1, 'Alice')").unwrap();
        db.execute("ALTER TABLE t ADD COLUMN age INTEGER").unwrap();
        // Existing rows should have NULL for the new column.
        let r = db.execute("SELECT age FROM t WHERE id = 1").unwrap();
        assert_eq!(r.rows[0].values[0], Value::Null);
        // New inserts should include the column.
        db.execute("INSERT INTO t VALUES (2, 'Bob', 30)").unwrap();
        let r = db.execute("SELECT age FROM t WHERE id = 2").unwrap();
        assert_eq!(r.rows[0].values[0], Value::Integer(30));
    }

    #[test]
    fn test_alter_table_rename_column() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, old_col TEXT)")
            .unwrap();
        db.execute("INSERT INTO t VALUES (1, 'test')").unwrap();
        db.execute("ALTER TABLE t RENAME COLUMN old_col TO new_col")
            .unwrap();
        let r = db.execute("SELECT new_col FROM t WHERE id = 1").unwrap();
        assert_eq!(r.rows[0].values[0], Value::Text("test".into()));
    }

    #[test]
    fn test_alter_table_drop_column() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, a TEXT, b TEXT)")
            .unwrap();
        db.execute("INSERT INTO t VALUES (1, 'keep', 'drop')")
            .unwrap();
        db.execute("ALTER TABLE t DROP COLUMN b").unwrap();
        let r = db.execute("SELECT * FROM t WHERE id = 1").unwrap();
        assert_eq!(r.rows[0].values.len(), 2); // id + a
        assert_eq!(r.rows[0].values[1], Value::Text("keep".into()));
    }

    // -- INSERT...SELECT tests --

    #[test]
    fn test_insert_select() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE src (id INTEGER PRIMARY KEY, val TEXT)")
            .unwrap();
        db.execute("INSERT INTO src VALUES (1, 'a')").unwrap();
        db.execute("INSERT INTO src VALUES (2, 'b')").unwrap();
        db.execute("CREATE TABLE dst (id INTEGER PRIMARY KEY, val TEXT)")
            .unwrap();
        db.execute("INSERT INTO dst SELECT * FROM src").unwrap();
        let r = db.execute("SELECT COUNT(*) FROM dst").unwrap();
        assert_eq!(r.rows[0].values[0], Value::Integer(2));
    }

    // -- UNION / INTERSECT / EXCEPT tests --

    #[test]
    fn test_union_all() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE t (v INTEGER)").unwrap();
        db.execute("INSERT INTO t VALUES (1)").unwrap();
        db.execute("INSERT INTO t VALUES (2)").unwrap();
        let r = db
            .execute("SELECT v FROM t UNION ALL SELECT v FROM t")
            .unwrap();
        assert_eq!(r.rows.len(), 4);
    }

    #[test]
    fn test_union_dedup() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE t (v INTEGER)").unwrap();
        db.execute("INSERT INTO t VALUES (1)").unwrap();
        db.execute("INSERT INTO t VALUES (2)").unwrap();
        let r = db.execute("SELECT v FROM t UNION SELECT v FROM t").unwrap();
        assert_eq!(r.rows.len(), 2);
    }

    #[test]
    fn test_intersect() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE a (v INTEGER)").unwrap();
        db.execute("CREATE TABLE b (v INTEGER)").unwrap();
        db.execute("INSERT INTO a VALUES (1)").unwrap();
        db.execute("INSERT INTO a VALUES (2)").unwrap();
        db.execute("INSERT INTO a VALUES (3)").unwrap();
        db.execute("INSERT INTO b VALUES (2)").unwrap();
        db.execute("INSERT INTO b VALUES (3)").unwrap();
        db.execute("INSERT INTO b VALUES (4)").unwrap();
        let r = db
            .execute("SELECT v FROM a INTERSECT SELECT v FROM b")
            .unwrap();
        assert_eq!(r.rows.len(), 2);
    }

    #[test]
    fn test_except() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE a (v INTEGER)").unwrap();
        db.execute("CREATE TABLE b (v INTEGER)").unwrap();
        db.execute("INSERT INTO a VALUES (1)").unwrap();
        db.execute("INSERT INTO a VALUES (2)").unwrap();
        db.execute("INSERT INTO a VALUES (3)").unwrap();
        db.execute("INSERT INTO b VALUES (2)").unwrap();
        let r = db
            .execute("SELECT v FROM a EXCEPT SELECT v FROM b")
            .unwrap();
        assert_eq!(r.rows.len(), 2);
    }

    // -- Expression LIMIT/OFFSET tests --

    #[test]
    fn test_expression_limit() {
        let mut db = Database::in_memory();
        db.execute("CREATE TABLE t (v INTEGER)").unwrap();
        for i in 1..=10 {
            db.execute(&format!("INSERT INTO t VALUES ({i})")).unwrap();
        }
        let r = db
            .execute("SELECT v FROM t ORDER BY v LIMIT 1 + 2")
            .unwrap();
        assert_eq!(r.rows.len(), 3);
    }
}
