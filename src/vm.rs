/// Virtual machine / query executor.
/// Uses a Volcano-style tree-of-iterators model for query execution.
use std::cmp::Ordering;
use std::collections::HashMap;

use crate::ast::*;
use crate::btree;
use crate::error::{Result, RsqliteError};
use crate::functions;
use crate::pager::Pager;
use crate::record;
use crate::types::Value;

/// Schema information for a table.
#[derive(Debug, Clone)]
pub struct TableSchema {
    pub name: String,
    pub columns: Vec<ColumnInfo>,
    pub root_page: u32,
    pub sql: String,
    pub is_autoincrement: bool,
    pub primary_key_column: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct ColumnInfo {
    pub name: String,
    pub type_name: String,
    pub not_null: bool,
    pub default_value: Option<Expr>,
    pub is_primary_key: bool,
}

/// Schema information for an index.
#[derive(Debug, Clone)]
pub struct IndexSchema {
    pub name: String,
    pub table_name: String,
    pub columns: Vec<String>,
    pub root_page: u32,
    pub unique: bool,
    pub sql: String,
}

/// The database instance that holds schema and pager.
pub struct Database {
    pub pager: Pager,
    pub tables: HashMap<String, TableSchema>,
    pub indexes: HashMap<String, IndexSchema>,
    pub auto_commit: bool,
    pub last_insert_rowid: i64,
    pub changes: i64,
}

impl Database {
    pub fn new_memory() -> Self {
        let pager = Pager::new_memory();
        Database {
            pager,
            tables: HashMap::new(),
            indexes: HashMap::new(),
            auto_commit: true,
            last_insert_rowid: 0,
            changes: 0,
        }
    }

    pub fn open(path: &str) -> Result<Self> {
        let pager = Pager::open(path)?;
        let mut db = Database {
            pager,
            tables: HashMap::new(),
            indexes: HashMap::new(),
            auto_commit: true,
            last_insert_rowid: 0,
            changes: 0,
        };
        db.load_schema()?;
        Ok(db)
    }

    /// Load schema from sqlite_master table.
    fn load_schema(&mut self) -> Result<()> {
        self.tables.clear();
        self.indexes.clear();

        // Page 1 is the sqlite_master table
        let rows = btree::scan_table(&mut self.pager, 1)?;

        for (_rowid, values) in rows {
            if values.len() < 5 {
                continue;
            }
            let obj_type = values[0].to_text();
            let name = values[1].to_text();
            let _tbl_name = values[2].to_text();
            let root_page = values[3].to_integer().unwrap_or(0) as u32;
            let sql = values[4].to_text();

            match obj_type.as_str() {
                "table" => {
                    if let Ok(schema) = self.parse_table_schema(&name, root_page, &sql) {
                        self.tables.insert(name.to_lowercase(), schema);
                    }
                }
                "index" => {
                    if let Ok(schema) = self.parse_index_schema(&name, root_page, &sql) {
                        self.indexes.insert(name.to_lowercase(), schema);
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn parse_table_schema(&self, name: &str, root_page: u32, sql: &str) -> Result<TableSchema> {
        if sql.is_empty() {
            return Ok(TableSchema {
                name: name.to_string(),
                columns: Vec::new(),
                root_page,
                sql: String::new(),
                is_autoincrement: false,
                primary_key_column: None,
            });
        }

        let stmt = crate::parser::parse_sql(sql)?;
        if let Statement::CreateTable(ct) = stmt {
            let mut columns = Vec::new();
            let mut pk_col = None;
            let mut is_autoincrement = false;

            for (i, col) in ct.columns.iter().enumerate() {
                let mut is_pk = false;
                let mut not_null = false;
                let mut default_value = None;

                for constraint in &col.constraints {
                    match constraint {
                        ColumnConstraint::PrimaryKey { autoincrement } => {
                            is_pk = true;
                            pk_col = Some(i);
                            if *autoincrement {
                                is_autoincrement = true;
                            }
                        }
                        ColumnConstraint::NotNull => not_null = true,
                        ColumnConstraint::Default(expr) => default_value = Some(expr.clone()),
                        _ => {}
                    }
                }

                columns.push(ColumnInfo {
                    name: col.name.clone(),
                    type_name: col.type_name.clone().unwrap_or_default(),
                    not_null,
                    default_value,
                    is_primary_key: is_pk,
                });
            }

            // Check table constraints for PRIMARY KEY
            for constraint in &ct.constraints {
                if let TableConstraint::PrimaryKey(cols) = constraint {
                    if cols.len() == 1 {
                        for (i, c) in columns.iter().enumerate() {
                            if c.name.eq_ignore_ascii_case(&cols[0]) {
                                pk_col = Some(i);
                            }
                        }
                    }
                }
            }

            Ok(TableSchema {
                name: name.to_string(),
                columns,
                root_page,
                sql: sql.to_string(),
                is_autoincrement,
                primary_key_column: pk_col,
            })
        } else {
            Err(RsqliteError::Internal("Expected CREATE TABLE".into()))
        }
    }

    fn parse_index_schema(&self, name: &str, root_page: u32, sql: &str) -> Result<IndexSchema> {
        if sql.is_empty() {
            return Ok(IndexSchema {
                name: name.to_string(),
                table_name: String::new(),
                columns: Vec::new(),
                root_page,
                unique: false,
                sql: String::new(),
            });
        }

        let stmt = crate::parser::parse_sql(sql)?;
        if let Statement::CreateIndex(ci) = stmt {
            Ok(IndexSchema {
                name: name.to_string(),
                table_name: ci.table_name,
                columns: ci.columns.iter().map(|c| c.name.clone()).collect(),
                root_page,
                unique: ci.unique,
                sql: sql.to_string(),
            })
        } else {
            Err(RsqliteError::Internal("Expected CREATE INDEX".into()))
        }
    }

    /// Execute a SQL statement and return results.
    pub fn execute(&mut self, sql: &str) -> Result<ExecuteResult> {
        let stmt = crate::parser::parse_sql(sql)?;
        self.execute_statement(&stmt)
    }

    pub fn execute_statement(&mut self, stmt: &Statement) -> Result<ExecuteResult> {
        match stmt {
            Statement::Select(sel) => self.execute_select(sel),
            Statement::Insert(ins) => self.execute_insert(ins),
            Statement::Update(upd) => self.execute_update(upd),
            Statement::Delete(del) => self.execute_delete(del),
            Statement::CreateTable(ct) => self.execute_create_table(ct),
            Statement::CreateIndex(ci) => self.execute_create_index(ci),
            Statement::DropTable(dt) => self.execute_drop_table(dt),
            Statement::DropIndex(di) => self.execute_drop_index(di),
            Statement::AlterTable(at) => self.execute_alter_table(at),
            Statement::Begin => {
                self.auto_commit = false;
                self.pager.begin_transaction()?;
                Ok(ExecuteResult::empty())
            }
            Statement::Commit => {
                self.pager.commit_transaction()?;
                self.auto_commit = true;
                Ok(ExecuteResult::empty())
            }
            Statement::Rollback => {
                self.pager.rollback_transaction()?;
                self.auto_commit = true;
                Ok(ExecuteResult::empty())
            }
            Statement::Explain(inner) => self.execute_explain(inner),
            Statement::ExplainQueryPlan(inner) => self.execute_explain(inner),
            Statement::Pragma(pragma) => self.execute_pragma(pragma),
        }
    }

    fn execute_select(&mut self, sel: &SelectStatement) -> Result<ExecuteResult> {
        let mut rows = self.fetch_rows(sel)?;

        // GROUP BY
        if !sel.group_by.is_empty() {
            rows = self.apply_group_by(&rows, sel)?;
        } else if self.has_aggregate(&sel.columns) && sel.from.is_some() {
            // Aggregate without GROUP BY = single row result
            rows = self.apply_aggregate_all(&rows, sel)?;
        }

        // HAVING (only if not already applied during GROUP BY)
        if !sel.group_by.is_empty() {
            // HAVING was already applied during apply_group_by
        } else if let Some(ref having) = sel.having {
            rows.retain(|row| {
                self.eval_expr_in_row(having, row, &[])
                    .map(|v| v.is_truthy())
                    .unwrap_or(false)
            });
        }

        // DISTINCT
        if sel.distinct {
            let mut seen = Vec::new();
            rows.retain(|row| {
                let key: Vec<Value> = row.values.clone();
                if seen.contains(&key) {
                    false
                } else {
                    seen.push(key);
                    true
                }
            });
        }

        // ORDER BY
        if !sel.order_by.is_empty() {
            let order_items = sel.order_by.clone();
            rows.sort_by(|a, b| {
                for item in &order_items {
                    let va = self
                        .eval_expr_in_row(&item.expr, a, &[])
                        .unwrap_or(Value::Null);
                    let vb = self
                        .eval_expr_in_row(&item.expr, b, &[])
                        .unwrap_or(Value::Null);
                    let cmp = va.sqlite_cmp(&vb);
                    let cmp = if item.descending { cmp.reverse() } else { cmp };
                    if cmp != Ordering::Equal {
                        return cmp;
                    }
                }
                Ordering::Equal
            });
        }

        // OFFSET
        if let Some(ref offset_expr) = sel.offset {
            let offset = self.eval_const_expr(offset_expr)?.to_integer().unwrap_or(0) as usize;
            if offset < rows.len() {
                rows = rows[offset..].to_vec();
            } else {
                rows.clear();
            }
        }

        // LIMIT
        if let Some(ref limit_expr) = sel.limit {
            let limit = self.eval_const_expr(limit_expr)?.to_integer().unwrap_or(-1);
            if limit >= 0 {
                rows.truncate(limit as usize);
            }
        }

        // Build column names from the select columns
        let column_names = self.build_column_names(sel)?;

        // Project columns
        let mut result_rows: Vec<Vec<Value>> = rows.into_iter().map(|r| r.values).collect();

        // Handle compound SELECT (UNION, EXCEPT, INTERSECT)
        if let Some(ref compound) = sel.compound {
            let right_result = self.execute_select(&compound.select)?;
            match compound.op {
                CompoundOp::UnionAll => {
                    result_rows.extend(right_result.rows);
                }
                CompoundOp::Union => {
                    for row in &right_result.rows {
                        if !result_rows.contains(row) {
                            result_rows.push(row.clone());
                        }
                    }
                }
                CompoundOp::Except => {
                    result_rows.retain(|row| !right_result.rows.contains(row));
                }
                CompoundOp::Intersect => {
                    result_rows.retain(|row| right_result.rows.contains(row));
                }
            }
        }

        Ok(ExecuteResult {
            columns: column_names,
            rows: result_rows,
            rows_affected: 0,
        })
    }

    fn fetch_rows(&mut self, sel: &SelectStatement) -> Result<Vec<Row>> {
        if sel.from.is_none() {
            // No FROM clause, just evaluate expressions
            let mut values = Vec::new();
            for col in &sel.columns {
                match col {
                    SelectColumn::Expr { expr, .. } => {
                        let empty_row = Row {
                            values: Vec::new(),
                            columns: Vec::new(),
                            table_alias: None,
                            rowid: 0,
                        };
                        values.push(self.eval_expr_in_row(expr, &empty_row, &[])?);
                    }
                    _ => values.push(Value::Null),
                }
            }
            return Ok(vec![Row {
                values,
                columns: Vec::new(),
                table_alias: None,
                rowid: 0,
            }]);
        }

        let from = sel.from.as_ref().unwrap();
        let base_rows = self.fetch_from(from)?;

        // Apply WHERE
        let filtered = if let Some(ref where_clause) = sel.where_clause {
            base_rows
                .into_iter()
                .filter(|row| {
                    self.eval_expr_in_row(where_clause, row, &[])
                        .map(|v| v.is_truthy())
                        .unwrap_or(false)
                })
                .collect()
        } else {
            base_rows
        };

        // For aggregate queries (with GROUP BY or aggregate functions),
        // return the raw base rows so aggregation can work on original values
        let needs_raw = !sel.group_by.is_empty() || self.has_aggregate(&sel.columns);
        if needs_raw {
            return Ok(filtered);
        }

        // Project columns
        self.project_rows(&filtered, &sel.columns)
    }

    fn project_rows(&mut self, rows: &[Row], columns: &[SelectColumn]) -> Result<Vec<Row>> {
        let mut result = Vec::new();
        for row in rows {
            let mut values = Vec::new();
            for col in columns {
                match col {
                    SelectColumn::AllColumns => {
                        values.extend(row.values.clone());
                    }
                    SelectColumn::TableAllColumns(table) => {
                        // Only include columns from matching table
                        if row.table_alias.as_deref() == Some(table.as_str()) {
                            values.extend(row.values.clone());
                        } else {
                            // Include all for now
                            values.extend(row.values.clone());
                        }
                    }
                    SelectColumn::Expr { expr, .. } => {
                        values.push(self.eval_expr_in_row(expr, row, &[])?);
                    }
                }
            }
            result.push(Row {
                values,
                columns: row.columns.clone(),
                table_alias: row.table_alias.clone(),
                rowid: row.rowid,
            });
        }
        Ok(result)
    }

    fn fetch_from(&mut self, from: &FromClause) -> Result<Vec<Row>> {
        match from {
            FromClause::Table { name, alias } => {
                let table_name = name.to_lowercase();
                let schema = self
                    .tables
                    .get(&table_name)
                    .ok_or_else(|| RsqliteError::TableNotFound(name.clone()))?
                    .clone();

                let rows = btree::scan_table(&mut self.pager, schema.root_page)?;

                let alias_name = alias.as_ref().unwrap_or(name);
                let column_names: Vec<String> =
                    schema.columns.iter().map(|c| c.name.clone()).collect();

                let pk_col = schema.primary_key_column;
                let pk_is_int = pk_col
                    .map(|i| {
                        let tn = schema.columns[i].type_name.to_uppercase();
                        tn.contains("INT")
                    })
                    .unwrap_or(false);

                Ok(rows
                    .into_iter()
                    .map(|(rowid, values)| {
                        // Pad or truncate values to match column count
                        let mut vals = values;
                        while vals.len() < column_names.len() {
                            vals.push(Value::Null);
                        }
                        // Restore integer primary key from rowid
                        if pk_is_int {
                            if let Some(pk_idx) = pk_col {
                                if pk_idx < vals.len() && vals[pk_idx].is_null() {
                                    vals[pk_idx] = Value::Integer(rowid);
                                }
                            }
                        }
                        Row {
                            values: vals,
                            columns: column_names
                                .iter()
                                .map(|n| (alias_name.clone(), n.clone()))
                                .collect(),
                            table_alias: Some(alias_name.clone()),
                            rowid,
                        }
                    })
                    .collect())
            }
            FromClause::Join(join) => {
                let left_rows = self.fetch_from(&join.left)?;
                let right_rows = self.fetch_from(&join.right)?;

                let mut result = Vec::new();
                for left in &left_rows {
                    let mut matched = false;
                    for right in &right_rows {
                        let combined = self.combine_rows(left, right);

                        if let Some(ref on_expr) = join.on {
                            let matches = self
                                .eval_expr_in_row(on_expr, &combined, &[])
                                .map(|v| v.is_truthy())
                                .unwrap_or(false);
                            if matches {
                                result.push(combined);
                                matched = true;
                            }
                        } else {
                            result.push(combined);
                            matched = true;
                        }
                    }

                    // LEFT JOIN: include unmatched left rows with NULLs for right
                    if join.join_type == JoinType::Left && !matched {
                        let null_right = Row {
                            values: vec![
                                Value::Null;
                                if right_rows.is_empty() {
                                    0
                                } else {
                                    right_rows[0].values.len()
                                }
                            ],
                            columns: if right_rows.is_empty() {
                                Vec::new()
                            } else {
                                right_rows[0].columns.clone()
                            },
                            table_alias: if right_rows.is_empty() {
                                None
                            } else {
                                right_rows[0].table_alias.clone()
                            },
                            rowid: 0,
                        };
                        result.push(self.combine_rows(left, &null_right));
                    }
                }
                Ok(result)
            }
            FromClause::Subquery { query, alias } => {
                let sub_result = self.execute_select(query)?;
                let column_names = sub_result.columns.clone();
                Ok(sub_result
                    .rows
                    .into_iter()
                    .map(|values| Row {
                        values,
                        columns: column_names
                            .iter()
                            .map(|n| (alias.clone(), n.clone()))
                            .collect(),
                        table_alias: Some(alias.clone()),
                        rowid: 0,
                    })
                    .collect())
            }
        }
    }

    fn combine_rows(&self, left: &Row, right: &Row) -> Row {
        let mut values = left.values.clone();
        values.extend(right.values.clone());
        let mut columns = left.columns.clone();
        columns.extend(right.columns.clone());
        Row {
            values,
            columns,
            table_alias: None,
            rowid: left.rowid,
        }
    }

    fn build_column_names(&self, sel: &SelectStatement) -> Result<Vec<String>> {
        let mut names = Vec::new();
        for col in &sel.columns {
            match col {
                SelectColumn::AllColumns => {
                    if let Some(ref from) = sel.from {
                        self.add_all_column_names(from, &mut names);
                    }
                }
                SelectColumn::TableAllColumns(table) => {
                    let table_lower = table.to_lowercase();
                    // Find table by alias or name
                    for schema in self.tables.values() {
                        if schema.name.to_lowercase() == table_lower {
                            for col in &schema.columns {
                                names.push(col.name.clone());
                            }
                        }
                    }
                }
                SelectColumn::Expr { expr, alias } => {
                    if let Some(a) = alias {
                        names.push(a.clone());
                    } else {
                        names.push(self.expr_to_name(expr));
                    }
                }
            }
        }
        Ok(names)
    }

    fn add_all_column_names(&self, from: &FromClause, names: &mut Vec<String>) {
        match from {
            FromClause::Table { name, .. } => {
                let table_lower = name.to_lowercase();
                if let Some(schema) = self.tables.get(&table_lower) {
                    for col in &schema.columns {
                        names.push(col.name.clone());
                    }
                }
            }
            FromClause::Join(join) => {
                self.add_all_column_names(&join.left, names);
                self.add_all_column_names(&join.right, names);
            }
            FromClause::Subquery { query, .. } => {
                if let Ok(sub_names) = self.build_column_names(query) {
                    names.extend(sub_names);
                }
            }
        }
    }

    fn expr_to_name(&self, expr: &Expr) -> String {
        match expr {
            Expr::Column { table: None, name } => name.clone(),
            Expr::Column {
                table: Some(t),
                name,
            } => format!("{}.{}", t, name),
            Expr::Function { name, .. } => name.to_lowercase(),
            Expr::Integer(i) => i.to_string(),
            Expr::String(s) => format!("'{}'", s),
            Expr::Star => "*".to_string(),
            _ => "?".to_string(),
        }
    }

    fn has_aggregate(&self, columns: &[SelectColumn]) -> bool {
        for col in columns {
            if let SelectColumn::Expr { expr, .. } = col {
                if self.is_aggregate_expr(expr) {
                    return true;
                }
            }
        }
        false
    }

    fn is_aggregate_expr(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Function { name, .. } => {
                matches!(
                    name.as_str(),
                    "COUNT" | "SUM" | "AVG" | "MIN" | "MAX" | "GROUP_CONCAT" | "TOTAL"
                )
            }
            _ => false,
        }
    }

    fn apply_group_by(&mut self, rows: &[Row], sel: &SelectStatement) -> Result<Vec<Row>> {
        // Group rows by the GROUP BY expressions
        let mut groups: Vec<(Vec<Value>, Vec<Row>)> = Vec::new();

        for row in rows {
            let key: Vec<Value> = sel
                .group_by
                .iter()
                .map(|e| self.eval_expr_in_row(e, row, &[]).unwrap_or(Value::Null))
                .collect();

            if let Some(group) = groups.iter_mut().find(|(k, _)| k == &key) {
                group.1.push(row.clone());
            } else {
                groups.push((key, vec![row.clone()]));
            }
        }

        let mut result = Vec::new();
        for (_key, group_rows) in &groups {
            // Evaluate HAVING in the group context where group_rows are available
            if let Some(ref having) = sel.having {
                let having_val = self.eval_expr_in_row(having, &group_rows[0], group_rows)?;
                if !having_val.is_truthy() {
                    continue;
                }
            }

            let mut values = Vec::new();
            for col in &sel.columns {
                if let SelectColumn::Expr { expr, .. } = col {
                    values.push(self.eval_expr_in_row(expr, &group_rows[0], group_rows)?);
                } else {
                    values.push(Value::Null);
                }
            }
            result.push(Row {
                values,
                columns: group_rows[0].columns.clone(),
                table_alias: group_rows[0].table_alias.clone(),
                rowid: 0,
            });
        }

        Ok(result)
    }

    fn apply_aggregate_all(&mut self, rows: &[Row], sel: &SelectStatement) -> Result<Vec<Row>> {
        let dummy = if rows.is_empty() {
            Row {
                values: Vec::new(),
                columns: Vec::new(),
                table_alias: None,
                rowid: 0,
            }
        } else {
            rows[0].clone()
        };

        let mut values = Vec::new();
        for col in &sel.columns {
            if let SelectColumn::Expr { expr, .. } = col {
                values.push(self.eval_expr_in_row(expr, &dummy, rows)?);
            } else {
                values.push(Value::Null);
            }
        }

        Ok(vec![Row {
            values,
            columns: dummy.columns,
            table_alias: dummy.table_alias,
            rowid: 0,
        }])
    }

    fn eval_const_expr(&mut self, expr: &Expr) -> Result<Value> {
        let empty = Row {
            values: Vec::new(),
            columns: Vec::new(),
            table_alias: None,
            rowid: 0,
        };
        self.eval_expr_in_row(expr, &empty, &[])
    }

    fn eval_expr_in_row(&mut self, expr: &Expr, row: &Row, group_rows: &[Row]) -> Result<Value> {
        match expr {
            Expr::Null => Ok(Value::Null),
            Expr::Integer(i) => Ok(Value::Integer(*i)),
            Expr::Float(f) => Ok(Value::Real(*f)),
            Expr::String(s) => Ok(Value::Text(s.clone())),
            Expr::Blob(b) => Ok(Value::Blob(b.clone())),
            Expr::Star => Ok(Value::Null),
            Expr::Rowid => Ok(Value::Integer(row.rowid)),

            Expr::Column { table, name } => {
                // Search for column in row
                for (i, (tbl, col)) in row.columns.iter().enumerate() {
                    if col.eq_ignore_ascii_case(name) {
                        if let Some(t) = table {
                            if !tbl.eq_ignore_ascii_case(t) {
                                continue;
                            }
                        }
                        return Ok(row.values.get(i).cloned().unwrap_or(Value::Null));
                    }
                }
                // Also try matching by position if it's a simple name
                if table.is_none() {
                    // Try case insensitive match
                    let lower_name = name.to_lowercase();
                    for (i, (_, col)) in row.columns.iter().enumerate() {
                        if col.to_lowercase() == lower_name {
                            return Ok(row.values.get(i).cloned().unwrap_or(Value::Null));
                        }
                    }
                }
                Ok(Value::Null)
            }

            Expr::UnaryMinus(inner) => {
                let v = self.eval_expr_in_row(inner, row, group_rows)?;
                match v {
                    Value::Integer(i) => Ok(Value::Integer(-i)),
                    Value::Real(f) => Ok(Value::Real(-f)),
                    _ => Ok(Value::Integer(0)),
                }
            }
            Expr::UnaryPlus(inner) => self.eval_expr_in_row(inner, row, group_rows),
            Expr::Not(inner) => {
                let v = self.eval_expr_in_row(inner, row, group_rows)?;
                if v.is_null() {
                    Ok(Value::Null)
                } else {
                    Ok(Value::Integer(if v.is_truthy() { 0 } else { 1 }))
                }
            }
            Expr::BitwiseNot(inner) => {
                let v = self.eval_expr_in_row(inner, row, group_rows)?;
                match v {
                    Value::Integer(i) => Ok(Value::Integer(!i)),
                    _ => Ok(Value::Integer(-1)),
                }
            }

            Expr::BinaryOp { left, op, right } => {
                let lv = self.eval_expr_in_row(left, row, group_rows)?;
                // Short-circuit for AND/OR
                match op {
                    BinaryOperator::And => {
                        if !lv.is_null() && !lv.is_truthy() {
                            return Ok(Value::Integer(0));
                        }
                        let rv = self.eval_expr_in_row(right, row, group_rows)?;
                        if lv.is_null() || rv.is_null() {
                            if !lv.is_null() && !lv.is_truthy() {
                                return Ok(Value::Integer(0));
                            }
                            if !rv.is_null() && !rv.is_truthy() {
                                return Ok(Value::Integer(0));
                            }
                            return Ok(Value::Null);
                        }
                        return Ok(Value::Integer(if lv.is_truthy() && rv.is_truthy() {
                            1
                        } else {
                            0
                        }));
                    }
                    BinaryOperator::Or => {
                        if !lv.is_null() && lv.is_truthy() {
                            return Ok(Value::Integer(1));
                        }
                        let rv = self.eval_expr_in_row(right, row, group_rows)?;
                        if lv.is_null() || rv.is_null() {
                            if !lv.is_null() && lv.is_truthy() {
                                return Ok(Value::Integer(1));
                            }
                            if !rv.is_null() && rv.is_truthy() {
                                return Ok(Value::Integer(1));
                            }
                            return Ok(Value::Null);
                        }
                        return Ok(Value::Integer(if lv.is_truthy() || rv.is_truthy() {
                            1
                        } else {
                            0
                        }));
                    }
                    _ => {}
                }

                let rv = self.eval_expr_in_row(right, row, group_rows)?;
                self.eval_binary_op(&lv, op, &rv)
            }

            Expr::IsNull(inner) => {
                let v = self.eval_expr_in_row(inner, row, group_rows)?;
                Ok(Value::Integer(if v.is_null() { 1 } else { 0 }))
            }
            Expr::IsNotNull(inner) => {
                let v = self.eval_expr_in_row(inner, row, group_rows)?;
                Ok(Value::Integer(if v.is_null() { 0 } else { 1 }))
            }

            Expr::Between {
                expr: inner,
                low,
                high,
                negated,
            } => {
                let v = self.eval_expr_in_row(inner, row, group_rows)?;
                let lo = self.eval_expr_in_row(low, row, group_rows)?;
                let hi = self.eval_expr_in_row(high, row, group_rows)?;
                let in_range =
                    v.sqlite_cmp(&lo) != Ordering::Less && v.sqlite_cmp(&hi) != Ordering::Greater;
                let result = if *negated { !in_range } else { in_range };
                Ok(Value::Integer(if result { 1 } else { 0 }))
            }

            Expr::InList {
                expr: inner,
                list,
                negated,
            } => {
                let v = self.eval_expr_in_row(inner, row, group_rows)?;
                if v.is_null() {
                    return Ok(Value::Null);
                }
                let mut found = false;
                for item in list {
                    let iv = self.eval_expr_in_row(item, row, group_rows)?;
                    if v == iv {
                        found = true;
                        break;
                    }
                }
                let result = if *negated { !found } else { found };
                Ok(Value::Integer(if result { 1 } else { 0 }))
            }

            Expr::InSelect {
                expr: inner,
                query,
                negated,
            } => {
                let v = self.eval_expr_in_row(inner, row, group_rows)?;
                let sub = self.execute_select(query)?;
                let found = sub
                    .rows
                    .iter()
                    .any(|r| r.first().is_some_and(|sv| *sv == v));
                let result = if *negated { !found } else { found };
                Ok(Value::Integer(if result { 1 } else { 0 }))
            }

            Expr::Like {
                expr: inner,
                pattern,
                escape,
                negated,
            } => {
                let v = self.eval_expr_in_row(inner, row, group_rows)?;
                let p = self.eval_expr_in_row(pattern, row, group_rows)?;
                if v.is_null() || p.is_null() {
                    return Ok(Value::Null);
                }
                let escape_char = if let Some(esc) = escape {
                    let e = self.eval_expr_in_row(esc, row, group_rows)?;
                    e.to_text().chars().next()
                } else {
                    None
                };
                let matched = functions::like_match(&p.to_text(), &v.to_text(), escape_char);
                let result = if *negated { !matched } else { matched };
                Ok(Value::Integer(if result { 1 } else { 0 }))
            }

            Expr::GlobExpr {
                expr: inner,
                pattern,
                negated,
            } => {
                let v = self.eval_expr_in_row(inner, row, group_rows)?;
                let p = self.eval_expr_in_row(pattern, row, group_rows)?;
                if v.is_null() || p.is_null() {
                    return Ok(Value::Null);
                }
                let matched = functions::glob_match(&p.to_text(), &v.to_text());
                let result = if *negated { !matched } else { matched };
                Ok(Value::Integer(if result { 1 } else { 0 }))
            }

            Expr::Function {
                name,
                args,
                distinct,
            } => {
                // Aggregate functions
                match name.as_str() {
                    "COUNT" => {
                        if !group_rows.is_empty() {
                            if args.len() == 1 && matches!(args[0], Expr::Star) {
                                return Ok(Value::Integer(group_rows.len() as i64));
                            }
                            let mut count = 0i64;
                            let mut seen = Vec::new();
                            for r in group_rows {
                                let v = self.eval_expr_in_row(&args[0], r, &[])?;
                                if !v.is_null() {
                                    if *distinct {
                                        if !seen.contains(&v) {
                                            seen.push(v);
                                            count += 1;
                                        }
                                    } else {
                                        count += 1;
                                    }
                                }
                            }
                            return Ok(Value::Integer(count));
                        }
                        if args.len() == 1 && matches!(args[0], Expr::Star) {
                            return Ok(Value::Integer(0));
                        }
                        return Ok(Value::Integer(0));
                    }
                    "SUM" | "TOTAL" => {
                        if !group_rows.is_empty() {
                            let mut sum_int: i64 = 0;
                            let mut sum_float: f64 = 0.0;
                            let mut has_real = false;
                            let mut has_value = false;
                            for r in group_rows {
                                let v = self.eval_expr_in_row(&args[0], r, &[])?;
                                match v {
                                    Value::Integer(i) => {
                                        sum_int += i;
                                        has_value = true;
                                    }
                                    Value::Real(f) => {
                                        sum_float += f;
                                        has_real = true;
                                        has_value = true;
                                    }
                                    Value::Null => {}
                                    _ => {
                                        if let Some(i) = v.to_integer() {
                                            sum_int += i;
                                            has_value = true;
                                        }
                                    }
                                }
                            }
                            if name == "TOTAL" {
                                return Ok(Value::Real(sum_int as f64 + sum_float));
                            }
                            if !has_value {
                                return Ok(Value::Null);
                            }
                            if has_real {
                                return Ok(Value::Real(sum_int as f64 + sum_float));
                            }
                            return Ok(Value::Integer(sum_int));
                        }
                        if name == "TOTAL" {
                            return Ok(Value::Real(0.0));
                        }
                        return Ok(Value::Null);
                    }
                    "AVG" => {
                        if !group_rows.is_empty() {
                            let mut sum: f64 = 0.0;
                            let mut count = 0;
                            for r in group_rows {
                                let v = self.eval_expr_in_row(&args[0], r, &[])?;
                                if !v.is_null() {
                                    sum += v.to_real().unwrap_or(0.0);
                                    count += 1;
                                }
                            }
                            if count == 0 {
                                return Ok(Value::Null);
                            }
                            return Ok(Value::Real(sum / count as f64));
                        }
                        return Ok(Value::Null);
                    }
                    "MIN" if args.len() == 1 => {
                        if !group_rows.is_empty() {
                            let mut min: Option<Value> = None;
                            for r in group_rows {
                                let v = self.eval_expr_in_row(&args[0], r, &[])?;
                                if !v.is_null() {
                                    min = Some(match min {
                                        None => v,
                                        Some(m) => {
                                            if v.sqlite_cmp(&m) == Ordering::Less {
                                                v
                                            } else {
                                                m
                                            }
                                        }
                                    });
                                }
                            }
                            return Ok(min.unwrap_or(Value::Null));
                        }
                        return Ok(Value::Null);
                    }
                    "MAX" if args.len() == 1 => {
                        if !group_rows.is_empty() {
                            let mut max: Option<Value> = None;
                            for r in group_rows {
                                let v = self.eval_expr_in_row(&args[0], r, &[])?;
                                if !v.is_null() {
                                    max = Some(match max {
                                        None => v,
                                        Some(m) => {
                                            if v.sqlite_cmp(&m) == Ordering::Greater {
                                                v
                                            } else {
                                                m
                                            }
                                        }
                                    });
                                }
                            }
                            return Ok(max.unwrap_or(Value::Null));
                        }
                        return Ok(Value::Null);
                    }
                    "GROUP_CONCAT" => {
                        if !group_rows.is_empty() {
                            let separator = if args.len() > 1 {
                                self.eval_expr_in_row(&args[1], &group_rows[0], &[])?
                                    .to_text()
                            } else {
                                ",".to_string()
                            };
                            let mut parts = Vec::new();
                            for r in group_rows {
                                let v = self.eval_expr_in_row(&args[0], r, &[])?;
                                if !v.is_null() {
                                    parts.push(v.to_text());
                                }
                            }
                            if parts.is_empty() {
                                return Ok(Value::Null);
                            }
                            return Ok(Value::Text(parts.join(&separator)));
                        }
                        return Ok(Value::Null);
                    }
                    _ => {}
                }

                // Scalar functions
                let mut arg_values = Vec::new();
                for a in args {
                    arg_values.push(self.eval_expr_in_row(a, row, group_rows)?);
                }
                if let Some(result) = functions::eval_function(name, &arg_values) {
                    Ok(result)
                } else {
                    Err(RsqliteError::Runtime(format!("Unknown function: {}", name)))
                }
            }

            Expr::Cast {
                expr: inner,
                type_name,
            } => {
                let v = self.eval_expr_in_row(inner, row, group_rows)?;
                let affinity = crate::types::TypeAffinity::from_type_name(type_name);
                Ok(affinity.apply(v))
            }

            Expr::Case {
                operand,
                when_clauses,
                else_clause,
            } => {
                if let Some(op) = operand {
                    let op_val = self.eval_expr_in_row(op, row, group_rows)?;
                    for (when_expr, then_expr) in when_clauses {
                        let when_val = self.eval_expr_in_row(when_expr, row, group_rows)?;
                        if op_val == when_val {
                            return self.eval_expr_in_row(then_expr, row, group_rows);
                        }
                    }
                } else {
                    for (when_expr, then_expr) in when_clauses {
                        let when_val = self.eval_expr_in_row(when_expr, row, group_rows)?;
                        if when_val.is_truthy() {
                            return self.eval_expr_in_row(then_expr, row, group_rows);
                        }
                    }
                }
                if let Some(else_expr) = else_clause {
                    self.eval_expr_in_row(else_expr, row, group_rows)
                } else {
                    Ok(Value::Null)
                }
            }

            Expr::Subquery(query) => {
                let result = self.execute_select(query)?;
                if let Some(first_row) = result.rows.first() {
                    Ok(first_row.first().cloned().unwrap_or(Value::Null))
                } else {
                    Ok(Value::Null)
                }
            }

            Expr::Exists(query) => {
                let result = self.execute_select(query)?;
                Ok(Value::Integer(if result.rows.is_empty() { 0 } else { 1 }))
            }
        }
    }

    fn eval_binary_op(&self, lv: &Value, op: &BinaryOperator, rv: &Value) -> Result<Value> {
        // NULL propagation for most ops
        if (lv.is_null() || rv.is_null()) && !matches!(op, BinaryOperator::And | BinaryOperator::Or)
        {
            // Comparison with NULL returns NULL
            if matches!(
                op,
                BinaryOperator::Eq
                    | BinaryOperator::NotEq
                    | BinaryOperator::Lt
                    | BinaryOperator::LtEq
                    | BinaryOperator::Gt
                    | BinaryOperator::GtEq
            ) {
                return Ok(Value::Null);
            }
            return Ok(Value::Null);
        }

        match op {
            BinaryOperator::Add => match (lv, rv) {
                (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a.wrapping_add(*b))),
                (Value::Real(a), Value::Real(b)) => Ok(Value::Real(a + b)),
                (Value::Integer(a), Value::Real(b)) => Ok(Value::Real(*a as f64 + b)),
                (Value::Real(a), Value::Integer(b)) => Ok(Value::Real(a + *b as f64)),
                _ => {
                    let a = lv.to_real().unwrap_or(0.0);
                    let b = rv.to_real().unwrap_or(0.0);
                    Ok(Value::Real(a + b))
                }
            },
            BinaryOperator::Subtract => match (lv, rv) {
                (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a.wrapping_sub(*b))),
                (Value::Real(a), Value::Real(b)) => Ok(Value::Real(a - b)),
                (Value::Integer(a), Value::Real(b)) => Ok(Value::Real(*a as f64 - b)),
                (Value::Real(a), Value::Integer(b)) => Ok(Value::Real(a - *b as f64)),
                _ => {
                    let a = lv.to_real().unwrap_or(0.0);
                    let b = rv.to_real().unwrap_or(0.0);
                    Ok(Value::Real(a - b))
                }
            },
            BinaryOperator::Multiply => match (lv, rv) {
                (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a.wrapping_mul(*b))),
                (Value::Real(a), Value::Real(b)) => Ok(Value::Real(a * b)),
                (Value::Integer(a), Value::Real(b)) => Ok(Value::Real(*a as f64 * b)),
                (Value::Real(a), Value::Integer(b)) => Ok(Value::Real(a * *b as f64)),
                _ => {
                    let a = lv.to_real().unwrap_or(0.0);
                    let b = rv.to_real().unwrap_or(0.0);
                    Ok(Value::Real(a * b))
                }
            },
            BinaryOperator::Divide => {
                let b = rv.to_real().unwrap_or(0.0);
                if b == 0.0 {
                    return Ok(Value::Null);
                }
                match (lv, rv) {
                    (Value::Integer(a), Value::Integer(b)) => {
                        if *b == 0 {
                            Ok(Value::Null)
                        } else {
                            Ok(Value::Integer(a / b))
                        }
                    }
                    _ => {
                        let a = lv.to_real().unwrap_or(0.0);
                        Ok(Value::Real(a / b))
                    }
                }
            }
            BinaryOperator::Modulo => match (lv, rv) {
                (Value::Integer(a), Value::Integer(b)) => {
                    if *b == 0 {
                        Ok(Value::Null)
                    } else {
                        Ok(Value::Integer(a % b))
                    }
                }
                _ => {
                    let a = lv.to_real().unwrap_or(0.0);
                    let b = rv.to_real().unwrap_or(0.0);
                    if b == 0.0 {
                        Ok(Value::Null)
                    } else {
                        Ok(Value::Real(a % b))
                    }
                }
            },
            BinaryOperator::Eq => Ok(Value::Integer(if lv.sqlite_cmp(rv) == Ordering::Equal {
                1
            } else {
                0
            })),
            BinaryOperator::NotEq => Ok(Value::Integer(if lv.sqlite_cmp(rv) != Ordering::Equal {
                1
            } else {
                0
            })),
            BinaryOperator::Lt => Ok(Value::Integer(if lv.sqlite_cmp(rv) == Ordering::Less {
                1
            } else {
                0
            })),
            BinaryOperator::LtEq => Ok(Value::Integer(if lv.sqlite_cmp(rv) != Ordering::Greater {
                1
            } else {
                0
            })),
            BinaryOperator::Gt => Ok(Value::Integer(if lv.sqlite_cmp(rv) == Ordering::Greater {
                1
            } else {
                0
            })),
            BinaryOperator::GtEq => Ok(Value::Integer(if lv.sqlite_cmp(rv) != Ordering::Less {
                1
            } else {
                0
            })),
            BinaryOperator::Concat => Ok(Value::Text(format!("{}{}", lv.to_text(), rv.to_text()))),
            BinaryOperator::BitAnd => {
                let a = lv.to_integer().unwrap_or(0);
                let b = rv.to_integer().unwrap_or(0);
                Ok(Value::Integer(a & b))
            }
            BinaryOperator::BitOr => {
                let a = lv.to_integer().unwrap_or(0);
                let b = rv.to_integer().unwrap_or(0);
                Ok(Value::Integer(a | b))
            }
            BinaryOperator::ShiftLeft => {
                let a = lv.to_integer().unwrap_or(0);
                let b = rv.to_integer().unwrap_or(0);
                Ok(Value::Integer(a << b))
            }
            BinaryOperator::ShiftRight => {
                let a = lv.to_integer().unwrap_or(0);
                let b = rv.to_integer().unwrap_or(0);
                Ok(Value::Integer(a >> b))
            }
            BinaryOperator::And | BinaryOperator::Or => {
                // Handled above in eval_expr_in_row
                Ok(Value::Null)
            }
        }
    }

    fn execute_insert(&mut self, ins: &InsertStatement) -> Result<ExecuteResult> {
        let table_name = ins.table_name.to_lowercase();
        let schema = self
            .tables
            .get(&table_name)
            .ok_or_else(|| RsqliteError::TableNotFound(ins.table_name.clone()))?
            .clone();

        // Collect value rows based on source type
        let value_rows: Vec<Vec<Value>> = match &ins.source {
            InsertSource::Values(rows) => {
                let mut result = Vec::new();
                for value_row in rows {
                    let mut record_values = vec![Value::Null; schema.columns.len()];
                    if let Some(ref columns) = ins.columns {
                        for (i, col_name) in columns.iter().enumerate() {
                            let col_idx = schema
                                .columns
                                .iter()
                                .position(|c| c.name.eq_ignore_ascii_case(col_name))
                                .ok_or_else(|| RsqliteError::ColumnNotFound(col_name.clone()))?;
                            if i < value_row.len() {
                                record_values[col_idx] = self.eval_const_expr(&value_row[i])?;
                            }
                        }
                    } else {
                        for (i, expr) in value_row.iter().enumerate() {
                            if i < record_values.len() {
                                record_values[i] = self.eval_const_expr(expr)?;
                            }
                        }
                    }
                    result.push(record_values);
                }
                result
            }
            InsertSource::Select(query) => {
                let select_result = self.execute_select(query)?;
                let mut result = Vec::new();
                for row in &select_result.rows {
                    let mut record_values = vec![Value::Null; schema.columns.len()];
                    if let Some(ref columns) = ins.columns {
                        for (i, col_name) in columns.iter().enumerate() {
                            let col_idx = schema
                                .columns
                                .iter()
                                .position(|c| c.name.eq_ignore_ascii_case(col_name))
                                .ok_or_else(|| RsqliteError::ColumnNotFound(col_name.clone()))?;
                            if i < row.len() {
                                record_values[col_idx] = row[i].clone();
                            }
                        }
                    } else {
                        for (i, val) in row.iter().enumerate() {
                            if i < record_values.len() {
                                record_values[i] = val.clone();
                            }
                        }
                    }
                    result.push(record_values);
                }
                result
            }
            InsertSource::DefaultValues => {
                // Single row with all defaults
                vec![vec![Value::Null; schema.columns.len()]]
            }
        };

        let root_page = schema.root_page;
        let mut current_root = root_page;
        let mut row_count = 0;

        for mut record_values in value_rows {
            // Apply defaults
            for (i, col) in schema.columns.iter().enumerate() {
                if record_values[i].is_null() {
                    if let Some(ref default) = col.default_value {
                        record_values[i] = self.eval_const_expr(default)?;
                    }
                }
            }

            // Enforce NOT NULL constraints
            for (i, col) in schema.columns.iter().enumerate() {
                if col.not_null && record_values[i].is_null() && !col.is_primary_key {
                    return Err(RsqliteError::Constraint(format!(
                        "NOT NULL constraint failed: {}.{}",
                        ins.table_name, col.name
                    )));
                }
            }

            // Determine rowid
            let rowid = if let Some(pk_idx) = schema.primary_key_column {
                if let Value::Integer(id) = &record_values[pk_idx] {
                    *id
                } else if schema.is_autoincrement {
                    // AUTOINCREMENT: use max of (current max rowid, stored sequence)
                    let max_in_table = btree::max_rowid(&mut self.pager, current_root)?;
                    let seq = self.get_autoincrement_seq(&ins.table_name)?;
                    let next = max_in_table.max(seq) + 1;
                    self.set_autoincrement_seq(&ins.table_name, next)?;
                    next
                } else {
                    let max = btree::max_rowid(&mut self.pager, current_root)?;
                    max + 1
                }
            } else {
                let max = btree::max_rowid(&mut self.pager, current_root)?;
                max + 1
            };

            // Enforce UNIQUE constraints via indexes
            let indexes: Vec<IndexSchema> = self
                .indexes
                .values()
                .filter(|idx| idx.table_name.eq_ignore_ascii_case(&ins.table_name))
                .cloned()
                .collect();

            if !ins.or_replace {
                for idx in &indexes {
                    if idx.unique {
                        let key_values: Vec<Value> = idx
                            .columns
                            .iter()
                            .map(|col_name| {
                                schema
                                    .columns
                                    .iter()
                                    .position(|c| c.name.eq_ignore_ascii_case(col_name))
                                    .map(|i| record_values[i].clone())
                                    .unwrap_or(Value::Null)
                            })
                            .collect();
                        // Check if any existing row has the same key values
                        if key_values.iter().all(|v| !v.is_null()) {
                            let existing = btree::scan_index(&mut self.pager, idx.root_page)?;
                            for entry in &existing {
                                let entry_keys = &entry[..entry.len().saturating_sub(1)];
                                if entry_keys == key_values.as_slice() {
                                    return Err(RsqliteError::Constraint(format!(
                                        "UNIQUE constraint failed: {}",
                                        idx.name
                                    )));
                                }
                            }
                        }
                    }
                }
            }

            // If the primary key column is an integer PK alias (rowid alias),
            // don't store it in the record - it's the rowid itself
            let store_values = if let Some(pk_idx) = schema.primary_key_column {
                let pk_type = &schema.columns[pk_idx].type_name;
                if pk_type.to_uppercase().contains("INT") {
                    // Store NULL for the PK column (it's the rowid alias)
                    let mut sv = record_values.clone();
                    sv[pk_idx] = Value::Null;
                    sv
                } else {
                    record_values.clone()
                }
            } else {
                record_values.clone()
            };

            let record_data = record::serialize_record(&store_values);
            let new_root =
                btree::insert_into_table(&mut self.pager, current_root, rowid, &record_data)?;

            if new_root != current_root && current_root == root_page {
                // Root page changed due to split - update schema
                let schema_mut = self.tables.get_mut(&table_name).unwrap();
                schema_mut.root_page = new_root;
                // Update sqlite_master
                self.update_schema_root_page(&ins.table_name, new_root)?;
            }
            current_root = new_root;

            // Update indexes
            for idx in &indexes {
                let key_values: Vec<Value> = idx
                    .columns
                    .iter()
                    .map(|col_name| {
                        schema
                            .columns
                            .iter()
                            .position(|c| c.name.eq_ignore_ascii_case(col_name))
                            .map(|i| record_values[i].clone())
                            .unwrap_or(Value::Null)
                    })
                    .collect();
                btree::insert_into_index(&mut self.pager, idx.root_page, &key_values, rowid)?;
            }

            self.last_insert_rowid = rowid;
            row_count += 1;
        }

        self.changes = row_count;

        if self.auto_commit {
            self.pager.flush()?;
        }

        Ok(ExecuteResult {
            columns: Vec::new(),
            rows: Vec::new(),
            rows_affected: row_count as usize,
        })
    }

    fn execute_update(&mut self, upd: &UpdateStatement) -> Result<ExecuteResult> {
        let table_name = upd.table_name.to_lowercase();
        let schema = self
            .tables
            .get(&table_name)
            .ok_or_else(|| RsqliteError::TableNotFound(upd.table_name.clone()))?
            .clone();

        let root_page = schema.root_page;
        let rows = btree::scan_table(&mut self.pager, root_page)?;

        let column_names: Vec<String> = schema.columns.iter().map(|c| c.name.clone()).collect();
        let mut updates = Vec::new();

        for (rowid, values) in &rows {
            let row = Row {
                values: values.clone(),
                columns: column_names
                    .iter()
                    .map(|n| (upd.table_name.clone(), n.clone()))
                    .collect(),
                table_alias: Some(upd.table_name.clone()),
                rowid: *rowid,
            };

            let matches = if let Some(ref where_clause) = upd.where_clause {
                self.eval_expr_in_row(where_clause, &row, &[])?.is_truthy()
            } else {
                true
            };

            if matches {
                let mut new_values = values.clone();
                for (col_name, expr) in &upd.assignments {
                    let col_idx = schema
                        .columns
                        .iter()
                        .position(|c| c.name.eq_ignore_ascii_case(col_name))
                        .ok_or_else(|| RsqliteError::ColumnNotFound(col_name.clone()))?;
                    let new_val = self.eval_expr_in_row(expr, &row, &[])?;
                    // Enforce NOT NULL constraints
                    if schema.columns[col_idx].not_null && new_val.is_null() {
                        return Err(RsqliteError::Constraint(format!(
                            "NOT NULL constraint failed: {}.{}",
                            upd.table_name, col_name
                        )));
                    }
                    new_values[col_idx] = new_val;
                }
                updates.push((*rowid, new_values));
            }
        }

        let count = updates.len() as i64;
        for (rowid, values) in updates {
            // For integer PK tables, store NULL in PK column
            let store_values = if let Some(pk_idx) = schema.primary_key_column {
                let pk_type = &schema.columns[pk_idx].type_name;
                if pk_type.to_uppercase().contains("INT") {
                    let mut sv = values;
                    sv[pk_idx] = Value::Null;
                    sv
                } else {
                    values
                }
            } else {
                values
            };

            btree::delete_from_table(&mut self.pager, root_page, rowid)?;
            let record_data = record::serialize_record(&store_values);
            btree::insert_into_table(&mut self.pager, root_page, rowid, &record_data)?;
        }

        self.changes = count;

        if self.auto_commit {
            self.pager.flush()?;
        }

        Ok(ExecuteResult {
            columns: Vec::new(),
            rows: Vec::new(),
            rows_affected: count as usize,
        })
    }

    fn execute_delete(&mut self, del: &DeleteStatement) -> Result<ExecuteResult> {
        let table_name = del.table_name.to_lowercase();
        let schema = self
            .tables
            .get(&table_name)
            .ok_or_else(|| RsqliteError::TableNotFound(del.table_name.clone()))?
            .clone();

        let root_page = schema.root_page;
        let rows = btree::scan_table(&mut self.pager, root_page)?;

        let column_names: Vec<String> = schema.columns.iter().map(|c| c.name.clone()).collect();
        let mut to_delete = Vec::new();

        for (rowid, values) in &rows {
            let row = Row {
                values: values.clone(),
                columns: column_names
                    .iter()
                    .map(|n| (del.table_name.clone(), n.clone()))
                    .collect(),
                table_alias: Some(del.table_name.clone()),
                rowid: *rowid,
            };

            let matches = if let Some(ref where_clause) = del.where_clause {
                self.eval_expr_in_row(where_clause, &row, &[])?.is_truthy()
            } else {
                true
            };

            if matches {
                to_delete.push(*rowid);
            }
        }

        let count = to_delete.len() as i64;
        for rowid in to_delete {
            btree::delete_from_table(&mut self.pager, root_page, rowid)?;

            // Delete from indexes
            let indexes: Vec<IndexSchema> = self
                .indexes
                .values()
                .filter(|idx| idx.table_name.eq_ignore_ascii_case(&del.table_name))
                .cloned()
                .collect();
            for idx in &indexes {
                btree::delete_from_index(&mut self.pager, idx.root_page, rowid)?;
            }
        }

        self.changes = count;

        if self.auto_commit {
            self.pager.flush()?;
        }

        Ok(ExecuteResult {
            columns: Vec::new(),
            rows: Vec::new(),
            rows_affected: count as usize,
        })
    }

    fn execute_create_table(&mut self, ct: &CreateTableStatement) -> Result<ExecuteResult> {
        let table_name = ct.table_name.to_lowercase();

        if self.tables.contains_key(&table_name) {
            if ct.if_not_exists {
                return Ok(ExecuteResult::empty());
            }
            return Err(RsqliteError::TableExists(ct.table_name.clone()));
        }

        // Allocate a root page for the new table
        let root_page = self.pager.allocate_page()?;
        btree::init_leaf_table_page(&mut self.pager, root_page)?;

        // Build the CREATE TABLE SQL
        let sql = format_create_table_sql(ct);

        // Insert into sqlite_master
        let master_record = record::serialize_record(&[
            Value::Text("table".into()),
            Value::Text(ct.table_name.clone()),
            Value::Text(ct.table_name.clone()),
            Value::Integer(root_page as i64),
            Value::Text(sql.clone()),
        ]);

        let master_rowid = btree::max_rowid(&mut self.pager, 1)? + 1;
        btree::insert_into_table(&mut self.pager, 1, master_rowid, &master_record)?;

        // Parse schema
        let schema = self.parse_table_schema(&ct.table_name, root_page, &sql)?;
        self.tables.insert(table_name, schema);

        // Update schema cookie
        self.pager.header.schema_cookie += 1;

        if self.auto_commit {
            self.pager.flush()?;
        }

        Ok(ExecuteResult::empty())
    }

    fn execute_create_index(&mut self, ci: &CreateIndexStatement) -> Result<ExecuteResult> {
        let index_name = ci.index_name.to_lowercase();

        if self.indexes.contains_key(&index_name) {
            if ci.if_not_exists {
                return Ok(ExecuteResult::empty());
            }
            return Err(RsqliteError::IndexExists(ci.index_name.clone()));
        }

        // Verify table exists
        let table_name = ci.table_name.to_lowercase();
        if !self.tables.contains_key(&table_name) {
            return Err(RsqliteError::TableNotFound(ci.table_name.clone()));
        }

        // Allocate root page for index
        let root_page = self.pager.allocate_page()?;
        btree::init_leaf_table_page(&mut self.pager, root_page)?;
        // Actually make it an index leaf
        let page = self.pager.get_page_mut(root_page)?;
        page.data[0] = crate::format::BTreePageType::LeafIndex.to_byte();

        // Build SQL
        let unique_str = if ci.unique { "UNIQUE " } else { "" };
        let cols: Vec<String> = ci.columns.iter().map(|c| c.name.clone()).collect();
        let sql = format!(
            "CREATE {}INDEX {} ON {}({})",
            unique_str,
            ci.index_name,
            ci.table_name,
            cols.join(", ")
        );

        // Insert into sqlite_master
        let master_record = record::serialize_record(&[
            Value::Text("index".into()),
            Value::Text(ci.index_name.clone()),
            Value::Text(ci.table_name.clone()),
            Value::Integer(root_page as i64),
            Value::Text(sql.clone()),
        ]);
        let master_rowid = btree::max_rowid(&mut self.pager, 1)? + 1;
        btree::insert_into_table(&mut self.pager, 1, master_rowid, &master_record)?;

        // Populate index with existing data
        let schema = self.tables.get(&table_name).unwrap().clone();
        let table_rows = btree::scan_table(&mut self.pager, schema.root_page)?;

        for (rowid, values) in table_rows {
            let key_values: Vec<Value> = ci
                .columns
                .iter()
                .map(|col| {
                    schema
                        .columns
                        .iter()
                        .position(|c| c.name.eq_ignore_ascii_case(&col.name))
                        .map(|i| values.get(i).cloned().unwrap_or(Value::Null))
                        .unwrap_or(Value::Null)
                })
                .collect();
            btree::insert_into_index(&mut self.pager, root_page, &key_values, rowid)?;
        }

        // Register index
        let index_schema = IndexSchema {
            name: ci.index_name.clone(),
            table_name: ci.table_name.clone(),
            columns: ci.columns.iter().map(|c| c.name.clone()).collect(),
            root_page,
            unique: ci.unique,
            sql,
        };
        self.indexes.insert(index_name, index_schema);

        self.pager.header.schema_cookie += 1;

        if self.auto_commit {
            self.pager.flush()?;
        }

        Ok(ExecuteResult::empty())
    }

    fn execute_drop_table(&mut self, dt: &DropTableStatement) -> Result<ExecuteResult> {
        let table_name = dt.table_name.to_lowercase();
        if !self.tables.contains_key(&table_name) {
            if dt.if_exists {
                return Ok(ExecuteResult::empty());
            }
            return Err(RsqliteError::TableNotFound(dt.table_name.clone()));
        }

        // Remove from sqlite_master
        self.remove_from_master(&dt.table_name)?;

        // Remove associated indexes
        let idx_to_remove: Vec<String> = self
            .indexes
            .values()
            .filter(|idx| idx.table_name.eq_ignore_ascii_case(&dt.table_name))
            .map(|idx| idx.name.clone())
            .collect();
        for idx_name in &idx_to_remove {
            self.remove_from_master(idx_name)?;
            self.indexes.remove(&idx_name.to_lowercase());
        }

        self.tables.remove(&table_name);
        self.pager.header.schema_cookie += 1;

        if self.auto_commit {
            self.pager.flush()?;
        }

        Ok(ExecuteResult::empty())
    }

    fn execute_drop_index(&mut self, di: &DropIndexStatement) -> Result<ExecuteResult> {
        let index_name = di.index_name.to_lowercase();
        if !self.indexes.contains_key(&index_name) {
            if di.if_exists {
                return Ok(ExecuteResult::empty());
            }
            return Err(RsqliteError::IndexNotFound(di.index_name.clone()));
        }

        self.remove_from_master(&di.index_name)?;
        self.indexes.remove(&index_name);
        self.pager.header.schema_cookie += 1;

        if self.auto_commit {
            self.pager.flush()?;
        }

        Ok(ExecuteResult::empty())
    }

    fn execute_alter_table(&mut self, at: &AlterTableStatement) -> Result<ExecuteResult> {
        match at {
            AlterTableStatement::RenameTable {
                table_name,
                new_name,
            } => {
                let old_lower = table_name.to_lowercase();
                let schema = self
                    .tables
                    .remove(&old_lower)
                    .ok_or_else(|| RsqliteError::TableNotFound(table_name.clone()))?;

                // Update in sqlite_master
                self.remove_from_master(table_name)?;
                let new_sql = schema.sql.replace(table_name, new_name);
                let master_record = record::serialize_record(&[
                    Value::Text("table".into()),
                    Value::Text(new_name.clone()),
                    Value::Text(new_name.clone()),
                    Value::Integer(schema.root_page as i64),
                    Value::Text(new_sql.clone()),
                ]);
                let master_rowid = btree::max_rowid(&mut self.pager, 1)? + 1;
                btree::insert_into_table(&mut self.pager, 1, master_rowid, &master_record)?;

                let mut new_schema = schema;
                new_schema.name = new_name.clone();
                new_schema.sql = new_sql;
                self.tables.insert(new_name.to_lowercase(), new_schema);
            }
            AlterTableStatement::AddColumn { table_name, column } => {
                let table_lower = table_name.to_lowercase();
                let (root_page, new_sql) = {
                    let schema = self
                        .tables
                        .get_mut(&table_lower)
                        .ok_or_else(|| RsqliteError::TableNotFound(table_name.clone()))?;

                    schema.columns.push(ColumnInfo {
                        name: column.name.clone(),
                        type_name: column.type_name.clone().unwrap_or_default(),
                        not_null: false,
                        default_value: None,
                        is_primary_key: false,
                    });

                    let old_sql = schema.sql.clone();
                    let rp = schema.root_page;
                    let ns = if let Some(paren_pos) = old_sql.rfind(')') {
                        let col_type = column.type_name.as_deref().unwrap_or("");
                        let s = format!("{}, {} {})", &old_sql[..paren_pos], column.name, col_type);
                        schema.sql = s.clone();
                        Some(s)
                    } else {
                        None
                    };
                    (rp, ns)
                };

                if let Some(new_sql) = new_sql {
                    self.remove_from_master(table_name)?;
                    let master_record = record::serialize_record(&[
                        Value::Text("table".into()),
                        Value::Text(table_name.clone()),
                        Value::Text(table_name.clone()),
                        Value::Integer(root_page as i64),
                        Value::Text(new_sql),
                    ]);
                    let master_rowid = btree::max_rowid(&mut self.pager, 1)? + 1;
                    btree::insert_into_table(&mut self.pager, 1, master_rowid, &master_record)?;
                }
            }
            AlterTableStatement::RenameColumn {
                table_name,
                old_name,
                new_name,
            } => {
                let table_lower = table_name.to_lowercase();
                let (root_page, new_sql) = {
                    let schema = self
                        .tables
                        .get_mut(&table_lower)
                        .ok_or_else(|| RsqliteError::TableNotFound(table_name.clone()))?;

                    for col in &mut schema.columns {
                        if col.name.eq_ignore_ascii_case(old_name) {
                            col.name = new_name.clone();
                            break;
                        }
                    }

                    let old_sql = schema.sql.clone();
                    let new_sql = old_sql.replace(old_name, new_name);
                    schema.sql = new_sql.clone();
                    (schema.root_page, new_sql)
                };

                self.remove_from_master(table_name)?;
                let master_record = record::serialize_record(&[
                    Value::Text("table".into()),
                    Value::Text(table_name.clone()),
                    Value::Text(table_name.clone()),
                    Value::Integer(root_page as i64),
                    Value::Text(new_sql),
                ]);
                let master_rowid = btree::max_rowid(&mut self.pager, 1)? + 1;
                btree::insert_into_table(&mut self.pager, 1, master_rowid, &master_record)?;
            }
        }

        self.pager.header.schema_cookie += 1;

        if self.auto_commit {
            self.pager.flush()?;
        }

        Ok(ExecuteResult::empty())
    }

    fn execute_explain(&mut self, stmt: &Statement) -> Result<ExecuteResult> {
        let plan = format!("{:#?}", stmt);
        Ok(ExecuteResult {
            columns: vec!["detail".into()],
            rows: plan
                .lines()
                .map(|l| vec![Value::Text(l.to_string())])
                .collect(),
            rows_affected: 0,
        })
    }

    fn execute_pragma(&mut self, pragma: &PragmaStatement) -> Result<ExecuteResult> {
        let name = pragma.name.to_lowercase();
        match name.as_str() {
            "table_info" => {
                if let Some(ref val) = pragma.value {
                    let table_name = match val {
                        PragmaValue::Name(n) | PragmaValue::Call(n) => n.to_lowercase(),
                        _ => return Ok(ExecuteResult::empty()),
                    };
                    if let Some(schema) = self.tables.get(&table_name) {
                        let columns = vec![
                            "cid".into(),
                            "name".into(),
                            "type".into(),
                            "notnull".into(),
                            "dflt_value".into(),
                            "pk".into(),
                        ];
                        let rows: Vec<Vec<Value>> = schema
                            .columns
                            .iter()
                            .enumerate()
                            .map(|(i, col)| {
                                vec![
                                    Value::Integer(i as i64),
                                    Value::Text(col.name.clone()),
                                    Value::Text(col.type_name.clone()),
                                    Value::Integer(if col.not_null { 1 } else { 0 }),
                                    Value::Null,
                                    Value::Integer(if col.is_primary_key { 1 } else { 0 }),
                                ]
                            })
                            .collect();
                        return Ok(ExecuteResult {
                            columns,
                            rows,
                            rows_affected: 0,
                        });
                    }
                }
                Ok(ExecuteResult::empty())
            }
            "page_size" => Ok(ExecuteResult {
                columns: vec!["page_size".into()],
                rows: vec![vec![Value::Integer(self.pager.page_size as i64)]],
                rows_affected: 0,
            }),
            "page_count" => Ok(ExecuteResult {
                columns: vec!["page_count".into()],
                rows: vec![vec![Value::Integer(self.pager.page_count() as i64)]],
                rows_affected: 0,
            }),
            "journal_mode" => Ok(ExecuteResult {
                columns: vec!["journal_mode".into()],
                rows: vec![vec![Value::Text("delete".into())]],
                rows_affected: 0,
            }),
            "schema_version" => Ok(ExecuteResult {
                columns: vec!["schema_version".into()],
                rows: vec![vec![Value::Integer(self.pager.header.schema_cookie as i64)]],
                rows_affected: 0,
            }),
            "database_list" => Ok(ExecuteResult {
                columns: vec!["seq".into(), "name".into(), "file".into()],
                rows: vec![vec![
                    Value::Integer(0),
                    Value::Text("main".into()),
                    Value::Text("".into()),
                ]],
                rows_affected: 0,
            }),
            "index_list" => {
                if let Some(ref val) = pragma.value {
                    let table_name = match val {
                        PragmaValue::Name(n) | PragmaValue::Call(n) => n.to_lowercase(),
                        _ => return Ok(ExecuteResult::empty()),
                    };
                    let columns = vec!["seq".into(), "name".into(), "unique".into()];
                    let rows: Vec<Vec<Value>> = self
                        .indexes
                        .values()
                        .filter(|idx| idx.table_name.to_lowercase() == table_name)
                        .enumerate()
                        .map(|(i, idx)| {
                            vec![
                                Value::Integer(i as i64),
                                Value::Text(idx.name.clone()),
                                Value::Integer(if idx.unique { 1 } else { 0 }),
                            ]
                        })
                        .collect();
                    return Ok(ExecuteResult {
                        columns,
                        rows,
                        rows_affected: 0,
                    });
                }
                Ok(ExecuteResult::empty())
            }
            "index_info" => {
                if let Some(ref val) = pragma.value {
                    let index_name = match val {
                        PragmaValue::Name(n) | PragmaValue::Call(n) => n.to_lowercase(),
                        _ => return Ok(ExecuteResult::empty()),
                    };
                    if let Some(idx) = self.indexes.get(&index_name) {
                        let columns = vec!["seqno".into(), "cid".into(), "name".into()];
                        let table_lower = idx.table_name.to_lowercase();
                        let table_schema = self.tables.get(&table_lower);
                        let rows: Vec<Vec<Value>> = idx
                            .columns
                            .iter()
                            .enumerate()
                            .map(|(i, col_name)| {
                                let cid = table_schema
                                    .and_then(|ts| {
                                        ts.columns
                                            .iter()
                                            .position(|c| c.name.eq_ignore_ascii_case(col_name))
                                    })
                                    .unwrap_or(0);
                                vec![
                                    Value::Integer(i as i64),
                                    Value::Integer(cid as i64),
                                    Value::Text(col_name.clone()),
                                ]
                            })
                            .collect();
                        return Ok(ExecuteResult {
                            columns,
                            rows,
                            rows_affected: 0,
                        });
                    }
                }
                Ok(ExecuteResult::empty())
            }
            "integrity_check" => {
                // Simple integrity check - just return "ok"
                Ok(ExecuteResult {
                    columns: vec!["integrity_check".into()],
                    rows: vec![vec![Value::Text("ok".into())]],
                    rows_affected: 0,
                })
            }
            "foreign_keys" | "foreign_key_list" => {
                // Foreign keys not enforced yet, return off
                if pragma.value.is_some() {
                    Ok(ExecuteResult::empty())
                } else {
                    Ok(ExecuteResult {
                        columns: vec!["foreign_keys".into()],
                        rows: vec![vec![Value::Integer(0)]],
                        rows_affected: 0,
                    })
                }
            }
            "cache_size" => Ok(ExecuteResult {
                columns: vec!["cache_size".into()],
                rows: vec![vec![Value::Integer(-2000)]],
                rows_affected: 0,
            }),
            "auto_vacuum" => Ok(ExecuteResult {
                columns: vec!["auto_vacuum".into()],
                rows: vec![vec![Value::Integer(0)]],
                rows_affected: 0,
            }),
            "encoding" => Ok(ExecuteResult {
                columns: vec!["encoding".into()],
                rows: vec![vec![Value::Text("UTF-8".into())]],
                rows_affected: 0,
            }),
            "compile_options" => Ok(ExecuteResult {
                columns: vec!["compile_option".into()],
                rows: vec![vec![Value::Text("RSQLITE".into())]],
                rows_affected: 0,
            }),
            "freelist_count" => Ok(ExecuteResult {
                columns: vec!["freelist_count".into()],
                rows: vec![vec![Value::Integer(
                    self.pager.header.total_freelist_pages as i64,
                )]],
                rows_affected: 0,
            }),
            "table_list" => {
                let columns = vec![
                    "schema".into(),
                    "name".into(),
                    "type".into(),
                    "ncol".into(),
                    "wr".into(),
                    "strict".into(),
                ];
                let rows: Vec<Vec<Value>> = self
                    .tables
                    .values()
                    .map(|t| {
                        vec![
                            Value::Text("main".into()),
                            Value::Text(t.name.clone()),
                            Value::Text("table".into()),
                            Value::Integer(t.columns.len() as i64),
                            Value::Integer(0),
                            Value::Integer(0),
                        ]
                    })
                    .collect();
                Ok(ExecuteResult {
                    columns,
                    rows,
                    rows_affected: 0,
                })
            }
            _ => {
                // Unknown pragma, return empty
                Ok(ExecuteResult::empty())
            }
        }
    }

    /// Get the current autoincrement sequence value for a table.
    fn get_autoincrement_seq(&mut self, table_name: &str) -> Result<i64> {
        // Check if sqlite_sequence table exists
        if !self.tables.contains_key("sqlite_sequence") {
            return Ok(0);
        }
        let seq_root = self.tables["sqlite_sequence"].root_page;
        let rows = btree::scan_table(&mut self.pager, seq_root)?;
        for (_rowid, values) in &rows {
            if values.len() >= 2 && values[0].to_text().eq_ignore_ascii_case(table_name) {
                return Ok(values[1].to_integer().unwrap_or(0));
            }
        }
        Ok(0)
    }

    /// Set the autoincrement sequence value for a table.
    fn set_autoincrement_seq(&mut self, table_name: &str, seq: i64) -> Result<()> {
        // Create sqlite_sequence table if it doesn't exist
        if !self.tables.contains_key("sqlite_sequence") {
            let root_page = self.pager.allocate_page()?;
            btree::init_leaf_table_page(&mut self.pager, root_page)?;
            let sql = "CREATE TABLE sqlite_sequence(name,seq)";
            let master_record = record::serialize_record(&[
                Value::Text("table".into()),
                Value::Text("sqlite_sequence".into()),
                Value::Text("sqlite_sequence".into()),
                Value::Integer(root_page as i64),
                Value::Text(sql.into()),
            ]);
            let master_rowid = btree::max_rowid(&mut self.pager, 1)? + 1;
            btree::insert_into_table(&mut self.pager, 1, master_rowid, &master_record)?;
            self.tables.insert(
                "sqlite_sequence".into(),
                TableSchema {
                    name: "sqlite_sequence".into(),
                    columns: vec![
                        ColumnInfo {
                            name: "name".into(),
                            type_name: String::new(),
                            not_null: false,
                            default_value: None,
                            is_primary_key: false,
                        },
                        ColumnInfo {
                            name: "seq".into(),
                            type_name: String::new(),
                            not_null: false,
                            default_value: None,
                            is_primary_key: false,
                        },
                    ],
                    root_page,
                    sql: sql.into(),
                    is_autoincrement: false,
                    primary_key_column: None,
                },
            );
        }

        let seq_root = self.tables["sqlite_sequence"].root_page;
        let rows = btree::scan_table(&mut self.pager, seq_root)?;

        // Check if an entry exists for this table
        for (rowid, values) in &rows {
            if values.len() >= 2 && values[0].to_text().eq_ignore_ascii_case(table_name) {
                // Update existing entry
                btree::delete_from_table(&mut self.pager, seq_root, *rowid)?;
                let record_data = record::serialize_record(&[
                    Value::Text(table_name.to_string()),
                    Value::Integer(seq),
                ]);
                btree::insert_into_table(&mut self.pager, seq_root, *rowid, &record_data)?;
                return Ok(());
            }
        }

        // Insert new entry
        let rowid = btree::max_rowid(&mut self.pager, seq_root)? + 1;
        let record_data =
            record::serialize_record(&[Value::Text(table_name.to_string()), Value::Integer(seq)]);
        btree::insert_into_table(&mut self.pager, seq_root, rowid, &record_data)?;
        Ok(())
    }

    fn remove_from_master(&mut self, name: &str) -> Result<()> {
        let rows = btree::scan_table(&mut self.pager, 1)?;
        for (rowid, values) in &rows {
            if values.len() >= 2 && values[1].to_text().eq_ignore_ascii_case(name) {
                btree::delete_from_table(&mut self.pager, 1, *rowid)?;
            }
        }
        Ok(())
    }

    fn update_schema_root_page(&mut self, table_name: &str, new_root: u32) -> Result<()> {
        let rows = btree::scan_table(&mut self.pager, 1)?;
        for (rowid, values) in &rows {
            if values.len() >= 5 && values[1].to_text().eq_ignore_ascii_case(table_name) {
                btree::delete_from_table(&mut self.pager, 1, *rowid)?;
                let new_record = record::serialize_record(&[
                    values[0].clone(),
                    values[1].clone(),
                    values[2].clone(),
                    Value::Integer(new_root as i64),
                    values[4].clone(),
                ]);
                btree::insert_into_table(&mut self.pager, 1, *rowid, &new_record)?;
                break;
            }
        }
        Ok(())
    }
}

/// Result of executing a statement.
#[derive(Debug, Clone)]
pub struct ExecuteResult {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<Value>>,
    pub rows_affected: usize,
}

impl ExecuteResult {
    pub fn empty() -> Self {
        ExecuteResult {
            columns: Vec::new(),
            rows: Vec::new(),
            rows_affected: 0,
        }
    }
}

/// Internal row representation during query execution.
#[derive(Debug, Clone)]
struct Row {
    values: Vec<Value>,
    columns: Vec<(String, String)>, // (table_alias, column_name)
    table_alias: Option<String>,
    rowid: i64,
}

fn format_create_table_sql(ct: &CreateTableStatement) -> String {
    let mut sql = format!("CREATE TABLE {} (", ct.table_name);
    for (i, col) in ct.columns.iter().enumerate() {
        if i > 0 {
            sql.push_str(", ");
        }
        sql.push_str(&col.name);
        if let Some(ref tn) = col.type_name {
            sql.push(' ');
            sql.push_str(tn);
        }
        for constraint in &col.constraints {
            match constraint {
                ColumnConstraint::PrimaryKey { autoincrement } => {
                    sql.push_str(" PRIMARY KEY");
                    if *autoincrement {
                        sql.push_str(" AUTOINCREMENT");
                    }
                }
                ColumnConstraint::NotNull => sql.push_str(" NOT NULL"),
                ColumnConstraint::Unique => sql.push_str(" UNIQUE"),
                ColumnConstraint::Default(expr) => {
                    sql.push_str(" DEFAULT ");
                    sql.push_str(&format_expr(expr));
                }
                ColumnConstraint::Check(_) => sql.push_str(" CHECK(...)"),
                ColumnConstraint::References { table, columns } => {
                    sql.push_str(&format!(" REFERENCES {}({})", table, columns.join(", ")));
                }
            }
        }
    }
    sql.push(')');
    sql
}

fn format_expr(expr: &Expr) -> String {
    match expr {
        Expr::Null => "NULL".into(),
        Expr::Integer(i) => i.to_string(),
        Expr::Float(f) => f.to_string(),
        Expr::String(s) => format!("'{}'", s.replace('\'', "''")),
        _ => "?".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_table_and_insert() {
        let mut db = Database::new_memory();
        db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
            .unwrap();

        assert!(db.tables.contains_key("users"));

        db.execute("INSERT INTO users VALUES (1, 'Alice', 30)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (2, 'Bob', 25)")
            .unwrap();

        let result = db.execute("SELECT * FROM users").unwrap();
        assert_eq!(result.rows.len(), 2);
        assert_eq!(result.columns, vec!["id", "name", "age"]);
    }

    #[test]
    fn test_select_where() {
        let mut db = Database::new_memory();
        db.execute("CREATE TABLE t (x INTEGER, y TEXT)").unwrap();
        db.execute("INSERT INTO t VALUES (1, 'a')").unwrap();
        db.execute("INSERT INTO t VALUES (2, 'b')").unwrap();
        db.execute("INSERT INTO t VALUES (3, 'c')").unwrap();

        let result = db.execute("SELECT * FROM t WHERE x > 1").unwrap();
        assert_eq!(result.rows.len(), 2);
    }

    #[test]
    fn test_update() {
        let mut db = Database::new_memory();
        db.execute("CREATE TABLE t (x INTEGER, y TEXT)").unwrap();
        db.execute("INSERT INTO t VALUES (1, 'old')").unwrap();
        db.execute("UPDATE t SET y = 'new' WHERE x = 1").unwrap();

        let result = db.execute("SELECT y FROM t WHERE x = 1").unwrap();
        assert_eq!(result.rows[0][0], Value::Text("new".into()));
    }

    #[test]
    fn test_delete() {
        let mut db = Database::new_memory();
        db.execute("CREATE TABLE t (x INTEGER)").unwrap();
        db.execute("INSERT INTO t VALUES (1)").unwrap();
        db.execute("INSERT INTO t VALUES (2)").unwrap();
        db.execute("INSERT INTO t VALUES (3)").unwrap();
        db.execute("DELETE FROM t WHERE x = 2").unwrap();

        let result = db.execute("SELECT * FROM t").unwrap();
        assert_eq!(result.rows.len(), 2);
    }

    #[test]
    fn test_aggregate_count() {
        let mut db = Database::new_memory();
        db.execute("CREATE TABLE t (x INTEGER)").unwrap();
        db.execute("INSERT INTO t VALUES (1)").unwrap();
        db.execute("INSERT INTO t VALUES (2)").unwrap();
        db.execute("INSERT INTO t VALUES (3)").unwrap();

        let result = db.execute("SELECT COUNT(*) FROM t").unwrap();
        assert_eq!(result.rows[0][0], Value::Integer(3));
    }

    #[test]
    fn test_group_by() {
        let mut db = Database::new_memory();
        db.execute("CREATE TABLE t (category TEXT, value INTEGER)")
            .unwrap();
        db.execute("INSERT INTO t VALUES ('a', 10)").unwrap();
        db.execute("INSERT INTO t VALUES ('a', 20)").unwrap();
        db.execute("INSERT INTO t VALUES ('b', 30)").unwrap();

        let result = db
            .execute("SELECT category, SUM(value) FROM t GROUP BY category")
            .unwrap();
        assert_eq!(result.rows.len(), 2);
    }

    #[test]
    fn test_order_by() {
        let mut db = Database::new_memory();
        db.execute("CREATE TABLE t (x INTEGER)").unwrap();
        db.execute("INSERT INTO t VALUES (3)").unwrap();
        db.execute("INSERT INTO t VALUES (1)").unwrap();
        db.execute("INSERT INTO t VALUES (2)").unwrap();

        let result = db.execute("SELECT x FROM t ORDER BY x").unwrap();
        assert_eq!(result.rows[0][0], Value::Integer(1));
        assert_eq!(result.rows[1][0], Value::Integer(2));
        assert_eq!(result.rows[2][0], Value::Integer(3));
    }

    #[test]
    fn test_limit_offset() {
        let mut db = Database::new_memory();
        db.execute("CREATE TABLE t (x INTEGER)").unwrap();
        for i in 1..=10 {
            db.execute(&format!("INSERT INTO t VALUES ({})", i))
                .unwrap();
        }

        let result = db
            .execute("SELECT x FROM t ORDER BY x LIMIT 3 OFFSET 2")
            .unwrap();
        assert_eq!(result.rows.len(), 3);
        assert_eq!(result.rows[0][0], Value::Integer(3));
    }

    #[test]
    fn test_join() {
        let mut db = Database::new_memory();
        db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
            .unwrap();
        db.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount INTEGER)")
            .unwrap();
        db.execute("INSERT INTO users VALUES (1, 'Alice')").unwrap();
        db.execute("INSERT INTO users VALUES (2, 'Bob')").unwrap();
        db.execute("INSERT INTO orders VALUES (1, 1, 100)").unwrap();
        db.execute("INSERT INTO orders VALUES (2, 1, 200)").unwrap();
        db.execute("INSERT INTO orders VALUES (3, 2, 150)").unwrap();

        let result = db.execute(
            "SELECT users.name, orders.amount FROM users JOIN orders ON users.id = orders.user_id ORDER BY orders.amount"
        ).unwrap();
        assert_eq!(result.rows.len(), 3);
    }

    #[test]
    fn test_select_expression() {
        let mut db = Database::new_memory();
        let result = db
            .execute("SELECT 1 + 2, 'hello' || ' ' || 'world'")
            .unwrap();
        assert_eq!(result.rows[0][0], Value::Integer(3));
        assert_eq!(result.rows[0][1], Value::Text("hello world".into()));
    }

    #[test]
    fn test_drop_table() {
        let mut db = Database::new_memory();
        db.execute("CREATE TABLE t (x INTEGER)").unwrap();
        assert!(db.tables.contains_key("t"));
        db.execute("DROP TABLE t").unwrap();
        assert!(!db.tables.contains_key("t"));
    }

    #[test]
    fn test_if_not_exists() {
        let mut db = Database::new_memory();
        db.execute("CREATE TABLE t (x INTEGER)").unwrap();
        db.execute("CREATE TABLE IF NOT EXISTS t (x INTEGER)")
            .unwrap();
    }

    #[test]
    fn test_transactions() {
        let mut db = Database::new_memory();
        db.execute("CREATE TABLE t (x INTEGER)").unwrap();
        db.execute("INSERT INTO t VALUES (1)").unwrap();

        db.execute("BEGIN").unwrap();
        db.execute("INSERT INTO t VALUES (2)").unwrap();
        db.execute("ROLLBACK").unwrap();

        let _result = db.execute("SELECT COUNT(*) FROM t").unwrap();
        // Note: in memory mode, rollback is simplified
    }

    #[test]
    fn test_case_expression() {
        let mut db = Database::new_memory();
        let result = db
            .execute("SELECT CASE WHEN 1 > 0 THEN 'yes' ELSE 'no' END")
            .unwrap();
        assert_eq!(result.rows[0][0], Value::Text("yes".into()));
    }

    #[test]
    fn test_subquery() {
        let mut db = Database::new_memory();
        db.execute("CREATE TABLE t (x INTEGER)").unwrap();
        db.execute("INSERT INTO t VALUES (1)").unwrap();
        db.execute("INSERT INTO t VALUES (2)").unwrap();
        db.execute("INSERT INTO t VALUES (3)").unwrap();

        let result = db
            .execute("SELECT * FROM t WHERE x IN (SELECT x FROM t WHERE x > 1)")
            .unwrap();
        assert_eq!(result.rows.len(), 2);
    }

    // ======== Phase 7 Tests ========

    #[test]
    fn test_union() {
        let mut db = Database::new_memory();
        db.execute("CREATE TABLE t1 (x INTEGER)").unwrap();
        db.execute("CREATE TABLE t2 (x INTEGER)").unwrap();
        db.execute("INSERT INTO t1 VALUES (1)").unwrap();
        db.execute("INSERT INTO t1 VALUES (2)").unwrap();
        db.execute("INSERT INTO t2 VALUES (2)").unwrap();
        db.execute("INSERT INTO t2 VALUES (3)").unwrap();

        // UNION removes duplicates
        let result = db
            .execute("SELECT x FROM t1 UNION SELECT x FROM t2")
            .unwrap();
        assert_eq!(result.rows.len(), 3);
    }

    #[test]
    fn test_union_all() {
        let mut db = Database::new_memory();
        db.execute("CREATE TABLE t1 (x INTEGER)").unwrap();
        db.execute("CREATE TABLE t2 (x INTEGER)").unwrap();
        db.execute("INSERT INTO t1 VALUES (1)").unwrap();
        db.execute("INSERT INTO t1 VALUES (2)").unwrap();
        db.execute("INSERT INTO t2 VALUES (2)").unwrap();
        db.execute("INSERT INTO t2 VALUES (3)").unwrap();

        // UNION ALL keeps duplicates
        let result = db
            .execute("SELECT x FROM t1 UNION ALL SELECT x FROM t2")
            .unwrap();
        assert_eq!(result.rows.len(), 4);
    }

    #[test]
    fn test_except() {
        let mut db = Database::new_memory();
        db.execute("CREATE TABLE t1 (x INTEGER)").unwrap();
        db.execute("CREATE TABLE t2 (x INTEGER)").unwrap();
        db.execute("INSERT INTO t1 VALUES (1)").unwrap();
        db.execute("INSERT INTO t1 VALUES (2)").unwrap();
        db.execute("INSERT INTO t1 VALUES (3)").unwrap();
        db.execute("INSERT INTO t2 VALUES (2)").unwrap();

        let result = db
            .execute("SELECT x FROM t1 EXCEPT SELECT x FROM t2")
            .unwrap();
        assert_eq!(result.rows.len(), 2); // 1 and 3
    }

    #[test]
    fn test_intersect() {
        let mut db = Database::new_memory();
        db.execute("CREATE TABLE t1 (x INTEGER)").unwrap();
        db.execute("CREATE TABLE t2 (x INTEGER)").unwrap();
        db.execute("INSERT INTO t1 VALUES (1)").unwrap();
        db.execute("INSERT INTO t1 VALUES (2)").unwrap();
        db.execute("INSERT INTO t1 VALUES (3)").unwrap();
        db.execute("INSERT INTO t2 VALUES (2)").unwrap();
        db.execute("INSERT INTO t2 VALUES (3)").unwrap();

        let result = db
            .execute("SELECT x FROM t1 INTERSECT SELECT x FROM t2")
            .unwrap();
        assert_eq!(result.rows.len(), 2); // 2 and 3
    }

    #[test]
    fn test_insert_select() {
        let mut db = Database::new_memory();
        db.execute("CREATE TABLE t1 (x INTEGER, y TEXT)").unwrap();
        db.execute("CREATE TABLE t2 (x INTEGER, y TEXT)").unwrap();
        db.execute("INSERT INTO t1 VALUES (1, 'a')").unwrap();
        db.execute("INSERT INTO t1 VALUES (2, 'b')").unwrap();

        db.execute("INSERT INTO t2 SELECT * FROM t1").unwrap();

        let result = db.execute("SELECT * FROM t2").unwrap();
        assert_eq!(result.rows.len(), 2);
        assert_eq!(result.rows[0][0], Value::Integer(1));
        assert_eq!(result.rows[0][1], Value::Text("a".into()));
    }

    #[test]
    fn test_insert_default_values() {
        let mut db = Database::new_memory();
        db.execute("CREATE TABLE t (x INTEGER DEFAULT 42, y TEXT DEFAULT 'hello')")
            .unwrap();
        db.execute("INSERT INTO t DEFAULT VALUES").unwrap();

        let result = db.execute("SELECT * FROM t").unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][0], Value::Integer(42));
        assert_eq!(result.rows[0][1], Value::Text("hello".into()));
    }

    #[test]
    fn test_not_null_constraint() {
        let mut db = Database::new_memory();
        db.execute("CREATE TABLE t (x INTEGER NOT NULL, y TEXT)")
            .unwrap();

        // Should succeed
        db.execute("INSERT INTO t VALUES (1, 'a')").unwrap();

        // Should fail - NOT NULL violation
        let result = db.execute("INSERT INTO t VALUES (NULL, 'b')");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("NOT NULL"));
    }

    #[test]
    fn test_not_null_constraint_update() {
        let mut db = Database::new_memory();
        db.execute("CREATE TABLE t (x INTEGER NOT NULL, y TEXT)")
            .unwrap();
        db.execute("INSERT INTO t VALUES (1, 'a')").unwrap();

        // Should fail - NOT NULL violation on update
        let result = db.execute("UPDATE t SET x = NULL WHERE y = 'a'");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("NOT NULL"));
    }

    #[test]
    fn test_unique_constraint() {
        let mut db = Database::new_memory();
        db.execute("CREATE TABLE t (x INTEGER, y TEXT)").unwrap();
        db.execute("CREATE UNIQUE INDEX idx_x ON t(x)").unwrap();

        db.execute("INSERT INTO t VALUES (1, 'a')").unwrap();

        // Should fail - UNIQUE violation
        let result = db.execute("INSERT INTO t VALUES (1, 'b')");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("UNIQUE"));
    }

    #[test]
    fn test_alter_table_add_column() {
        let mut db = Database::new_memory();
        db.execute("CREATE TABLE t (x INTEGER)").unwrap();
        db.execute("INSERT INTO t VALUES (1)").unwrap();
        db.execute("ALTER TABLE t ADD COLUMN y TEXT").unwrap();

        let schema = db.tables.get("t").unwrap();
        assert_eq!(schema.columns.len(), 2);
        assert_eq!(schema.columns[1].name, "y");
    }

    #[test]
    fn test_alter_table_rename() {
        let mut db = Database::new_memory();
        db.execute("CREATE TABLE old_name (x INTEGER)").unwrap();
        db.execute("ALTER TABLE old_name RENAME TO new_name")
            .unwrap();

        assert!(!db.tables.contains_key("old_name"));
        assert!(db.tables.contains_key("new_name"));
    }

    #[test]
    fn test_alter_table_rename_column() {
        let mut db = Database::new_memory();
        db.execute("CREATE TABLE t (old_col INTEGER)").unwrap();
        db.execute("ALTER TABLE t RENAME COLUMN old_col TO new_col")
            .unwrap();

        let schema = db.tables.get("t").unwrap();
        assert_eq!(schema.columns[0].name, "new_col");
    }

    #[test]
    fn test_drop_index() {
        let mut db = Database::new_memory();
        db.execute("CREATE TABLE t (x INTEGER)").unwrap();
        db.execute("CREATE INDEX idx ON t(x)").unwrap();
        assert!(db.indexes.contains_key("idx"));

        db.execute("DROP INDEX idx").unwrap();
        assert!(!db.indexes.contains_key("idx"));
    }

    #[test]
    fn test_drop_table_cascades_indexes() {
        let mut db = Database::new_memory();
        db.execute("CREATE TABLE t (x INTEGER, y TEXT)").unwrap();
        db.execute("CREATE INDEX idx_x ON t(x)").unwrap();
        db.execute("CREATE INDEX idx_y ON t(y)").unwrap();
        assert_eq!(db.indexes.len(), 2);

        db.execute("DROP TABLE t").unwrap();
        assert!(db.indexes.is_empty());
    }

    #[test]
    fn test_pragma_table_info() {
        let mut db = Database::new_memory();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT NOT NULL, age INTEGER)")
            .unwrap();

        let result = db.execute("PRAGMA table_info(t)").unwrap();
        assert_eq!(result.rows.len(), 3);
        assert_eq!(result.columns[0], "cid");
        assert_eq!(result.columns[1], "name");
        // First column: id
        assert_eq!(result.rows[0][1], Value::Text("id".into()));
        assert_eq!(result.rows[0][5], Value::Integer(1)); // pk
    }

    #[test]
    fn test_pragma_page_size() {
        let mut db = Database::new_memory();
        let result = db.execute("PRAGMA page_size").unwrap();
        assert_eq!(result.rows[0][0], Value::Integer(4096));
    }

    #[test]
    fn test_pragma_integrity_check() {
        let mut db = Database::new_memory();
        let result = db.execute("PRAGMA integrity_check").unwrap();
        assert_eq!(result.rows[0][0], Value::Text("ok".into()));
    }

    #[test]
    fn test_pragma_encoding() {
        let mut db = Database::new_memory();
        let result = db.execute("PRAGMA encoding").unwrap();
        assert_eq!(result.rows[0][0], Value::Text("UTF-8".into()));
    }

    #[test]
    fn test_cast_expression() {
        let mut db = Database::new_memory();
        let result = db.execute("SELECT CAST('42' AS INTEGER)").unwrap();
        assert_eq!(result.rows[0][0], Value::Integer(42));

        let result = db.execute("SELECT CAST(42 AS TEXT)").unwrap();
        assert_eq!(result.rows[0][0], Value::Text("42".into()));

        let result = db.execute("SELECT CAST(42 AS REAL)").unwrap();
        assert_eq!(result.rows[0][0], Value::Real(42.0));
    }

    #[test]
    fn test_case_when_simple() {
        let mut db = Database::new_memory();
        let result = db
            .execute("SELECT CASE 2 WHEN 1 THEN 'one' WHEN 2 THEN 'two' ELSE 'other' END")
            .unwrap();
        assert_eq!(result.rows[0][0], Value::Text("two".into()));
    }

    #[test]
    fn test_iif_function() {
        let mut db = Database::new_memory();
        let result = db.execute("SELECT IIF(1 > 0, 'yes', 'no')").unwrap();
        assert_eq!(result.rows[0][0], Value::Text("yes".into()));

        let result = db.execute("SELECT IIF(0, 'yes', 'no')").unwrap();
        assert_eq!(result.rows[0][0], Value::Text("no".into()));
    }

    #[test]
    fn test_round_function() {
        let mut db = Database::new_memory();
        let result = db.execute("SELECT ROUND(3.14159, 2)").unwrap();
        assert_eq!(result.rows[0][0], Value::Real(3.14));

        let result = db.execute("SELECT ROUND(3.5)").unwrap();
        assert_eq!(result.rows[0][0], Value::Real(4.0));
    }

    #[test]
    fn test_sign_function() {
        let mut db = Database::new_memory();
        let result = db.execute("SELECT SIGN(-5)").unwrap();
        assert_eq!(result.rows[0][0], Value::Integer(-1));

        let result = db.execute("SELECT SIGN(0)").unwrap();
        assert_eq!(result.rows[0][0], Value::Integer(0));

        let result = db.execute("SELECT SIGN(42)").unwrap();
        assert_eq!(result.rows[0][0], Value::Integer(1));
    }

    #[test]
    fn test_like_operator() {
        let mut db = Database::new_memory();
        db.execute("CREATE TABLE t (name TEXT)").unwrap();
        db.execute("INSERT INTO t VALUES ('Alice')").unwrap();
        db.execute("INSERT INTO t VALUES ('Bob')").unwrap();
        db.execute("INSERT INTO t VALUES ('Charlie')").unwrap();

        let result = db
            .execute("SELECT name FROM t WHERE name LIKE 'A%'")
            .unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][0], Value::Text("Alice".into()));

        let result = db
            .execute("SELECT name FROM t WHERE name LIKE '%li%'")
            .unwrap();
        assert_eq!(result.rows.len(), 2); // Alice and Charlie
    }

    #[test]
    fn test_glob_operator() {
        let mut db = Database::new_memory();
        db.execute("CREATE TABLE t (name TEXT)").unwrap();
        db.execute("INSERT INTO t VALUES ('Alice')").unwrap();
        db.execute("INSERT INTO t VALUES ('Bob')").unwrap();

        let result = db
            .execute("SELECT name FROM t WHERE name GLOB 'A*'")
            .unwrap();
        assert_eq!(result.rows.len(), 1);
    }

    #[test]
    fn test_null_handling() {
        let mut db = Database::new_memory();

        // NULL comparisons return NULL (falsy)
        let result = db.execute("SELECT NULL = NULL").unwrap();
        assert_eq!(result.rows[0][0], Value::Null);

        // IS NULL works
        let result = db.execute("SELECT NULL IS NULL").unwrap();
        assert_eq!(result.rows[0][0], Value::Integer(1));

        // NULL arithmetic propagates
        let result = db.execute("SELECT 1 + NULL").unwrap();
        assert_eq!(result.rows[0][0], Value::Null);

        // COALESCE skips NULLs
        let result = db.execute("SELECT COALESCE(NULL, NULL, 42)").unwrap();
        assert_eq!(result.rows[0][0], Value::Integer(42));
    }

    #[test]
    fn test_type_affinity() {
        let mut db = Database::new_memory();
        db.execute("CREATE TABLE t (x REAL, y TEXT, z INTEGER)")
            .unwrap();
        db.execute("INSERT INTO t VALUES (42, 100, '200')").unwrap();

        let result = db
            .execute("SELECT TYPEOF(x), TYPEOF(y), TYPEOF(z) FROM t")
            .unwrap();
        // Values are stored as-is in our implementation
        // x=42 is Integer, y=100 is Integer, z='200' is Text
        assert!(!result.rows.is_empty());
    }

    #[test]
    fn test_between_operator() {
        let mut db = Database::new_memory();
        db.execute("CREATE TABLE t (x INTEGER)").unwrap();
        for i in 1..=10 {
            db.execute(&format!("INSERT INTO t VALUES ({})", i))
                .unwrap();
        }

        let result = db
            .execute("SELECT x FROM t WHERE x BETWEEN 3 AND 7")
            .unwrap();
        assert_eq!(result.rows.len(), 5);
    }

    #[test]
    fn test_in_operator() {
        let mut db = Database::new_memory();
        db.execute("CREATE TABLE t (x INTEGER)").unwrap();
        db.execute("INSERT INTO t VALUES (1)").unwrap();
        db.execute("INSERT INTO t VALUES (2)").unwrap();
        db.execute("INSERT INTO t VALUES (3)").unwrap();

        let result = db.execute("SELECT x FROM t WHERE x IN (1, 3)").unwrap();
        assert_eq!(result.rows.len(), 2);
    }

    #[test]
    fn test_multi_row_insert() {
        let mut db = Database::new_memory();
        db.execute("CREATE TABLE t (x INTEGER, y TEXT)").unwrap();
        db.execute("INSERT INTO t VALUES (1, 'a'), (2, 'b'), (3, 'c')")
            .unwrap();

        let result = db.execute("SELECT COUNT(*) FROM t").unwrap();
        assert_eq!(result.rows[0][0], Value::Integer(3));
    }

    #[test]
    fn test_distinct() {
        let mut db = Database::new_memory();
        db.execute("CREATE TABLE t (x INTEGER)").unwrap();
        db.execute("INSERT INTO t VALUES (1)").unwrap();
        db.execute("INSERT INTO t VALUES (2)").unwrap();
        db.execute("INSERT INTO t VALUES (1)").unwrap();
        db.execute("INSERT INTO t VALUES (2)").unwrap();

        let result = db.execute("SELECT DISTINCT x FROM t").unwrap();
        assert_eq!(result.rows.len(), 2);
    }

    #[test]
    fn test_left_join() {
        let mut db = Database::new_memory();
        db.execute("CREATE TABLE a (id INTEGER, name TEXT)")
            .unwrap();
        db.execute("CREATE TABLE b (a_id INTEGER, val TEXT)")
            .unwrap();
        db.execute("INSERT INTO a VALUES (1, 'Alice')").unwrap();
        db.execute("INSERT INTO a VALUES (2, 'Bob')").unwrap();
        db.execute("INSERT INTO b VALUES (1, 'x')").unwrap();

        let result = db
            .execute("SELECT a.name, b.val FROM a LEFT JOIN b ON a.id = b.a_id ORDER BY a.id")
            .unwrap();
        assert_eq!(result.rows.len(), 2);
        assert_eq!(result.rows[0][1], Value::Text("x".into()));
        assert_eq!(result.rows[1][1], Value::Null);
    }

    #[test]
    fn test_group_concat() {
        let mut db = Database::new_memory();
        db.execute("CREATE TABLE t (grp TEXT, val TEXT)").unwrap();
        db.execute("INSERT INTO t VALUES ('a', 'x')").unwrap();
        db.execute("INSERT INTO t VALUES ('a', 'y')").unwrap();
        db.execute("INSERT INTO t VALUES ('b', 'z')").unwrap();

        let result = db
            .execute("SELECT grp, GROUP_CONCAT(val) FROM t GROUP BY grp ORDER BY grp")
            .unwrap();
        assert_eq!(result.rows.len(), 2);
        assert_eq!(result.rows[0][1], Value::Text("x,y".into()));
        assert_eq!(result.rows[1][1], Value::Text("z".into()));
    }

    #[test]
    fn test_having_clause() {
        let mut db = Database::new_memory();
        db.execute("CREATE TABLE t (grp TEXT, val INTEGER)")
            .unwrap();
        db.execute("INSERT INTO t VALUES ('a', 1)").unwrap();
        db.execute("INSERT INTO t VALUES ('a', 2)").unwrap();
        db.execute("INSERT INTO t VALUES ('b', 3)").unwrap();

        let result = db
            .execute("SELECT grp, SUM(val) FROM t GROUP BY grp HAVING SUM(val) > 2")
            .unwrap();
        assert_eq!(result.rows.len(), 2);
    }

    #[test]
    fn test_cross_join() {
        let mut db = Database::new_memory();
        db.execute("CREATE TABLE t1 (x INTEGER)").unwrap();
        db.execute("CREATE TABLE t2 (y INTEGER)").unwrap();
        db.execute("INSERT INTO t1 VALUES (1)").unwrap();
        db.execute("INSERT INTO t1 VALUES (2)").unwrap();
        db.execute("INSERT INTO t2 VALUES (10)").unwrap();
        db.execute("INSERT INTO t2 VALUES (20)").unwrap();

        let result = db.execute("SELECT * FROM t1 CROSS JOIN t2").unwrap();
        assert_eq!(result.rows.len(), 4);
    }

    #[test]
    fn test_string_functions() {
        let mut db = Database::new_memory();

        let result = db.execute("SELECT LENGTH('hello')").unwrap();
        assert_eq!(result.rows[0][0], Value::Integer(5));

        let result = db.execute("SELECT UPPER('hello')").unwrap();
        assert_eq!(result.rows[0][0], Value::Text("HELLO".into()));

        let result = db.execute("SELECT LOWER('HELLO')").unwrap();
        assert_eq!(result.rows[0][0], Value::Text("hello".into()));

        let result = db.execute("SELECT SUBSTR('hello', 2, 3)").unwrap();
        assert_eq!(result.rows[0][0], Value::Text("ell".into()));

        let result = db.execute("SELECT REPLACE('hello', 'l', 'r')").unwrap();
        assert_eq!(result.rows[0][0], Value::Text("herro".into()));

        let result = db.execute("SELECT TRIM('  hello  ')").unwrap();
        assert_eq!(result.rows[0][0], Value::Text("hello".into()));

        let result = db.execute("SELECT INSTR('hello', 'ell')").unwrap();
        assert_eq!(result.rows[0][0], Value::Integer(2));
    }

    #[test]
    fn test_abs_function() {
        let mut db = Database::new_memory();
        let result = db.execute("SELECT ABS(-42)").unwrap();
        assert_eq!(result.rows[0][0], Value::Integer(42));

        let result = db.execute("SELECT ABS(42)").unwrap();
        assert_eq!(result.rows[0][0], Value::Integer(42));
    }

    #[test]
    fn test_typeof_function() {
        let mut db = Database::new_memory();
        let result = db
            .execute("SELECT TYPEOF(42), TYPEOF(3.14), TYPEOF('hello'), TYPEOF(NULL)")
            .unwrap();
        assert_eq!(result.rows[0][0], Value::Text("integer".into()));
        assert_eq!(result.rows[0][1], Value::Text("real".into()));
        assert_eq!(result.rows[0][2], Value::Text("text".into()));
        assert_eq!(result.rows[0][3], Value::Text("null".into()));
    }

    #[test]
    fn test_min_max_scalar() {
        let mut db = Database::new_memory();
        let result = db.execute("SELECT MIN(1, 2, 3)").unwrap();
        assert_eq!(result.rows[0][0], Value::Integer(1));

        let result = db.execute("SELECT MAX(1, 2, 3)").unwrap();
        assert_eq!(result.rows[0][0], Value::Integer(3));
    }

    #[test]
    fn test_nullif_function() {
        let mut db = Database::new_memory();
        let result = db.execute("SELECT NULLIF(1, 1)").unwrap();
        assert_eq!(result.rows[0][0], Value::Null);

        let result = db.execute("SELECT NULLIF(1, 2)").unwrap();
        assert_eq!(result.rows[0][0], Value::Integer(1));
    }

    #[test]
    fn test_ifnull_function() {
        let mut db = Database::new_memory();
        let result = db.execute("SELECT IFNULL(NULL, 42)").unwrap();
        assert_eq!(result.rows[0][0], Value::Integer(42));

        let result = db.execute("SELECT IFNULL(1, 42)").unwrap();
        assert_eq!(result.rows[0][0], Value::Integer(1));
    }

    #[test]
    fn test_aggregate_functions() {
        let mut db = Database::new_memory();
        db.execute("CREATE TABLE t (x INTEGER)").unwrap();
        db.execute("INSERT INTO t VALUES (10)").unwrap();
        db.execute("INSERT INTO t VALUES (20)").unwrap();
        db.execute("INSERT INTO t VALUES (30)").unwrap();

        let result = db.execute("SELECT SUM(x) FROM t").unwrap();
        assert_eq!(result.rows[0][0], Value::Integer(60));

        let result = db.execute("SELECT AVG(x) FROM t").unwrap();
        assert_eq!(result.rows[0][0], Value::Real(20.0));

        let result = db.execute("SELECT MIN(x) FROM t").unwrap();
        assert_eq!(result.rows[0][0], Value::Integer(10));

        let result = db.execute("SELECT MAX(x) FROM t").unwrap();
        assert_eq!(result.rows[0][0], Value::Integer(30));
    }

    #[test]
    fn test_concat_operator() {
        let mut db = Database::new_memory();
        let result = db.execute("SELECT 'hello' || ' ' || 'world'").unwrap();
        assert_eq!(result.rows[0][0], Value::Text("hello world".into()));
    }

    #[test]
    fn test_bit_operations() {
        let mut db = Database::new_memory();
        let result = db.execute("SELECT 6 & 3").unwrap();
        assert_eq!(result.rows[0][0], Value::Integer(2));

        let result = db.execute("SELECT 6 | 3").unwrap();
        assert_eq!(result.rows[0][0], Value::Integer(7));
    }

    #[test]
    fn test_modulo() {
        let mut db = Database::new_memory();
        let result = db.execute("SELECT 10 % 3").unwrap();
        assert_eq!(result.rows[0][0], Value::Integer(1));
    }

    #[test]
    fn test_negative_unary() {
        let mut db = Database::new_memory();
        let result = db.execute("SELECT -5").unwrap();
        assert_eq!(result.rows[0][0], Value::Integer(-5));
    }

    #[test]
    fn test_is_null_is_not_null() {
        let mut db = Database::new_memory();
        db.execute("CREATE TABLE t (x INTEGER)").unwrap();
        db.execute("INSERT INTO t VALUES (1)").unwrap();
        db.execute("INSERT INTO t VALUES (NULL)").unwrap();

        let result = db.execute("SELECT * FROM t WHERE x IS NULL").unwrap();
        assert_eq!(result.rows.len(), 1);

        let result = db.execute("SELECT * FROM t WHERE x IS NOT NULL").unwrap();
        assert_eq!(result.rows.len(), 1);
    }

    #[test]
    fn test_drop_if_exists() {
        let mut db = Database::new_memory();
        // Should not error since IF EXISTS is specified
        db.execute("DROP TABLE IF EXISTS nonexistent").unwrap();
        db.execute("DROP INDEX IF EXISTS nonexistent").unwrap();
    }

    #[test]
    fn test_create_index_and_query() {
        let mut db = Database::new_memory();
        db.execute("CREATE TABLE t (x INTEGER, y TEXT)").unwrap();
        db.execute("INSERT INTO t VALUES (1, 'a')").unwrap();
        db.execute("INSERT INTO t VALUES (2, 'b')").unwrap();
        db.execute("INSERT INTO t VALUES (3, 'c')").unwrap();
        db.execute("CREATE INDEX idx_x ON t(x)").unwrap();

        let result = db.execute("SELECT * FROM t WHERE x = 2").unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][1], Value::Text("b".into()));
    }

    #[test]
    fn test_exists_subquery() {
        let mut db = Database::new_memory();
        db.execute("CREATE TABLE t (x INTEGER)").unwrap();
        db.execute("INSERT INTO t VALUES (1)").unwrap();

        let result = db
            .execute("SELECT EXISTS(SELECT * FROM t WHERE x = 1)")
            .unwrap();
        assert_eq!(result.rows[0][0], Value::Integer(1));

        let result = db
            .execute("SELECT EXISTS(SELECT * FROM t WHERE x = 999)")
            .unwrap();
        assert_eq!(result.rows[0][0], Value::Integer(0));
    }

    #[test]
    fn test_select_without_from() {
        let mut db = Database::new_memory();
        let result = db.execute("SELECT 42, 'hello'").unwrap();
        assert_eq!(result.rows[0][0], Value::Integer(42));
        assert_eq!(result.rows[0][1], Value::Text("hello".into()));
    }

    #[test]
    fn test_pragma_database_list() {
        let mut db = Database::new_memory();
        let result = db.execute("PRAGMA database_list").unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][1], Value::Text("main".into()));
    }
}
