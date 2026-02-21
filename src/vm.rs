// Virtual machine / query executor.
//
// Uses a Volcano (iterator) model: each operator implements open/next/close.
// The executor evaluates expressions, applies predicates, and produces rows.

use crate::ast::{
    BinaryOp, Expr, FunctionArgs, LiteralValue, ResultColumn, SelectStatement, UnaryOp,
};
use crate::btree::BTreeCursor;
use crate::error::{Result, RsqliteError};
use crate::pager::Pager;
use crate::record;
use crate::types::Value;

/// A row is a vector of named columns with their values.
#[derive(Debug, Clone)]
pub struct Row {
    pub values: Vec<Value>,
}

/// Column metadata for a result set.
#[derive(Debug, Clone)]
pub struct ColumnInfo {
    pub name: String,
    pub table: Option<String>,
}

/// Schema information for a table, used during query execution.
#[derive(Debug, Clone)]
pub struct TableSchema {
    pub name: String,
    pub columns: Vec<String>,
    pub root_page: u32,
    /// Index of the column that is INTEGER PRIMARY KEY (rowid alias), if any.
    /// When set, this column's value is not stored in the record payload but
    /// is instead the rowid of the B-tree entry.
    pub rowid_column: Option<usize>,
}

/// Result of executing a query.
#[derive(Debug)]
pub struct QueryResult {
    pub columns: Vec<ColumnInfo>,
    pub rows: Vec<Row>,
}

/// Execute a SELECT statement against the database.
pub fn execute_select(
    pager: &mut Pager,
    stmt: &SelectStatement,
    tables: &[TableSchema],
) -> Result<QueryResult> {
    // Determine the source table.
    let (table_schema, _table_alias) = match &stmt.from {
        Some(from) => {
            let (table_name, alias) = match &from.table {
                crate::ast::TableRef::Table { name, alias } => {
                    (name.as_str(), alias.as_deref().unwrap_or(name.as_str()))
                }
                crate::ast::TableRef::Subquery { .. } => {
                    return Err(RsqliteError::NotImplemented(
                        "subqueries in FROM".into(),
                    ));
                }
            };
            let schema = tables
                .iter()
                .find(|t| t.name.eq_ignore_ascii_case(table_name))
                .ok_or_else(|| {
                    RsqliteError::Runtime(format!("no such table: {table_name}"))
                })?;
            (Some(schema), alias.to_string())
        }
        None => (None, String::new()),
    };

    if !stmt.from.as_ref().map(|f| f.joins.is_empty()).unwrap_or(true) {
        return Err(RsqliteError::NotImplemented("JOINs".into()));
    }

    // Scan the table.
    let mut raw_rows: Vec<(i64, Vec<Value>)> = Vec::new();

    if let Some(schema) = table_schema {
        let mut cursor = BTreeCursor::new(schema.root_page);
        cursor.move_to_first(pager)?;

        while cursor.is_valid() {
            let rowid = cursor.current_rowid(pager)?;
            let payload = cursor.current_payload(pager)?;
            let mut values = record::decode_record(&payload)?;

            // If this table has an INTEGER PRIMARY KEY column (rowid alias),
            // the record stores NULL for that column. Replace it with the rowid.
            if let Some(pk_idx) = schema.rowid_column {
                if pk_idx < values.len() {
                    values[pk_idx] = Value::Integer(rowid);
                } else if values.len() < schema.columns.len() {
                    // Some schemas omit the column entirely.
                    values.insert(pk_idx, Value::Integer(rowid));
                }
            }

            raw_rows.push((rowid, values));
            cursor.move_to_next(pager)?;
        }
    }

    // Apply WHERE filter.
    let column_names: Vec<&str> = table_schema
        .map(|s| s.columns.iter().map(|c| c.as_str()).collect())
        .unwrap_or_default();
    let table_name = table_schema.map(|s| s.name.as_str());

    let mut filtered_rows: Vec<(i64, Vec<Value>)> = Vec::new();
    for (rowid, values) in &raw_rows {
        if let Some(ref where_expr) = stmt.where_clause {
            let result = eval_expr(where_expr, &column_names, values, *rowid, table_name)?;
            if result.is_truthy() {
                filtered_rows.push((*rowid, values.clone()));
            }
        } else {
            filtered_rows.push((*rowid, values.clone()));
        }
    }

    // Build result columns.
    let (result_columns, result_rows) =
        project_columns(&stmt.columns, &column_names, &filtered_rows, table_name)?;

    // Apply ORDER BY.
    let mut result_rows = result_rows;
    if let Some(ref order_by) = stmt.order_by {
        // Build sort keys for each row.
        let mut indexed: Vec<(usize, Vec<Value>)> = Vec::new();
        for (i, (rowid, values)) in filtered_rows.iter().enumerate() {
            let mut keys = Vec::new();
            for item in order_by {
                let key = eval_expr(&item.expr, &column_names, values, *rowid, table_name)?;
                keys.push(key);
            }
            indexed.push((i, keys));
        }

        indexed.sort_by(|a, b| {
            for (idx, item) in order_by.iter().enumerate() {
                let cmp = crate::types::sqlite_cmp(&a.1[idx], &b.1[idx]);
                let cmp = if item.direction == crate::ast::SortDirection::Desc {
                    cmp.reverse()
                } else {
                    cmp
                };
                if cmp != std::cmp::Ordering::Equal {
                    return cmp;
                }
            }
            std::cmp::Ordering::Equal
        });

        let sorted_rows: Vec<Row> = indexed
            .iter()
            .map(|(i, _)| result_rows[*i].clone())
            .collect();
        result_rows = sorted_rows;
    }

    // Apply LIMIT and OFFSET.
    if let Some(ref limit_clause) = stmt.limit {
        let limit_val = match &limit_clause.limit {
            Expr::Literal(LiteralValue::Integer(n)) => *n as usize,
            _ => {
                return Err(RsqliteError::NotImplemented(
                    "non-literal LIMIT".into(),
                ))
            }
        };
        let offset_val = match &limit_clause.offset {
            Some(Expr::Literal(LiteralValue::Integer(n))) => *n as usize,
            Some(_) => {
                return Err(RsqliteError::NotImplemented(
                    "non-literal OFFSET".into(),
                ))
            }
            None => 0,
        };
        let start = offset_val.min(result_rows.len());
        let end = (start + limit_val).min(result_rows.len());
        result_rows = result_rows[start..end].to_vec();
    }

    Ok(QueryResult {
        columns: result_columns,
        rows: result_rows,
    })
}

/// Project raw rows into result columns based on the SELECT column list.
fn project_columns(
    select_columns: &[ResultColumn],
    column_names: &[&str],
    rows: &[(i64, Vec<Value>)],
    table_name: Option<&str>,
) -> Result<(Vec<ColumnInfo>, Vec<Row>)> {
    // First, resolve what each result column maps to.
    let mut col_infos: Vec<ColumnInfo> = Vec::new();
    let mut projections: Vec<ColumnProjection> = Vec::new();

    for rc in select_columns {
        match rc {
            ResultColumn::AllColumns => {
                for (i, &name) in column_names.iter().enumerate() {
                    col_infos.push(ColumnInfo {
                        name: name.to_string(),
                        table: table_name.map(|s| s.to_string()),
                    });
                    projections.push(ColumnProjection::Index(i));
                }
            }
            ResultColumn::TableAllColumns(tname) => {
                // Only support single-table for now.
                if table_name.is_some()
                    && table_name.unwrap().eq_ignore_ascii_case(tname)
                {
                    for (i, &name) in column_names.iter().enumerate() {
                        col_infos.push(ColumnInfo {
                            name: name.to_string(),
                            table: Some(tname.clone()),
                        });
                        projections.push(ColumnProjection::Index(i));
                    }
                } else {
                    return Err(RsqliteError::Runtime(format!(
                        "no such table: {tname}"
                    )));
                }
            }
            ResultColumn::Expr { expr, alias } => {
                let name = alias.clone().unwrap_or_else(|| expr_display_name(expr));
                col_infos.push(ColumnInfo {
                    name,
                    table: None,
                });
                projections.push(ColumnProjection::Expr(expr.clone()));
            }
        }
    }

    // If no columns specified (shouldn't happen in valid SQL, but handle gracefully).
    if col_infos.is_empty() && column_names.is_empty() {
        return Ok((vec![], vec![]));
    }

    // Project each row.
    let mut result_rows = Vec::with_capacity(rows.len());
    for (rowid, values) in rows {
        let mut row_values = Vec::with_capacity(projections.len());
        for proj in &projections {
            match proj {
                ColumnProjection::Index(i) => {
                    let val = values.get(*i).cloned().unwrap_or(Value::Null);
                    row_values.push(val);
                }
                ColumnProjection::Expr(expr) => {
                    let val =
                        eval_expr(expr, column_names, values, *rowid, table_name)?;
                    row_values.push(val);
                }
            }
        }
        result_rows.push(Row {
            values: row_values,
        });
    }

    Ok((col_infos, result_rows))
}

#[derive(Clone)]
enum ColumnProjection {
    Index(usize),
    Expr(Expr),
}

/// Generate a display name for an expression (used when no alias is given).
fn expr_display_name(expr: &Expr) -> String {
    match expr {
        Expr::ColumnRef { table, column } => {
            if let Some(t) = table {
                format!("{t}.{column}")
            } else {
                column.clone()
            }
        }
        Expr::Literal(LiteralValue::Integer(n)) => n.to_string(),
        Expr::Literal(LiteralValue::Real(f)) => f.to_string(),
        Expr::Literal(LiteralValue::String(s)) => format!("'{s}'"),
        Expr::Literal(LiteralValue::Null) => "NULL".to_string(),
        Expr::FunctionCall { name, .. } => format!("{name}(...)"),
        _ => "?".to_string(),
    }
}

/// Resolve a column name to its index in the row.
fn resolve_column(
    column: &str,
    table: Option<&str>,
    column_names: &[&str],
    row_table: Option<&str>,
) -> Result<usize> {
    // Check table qualifier matches if provided.
    if let Some(t) = table {
        if let Some(rt) = row_table {
            if !t.eq_ignore_ascii_case(rt) {
                return Err(RsqliteError::Runtime(format!(
                    "no such table: {t}"
                )));
            }
        }
    }

    // Handle rowid aliases.
    if column.eq_ignore_ascii_case("rowid")
        || column.eq_ignore_ascii_case("_rowid_")
        || column.eq_ignore_ascii_case("oid")
    {
        return Err(RsqliteError::Runtime("__rowid__".into())); // Special marker
    }

    column_names
        .iter()
        .position(|&n| n.eq_ignore_ascii_case(column))
        .ok_or_else(|| RsqliteError::Runtime(format!("no such column: {column}")))
}

/// Evaluate an expression given a row's values and column names.
pub fn eval_expr(
    expr: &Expr,
    column_names: &[&str],
    values: &[Value],
    rowid: i64,
    table_name: Option<&str>,
) -> Result<Value> {
    match expr {
        Expr::Literal(lit) => Ok(match lit {
            LiteralValue::Integer(n) => Value::Integer(*n),
            LiteralValue::Real(f) => Value::Real(*f),
            LiteralValue::String(s) => Value::Text(s.clone()),
            LiteralValue::Blob(b) => Value::Blob(b.clone()),
            LiteralValue::Null => Value::Null,
            LiteralValue::CurrentTime
            | LiteralValue::CurrentDate
            | LiteralValue::CurrentTimestamp => {
                Value::Text("(not implemented)".into())
            }
        }),

        Expr::ColumnRef { table, column } => {
            match resolve_column(column, table.as_deref(), column_names, table_name) {
                Ok(idx) => Ok(values.get(idx).cloned().unwrap_or(Value::Null)),
                Err(RsqliteError::Runtime(ref msg)) if msg == "__rowid__" => {
                    Ok(Value::Integer(rowid))
                }
                Err(e) => Err(e),
            }
        }

        Expr::UnaryOp { op, operand } => {
            let val = eval_expr(operand, column_names, values, rowid, table_name)?;
            match op {
                UnaryOp::Negate => match val {
                    Value::Integer(n) => Ok(Value::Integer(-n)),
                    Value::Real(f) => Ok(Value::Real(-f)),
                    Value::Null => Ok(Value::Null),
                    _ => Ok(Value::Integer(0)),
                },
                UnaryOp::Plus => Ok(val),
                UnaryOp::Not => Ok(Value::Integer(if val.is_truthy() { 0 } else { 1 })),
                UnaryOp::BitwiseNot => match val {
                    Value::Integer(n) => Ok(Value::Integer(!n)),
                    Value::Null => Ok(Value::Null),
                    _ => Ok(Value::Integer(0)),
                },
            }
        }

        Expr::BinaryOp { left, op, right } => {
            // Short-circuit for AND/OR.
            match op {
                BinaryOp::And => {
                    let lv = eval_expr(left, column_names, values, rowid, table_name)?;
                    if lv.is_null() {
                        let rv = eval_expr(right, column_names, values, rowid, table_name)?;
                        if rv.is_truthy() {
                            return Ok(Value::Null);
                        }
                        return Ok(Value::Integer(0));
                    }
                    if !lv.is_truthy() {
                        return Ok(Value::Integer(0));
                    }
                    let rv = eval_expr(right, column_names, values, rowid, table_name)?;
                    return Ok(Value::Integer(if rv.is_truthy() { 1 } else { 0 }));
                }
                BinaryOp::Or => {
                    let lv = eval_expr(left, column_names, values, rowid, table_name)?;
                    if !lv.is_null() && lv.is_truthy() {
                        return Ok(Value::Integer(1));
                    }
                    let rv = eval_expr(right, column_names, values, rowid, table_name)?;
                    if rv.is_truthy() {
                        return Ok(Value::Integer(1));
                    }
                    if lv.is_null() || rv.is_null() {
                        return Ok(Value::Null);
                    }
                    return Ok(Value::Integer(0));
                }
                _ => {}
            }

            let lv = eval_expr(left, column_names, values, rowid, table_name)?;
            let rv = eval_expr(right, column_names, values, rowid, table_name)?;

            eval_binary_op(op, &lv, &rv)
        }

        Expr::IsNull { operand, negated } => {
            let val = eval_expr(operand, column_names, values, rowid, table_name)?;
            let is_null = val.is_null();
            let result = if *negated { !is_null } else { is_null };
            Ok(Value::Integer(if result { 1 } else { 0 }))
        }

        Expr::Between {
            operand,
            low,
            high,
            negated,
        } => {
            let val = eval_expr(operand, column_names, values, rowid, table_name)?;
            let lo = eval_expr(low, column_names, values, rowid, table_name)?;
            let hi = eval_expr(high, column_names, values, rowid, table_name)?;

            let ge_lo = crate::types::sqlite_cmp(&val, &lo) != std::cmp::Ordering::Less;
            let le_hi = crate::types::sqlite_cmp(&val, &hi) != std::cmp::Ordering::Greater;
            let in_range = ge_lo && le_hi;
            let result = if *negated { !in_range } else { in_range };
            Ok(Value::Integer(if result { 1 } else { 0 }))
        }

        Expr::In {
            operand,
            list,
            negated,
        } => {
            let val = eval_expr(operand, column_names, values, rowid, table_name)?;
            match list {
                crate::ast::InList::Values(exprs) => {
                    let mut found = false;
                    for e in exprs {
                        let item = eval_expr(e, column_names, values, rowid, table_name)?;
                        if crate::types::sqlite_cmp(&val, &item) == std::cmp::Ordering::Equal {
                            found = true;
                            break;
                        }
                    }
                    let result = if *negated { !found } else { found };
                    Ok(Value::Integer(if result { 1 } else { 0 }))
                }
                crate::ast::InList::Subquery(_) => Err(RsqliteError::NotImplemented(
                    "IN subquery".into(),
                )),
            }
        }

        Expr::Like {
            operand,
            pattern,
            escape: _,
            negated,
        } => {
            let val = eval_expr(operand, column_names, values, rowid, table_name)?;
            let pat = eval_expr(pattern, column_names, values, rowid, table_name)?;

            match (&val, &pat) {
                (Value::Null, _) | (_, Value::Null) => Ok(Value::Null),
                (Value::Text(s), Value::Text(p)) => {
                    let matches = sqlite_like(p, s);
                    let result = if *negated { !matches } else { matches };
                    Ok(Value::Integer(if result { 1 } else { 0 }))
                }
                _ => {
                    let s = val.to_text().unwrap_or_default();
                    let p = pat.to_text().unwrap_or_default();
                    let matches = sqlite_like(&p, &s);
                    let result = if *negated { !matches } else { matches };
                    Ok(Value::Integer(if result { 1 } else { 0 }))
                }
            }
        }

        Expr::FunctionCall { name, args } => {
            eval_function(name, args, column_names, values, rowid, table_name)
        }

        Expr::Parenthesized(inner) => {
            eval_expr(inner, column_names, values, rowid, table_name)
        }

        Expr::Cast { expr, type_name } => {
            let val = eval_expr(expr, column_names, values, rowid, table_name)?;
            let affinity = crate::types::determine_affinity(type_name);
            Ok(crate::types::apply_affinity(val, affinity))
        }

        Expr::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            if let Some(base) = operand {
                let base_val = eval_expr(base, column_names, values, rowid, table_name)?;
                for (when_expr, then_expr) in when_clauses {
                    let when_val =
                        eval_expr(when_expr, column_names, values, rowid, table_name)?;
                    if crate::types::sqlite_cmp(&base_val, &when_val)
                        == std::cmp::Ordering::Equal
                    {
                        return eval_expr(then_expr, column_names, values, rowid, table_name);
                    }
                }
            } else {
                for (when_expr, then_expr) in when_clauses {
                    let when_val =
                        eval_expr(when_expr, column_names, values, rowid, table_name)?;
                    if when_val.is_truthy() {
                        return eval_expr(then_expr, column_names, values, rowid, table_name);
                    }
                }
            }
            if let Some(else_expr) = else_clause {
                eval_expr(else_expr, column_names, values, rowid, table_name)
            } else {
                Ok(Value::Null)
            }
        }

        Expr::Collate { expr, .. } => {
            // Ignore collation for now, just evaluate the expression.
            eval_expr(expr, column_names, values, rowid, table_name)
        }

        _ => Err(RsqliteError::NotImplemented(format!(
            "expression type: {expr:?}"
        ))),
    }
}

/// Evaluate a binary operation on two values.
fn eval_binary_op(op: &BinaryOp, lv: &Value, rv: &Value) -> Result<Value> {
    // NULL propagation for most operators.
    if (lv.is_null() || rv.is_null())
        && !matches!(op, BinaryOp::Is | BinaryOp::IsNot)
    {
        return Ok(Value::Null);
    }

    match op {
        BinaryOp::Add => numeric_op(lv, rv, |a, b| a + b, |a, b| a + b),
        BinaryOp::Subtract => numeric_op(lv, rv, |a, b| a - b, |a, b| a - b),
        BinaryOp::Multiply => numeric_op(lv, rv, |a, b| a * b, |a, b| a * b),
        BinaryOp::Divide => {
            // Check for division by zero.
            match rv {
                Value::Integer(0) => Ok(Value::Null),
                Value::Real(f) if *f == 0.0 => Ok(Value::Null),
                _ => numeric_op(lv, rv, |a, b| a / b, |a, b| a / b),
            }
        }
        BinaryOp::Modulo => match (lv, rv) {
            (Value::Integer(a), Value::Integer(b)) => {
                if *b == 0 {
                    Ok(Value::Null)
                } else {
                    Ok(Value::Integer(a % b))
                }
            }
            _ => Ok(Value::Null),
        },

        BinaryOp::Eq => Ok(Value::Integer(
            if crate::types::sqlite_cmp(lv, rv) == std::cmp::Ordering::Equal {
                1
            } else {
                0
            },
        )),
        BinaryOp::NotEq => Ok(Value::Integer(
            if crate::types::sqlite_cmp(lv, rv) != std::cmp::Ordering::Equal {
                1
            } else {
                0
            },
        )),
        BinaryOp::Lt => Ok(Value::Integer(
            if crate::types::sqlite_cmp(lv, rv) == std::cmp::Ordering::Less {
                1
            } else {
                0
            },
        )),
        BinaryOp::Gt => Ok(Value::Integer(
            if crate::types::sqlite_cmp(lv, rv) == std::cmp::Ordering::Greater {
                1
            } else {
                0
            },
        )),
        BinaryOp::Le => Ok(Value::Integer(
            if crate::types::sqlite_cmp(lv, rv) != std::cmp::Ordering::Greater {
                1
            } else {
                0
            },
        )),
        BinaryOp::Ge => Ok(Value::Integer(
            if crate::types::sqlite_cmp(lv, rv) != std::cmp::Ordering::Less {
                1
            } else {
                0
            },
        )),

        BinaryOp::Is => Ok(Value::Integer(
            if crate::types::sqlite_cmp(lv, rv) == std::cmp::Ordering::Equal {
                1
            } else {
                0
            },
        )),
        BinaryOp::IsNot => Ok(Value::Integer(
            if crate::types::sqlite_cmp(lv, rv) != std::cmp::Ordering::Equal {
                1
            } else {
                0
            },
        )),

        BinaryOp::Concat => {
            let ls = lv.to_text().unwrap_or_default();
            let rs = rv.to_text().unwrap_or_default();
            Ok(Value::Text(format!("{ls}{rs}")))
        }

        BinaryOp::BitAnd => match (lv, rv) {
            (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a & b)),
            _ => Ok(Value::Integer(0)),
        },
        BinaryOp::BitOr => match (lv, rv) {
            (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a | b)),
            _ => Ok(Value::Integer(0)),
        },
        BinaryOp::ShiftLeft => match (lv, rv) {
            (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a << b)),
            _ => Ok(Value::Integer(0)),
        },
        BinaryOp::ShiftRight => match (lv, rv) {
            (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a >> b)),
            _ => Ok(Value::Integer(0)),
        },

        BinaryOp::Like => {
            let s = lv.to_text().unwrap_or_default();
            let p = rv.to_text().unwrap_or_default();
            Ok(Value::Integer(if sqlite_like(&p, &s) { 1 } else { 0 }))
        }

        BinaryOp::Glob => {
            let s = lv.to_text().unwrap_or_default();
            let p = rv.to_text().unwrap_or_default();
            Ok(Value::Integer(if sqlite_glob(&p, &s) { 1 } else { 0 }))
        }

        // AND/OR handled in eval_expr for short-circuit.
        BinaryOp::And | BinaryOp::Or => unreachable!(),
    }
}

/// Perform a numeric operation, coercing to int or float as needed.
fn numeric_op(
    lv: &Value,
    rv: &Value,
    int_op: impl Fn(i64, i64) -> i64,
    float_op: impl Fn(f64, f64) -> f64,
) -> Result<Value> {
    match (lv, rv) {
        (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(int_op(*a, *b))),
        (Value::Real(a), Value::Real(b)) => Ok(Value::Real(float_op(*a, *b))),
        (Value::Integer(a), Value::Real(b)) => Ok(Value::Real(float_op(*a as f64, *b))),
        (Value::Real(a), Value::Integer(b)) => Ok(Value::Real(float_op(*a, *b as f64))),
        _ => {
            // Try to coerce to numeric.
            let a = lv.to_real().unwrap_or(0.0);
            let b = rv.to_real().unwrap_or(0.0);
            Ok(Value::Real(float_op(a, b)))
        }
    }
}

/// Evaluate a scalar function call (non-aggregate, per-row).
fn eval_function(
    name: &str,
    args: &FunctionArgs,
    column_names: &[&str],
    values: &[Value],
    rowid: i64,
    table_name: Option<&str>,
) -> Result<Value> {
    let upper = name.to_uppercase();

    let arg_values: Vec<Value> = match args {
        FunctionArgs::Wildcard => vec![],
        FunctionArgs::Exprs { args: exprs, .. } => {
            let mut vals = Vec::new();
            for e in exprs {
                vals.push(eval_expr(e, column_names, values, rowid, table_name)?);
            }
            vals
        }
    };

    match upper.as_str() {
        "LENGTH" => {
            if arg_values.len() != 1 {
                return Err(RsqliteError::Runtime("length() takes 1 argument".into()));
            }
            match &arg_values[0] {
                Value::Null => Ok(Value::Null),
                Value::Text(s) => Ok(Value::Integer(s.len() as i64)),
                Value::Blob(b) => Ok(Value::Integer(b.len() as i64)),
                other => Ok(Value::Integer(
                    other.to_text().unwrap_or_default().len() as i64,
                )),
            }
        }
        "UPPER" => {
            if arg_values.len() != 1 {
                return Err(RsqliteError::Runtime("upper() takes 1 argument".into()));
            }
            match &arg_values[0] {
                Value::Null => Ok(Value::Null),
                Value::Text(s) => Ok(Value::Text(s.to_uppercase())),
                other => Ok(Value::Text(other.to_text().unwrap_or_default().to_uppercase())),
            }
        }
        "LOWER" => {
            if arg_values.len() != 1 {
                return Err(RsqliteError::Runtime("lower() takes 1 argument".into()));
            }
            match &arg_values[0] {
                Value::Null => Ok(Value::Null),
                Value::Text(s) => Ok(Value::Text(s.to_lowercase())),
                other => Ok(Value::Text(other.to_text().unwrap_or_default().to_lowercase())),
            }
        }
        "ABS" => {
            if arg_values.len() != 1 {
                return Err(RsqliteError::Runtime("abs() takes 1 argument".into()));
            }
            match &arg_values[0] {
                Value::Null => Ok(Value::Null),
                Value::Integer(n) => Ok(Value::Integer(n.abs())),
                Value::Real(f) => Ok(Value::Real(f.abs())),
                _ => Ok(Value::Integer(0)),
            }
        }
        "TYPEOF" => {
            if arg_values.len() != 1 {
                return Err(RsqliteError::Runtime("typeof() takes 1 argument".into()));
            }
            Ok(Value::Text(arg_values[0].type_name().to_string()))
        }
        "COALESCE" => {
            for v in &arg_values {
                if !v.is_null() {
                    return Ok(v.clone());
                }
            }
            Ok(Value::Null)
        }
        "IFNULL" => {
            if arg_values.len() != 2 {
                return Err(RsqliteError::Runtime("ifnull() takes 2 arguments".into()));
            }
            if arg_values[0].is_null() {
                Ok(arg_values[1].clone())
            } else {
                Ok(arg_values[0].clone())
            }
        }
        "NULLIF" => {
            if arg_values.len() != 2 {
                return Err(RsqliteError::Runtime("nullif() takes 2 arguments".into()));
            }
            if crate::types::sqlite_cmp(&arg_values[0], &arg_values[1])
                == std::cmp::Ordering::Equal
            {
                Ok(Value::Null)
            } else {
                Ok(arg_values[0].clone())
            }
        }
        "MAX" => {
            if arg_values.is_empty() {
                return Ok(Value::Null);
            }
            let mut max = &arg_values[0];
            for v in &arg_values[1..] {
                if crate::types::sqlite_cmp(v, max) == std::cmp::Ordering::Greater {
                    max = v;
                }
            }
            Ok(max.clone())
        }
        "MIN" => {
            if arg_values.is_empty() {
                return Ok(Value::Null);
            }
            let mut min = &arg_values[0];
            for v in &arg_values[1..] {
                if crate::types::sqlite_cmp(v, min) == std::cmp::Ordering::Less {
                    min = v;
                }
            }
            Ok(min.clone())
        }
        _ => Err(RsqliteError::Runtime(format!(
            "no such function: {name}"
        ))),
    }
}

/// SQLite LIKE pattern matching (case-insensitive for ASCII).
/// `%` matches any sequence, `_` matches any single character.
fn sqlite_like(pattern: &str, string: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let s: Vec<char> = string.chars().collect();
    like_match(&p, 0, &s, 0)
}

fn like_match(p: &[char], pi: usize, s: &[char], si: usize) -> bool {
    if pi == p.len() {
        return si == s.len();
    }

    match p[pi] {
        '%' => {
            // Match zero or more characters.
            for i in si..=s.len() {
                if like_match(p, pi + 1, s, i) {
                    return true;
                }
            }
            false
        }
        '_' => {
            // Match exactly one character.
            if si < s.len() {
                like_match(p, pi + 1, s, si + 1)
            } else {
                false
            }
        }
        c => {
            if si < s.len()
                && c.to_ascii_lowercase() == s[si].to_ascii_lowercase()
            {
                like_match(p, pi + 1, s, si + 1)
            } else {
                false
            }
        }
    }
}

/// SQLite GLOB pattern matching (case-sensitive).
/// `*` matches any sequence, `?` matches any single character.
fn sqlite_glob(pattern: &str, string: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let s: Vec<char> = string.chars().collect();
    glob_match(&p, 0, &s, 0)
}

fn glob_match(p: &[char], pi: usize, s: &[char], si: usize) -> bool {
    if pi == p.len() {
        return si == s.len();
    }

    match p[pi] {
        '*' => {
            for i in si..=s.len() {
                if glob_match(p, pi + 1, s, i) {
                    return true;
                }
            }
            false
        }
        '?' => {
            if si < s.len() {
                glob_match(p, pi + 1, s, si + 1)
            } else {
                false
            }
        }
        c => {
            if si < s.len() && c == s[si] {
                glob_match(p, pi + 1, s, si + 1)
            } else {
                false
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Value;

    #[test]
    fn test_like_matching() {
        assert!(sqlite_like("%", "anything"));
        assert!(sqlite_like("%", ""));
        assert!(sqlite_like("hello", "hello"));
        assert!(sqlite_like("hello", "HELLO")); // case-insensitive
        assert!(sqlite_like("h%o", "hello"));
        assert!(sqlite_like("h_llo", "hello"));
        assert!(!sqlite_like("h_llo", "hllo"));
        assert!(sqlite_like("%world", "hello world"));
        assert!(sqlite_like("hello%", "hello world"));
        assert!(sqlite_like("%ll%", "hello"));
        assert!(!sqlite_like("xyz", "hello"));
    }

    #[test]
    fn test_glob_matching() {
        assert!(sqlite_glob("*", "anything"));
        assert!(sqlite_glob("*", ""));
        assert!(sqlite_glob("hello", "hello"));
        assert!(!sqlite_glob("hello", "HELLO")); // case-sensitive
        assert!(sqlite_glob("h*o", "hello"));
        assert!(sqlite_glob("h?llo", "hello"));
        assert!(!sqlite_glob("h?llo", "hllo"));
    }

    #[test]
    fn test_eval_literal() {
        let val = eval_expr(
            &Expr::Literal(LiteralValue::Integer(42)),
            &[],
            &[],
            0,
            None,
        )
        .unwrap();
        assert_eq!(val, Value::Integer(42));
    }

    #[test]
    fn test_eval_column_ref() {
        let cols = ["name", "age"];
        let values = [Value::Text("Alice".into()), Value::Integer(30)];
        let expr = Expr::ColumnRef {
            table: None,
            column: "age".into(),
        };
        let val = eval_expr(&expr, &cols, &values, 1, None).unwrap();
        assert_eq!(val, Value::Integer(30));
    }

    #[test]
    fn test_eval_binary_comparison() {
        let cols = ["x"];
        let values = [Value::Integer(10)];
        let expr = Expr::BinaryOp {
            left: Box::new(Expr::ColumnRef {
                table: None,
                column: "x".into(),
            }),
            op: BinaryOp::Gt,
            right: Box::new(Expr::Literal(LiteralValue::Integer(5))),
        };
        let val = eval_expr(&expr, &cols, &values, 1, None).unwrap();
        assert_eq!(val, Value::Integer(1));
    }

    #[test]
    fn test_eval_and_or() {
        let cols: &[&str] = &[];
        let values: &[Value] = &[];

        // true AND false = false
        let expr = Expr::BinaryOp {
            left: Box::new(Expr::Literal(LiteralValue::Integer(1))),
            op: BinaryOp::And,
            right: Box::new(Expr::Literal(LiteralValue::Integer(0))),
        };
        let val = eval_expr(&expr, cols, values, 1, None).unwrap();
        assert_eq!(val, Value::Integer(0));

        // false OR true = true
        let expr = Expr::BinaryOp {
            left: Box::new(Expr::Literal(LiteralValue::Integer(0))),
            op: BinaryOp::Or,
            right: Box::new(Expr::Literal(LiteralValue::Integer(1))),
        };
        let val = eval_expr(&expr, cols, values, 1, None).unwrap();
        assert_eq!(val, Value::Integer(1));
    }

    #[test]
    fn test_eval_is_null() {
        let cols: &[&str] = &[];
        let values: &[Value] = &[];
        let expr = Expr::IsNull {
            operand: Box::new(Expr::Literal(LiteralValue::Null)),
            negated: false,
        };
        let val = eval_expr(&expr, cols, values, 1, None).unwrap();
        assert_eq!(val, Value::Integer(1));

        let expr = Expr::IsNull {
            operand: Box::new(Expr::Literal(LiteralValue::Integer(5))),
            negated: false,
        };
        let val = eval_expr(&expr, cols, values, 1, None).unwrap();
        assert_eq!(val, Value::Integer(0));
    }

    #[test]
    fn test_eval_between() {
        let cols: &[&str] = &[];
        let values: &[Value] = &[];
        let expr = Expr::Between {
            operand: Box::new(Expr::Literal(LiteralValue::Integer(5))),
            low: Box::new(Expr::Literal(LiteralValue::Integer(1))),
            high: Box::new(Expr::Literal(LiteralValue::Integer(10))),
            negated: false,
        };
        let val = eval_expr(&expr, cols, values, 1, None).unwrap();
        assert_eq!(val, Value::Integer(1));
    }

    #[test]
    fn test_eval_in() {
        let cols: &[&str] = &[];
        let values: &[Value] = &[];
        let expr = Expr::In {
            operand: Box::new(Expr::Literal(LiteralValue::Integer(3))),
            list: crate::ast::InList::Values(vec![
                Expr::Literal(LiteralValue::Integer(1)),
                Expr::Literal(LiteralValue::Integer(2)),
                Expr::Literal(LiteralValue::Integer(3)),
            ]),
            negated: false,
        };
        let val = eval_expr(&expr, cols, values, 1, None).unwrap();
        assert_eq!(val, Value::Integer(1));
    }

    #[test]
    fn test_eval_arithmetic() {
        let cols: &[&str] = &[];
        let values: &[Value] = &[];
        let expr = Expr::BinaryOp {
            left: Box::new(Expr::Literal(LiteralValue::Integer(10))),
            op: BinaryOp::Add,
            right: Box::new(Expr::Literal(LiteralValue::Integer(5))),
        };
        let val = eval_expr(&expr, cols, values, 1, None).unwrap();
        assert_eq!(val, Value::Integer(15));
    }

    #[test]
    fn test_eval_concat() {
        let cols: &[&str] = &[];
        let values: &[Value] = &[];
        let expr = Expr::BinaryOp {
            left: Box::new(Expr::Literal(LiteralValue::String("hello ".into()))),
            op: BinaryOp::Concat,
            right: Box::new(Expr::Literal(LiteralValue::String("world".into()))),
        };
        let val = eval_expr(&expr, cols, values, 1, None).unwrap();
        assert_eq!(val, Value::Text("hello world".into()));
    }

    #[test]
    fn test_eval_case() {
        let cols: &[&str] = &[];
        let values: &[Value] = &[];
        let expr = Expr::Case {
            operand: None,
            when_clauses: vec![
                (
                    Expr::Literal(LiteralValue::Integer(0)),
                    Expr::Literal(LiteralValue::String("no".into())),
                ),
                (
                    Expr::Literal(LiteralValue::Integer(1)),
                    Expr::Literal(LiteralValue::String("yes".into())),
                ),
            ],
            else_clause: Some(Box::new(Expr::Literal(LiteralValue::String(
                "maybe".into(),
            )))),
        };
        let val = eval_expr(&expr, cols, values, 1, None).unwrap();
        assert_eq!(val, Value::Text("yes".into()));
    }

    #[test]
    fn test_eval_typeof() {
        let cols: &[&str] = &[];
        let values: &[Value] = &[];
        let expr = Expr::FunctionCall {
            name: "typeof".into(),
            args: FunctionArgs::Exprs {
                distinct: false,
                args: vec![Expr::Literal(LiteralValue::Integer(42))],
            },
        };
        let val = eval_expr(&expr, cols, values, 1, None).unwrap();
        assert_eq!(val, Value::Text("integer".into()));
    }

    #[test]
    fn test_eval_coalesce() {
        let cols: &[&str] = &[];
        let values: &[Value] = &[];
        let expr = Expr::FunctionCall {
            name: "coalesce".into(),
            args: FunctionArgs::Exprs {
                distinct: false,
                args: vec![
                    Expr::Literal(LiteralValue::Null),
                    Expr::Literal(LiteralValue::Null),
                    Expr::Literal(LiteralValue::Integer(42)),
                ],
            },
        };
        let val = eval_expr(&expr, cols, values, 1, None).unwrap();
        assert_eq!(val, Value::Integer(42));
    }

    #[test]
    fn test_eval_length() {
        let cols: &[&str] = &[];
        let values: &[Value] = &[];
        let expr = Expr::FunctionCall {
            name: "length".into(),
            args: FunctionArgs::Exprs {
                distinct: false,
                args: vec![Expr::Literal(LiteralValue::String("hello".into()))],
            },
        };
        let val = eval_expr(&expr, cols, values, 1, None).unwrap();
        assert_eq!(val, Value::Integer(5));
    }

    #[test]
    fn test_eval_upper_lower() {
        let cols: &[&str] = &[];
        let values: &[Value] = &[];
        let expr = Expr::FunctionCall {
            name: "upper".into(),
            args: FunctionArgs::Exprs {
                distinct: false,
                args: vec![Expr::Literal(LiteralValue::String("hello".into()))],
            },
        };
        let val = eval_expr(&expr, cols, values, 1, None).unwrap();
        assert_eq!(val, Value::Text("HELLO".into()));
    }

    #[test]
    fn test_null_propagation() {
        let cols: &[&str] = &[];
        let values: &[Value] = &[];
        // NULL + 5 = NULL
        let expr = Expr::BinaryOp {
            left: Box::new(Expr::Literal(LiteralValue::Null)),
            op: BinaryOp::Add,
            right: Box::new(Expr::Literal(LiteralValue::Integer(5))),
        };
        let val = eval_expr(&expr, cols, values, 1, None).unwrap();
        assert_eq!(val, Value::Null);
    }

    #[test]
    fn test_division_by_zero() {
        let cols: &[&str] = &[];
        let values: &[Value] = &[];
        let expr = Expr::BinaryOp {
            left: Box::new(Expr::Literal(LiteralValue::Integer(10))),
            op: BinaryOp::Divide,
            right: Box::new(Expr::Literal(LiteralValue::Integer(0))),
        };
        let val = eval_expr(&expr, cols, values, 1, None).unwrap();
        assert_eq!(val, Value::Null);
    }

    #[test]
    fn test_rowid_access() {
        let cols = ["name"];
        let values = [Value::Text("Alice".into())];
        let expr = Expr::ColumnRef {
            table: None,
            column: "rowid".into(),
        };
        let val = eval_expr(&expr, &cols, &values, 42, None).unwrap();
        assert_eq!(val, Value::Integer(42));
    }
}
