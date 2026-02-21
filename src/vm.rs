// Virtual machine / query executor.
//
// Uses a Volcano (iterator) model: each operator implements open/next/close.
// The executor evaluates expressions, applies predicates, and produces rows.

use crate::ast::{
    BinaryOp, Expr, FunctionArgs, JoinConstraint, JoinType, LiteralValue, ResultColumn,
    SelectStatement, UnaryOp,
};
use crate::btree::BTreeCursor;
use crate::error::{Result, RsqliteError};
use crate::pager::Pager;
use crate::record;
use crate::types::Value;

/// A group is a key (group-by values) and its matching rows.
type GroupEntry<'a> = (Vec<Value>, Vec<&'a (i64, Vec<Value>)>);

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

/// Per-column constraint information extracted from CREATE TABLE.
#[derive(Debug, Clone, Default)]
pub struct ColumnConstraints {
    /// Column must not be NULL.
    pub not_null: bool,
    /// Default value expression (already evaluated to a constant).
    pub default_value: Option<Value>,
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
    /// Per-column constraints (parallel to `columns`).
    pub column_constraints: Vec<ColumnConstraints>,
}

/// Result of executing a query.
#[derive(Debug)]
pub struct QueryResult {
    pub columns: Vec<ColumnInfo>,
    pub rows: Vec<Row>,
}

/// Scan all rows from a table, handling INTEGER PRIMARY KEY substitution.
fn scan_table(pager: &mut Pager, schema: &TableSchema) -> Result<Vec<(i64, Vec<Value>)>> {
    let mut rows = Vec::new();
    let mut cursor = BTreeCursor::new(schema.root_page);
    cursor.move_to_first(pager)?;

    while cursor.is_valid() {
        let rowid = cursor.current_rowid(pager)?;
        let payload = cursor.current_payload(pager)?;
        let mut values = record::decode_record(&payload)?;

        if let Some(pk_idx) = schema.rowid_column {
            if pk_idx < values.len() {
                values[pk_idx] = Value::Integer(rowid);
            } else if values.len() < schema.columns.len() {
                values.insert(pk_idx, Value::Integer(rowid));
            }
        }

        rows.push((rowid, values));
        cursor.move_to_next(pager)?;
    }

    Ok(rows)
}

/// Execute a SELECT statement against the database.
pub fn execute_select(
    pager: &mut Pager,
    stmt: &SelectStatement,
    tables: &[TableSchema],
) -> Result<QueryResult> {
    // Determine the source table(s).
    let (table_schema, _table_alias) = match &stmt.from {
        Some(from) => {
            let (table_name, alias) = match &from.table {
                crate::ast::TableRef::Table { name, alias } => {
                    (name.as_str(), alias.as_deref().unwrap_or(name.as_str()))
                }
                crate::ast::TableRef::Subquery { .. } => {
                    return Err(RsqliteError::NotImplemented("subqueries in FROM".into()));
                }
            };
            let schema = tables
                .iter()
                .find(|t| t.name.eq_ignore_ascii_case(table_name))
                .ok_or_else(|| RsqliteError::Runtime(format!("no such table: {table_name}")))?;
            (Some(schema), alias.to_string())
        }
        None => (None, String::new()),
    };

    // Build the combined column names and rows, handling JOINs.
    let has_joins = stmt
        .from
        .as_ref()
        .map(|f| !f.joins.is_empty())
        .unwrap_or(false);

    let (column_names_owned, raw_rows, table_name) = if has_joins {
        let from = stmt.from.as_ref().unwrap();
        let left_schema = table_schema.unwrap();
        let left_rows = scan_table(pager, left_schema)?;

        // Determine the left table alias.
        let left_alias = match &from.table {
            crate::ast::TableRef::Table { name, alias } => {
                alias.as_deref().unwrap_or(name.as_str()).to_string()
            }
            _ => left_schema.name.clone(),
        };

        // Start with the left table's columns (qualified with alias) and rows.
        let mut combined_columns: Vec<String> = left_schema
            .columns
            .iter()
            .map(|c| format!("{}.{}", left_alias, c))
            .collect();
        let mut combined_rows: Vec<(i64, Vec<Value>)> = left_rows;

        for join in &from.joins {
            let (right_name, right_alias) = match &join.table {
                crate::ast::TableRef::Table { name, alias } => {
                    (name.as_str(), alias.as_deref().unwrap_or(name.as_str()))
                }
                _ => return Err(RsqliteError::NotImplemented("subquery in JOIN".into())),
            };
            let right_schema = tables
                .iter()
                .find(|t| t.name.eq_ignore_ascii_case(right_name))
                .ok_or_else(|| RsqliteError::Runtime(format!("no such table: {right_name}")))?;

            let right_rows = scan_table(pager, right_schema)?;

            // For USING clause, find the common column indices.
            let using_cols: Option<Vec<(usize, usize)>> = match &join.constraint {
                Some(JoinConstraint::Using(cols)) => {
                    let mut pairs = Vec::new();
                    for col_name in cols {
                        // Left columns are qualified; search by bare name.
                        let left_idx = combined_columns
                            .iter()
                            .position(|c| {
                                let bare = c.rfind('.').map(|p| &c[p + 1..]).unwrap_or(c);
                                bare.eq_ignore_ascii_case(col_name)
                            })
                            .ok_or_else(|| {
                                RsqliteError::Runtime(format!(
                                    "column {col_name} not found in left table"
                                ))
                            })?;
                        let right_idx = right_schema
                            .columns
                            .iter()
                            .position(|c| c.eq_ignore_ascii_case(col_name))
                            .ok_or_else(|| {
                                RsqliteError::Runtime(format!(
                                    "column {col_name} not found in table {right_name}"
                                ))
                            })?;
                        pairs.push((left_idx, right_idx));
                    }
                    Some(pairs)
                }
                _ => None,
            };

            // Build combined columns (for USING, skip duplicate columns from right).
            let right_col_start = combined_columns.len();
            let mut skip_right_cols: Vec<bool> = vec![false; right_schema.columns.len()];
            if let Some(ref pairs) = using_cols {
                for &(_, right_idx) in pairs {
                    skip_right_cols[right_idx] = true;
                }
            }
            for (i, col) in right_schema.columns.iter().enumerate() {
                if !skip_right_cols[i] {
                    combined_columns.push(format!("{}.{}", right_alias, col));
                }
            }

            // Build the all_columns list for expression evaluation during ON.
            let all_columns: Vec<&str> = combined_columns.iter().map(|c| c.as_str()).collect();

            // Nested loop join.
            let mut new_rows: Vec<(i64, Vec<Value>)> = Vec::new();

            for (left_rowid, left_values) in &combined_rows {
                let mut matched = false;

                for (_, right_values) in &right_rows {
                    // Build combined row.
                    let mut combined_values = left_values.clone();
                    for (i, val) in right_values.iter().enumerate() {
                        if !skip_right_cols[i] {
                            combined_values.push(val.clone());
                        }
                    }

                    // Check join condition.
                    let passes = match &join.constraint {
                        Some(JoinConstraint::On(expr)) => {
                            let result =
                                eval_expr(expr, &all_columns, &combined_values, *left_rowid, None)?;
                            result.is_truthy()
                        }
                        Some(JoinConstraint::Using(_)) => {
                            // Check that the USING columns match.
                            let pairs = using_cols.as_ref().unwrap();
                            pairs.iter().all(|&(left_idx, right_idx)| {
                                left_values.get(left_idx) == right_values.get(right_idx)
                            })
                        }
                        None => true, // CROSS JOIN
                    };

                    if passes {
                        matched = true;
                        new_rows.push((*left_rowid, combined_values));
                    }
                }

                // LEFT JOIN: if no match, include left row with NULLs for right columns.
                if !matched && join.join_type == JoinType::Left {
                    let mut combined_values = left_values.clone();
                    let extra_cols = combined_columns.len() - right_col_start;
                    for _ in 0..extra_cols {
                        combined_values.push(Value::Null);
                    }
                    new_rows.push((*left_rowid, combined_values));
                }
            }

            combined_rows = new_rows;
        }

        let col_names: Vec<String> = combined_columns;
        (col_names, combined_rows, None)
    } else {
        // No JOINs - single table scan.
        let mut raw_rows: Vec<(i64, Vec<Value>)> = Vec::new();
        if let Some(schema) = table_schema {
            raw_rows = scan_table(pager, schema)?;
        } else {
            // No FROM clause: produce a single synthetic row for expression evaluation.
            raw_rows.push((0, vec![]));
        }
        let col_names: Vec<String> = table_schema.map(|s| s.columns.clone()).unwrap_or_default();
        let tname = table_schema.map(|s| s.name.as_str());
        (col_names, raw_rows, tname)
    };

    let column_names: Vec<&str> = column_names_owned.iter().map(|c| c.as_str()).collect();

    // Pre-resolve subqueries in the WHERE clause and SELECT columns.
    let resolved_where = if let Some(ref where_expr) = stmt.where_clause {
        Some(resolve_subqueries(where_expr, pager, tables)?)
    } else {
        None
    };

    let resolved_columns: Vec<ResultColumn> = stmt
        .columns
        .iter()
        .map(|rc| match rc {
            ResultColumn::Expr { expr, alias } => {
                let resolved = resolve_subqueries(expr, pager, tables)?;
                Ok(ResultColumn::Expr {
                    expr: resolved,
                    alias: alias.clone(),
                })
            }
            other => Ok(other.clone()),
        })
        .collect::<Result<_>>()?;

    let resolved_order_by: Option<Vec<crate::ast::OrderByItem>> =
        if let Some(ref order_by) = stmt.order_by {
            Some(
                order_by
                    .iter()
                    .map(|item| {
                        Ok(crate::ast::OrderByItem {
                            expr: resolve_subqueries(&item.expr, pager, tables)?,
                            direction: item.direction,
                        })
                    })
                    .collect::<Result<_>>()?,
            )
        } else {
            None
        };

    let resolved_having = if let Some(ref having_expr) = stmt.having {
        Some(resolve_subqueries(having_expr, pager, tables)?)
    } else {
        None
    };

    // Apply WHERE filter.
    let mut filtered_rows: Vec<(i64, Vec<Value>)> = Vec::new();
    for (rowid, values) in &raw_rows {
        if let Some(ref where_expr) = resolved_where {
            let result = eval_expr(where_expr, &column_names, values, *rowid, table_name)?;
            if result.is_truthy() {
                filtered_rows.push((*rowid, values.clone()));
            }
        } else {
            filtered_rows.push((*rowid, values.clone()));
        }
    }

    // Handle GROUP BY and aggregates.
    let (result_columns, result_rows, filtered_rows) =
        if stmt.group_by.is_some() || has_aggregate(&resolved_columns) {
            let (cols, rows) = execute_group_by(
                &resolved_columns,
                &stmt.group_by,
                &resolved_having,
                &column_names,
                &filtered_rows,
                table_name,
            )?;
            // For ORDER BY after GROUP BY, we need the grouped "filtered_rows" equivalent.
            // Create synthetic filtered rows from the result for ORDER BY.
            let synth_rows: Vec<(i64, Vec<Value>)> = rows
                .iter()
                .enumerate()
                .map(|(i, r)| (i as i64, r.values.clone()))
                .collect();
            (cols, rows, synth_rows)
        } else {
            let (cols, rows) =
                project_columns(&resolved_columns, &column_names, &filtered_rows, table_name)?;
            (cols, rows, filtered_rows)
        };

    // Apply ORDER BY.
    let mut result_rows = result_rows;
    if let Some(ref order_by) = resolved_order_by {
        // For ORDER BY after GROUP BY, evaluate expressions against result columns.
        // For regular queries, evaluate against source columns.
        let order_col_names: Vec<&str> =
            if stmt.group_by.is_some() || has_aggregate(&resolved_columns) {
                result_columns.iter().map(|c| c.name.as_str()).collect()
            } else {
                column_names.clone()
            };

        let mut indexed: Vec<(usize, Vec<Value>)> = Vec::new();
        for (i, (rowid, values)) in filtered_rows.iter().enumerate() {
            let mut keys = Vec::new();
            for item in order_by {
                let key = eval_expr(&item.expr, &order_col_names, values, *rowid, table_name)?;
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

    // Apply DISTINCT.
    if stmt.distinct {
        let mut seen: Vec<Vec<Value>> = Vec::new();
        result_rows.retain(|row| {
            if seen.iter().any(|s| {
                s.len() == row.values.len()
                    && s.iter()
                        .zip(row.values.iter())
                        .all(|(a, b)| crate::types::sqlite_cmp(a, b) == std::cmp::Ordering::Equal)
            }) {
                false
            } else {
                seen.push(row.values.clone());
                true
            }
        });
    }

    // Apply LIMIT and OFFSET.
    if let Some(ref limit_clause) = stmt.limit {
        let limit_val = match eval_expr(&limit_clause.limit, &[], &[], 0, None)? {
            Value::Integer(n) => n as usize,
            Value::Null => usize::MAX, // NULL limit means no limit
            other => {
                return Err(RsqliteError::Runtime(format!(
                    "LIMIT must evaluate to an integer, got: {other}"
                )))
            }
        };
        let offset_val = match &limit_clause.offset {
            Some(expr) => match eval_expr(expr, &[], &[], 0, None)? {
                Value::Integer(n) => n as usize,
                Value::Null => 0,
                other => {
                    return Err(RsqliteError::Runtime(format!(
                        "OFFSET must evaluate to an integer, got: {other}"
                    )))
                }
            },
            None => 0,
        };
        let start = offset_val.min(result_rows.len());
        let end = (start + limit_val).min(result_rows.len());
        result_rows = result_rows[start..end].to_vec();
    }

    // Handle compound operations (UNION, UNION ALL, INTERSECT, EXCEPT).
    if !stmt.compound.is_empty() {
        for compound in &stmt.compound {
            let rhs_result = execute_select(pager, &compound.select, tables)?;
            match compound.op {
                crate::ast::CompoundOp::UnionAll => {
                    result_rows.extend(rhs_result.rows);
                }
                crate::ast::CompoundOp::Union => {
                    // UNION = UNION ALL + dedup.
                    for row in rhs_result.rows {
                        if !result_rows
                            .iter()
                            .any(|r| rows_equal(&r.values, &row.values))
                        {
                            result_rows.push(row);
                        }
                    }
                }
                crate::ast::CompoundOp::Intersect => {
                    result_rows.retain(|row| {
                        rhs_result
                            .rows
                            .iter()
                            .any(|r| rows_equal(&r.values, &row.values))
                    });
                }
                crate::ast::CompoundOp::Except => {
                    result_rows.retain(|row| {
                        !rhs_result
                            .rows
                            .iter()
                            .any(|r| rows_equal(&r.values, &row.values))
                    });
                }
            }
        }
    }

    Ok(QueryResult {
        columns: result_columns,
        rows: result_rows,
    })
}

/// Check if two rows have the same values (for UNION dedup).
fn rows_equal(a: &[Value], b: &[Value]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(va, vb)| match (va, vb) {
        (Value::Null, Value::Null) => true,
        (Value::Integer(a), Value::Integer(b)) => a == b,
        (Value::Real(a), Value::Real(b)) => a == b,
        (Value::Text(a), Value::Text(b)) => a == b,
        (Value::Blob(a), Value::Blob(b)) => a == b,
        _ => false,
    })
}

/// Check if any result column contains an aggregate function.
fn has_aggregate(columns: &[ResultColumn]) -> bool {
    columns.iter().any(|rc| match rc {
        ResultColumn::Expr { expr, .. } => expr_has_aggregate(expr),
        _ => false,
    })
}

/// Check if an expression contains an aggregate function call.
fn expr_has_aggregate(expr: &Expr) -> bool {
    match expr {
        Expr::FunctionCall { name, .. } => is_aggregate_fn(name),
        Expr::BinaryOp { left, right, .. } => expr_has_aggregate(left) || expr_has_aggregate(right),
        Expr::UnaryOp { operand, .. } => expr_has_aggregate(operand),
        Expr::Parenthesized(inner) => expr_has_aggregate(inner),
        Expr::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            operand.as_ref().is_some_and(|e| expr_has_aggregate(e))
                || when_clauses
                    .iter()
                    .any(|(w, t)| expr_has_aggregate(w) || expr_has_aggregate(t))
                || else_clause.as_ref().is_some_and(|e| expr_has_aggregate(e))
        }
        _ => false,
    }
}

/// Check if a function name is an aggregate function.
fn is_aggregate_fn(name: &str) -> bool {
    matches!(
        name.to_uppercase().as_str(),
        "COUNT" | "SUM" | "AVG" | "MIN" | "MAX" | "GROUP_CONCAT" | "TOTAL"
    )
}

/// Execute GROUP BY with aggregate functions.
fn execute_group_by(
    select_columns: &[ResultColumn],
    group_by: &Option<Vec<Expr>>,
    having: &Option<Expr>,
    column_names: &[&str],
    rows: &[(i64, Vec<Value>)],
    table_name: Option<&str>,
) -> Result<(Vec<ColumnInfo>, Vec<Row>)> {
    // Group rows by the GROUP BY key expressions.
    let groups: Vec<GroupEntry<'_>> = if let Some(ref group_exprs) = group_by {
        let mut group_map: Vec<GroupEntry<'_>> = Vec::new();

        for row in rows {
            let key: Vec<Value> = group_exprs
                .iter()
                .map(|e| eval_expr(e, column_names, &row.1, row.0, table_name))
                .collect::<Result<_>>()?;

            // Find existing group with same key.
            let found = group_map.iter_mut().find(|(k, _)| {
                k.len() == key.len()
                    && k.iter()
                        .zip(key.iter())
                        .all(|(a, b)| crate::types::sqlite_cmp(a, b) == std::cmp::Ordering::Equal)
            });

            if let Some((_, group_rows)) = found {
                group_rows.push(row);
            } else {
                group_map.push((key, vec![row]));
            }
        }

        group_map
    } else {
        // No GROUP BY but has aggregates — treat all rows as one group.
        if rows.is_empty() {
            vec![(vec![], vec![])]
        } else {
            vec![(vec![], rows.iter().collect())]
        }
    };

    // Build result columns info.
    let mut col_infos: Vec<ColumnInfo> = Vec::new();
    for rc in select_columns {
        match rc {
            ResultColumn::AllColumns => {
                for name in column_names {
                    col_infos.push(ColumnInfo {
                        name: name.to_string(),
                        table: table_name.map(|s| s.to_string()),
                    });
                }
            }
            ResultColumn::TableAllColumns(tname) => {
                for name in column_names {
                    col_infos.push(ColumnInfo {
                        name: name.to_string(),
                        table: Some(tname.clone()),
                    });
                }
            }
            ResultColumn::Expr { expr, alias } => {
                let name = alias.clone().unwrap_or_else(|| expr_display_name(expr));
                col_infos.push(ColumnInfo { name, table: None });
            }
        }
    }

    // Evaluate each group to produce result rows.
    let mut result_rows = Vec::new();

    for (_key, group_rows) in &groups {
        let mut row_values = Vec::new();

        for rc in select_columns {
            match rc {
                ResultColumn::AllColumns => {
                    // Use the first row's values for non-aggregate columns.
                    if let Some(first) = group_rows.first() {
                        for val in &first.1 {
                            row_values.push(val.clone());
                        }
                    } else {
                        for _ in column_names {
                            row_values.push(Value::Null);
                        }
                    }
                }
                ResultColumn::TableAllColumns(_) => {
                    if let Some(first) = group_rows.first() {
                        for val in &first.1 {
                            row_values.push(val.clone());
                        }
                    } else {
                        for _ in column_names {
                            row_values.push(Value::Null);
                        }
                    }
                }
                ResultColumn::Expr { expr, .. } => {
                    let val = eval_aggregate_expr(expr, column_names, group_rows, table_name)?;
                    row_values.push(val);
                }
            }
        }

        // Apply HAVING filter.
        if let Some(ref having_expr) = having {
            let having_val =
                eval_aggregate_expr(having_expr, column_names, group_rows, table_name)?;
            if !having_val.is_truthy() {
                continue;
            }
        }

        result_rows.push(Row { values: row_values });
    }

    Ok((col_infos, result_rows))
}

/// Evaluate an expression that may contain aggregate functions, given a group of rows.
fn eval_aggregate_expr(
    expr: &Expr,
    column_names: &[&str],
    group_rows: &[&(i64, Vec<Value>)],
    table_name: Option<&str>,
) -> Result<Value> {
    match expr {
        Expr::FunctionCall { name, args } if is_aggregate_fn(name) => {
            eval_aggregate_function(name, args, column_names, group_rows, table_name)
        }
        // For non-aggregate expressions, use the first row of the group.
        _ => {
            if let Some(first) = group_rows.first() {
                eval_expr(expr, column_names, &first.1, first.0, table_name)
            } else {
                Ok(Value::Null)
            }
        }
    }
}

/// Evaluate an aggregate function over a group of rows.
fn eval_aggregate_function(
    name: &str,
    args: &FunctionArgs,
    column_names: &[&str],
    group_rows: &[&(i64, Vec<Value>)],
    table_name: Option<&str>,
) -> Result<Value> {
    let upper = name.to_uppercase();

    match upper.as_str() {
        "COUNT" => {
            match args {
                FunctionArgs::Wildcard => {
                    // COUNT(*) counts all rows including NULLs.
                    Ok(Value::Integer(group_rows.len() as i64))
                }
                FunctionArgs::Exprs {
                    args: exprs,
                    distinct,
                } => {
                    if exprs.is_empty() {
                        return Ok(Value::Integer(group_rows.len() as i64));
                    }
                    let expr = &exprs[0];
                    if *distinct {
                        let mut seen: Vec<Value> = Vec::new();
                        for row in group_rows {
                            let val = eval_expr(expr, column_names, &row.1, row.0, table_name)?;
                            if !val.is_null() {
                                let already = seen.iter().any(|s| {
                                    crate::types::sqlite_cmp(s, &val) == std::cmp::Ordering::Equal
                                });
                                if !already {
                                    seen.push(val);
                                }
                            }
                        }
                        Ok(Value::Integer(seen.len() as i64))
                    } else {
                        let mut count = 0i64;
                        for row in group_rows {
                            let val = eval_expr(expr, column_names, &row.1, row.0, table_name)?;
                            if !val.is_null() {
                                count += 1;
                            }
                        }
                        Ok(Value::Integer(count))
                    }
                }
            }
        }
        "SUM" => {
            let expr = get_aggregate_arg(args)?;
            let mut int_sum: i64 = 0;
            let mut real_sum: f64 = 0.0;
            let mut has_real = false;
            let mut all_null = true;

            for row in group_rows {
                let val = eval_expr(expr, column_names, &row.1, row.0, table_name)?;
                match val {
                    Value::Integer(n) => {
                        int_sum += n;
                        all_null = false;
                    }
                    Value::Real(f) => {
                        real_sum += f;
                        has_real = true;
                        all_null = false;
                    }
                    Value::Null => {}
                    Value::Text(ref s) => {
                        if let Ok(n) = s.parse::<i64>() {
                            int_sum += n;
                            all_null = false;
                        } else if let Ok(f) = s.parse::<f64>() {
                            real_sum += f;
                            has_real = true;
                            all_null = false;
                        }
                    }
                    _ => {}
                }
            }

            if all_null {
                Ok(Value::Null)
            } else if has_real {
                Ok(Value::Real(real_sum + int_sum as f64))
            } else {
                Ok(Value::Integer(int_sum))
            }
        }
        "TOTAL" => {
            let expr = get_aggregate_arg(args)?;
            let mut total: f64 = 0.0;

            for row in group_rows {
                let val = eval_expr(expr, column_names, &row.1, row.0, table_name)?;
                match val {
                    Value::Integer(n) => total += n as f64,
                    Value::Real(f) => total += f,
                    _ => {}
                }
            }

            Ok(Value::Real(total))
        }
        "AVG" => {
            let expr = get_aggregate_arg(args)?;
            let mut sum: f64 = 0.0;
            let mut count: i64 = 0;

            for row in group_rows {
                let val = eval_expr(expr, column_names, &row.1, row.0, table_name)?;
                match val {
                    Value::Integer(n) => {
                        sum += n as f64;
                        count += 1;
                    }
                    Value::Real(f) => {
                        sum += f;
                        count += 1;
                    }
                    Value::Null => {}
                    Value::Text(ref s) => {
                        if let Ok(f) = s.parse::<f64>() {
                            sum += f;
                            count += 1;
                        }
                    }
                    _ => {}
                }
            }

            if count == 0 {
                Ok(Value::Null)
            } else {
                Ok(Value::Real(sum / count as f64))
            }
        }
        "MIN" => {
            let expr = get_aggregate_arg(args)?;
            let mut min: Option<Value> = None;

            for row in group_rows {
                let val = eval_expr(expr, column_names, &row.1, row.0, table_name)?;
                if val.is_null() {
                    continue;
                }
                min = Some(match min {
                    Some(ref m) => {
                        if crate::types::sqlite_cmp(&val, m) == std::cmp::Ordering::Less {
                            val
                        } else {
                            m.clone()
                        }
                    }
                    None => val,
                });
            }

            Ok(min.unwrap_or(Value::Null))
        }
        "MAX" => {
            let expr = get_aggregate_arg(args)?;
            let mut max: Option<Value> = None;

            for row in group_rows {
                let val = eval_expr(expr, column_names, &row.1, row.0, table_name)?;
                if val.is_null() {
                    continue;
                }
                max = Some(match max {
                    Some(ref m) => {
                        if crate::types::sqlite_cmp(&val, m) == std::cmp::Ordering::Greater {
                            val
                        } else {
                            m.clone()
                        }
                    }
                    None => val,
                });
            }

            Ok(max.unwrap_or(Value::Null))
        }
        "GROUP_CONCAT" => match args {
            FunctionArgs::Wildcard => Ok(Value::Null),
            FunctionArgs::Exprs { args: exprs, .. } => {
                if exprs.is_empty() {
                    return Ok(Value::Null);
                }
                let expr = &exprs[0];
                let separator = if exprs.len() > 1 {
                    if let Some(first) = group_rows.first() {
                        eval_expr(&exprs[1], column_names, &first.1, first.0, table_name)?
                            .to_text()
                            .unwrap_or_else(|| ",".to_string())
                    } else {
                        ",".to_string()
                    }
                } else {
                    ",".to_string()
                };

                let mut parts = Vec::new();
                for row in group_rows {
                    let val = eval_expr(expr, column_names, &row.1, row.0, table_name)?;
                    if !val.is_null() {
                        parts.push(val.to_text().unwrap_or_default());
                    }
                }

                if parts.is_empty() {
                    Ok(Value::Null)
                } else {
                    Ok(Value::Text(parts.join(&separator)))
                }
            }
        },
        _ => Err(RsqliteError::Runtime(format!(
            "unknown aggregate function: {name}"
        ))),
    }
}

/// Extract the single argument expression from aggregate function args.
fn get_aggregate_arg(args: &FunctionArgs) -> Result<&Expr> {
    match args {
        FunctionArgs::Wildcard => Err(RsqliteError::Runtime(
            "aggregate function does not accept *".into(),
        )),
        FunctionArgs::Exprs { args: exprs, .. } => {
            if exprs.len() != 1 {
                Err(RsqliteError::Runtime(
                    "aggregate function takes exactly 1 argument".into(),
                ))
            } else {
                Ok(&exprs[0])
            }
        }
    }
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
                    // In JOIN context, column names may be qualified like "table.col".
                    // Display just the bare column name.
                    let display_name = if let Some(dot_pos) = name.rfind('.') {
                        &name[dot_pos + 1..]
                    } else {
                        name
                    };
                    col_infos.push(ColumnInfo {
                        name: display_name.to_string(),
                        table: table_name.map(|s| s.to_string()),
                    });
                    projections.push(ColumnProjection::Index(i));
                }
            }
            ResultColumn::TableAllColumns(tname) => {
                if table_name.is_some() && table_name.unwrap().eq_ignore_ascii_case(tname) {
                    // Single-table context.
                    for (i, &name) in column_names.iter().enumerate() {
                        col_infos.push(ColumnInfo {
                            name: name.to_string(),
                            table: Some(tname.clone()),
                        });
                        projections.push(ColumnProjection::Index(i));
                    }
                } else {
                    // JOIN context: find columns with the matching table prefix.
                    let prefix = format!("{}.", tname);
                    let mut found = false;
                    for (i, &name) in column_names.iter().enumerate() {
                        if name
                            .to_ascii_lowercase()
                            .starts_with(&prefix.to_ascii_lowercase())
                        {
                            let bare = &name[prefix.len()..];
                            col_infos.push(ColumnInfo {
                                name: bare.to_string(),
                                table: Some(tname.clone()),
                            });
                            projections.push(ColumnProjection::Index(i));
                            found = true;
                        }
                    }
                    if !found {
                        return Err(RsqliteError::Runtime(format!("no such table: {tname}")));
                    }
                }
            }
            ResultColumn::Expr { expr, alias } => {
                let name = alias.clone().unwrap_or_else(|| expr_display_name(expr));
                col_infos.push(ColumnInfo { name, table: None });
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
                    let val = eval_expr(expr, column_names, values, *rowid, table_name)?;
                    row_values.push(val);
                }
            }
        }
        result_rows.push(Row { values: row_values });
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
    // Handle rowid aliases.
    if column.eq_ignore_ascii_case("rowid")
        || column.eq_ignore_ascii_case("_rowid_")
        || column.eq_ignore_ascii_case("oid")
    {
        return Err(RsqliteError::Runtime("__rowid__".into())); // Special marker
    }

    // Single-table context: check table qualifier matches.
    if let Some(rt) = row_table {
        if let Some(t) = table {
            if !t.eq_ignore_ascii_case(rt) {
                return Err(RsqliteError::Runtime(format!("no such table: {t}")));
            }
        }
        return column_names
            .iter()
            .position(|&n| n.eq_ignore_ascii_case(column))
            .ok_or_else(|| RsqliteError::Runtime(format!("no such column: {column}")));
    }

    // JOIN context (row_table is None): column names may be qualified like "table.col".
    if let Some(t) = table {
        // Try exact qualified match: "table.column".
        let qualified = format!("{}.{}", t, column);
        if let Some(pos) = column_names
            .iter()
            .position(|&n| n.eq_ignore_ascii_case(&qualified))
        {
            return Ok(pos);
        }
    }

    // Try unqualified match: search for column names that end with ".column" or are exactly "column".
    let mut found = None;
    for (i, &name) in column_names.iter().enumerate() {
        let bare = if let Some(dot_pos) = name.rfind('.') {
            &name[dot_pos + 1..]
        } else {
            name
        };
        if bare.eq_ignore_ascii_case(column) {
            if found.is_some() {
                return Err(RsqliteError::Runtime(format!(
                    "ambiguous column name: {column}"
                )));
            }
            found = Some(i);
        }
    }

    found.ok_or_else(|| RsqliteError::Runtime(format!("no such column: {column}")))
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
            | LiteralValue::CurrentTimestamp => Value::Text("(not implemented)".into()),
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
                crate::ast::InList::Subquery(_) => {
                    Err(RsqliteError::NotImplemented("IN subquery".into()))
                }
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

        Expr::Parenthesized(inner) => eval_expr(inner, column_names, values, rowid, table_name),

        Expr::Cast { expr, type_name } => {
            let val = eval_expr(expr, column_names, values, rowid, table_name)?;
            let upper_type = type_name.to_uppercase();
            // CAST performs hard conversion, unlike column affinity.
            if upper_type.contains("INT") {
                match &val {
                    Value::Null => Ok(Value::Null),
                    Value::Integer(_) => Ok(val),
                    Value::Real(f) => Ok(Value::Integer(*f as i64)),
                    Value::Text(s) => {
                        if let Ok(i) = s.parse::<i64>() {
                            Ok(Value::Integer(i))
                        } else if let Ok(f) = s.parse::<f64>() {
                            Ok(Value::Integer(f as i64))
                        } else {
                            Ok(Value::Integer(0))
                        }
                    }
                    Value::Blob(_) => Ok(Value::Integer(0)),
                }
            } else if upper_type.contains("REAL")
                || upper_type.contains("FLOA")
                || upper_type.contains("DOUB")
            {
                match &val {
                    Value::Null => Ok(Value::Null),
                    Value::Real(_) => Ok(val),
                    Value::Integer(n) => Ok(Value::Real(*n as f64)),
                    Value::Text(s) => {
                        if let Ok(f) = s.parse::<f64>() {
                            Ok(Value::Real(f))
                        } else {
                            Ok(Value::Real(0.0))
                        }
                    }
                    Value::Blob(_) => Ok(Value::Real(0.0)),
                }
            } else if upper_type.contains("TEXT")
                || upper_type.contains("CHAR")
                || upper_type.contains("CLOB")
                || upper_type.contains("VAR")
            {
                match &val {
                    Value::Null => Ok(Value::Null),
                    Value::Text(_) => Ok(val),
                    Value::Integer(n) => Ok(Value::Text(n.to_string())),
                    Value::Real(f) => Ok(Value::Text(format!("{f}"))),
                    Value::Blob(b) => Ok(Value::Text(String::from_utf8_lossy(b).to_string())),
                }
            } else if upper_type.contains("BLOB") {
                match &val {
                    Value::Null => Ok(Value::Null),
                    Value::Blob(_) => Ok(val),
                    Value::Text(s) => Ok(Value::Blob(s.as_bytes().to_vec())),
                    _ => Ok(val), // SQLite preserves other types as-is for BLOB cast
                }
            } else {
                // Unknown type: use affinity-based conversion.
                let affinity = crate::types::determine_affinity(type_name);
                Ok(crate::types::apply_affinity(val, affinity))
            }
        }

        Expr::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            if let Some(base) = operand {
                let base_val = eval_expr(base, column_names, values, rowid, table_name)?;
                for (when_expr, then_expr) in when_clauses {
                    let when_val = eval_expr(when_expr, column_names, values, rowid, table_name)?;
                    if crate::types::sqlite_cmp(&base_val, &when_val) == std::cmp::Ordering::Equal {
                        return eval_expr(then_expr, column_names, values, rowid, table_name);
                    }
                }
            } else {
                for (when_expr, then_expr) in when_clauses {
                    let when_val = eval_expr(when_expr, column_names, values, rowid, table_name)?;
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
    if (lv.is_null() || rv.is_null()) && !matches!(op, BinaryOp::Is | BinaryOp::IsNot) {
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
                    other.to_text().unwrap_or_default().len() as i64
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
                other => Ok(Value::Text(
                    other.to_text().unwrap_or_default().to_uppercase(),
                )),
            }
        }
        "LOWER" => {
            if arg_values.len() != 1 {
                return Err(RsqliteError::Runtime("lower() takes 1 argument".into()));
            }
            match &arg_values[0] {
                Value::Null => Ok(Value::Null),
                Value::Text(s) => Ok(Value::Text(s.to_lowercase())),
                other => Ok(Value::Text(
                    other.to_text().unwrap_or_default().to_lowercase(),
                )),
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
            if crate::types::sqlite_cmp(&arg_values[0], &arg_values[1]) == std::cmp::Ordering::Equal
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
        "SUBSTR" | "SUBSTRING" => {
            if arg_values.len() < 2 || arg_values.len() > 3 {
                return Err(RsqliteError::Runtime(
                    "substr() takes 2 or 3 arguments".into(),
                ));
            }
            if arg_values[0].is_null() {
                return Ok(Value::Null);
            }
            let s = arg_values[0].to_text().unwrap_or_default();
            let chars: Vec<char> = s.chars().collect();
            // SQLite substr is 1-based. Negative means from the end.
            let start = match &arg_values[1] {
                Value::Integer(n) => *n,
                _ => return Ok(Value::Null),
            };
            let len = if arg_values.len() == 3 {
                match &arg_values[2] {
                    Value::Integer(n) => Some(*n),
                    _ => return Ok(Value::Null),
                }
            } else {
                None
            };
            // Convert to 0-based index.
            let start_idx = if start > 0 {
                (start - 1) as usize
            } else if start < 0 {
                let from_end = (-start) as usize;
                if from_end > chars.len() {
                    0
                } else {
                    chars.len() - from_end
                }
            } else {
                // start == 0 is treated as start == 1 but the first char is skipped.
                0
            };
            let result: String = if let Some(l) = len {
                if l < 0 {
                    String::new()
                } else {
                    let actual_start = if start == 0 {
                        // When start=0, length is effectively reduced by 1.
                        let take = (l as usize).saturating_sub(1);
                        chars.iter().skip(start_idx).take(take).collect()
                    } else {
                        chars.iter().skip(start_idx).take(l as usize).collect()
                    };
                    actual_start
                }
            } else {
                chars[start_idx.min(chars.len())..].iter().collect()
            };
            Ok(Value::Text(result))
        }
        "TRIM" => {
            if arg_values.len() != 1 {
                return Err(RsqliteError::Runtime("trim() takes 1 argument".into()));
            }
            match &arg_values[0] {
                Value::Null => Ok(Value::Null),
                Value::Text(s) => Ok(Value::Text(s.trim().to_string())),
                other => Ok(Value::Text(
                    other.to_text().unwrap_or_default().trim().to_string(),
                )),
            }
        }
        "LTRIM" => {
            if arg_values.len() != 1 {
                return Err(RsqliteError::Runtime("ltrim() takes 1 argument".into()));
            }
            match &arg_values[0] {
                Value::Null => Ok(Value::Null),
                Value::Text(s) => Ok(Value::Text(s.trim_start().to_string())),
                other => Ok(Value::Text(
                    other.to_text().unwrap_or_default().trim_start().to_string(),
                )),
            }
        }
        "RTRIM" => {
            if arg_values.len() != 1 {
                return Err(RsqliteError::Runtime("rtrim() takes 1 argument".into()));
            }
            match &arg_values[0] {
                Value::Null => Ok(Value::Null),
                Value::Text(s) => Ok(Value::Text(s.trim_end().to_string())),
                other => Ok(Value::Text(
                    other.to_text().unwrap_or_default().trim_end().to_string(),
                )),
            }
        }
        "REPLACE" => {
            if arg_values.len() != 3 {
                return Err(RsqliteError::Runtime("replace() takes 3 arguments".into()));
            }
            if arg_values[0].is_null() {
                return Ok(Value::Null);
            }
            let s = arg_values[0].to_text().unwrap_or_default();
            let from = arg_values[1].to_text().unwrap_or_default();
            let to = arg_values[2].to_text().unwrap_or_default();
            Ok(Value::Text(s.replace(&from, &to)))
        }
        "HEX" => {
            if arg_values.len() != 1 {
                return Err(RsqliteError::Runtime("hex() takes 1 argument".into()));
            }
            match &arg_values[0] {
                Value::Null => Ok(Value::Text(String::new())),
                Value::Blob(b) => {
                    let hex: String = b.iter().map(|byte| format!("{byte:02X}")).collect();
                    Ok(Value::Text(hex))
                }
                other => {
                    let s = other.to_text().unwrap_or_default();
                    let hex: String = s.bytes().map(|byte| format!("{byte:02X}")).collect();
                    Ok(Value::Text(hex))
                }
            }
        }
        "QUOTE" => {
            if arg_values.len() != 1 {
                return Err(RsqliteError::Runtime("quote() takes 1 argument".into()));
            }
            match &arg_values[0] {
                Value::Null => Ok(Value::Text("NULL".to_string())),
                Value::Integer(n) => Ok(Value::Text(n.to_string())),
                Value::Real(f) => Ok(Value::Text(format!("{f}"))),
                Value::Text(s) => {
                    let escaped = s.replace('\'', "''");
                    Ok(Value::Text(format!("'{escaped}'")))
                }
                Value::Blob(b) => {
                    let hex: String = b.iter().map(|byte| format!("{byte:02X}")).collect();
                    Ok(Value::Text(format!("X'{hex}'")))
                }
            }
        }
        "RANDOM" => {
            // Return a random integer between i64::MIN and i64::MAX.
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
                .hash(&mut hasher);
            Ok(Value::Integer(hasher.finish() as i64))
        }
        "INSTR" => {
            if arg_values.len() != 2 {
                return Err(RsqliteError::Runtime("instr() takes 2 arguments".into()));
            }
            if arg_values[0].is_null() || arg_values[1].is_null() {
                return Ok(Value::Null);
            }
            let haystack = arg_values[0].to_text().unwrap_or_default();
            let needle = arg_values[1].to_text().unwrap_or_default();
            match haystack.find(&needle) {
                Some(pos) => {
                    // SQLite returns 1-based position (counting characters).
                    let char_pos = haystack[..pos].chars().count() + 1;
                    Ok(Value::Integer(char_pos as i64))
                }
                None => Ok(Value::Integer(0)),
            }
        }
        "UNICODE" => {
            if arg_values.len() != 1 {
                return Err(RsqliteError::Runtime("unicode() takes 1 argument".into()));
            }
            match &arg_values[0] {
                Value::Null => Ok(Value::Null),
                Value::Text(s) => {
                    if let Some(c) = s.chars().next() {
                        Ok(Value::Integer(c as i64))
                    } else {
                        Ok(Value::Null)
                    }
                }
                _ => Ok(Value::Null),
            }
        }
        "ZEROBLOB" => {
            if arg_values.len() != 1 {
                return Err(RsqliteError::Runtime("zeroblob() takes 1 argument".into()));
            }
            match &arg_values[0] {
                Value::Integer(n) => {
                    let size = (*n).max(0) as usize;
                    Ok(Value::Blob(vec![0u8; size]))
                }
                _ => Ok(Value::Blob(vec![])),
            }
        }
        "ROUND" => {
            if arg_values.is_empty() || arg_values.len() > 2 {
                return Err(RsqliteError::Runtime(
                    "round() takes 1 or 2 arguments".into(),
                ));
            }
            if arg_values[0].is_null() {
                return Ok(Value::Null);
            }
            let val = match &arg_values[0] {
                Value::Integer(n) => *n as f64,
                Value::Real(f) => *f,
                _ => return Ok(Value::Real(0.0)),
            };
            let decimals = if arg_values.len() == 2 {
                match &arg_values[1] {
                    Value::Integer(n) => *n as i32,
                    _ => 0,
                }
            } else {
                0
            };
            let factor = 10f64.powi(decimals);
            Ok(Value::Real((val * factor).round() / factor))
        }
        "CHAR" => {
            // char(X1,X2,...) returns a string from Unicode code points.
            let s: String = arg_values
                .iter()
                .filter_map(|v| match v {
                    Value::Integer(n) => char::from_u32(*n as u32),
                    _ => None,
                })
                .collect();
            Ok(Value::Text(s))
        }
        "PRINTF" | "FORMAT" => {
            // Simplified printf: just handle %d, %s, %f, %%.
            if arg_values.is_empty() {
                return Err(RsqliteError::Runtime(
                    "printf() requires at least 1 argument".into(),
                ));
            }
            let fmt = arg_values[0].to_text().unwrap_or_default();
            let mut result = String::new();
            let mut arg_idx = 1;
            let chars: Vec<char> = fmt.chars().collect();
            let mut i = 0;
            while i < chars.len() {
                if chars[i] == '%' && i + 1 < chars.len() {
                    match chars[i + 1] {
                        'd' | 'i' => {
                            if arg_idx < arg_values.len() {
                                match &arg_values[arg_idx] {
                                    Value::Integer(n) => result.push_str(&n.to_string()),
                                    Value::Real(f) => result.push_str(&(*f as i64).to_string()),
                                    _ => result.push('0'),
                                }
                                arg_idx += 1;
                            }
                            i += 2;
                        }
                        'f' => {
                            if arg_idx < arg_values.len() {
                                match &arg_values[arg_idx] {
                                    Value::Real(f) => result.push_str(&format!("{f:.6}")),
                                    Value::Integer(n) => {
                                        result.push_str(&format!("{:.6}", *n as f64))
                                    }
                                    _ => result.push_str("0.000000"),
                                }
                                arg_idx += 1;
                            }
                            i += 2;
                        }
                        's' => {
                            if arg_idx < arg_values.len() {
                                result.push_str(&arg_values[arg_idx].to_text().unwrap_or_default());
                                arg_idx += 1;
                            }
                            i += 2;
                        }
                        '%' => {
                            result.push('%');
                            i += 2;
                        }
                        _ => {
                            result.push('%');
                            i += 1;
                        }
                    }
                } else {
                    result.push(chars[i]);
                    i += 1;
                }
            }
            Ok(Value::Text(result))
        }
        "TOTAL_CHANGES" | "CHANGES" | "LAST_INSERT_ROWID" => {
            // Stub: return 0 for now.
            Ok(Value::Integer(0))
        }
        _ => Err(RsqliteError::Runtime(format!("no such function: {name}"))),
    }
}

/// Resolve subqueries in an expression tree by executing them and
/// replacing them with their results (literal values).
fn resolve_subqueries(expr: &Expr, pager: &mut Pager, tables: &[TableSchema]) -> Result<Expr> {
    match expr {
        // IN (SELECT ...) → IN (value1, value2, ...)
        Expr::In {
            operand,
            list: crate::ast::InList::Subquery(sub_select),
            negated,
        } => {
            let result = execute_select(pager, sub_select, tables)?;
            // Collect first column values from the subquery result.
            let value_exprs: Vec<Expr> = result
                .rows
                .iter()
                .map(|row| {
                    let val = row.values.first().cloned().unwrap_or(Value::Null);
                    value_to_literal_expr(&val)
                })
                .collect();
            Ok(Expr::In {
                operand: Box::new(resolve_subqueries(operand, pager, tables)?),
                list: crate::ast::InList::Values(value_exprs),
                negated: *negated,
            })
        }
        // EXISTS (SELECT ...) → 1 or 0
        Expr::Exists { subquery, negated } => {
            let result = execute_select(pager, subquery, tables)?;
            let exists = !result.rows.is_empty();
            let truth = if *negated { !exists } else { exists };
            Ok(Expr::Literal(LiteralValue::Integer(if truth {
                1
            } else {
                0
            })))
        }
        // Scalar subquery: (SELECT x) → literal value
        Expr::Subquery(sub_select) => {
            let result = execute_select(pager, sub_select, tables)?;
            let val = result
                .rows
                .first()
                .and_then(|r| r.values.first())
                .cloned()
                .unwrap_or(Value::Null);
            Ok(value_to_literal_expr(&val))
        }
        // Recursively resolve subqueries in compound expressions.
        Expr::BinaryOp { left, op, right } => Ok(Expr::BinaryOp {
            left: Box::new(resolve_subqueries(left, pager, tables)?),
            op: *op,
            right: Box::new(resolve_subqueries(right, pager, tables)?),
        }),
        Expr::UnaryOp { op, operand } => Ok(Expr::UnaryOp {
            op: *op,
            operand: Box::new(resolve_subqueries(operand, pager, tables)?),
        }),
        Expr::Parenthesized(inner) => Ok(Expr::Parenthesized(Box::new(resolve_subqueries(
            inner, pager, tables,
        )?))),
        Expr::IsNull { operand, negated } => Ok(Expr::IsNull {
            operand: Box::new(resolve_subqueries(operand, pager, tables)?),
            negated: *negated,
        }),
        Expr::Between {
            operand,
            low,
            high,
            negated,
        } => Ok(Expr::Between {
            operand: Box::new(resolve_subqueries(operand, pager, tables)?),
            low: Box::new(resolve_subqueries(low, pager, tables)?),
            high: Box::new(resolve_subqueries(high, pager, tables)?),
            negated: *negated,
        }),
        Expr::In {
            operand,
            list: crate::ast::InList::Values(exprs),
            negated,
        } => {
            let resolved_exprs: Vec<Expr> = exprs
                .iter()
                .map(|e| resolve_subqueries(e, pager, tables))
                .collect::<Result<_>>()?;
            Ok(Expr::In {
                operand: Box::new(resolve_subqueries(operand, pager, tables)?),
                list: crate::ast::InList::Values(resolved_exprs),
                negated: *negated,
            })
        }
        Expr::Like {
            operand,
            pattern,
            negated,
            escape,
        } => Ok(Expr::Like {
            operand: Box::new(resolve_subqueries(operand, pager, tables)?),
            pattern: Box::new(resolve_subqueries(pattern, pager, tables)?),
            negated: *negated,
            escape: match escape {
                Some(e) => Some(Box::new(resolve_subqueries(e, pager, tables)?)),
                None => None,
            },
        }),
        // Leaf expressions: no subqueries to resolve.
        _ => Ok(expr.clone()),
    }
}

/// Convert a Value to a literal Expr.
fn value_to_literal_expr(val: &Value) -> Expr {
    match val {
        Value::Null => Expr::Literal(LiteralValue::Null),
        Value::Integer(n) => Expr::Literal(LiteralValue::Integer(*n)),
        Value::Real(f) => Expr::Literal(LiteralValue::Real(*f)),
        Value::Text(s) => Expr::Literal(LiteralValue::String(s.clone())),
        Value::Blob(b) => Expr::Literal(LiteralValue::Blob(b.clone())),
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
            if si < s.len() && c.eq_ignore_ascii_case(&s[si]) {
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
        let val = eval_expr(&Expr::Literal(LiteralValue::Integer(42)), &[], &[], 0, None).unwrap();
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

    // -- Aggregate tests --

    fn make_rows(data: &[(i64, Vec<Value>)]) -> Vec<(i64, Vec<Value>)> {
        data.to_vec()
    }

    #[test]
    fn test_count_star() {
        let rows = make_rows(&[
            (1, vec![Value::Text("a".into()), Value::Integer(10)]),
            (2, vec![Value::Text("b".into()), Value::Integer(20)]),
            (3, vec![Value::Text("c".into()), Value::Integer(30)]),
        ]);
        let group_rows: Vec<&(i64, Vec<Value>)> = rows.iter().collect();
        let result = eval_aggregate_function(
            "COUNT",
            &FunctionArgs::Wildcard,
            &["name", "value"],
            &group_rows,
            None,
        )
        .unwrap();
        assert_eq!(result, Value::Integer(3));
    }

    #[test]
    fn test_count_column() {
        let rows = make_rows(&[
            (1, vec![Value::Integer(10)]),
            (2, vec![Value::Null]),
            (3, vec![Value::Integer(30)]),
        ]);
        let group_rows: Vec<&(i64, Vec<Value>)> = rows.iter().collect();
        let result = eval_aggregate_function(
            "COUNT",
            &FunctionArgs::Exprs {
                distinct: false,
                args: vec![Expr::ColumnRef {
                    table: None,
                    column: "x".into(),
                }],
            },
            &["x"],
            &group_rows,
            None,
        )
        .unwrap();
        // COUNT(x) should not count NULLs.
        assert_eq!(result, Value::Integer(2));
    }

    #[test]
    fn test_sum() {
        let rows = make_rows(&[
            (1, vec![Value::Integer(10)]),
            (2, vec![Value::Integer(20)]),
            (3, vec![Value::Integer(30)]),
        ]);
        let group_rows: Vec<&(i64, Vec<Value>)> = rows.iter().collect();
        let result = eval_aggregate_function(
            "SUM",
            &FunctionArgs::Exprs {
                distinct: false,
                args: vec![Expr::ColumnRef {
                    table: None,
                    column: "x".into(),
                }],
            },
            &["x"],
            &group_rows,
            None,
        )
        .unwrap();
        assert_eq!(result, Value::Integer(60));
    }

    #[test]
    fn test_avg() {
        let rows = make_rows(&[
            (1, vec![Value::Integer(10)]),
            (2, vec![Value::Integer(20)]),
            (3, vec![Value::Integer(30)]),
        ]);
        let group_rows: Vec<&(i64, Vec<Value>)> = rows.iter().collect();
        let result = eval_aggregate_function(
            "AVG",
            &FunctionArgs::Exprs {
                distinct: false,
                args: vec![Expr::ColumnRef {
                    table: None,
                    column: "x".into(),
                }],
            },
            &["x"],
            &group_rows,
            None,
        )
        .unwrap();
        assert_eq!(result, Value::Real(20.0));
    }

    #[test]
    fn test_min_max() {
        let rows = make_rows(&[
            (1, vec![Value::Integer(10)]),
            (2, vec![Value::Integer(5)]),
            (3, vec![Value::Integer(30)]),
        ]);
        let group_rows: Vec<&(i64, Vec<Value>)> = rows.iter().collect();

        let min = eval_aggregate_function(
            "MIN",
            &FunctionArgs::Exprs {
                distinct: false,
                args: vec![Expr::ColumnRef {
                    table: None,
                    column: "x".into(),
                }],
            },
            &["x"],
            &group_rows,
            None,
        )
        .unwrap();
        assert_eq!(min, Value::Integer(5));

        let max = eval_aggregate_function(
            "MAX",
            &FunctionArgs::Exprs {
                distinct: false,
                args: vec![Expr::ColumnRef {
                    table: None,
                    column: "x".into(),
                }],
            },
            &["x"],
            &group_rows,
            None,
        )
        .unwrap();
        assert_eq!(max, Value::Integer(30));
    }

    #[test]
    fn test_sum_with_nulls() {
        let rows = make_rows(&[
            (1, vec![Value::Integer(10)]),
            (2, vec![Value::Null]),
            (3, vec![Value::Integer(30)]),
        ]);
        let group_rows: Vec<&(i64, Vec<Value>)> = rows.iter().collect();
        let result = eval_aggregate_function(
            "SUM",
            &FunctionArgs::Exprs {
                distinct: false,
                args: vec![Expr::ColumnRef {
                    table: None,
                    column: "x".into(),
                }],
            },
            &["x"],
            &group_rows,
            None,
        )
        .unwrap();
        assert_eq!(result, Value::Integer(40));
    }

    #[test]
    fn test_sum_all_null() {
        let rows = make_rows(&[(1, vec![Value::Null]), (2, vec![Value::Null])]);
        let group_rows: Vec<&(i64, Vec<Value>)> = rows.iter().collect();
        let result = eval_aggregate_function(
            "SUM",
            &FunctionArgs::Exprs {
                distinct: false,
                args: vec![Expr::ColumnRef {
                    table: None,
                    column: "x".into(),
                }],
            },
            &["x"],
            &group_rows,
            None,
        )
        .unwrap();
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_count_distinct() {
        let rows = make_rows(&[
            (1, vec![Value::Text("a".into())]),
            (2, vec![Value::Text("b".into())]),
            (3, vec![Value::Text("a".into())]),
            (4, vec![Value::Null]),
        ]);
        let group_rows: Vec<&(i64, Vec<Value>)> = rows.iter().collect();
        let result = eval_aggregate_function(
            "COUNT",
            &FunctionArgs::Exprs {
                distinct: true,
                args: vec![Expr::ColumnRef {
                    table: None,
                    column: "x".into(),
                }],
            },
            &["x"],
            &group_rows,
            None,
        )
        .unwrap();
        // 'a' and 'b' are distinct non-null values.
        assert_eq!(result, Value::Integer(2));
    }

    #[test]
    fn test_group_concat() {
        let rows = make_rows(&[
            (1, vec![Value::Text("a".into())]),
            (2, vec![Value::Text("b".into())]),
            (3, vec![Value::Text("c".into())]),
        ]);
        let group_rows: Vec<&(i64, Vec<Value>)> = rows.iter().collect();
        let result = eval_aggregate_function(
            "GROUP_CONCAT",
            &FunctionArgs::Exprs {
                distinct: false,
                args: vec![Expr::ColumnRef {
                    table: None,
                    column: "x".into(),
                }],
            },
            &["x"],
            &group_rows,
            None,
        )
        .unwrap();
        assert_eq!(result, Value::Text("a,b,c".into()));
    }

    #[test]
    fn test_has_aggregate() {
        let cols = vec![ResultColumn::Expr {
            expr: Expr::FunctionCall {
                name: "COUNT".into(),
                args: FunctionArgs::Wildcard,
            },
            alias: None,
        }];
        assert!(has_aggregate(&cols));

        let cols = vec![ResultColumn::AllColumns];
        assert!(!has_aggregate(&cols));
    }
}
