/// Query planner module.
///
/// Design decision: We use a Volcano-style tree-of-iterators model rather than
/// a bytecode VM. The Volcano model is simpler to implement, easier to debug,
/// and naturally composable. Each operator in the query plan implements a simple
/// interface: produce the next row. The query planner translates the AST into a
/// tree of operators that can be evaluated top-down.
///
/// The actual query execution is done in the `vm` module. This module provides
/// the plan representation and any optimizations (currently minimal).
use crate::ast::*;

/// A query plan node.
#[derive(Debug, Clone)]
pub enum PlanNode {
    /// Full table scan
    TableScan { table_name: String },
    /// Index scan
    IndexScan {
        index_name: String,
        table_name: String,
    },
    /// Filter (WHERE clause)
    Filter {
        source: Box<PlanNode>,
        predicate: Expr,
    },
    /// Projection (SELECT columns)
    Project {
        source: Box<PlanNode>,
        columns: Vec<SelectColumn>,
    },
    /// Sort (ORDER BY)
    Sort {
        source: Box<PlanNode>,
        order_by: Vec<OrderByItem>,
    },
    /// Limit
    Limit {
        source: Box<PlanNode>,
        count: Expr,
        offset: Option<Expr>,
    },
    /// Nested loop join
    NestedLoopJoin {
        left: Box<PlanNode>,
        right: Box<PlanNode>,
        join_type: JoinType,
        on: Option<Expr>,
    },
    /// Aggregate
    Aggregate {
        source: Box<PlanNode>,
        group_by: Vec<Expr>,
        aggregates: Vec<SelectColumn>,
    },
    /// Values (no FROM clause)
    Values { exprs: Vec<Vec<Expr>> },
}

impl std::fmt::Display for PlanNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.format(f, 0)
    }
}

impl PlanNode {
    fn format(&self, f: &mut std::fmt::Formatter<'_>, indent: usize) -> std::fmt::Result {
        let pad = " ".repeat(indent);
        match self {
            PlanNode::TableScan { table_name } => {
                writeln!(f, "{}SCAN TABLE {}", pad, table_name)
            }
            PlanNode::IndexScan {
                index_name,
                table_name,
            } => {
                writeln!(f, "{}SEARCH {} USING INDEX {}", pad, table_name, index_name)
            }
            PlanNode::Filter { source, .. } => {
                writeln!(f, "{}FILTER", pad)?;
                source.format(f, indent + 2)
            }
            PlanNode::Project { source, .. } => {
                writeln!(f, "{}PROJECT", pad)?;
                source.format(f, indent + 2)
            }
            PlanNode::Sort { source, .. } => {
                writeln!(f, "{}SORT", pad)?;
                source.format(f, indent + 2)
            }
            PlanNode::Limit { source, .. } => {
                writeln!(f, "{}LIMIT", pad)?;
                source.format(f, indent + 2)
            }
            PlanNode::NestedLoopJoin {
                left,
                right,
                join_type,
                ..
            } => {
                writeln!(f, "{}NESTED LOOP {:?} JOIN", pad, join_type)?;
                left.format(f, indent + 2)?;
                right.format(f, indent + 2)
            }
            PlanNode::Aggregate { source, .. } => {
                writeln!(f, "{}AGGREGATE", pad)?;
                source.format(f, indent + 2)
            }
            PlanNode::Values { .. } => {
                writeln!(f, "{}VALUES", pad)
            }
        }
    }
}

/// Create a basic query plan from a SELECT statement.
/// Currently just does straightforward translation without optimization.
pub fn plan_select(select: &SelectStatement) -> PlanNode {
    let mut plan = if let Some(ref from) = select.from {
        plan_from(from)
    } else {
        PlanNode::Values {
            exprs: vec![select
                .columns
                .iter()
                .map(|c| {
                    if let SelectColumn::Expr { expr, .. } = c {
                        expr.clone()
                    } else {
                        Expr::Null
                    }
                })
                .collect()],
        }
    };

    // WHERE
    if let Some(ref where_clause) = select.where_clause {
        plan = PlanNode::Filter {
            source: Box::new(plan),
            predicate: where_clause.clone(),
        };
    }

    // GROUP BY / Aggregates
    if !select.group_by.is_empty() || has_aggregate_columns(&select.columns) {
        plan = PlanNode::Aggregate {
            source: Box::new(plan),
            group_by: select.group_by.clone(),
            aggregates: select.columns.clone(),
        };
    }

    // PROJECT
    plan = PlanNode::Project {
        source: Box::new(plan),
        columns: select.columns.clone(),
    };

    // ORDER BY
    if !select.order_by.is_empty() {
        plan = PlanNode::Sort {
            source: Box::new(plan),
            order_by: select.order_by.clone(),
        };
    }

    // LIMIT/OFFSET
    if let Some(ref limit) = select.limit {
        plan = PlanNode::Limit {
            source: Box::new(plan),
            count: limit.clone(),
            offset: select.offset.clone(),
        };
    }

    plan
}

fn plan_from(from: &FromClause) -> PlanNode {
    match from {
        FromClause::Table { name, .. } => PlanNode::TableScan {
            table_name: name.clone(),
        },
        FromClause::Join(join) => {
            let left = plan_from(&join.left);
            let right = plan_from(&join.right);
            PlanNode::NestedLoopJoin {
                left: Box::new(left),
                right: Box::new(right),
                join_type: join.join_type,
                on: join.on.clone(),
            }
        }
        FromClause::Subquery { .. } => PlanNode::TableScan {
            table_name: "<subquery>".into(),
        },
    }
}

fn has_aggregate_columns(columns: &[SelectColumn]) -> bool {
    for col in columns {
        if let SelectColumn::Expr { expr, .. } = col {
            if is_aggregate(expr) {
                return true;
            }
        }
    }
    false
}

fn is_aggregate(expr: &Expr) -> bool {
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
