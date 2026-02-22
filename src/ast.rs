// SQL Abstract Syntax Tree node definitions.
//
// These types represent the parsed structure of SQL statements.
// The parser produces these from a token stream, and the planner
// consumes them to produce query execution plans.

/// A complete SQL statement.
#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    Select(Box<SelectStatement>),
    Insert(InsertStatement),
    Update(UpdateStatement),
    Delete(DeleteStatement),
    CreateTable(CreateTableStatement),
    CreateIndex(CreateIndexStatement),
    DropTable(DropTableStatement),
    DropIndex(DropIndexStatement),
    AlterTable(AlterTableStatement),
    Begin(Option<TransactionType>),
    Commit,
    Rollback,
    Pragma(PragmaStatement),
    Explain(Box<Statement>),
    ExplainQueryPlan(Box<Statement>),
}

/// Transaction type for BEGIN.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransactionType {
    Deferred,
    Immediate,
    Exclusive,
}

// -- SELECT --

#[derive(Debug, Clone, PartialEq)]
pub struct SelectStatement {
    pub distinct: bool,
    pub columns: Vec<ResultColumn>,
    pub from: Option<FromClause>,
    pub where_clause: Option<Expr>,
    pub group_by: Option<Vec<Expr>>,
    pub having: Option<Expr>,
    pub order_by: Option<Vec<OrderByItem>>,
    pub limit: Option<LimitClause>,
    /// Compound operations (UNION, UNION ALL, INTERSECT, EXCEPT).
    pub compound: Vec<CompoundClause>,
    /// Common Table Expressions (WITH clause).
    pub ctes: Vec<Cte>,
}

/// A single Common Table Expression definition.
#[derive(Debug, Clone, PartialEq)]
pub struct Cte {
    pub name: String,
    pub columns: Option<Vec<String>>,
    pub query: Box<SelectStatement>,
    pub recursive: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompoundClause {
    pub op: CompoundOp,
    pub select: SelectStatement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompoundOp {
    Union,
    UnionAll,
    Intersect,
    Except,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ResultColumn {
    /// `*`
    AllColumns,
    /// `table.*`
    TableAllColumns(String),
    /// An expression, optionally aliased: `expr AS alias`
    Expr { expr: Expr, alias: Option<String> },
}

#[derive(Debug, Clone, PartialEq)]
pub struct FromClause {
    pub table: TableRef,
    pub joins: Vec<JoinClause>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TableRef {
    /// A simple table name, optionally aliased.
    Table { name: String, alias: Option<String> },
    /// A subquery in the FROM clause.
    Subquery {
        select: Box<SelectStatement>,
        alias: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct JoinClause {
    pub join_type: JoinType,
    pub table: TableRef,
    pub constraint: Option<JoinConstraint>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Cross,
    /// NATURAL is a modifier — the actual join type is stored here.
    NaturalInner,
    NaturalLeft,
    NaturalRight,
}

#[derive(Debug, Clone, PartialEq)]
pub enum JoinConstraint {
    On(Expr),
    Using(Vec<String>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct OrderByItem {
    pub expr: Expr,
    pub direction: SortDirection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortDirection {
    Asc,
    Desc,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LimitClause {
    pub limit: Expr,
    pub offset: Option<Expr>,
}

// -- INSERT --

#[derive(Debug, Clone, PartialEq)]
pub struct InsertStatement {
    pub table: String,
    pub columns: Option<Vec<String>>,
    pub source: InsertSource,
    pub or_conflict: Option<ConflictResolution>,
}

/// Conflict resolution strategy for INSERT OR ... and ON CONFLICT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictResolution {
    Abort,
    Fail,
    Ignore,
    Replace,
    Rollback,
}

#[derive(Debug, Clone, PartialEq)]
pub enum InsertSource {
    Values(Vec<Vec<Expr>>),
    Select(Box<SelectStatement>),
    DefaultValues,
}

// -- UPDATE --

#[derive(Debug, Clone, PartialEq)]
pub struct UpdateStatement {
    pub table: String,
    pub assignments: Vec<Assignment>,
    pub where_clause: Option<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Assignment {
    pub column: String,
    pub value: Expr,
}

// -- DELETE --

#[derive(Debug, Clone, PartialEq)]
pub struct DeleteStatement {
    pub table: String,
    pub where_clause: Option<Expr>,
}

// -- CREATE TABLE --

#[derive(Debug, Clone, PartialEq)]
pub struct CreateTableStatement {
    pub if_not_exists: bool,
    pub name: String,
    pub columns: Vec<ColumnDef>,
    pub constraints: Vec<TableConstraint>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ColumnDef {
    pub name: String,
    pub type_name: Option<String>,
    pub constraints: Vec<ColumnConstraint>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ColumnConstraint {
    PrimaryKey {
        direction: Option<SortDirection>,
        autoincrement: bool,
    },
    NotNull,
    Unique,
    Default(Expr),
    Check(Expr),
    References {
        table: String,
        columns: Vec<String>,
    },
    Collate(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum TableConstraint {
    PrimaryKey(Vec<IndexedColumn>),
    Unique(Vec<IndexedColumn>),
    Check(Expr),
    ForeignKey {
        columns: Vec<String>,
        ref_table: String,
        ref_columns: Vec<String>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct IndexedColumn {
    pub name: String,
    pub direction: Option<SortDirection>,
}

// -- CREATE INDEX --

#[derive(Debug, Clone, PartialEq)]
pub struct CreateIndexStatement {
    pub unique: bool,
    pub if_not_exists: bool,
    pub name: String,
    pub table: String,
    pub columns: Vec<IndexedColumn>,
    pub where_clause: Option<Expr>,
}

// -- DROP TABLE / DROP INDEX --

#[derive(Debug, Clone, PartialEq)]
pub struct DropTableStatement {
    pub if_exists: bool,
    pub name: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DropIndexStatement {
    pub if_exists: bool,
    pub name: String,
}

// -- ALTER TABLE --

#[derive(Debug, Clone, PartialEq)]
pub struct AlterTableStatement {
    pub table: String,
    pub action: AlterTableAction,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AlterTableAction {
    RenameTable(String),
    RenameColumn { old: String, new: String },
    AddColumn(ColumnDef),
    DropColumn(String),
}

// -- PRAGMA --

#[derive(Debug, Clone, PartialEq)]
pub struct PragmaStatement {
    pub name: String,
    pub value: Option<PragmaValue>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PragmaValue {
    Name(String),
    Number(i64),
    StringLiteral(String),
}

// -- Expressions --

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// A literal value.
    Literal(LiteralValue),
    /// A column reference, optionally qualified: `table.column`
    ColumnRef {
        table: Option<String>,
        column: String,
    },
    /// A bind parameter: `?` or `?NNN`
    BindParameter(Option<u32>),
    /// A unary operation: `-expr`, `NOT expr`, `~expr`
    UnaryOp { op: UnaryOp, operand: Box<Expr> },
    /// A binary operation: `a + b`, `a AND b`, etc.
    BinaryOp {
        left: Box<Expr>,
        op: BinaryOp,
        right: Box<Expr>,
    },
    /// `expr IS NULL` or `expr IS NOT NULL`
    IsNull { operand: Box<Expr>, negated: bool },
    /// `expr BETWEEN low AND high`
    Between {
        operand: Box<Expr>,
        low: Box<Expr>,
        high: Box<Expr>,
        negated: bool,
    },
    /// `expr IN (value_list)` or `expr IN (subquery)`
    In {
        operand: Box<Expr>,
        list: InList,
        negated: bool,
    },
    /// `expr LIKE pattern [ESCAPE escape_char]`
    Like {
        operand: Box<Expr>,
        pattern: Box<Expr>,
        escape: Option<Box<Expr>>,
        negated: bool,
    },
    /// `CASE [operand] WHEN ... THEN ... [ELSE ...] END`
    Case {
        operand: Option<Box<Expr>>,
        when_clauses: Vec<(Expr, Expr)>,
        else_clause: Option<Box<Expr>>,
    },
    /// `CAST(expr AS type_name)`
    Cast { expr: Box<Expr>, type_name: String },
    /// A function call: `func(args...)`
    FunctionCall { name: String, args: FunctionArgs },
    /// A parenthesized expression.
    Parenthesized(Box<Expr>),
    /// A subquery expression: `(SELECT ...)`
    Subquery(Box<SelectStatement>),
    /// `EXISTS (SELECT ...)`
    Exists {
        subquery: Box<SelectStatement>,
        negated: bool,
    },
    /// `expr COLLATE collation_name`
    Collate { expr: Box<Expr>, collation: String },
}

#[derive(Debug, Clone, PartialEq)]
pub enum LiteralValue {
    Integer(i64),
    Real(f64),
    String(String),
    Blob(Vec<u8>),
    Null,
    CurrentTime,
    CurrentDate,
    CurrentTimestamp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Negate,
    Not,
    BitwiseNot,
    Plus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    Eq,
    NotEq,
    Lt,
    Gt,
    Le,
    Ge,
    And,
    Or,
    Concat,
    BitAnd,
    BitOr,
    ShiftLeft,
    ShiftRight,
    Is,
    IsNot,
    Glob,
    Like,
}

#[derive(Debug, Clone, PartialEq)]
pub enum InList {
    Values(Vec<Expr>),
    Subquery(Box<SelectStatement>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum FunctionArgs {
    /// `func(*)` (e.g., COUNT(*))
    Wildcard,
    /// `func(expr, ...)` or `func(DISTINCT expr, ...)`
    Exprs { distinct: bool, args: Vec<Expr> },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_literal_values() {
        let lit = LiteralValue::Integer(42);
        assert_eq!(lit, LiteralValue::Integer(42));

        let lit = LiteralValue::String("hello".into());
        assert_eq!(lit, LiteralValue::String("hello".into()));
    }

    #[test]
    fn test_select_statement_construction() {
        let stmt = SelectStatement {
            distinct: false,
            columns: vec![ResultColumn::AllColumns],
            from: Some(FromClause {
                table: TableRef::Table {
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
        assert!(!stmt.distinct);
        assert_eq!(stmt.columns.len(), 1);
    }

    #[test]
    fn test_expr_construction() {
        let expr = Expr::BinaryOp {
            left: Box::new(Expr::ColumnRef {
                table: None,
                column: "age".into(),
            }),
            op: BinaryOp::Gt,
            right: Box::new(Expr::Literal(LiteralValue::Integer(18))),
        };
        match &expr {
            Expr::BinaryOp { op, .. } => assert_eq!(*op, BinaryOp::Gt),
            _ => panic!("expected BinaryOp"),
        }
    }
}
