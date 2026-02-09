//! AST node definitions for parsed SQL statements.

/// A complete SQL statement.
#[derive(Debug, Clone)]
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
    Begin,
    Commit,
    Rollback,
    Explain(Box<Statement>),
    ExplainQueryPlan(Box<Statement>),
    Pragma(PragmaStatement),
}

/// SELECT statement.
#[derive(Debug, Clone)]
pub struct SelectStatement {
    pub distinct: bool,
    pub columns: Vec<SelectColumn>,
    pub from: Option<FromClause>,
    pub where_clause: Option<Expr>,
    pub group_by: Vec<Expr>,
    pub having: Option<Expr>,
    pub order_by: Vec<OrderByItem>,
    pub limit: Option<Expr>,
    pub offset: Option<Expr>,
}

#[derive(Debug, Clone)]
pub enum SelectColumn {
    AllColumns,              // *
    TableAllColumns(String), // table.*
    Expr { expr: Expr, alias: Option<String> },
}

/// FROM clause.
#[derive(Debug, Clone)]
pub enum FromClause {
    Table {
        name: String,
        alias: Option<String>,
    },
    Join(Box<JoinClause>),
    Subquery {
        query: Box<SelectStatement>,
        alias: String,
    },
}

#[derive(Debug, Clone)]
pub struct JoinClause {
    pub left: FromClause,
    pub right: FromClause,
    pub join_type: JoinType,
    pub on: Option<Expr>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    Inner,
    Left,
    Cross,
}

/// ORDER BY item.
#[derive(Debug, Clone)]
pub struct OrderByItem {
    pub expr: Expr,
    pub descending: bool,
}

/// Expression AST node.
#[derive(Debug, Clone)]
pub enum Expr {
    // Literals
    Null,
    Integer(i64),
    Float(f64),
    String(String),
    Blob(Vec<u8>),

    // Column reference
    Column {
        table: Option<String>,
        name: String,
    },

    // Unary operators
    UnaryMinus(Box<Expr>),
    UnaryPlus(Box<Expr>),
    Not(Box<Expr>),
    BitwiseNot(Box<Expr>),

    // Binary operators
    BinaryOp {
        left: Box<Expr>,
        op: BinaryOperator,
        right: Box<Expr>,
    },

    // IS NULL / IS NOT NULL
    IsNull(Box<Expr>),
    IsNotNull(Box<Expr>),

    // BETWEEN
    Between {
        expr: Box<Expr>,
        low: Box<Expr>,
        high: Box<Expr>,
        negated: bool,
    },

    // IN
    InList {
        expr: Box<Expr>,
        list: Vec<Expr>,
        negated: bool,
    },
    InSelect {
        expr: Box<Expr>,
        query: Box<SelectStatement>,
        negated: bool,
    },

    // LIKE
    Like {
        expr: Box<Expr>,
        pattern: Box<Expr>,
        escape: Option<Box<Expr>>,
        negated: bool,
    },

    // GLOB
    GlobExpr {
        expr: Box<Expr>,
        pattern: Box<Expr>,
        negated: bool,
    },

    // Function call
    Function {
        name: String,
        args: Vec<Expr>,
        distinct: bool,
    },

    // CAST
    Cast {
        expr: Box<Expr>,
        type_name: String,
    },

    // CASE
    Case {
        operand: Option<Box<Expr>>,
        when_clauses: Vec<(Expr, Expr)>,
        else_clause: Option<Box<Expr>>,
    },

    // Subquery that returns a single value
    Subquery(Box<SelectStatement>),

    // EXISTS subquery
    Exists(Box<SelectStatement>),

    // Rowid reference
    Rowid,

    // Star (for COUNT(*))
    Star,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    And,
    Or,
    Concat,
    BitAnd,
    BitOr,
    ShiftLeft,
    ShiftRight,
}

/// INSERT statement.
#[derive(Debug, Clone)]
pub struct InsertStatement {
    pub table_name: String,
    pub columns: Option<Vec<String>>,
    pub values: Vec<Vec<Expr>>,
    pub or_replace: bool,
}

/// UPDATE statement.
#[derive(Debug, Clone)]
pub struct UpdateStatement {
    pub table_name: String,
    pub assignments: Vec<(String, Expr)>,
    pub where_clause: Option<Expr>,
}

/// DELETE statement.
#[derive(Debug, Clone)]
pub struct DeleteStatement {
    pub table_name: String,
    pub where_clause: Option<Expr>,
}

/// CREATE TABLE statement.
#[derive(Debug, Clone)]
pub struct CreateTableStatement {
    pub if_not_exists: bool,
    pub table_name: String,
    pub columns: Vec<ColumnDef>,
    pub constraints: Vec<TableConstraint>,
    pub without_rowid: bool,
}

/// Column definition.
#[derive(Debug, Clone)]
pub struct ColumnDef {
    pub name: String,
    pub type_name: Option<String>,
    pub constraints: Vec<ColumnConstraint>,
}

/// Column constraint.
#[derive(Debug, Clone)]
pub enum ColumnConstraint {
    PrimaryKey { autoincrement: bool },
    NotNull,
    Unique,
    Default(Expr),
    Check(Expr),
    References { table: String, columns: Vec<String> },
}

/// Table-level constraint.
#[derive(Debug, Clone)]
pub enum TableConstraint {
    PrimaryKey(Vec<String>),
    Unique(Vec<String>),
    Check(Expr),
    ForeignKey {
        columns: Vec<String>,
        ref_table: String,
        ref_columns: Vec<String>,
    },
}

/// CREATE INDEX statement.
#[derive(Debug, Clone)]
pub struct CreateIndexStatement {
    pub if_not_exists: bool,
    pub unique: bool,
    pub index_name: String,
    pub table_name: String,
    pub columns: Vec<IndexColumn>,
    pub where_clause: Option<Expr>,
}

#[derive(Debug, Clone)]
pub struct IndexColumn {
    pub name: String,
    pub descending: bool,
}

/// DROP TABLE statement.
#[derive(Debug, Clone)]
pub struct DropTableStatement {
    pub if_exists: bool,
    pub table_name: String,
}

/// DROP INDEX statement.
#[derive(Debug, Clone)]
pub struct DropIndexStatement {
    pub if_exists: bool,
    pub index_name: String,
}

/// ALTER TABLE statement.
#[derive(Debug, Clone)]
pub enum AlterTableStatement {
    AddColumn {
        table_name: String,
        column: ColumnDef,
    },
    RenameTable {
        table_name: String,
        new_name: String,
    },
    RenameColumn {
        table_name: String,
        old_name: String,
        new_name: String,
    },
}

/// PRAGMA statement.
#[derive(Debug, Clone)]
pub struct PragmaStatement {
    pub name: String,
    pub value: Option<PragmaValue>,
}

#[derive(Debug, Clone)]
pub enum PragmaValue {
    Name(String),
    Number(String),
    String(String),
    Call(String),
}
