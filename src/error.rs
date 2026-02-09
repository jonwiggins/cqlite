use thiserror::Error;

#[derive(Error, Debug)]
pub enum RsqliteError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Runtime error: {0}")]
    Runtime(String),

    #[error("Constraint violation: {0}")]
    Constraint(String),

    #[error("Corruption: {0}")]
    Corruption(String),

    #[error("Not a database file or unsupported format")]
    NotADatabase,

    #[error("Table not found: {0}")]
    TableNotFound(String),

    #[error("Column not found: {0}")]
    ColumnNotFound(String),

    #[error("Index not found: {0}")]
    IndexNotFound(String),

    #[error("Table already exists: {0}")]
    TableExists(String),

    #[error("Index already exists: {0}")]
    IndexExists(String),

    #[error("Type mismatch: {0}")]
    TypeMismatch(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, RsqliteError>;
