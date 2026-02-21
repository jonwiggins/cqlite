use thiserror::Error;

/// Top-level error type for rsqlite.
#[derive(Debug, Error)]
pub enum RsqliteError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("parse error: {0}")]
    Parse(String),

    #[error("runtime error: {0}")]
    Runtime(String),

    #[error("constraint violation: {0}")]
    Constraint(String),

    #[error("database is corrupt: {0}")]
    Corrupt(String),

    #[error("not implemented: {0}")]
    NotImplemented(String),
}

pub type Result<T> = std::result::Result<T, RsqliteError>;
