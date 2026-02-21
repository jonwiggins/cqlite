// SQLite value types, type affinity, and coercion rules.
// See: https://www.sqlite.org/datatype3.html

use std::cmp::Ordering;
use std::fmt;

/// SQLite storage classes — the actual types values can have at runtime.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Integer(i64),
    Real(f64),
    Text(String),
    Blob(Vec<u8>),
}

impl Value {
    /// Returns the storage class name as SQLite's `typeof()` would.
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Null => "null",
            Value::Integer(_) => "integer",
            Value::Real(_) => "real",
            Value::Text(_) => "text",
            Value::Blob(_) => "blob",
        }
    }

    /// Returns true if this value is NULL.
    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    /// Interpret this value as a boolean (for WHERE clause evaluation).
    /// SQLite treats 0 and NULL as false, everything else as true.
    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Null => false,
            Value::Integer(0) => false,
            Value::Integer(_) => true,
            Value::Real(f) => *f != 0.0,
            Value::Text(s) => {
                // SQLite coerces text to numeric for boolean context.
                if let Ok(i) = s.parse::<i64>() {
                    i != 0
                } else if let Ok(f) = s.parse::<f64>() {
                    f != 0.0
                } else {
                    false
                }
            }
            Value::Blob(_) => false,
        }
    }

    /// Convert to i64 following SQLite coercion rules.
    pub fn to_integer(&self) -> Option<i64> {
        match self {
            Value::Null => None,
            Value::Integer(i) => Some(*i),
            Value::Real(f) => Some(*f as i64),
            Value::Text(s) => s
                .parse::<i64>()
                .ok()
                .or_else(|| s.parse::<f64>().ok().map(|f| f as i64)),
            Value::Blob(_) => Some(0),
        }
    }

    /// Convert to f64 following SQLite coercion rules.
    pub fn to_real(&self) -> Option<f64> {
        match self {
            Value::Null => None,
            Value::Integer(i) => Some(*i as f64),
            Value::Real(f) => Some(*f),
            Value::Text(s) => s.parse::<f64>().ok(),
            Value::Blob(_) => Some(0.0),
        }
    }

    /// Convert to text representation.
    pub fn to_text(&self) -> Option<String> {
        match self {
            Value::Null => None,
            Value::Integer(i) => Some(i.to_string()),
            Value::Real(f) => Some(format_real(*f)),
            Value::Text(s) => Some(s.clone()),
            Value::Blob(b) => Some(format_blob_as_hex(b)),
        }
    }
}

/// Format a float the way SQLite does.
fn format_real(f: f64) -> String {
    if f == f.trunc() && f.abs() < 1e15 {
        // SQLite displays integer-valued floats with ".0"
        format!("{f:.1}")
    } else {
        format!("{f}")
    }
}

/// Format a blob as a hex string (X'...' style without the quotes).
fn format_blob_as_hex(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2);
    for byte in b {
        s.push_str(&format!("{byte:02X}"));
    }
    s
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Null => write!(f, "NULL"),
            Value::Integer(i) => write!(f, "{i}"),
            Value::Real(r) => write!(f, "{}", format_real(*r)),
            Value::Text(s) => write!(f, "{s}"),
            Value::Blob(b) => write!(f, "X'{}'", format_blob_as_hex(b)),
        }
    }
}

/// Compare two Values using SQLite's comparison rules:
///   NULL < INTEGER/REAL < TEXT < BLOB
/// Within the same type, normal ordering applies.
/// Cross-type numeric comparisons (INTEGER vs REAL) compare numerically.
impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(sqlite_cmp(self, other))
    }
}

/// SQLite comparison ordering.
pub fn sqlite_cmp(a: &Value, b: &Value) -> Ordering {
    use Value::*;
    match (a, b) {
        // NULLs
        (Null, Null) => Ordering::Equal,
        (Null, _) => Ordering::Less,
        (_, Null) => Ordering::Greater,

        // Same-type comparisons
        (Integer(x), Integer(y)) => x.cmp(y),
        (Real(x), Real(y)) => x.partial_cmp(y).unwrap_or(Ordering::Equal),
        (Text(x), Text(y)) => x.cmp(y),
        (Blob(x), Blob(y)) => x.cmp(y),

        // Cross-type numeric: INTEGER vs REAL
        (Integer(x), Real(y)) => (*x as f64).partial_cmp(y).unwrap_or(Ordering::Equal),
        (Real(x), Integer(y)) => x.partial_cmp(&(*y as f64)).unwrap_or(Ordering::Equal),

        // Sort order: numeric < text < blob
        (Integer(_) | Real(_), Text(_) | Blob(_)) => Ordering::Less,
        (Text(_) | Blob(_), Integer(_) | Real(_)) => Ordering::Greater,
        (Text(_), Blob(_)) => Ordering::Less,
        (Blob(_), Text(_)) => Ordering::Greater,
    }
}

/// Column type affinity as determined by the declared column type.
/// See: https://www.sqlite.org/datatype3.html#type_affinity
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeAffinity {
    Integer,
    Text,
    Blob,
    Real,
    Numeric,
}

/// Determine type affinity from a column's declared type string.
/// Follows the SQLite rules:
/// 1. If the type contains "INT" -> INTEGER
/// 2. If the type contains "CHAR", "CLOB", or "TEXT" -> TEXT
/// 3. If the type contains "BLOB" or is empty -> BLOB
/// 4. If the type contains "REAL", "FLOA", or "DOUB" -> REAL
/// 5. Otherwise -> NUMERIC
pub fn determine_affinity(declared_type: &str) -> TypeAffinity {
    let upper = declared_type.to_uppercase();

    if upper.contains("INT") {
        TypeAffinity::Integer
    } else if upper.contains("CHAR") || upper.contains("CLOB") || upper.contains("TEXT") {
        TypeAffinity::Text
    } else if upper.contains("BLOB") || upper.is_empty() {
        TypeAffinity::Blob
    } else if upper.contains("REAL") || upper.contains("FLOA") || upper.contains("DOUB") {
        TypeAffinity::Real
    } else {
        TypeAffinity::Numeric
    }
}

/// Apply type affinity to a value (used when storing values into columns).
pub fn apply_affinity(value: Value, affinity: TypeAffinity) -> Value {
    match affinity {
        TypeAffinity::Integer | TypeAffinity::Numeric => match &value {
            Value::Text(s) => {
                if let Ok(i) = s.parse::<i64>() {
                    Value::Integer(i)
                } else if let Ok(f) = s.parse::<f64>() {
                    if affinity == TypeAffinity::Integer {
                        // INTEGER affinity: try to convert real to integer if lossless
                        if f == (f as i64) as f64 {
                            Value::Integer(f as i64)
                        } else {
                            Value::Real(f)
                        }
                    } else {
                        Value::Real(f)
                    }
                } else {
                    value
                }
            }
            Value::Real(f) if affinity == TypeAffinity::Integer => {
                if *f == (*f as i64) as f64 {
                    Value::Integer(*f as i64)
                } else {
                    value
                }
            }
            _ => value,
        },
        TypeAffinity::Text => match &value {
            Value::Integer(i) => Value::Text(i.to_string()),
            Value::Real(f) => Value::Text(format_real(*f)),
            _ => value,
        },
        TypeAffinity::Real => match &value {
            Value::Text(s) => {
                if let Ok(f) = s.parse::<f64>() {
                    Value::Real(f)
                } else {
                    value
                }
            }
            Value::Integer(i) => Value::Real(*i as f64),
            _ => value,
        },
        TypeAffinity::Blob => value, // BLOB affinity: no coercion
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_name() {
        assert_eq!(Value::Null.type_name(), "null");
        assert_eq!(Value::Integer(42).type_name(), "integer");
        assert_eq!(Value::Real(3.14).type_name(), "real");
        assert_eq!(Value::Text("hi".into()).type_name(), "text");
        assert_eq!(Value::Blob(vec![1, 2]).type_name(), "blob");
    }

    #[test]
    fn test_truthiness() {
        assert!(!Value::Null.is_truthy());
        assert!(!Value::Integer(0).is_truthy());
        assert!(Value::Integer(1).is_truthy());
        assert!(Value::Integer(-1).is_truthy());
        assert!(!Value::Real(0.0).is_truthy());
        assert!(Value::Real(0.1).is_truthy());
        assert!(!Value::Text("0".into()).is_truthy());
        assert!(Value::Text("1".into()).is_truthy());
        assert!(!Value::Text("abc".into()).is_truthy());
    }

    #[test]
    fn test_comparison_ordering() {
        // NULL < everything
        assert!(Value::Null < Value::Integer(0));
        assert!(Value::Null < Value::Text("".into()));

        // INTEGER < TEXT < BLOB
        assert!(Value::Integer(999) < Value::Text("a".into()));
        assert!(Value::Text("z".into()) < Value::Blob(vec![]));

        // Same type comparisons
        assert!(Value::Integer(1) < Value::Integer(2));
        assert!(Value::Text("a".into()) < Value::Text("b".into()));

        // Cross-type numeric
        assert_eq!(
            sqlite_cmp(&Value::Integer(1), &Value::Real(1.0)),
            Ordering::Equal
        );
        assert_eq!(
            sqlite_cmp(&Value::Integer(1), &Value::Real(1.5)),
            Ordering::Less
        );
    }

    #[test]
    fn test_determine_affinity() {
        assert_eq!(determine_affinity("INTEGER"), TypeAffinity::Integer);
        assert_eq!(determine_affinity("INT"), TypeAffinity::Integer);
        assert_eq!(determine_affinity("BIGINT"), TypeAffinity::Integer);
        assert_eq!(determine_affinity("TINYINT"), TypeAffinity::Integer);

        assert_eq!(determine_affinity("TEXT"), TypeAffinity::Text);
        assert_eq!(determine_affinity("VARCHAR(255)"), TypeAffinity::Text);
        assert_eq!(determine_affinity("CLOB"), TypeAffinity::Text);
        assert_eq!(determine_affinity("CHARACTER(20)"), TypeAffinity::Text);

        assert_eq!(determine_affinity("BLOB"), TypeAffinity::Blob);
        assert_eq!(determine_affinity(""), TypeAffinity::Blob);

        assert_eq!(determine_affinity("REAL"), TypeAffinity::Real);
        assert_eq!(determine_affinity("DOUBLE"), TypeAffinity::Real);
        assert_eq!(determine_affinity("FLOAT"), TypeAffinity::Real);

        assert_eq!(determine_affinity("NUMERIC"), TypeAffinity::Numeric);
        assert_eq!(determine_affinity("DECIMAL"), TypeAffinity::Numeric);
        assert_eq!(determine_affinity("BOOLEAN"), TypeAffinity::Numeric);
    }

    #[test]
    fn test_apply_affinity_integer() {
        assert_eq!(
            apply_affinity(Value::Text("42".into()), TypeAffinity::Integer),
            Value::Integer(42)
        );
        assert_eq!(
            apply_affinity(Value::Text("3.0".into()), TypeAffinity::Integer),
            Value::Integer(3)
        );
        assert_eq!(
            apply_affinity(Value::Text("3.5".into()), TypeAffinity::Integer),
            Value::Real(3.5)
        );
        assert_eq!(
            apply_affinity(Value::Text("abc".into()), TypeAffinity::Integer),
            Value::Text("abc".into())
        );
    }

    #[test]
    fn test_apply_affinity_text() {
        assert_eq!(
            apply_affinity(Value::Integer(42), TypeAffinity::Text),
            Value::Text("42".into())
        );
        assert_eq!(
            apply_affinity(Value::Real(3.0), TypeAffinity::Text),
            Value::Text("3.0".into())
        );
    }

    #[test]
    fn test_apply_affinity_blob() {
        // BLOB affinity never coerces.
        assert_eq!(
            apply_affinity(Value::Integer(42), TypeAffinity::Blob),
            Value::Integer(42)
        );
        assert_eq!(
            apply_affinity(Value::Text("hi".into()), TypeAffinity::Blob),
            Value::Text("hi".into())
        );
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", Value::Null), "NULL");
        assert_eq!(format!("{}", Value::Integer(42)), "42");
        assert_eq!(format!("{}", Value::Real(3.0)), "3.0");
        assert_eq!(format!("{}", Value::Text("hello".into())), "hello");
        assert_eq!(format!("{}", Value::Blob(vec![0xDE, 0xAD])), "X'DEAD'");
    }

    #[test]
    fn test_to_integer() {
        assert_eq!(Value::Null.to_integer(), None);
        assert_eq!(Value::Integer(42).to_integer(), Some(42));
        assert_eq!(Value::Real(3.7).to_integer(), Some(3));
        assert_eq!(Value::Text("100".into()).to_integer(), Some(100));
        assert_eq!(Value::Text("abc".into()).to_integer(), None);
    }

    #[test]
    fn test_to_real() {
        assert_eq!(Value::Null.to_real(), None);
        assert_eq!(Value::Integer(42).to_real(), Some(42.0));
        assert_eq!(Value::Real(3.14).to_real(), Some(3.14));
        assert_eq!(Value::Text("2.5".into()).to_real(), Some(2.5));
        assert_eq!(Value::Text("abc".into()).to_real(), None);
    }
}
