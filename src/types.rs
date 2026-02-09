use std::cmp::Ordering;
use std::fmt;

/// SQLite storage classes / value types.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Integer(i64),
    Real(f64),
    Text(String),
    Blob(Vec<u8>),
}

impl Value {
    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    /// Returns the SQLite type name.
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Null => "null",
            Value::Integer(_) => "integer",
            Value::Real(_) => "real",
            Value::Text(_) => "text",
            Value::Blob(_) => "blob",
        }
    }

    /// Coerce to integer if possible.
    pub fn to_integer(&self) -> Option<i64> {
        match self {
            Value::Integer(i) => Some(*i),
            Value::Real(f) => Some(*f as i64),
            Value::Text(s) => s.parse::<i64>().ok(),
            _ => None,
        }
    }

    /// Coerce to real if possible.
    pub fn to_real(&self) -> Option<f64> {
        match self {
            Value::Integer(i) => Some(*i as f64),
            Value::Real(f) => Some(*f),
            Value::Text(s) => s.parse::<f64>().ok(),
            _ => None,
        }
    }

    /// Coerce to text.
    pub fn to_text(&self) -> String {
        match self {
            Value::Null => String::new(),
            Value::Integer(i) => i.to_string(),
            Value::Real(f) => format_real(*f),
            Value::Text(s) => s.clone(),
            Value::Blob(b) => format!("X'{}'", hex_encode(b)),
        }
    }

    /// Check truthiness (SQLite semantics: 0 and NULL are false).
    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Null => false,
            Value::Integer(i) => *i != 0,
            Value::Real(f) => *f != 0.0,
            Value::Text(s) => {
                // Try to parse as number
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

    /// SQLite comparison ordering.
    /// NULL < INTEGER/REAL < TEXT < BLOB
    pub fn sqlite_cmp(&self, other: &Value) -> Ordering {
        match (self, other) {
            (Value::Null, Value::Null) => Ordering::Equal,
            (Value::Null, _) => Ordering::Less,
            (_, Value::Null) => Ordering::Greater,

            (Value::Integer(a), Value::Integer(b)) => a.cmp(b),
            (Value::Integer(a), Value::Real(b)) => {
                (*a as f64).partial_cmp(b).unwrap_or(Ordering::Less)
            }
            (Value::Real(a), Value::Integer(b)) => {
                a.partial_cmp(&(*b as f64)).unwrap_or(Ordering::Less)
            }
            (Value::Real(a), Value::Real(b)) => a.partial_cmp(b).unwrap_or(Ordering::Less),

            (Value::Integer(_) | Value::Real(_), Value::Text(_) | Value::Blob(_)) => Ordering::Less,
            (Value::Text(_) | Value::Blob(_), Value::Integer(_) | Value::Real(_)) => {
                Ordering::Greater
            }

            (Value::Text(a), Value::Text(b)) => a.cmp(b),
            (Value::Text(_), Value::Blob(_)) => Ordering::Less,
            (Value::Blob(_), Value::Text(_)) => Ordering::Greater,
            (Value::Blob(a), Value::Blob(b)) => a.cmp(b),
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Null => write!(f, ""),
            Value::Integer(i) => write!(f, "{}", i),
            Value::Real(r) => write!(f, "{}", format_real(*r)),
            Value::Text(s) => write!(f, "{}", s),
            Value::Blob(b) => write!(f, "X'{}'", hex_encode(b)),
        }
    }
}

impl Eq for Value {}

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(std::cmp::Ord::cmp(self, other))
    }
}

impl Ord for Value {
    fn cmp(&self, other: &Self) -> Ordering {
        self.sqlite_cmp(other)
    }
}

/// Column type affinity (SQLite rules).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeAffinity {
    Text,
    Numeric,
    Integer,
    Real,
    Blob,
}

impl TypeAffinity {
    /// Determine type affinity from a column type declaration string.
    /// Follows SQLite's rules from section 3.1 of the datatype doc.
    pub fn from_type_name(type_name: &str) -> TypeAffinity {
        let upper = type_name.to_uppercase();
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

    /// Apply type affinity to a value.
    pub fn apply(&self, value: Value) -> Value {
        match self {
            TypeAffinity::Integer | TypeAffinity::Numeric => match &value {
                Value::Text(s) => {
                    if let Ok(i) = s.trim().parse::<i64>() {
                        Value::Integer(i)
                    } else if let Ok(f) = s.trim().parse::<f64>() {
                        if *self == TypeAffinity::Integer {
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
                _ => value,
            },
            TypeAffinity::Real => match &value {
                Value::Integer(i) => Value::Real(*i as f64),
                Value::Text(s) => {
                    if let Ok(f) = s.trim().parse::<f64>() {
                        Value::Real(f)
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
            TypeAffinity::Blob => value,
        }
    }
}

fn format_real(f: f64) -> String {
    if f == f.trunc() && f.abs() < 1e15 {
        format!("{:.1}", f)
    } else {
        format!("{}", f)
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02X}", b)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_affinity_rules() {
        assert_eq!(
            TypeAffinity::from_type_name("INTEGER"),
            TypeAffinity::Integer
        );
        assert_eq!(TypeAffinity::from_type_name("INT"), TypeAffinity::Integer);
        assert_eq!(
            TypeAffinity::from_type_name("TINYINT"),
            TypeAffinity::Integer
        );
        assert_eq!(
            TypeAffinity::from_type_name("BIGINT"),
            TypeAffinity::Integer
        );
        assert_eq!(TypeAffinity::from_type_name("TEXT"), TypeAffinity::Text);
        assert_eq!(
            TypeAffinity::from_type_name("VARCHAR(255)"),
            TypeAffinity::Text
        );
        assert_eq!(TypeAffinity::from_type_name("CLOB"), TypeAffinity::Text);
        assert_eq!(TypeAffinity::from_type_name("BLOB"), TypeAffinity::Blob);
        assert_eq!(TypeAffinity::from_type_name(""), TypeAffinity::Blob);
        assert_eq!(TypeAffinity::from_type_name("REAL"), TypeAffinity::Real);
        assert_eq!(TypeAffinity::from_type_name("FLOAT"), TypeAffinity::Real);
        assert_eq!(TypeAffinity::from_type_name("DOUBLE"), TypeAffinity::Real);
        assert_eq!(
            TypeAffinity::from_type_name("NUMERIC"),
            TypeAffinity::Numeric
        );
        assert_eq!(
            TypeAffinity::from_type_name("DECIMAL"),
            TypeAffinity::Numeric
        );
        assert_eq!(
            TypeAffinity::from_type_name("BOOLEAN"),
            TypeAffinity::Numeric
        );
    }

    #[test]
    fn test_value_comparison() {
        assert_eq!(Value::Null.sqlite_cmp(&Value::Integer(1)), Ordering::Less);
        assert_eq!(
            Value::Integer(1).sqlite_cmp(&Value::Integer(2)),
            Ordering::Less
        );
        assert_eq!(
            Value::Integer(1).sqlite_cmp(&Value::Text("a".into())),
            Ordering::Less
        );
        assert_eq!(
            Value::Text("a".into()).sqlite_cmp(&Value::Blob(vec![1])),
            Ordering::Less
        );
    }

    #[test]
    fn test_value_truthiness() {
        assert!(!Value::Null.is_truthy());
        assert!(!Value::Integer(0).is_truthy());
        assert!(Value::Integer(1).is_truthy());
        assert!(Value::Integer(-1).is_truthy());
        assert!(!Value::Real(0.0).is_truthy());
        assert!(Value::Real(1.0).is_truthy());
    }
}
