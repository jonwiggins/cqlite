// Built-in scalar functions.
//
// Each function takes a slice of already-evaluated argument values and
// returns a single Value. The VM calls into this module from `eval_function`.

use crate::error::{Result, RsqliteError};
use crate::types::Value;

/// Evaluate a built-in scalar function by name.
/// `args` are the already-evaluated argument values.
pub fn call_scalar(name: &str, args: &[Value]) -> Result<Value> {
    let upper = name.to_uppercase();

    match upper.as_str() {
        "LENGTH" => {
            check_args(name, args, 1)?;
            match &args[0] {
                Value::Null => Ok(Value::Null),
                Value::Text(s) => Ok(Value::Integer(s.len() as i64)),
                Value::Blob(b) => Ok(Value::Integer(b.len() as i64)),
                other => Ok(Value::Integer(
                    other.to_text().unwrap_or_default().len() as i64
                )),
            }
        }
        "UPPER" => {
            check_args(name, args, 1)?;
            match &args[0] {
                Value::Null => Ok(Value::Null),
                Value::Text(s) => Ok(Value::Text(s.to_uppercase())),
                other => Ok(Value::Text(
                    other.to_text().unwrap_or_default().to_uppercase(),
                )),
            }
        }
        "LOWER" => {
            check_args(name, args, 1)?;
            match &args[0] {
                Value::Null => Ok(Value::Null),
                Value::Text(s) => Ok(Value::Text(s.to_lowercase())),
                other => Ok(Value::Text(
                    other.to_text().unwrap_or_default().to_lowercase(),
                )),
            }
        }
        "ABS" => {
            check_args(name, args, 1)?;
            match &args[0] {
                Value::Null => Ok(Value::Null),
                Value::Integer(n) => Ok(Value::Integer(n.abs())),
                Value::Real(f) => Ok(Value::Real(f.abs())),
                _ => Ok(Value::Integer(0)),
            }
        }
        "TYPEOF" => {
            check_args(name, args, 1)?;
            Ok(Value::Text(args[0].type_name().to_string()))
        }
        "COALESCE" => {
            for v in args {
                if !v.is_null() {
                    return Ok(v.clone());
                }
            }
            Ok(Value::Null)
        }
        "IFNULL" => {
            check_args(name, args, 2)?;
            if args[0].is_null() {
                Ok(args[1].clone())
            } else {
                Ok(args[0].clone())
            }
        }
        "NULLIF" => {
            check_args(name, args, 2)?;
            if crate::types::sqlite_cmp(&args[0], &args[1]) == std::cmp::Ordering::Equal {
                Ok(Value::Null)
            } else {
                Ok(args[0].clone())
            }
        }
        "MAX" => {
            if args.is_empty() {
                return Ok(Value::Null);
            }
            let mut max = &args[0];
            for v in &args[1..] {
                if crate::types::sqlite_cmp(v, max) == std::cmp::Ordering::Greater {
                    max = v;
                }
            }
            Ok(max.clone())
        }
        "MIN" => {
            if args.is_empty() {
                return Ok(Value::Null);
            }
            let mut min = &args[0];
            for v in &args[1..] {
                if crate::types::sqlite_cmp(v, min) == std::cmp::Ordering::Less {
                    min = v;
                }
            }
            Ok(min.clone())
        }
        "SUBSTR" | "SUBSTRING" => {
            if args.len() < 2 || args.len() > 3 {
                return Err(RsqliteError::Runtime(
                    "substr() takes 2 or 3 arguments".into(),
                ));
            }
            if args[0].is_null() {
                return Ok(Value::Null);
            }
            let s = args[0].to_text().unwrap_or_default();
            let chars: Vec<char> = s.chars().collect();
            let start = match &args[1] {
                Value::Integer(n) => *n,
                _ => return Ok(Value::Null),
            };
            let len = if args.len() == 3 {
                match &args[2] {
                    Value::Integer(n) => Some(*n),
                    _ => return Ok(Value::Null),
                }
            } else {
                None
            };
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
                0
            };
            let result: String = if let Some(l) = len {
                if l < 0 {
                    String::new()
                } else if start == 0 {
                    let take = (l as usize).saturating_sub(1);
                    chars.iter().skip(start_idx).take(take).collect()
                } else {
                    chars.iter().skip(start_idx).take(l as usize).collect()
                }
            } else {
                chars[start_idx.min(chars.len())..].iter().collect()
            };
            Ok(Value::Text(result))
        }
        "TRIM" => {
            check_args(name, args, 1)?;
            match &args[0] {
                Value::Null => Ok(Value::Null),
                Value::Text(s) => Ok(Value::Text(s.trim().to_string())),
                other => Ok(Value::Text(
                    other.to_text().unwrap_or_default().trim().to_string(),
                )),
            }
        }
        "LTRIM" => {
            check_args(name, args, 1)?;
            match &args[0] {
                Value::Null => Ok(Value::Null),
                Value::Text(s) => Ok(Value::Text(s.trim_start().to_string())),
                other => Ok(Value::Text(
                    other.to_text().unwrap_or_default().trim_start().to_string(),
                )),
            }
        }
        "RTRIM" => {
            check_args(name, args, 1)?;
            match &args[0] {
                Value::Null => Ok(Value::Null),
                Value::Text(s) => Ok(Value::Text(s.trim_end().to_string())),
                other => Ok(Value::Text(
                    other.to_text().unwrap_or_default().trim_end().to_string(),
                )),
            }
        }
        "REPLACE" => {
            check_args(name, args, 3)?;
            if args[0].is_null() {
                return Ok(Value::Null);
            }
            let s = args[0].to_text().unwrap_or_default();
            let from = args[1].to_text().unwrap_or_default();
            let to = args[2].to_text().unwrap_or_default();
            Ok(Value::Text(s.replace(&from, &to)))
        }
        "HEX" => {
            check_args(name, args, 1)?;
            match &args[0] {
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
            check_args(name, args, 1)?;
            match &args[0] {
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
            check_args(name, args, 2)?;
            if args[0].is_null() || args[1].is_null() {
                return Ok(Value::Null);
            }
            let haystack = args[0].to_text().unwrap_or_default();
            let needle = args[1].to_text().unwrap_or_default();
            match haystack.find(&needle) {
                Some(pos) => {
                    let char_pos = haystack[..pos].chars().count() + 1;
                    Ok(Value::Integer(char_pos as i64))
                }
                None => Ok(Value::Integer(0)),
            }
        }
        "UNICODE" => {
            check_args(name, args, 1)?;
            match &args[0] {
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
            check_args(name, args, 1)?;
            match &args[0] {
                Value::Integer(n) => {
                    let size = (*n).max(0) as usize;
                    Ok(Value::Blob(vec![0u8; size]))
                }
                _ => Ok(Value::Blob(vec![])),
            }
        }
        "ROUND" => {
            if args.is_empty() || args.len() > 2 {
                return Err(RsqliteError::Runtime(
                    "round() takes 1 or 2 arguments".into(),
                ));
            }
            if args[0].is_null() {
                return Ok(Value::Null);
            }
            let val = match &args[0] {
                Value::Integer(n) => *n as f64,
                Value::Real(f) => *f,
                _ => return Ok(Value::Real(0.0)),
            };
            let decimals = if args.len() == 2 {
                match &args[1] {
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
            let s: String = args
                .iter()
                .filter_map(|v| match v {
                    Value::Integer(n) => char::from_u32(*n as u32),
                    _ => None,
                })
                .collect();
            Ok(Value::Text(s))
        }
        "PRINTF" | "FORMAT" => {
            if args.is_empty() {
                return Err(RsqliteError::Runtime(
                    "printf() requires at least 1 argument".into(),
                ));
            }
            let fmt = args[0].to_text().unwrap_or_default();
            let mut result = String::new();
            let mut arg_idx = 1;
            let chars: Vec<char> = fmt.chars().collect();
            let mut i = 0;
            while i < chars.len() {
                if chars[i] == '%' && i + 1 < chars.len() {
                    match chars[i + 1] {
                        'd' | 'i' => {
                            if arg_idx < args.len() {
                                match &args[arg_idx] {
                                    Value::Integer(n) => result.push_str(&n.to_string()),
                                    Value::Real(f) => result.push_str(&(*f as i64).to_string()),
                                    _ => result.push('0'),
                                }
                                arg_idx += 1;
                            }
                            i += 2;
                        }
                        'f' => {
                            if arg_idx < args.len() {
                                match &args[arg_idx] {
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
                            if arg_idx < args.len() {
                                result.push_str(&args[arg_idx].to_text().unwrap_or_default());
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
        "TOTAL_CHANGES" | "CHANGES" | "LAST_INSERT_ROWID" => Ok(Value::Integer(0)),
        _ => Err(RsqliteError::Runtime(format!("no such function: {name}"))),
    }
}

fn check_args(name: &str, args: &[Value], expected: usize) -> Result<()> {
    if args.len() != expected {
        return Err(RsqliteError::Runtime(format!(
            "{name}() takes {expected} argument(s), got {}",
            args.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::approx_constant)]
mod tests {
    use super::*;

    #[test]
    fn test_length() {
        assert_eq!(
            call_scalar("length", &[Value::Text("hello".into())]).unwrap(),
            Value::Integer(5)
        );
        assert_eq!(call_scalar("length", &[Value::Null]).unwrap(), Value::Null);
    }

    #[test]
    fn test_upper_lower() {
        assert_eq!(
            call_scalar("upper", &[Value::Text("hello".into())]).unwrap(),
            Value::Text("HELLO".into())
        );
        assert_eq!(
            call_scalar("lower", &[Value::Text("HELLO".into())]).unwrap(),
            Value::Text("hello".into())
        );
    }

    #[test]
    fn test_abs() {
        assert_eq!(
            call_scalar("abs", &[Value::Integer(-42)]).unwrap(),
            Value::Integer(42)
        );
        assert_eq!(
            call_scalar("abs", &[Value::Real(-3.14)]).unwrap(),
            Value::Real(3.14)
        );
    }

    #[test]
    fn test_coalesce() {
        assert_eq!(
            call_scalar("coalesce", &[Value::Null, Value::Integer(1)]).unwrap(),
            Value::Integer(1)
        );
        assert_eq!(
            call_scalar("coalesce", &[Value::Null, Value::Null]).unwrap(),
            Value::Null
        );
    }

    #[test]
    fn test_typeof() {
        assert_eq!(
            call_scalar("typeof", &[Value::Integer(1)]).unwrap(),
            Value::Text("integer".into())
        );
        assert_eq!(
            call_scalar("typeof", &[Value::Text("hi".into())]).unwrap(),
            Value::Text("text".into())
        );
    }

    #[test]
    fn test_substr() {
        assert_eq!(
            call_scalar(
                "substr",
                &[
                    Value::Text("hello".into()),
                    Value::Integer(2),
                    Value::Integer(3)
                ]
            )
            .unwrap(),
            Value::Text("ell".into())
        );
    }

    #[test]
    fn test_replace() {
        assert_eq!(
            call_scalar(
                "replace",
                &[
                    Value::Text("hello world".into()),
                    Value::Text("world".into()),
                    Value::Text("rust".into())
                ]
            )
            .unwrap(),
            Value::Text("hello rust".into())
        );
    }

    #[test]
    fn test_round() {
        assert_eq!(
            call_scalar("round", &[Value::Real(3.14159), Value::Integer(2)]).unwrap(),
            Value::Real(3.14)
        );
    }

    #[test]
    fn test_instr() {
        assert_eq!(
            call_scalar(
                "instr",
                &[
                    Value::Text("hello world".into()),
                    Value::Text("world".into())
                ]
            )
            .unwrap(),
            Value::Integer(7)
        );
    }
}
