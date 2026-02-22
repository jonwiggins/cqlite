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
        // IIF(cond, true_val, false_val) — shorthand for CASE WHEN
        "IIF" => {
            check_args(name, args, 3)?;
            if args[0].is_truthy() {
                Ok(args[1].clone())
            } else {
                Ok(args[2].clone())
            }
        }
        // SIGN(x) — returns -1, 0, or 1
        "SIGN" => {
            check_args(name, args, 1)?;
            match &args[0] {
                Value::Null => Ok(Value::Null),
                Value::Integer(n) => Ok(Value::Integer(n.signum())),
                Value::Real(f) => {
                    if f.is_nan() {
                        Ok(Value::Null)
                    } else if *f > 0.0 {
                        Ok(Value::Integer(1))
                    } else if *f < 0.0 {
                        Ok(Value::Integer(-1))
                    } else {
                        Ok(Value::Integer(0))
                    }
                }
                _ => Ok(Value::Integer(0)),
            }
        }
        // LIKELY/UNLIKELY — optimization hints, just return the value
        "LIKELY" | "UNLIKELY" => {
            check_args(name, args, 1)?;
            Ok(args[0].clone())
        }
        // LIKE(pattern, string) — function form of LIKE operator
        "LIKE" => {
            if args.len() < 2 || args.len() > 3 {
                return Err(RsqliteError::Runtime(
                    "like() takes 2 or 3 arguments".into(),
                ));
            }
            if args[0].is_null() || args[1].is_null() {
                return Ok(Value::Null);
            }
            let pattern = args[0].to_text().unwrap_or_default();
            let text = args[1].to_text().unwrap_or_default();
            let escape = if args.len() == 3 {
                args[2].to_text().unwrap_or_default().chars().next()
            } else {
                None
            };
            Ok(Value::Integer(if like_match(&pattern, &text, escape) {
                1
            } else {
                0
            }))
        }
        // GLOB(pattern, string) — function form of GLOB operator
        "GLOB" => {
            check_args(name, args, 2)?;
            if args[0].is_null() || args[1].is_null() {
                return Ok(Value::Null);
            }
            let pattern = args[0].to_text().unwrap_or_default();
            let text = args[1].to_text().unwrap_or_default();
            Ok(Value::Integer(if glob_match(&pattern, &text) {
                1
            } else {
                0
            }))
        }
        // Date/time functions
        "DATE" => call_date(args),
        "TIME" => call_time(args),
        "DATETIME" => call_datetime(args),
        "JULIANDAY" => call_julianday(args),
        "UNIXEPOCH" => call_unixepoch(args),
        "STRFTIME" => call_strftime(args),
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

// ── LIKE pattern matching (case-insensitive) ──

fn like_match(pattern: &str, text: &str, escape: Option<char>) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    like_match_inner(&p, &t, 0, 0, escape)
}

fn like_match_inner(p: &[char], t: &[char], pi: usize, ti: usize, escape: Option<char>) -> bool {
    if pi == p.len() {
        return ti == t.len();
    }
    if Some(p[pi]) == escape && pi + 1 < p.len() {
        if ti < t.len() && p[pi + 1].to_lowercase().eq(t[ti].to_lowercase()) {
            return like_match_inner(p, t, pi + 2, ti + 1, escape);
        }
        return false;
    }
    match p[pi] {
        '%' => {
            for i in ti..=t.len() {
                if like_match_inner(p, t, pi + 1, i, escape) {
                    return true;
                }
            }
            false
        }
        '_' => {
            if ti < t.len() {
                like_match_inner(p, t, pi + 1, ti + 1, escape)
            } else {
                false
            }
        }
        c => {
            if ti < t.len() && c.to_lowercase().eq(t[ti].to_lowercase()) {
                like_match_inner(p, t, pi + 1, ti + 1, escape)
            } else {
                false
            }
        }
    }
}

// ── GLOB pattern matching (case-sensitive) ──

fn glob_match(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    glob_match_inner(&p, &t, 0, 0)
}

fn glob_match_inner(p: &[char], t: &[char], pi: usize, ti: usize) -> bool {
    if pi == p.len() {
        return ti == t.len();
    }
    match p[pi] {
        '*' => {
            for i in ti..=t.len() {
                if glob_match_inner(p, t, pi + 1, i) {
                    return true;
                }
            }
            false
        }
        '?' => {
            if ti < t.len() {
                glob_match_inner(p, t, pi + 1, ti + 1)
            } else {
                false
            }
        }
        '[' => {
            if ti >= t.len() {
                return false;
            }
            let mut j = pi + 1;
            let negate = j < p.len() && p[j] == '^';
            if negate {
                j += 1;
            }
            let mut matched = false;
            while j < p.len() && p[j] != ']' {
                if j + 2 < p.len() && p[j + 1] == '-' {
                    if t[ti] >= p[j] && t[ti] <= p[j + 2] {
                        matched = true;
                    }
                    j += 3;
                } else {
                    if t[ti] == p[j] {
                        matched = true;
                    }
                    j += 1;
                }
            }
            if negate {
                matched = !matched;
            }
            if matched && j < p.len() {
                glob_match_inner(p, t, j + 1, ti + 1)
            } else {
                false
            }
        }
        c => {
            if ti < t.len() && c == t[ti] {
                glob_match_inner(p, t, pi + 1, ti + 1)
            } else {
                false
            }
        }
    }
}

// ── Date/time helpers ──

/// Parse a time string into (year, month, day, hour, minute, second).
/// Supports: 'now', 'YYYY-MM-DD', 'YYYY-MM-DD HH:MM:SS', 'YYYY-MM-DD HH:MM',
/// unix timestamps (numeric), Julian day numbers.
fn parse_datetime(s: &str) -> Option<(i64, u32, u32, u32, u32, u32)> {
    let s = s.trim();
    if s.eq_ignore_ascii_case("now") {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        return Some(unix_to_datetime(now));
    }

    // Try YYYY-MM-DD HH:MM:SS
    if s.len() >= 19 && s.as_bytes()[4] == b'-' && s.as_bytes()[10] == b' ' {
        let year = s[0..4].parse::<i64>().ok()?;
        let month = s[5..7].parse::<u32>().ok()?;
        let day = s[8..10].parse::<u32>().ok()?;
        let hour = s[11..13].parse::<u32>().ok()?;
        let minute = s[14..16].parse::<u32>().ok()?;
        let second = s[17..19].parse::<u32>().ok()?;
        return Some((year, month, day, hour, minute, second));
    }

    // Try YYYY-MM-DD HH:MM
    if s.len() >= 16 && s.as_bytes()[4] == b'-' && s.as_bytes()[10] == b' ' {
        let year = s[0..4].parse::<i64>().ok()?;
        let month = s[5..7].parse::<u32>().ok()?;
        let day = s[8..10].parse::<u32>().ok()?;
        let hour = s[11..13].parse::<u32>().ok()?;
        let minute = s[14..16].parse::<u32>().ok()?;
        return Some((year, month, day, hour, minute, 0));
    }

    // Try YYYY-MM-DD
    if s.len() >= 10 && s.as_bytes()[4] == b'-' {
        let year = s[0..4].parse::<i64>().ok()?;
        let month = s[5..7].parse::<u32>().ok()?;
        let day = s[8..10].parse::<u32>().ok()?;
        return Some((year, month, day, 0, 0, 0));
    }

    // Try HH:MM:SS
    if s.len() >= 8 && s.as_bytes()[2] == b':' && s.as_bytes()[5] == b':' {
        let hour = s[0..2].parse::<u32>().ok()?;
        let minute = s[3..5].parse::<u32>().ok()?;
        let second = s[6..8].parse::<u32>().ok()?;
        return Some((2000, 1, 1, hour, minute, second));
    }

    // Try HH:MM
    if s.len() >= 5 && s.as_bytes()[2] == b':' {
        let hour = s[0..2].parse::<u32>().ok()?;
        let minute = s[3..5].parse::<u32>().ok()?;
        return Some((2000, 1, 1, hour, minute, 0));
    }

    // Try unix timestamp (numeric)
    if let Ok(ts) = s.parse::<i64>() {
        return Some(unix_to_datetime(ts));
    }
    if let Ok(ts) = s.parse::<f64>() {
        return Some(unix_to_datetime(ts as i64));
    }

    None
}

/// Resolve the time value from args[0] (or 'now' if empty).
fn resolve_time_arg(args: &[Value]) -> Option<(i64, u32, u32, u32, u32, u32)> {
    if args.is_empty() {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        return Some(unix_to_datetime(now));
    }
    match &args[0] {
        Value::Null => None,
        Value::Text(s) => parse_datetime(s),
        Value::Integer(n) => {
            // Treat as unix timestamp
            Some(unix_to_datetime(*n))
        }
        Value::Real(f) => Some(unix_to_datetime(*f as i64)),
        _ => None,
    }
}

fn is_leap_year(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

fn days_in_month(y: i64, m: u32) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap_year(y) {
                29
            } else {
                28
            }
        }
        _ => 30,
    }
}

/// Convert (year, month, day) to Julian Day Number.
fn date_to_jdn(y: i64, m: u32, d: u32) -> i64 {
    let a = (14 - m as i64) / 12;
    let y2 = y + 4800 - a;
    let m2 = m as i64 + 12 * a - 3;
    d as i64 + (153 * m2 + 2) / 5 + 365 * y2 + y2 / 4 - y2 / 100 + y2 / 400 - 32045
}

/// Convert Julian Day Number to (year, month, day).
fn jdn_to_date(jdn: i64) -> (i64, u32, u32) {
    let a = jdn + 32044;
    let b = (4 * a + 3) / 146097;
    let c = a - 146097 * b / 4;
    let d = (4 * c + 3) / 1461;
    let e = c - 1461 * d / 4;
    let m = (5 * e + 2) / 153;
    let day = (e - (153 * m + 2) / 5 + 1) as u32;
    let month = (m + 3 - 12 * (m / 10)) as u32;
    let year = 100 * b + d - 4800 + m / 10;
    (year, month, day)
}

/// Convert datetime tuple to unix timestamp.
fn datetime_to_unix(y: i64, m: u32, d: u32, h: u32, min: u32, s: u32) -> i64 {
    let jdn = date_to_jdn(y, m, d);
    let epoch_jdn = date_to_jdn(1970, 1, 1);
    let days = jdn - epoch_jdn;
    days * 86400 + h as i64 * 3600 + min as i64 * 60 + s as i64
}

/// Convert unix timestamp to datetime tuple.
fn unix_to_datetime(ts: i64) -> (i64, u32, u32, u32, u32, u32) {
    let epoch_jdn = date_to_jdn(1970, 1, 1);
    let total_secs = ts;
    let days = total_secs.div_euclid(86400);
    let day_secs = total_secs.rem_euclid(86400);
    let (y, m, d) = jdn_to_date(epoch_jdn + days);
    let h = (day_secs / 3600) as u32;
    let min = ((day_secs % 3600) / 60) as u32;
    let s = (day_secs % 60) as u32;
    (y, m, d, h, min, s)
}

/// Compute day of week (0=Sunday, 6=Saturday).
fn day_of_week(y: i64, m: u32, d: u32) -> u32 {
    let jdn = date_to_jdn(y, m, d);
    ((jdn + 1) % 7) as u32
}

/// Compute day of year (1-366).
fn day_of_year(y: i64, m: u32, d: u32) -> u32 {
    let jan1 = date_to_jdn(y, 1, 1);
    let today = date_to_jdn(y, m, d);
    (today - jan1 + 1) as u32
}

/// Apply modifiers to a datetime tuple. Supports:
/// '+N days', '-N hours', 'start of month', 'start of year', 'start of day',
/// 'weekday N', 'unixepoch', 'localtime', 'utc'
fn apply_modifiers(dt: &mut (i64, u32, u32, u32, u32, u32), args: &[Value], start: usize) {
    for arg in args.iter().skip(start) {
        let s = match arg {
            Value::Text(s) => s.trim().to_lowercase(),
            _ => continue,
        };
        if s == "unixepoch" {
            // Handled at parse time by resolve_datetime_with_modifiers
            continue;
        }
        if s == "localtime" || s == "utc" {
            continue; // We don't handle timezone conversion
        }
        if s == "start of month" {
            dt.2 = 1;
            dt.3 = 0;
            dt.4 = 0;
            dt.5 = 0;
            continue;
        }
        if s == "start of year" {
            dt.1 = 1;
            dt.2 = 1;
            dt.3 = 0;
            dt.4 = 0;
            dt.5 = 0;
            continue;
        }
        if s == "start of day" {
            dt.3 = 0;
            dt.4 = 0;
            dt.5 = 0;
            continue;
        }
        if let Some(rest) = s.strip_prefix("weekday ") {
            if let Ok(wd) = rest.parse::<u32>() {
                let current = day_of_week(dt.0, dt.1, dt.2);
                let advance = if current <= wd {
                    wd - current
                } else {
                    7 - current + wd
                };
                if advance > 0 {
                    let ts = datetime_to_unix(dt.0, dt.1, dt.2, dt.3, dt.4, dt.5)
                        + advance as i64 * 86400;
                    *dt = unix_to_datetime(ts);
                }
            }
            continue;
        }
        // Try +/-N unit modifiers
        let (sign, rest) = if let Some(r) = s.strip_prefix('+') {
            (1i64, r.trim())
        } else if let Some(r) = s.strip_prefix('-') {
            (-1i64, r.trim())
        } else {
            continue;
        };
        let parts: Vec<&str> = rest.splitn(2, ' ').collect();
        if parts.len() != 2 {
            continue;
        }
        let n: i64 = match parts[0].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let unit = parts[1].trim_end_matches('s'); // "days" -> "day"
        match unit {
            "second" => {
                let ts = datetime_to_unix(dt.0, dt.1, dt.2, dt.3, dt.4, dt.5) + sign * n;
                *dt = unix_to_datetime(ts);
            }
            "minute" => {
                let ts = datetime_to_unix(dt.0, dt.1, dt.2, dt.3, dt.4, dt.5) + sign * n * 60;
                *dt = unix_to_datetime(ts);
            }
            "hour" => {
                let ts = datetime_to_unix(dt.0, dt.1, dt.2, dt.3, dt.4, dt.5) + sign * n * 3600;
                *dt = unix_to_datetime(ts);
            }
            "day" => {
                let ts = datetime_to_unix(dt.0, dt.1, dt.2, dt.3, dt.4, dt.5) + sign * n * 86400;
                *dt = unix_to_datetime(ts);
            }
            "month" => {
                let total_months = dt.0 * 12 + dt.1 as i64 - 1 + sign * n;
                dt.0 = total_months.div_euclid(12);
                dt.1 = (total_months.rem_euclid(12) + 1) as u32;
                let max_day = days_in_month(dt.0, dt.1);
                if dt.2 > max_day {
                    dt.2 = max_day;
                }
            }
            "year" => {
                dt.0 += sign * n;
                let max_day = days_in_month(dt.0, dt.1);
                if dt.2 > max_day {
                    dt.2 = max_day;
                }
            }
            _ => {}
        }
    }
}

/// Resolve the first arg, handling 'unixepoch' modifier.
fn resolve_datetime_with_modifiers(args: &[Value]) -> Option<(i64, u32, u32, u32, u32, u32)> {
    // Check if 'unixepoch' modifier is present
    let has_unixepoch = args
        .iter()
        .skip(1)
        .any(|a| matches!(a, Value::Text(s) if s.trim().eq_ignore_ascii_case("unixepoch")));

    let mut dt = if has_unixepoch {
        // First arg is a unix timestamp
        match &args[0] {
            Value::Integer(n) => unix_to_datetime(*n),
            Value::Real(f) => unix_to_datetime(*f as i64),
            Value::Text(s) => {
                if let Ok(n) = s.parse::<i64>() {
                    unix_to_datetime(n)
                } else if let Ok(f) = s.parse::<f64>() {
                    unix_to_datetime(f as i64)
                } else {
                    return None;
                }
            }
            _ => return None,
        }
    } else {
        resolve_time_arg(args)?
    };

    if args.len() > 1 {
        apply_modifiers(&mut dt, args, 1);
    }
    Some(dt)
}

fn call_date(args: &[Value]) -> Result<Value> {
    match resolve_datetime_with_modifiers(args) {
        Some((y, m, d, _, _, _)) => Ok(Value::Text(format!("{y:04}-{m:02}-{d:02}"))),
        None => Ok(Value::Null),
    }
}

fn call_time(args: &[Value]) -> Result<Value> {
    match resolve_datetime_with_modifiers(args) {
        Some((_, _, _, h, min, s)) => Ok(Value::Text(format!("{h:02}:{min:02}:{s:02}"))),
        None => Ok(Value::Null),
    }
}

fn call_datetime(args: &[Value]) -> Result<Value> {
    match resolve_datetime_with_modifiers(args) {
        Some((y, m, d, h, min, s)) => Ok(Value::Text(format!(
            "{y:04}-{m:02}-{d:02} {h:02}:{min:02}:{s:02}"
        ))),
        None => Ok(Value::Null),
    }
}

fn call_julianday(args: &[Value]) -> Result<Value> {
    match resolve_datetime_with_modifiers(args) {
        Some((y, m, d, h, min, s)) => {
            let jdn = date_to_jdn(y, m, d) as f64;
            let frac = (h as f64 * 3600.0 + min as f64 * 60.0 + s as f64) / 86400.0;
            Ok(Value::Real(jdn as f64 + frac - 0.5))
        }
        None => Ok(Value::Null),
    }
}

fn call_unixepoch(args: &[Value]) -> Result<Value> {
    match resolve_datetime_with_modifiers(args) {
        Some((y, m, d, h, min, s)) => Ok(Value::Integer(datetime_to_unix(y, m, d, h, min, s))),
        None => Ok(Value::Null),
    }
}

fn call_strftime(args: &[Value]) -> Result<Value> {
    if args.is_empty() {
        return Err(RsqliteError::Runtime(
            "strftime() requires at least 1 argument".into(),
        ));
    }
    let fmt = match &args[0] {
        Value::Text(s) => s.clone(),
        _ => return Ok(Value::Null),
    };
    let dt_args = &args[1..];
    let (y, m, d, h, min, s) = match resolve_datetime_with_modifiers(dt_args) {
        Some(dt) => dt,
        None => return Ok(Value::Null),
    };

    let mut result = String::new();
    let chars: Vec<char> = fmt.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '%' && i + 1 < chars.len() {
            match chars[i + 1] {
                'Y' => result.push_str(&format!("{y:04}")),
                'm' => result.push_str(&format!("{m:02}")),
                'd' => result.push_str(&format!("{d:02}")),
                'H' => result.push_str(&format!("{h:02}")),
                'M' => result.push_str(&format!("{min:02}")),
                'S' => result.push_str(&format!("{s:02}")),
                'j' => result.push_str(&format!("{:03}", day_of_year(y, m, d))),
                'w' => result.push_str(&format!("{}", day_of_week(y, m, d))),
                'W' => {
                    // ISO week number (approximate)
                    let doy = day_of_year(y, m, d);
                    let dow = day_of_week(y, m, d);
                    let week = (doy + 6 - dow) / 7;
                    result.push_str(&format!("{week:02}"));
                }
                's' => {
                    result.push_str(&format!("{}", datetime_to_unix(y, m, d, h, min, s)));
                }
                'f' => {
                    result.push_str(&format!("{s:02}.000"));
                }
                'J' => {
                    let jdn = date_to_jdn(y, m, d) as f64;
                    let frac = (h as f64 * 3600.0 + min as f64 * 60.0 + s as f64) / 86400.0;
                    result.push_str(&format!("{:.6}", jdn + frac - 0.5));
                }
                '%' => result.push('%'),
                _ => {
                    result.push('%');
                    result.push(chars[i + 1]);
                }
            }
            i += 2;
        } else {
            result.push(chars[i]);
            i += 1;
        }
    }
    Ok(Value::Text(result))
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

    #[test]
    fn test_iif() {
        assert_eq!(
            call_scalar(
                "iif",
                &[
                    Value::Integer(1),
                    Value::Text("yes".into()),
                    Value::Text("no".into())
                ]
            )
            .unwrap(),
            Value::Text("yes".into())
        );
        assert_eq!(
            call_scalar(
                "iif",
                &[
                    Value::Integer(0),
                    Value::Text("yes".into()),
                    Value::Text("no".into())
                ]
            )
            .unwrap(),
            Value::Text("no".into())
        );
    }

    #[test]
    fn test_sign() {
        assert_eq!(
            call_scalar("sign", &[Value::Integer(42)]).unwrap(),
            Value::Integer(1)
        );
        assert_eq!(
            call_scalar("sign", &[Value::Integer(-5)]).unwrap(),
            Value::Integer(-1)
        );
        assert_eq!(
            call_scalar("sign", &[Value::Integer(0)]).unwrap(),
            Value::Integer(0)
        );
        assert_eq!(
            call_scalar("sign", &[Value::Real(-3.14)]).unwrap(),
            Value::Integer(-1)
        );
    }

    #[test]
    fn test_likely_unlikely() {
        assert_eq!(
            call_scalar("likely", &[Value::Integer(42)]).unwrap(),
            Value::Integer(42)
        );
        assert_eq!(
            call_scalar("unlikely", &[Value::Integer(0)]).unwrap(),
            Value::Integer(0)
        );
    }

    #[test]
    fn test_like_function() {
        assert_eq!(
            call_scalar(
                "like",
                &[Value::Text("%ello".into()), Value::Text("hello".into())]
            )
            .unwrap(),
            Value::Integer(1)
        );
        assert_eq!(
            call_scalar(
                "like",
                &[Value::Text("%xyz%".into()), Value::Text("hello".into())]
            )
            .unwrap(),
            Value::Integer(0)
        );
    }

    #[test]
    fn test_glob_function() {
        assert_eq!(
            call_scalar(
                "glob",
                &[Value::Text("h*o".into()), Value::Text("hello".into())]
            )
            .unwrap(),
            Value::Integer(1)
        );
        assert_eq!(
            call_scalar(
                "glob",
                &[Value::Text("xyz*".into()), Value::Text("hello".into())]
            )
            .unwrap(),
            Value::Integer(0)
        );
    }

    #[test]
    fn test_date_function() {
        assert_eq!(
            call_scalar("date", &[Value::Text("2023-06-15".into())]).unwrap(),
            Value::Text("2023-06-15".into())
        );
        assert_eq!(
            call_scalar("date", &[Value::Text("2023-06-15 14:30:00".into())]).unwrap(),
            Value::Text("2023-06-15".into())
        );
    }

    #[test]
    fn test_time_function() {
        assert_eq!(
            call_scalar("time", &[Value::Text("2023-06-15 14:30:45".into())]).unwrap(),
            Value::Text("14:30:45".into())
        );
    }

    #[test]
    fn test_datetime_function() {
        assert_eq!(
            call_scalar("datetime", &[Value::Text("2023-06-15 14:30:45".into())]).unwrap(),
            Value::Text("2023-06-15 14:30:45".into())
        );
    }

    #[test]
    fn test_datetime_modifiers() {
        assert_eq!(
            call_scalar(
                "datetime",
                &[
                    Value::Text("2023-06-15 14:30:00".into()),
                    Value::Text("+1 day".into()),
                ]
            )
            .unwrap(),
            Value::Text("2023-06-16 14:30:00".into())
        );
        assert_eq!(
            call_scalar(
                "datetime",
                &[
                    Value::Text("2023-06-15 14:30:00".into()),
                    Value::Text("start of month".into()),
                ]
            )
            .unwrap(),
            Value::Text("2023-06-01 00:00:00".into())
        );
    }

    #[test]
    fn test_julianday() {
        // Julian day of 2000-01-01 12:00:00 should be 2451545.0
        let result =
            call_scalar("julianday", &[Value::Text("2000-01-01 12:00:00".into())]).unwrap();
        match result {
            Value::Real(f) => assert!((f - 2451545.0).abs() < 0.001, "got {f}"),
            other => panic!("expected Real, got {other:?}"),
        }
    }

    #[test]
    fn test_unixepoch() {
        assert_eq!(
            call_scalar("unixepoch", &[Value::Text("1970-01-01 00:00:00".into())]).unwrap(),
            Value::Integer(0)
        );
        assert_eq!(
            call_scalar("unixepoch", &[Value::Text("2000-01-01 00:00:00".into())]).unwrap(),
            Value::Integer(946684800)
        );
    }

    #[test]
    fn test_strftime() {
        assert_eq!(
            call_scalar(
                "strftime",
                &[
                    Value::Text("%Y-%m-%d".into()),
                    Value::Text("2023-06-15 14:30:00".into()),
                ]
            )
            .unwrap(),
            Value::Text("2023-06-15".into())
        );
        assert_eq!(
            call_scalar(
                "strftime",
                &[
                    Value::Text("%H:%M".into()),
                    Value::Text("2023-06-15 14:30:00".into()),
                ]
            )
            .unwrap(),
            Value::Text("14:30".into())
        );
    }

    #[test]
    fn test_date_unix_modifier() {
        assert_eq!(
            call_scalar(
                "datetime",
                &[Value::Text("0".into()), Value::Text("unixepoch".into()),]
            )
            .unwrap(),
            Value::Text("1970-01-01 00:00:00".into())
        );
    }
}
