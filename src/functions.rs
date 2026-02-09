/// Built-in scalar functions for SQLite compatibility.
use crate::types::Value;

/// Evaluate a built-in scalar function.
pub fn eval_function(name: &str, args: &[Value]) -> Option<Value> {
    match name {
        "LENGTH" => Some(fn_length(args)),
        "UPPER" => Some(fn_upper(args)),
        "LOWER" => Some(fn_lower(args)),
        "TYPEOF" => Some(fn_typeof(args)),
        "ABS" => Some(fn_abs(args)),
        "COALESCE" => Some(fn_coalesce(args)),
        "IFNULL" => Some(fn_ifnull(args)),
        "NULLIF" => Some(fn_nullif(args)),
        "SUBSTR" | "SUBSTRING" => Some(fn_substr(args)),
        "TRIM" => Some(fn_trim(args)),
        "LTRIM" => Some(fn_ltrim(args)),
        "RTRIM" => Some(fn_rtrim(args)),
        "REPLACE" => Some(fn_replace(args)),
        "HEX" => Some(fn_hex(args)),
        "QUOTE" => Some(fn_quote(args)),
        "RANDOM" => Some(fn_random()),
        "RANDOMBLOB" => Some(fn_randomblob(args)),
        "ZEROBLOB" => Some(fn_zeroblob(args)),
        "UNICODE" => Some(fn_unicode(args)),
        "CHAR" => Some(fn_char(args)),
        "INSTR" => Some(fn_instr(args)),
        "MIN" => {
            if args.len() >= 2 {
                Some(fn_min_scalar(args))
            } else {
                None // aggregate
            }
        }
        "MAX" => {
            if args.len() >= 2 {
                Some(fn_max_scalar(args))
            } else {
                None // aggregate
            }
        }
        "TOTAL_CHANGES" => Some(Value::Integer(0)),
        "CHANGES" => Some(Value::Integer(0)),
        "LAST_INSERT_ROWID" => Some(Value::Integer(0)),
        "SQLITE_VERSION" => Some(Value::Text("3.39.0".into())),
        "PRINTF" | "FORMAT" => Some(fn_printf(args)),
        "LIKE" => {
            if args.len() >= 2 {
                let pattern = args[0].to_text();
                let text = args[1].to_text();
                Some(Value::Integer(if like_match(&pattern, &text, None) {
                    1
                } else {
                    0
                }))
            } else {
                Some(Value::Null)
            }
        }
        "GLOB" => {
            if args.len() >= 2 {
                let pattern = args[0].to_text();
                let text = args[1].to_text();
                Some(Value::Integer(if glob_match(&pattern, &text) {
                    1
                } else {
                    0
                }))
            } else {
                Some(Value::Null)
            }
        }
        _ => None,
    }
}

fn fn_length(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Null;
    }
    match &args[0] {
        Value::Null => Value::Null,
        Value::Text(s) => Value::Integer(s.chars().count() as i64),
        Value::Blob(b) => Value::Integer(b.len() as i64),
        Value::Integer(i) => Value::Integer(i.to_string().len() as i64),
        Value::Real(f) => Value::Integer(f.to_string().len() as i64),
    }
}

fn fn_upper(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Null;
    }
    match &args[0] {
        Value::Null => Value::Null,
        v => Value::Text(v.to_text().to_uppercase()),
    }
}

fn fn_lower(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Null;
    }
    match &args[0] {
        Value::Null => Value::Null,
        v => Value::Text(v.to_text().to_lowercase()),
    }
}

fn fn_typeof(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Null;
    }
    Value::Text(args[0].type_name().to_string())
}

fn fn_abs(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Null;
    }
    match &args[0] {
        Value::Null => Value::Null,
        Value::Integer(i) => Value::Integer(i.abs()),
        Value::Real(f) => Value::Real(f.abs()),
        Value::Text(s) => {
            if let Ok(i) = s.parse::<i64>() {
                Value::Integer(i.abs())
            } else if let Ok(f) = s.parse::<f64>() {
                Value::Real(f.abs())
            } else {
                Value::Integer(0)
            }
        }
        _ => Value::Integer(0),
    }
}

fn fn_coalesce(args: &[Value]) -> Value {
    for arg in args {
        if !arg.is_null() {
            return arg.clone();
        }
    }
    Value::Null
}

fn fn_ifnull(args: &[Value]) -> Value {
    if args.len() < 2 {
        return Value::Null;
    }
    if args[0].is_null() {
        args[1].clone()
    } else {
        args[0].clone()
    }
}

fn fn_nullif(args: &[Value]) -> Value {
    if args.len() < 2 {
        return Value::Null;
    }
    if args[0] == args[1] {
        Value::Null
    } else {
        args[0].clone()
    }
}

fn fn_substr(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Null;
    }
    if args[0].is_null() {
        return Value::Null;
    }
    let s = args[0].to_text();
    let chars: Vec<char> = s.chars().collect();

    let start = if args.len() > 1 {
        args[1].to_integer().unwrap_or(1)
    } else {
        1
    };

    // SQLite uses 1-based indexing
    let start_idx = if start > 0 {
        (start - 1) as usize
    } else if start < 0 {
        let idx = chars.len() as i64 + start;
        if idx < 0 {
            0
        } else {
            idx as usize
        }
    } else {
        0
    };

    if args.len() > 2 {
        let len = args[2].to_integer().unwrap_or(chars.len() as i64);
        if len < 0 {
            return Value::Text(String::new());
        }
        let end_idx = (start_idx + len as usize).min(chars.len());
        let result: String = chars[start_idx.min(chars.len())..end_idx].iter().collect();
        Value::Text(result)
    } else {
        let result: String = chars[start_idx.min(chars.len())..].iter().collect();
        Value::Text(result)
    }
}

fn fn_trim(args: &[Value]) -> Value {
    if args.is_empty() || args[0].is_null() {
        return Value::Null;
    }
    Value::Text(args[0].to_text().trim().to_string())
}

fn fn_ltrim(args: &[Value]) -> Value {
    if args.is_empty() || args[0].is_null() {
        return Value::Null;
    }
    if args.len() > 1 && !args[1].is_null() {
        let chars_to_trim: Vec<char> = args[1].to_text().chars().collect();
        Value::Text(
            args[0]
                .to_text()
                .trim_start_matches(|c| chars_to_trim.contains(&c))
                .to_string(),
        )
    } else {
        Value::Text(args[0].to_text().trim_start().to_string())
    }
}

fn fn_rtrim(args: &[Value]) -> Value {
    if args.is_empty() || args[0].is_null() {
        return Value::Null;
    }
    if args.len() > 1 && !args[1].is_null() {
        let chars_to_trim: Vec<char> = args[1].to_text().chars().collect();
        Value::Text(
            args[0]
                .to_text()
                .trim_end_matches(|c| chars_to_trim.contains(&c))
                .to_string(),
        )
    } else {
        Value::Text(args[0].to_text().trim_end().to_string())
    }
}

fn fn_replace(args: &[Value]) -> Value {
    if args.len() < 3 || args[0].is_null() {
        return Value::Null;
    }
    let s = args[0].to_text();
    let from = args[1].to_text();
    let to = args[2].to_text();
    Value::Text(s.replace(&from, &to))
}

fn fn_hex(args: &[Value]) -> Value {
    if args.is_empty() || args[0].is_null() {
        return Value::Null;
    }
    match &args[0] {
        Value::Blob(b) => Value::Text(b.iter().map(|byte| format!("{:02X}", byte)).collect()),
        v => {
            let s = v.to_text();
            Value::Text(s.bytes().map(|b| format!("{:02X}", b)).collect())
        }
    }
}

fn fn_quote(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Null;
    }
    match &args[0] {
        Value::Null => Value::Text("NULL".into()),
        Value::Integer(i) => Value::Text(i.to_string()),
        Value::Real(f) => Value::Text(f.to_string()),
        Value::Text(s) => {
            let escaped = s.replace('\'', "''");
            Value::Text(format!("'{}'", escaped))
        }
        Value::Blob(b) => {
            let hex: String = b.iter().map(|byte| format!("{:02X}", byte)).collect();
            Value::Text(format!("X'{}'", hex))
        }
    }
}

fn fn_random() -> Value {
    // Simple random using a fast method
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::SystemTime;
    let mut hasher = DefaultHasher::new();
    SystemTime::now().hash(&mut hasher);
    std::thread::current().id().hash(&mut hasher);
    Value::Integer(hasher.finish() as i64)
}

fn fn_randomblob(args: &[Value]) -> Value {
    let n = if let Some(i) = args.first().and_then(|v| v.to_integer()) {
        i.max(0) as usize
    } else {
        0
    };
    let blob: Vec<u8> = (0..n).map(|i| (i * 7 + 13) as u8).collect();
    Value::Blob(blob)
}

fn fn_zeroblob(args: &[Value]) -> Value {
    let n = if let Some(i) = args.first().and_then(|v| v.to_integer()) {
        i.max(0) as usize
    } else {
        0
    };
    Value::Blob(vec![0u8; n])
}

fn fn_unicode(args: &[Value]) -> Value {
    if args.is_empty() || args[0].is_null() {
        return Value::Null;
    }
    let s = args[0].to_text();
    if let Some(c) = s.chars().next() {
        Value::Integer(c as i64)
    } else {
        Value::Null
    }
}

fn fn_char(args: &[Value]) -> Value {
    let mut s = String::new();
    for arg in args {
        if let Some(i) = arg.to_integer() {
            if let Some(c) = char::from_u32(i as u32) {
                s.push(c);
            }
        }
    }
    Value::Text(s)
}

fn fn_instr(args: &[Value]) -> Value {
    if args.len() < 2 || args[0].is_null() || args[1].is_null() {
        return Value::Null;
    }
    let haystack = args[0].to_text();
    let needle = args[1].to_text();
    match haystack.find(&needle) {
        Some(pos) => Value::Integer((pos + 1) as i64), // 1-based
        None => Value::Integer(0),
    }
}

fn fn_min_scalar(args: &[Value]) -> Value {
    let mut min = &args[0];
    for arg in &args[1..] {
        if arg.is_null() {
            return Value::Null;
        }
        if min.is_null() || arg.sqlite_cmp(min) == std::cmp::Ordering::Less {
            min = arg;
        }
    }
    min.clone()
}

fn fn_max_scalar(args: &[Value]) -> Value {
    let mut max = &args[0];
    for arg in &args[1..] {
        if arg.is_null() {
            return Value::Null;
        }
        if max.is_null() || arg.sqlite_cmp(max) == std::cmp::Ordering::Greater {
            max = arg;
        }
    }
    max.clone()
}

fn fn_printf(args: &[Value]) -> Value {
    if args.is_empty() {
        return Value::Null;
    }
    // Simplified printf: just return the format string with %s/%d replaced
    let fmt = args[0].to_text();
    if args.len() == 1 {
        return Value::Text(fmt);
    }

    let mut result = String::new();
    let mut arg_idx = 1;
    let chars: Vec<char> = fmt.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '%' && i + 1 < chars.len() {
            match chars[i + 1] {
                'd' | 'i' => {
                    if arg_idx < args.len() {
                        result.push_str(&args[arg_idx].to_integer().unwrap_or(0).to_string());
                        arg_idx += 1;
                    }
                    i += 2;
                }
                's' => {
                    if arg_idx < args.len() {
                        result.push_str(&args[arg_idx].to_text());
                        arg_idx += 1;
                    }
                    i += 2;
                }
                'f' => {
                    if arg_idx < args.len() {
                        result.push_str(&format!("{:.6}", args[arg_idx].to_real().unwrap_or(0.0)));
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
    Value::Text(result)
}

/// LIKE pattern matching (case-insensitive by default).
pub fn like_match(pattern: &str, text: &str, escape_char: Option<char>) -> bool {
    let pattern: Vec<char> = pattern.chars().collect();
    let text: Vec<char> = text.chars().collect();
    like_match_impl(&pattern, &text, 0, 0, escape_char)
}

fn like_match_impl(
    pattern: &[char],
    text: &[char],
    mut pi: usize,
    mut ti: usize,
    escape_char: Option<char>,
) -> bool {
    while pi < pattern.len() {
        if Some(pattern[pi]) == escape_char {
            pi += 1;
            if pi >= pattern.len() {
                return false;
            }
            if ti >= text.len() || !chars_eq_ci(pattern[pi], text[ti]) {
                return false;
            }
            pi += 1;
            ti += 1;
        } else if pattern[pi] == '%' {
            // Skip consecutive %
            while pi < pattern.len() && pattern[pi] == '%' {
                pi += 1;
            }
            if pi >= pattern.len() {
                return true;
            }
            for i in ti..=text.len() {
                if like_match_impl(pattern, text, pi, i, escape_char) {
                    return true;
                }
            }
            return false;
        } else if pattern[pi] == '_' {
            if ti >= text.len() {
                return false;
            }
            pi += 1;
            ti += 1;
        } else {
            if ti >= text.len() || !chars_eq_ci(pattern[pi], text[ti]) {
                return false;
            }
            pi += 1;
            ti += 1;
        }
    }
    ti >= text.len()
}

fn chars_eq_ci(a: char, b: char) -> bool {
    a.eq_ignore_ascii_case(&b)
}

/// GLOB pattern matching (case-sensitive).
pub fn glob_match(pattern: &str, text: &str) -> bool {
    let pattern: Vec<char> = pattern.chars().collect();
    let text: Vec<char> = text.chars().collect();
    glob_match_impl(&pattern, &text, 0, 0)
}

fn glob_match_impl(pattern: &[char], text: &[char], mut pi: usize, mut ti: usize) -> bool {
    while pi < pattern.len() {
        match pattern[pi] {
            '*' => {
                while pi < pattern.len() && pattern[pi] == '*' {
                    pi += 1;
                }
                if pi >= pattern.len() {
                    return true;
                }
                for i in ti..=text.len() {
                    if glob_match_impl(pattern, text, pi, i) {
                        return true;
                    }
                }
                return false;
            }
            '?' => {
                if ti >= text.len() {
                    return false;
                }
                pi += 1;
                ti += 1;
            }
            '[' => {
                pi += 1;
                if ti >= text.len() {
                    return false;
                }
                let negate = pi < pattern.len() && pattern[pi] == '^';
                if negate {
                    pi += 1;
                }
                let mut found = false;
                while pi < pattern.len() && pattern[pi] != ']' {
                    let start = pattern[pi];
                    pi += 1;
                    if pi + 1 < pattern.len() && pattern[pi] == '-' {
                        pi += 1;
                        let end = pattern[pi];
                        pi += 1;
                        if text[ti] >= start && text[ti] <= end {
                            found = true;
                        }
                    } else if text[ti] == start {
                        found = true;
                    }
                }
                if pi < pattern.len() {
                    pi += 1; // skip ']'
                }
                if found == negate {
                    return false;
                }
                ti += 1;
            }
            c => {
                if ti >= text.len() || c != text[ti] {
                    return false;
                }
                pi += 1;
                ti += 1;
            }
        }
    }
    ti >= text.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_like_match() {
        assert!(like_match("%", "anything", None));
        assert!(like_match("hello", "hello", None));
        assert!(like_match("hello", "HELLO", None)); // case insensitive
        assert!(like_match("h%o", "hello", None));
        assert!(like_match("h_llo", "hello", None));
        assert!(!like_match("h_llo", "hllo", None));
        assert!(like_match("%world%", "hello world!", None));
        assert!(!like_match("xyz", "abc", None));
    }

    #[test]
    fn test_glob_match() {
        assert!(glob_match("*", "anything"));
        assert!(glob_match("hello", "hello"));
        assert!(!glob_match("hello", "HELLO")); // case sensitive
        assert!(glob_match("h*o", "hello"));
        assert!(glob_match("h?llo", "hello"));
        assert!(!glob_match("h?llo", "hllo"));
    }

    #[test]
    fn test_length() {
        assert_eq!(fn_length(&[Value::Text("hello".into())]), Value::Integer(5));
        assert_eq!(fn_length(&[Value::Null]), Value::Null);
        assert_eq!(fn_length(&[Value::Blob(vec![1, 2, 3])]), Value::Integer(3));
    }

    #[test]
    fn test_upper_lower() {
        assert_eq!(
            fn_upper(&[Value::Text("hello".into())]),
            Value::Text("HELLO".into())
        );
        assert_eq!(
            fn_lower(&[Value::Text("HELLO".into())]),
            Value::Text("hello".into())
        );
    }

    #[test]
    fn test_coalesce() {
        assert_eq!(
            fn_coalesce(&[Value::Null, Value::Null, Value::Integer(42)]),
            Value::Integer(42)
        );
        assert_eq!(fn_coalesce(&[Value::Null, Value::Null]), Value::Null);
    }

    #[test]
    fn test_substr() {
        assert_eq!(
            fn_substr(&[
                Value::Text("hello".into()),
                Value::Integer(2),
                Value::Integer(3)
            ]),
            Value::Text("ell".into())
        );
    }
}
