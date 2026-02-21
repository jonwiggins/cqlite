// sqllogictest runner for cqlite.
//
// Parses .test files in the sqllogictest format and executes them against
// our database engine. Reports pass/fail counts per file.

use rsqlite::planner::Database;
use rsqlite::types::Value;
use std::path::Path;

/// A parsed record from a sqllogictest file.
#[derive(Debug)]
enum Record {
    Statement {
        expected_error: bool,
        sql: String,
    },
    Query {
        _types: String,
        sort_mode: SortMode,
        sql: String,
        expected: ExpectedResult,
    },
    Halt,
}

/// What we expect from a query.
#[derive(Debug)]
enum ExpectedResult {
    /// Exact values, one per line.
    Values(Vec<String>),
    /// Hash-based: "N values hashing to HASH".
    Hash { count: usize, md5: String },
    /// No expected result provided.
    None,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum SortMode {
    NoSort,
    RowSort,
    ValueSort,
}

/// Parse a sqllogictest file into records.
fn parse_test_file(content: &str) -> Vec<Record> {
    let mut records = Vec::new();
    let lines: Vec<&str> = content.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i].trim();

        // Skip blank lines and comments.
        if line.is_empty() || line.starts_with('#') {
            i += 1;
            continue;
        }

        // Skip conditional directives (skipif/onlyif) — we run everything.
        if line.starts_with("skipif") || line.starts_with("onlyif") {
            i += 1;
            continue;
        }

        if line == "halt" {
            records.push(Record::Halt);
            break;
        }

        if line.starts_with("hash-threshold") {
            i += 1;
            continue;
        }

        if line.starts_with("statement") {
            let expected_error = line.contains("error");
            i += 1;

            // Collect SQL lines until blank line or EOF.
            let mut sql_lines = Vec::new();
            while i < lines.len() && !lines[i].trim().is_empty() {
                if !lines[i].trim().starts_with('#') {
                    sql_lines.push(lines[i]);
                }
                i += 1;
            }

            records.push(Record::Statement {
                expected_error,
                sql: sql_lines.join("\n"),
            });
        } else if line.starts_with("query") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            let types = parts.get(1).unwrap_or(&"").to_string();
            let sort_mode = match parts.get(2).copied() {
                Some("rowsort") => SortMode::RowSort,
                Some("valuesort") => SortMode::ValueSort,
                _ => SortMode::NoSort,
            };
            i += 1;

            // Collect SQL lines until ---- or blank line.
            let mut sql_lines = Vec::new();
            while i < lines.len() && lines[i].trim() != "----" && !lines[i].trim().is_empty() {
                if !lines[i].trim().starts_with('#') {
                    sql_lines.push(lines[i]);
                }
                i += 1;
            }

            // Collect expected values after ----.
            let expected = if i < lines.len() && lines[i].trim() == "----" {
                i += 1;
                // Check if the next line is a hash result: "N values hashing to HASH"
                if i < lines.len() && lines[i].contains("values hashing to") {
                    let hash_line = lines[i].trim();
                    i += 1;
                    // Parse "N values hashing to HASH"
                    let parts: Vec<&str> = hash_line.split_whitespace().collect();
                    if parts.len() >= 5 {
                        let count = parts[0].parse::<usize>().unwrap_or(0);
                        let md5 = parts[4].to_string();
                        ExpectedResult::Hash { count, md5 }
                    } else {
                        ExpectedResult::None
                    }
                } else {
                    let mut values = Vec::new();
                    while i < lines.len() && !lines[i].trim().is_empty() {
                        values.push(lines[i].to_string());
                        i += 1;
                    }
                    ExpectedResult::Values(values)
                }
            } else {
                ExpectedResult::None
            };

            records.push(Record::Query {
                _types: types,
                sort_mode,
                sql: sql_lines.join("\n"),
                expected,
            });
        } else {
            // Unknown line, skip.
            i += 1;
        }
    }

    records
}

/// Format a Value for sqllogictest comparison.
fn format_value(val: &Value) -> String {
    match val {
        Value::Null => "NULL".to_string(),
        Value::Integer(i) => format!("{i}"),
        Value::Real(r) => format!("{r:.3}"),
        Value::Text(s) => {
            if s.is_empty() {
                "(empty)".to_string()
            } else {
                s.clone()
            }
        }
        Value::Blob(b) => {
            if b.is_empty() {
                "(empty)".to_string()
            } else {
                format!(
                    "X'{}'",
                    b.iter()
                        .map(|byte| format!("{byte:02X}"))
                        .collect::<String>()
                )
            }
        }
    }
}

/// Compute MD5 hash of values in sqllogictest format.
/// Values are joined with newlines and a trailing newline.
fn md5_hash(values: &[String]) -> String {
    // Simple MD5 implementation for the harness.
    // We compute it by joining values with \n and hashing.
    let input = values.iter().map(|v| format!("{v}\n")).collect::<String>();
    md5_digest(input.as_bytes())
}

/// Minimal MD5 implementation (RFC 1321) — just enough for the harness.
fn md5_digest(data: &[u8]) -> String {
    let s = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5,
        9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10,
        15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21u32,
    ];
    let k: [u32; 64] = [
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613,
        0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193,
        0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d,
        0x02441453, 0xd8a1e681, 0xe7d3fbc8, 0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
        0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122,
        0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
        0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665, 0xf4292244,
        0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb,
        0xeb86d391,
    ];

    let mut a0: u32 = 0x67452301;
    let mut b0: u32 = 0xefcdab89;
    let mut c0: u32 = 0x98badcfe;
    let mut d0: u32 = 0x10325476;

    // Pre-processing: adding padding bits
    let orig_len_bits = (data.len() as u64) * 8;
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&orig_len_bits.to_le_bytes());

    // Process each 512-bit chunk
    for chunk in msg.chunks(64) {
        let mut m = [0u32; 16];
        for (j, word) in m.iter_mut().enumerate() {
            *word = u32::from_le_bytes([
                chunk[j * 4],
                chunk[j * 4 + 1],
                chunk[j * 4 + 2],
                chunk[j * 4 + 3],
            ]);
        }

        let (mut a, mut b, mut c, mut d) = (a0, b0, c0, d0);

        for i in 0..64 {
            let (f, g) = match i {
                0..=15 => ((b & c) | ((!b) & d), i),
                16..=31 => ((d & b) | ((!d) & c), (5 * i + 1) % 16),
                32..=47 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | (!d)), (7 * i) % 16),
            };

            let f = f.wrapping_add(a).wrapping_add(k[i]).wrapping_add(m[g]);
            a = d;
            d = c;
            c = b;
            b = b.wrapping_add(f.rotate_left(s[i]));
        }

        a0 = a0.wrapping_add(a);
        b0 = b0.wrapping_add(b);
        c0 = c0.wrapping_add(c);
        d0 = d0.wrapping_add(d);
    }

    format!(
        "{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        a0 as u8, (a0 >> 8) as u8, (a0 >> 16) as u8, (a0 >> 24) as u8,
        b0 as u8, (b0 >> 8) as u8, (b0 >> 16) as u8, (b0 >> 24) as u8,
        c0 as u8, (c0 >> 8) as u8, (c0 >> 16) as u8, (c0 >> 24) as u8,
        d0 as u8, (d0 >> 8) as u8, (d0 >> 16) as u8, (d0 >> 24) as u8,
    )
}

/// Run a single test file and return (passed, failed, errors).
fn run_test_file(path: &Path) -> (usize, usize, Vec<String>) {
    let content = std::fs::read_to_string(path).unwrap();
    let records = parse_test_file(&content);

    let mut db = Database::in_memory();
    let mut passed = 0;
    let mut failed = 0;
    let mut errors = Vec::new();

    for record in &records {
        match record {
            Record::Halt => break,
            Record::Statement {
                expected_error,
                sql,
            } => {
                let result = db.execute(sql);
                match (result.is_ok(), expected_error) {
                    (true, false) | (false, true) => passed += 1,
                    (true, true) => {
                        failed += 1;
                        errors.push(format!("Expected error but got ok: {sql}"));
                    }
                    (false, false) => {
                        failed += 1;
                        errors.push(format!(
                            "Expected ok but got error: {sql}\n  error: {}",
                            result.unwrap_err()
                        ));
                    }
                }
            }
            Record::Query {
                sort_mode,
                sql,
                expected,
                ..
            } => {
                match db.execute(sql) {
                    Ok(result) => {
                        // Flatten results: one value per line, row by row.
                        let mut actual: Vec<String> = Vec::new();
                        for row in &result.rows {
                            for val in &row.values {
                                actual.push(format_value(val));
                            }
                        }

                        match expected {
                            ExpectedResult::None => {
                                // No expected result — just check it ran ok.
                                passed += 1;
                            }
                            ExpectedResult::Values(expected_values) => {
                                let mut sorted_actual = actual.clone();
                                let mut sorted_expected = expected_values.clone();

                                // Apply sorting.
                                match sort_mode {
                                    SortMode::RowSort => {
                                        let ncols = result.columns.len().max(1);
                                        let mut actual_rows: Vec<Vec<String>> = sorted_actual
                                            .chunks(ncols)
                                            .map(|c| c.to_vec())
                                            .collect();
                                        actual_rows.sort();
                                        sorted_actual = actual_rows.into_iter().flatten().collect();

                                        let mut expected_rows: Vec<Vec<String>> = sorted_expected
                                            .chunks(ncols)
                                            .map(|c| c.to_vec())
                                            .collect();
                                        expected_rows.sort();
                                        sorted_expected =
                                            expected_rows.into_iter().flatten().collect();
                                    }
                                    SortMode::ValueSort => {
                                        sorted_actual.sort();
                                        sorted_expected.sort();
                                    }
                                    SortMode::NoSort => {}
                                }

                                if sorted_actual == sorted_expected {
                                    passed += 1;
                                } else {
                                    failed += 1;
                                    let max_show = 10;
                                    let exp_show: Vec<&str> = sorted_expected
                                        .iter()
                                        .take(max_show)
                                        .map(|s| s.as_str())
                                        .collect();
                                    let act_show: Vec<&str> = sorted_actual
                                        .iter()
                                        .take(max_show)
                                        .map(|s| s.as_str())
                                        .collect();
                                    errors.push(format!(
                                        "Query mismatch: {sql}\n  expected ({} values): {:?}{}\n  actual   ({} values): {:?}{}",
                                        sorted_expected.len(),
                                        exp_show,
                                        if sorted_expected.len() > max_show { "..." } else { "" },
                                        sorted_actual.len(),
                                        act_show,
                                        if sorted_actual.len() > max_show { "..." } else { "" },
                                    ));
                                }
                            }
                            ExpectedResult::Hash { count, md5 } => {
                                // Sort values according to sort mode before hashing.
                                let mut to_hash = actual.clone();
                                match sort_mode {
                                    SortMode::RowSort => {
                                        let ncols = result.columns.len().max(1);
                                        let mut rows: Vec<Vec<String>> =
                                            to_hash.chunks(ncols).map(|c| c.to_vec()).collect();
                                        rows.sort();
                                        to_hash = rows.into_iter().flatten().collect();
                                    }
                                    SortMode::ValueSort => {
                                        to_hash.sort();
                                    }
                                    SortMode::NoSort => {}
                                }

                                let actual_hash = md5_hash(&to_hash);
                                if to_hash.len() == *count && actual_hash == *md5 {
                                    passed += 1;
                                } else {
                                    failed += 1;
                                    if to_hash.len() != *count {
                                        errors.push(format!(
                                            "Query count mismatch: {sql}\n  expected {count} values, got {}",
                                            to_hash.len()
                                        ));
                                    } else {
                                        errors.push(format!(
                                            "Query hash mismatch: {sql}\n  expected hash {md5}\n  actual   hash {actual_hash}"
                                        ));
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        failed += 1;
                        errors.push(format!("Query error: {sql}\n  error: {e}"));
                    }
                }
            }
        }
    }

    (passed, failed, errors)
}

#[test]
fn test_sqllogictest_basic() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/sqllogictest/basic.test");
    let (passed, failed, errors) = run_test_file(&path);

    if !errors.is_empty() {
        eprintln!("\n--- sqllogictest failures ---");
        for e in &errors {
            eprintln!("{e}\n");
        }
    }

    assert_eq!(
        failed,
        0,
        "{failed} tests failed out of {}",
        passed + failed
    );
    assert!(passed > 0, "no tests ran");
    eprintln!("sqllogictest basic: {passed} passed, {failed} failed");
}

/// Run all .test files in the sqllogictest directory and print a summary.
/// Individual file failures don't fail the overall test — this is a progress tracker.
#[test]
fn test_sqllogictest_suite_summary() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/sqllogictest");
    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "test")
                .unwrap_or(false)
        })
        .collect();
    entries.sort_by_key(|e| e.path());

    let mut total_passed = 0;
    let mut total_failed = 0;

    eprintln!("\n=== sqllogictest suite ===");
    for entry in &entries {
        let path = entry.path();
        let name = path.file_name().unwrap().to_string_lossy();
        let (passed, failed, _errors) = run_test_file(&path);
        let pct = if passed + failed > 0 {
            100.0 * passed as f64 / (passed + failed) as f64
        } else {
            0.0
        };
        eprintln!("  {name}: {passed} passed, {failed} failed ({pct:.0}%)");
        total_passed += passed;
        total_failed += failed;
    }

    let total = total_passed + total_failed;
    let pct = if total > 0 {
        100.0 * total_passed as f64 / total as f64
    } else {
        0.0
    };
    eprintln!("=== total: {total_passed}/{total} passed ({pct:.1}%) ===\n");
}

#[test]
fn test_md5_known_value() {
    // Verify our MD5 implementation against a known value.
    assert_eq!(md5_digest(b""), "d41d8cd98f00b204e9800998ecf8427e");
    assert_eq!(md5_digest(b"abc"), "900150983cd24fb0d6963f7d28e17f72");
}
