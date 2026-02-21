use rsqlite::planner::Database;
use rsqlite::types::Value;
use rsqlite::vm::QueryResult;
use std::io::{self, BufRead, Write};

/// Output display mode for query results.
#[derive(Debug, Clone, Copy, PartialEq)]
enum OutputMode {
    Column,
    Csv,
    Line,
}

/// REPL configuration state.
struct ReplState {
    headers: bool,
    mode: OutputMode,
}

impl Default for ReplState {
    fn default() -> Self {
        Self {
            headers: true,
            mode: OutputMode::Column,
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut db = if args.len() > 1 {
        let path = &args[1];
        match Database::open(path) {
            Ok(db) => {
                eprintln!("Connected to {path}");
                db
            }
            Err(e) => {
                eprintln!("Error: unable to open database \"{path}\": {e}");
                std::process::exit(1);
            }
        }
    } else {
        eprintln!("Connected to a transient in-memory database.");
        eprintln!("Use \".open FILENAME\" to reopen on a persistent database.");
        Database::in_memory()
    };

    let mut state = ReplState::default();

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut reader = stdin.lock();
    let mut input_buf = String::new();
    let is_tty = atty_detect();

    loop {
        // Print prompt.
        if is_tty {
            let prompt = if input_buf.is_empty() {
                "rsqlite> "
            } else {
                "   ...> "
            };
            let mut out = stdout.lock();
            let _ = out.write_all(prompt.as_bytes());
            let _ = out.flush();
        }

        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => {
                // EOF
                if is_tty {
                    println!();
                }
                break;
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("Error reading input: {e}");
                break;
            }
        }

        // If we have no accumulated input, check for dot-commands.
        if input_buf.is_empty() {
            let trimmed = line.trim();
            if trimmed.starts_with('.') {
                handle_dot_command(trimmed, &mut db, &mut state);
                continue;
            }
        }

        input_buf.push_str(&line);

        // Check if the accumulated input contains a complete SQL statement
        // (terminated by a semicolon outside of quotes).
        if !has_complete_statement(&input_buf) {
            continue;
        }

        // Extract and execute all complete statements from the buffer.
        let sql = input_buf.trim().to_string();
        input_buf.clear();

        if sql.is_empty() {
            continue;
        }

        // Split on semicolons at the top level and execute each statement.
        for stmt_sql in split_statements(&sql) {
            let trimmed = stmt_sql.trim();
            if trimmed.is_empty() {
                continue;
            }
            match db.execute(trimmed) {
                Ok(result) => {
                    print_result(&result, &state);
                }
                Err(e) => {
                    eprintln!("Error: {e}");
                }
            }
        }
    }
}

/// Handle a dot-command.
fn handle_dot_command(input: &str, db: &mut Database, state: &mut ReplState) {
    let parts: Vec<&str> = input.splitn(2, char::is_whitespace).collect();
    let cmd = parts[0].to_lowercase();
    let arg = parts.get(1).map(|s| s.trim()).unwrap_or("");

    match cmd.as_str() {
        ".quit" | ".exit" => {
            std::process::exit(0);
        }
        ".tables" => {
            match db.table_names() {
                Ok(names) => {
                    if !names.is_empty() {
                        println!("{}", names.join("  "));
                    }
                }
                Err(e) => eprintln!("Error: {e}"),
            }
        }
        ".schema" => {
            match db.schema() {
                Ok(entries) => {
                    for entry in &entries {
                        if !arg.is_empty()
                            && !entry.name.eq_ignore_ascii_case(arg)
                        {
                            continue;
                        }
                        if let Some(ref sql) = entry.sql {
                            println!("{sql};");
                        }
                    }
                }
                Err(e) => eprintln!("Error: {e}"),
            }
        }
        ".headers" => match arg.to_lowercase().as_str() {
            "on" => state.headers = true,
            "off" => state.headers = false,
            _ => eprintln!("Usage: .headers on|off"),
        },
        ".mode" => match arg.to_lowercase().as_str() {
            "column" => state.mode = OutputMode::Column,
            "csv" => state.mode = OutputMode::Csv,
            "line" => state.mode = OutputMode::Line,
            _ => eprintln!("Usage: .mode column|csv|line"),
        },
        ".help" => {
            println!(".exit                  Exit this program");
            println!(".headers on|off        Turn display of headers on or off");
            println!(".help                  Show this help");
            println!(".mode column|csv|line  Set output mode");
            println!(".quit                  Exit this program");
            println!(".schema ?TABLE?        Show the CREATE statements");
            println!(".tables                List names of tables");
        }
        ".open" => {
            if arg.is_empty() {
                eprintln!("Usage: .open FILENAME");
            } else {
                match Database::open(arg) {
                    Ok(new_db) => {
                        *db = new_db;
                        eprintln!("Connected to {arg}");
                    }
                    Err(e) => eprintln!("Error: unable to open database \"{arg}\": {e}"),
                }
            }
        }
        _ => {
            eprintln!("Error: unknown command or invalid arguments: \"{cmd}\". Enter \".help\" for help");
        }
    }
}

/// Print a query result according to the current output mode.
fn print_result(result: &QueryResult, state: &ReplState) {
    if result.columns.is_empty() && result.rows.is_empty() {
        return;
    }

    match state.mode {
        OutputMode::Column => print_column_mode(result, state.headers),
        OutputMode::Csv => print_csv_mode(result, state.headers),
        OutputMode::Line => print_line_mode(result),
    }
}

/// Format a Value for display. Returns the display string and whether the value
/// is an integer (for alignment purposes).
fn format_value(val: &Value) -> (String, bool) {
    match val {
        Value::Null => ("NULL".to_string(), false),
        Value::Integer(i) => (i.to_string(), true),
        Value::Real(f) => (format!("{f}"), true),
        Value::Text(s) => (s.clone(), false),
        Value::Blob(b) => (format!("{}", Value::Blob(b.clone())), false),
    }
}

/// Print results in column mode with aligned columns.
fn print_column_mode(result: &QueryResult, headers: bool) {
    let num_cols = result.columns.len();
    if num_cols == 0 {
        return;
    }

    // Compute the display strings and column widths.
    let mut col_widths: Vec<usize> = result.columns.iter().map(|c| c.name.len()).collect();
    let mut is_integer_col: Vec<bool> = vec![true; num_cols];

    let mut formatted_rows: Vec<Vec<(String, bool)>> = Vec::with_capacity(result.rows.len());

    for row in &result.rows {
        let mut formatted: Vec<(String, bool)> = Vec::with_capacity(num_cols);
        for (i, val) in row.values.iter().enumerate() {
            let (s, is_int) = format_value(val);
            if i < col_widths.len() {
                if s.len() > col_widths[i] {
                    col_widths[i] = s.len();
                }
                if !is_int {
                    is_integer_col[i] = false;
                }
            }
            formatted.push((s, is_int));
        }
        // Handle rows with fewer values than columns.
        while formatted.len() < num_cols {
            formatted.push(("NULL".to_string(), false));
        }
        formatted_rows.push(formatted);
    }

    // Ensure minimum column width.
    for w in &mut col_widths {
        if *w < 1 {
            *w = 1;
        }
    }

    // Print header.
    if headers {
        let header_parts: Vec<String> = result
            .columns
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let w = col_widths[i];
                format!("{:<width$}", c.name, width = w)
            })
            .collect();
        println!("{}", header_parts.join("  "));

        // Print separator.
        let sep_parts: Vec<String> = col_widths
            .iter()
            .map(|w| "-".repeat(*w))
            .collect();
        println!("{}", sep_parts.join("  "));
    }

    // Print rows.
    for formatted in &formatted_rows {
        let parts: Vec<String> = formatted
            .iter()
            .enumerate()
            .map(|(i, (s, is_int))| {
                let w = col_widths.get(i).copied().unwrap_or(1);
                if *is_int && is_integer_col[i] {
                    format!("{:>width$}", s, width = w)
                } else {
                    format!("{:<width$}", s, width = w)
                }
            })
            .collect();
        println!("{}", parts.join("  "));
    }
}

/// Print results in CSV mode.
fn print_csv_mode(result: &QueryResult, headers: bool) {
    if headers && !result.columns.is_empty() {
        let header_parts: Vec<String> = result
            .columns
            .iter()
            .map(|c| csv_escape(&c.name))
            .collect();
        println!("{}", header_parts.join(","));
    }

    for row in &result.rows {
        let parts: Vec<String> = row
            .values
            .iter()
            .map(|val| {
                let (s, _) = format_value(val);
                csv_escape(&s)
            })
            .collect();
        println!("{}", parts.join(","));
    }
}

/// Escape a string for CSV output: quote if it contains commas, quotes, or newlines.
fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') || s.contains('\r') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// Print results in line mode (each column on its own line).
fn print_line_mode(result: &QueryResult) {
    let max_name_len = result
        .columns
        .iter()
        .map(|c| c.name.len())
        .max()
        .unwrap_or(0);

    for (row_idx, row) in result.rows.iter().enumerate() {
        if row_idx > 0 {
            println!();
        }
        for (i, val) in row.values.iter().enumerate() {
            let name = result
                .columns
                .get(i)
                .map(|c| c.name.as_str())
                .unwrap_or("?");
            let (s, _) = format_value(val);
            println!("{:>width$} = {}", name, s, width = max_name_len);
        }
    }
}

/// Check whether the input buffer contains at least one complete SQL statement.
/// A statement is complete when we find a semicolon that is not inside a string literal.
fn has_complete_statement(input: &str) -> bool {
    let mut in_single_quote = false;
    let mut in_double_quote = false;

    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let ch = chars[i];
        if in_single_quote {
            if ch == '\'' {
                // Check for escaped quote ('').
                if i + 1 < chars.len() && chars[i + 1] == '\'' {
                    i += 2; // Skip both quotes.
                    continue;
                }
                in_single_quote = false;
            }
        } else if in_double_quote {
            if ch == '"' {
                if i + 1 < chars.len() && chars[i + 1] == '"' {
                    i += 2;
                    continue;
                }
                in_double_quote = false;
            }
        } else {
            match ch {
                '\'' => in_single_quote = true,
                '"' => in_double_quote = true,
                ';' => return true,
                _ => {}
            }
        }
        i += 1;
    }

    false
}

/// Split input into individual SQL statements on semicolons (respecting quotes).
fn split_statements(input: &str) -> Vec<String> {
    let mut statements = Vec::new();
    let mut current = String::new();
    let mut in_single_quote = false;
    let mut in_double_quote = false;

    for ch in input.chars() {
        if in_single_quote {
            current.push(ch);
            if ch == '\'' {
                in_single_quote = false;
            }
        } else if in_double_quote {
            current.push(ch);
            if ch == '"' {
                in_double_quote = false;
            }
        } else {
            match ch {
                '\'' => {
                    in_single_quote = true;
                    current.push(ch);
                }
                '"' => {
                    in_double_quote = true;
                    current.push(ch);
                }
                ';' => {
                    current.push(ch);
                    let trimmed = current.trim().to_string();
                    if !trimmed.is_empty() {
                        statements.push(trimmed);
                    }
                    current.clear();
                }
                _ => current.push(ch),
            }
        }
    }

    // If there is leftover non-empty content (no trailing semicolon), include it.
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        statements.push(trimmed);
    }

    statements
}

/// Detect if stdin is a TTY. We use a simple heuristic: check if stdin is a terminal
/// using libc on Unix. Falls back to true.
fn atty_detect() -> bool {
    #[cfg(unix)]
    {
        unsafe { libc_isatty(0) != 0 }
    }
    #[cfg(not(unix))]
    {
        true
    }
}

#[cfg(unix)]
extern "C" {
    fn isatty(fd: i32) -> i32;
}

#[cfg(unix)]
unsafe fn libc_isatty(fd: i32) -> i32 {
    unsafe { isatty(fd) }
}
