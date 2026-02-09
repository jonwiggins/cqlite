use std::env;
use std::io::{self, BufRead, Write};

use rsqlite::{Database, Value};

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut db = if args.len() > 1 {
        match Database::open(&args[1]) {
            Ok(db) => db,
            Err(e) => {
                eprintln!("Error opening database: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        Database::new_memory()
    };

    let mut headers_on = false;
    let mut mode = OutputMode::Column;

    let is_tty = env::var("TERM").is_ok();

    if is_tty {
        println!("rsqlite version 0.1.0");
        println!("Enter \".help\" for usage hints.");
    }

    let stdin = io::stdin();
    let mut line_buffer = String::new();

    loop {
        if is_tty {
            if line_buffer.is_empty() {
                print!("rsqlite> ");
            } else {
                print!("   ...> ");
            }
            io::stdout().flush().ok();
        }

        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(_) => break,
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if line_buffer.is_empty() && trimmed.starts_with('.') {
            handle_dot_command(trimmed, &mut db, &mut headers_on, &mut mode);
            continue;
        }

        line_buffer.push_str(&line);

        let buf_trimmed = line_buffer.trim();
        if !buf_trimmed.ends_with(';') {
            continue;
        }

        let sql = line_buffer.trim().to_string();
        line_buffer.clear();

        match db.execute(&sql) {
            Ok(result) => {
                if !result.columns.is_empty() || !result.rows.is_empty() {
                    print_result(&result, headers_on, &mode);
                }
            }
            Err(e) => {
                eprintln!("Error: {}", e);
            }
        }
    }
}

#[derive(Clone)]
enum OutputMode {
    Column,
    Csv,
    Line,
    List,
    Tabs,
}

fn handle_dot_command(cmd: &str, db: &mut Database, headers_on: &mut bool, mode: &mut OutputMode) {
    let parts: Vec<&str> = cmd.splitn(2, char::is_whitespace).collect();
    let command = parts[0].to_lowercase();
    let arg = parts.get(1).map(|s| s.trim()).unwrap_or("");

    match command.as_str() {
        ".quit" | ".exit" => std::process::exit(0),
        ".help" => {
            println!(".headers on|off     Turn display of headers on or off");
            println!(".help               Show this message");
            println!(".mode MODE          Set output mode (column, csv, line, list, tabs)");
            println!(".open FILENAME      Open a database file");
            println!(".quit               Exit this program");
            println!(".read FILENAME      Execute SQL from a file");
            println!(".schema ?TABLE?     Show CREATE statements");
            println!(".tables             List names of tables");
        }
        ".tables" => {
            let mut names: Vec<String> = db.tables.keys().cloned().collect();
            names.sort();
            for name in &names {
                print!("{}  ", name);
            }
            if !names.is_empty() {
                println!();
            }
        }
        ".schema" => {
            if arg.is_empty() {
                let mut schemas: Vec<String> = db
                    .tables
                    .values()
                    .map(|t| t.sql.clone())
                    .filter(|s| !s.is_empty())
                    .collect();
                schemas.sort();
                for sql in schemas {
                    println!("{};", sql);
                }
                for idx in db.indexes.values() {
                    if !idx.sql.is_empty() {
                        println!("{};", idx.sql);
                    }
                }
            } else {
                let table_name = arg.to_lowercase();
                if let Some(schema) = db.tables.get(&table_name) {
                    if !schema.sql.is_empty() {
                        println!("{};", schema.sql);
                    }
                }
                for idx in db.indexes.values() {
                    if idx.table_name.to_lowercase() == table_name && !idx.sql.is_empty() {
                        println!("{};", idx.sql);
                    }
                }
            }
        }
        ".headers" => {
            *headers_on = arg.eq_ignore_ascii_case("on");
        }
        ".mode" => {
            *mode = match arg.to_lowercase().as_str() {
                "column" => OutputMode::Column,
                "csv" => OutputMode::Csv,
                "line" => OutputMode::Line,
                "list" => OutputMode::List,
                "tabs" => OutputMode::Tabs,
                _ => {
                    eprintln!("Unknown mode: {}", arg);
                    return;
                }
            };
        }
        ".open" => {
            if arg.is_empty() {
                eprintln!("Usage: .open FILENAME");
                return;
            }
            match Database::open(arg) {
                Ok(new_db) => *db = new_db,
                Err(e) => eprintln!("Error: {}", e),
            }
        }
        ".read" => {
            if arg.is_empty() {
                eprintln!("Usage: .read FILENAME");
                return;
            }
            match std::fs::read_to_string(arg) {
                Ok(contents) => {
                    for stmt_str in contents.split(';') {
                        let sql = stmt_str.trim();
                        if sql.is_empty() {
                            continue;
                        }
                        let sql_with_semi = format!("{};", sql);
                        match db.execute(&sql_with_semi) {
                            Ok(result) => {
                                if !result.columns.is_empty() || !result.rows.is_empty() {
                                    print_result(&result, *headers_on, mode);
                                }
                            }
                            Err(e) => eprintln!("Error: {}", e),
                        }
                    }
                }
                Err(e) => eprintln!("Error reading file: {}", e),
            }
        }
        _ => {
            eprintln!("Unknown command: {}", command);
        }
    }
}

fn print_result(result: &rsqlite::ExecuteResult, headers: bool, mode: &OutputMode) {
    match mode {
        OutputMode::Column => print_column(result, headers),
        OutputMode::Csv => print_csv(result, headers),
        OutputMode::Line => print_line(result),
        OutputMode::List => print_list(result, headers),
        OutputMode::Tabs => print_tabs(result, headers),
    }
}

fn print_column(result: &rsqlite::ExecuteResult, headers: bool) {
    if result.rows.is_empty() && result.columns.is_empty() {
        return;
    }

    let num_cols = result
        .columns
        .len()
        .max(result.rows.first().map_or(0, |r| r.len()));

    let mut widths = vec![1usize; num_cols];
    for (i, col) in result.columns.iter().enumerate() {
        widths[i] = widths[i].max(col.len());
    }
    for row in &result.rows {
        for (i, val) in row.iter().enumerate() {
            if i < widths.len() {
                widths[i] = widths[i].max(format_value(val).len());
            }
        }
    }

    if headers && !result.columns.is_empty() {
        for (i, col) in result.columns.iter().enumerate() {
            if i > 0 {
                print!("  ");
            }
            print!("{:<width$}", col, width = widths[i]);
        }
        println!();
        for (i, w) in widths.iter().enumerate() {
            if i > 0 {
                print!("  ");
            }
            print!("{}", "-".repeat(*w));
        }
        println!();
    }

    for row in &result.rows {
        for (i, val) in row.iter().enumerate() {
            if i > 0 {
                print!("  ");
            }
            let w = widths.get(i).copied().unwrap_or(1);
            print!("{:<width$}", format_value(val), width = w);
        }
        println!();
    }
}

fn print_csv(result: &rsqlite::ExecuteResult, headers: bool) {
    if headers {
        println!("{}", result.columns.join(","));
    }
    for row in &result.rows {
        let vals: Vec<String> = row.iter().map(|v| csv_escape(&format_value(v))).collect();
        println!("{}", vals.join(","));
    }
}

fn print_list(result: &rsqlite::ExecuteResult, headers: bool) {
    if headers {
        println!("{}", result.columns.join("|"));
    }
    for row in &result.rows {
        let vals: Vec<String> = row.iter().map(format_value).collect();
        println!("{}", vals.join("|"));
    }
}

fn print_tabs(result: &rsqlite::ExecuteResult, headers: bool) {
    if headers {
        println!("{}", result.columns.join("\t"));
    }
    for row in &result.rows {
        let vals: Vec<String> = row.iter().map(format_value).collect();
        println!("{}", vals.join("\t"));
    }
}

fn print_line(result: &rsqlite::ExecuteResult) {
    for row in &result.rows {
        for (i, val) in row.iter().enumerate() {
            let col_name = result.columns.get(i).map(|s| s.as_str()).unwrap_or("?");
            println!("{} = {}", col_name, format_value(val));
        }
        println!();
    }
}

fn format_value(val: &Value) -> String {
    match val {
        Value::Null => String::new(),
        Value::Integer(i) => i.to_string(),
        Value::Real(f) => {
            if *f == f.trunc() && f.abs() < 1e15 {
                format!("{:.1}", f)
            } else {
                format!("{}", f)
            }
        }
        Value::Text(s) => s.clone(),
        Value::Blob(b) => {
            let hex: String = b.iter().map(|byte| format!("{:02X}", byte)).collect();
            format!("X'{}'", hex)
        }
    }
}

fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}
