// Hand-written SQL tokenizer/lexer.

use crate::error::{Result, RsqliteError};

// ---------------------------------------------------------------------------
// Token
// ---------------------------------------------------------------------------

/// A single SQL token produced by the lexer.
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // ---- Keywords (alphabetical) ----
    Abort,
    Add,
    All,
    Alter,
    And,
    As,
    Asc,
    Autoincrement,
    Begin,
    Between,
    BlobKw,
    By,
    Case,
    Cast,
    Check,
    Collate,
    Column,
    Commit,
    Conflict,
    Constraint,
    Create,
    Cross,
    CurrentDate,
    CurrentTime,
    CurrentTimestamp,
    Default,
    Delete,
    Desc,
    Distinct,
    Drop,
    Else,
    End,
    Escape,
    Except,
    Exists,
    Explain,
    Fail,
    Foreign,
    From,
    Glob,
    Group,
    Having,
    If,
    Ignore,
    In,
    Index,
    Inner,
    Insert,
    IntegerKw,
    Intersect,
    Into,
    Is,
    Join,
    Key,
    Left,
    Like,
    Limit,
    Not,
    Null,
    NumericKw,
    Offset,
    On,
    Or,
    Order,
    Outer,
    Plan,
    Pragma,
    Primary,
    Query,
    RealKw,
    Recursive,
    References,
    Release,
    Rename,
    Replace,
    Right,
    Rollback,
    Savepoint,
    Select,
    Set,
    Table,
    TextKw,
    Then,
    To,
    Transaction,
    Union,
    Unique,
    Update,
    Values,
    When,
    Where,
    With,

    // ---- Literals ----
    IntegerLiteral(i64),
    RealLiteral(f64),
    StringLiteral(String),
    BlobLiteral(Vec<u8>),

    // ---- Identifiers ----
    Ident(String),

    // ---- Operators ----
    Plus,       // +
    Minus,      // -
    Star,       // *
    Slash,      // /
    Percent,    // %
    Eq,         // =  or  ==
    NotEq,      // !=  or  <>
    Lt,         // <
    Gt,         // >
    Le,         // <=
    Ge,         // >=
    Concat,     // ||
    BitAnd,     // &
    BitOr,      // |  (single)
    BitNot,     // ~
    ShiftLeft,  // <<
    ShiftRight, // >>

    // ---- Punctuation ----
    LeftParen,    // (
    RightParen,   // )
    Comma,        // ,
    Semicolon,    // ;
    Dot,          // .
    QuestionMark, // ?

    // ---- Special ----
    Eof,
}

impl std::fmt::Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Token::Abort => write!(f, "ABORT"),
            Token::Add => write!(f, "ADD"),
            Token::All => write!(f, "ALL"),
            Token::Alter => write!(f, "ALTER"),
            Token::And => write!(f, "AND"),
            Token::As => write!(f, "AS"),
            Token::Asc => write!(f, "ASC"),
            Token::Autoincrement => write!(f, "AUTOINCREMENT"),
            Token::Begin => write!(f, "BEGIN"),
            Token::Between => write!(f, "BETWEEN"),
            Token::BlobKw => write!(f, "BLOB"),
            Token::By => write!(f, "BY"),
            Token::Case => write!(f, "CASE"),
            Token::Cast => write!(f, "CAST"),
            Token::Check => write!(f, "CHECK"),
            Token::Collate => write!(f, "COLLATE"),
            Token::Column => write!(f, "COLUMN"),
            Token::Commit => write!(f, "COMMIT"),
            Token::Conflict => write!(f, "CONFLICT"),
            Token::Constraint => write!(f, "CONSTRAINT"),
            Token::Create => write!(f, "CREATE"),
            Token::Cross => write!(f, "CROSS"),
            Token::CurrentDate => write!(f, "CURRENT_DATE"),
            Token::CurrentTime => write!(f, "CURRENT_TIME"),
            Token::CurrentTimestamp => write!(f, "CURRENT_TIMESTAMP"),
            Token::Default => write!(f, "DEFAULT"),
            Token::Delete => write!(f, "DELETE"),
            Token::Desc => write!(f, "DESC"),
            Token::Distinct => write!(f, "DISTINCT"),
            Token::Drop => write!(f, "DROP"),
            Token::Else => write!(f, "ELSE"),
            Token::End => write!(f, "END"),
            Token::Escape => write!(f, "ESCAPE"),
            Token::Except => write!(f, "EXCEPT"),
            Token::Exists => write!(f, "EXISTS"),
            Token::Explain => write!(f, "EXPLAIN"),
            Token::Fail => write!(f, "FAIL"),
            Token::Foreign => write!(f, "FOREIGN"),
            Token::From => write!(f, "FROM"),
            Token::Glob => write!(f, "GLOB"),
            Token::Group => write!(f, "GROUP"),
            Token::Having => write!(f, "HAVING"),
            Token::If => write!(f, "IF"),
            Token::Ignore => write!(f, "IGNORE"),
            Token::In => write!(f, "IN"),
            Token::Index => write!(f, "INDEX"),
            Token::Inner => write!(f, "INNER"),
            Token::Insert => write!(f, "INSERT"),
            Token::IntegerKw => write!(f, "INTEGER"),
            Token::Intersect => write!(f, "INTERSECT"),
            Token::Into => write!(f, "INTO"),
            Token::Is => write!(f, "IS"),
            Token::Join => write!(f, "JOIN"),
            Token::Key => write!(f, "KEY"),
            Token::Left => write!(f, "LEFT"),
            Token::Like => write!(f, "LIKE"),
            Token::Limit => write!(f, "LIMIT"),
            Token::Not => write!(f, "NOT"),
            Token::Null => write!(f, "NULL"),
            Token::NumericKw => write!(f, "NUMERIC"),
            Token::Offset => write!(f, "OFFSET"),
            Token::On => write!(f, "ON"),
            Token::Or => write!(f, "OR"),
            Token::Order => write!(f, "ORDER"),
            Token::Outer => write!(f, "OUTER"),
            Token::Plan => write!(f, "PLAN"),
            Token::Pragma => write!(f, "PRAGMA"),
            Token::Primary => write!(f, "PRIMARY"),
            Token::Query => write!(f, "QUERY"),
            Token::RealKw => write!(f, "REAL"),
            Token::Recursive => write!(f, "RECURSIVE"),
            Token::References => write!(f, "REFERENCES"),
            Token::Release => write!(f, "RELEASE"),
            Token::Rename => write!(f, "RENAME"),
            Token::Replace => write!(f, "REPLACE"),
            Token::Right => write!(f, "RIGHT"),
            Token::Rollback => write!(f, "ROLLBACK"),
            Token::Savepoint => write!(f, "SAVEPOINT"),
            Token::Select => write!(f, "SELECT"),
            Token::Set => write!(f, "SET"),
            Token::Table => write!(f, "TABLE"),
            Token::TextKw => write!(f, "TEXT"),
            Token::Then => write!(f, "THEN"),
            Token::To => write!(f, "TO"),
            Token::Transaction => write!(f, "TRANSACTION"),
            Token::Union => write!(f, "UNION"),
            Token::Unique => write!(f, "UNIQUE"),
            Token::Update => write!(f, "UPDATE"),
            Token::Values => write!(f, "VALUES"),
            Token::When => write!(f, "WHEN"),
            Token::Where => write!(f, "WHERE"),
            Token::With => write!(f, "WITH"),
            Token::IntegerLiteral(v) => write!(f, "{v}"),
            Token::RealLiteral(v) => write!(f, "{v}"),
            Token::StringLiteral(v) => write!(f, "'{v}'"),
            Token::BlobLiteral(_) => write!(f, "<blob>"),
            Token::Ident(v) => write!(f, "{v}"),
            Token::Plus => write!(f, "+"),
            Token::Minus => write!(f, "-"),
            Token::Star => write!(f, "*"),
            Token::Slash => write!(f, "/"),
            Token::Percent => write!(f, "%"),
            Token::Eq => write!(f, "="),
            Token::NotEq => write!(f, "!="),
            Token::Lt => write!(f, "<"),
            Token::Gt => write!(f, ">"),
            Token::Le => write!(f, "<="),
            Token::Ge => write!(f, ">="),
            Token::Concat => write!(f, "||"),
            Token::BitAnd => write!(f, "&"),
            Token::BitOr => write!(f, "|"),
            Token::BitNot => write!(f, "~"),
            Token::ShiftLeft => write!(f, "<<"),
            Token::ShiftRight => write!(f, ">>"),
            Token::LeftParen => write!(f, "("),
            Token::RightParen => write!(f, ")"),
            Token::Comma => write!(f, ","),
            Token::Semicolon => write!(f, ";"),
            Token::Dot => write!(f, "."),
            Token::QuestionMark => write!(f, "?"),
            Token::Eof => write!(f, "<EOF>"),
        }
    }
}

// ---------------------------------------------------------------------------
// Keyword lookup
// ---------------------------------------------------------------------------

/// Map an identifier (already uppercased) to a keyword token, if it matches.
fn lookup_keyword(upper: &str) -> Option<Token> {
    match upper {
        "ABORT" => Some(Token::Abort),
        "ADD" => Some(Token::Add),
        "ALL" => Some(Token::All),
        "ALTER" => Some(Token::Alter),
        "AND" => Some(Token::And),
        "AS" => Some(Token::As),
        "ASC" => Some(Token::Asc),
        "AUTOINCREMENT" => Some(Token::Autoincrement),
        "BEGIN" => Some(Token::Begin),
        "BETWEEN" => Some(Token::Between),
        "BLOB" => Some(Token::BlobKw),
        "BY" => Some(Token::By),
        "CASE" => Some(Token::Case),
        "CAST" => Some(Token::Cast),
        "CHECK" => Some(Token::Check),
        "COLLATE" => Some(Token::Collate),
        "COLUMN" => Some(Token::Column),
        "COMMIT" => Some(Token::Commit),
        "CONFLICT" => Some(Token::Conflict),
        "CONSTRAINT" => Some(Token::Constraint),
        "CREATE" => Some(Token::Create),
        "CROSS" => Some(Token::Cross),
        "CURRENT_DATE" => Some(Token::CurrentDate),
        "CURRENT_TIME" => Some(Token::CurrentTime),
        "CURRENT_TIMESTAMP" => Some(Token::CurrentTimestamp),
        "DEFAULT" => Some(Token::Default),
        "DELETE" => Some(Token::Delete),
        "DESC" => Some(Token::Desc),
        "DISTINCT" => Some(Token::Distinct),
        "DROP" => Some(Token::Drop),
        "ELSE" => Some(Token::Else),
        "END" => Some(Token::End),
        "ESCAPE" => Some(Token::Escape),
        "EXCEPT" => Some(Token::Except),
        "EXISTS" => Some(Token::Exists),
        "EXPLAIN" => Some(Token::Explain),
        "FAIL" => Some(Token::Fail),
        "FOREIGN" => Some(Token::Foreign),
        "FROM" => Some(Token::From),
        "GLOB" => Some(Token::Glob),
        "GROUP" => Some(Token::Group),
        "HAVING" => Some(Token::Having),
        "IF" => Some(Token::If),
        "IGNORE" => Some(Token::Ignore),
        "IN" => Some(Token::In),
        "INDEX" => Some(Token::Index),
        "INNER" => Some(Token::Inner),
        "INSERT" => Some(Token::Insert),
        "INTEGER" => Some(Token::IntegerKw),
        "INTERSECT" => Some(Token::Intersect),
        "INTO" => Some(Token::Into),
        "IS" => Some(Token::Is),
        "JOIN" => Some(Token::Join),
        "KEY" => Some(Token::Key),
        "LEFT" => Some(Token::Left),
        "LIKE" => Some(Token::Like),
        "LIMIT" => Some(Token::Limit),
        "NOT" => Some(Token::Not),
        "NULL" => Some(Token::Null),
        "NUMERIC" => Some(Token::NumericKw),
        "OFFSET" => Some(Token::Offset),
        "ON" => Some(Token::On),
        "OR" => Some(Token::Or),
        "ORDER" => Some(Token::Order),
        "OUTER" => Some(Token::Outer),
        "PLAN" => Some(Token::Plan),
        "PRAGMA" => Some(Token::Pragma),
        "PRIMARY" => Some(Token::Primary),
        "QUERY" => Some(Token::Query),
        "REAL" => Some(Token::RealKw),
        "RECURSIVE" => Some(Token::Recursive),
        "REFERENCES" => Some(Token::References),
        "RELEASE" => Some(Token::Release),
        "RENAME" => Some(Token::Rename),
        "REPLACE" => Some(Token::Replace),
        "RIGHT" => Some(Token::Right),
        "ROLLBACK" => Some(Token::Rollback),
        "SAVEPOINT" => Some(Token::Savepoint),
        "SELECT" => Some(Token::Select),
        "SET" => Some(Token::Set),
        "TABLE" => Some(Token::Table),
        "TEXT" => Some(Token::TextKw),
        "THEN" => Some(Token::Then),
        "TO" => Some(Token::To),
        "TRANSACTION" => Some(Token::Transaction),
        "UNION" => Some(Token::Union),
        "UNIQUE" => Some(Token::Unique),
        "UPDATE" => Some(Token::Update),
        "VALUES" => Some(Token::Values),
        "WHEN" => Some(Token::When),
        "WHERE" => Some(Token::Where),
        "WITH" => Some(Token::With),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

/// A hand-written SQL lexer that produces a stream of [`Token`]s.
///
/// Create one via [`Tokenizer::new`] and repeatedly call [`Tokenizer::next_token`]
/// until you receive [`Token::Eof`].
pub struct Tokenizer<'a> {
    /// The full input source (UTF-8).
    input: &'a str,
    /// Byte-offset cursor into `input`. Always points at the start of the next
    /// unconsumed character (or `input.len()` when exhausted).
    pos: usize,
}

impl<'a> Tokenizer<'a> {
    /// Create a new tokenizer for the given SQL input string.
    pub fn new(input: &'a str) -> Self {
        Tokenizer { input, pos: 0 }
    }

    // -- helpers ----------------------------------------------------------

    /// Peek at the current byte without consuming it. Returns `None` at EOF.
    fn peek(&self) -> Option<u8> {
        self.input.as_bytes().get(self.pos).copied()
    }

    /// Peek at the byte `offset` positions ahead of the cursor.
    fn peek_at(&self, offset: usize) -> Option<u8> {
        self.input.as_bytes().get(self.pos + offset).copied()
    }

    /// Advance the cursor by one byte and return it.
    fn advance(&mut self) -> Option<u8> {
        let b = self.peek()?;
        self.pos += 1;
        Some(b)
    }

    /// Advance the cursor by `n` bytes.
    fn advance_by(&mut self, n: usize) {
        self.pos = (self.pos + n).min(self.input.len());
    }

    /// Return the remaining (unconsumed) slice of the input.
    fn remaining(&self) -> &'a str {
        &self.input[self.pos..]
    }

    /// Produce a parse-error [`Result`].
    fn error<T>(&self, msg: impl Into<String>) -> Result<T> {
        Err(RsqliteError::Parse(msg.into()))
    }

    // -- whitespace & comments -------------------------------------------

    /// Skip over whitespace and comments. Returns `true` if any were skipped.
    fn skip_whitespace_and_comments(&mut self) -> Result<()> {
        loop {
            // Skip whitespace bytes
            while let Some(b) = self.peek() {
                if b.is_ascii_whitespace() {
                    self.advance();
                } else {
                    break;
                }
            }

            // Try to skip a line comment: -- ...
            if self.peek() == Some(b'-') && self.peek_at(1) == Some(b'-') {
                self.advance_by(2);
                while let Some(b) = self.peek() {
                    self.advance();
                    if b == b'\n' {
                        break;
                    }
                }
                continue; // re-check for more whitespace / comments
            }

            // Try to skip a block comment: /* ... */
            if self.peek() == Some(b'/') && self.peek_at(1) == Some(b'*') {
                self.advance_by(2);
                let mut depth = 1u32;
                while depth > 0 {
                    match self.advance() {
                        None => {
                            return self.error("unterminated block comment");
                        }
                        Some(b'/') if self.peek() == Some(b'*') => {
                            self.advance();
                            depth += 1;
                        }
                        Some(b'*') if self.peek() == Some(b'/') => {
                            self.advance();
                            depth -= 1;
                        }
                        _ => {}
                    }
                }
                continue;
            }

            // Nothing more to skip.
            return Ok(());
        }
    }

    // -- number literals --------------------------------------------------

    /// Read a numeric literal (integer, real, or hex).
    fn read_number(&mut self) -> Result<Token> {
        let start = self.pos;

        // Hex literal: 0x...
        if self.peek() == Some(b'0') && matches!(self.peek_at(1), Some(b'x') | Some(b'X')) {
            self.advance_by(2); // consume 0x
            let hex_start = self.pos;
            while let Some(b) = self.peek() {
                if b.is_ascii_hexdigit() {
                    self.advance();
                } else {
                    break;
                }
            }
            if self.pos == hex_start {
                return self.error("expected hex digits after 0x");
            }
            let hex_str = &self.input[hex_start..self.pos];
            let value =
                i64::from_str_radix(hex_str, 16).map_err(|e| RsqliteError::Parse(e.to_string()))?;
            return Ok(Token::IntegerLiteral(value));
        }

        // Consume leading digits
        while let Some(b) = self.peek() {
            if b.is_ascii_digit() {
                self.advance();
            } else {
                break;
            }
        }

        let mut is_real = false;

        // Decimal point (only if followed by a digit — otherwise `.` is a
        // separate token, e.g. `t.col`).
        if self.peek() == Some(b'.') && matches!(self.peek_at(1), Some(b'0'..=b'9')) {
            is_real = true;
            self.advance(); // consume '.'
            while let Some(b) = self.peek() {
                if b.is_ascii_digit() {
                    self.advance();
                } else {
                    break;
                }
            }
        }

        // Exponent part
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            is_real = true;
            self.advance(); // consume 'e'/'E'
            if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                self.advance();
            }
            let exp_start = self.pos;
            while let Some(b) = self.peek() {
                if b.is_ascii_digit() {
                    self.advance();
                } else {
                    break;
                }
            }
            if self.pos == exp_start {
                return self.error("expected digits in exponent");
            }
        }

        let text = &self.input[start..self.pos];

        if is_real {
            let value: f64 = text
                .parse()
                .map_err(|e: std::num::ParseFloatError| RsqliteError::Parse(e.to_string()))?;
            Ok(Token::RealLiteral(value))
        } else {
            let value: i64 = text
                .parse()
                .map_err(|e: std::num::ParseIntError| RsqliteError::Parse(e.to_string()))?;
            Ok(Token::IntegerLiteral(value))
        }
    }

    // -- string / blob literals -------------------------------------------

    /// Read a single-quoted string literal, handling '' escapes.
    fn read_string_literal(&mut self) -> Result<Token> {
        debug_assert_eq!(self.peek(), Some(b'\''));
        self.advance(); // opening quote

        let mut value = String::new();
        loop {
            match self.advance() {
                None => {
                    return self.error("unterminated string literal");
                }
                Some(b'\'') => {
                    // '' is an escaped embedded quote
                    if self.peek() == Some(b'\'') {
                        self.advance();
                        value.push('\'');
                    } else {
                        return Ok(Token::StringLiteral(value));
                    }
                }
                Some(b) => {
                    // The input is UTF-8; we're iterating bytes. For non-ASCII
                    // we need to handle multi-byte sequences correctly. Since
                    // the input slice is known-good UTF-8 we can copy byte by
                    // byte into the String via char reconstruction after the
                    // loop, but the simplest correct approach is to track a
                    // start offset instead. However, because of '' escaping we
                    // already need a mutable buffer. For simplicity and
                    // correctness we push the byte (ASCII) directly; for
                    // multi-byte UTF-8 we fall through to pushing from the
                    // source slice.
                    if b.is_ascii() {
                        value.push(b as char);
                    } else {
                        // Back up one byte and decode the full char from the
                        // source.
                        self.pos -= 1;
                        let ch = self.remaining().chars().next().unwrap();
                        value.push(ch);
                        self.pos += ch.len_utf8();
                    }
                }
            }
        }
    }

    /// Read a blob literal: the opening `X` or `x` has already been peeked
    /// but NOT consumed. We expect `X'<hex pairs>'`.
    fn read_blob_literal(&mut self) -> Result<Token> {
        // Consume 'x' or 'X'
        self.advance();
        // Consume opening quote
        if self.peek() != Some(b'\'') {
            return self.error("expected ' after X in blob literal");
        }
        self.advance();

        let mut hex_str = String::new();
        loop {
            match self.advance() {
                None => {
                    return self.error("unterminated blob literal");
                }
                Some(b'\'') => {
                    break;
                }
                Some(b) if (b as char).is_ascii_hexdigit() => {
                    hex_str.push(b as char);
                }
                Some(b) => {
                    return self
                        .error(format!("invalid character '{}' in blob literal", b as char));
                }
            }
        }

        if !hex_str.len().is_multiple_of(2) {
            return self.error("blob literal must contain an even number of hex digits");
        }

        let bytes: Vec<u8> = (0..hex_str.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex_str[i..i + 2], 16).unwrap())
            .collect();

        Ok(Token::BlobLiteral(bytes))
    }

    // -- quoted identifiers -----------------------------------------------

    /// Read a double-quoted identifier: `"ident"`.
    fn read_double_quoted_ident(&mut self) -> Result<Token> {
        debug_assert_eq!(self.peek(), Some(b'"'));
        self.advance(); // opening "

        let mut value = String::new();
        loop {
            match self.advance() {
                None => {
                    return self.error("unterminated double-quoted identifier");
                }
                Some(b'"') => {
                    // "" is an escaped embedded double-quote
                    if self.peek() == Some(b'"') {
                        self.advance();
                        value.push('"');
                    } else {
                        return Ok(Token::Ident(value));
                    }
                }
                Some(b) => {
                    if b.is_ascii() {
                        value.push(b as char);
                    } else {
                        self.pos -= 1;
                        let ch = self.remaining().chars().next().unwrap();
                        value.push(ch);
                        self.pos += ch.len_utf8();
                    }
                }
            }
        }
    }

    /// Read a backtick-quoted identifier: `` `ident` ``.
    fn read_backtick_quoted_ident(&mut self) -> Result<Token> {
        debug_assert_eq!(self.peek(), Some(b'`'));
        self.advance(); // opening `

        let mut value = String::new();
        loop {
            match self.advance() {
                None => {
                    return self.error("unterminated backtick-quoted identifier");
                }
                Some(b'`') => {
                    // `` is an escaped embedded backtick
                    if self.peek() == Some(b'`') {
                        self.advance();
                        value.push('`');
                    } else {
                        return Ok(Token::Ident(value));
                    }
                }
                Some(b) => {
                    if b.is_ascii() {
                        value.push(b as char);
                    } else {
                        self.pos -= 1;
                        let ch = self.remaining().chars().next().unwrap();
                        value.push(ch);
                        self.pos += ch.len_utf8();
                    }
                }
            }
        }
    }

    /// Read a bracket-quoted identifier: `[ident]`.
    fn read_bracket_quoted_ident(&mut self) -> Result<Token> {
        debug_assert_eq!(self.peek(), Some(b'['));
        self.advance(); // opening [

        let mut value = String::new();
        loop {
            match self.advance() {
                None => {
                    return self.error("unterminated bracket-quoted identifier");
                }
                Some(b']') => {
                    return Ok(Token::Ident(value));
                }
                Some(b) => {
                    if b.is_ascii() {
                        value.push(b as char);
                    } else {
                        self.pos -= 1;
                        let ch = self.remaining().chars().next().unwrap();
                        value.push(ch);
                        self.pos += ch.len_utf8();
                    }
                }
            }
        }
    }

    // -- identifiers / keywords ------------------------------------------

    /// Read a bare (unquoted) identifier or keyword. The first character has
    /// already been validated as alphabetic or `_`.
    fn read_word(&mut self) -> Token {
        let start = self.pos;
        while let Some(b) = self.peek() {
            if b.is_ascii_alphanumeric() || b == b'_' {
                self.advance();
            } else {
                break;
            }
        }
        let word = &self.input[start..self.pos];
        let upper = word.to_ascii_uppercase();
        lookup_keyword(&upper).unwrap_or_else(|| Token::Ident(word.to_owned()))
    }

    // -- main entry point -------------------------------------------------

    /// Return the next token from the input.
    ///
    /// Returns [`Token::Eof`] when the input is exhausted. After `Eof` is
    /// returned, subsequent calls will continue to return `Eof`.
    pub fn next_token(&mut self) -> Result<Token> {
        self.skip_whitespace_and_comments()?;

        let b = match self.peek() {
            None => return Ok(Token::Eof),
            Some(b) => b,
        };

        match b {
            // ---- single-quoted string literal ----
            b'\'' => self.read_string_literal(),

            // ---- double-quoted identifier ----
            b'"' => self.read_double_quoted_ident(),

            // ---- backtick-quoted identifier ----
            b'`' => self.read_backtick_quoted_ident(),

            // ---- bracket-quoted identifier ----
            b'[' => self.read_bracket_quoted_ident(),

            // ---- blob literal or identifier starting with x/X ----
            b'x' | b'X' if self.peek_at(1) == Some(b'\'') => self.read_blob_literal(),

            // ---- numeric literals ----
            b'0'..=b'9' => self.read_number(),

            // Dot followed by digit => real literal (e.g. .5)
            b'.' if matches!(self.peek_at(1), Some(b'0'..=b'9')) => {
                // We don't consume the dot here; read_number handles the
                // integer part being empty by starting at the current pos.
                // Actually, read_number expects leading digits first. We need
                // to handle this case specially.
                let start = self.pos;
                self.advance(); // consume '.'
                while let Some(d) = self.peek() {
                    if d.is_ascii_digit() {
                        self.advance();
                    } else {
                        break;
                    }
                }
                // Exponent
                if matches!(self.peek(), Some(b'e') | Some(b'E')) {
                    self.advance();
                    if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                        self.advance();
                    }
                    let exp_start = self.pos;
                    while let Some(d) = self.peek() {
                        if d.is_ascii_digit() {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                    if self.pos == exp_start {
                        return self.error("expected digits in exponent");
                    }
                }
                let text = &self.input[start..self.pos];
                let value: f64 = text
                    .parse()
                    .map_err(|e: std::num::ParseFloatError| RsqliteError::Parse(e.to_string()))?;
                Ok(Token::RealLiteral(value))
            }

            // ---- identifiers / keywords ----
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => Ok(self.read_word()),

            // ---- operators & punctuation ----
            b'+' => {
                self.advance();
                Ok(Token::Plus)
            }
            b'-' => {
                self.advance();
                Ok(Token::Minus)
            }
            b'*' => {
                self.advance();
                Ok(Token::Star)
            }
            b'/' => {
                self.advance();
                Ok(Token::Slash)
            }
            b'%' => {
                self.advance();
                Ok(Token::Percent)
            }
            b'&' => {
                self.advance();
                Ok(Token::BitAnd)
            }
            b'~' => {
                self.advance();
                Ok(Token::BitNot)
            }
            b'(' => {
                self.advance();
                Ok(Token::LeftParen)
            }
            b')' => {
                self.advance();
                Ok(Token::RightParen)
            }
            b',' => {
                self.advance();
                Ok(Token::Comma)
            }
            b';' => {
                self.advance();
                Ok(Token::Semicolon)
            }
            b'.' => {
                self.advance();
                Ok(Token::Dot)
            }
            b'?' => {
                self.advance();
                Ok(Token::QuestionMark)
            }

            // = or ==
            b'=' => {
                self.advance();
                if self.peek() == Some(b'=') {
                    self.advance();
                }
                Ok(Token::Eq)
            }

            // ! must be followed by = for !=
            b'!' => {
                self.advance();
                if self.peek() == Some(b'=') {
                    self.advance();
                    Ok(Token::NotEq)
                } else {
                    self.error("expected '=' after '!'")
                }
            }

            // < <= <> <<
            b'<' => {
                self.advance();
                match self.peek() {
                    Some(b'=') => {
                        self.advance();
                        Ok(Token::Le)
                    }
                    Some(b'>') => {
                        self.advance();
                        Ok(Token::NotEq)
                    }
                    Some(b'<') => {
                        self.advance();
                        Ok(Token::ShiftLeft)
                    }
                    _ => Ok(Token::Lt),
                }
            }

            // > >= >>
            b'>' => {
                self.advance();
                match self.peek() {
                    Some(b'=') => {
                        self.advance();
                        Ok(Token::Ge)
                    }
                    Some(b'>') => {
                        self.advance();
                        Ok(Token::ShiftRight)
                    }
                    _ => Ok(Token::Gt),
                }
            }

            // | || (BitOr vs Concat)
            b'|' => {
                self.advance();
                if self.peek() == Some(b'|') {
                    self.advance();
                    Ok(Token::Concat)
                } else {
                    Ok(Token::BitOr)
                }
            }

            _ => self.error(format!(
                "unexpected character '{}'",
                self.remaining().chars().next().unwrap()
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience function
// ---------------------------------------------------------------------------

/// Tokenize the full input into a vector of tokens (not including the final
/// [`Token::Eof`]).
pub fn tokenize(input: &str) -> Result<Vec<Token>> {
    let mut tokenizer = Tokenizer::new(input);
    let mut tokens = Vec::new();
    loop {
        let tok = tokenizer.next_token()?;
        if tok == Token::Eof {
            break;
        }
        tokens.push(tok);
    }
    Ok(tokens)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::approx_constant)]
mod tests {
    use super::*;

    /// Helper: tokenize input and return all non-EOF tokens.
    fn tok(input: &str) -> Vec<Token> {
        tokenize(input).expect("tokenize failed")
    }

    /// Helper: tokenize input and expect exactly one non-EOF token.
    fn tok1(input: &str) -> Token {
        let tokens = tok(input);
        assert_eq!(tokens.len(), 1, "expected 1 token, got {:?}", tokens);
        tokens.into_iter().next().unwrap()
    }

    // ----------------------------------------------------------------
    // Keywords
    // ----------------------------------------------------------------

    #[test]
    fn keywords_case_insensitive() {
        assert_eq!(tok1("SELECT"), Token::Select);
        assert_eq!(tok1("select"), Token::Select);
        assert_eq!(tok1("SeLeCt"), Token::Select);
    }

    #[test]
    fn all_keywords() {
        let cases: Vec<(&str, Token)> = vec![
            ("ABORT", Token::Abort),
            ("ADD", Token::Add),
            ("ALL", Token::All),
            ("ALTER", Token::Alter),
            ("AND", Token::And),
            ("AS", Token::As),
            ("ASC", Token::Asc),
            ("AUTOINCREMENT", Token::Autoincrement),
            ("BEGIN", Token::Begin),
            ("BETWEEN", Token::Between),
            ("BLOB", Token::BlobKw),
            ("BY", Token::By),
            ("CASE", Token::Case),
            ("CAST", Token::Cast),
            ("CHECK", Token::Check),
            ("COLLATE", Token::Collate),
            ("COLUMN", Token::Column),
            ("COMMIT", Token::Commit),
            ("CONFLICT", Token::Conflict),
            ("CONSTRAINT", Token::Constraint),
            ("CREATE", Token::Create),
            ("CROSS", Token::Cross),
            ("CURRENT_DATE", Token::CurrentDate),
            ("CURRENT_TIME", Token::CurrentTime),
            ("CURRENT_TIMESTAMP", Token::CurrentTimestamp),
            ("DEFAULT", Token::Default),
            ("DELETE", Token::Delete),
            ("DESC", Token::Desc),
            ("DISTINCT", Token::Distinct),
            ("DROP", Token::Drop),
            ("ELSE", Token::Else),
            ("END", Token::End),
            ("ESCAPE", Token::Escape),
            ("EXCEPT", Token::Except),
            ("EXISTS", Token::Exists),
            ("EXPLAIN", Token::Explain),
            ("FAIL", Token::Fail),
            ("FOREIGN", Token::Foreign),
            ("FROM", Token::From),
            ("GLOB", Token::Glob),
            ("GROUP", Token::Group),
            ("HAVING", Token::Having),
            ("IF", Token::If),
            ("IGNORE", Token::Ignore),
            ("IN", Token::In),
            ("INDEX", Token::Index),
            ("INNER", Token::Inner),
            ("INSERT", Token::Insert),
            ("INTEGER", Token::IntegerKw),
            ("INTERSECT", Token::Intersect),
            ("INTO", Token::Into),
            ("IS", Token::Is),
            ("JOIN", Token::Join),
            ("KEY", Token::Key),
            ("LEFT", Token::Left),
            ("LIKE", Token::Like),
            ("LIMIT", Token::Limit),
            ("NOT", Token::Not),
            ("NULL", Token::Null),
            ("NUMERIC", Token::NumericKw),
            ("OFFSET", Token::Offset),
            ("ON", Token::On),
            ("OR", Token::Or),
            ("ORDER", Token::Order),
            ("OUTER", Token::Outer),
            ("PLAN", Token::Plan),
            ("PRAGMA", Token::Pragma),
            ("PRIMARY", Token::Primary),
            ("QUERY", Token::Query),
            ("REAL", Token::RealKw),
            ("RECURSIVE", Token::Recursive),
            ("REFERENCES", Token::References),
            ("RELEASE", Token::Release),
            ("RENAME", Token::Rename),
            ("REPLACE", Token::Replace),
            ("RIGHT", Token::Right),
            ("ROLLBACK", Token::Rollback),
            ("SAVEPOINT", Token::Savepoint),
            ("SELECT", Token::Select),
            ("SET", Token::Set),
            ("TABLE", Token::Table),
            ("TEXT", Token::TextKw),
            ("THEN", Token::Then),
            ("TO", Token::To),
            ("TRANSACTION", Token::Transaction),
            ("UNION", Token::Union),
            ("UNIQUE", Token::Unique),
            ("UPDATE", Token::Update),
            ("VALUES", Token::Values),
            ("WHEN", Token::When),
            ("WHERE", Token::Where),
            ("WITH", Token::With),
        ];
        for (input, expected) in cases {
            assert_eq!(tok1(input), expected, "keyword mismatch for {:?}", input);
            // Also verify lowercase
            assert_eq!(
                tok1(&input.to_lowercase()),
                expected,
                "lowercase keyword mismatch for {:?}",
                input
            );
        }
    }

    // ----------------------------------------------------------------
    // String literals
    // ----------------------------------------------------------------

    #[test]
    fn string_literal_simple() {
        assert_eq!(tok1("'hello'"), Token::StringLiteral("hello".into()));
    }

    #[test]
    fn string_literal_empty() {
        assert_eq!(tok1("''"), Token::StringLiteral("".into()));
    }

    #[test]
    fn string_literal_escaped_quote() {
        assert_eq!(tok1("'it''s'"), Token::StringLiteral("it's".into()));
    }

    #[test]
    fn string_literal_double_escaped() {
        assert_eq!(tok1("'a''''b'"), Token::StringLiteral("a''b".into()));
    }

    #[test]
    fn string_literal_unterminated() {
        assert!(tokenize("'unterminated").is_err());
    }

    // ----------------------------------------------------------------
    // Numeric literals
    // ----------------------------------------------------------------

    #[test]
    fn integer_literal() {
        assert_eq!(tok1("42"), Token::IntegerLiteral(42));
        assert_eq!(tok1("0"), Token::IntegerLiteral(0));
        assert_eq!(tok1("1234567890"), Token::IntegerLiteral(1234567890));
    }

    #[test]
    fn real_literal_with_dot() {
        assert_eq!(tok1("3.14"), Token::RealLiteral(3.14));
        assert_eq!(tok1("0.5"), Token::RealLiteral(0.5));
    }

    #[test]
    fn real_literal_leading_dot() {
        assert_eq!(tok1(".5"), Token::RealLiteral(0.5));
        assert_eq!(tok1(".123"), Token::RealLiteral(0.123));
    }

    #[test]
    fn real_literal_with_exponent() {
        assert_eq!(tok1("1e10"), Token::RealLiteral(1e10));
        assert_eq!(tok1("1E10"), Token::RealLiteral(1e10));
        assert_eq!(tok1("1.5e2"), Token::RealLiteral(150.0));
        assert_eq!(tok1("1e+3"), Token::RealLiteral(1e3));
        assert_eq!(tok1("1e-3"), Token::RealLiteral(1e-3));
    }

    #[test]
    fn real_literal_dot_with_exponent() {
        assert_eq!(tok1(".5e2"), Token::RealLiteral(50.0));
    }

    #[test]
    fn hex_integer_literal() {
        assert_eq!(tok1("0xFF"), Token::IntegerLiteral(255));
        assert_eq!(tok1("0x0"), Token::IntegerLiteral(0));
        assert_eq!(tok1("0X1A"), Token::IntegerLiteral(26));
    }

    #[test]
    fn hex_literal_no_digits_error() {
        assert!(tokenize("0x").is_err());
    }

    #[test]
    fn exponent_no_digits_error() {
        assert!(tokenize("1e").is_err());
    }

    // ----------------------------------------------------------------
    // Blob literals
    // ----------------------------------------------------------------

    #[test]
    fn blob_literal() {
        assert_eq!(
            tok1("X'48656C6C6F'"),
            Token::BlobLiteral(vec![0x48, 0x65, 0x6C, 0x6C, 0x6F])
        );
    }

    #[test]
    fn blob_literal_lowercase_x() {
        assert_eq!(tok1("x'FF00'"), Token::BlobLiteral(vec![0xFF, 0x00]));
    }

    #[test]
    fn blob_literal_empty() {
        assert_eq!(tok1("X''"), Token::BlobLiteral(vec![]));
    }

    #[test]
    fn blob_literal_odd_digits_error() {
        assert!(tokenize("X'ABC'").is_err());
    }

    #[test]
    fn blob_literal_unterminated() {
        assert!(tokenize("X'AB").is_err());
    }

    #[test]
    fn blob_literal_invalid_char() {
        assert!(tokenize("X'GG'").is_err());
    }

    // ----------------------------------------------------------------
    // Identifiers
    // ----------------------------------------------------------------

    #[test]
    fn bare_identifier() {
        assert_eq!(tok1("foo"), Token::Ident("foo".into()));
        assert_eq!(tok1("_bar"), Token::Ident("_bar".into()));
        assert_eq!(tok1("a1b2"), Token::Ident("a1b2".into()));
    }

    #[test]
    fn double_quoted_identifier() {
        assert_eq!(tok1("\"my table\""), Token::Ident("my table".into()));
    }

    #[test]
    fn double_quoted_with_escaped_quote() {
        assert_eq!(
            tok1("\"say \"\"hello\"\"\""),
            Token::Ident("say \"hello\"".into())
        );
    }

    #[test]
    fn backtick_quoted_identifier() {
        assert_eq!(tok1("`my col`"), Token::Ident("my col".into()));
    }

    #[test]
    fn bracket_quoted_identifier() {
        assert_eq!(tok1("[my col]"), Token::Ident("my col".into()));
    }

    #[test]
    fn unterminated_double_quote() {
        assert!(tokenize("\"abc").is_err());
    }

    #[test]
    fn unterminated_backtick() {
        assert!(tokenize("`abc").is_err());
    }

    #[test]
    fn unterminated_bracket() {
        assert!(tokenize("[abc").is_err());
    }

    // ----------------------------------------------------------------
    // Operators
    // ----------------------------------------------------------------

    #[test]
    fn single_char_operators() {
        assert_eq!(tok1("+"), Token::Plus);
        assert_eq!(tok1("-"), Token::Minus);
        assert_eq!(tok1("*"), Token::Star);
        assert_eq!(tok1("/"), Token::Slash);
        assert_eq!(tok1("%"), Token::Percent);
        assert_eq!(tok1("&"), Token::BitAnd);
        assert_eq!(tok1("~"), Token::BitNot);
    }

    #[test]
    fn comparison_operators() {
        assert_eq!(tok1("="), Token::Eq);
        assert_eq!(tok1("=="), Token::Eq);
        assert_eq!(tok1("!="), Token::NotEq);
        assert_eq!(tok1("<>"), Token::NotEq);
        assert_eq!(tok1("<"), Token::Lt);
        assert_eq!(tok1(">"), Token::Gt);
        assert_eq!(tok1("<="), Token::Le);
        assert_eq!(tok1(">="), Token::Ge);
    }

    #[test]
    fn shift_operators() {
        assert_eq!(tok1("<<"), Token::ShiftLeft);
        assert_eq!(tok1(">>"), Token::ShiftRight);
    }

    #[test]
    fn pipe_operators() {
        assert_eq!(tok1("|"), Token::BitOr);
        assert_eq!(tok1("||"), Token::Concat);
    }

    #[test]
    fn bang_without_equals_is_error() {
        assert!(tokenize("!").is_err());
    }

    // ----------------------------------------------------------------
    // Punctuation
    // ----------------------------------------------------------------

    #[test]
    fn punctuation() {
        assert_eq!(tok1("("), Token::LeftParen);
        assert_eq!(tok1(")"), Token::RightParen);
        assert_eq!(tok1(","), Token::Comma);
        assert_eq!(tok1(";"), Token::Semicolon);
        assert_eq!(tok1("."), Token::Dot);
        assert_eq!(tok1("?"), Token::QuestionMark);
    }

    // ----------------------------------------------------------------
    // Comments
    // ----------------------------------------------------------------

    #[test]
    fn line_comment() {
        let tokens = tok("SELECT -- this is a comment\n42");
        assert_eq!(tokens, vec![Token::Select, Token::IntegerLiteral(42)]);
    }

    #[test]
    fn line_comment_at_end() {
        let tokens = tok("SELECT -- trailing");
        assert_eq!(tokens, vec![Token::Select]);
    }

    #[test]
    fn block_comment() {
        let tokens = tok("SELECT /* comment */ 42");
        assert_eq!(tokens, vec![Token::Select, Token::IntegerLiteral(42)]);
    }

    #[test]
    fn block_comment_multiline() {
        let tokens = tok("SELECT /* multi\nline\ncomment */ 42");
        assert_eq!(tokens, vec![Token::Select, Token::IntegerLiteral(42)]);
    }

    #[test]
    fn nested_block_comment() {
        let tokens = tok("SELECT /* outer /* inner */ still comment */ 42");
        assert_eq!(tokens, vec![Token::Select, Token::IntegerLiteral(42)]);
    }

    #[test]
    fn unterminated_block_comment() {
        assert!(tokenize("SELECT /* unterminated").is_err());
    }

    // ----------------------------------------------------------------
    // Whitespace
    // ----------------------------------------------------------------

    #[test]
    fn various_whitespace() {
        let tokens = tok("  SELECT \t\n\r FROM  ");
        assert_eq!(tokens, vec![Token::Select, Token::From]);
    }

    #[test]
    fn empty_input() {
        let tokens = tok("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn whitespace_only() {
        let tokens = tok("   \t\n  ");
        assert!(tokens.is_empty());
    }

    // ----------------------------------------------------------------
    // Error cases
    // ----------------------------------------------------------------

    #[test]
    fn invalid_character() {
        assert!(tokenize("SELECT @foo").is_err());
    }

    #[test]
    fn invalid_character_hash() {
        assert!(tokenize("#").is_err());
    }

    // ----------------------------------------------------------------
    // x/X as identifier (not blob literal)
    // ----------------------------------------------------------------

    #[test]
    fn x_as_identifier() {
        // x not followed by ' should be an identifier
        assert_eq!(tok1("x"), Token::Ident("x".into()));
        assert_eq!(tok1("X"), Token::Ident("X".into()));
        assert_eq!(tok1("xyz"), Token::Ident("xyz".into()));
    }

    // ----------------------------------------------------------------
    // Dot vs real literal disambiguation
    // ----------------------------------------------------------------

    #[test]
    fn dot_then_ident() {
        let tokens = tok("t.col");
        assert_eq!(
            tokens,
            vec![
                Token::Ident("t".into()),
                Token::Dot,
                Token::Ident("col".into()),
            ]
        );
    }

    #[test]
    fn integer_dot_no_fraction() {
        // "42." followed by a non-digit should be integer + dot
        let tokens = tok("42.col");
        assert_eq!(
            tokens,
            vec![
                Token::IntegerLiteral(42),
                Token::Dot,
                Token::Ident("col".into()),
            ]
        );
    }

    // ----------------------------------------------------------------
    // Full SQL statements
    // ----------------------------------------------------------------

    #[test]
    fn select_statement() {
        let tokens = tok("SELECT a, b FROM t WHERE a = 1;");
        assert_eq!(
            tokens,
            vec![
                Token::Select,
                Token::Ident("a".into()),
                Token::Comma,
                Token::Ident("b".into()),
                Token::From,
                Token::Ident("t".into()),
                Token::Where,
                Token::Ident("a".into()),
                Token::Eq,
                Token::IntegerLiteral(1),
                Token::Semicolon,
            ]
        );
    }

    #[test]
    fn create_table_statement() {
        let tokens = tok(
            "CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL);",
        );
        assert_eq!(
            tokens,
            vec![
                Token::Create,
                Token::Table,
                Token::If,
                Token::Not,
                Token::Exists,
                Token::Ident("users".into()),
                Token::LeftParen,
                Token::Ident("id".into()),
                Token::IntegerKw,
                Token::Primary,
                Token::Key,
                Token::Autoincrement,
                Token::Comma,
                Token::Ident("name".into()),
                Token::TextKw,
                Token::Not,
                Token::Null,
                Token::RightParen,
                Token::Semicolon,
            ]
        );
    }

    #[test]
    fn insert_statement() {
        let tokens = tok("INSERT INTO t VALUES (1, 'hello', 3.14);");
        assert_eq!(
            tokens,
            vec![
                Token::Insert,
                Token::Into,
                Token::Ident("t".into()),
                Token::Values,
                Token::LeftParen,
                Token::IntegerLiteral(1),
                Token::Comma,
                Token::StringLiteral("hello".into()),
                Token::Comma,
                Token::RealLiteral(3.14),
                Token::RightParen,
                Token::Semicolon,
            ]
        );
    }

    #[test]
    fn update_statement() {
        let tokens = tok("UPDATE t SET a = 1, b = 'two' WHERE id = 3;");
        assert_eq!(
            tokens,
            vec![
                Token::Update,
                Token::Ident("t".into()),
                Token::Set,
                Token::Ident("a".into()),
                Token::Eq,
                Token::IntegerLiteral(1),
                Token::Comma,
                Token::Ident("b".into()),
                Token::Eq,
                Token::StringLiteral("two".into()),
                Token::Where,
                Token::Ident("id".into()),
                Token::Eq,
                Token::IntegerLiteral(3),
                Token::Semicolon,
            ]
        );
    }

    #[test]
    fn delete_statement() {
        let tokens = tok("DELETE FROM t WHERE id > 10;");
        assert_eq!(
            tokens,
            vec![
                Token::Delete,
                Token::From,
                Token::Ident("t".into()),
                Token::Where,
                Token::Ident("id".into()),
                Token::Gt,
                Token::IntegerLiteral(10),
                Token::Semicolon,
            ]
        );
    }

    #[test]
    fn complex_select_with_join() {
        let tokens = tok(
            "SELECT a.id, b.name FROM a INNER JOIN b ON a.id = b.a_id WHERE a.x BETWEEN 1 AND 10 ORDER BY b.name ASC LIMIT 5 OFFSET 10;",
        );
        assert_eq!(
            tokens,
            vec![
                Token::Select,
                Token::Ident("a".into()),
                Token::Dot,
                Token::Ident("id".into()),
                Token::Comma,
                Token::Ident("b".into()),
                Token::Dot,
                Token::Ident("name".into()),
                Token::From,
                Token::Ident("a".into()),
                Token::Inner,
                Token::Join,
                Token::Ident("b".into()),
                Token::On,
                Token::Ident("a".into()),
                Token::Dot,
                Token::Ident("id".into()),
                Token::Eq,
                Token::Ident("b".into()),
                Token::Dot,
                Token::Ident("a_id".into()),
                Token::Where,
                Token::Ident("a".into()),
                Token::Dot,
                Token::Ident("x".into()),
                Token::Between,
                Token::IntegerLiteral(1),
                Token::And,
                Token::IntegerLiteral(10),
                Token::Order,
                Token::By,
                Token::Ident("b".into()),
                Token::Dot,
                Token::Ident("name".into()),
                Token::Asc,
                Token::Limit,
                Token::IntegerLiteral(5),
                Token::Offset,
                Token::IntegerLiteral(10),
                Token::Semicolon,
            ]
        );
    }

    #[test]
    fn concat_operator_in_expression() {
        let tokens = tok("SELECT 'a' || 'b'");
        assert_eq!(
            tokens,
            vec![
                Token::Select,
                Token::StringLiteral("a".into()),
                Token::Concat,
                Token::StringLiteral("b".into()),
            ]
        );
    }

    #[test]
    fn negative_number_as_tokens() {
        // The tokenizer produces Minus and IntegerLiteral separately;
        // the parser is responsible for combining them.
        let tokens = tok("-42");
        assert_eq!(tokens, vec![Token::Minus, Token::IntegerLiteral(42)]);
    }

    #[test]
    fn group_by_having() {
        let tokens = tok("SELECT a, COUNT(*) FROM t GROUP BY a HAVING COUNT(*) > 1");
        assert_eq!(
            tokens,
            vec![
                Token::Select,
                Token::Ident("a".into()),
                Token::Comma,
                Token::Ident("COUNT".into()),
                Token::LeftParen,
                Token::Star,
                Token::RightParen,
                Token::From,
                Token::Ident("t".into()),
                Token::Group,
                Token::By,
                Token::Ident("a".into()),
                Token::Having,
                Token::Ident("COUNT".into()),
                Token::LeftParen,
                Token::Star,
                Token::RightParen,
                Token::Gt,
                Token::IntegerLiteral(1),
            ]
        );
    }

    #[test]
    fn case_expression() {
        let tokens = tok("CASE WHEN x = 1 THEN 'one' ELSE 'other' END");
        assert_eq!(
            tokens,
            vec![
                Token::Case,
                Token::When,
                Token::Ident("x".into()),
                Token::Eq,
                Token::IntegerLiteral(1),
                Token::Then,
                Token::StringLiteral("one".into()),
                Token::Else,
                Token::StringLiteral("other".into()),
                Token::End,
            ]
        );
    }

    #[test]
    fn transaction_statements() {
        let tokens = tok("BEGIN TRANSACTION; COMMIT; ROLLBACK;");
        assert_eq!(
            tokens,
            vec![
                Token::Begin,
                Token::Transaction,
                Token::Semicolon,
                Token::Commit,
                Token::Semicolon,
                Token::Rollback,
                Token::Semicolon,
            ]
        );
    }

    #[test]
    fn pragma_statement() {
        let tokens = tok("PRAGMA table_info('users');");
        assert_eq!(
            tokens,
            vec![
                Token::Pragma,
                Token::Ident("table_info".into()),
                Token::LeftParen,
                Token::StringLiteral("users".into()),
                Token::RightParen,
                Token::Semicolon,
            ]
        );
    }

    #[test]
    fn explain_query_plan() {
        let tokens = tok("EXPLAIN QUERY PLAN SELECT * FROM t;");
        assert_eq!(
            tokens,
            vec![
                Token::Explain,
                Token::Query,
                Token::Plan,
                Token::Select,
                Token::Star,
                Token::From,
                Token::Ident("t".into()),
                Token::Semicolon,
            ]
        );
    }

    #[test]
    fn is_null_is_not_null() {
        let tokens = tok("x IS NULL AND y IS NOT NULL");
        assert_eq!(
            tokens,
            vec![
                Token::Ident("x".into()),
                Token::Is,
                Token::Null,
                Token::And,
                Token::Ident("y".into()),
                Token::Is,
                Token::Not,
                Token::Null,
            ]
        );
    }

    #[test]
    fn in_subquery() {
        let tokens = tok("WHERE id IN (SELECT id FROM other)");
        assert_eq!(
            tokens,
            vec![
                Token::Where,
                Token::Ident("id".into()),
                Token::In,
                Token::LeftParen,
                Token::Select,
                Token::Ident("id".into()),
                Token::From,
                Token::Ident("other".into()),
                Token::RightParen,
            ]
        );
    }

    #[test]
    fn like_expression() {
        let tokens = tok("name LIKE '%foo%'");
        assert_eq!(
            tokens,
            vec![
                Token::Ident("name".into()),
                Token::Like,
                Token::StringLiteral("%foo%".into()),
            ]
        );
    }

    #[test]
    fn cast_expression() {
        let tokens = tok("CAST(x AS INTEGER)");
        assert_eq!(
            tokens,
            vec![
                Token::Cast,
                Token::LeftParen,
                Token::Ident("x".into()),
                Token::As,
                Token::IntegerKw,
                Token::RightParen,
            ]
        );
    }

    #[test]
    fn multiple_semicolons() {
        let tokens = tok(";;;");
        assert_eq!(
            tokens,
            vec![Token::Semicolon, Token::Semicolon, Token::Semicolon]
        );
    }

    #[test]
    fn eof_repeated() {
        let mut t = Tokenizer::new("");
        assert_eq!(t.next_token().unwrap(), Token::Eof);
        assert_eq!(t.next_token().unwrap(), Token::Eof);
        assert_eq!(t.next_token().unwrap(), Token::Eof);
    }

    #[test]
    fn question_mark_placeholder() {
        let tokens = tok("SELECT * FROM t WHERE id = ?");
        assert_eq!(
            tokens,
            vec![
                Token::Select,
                Token::Star,
                Token::From,
                Token::Ident("t".into()),
                Token::Where,
                Token::Ident("id".into()),
                Token::Eq,
                Token::QuestionMark,
            ]
        );
    }

    #[test]
    fn bitwise_operators() {
        let tokens = tok("a & b | ~c << 2 >> 1");
        assert_eq!(
            tokens,
            vec![
                Token::Ident("a".into()),
                Token::BitAnd,
                Token::Ident("b".into()),
                Token::BitOr,
                Token::BitNot,
                Token::Ident("c".into()),
                Token::ShiftLeft,
                Token::IntegerLiteral(2),
                Token::ShiftRight,
                Token::IntegerLiteral(1),
            ]
        );
    }

    #[test]
    fn distinct_union_except_intersect() {
        let tokens = tok("SELECT DISTINCT a FROM t1 UNION SELECT a FROM t2 EXCEPT SELECT a FROM t3 INTERSECT SELECT a FROM t4");
        assert_eq!(
            tokens,
            vec![
                Token::Select,
                Token::Distinct,
                Token::Ident("a".into()),
                Token::From,
                Token::Ident("t1".into()),
                Token::Union,
                Token::Select,
                Token::Ident("a".into()),
                Token::From,
                Token::Ident("t2".into()),
                Token::Except,
                Token::Select,
                Token::Ident("a".into()),
                Token::From,
                Token::Ident("t3".into()),
                Token::Intersect,
                Token::Select,
                Token::Ident("a".into()),
                Token::From,
                Token::Ident("t4".into()),
            ]
        );
    }

    #[test]
    fn with_recursive() {
        let tokens = tok("WITH RECURSIVE cte AS (SELECT 1)");
        assert_eq!(
            tokens,
            vec![
                Token::With,
                Token::Recursive,
                Token::Ident("cte".into()),
                Token::As,
                Token::LeftParen,
                Token::Select,
                Token::IntegerLiteral(1),
                Token::RightParen,
            ]
        );
    }

    #[test]
    fn glob_keyword() {
        let tokens = tok("name GLOB 'foo*'");
        assert_eq!(
            tokens,
            vec![
                Token::Ident("name".into()),
                Token::Glob,
                Token::StringLiteral("foo*".into()),
            ]
        );
    }

    #[test]
    fn escape_keyword() {
        let tokens = tok("LIKE '%x%' ESCAPE '\\'");
        assert_eq!(
            tokens,
            vec![
                Token::Like,
                Token::StringLiteral("%x%".into()),
                Token::Escape,
                Token::StringLiteral("\\".into()),
            ]
        );
    }

    #[test]
    fn create_index_statement() {
        let tokens = tok("CREATE UNIQUE INDEX idx_name ON t(col);");
        assert_eq!(
            tokens,
            vec![
                Token::Create,
                Token::Unique,
                Token::Index,
                Token::Ident("idx_name".into()),
                Token::On,
                Token::Ident("t".into()),
                Token::LeftParen,
                Token::Ident("col".into()),
                Token::RightParen,
                Token::Semicolon,
            ]
        );
    }

    #[test]
    fn alter_table() {
        let tokens = tok("ALTER TABLE t ADD COLUMN c TEXT DEFAULT 'x';");
        assert_eq!(
            tokens,
            vec![
                Token::Alter,
                Token::Table,
                Token::Ident("t".into()),
                Token::Add,
                Token::Column,
                Token::Ident("c".into()),
                Token::TextKw,
                Token::Default,
                Token::StringLiteral("x".into()),
                Token::Semicolon,
            ]
        );
    }

    #[test]
    fn rename_table() {
        let tokens = tok("ALTER TABLE t RENAME TO t2;");
        assert_eq!(
            tokens,
            vec![
                Token::Alter,
                Token::Table,
                Token::Ident("t".into()),
                Token::Rename,
                Token::To,
                Token::Ident("t2".into()),
                Token::Semicolon,
            ]
        );
    }

    #[test]
    fn foreign_key_references() {
        let tokens = tok("FOREIGN KEY (a) REFERENCES other(b)");
        assert_eq!(
            tokens,
            vec![
                Token::Foreign,
                Token::Key,
                Token::LeftParen,
                Token::Ident("a".into()),
                Token::RightParen,
                Token::References,
                Token::Ident("other".into()),
                Token::LeftParen,
                Token::Ident("b".into()),
                Token::RightParen,
            ]
        );
    }

    #[test]
    fn savepoint_release() {
        let tokens = tok("SAVEPOINT sp1; RELEASE SAVEPOINT sp1;");
        assert_eq!(
            tokens,
            vec![
                Token::Savepoint,
                Token::Ident("sp1".into()),
                Token::Semicolon,
                Token::Release,
                Token::Savepoint,
                Token::Ident("sp1".into()),
                Token::Semicolon,
            ]
        );
    }

    #[test]
    fn on_conflict_clause() {
        let tokens = tok("ON CONFLICT ABORT");
        assert_eq!(tokens, vec![Token::On, Token::Conflict, Token::Abort]);
    }

    #[test]
    fn replace_insert() {
        let tokens = tok("REPLACE INTO t VALUES (1);");
        assert_eq!(
            tokens,
            vec![
                Token::Replace,
                Token::Into,
                Token::Ident("t".into()),
                Token::Values,
                Token::LeftParen,
                Token::IntegerLiteral(1),
                Token::RightParen,
                Token::Semicolon,
            ]
        );
    }

    #[test]
    fn current_date_time_timestamp() {
        let tokens = tok("SELECT CURRENT_DATE, CURRENT_TIME, CURRENT_TIMESTAMP");
        assert_eq!(
            tokens,
            vec![
                Token::Select,
                Token::CurrentDate,
                Token::Comma,
                Token::CurrentTime,
                Token::Comma,
                Token::CurrentTimestamp,
            ]
        );
    }

    #[test]
    fn collate_keyword() {
        let tokens = tok("ORDER BY name COLLATE NOCASE");
        assert_eq!(
            tokens,
            vec![
                Token::Order,
                Token::By,
                Token::Ident("name".into()),
                Token::Collate,
                Token::Ident("NOCASE".into()),
            ]
        );
    }

    #[test]
    fn constraint_check() {
        let tokens = tok("CONSTRAINT chk CHECK (x > 0)");
        assert_eq!(
            tokens,
            vec![
                Token::Constraint,
                Token::Ident("chk".into()),
                Token::Check,
                Token::LeftParen,
                Token::Ident("x".into()),
                Token::Gt,
                Token::IntegerLiteral(0),
                Token::RightParen,
            ]
        );
    }

    #[test]
    fn left_right_cross_outer_join() {
        let tokens = tok("LEFT OUTER JOIN t ON 1 = 1");
        assert_eq!(
            tokens,
            vec![
                Token::Left,
                Token::Outer,
                Token::Join,
                Token::Ident("t".into()),
                Token::On,
                Token::IntegerLiteral(1),
                Token::Eq,
                Token::IntegerLiteral(1),
            ]
        );

        let tokens = tok("RIGHT JOIN t ON 1 = 1");
        assert_eq!(
            tokens,
            vec![
                Token::Right,
                Token::Join,
                Token::Ident("t".into()),
                Token::On,
                Token::IntegerLiteral(1),
                Token::Eq,
                Token::IntegerLiteral(1),
            ]
        );

        let tokens = tok("CROSS JOIN t");
        assert_eq!(
            tokens,
            vec![Token::Cross, Token::Join, Token::Ident("t".into())]
        );
    }

    #[test]
    fn display_impl() {
        // Spot-check a few Display implementations
        assert_eq!(Token::Select.to_string(), "SELECT");
        assert_eq!(Token::IntegerLiteral(42).to_string(), "42");
        assert_eq!(Token::StringLiteral("hi".into()).to_string(), "'hi'");
        assert_eq!(Token::Concat.to_string(), "||");
        assert_eq!(Token::Eof.to_string(), "<EOF>");
    }
}
