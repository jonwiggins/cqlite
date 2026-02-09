/// SQL tokenizer - hand-written lexer for SQL statements.

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Keywords
    Select,
    From,
    Where,
    And,
    Or,
    Not,
    Insert,
    Into,
    Values,
    Update,
    Set,
    Delete,
    Create,
    Drop,
    Table,
    Index,
    If,
    Exists,
    Alter,
    Add,
    Rename,
    Column,
    To,
    Primary,
    Key,
    Autoincrement,
    Unique,
    Null,
    Default,
    As,
    Is,
    In,
    Like,
    Between,
    Join,
    Inner,
    Left,
    Outer,
    Cross,
    On,
    Order,
    By,
    Asc,
    Desc,
    Group,
    Having,
    Limit,
    Offset,
    Begin,
    Commit,
    Rollback,
    Transaction,
    Explain,
    Query,
    Plan,
    Pragma,
    Case,
    When,
    Then,
    Else,
    End,
    Cast,
    Distinct,
    All,
    Union,
    Except,
    Intersect,
    Glob,
    Escape,
    Collate,
    Abort,
    Conflict,
    Fail,
    Ignore,
    Replace,
    Check,
    Constraint,
    Foreign,
    References,
    Integer,   // keyword INTEGER for type declarations
    Text,      // keyword TEXT for type declarations
    Real,      // keyword REAL for type declarations
    Blob,      // keyword BLOB for type declarations
    WithoutKw, // WITHOUT keyword
    Rowid,     // keyword ROWID

    // Literals
    StringLiteral(String),
    NumericLiteral(String),
    BlobLiteral(Vec<u8>),

    // Identifiers
    Identifier(String),

    // Operators
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Eq,         // =
    EqEq,       // ==
    NotEq,      // != or <>
    Lt,         // <
    LtEq,       // <=
    Gt,         // >
    GtEq,       // >=
    Concat,     // ||
    BitAnd,     // &
    BitOr,      // |
    ShiftLeft,  // <<
    ShiftRight, // >>
    Tilde,      // ~

    // Punctuation
    LeftParen,
    RightParen,
    Comma,
    Semicolon,
    Dot,

    // Special
    Eof,
}

impl Token {
    pub fn is_keyword(&self) -> bool {
        !matches!(
            self,
            Token::StringLiteral(_)
                | Token::NumericLiteral(_)
                | Token::BlobLiteral(_)
                | Token::Identifier(_)
                | Token::Plus
                | Token::Minus
                | Token::Star
                | Token::Slash
                | Token::Percent
                | Token::Eq
                | Token::EqEq
                | Token::NotEq
                | Token::Lt
                | Token::LtEq
                | Token::Gt
                | Token::GtEq
                | Token::Concat
                | Token::BitAnd
                | Token::BitOr
                | Token::ShiftLeft
                | Token::ShiftRight
                | Token::Tilde
                | Token::LeftParen
                | Token::RightParen
                | Token::Comma
                | Token::Semicolon
                | Token::Dot
                | Token::Eof
        )
    }
}

pub struct Tokenizer {
    input: Vec<char>,
    pos: usize,
}

impl Tokenizer {
    pub fn new(input: &str) -> Self {
        Tokenizer {
            input: input.chars().collect(),
            pos: 0,
        }
    }

    pub fn tokenize(&mut self) -> crate::error::Result<Vec<Token>> {
        let mut tokens = Vec::new();
        loop {
            let token = self.next_token()?;
            if token == Token::Eof {
                tokens.push(Token::Eof);
                break;
            }
            tokens.push(token);
        }
        Ok(tokens)
    }

    fn next_token(&mut self) -> crate::error::Result<Token> {
        self.skip_whitespace();

        if self.pos >= self.input.len() {
            return Ok(Token::Eof);
        }

        let ch = self.input[self.pos];

        // Skip comments
        if ch == '-' && self.peek(1) == Some('-') {
            // Line comment
            while self.pos < self.input.len() && self.input[self.pos] != '\n' {
                self.pos += 1;
            }
            return self.next_token();
        }
        if ch == '/' && self.peek(1) == Some('*') {
            // Block comment
            self.pos += 2;
            while self.pos + 1 < self.input.len() {
                if self.input[self.pos] == '*' && self.input[self.pos + 1] == '/' {
                    self.pos += 2;
                    return self.next_token();
                }
                self.pos += 1;
            }
            return Err(crate::error::RsqliteError::Parse(
                "Unterminated block comment".into(),
            ));
        }

        match ch {
            '(' => {
                self.pos += 1;
                Ok(Token::LeftParen)
            }
            ')' => {
                self.pos += 1;
                Ok(Token::RightParen)
            }
            ',' => {
                self.pos += 1;
                Ok(Token::Comma)
            }
            ';' => {
                self.pos += 1;
                Ok(Token::Semicolon)
            }
            '+' => {
                self.pos += 1;
                Ok(Token::Plus)
            }
            '-' => {
                self.pos += 1;
                Ok(Token::Minus)
            }
            '*' => {
                self.pos += 1;
                Ok(Token::Star)
            }
            '/' => {
                self.pos += 1;
                Ok(Token::Slash)
            }
            '%' => {
                self.pos += 1;
                Ok(Token::Percent)
            }
            '~' => {
                self.pos += 1;
                Ok(Token::Tilde)
            }
            '&' => {
                self.pos += 1;
                Ok(Token::BitAnd)
            }
            '.' => {
                self.pos += 1;
                Ok(Token::Dot)
            }
            '=' => {
                self.pos += 1;
                if self.peek(0) == Some('=') {
                    self.pos += 1;
                    Ok(Token::EqEq)
                } else {
                    Ok(Token::Eq)
                }
            }
            '!' => {
                self.pos += 1;
                if self.peek(0) == Some('=') {
                    self.pos += 1;
                    Ok(Token::NotEq)
                } else {
                    Err(crate::error::RsqliteError::Parse(
                        "Expected '=' after '!'".into(),
                    ))
                }
            }
            '<' => {
                self.pos += 1;
                match self.peek(0) {
                    Some('=') => {
                        self.pos += 1;
                        Ok(Token::LtEq)
                    }
                    Some('>') => {
                        self.pos += 1;
                        Ok(Token::NotEq)
                    }
                    Some('<') => {
                        self.pos += 1;
                        Ok(Token::ShiftLeft)
                    }
                    _ => Ok(Token::Lt),
                }
            }
            '>' => {
                self.pos += 1;
                match self.peek(0) {
                    Some('=') => {
                        self.pos += 1;
                        Ok(Token::GtEq)
                    }
                    Some('>') => {
                        self.pos += 1;
                        Ok(Token::ShiftRight)
                    }
                    _ => Ok(Token::Gt),
                }
            }
            '|' => {
                self.pos += 1;
                if self.peek(0) == Some('|') {
                    self.pos += 1;
                    Ok(Token::Concat)
                } else {
                    Ok(Token::BitOr)
                }
            }
            '\'' => self.read_string(),
            '"' => self.read_quoted_identifier(),
            '`' => self.read_backtick_identifier(),
            '[' => self.read_bracket_identifier(),
            _ if ch == 'x' || ch == 'X' => {
                if self.peek(1) == Some('\'') {
                    self.read_blob_literal()
                } else {
                    self.read_identifier_or_keyword()
                }
            }
            _ if ch.is_ascii_digit() => self.read_number(),
            _ if ch.is_ascii_alphabetic() || ch == '_' => self.read_identifier_or_keyword(),
            _ => {
                self.pos += 1;
                Err(crate::error::RsqliteError::Parse(format!(
                    "Unexpected character: '{}'",
                    ch
                )))
            }
        }
    }

    fn skip_whitespace(&mut self) {
        while self.pos < self.input.len() && self.input[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&self, offset: usize) -> Option<char> {
        self.input.get(self.pos + offset).copied()
    }

    fn read_string(&mut self) -> crate::error::Result<Token> {
        self.pos += 1; // skip opening quote
        let mut s = String::new();
        while self.pos < self.input.len() {
            let ch = self.input[self.pos];
            if ch == '\'' {
                if self.peek(1) == Some('\'') {
                    s.push('\'');
                    self.pos += 2;
                } else {
                    self.pos += 1;
                    return Ok(Token::StringLiteral(s));
                }
            } else {
                s.push(ch);
                self.pos += 1;
            }
        }
        Err(crate::error::RsqliteError::Parse(
            "Unterminated string literal".into(),
        ))
    }

    fn read_quoted_identifier(&mut self) -> crate::error::Result<Token> {
        self.pos += 1;
        let mut s = String::new();
        while self.pos < self.input.len() {
            let ch = self.input[self.pos];
            if ch == '"' {
                if self.peek(1) == Some('"') {
                    s.push('"');
                    self.pos += 2;
                } else {
                    self.pos += 1;
                    return Ok(Token::Identifier(s));
                }
            } else {
                s.push(ch);
                self.pos += 1;
            }
        }
        Err(crate::error::RsqliteError::Parse(
            "Unterminated quoted identifier".into(),
        ))
    }

    fn read_backtick_identifier(&mut self) -> crate::error::Result<Token> {
        self.pos += 1;
        let mut s = String::new();
        while self.pos < self.input.len() {
            let ch = self.input[self.pos];
            if ch == '`' {
                self.pos += 1;
                return Ok(Token::Identifier(s));
            }
            s.push(ch);
            self.pos += 1;
        }
        Err(crate::error::RsqliteError::Parse(
            "Unterminated backtick identifier".into(),
        ))
    }

    fn read_bracket_identifier(&mut self) -> crate::error::Result<Token> {
        self.pos += 1;
        let mut s = String::new();
        while self.pos < self.input.len() {
            let ch = self.input[self.pos];
            if ch == ']' {
                self.pos += 1;
                return Ok(Token::Identifier(s));
            }
            s.push(ch);
            self.pos += 1;
        }
        Err(crate::error::RsqliteError::Parse(
            "Unterminated bracket identifier".into(),
        ))
    }

    fn read_blob_literal(&mut self) -> crate::error::Result<Token> {
        self.pos += 2; // skip x'
        let mut hex = String::new();
        while self.pos < self.input.len() {
            let ch = self.input[self.pos];
            if ch == '\'' {
                self.pos += 1;
                let bytes = hex_decode(&hex).map_err(|e| {
                    crate::error::RsqliteError::Parse(format!("Invalid blob literal: {}", e))
                })?;
                return Ok(Token::BlobLiteral(bytes));
            }
            hex.push(ch);
            self.pos += 1;
        }
        Err(crate::error::RsqliteError::Parse(
            "Unterminated blob literal".into(),
        ))
    }

    fn read_number(&mut self) -> crate::error::Result<Token> {
        let start = self.pos;

        // Check for hex: 0x...
        if self.input[self.pos] == '0' && self.peek(1).is_some_and(|c| c == 'x' || c == 'X') {
            self.pos += 2;
            while self.pos < self.input.len() && self.input[self.pos].is_ascii_hexdigit() {
                self.pos += 1;
            }
            let s: String = self.input[start..self.pos].iter().collect();
            return Ok(Token::NumericLiteral(s));
        }

        while self.pos < self.input.len() && self.input[self.pos].is_ascii_digit() {
            self.pos += 1;
        }

        // Decimal point
        if self.pos < self.input.len() && self.input[self.pos] == '.' {
            self.pos += 1;
            while self.pos < self.input.len() && self.input[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
        }

        // Exponent
        if self.pos < self.input.len()
            && (self.input[self.pos] == 'e' || self.input[self.pos] == 'E')
        {
            self.pos += 1;
            if self.pos < self.input.len()
                && (self.input[self.pos] == '+' || self.input[self.pos] == '-')
            {
                self.pos += 1;
            }
            while self.pos < self.input.len() && self.input[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
        }

        let s: String = self.input[start..self.pos].iter().collect();
        Ok(Token::NumericLiteral(s))
    }

    fn read_identifier_or_keyword(&mut self) -> crate::error::Result<Token> {
        let start = self.pos;
        while self.pos < self.input.len()
            && (self.input[self.pos].is_ascii_alphanumeric() || self.input[self.pos] == '_')
        {
            self.pos += 1;
        }

        let word: String = self.input[start..self.pos].iter().collect();
        let upper = word.to_uppercase();

        let token = match upper.as_str() {
            "SELECT" => Token::Select,
            "FROM" => Token::From,
            "WHERE" => Token::Where,
            "AND" => Token::And,
            "OR" => Token::Or,
            "NOT" => Token::Not,
            "INSERT" => Token::Insert,
            "INTO" => Token::Into,
            "VALUES" => Token::Values,
            "UPDATE" => Token::Update,
            "SET" => Token::Set,
            "DELETE" => Token::Delete,
            "CREATE" => Token::Create,
            "DROP" => Token::Drop,
            "TABLE" => Token::Table,
            "INDEX" => Token::Index,
            "IF" => Token::If,
            "EXISTS" => Token::Exists,
            "ALTER" => Token::Alter,
            "ADD" => Token::Add,
            "RENAME" => Token::Rename,
            "COLUMN" => Token::Column,
            "TO" => Token::To,
            "PRIMARY" => Token::Primary,
            "KEY" => Token::Key,
            "AUTOINCREMENT" => Token::Autoincrement,
            "UNIQUE" => Token::Unique,
            "NULL" => Token::Null,
            "DEFAULT" => Token::Default,
            "AS" => Token::As,
            "IS" => Token::Is,
            "IN" => Token::In,
            "LIKE" => Token::Like,
            "BETWEEN" => Token::Between,
            "JOIN" => Token::Join,
            "INNER" => Token::Inner,
            "LEFT" => Token::Left,
            "OUTER" => Token::Outer,
            "CROSS" => Token::Cross,
            "ON" => Token::On,
            "ORDER" => Token::Order,
            "BY" => Token::By,
            "ASC" => Token::Asc,
            "DESC" => Token::Desc,
            "GROUP" => Token::Group,
            "HAVING" => Token::Having,
            "LIMIT" => Token::Limit,
            "OFFSET" => Token::Offset,
            "BEGIN" => Token::Begin,
            "COMMIT" => Token::Commit,
            "ROLLBACK" => Token::Rollback,
            "TRANSACTION" => Token::Transaction,
            "EXPLAIN" => Token::Explain,
            "QUERY" => Token::Query,
            "PLAN" => Token::Plan,
            "PRAGMA" => Token::Pragma,
            "CASE" => Token::Case,
            "WHEN" => Token::When,
            "THEN" => Token::Then,
            "ELSE" => Token::Else,
            "END" => Token::End,
            "CAST" => Token::Cast,
            "DISTINCT" => Token::Distinct,
            "ALL" => Token::All,
            "UNION" => Token::Union,
            "EXCEPT" => Token::Except,
            "INTERSECT" => Token::Intersect,
            "GLOB" => Token::Glob,
            "ESCAPE" => Token::Escape,
            "COLLATE" => Token::Collate,
            "ABORT" => Token::Abort,
            "CONFLICT" => Token::Conflict,
            "FAIL" => Token::Fail,
            "IGNORE" => Token::Ignore,
            "REPLACE" => Token::Replace,
            "CHECK" => Token::Check,
            "CONSTRAINT" => Token::Constraint,
            "FOREIGN" => Token::Foreign,
            "REFERENCES" => Token::References,
            "INTEGER" => Token::Integer,
            "TEXT" => Token::Text,
            "REAL" => Token::Real,
            "BLOB" => Token::Blob,
            "WITHOUT" => Token::WithoutKw,
            "ROWID" => Token::Rowid,
            "TRUE" => Token::NumericLiteral("1".into()),
            "FALSE" => Token::NumericLiteral("0".into()),
            _ => Token::Identifier(word),
        };

        Ok(token)
    }
}

fn hex_decode(hex: &str) -> std::result::Result<Vec<u8>, String> {
    if !hex.len().is_multiple_of(2) {
        return Err("Odd number of hex digits".into());
    }
    let mut bytes = Vec::with_capacity(hex.len() / 2);
    for i in (0..hex.len()).step_by(2) {
        let byte =
            u8::from_str_radix(&hex[i..i + 2], 16).map_err(|e| format!("Invalid hex: {}", e))?;
        bytes.push(byte);
    }
    Ok(bytes)
}

/// Helper to get a string representation of a token for error messages.
pub fn token_name(token: &Token) -> &'static str {
    match token {
        Token::Select => "SELECT",
        Token::From => "FROM",
        Token::Where => "WHERE",
        Token::And => "AND",
        Token::Or => "OR",
        Token::Not => "NOT",
        Token::Insert => "INSERT",
        Token::Into => "INTO",
        Token::Values => "VALUES",
        Token::Update => "UPDATE",
        Token::Set => "SET",
        Token::Delete => "DELETE",
        Token::Create => "CREATE",
        Token::Drop => "DROP",
        Token::Table => "TABLE",
        Token::Index => "INDEX",
        Token::Alter => "ALTER",
        Token::Primary => "PRIMARY",
        Token::Key => "KEY",
        Token::Unique => "UNIQUE",
        Token::Null => "NULL",
        Token::Default => "DEFAULT",
        Token::As => "AS",
        Token::Is => "IS",
        Token::In => "IN",
        Token::Like => "LIKE",
        Token::Between => "BETWEEN",
        Token::Join => "JOIN",
        Token::Inner => "INNER",
        Token::Left => "LEFT",
        Token::Order => "ORDER",
        Token::By => "BY",
        Token::Group => "GROUP",
        Token::Having => "HAVING",
        Token::Limit => "LIMIT",
        Token::Offset => "OFFSET",
        Token::Begin => "BEGIN",
        Token::Commit => "COMMIT",
        Token::Rollback => "ROLLBACK",
        Token::Explain => "EXPLAIN",
        Token::Pragma => "PRAGMA",
        Token::Case => "CASE",
        Token::Cast => "CAST",
        Token::Distinct => "DISTINCT",
        Token::Star => "*",
        Token::Eof => "EOF",
        Token::LeftParen => "(",
        Token::RightParen => ")",
        Token::Comma => ",",
        Token::Semicolon => ";",
        Token::Dot => ".",
        _ => "token",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_select() {
        let mut tok = Tokenizer::new("SELECT * FROM users WHERE id = 1;");
        let tokens = tok.tokenize().unwrap();
        assert_eq!(tokens[0], Token::Select);
        assert_eq!(tokens[1], Token::Star);
        assert_eq!(tokens[2], Token::From);
        assert_eq!(tokens[3], Token::Identifier("users".into()));
        assert_eq!(tokens[4], Token::Where);
        assert_eq!(tokens[5], Token::Identifier("id".into()));
        assert_eq!(tokens[6], Token::Eq);
        assert_eq!(tokens[7], Token::NumericLiteral("1".into()));
        assert_eq!(tokens[8], Token::Semicolon);
        assert_eq!(tokens[9], Token::Eof);
    }

    #[test]
    fn test_string_literal() {
        let mut tok = Tokenizer::new("'hello ''world'''");
        let tokens = tok.tokenize().unwrap();
        assert_eq!(tokens[0], Token::StringLiteral("hello 'world'".into()));
    }

    #[test]
    fn test_quoted_identifier() {
        let mut tok = Tokenizer::new("\"my table\"");
        let tokens = tok.tokenize().unwrap();
        assert_eq!(tokens[0], Token::Identifier("my table".into()));
    }

    #[test]
    fn test_numeric_literals() {
        let mut tok = Tokenizer::new("42 3.14 1e10 0xFF");
        let tokens = tok.tokenize().unwrap();
        assert_eq!(tokens[0], Token::NumericLiteral("42".into()));
        assert_eq!(tokens[1], Token::NumericLiteral("3.14".into()));
        assert_eq!(tokens[2], Token::NumericLiteral("1e10".into()));
        assert_eq!(tokens[3], Token::NumericLiteral("0xFF".into()));
    }

    #[test]
    fn test_blob_literal() {
        let mut tok = Tokenizer::new("X'48454C4C4F'");
        let tokens = tok.tokenize().unwrap();
        assert_eq!(tokens[0], Token::BlobLiteral(b"HELLO".to_vec()));
    }

    #[test]
    fn test_operators() {
        let mut tok = Tokenizer::new("= == != <> < <= > >= || + - * / %");
        let tokens = tok.tokenize().unwrap();
        assert_eq!(tokens[0], Token::Eq);
        assert_eq!(tokens[1], Token::EqEq);
        assert_eq!(tokens[2], Token::NotEq);
        assert_eq!(tokens[3], Token::NotEq);
        assert_eq!(tokens[4], Token::Lt);
        assert_eq!(tokens[5], Token::LtEq);
        assert_eq!(tokens[6], Token::Gt);
        assert_eq!(tokens[7], Token::GtEq);
        assert_eq!(tokens[8], Token::Concat);
    }

    #[test]
    fn test_comments() {
        let mut tok = Tokenizer::new("SELECT -- comment\n* /* block */ FROM t");
        let tokens = tok.tokenize().unwrap();
        assert_eq!(tokens[0], Token::Select);
        assert_eq!(tokens[1], Token::Star);
        assert_eq!(tokens[2], Token::From);
        assert_eq!(tokens[3], Token::Identifier("t".into()));
    }
}
