// Recursive descent SQL parser.
//
// Parses SQL text into AST nodes defined in ast.rs.
// Uses the tokenizer to produce a token stream, then applies
// recursive descent parsing with operator precedence climbing.

use crate::ast::*;
use crate::error::{Result, RsqliteError};
use crate::tokenizer::{Token, Tokenizer};

/// Parse a SQL string into an AST Statement.
pub fn parse(sql: &str) -> Result<Statement> {
    let mut parser = Parser::new(sql)?;
    parser.parse_statement()
}

struct Parser<'a> {
    tokenizer: Tokenizer<'a>,
    current: Token,
}

impl<'a> Parser<'a> {
    fn new(sql: &'a str) -> Result<Self> {
        let mut tokenizer = Tokenizer::new(sql);
        let current = tokenizer
            .next_token()
            .map_err(|e| RsqliteError::Parse(e.to_string()))?;
        Ok(Self { tokenizer, current })
    }

    fn advance(&mut self) -> Result<Token> {
        let prev = self.current.clone();
        self.current = self
            .tokenizer
            .next_token()
            .map_err(|e| RsqliteError::Parse(e.to_string()))?;
        Ok(prev)
    }

    fn expect(&mut self, expected: &Token) -> Result<()> {
        if &self.current == expected {
            self.advance()?;
            Ok(())
        } else {
            Err(RsqliteError::Parse(format!(
                "expected {expected:?}, got {:?}",
                self.current
            )))
        }
    }

    fn eat_if(&mut self, token: &Token) -> Result<bool> {
        if &self.current == token {
            self.advance()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Check if current token is an identifier matching the given keyword (case-insensitive).
    fn is_ident_keyword(&self, keyword: &str) -> bool {
        if let Token::Ident(ref s) = self.current {
            s.eq_ignore_ascii_case(keyword)
        } else {
            false
        }
    }

    /// Eat an identifier that matches a keyword (case-insensitive).
    fn eat_ident_keyword(&mut self, keyword: &str) -> Result<bool> {
        if self.is_ident_keyword(keyword) {
            self.advance()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn parse_statement(&mut self) -> Result<Statement> {
        let stmt = match &self.current {
            Token::Select => self
                .parse_select_statement()
                .map(|s| Statement::Select(Box::new(s))),
            Token::With => self
                .parse_with_select()
                .map(|s| Statement::Select(Box::new(s))),
            Token::Insert | Token::Replace => self.parse_insert_statement().map(Statement::Insert),
            Token::Update => self.parse_update_statement().map(Statement::Update),
            Token::Delete => self.parse_delete_statement().map(Statement::Delete),
            Token::Create => self.parse_create_statement(),
            Token::Drop => self.parse_drop_statement(),
            Token::Alter => self.parse_alter_statement().map(Statement::AlterTable),
            Token::Begin => self.parse_begin_statement(),
            Token::Commit | Token::End => {
                self.advance()?;
                Ok(Statement::Commit)
            }
            Token::Rollback => {
                self.advance()?;
                Ok(Statement::Rollback)
            }
            Token::Pragma => self.parse_pragma_statement().map(Statement::Pragma),
            Token::Explain => self.parse_explain_statement(),
            _ => Err(RsqliteError::Parse(format!(
                "unexpected token: {:?}",
                self.current
            ))),
        }?;

        // Consume optional semicolon.
        self.eat_if(&Token::Semicolon)?;
        Ok(stmt)
    }

    // -------------------------------------------------------------------------
    // SELECT
    // -------------------------------------------------------------------------

    /// Parse WITH ... SELECT (common table expressions).
    fn parse_with_select(&mut self) -> Result<SelectStatement> {
        self.expect(&Token::With)?;

        let is_recursive = self.eat_if(&Token::Recursive)?;

        let mut ctes = Vec::new();
        loop {
            let name = self.parse_identifier()?;

            // Optional column list: WITH name(col1, col2, ...) AS (...)
            let columns = if self.eat_if(&Token::LeftParen)? {
                let mut cols = vec![self.parse_identifier()?];
                while self.eat_if(&Token::Comma)? {
                    cols.push(self.parse_identifier()?);
                }
                self.expect(&Token::RightParen)?;
                Some(cols)
            } else {
                None
            };

            self.expect(&Token::As)?;
            self.expect(&Token::LeftParen)?;
            let query = self.parse_select_statement()?;
            self.expect(&Token::RightParen)?;

            ctes.push(Cte {
                name,
                columns,
                query: Box::new(query),
                recursive: is_recursive,
            });

            if !self.eat_if(&Token::Comma)? {
                break;
            }
        }

        let mut select = self.parse_select_statement()?;
        select.ctes = ctes;
        Ok(select)
    }

    fn parse_select_statement(&mut self) -> Result<SelectStatement> {
        let mut stmt = self.parse_select_core()?;

        // Parse compound operations (UNION, UNION ALL, INTERSECT, EXCEPT).
        loop {
            let op = if self.eat_if(&Token::Union)? {
                if self.eat_if(&Token::All)? {
                    Some(crate::ast::CompoundOp::UnionAll)
                } else {
                    Some(crate::ast::CompoundOp::Union)
                }
            } else if self.eat_if(&Token::Intersect)? {
                Some(crate::ast::CompoundOp::Intersect)
            } else if self.eat_if(&Token::Except)? {
                Some(crate::ast::CompoundOp::Except)
            } else {
                None
            };

            if let Some(op) = op {
                let rhs = self.parse_select_core()?;
                stmt.compound
                    .push(crate::ast::CompoundClause { op, select: rhs });
            } else {
                break;
            }
        }

        // Parse ORDER BY (applies to entire compound result).
        if self.eat_if(&Token::Order)? {
            self.expect(&Token::By)?;
            let mut items = vec![self.parse_order_by_item()?];
            while self.eat_if(&Token::Comma)? {
                items.push(self.parse_order_by_item()?);
            }
            stmt.order_by = Some(items);
        }

        // Parse LIMIT (applies to entire compound result).
        if self.eat_if(&Token::Limit)? {
            let limit_expr = self.parse_expr()?;
            let offset = if self.eat_if(&Token::Offset)? {
                Some(self.parse_expr()?)
            } else if self.eat_if(&Token::Comma)? {
                // LIMIT x, y means LIMIT y OFFSET x in SQLite.
                let real_limit = self.parse_expr()?;
                stmt.limit = Some(LimitClause {
                    limit: real_limit,
                    offset: Some(limit_expr),
                });
                return Ok(stmt);
            } else {
                None
            };
            stmt.limit = Some(LimitClause {
                limit: limit_expr,
                offset,
            });
        }

        Ok(stmt)
    }

    /// Parse a single SELECT core (no ORDER BY, LIMIT, or compound ops).
    fn parse_select_core(&mut self) -> Result<SelectStatement> {
        self.expect(&Token::Select)?;

        let distinct = self.eat_if(&Token::Distinct)?;

        let columns = self.parse_result_columns()?;

        let from = if self.eat_if(&Token::From)? {
            Some(self.parse_from_clause()?)
        } else {
            None
        };

        let where_clause = if self.eat_if(&Token::Where)? {
            Some(self.parse_expr()?)
        } else {
            None
        };

        let group_by = if self.eat_if(&Token::Group)? {
            self.expect(&Token::By)?;
            let mut exprs = vec![self.parse_expr()?];
            while self.eat_if(&Token::Comma)? {
                exprs.push(self.parse_expr()?);
            }
            Some(exprs)
        } else {
            None
        };

        let having = if self.eat_if(&Token::Having)? {
            Some(self.parse_expr()?)
        } else {
            None
        };

        Ok(SelectStatement {
            distinct,
            columns,
            from,
            where_clause,
            group_by,
            having,
            order_by: None,
            limit: None,
            compound: vec![],
            ctes: vec![],
        })
    }

    fn parse_result_columns(&mut self) -> Result<Vec<ResultColumn>> {
        let mut cols = vec![self.parse_result_column()?];
        while self.eat_if(&Token::Comma)? {
            cols.push(self.parse_result_column()?);
        }
        Ok(cols)
    }

    fn parse_result_column(&mut self) -> Result<ResultColumn> {
        if self.eat_if(&Token::Star)? {
            return Ok(ResultColumn::AllColumns);
        }

        // Parse as expression.
        let expr = self.parse_expr()?;

        // Check for table.* pattern after parsing table name as ColumnRef.
        if let Expr::ColumnRef {
            table: None,
            column: ref tname,
        } = expr
        {
            if self.eat_if(&Token::Dot)? {
                if self.eat_if(&Token::Star)? {
                    return Ok(ResultColumn::TableAllColumns(tname.clone()));
                }
                // It's table.column
                let col_name = self.parse_identifier()?;
                let qualified_expr = Expr::ColumnRef {
                    table: Some(tname.clone()),
                    column: col_name,
                };
                let alias = self.parse_optional_alias()?;
                return Ok(ResultColumn::Expr {
                    expr: qualified_expr,
                    alias,
                });
            }
        }

        let alias = self.parse_optional_alias()?;
        Ok(ResultColumn::Expr { expr, alias })
    }

    fn parse_optional_alias(&mut self) -> Result<Option<String>> {
        if self.eat_if(&Token::As)? {
            Ok(Some(self.parse_identifier()?))
        } else if let Token::Ident(_) = &self.current {
            // Implicit alias (no AS keyword).
            // Only if the next token looks like an identifier and not a keyword.
            Ok(Some(self.parse_identifier()?))
        } else if let Token::StringLiteral(_) = &self.current {
            if let Token::StringLiteral(s) = self.advance()? {
                Ok(Some(s))
            } else {
                unreachable!()
            }
        } else {
            Ok(None)
        }
    }

    fn parse_from_clause(&mut self) -> Result<FromClause> {
        let table = self.parse_table_ref()?;
        let mut joins = Vec::new();

        loop {
            let join_type = match &self.current {
                Token::Natural => {
                    self.advance()?;
                    // NATURAL [LEFT|RIGHT|INNER] JOIN
                    match &self.current {
                        Token::Left => {
                            self.advance()?;
                            self.eat_if(&Token::Outer)?;
                            self.expect(&Token::Join)?;
                            JoinType::NaturalLeft
                        }
                        Token::Right => {
                            self.advance()?;
                            self.eat_if(&Token::Outer)?;
                            self.expect(&Token::Join)?;
                            JoinType::NaturalRight
                        }
                        Token::Inner => {
                            self.advance()?;
                            self.expect(&Token::Join)?;
                            JoinType::NaturalInner
                        }
                        Token::Join => {
                            self.advance()?;
                            JoinType::NaturalInner
                        }
                        _ => {
                            return Err(RsqliteError::Parse("expected JOIN after NATURAL".into()));
                        }
                    }
                }
                Token::Inner => {
                    self.advance()?;
                    self.expect(&Token::Join)?;
                    JoinType::Inner
                }
                Token::Left => {
                    self.advance()?;
                    self.eat_if(&Token::Outer)?;
                    self.expect(&Token::Join)?;
                    JoinType::Left
                }
                Token::Right => {
                    self.advance()?;
                    self.eat_if(&Token::Outer)?;
                    self.expect(&Token::Join)?;
                    JoinType::Right
                }
                Token::Cross => {
                    self.advance()?;
                    self.expect(&Token::Join)?;
                    JoinType::Cross
                }
                Token::Join => {
                    self.advance()?;
                    JoinType::Inner
                }
                Token::Comma => {
                    self.advance()?;
                    JoinType::Cross
                }
                _ => break,
            };

            let table = self.parse_table_ref()?;
            let constraint = if self.eat_if(&Token::On)? {
                Some(JoinConstraint::On(self.parse_expr()?))
            } else if self.eat_ident_keyword("USING")? {
                self.expect(&Token::LeftParen)?;
                let mut cols = vec![self.parse_identifier()?];
                while self.eat_if(&Token::Comma)? {
                    cols.push(self.parse_identifier()?);
                }
                self.expect(&Token::RightParen)?;
                Some(JoinConstraint::Using(cols))
            } else {
                None
            };

            joins.push(JoinClause {
                join_type,
                table,
                constraint,
            });
        }

        Ok(FromClause { table, joins })
    }

    fn parse_table_ref(&mut self) -> Result<TableRef> {
        if self.eat_if(&Token::LeftParen)? {
            // Subquery or parenthesized table ref.
            if self.current == Token::Select {
                let select = self.parse_select_statement()?;
                self.expect(&Token::RightParen)?;
                // AS keyword is optional for subquery aliases.
                let _ = self.eat_if(&Token::As)?;
                let alias = self.parse_identifier()?;
                return Ok(TableRef::Subquery {
                    select: Box::new(select),
                    alias,
                });
            }
            return Err(RsqliteError::Parse("expected SELECT in subquery".into()));
        }

        let name = self.parse_identifier()?;
        let alias = if self.eat_if(&Token::As)? {
            Some(self.parse_identifier()?)
        } else if let Token::Ident(_) = &self.current {
            // Check it's not a keyword that would start a new clause.
            if !self.is_clause_keyword() {
                Some(self.parse_identifier()?)
            } else {
                None
            }
        } else {
            None
        };

        Ok(TableRef::Table { name, alias })
    }

    fn is_clause_keyword(&self) -> bool {
        matches!(
            self.current,
            Token::Where
                | Token::Group
                | Token::Having
                | Token::Order
                | Token::Limit
                | Token::Inner
                | Token::Left
                | Token::Cross
                | Token::Join
                | Token::On
                | Token::Set
                | Token::Values
        ) || self.is_ident_keyword("USING")
    }

    fn parse_order_by_item(&mut self) -> Result<OrderByItem> {
        let expr = self.parse_expr()?;
        let direction = if self.eat_if(&Token::Desc)? {
            SortDirection::Desc
        } else {
            self.eat_if(&Token::Asc)?;
            SortDirection::Asc
        };
        Ok(OrderByItem { expr, direction })
    }

    // -------------------------------------------------------------------------
    // INSERT
    // -------------------------------------------------------------------------

    fn parse_insert_statement(&mut self) -> Result<InsertStatement> {
        // Consume INSERT or REPLACE
        let is_replace = self.current == Token::Replace;
        self.advance()?;

        // Parse OR conflict-resolution if present: INSERT OR REPLACE/IGNORE/ABORT/FAIL/ROLLBACK
        let or_conflict = if is_replace {
            Some(ConflictResolution::Replace)
        } else if self.eat_if(&Token::Or)? {
            let resolution = match &self.current {
                Token::Replace => ConflictResolution::Replace,
                Token::Ignore => ConflictResolution::Ignore,
                Token::Abort => ConflictResolution::Abort,
                Token::Fail => ConflictResolution::Fail,
                Token::Rollback => ConflictResolution::Rollback,
                _ => {
                    return Err(RsqliteError::Parse(
                        "expected REPLACE, IGNORE, ABORT, FAIL, or ROLLBACK after INSERT OR".into(),
                    ));
                }
            };
            self.advance()?;
            Some(resolution)
        } else {
            None
        };

        self.eat_if(&Token::Into)?;

        let table = self.parse_identifier()?;

        // Optional column list.
        let columns = if self.eat_if(&Token::LeftParen)? {
            let mut cols = vec![self.parse_identifier()?];
            while self.eat_if(&Token::Comma)? {
                cols.push(self.parse_identifier()?);
            }
            self.expect(&Token::RightParen)?;
            Some(cols)
        } else {
            None
        };

        let source = if self.eat_if(&Token::Default)? {
            self.expect(&Token::Values)?;
            InsertSource::DefaultValues
        } else if self.eat_if(&Token::Values)? {
            let mut rows = vec![self.parse_value_row()?];
            while self.eat_if(&Token::Comma)? {
                rows.push(self.parse_value_row()?);
            }
            InsertSource::Values(rows)
        } else if self.current == Token::Select {
            let select = self.parse_select_statement()?;
            InsertSource::Select(Box::new(select))
        } else {
            return Err(RsqliteError::Parse(
                "expected VALUES, DEFAULT VALUES, or SELECT after INSERT INTO".into(),
            ));
        };

        Ok(InsertStatement {
            table,
            columns,
            source,
            or_conflict,
        })
    }

    fn parse_value_row(&mut self) -> Result<Vec<Expr>> {
        self.expect(&Token::LeftParen)?;
        let mut exprs = vec![self.parse_expr()?];
        while self.eat_if(&Token::Comma)? {
            exprs.push(self.parse_expr()?);
        }
        self.expect(&Token::RightParen)?;
        Ok(exprs)
    }

    // -------------------------------------------------------------------------
    // UPDATE
    // -------------------------------------------------------------------------

    fn parse_update_statement(&mut self) -> Result<UpdateStatement> {
        self.expect(&Token::Update)?;
        let table = self.parse_identifier()?;
        self.expect(&Token::Set)?;

        let mut assignments = vec![self.parse_assignment()?];
        while self.eat_if(&Token::Comma)? {
            assignments.push(self.parse_assignment()?);
        }

        let where_clause = if self.eat_if(&Token::Where)? {
            Some(self.parse_expr()?)
        } else {
            None
        };

        Ok(UpdateStatement {
            table,
            assignments,
            where_clause,
        })
    }

    fn parse_assignment(&mut self) -> Result<Assignment> {
        let column = self.parse_identifier()?;
        self.expect(&Token::Eq)?;
        let value = self.parse_expr()?;
        Ok(Assignment { column, value })
    }

    // -------------------------------------------------------------------------
    // DELETE
    // -------------------------------------------------------------------------

    fn parse_delete_statement(&mut self) -> Result<DeleteStatement> {
        self.expect(&Token::Delete)?;
        self.expect(&Token::From)?;
        let table = self.parse_identifier()?;

        let where_clause = if self.eat_if(&Token::Where)? {
            Some(self.parse_expr()?)
        } else {
            None
        };

        Ok(DeleteStatement {
            table,
            where_clause,
        })
    }

    // -------------------------------------------------------------------------
    // CREATE TABLE / CREATE INDEX
    // -------------------------------------------------------------------------

    fn parse_create_statement(&mut self) -> Result<Statement> {
        self.expect(&Token::Create)?;

        if self.eat_if(&Token::Unique)? {
            // CREATE UNIQUE INDEX
            self.expect(&Token::Index)?;
            return self.parse_create_index(true).map(Statement::CreateIndex);
        }

        match &self.current {
            Token::Table => {
                self.advance()?;
                self.parse_create_table().map(Statement::CreateTable)
            }
            Token::Index => {
                self.advance()?;
                self.parse_create_index(false).map(Statement::CreateIndex)
            }
            _ => Err(RsqliteError::Parse(format!(
                "expected TABLE or INDEX after CREATE, got {:?}",
                self.current
            ))),
        }
    }

    fn parse_create_table(&mut self) -> Result<CreateTableStatement> {
        let if_not_exists = if self.eat_if(&Token::If)? {
            self.expect(&Token::Not)?;
            self.expect(&Token::Exists)?;
            true
        } else {
            false
        };

        let name = self.parse_identifier()?;
        self.expect(&Token::LeftParen)?;

        let mut columns = Vec::new();
        let mut constraints = Vec::new();

        loop {
            // Check for table-level constraints.
            match &self.current {
                Token::Primary => {
                    self.advance()?;
                    self.expect(&Token::Key)?;
                    self.expect(&Token::LeftParen)?;
                    let cols = self.parse_indexed_columns()?;
                    self.expect(&Token::RightParen)?;
                    constraints.push(TableConstraint::PrimaryKey(cols));
                }
                Token::Unique => {
                    self.advance()?;
                    self.expect(&Token::LeftParen)?;
                    let cols = self.parse_indexed_columns()?;
                    self.expect(&Token::RightParen)?;
                    constraints.push(TableConstraint::Unique(cols));
                }
                Token::Check => {
                    self.advance()?;
                    self.expect(&Token::LeftParen)?;
                    let expr = self.parse_expr()?;
                    self.expect(&Token::RightParen)?;
                    constraints.push(TableConstraint::Check(expr));
                }
                Token::Foreign => {
                    self.advance()?;
                    self.expect(&Token::Key)?;
                    self.expect(&Token::LeftParen)?;
                    let mut cols = vec![self.parse_identifier()?];
                    while self.eat_if(&Token::Comma)? {
                        cols.push(self.parse_identifier()?);
                    }
                    self.expect(&Token::RightParen)?;
                    self.expect(&Token::References)?;
                    let ref_table = self.parse_identifier()?;
                    self.expect(&Token::LeftParen)?;
                    let mut ref_cols = vec![self.parse_identifier()?];
                    while self.eat_if(&Token::Comma)? {
                        ref_cols.push(self.parse_identifier()?);
                    }
                    self.expect(&Token::RightParen)?;
                    constraints.push(TableConstraint::ForeignKey {
                        columns: cols,
                        ref_table,
                        ref_columns: ref_cols,
                    });
                }
                Token::Constraint => {
                    self.advance()?;
                    // Consume constraint name.
                    self.parse_identifier()?;
                    // Continue to next iteration which will parse the actual constraint.
                    if self.eat_if(&Token::Comma)? {
                        continue;
                    }
                    continue;
                }
                _ => {
                    // Column definition.
                    columns.push(self.parse_column_def()?);
                }
            }

            if !self.eat_if(&Token::Comma)? {
                break;
            }
        }

        self.expect(&Token::RightParen)?;

        Ok(CreateTableStatement {
            if_not_exists,
            name,
            columns,
            constraints,
        })
    }

    fn parse_column_def(&mut self) -> Result<ColumnDef> {
        let name = self.parse_identifier()?;

        // Optional type name.
        let type_name = if !matches!(
            self.current,
            Token::Comma
                | Token::RightParen
                | Token::Primary
                | Token::Not
                | Token::Unique
                | Token::Default
                | Token::Check
                | Token::References
                | Token::Collate
                | Token::Constraint
        ) {
            Some(self.parse_type_name()?)
        } else {
            None
        };

        // Column constraints.
        let mut constraints = Vec::new();
        loop {
            match &self.current {
                Token::Primary => {
                    self.advance()?;
                    self.expect(&Token::Key)?;
                    let direction = if self.eat_if(&Token::Asc)? {
                        Some(SortDirection::Asc)
                    } else if self.eat_if(&Token::Desc)? {
                        Some(SortDirection::Desc)
                    } else {
                        None
                    };
                    let autoincrement = self.eat_if(&Token::Autoincrement)?;
                    constraints.push(ColumnConstraint::PrimaryKey {
                        direction,
                        autoincrement,
                    });
                }
                Token::Not => {
                    self.advance()?;
                    self.expect(&Token::Null)?;
                    constraints.push(ColumnConstraint::NotNull);
                }
                Token::Unique => {
                    self.advance()?;
                    constraints.push(ColumnConstraint::Unique);
                }
                Token::Default => {
                    self.advance()?;
                    let value = if self.eat_if(&Token::LeftParen)? {
                        let expr = self.parse_expr()?;
                        self.expect(&Token::RightParen)?;
                        expr
                    } else {
                        self.parse_literal_or_signed()?
                    };
                    constraints.push(ColumnConstraint::Default(value));
                }
                Token::Check => {
                    self.advance()?;
                    self.expect(&Token::LeftParen)?;
                    let expr = self.parse_expr()?;
                    self.expect(&Token::RightParen)?;
                    constraints.push(ColumnConstraint::Check(expr));
                }
                Token::References => {
                    self.advance()?;
                    let table = self.parse_identifier()?;
                    let columns = if self.eat_if(&Token::LeftParen)? {
                        let mut cols = vec![self.parse_identifier()?];
                        while self.eat_if(&Token::Comma)? {
                            cols.push(self.parse_identifier()?);
                        }
                        self.expect(&Token::RightParen)?;
                        cols
                    } else {
                        vec![]
                    };
                    constraints.push(ColumnConstraint::References { table, columns });
                }
                Token::Collate => {
                    self.advance()?;
                    let collation = self.parse_identifier()?;
                    constraints.push(ColumnConstraint::Collate(collation));
                }
                Token::Constraint => {
                    // Consume named constraint prefix.
                    self.advance()?;
                    self.parse_identifier()?;
                    continue;
                }
                _ => break,
            }
        }

        Ok(ColumnDef {
            name,
            type_name,
            constraints,
        })
    }

    fn parse_type_name(&mut self) -> Result<String> {
        let mut name = self.parse_identifier_preserve_case()?;

        // Handle multi-word types like "DOUBLE PRECISION", "VARYING CHARACTER(100)".
        // Keep consuming identifiers that look like type modifiers.
        while let Token::Ident(_) = &self.current {
            let next = self.parse_identifier_preserve_case()?;
            name.push(' ');
            name.push_str(&next);
        }

        // Handle optional (N) or (N,M) size spec.
        if self.eat_if(&Token::LeftParen)? {
            name.push('(');
            let n = self.parse_identifier_or_number()?;
            name.push_str(&n);
            if self.eat_if(&Token::Comma)? {
                name.push(',');
                let m = self.parse_identifier_or_number()?;
                name.push_str(&m);
            }
            self.expect(&Token::RightParen)?;
            name.push(')');
        }

        Ok(name)
    }

    fn parse_identifier_or_number(&mut self) -> Result<String> {
        match &self.current {
            Token::Ident(s) => {
                let s = s.clone();
                self.advance()?;
                Ok(s)
            }
            Token::IntegerLiteral(n) => {
                let s = n.to_string();
                self.advance()?;
                Ok(s)
            }
            _ => Err(RsqliteError::Parse(format!(
                "expected identifier or number, got {:?}",
                self.current
            ))),
        }
    }

    fn parse_literal_or_signed(&mut self) -> Result<Expr> {
        // Handle signed numbers.
        if self.eat_if(&Token::Minus)? {
            match &self.current {
                Token::IntegerLiteral(n) => {
                    let n = -*n;
                    self.advance()?;
                    Ok(Expr::Literal(LiteralValue::Integer(n)))
                }
                Token::RealLiteral(f) => {
                    let f = -*f;
                    self.advance()?;
                    Ok(Expr::Literal(LiteralValue::Real(f)))
                }
                _ => Err(RsqliteError::Parse("expected number after minus".into())),
            }
        } else if self.eat_if(&Token::Plus)? {
            match &self.current {
                Token::IntegerLiteral(n) => {
                    let n = *n;
                    self.advance()?;
                    Ok(Expr::Literal(LiteralValue::Integer(n)))
                }
                Token::RealLiteral(f) => {
                    let f = *f;
                    self.advance()?;
                    Ok(Expr::Literal(LiteralValue::Real(f)))
                }
                _ => Err(RsqliteError::Parse("expected number after plus".into())),
            }
        } else {
            self.parse_primary()
        }
    }

    fn parse_indexed_columns(&mut self) -> Result<Vec<IndexedColumn>> {
        let mut cols = vec![self.parse_indexed_column()?];
        while self.eat_if(&Token::Comma)? {
            cols.push(self.parse_indexed_column()?);
        }
        Ok(cols)
    }

    fn parse_indexed_column(&mut self) -> Result<IndexedColumn> {
        let name = self.parse_identifier()?;
        let direction = if self.eat_if(&Token::Asc)? {
            Some(SortDirection::Asc)
        } else if self.eat_if(&Token::Desc)? {
            Some(SortDirection::Desc)
        } else {
            None
        };
        Ok(IndexedColumn { name, direction })
    }

    fn parse_create_index(&mut self, unique: bool) -> Result<CreateIndexStatement> {
        let if_not_exists = if self.eat_if(&Token::If)? {
            self.expect(&Token::Not)?;
            self.expect(&Token::Exists)?;
            true
        } else {
            false
        };

        let name = self.parse_identifier()?;
        self.expect(&Token::On)?;
        let table = self.parse_identifier()?;
        self.expect(&Token::LeftParen)?;
        let columns = self.parse_indexed_columns()?;
        self.expect(&Token::RightParen)?;

        let where_clause = if self.eat_if(&Token::Where)? {
            Some(self.parse_expr()?)
        } else {
            None
        };

        Ok(CreateIndexStatement {
            unique,
            if_not_exists,
            name,
            table,
            columns,
            where_clause,
        })
    }

    // -------------------------------------------------------------------------
    // DROP TABLE / DROP INDEX
    // -------------------------------------------------------------------------

    fn parse_drop_statement(&mut self) -> Result<Statement> {
        self.expect(&Token::Drop)?;
        match &self.current {
            Token::Table => {
                self.advance()?;
                let if_exists = if self.eat_if(&Token::If)? {
                    self.expect(&Token::Exists)?;
                    true
                } else {
                    false
                };
                let name = self.parse_identifier()?;
                Ok(Statement::DropTable(DropTableStatement { if_exists, name }))
            }
            Token::Index => {
                self.advance()?;
                let if_exists = if self.eat_if(&Token::If)? {
                    self.expect(&Token::Exists)?;
                    true
                } else {
                    false
                };
                let name = self.parse_identifier()?;
                Ok(Statement::DropIndex(DropIndexStatement { if_exists, name }))
            }
            _ => Err(RsqliteError::Parse(
                "expected TABLE or INDEX after DROP".into(),
            )),
        }
    }

    // -------------------------------------------------------------------------
    // ALTER TABLE
    // -------------------------------------------------------------------------

    fn parse_alter_statement(&mut self) -> Result<AlterTableStatement> {
        self.expect(&Token::Alter)?;
        self.expect(&Token::Table)?;
        let table = self.parse_identifier()?;

        let action = if self.eat_if(&Token::Rename)? {
            if self.eat_if(&Token::To)? {
                AlterTableAction::RenameTable(self.parse_identifier()?)
            } else if self.eat_if(&Token::Column)? {
                let old = self.parse_identifier()?;
                self.expect(&Token::To)?;
                let new = self.parse_identifier()?;
                AlterTableAction::RenameColumn { old, new }
            } else {
                // RENAME old TO new (without COLUMN keyword)
                let old = self.parse_identifier()?;
                self.expect(&Token::To)?;
                let new = self.parse_identifier()?;
                AlterTableAction::RenameColumn { old, new }
            }
        } else if self.eat_if(&Token::Add)? {
            self.eat_if(&Token::Column)?;
            let col_def = self.parse_column_def()?;
            AlterTableAction::AddColumn(col_def)
        } else if self.eat_if(&Token::Drop)? {
            self.eat_if(&Token::Column)?;
            let name = self.parse_identifier()?;
            AlterTableAction::DropColumn(name)
        } else {
            return Err(RsqliteError::Parse(
                "expected RENAME, ADD, or DROP after ALTER TABLE".into(),
            ));
        };

        Ok(AlterTableStatement { table, action })
    }

    // -------------------------------------------------------------------------
    // BEGIN / COMMIT / ROLLBACK
    // -------------------------------------------------------------------------

    fn parse_begin_statement(&mut self) -> Result<Statement> {
        self.expect(&Token::Begin)?;

        let tx_type = if self.eat_ident_keyword("DEFERRED")? {
            Some(TransactionType::Deferred)
        } else if self.eat_ident_keyword("IMMEDIATE")? {
            Some(TransactionType::Immediate)
        } else if self.eat_ident_keyword("EXCLUSIVE")? {
            Some(TransactionType::Exclusive)
        } else {
            None
        };

        // Optional TRANSACTION keyword.
        self.eat_if(&Token::Transaction)?;

        Ok(Statement::Begin(tx_type))
    }

    // -------------------------------------------------------------------------
    // PRAGMA
    // -------------------------------------------------------------------------

    fn parse_pragma_statement(&mut self) -> Result<PragmaStatement> {
        self.expect(&Token::Pragma)?;
        let name = self.parse_identifier()?;

        let value = if self.eat_if(&Token::Eq)? {
            Some(self.parse_pragma_value()?)
        } else if self.eat_if(&Token::LeftParen)? {
            let val = self.parse_pragma_value()?;
            self.expect(&Token::RightParen)?;
            Some(val)
        } else {
            None
        };

        Ok(PragmaStatement { name, value })
    }

    fn parse_pragma_value(&mut self) -> Result<PragmaValue> {
        match &self.current {
            Token::IntegerLiteral(n) => {
                let n = *n;
                self.advance()?;
                Ok(PragmaValue::Number(n))
            }
            Token::StringLiteral(s) => {
                let s = s.clone();
                self.advance()?;
                Ok(PragmaValue::StringLiteral(s))
            }
            Token::Ident(s) => {
                let s = s.clone();
                self.advance()?;
                Ok(PragmaValue::Name(s))
            }
            Token::Minus => {
                self.advance()?;
                if let Token::IntegerLiteral(n) = &self.current {
                    let n = -*n;
                    self.advance()?;
                    Ok(PragmaValue::Number(n))
                } else {
                    Err(RsqliteError::Parse(
                        "expected number after minus in PRAGMA value".into(),
                    ))
                }
            }
            _ => Err(RsqliteError::Parse(format!(
                "expected PRAGMA value, got {:?}",
                self.current
            ))),
        }
    }

    // -------------------------------------------------------------------------
    // EXPLAIN
    // -------------------------------------------------------------------------

    fn parse_explain_statement(&mut self) -> Result<Statement> {
        self.expect(&Token::Explain)?;

        if self.eat_if(&Token::Query)? {
            self.expect(&Token::Plan)?;
            let inner = self.parse_statement()?;
            Ok(Statement::ExplainQueryPlan(Box::new(inner)))
        } else {
            let inner = self.parse_statement()?;
            Ok(Statement::Explain(Box::new(inner)))
        }
    }

    // -------------------------------------------------------------------------
    // Expressions (precedence climbing)
    // -------------------------------------------------------------------------

    fn parse_expr(&mut self) -> Result<Expr> {
        self.parse_or_expr()
    }

    fn parse_or_expr(&mut self) -> Result<Expr> {
        let mut left = self.parse_and_expr()?;
        while self.eat_if(&Token::Or)? {
            let right = self.parse_and_expr()?;
            left = Expr::BinaryOp {
                left: Box::new(left),
                op: BinaryOp::Or,
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_and_expr(&mut self) -> Result<Expr> {
        let mut left = self.parse_not_expr()?;
        while self.eat_if(&Token::And)? {
            let right = self.parse_not_expr()?;
            left = Expr::BinaryOp {
                left: Box::new(left),
                op: BinaryOp::And,
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_not_expr(&mut self) -> Result<Expr> {
        if self.eat_if(&Token::Not)? {
            let operand = self.parse_not_expr()?;
            Ok(Expr::UnaryOp {
                op: UnaryOp::Not,
                operand: Box::new(operand),
            })
        } else {
            self.parse_comparison()
        }
    }

    fn parse_comparison(&mut self) -> Result<Expr> {
        let mut left = self.parse_addition()?;

        loop {
            // IS [NOT] NULL / IS [NOT] expr
            if self.eat_if(&Token::Is)? {
                if self.eat_if(&Token::Not)? {
                    if self.eat_if(&Token::Null)? {
                        left = Expr::IsNull {
                            operand: Box::new(left),
                            negated: true,
                        };
                    } else {
                        let right = self.parse_addition()?;
                        left = Expr::BinaryOp {
                            left: Box::new(left),
                            op: BinaryOp::IsNot,
                            right: Box::new(right),
                        };
                    }
                } else if self.eat_if(&Token::Null)? {
                    left = Expr::IsNull {
                        operand: Box::new(left),
                        negated: false,
                    };
                } else {
                    let right = self.parse_addition()?;
                    left = Expr::BinaryOp {
                        left: Box::new(left),
                        op: BinaryOp::Is,
                        right: Box::new(right),
                    };
                }
                continue;
            }

            // [NOT] BETWEEN low AND high
            let negated = self.eat_if(&Token::Not)?;
            if self.eat_if(&Token::Between)? {
                let low = self.parse_addition()?;
                self.expect(&Token::And)?;
                let high = self.parse_addition()?;
                left = Expr::Between {
                    operand: Box::new(left),
                    low: Box::new(low),
                    high: Box::new(high),
                    negated,
                };
                continue;
            }

            // [NOT] IN (...)
            if self.eat_if(&Token::In)? {
                self.expect(&Token::LeftParen)?;
                let list = if self.current == Token::Select {
                    let select = self.parse_select_statement()?;
                    InList::Subquery(Box::new(select))
                } else {
                    let mut vals = vec![self.parse_expr()?];
                    while self.eat_if(&Token::Comma)? {
                        vals.push(self.parse_expr()?);
                    }
                    InList::Values(vals)
                };
                self.expect(&Token::RightParen)?;
                left = Expr::In {
                    operand: Box::new(left),
                    list,
                    negated,
                };
                continue;
            }

            // [NOT] LIKE pattern [ESCAPE escape]
            if self.eat_if(&Token::Like)? {
                let pattern = self.parse_addition()?;
                let escape = if self.eat_if(&Token::Escape)? {
                    Some(Box::new(self.parse_addition()?))
                } else {
                    None
                };
                left = Expr::Like {
                    operand: Box::new(left),
                    pattern: Box::new(pattern),
                    escape,
                    negated,
                };
                continue;
            }

            // [NOT] GLOB pattern
            if self.eat_if(&Token::Glob)? {
                let pattern = self.parse_addition()?;
                left = Expr::BinaryOp {
                    left: Box::new(left),
                    op: if negated {
                        // NOT GLOB: we'll wrap in NOT
                        BinaryOp::Glob
                    } else {
                        BinaryOp::Glob
                    },
                    right: Box::new(pattern),
                };
                if negated {
                    left = Expr::UnaryOp {
                        op: UnaryOp::Not,
                        operand: Box::new(left),
                    };
                }
                continue;
            }

            // If we consumed NOT but didn't match BETWEEN/IN/LIKE/GLOB, put it back.
            if negated {
                // This is a parse error: NOT without a valid following keyword.
                return Err(RsqliteError::Parse(
                    "expected BETWEEN, IN, LIKE, or GLOB after NOT".into(),
                ));
            }

            // Standard comparison operators.
            let op = match &self.current {
                Token::Eq => Some(BinaryOp::Eq),
                Token::NotEq => Some(BinaryOp::NotEq),
                Token::Lt => Some(BinaryOp::Lt),
                Token::Gt => Some(BinaryOp::Gt),
                Token::Le => Some(BinaryOp::Le),
                Token::Ge => Some(BinaryOp::Ge),
                _ => None,
            };

            if let Some(op) = op {
                self.advance()?;
                let right = self.parse_addition()?;
                left = Expr::BinaryOp {
                    left: Box::new(left),
                    op,
                    right: Box::new(right),
                };
            } else {
                break;
            }
        }

        Ok(left)
    }

    fn parse_addition(&mut self) -> Result<Expr> {
        let mut left = self.parse_multiplication()?;
        loop {
            let op = match &self.current {
                Token::Plus => Some(BinaryOp::Add),
                Token::Minus => Some(BinaryOp::Subtract),
                Token::Concat => Some(BinaryOp::Concat),
                Token::BitAnd => Some(BinaryOp::BitAnd),
                Token::BitOr => Some(BinaryOp::BitOr),
                Token::ShiftLeft => Some(BinaryOp::ShiftLeft),
                Token::ShiftRight => Some(BinaryOp::ShiftRight),
                _ => None,
            };
            if let Some(op) = op {
                self.advance()?;
                let right = self.parse_multiplication()?;
                left = Expr::BinaryOp {
                    left: Box::new(left),
                    op,
                    right: Box::new(right),
                };
            } else {
                break;
            }
        }
        Ok(left)
    }

    fn parse_multiplication(&mut self) -> Result<Expr> {
        let mut left = self.parse_unary()?;
        loop {
            let op = match &self.current {
                Token::Star => Some(BinaryOp::Multiply),
                Token::Slash => Some(BinaryOp::Divide),
                Token::Percent => Some(BinaryOp::Modulo),
                _ => None,
            };
            if let Some(op) = op {
                self.advance()?;
                let right = self.parse_unary()?;
                left = Expr::BinaryOp {
                    left: Box::new(left),
                    op,
                    right: Box::new(right),
                };
            } else {
                break;
            }
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Expr> {
        match &self.current {
            Token::Minus => {
                self.advance()?;
                let operand = self.parse_unary()?;
                Ok(Expr::UnaryOp {
                    op: UnaryOp::Negate,
                    operand: Box::new(operand),
                })
            }
            Token::Plus => {
                self.advance()?;
                let operand = self.parse_unary()?;
                Ok(Expr::UnaryOp {
                    op: UnaryOp::Plus,
                    operand: Box::new(operand),
                })
            }
            Token::BitNot => {
                self.advance()?;
                let operand = self.parse_unary()?;
                Ok(Expr::UnaryOp {
                    op: UnaryOp::BitwiseNot,
                    operand: Box::new(operand),
                })
            }
            _ => self.parse_postfix(),
        }
    }

    fn parse_postfix(&mut self) -> Result<Expr> {
        let mut expr = self.parse_primary()?;

        // Handle COLLATE postfix.
        if self.eat_if(&Token::Collate)? {
            let collation = self.parse_identifier()?;
            expr = Expr::Collate {
                expr: Box::new(expr),
                collation,
            };
        }

        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<Expr> {
        match &self.current {
            // Integer literal
            Token::IntegerLiteral(n) => {
                let n = *n;
                self.advance()?;
                Ok(Expr::Literal(LiteralValue::Integer(n)))
            }
            // Float literal
            Token::RealLiteral(f) => {
                let f = *f;
                self.advance()?;
                Ok(Expr::Literal(LiteralValue::Real(f)))
            }
            // String literal
            Token::StringLiteral(s) => {
                let s = s.clone();
                self.advance()?;
                Ok(Expr::Literal(LiteralValue::String(s)))
            }
            // Blob literal
            Token::BlobLiteral(b) => {
                let b = b.clone();
                self.advance()?;
                Ok(Expr::Literal(LiteralValue::Blob(b)))
            }
            // NULL
            Token::Null => {
                self.advance()?;
                Ok(Expr::Literal(LiteralValue::Null))
            }
            // CURRENT_TIME / CURRENT_DATE / CURRENT_TIMESTAMP
            Token::CurrentTime => {
                self.advance()?;
                Ok(Expr::Literal(LiteralValue::CurrentTime))
            }
            Token::CurrentDate => {
                self.advance()?;
                Ok(Expr::Literal(LiteralValue::CurrentDate))
            }
            Token::CurrentTimestamp => {
                self.advance()?;
                Ok(Expr::Literal(LiteralValue::CurrentTimestamp))
            }
            // Bind parameter
            Token::QuestionMark => {
                self.advance()?;
                // Check for ?NNN
                if let Token::IntegerLiteral(n) = &self.current {
                    let n = *n as u32;
                    self.advance()?;
                    Ok(Expr::BindParameter(Some(n)))
                } else {
                    Ok(Expr::BindParameter(None))
                }
            }
            // CASE expression
            Token::Case => self.parse_case_expr(),
            // CAST expression
            Token::Cast => self.parse_cast_expr(),
            // EXISTS subquery
            Token::Exists => {
                self.advance()?;
                self.expect(&Token::LeftParen)?;
                let select = self.parse_select_statement()?;
                self.expect(&Token::RightParen)?;
                Ok(Expr::Exists {
                    subquery: Box::new(select),
                    negated: false,
                })
            }
            Token::Not => {
                // NOT EXISTS
                self.advance()?;
                if self.eat_if(&Token::Exists)? {
                    self.expect(&Token::LeftParen)?;
                    let select = self.parse_select_statement()?;
                    self.expect(&Token::RightParen)?;
                    Ok(Expr::Exists {
                        subquery: Box::new(select),
                        negated: true,
                    })
                } else {
                    let operand = self.parse_not_expr()?;
                    Ok(Expr::UnaryOp {
                        op: UnaryOp::Not,
                        operand: Box::new(operand),
                    })
                }
            }
            // Parenthesized expression or subquery
            Token::LeftParen => {
                self.advance()?;
                if self.current == Token::Select {
                    let select = self.parse_select_statement()?;
                    self.expect(&Token::RightParen)?;
                    Ok(Expr::Subquery(Box::new(select)))
                } else {
                    let expr = self.parse_expr()?;
                    self.expect(&Token::RightParen)?;
                    Ok(Expr::Parenthesized(Box::new(expr)))
                }
            }
            // Identifier or keyword-as-identifier (column ref or function call)
            Token::Ident(_)
            | Token::Replace
            | Token::Abort
            | Token::Conflict
            | Token::Fail
            | Token::Ignore
            | Token::Table
            | Token::Index
            | Token::Key
            | Token::Column
            | Token::IntegerKw
            | Token::RealKw
            | Token::TextKw
            | Token::BlobKw
            | Token::NumericKw
            | Token::Plan
            | Token::Query
            | Token::Rename
            | Token::Savepoint
            | Token::Transaction
            | Token::Release
            | Token::Recursive
            | Token::Right
            | Token::Outer => {
                let name = self.parse_identifier()?;

                // Check for function call: name(...)
                if self.eat_if(&Token::LeftParen)? {
                    let args = self.parse_function_args()?;
                    self.expect(&Token::RightParen)?;
                    return Ok(Expr::FunctionCall { name, args });
                }

                // Check for table.column
                if self.eat_if(&Token::Dot)? {
                    let column = self.parse_identifier()?;
                    return Ok(Expr::ColumnRef {
                        table: Some(name),
                        column,
                    });
                }

                Ok(Expr::ColumnRef {
                    table: None,
                    column: name,
                })
            }
            _ => Err(RsqliteError::Parse(format!(
                "unexpected token in expression: {:?}",
                self.current
            ))),
        }
    }

    fn parse_case_expr(&mut self) -> Result<Expr> {
        self.expect(&Token::Case)?;

        let operand = if self.current != Token::When {
            Some(Box::new(self.parse_expr()?))
        } else {
            None
        };

        let mut when_clauses = Vec::new();
        while self.eat_if(&Token::When)? {
            let when_expr = self.parse_expr()?;
            self.expect(&Token::Then)?;
            let then_expr = self.parse_expr()?;
            when_clauses.push((when_expr, then_expr));
        }

        let else_clause = if self.eat_if(&Token::Else)? {
            Some(Box::new(self.parse_expr()?))
        } else {
            None
        };

        self.expect(&Token::End)?;

        Ok(Expr::Case {
            operand,
            when_clauses,
            else_clause,
        })
    }

    fn parse_cast_expr(&mut self) -> Result<Expr> {
        self.expect(&Token::Cast)?;
        self.expect(&Token::LeftParen)?;
        let expr = self.parse_expr()?;
        self.expect(&Token::As)?;
        let type_name = self.parse_type_name()?;
        self.expect(&Token::RightParen)?;
        Ok(Expr::Cast {
            expr: Box::new(expr),
            type_name,
        })
    }

    fn parse_function_args(&mut self) -> Result<FunctionArgs> {
        if self.eat_if(&Token::Star)? {
            return Ok(FunctionArgs::Wildcard);
        }

        if self.current == Token::RightParen {
            return Ok(FunctionArgs::Exprs {
                distinct: false,
                args: vec![],
            });
        }

        let distinct = self.eat_if(&Token::Distinct)?;
        let mut args = vec![self.parse_expr()?];
        while self.eat_if(&Token::Comma)? {
            args.push(self.parse_expr()?);
        }

        Ok(FunctionArgs::Exprs { distinct, args })
    }

    /// Like parse_identifier but preserves the original case for keyword tokens.
    /// Used for type names where we want "INTEGER" not "integer".
    fn parse_identifier_preserve_case(&mut self) -> Result<String> {
        match &self.current {
            Token::Ident(s) => {
                let s = s.clone();
                self.advance()?;
                Ok(s)
            }
            Token::Table
            | Token::Index
            | Token::Key
            | Token::Column
            | Token::IntegerKw
            | Token::RealKw
            | Token::TextKw
            | Token::BlobKw
            | Token::NumericKw
            | Token::Replace
            | Token::Abort
            | Token::Conflict
            | Token::Fail
            | Token::Ignore
            | Token::Plan
            | Token::Query
            | Token::Rename
            | Token::Savepoint
            | Token::Transaction
            | Token::Release
            | Token::Recursive
            | Token::Right
            | Token::Outer => {
                let name = format!("{}", self.current).to_uppercase();
                self.advance()?;
                Ok(name)
            }
            _ => Err(RsqliteError::Parse(format!(
                "expected identifier, got {:?}",
                self.current
            ))),
        }
    }

    fn parse_identifier(&mut self) -> Result<String> {
        match &self.current {
            Token::Ident(s) => {
                let s = s.clone();
                self.advance()?;
                Ok(s)
            }
            // Allow certain keywords to be used as identifiers.
            // Use Display impl to get the keyword text as lowercase.
            Token::Table
            | Token::Index
            | Token::Key
            | Token::Column
            | Token::IntegerKw
            | Token::RealKw
            | Token::TextKw
            | Token::BlobKw
            | Token::NumericKw
            | Token::Replace
            | Token::Abort
            | Token::Conflict
            | Token::Fail
            | Token::Ignore
            | Token::Plan
            | Token::Query
            | Token::Rename
            | Token::Savepoint
            | Token::Transaction
            | Token::Release
            | Token::Recursive
            | Token::Right
            | Token::Outer => {
                let name = format!("{}", self.current).to_lowercase();
                self.advance()?;
                Ok(name)
            }
            _ => Err(RsqliteError::Parse(format!(
                "expected identifier, got {:?}",
                self.current
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_select_star() {
        let stmt = parse("SELECT * FROM users").unwrap();
        if let Statement::Select(s) = stmt {
            assert_eq!(s.columns.len(), 1);
            assert!(matches!(s.columns[0], ResultColumn::AllColumns));
            assert!(s.from.is_some());
        } else {
            panic!("expected SELECT statement");
        }
    }

    #[test]
    fn test_parse_select_columns() {
        let stmt = parse("SELECT id, name FROM users").unwrap();
        if let Statement::Select(s) = stmt {
            assert_eq!(s.columns.len(), 2);
        } else {
            panic!("expected SELECT statement");
        }
    }

    #[test]
    fn test_parse_select_where() {
        let stmt = parse("SELECT * FROM users WHERE age > 18").unwrap();
        if let Statement::Select(s) = stmt {
            assert!(s.where_clause.is_some());
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn test_parse_select_order_by() {
        let stmt = parse("SELECT * FROM items ORDER BY price DESC").unwrap();
        if let Statement::Select(s) = stmt {
            assert!(s.order_by.is_some());
            let ob = s.order_by.unwrap();
            assert_eq!(ob.len(), 1);
            assert_eq!(ob[0].direction, SortDirection::Desc);
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn test_parse_select_limit_offset() {
        let stmt = parse("SELECT * FROM t LIMIT 10 OFFSET 5").unwrap();
        if let Statement::Select(s) = stmt {
            let lim = s.limit.unwrap();
            assert!(matches!(
                lim.limit,
                Expr::Literal(LiteralValue::Integer(10))
            ));
            assert!(lim.offset.is_some());
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn test_parse_select_with_alias() {
        let stmt = parse("SELECT id AS user_id FROM users").unwrap();
        if let Statement::Select(s) = stmt {
            if let ResultColumn::Expr { alias, .. } = &s.columns[0] {
                assert_eq!(alias.as_deref(), Some("user_id"));
            } else {
                panic!("expected Expr column");
            }
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn test_parse_select_distinct() {
        let stmt = parse("SELECT DISTINCT name FROM users").unwrap();
        if let Statement::Select(s) = stmt {
            assert!(s.distinct);
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn test_parse_select_group_by() {
        let stmt = parse("SELECT name, COUNT(*) FROM users GROUP BY name").unwrap();
        if let Statement::Select(s) = stmt {
            assert!(s.group_by.is_some());
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn test_parse_select_having() {
        let stmt =
            parse("SELECT name, COUNT(*) FROM users GROUP BY name HAVING COUNT(*) > 1").unwrap();
        if let Statement::Select(s) = stmt {
            assert!(s.having.is_some());
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn test_parse_select_join() {
        let stmt =
            parse("SELECT u.name, o.total FROM users u INNER JOIN orders o ON u.id = o.user_id")
                .unwrap();
        if let Statement::Select(s) = stmt {
            let from = s.from.unwrap();
            assert_eq!(from.joins.len(), 1);
            assert_eq!(from.joins[0].join_type, JoinType::Inner);
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn test_parse_select_left_join() {
        let stmt = parse("SELECT * FROM a LEFT JOIN b ON a.id = b.a_id").unwrap();
        if let Statement::Select(s) = stmt {
            let from = s.from.unwrap();
            assert_eq!(from.joins.len(), 1);
            assert_eq!(from.joins[0].join_type, JoinType::Left);
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn test_parse_insert_values() {
        let stmt = parse("INSERT INTO users (name, age) VALUES ('Alice', 30)").unwrap();
        if let Statement::Insert(ins) = stmt {
            assert_eq!(ins.table, "users");
            assert_eq!(ins.columns, Some(vec!["name".into(), "age".into()]));
            if let InsertSource::Values(rows) = ins.source {
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0].len(), 2);
            } else {
                panic!("expected VALUES");
            }
        } else {
            panic!("expected INSERT");
        }
    }

    #[test]
    fn test_parse_insert_multi_row() {
        let stmt = parse("INSERT INTO t VALUES (1), (2), (3)").unwrap();
        if let Statement::Insert(ins) = stmt {
            if let InsertSource::Values(rows) = ins.source {
                assert_eq!(rows.len(), 3);
            } else {
                panic!("expected VALUES");
            }
        } else {
            panic!("expected INSERT");
        }
    }

    #[test]
    fn test_parse_insert_default_values() {
        let stmt = parse("INSERT INTO t DEFAULT VALUES").unwrap();
        if let Statement::Insert(ins) = stmt {
            assert!(matches!(ins.source, InsertSource::DefaultValues));
        } else {
            panic!("expected INSERT");
        }
    }

    #[test]
    fn test_parse_update() {
        let stmt = parse("UPDATE users SET name = 'Bob' WHERE id = 1").unwrap();
        if let Statement::Update(upd) = stmt {
            assert_eq!(upd.table, "users");
            assert_eq!(upd.assignments.len(), 1);
            assert_eq!(upd.assignments[0].column, "name");
            assert!(upd.where_clause.is_some());
        } else {
            panic!("expected UPDATE");
        }
    }

    #[test]
    fn test_parse_delete() {
        let stmt = parse("DELETE FROM users WHERE id = 1").unwrap();
        if let Statement::Delete(del) = stmt {
            assert_eq!(del.table, "users");
            assert!(del.where_clause.is_some());
        } else {
            panic!("expected DELETE");
        }
    }

    #[test]
    fn test_parse_create_table() {
        let stmt =
            parse("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, age INTEGER)")
                .unwrap();
        if let Statement::CreateTable(ct) = stmt {
            assert_eq!(ct.name, "users");
            assert_eq!(ct.columns.len(), 3);
            assert_eq!(ct.columns[0].name, "id");
            assert_eq!(ct.columns[1].name, "name");
            assert_eq!(ct.columns[2].name, "age");
        } else {
            panic!("expected CREATE TABLE");
        }
    }

    #[test]
    fn test_parse_create_table_if_not_exists() {
        let stmt = parse("CREATE TABLE IF NOT EXISTS t (id INTEGER)").unwrap();
        if let Statement::CreateTable(ct) = stmt {
            assert!(ct.if_not_exists);
        } else {
            panic!("expected CREATE TABLE");
        }
    }

    #[test]
    fn test_parse_create_index() {
        let stmt = parse("CREATE INDEX idx_name ON users (name)").unwrap();
        if let Statement::CreateIndex(ci) = stmt {
            assert!(!ci.unique);
            assert_eq!(ci.name, "idx_name");
            assert_eq!(ci.table, "users");
            assert_eq!(ci.columns.len(), 1);
        } else {
            panic!("expected CREATE INDEX");
        }
    }

    #[test]
    fn test_parse_create_unique_index() {
        let stmt = parse("CREATE UNIQUE INDEX idx ON t (a, b)").unwrap();
        if let Statement::CreateIndex(ci) = stmt {
            assert!(ci.unique);
            assert_eq!(ci.columns.len(), 2);
        } else {
            panic!("expected CREATE INDEX");
        }
    }

    #[test]
    fn test_parse_drop_table() {
        let stmt = parse("DROP TABLE IF EXISTS users").unwrap();
        if let Statement::DropTable(dt) = stmt {
            assert!(dt.if_exists);
            assert_eq!(dt.name, "users");
        } else {
            panic!("expected DROP TABLE");
        }
    }

    #[test]
    fn test_parse_drop_index() {
        let stmt = parse("DROP INDEX idx_name").unwrap();
        if let Statement::DropIndex(di) = stmt {
            assert!(!di.if_exists);
            assert_eq!(di.name, "idx_name");
        } else {
            panic!("expected DROP INDEX");
        }
    }

    #[test]
    fn test_parse_alter_rename_table() {
        let stmt = parse("ALTER TABLE users RENAME TO customers").unwrap();
        if let Statement::AlterTable(at) = stmt {
            assert_eq!(at.table, "users");
            assert!(matches!(at.action, AlterTableAction::RenameTable(ref n) if n == "customers"));
        } else {
            panic!("expected ALTER TABLE");
        }
    }

    #[test]
    fn test_parse_alter_add_column() {
        let stmt = parse("ALTER TABLE users ADD COLUMN email TEXT").unwrap();
        if let Statement::AlterTable(at) = stmt {
            if let AlterTableAction::AddColumn(col) = at.action {
                assert_eq!(col.name, "email");
            } else {
                panic!("expected ADD COLUMN");
            }
        } else {
            panic!("expected ALTER TABLE");
        }
    }

    #[test]
    fn test_parse_alter_drop_column() {
        let stmt = parse("ALTER TABLE users DROP COLUMN age").unwrap();
        if let Statement::AlterTable(at) = stmt {
            assert!(matches!(at.action, AlterTableAction::DropColumn(ref n) if n == "age"));
        } else {
            panic!("expected ALTER TABLE");
        }
    }

    #[test]
    fn test_parse_begin_commit_rollback() {
        let stmt = parse("BEGIN").unwrap();
        assert!(matches!(stmt, Statement::Begin(None)));

        let stmt = parse("BEGIN IMMEDIATE").unwrap();
        assert!(matches!(
            stmt,
            Statement::Begin(Some(TransactionType::Immediate))
        ));

        let stmt = parse("COMMIT").unwrap();
        assert!(matches!(stmt, Statement::Commit));

        let stmt = parse("ROLLBACK").unwrap();
        assert!(matches!(stmt, Statement::Rollback));
    }

    #[test]
    fn test_parse_pragma() {
        let stmt = parse("PRAGMA page_size").unwrap();
        if let Statement::Pragma(p) = stmt {
            assert_eq!(p.name, "page_size");
            assert!(p.value.is_none());
        } else {
            panic!("expected PRAGMA");
        }
    }

    #[test]
    fn test_parse_pragma_with_value() {
        let stmt = parse("PRAGMA page_size = 4096").unwrap();
        if let Statement::Pragma(p) = stmt {
            assert_eq!(p.name, "page_size");
            assert!(matches!(p.value, Some(PragmaValue::Number(4096))));
        } else {
            panic!("expected PRAGMA");
        }
    }

    #[test]
    fn test_parse_pragma_with_paren() {
        let stmt = parse("PRAGMA table_info(users)").unwrap();
        if let Statement::Pragma(p) = stmt {
            assert_eq!(p.name, "table_info");
            assert!(matches!(p.value, Some(PragmaValue::Name(ref n)) if n == "users"));
        } else {
            panic!("expected PRAGMA");
        }
    }

    #[test]
    fn test_parse_explain() {
        let stmt = parse("EXPLAIN SELECT * FROM t").unwrap();
        assert!(matches!(stmt, Statement::Explain(_)));
    }

    #[test]
    fn test_parse_explain_query_plan() {
        let stmt = parse("EXPLAIN QUERY PLAN SELECT * FROM t").unwrap();
        assert!(matches!(stmt, Statement::ExplainQueryPlan(_)));
    }

    #[test]
    fn test_parse_expr_between() {
        let stmt = parse("SELECT * FROM t WHERE x BETWEEN 1 AND 10").unwrap();
        if let Statement::Select(s) = stmt {
            assert!(matches!(
                s.where_clause,
                Some(Expr::Between { negated: false, .. })
            ));
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn test_parse_expr_in_values() {
        let stmt = parse("SELECT * FROM t WHERE x IN (1, 2, 3)").unwrap();
        if let Statement::Select(s) = stmt {
            assert!(matches!(
                s.where_clause,
                Some(Expr::In { negated: false, .. })
            ));
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn test_parse_expr_like() {
        let stmt = parse("SELECT * FROM t WHERE name LIKE '%alice%'").unwrap();
        if let Statement::Select(s) = stmt {
            assert!(matches!(
                s.where_clause,
                Some(Expr::Like { negated: false, .. })
            ));
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn test_parse_expr_not_like() {
        let stmt = parse("SELECT * FROM t WHERE name NOT LIKE '%test%'").unwrap();
        if let Statement::Select(s) = stmt {
            assert!(matches!(
                s.where_clause,
                Some(Expr::Like { negated: true, .. })
            ));
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn test_parse_expr_is_null() {
        let stmt = parse("SELECT * FROM t WHERE x IS NULL").unwrap();
        if let Statement::Select(s) = stmt {
            assert!(matches!(
                s.where_clause,
                Some(Expr::IsNull { negated: false, .. })
            ));
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn test_parse_expr_is_not_null() {
        let stmt = parse("SELECT * FROM t WHERE x IS NOT NULL").unwrap();
        if let Statement::Select(s) = stmt {
            assert!(matches!(
                s.where_clause,
                Some(Expr::IsNull { negated: true, .. })
            ));
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn test_parse_case_expr() {
        let stmt =
            parse("SELECT CASE WHEN x > 0 THEN 'positive' ELSE 'non-positive' END FROM t").unwrap();
        if let Statement::Select(s) = stmt {
            if let ResultColumn::Expr { expr, .. } = &s.columns[0] {
                assert!(matches!(expr, Expr::Case { .. }));
            } else {
                panic!("expected Expr column");
            }
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn test_parse_cast_expr() {
        let stmt = parse("SELECT CAST(x AS INTEGER) FROM t").unwrap();
        if let Statement::Select(s) = stmt {
            if let ResultColumn::Expr { expr, .. } = &s.columns[0] {
                assert!(matches!(expr, Expr::Cast { .. }));
            } else {
                panic!("expected Expr column");
            }
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn test_parse_function_call() {
        let stmt = parse("SELECT COUNT(*) FROM t").unwrap();
        if let Statement::Select(s) = stmt {
            if let ResultColumn::Expr { expr, .. } = &s.columns[0] {
                if let Expr::FunctionCall { name, args } = expr {
                    assert_eq!(name, "COUNT");
                    assert!(matches!(args, FunctionArgs::Wildcard));
                } else {
                    panic!("expected FunctionCall");
                }
            } else {
                panic!("expected Expr column");
            }
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn test_parse_subquery_in_where() {
        let stmt = parse("SELECT * FROM t WHERE id IN (SELECT id FROM t2)").unwrap();
        if let Statement::Select(s) = stmt {
            if let Some(Expr::In { list, .. }) = &s.where_clause {
                assert!(matches!(list, InList::Subquery(_)));
            } else {
                panic!("expected IN with subquery");
            }
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn test_parse_nested_arithmetic() {
        let stmt = parse("SELECT (1 + 2) * 3").unwrap();
        if let Statement::Select(s) = stmt {
            if let ResultColumn::Expr { expr, .. } = &s.columns[0] {
                // Should be Multiply(Parenthesized(Add(1,2)), 3)
                assert!(matches!(
                    expr,
                    Expr::BinaryOp {
                        op: BinaryOp::Multiply,
                        ..
                    }
                ));
            } else {
                panic!("expected Expr column");
            }
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn test_parse_error_missing_from() {
        // This should still parse (SELECT without FROM is valid).
        let result = parse("SELECT 1 + 2");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_error_garbage() {
        let result = parse("GARBAGE INPUT");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_select_no_from() {
        let stmt = parse("SELECT 42").unwrap();
        if let Statement::Select(s) = stmt {
            assert!(s.from.is_none());
            assert_eq!(s.columns.len(), 1);
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn test_parse_create_table_with_autoincrement() {
        let stmt =
            parse("CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT)").unwrap();
        if let Statement::CreateTable(ct) = stmt {
            assert_eq!(ct.columns[0].name, "id");
            let pk = &ct.columns[0].constraints[0];
            assert!(matches!(
                pk,
                ColumnConstraint::PrimaryKey {
                    autoincrement: true,
                    ..
                }
            ));
        } else {
            panic!("expected CREATE TABLE");
        }
    }

    #[test]
    fn test_parse_insert_select() {
        let stmt = parse("INSERT INTO t2 SELECT * FROM t1").unwrap();
        if let Statement::Insert(ins) = stmt {
            assert!(matches!(ins.source, InsertSource::Select(_)));
        } else {
            panic!("expected INSERT");
        }
    }

    #[test]
    fn test_parse_select_with_using_join() {
        let stmt = parse("SELECT * FROM a JOIN b USING (id)").unwrap();
        if let Statement::Select(s) = stmt {
            let from = s.from.unwrap();
            assert_eq!(from.joins.len(), 1);
            assert!(matches!(
                from.joins[0].constraint,
                Some(JoinConstraint::Using(_))
            ));
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn test_parse_exists_subquery() {
        let stmt =
            parse("SELECT * FROM t WHERE EXISTS (SELECT 1 FROM t2 WHERE t2.id = t.id)").unwrap();
        if let Statement::Select(s) = stmt {
            assert!(matches!(
                s.where_clause,
                Some(Expr::Exists { negated: false, .. })
            ));
        } else {
            panic!("expected SELECT");
        }
    }
}
