/// SQL Parser - recursive descent parser producing a typed AST.
use crate::ast::*;
use crate::error::{Result, RsqliteError};
use crate::tokenizer::Token;

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Parser { tokens, pos: 0 }
    }

    pub fn parse(&mut self) -> Result<Statement> {
        let stmt = self.parse_statement()?;
        // Consume optional semicolon
        if self.check(&Token::Semicolon) {
            self.advance();
        }
        Ok(stmt)
    }

    pub fn parse_multiple(&mut self) -> Result<Vec<Statement>> {
        let mut stmts = Vec::new();
        while !self.check(&Token::Eof) {
            if self.check(&Token::Semicolon) {
                self.advance();
                continue;
            }
            stmts.push(self.parse_statement()?);
            if self.check(&Token::Semicolon) {
                self.advance();
            }
        }
        Ok(stmts)
    }

    fn parse_statement(&mut self) -> Result<Statement> {
        match self.peek() {
            Token::Select => self.parse_select_stmt(),
            Token::Insert | Token::Replace => self.parse_insert_stmt(),
            Token::Update => self.parse_update_stmt(),
            Token::Delete => self.parse_delete_stmt(),
            Token::Create => self.parse_create_stmt(),
            Token::Drop => self.parse_drop_stmt(),
            Token::Alter => self.parse_alter_stmt(),
            Token::Begin => {
                self.advance();
                // Optional TRANSACTION
                if self.check(&Token::Transaction) {
                    self.advance();
                }
                Ok(Statement::Begin)
            }
            Token::Commit | Token::End => {
                self.advance();
                if self.check(&Token::Transaction) {
                    self.advance();
                }
                Ok(Statement::Commit)
            }
            Token::Rollback => {
                self.advance();
                if self.check(&Token::Transaction) {
                    self.advance();
                }
                Ok(Statement::Rollback)
            }
            Token::Explain => {
                self.advance();
                if self.check(&Token::Query) {
                    self.advance();
                    self.expect(&Token::Plan)?;
                    let stmt = self.parse_statement()?;
                    Ok(Statement::ExplainQueryPlan(Box::new(stmt)))
                } else {
                    let stmt = self.parse_statement()?;
                    Ok(Statement::Explain(Box::new(stmt)))
                }
            }
            Token::Pragma => self.parse_pragma_stmt(),
            _ => Err(RsqliteError::Parse(format!(
                "Unexpected token: {:?}",
                self.peek()
            ))),
        }
    }

    fn parse_select_stmt(&mut self) -> Result<Statement> {
        let select = self.parse_select()?;
        Ok(Statement::Select(Box::new(select)))
    }

    fn parse_select(&mut self) -> Result<SelectStatement> {
        self.expect(&Token::Select)?;

        let distinct = if self.check(&Token::Distinct) {
            self.advance();
            true
        } else {
            if self.check(&Token::All) {
                self.advance();
            }
            false
        };

        let columns = self.parse_select_columns()?;

        let from = if self.check(&Token::From) {
            self.advance();
            Some(self.parse_from()?)
        } else {
            None
        };

        let where_clause = if self.check(&Token::Where) {
            self.advance();
            Some(self.parse_expr()?)
        } else {
            None
        };

        let group_by = if self.check(&Token::Group) {
            self.advance();
            self.expect(&Token::By)?;
            self.parse_expr_list()?
        } else {
            Vec::new()
        };

        let having = if self.check(&Token::Having) {
            self.advance();
            Some(self.parse_expr()?)
        } else {
            None
        };

        let order_by = if self.check(&Token::Order) {
            self.advance();
            self.expect(&Token::By)?;
            self.parse_order_by_list()?
        } else {
            Vec::new()
        };

        let limit = if self.check(&Token::Limit) {
            self.advance();
            Some(self.parse_expr()?)
        } else {
            None
        };

        let offset = if self.check(&Token::Offset) {
            self.advance();
            Some(self.parse_expr()?)
        } else if limit.is_some() && self.check(&Token::Comma) {
            // LIMIT x, y syntax (x is offset, y is limit)
            // Actually in SQLite: LIMIT count OFFSET offset
            // or LIMIT offset, count
            self.advance();
            let count = self.parse_expr()?;
            // Swap: the first expr was actually offset
            // We need to return (limit=count, offset=original_limit)
            // This is handled by returning count as the new limit
            // and the original limit value as offset
            let offset_expr = limit.clone();
            let compound = self.parse_compound_tail()?;
            return Ok(SelectStatement {
                distinct,
                columns,
                from,
                where_clause,
                group_by,
                having,
                order_by,
                limit: Some(count),
                offset: offset_expr,
                compound,
            });
        } else {
            None
        };

        let mut sel = SelectStatement {
            distinct,
            columns,
            from,
            where_clause,
            group_by,
            having,
            order_by,
            limit,
            offset,
            compound: None,
        };

        // Check for compound operators: UNION, EXCEPT, INTERSECT
        sel.compound = self.parse_compound_tail()?;

        Ok(sel)
    }

    fn parse_compound_tail(&mut self) -> Result<Option<Box<CompoundSelect>>> {
        let op = if self.check(&Token::Union) {
            self.advance();
            if self.check(&Token::All) {
                self.advance();
                CompoundOp::UnionAll
            } else {
                CompoundOp::Union
            }
        } else if self.check(&Token::Except) {
            self.advance();
            CompoundOp::Except
        } else if self.check(&Token::Intersect) {
            self.advance();
            CompoundOp::Intersect
        } else {
            return Ok(None);
        };

        let select = self.parse_select()?;
        Ok(Some(Box::new(CompoundSelect { op, select })))
    }

    fn parse_select_columns(&mut self) -> Result<Vec<SelectColumn>> {
        let mut columns = Vec::new();
        loop {
            if self.check(&Token::Star) {
                self.advance();
                columns.push(SelectColumn::AllColumns);
            } else {
                let expr = self.parse_expr()?;

                // Check for table.* pattern
                if let Expr::Column {
                    table: Some(t),
                    name,
                } = &expr
                {
                    if name == "*" {
                        columns.push(SelectColumn::TableAllColumns(t.clone()));
                        if !self.check(&Token::Comma) {
                            break;
                        }
                        self.advance();
                        continue;
                    }
                }

                let alias = if self.check(&Token::As) {
                    self.advance();
                    Some(self.parse_identifier()?)
                } else if matches!(self.peek(), Token::Identifier(_))
                    && !self.is_keyword_at_current()
                {
                    Some(self.parse_identifier()?)
                } else {
                    None
                };

                columns.push(SelectColumn::Expr { expr, alias });
            }

            if !self.check(&Token::Comma) {
                break;
            }
            self.advance();
        }
        Ok(columns)
    }

    fn is_keyword_at_current(&self) -> bool {
        matches!(
            self.peek(),
            Token::From
                | Token::Where
                | Token::Group
                | Token::Having
                | Token::Order
                | Token::Limit
                | Token::Offset
                | Token::Join
                | Token::Inner
                | Token::Left
                | Token::Cross
                | Token::On
                | Token::Union
                | Token::Except
                | Token::Intersect
        )
    }

    fn parse_from(&mut self) -> Result<FromClause> {
        let mut from = self.parse_table_ref()?;

        // Check for joins
        loop {
            let join_type = if self.check(&Token::Join)
                || self.check(&Token::Inner)
                || self.check(&Token::Comma)
            {
                if self.check(&Token::Comma) {
                    self.advance();
                    JoinType::Cross
                } else {
                    if self.check(&Token::Inner) {
                        self.advance();
                    }
                    self.expect(&Token::Join)?;
                    JoinType::Inner
                }
            } else if self.check(&Token::Left) {
                self.advance();
                if self.check(&Token::Outer) {
                    self.advance();
                }
                self.expect(&Token::Join)?;
                JoinType::Left
            } else if self.check(&Token::Cross) {
                self.advance();
                self.expect(&Token::Join)?;
                JoinType::Cross
            } else {
                break;
            };

            let right = self.parse_table_ref()?;
            let on = if self.check(&Token::On) {
                self.advance();
                Some(self.parse_expr()?)
            } else {
                None
            };

            from = FromClause::Join(Box::new(JoinClause {
                left: from,
                right,
                join_type,
                on,
            }));
        }

        Ok(from)
    }

    fn parse_table_ref(&mut self) -> Result<FromClause> {
        if self.check(&Token::LeftParen) {
            self.advance();
            if self.check(&Token::Select) {
                let query = self.parse_select()?;
                self.expect(&Token::RightParen)?;
                let alias = if self.check(&Token::As) {
                    self.advance();
                    self.parse_identifier()?
                } else {
                    self.parse_identifier()?
                };
                return Ok(FromClause::Subquery {
                    query: Box::new(query),
                    alias,
                });
            }
            // Could be a parenthesized join
            let from = self.parse_from()?;
            self.expect(&Token::RightParen)?;
            return Ok(from);
        }

        let name = self.parse_identifier()?;
        let alias = if self.check(&Token::As) {
            self.advance();
            Some(self.parse_identifier()?)
        } else if matches!(self.peek(), Token::Identifier(_)) && !self.is_keyword_at_current() {
            Some(self.parse_identifier()?)
        } else {
            None
        };

        Ok(FromClause::Table { name, alias })
    }

    fn parse_order_by_list(&mut self) -> Result<Vec<OrderByItem>> {
        let mut items = Vec::new();
        loop {
            let expr = self.parse_expr()?;
            let descending = if self.check(&Token::Desc) {
                self.advance();
                true
            } else {
                if self.check(&Token::Asc) {
                    self.advance();
                }
                false
            };
            items.push(OrderByItem { expr, descending });
            if !self.check(&Token::Comma) {
                break;
            }
            self.advance();
        }
        Ok(items)
    }

    fn parse_insert_stmt(&mut self) -> Result<Statement> {
        let on_conflict = if self.check(&Token::Replace) {
            self.advance();
            ConflictClause::Replace
        } else {
            self.expect(&Token::Insert)?;
            if self.check(&Token::Or) {
                self.advance();
                if self.check(&Token::Replace) {
                    self.advance();
                    ConflictClause::Replace
                } else if self.check(&Token::Ignore) {
                    self.advance();
                    ConflictClause::Ignore
                } else if self.check(&Token::Rollback) {
                    self.advance();
                    ConflictClause::Rollback
                } else if self.check(&Token::Abort) {
                    self.advance();
                    ConflictClause::Abort
                } else if self.check(&Token::Fail) {
                    self.advance();
                    ConflictClause::Fail
                } else {
                    self.advance(); // skip unknown conflict resolution
                    ConflictClause::Abort
                }
            } else {
                ConflictClause::Abort
            }
        };

        self.expect(&Token::Into)?;
        let table_name = self.parse_identifier()?;

        let columns = if self.check(&Token::LeftParen) {
            // Could be column list or could be subquery in INSERT ... SELECT
            // Peek ahead to check if it's a SELECT
            self.advance();
            if self.check(&Token::Select) {
                // It's INSERT INTO table (SELECT ...)
                let query = self.parse_select()?;
                self.expect(&Token::RightParen)?;
                return Ok(Statement::Insert(InsertStatement {
                    table_name,
                    columns: None,
                    source: InsertSource::Select(Box::new(query)),
                    on_conflict,
                }));
            }
            let cols = self.parse_identifier_list()?;
            self.expect(&Token::RightParen)?;
            Some(cols)
        } else {
            None
        };

        // DEFAULT VALUES
        if self.check(&Token::Default) {
            self.advance();
            self.expect(&Token::Values)?;
            return Ok(Statement::Insert(InsertStatement {
                table_name,
                columns,
                source: InsertSource::DefaultValues,
                on_conflict,
            }));
        }

        // INSERT ... SELECT
        if self.check(&Token::Select) {
            let query = self.parse_select()?;
            return Ok(Statement::Insert(InsertStatement {
                table_name,
                columns,
                source: InsertSource::Select(Box::new(query)),
                on_conflict,
            }));
        }

        self.expect(&Token::Values)?;

        let mut values = Vec::new();
        loop {
            self.expect(&Token::LeftParen)?;
            let row = self.parse_expr_list()?;
            self.expect(&Token::RightParen)?;
            values.push(row);
            if !self.check(&Token::Comma) {
                break;
            }
            self.advance();
        }

        Ok(Statement::Insert(InsertStatement {
            table_name,
            columns,
            source: InsertSource::Values(values),
            on_conflict,
        }))
    }

    fn parse_update_stmt(&mut self) -> Result<Statement> {
        self.expect(&Token::Update)?;
        let table_name = self.parse_identifier()?;
        self.expect(&Token::Set)?;

        let mut assignments = Vec::new();
        loop {
            let col = self.parse_identifier()?;
            self.expect(&Token::Eq)?;
            let val = self.parse_expr()?;
            assignments.push((col, val));
            if !self.check(&Token::Comma) {
                break;
            }
            self.advance();
        }

        let where_clause = if self.check(&Token::Where) {
            self.advance();
            Some(self.parse_expr()?)
        } else {
            None
        };

        Ok(Statement::Update(UpdateStatement {
            table_name,
            assignments,
            where_clause,
        }))
    }

    fn parse_delete_stmt(&mut self) -> Result<Statement> {
        self.expect(&Token::Delete)?;
        self.expect(&Token::From)?;
        let table_name = self.parse_identifier()?;

        let where_clause = if self.check(&Token::Where) {
            self.advance();
            Some(self.parse_expr()?)
        } else {
            None
        };

        Ok(Statement::Delete(DeleteStatement {
            table_name,
            where_clause,
        }))
    }

    fn parse_create_stmt(&mut self) -> Result<Statement> {
        self.expect(&Token::Create)?;

        if self.check(&Token::Unique) {
            self.advance();
            self.expect(&Token::Index)?;
            return self.parse_create_index(true);
        }

        if self.check(&Token::Index) {
            self.advance();
            return self.parse_create_index(false);
        }

        self.expect(&Token::Table)?;

        let if_not_exists = if self.check(&Token::If) {
            self.advance();
            self.expect(&Token::Not)?;
            self.expect(&Token::Exists)?;
            true
        } else {
            false
        };

        let table_name = self.parse_identifier()?;
        self.expect(&Token::LeftParen)?;

        let mut columns = Vec::new();
        let mut constraints = Vec::new();

        loop {
            // Check for table constraints
            if self.check(&Token::Primary)
                || self.check(&Token::Unique)
                || self.check(&Token::Check)
                || self.check(&Token::Foreign)
                || self.check(&Token::Constraint)
            {
                constraints.push(self.parse_table_constraint()?);
            } else {
                columns.push(self.parse_column_def()?);
            }

            if !self.check(&Token::Comma) {
                break;
            }
            self.advance();
        }

        self.expect(&Token::RightParen)?;

        let without_rowid = if self.check(&Token::WithoutKw) {
            self.advance();
            self.expect(&Token::Rowid)?;
            true
        } else {
            false
        };

        Ok(Statement::CreateTable(CreateTableStatement {
            if_not_exists,
            table_name,
            columns,
            constraints,
            without_rowid,
        }))
    }

    fn parse_column_def(&mut self) -> Result<ColumnDef> {
        let name = self.parse_identifier()?;

        // Optional type name
        let type_name = if !self.check(&Token::Comma)
            && !self.check(&Token::RightParen)
            && !self.check(&Token::Primary)
            && !self.check(&Token::Not)
            && !self.check(&Token::Unique)
            && !self.check(&Token::Default)
            && !self.check(&Token::Check)
            && !self.check(&Token::References)
            && !self.check(&Token::Constraint)
            && !self.check(&Token::Eof)
        {
            let tn = self.parse_type_name()?;
            if tn.is_empty() {
                None
            } else {
                Some(tn)
            }
        } else {
            None
        };

        let mut constraints = Vec::new();
        loop {
            if self.check(&Token::Primary) {
                self.advance();
                self.expect(&Token::Key)?;
                let autoincrement = if self.check(&Token::Autoincrement) {
                    self.advance();
                    true
                } else {
                    false
                };
                // Optional ASC/DESC
                if self.check(&Token::Asc) || self.check(&Token::Desc) {
                    self.advance();
                }
                constraints.push(ColumnConstraint::PrimaryKey { autoincrement });
            } else if self.check(&Token::Not) {
                self.advance();
                self.expect(&Token::Null)?;
                constraints.push(ColumnConstraint::NotNull);
            } else if self.check(&Token::Unique) {
                self.advance();
                constraints.push(ColumnConstraint::Unique);
            } else if self.check(&Token::Default) {
                self.advance();
                let expr = if self.check(&Token::LeftParen) {
                    self.advance();
                    let e = self.parse_expr()?;
                    self.expect(&Token::RightParen)?;
                    e
                } else {
                    self.parse_primary_expr()?
                };
                constraints.push(ColumnConstraint::Default(expr));
            } else if self.check(&Token::Check) {
                self.advance();
                self.expect(&Token::LeftParen)?;
                let expr = self.parse_expr()?;
                self.expect(&Token::RightParen)?;
                constraints.push(ColumnConstraint::Check(expr));
            } else if self.check(&Token::References) {
                self.advance();
                let table = self.parse_identifier()?;
                let cols = if self.check(&Token::LeftParen) {
                    self.advance();
                    let c = self.parse_identifier_list()?;
                    self.expect(&Token::RightParen)?;
                    c
                } else {
                    Vec::new()
                };
                constraints.push(ColumnConstraint::References {
                    table,
                    columns: cols,
                });
            } else if self.check(&Token::Constraint) {
                self.advance();
                self.parse_identifier()?; // constraint name, ignored for now
            } else {
                break;
            }
        }

        Ok(ColumnDef {
            name,
            type_name,
            constraints,
        })
    }

    fn parse_type_name(&mut self) -> Result<String> {
        let mut name = String::new();

        // Type name can be multiple words and include parenthesized params
        match self.peek() {
            Token::Integer => {
                name.push_str("INTEGER");
                self.advance();
            }
            Token::Text => {
                name.push_str("TEXT");
                self.advance();
            }
            Token::Real => {
                name.push_str("REAL");
                self.advance();
            }
            Token::Blob => {
                name.push_str("BLOB");
                self.advance();
            }
            Token::Identifier(ref s) => {
                name = s.clone();
                self.advance();
            }
            _ => return Ok(String::new()),
        }

        // Handle additional type words like VARCHAR(255) or DOUBLE PRECISION
        while matches!(self.peek(), Token::Identifier(_)) {
            if let Token::Identifier(s) = self.peek().clone() {
                name.push(' ');
                name.push_str(&s);
                self.advance();
            }
        }

        if self.check(&Token::LeftParen) {
            self.advance();
            name.push('(');
            // Read numbers/identifiers until closing paren
            while !self.check(&Token::RightParen) && !self.check(&Token::Eof) {
                match self.peek().clone() {
                    Token::NumericLiteral(n) => {
                        name.push_str(&n);
                        self.advance();
                    }
                    Token::Comma => {
                        name.push(',');
                        self.advance();
                    }
                    _ => {
                        self.advance();
                    }
                }
            }
            self.expect(&Token::RightParen)?;
            name.push(')');
        }

        Ok(name)
    }

    fn parse_table_constraint(&mut self) -> Result<TableConstraint> {
        if self.check(&Token::Constraint) {
            self.advance();
            self.parse_identifier()?; // constraint name
        }

        if self.check(&Token::Primary) {
            self.advance();
            self.expect(&Token::Key)?;
            self.expect(&Token::LeftParen)?;
            let cols = self.parse_identifier_list()?;
            self.expect(&Token::RightParen)?;
            Ok(TableConstraint::PrimaryKey(cols))
        } else if self.check(&Token::Unique) {
            self.advance();
            self.expect(&Token::LeftParen)?;
            let cols = self.parse_identifier_list()?;
            self.expect(&Token::RightParen)?;
            Ok(TableConstraint::Unique(cols))
        } else if self.check(&Token::Check) {
            self.advance();
            self.expect(&Token::LeftParen)?;
            let expr = self.parse_expr()?;
            self.expect(&Token::RightParen)?;
            Ok(TableConstraint::Check(expr))
        } else if self.check(&Token::Foreign) {
            self.advance();
            self.expect(&Token::Key)?;
            self.expect(&Token::LeftParen)?;
            let cols = self.parse_identifier_list()?;
            self.expect(&Token::RightParen)?;
            self.expect(&Token::References)?;
            let ref_table = self.parse_identifier()?;
            let ref_cols = if self.check(&Token::LeftParen) {
                self.advance();
                let c = self.parse_identifier_list()?;
                self.expect(&Token::RightParen)?;
                c
            } else {
                Vec::new()
            };
            Ok(TableConstraint::ForeignKey {
                columns: cols,
                ref_table,
                ref_columns: ref_cols,
            })
        } else {
            Err(RsqliteError::Parse("Expected table constraint".into()))
        }
    }

    fn parse_create_index(&mut self, unique: bool) -> Result<Statement> {
        let if_not_exists = if self.check(&Token::If) {
            self.advance();
            self.expect(&Token::Not)?;
            self.expect(&Token::Exists)?;
            true
        } else {
            false
        };

        let index_name = self.parse_identifier()?;
        self.expect(&Token::On)?;
        let table_name = self.parse_identifier()?;

        self.expect(&Token::LeftParen)?;
        let mut columns = Vec::new();
        loop {
            let name = self.parse_identifier()?;
            let descending = if self.check(&Token::Desc) {
                self.advance();
                true
            } else {
                if self.check(&Token::Asc) {
                    self.advance();
                }
                false
            };
            columns.push(IndexColumn { name, descending });
            if !self.check(&Token::Comma) {
                break;
            }
            self.advance();
        }
        self.expect(&Token::RightParen)?;

        let where_clause = if self.check(&Token::Where) {
            self.advance();
            Some(self.parse_expr()?)
        } else {
            None
        };

        Ok(Statement::CreateIndex(CreateIndexStatement {
            if_not_exists,
            unique,
            index_name,
            table_name,
            columns,
            where_clause,
        }))
    }

    fn parse_drop_stmt(&mut self) -> Result<Statement> {
        self.expect(&Token::Drop)?;

        if self.check(&Token::Table) {
            self.advance();
            let if_exists = if self.check(&Token::If) {
                self.advance();
                self.expect(&Token::Exists)?;
                true
            } else {
                false
            };
            let table_name = self.parse_identifier()?;
            Ok(Statement::DropTable(DropTableStatement {
                if_exists,
                table_name,
            }))
        } else if self.check(&Token::Index) {
            self.advance();
            let if_exists = if self.check(&Token::If) {
                self.advance();
                self.expect(&Token::Exists)?;
                true
            } else {
                false
            };
            let index_name = self.parse_identifier()?;
            Ok(Statement::DropIndex(DropIndexStatement {
                if_exists,
                index_name,
            }))
        } else {
            Err(RsqliteError::Parse(
                "Expected TABLE or INDEX after DROP".into(),
            ))
        }
    }

    fn parse_alter_stmt(&mut self) -> Result<Statement> {
        self.expect(&Token::Alter)?;
        self.expect(&Token::Table)?;
        let table_name = self.parse_identifier()?;

        if self.check(&Token::Add) {
            self.advance();
            if self.check(&Token::Column) {
                self.advance();
            }
            let column = self.parse_column_def()?;
            Ok(Statement::AlterTable(AlterTableStatement::AddColumn {
                table_name,
                column,
            }))
        } else if self.check(&Token::Rename) {
            self.advance();
            if self.check(&Token::To) {
                self.advance();
                let new_name = self.parse_identifier()?;
                Ok(Statement::AlterTable(AlterTableStatement::RenameTable {
                    table_name,
                    new_name,
                }))
            } else if self.check(&Token::Column) {
                self.advance();
                let old_name = self.parse_identifier()?;
                self.expect(&Token::To)?;
                let new_name = self.parse_identifier()?;
                Ok(Statement::AlterTable(AlterTableStatement::RenameColumn {
                    table_name,
                    old_name,
                    new_name,
                }))
            } else {
                let old_name = self.parse_identifier()?;
                self.expect(&Token::To)?;
                let new_name = self.parse_identifier()?;
                Ok(Statement::AlterTable(AlterTableStatement::RenameColumn {
                    table_name,
                    old_name,
                    new_name,
                }))
            }
        } else {
            Err(RsqliteError::Parse(
                "Expected ADD or RENAME after ALTER TABLE".into(),
            ))
        }
    }

    fn parse_pragma_stmt(&mut self) -> Result<Statement> {
        self.expect(&Token::Pragma)?;
        let name = self.parse_identifier()?;

        let value = if self.check(&Token::Eq) {
            self.advance();
            Some(self.parse_pragma_value()?)
        } else if self.check(&Token::LeftParen) {
            self.advance();
            let val = self.parse_pragma_value()?;
            self.expect(&Token::RightParen)?;
            Some(match val {
                PragmaValue::Name(n) => PragmaValue::Call(n),
                PragmaValue::Number(n) => PragmaValue::Call(n),
                PragmaValue::String(s) => PragmaValue::Call(s),
                PragmaValue::Call(c) => PragmaValue::Call(c),
            })
        } else {
            None
        };

        Ok(Statement::Pragma(PragmaStatement { name, value }))
    }

    fn parse_pragma_value(&mut self) -> Result<PragmaValue> {
        match self.peek().clone() {
            Token::NumericLiteral(n) => {
                self.advance();
                Ok(PragmaValue::Number(n))
            }
            Token::StringLiteral(s) => {
                self.advance();
                Ok(PragmaValue::String(s))
            }
            Token::Identifier(s) => {
                self.advance();
                Ok(PragmaValue::Name(s))
            }
            Token::Minus => {
                self.advance();
                if let Token::NumericLiteral(n) = self.peek().clone() {
                    self.advance();
                    Ok(PragmaValue::Number(format!("-{}", n)))
                } else {
                    Err(RsqliteError::Parse("Expected number after minus".into()))
                }
            }
            _ => {
                // Try to use keyword as identifier
                let name = self.parse_identifier()?;
                Ok(PragmaValue::Name(name))
            }
        }
    }

    // Expression parsing with precedence climbing

    fn parse_expr(&mut self) -> Result<Expr> {
        self.parse_or_expr()
    }

    fn parse_or_expr(&mut self) -> Result<Expr> {
        let mut left = self.parse_and_expr()?;
        while self.check(&Token::Or) {
            self.advance();
            let right = self.parse_and_expr()?;
            left = Expr::BinaryOp {
                left: Box::new(left),
                op: BinaryOperator::Or,
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_and_expr(&mut self) -> Result<Expr> {
        let mut left = self.parse_not_expr()?;
        while self.check(&Token::And) {
            self.advance();
            let right = self.parse_not_expr()?;
            left = Expr::BinaryOp {
                left: Box::new(left),
                op: BinaryOperator::And,
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_not_expr(&mut self) -> Result<Expr> {
        if self.check(&Token::Not) {
            self.advance();
            let expr = self.parse_not_expr()?;
            Ok(Expr::Not(Box::new(expr)))
        } else {
            self.parse_comparison_expr()
        }
    }

    fn parse_comparison_expr(&mut self) -> Result<Expr> {
        let mut left = self.parse_addition_expr()?;

        loop {
            if self.check(&Token::Eq) || self.check(&Token::EqEq) {
                self.advance();
                let right = self.parse_addition_expr()?;
                left = Expr::BinaryOp {
                    left: Box::new(left),
                    op: BinaryOperator::Eq,
                    right: Box::new(right),
                };
            } else if self.check(&Token::NotEq) {
                self.advance();
                let right = self.parse_addition_expr()?;
                left = Expr::BinaryOp {
                    left: Box::new(left),
                    op: BinaryOperator::NotEq,
                    right: Box::new(right),
                };
            } else if self.check(&Token::Lt) {
                self.advance();
                let right = self.parse_addition_expr()?;
                left = Expr::BinaryOp {
                    left: Box::new(left),
                    op: BinaryOperator::Lt,
                    right: Box::new(right),
                };
            } else if self.check(&Token::LtEq) {
                self.advance();
                let right = self.parse_addition_expr()?;
                left = Expr::BinaryOp {
                    left: Box::new(left),
                    op: BinaryOperator::LtEq,
                    right: Box::new(right),
                };
            } else if self.check(&Token::Gt) {
                self.advance();
                let right = self.parse_addition_expr()?;
                left = Expr::BinaryOp {
                    left: Box::new(left),
                    op: BinaryOperator::Gt,
                    right: Box::new(right),
                };
            } else if self.check(&Token::GtEq) {
                self.advance();
                let right = self.parse_addition_expr()?;
                left = Expr::BinaryOp {
                    left: Box::new(left),
                    op: BinaryOperator::GtEq,
                    right: Box::new(right),
                };
            } else if self.check(&Token::Is) {
                self.advance();
                if self.check(&Token::Not) {
                    self.advance();
                    if self.check(&Token::Null) {
                        self.advance();
                        left = Expr::IsNotNull(Box::new(left));
                    } else {
                        let right = self.parse_addition_expr()?;
                        // IS NOT expr is treated as != for non-NULL
                        left = Expr::BinaryOp {
                            left: Box::new(left),
                            op: BinaryOperator::NotEq,
                            right: Box::new(right),
                        };
                    }
                } else if self.check(&Token::Null) {
                    self.advance();
                    left = Expr::IsNull(Box::new(left));
                } else {
                    let right = self.parse_addition_expr()?;
                    left = Expr::BinaryOp {
                        left: Box::new(left),
                        op: BinaryOperator::Eq,
                        right: Box::new(right),
                    };
                }
            } else if self.check(&Token::Not) {
                // NOT IN, NOT LIKE, NOT BETWEEN, NOT GLOB
                self.advance();
                if self.check(&Token::In) {
                    self.advance();
                    left = self.parse_in_expr(left, true)?;
                } else if self.check(&Token::Like) {
                    self.advance();
                    left = self.parse_like_expr(left, true)?;
                } else if self.check(&Token::Between) {
                    self.advance();
                    left = self.parse_between_expr(left, true)?;
                } else if self.check(&Token::Glob) {
                    self.advance();
                    left = self.parse_glob_expr(left, true)?;
                } else {
                    return Err(RsqliteError::Parse(
                        "Expected IN, LIKE, BETWEEN, or GLOB after NOT".into(),
                    ));
                }
            } else if self.check(&Token::In) {
                self.advance();
                left = self.parse_in_expr(left, false)?;
            } else if self.check(&Token::Like) {
                self.advance();
                left = self.parse_like_expr(left, false)?;
            } else if self.check(&Token::Between) {
                self.advance();
                left = self.parse_between_expr(left, false)?;
            } else if self.check(&Token::Glob) {
                self.advance();
                left = self.parse_glob_expr(left, false)?;
            } else {
                break;
            }
        }
        Ok(left)
    }

    fn parse_in_expr(&mut self, expr: Expr, negated: bool) -> Result<Expr> {
        self.expect(&Token::LeftParen)?;
        if self.check(&Token::Select) {
            let query = self.parse_select()?;
            self.expect(&Token::RightParen)?;
            Ok(Expr::InSelect {
                expr: Box::new(expr),
                query: Box::new(query),
                negated,
            })
        } else {
            let list = self.parse_expr_list()?;
            self.expect(&Token::RightParen)?;
            Ok(Expr::InList {
                expr: Box::new(expr),
                list,
                negated,
            })
        }
    }

    fn parse_like_expr(&mut self, expr: Expr, negated: bool) -> Result<Expr> {
        let pattern = self.parse_addition_expr()?;
        let escape = if self.check(&Token::Escape) {
            self.advance();
            Some(Box::new(self.parse_addition_expr()?))
        } else {
            None
        };
        Ok(Expr::Like {
            expr: Box::new(expr),
            pattern: Box::new(pattern),
            escape,
            negated,
        })
    }

    fn parse_glob_expr(&mut self, expr: Expr, negated: bool) -> Result<Expr> {
        let pattern = self.parse_addition_expr()?;
        Ok(Expr::GlobExpr {
            expr: Box::new(expr),
            pattern: Box::new(pattern),
            negated,
        })
    }

    fn parse_between_expr(&mut self, expr: Expr, negated: bool) -> Result<Expr> {
        let low = self.parse_addition_expr()?;
        self.expect(&Token::And)?;
        let high = self.parse_addition_expr()?;
        Ok(Expr::Between {
            expr: Box::new(expr),
            low: Box::new(low),
            high: Box::new(high),
            negated,
        })
    }

    fn parse_addition_expr(&mut self) -> Result<Expr> {
        let mut left = self.parse_multiplication_expr()?;
        loop {
            if self.check(&Token::Plus) {
                self.advance();
                let right = self.parse_multiplication_expr()?;
                left = Expr::BinaryOp {
                    left: Box::new(left),
                    op: BinaryOperator::Add,
                    right: Box::new(right),
                };
            } else if self.check(&Token::Minus) {
                self.advance();
                let right = self.parse_multiplication_expr()?;
                left = Expr::BinaryOp {
                    left: Box::new(left),
                    op: BinaryOperator::Subtract,
                    right: Box::new(right),
                };
            } else if self.check(&Token::Concat) {
                self.advance();
                let right = self.parse_multiplication_expr()?;
                left = Expr::BinaryOp {
                    left: Box::new(left),
                    op: BinaryOperator::Concat,
                    right: Box::new(right),
                };
            } else if self.check(&Token::BitAnd) {
                self.advance();
                let right = self.parse_multiplication_expr()?;
                left = Expr::BinaryOp {
                    left: Box::new(left),
                    op: BinaryOperator::BitAnd,
                    right: Box::new(right),
                };
            } else if self.check(&Token::BitOr) {
                self.advance();
                let right = self.parse_multiplication_expr()?;
                left = Expr::BinaryOp {
                    left: Box::new(left),
                    op: BinaryOperator::BitOr,
                    right: Box::new(right),
                };
            } else if self.check(&Token::ShiftLeft) {
                self.advance();
                let right = self.parse_multiplication_expr()?;
                left = Expr::BinaryOp {
                    left: Box::new(left),
                    op: BinaryOperator::ShiftLeft,
                    right: Box::new(right),
                };
            } else if self.check(&Token::ShiftRight) {
                self.advance();
                let right = self.parse_multiplication_expr()?;
                left = Expr::BinaryOp {
                    left: Box::new(left),
                    op: BinaryOperator::ShiftRight,
                    right: Box::new(right),
                };
            } else {
                break;
            }
        }
        Ok(left)
    }

    fn parse_multiplication_expr(&mut self) -> Result<Expr> {
        let mut left = self.parse_unary_expr()?;
        loop {
            if self.check(&Token::Star) {
                self.advance();
                let right = self.parse_unary_expr()?;
                left = Expr::BinaryOp {
                    left: Box::new(left),
                    op: BinaryOperator::Multiply,
                    right: Box::new(right),
                };
            } else if self.check(&Token::Slash) {
                self.advance();
                let right = self.parse_unary_expr()?;
                left = Expr::BinaryOp {
                    left: Box::new(left),
                    op: BinaryOperator::Divide,
                    right: Box::new(right),
                };
            } else if self.check(&Token::Percent) {
                self.advance();
                let right = self.parse_unary_expr()?;
                left = Expr::BinaryOp {
                    left: Box::new(left),
                    op: BinaryOperator::Modulo,
                    right: Box::new(right),
                };
            } else {
                break;
            }
        }
        Ok(left)
    }

    fn parse_unary_expr(&mut self) -> Result<Expr> {
        if self.check(&Token::Minus) {
            self.advance();
            let expr = self.parse_unary_expr()?;
            // Constant fold negative numbers
            if let Expr::Integer(n) = expr {
                Ok(Expr::Integer(-n))
            } else if let Expr::Float(f) = expr {
                Ok(Expr::Float(-f))
            } else {
                Ok(Expr::UnaryMinus(Box::new(expr)))
            }
        } else if self.check(&Token::Plus) {
            self.advance();
            let expr = self.parse_unary_expr()?;
            Ok(Expr::UnaryPlus(Box::new(expr)))
        } else if self.check(&Token::Tilde) {
            self.advance();
            let expr = self.parse_unary_expr()?;
            Ok(Expr::BitwiseNot(Box::new(expr)))
        } else {
            self.parse_primary_expr()
        }
    }

    fn parse_primary_expr(&mut self) -> Result<Expr> {
        let token = self.peek().clone();
        match token {
            Token::NumericLiteral(ref s) => {
                let s = s.clone();
                self.advance();
                if s.starts_with("0x") || s.starts_with("0X") {
                    let n = i64::from_str_radix(&s[2..], 16)
                        .map_err(|e| RsqliteError::Parse(format!("Invalid hex: {}", e)))?;
                    Ok(Expr::Integer(n))
                } else if s.contains('.') || s.contains('e') || s.contains('E') {
                    let f: f64 = s
                        .parse()
                        .map_err(|e| RsqliteError::Parse(format!("Invalid float: {}", e)))?;
                    Ok(Expr::Float(f))
                } else {
                    let n: i64 = s
                        .parse()
                        .map_err(|e| RsqliteError::Parse(format!("Invalid integer: {}", e)))?;
                    Ok(Expr::Integer(n))
                }
            }
            Token::StringLiteral(ref s) => {
                let s = s.clone();
                self.advance();
                Ok(Expr::String(s))
            }
            Token::BlobLiteral(ref b) => {
                let b = b.clone();
                self.advance();
                Ok(Expr::Blob(b))
            }
            Token::Null => {
                self.advance();
                Ok(Expr::Null)
            }
            Token::Star => {
                self.advance();
                Ok(Expr::Star)
            }
            Token::LeftParen => {
                self.advance();
                if self.check(&Token::Select) {
                    let query = self.parse_select()?;
                    self.expect(&Token::RightParen)?;
                    Ok(Expr::Subquery(Box::new(query)))
                } else {
                    let expr = self.parse_expr()?;
                    self.expect(&Token::RightParen)?;
                    Ok(expr)
                }
            }
            Token::Case => {
                self.advance();
                let operand = if !self.check(&Token::When) {
                    Some(Box::new(self.parse_expr()?))
                } else {
                    None
                };

                let mut when_clauses = Vec::new();
                while self.check(&Token::When) {
                    self.advance();
                    let when_expr = self.parse_expr()?;
                    self.expect(&Token::Then)?;
                    let then_expr = self.parse_expr()?;
                    when_clauses.push((when_expr, then_expr));
                }

                let else_clause = if self.check(&Token::Else) {
                    self.advance();
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
            Token::Cast => {
                self.advance();
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
            Token::Exists => {
                self.advance();
                self.expect(&Token::LeftParen)?;
                let query = self.parse_select()?;
                self.expect(&Token::RightParen)?;
                Ok(Expr::Exists(Box::new(query)))
            }
            Token::Not => {
                self.advance();
                if self.check(&Token::Exists) {
                    self.advance();
                    self.expect(&Token::LeftParen)?;
                    let query = self.parse_select()?;
                    self.expect(&Token::RightParen)?;
                    Ok(Expr::Not(Box::new(Expr::Exists(Box::new(query)))))
                } else {
                    let expr = self.parse_primary_expr()?;
                    Ok(Expr::Not(Box::new(expr)))
                }
            }
            Token::Identifier(_) | Token::Replace => {
                let name = self.parse_identifier()?;

                // Check for function call
                if self.check(&Token::LeftParen) {
                    self.advance();
                    let distinct = if self.check(&Token::Distinct) {
                        self.advance();
                        true
                    } else {
                        false
                    };
                    let args = if self.check(&Token::RightParen) {
                        Vec::new()
                    } else if self.check(&Token::Star) {
                        self.advance();
                        vec![Expr::Star]
                    } else {
                        self.parse_expr_list()?
                    };
                    self.expect(&Token::RightParen)?;
                    return Ok(Expr::Function {
                        name: name.to_uppercase(),
                        args,
                        distinct,
                    });
                }

                // Check for table.column
                if self.check(&Token::Dot) {
                    self.advance();
                    if self.check(&Token::Star) {
                        self.advance();
                        return Ok(Expr::Column {
                            table: Some(name),
                            name: "*".into(),
                        });
                    }
                    let col_name = self.parse_identifier()?;
                    return Ok(Expr::Column {
                        table: Some(name),
                        name: col_name,
                    });
                }

                Ok(Expr::Column { table: None, name })
            }
            // Handle keywords that can be used as identifiers in expressions
            Token::Rowid => {
                self.advance();
                Ok(Expr::Rowid)
            }
            _ => {
                // Try to treat keywords as identifiers
                if token.is_keyword() {
                    let name = self.keyword_as_identifier()?;
                    if self.check(&Token::LeftParen) {
                        self.advance();
                        let distinct = if self.check(&Token::Distinct) {
                            self.advance();
                            true
                        } else {
                            false
                        };
                        let args = if self.check(&Token::RightParen) {
                            Vec::new()
                        } else if self.check(&Token::Star) {
                            self.advance();
                            vec![Expr::Star]
                        } else {
                            self.parse_expr_list()?
                        };
                        self.expect(&Token::RightParen)?;
                        return Ok(Expr::Function {
                            name: name.to_uppercase(),
                            args,
                            distinct,
                        });
                    }
                    if self.check(&Token::Dot) {
                        self.advance();
                        let col_name = self.parse_identifier()?;
                        return Ok(Expr::Column {
                            table: Some(name),
                            name: col_name,
                        });
                    }
                    return Ok(Expr::Column { table: None, name });
                }

                Err(RsqliteError::Parse(format!(
                    "Unexpected token in expression: {:?}",
                    token
                )))
            }
        }
    }

    fn parse_expr_list(&mut self) -> Result<Vec<Expr>> {
        let mut exprs = Vec::new();
        loop {
            exprs.push(self.parse_expr()?);
            if !self.check(&Token::Comma) {
                break;
            }
            self.advance();
        }
        Ok(exprs)
    }

    fn parse_identifier_list(&mut self) -> Result<Vec<String>> {
        let mut names = Vec::new();
        loop {
            names.push(self.parse_identifier()?);
            if !self.check(&Token::Comma) {
                break;
            }
            self.advance();
        }
        Ok(names)
    }

    fn parse_identifier(&mut self) -> Result<String> {
        match self.peek().clone() {
            Token::Identifier(s) => {
                self.advance();
                Ok(s)
            }
            // Allow certain keywords as identifiers
            token if token.is_keyword() => self.keyword_as_identifier(),
            _ => Err(RsqliteError::Parse(format!(
                "Expected identifier, got {:?}",
                self.peek()
            ))),
        }
    }

    fn keyword_as_identifier(&mut self) -> Result<String> {
        let name = match self.peek() {
            Token::Table => "table",
            Token::Index => "index",
            Token::Key => "key",
            Token::Column => "column",
            Token::Values => "values",
            Token::Set => "set",
            Token::Default => "default",
            Token::Add => "add",
            Token::Integer => "INTEGER",
            Token::Text => "TEXT",
            Token::Real => "REAL",
            Token::Blob => "BLOB",
            Token::Replace => "replace",
            Token::If => "if",
            Token::Exists => "exists",
            Token::Begin => "begin",
            Token::End => "end",
            Token::Abort => "abort",
            Token::Conflict => "conflict",
            Token::Fail => "fail",
            Token::Ignore => "ignore",
            Token::Rollback => "rollback",
            Token::Asc => "asc",
            Token::Desc => "desc",
            Token::Null => "null",
            Token::Rename => "rename",
            Token::To => "to",
            Token::Plan => "plan",
            Token::Query => "query",
            Token::Escape => "escape",
            Token::Collate => "collate",
            Token::Rowid => "rowid",
            Token::Transaction => "transaction",
            Token::Check => "check",
            Token::Constraint => "constraint",
            Token::Foreign => "foreign",
            Token::References => "references",
            Token::Primary => "primary",
            Token::Unique => "unique",
            Token::Autoincrement => "autoincrement",
            Token::Pragma => "pragma",
            _ => {
                return Err(RsqliteError::Parse(format!(
                    "Cannot use {:?} as identifier",
                    self.peek()
                )))
            }
        };
        self.advance();
        Ok(name.to_string())
    }

    // Token helpers

    fn peek(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or(&Token::Eof)
    }

    fn check(&self, token: &Token) -> bool {
        std::mem::discriminant(self.peek()) == std::mem::discriminant(token)
    }

    fn advance(&mut self) -> &Token {
        let token = self.tokens.get(self.pos).unwrap_or(&Token::Eof);
        self.pos += 1;
        token
    }

    fn expect(&mut self, expected: &Token) -> Result<()> {
        if self.check(expected) {
            self.advance();
            Ok(())
        } else {
            Err(RsqliteError::Parse(format!(
                "Expected {:?}, got {:?}",
                expected,
                self.peek()
            )))
        }
    }
}

/// Parse a SQL string into a statement.
pub fn parse_sql(sql: &str) -> Result<Statement> {
    let mut tokenizer = crate::tokenizer::Tokenizer::new(sql);
    let tokens = tokenizer.tokenize()?;
    let mut parser = Parser::new(tokens);
    parser.parse()
}

/// Parse a SQL string into multiple statements.
pub fn parse_sql_multi(sql: &str) -> Result<Vec<Statement>> {
    let mut tokenizer = crate::tokenizer::Tokenizer::new(sql);
    let tokens = tokenizer.tokenize()?;
    let mut parser = Parser::new(tokens);
    parser.parse_multiple()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_select_star() {
        let stmt = parse_sql("SELECT * FROM users").unwrap();
        if let Statement::Select(sel) = stmt {
            assert_eq!(sel.columns.len(), 1);
            assert!(matches!(sel.columns[0], SelectColumn::AllColumns));
            if let Some(FromClause::Table { name, alias }) = &sel.from {
                assert_eq!(name, "users");
                assert!(alias.is_none());
            } else {
                panic!("Expected table from clause");
            }
        } else {
            panic!("Expected SELECT statement");
        }
    }

    #[test]
    fn test_parse_select_where() {
        let stmt = parse_sql("SELECT id, name FROM users WHERE id = 1").unwrap();
        if let Statement::Select(sel) = stmt {
            assert_eq!(sel.columns.len(), 2);
            assert!(sel.where_clause.is_some());
        } else {
            panic!("Expected SELECT");
        }
    }

    #[test]
    fn test_parse_insert() {
        let stmt = parse_sql("INSERT INTO users (name, age) VALUES ('Alice', 30)").unwrap();
        if let Statement::Insert(ins) = stmt {
            assert_eq!(ins.table_name, "users");
            assert_eq!(ins.columns.as_ref().unwrap().len(), 2);
            if let InsertSource::Values(values) = &ins.source {
                assert_eq!(values.len(), 1);
                assert_eq!(values[0].len(), 2);
            } else {
                panic!("Expected VALUES source");
            }
        } else {
            panic!("Expected INSERT");
        }
    }

    #[test]
    fn test_parse_create_table() {
        let stmt = parse_sql(
            "CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, age INTEGER DEFAULT 0)"
        ).unwrap();
        if let Statement::CreateTable(ct) = stmt {
            assert_eq!(ct.table_name, "users");
            assert_eq!(ct.columns.len(), 3);
            assert_eq!(ct.columns[0].name, "id");
            assert!(matches!(
                ct.columns[0].constraints[0],
                ColumnConstraint::PrimaryKey {
                    autoincrement: true
                }
            ));
        } else {
            panic!("Expected CREATE TABLE");
        }
    }

    #[test]
    fn test_parse_update() {
        let stmt = parse_sql("UPDATE users SET name = 'Bob' WHERE id = 1").unwrap();
        assert!(matches!(stmt, Statement::Update(_)));
    }

    #[test]
    fn test_parse_delete() {
        let stmt = parse_sql("DELETE FROM users WHERE id = 1").unwrap();
        assert!(matches!(stmt, Statement::Delete(_)));
    }

    #[test]
    fn test_parse_join() {
        let stmt =
            parse_sql("SELECT u.name, o.amount FROM users u JOIN orders o ON u.id = o.user_id")
                .unwrap();
        if let Statement::Select(sel) = stmt {
            assert!(matches!(sel.from, Some(FromClause::Join(_))));
        } else {
            panic!("Expected SELECT");
        }
    }

    #[test]
    fn test_parse_order_by_limit() {
        let stmt = parse_sql("SELECT * FROM users ORDER BY name ASC LIMIT 10 OFFSET 5").unwrap();
        if let Statement::Select(sel) = stmt {
            assert_eq!(sel.order_by.len(), 1);
            assert!(!sel.order_by[0].descending);
            assert!(sel.limit.is_some());
            assert!(sel.offset.is_some());
        } else {
            panic!("Expected SELECT");
        }
    }

    #[test]
    fn test_parse_group_by_having() {
        let stmt = parse_sql(
            "SELECT department, COUNT(*) FROM employees GROUP BY department HAVING COUNT(*) > 5",
        )
        .unwrap();
        if let Statement::Select(sel) = stmt {
            assert_eq!(sel.group_by.len(), 1);
            assert!(sel.having.is_some());
        } else {
            panic!("Expected SELECT");
        }
    }

    #[test]
    fn test_parse_transactions() {
        assert!(matches!(parse_sql("BEGIN").unwrap(), Statement::Begin));
        assert!(matches!(parse_sql("COMMIT").unwrap(), Statement::Commit));
        assert!(matches!(
            parse_sql("ROLLBACK").unwrap(),
            Statement::Rollback
        ));
    }

    #[test]
    fn test_parse_case() {
        let stmt =
            parse_sql("SELECT CASE WHEN x > 0 THEN 'positive' ELSE 'non-positive' END FROM t")
                .unwrap();
        assert!(matches!(stmt, Statement::Select(_)));
    }

    #[test]
    fn test_parse_create_index() {
        let stmt = parse_sql("CREATE UNIQUE INDEX idx_name ON users (name)").unwrap();
        if let Statement::CreateIndex(ci) = stmt {
            assert!(ci.unique);
            assert_eq!(ci.index_name, "idx_name");
            assert_eq!(ci.table_name, "users");
            assert_eq!(ci.columns.len(), 1);
        } else {
            panic!("Expected CREATE INDEX");
        }
    }

    // ======== Phase 7 Parser Tests ========

    #[test]
    fn test_parse_union() {
        let stmt = parse_sql("SELECT x FROM t1 UNION SELECT x FROM t2").unwrap();
        if let Statement::Select(sel) = stmt {
            assert!(sel.compound.is_some());
            let compound = sel.compound.unwrap();
            assert_eq!(compound.op, CompoundOp::Union);
        } else {
            panic!("Expected SELECT");
        }
    }

    #[test]
    fn test_parse_union_all() {
        let stmt = parse_sql("SELECT x FROM t1 UNION ALL SELECT x FROM t2").unwrap();
        if let Statement::Select(sel) = stmt {
            assert!(sel.compound.is_some());
            let compound = sel.compound.unwrap();
            assert_eq!(compound.op, CompoundOp::UnionAll);
        } else {
            panic!("Expected SELECT");
        }
    }

    #[test]
    fn test_parse_except() {
        let stmt = parse_sql("SELECT x FROM t1 EXCEPT SELECT x FROM t2").unwrap();
        if let Statement::Select(sel) = stmt {
            assert!(sel.compound.is_some());
            let compound = sel.compound.unwrap();
            assert_eq!(compound.op, CompoundOp::Except);
        } else {
            panic!("Expected SELECT");
        }
    }

    #[test]
    fn test_parse_intersect() {
        let stmt = parse_sql("SELECT x FROM t1 INTERSECT SELECT x FROM t2").unwrap();
        if let Statement::Select(sel) = stmt {
            assert!(sel.compound.is_some());
            let compound = sel.compound.unwrap();
            assert_eq!(compound.op, CompoundOp::Intersect);
        } else {
            panic!("Expected SELECT");
        }
    }

    #[test]
    fn test_parse_insert_select() {
        let stmt = parse_sql("INSERT INTO t2 SELECT * FROM t1").unwrap();
        if let Statement::Insert(ins) = stmt {
            assert_eq!(ins.table_name, "t2");
            assert!(matches!(ins.source, InsertSource::Select(_)));
        } else {
            panic!("Expected INSERT");
        }
    }

    #[test]
    fn test_parse_insert_default_values() {
        let stmt = parse_sql("INSERT INTO t DEFAULT VALUES").unwrap();
        if let Statement::Insert(ins) = stmt {
            assert_eq!(ins.table_name, "t");
            assert!(matches!(ins.source, InsertSource::DefaultValues));
        } else {
            panic!("Expected INSERT");
        }
    }

    #[test]
    fn test_parse_drop_table() {
        let stmt = parse_sql("DROP TABLE IF EXISTS users").unwrap();
        if let Statement::DropTable(dt) = stmt {
            assert_eq!(dt.table_name, "users");
            assert!(dt.if_exists);
        } else {
            panic!("Expected DROP TABLE");
        }
    }

    #[test]
    fn test_parse_drop_index() {
        let stmt = parse_sql("DROP INDEX IF EXISTS idx_name").unwrap();
        if let Statement::DropIndex(di) = stmt {
            assert_eq!(di.index_name, "idx_name");
            assert!(di.if_exists);
        } else {
            panic!("Expected DROP INDEX");
        }
    }

    #[test]
    fn test_parse_alter_table_rename() {
        let stmt = parse_sql("ALTER TABLE users RENAME TO people").unwrap();
        if let Statement::AlterTable(AlterTableStatement::RenameTable {
            table_name,
            new_name,
        }) = stmt
        {
            assert_eq!(table_name, "users");
            assert_eq!(new_name, "people");
        } else {
            panic!("Expected ALTER TABLE RENAME");
        }
    }

    #[test]
    fn test_parse_alter_table_add_column() {
        let stmt = parse_sql("ALTER TABLE users ADD COLUMN email TEXT").unwrap();
        if let Statement::AlterTable(AlterTableStatement::AddColumn { table_name, column }) = stmt {
            assert_eq!(table_name, "users");
            assert_eq!(column.name, "email");
            assert_eq!(column.type_name.as_deref(), Some("TEXT"));
        } else {
            panic!("Expected ALTER TABLE ADD COLUMN");
        }
    }

    #[test]
    fn test_parse_pragma() {
        let stmt = parse_sql("PRAGMA table_info(users)").unwrap();
        if let Statement::Pragma(p) = stmt {
            assert_eq!(p.name, "table_info");
            assert!(p.value.is_some());
        } else {
            panic!("Expected PRAGMA");
        }
    }

    #[test]
    fn test_parse_multi_row_insert() {
        let stmt = parse_sql("INSERT INTO t VALUES (1, 'a'), (2, 'b'), (3, 'c')").unwrap();
        if let Statement::Insert(ins) = stmt {
            if let InsertSource::Values(values) = &ins.source {
                assert_eq!(values.len(), 3);
            } else {
                panic!("Expected VALUES source");
            }
        } else {
            panic!("Expected INSERT");
        }
    }

    #[test]
    fn test_parse_explain() {
        let stmt = parse_sql("EXPLAIN SELECT * FROM t").unwrap();
        assert!(matches!(stmt, Statement::Explain(_)));
    }
}
