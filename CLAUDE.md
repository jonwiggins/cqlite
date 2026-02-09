# Project: Implement SQLite from Scratch in Rust

## Objective

Build a fully functional SQLite-compatible database engine in Rust from scratch. The implementation should be able to read and write SQLite3-compatible database files, parse and execute SQL statements, and pass progressively more of SQLite's official test suite.

Do NOT use any existing SQL parsing libraries, database engines, or SQLite bindings. Every component must be implemented from first principles. You MAY use general-purpose Rust crates for things like I/O, byte manipulation, and async runtime, but the core database logic must be yours.

## Project Setup

- Initialize a new Rust project using `cargo init --name rsqlite`
- Use Rust stable (latest edition)
- Set up the project as both a library (`lib.rs`) and a binary (`main.rs`) that provides a REPL/CLI interface similar to the `sqlite3` command-line tool
- Use `cargo test` as the primary test runner
- Structure the codebase into well-separated modules from the start (see Architecture below)

## Architecture

The implementation must be organized into these core modules/components. Plan and design the interfaces between them BEFORE writing implementation code.

### 1. SQL Tokenizer (`tokenizer` module)
- Hand-written lexer (no parser generator dependencies)
- Tokenize SQL input into a stream of typed tokens
- Handle: keywords, identifiers, string literals, numeric literals, operators, punctuation, comments
- Support single-quoted strings, double-quoted identifiers, backtick identifiers
- Handle edge cases: escaped quotes, hex literals, blob literals (X'...')

### 2. SQL Parser (`parser` module)
- Recursive descent parser producing a typed AST
- Define a clean AST representation using Rust enums and structs
- Supported statements (implement in this priority order):
  1. `CREATE TABLE` (column definitions, types, constraints: PRIMARY KEY, NOT NULL, DEFAULT, UNIQUE, AUTOINCREMENT)
  2. `INSERT INTO` (with VALUES, multi-row insert)
  3. `SELECT` (columns, *, expressions, aliases)
  4. `WHERE` clauses (comparison operators, AND/OR/NOT, IS NULL, IS NOT NULL, BETWEEN, IN, LIKE)
  5. `UPDATE` with SET and WHERE
  6. `DELETE FROM` with WHERE
  7. `DROP TABLE`, `ALTER TABLE` (ADD COLUMN, RENAME)
  8. `CREATE INDEX`, `DROP INDEX`
  9. `ORDER BY`, `GROUP BY`, `HAVING`, `LIMIT`, `OFFSET`
  10. `JOIN` (INNER, LEFT, CROSS), subqueries
  11. Aggregate functions: COUNT, SUM, AVG, MIN, MAX
  12. `BEGIN`, `COMMIT`, `ROLLBACK` (transactions)
  13. `EXPLAIN` and `EXPLAIN QUERY PLAN`
  14. `PRAGMA` statements
  15. Common scalar functions: length, upper, lower, typeof, abs, coalesce, ifnull, nullif, substr, trim, replace, hex, quote, random

### 3. Query Planner / Compiler (`planner` module)
- Translate the AST into a query execution plan
- Use a bytecode VM approach (like real SQLite) OR a tree-of-iterators (Volcano model) — choose one and document why
- Plan representation should be inspectable (for EXPLAIN support)
- Index selection: when an index exists that satisfies a WHERE clause, use it
- Basic optimizations:
  - Constant folding
  - Predicate pushdown
  - Index scan vs. full table scan decision

### 4. Virtual Machine / Executor (`vm` module)
- Execute query plans against the storage engine
- If using bytecode VM: define an opcode set inspired by (but not necessarily identical to) SQLite's opcodes
- If using Volcano model: implement iterator trait with `open()`, `next()`, `close()`
- Handle expression evaluation, type coercion (SQLite's type affinity rules), NULL semantics
- Implement SQLite's type affinity system:
  - Storage classes: NULL, INTEGER, REAL, TEXT, BLOB
  - Type affinity for columns: TEXT, NUMERIC, INTEGER, REAL, BLOB
  - Comparison rules between different types

### 5. B-Tree Engine (`btree` module)
- Implement both B-tree (for indexes) and B+tree (for tables) variants
- This is the core data structure — get it right
- Operations: search, insert, delete, sequential scan, range scan
- Handle node splitting and merging
- Support variable-length keys and payloads
- Implement overflow pages for large records
- Cursor-based iteration API

### 6. Pager / Buffer Pool (`pager` module)
- Page-based I/O layer between B-tree and the OS filesystem
- Fixed page size (default 4096 bytes, configurable via PRAGMA)
- Page cache with configurable size (LRU eviction)
- Read/write pages by page number
- Manage free page list (freelist)
- Coordinate with the WAL or journal for crash safety

### 7. File Format (`format` module)
- Implement the SQLite3 file format: https://www.sqlite.org/fileformat2.html
- This is critical for compatibility — read this spec thoroughly
- Database header (first 100 bytes): magic string, page size, format versions, etc.
- Table and index interior/leaf pages
- Record format: serial type codes, varint encoding
- The `sqlite_master` table (schema storage)
- Freelist trunk and leaf pages

### 8. Transaction & Journal (`journal` module)
- Implement rollback journal (simpler) first, WAL (write-ahead log) as stretch goal
- ACID guarantees:
  - Atomicity: all-or-nothing commits via journal
  - Consistency: constraint enforcement
  - Isolation: at minimum, serialized access (single-writer). Stretch: WAL for concurrent readers
  - Durability: fsync at commit
- Lock states: UNLOCKED, SHARED, RESERVED, PENDING, EXCLUSIVE
- Savepoints (stretch goal)

### 9. CLI / REPL (`cli` module)
- Interactive prompt with `rsqlite>` prefix
- Execute SQL statements terminated by `;`
- Support dot-commands: `.tables`, `.schema`, `.quit`, `.open`, `.headers on/off`, `.mode` (column, csv, etc.)
- Pretty-print query results in column format
- Read and execute `.sql` files via `.read` or command-line argument
- Accept database file path as CLI argument: `rsqlite mydb.db`

## Implementation Phases

Follow these phases strictly. Do not skip ahead. Each phase should be fully tested before moving to the next.

### Phase 1: Foundation (Storage Layer)
1. Implement varint encoding/decoding (SQLite's variable-length integer format)
2. Implement the pager: read/write fixed-size pages to/from a file
3. Implement the database file header (100-byte header, parse and write)
4. Implement B-tree page structure: parse cell pointers, cell content
5. Implement the `sqlite_master` table reader
6. **Milestone test**: Open a real SQLite database file created by the official `sqlite3` tool and read its schema

### Phase 2: Read Path
1. Implement the SQL tokenizer
2. Implement the parser for SELECT, WHERE
3. Implement a basic query executor (full table scan)
4. Implement record deserialization (decode stored rows into values)
5. Implement B-tree cursor for sequential scan
6. Implement expression evaluation (comparisons, AND/OR, arithmetic)
7. **Milestone test**: `SELECT * FROM table WHERE condition` works against a real SQLite DB file

### Phase 3: Write Path
1. Implement CREATE TABLE (write to sqlite_master, allocate B-tree root page)
2. Implement INSERT (serialize records, insert into B-tree, handle page splits)
3. Implement DELETE (remove from B-tree, handle merging)
4. Implement UPDATE (delete + insert, or in-place when possible)
5. Implement the freelist (recycle deleted pages)
6. **Milestone test**: Create a database, insert data, close and reopen, data persists. Verify the file is readable by official `sqlite3`.

### Phase 4: Transactions & Crash Safety
1. Implement rollback journal (write original pages before modifying)
2. Implement BEGIN/COMMIT/ROLLBACK
3. Implement auto-commit mode (each statement is its own transaction)
4. Hot journal recovery on open
5. **Milestone test**: Kill the process mid-transaction, restart, verify database integrity

### Phase 5: Indexes & Query Planning
1. Implement CREATE INDEX (separate B-tree for index entries)
2. Modify the query planner to detect usable indexes
3. Implement index scan in the executor
4. Implement UNIQUE constraint enforcement via index
5. Implement PRIMARY KEY as implicit index (rowid alias)
6. **Milestone test**: Create an index, verify queries use it (via EXPLAIN), verify performance improvement

### Phase 6: Advanced SQL
1. ORDER BY (external sort for large results)
2. GROUP BY and aggregate functions
3. JOIN (start with nested loop join)
4. Subqueries (in WHERE and FROM clauses)
5. LIMIT and OFFSET
6. HAVING
7. **Milestone test**: Complex multi-table queries with joins, aggregates, and ordering

### Phase 7: Completeness & Compatibility
1. ALTER TABLE, DROP TABLE, DROP INDEX
2. AUTOINCREMENT
3. PRAGMA support (page_size, table_info, journal_mode, etc.)
4. Built-in scalar functions
5. LIKE operator with % and _ wildcards
6. CAST expressions
7. CASE/WHEN expressions
8. NULL handling edge cases
9. Type affinity edge cases
10. **Milestone test**: Run against SQLite's official test vectors

## Testing Strategy

### Unit Tests
- Every module should have comprehensive unit tests
- Test the tokenizer with edge cases (unicode, escape sequences, all keyword types)
- Test the parser by verifying AST structure for known SQL inputs
- Test the B-tree with random insert/delete/search sequences
- Test varint encoding round-trips
- Test record serialization round-trips

### Integration Tests
- Create a `tests/` directory with integration tests
- Each test should create a fresh database, perform operations, and verify results
- Cross-compatibility tests: write with rsqlite, read with official sqlite3, and vice versa
- Fuzz the parser with malformed SQL (should return errors, never panic)

### External Test Suites

#### SQLite's TCL Test Suite
- Clone the SQLite source: https://github.com/sqlite/sqlite (or fossil repo)
- The test suite is in `test/` directory
- Many tests are TCL scripts — you'll need to adapt or selectively run them
- Start with these test files as priorities:
  - `select1.test` through `select9.test`
  - `insert.test`, `insert2.test`, `insert3.test`
  - `delete.test`
  - `update.test`
  - `index.test`
  - `join.test` through `join6.test`
  - `expr.test`
  - `types.test`, `types2.test`
  - `null.test`
  - `trans.test`
  - `crash.test` (critical for journal correctness)

#### sqllogictest
- https://www.sqlite.org/sqllogictest/doc/trunk/about.wiki
- This is the BEST external test suite for a from-scratch implementation
- Contains millions of SQL statements with expected results
- Language-agnostic (just SQL in, results out)
- Start with the "select" test files which are simpler
- Build a test harness that reads `.test` files and executes them against your engine
- Track your pass rate as a progress metric

#### Write Your Own Conformance Tests
- Build a test harness that runs the same SQL against both rsqlite and official sqlite3
- Compare outputs byte-for-byte
- Automate this as a cargo test target
- This catches subtle behavioral differences

### Performance Benchmarks
- Implement benchmarks using `criterion` crate
- Key benchmarks:
  - Bulk insert throughput (1M rows)
  - Point query latency (by rowid, by indexed column)
  - Full table scan throughput
  - Complex query (join + aggregate + order by)
- Compare against official SQLite via its C library for context (not to match — just to measure the gap)

## Rust-Specific Guidance

### Error Handling
- Define a custom error enum (`RsqliteError`) using `thiserror`
- Use `Result<T, RsqliteError>` everywhere — never panic on user input
- Distinguish between: parse errors, runtime errors, I/O errors, constraint violations, corruption errors

### Memory Management for Trees
- Use index-based arena allocation for B-tree nodes rather than `Rc<RefCell<>>`
- The `slotmap` crate works well for this pattern
- Alternatively, nodes can be identified by page number (natural since they map to disk pages)

### Byte Manipulation
- Use the `bytes` crate for efficient buffer management in the pager
- Implement `byteorder`-style helpers for reading big-endian integers (SQLite uses big-endian on disk)
- Consider `zerocopy` for safe reinterpretation of page buffers

### File I/O
- Use `std::fs::File` with `seek` and `read_exact`/`write_all`
- Do NOT use memory-mapped I/O initially (adds complexity)
- Consider `BufReader`/`BufWriter` but be careful with consistency — the pager should manage its own buffering

### Concurrency (if implementing WAL)
- Use `std::sync::RwLock` for reader/writer coordination
- File-level locking via `fs2` crate for cross-process coordination
- Keep it single-threaded initially — correctness first

### Project Structure
```
rsqlite/
├── Cargo.toml
├── src/
│   ├── main.rs          # CLI entry point
│   ├── lib.rs           # Public API
│   ├── tokenizer.rs     # SQL tokenizer
│   ├── parser.rs        # SQL parser + AST types
│   ├── ast.rs           # AST node definitions
│   ├── planner.rs       # Query planner
│   ├── vm.rs            # Virtual machine / executor
│   ├── btree.rs         # B-tree implementation
│   ├── pager.rs         # Page cache and I/O
│   ├── format.rs        # File format constants and helpers
│   ├── journal.rs       # Transaction journal
│   ├── record.rs        # Record serialization
│   ├── varint.rs        # Variable-length integer encoding
│   ├── types.rs         # Value types, affinity, coercion
│   ├── functions.rs     # Built-in scalar functions
│   └── error.rs         # Error types
├── tests/
│   ├── integration/     # Integration tests
│   ├── compat/          # Cross-compatibility tests with sqlite3
│   └── sqllogictest/    # sqllogictest harness
└── benches/
    └── benchmarks.rs    # Criterion benchmarks
```

## Definition of Done

The project is considered complete when:

1. **File format compatibility**: Can read databases created by official SQLite3 and vice versa
2. **SQL coverage**: Supports SELECT (with WHERE, JOIN, GROUP BY, ORDER BY, LIMIT, subqueries), INSERT, UPDATE, DELETE, CREATE TABLE/INDEX, DROP TABLE/INDEX, ALTER TABLE, transactions
3. **ACID compliance**: Transactions are atomic (journal-based recovery works), data survives crashes
4. **CLI works**: Interactive REPL that feels like the real `sqlite3` CLI
5. **Test passage**: Passes >50% of sqllogictest "select" test files
6. **No unsafe**: Zero `unsafe` blocks unless absolutely necessary for performance-critical paths, and each one is documented with a safety comment
7. **No panics**: All errors are handled gracefully with descriptive error messages. Malformed input never causes a crash.

## Reference Materials

- SQLite file format spec: https://www.sqlite.org/fileformat2.html
- SQLite architecture doc: https://www.sqlite.org/arch.html
- SQLite virtual machine opcodes: https://www.sqlite.org/opcode.html
- SQLite query planner: https://www.sqlite.org/queryplanner.html
- SQLite SQL syntax diagrams: https://www.sqlite.org/syntaxdiagrams.html
- Architecture of SQLite (detailed): https://www.sqlite.org/zipvfs/doc/trunk/www/howitworks.wiki
- sqllogictest: https://www.sqlite.org/sqllogictest/doc/trunk/about.wiki
- "SQLite: Past, Present, and Future" (VLDB 2022 paper)
- Rust mini-lsm tutorial (similar storage engine concepts): https://github.com/skyzh/mini-lsm

## Important Notes

- Always run `cargo clippy` and fix warnings before considering any phase complete
- Run `cargo fmt` to maintain consistent formatting
- Commit after each milestone with a descriptive message
- If you get stuck on a design decision, document the trade-offs and pick the simpler option
- Correctness over performance. Always. You can optimize later.
- When in doubt about SQLite behavior, test against the official `sqlite3` binary — it is the spec
