# rsqlite

A fully functional SQLite-compatible database engine built from scratch in Rust — written entirely by AI.

## Goal

Build a complete SQLite implementation from first principles using only AI-generated code. No existing SQL parsing libraries, database engines, or SQLite bindings. Every component — from the B-tree storage engine to the SQL parser to the query executor — is implemented from scratch.

The target is to produce a database engine that:

- **Reads and writes SQLite3-compatible database files** — files created by rsqlite should be readable by the official `sqlite3` tool, and vice versa
- **Parses and executes SQL** — supports SELECT, INSERT, UPDATE, DELETE, CREATE TABLE/INDEX, JOINs, aggregates, subqueries, transactions, and more
- **Passes major SQLite test suites** — specifically [sqllogictest](https://www.sqlite.org/sqllogictest/doc/trunk/about.wiki) and portions of SQLite's official TCL test suite
- **Provides a CLI** — an interactive REPL similar to the `sqlite3` command-line tool

## Status

Under active development. Tracking progress against Phase 1 (storage layer foundation) of the implementation plan.

## Building

```
cargo build
```

## Testing

```
cargo test
```

## License

MIT
