// Integration tests for rsqlite.
//
// These tests exercise the full stack: parsing, planning, execution, and storage.
// Each test creates a fresh in-memory or file-backed database.

use rsqlite::planner::Database;
use rsqlite::types::Value;

/// Helper: execute SQL and return (columns, rows_as_vec_of_vec).
fn query(db: &mut Database, sql: &str) -> Vec<Vec<Value>> {
    let result = db.execute(sql).unwrap();
    result.rows.into_iter().map(|r| r.values).collect()
}

/// Helper: execute SQL, expect success, ignore result.
fn exec(db: &mut Database, sql: &str) {
    db.execute(sql).unwrap();
}

// ---- Basic CRUD ----

#[test]
fn test_create_insert_select() {
    let mut db = Database::in_memory();
    exec(&mut db, "CREATE TABLE t (a INTEGER, b TEXT);");
    exec(&mut db, "INSERT INTO t VALUES (1, 'hello');");
    exec(&mut db, "INSERT INTO t VALUES (2, 'world');");

    let rows = query(&mut db, "SELECT * FROM t;");
    assert_eq!(rows.len(), 2);
    assert_eq!(
        rows[0],
        vec![Value::Integer(1), Value::Text("hello".into())]
    );
    assert_eq!(
        rows[1],
        vec![Value::Integer(2), Value::Text("world".into())]
    );
}

#[test]
fn test_update() {
    let mut db = Database::in_memory();
    exec(
        &mut db,
        "CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT);",
    );
    exec(&mut db, "INSERT INTO t VALUES (1, 'alice');");
    exec(&mut db, "INSERT INTO t VALUES (2, 'bob');");
    exec(&mut db, "UPDATE t SET name = 'ALICE' WHERE id = 1;");

    let rows = query(&mut db, "SELECT name FROM t WHERE id = 1;");
    assert_eq!(rows, vec![vec![Value::Text("ALICE".into())]]);
}

#[test]
fn test_delete() {
    let mut db = Database::in_memory();
    exec(&mut db, "CREATE TABLE t (x INTEGER);");
    exec(&mut db, "INSERT INTO t VALUES (1);");
    exec(&mut db, "INSERT INTO t VALUES (2);");
    exec(&mut db, "INSERT INTO t VALUES (3);");
    exec(&mut db, "DELETE FROM t WHERE x = 2;");

    let rows = query(&mut db, "SELECT x FROM t;");
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0], vec![Value::Integer(1)]);
    assert_eq!(rows[1], vec![Value::Integer(3)]);
}

// ---- WHERE clause ----

#[test]
fn test_where_and_or() {
    let mut db = Database::in_memory();
    exec(&mut db, "CREATE TABLE t (a INTEGER, b INTEGER);");
    exec(&mut db, "INSERT INTO t VALUES (1, 10);");
    exec(&mut db, "INSERT INTO t VALUES (2, 20);");
    exec(&mut db, "INSERT INTO t VALUES (3, 30);");

    let rows = query(&mut db, "SELECT a FROM t WHERE a > 1 AND b < 30;");
    assert_eq!(rows, vec![vec![Value::Integer(2)]]);

    let rows = query(&mut db, "SELECT a FROM t WHERE a = 1 OR a = 3;");
    assert_eq!(rows.len(), 2);
}

#[test]
fn test_where_between() {
    let mut db = Database::in_memory();
    exec(&mut db, "CREATE TABLE t (x INTEGER);");
    for i in 1..=10 {
        exec(&mut db, &format!("INSERT INTO t VALUES ({i});"));
    }

    let rows = query(&mut db, "SELECT x FROM t WHERE x BETWEEN 3 AND 7;");
    assert_eq!(rows.len(), 5);
}

#[test]
fn test_where_in() {
    let mut db = Database::in_memory();
    exec(&mut db, "CREATE TABLE t (x INTEGER);");
    for i in 1..=5 {
        exec(&mut db, &format!("INSERT INTO t VALUES ({i});"));
    }

    let rows = query(&mut db, "SELECT x FROM t WHERE x IN (2, 4);");
    assert_eq!(rows.len(), 2);
}

#[test]
fn test_where_like() {
    let mut db = Database::in_memory();
    exec(&mut db, "CREATE TABLE t (name TEXT);");
    exec(&mut db, "INSERT INTO t VALUES ('alice');");
    exec(&mut db, "INSERT INTO t VALUES ('bob');");
    exec(&mut db, "INSERT INTO t VALUES ('alicia');");

    let rows = query(&mut db, "SELECT name FROM t WHERE name LIKE 'ali%';");
    assert_eq!(rows.len(), 2);
}

#[test]
fn test_where_is_null() {
    let mut db = Database::in_memory();
    exec(&mut db, "CREATE TABLE t (x INTEGER);");
    exec(&mut db, "INSERT INTO t VALUES (1);");
    exec(&mut db, "INSERT INTO t VALUES (NULL);");
    exec(&mut db, "INSERT INTO t VALUES (3);");

    let rows = query(&mut db, "SELECT x FROM t WHERE x IS NULL;");
    assert_eq!(rows, vec![vec![Value::Null]]);

    let rows = query(&mut db, "SELECT x FROM t WHERE x IS NOT NULL;");
    assert_eq!(rows.len(), 2);
}

// ---- ORDER BY, LIMIT, OFFSET ----

#[test]
fn test_order_by() {
    let mut db = Database::in_memory();
    exec(&mut db, "CREATE TABLE t (x INTEGER);");
    exec(&mut db, "INSERT INTO t VALUES (3);");
    exec(&mut db, "INSERT INTO t VALUES (1);");
    exec(&mut db, "INSERT INTO t VALUES (2);");

    let rows = query(&mut db, "SELECT x FROM t ORDER BY x ASC;");
    assert_eq!(
        rows,
        vec![
            vec![Value::Integer(1)],
            vec![Value::Integer(2)],
            vec![Value::Integer(3)],
        ]
    );

    let rows = query(&mut db, "SELECT x FROM t ORDER BY x DESC;");
    assert_eq!(
        rows,
        vec![
            vec![Value::Integer(3)],
            vec![Value::Integer(2)],
            vec![Value::Integer(1)],
        ]
    );
}

#[test]
fn test_limit_offset() {
    let mut db = Database::in_memory();
    exec(&mut db, "CREATE TABLE t (x INTEGER);");
    for i in 1..=10 {
        exec(&mut db, &format!("INSERT INTO t VALUES ({i});"));
    }

    let rows = query(&mut db, "SELECT x FROM t ORDER BY x LIMIT 3;");
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0], vec![Value::Integer(1)]);

    let rows = query(&mut db, "SELECT x FROM t ORDER BY x LIMIT 3 OFFSET 5;");
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0], vec![Value::Integer(6)]);
}

// ---- Aggregates ----

#[test]
fn test_aggregates() {
    let mut db = Database::in_memory();
    exec(&mut db, "CREATE TABLE t (x INTEGER);");
    exec(&mut db, "INSERT INTO t VALUES (10);");
    exec(&mut db, "INSERT INTO t VALUES (20);");
    exec(&mut db, "INSERT INTO t VALUES (30);");

    let rows = query(
        &mut db,
        "SELECT COUNT(*), SUM(x), AVG(x), MIN(x), MAX(x) FROM t;",
    );
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0][0], Value::Integer(3));
    assert_eq!(rows[0][1], Value::Integer(60));
    assert_eq!(rows[0][3], Value::Integer(10));
    assert_eq!(rows[0][4], Value::Integer(30));
}

#[test]
fn test_group_by() {
    let mut db = Database::in_memory();
    exec(&mut db, "CREATE TABLE t (category TEXT, amount INTEGER);");
    exec(&mut db, "INSERT INTO t VALUES ('a', 10);");
    exec(&mut db, "INSERT INTO t VALUES ('b', 20);");
    exec(&mut db, "INSERT INTO t VALUES ('a', 30);");
    exec(&mut db, "INSERT INTO t VALUES ('b', 40);");

    let rows = query(
        &mut db,
        "SELECT category, SUM(amount) FROM t GROUP BY category ORDER BY category;",
    );
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0], vec![Value::Text("a".into()), Value::Integer(40)]);
    assert_eq!(rows[1], vec![Value::Text("b".into()), Value::Integer(60)]);
}

#[test]
fn test_having() {
    let mut db = Database::in_memory();
    exec(&mut db, "CREATE TABLE t (category TEXT, amount INTEGER);");
    exec(&mut db, "INSERT INTO t VALUES ('a', 10);");
    exec(&mut db, "INSERT INTO t VALUES ('b', 20);");
    exec(&mut db, "INSERT INTO t VALUES ('a', 30);");
    exec(&mut db, "INSERT INTO t VALUES ('b', 40);");

    // HAVING on the group key (supported).
    let rows = query(
        &mut db,
        "SELECT category, SUM(amount) FROM t GROUP BY category HAVING category = 'b';",
    );
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0][0], Value::Text("b".into()));
    assert_eq!(rows[0][1], Value::Integer(60));
}

// ---- JOINs ----

#[test]
fn test_inner_join() {
    let mut db = Database::in_memory();
    exec(
        &mut db,
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);",
    );
    exec(
        &mut db,
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, item TEXT);",
    );
    exec(&mut db, "INSERT INTO users VALUES (1, 'alice');");
    exec(&mut db, "INSERT INTO users VALUES (2, 'bob');");
    exec(&mut db, "INSERT INTO orders VALUES (1, 1, 'book');");
    exec(&mut db, "INSERT INTO orders VALUES (2, 1, 'pen');");
    exec(&mut db, "INSERT INTO orders VALUES (3, 2, 'hat');");

    let rows = query(
        &mut db,
        "SELECT users.name, orders.item FROM users INNER JOIN orders ON users.id = orders.user_id ORDER BY orders.id;",
    );
    assert_eq!(rows.len(), 3);
    assert_eq!(
        rows[0],
        vec![Value::Text("alice".into()), Value::Text("book".into())]
    );
}

#[test]
fn test_left_join() {
    let mut db = Database::in_memory();
    exec(&mut db, "CREATE TABLE a (id INTEGER, val TEXT);");
    exec(&mut db, "CREATE TABLE b (id INTEGER, ref_id INTEGER);");
    exec(&mut db, "INSERT INTO a VALUES (1, 'x');");
    exec(&mut db, "INSERT INTO a VALUES (2, 'y');");
    exec(&mut db, "INSERT INTO b VALUES (1, 1);");

    let rows = query(
        &mut db,
        "SELECT a.val, b.id FROM a LEFT JOIN b ON a.id = b.ref_id ORDER BY a.id;",
    );
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[1][1], Value::Null); // no match for a.id=2
}

// ---- Subqueries ----

#[test]
fn test_in_subquery() {
    let mut db = Database::in_memory();
    exec(&mut db, "CREATE TABLE t1 (x INTEGER);");
    exec(&mut db, "CREATE TABLE t2 (y INTEGER);");
    exec(&mut db, "INSERT INTO t1 VALUES (1);");
    exec(&mut db, "INSERT INTO t1 VALUES (2);");
    exec(&mut db, "INSERT INTO t1 VALUES (3);");
    exec(&mut db, "INSERT INTO t2 VALUES (2);");
    exec(&mut db, "INSERT INTO t2 VALUES (3);");

    let rows = query(&mut db, "SELECT x FROM t1 WHERE x IN (SELECT y FROM t2);");
    assert_eq!(rows.len(), 2);
}

#[test]
fn test_exists_subquery() {
    let mut db = Database::in_memory();
    exec(&mut db, "CREATE TABLE t1 (x INTEGER);");
    exec(&mut db, "CREATE TABLE t2 (y INTEGER);");
    exec(&mut db, "INSERT INTO t1 VALUES (1);");
    exec(&mut db, "INSERT INTO t2 VALUES (99);");

    let rows = query(&mut db, "SELECT x FROM t1 WHERE EXISTS (SELECT 1 FROM t2);");
    assert_eq!(rows.len(), 1);
}

// ---- DISTINCT ----

#[test]
fn test_select_distinct() {
    let mut db = Database::in_memory();
    exec(&mut db, "CREATE TABLE t (x INTEGER);");
    exec(&mut db, "INSERT INTO t VALUES (1);");
    exec(&mut db, "INSERT INTO t VALUES (2);");
    exec(&mut db, "INSERT INTO t VALUES (1);");
    exec(&mut db, "INSERT INTO t VALUES (2);");

    let rows = query(&mut db, "SELECT DISTINCT x FROM t ORDER BY x;");
    assert_eq!(rows.len(), 2);
}

// ---- UNION / INTERSECT / EXCEPT ----

#[test]
fn test_union() {
    let mut db = Database::in_memory();
    exec(&mut db, "CREATE TABLE t1 (x INTEGER);");
    exec(&mut db, "CREATE TABLE t2 (y INTEGER);");
    exec(&mut db, "INSERT INTO t1 VALUES (1);");
    exec(&mut db, "INSERT INTO t1 VALUES (2);");
    exec(&mut db, "INSERT INTO t2 VALUES (2);");
    exec(&mut db, "INSERT INTO t2 VALUES (3);");

    let rows = query(&mut db, "SELECT x FROM t1 UNION SELECT y FROM t2;");
    assert_eq!(rows.len(), 3); // 1, 2, 3 (deduped)
}

#[test]
fn test_union_all() {
    let mut db = Database::in_memory();
    exec(&mut db, "CREATE TABLE t1 (x INTEGER);");
    exec(&mut db, "CREATE TABLE t2 (y INTEGER);");
    exec(&mut db, "INSERT INTO t1 VALUES (1);");
    exec(&mut db, "INSERT INTO t1 VALUES (2);");
    exec(&mut db, "INSERT INTO t2 VALUES (2);");
    exec(&mut db, "INSERT INTO t2 VALUES (3);");

    let rows = query(&mut db, "SELECT x FROM t1 UNION ALL SELECT y FROM t2;");
    assert_eq!(rows.len(), 4); // 1, 2, 2, 3 (no dedup)
}

// ---- Transactions ----

#[test]
fn test_transaction_commit() {
    let mut db = Database::in_memory();
    exec(&mut db, "CREATE TABLE t (x INTEGER);");
    exec(&mut db, "BEGIN;");
    exec(&mut db, "INSERT INTO t VALUES (1);");
    exec(&mut db, "INSERT INTO t VALUES (2);");
    exec(&mut db, "COMMIT;");

    let rows = query(&mut db, "SELECT x FROM t;");
    assert_eq!(rows.len(), 2);
}

#[test]
fn test_transaction_rollback() {
    let mut db = Database::in_memory();
    exec(&mut db, "CREATE TABLE t (x INTEGER);");
    exec(&mut db, "INSERT INTO t VALUES (1);");
    exec(&mut db, "BEGIN;");
    exec(&mut db, "INSERT INTO t VALUES (2);");
    exec(&mut db, "INSERT INTO t VALUES (3);");
    exec(&mut db, "ROLLBACK;");

    let rows = query(&mut db, "SELECT x FROM t;");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0], vec![Value::Integer(1)]);
}

// ---- Indexes ----

#[test]
fn test_index_maintains_correctness() {
    let mut db = Database::in_memory();
    exec(
        &mut db,
        "CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT);",
    );
    exec(&mut db, "CREATE INDEX idx_name ON t (name);");
    exec(&mut db, "INSERT INTO t VALUES (1, 'charlie');");
    exec(&mut db, "INSERT INTO t VALUES (2, 'alice');");
    exec(&mut db, "INSERT INTO t VALUES (3, 'bob');");

    // Index should not change query results.
    let rows = query(&mut db, "SELECT name FROM t ORDER BY name;");
    assert_eq!(
        rows,
        vec![
            vec![Value::Text("alice".into())],
            vec![Value::Text("bob".into())],
            vec![Value::Text("charlie".into())],
        ]
    );

    // Delete and verify index stays consistent.
    exec(&mut db, "DELETE FROM t WHERE name = 'bob';");
    let rows = query(&mut db, "SELECT name FROM t ORDER BY name;");
    assert_eq!(rows.len(), 2);
}

// ---- Constraints ----

#[test]
fn test_not_null_constraint() {
    let mut db = Database::in_memory();
    exec(&mut db, "CREATE TABLE t (x INTEGER NOT NULL);");
    let err = db.execute("INSERT INTO t VALUES (NULL);");
    assert!(err.is_err());
}

#[test]
fn test_default_values() {
    let mut db = Database::in_memory();
    exec(
        &mut db,
        "CREATE TABLE t (id INTEGER PRIMARY KEY, status TEXT DEFAULT 'active');",
    );
    exec(&mut db, "INSERT INTO t (id) VALUES (1);");

    let rows = query(&mut db, "SELECT status FROM t;");
    assert_eq!(rows, vec![vec![Value::Text("active".into())]]);
}

// ---- Scalar functions ----

#[test]
fn test_scalar_functions() {
    let mut db = Database::in_memory();
    let rows = query(
        &mut db,
        "SELECT length('hello'), upper('hello'), lower('WORLD');",
    );
    assert_eq!(rows[0][0], Value::Integer(5));
    assert_eq!(rows[0][1], Value::Text("HELLO".into()));
    assert_eq!(rows[0][2], Value::Text("world".into()));
}

#[test]
fn test_typeof_function() {
    let mut db = Database::in_memory();
    let rows = query(
        &mut db,
        "SELECT typeof(1), typeof(1.5), typeof('hi'), typeof(NULL), typeof(X'00');",
    );
    assert_eq!(rows[0][0], Value::Text("integer".into()));
    assert_eq!(rows[0][1], Value::Text("real".into()));
    assert_eq!(rows[0][2], Value::Text("text".into()));
    assert_eq!(rows[0][3], Value::Text("null".into()));
    assert_eq!(rows[0][4], Value::Text("blob".into()));
}

#[test]
fn test_coalesce() {
    let mut db = Database::in_memory();
    let rows = query(&mut db, "SELECT coalesce(NULL, NULL, 42, 99);");
    assert_eq!(rows[0][0], Value::Integer(42));
}

// ---- CAST ----

#[test]
fn test_cast() {
    let mut db = Database::in_memory();
    let rows = query(&mut db, "SELECT CAST('123' AS INTEGER), CAST(42 AS TEXT);");
    assert_eq!(rows[0][0], Value::Integer(123));
    assert_eq!(rows[0][1], Value::Text("42".into()));
}

// ---- CASE/WHEN ----

#[test]
fn test_case_when() {
    let mut db = Database::in_memory();
    exec(&mut db, "CREATE TABLE t (x INTEGER);");
    exec(&mut db, "INSERT INTO t VALUES (1);");
    exec(&mut db, "INSERT INTO t VALUES (2);");
    exec(&mut db, "INSERT INTO t VALUES (3);");

    let rows = query(
        &mut db,
        "SELECT x, CASE WHEN x = 1 THEN 'one' WHEN x = 2 THEN 'two' ELSE 'other' END FROM t ORDER BY x;",
    );
    assert_eq!(rows[0][1], Value::Text("one".into()));
    assert_eq!(rows[1][1], Value::Text("two".into()));
    assert_eq!(rows[2][1], Value::Text("other".into()));
}

// ---- Multi-row INSERT ----

#[test]
fn test_multi_row_insert() {
    let mut db = Database::in_memory();
    exec(&mut db, "CREATE TABLE t (x INTEGER, y TEXT);");
    exec(
        &mut db,
        "INSERT INTO t VALUES (1, 'a'), (2, 'b'), (3, 'c');",
    );

    let rows = query(&mut db, "SELECT COUNT(*) FROM t;");
    assert_eq!(rows[0][0], Value::Integer(3));
}

// ---- INSERT ... SELECT ----

#[test]
fn test_insert_select() {
    let mut db = Database::in_memory();
    exec(&mut db, "CREATE TABLE src (x INTEGER);");
    exec(&mut db, "CREATE TABLE dst (x INTEGER);");
    exec(&mut db, "INSERT INTO src VALUES (1), (2), (3);");
    exec(&mut db, "INSERT INTO dst SELECT x FROM src WHERE x > 1;");

    let rows = query(&mut db, "SELECT x FROM dst ORDER BY x;");
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0], vec![Value::Integer(2)]);
}

// ---- ALTER TABLE ----

#[test]
fn test_alter_table_add_column() {
    let mut db = Database::in_memory();
    exec(&mut db, "CREATE TABLE t (x INTEGER);");
    exec(&mut db, "INSERT INTO t VALUES (1);");
    exec(&mut db, "ALTER TABLE t ADD COLUMN y TEXT;");

    let rows = query(&mut db, "SELECT x, y FROM t;");
    assert_eq!(rows[0], vec![Value::Integer(1), Value::Null]);
}

#[test]
fn test_alter_table_rename() {
    let mut db = Database::in_memory();
    exec(&mut db, "CREATE TABLE old_name (x INTEGER);");
    exec(&mut db, "INSERT INTO old_name VALUES (1);");
    exec(&mut db, "ALTER TABLE old_name RENAME TO new_name;");

    let rows = query(&mut db, "SELECT x FROM new_name;");
    assert_eq!(rows.len(), 1);
}

// ---- DROP TABLE ----

#[test]
fn test_drop_table() {
    let mut db = Database::in_memory();
    exec(&mut db, "CREATE TABLE t (x INTEGER);");
    exec(&mut db, "INSERT INTO t VALUES (1);");
    exec(&mut db, "DROP TABLE t;");

    let err = db.execute("SELECT * FROM t;");
    assert!(err.is_err());
}

// ---- Expressions in SELECT ----

#[test]
fn test_expression_select() {
    let mut db = Database::in_memory();
    let rows = query(
        &mut db,
        "SELECT 1 + 2, 10 * 3 - 5, 'hello' || ' ' || 'world';",
    );
    assert_eq!(rows[0][0], Value::Integer(3));
    assert_eq!(rows[0][1], Value::Integer(25));
    assert_eq!(rows[0][2], Value::Text("hello world".into()));
}

// ---- File-backed database persistence ----

#[test]
fn test_file_persistence() {
    let dir = std::env::temp_dir().join("rsqlite_integration_test");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let db_path = dir.join("test.db");

    // Write data.
    {
        let mut db = Database::open(db_path.to_str().unwrap()).unwrap();
        exec(
            &mut db,
            "CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT);",
        );
        exec(&mut db, "INSERT INTO t VALUES (1, 'alice');");
        exec(&mut db, "INSERT INTO t VALUES (2, 'bob');");
    }

    // Re-open and verify data persists.
    {
        let mut db = Database::open(db_path.to_str().unwrap()).unwrap();
        let rows = query(&mut db, "SELECT name FROM t ORDER BY id;");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0], vec![Value::Text("alice".into())]);
        assert_eq!(rows[1], vec![Value::Text("bob".into())]);
    }

    let _ = std::fs::remove_dir_all(&dir);
}

// ---- Complex query combining multiple features ----

#[test]
fn test_complex_query() {
    let mut db = Database::in_memory();
    exec(
        &mut db,
        "CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, category TEXT, price REAL);",
    );
    exec(
        &mut db,
        "INSERT INTO products VALUES (1, 'Widget', 'A', 9.99);",
    );
    exec(
        &mut db,
        "INSERT INTO products VALUES (2, 'Gadget', 'B', 24.99);",
    );
    exec(
        &mut db,
        "INSERT INTO products VALUES (3, 'Doohickey', 'A', 14.99);",
    );
    exec(
        &mut db,
        "INSERT INTO products VALUES (4, 'Thingamajig', 'B', 34.99);",
    );
    exec(
        &mut db,
        "INSERT INTO products VALUES (5, 'Whatchamacallit', 'A', 4.99);",
    );

    // Aggregate with GROUP BY and ORDER BY.
    let rows = query(
        &mut db,
        "SELECT category, COUNT(*), SUM(price) FROM products GROUP BY category ORDER BY category;",
    );
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0][0], Value::Text("A".into()));
    assert_eq!(rows[0][1], Value::Integer(3));
    assert_eq!(rows[1][0], Value::Text("B".into()));
    assert_eq!(rows[1][1], Value::Integer(2));
}
