/// Integration tests for rsqlite database.
/// These tests verify end-to-end behavior including persistence, complex queries,
/// and multi-statement workflows.
use rsqlite::{Database, Value};

#[test]
fn test_full_crud_workflow() {
    let mut db = Database::new_memory();

    // CREATE
    db.execute("CREATE TABLE employees (name TEXT NOT NULL, salary REAL)")
        .unwrap();

    // INSERT
    db.execute("INSERT INTO employees VALUES ('Alice', 75000.0)")
        .unwrap();
    db.execute("INSERT INTO employees VALUES ('Bob', 85000.0)")
        .unwrap();
    db.execute("INSERT INTO employees VALUES ('Carol', 65000.0)")
        .unwrap();

    // READ
    let result = db
        .execute("SELECT name, salary FROM employees ORDER BY salary DESC")
        .unwrap();
    assert_eq!(result.rows.len(), 3);
    // Bob has highest salary (85000)
    assert_eq!(
        result.rows[0][0],
        Value::Text("Bob".into()),
        "rows: {:?}",
        result.rows
    );

    // UPDATE
    db.execute("UPDATE employees SET salary = 90000.0 WHERE name = 'Alice'")
        .unwrap();
    let result = db
        .execute("SELECT salary FROM employees WHERE name = 'Alice'")
        .unwrap();
    assert_eq!(result.rows[0][0], Value::Real(90000.0));

    // DELETE
    db.execute("DELETE FROM employees WHERE name = 'Carol'")
        .unwrap();
    let result = db.execute("SELECT COUNT(*) FROM employees").unwrap();
    assert_eq!(result.rows[0][0], Value::Integer(2));
}

#[test]
fn test_complex_query_with_joins_and_aggregates() {
    let mut db = Database::new_memory();

    db.execute("CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT)")
        .unwrap();
    db.execute(
        "CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, dept_id INTEGER, salary REAL)",
    )
    .unwrap();

    db.execute("INSERT INTO departments VALUES (1, 'Engineering')")
        .unwrap();
    db.execute("INSERT INTO departments VALUES (2, 'Sales')")
        .unwrap();
    db.execute("INSERT INTO departments VALUES (3, 'Marketing')")
        .unwrap();

    db.execute("INSERT INTO employees VALUES (1, 'Alice', 1, 90000)")
        .unwrap();
    db.execute("INSERT INTO employees VALUES (2, 'Bob', 1, 85000)")
        .unwrap();
    db.execute("INSERT INTO employees VALUES (3, 'Carol', 2, 75000)")
        .unwrap();
    db.execute("INSERT INTO employees VALUES (4, 'Dave', 2, 70000)")
        .unwrap();
    db.execute("INSERT INTO employees VALUES (5, 'Eve', 3, 80000)")
        .unwrap();

    // JOIN with aggregate
    let result = db
        .execute(
            "SELECT d.name, COUNT(*), AVG(e.salary) FROM employees e \
             INNER JOIN departments d ON e.dept_id = d.id \
             GROUP BY d.name ORDER BY d.name",
        )
        .unwrap();

    assert_eq!(result.rows.len(), 3);
    // Engineering: 2 employees, avg 87500
    assert_eq!(result.rows[0][0], Value::Text("Engineering".into()));
    assert_eq!(result.rows[0][1], Value::Integer(2));

    // LEFT JOIN - departments with no employees
    let result = db
        .execute(
            "SELECT d.name, e.name FROM departments d \
             LEFT JOIN employees e ON d.id = e.dept_id \
             ORDER BY d.name, e.name",
        )
        .unwrap();
    assert!(result.rows.len() >= 5);
}

#[test]
fn test_subqueries() {
    let mut db = Database::new_memory();

    db.execute("CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price REAL)")
        .unwrap();
    db.execute("INSERT INTO products VALUES (1, 'Widget', 9.99)")
        .unwrap();
    db.execute("INSERT INTO products VALUES (2, 'Gadget', 24.99)")
        .unwrap();
    db.execute("INSERT INTO products VALUES (3, 'Doohickey', 4.99)")
        .unwrap();

    // Subquery in WHERE
    let result = db
        .execute("SELECT name FROM products WHERE price > (SELECT AVG(price) FROM products)")
        .unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::Text("Gadget".into()));

    // EXISTS subquery
    let result = db
        .execute("SELECT EXISTS(SELECT 1 FROM products WHERE price < 5)")
        .unwrap();
    assert_eq!(result.rows[0][0], Value::Integer(1));

    // IN subquery
    let result = db
        .execute("SELECT name FROM products WHERE id IN (SELECT id FROM products WHERE price > 10)")
        .unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::Text("Gadget".into()));
}

#[test]
fn test_transactions() {
    let mut db = Database::new_memory();
    db.execute("CREATE TABLE t (x INTEGER)").unwrap();
    db.execute("INSERT INTO t VALUES (1)").unwrap();

    // Begin transaction, make changes, then rollback
    db.execute("BEGIN").unwrap();
    db.execute("INSERT INTO t VALUES (2)").unwrap();
    db.execute("INSERT INTO t VALUES (3)").unwrap();

    let result = db.execute("SELECT COUNT(*) FROM t").unwrap();
    assert_eq!(result.rows[0][0], Value::Integer(3));

    db.execute("ROLLBACK").unwrap();

    let result = db.execute("SELECT COUNT(*) FROM t").unwrap();
    assert_eq!(result.rows[0][0], Value::Integer(1));

    // Begin, commit
    db.execute("BEGIN").unwrap();
    db.execute("INSERT INTO t VALUES (10)").unwrap();
    db.execute("COMMIT").unwrap();

    let result = db.execute("SELECT COUNT(*) FROM t").unwrap();
    assert_eq!(result.rows[0][0], Value::Integer(2));
}

#[test]
fn test_compound_selects() {
    let mut db = Database::new_memory();
    db.execute("CREATE TABLE a (x INTEGER)").unwrap();
    db.execute("CREATE TABLE b (x INTEGER)").unwrap();

    db.execute("INSERT INTO a VALUES (1)").unwrap();
    db.execute("INSERT INTO a VALUES (2)").unwrap();
    db.execute("INSERT INTO a VALUES (3)").unwrap();

    db.execute("INSERT INTO b VALUES (2)").unwrap();
    db.execute("INSERT INTO b VALUES (3)").unwrap();
    db.execute("INSERT INTO b VALUES (4)").unwrap();

    // UNION (deduplicates)
    let result = db
        .execute("SELECT x FROM a UNION SELECT x FROM b ORDER BY x")
        .unwrap();
    assert_eq!(result.rows.len(), 4); // 1,2,3,4

    // UNION ALL (no dedup)
    let result = db
        .execute("SELECT x FROM a UNION ALL SELECT x FROM b ORDER BY x")
        .unwrap();
    assert_eq!(result.rows.len(), 6);

    // INTERSECT
    let result = db
        .execute("SELECT x FROM a INTERSECT SELECT x FROM b ORDER BY x")
        .unwrap();
    assert_eq!(result.rows.len(), 2); // 2,3

    // EXCEPT
    let result = db
        .execute("SELECT x FROM a EXCEPT SELECT x FROM b ORDER BY x")
        .unwrap();
    assert_eq!(result.rows.len(), 1); // 1
    assert_eq!(result.rows[0][0], Value::Integer(1));
}

#[test]
fn test_index_creation_and_constraint_enforcement() {
    let mut db = Database::new_memory();
    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT, name TEXT)")
        .unwrap();
    db.execute("CREATE UNIQUE INDEX idx_email ON users(email)")
        .unwrap();

    db.execute("INSERT INTO users VALUES (1, 'alice@test.com', 'Alice')")
        .unwrap();
    db.execute("INSERT INTO users VALUES (2, 'bob@test.com', 'Bob')")
        .unwrap();

    // Duplicate email should fail
    let result = db.execute("INSERT INTO users VALUES (3, 'alice@test.com', 'Alice2')");
    assert!(result.is_err());

    // INSERT OR IGNORE should skip
    db.execute("INSERT OR IGNORE INTO users VALUES (3, 'alice@test.com', 'Alice2')")
        .unwrap();
    let result = db.execute("SELECT COUNT(*) FROM users").unwrap();
    assert_eq!(result.rows[0][0], Value::Integer(2));

    // INSERT OR REPLACE should replace
    db.execute("INSERT OR REPLACE INTO users VALUES (3, 'alice@test.com', 'Alice2')")
        .unwrap();
    let result = db
        .execute("SELECT name FROM users WHERE email = 'alice@test.com'")
        .unwrap();
    assert_eq!(result.rows[0][0], Value::Text("Alice2".into()));
}

#[test]
fn test_schema_modifications() {
    let mut db = Database::new_memory();
    db.execute("CREATE TABLE t (a INTEGER, b TEXT)").unwrap();
    db.execute("INSERT INTO t VALUES (1, 'hello')").unwrap();

    // ALTER TABLE ADD COLUMN
    db.execute("ALTER TABLE t ADD COLUMN c REAL").unwrap();
    let result = db.execute("SELECT * FROM t").unwrap();
    assert_eq!(result.columns.len(), 3);

    // ALTER TABLE RENAME
    db.execute("ALTER TABLE t RENAME TO t2").unwrap();
    assert!(!db.tables.contains_key("t"));
    assert!(db.tables.contains_key("t2"));

    // DROP TABLE
    db.execute("DROP TABLE t2").unwrap();
    assert!(!db.tables.contains_key("t2"));

    // DROP TABLE IF EXISTS (should not error)
    db.execute("DROP TABLE IF EXISTS t2").unwrap();
}

#[test]
fn test_type_coercion_and_affinity() {
    let mut db = Database::new_memory();
    db.execute("CREATE TABLE t (i INTEGER, r REAL, t TEXT)")
        .unwrap();

    // Insert typed values directly
    db.execute("INSERT INTO t VALUES (42, 3.14, 'hello')")
        .unwrap();

    let result = db
        .execute("SELECT typeof(i), typeof(r), typeof(t) FROM t")
        .unwrap();
    assert_eq!(result.rows[0][0], Value::Text("integer".into()));
    assert_eq!(result.rows[0][1], Value::Text("real".into()));
    assert_eq!(result.rows[0][2], Value::Text("text".into()));

    // Type affinity with CAST
    let result = db.execute("SELECT CAST('42' AS INTEGER)").unwrap();
    assert_eq!(result.rows[0][0], Value::Integer(42));

    let result = db.execute("SELECT CAST(42 AS REAL)").unwrap();
    assert_eq!(result.rows[0][0], Value::Real(42.0));
}

#[test]
fn test_null_handling_comprehensive() {
    let mut db = Database::new_memory();
    db.execute("CREATE TABLE t (x INTEGER, y TEXT)").unwrap();
    db.execute("INSERT INTO t VALUES (1, 'a')").unwrap();
    db.execute("INSERT INTO t VALUES (NULL, 'b')").unwrap();
    db.execute("INSERT INTO t VALUES (3, NULL)").unwrap();

    // NULL comparison
    let result = db.execute("SELECT * FROM t WHERE x IS NULL").unwrap();
    assert_eq!(result.rows.len(), 1);

    // COALESCE
    let result = db
        .execute("SELECT COALESCE(y, 'default') FROM t ORDER BY COALESCE(x, 999)")
        .unwrap();
    assert_eq!(result.rows.len(), 3);
    assert_eq!(result.rows[2][0], Value::Text("default".into()));

    // IFNULL
    let result = db
        .execute("SELECT IFNULL(x, -1) FROM t WHERE y = 'b'")
        .unwrap();
    assert_eq!(result.rows[0][0], Value::Integer(-1));

    // NULLIF
    let result = db.execute("SELECT NULLIF(1, 1)").unwrap();
    assert_eq!(result.rows[0][0], Value::Null);
    let result = db.execute("SELECT NULLIF(1, 2)").unwrap();
    assert_eq!(result.rows[0][0], Value::Integer(1));
}

#[test]
fn test_scalar_functions() {
    let mut db = Database::new_memory();

    // String functions
    let result = db.execute("SELECT LENGTH('hello')").unwrap();
    assert_eq!(result.rows[0][0], Value::Integer(5));

    let result = db.execute("SELECT UPPER('hello')").unwrap();
    assert_eq!(result.rows[0][0], Value::Text("HELLO".into()));

    let result = db.execute("SELECT LOWER('HELLO')").unwrap();
    assert_eq!(result.rows[0][0], Value::Text("hello".into()));

    let result = db.execute("SELECT SUBSTR('hello', 2, 3)").unwrap();
    assert_eq!(result.rows[0][0], Value::Text("ell".into()));

    let result = db
        .execute("SELECT REPLACE('hello world', 'world', 'rust')")
        .unwrap();
    assert_eq!(result.rows[0][0], Value::Text("hello rust".into()));

    let result = db.execute("SELECT TRIM('  hello  ')").unwrap();
    assert_eq!(result.rows[0][0], Value::Text("hello".into()));

    // Math functions
    let result = db.execute("SELECT ABS(-42)").unwrap();
    assert_eq!(result.rows[0][0], Value::Integer(42));

    let result = db.execute("SELECT ROUND(3.14159, 2)").unwrap();
    assert_eq!(result.rows[0][0], Value::Real(3.14));

    let result = db.execute("SELECT SIGN(-5)").unwrap();
    assert_eq!(result.rows[0][0], Value::Integer(-1));

    // Conditional
    let result = db.execute("SELECT IIF(1 > 0, 'yes', 'no')").unwrap();
    assert_eq!(result.rows[0][0], Value::Text("yes".into()));
}

#[test]
fn test_expressions_and_operators() {
    let mut db = Database::new_memory();

    // Arithmetic
    let result = db
        .execute("SELECT 10 + 20, 50 - 30, 6 * 7, 100 / 3, 17 % 5")
        .unwrap();
    assert_eq!(result.rows[0][0], Value::Integer(30));
    assert_eq!(result.rows[0][1], Value::Integer(20));
    assert_eq!(result.rows[0][2], Value::Integer(42));
    assert_eq!(result.rows[0][3], Value::Integer(33));
    assert_eq!(result.rows[0][4], Value::Integer(2));

    // String concatenation
    let result = db.execute("SELECT 'hello' || ' ' || 'world'").unwrap();
    assert_eq!(result.rows[0][0], Value::Text("hello world".into()));

    // BETWEEN
    let result = db.execute("SELECT 5 BETWEEN 1 AND 10").unwrap();
    assert_eq!(result.rows[0][0], Value::Integer(1));

    // IN
    let result = db.execute("SELECT 3 IN (1, 2, 3, 4)").unwrap();
    assert_eq!(result.rows[0][0], Value::Integer(1));

    // CASE
    let result = db
        .execute("SELECT CASE WHEN 1 > 0 THEN 'positive' ELSE 'non-positive' END")
        .unwrap();
    assert_eq!(result.rows[0][0], Value::Text("positive".into()));

    // CAST
    let result = db.execute("SELECT CAST('42' AS INTEGER)").unwrap();
    assert_eq!(result.rows[0][0], Value::Integer(42));
}

#[test]
fn test_file_persistence() {
    let tmp_path = "/tmp/rsqlite_test_persist.db";

    // Clean up from prior runs
    let _ = std::fs::remove_file(tmp_path);

    {
        let mut db = Database::open(tmp_path).unwrap();
        db.execute("CREATE TABLE persist_test (id INTEGER PRIMARY KEY, data TEXT)")
            .unwrap();
        db.execute("INSERT INTO persist_test VALUES (1, 'hello')")
            .unwrap();
        db.execute("INSERT INTO persist_test VALUES (2, 'world')")
            .unwrap();
    }

    {
        let mut db = Database::open(tmp_path).unwrap();
        let result = db
            .execute("SELECT data FROM persist_test ORDER BY id")
            .unwrap();
        assert_eq!(result.rows.len(), 2);
        assert_eq!(result.rows[0][0], Value::Text("hello".into()));
        assert_eq!(result.rows[1][0], Value::Text("world".into()));
    }

    let _ = std::fs::remove_file(tmp_path);
}

#[test]
fn test_error_handling() {
    let mut db = Database::new_memory();

    // Table not found
    let result = db.execute("SELECT * FROM nonexistent");
    assert!(result.is_err());

    // Duplicate table
    db.execute("CREATE TABLE t (x INTEGER)").unwrap();
    let result = db.execute("CREATE TABLE t (y TEXT)");
    assert!(result.is_err());

    // Syntax error
    let result = db.execute("SELEC * FROM t");
    assert!(result.is_err());

    // NOT NULL violation
    db.execute("CREATE TABLE nn (x INTEGER NOT NULL)").unwrap();
    let result = db.execute("INSERT INTO nn VALUES (NULL)");
    assert!(result.is_err());
}

#[test]
fn test_pragma_statements() {
    let mut db = Database::new_memory();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT NOT NULL, age INTEGER)")
        .unwrap();

    // PRAGMA table_info
    let result = db.execute("PRAGMA table_info(t)").unwrap();
    assert_eq!(result.rows.len(), 3);

    // PRAGMA page_size
    let result = db.execute("PRAGMA page_size").unwrap();
    assert_eq!(result.rows[0][0], Value::Integer(4096));

    // PRAGMA integrity_check
    let result = db.execute("PRAGMA integrity_check").unwrap();
    assert_eq!(result.rows[0][0], Value::Text("ok".into()));

    // PRAGMA encoding
    let result = db.execute("PRAGMA encoding").unwrap();
    assert_eq!(result.rows[0][0], Value::Text("UTF-8".into()));

    // PRAGMA database_list
    let result = db.execute("PRAGMA database_list").unwrap();
    assert!(!result.rows.is_empty());
}
