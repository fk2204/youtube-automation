"""
Database optimization utilities.

Provides:
- Index management
- Query analysis
- Vacuum/optimize operations
- Connection pooling

Usage:
    from src.utils.db_optimizer import DatabaseOptimizer, ConnectionPool, QueryCache

    # Optimize a specific database
    optimizer = DatabaseOptimizer("data/token_usage.db")
    optimizer.vacuum()
    optimizer.create_recommended_indexes()

    # Use connection pool
    pool = ConnectionPool("data/pipeline.db", max_connections=5)
    with pool.get_connection() as conn:
        cursor = conn.execute("SELECT * FROM pipelines")
        results = cursor.fetchall()

    # Optimize all databases in the project
    optimize_all_databases()
"""

import sqlite3
import time
import threading
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict
from loguru import logger


@dataclass
class QueryStats:
    """Statistics for a single query execution."""
    query: str
    execution_time_ms: float
    rows_affected: int
    uses_index: bool
    scan_type: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query[:100] + "..." if len(self.query) > 100 else self.query,
            "execution_time_ms": self.execution_time_ms,
            "rows_affected": self.rows_affected,
            "uses_index": self.uses_index,
            "scan_type": self.scan_type
        }


@dataclass
class TableInfo:
    """Information about a database table."""
    name: str
    row_count: int
    column_count: int
    columns: List[str]
    indexes: List[str]
    size_estimate_kb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": self.columns,
            "indexes": self.indexes,
            "size_estimate_kb": self.size_estimate_kb
        }


@dataclass
class IndexRecommendation:
    """A recommended index to create."""
    table_name: str
    column_name: str
    index_name: str
    reason: str
    priority: str = "medium"  # low, medium, high

    def get_create_sql(self) -> str:
        return f"CREATE INDEX IF NOT EXISTS {self.index_name} ON {self.table_name}({self.column_name})"


class DatabaseOptimizer:
    """Optimize SQLite database performance."""

    # Common columns that benefit from indexing
    INDEX_CANDIDATES = [
        "id", "date", "timestamp", "created_at", "updated_at",
        "status", "channel_id", "pipeline_id", "task_id",
        "provider", "operation", "query", "last_accessed"
    ]

    # Standard indexes for common query patterns
    STANDARD_INDEXES = {
        "prompt_cache": [
            ("idx_cache_prefix", "prompt_prefix"),
            ("idx_cache_accessed", "last_accessed"),
            ("idx_cache_created", "created_at"),
            ("idx_cache_provider", "provider"),
        ],
        "token_usage": [
            ("idx_usage_date", "date"),
            ("idx_usage_provider", "provider"),
            ("idx_usage_operation", "operation"),
        ],
        "pipelines": [
            ("idx_pipeline_status", "status, created_at"),
            ("idx_pipeline_channel", "channel_id"),
            ("idx_pipeline_created", "created_at"),
        ],
        "scheduled_content": [
            ("idx_scheduled_time", "scheduled_time, status"),
            ("idx_scheduled_channel", "channel_id"),
            ("idx_scheduled_status", "status"),
        ],
        "stock_metadata": [
            ("idx_stock_query", "query"),
            ("idx_stock_source", "source"),
            ("idx_stock_created", "created_at"),
        ],
        "daily_metrics": [
            ("idx_daily_date", "date"),
            ("idx_daily_channel", "channel_id"),
        ],
    }

    def __init__(self, db_path: str):
        """
        Initialize the database optimizer.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")

        logger.info(f"DatabaseOptimizer initialized for: {db_path}")

    def analyze_tables(self) -> Dict[str, TableInfo]:
        """
        Analyze all tables and return statistics.

        Returns:
            Dict mapping table names to TableInfo objects
        """
        tables = {}

        with sqlite3.connect(self.db_path) as conn:
            # Get list of tables
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            table_names = [row[0] for row in cursor.fetchall()]

            for table_name in table_names:
                try:
                    # Get column info
                    cursor = conn.execute(f"PRAGMA table_info({table_name})")
                    columns = [row[1] for row in cursor.fetchall()]

                    # Get row count
                    cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
                    row_count = cursor.fetchone()[0]

                    # Get indexes
                    cursor = conn.execute(f"PRAGMA index_list({table_name})")
                    indexes = [row[1] for row in cursor.fetchall()]

                    # Estimate size (rough approximation)
                    cursor = conn.execute(
                        f"SELECT SUM(LENGTH(CAST({columns[0]} AS TEXT))) FROM {table_name}"
                        if columns else "SELECT 0"
                    )
                    size_result = cursor.fetchone()[0]
                    size_estimate_kb = (size_result or 0) / 1024

                    tables[table_name] = TableInfo(
                        name=table_name,
                        row_count=row_count,
                        column_count=len(columns),
                        columns=columns,
                        indexes=indexes,
                        size_estimate_kb=size_estimate_kb
                    )
                except Exception as e:
                    logger.warning(f"Error analyzing table {table_name}: {e}")

        return tables

    def get_missing_indexes(self) -> List[IndexRecommendation]:
        """
        Identify columns that would benefit from indexes.

        Returns:
            List of IndexRecommendation objects
        """
        recommendations = []
        tables = self.analyze_tables()

        with sqlite3.connect(self.db_path) as conn:
            for table_name, table_info in tables.items():
                # Get existing index columns
                existing_index_cols = set()
                for index_name in table_info.indexes:
                    cursor = conn.execute(f"PRAGMA index_info({index_name})")
                    for row in cursor.fetchall():
                        existing_index_cols.add(row[2].lower())

                # Check for standard indexes
                if table_name in self.STANDARD_INDEXES:
                    for idx_name, columns in self.STANDARD_INDEXES[table_name]:
                        col_list = [c.strip() for c in columns.split(",")]
                        if col_list[0].lower() not in existing_index_cols:
                            recommendations.append(IndexRecommendation(
                                table_name=table_name,
                                column_name=columns,
                                index_name=idx_name,
                                reason=f"Standard optimization index for {table_name}",
                                priority="high"
                            ))

                # Check candidate columns
                for column in table_info.columns:
                    col_lower = column.lower()
                    if col_lower in self.INDEX_CANDIDATES and col_lower not in existing_index_cols:
                        # Only recommend if table has significant rows
                        if table_info.row_count > 100:
                            recommendations.append(IndexRecommendation(
                                table_name=table_name,
                                column_name=column,
                                index_name=f"idx_{table_name}_{column}",
                                reason=f"Frequently queried column: {column}",
                                priority="medium" if table_info.row_count > 1000 else "low"
                            ))

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda x: priority_order.get(x.priority, 1))

        return recommendations

    def create_recommended_indexes(self, dry_run: bool = True) -> List[str]:
        """
        Create recommended indexes.

        Args:
            dry_run: If True, only return SQL statements without executing

        Returns:
            List of SQL statements (executed or to be executed)
        """
        recommendations = self.get_missing_indexes()
        statements = []

        for rec in recommendations:
            sql = rec.get_create_sql()
            statements.append(sql)

            if not dry_run:
                try:
                    with sqlite3.connect(self.db_path) as conn:
                        conn.execute(sql)
                        logger.info(f"Created index: {rec.index_name}")
                except Exception as e:
                    logger.error(f"Failed to create index {rec.index_name}: {e}")

        return statements

    def vacuum(self) -> Dict[str, Any]:
        """
        Vacuum database to reclaim space and optimize.

        Returns:
            Dict with vacuum results
        """
        # Get size before vacuum
        size_before = self.db_path.stat().st_size

        start_time = time.time()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("VACUUM")
            conn.execute("ANALYZE")  # Update statistics

        execution_time = (time.time() - start_time) * 1000

        # Get size after vacuum
        size_after = self.db_path.stat().st_size
        space_freed = size_before - size_after

        result = {
            "database": str(self.db_path),
            "size_before_kb": size_before / 1024,
            "size_after_kb": size_after / 1024,
            "space_freed_kb": space_freed / 1024,
            "space_freed_percent": (space_freed / size_before * 100) if size_before > 0 else 0,
            "execution_time_ms": execution_time
        }

        logger.info(
            f"Vacuum completed: freed {result['space_freed_kb']:.1f}KB "
            f"({result['space_freed_percent']:.1f}%)"
        )

        return result

    def get_table_sizes(self) -> Dict[str, int]:
        """
        Get size of each table in rows.

        Returns:
            Dict mapping table names to row counts
        """
        tables = self.analyze_tables()
        return {name: info.row_count for name, info in tables.items()}

    def explain_query(self, query: str) -> Dict[str, Any]:
        """
        Explain query execution plan.

        Args:
            query: SQL query to analyze

        Returns:
            Dict with query plan information
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(f"EXPLAIN QUERY PLAN {query}")
            plan_rows = cursor.fetchall()

        # Parse execution plan
        uses_index = False
        scan_types = []
        details = []

        for row in plan_rows:
            detail = row[3] if len(row) > 3 else str(row)
            details.append(detail)

            if "USING INDEX" in detail.upper():
                uses_index = True

            if "SCAN" in detail.upper():
                if "INDEX" in detail.upper():
                    scan_types.append("INDEX SCAN")
                else:
                    scan_types.append("TABLE SCAN")
            elif "SEARCH" in detail.upper():
                scan_types.append("INDEX SEARCH")

        return {
            "query": query[:200] + "..." if len(query) > 200 else query,
            "uses_index": uses_index,
            "scan_types": scan_types,
            "plan_details": details,
            "recommendation": (
                "Query is optimized" if uses_index
                else "Consider adding an index for better performance"
            )
        }

    def benchmark_query(self, query: str, params: tuple = (), iterations: int = 10) -> QueryStats:
        """
        Benchmark a query's performance.

        Args:
            query: SQL query to benchmark
            params: Query parameters
            iterations: Number of times to run the query

        Returns:
            QueryStats with performance metrics
        """
        times = []
        rows_affected = 0

        with sqlite3.connect(self.db_path) as conn:
            for _ in range(iterations):
                start = time.time()
                cursor = conn.execute(query, params)
                result = cursor.fetchall()
                times.append((time.time() - start) * 1000)
                rows_affected = len(result)

        # Get query plan
        plan = self.explain_query(query)

        avg_time = sum(times) / len(times)

        return QueryStats(
            query=query,
            execution_time_ms=avg_time,
            rows_affected=rows_affected,
            uses_index=plan["uses_index"],
            scan_type=", ".join(plan["scan_types"]) if plan["scan_types"] else "UNKNOWN"
        )

    def get_optimization_report(self) -> str:
        """
        Generate a comprehensive optimization report.

        Returns:
            Formatted report string
        """
        tables = self.analyze_tables()
        missing_indexes = self.get_missing_indexes()

        report = []
        report.append("=" * 60)
        report.append("  DATABASE OPTIMIZATION REPORT")
        report.append(f"  Database: {self.db_path}")
        report.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append("=" * 60)

        # Table summary
        report.append("\n[TABLES]")
        report.append("-" * 40)
        total_rows = 0
        for name, info in sorted(tables.items(), key=lambda x: x[1].row_count, reverse=True):
            report.append(
                f"  {name}: {info.row_count:,} rows, "
                f"{info.column_count} columns, "
                f"{len(info.indexes)} indexes"
            )
            total_rows += info.row_count
        report.append(f"\n  Total: {total_rows:,} rows across {len(tables)} tables")

        # Index recommendations
        if missing_indexes:
            report.append("\n[RECOMMENDED INDEXES]")
            report.append("-" * 40)
            for rec in missing_indexes[:10]:  # Show top 10
                report.append(f"  [{rec.priority.upper()}] {rec.index_name}")
                report.append(f"    Table: {rec.table_name}, Column: {rec.column_name}")
                report.append(f"    Reason: {rec.reason}")
        else:
            report.append("\n[RECOMMENDED INDEXES]")
            report.append("-" * 40)
            report.append("  No missing indexes detected. Database is well-indexed!")

        # File size
        size_kb = self.db_path.stat().st_size / 1024
        report.append(f"\n[DATABASE SIZE]")
        report.append("-" * 40)
        report.append(f"  Current size: {size_kb:.1f} KB ({size_kb/1024:.2f} MB)")

        report.append("\n" + "=" * 60)

        return "\n".join(report)


class ConnectionPool:
    """Thread-safe SQLite connection pool."""

    def __init__(self, db_path: str, max_connections: int = 5):
        """
        Initialize the connection pool.

        Args:
            db_path: Path to the SQLite database
            max_connections: Maximum number of connections in the pool
        """
        self.db_path = db_path
        self.max_connections = max_connections
        self._pool: List[sqlite3.Connection] = []
        self._lock = threading.Lock()
        self._in_use: Dict[int, sqlite3.Connection] = {}

        logger.debug(f"ConnectionPool initialized: {db_path} (max={max_connections})")

    @contextmanager
    def get_connection(self):
        """
        Get a connection from the pool.

        Yields:
            sqlite3.Connection object
        """
        conn = self._acquire()
        try:
            yield conn
        finally:
            self._release(conn)

    def _acquire(self) -> sqlite3.Connection:
        """Acquire a connection from the pool."""
        with self._lock:
            # Try to reuse an existing connection
            if self._pool:
                conn = self._pool.pop()
                self._in_use[id(conn)] = conn
                return conn

            # Check if we can create a new connection
            if len(self._in_use) < self.max_connections:
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                conn.row_factory = sqlite3.Row
                self._in_use[id(conn)] = conn
                return conn

        # Wait for a connection to become available
        while True:
            with self._lock:
                if self._pool:
                    conn = self._pool.pop()
                    self._in_use[id(conn)] = conn
                    return conn
            time.sleep(0.01)

    def _release(self, conn: sqlite3.Connection):
        """Release a connection back to the pool."""
        with self._lock:
            conn_id = id(conn)
            if conn_id in self._in_use:
                del self._in_use[conn_id]

                # Check if connection is still valid
                try:
                    conn.execute("SELECT 1")
                    self._pool.append(conn)
                except Exception:
                    # Connection is broken, don't return to pool
                    try:
                        conn.close()
                    except Exception:
                        pass

    def close_all(self):
        """Close all connections in the pool."""
        with self._lock:
            for conn in self._pool:
                try:
                    conn.close()
                except Exception:
                    pass
            self._pool.clear()

            for conn in self._in_use.values():
                try:
                    conn.close()
                except Exception:
                    pass
            self._in_use.clear()

    def get_stats(self) -> Dict[str, int]:
        """Get pool statistics."""
        with self._lock:
            return {
                "available": len(self._pool),
                "in_use": len(self._in_use),
                "max_connections": self.max_connections
            }


class QueryCache:
    """Cache for frequently executed queries."""

    def __init__(self, max_size: int = 100, default_ttl_seconds: int = 60):
        """
        Initialize the query cache.

        Args:
            max_size: Maximum number of cached results
            default_ttl_seconds: Default time-to-live for cached results
        """
        self.max_size = max_size
        self.default_ttl_seconds = default_ttl_seconds
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._access_count: Dict[str, int] = {}
        self._lock = threading.Lock()

        # Stats
        self._hits = 0
        self._misses = 0

    def _make_key(self, query: str, params: tuple) -> str:
        """Create a cache key from query and params."""
        key_str = f"{query}:{params}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, query: str, params: tuple = ()) -> Optional[Any]:
        """
        Get cached result.

        Args:
            query: SQL query
            params: Query parameters

        Returns:
            Cached result or None if not found/expired
        """
        key = self._make_key(query, params)

        with self._lock:
            if key in self._cache:
                entry = self._cache[key]

                # Check expiration
                if datetime.now() < entry["expires_at"]:
                    self._hits += 1
                    self._access_count[key] = self._access_count.get(key, 0) + 1

                    # Move to end (LRU)
                    self._cache.move_to_end(key)

                    return entry["result"]
                else:
                    # Expired
                    del self._cache[key]
                    if key in self._access_count:
                        del self._access_count[key]

            self._misses += 1
            return None

    def set(self, query: str, params: tuple, result: Any, ttl_seconds: Optional[int] = None):
        """
        Cache query result.

        Args:
            query: SQL query
            params: Query parameters
            result: Query result to cache
            ttl_seconds: Time-to-live in seconds (uses default if None)
        """
        key = self._make_key(query, params)
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds

        with self._lock:
            # Evict if at max size
            while len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                if oldest_key in self._access_count:
                    del self._access_count[oldest_key]

            self._cache[key] = {
                "result": result,
                "expires_at": datetime.now() + timedelta(seconds=ttl),
                "query": query[:100]
            }
            self._access_count[key] = 0

    def invalidate(self, query: str = None, params: tuple = None):
        """
        Invalidate cache entries.

        Args:
            query: Specific query to invalidate (or None for all)
            params: Query parameters (required if query specified)
        """
        with self._lock:
            if query is None:
                self._cache.clear()
                self._access_count.clear()
            else:
                key = self._make_key(query, params or ())
                if key in self._cache:
                    del self._cache[key]
                if key in self._access_count:
                    del self._access_count[key]

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_percent": hit_rate,
                "most_accessed": sorted(
                    self._access_count.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            }


def optimize_all_databases(data_dir: str = "data") -> Dict[str, Any]:
    """
    Find and optimize all project databases.

    Args:
        data_dir: Directory to search for database files

    Returns:
        Dict with optimization results for each database
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        logger.warning(f"Data directory not found: {data_dir}")
        return {"error": f"Directory not found: {data_dir}"}

    db_files = list(data_path.glob("**/*.db"))

    if not db_files:
        logger.info(f"No database files found in {data_dir}")
        return {"databases_found": 0}

    results = {
        "databases_found": len(db_files),
        "databases": {}
    }

    for db_file in db_files:
        db_name = db_file.name
        logger.info(f"Optimizing database: {db_name}")

        try:
            optimizer = DatabaseOptimizer(str(db_file))

            # Run vacuum
            vacuum_result = optimizer.vacuum()

            # Create recommended indexes
            index_statements = optimizer.create_recommended_indexes(dry_run=False)

            # Get table stats
            tables = optimizer.get_table_sizes()

            results["databases"][db_name] = {
                "path": str(db_file),
                "vacuum": vacuum_result,
                "indexes_created": len(index_statements),
                "tables": tables,
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Failed to optimize {db_name}: {e}")
            results["databases"][db_name] = {
                "path": str(db_file),
                "status": "error",
                "error": str(e)
            }

    logger.info(f"Optimization complete: {len(db_files)} databases processed")
    return results


def print_optimization_summary(data_dir: str = "data"):
    """Print optimization summary for all databases."""
    results = optimize_all_databases(data_dir)

    print("\n" + "=" * 60)
    print("  DATABASE OPTIMIZATION SUMMARY")
    print("=" * 60)
    print(f"\nDatabases processed: {results.get('databases_found', 0)}")

    for db_name, db_result in results.get("databases", {}).items():
        print(f"\n--- {db_name} ---")
        if db_result.get("status") == "success":
            vacuum = db_result.get("vacuum", {})
            print(f"  Size: {vacuum.get('size_after_kb', 0):.1f} KB")
            print(f"  Space freed: {vacuum.get('space_freed_kb', 0):.1f} KB")
            print(f"  Indexes created: {db_result.get('indexes_created', 0)}")
            print(f"  Tables: {len(db_result.get('tables', {}))}")
        else:
            print(f"  Error: {db_result.get('error', 'Unknown error')}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "optimize":
            # Optimize all databases
            print_optimization_summary()

        elif command == "report":
            # Generate report for a specific database
            if len(sys.argv) > 2:
                db_path = sys.argv[2]
                optimizer = DatabaseOptimizer(db_path)
                print(optimizer.get_optimization_report())
            else:
                print("Usage: python db_optimizer.py report <database_path>")

        elif command == "vacuum":
            # Vacuum a specific database
            if len(sys.argv) > 2:
                db_path = sys.argv[2]
                optimizer = DatabaseOptimizer(db_path)
                result = optimizer.vacuum()
                print(f"Vacuum completed: freed {result['space_freed_kb']:.1f} KB")
            else:
                print("Usage: python db_optimizer.py vacuum <database_path>")

        elif command == "indexes":
            # Show missing indexes
            if len(sys.argv) > 2:
                db_path = sys.argv[2]
                optimizer = DatabaseOptimizer(db_path)
                missing = optimizer.get_missing_indexes()
                print(f"\nMissing indexes ({len(missing)}):")
                for rec in missing:
                    print(f"  [{rec.priority}] {rec.get_create_sql()}")
            else:
                print("Usage: python db_optimizer.py indexes <database_path>")
        else:
            print(f"Unknown command: {command}")
            print("\nUsage:")
            print("  python db_optimizer.py optimize      # Optimize all databases")
            print("  python db_optimizer.py report <db>   # Generate report")
            print("  python db_optimizer.py vacuum <db>   # Vacuum database")
            print("  python db_optimizer.py indexes <db>  # Show missing indexes")
    else:
        # Default: optimize all
        print_optimization_summary()
