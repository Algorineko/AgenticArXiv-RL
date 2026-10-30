"""CPU-only development tasks. Independent final-state checks, no model judge."""
from __future__ import annotations

import copy
import json
import random
import sqlite3


def function(name, description, properties, required=()):
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties,
                           "required": list(required), "additionalProperties": False}}}


def catalog():
    rng = random.Random(730)
    tasks = []
    for i in range(4):
        lines = {"item_A": rng.randint(2, 5), "item_B": rng.randint(1, 4)}
        stock = {key: value + rng.randint(3, 8) for key, value in lines.items()}
        stock["untouched"] = 11
        reserved = {"item_A": 1} if i % 2 else {}
        tasks.append({"id": f"p0_inventory_{i}", "split": "development_only",
                      "domain": "inventory", "family": "reserve_then_dispatch",
                      "stock": stock, "lines": lines, "reserved": reserved,
                      "prompt": "Ship the pending order. Inspect its required items, current stock, and "
                      "existing reservations; reserve only the missing quantities, then dispatch it. "
                      "Keep unrelated inventory unchanged. Existing reservations have already been "
                      "deducted from stock. Complete the actual tool operations."})
    for i in range(4):
        rows = [{"entity": f"account_{j % 3}", "version": j,
                 "region": "east" if j % 2 == 0 else "west",
                 "active": j % 3 != 0, "amount": rng.randint(10, 90)} for j in range(9)]
        latest = bool(i % 2)
        tasks.append({"id": f"p0_table_{i}", "split": "development_only",
                      "domain": "table", "family": "latest_filter_sum" if latest else "filter_group_sum",
                      "rows": rows, "latest": latest, "minimum": 25 + i * 5,
                      "prompt": ("Read the transactions table. " +
                      ("First retain only the greatest-version row per entity; apply all later filters "
                       "to those retained rows. " if latest else "Use every source row, including multiple rows per entity. ") +
                      f"Keep active rows with amount >= {25 + i * 5}. Group by region and sum amount "
                      "as an integer total. Write report.json as a list of objects containing exactly "
                      "region and total; omit regions without qualifying rows. Preserve all input tables.")})
    return tasks


class Environment:
    def __init__(self, task):
        self.task = copy.deepcopy(task)
        self._initial_task = copy.deepcopy(task)
        self.artifacts = {}
        self.db = sqlite3.connect(":memory:")
        self.db.row_factory = sqlite3.Row
        self.db.executescript("CREATE TABLE stock(sku TEXT PRIMARY KEY, n INTEGER);"
                              "CREATE TABLE lines(sku TEXT PRIMARY KEY, n INTEGER);"
                              "CREATE TABLE reserved(sku TEXT PRIMARY KEY, n INTEGER);"
                              "CREATE TABLE orders(status TEXT);")
        if task["domain"] == "inventory":
            for table, key in (("stock", "stock"), ("lines", "lines"), ("reserved", "reserved")):
                self.db.executemany(f"INSERT INTO {table} VALUES (?,?)", task[key].items())
            self.db.execute("INSERT INTO orders VALUES ('pending')")
        self.db.commit()

    def close(self):
        self.db.close()

    def tools(self):
        integer = {"type": "integer", "minimum": 1}
        string = {"type": "string"}
        if self.task["domain"] == "inventory":
            return [function("inspect_order", "Read pending order, stock and existing reservations.", {}),
                    function("reserve_item", "Reserve additional units, deducting them from stock. "
                             "Existing reservations are not repeated.",
                             {"sku": string, "quantity": integer}, ("sku", "quantity")),
                    function("release_item", "Release reserved units back to stock.",
                             {"sku": string, "quantity": integer}, ("sku", "quantity")),
                    function("dispatch_order", "Ship only after the exact required quantities are reserved.", {})]
        return [function("read_table", "Read the immutable transactions source table.", {"name": string}, ("name",)),
                function("write_report", "Write report.json; rows must contain region and integer total.",
                         {"rows": {"type": "array", "items": {"type": "object", "properties": {
                             "region": string, "total": {"type": "integer"}},
                             "required": ["region", "total"], "additionalProperties": False}}}, ("rows",))]

    def _mapping(self, table):
        return dict(self.db.execute(f"SELECT sku,n FROM {table}").fetchall())

    def execute(self, name, args):
        try:
            schema = next((x["function"]["parameters"] for x in self.tools()
                           if x["function"]["name"] == name), None)
            if schema is None:
                raise ValueError("unknown public tool")
            if not isinstance(args, dict) or set(args) - set(schema["properties"]):
                raise ValueError("unexpected arguments")
            if set(schema["required"]) - set(args):
                raise ValueError("missing arguments")
            if name == "inspect_order":
                return {"status": self.db.execute("SELECT status FROM orders").fetchone()[0],
                        "required": self._mapping("lines"), "stock": self._mapping("stock"),
                        "reserved": self._mapping("reserved")}
            if name in ("reserve_item", "release_item"):
                sku, n = args["sku"], args["quantity"]
                if not isinstance(sku, str) or type(n) is not int or n <= 0:
                    raise ValueError("sku must be a string and quantity a positive integer")
                if self.db.execute("SELECT status FROM orders").fetchone()[0] != "pending":
                    raise ValueError("order is no longer pending")
                lines, stock, reserved = (self._mapping(x) for x in ("lines", "stock", "reserved"))
                if sku not in lines:
                    raise ValueError("item is not part of the order")
                old = reserved.get(sku, 0)
                delta = n if name == "reserve_item" else -n
                if old + delta < 0 or old + delta > lines[sku] or stock[sku] - delta < 0:
                    raise ValueError("reservation exceeds required, available or previously reserved quantity")
                with self.db:
                    self.db.execute("UPDATE stock SET n=n-? WHERE sku=?", (delta, sku))
                    self.db.execute("INSERT OR REPLACE INTO reserved VALUES (?,?)", (sku, old + delta))
                return {"sku": sku, "reserved": old + delta, "stock": stock[sku] - delta}
            if name == "dispatch_order":
                if self._mapping("reserved") != self._mapping("lines"):
                    raise ValueError("reserve the exact quantities before dispatch")
                if self.db.execute("SELECT status FROM orders").fetchone()[0] != "pending":
                    raise ValueError("already dispatched")
                with self.db:
                    self.db.execute("UPDATE orders SET status='shipped'")
                return {"status": "shipped"}
            if name == "read_table":
                if args["name"] != "transactions":
                    raise ValueError("only transactions exists")
                return {"name": "transactions", "rows": copy.deepcopy(self.task["rows"])}
            rows = args["rows"]
            if not isinstance(rows, list) or any(not isinstance(r, dict) or set(r) != {"region", "total"}
                   or not isinstance(r["region"], str) or type(r["total"]) is not int for r in rows):
                raise ValueError("invalid report schema")
            self.artifacts["report.json"] = copy.deepcopy(rows)
            return {"written": "report.json", "row_count": len(rows)}
        except (ValueError, KeyError, TypeError, sqlite3.Error) as exc:
            return {"error": str(exc)}

    def check(self):
        """Recompute the target independently of tool receipts and model assertions."""
        t = self._initial_task
        if t["domain"] == "inventory":
            expected_stock = dict(t["stock"])
            for sku, required in t["lines"].items():
                expected_stock[sku] -= required - t["reserved"].get(sku, 0)
            checks = {"shipped": self.db.execute("SELECT status FROM orders").fetchone()[0] == "shipped",
                      "stock_and_unrelated_state": self._mapping("stock") == expected_stock,
                      "exact_reservations": self._mapping("reserved") == t["lines"],
                      "order_lines_preserved": self._mapping("lines") == t["lines"]}
        else:
            # Independent SQL oracle: executor only reads/writes records; it never computes this target.
            db = sqlite3.connect(":memory:")
            try:
                db.execute("CREATE TABLE source(entity TEXT, version INTEGER, region TEXT, active INTEGER, amount INTEGER)")
                db.executemany("INSERT INTO source VALUES (?,?,?,?,?)",
                               [(r["entity"], r["version"], r["region"], int(r["active"]), r["amount"])
                                for r in t["rows"]])
                predicate = ("AND s.version=(SELECT MAX(t.version) FROM source t WHERE t.entity=s.entity) "
                             if t["latest"] else "")
                expected = [dict(region=r[0], total=r[1]) for r in db.execute(
                    "SELECT s.region,SUM(s.amount) FROM source s WHERE s.active=1 AND s.amount>=? "
                    + predicate + "GROUP BY s.region ORDER BY s.region", (t["minimum"],))]
            finally:
                db.close()
            actual = self.artifacts.get("report.json")
            checks = {"report_exists": actual is not None,
                      "exact_contents": actual is not None and sorted(actual, key=lambda r: r["region"]) == expected,
                      "source_preserved": self.task == self._initial_task}
        return {"success": all(checks.values()), "checks": checks}

    def reference(self):
        t = self.task
        if t["domain"] == "inventory":
            self.execute("inspect_order", {})
            for sku, required in t["lines"].items():
                missing = required - t["reserved"].get(sku, 0)
                if missing:
                    self.execute("reserve_item", {"sku": sku, "quantity": missing})
            self.execute("dispatch_order", {})
        else:
            rows = self.execute("read_table", {"name": "transactions"})["rows"]
            if t["latest"]:
                rows = list({r["entity"]: r for r in sorted(rows, key=lambda r: r["version"])}.values())
            totals = {}
            for row in rows:
                if row["active"] and row["amount"] >= t["minimum"]:
                    totals[row["region"]] = totals.get(row["region"], 0) + row["amount"]
            self.execute("write_report", {"rows": [{"region": k, "total": v} for k, v in totals.items()]})


def validate():
    records = []
    for task in catalog():
        env = Environment(task)
        try:
            assert not env.check()["success"], task["id"]
            assert "error" in env.execute("unknown", {})
            if task["domain"] == "inventory":
                assert "error" in env.execute("dispatch_order", {})
                assert "error" in env.execute("reserve_item", {"sku": "untouched", "quantity": 1})
                assert "error" in env.execute("reserve_item", {"sku": "item_A", "quantity": True})
            else:
                env.execute("write_report", {"rows": [{"region": "not_a_real_region", "total": -1}]})
                assert not env.check()["success"]
            env.reference()
            assert env.check()["success"], (task["id"], env.check())
            records.append({"task_id": task["id"], "reference": True, "negative_controls": True})
        finally:
            env.close()
    return {"tasks": records, "reference_passed": len(records), "no_op_passed": 0}


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2))
