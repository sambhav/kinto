from unittest import mock

import pytest

from kinto.core import DEFAULT_SETTINGS
from kinto.core.storage import (
    MISSING,
    Filter,
    Sort,
    StorageBase,
    exceptions,
    generators,
    memory,
    postgresql,
)
from kinto.core.storage.testing import StorageTest
from kinto.core.storage.utils import paginated
from kinto.core.testing import skip_if_no_postgresql, unittest
from kinto.core.utils import COMPARISON
from kinto.core.utils import sqlalchemy as sa


class GeneratorTest(unittest.TestCase):
    def test_generic_has_mandatory_override(self):
        self.assertRaises(NotImplementedError, generators.Generator)

    def test_id_generator_must_respect_storage_backends(self):
        class Dumb(generators.Generator):
            def __call__(self):
                return "*" * 80

        self.assertRaises(ValueError, Dumb)

    def test_default_generator_allow_underscores_dash_alphabet(self):
        class Dumb(generators.Generator):
            def __call__(self):
                return "1234"

        generator = Dumb()
        self.assertTrue(generator.match("1_2_3-abc"))
        self.assertTrue(generator.match("abc_123"))
        self.assertFalse(generator.match("-1_2_3-abc"))
        self.assertFalse(generator.match("_1_2_3-abc"))

    def test_uuid_generator_pattern_allows_uuid_only(self):
        invalid_uuid = "XXX-00000000-0000-5000-a000-000000000000"
        generator = generators.UUID4()
        self.assertFalse(generator.match(invalid_uuid))

    def test_uuid_generator_pattern_is_not_restricted_to_uuid4(self):
        generator = generators.UUID4()
        valid_uuid = "fd800e8d-e8e9-3cac-f502-816cbed9bb6c"
        self.assertTrue(generator.match(valid_uuid))
        invalid_uuid4 = "00000000-0000-5000-a000-000000000000"
        self.assertTrue(generator.match(invalid_uuid4))
        invalid_uuid4 = "00000000-0000-4000-e000-000000000000"
        self.assertTrue(generator.match(invalid_uuid4))


class StorageBaseTest(unittest.TestCase):
    def setUp(self):
        self.storage = StorageBase()

    def test_mandatory_overrides(self):
        calls = [
            (self.storage.initialize_schema,),
            (self.storage.flush,),
            (self.storage.resource_timestamp, "", ""),
            (self.storage.all_resources_timestamps, ""),
            (self.storage.create, "", "", {}),
            (self.storage.get, "", "", ""),
            (self.storage.update, "", "", "", {}),
            (self.storage.delete, "", "", ""),
            (self.storage.delete_all, "", ""),
            (self.storage.purge_deleted, "", ""),
            (self.storage.list_all, "", ""),
            (self.storage.count_all, "", ""),
            (self.storage.trim_objects, "", "", [], 0),
        ]
        for call in calls:
            self.assertRaises(NotImplementedError, *call)

    def test_backend_error_message_provides_given_message_if_defined(self):
        error = exceptions.BackendError(message="Connection Error")
        self.assertEqual(str(error), "Connection Error")

    def test_backenderror_message_default_to_original_exception_message(self):
        error = exceptions.BackendError(ValueError("Pool Error"))
        self.assertEqual(str(error), "ValueError: Pool Error")


class MemoryBasedStorageTest(unittest.TestCase):
    def test_backend_raise_not_implemented_error(self):
        storage = memory.MemoryBasedStorage()
        with pytest.raises(NotImplementedError):
            storage.bump_and_store_timestamp("object", "/school/foo/students/bar")


class MemoryStorageTest(StorageTest, unittest.TestCase):
    backend = memory

    def setUp(self):
        super().setUp()
        self.client_error_patcher = mock.patch.object(
            self.storage,
            "bump_and_store_timestamp",
            side_effect=exceptions.BackendError("Segmentation fault."),
        )

    def test_backend_error_provides_original_exception(self):
        pass

    def test_raises_backend_error_if_error_occurs_on_client(self):
        pass

    def test_backend_error_is_raised_anywhere(self):
        pass

    def test_backenderror_message_default_to_original_exception_message(self):
        pass

    def test_ping_logs_error_if_unavailable(self):
        pass

    def test_create_bytes_raises(self):
        data = {"steak": "haché".encode(encoding="utf-8")}
        self.assertIsInstance(data["steak"], bytes)
        self.assertRaises(TypeError, self.create_object, data)

    def test_update_bytes_raises(self):
        obj = self.create_object()

        new_object = {"steak": "haché".encode(encoding="utf-8")}
        self.assertIsInstance(new_object["steak"], bytes)

        self.assertRaises(
            TypeError, self.storage.update, object_id=obj["id"], obj=new_object, **self.storage_kw
        )


@skip_if_no_postgresql
class PostgreSQLStorageTest(StorageTest, unittest.TestCase):
    backend = postgresql
    settings = {
        "storage_max_fetch_size": 10000,
        "storage_backend": "kinto.core.storage.postgresql",
        "storage_poolclass": "sqlalchemy.pool.StaticPool",
        "storage_url": "postgresql://postgres:postgres@localhost:5432/testdb",
    }

    def setUp(self):
        super().setUp()
        self.client_error_patcher = mock.patch.object(
            self.storage.client, "session_factory", side_effect=sa.exc.SQLAlchemyError
        )

    def test_number_of_fetched_objects_can_be_limited_in_settings(self):
        for i in range(4):
            self.create_object({"phone": "tel-{}".format(i)})

        results = self.storage.list_all(**self.storage_kw)
        self.assertEqual(len(results), 4)

        settings = {**self.settings, "storage_max_fetch_size": 2}
        config = self._get_config(settings=settings)
        limited = self.backend.load_from_config(config)

        results = limited.list_all(**self.storage_kw)
        self.assertEqual(len(results), 2)
        count = limited.count_all(**self.storage_kw)
        self.assertEqual(count, 4)

    def test_number_of_fetched_objects_is_per_page(self):
        for i in range(10):
            self.create_object({"number": i})

        settings = {**self.settings, "storage_max_fetch_size": 2}
        config = self._get_config(settings=settings)
        backend = self.backend.load_from_config(config)

        results = backend.list_all(
            pagination_rules=[[Filter("number", 1, COMPARISON.GT)]], **self.storage_kw
        )
        self.assertEqual(len(results), 2)
        count = backend.count_all(**self.storage_kw)
        self.assertEqual(count, 10)

    def test_connection_is_rolledback_if_error_occurs(self):
        with self.storage.client.connect() as conn:
            query = "DELETE FROM objects WHERE resource_name = 'genre';"
            conn.execute(sa.text(query))

        try:
            with self.storage.client.connect() as conn:
                query = """
                INSERT INTO objects VALUES ('rock-and-roll', 'music', 'genre', NOW(), '{}', FALSE);
                """
                conn.execute(sa.text(query))
                conn.commit()

                query = """
                INSERT INTO objects VALUES ('jazz', 'music', 'genre', NOW(), '{}', FALSE);
                """
                conn.execute(sa.text(query))

                raise sa.exc.TimeoutError()
        except exceptions.BackendError:
            pass

        with self.storage.client.connect() as conn:
            query = "SELECT COUNT(*) FROM objects WHERE resource_name = 'genre';"
            result = conn.execute(sa.text(query))
            self.assertEqual(result.fetchone()[0], 1)

    def test_pool_object_is_shared_among_backend_instances(self):
        config = self._get_config()
        storage1 = self.backend.load_from_config(config)
        storage2 = self.backend.load_from_config(config)
        self.assertEqual(id(storage1.client), id(storage2.client))

    def test_warns_if_configured_pool_size_differs_for_same_backend_type(self):
        self.backend.load_from_config(self._get_config())
        settings = {**self.settings, "storage_pool_size": 1}
        msg = "Reuse existing PostgreSQL connection. Parameters storage_* will be ignored."
        with mock.patch("kinto.core.storage.postgresql.client.warnings.warn") as mocked:
            self.backend.load_from_config(self._get_config(settings=settings))
            mocked.assert_any_call(msg)

    def test_list_all_raises_if_missing_on_strange_query(self):
        with self.assertRaises(ValueError):
            self.storage.list_all(
                "some-resource", "some-parent", filters=[Filter("author", MISSING, COMPARISON.HAS)]
            )

    def test_integrity_error_rollsback_transaction(self):
        client = postgresql.create_from_config(
            self._get_config(), prefix="storage_", with_transaction=False
        )
        with self.assertRaises(exceptions.IntegrityError):
            with client.connect() as conn:
                # Make some change in a table.
                conn.execute(
                    sa.text(
                        """
                INSERT INTO objects
                VALUES ('rock-and-roll', 'music', 'genre', NOW(), '{}', FALSE);
                """
                    )
                )
                # Go into a failing integrity constraint.
                query = "INSERT INTO timestamps VALUES ('a', 'b', NOW());"
                conn.execute(sa.text(query))
                conn.execute(sa.text(query))
                conn.commit()
                conn.close()

        # Check that change in the above table was rolledback.
        with client.connect() as conn:
            result = conn.execute(
                sa.text(
                    """
            SELECT FROM objects
             WHERE parent_id = 'music'
               AND resource_name = 'genre';
            """
                )
            )
        self.assertEqual(result.rowcount, 0)

    def test_conflicts_handled_correctly(self):
        config = self._get_config()
        storage = self.backend.load_from_config(config)
        storage.create(resource_name="genre", parent_id="music", obj={"id": "rock-and-roll"})

        def object_not_found(*args, **kwargs):
            raise exceptions.ObjectNotFoundError()

        with mock.patch.object(storage, "get", side_effect=object_not_found):
            with self.assertRaises(exceptions.UnicityError):
                storage.create(
                    resource_name="genre", parent_id="music", obj={"id": "rock-and-roll"}
                )

    def test_supports_null_pool(self):
        settings = {
            **DEFAULT_SETTINGS,
            **self.settings,
            "storage_poolclass": "sqlalchemy.pool.NullPool",
        }
        config = self._get_config(settings=settings)
        self.backend.load_from_config(config)  # does not raise

    def test_pagination_with_modified_field_filter(self):
        """Functional test: pagination with last_modified filters works correctly."""
        # Create objects with different timestamps
        objects = []
        for i in range(10):
            obj = self.create_object({"number": i})
            objects.append(obj)

        # Get objects with pagination using last_modified filter
        # This simulates what happens during actual pagination
        before = objects[5]["last_modified"]
        filters = [Filter("last_modified", before, COMPARISON.LT)]

        results = self.storage.list_all(filters=filters, **self.storage_kw)

        # Should get the first 5 objects (created before object #5)
        self.assertEqual(len(results), 5)
        for obj in results:
            self.assertLess(obj["last_modified"], before)


@skip_if_no_postgresql
class PostgreSQLLastModifiedFilterTest(unittest.TestCase):
    """Tests for the from_epoch() optimization on last_modified queries.

    Verifies that moving as_epoch() from the column side to from_epoch()
    on the value side produces correct results for every comparison operator.
    """

    backend = postgresql
    settings = {
        "storage_max_fetch_size": 10000,
        "storage_backend": "kinto.core.storage.postgresql",
        "storage_poolclass": "sqlalchemy.pool.StaticPool",
        "storage_url": "postgresql://postgres:postgres@localhost:5432/testdb",
    }

    def setUp(self):
        from pyramid import testing as pyramid_testing

        config = pyramid_testing.setUp()
        config.add_settings(self.settings)
        self.storage = self.backend.load_from_config(config)
        self.storage.initialize_schema()
        self.storage_kw = {"resource_name": "test", "parent_id": "parent1"}

        # Create objects with known timestamps by using the storage API
        import time

        self.objects = []
        for i in range(5):
            obj = self.storage.create(obj={"index": i}, **self.storage_kw)
            self.objects.append(obj)
            time.sleep(0.01)

    def tearDown(self):
        self.storage.flush()

    # --- Scalar comparison operators on last_modified ---

    def test_filter_gt_last_modified(self):
        """GT filter with from_epoch() returns only later objects."""
        ts = self.objects[2]["last_modified"]
        filters = [Filter("last_modified", ts, COMPARISON.GT)]
        results = self.storage.list_all(filters=filters, **self.storage_kw)
        result_ids = {r["id"] for r in results}
        for obj in self.objects[:3]:
            self.assertNotIn(obj["id"], result_ids)
        for obj in self.objects[3:]:
            self.assertIn(obj["id"], result_ids)

    def test_filter_lt_last_modified(self):
        """LT filter with from_epoch() returns only earlier objects."""
        ts = self.objects[2]["last_modified"]
        filters = [Filter("last_modified", ts, COMPARISON.LT)]
        results = self.storage.list_all(filters=filters, **self.storage_kw)
        result_ids = {r["id"] for r in results}
        for obj in self.objects[:2]:
            self.assertIn(obj["id"], result_ids)
        for obj in self.objects[2:]:
            self.assertNotIn(obj["id"], result_ids)

    def test_filter_min_last_modified(self):
        """MIN (>=) filter with from_epoch() includes the boundary."""
        ts = self.objects[2]["last_modified"]
        filters = [Filter("last_modified", ts, COMPARISON.MIN)]
        results = self.storage.list_all(filters=filters, **self.storage_kw)
        result_ids = {r["id"] for r in results}
        self.assertIn(self.objects[2]["id"], result_ids)
        for obj in self.objects[3:]:
            self.assertIn(obj["id"], result_ids)
        for obj in self.objects[:2]:
            self.assertNotIn(obj["id"], result_ids)

    def test_filter_max_last_modified(self):
        """MAX (<=) filter with from_epoch() includes the boundary."""
        ts = self.objects[2]["last_modified"]
        filters = [Filter("last_modified", ts, COMPARISON.MAX)]
        results = self.storage.list_all(filters=filters, **self.storage_kw)
        result_ids = {r["id"] for r in results}
        for obj in self.objects[:3]:
            self.assertIn(obj["id"], result_ids)
        for obj in self.objects[3:]:
            self.assertNotIn(obj["id"], result_ids)

    def test_filter_eq_last_modified(self):
        """EQ filter with from_epoch() matches exactly one object."""
        ts = self.objects[2]["last_modified"]
        filters = [Filter("last_modified", ts, COMPARISON.EQ)]
        results = self.storage.list_all(filters=filters, **self.storage_kw)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], self.objects[2]["id"])

    def test_filter_not_last_modified(self):
        """NOT filter with from_epoch() excludes exactly one object."""
        ts = self.objects[2]["last_modified"]
        filters = [Filter("last_modified", ts, COMPARISON.NOT)]
        results = self.storage.list_all(filters=filters, **self.storage_kw)
        result_ids = {r["id"] for r in results}
        self.assertEqual(len(results), 4)
        self.assertNotIn(self.objects[2]["id"], result_ids)

    # --- IN / EXCLUDE operators (tuple expansion) ---

    def test_filter_in_last_modified(self):
        """IN filter expands each epoch value with from_epoch()."""
        timestamps = [self.objects[1]["last_modified"], self.objects[3]["last_modified"]]
        filters = [Filter("last_modified", timestamps, COMPARISON.IN)]
        results = self.storage.list_all(filters=filters, **self.storage_kw)
        result_ids = {r["id"] for r in results}
        self.assertEqual(len(results), 2)
        self.assertIn(self.objects[1]["id"], result_ids)
        self.assertIn(self.objects[3]["id"], result_ids)

    def test_filter_in_last_modified_single_value(self):
        """IN filter with a single value works correctly."""
        timestamps = [self.objects[0]["last_modified"]]
        filters = [Filter("last_modified", timestamps, COMPARISON.IN)]
        results = self.storage.list_all(filters=filters, **self.storage_kw)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], self.objects[0]["id"])

    def test_filter_in_last_modified_empty_list(self):
        """IN filter with empty list returns no results."""
        filters = [Filter("last_modified", [], COMPARISON.IN)]
        results = self.storage.list_all(filters=filters, **self.storage_kw)
        self.assertEqual(len(results), 0)

    def test_filter_in_last_modified_all_values(self):
        """IN filter with all timestamps returns all objects."""
        timestamps = [obj["last_modified"] for obj in self.objects]
        filters = [Filter("last_modified", timestamps, COMPARISON.IN)]
        results = self.storage.list_all(filters=filters, **self.storage_kw)
        self.assertEqual(len(results), 5)

    def test_filter_in_last_modified_nonexistent_value(self):
        """IN filter with a value that doesn't match returns nothing."""
        filters = [Filter("last_modified", [9999999999999], COMPARISON.IN)]
        results = self.storage.list_all(filters=filters, **self.storage_kw)
        self.assertEqual(len(results), 0)

    def test_filter_exclude_last_modified(self):
        """EXCLUDE filter with from_epoch() tuple expansion."""
        timestamps = [self.objects[1]["last_modified"], self.objects[3]["last_modified"]]
        filters = [Filter("last_modified", timestamps, COMPARISON.EXCLUDE)]
        results = self.storage.list_all(filters=filters, **self.storage_kw)
        result_ids = {r["id"] for r in results}
        self.assertEqual(len(results), 3)
        self.assertNotIn(self.objects[1]["id"], result_ids)
        self.assertNotIn(self.objects[3]["id"], result_ids)

    def test_filter_exclude_last_modified_empty_list(self):
        """EXCLUDE with empty list.

        Empty tuples are rewritten to (None,) to avoid SQL syntax errors.
        NOT IN (NULL) evaluates to NULL (falsy) for every row, so no
        results are returned.  This is pre-existing behaviour.
        """
        filters = [Filter("last_modified", [], COMPARISON.EXCLUDE)]
        results = self.storage.list_all(filters=filters, **self.storage_kw)
        self.assertEqual(len(results), 0)

    # --- Combined filters ---

    def test_combined_range_filters_on_last_modified(self):
        """MIN and MAX filters together select a range (inclusive)."""
        ts_low = self.objects[1]["last_modified"]
        ts_high = self.objects[3]["last_modified"]
        filters = [
            Filter("last_modified", ts_low, COMPARISON.MIN),
            Filter("last_modified", ts_high, COMPARISON.MAX),
        ]
        results = self.storage.list_all(filters=filters, **self.storage_kw)
        result_ids = {r["id"] for r in results}
        self.assertEqual(len(results), 3)
        for obj in self.objects[1:4]:
            self.assertIn(obj["id"], result_ids)

    def test_filter_last_modified_with_count(self):
        """count_all returns same count as len(list_all) for GT filter."""
        ts = self.objects[1]["last_modified"]
        filters = [Filter("last_modified", ts, COMPARISON.GT)]
        results = self.storage.list_all(filters=filters, **self.storage_kw)
        count = self.storage.count_all(filters=filters, **self.storage_kw)
        self.assertEqual(count, len(results))
        self.assertEqual(count, 3)

    def test_filter_last_modified_does_not_affect_other_fields(self):
        """Filters on non-modified fields are unaffected by the changes."""
        filters = [Filter("index", 2, COMPARISON.EQ)]
        results = self.storage.list_all(filters=filters, **self.storage_kw)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["index"], 2)

    # --- Sorting on last_modified ---

    def test_sort_by_last_modified_ascending(self):
        """Sorting by last_modified ASC returns objects in creation order."""
        sorting = [Sort("last_modified", 1)]
        results = self.storage.list_all(sorting=sorting, **self.storage_kw)
        timestamps = [r["last_modified"] for r in results]
        self.assertEqual(timestamps, sorted(timestamps))

    def test_sort_by_last_modified_descending(self):
        """Sorting by last_modified DESC returns objects in reverse order."""
        sorting = [Sort("last_modified", -1)]
        results = self.storage.list_all(sorting=sorting, **self.storage_kw)
        timestamps = [r["last_modified"] for r in results]
        self.assertEqual(timestamps, sorted(timestamps, reverse=True))

    # --- purge_deleted with from_epoch() ---

    def test_purge_deleted_with_before_timestamp(self):
        """purge_deleted uses from_epoch(:before) correctly."""
        import time

        # Use a dedicated parent_id to avoid interference from setUp objects.
        purge_kw = {"resource_name": "test", "parent_id": "purge_test1"}

        # Create and delete an object (produces a tombstone).
        obj1 = self.storage.create(obj={"val": 1}, **purge_kw)
        tombstone1 = self.storage.delete(object_id=obj1["id"], **purge_kw)
        time.sleep(0.01)

        # Create and delete another (later tombstone).
        obj2 = self.storage.create(obj={"val": 2}, **purge_kw)
        tombstone2 = self.storage.delete(object_id=obj2["id"], **purge_kw)
        time.sleep(0.01)

        # Use a boundary between the two tombstones (strictly after the first).
        boundary_ts = tombstone1["last_modified"] + 1

        # Purge only rows before boundary — should remove the first tombstone.
        num_purged = self.storage.purge_deleted(before=boundary_ts, **purge_kw)
        self.assertEqual(num_purged, 1)

        # The later tombstone should still exist.
        all_with_deleted = self.storage.list_all(
            include_deleted=True,
            filters=[Filter("last_modified", boundary_ts, COMPARISON.MIN)],
            **purge_kw,
        )
        tombstones = [r for r in all_with_deleted if r.get("deleted")]
        self.assertGreaterEqual(len(tombstones), 1)

    def test_purge_deleted_at_exact_boundary(self):
        """purge_deleted with before= exactly at a tombstone's timestamp.

        The condition is last_modified < from_epoch(:before), so the
        tombstone at the exact boundary should NOT be purged.
        """
        import time

        purge_kw = {"resource_name": "test", "parent_id": "purge_test2"}

        obj = self.storage.create(obj={"val": 1}, **purge_kw)
        deleted = self.storage.delete(object_id=obj["id"], **purge_kw)
        tombstone_ts = deleted["last_modified"]
        time.sleep(0.01)

        num_purged = self.storage.purge_deleted(before=tombstone_ts, **purge_kw)
        # The tombstone at exactly the boundary should not be purged (strict <).
        self.assertEqual(num_purged, 0)

    # --- resource_timestamp ORDER BY ---

    def test_resource_timestamp_returns_latest(self):
        """resource_timestamp returns the highest timestamp after ORDER BY change."""
        ts = self.storage.resource_timestamp(**self.storage_kw)
        latest_obj_ts = max(obj["last_modified"] for obj in self.objects)
        self.assertGreaterEqual(ts, latest_obj_ts)

    def test_resource_timestamp_after_delete_is_monotonic(self):
        """resource_timestamp stays monotonically increasing after deletions."""
        import time

        ts_before = self.storage.resource_timestamp(**self.storage_kw)
        last_obj = self.objects[-1]
        time.sleep(0.01)
        self.storage.delete(object_id=last_obj["id"], **self.storage_kw)
        ts_after = self.storage.resource_timestamp(**self.storage_kw)
        self.assertGreater(ts_after, ts_before)

    # --- bump_timestamp trigger ---

    def test_bump_timestamp_creates_monotonic_timestamps(self):
        """Records created in sequence get strictly increasing timestamps."""
        timestamps = []
        for i in range(10):
            obj = self.storage.create(obj={"seq": i}, **self.storage_kw)
            timestamps.append(obj["last_modified"])
        # All timestamps must be strictly increasing.
        for i in range(1, len(timestamps)):
            self.assertGreater(timestamps[i], timestamps[i - 1])

    def test_bump_timestamp_on_update(self):
        """An update bumps the last_modified value."""
        obj = self.objects[0]
        old_ts = obj["last_modified"]
        updated = self.storage.update(
            object_id=obj["id"], obj={"index": 0, "extra": True}, **self.storage_kw
        )
        self.assertGreater(updated["last_modified"], old_ts)

    # --- Pagination with last_modified ---

    def test_pagination_on_last_modified(self):
        """Pagination rules using last_modified work with from_epoch()."""
        # Get first 2 objects sorted by last_modified ASC
        sorting = [Sort("last_modified", 1)]
        first_page = self.storage.list_all(
            sorting=sorting, limit=2, **self.storage_kw
        )
        self.assertEqual(len(first_page), 2)

        # Paginate: get objects after the last one on the first page
        pagination_rules = [[
            Filter("last_modified", first_page[-1]["last_modified"], COMPARISON.GT),
        ]]
        second_page = self.storage.list_all(
            sorting=sorting, limit=2, pagination_rules=pagination_rules, **self.storage_kw
        )
        self.assertEqual(len(second_page), 2)

        # No overlap between pages
        first_ids = {r["id"] for r in first_page}
        second_ids = {r["id"] for r in second_page}
        self.assertEqual(len(first_ids & second_ids), 0)

    # --- Index verification ---

    def test_expression_index_does_not_exist(self):
        """The old idx_objects_last_modified_epoch index was dropped."""
        with self.storage.client.connect(readonly=True) as conn:
            result = conn.execute(sa.text("""
                SELECT 1 FROM pg_indexes
                WHERE indexname = 'idx_objects_last_modified_epoch'
            """))
            rows = result.fetchall()
        self.assertEqual(len(rows), 0)

    def test_composite_index_exists(self):
        """The composite index on (parent_id, resource_name, last_modified) exists."""
        with self.storage.client.connect(readonly=True) as conn:
            result = conn.execute(sa.text("""
                SELECT 1 FROM pg_indexes
                WHERE indexname = 'idx_objects_parent_id_resource_name_last_modified'
            """))
            rows = result.fetchall()
        self.assertEqual(len(rows), 1)


class PaginatedTest(unittest.TestCase):
    def setUp(self):
        self.storage = mock.Mock()
        self.sample_objects = [
            {"id": "object-01", "flavor": "strawberry"},
            {"id": "object-02", "flavor": "banana"},
            {"id": "object-03", "flavor": "mint"},
            {"id": "object-04", "flavor": "plain"},
            {"id": "object-05", "flavor": "peanut"},
        ]

        def sample_objects_side_effect(*args, **kwargs):
            return self.sample_objects

        self.storage.list_all.side_effect = sample_objects_side_effect

    def test_paginated_passes_sort(self):
        i = paginated(self.storage, sorting=[Sort("id", -1)])
        next(i)  # make the generator do anything
        self.storage.list_all.assert_called_with(
            sorting=[Sort("id", -1)], limit=25, pagination_rules=None
        )

    def test_paginated_passes_batch_size(self):
        i = paginated(self.storage, sorting=[Sort("id", -1)], batch_size=17)
        next(i)  # make the generator do anything
        self.storage.list_all.assert_called_with(
            sorting=[Sort("id", -1)], limit=17, pagination_rules=None
        )

    def test_paginated_yields_objects(self):
        iter = paginated(self.storage, sorting=[Sort("id", -1)])
        assert next(iter) == {"id": "object-01", "flavor": "strawberry"}

    def test_paginated_fetches_next_page(self):
        objects = self.sample_objects
        objects.reverse()

        def list_all_mock(*args, **kwargs):
            this_objects = objects[:3]
            del objects[:3]
            return this_objects

        self.storage.list_all.side_effect = list_all_mock

        list(paginated(self.storage, sorting=[Sort("id", -1)]))
        assert self.storage.list_all.call_args_list == [
            mock.call(sorting=[Sort("id", -1)], limit=25, pagination_rules=None),
            mock.call(
                sorting=[Sort("id", -1)],
                limit=25,
                pagination_rules=[[Filter("id", "object-03", COMPARISON.LT)]],
            ),
            mock.call(
                sorting=[Sort("id", -1)],
                limit=25,
                pagination_rules=[[Filter("id", "object-01", COMPARISON.LT)]],
            ),
        ]
