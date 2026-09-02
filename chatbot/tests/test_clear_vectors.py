from services.vector_cleanup import delete_all_vector_ids


class FakeVectorDB:
    def __init__(self, ids):
        self.ids = list(ids)
        self.deleted = []

    def get(self, limit=500):
        return {"ids": self.ids[:limit]}

    def delete(self, ids):
        self.deleted.extend(ids)
        remaining = set(ids)
        self.ids = [item for item in self.ids if item not in remaining]


def test_delete_all_vector_ids_removes_existing_ids():
    vector_db = FakeVectorDB(["a", "b", "c"])
    deleted = delete_all_vector_ids(vector_db, batch_size=2)
    assert deleted == 3
    assert vector_db.ids == []


def test_delete_all_vector_ids_empty_collection():
    vector_db = FakeVectorDB([])
    deleted = delete_all_vector_ids(vector_db)
    assert deleted == 0
    assert vector_db.deleted == []
