"""Delete embeddings from an existing Chroma collection."""


def delete_all_vector_ids(vector_db, batch_size: int = 500) -> int:
    """
    Delete every embedding in the existing Chroma collection.

    Keeps the same collection so later uploads can add documents
    the same way as today.
    """

    deleted = 0
    previous_ids = None

    while True:
        result = vector_db.get(limit=batch_size)
        ids = result.get("ids") or []
        if not ids or ids == previous_ids:
            break
        vector_db.delete(ids=ids)
        deleted += len(ids)
        previous_ids = ids

    return deleted
