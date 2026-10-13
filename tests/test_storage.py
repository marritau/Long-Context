from bikv.storage import ResultStore, stable_mask_key


def test_atomic_store_mirrors_and_recovers(tmp_path):
    local = tmp_path / "local"
    mirror = tmp_path / "mirror"
    store = ResultStore(local, "run", mirror)
    store.write_json("masks/sample", "one", {"value": 3})
    local_file = store.path("masks/sample", "one")
    assert local_file.exists()
    local_file.unlink()
    assert store.exists("masks/sample", "one")
    assert store.read_json("masks/sample", "one") == {"value": 3}
    store.write_text("reports", "pilot", "ok\n")
    bundle = store.bundle()
    assert bundle.exists()
    assert (mirror / bundle.name).exists()


def test_mask_key_is_order_independent():
    assert stable_mask_key([3, 1, 2]) == stable_mask_key([2, 3, 1])
