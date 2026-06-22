from mr_roboto.verify_contains_product_name import verify_contains_product_name


def test_present_whole_word_passes():
    res = verify_contains_product_name(
        product_name="FlowState",
        artifact_texts=["# Launch\nIntroducing FlowState, the app."],
    )
    assert res["ok"] is True
    assert res["found"] is True


def test_absent_fails():
    res = verify_contains_product_name(
        product_name="FlowState",
        artifact_texts=["# Launch\nIntroducing HabitTrack, the app."],
    )
    assert res["ok"] is False
    assert res["found"] is False


def test_case_insensitive():
    res = verify_contains_product_name(
        product_name="FlowState", artifact_texts=["we love flowstate here"],
    )
    assert res["ok"] is True


def test_substring_in_larger_word_does_not_match():
    # "Flow" must not match inside "Flowers"; whole-word only.
    res = verify_contains_product_name(
        product_name="Flow", artifact_texts=["Flowers everywhere, no product"],
    )
    assert res["found"] is False
    assert res["ok"] is False


def test_empty_name_is_defensive_skip():
    res = verify_contains_product_name(product_name="  ", artifact_texts=["anything"])
    assert res["ok"] is True
    assert res["skipped"] == "no product_name pinned"


def test_none_name_is_defensive_skip():
    res = verify_contains_product_name(product_name=None, artifact_texts=[])
    assert res["ok"] is True
