def test_verify_contains_product_name_registered_as_blocker_check():
    from general_beckman.posthooks import POST_HOOK_REGISTRY
    spec = POST_HOOK_REGISTRY["verify_contains_product_name"]
    assert spec.kind == "verify_contains_product_name"
    assert spec.verb == "verify_contains_product_name"
    assert spec.default_severity == "blocker"
    assert spec.auto_wire_triggers == []


def test_check_kind_is_derived_for_apply():
    # apply._CHECK_KINDS derives verify_* kinds from the registry by name.
    from general_beckman.apply import _CHECK_KINDS
    assert "verify_contains_product_name" in _CHECK_KINDS
