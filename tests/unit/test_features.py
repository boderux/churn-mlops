from churn.features import add_features, build_preprocessor


def test_engineered_columns(raw_df):
    out = add_features(raw_df.head(50))
    for c in ["avg_monthly_spend", "n_services", "tenure_group", "is_month_to_month", "has_internet"]:
        assert c in out
    assert out["n_services"].between(0, 8).all()
    assert out["avg_monthly_spend"].ge(0).all()


def test_add_features_does_not_mutate(raw_df):
    before = raw_df.head(5).copy()
    add_features(before)
    assert "n_services" not in before


def test_preprocessor_drops_sensitive(raw_df):
    prep = build_preprocessor(drop=["gender"])
    prep.fit(add_features(raw_df))
    names = prep.get_feature_names_out()
    assert not any("gender" in n for n in names)
    assert any("Contract" in n for n in names)
