from bikv.features import FEATURE_NAMES
from bikv.predictor import LinearInteractionModel, train_linear_predictor


def test_exported_predictor_runs_without_sklearn():
    model = LinearInteractionModel(
        feature_names=list(FEATURE_NAMES),
        mean=[0.0] * len(FEATURE_NAMES),
        scale=[1.0] * len(FEATURE_NAMES),
        coefficients=[1.0] + [0.0] * (len(FEATURE_NAMES) - 1),
        intercept=0.5,
    )
    row = {name: 0.0 for name in FEATURE_NAMES}
    row[FEATURE_NAMES[0]] = 2.0
    assert model.predict(row) == 2.5


def test_training_reports_document_ranking_and_group_metrics():
    def rows(split_offset):
        result = []
        for family in range(2):
            sample_id = f"sample-{split_offset}-{family}"
            for boundary in range(3):
                features = {name: 0.0 for name in FEATURE_NAMES}
                features["boundary_position"] = float(boundary)
                result.append(
                    {
                        "story_id": sample_id,
                        "sample_id": sample_id,
                        "counterfactual_group": f"family-{split_offset}-{family}",
                        "features": features,
                        "target": float(boundary),
                        "proposal_utility": float(boundary),
                        "is_support_boundary": boundary == 2,
                        "supporting_blocks": [2, 3],
                        "proposal_mask": [0, 2, 3] if boundary == 2 else [0, 1, 2],
                        "singleton_mask": [0, 1, 2],
                        "singleton_utility": 0.0,
                    }
                )
        return result

    model = train_linear_predictor(rows("train"), rows("validation"))
    metrics = model.validation_metrics
    assert metrics is not None
    assert metrics["train_counterfactual_groups"] == 2.0
    assert metrics["validation_pair_recall_at_1"] == 1.0
    assert metrics["validation_support_retention"] == 1.0
    assert metrics["validation_support_interaction_gap"] > 0.0
    assert metrics["validation_support_positive_interaction_rate"] == 1.0
