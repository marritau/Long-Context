from bikv.runner import interaction_contrast


def test_interaction_contrast_is_difference_in_differences():
    effects = interaction_contrast(
        {"both": 9.0, "left_only": 4.0, "right_only": 3.0, "neither": 1.0}
    )
    assert effects["interaction"] == 3.0
    assert effects["joint_gain"] == 8.0
    assert effects["left_conditional_on_right"] == 6.0
    assert effects["right_conditional_on_left"] == 5.0


def test_additive_pair_has_zero_interaction():
    effects = interaction_contrast(
        {"both": 8.0, "left_only": 5.0, "right_only": 4.0, "neither": 1.0}
    )
    assert effects["interaction"] == 0.0
