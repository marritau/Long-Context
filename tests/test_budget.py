from pearl.compression.budget_allocator import allocate_budgets


def test_allocate_budgets_respects_total():
    budgets = allocate_budgets(total_budget=100, importances=[3.0, 1.0, 1.0], min_budget=10)

    assert sum(budgets) == 100
    assert budgets[0] > budgets[1]
    assert all(budget >= 10 for budget in budgets)


def test_allocate_budgets_handles_tiny_total():
    budgets = allocate_budgets(total_budget=5, importances=[1.0, 1.0, 1.0], min_budget=10)

    assert sum(budgets) == 5
    assert all(budget >= 1 for budget in budgets)
