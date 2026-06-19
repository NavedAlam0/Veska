"""Tests for the Task Planner — dependency graph and execution order."""

from veska.core.task_planner import Task, TaskPlanner, TaskStatus


def test_independent_tasks_run_in_parallel():
    """Tasks with no dependencies should be in the same wave."""
    p = TaskPlanner()
    a = Task(id="a", name="A", agent="r")
    b = Task(id="b", name="B", agent="r")
    p.add_tasks([a, b])

    waves = p.get_execution_order()
    assert len(waves) == 1
    assert set(waves[0]) == {"a", "b"}


def test_dependent_task_runs_after():
    """A task depending on others should be in a later wave."""
    p = TaskPlanner()
    a = Task(id="a", name="A", agent="r")
    b = Task(id="b", name="B", agent="r")
    c = Task(id="c", name="C", agent="w", depends_on=["a", "b"])
    p.add_tasks([a, b, c])

    waves = p.get_execution_order()
    assert len(waves) == 2
    assert set(waves[0]) == {"a", "b"}
    assert waves[1] == ["c"]


def test_three_level_dependency_chain():
    """A → B → C should produce 3 waves."""
    p = TaskPlanner()
    p.add_tasks([
        Task(id="a", name="A", agent="r"),
        Task(id="b", name="B", agent="r", depends_on=["a"]),
        Task(id="c", name="C", agent="r", depends_on=["b"]),
    ])

    waves = p.get_execution_order()
    assert len(waves) == 3
    assert waves[0] == ["a"]
    assert waves[1] == ["b"]
    assert waves[2] == ["c"]


def test_circular_dependency_detected():
    """Circular dependencies should be caught by validate()."""
    p = TaskPlanner()
    p.add_tasks([
        Task(id="a", name="A", agent="r", depends_on=["b"]),
        Task(id="b", name="B", agent="r", depends_on=["a"]),
    ])

    errors = p.validate()
    assert len(errors) > 0
    assert "Circular" in errors[0]


def test_missing_dependency_detected():
    """A task depending on a non-existent task should be flagged."""
    p = TaskPlanner()
    p.add_task(Task(id="a", name="A", agent="r", depends_on=["missing"]))

    errors = p.validate()
    assert len(errors) > 0
    assert "missing" in errors[0]


def test_valid_plan_has_no_errors():
    """A valid plan should return empty errors list."""
    p = TaskPlanner()
    p.add_tasks([
        Task(id="a", name="A", agent="r"),
        Task(id="b", name="B", agent="r", depends_on=["a"]),
    ])

    assert p.validate() == []


def test_get_ready_tasks():
    """Only tasks with all dependencies met should be ready."""
    p = TaskPlanner()
    a = Task(id="a", name="A", agent="r")
    b = Task(id="b", name="B", agent="w", depends_on=["a"])
    p.add_tasks([a, b])

    ready = p.get_ready_tasks()
    assert len(ready) == 1
    assert ready[0].id == "a"


def test_completing_task_unlocks_dependents():
    """After completing a task, its dependents should become ready."""
    p = TaskPlanner()
    a = Task(id="a", name="A", agent="r")
    b = Task(id="b", name="B", agent="w", depends_on=["a"])
    p.add_tasks([a, b])

    p.complete_task("a", "done")
    ready = p.get_ready_tasks()
    assert len(ready) == 1
    assert ready[0].id == "b"


def test_task_retry():
    """A failed task should be retryable up to max_retries."""
    p = TaskPlanner()
    a = Task(id="a", name="A", agent="r", max_retries=2)
    p.add_task(a)

    p.fail_task("a", "error")
    assert p.get_task("a").can_retry

    p.retry_task("a")
    assert p.get_task("a").status == TaskStatus.RETRYING
    assert p.get_task("a").retries == 1


def test_progress_tracking():
    """Progress should reflect task statuses."""
    p = TaskPlanner()
    p.add_tasks([
        Task(id="a", name="A", agent="r"),
        Task(id="b", name="B", agent="r"),
    ])

    assert p.progress["total"] == 2
    assert p.progress["completed"] == 0

    p.complete_task("a", "done")
    assert p.progress["completed"] == 1
    assert p.progress["percentage"] == 50.0


def test_is_complete():
    """is_complete should be True only when all tasks are terminal."""
    p = TaskPlanner()
    p.add_tasks([
        Task(id="a", name="A", agent="r"),
        Task(id="b", name="B", agent="r"),
    ])

    assert not p.is_complete
    p.complete_task("a", "done")
    assert not p.is_complete
    p.complete_task("b", "done")
    assert p.is_complete
