"""Tests for the error recovery system — detection, classification, discussion room."""

from veska.recovery.error_recovery import (
    ErrorDetector,
    ErrorSeverity,
    ErrorCategory,
    DiscussionRoom,
    FixCoordinator,
    DetectedError,
)


def test_detect_code_error():
    """Code errors should be classified as LEVEL_2."""
    d = ErrorDetector()
    error = d.detect(agent_name="be", message="TypeError: cannot add str and int")

    assert error.category == ErrorCategory.CODE_ERROR
    assert error.severity == ErrorSeverity.LEVEL_2


def test_detect_timeout():
    """Timeouts should be classified as LEVEL_1."""
    d = ErrorDetector()
    error = d.detect(agent_name="be", message="Request timed out after 30s")

    assert error.category == ErrorCategory.TIMEOUT
    assert error.severity == ErrorSeverity.LEVEL_1


def test_detect_cross_agent_error():
    """Cross-agent errors should be classified as LEVEL_3."""
    d = ErrorDetector()
    error = d.detect(agent_name="fe", message="Interface mismatch with backend API")

    assert error.category == ErrorCategory.CROSS_AGENT
    assert error.severity == ErrorSeverity.LEVEL_3


def test_escalation_after_retries():
    """Repeated failures should escalate severity."""
    d = ErrorDetector()

    error1 = d.detect(agent_name="be", message="Tool failed", retry_count=0)
    assert error1.severity == ErrorSeverity.LEVEL_1

    error2 = d.detect(agent_name="be", message="Tool failed", retry_count=2)
    assert error2.severity == ErrorSeverity.LEVEL_2

    error3 = d.detect(agent_name="be", message="Tool failed", retry_count=3)
    assert error3.severity == ErrorSeverity.LEVEL_3


def test_resolve_error():
    """Resolving an error should mark it as resolved."""
    d = ErrorDetector()
    error = d.detect(agent_name="be", message="Some error")

    d.resolve(error.id, "Fixed it")
    assert error.resolved
    assert error.resolution == "Fixed it"


def test_unresolved_errors():
    """get_unresolved should return only unresolved errors."""
    d = ErrorDetector()
    e1 = d.detect(agent_name="be", message="Error 1")
    e2 = d.detect(agent_name="be", message="Error 2")

    d.resolve(e1.id)
    unresolved = d.get_unresolved()
    assert len(unresolved) == 1
    assert unresolved[0].id == e2.id


def test_discussion_room_messages():
    """Discussion room should track messages from participants."""
    error = DetectedError(agent_name="be", message="API broke")
    room = DiscussionRoom(error=error, participants=["be", "fe"])

    room.add_message("be", "I changed the API format")
    room.add_message("fe", "That broke my dashboard")

    assert room.message_count == 2


def test_discussion_room_ignores_non_participants():
    """Non-participants should not be able to add messages."""
    error = DetectedError(agent_name="be", message="API broke")
    room = DiscussionRoom(error=error, participants=["be", "fe"])

    room.add_message("random_agent", "I want to talk too")
    assert room.message_count == 0


def test_discussion_room_propose_fix():
    """Fix proposals should be tracked in the room."""
    error = DetectedError(agent_name="be", message="API broke")
    room = DiscussionRoom(error=error, participants=["be", "fe"])

    report = room.propose_fix(
        from_agent="fe",
        to_agent="be",
        problem="API format changed",
        suggestion="Add backwards compatibility",
    )

    result = room.conclude("Agents agreed on fix")
    assert len(result.fix_reports) == 1
    assert result.fix_reports[0].to_agent == "be"
    assert result.resolved


def test_fix_coordinator():
    """FixCoordinator should track and manage fix reports."""
    error = DetectedError(agent_name="be", message="API broke")
    room = DiscussionRoom(error=error, participants=["be", "fe"])
    report = room.propose_fix(from_agent="fe", to_agent="be", problem="Format changed")
    result = room.conclude()

    coord = FixCoordinator()
    coord.add_reports(result.fix_reports)

    pending = coord.get_pending("be")
    assert len(pending) == 1

    coord.complete(pending[0].id, "Fixed the format")
    assert coord.is_error_resolved(error.id)
