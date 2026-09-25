import io
from unittest.mock import MagicMock

from agents.manager_agent import ManagerResult
from cli import run


def _manager_returning(result: ManagerResult) -> MagicMock:
    manager = MagicMock()
    manager.handle.return_value = result
    return manager


def test_prints_banner_and_answer_then_exits_on_command():
    manager = _manager_returning(ManagerResult(final_answer="the answer"))
    stdin = io.StringIO("what is the policy?\nexit\n")
    stdout = io.StringIO()

    run(manager, input_stream=stdin, output_stream=stdout)

    output = stdout.getvalue()
    assert "Multi-Agent RAG System" in output
    assert "the answer" in output
    manager.handle.assert_called_once_with("what is the policy?")


def test_blank_lines_are_ignored_and_do_not_call_manager():
    manager = _manager_returning(ManagerResult(final_answer="unused"))
    stdin = io.StringIO("\n\nexit\n")
    stdout = io.StringIO()

    run(manager, input_stream=stdin, output_stream=stdout)

    manager.handle.assert_not_called()


def test_eof_ends_loop_without_exit_command():
    manager = _manager_returning(ManagerResult(final_answer="unused"))
    stdin = io.StringIO("")  # immediate EOF
    stdout = io.StringIO()

    run(manager, input_stream=stdin, output_stream=stdout)

    manager.handle.assert_not_called()
    assert "Multi-Agent RAG System" in stdout.getvalue()


def test_clarification_result_is_prefixed():
    manager = _manager_returning(
        ManagerResult(
            final_answer="please clarify",
            clarification_needed=True,
            clarification_question="please clarify",
        )
    )
    stdin = io.StringIO("vague question\nexit\n")
    stdout = io.StringIO()

    run(manager, input_stream=stdin, output_stream=stdout)

    assert "[Clarification needed] please clarify" in stdout.getvalue()
