"""Strict BFCL chain outcomes must reject malformed or altered calls."""
import pytest

from benchmark.tasks_bfcl import _parse_call, score_chain


@pytest.mark.parametrize("pred,gold", [
    (["foo(x=1)", "not a call"], ["foo(x=1)"]),
    (["b()", "a()"], ["a()", "b()"]),
    (["foo(x=1,y=2)"], ["foo(x=1)"]),
    (["foo(x=1)"], ["foo(x=1)", "invalid gold"]),
    (["foo(x=2)"], ["foo(x=1)"]),
    (["foo(x=1)", "foo(x=1)"], ["foo(x=1)"]),
])
def test_inexact_chain_cannot_receive_full_outcome(pred, gold):
    assert score_chain(pred, gold)["outcome"] == 0.0


@pytest.mark.parametrize("pred,gold", [
    (["foo(x=1,y='a')", "b()"], ["foo(y=\"a\",x=1)", "b()"]),
    (["foo(x=[1,2])"], ["foo(x=[1, 2])"]),
])
def test_structurally_identical_chain_passes(pred, gold):
    assert score_chain(pred, gold)["outcome"] == 1.0


@pytest.mark.parametrize("call", [
    "foo(x=dict.fromkeys(['a'],1))", "foo(x=1+2)",
    "foo(x=1,x=2)", "foo(**{'x':1})", "foo(1)",
])
def test_model_expressions_are_not_executed_as_arguments(call):
    assert _parse_call(call) is None


def test_nested_python_literal_arguments_are_supported():
    assert _parse_call("foo(x={'a':[1,True,None]}, y=-2)") == {
        "name": "foo", "args": {"x": {"a": [1, True, None]}, "y": -2}}
