"""PennyLane finite-qubit operations and fixed quantum-channel interchange."""
import numpy as np
from ..channels import KrausChannel
from ..numerics import Precision, embed_operator, unitary


def from_operations(operations, *, wire_order, precision=None):
    """Convert an explicit numeric operation sequence to a Kraus channel.

    Includes unitary and Channel operations. Rejects state preparation, CV gates,
    mid-circuit measurement and unsupported operations; never ignores them.
    Wire order is mandatory. This conversion is not differentiable.
    """
    import pennylane as qml
    p = precision or Precision()
    wires = tuple(wire_order)
    if not wires or len(set(wires)) != len(wires):
        raise ValueError("wire_order must be nonempty and contain unique wires")
    p.guard(2**len(wires))
    result = KrausChannel.unitary(np.eye(2**len(wires)), precision=p)
    for operation in operations:
        if any(w not in wires for w in operation.wires):
            raise ValueError("Operation contains wires outside wire_order")
        targets = [wires.index(w) for w in operation.wires]
        if isinstance(operation,qml.operation.Channel):
            matrices = [np.asarray(k,dtype=complex) for k in operation.kraus_matrices()]
        else:
            if isinstance(operation, (qml.operation.StatePrepBase, qml.operation.CV)) or not operation.has_matrix:
                raise ValueError(f"Unsupported PennyLane operation: {operation.name}")
            matrices = [unitary(np.asarray(qml.matrix(operation),dtype=complex), p.atol)]
        layer = KrausChannel([embed_operator(k,targets,len(wires)) for k in matrices], precision=p, name=operation.name)
        result = result.then(layer)
    return result


def to_operation(channel, wires):
    """Queue a fixed CPTP channel. Call .flagged() explicitly for success/failure maps.

    For algorithm gradients use QNode diff_method='parameter-shift'. Some SDK
    backprop paths are singular at rank-deficient flagged output states.
    """
    import pennylane as qml
    wires = tuple(wires)
    if not channel.trace_preserving:
        raise ValueError("A trace-decreasing map cannot be a QubitChannel. Use channel.flagged() and include the flag wire")
    if channel.input_dimension != channel.output_dimension or 2**len(wires) != channel.input_dimension:
        raise ValueError("The channel size must match 2**len(wires)")
    if len(set(wires)) != len(wires):
        raise ValueError("Duplicate wires")
    return qml.QubitChannel(list(channel.operators), wires=wires)


def logical_unitary(qfunc, *args, wire_order, **kwargs):
    """Evaluate a numeric unitary quantum function. Terminal measurements are rejected.

    Feed the resulting matrix to LogicalQPU.logical_gate. That gate remains an
    ideal encoded operation, not an automatically synthesized optical circuit.
    """
    import pennylane as qml
    tape = qml.tape.make_qscript(qfunc)(*args, **kwargs)
    if tape.measurements:
        raise ValueError("Provide an operation-only quantum function without terminal measurements")
    for op in tape.operations:
        if isinstance(op, (qml.operation.Channel, qml.operation.StatePrepBase, qml.operation.CV)) or not op.has_matrix:
            raise ValueError(f"The function is not unitary-only: {op.name}")
    channel = from_operations(tape.operations,wire_order=wire_order)
    return channel.operators[0].copy()
