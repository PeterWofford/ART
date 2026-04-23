import tempfile

from art.tinker_native.backend import TinkerNativeBackend


def test_build_datums_with_distributions_uses_full_topk() -> None:
    backend = TinkerNativeBackend(path=tempfile.mkdtemp(), tinker_api_key="test-key")

    prompt_tokens = [101, 102]
    completion_tokens = [201]
    student_topk = [[(201, -0.1), (202, -0.5), (203, -1.0)]]
    teacher_topk = [[(201, -0.2), (202, -0.4), (203, -1.2)]]
    student_tail_mass = [0.1]
    teacher_tail_mass = [0.2]

    datums = backend._build_datums_with_distributions(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        student_topk=student_topk,
        teacher_topk=teacher_topk,
        student_tail_mass=student_tail_mass,
        teacher_tail_mass=teacher_tail_mass,
    )

    assert len(datums) == 3
