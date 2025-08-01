import chex
import numpy as np


@chex.dataclass
class ButcherTableau:
    stages: int
    a: tuple[tuple[int, ...]]
    b: tuple[int, ...]
    c: tuple[int, ...]

    @classmethod
    def from_ab(cls, a, b):
        a = tuple(tuple(ai) for ai in a)
        b = tuple(b)
        c = tuple(sum(ai) for ai in a)

        assert all(len(ai) == len(c) for ai in a)
        assert len(b) == len(c)
        assert np.isclose(sum(b), 1)

        for j in range(len(c)):
            for i in range(j+1):
                assert a[i][j] == 0, 'only explicit methods supported'

        return cls(stages=len(c), a=a, b=b, c=c)


EULER = ButcherTableau.from_ab(
    a=[[0]],
    b=[1],
)

CG2 = ButcherTableau.from_ab(
    a=[[0, 0], [1/2, 0]],
    b=[0, 1],
)

CG3 = ButcherTableau.from_ab(
    a=[[0, 0, 0], [3/4, 0, 0], [119/216, 17/108, 0]],
    b=[13/51, -2/3, 24/17],
)
