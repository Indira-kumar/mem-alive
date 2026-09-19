from dataclasses import dataclass, field


@dataclass(frozen=True)
class MemorySeed:
    memory_type: str
    content: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class EvalCase:
    name: str
    memory_type: str
    memories: tuple[MemorySeed, ...]
    query: str
    expected_context: frozenset[str]
    expected_answer_terms: frozenset[str]
    metadata: dict = field(default_factory=dict)
    top_k: int = 5
    min_context_recall: float = 1.0
    min_context_precision: float = 1.0

    def __post_init__(self) -> None:
        if not self.expected_context:
            raise ValueError("expected_context must contain at least one memory")
        if not self.expected_answer_terms:
            raise ValueError("expected_answer_terms must contain at least one term")
        for name, threshold in (
            ("min_context_recall", self.min_context_recall),
            ("min_context_precision", self.min_context_precision),
        ):
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True)
class EvalCaseResult:
    name: str
    memory_type: str
    passed: bool
    context_recall: float
    context_precision: float
    answer_terms_found: float
    retrieved_context: tuple[str, ...]
    answer: str


@dataclass(frozen=True)
class EvalReport:
    results: tuple[EvalCaseResult, ...]

    @property
    def passed(self) -> bool:
        return bool(self.results) and all(result.passed for result in self.results)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(result.passed for result in self.results) / len(self.results)

    @property
    def mean_context_recall(self) -> float:
        if not self.results:
            return 0.0
        return sum(result.context_recall for result in self.results) / len(self.results)

    @property
    def mean_context_precision(self) -> float:
        if not self.results:
            return 0.0
        return sum(result.context_precision for result in self.results) / len(self.results)
