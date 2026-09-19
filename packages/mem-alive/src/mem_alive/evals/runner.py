from uuid import uuid4

from ..core.memory import Memory
from .rag_agent import RagAgent
from .schema import EvalCase, EvalCaseResult, EvalReport


class EvalRunner:
    def __init__(self, memory: Memory, agent: RagAgent):
        self._memory = memory
        self._agent = agent

    async def run(self, cases: tuple[EvalCase, ...]) -> EvalReport:
        run_id = uuid4().hex
        results = []
        for case in cases:
            results.append(await self._run_case(case, run_id))
        return EvalReport(results=tuple(results))

    async def _run_case(self, case: EvalCase, run_id: str) -> EvalCaseResult:
        namespace = f"eval:{run_id}:{case.name}"
        for seed in case.memories:
            await self._memory.remember(
                memory_type=seed.memory_type,
                namespace=namespace,
                fact=seed.content,
                metadata=seed.metadata,
            )

        response = await self._agent.answer(
            namespace=namespace,
            query=case.query,
            metadata=case.metadata,
            memory_type=case.memory_type,
            top_k=case.top_k,
        )
        retrieved = tuple(memory.content for memory in response.context)
        relevant = case.expected_context.intersection(retrieved)
        if case.expected_context:
            context_recall = len(relevant) / len(case.expected_context)
            context_precision = len(relevant) / len(retrieved) if retrieved else 0.0
        else:
            context_recall = float(not retrieved)
            context_precision = float(not retrieved)
        answer = response.answer.casefold()
        terms_found = sum(term.casefold() in answer for term in case.expected_answer_terms)
        answer_terms_found = terms_found / len(case.expected_answer_terms)
        passed = (
            context_recall >= case.min_context_recall
            and context_precision >= case.min_context_precision
            and answer_terms_found == 1.0
        )
        return EvalCaseResult(
            name=case.name,
            memory_type=case.memory_type,
            passed=passed,
            context_recall=context_recall,
            context_precision=context_precision,
            answer_terms_found=answer_terms_found,
            retrieved_context=retrieved,
            answer=response.answer,
        )
