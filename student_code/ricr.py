"""Connected-DAG Reasoning Inference Chain Retrieval for PANINI."""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence


_ENTITY = re.compile(r"<ENTITY_Q(\d+)>")


@dataclass(frozen=True)
class Candidate:
    qa_uid: str
    answer_names: tuple[str, ...]
    score: float
    question: str = ""
    answer_ids: tuple[str, ...] = ()
    answer_role_states: tuple[str, ...] = ()
    document_id: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ChainState:
    steps: tuple[Candidate, ...]
    answers_by_step: Mapping[int, str | tuple[str, ...]]
    score: float
    last_hop_score: float = 0.0

    @property
    def current_answers(self) -> tuple[str, ...]:
        return self.steps[-1].answer_names if self.steps else ()


@dataclass(frozen=True)
class RICRResult:
    components: tuple[tuple[int, ...], ...]
    chains: tuple[ChainState, ...]
    evidence: tuple[Candidate, ...]
    issued_queries: tuple[str, ...]
    fallback: bool = False
    trace: Mapping[str, object] = field(default_factory=dict)


def normalize_entity_name(value: str) -> str:
    return " ".join(str(value).casefold().split())


def geometric_mean(values: Sequence[float], epsilon: float = 1e-12) -> float:
    if not values:
        return 0.0
    vals = [max(float(v), epsilon) for v in values]
    return float(math.exp(sum(math.log(v) for v in vals) / len(vals)))


def panini_chain_score(steps: Sequence[Candidate]) -> float:
    transformed = [
        max(1e-6, min(1.0, 0.5 * (float(c.score) + 1.0)))
        for c in steps
    ]
    return geometric_mean(transformed, epsilon=1e-6)


def harmonic_mean(scores: Sequence[float]) -> float:
    vals = [float(s) for s in scores if float(s) > 1e-6]
    return len(vals) / sum(1.0 / v for v in vals) if vals else 1e-6


def instantiate_question(
    template: str,
    answers_by_step: Mapping[int, str | tuple[str, ...]],
) -> str:
    def repl(match: re.Match[str]) -> str:
        node = int(match.group(1))
        if node not in answers_by_step:
            raise KeyError(f"Unresolved Q{node}: {template}")
        value = answers_by_step[node]
        return ", ".join(value) if isinstance(value, tuple) else str(value)

    return _ENTITY.sub(repl, str(template))


def _toposort(nodes: set[int], edges: set[tuple[int, int]]) -> list[int]:
    incoming = {n: set() for n in nodes}
    outgoing = {n: set() for n in nodes}

    for parent, child in edges:
        if parent in nodes and child in nodes:
            outgoing[parent].add(child)
            incoming[child].add(parent)

    ready = sorted(n for n in nodes if not incoming[n])
    result = []

    while ready:
        node = ready.pop(0)
        result.append(node)
        for child in sorted(outgoing[node]):
            incoming[child].discard(node)
            if not incoming[child] and child not in ready:
                ready.append(child)
        ready.sort()

    if len(result) != len(nodes):
        raise ValueError("Cyclic retrieval dependency graph")

    return result


def _dependency_graph(
    plan: Sequence[Mapping[str, object]],
):
    total = len(plan)
    retrieval_nodes = {
        i for i, row in enumerate(plan, 1)
        if bool(row.get("requires_retrieval", True))
    }

    all_edges: set[tuple[int, int]] = set()
    retrieval_edges: set[tuple[int, int]] = set()

    for child, row in enumerate(plan, 1):
        refs = [int(x) for x in _ENTITY.findall(str(row.get("question", "")))]

        for parent in refs:
            if not 1 <= parent <= total:
                raise ValueError(f"Q{child} references missing Q{parent}")
            if parent >= child:
                raise ValueError(
                    f"Q{child} references non-earlier Q{parent}"
                )

            all_edges.add((parent, child))

            if parent in retrieval_nodes and child in retrieval_nodes:
                retrieval_edges.add((parent, child))

    _toposort(set(range(1, total + 1)), all_edges)

    parents = {n: set() for n in retrieval_nodes}
    children = {n: set() for n in retrieval_nodes}

    for parent, child in retrieval_edges:
        parents[child].add(parent)
        children[parent].add(child)

    return retrieval_nodes, parents, children, retrieval_edges


def identify_retrieval_components(
    decomposed_questions: Sequence[Mapping[str, object]],
) -> list[list[int]]:
    retrieval_nodes, parents, children, edges = _dependency_graph(
        decomposed_questions
    )

    remaining = set(retrieval_nodes)
    found = []

    while remaining:
        start = min(remaining)
        stack = [start]
        component = set()

        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend((parents[node] | children[node]) - component)

        remaining -= component

        if len(component) > 1:
            found.append(_toposort(component, edges))

    return sorted(found, key=lambda x: (x[0], tuple(x)))


def _candidate_info(candidate: Candidate) -> dict[str, object]:
    return {
        "qa_uid": str(candidate.qa_uid),
        "answer_names": list(candidate.answer_names),
        "answer_ids": list(candidate.answer_ids),
        "score": float(candidate.score),
        "question": candidate.question,
        "document_id": candidate.document_id,
    }


def _state_info(state: ChainState) -> dict[str, object]:
    return {
        "qa_ids": [c.qa_uid for c in state.steps],
        "answers_by_step": {
            str(k): v for k, v in state.answers_by_step.items()
        },
        "score": float(state.score),
        "last_hop_score": float(state.last_hop_score),
    }


def _state_order(state: ChainState):
    return (
        -float(state.score),
        -float(state.last_hop_score),
        tuple(c.qa_uid for c in state.steps),
        tuple(
            (int(k), str(v))
            for k, v in sorted(state.answers_by_step.items())
        ),
    )


def _limit_candidates(
    candidates: Sequence[Candidate],
    limit: int,
) -> list[Candidate]:
    best: dict[str, Candidate] = {}

    for candidate in candidates:
        if not isinstance(candidate, Candidate):
            raise TypeError("Retriever must return Candidate objects")

        uid = str(candidate.qa_uid)
        old = best.get(uid)

        if old is None or (
            float(candidate.score),
            tuple(candidate.answer_names),
            uid,
        ) > (
            float(old.score),
            tuple(old.answer_names),
            uid,
        ):
            best[uid] = candidate

    return sorted(
        best.values(),
        key=lambda c: (-float(c.score), str(c.qa_uid)),
    )[:max(0, limit)]


def _push(
    state: ChainState,
    node: int,
    candidate: Candidate,
    answer: str | tuple[str, ...],
) -> ChainState:
    answers = dict(state.answers_by_step)
    answers[node] = answer
    steps = state.steps + (candidate,)

    return ChainState(
        steps=steps,
        answers_by_step=answers,
        score=panini_chain_score(steps),
        last_hop_score=float(candidate.score),
    )


def _entity_key(
    candidate: Candidate,
    answer_index: int,
    answer_name: str,
) -> str:
    raw = (
        str(candidate.answer_ids[answer_index])
        if answer_index < len(candidate.answer_ids)
        else ""
    )

    if raw:
        if "::" in raw:
            return f"id::{raw}"

        document = str(candidate.document_id)
        gsw_file = str(candidate.metadata.get("gsw_file", ""))

        if document and gsw_file:
            return f"id::{document}::{gsw_file}::{raw}"
        if document:
            return f"id::{document}::{raw}"

        return f"id::{candidate.qa_uid}::{raw}"

    return f"name::{normalize_entity_name(answer_name)}"


def _expand_intermediate(
    states: Sequence[ChainState],
    node: int,
    candidates: Sequence[Candidate],
    width: int,
    group_entities: bool,
):
    generated: list[tuple[str, ChainState]] = []

    for state in states:
        for candidate in candidates:
            local_seen = set()

            for i, name in enumerate(candidate.answer_names):
                answer = str(name).strip()
                if not answer:
                    continue

                key = _entity_key(candidate, i, answer)
                if key in local_seen:
                    continue
                local_seen.add(key)

                generated.append(
                    (
                        key,
                        _push(state, node, candidate, answer),
                    )
                )

    if group_entities:
        best_by_entity: dict[str, ChainState] = {}

        for key, state in generated:
            old = best_by_entity.get(key)
            if old is None or _state_order(state) < _state_order(old):
                best_by_entity[key] = state

        pool = list(best_by_entity.values())
    else:
        pool = [state for _, state in generated]

    pool.sort(key=_state_order)
    return pool[:width], pool[width:]


def _expand_final(
    states: Sequence[ChainState],
    node: int,
    candidates: Sequence[Candidate],
    width: int,
):
    expanded = []

    for state in states:
        for candidate in candidates:
            if not candidate.answer_names:
                continue

            answer = (
                candidate.answer_names[0]
                if len(candidate.answer_names) == 1
                else tuple(candidate.answer_names)
            )

            expanded.append(
                _push(state, node, candidate, answer)
            )

    expanded.sort(key=_state_order)
    return expanded[:width], expanded[width:]


def _merge_states(
    states: Sequence[ChainState],
) -> ChainState | None:
    answers: dict[int, str | tuple[str, ...]] = {}
    steps: list[Candidate] = []
    seen = set()

    for state in states:
        for node, value in state.answers_by_step.items():
            if node in answers and answers[node] != value:
                return None
            answers[node] = value

        for candidate in state.steps:
            uid = str(candidate.qa_uid)
            if uid not in seen:
                seen.add(uid)
                steps.append(candidate)

    return ChainState(
        steps=tuple(steps),
        answers_by_step=answers,
        score=panini_chain_score(steps) if steps else 1.0,
        last_hop_score=float(steps[-1].score) if steps else 1.0,
    )


def _select_parent_products(
    parent_nodes: Sequence[int],
    beams: Mapping[int, Sequence[ChainState]],
    width: int,
    threshold: float,
):
    products = []

    for combo in itertools.product(
        *(beams.get(parent, ()) for parent in parent_nodes)
    ):
        merged = _merge_states(combo)
        if merged is None:
            continue

        h = harmonic_mean([state.score for state in combo])
        products.append((h, combo, merged))

    products.sort(
        key=lambda item: (
            -item[0],
            tuple(
                tuple(c.qa_uid for c in state.steps)
                for state in item[1]
            ),
        )
    )

    considered = products[:width]
    chosen = [item for item in considered if item[0] >= threshold]

    if not chosen and considered:
        chosen = [considered[0]]

    chosen_refs = {id(item) for item in chosen}

    trace = [
        {
            "parent_nodes": list(parent_nodes),
            "parent_qa_ids": [
                [c.qa_uid for c in state.steps]
                for state in combo
            ],
            "harmonic_score": float(h),
            "examined": index < width,
            "selected": id(item) in chosen_refs,
        }
        for index, item in enumerate(products)
        for h, combo, _ in [item]
    ]

    return [merged for _, _, merged in chosen], trace


def _union_evidence(chains: Sequence[ChainState]) -> tuple[Candidate, ...]:
    evidence = []
    seen = set()

    for chain in chains:
        for candidate in chain.steps:
            uid = str(candidate.qa_uid)
            if uid not in seen:
                seen.add(uid)
                evidence.append(candidate)

    return tuple(evidence)


def _missing_embedding(exc: Exception) -> bool:
    text = str(exc).casefold()
    name = type(exc).__name__

    return (
        name in {"MissingEmbedding", "MissingQueryEmbeddingError"}
        or "no supplied embedding" in text
        or "supplied embedding" in text
    )


def run_panini_ricr(
    decomposed_questions: Sequence[Mapping[str, object]],
    retrieve_and_score: Callable[[str, int], Sequence[Candidate]],
    *,
    original_question: str,
    beam_width: int = 5,
    candidates_per_hop: int = 15,
    multi_dependency_threshold: float = 0.3,
    unique_intermediate_entities: bool = True,
) -> RICRResult:

    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    if candidates_per_hop <= 0:
        raise ValueError("candidates_per_hop must be positive")
    if multi_dependency_threshold < 0:
        raise ValueError("threshold must be non-negative")

    components = identify_retrieval_components(decomposed_questions)
    _, parents, children, _ = _dependency_graph(decomposed_questions)

    cache: dict[str, list[Candidate]] = {}
    issued_queries: list[str] = []

    def retrieve(query: str) -> list[Candidate]:
        if query not in cache:
            issued_queries.append(query)
            cache[query] = _limit_candidates(
                tuple(
                    retrieve_and_score(
                        query,
                        candidates_per_hop,
                    )
                ),
                candidates_per_hop,
            )
        return cache[query]

    # Required singleton/original-question fallback.
    if not components:
        candidates = retrieve(original_question)

        seed = ChainState(
            steps=(),
            answers_by_step={},
            score=1.0,
            last_hop_score=1.0,
        )

        kept, pruned = _expand_final(
            [seed],
            1,
            candidates,
            beam_width,
        )

        return RICRResult(
            components=(),
            chains=tuple(kept),
            evidence=_union_evidence(kept),
            issued_queries=tuple(issued_queries),
            fallback=True,
            trace={
                "fallback": True,
                "query": original_question,
                "candidates": [
                    _candidate_info(c)
                    for c in candidates
                ],
                "retained_states": [
                    _state_info(s)
                    for s in kept
                ],
                "pruned_states": [
                    _state_info(s)
                    for s in pruned
                ],
            },
        )

    all_final: list[ChainState] = []
    component_logs = []

    for component in components:
        component_set = set(component)
        local_parents = {
            node: sorted(parents[node] & component_set)
            for node in component
        }
        local_children = {
            node: sorted(children[node] & component_set)
            for node in component
        }

        beams: dict[int, list[ChainState]] = {}
        node_logs = []

        for node in component:
            parent_nodes = local_parents[node]

            if not parent_nodes:
                base_states = [
                    ChainState(
                        steps=(),
                        answers_by_step={},
                        score=1.0,
                        last_hop_score=1.0,
                    )
                ]
                parent_log = []

            elif len(parent_nodes) == 1:
                base_states = list(beams[parent_nodes[0]])
                parent_log = []

            else:
                base_states, parent_log = _select_parent_products(
                    parent_nodes,
                    beams,
                    beam_width,
                    multi_dependency_threshold,
                )

            template = str(
                decomposed_questions[node - 1].get(
                    "question",
                    "",
                )
            )

            queries: dict[str, list[ChainState]] = {}
            failures = []

            for state in base_states:
                try:
                    concrete = instantiate_question(
                        template,
                        state.answers_by_step,
                    )
                except KeyError as exc:
                    failures.append(str(exc))
                    continue

                queries.setdefault(concrete, []).append(state)

            is_sink = not local_children[node]
            retained = []
            pruned = []
            query_logs = []

            for query in sorted(queries):
                try:
                    candidates = retrieve(query)
                except Exception as exc:
                    if not _missing_embedding(exc):
                        raise

                    failures.append(
                        f"{query}: {type(exc).__name__}: {exc}"
                    )
                    continue

                if is_sink:
                    good, bad = _expand_final(
                        queries[query],
                        node,
                        candidates,
                        beam_width,
                    )
                else:
                    good, bad = _expand_intermediate(
                        queries[query],
                        node,
                        candidates,
                        beam_width,
                        unique_intermediate_entities,
                    )

                retained.extend(good)
                pruned.extend(bad)

                query_logs.append({
                    "query": query,
                    "base_states": [
                        _state_info(s)
                        for s in queries[query]
                    ],
                    "candidates": [
                        _candidate_info(c)
                        for c in candidates
                    ],
                    "retained_states": [
                        _state_info(s)
                        for s in good
                    ],
                    "pruned_states": [
                        _state_info(s)
                        for s in bad
                    ],
                })

            pool = sorted(
                retained + pruned,
                key=_state_order,
            )

            beams[node] = pool[:beam_width]

            node_logs.append({
                "node": node,
                "template": template,
                "parents": parent_nodes,
                "children": local_children[node],
                "final": is_sink,
                "parent_combinations": parent_log,
                "query_errors": failures,
                "queries": query_logs,
                "retained_states": [
                    _state_info(s)
                    for s in beams[node]
                ],
                "pruned_states": [
                    _state_info(s)
                    for s in pool[beam_width:]
                ],
            })

        sinks = [
            node for node in component
            if not local_children[node]
        ]

        finals = sorted(
            [
                state
                for sink in sinks
                for state in beams.get(sink, ())
            ],
            key=_state_order,
        )[:beam_width]

        all_final.extend(finals)

        component_logs.append({
            "nodes": list(component),
            "parents": {
                str(node): local_parents[node]
                for node in component
            },
            "children": {
                str(node): local_children[node]
                for node in component
            },
            "node_traces": node_logs,
            "final_qa_ids": [
                [c.qa_uid for c in state.steps]
                for state in finals
            ],
        })

    evidence = _union_evidence(all_final)

    return RICRResult(
        components=tuple(tuple(c) for c in components),
        chains=tuple(all_final),
        evidence=evidence,
        issued_queries=tuple(issued_queries),
        fallback=False,
        trace={
            "fallback": False,
            "components": component_logs,
            "evidence_qa_ids": [
                c.qa_uid for c in evidence
            ],
        },
    )


def run_linear_ricr(
    decomposed_questions: Sequence[Mapping[str, object]],
    retrieve_and_score: Callable[[str, int], Sequence[Candidate]],
    *,
    beam_width: int = 5,
    candidates_per_hop: int = 15,
) -> list[ChainState]:

    first_question = next(
        (
            str(row.get("question", ""))
            for row in decomposed_questions
            if row.get("requires_retrieval", True)
        ),
        "",
    )

    result = run_panini_ricr(
        decomposed_questions,
        retrieve_and_score,
        original_question=first_question,
        beam_width=beam_width,
        candidates_per_hop=candidates_per_hop,
    )

    return list(result.chains)