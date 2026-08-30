import math
import pytest

from panini_course.ricr import (
    Candidate,
    harmonic_mean,
    identify_retrieval_components,
    instantiate_question,
    panini_chain_score,
    run_panini_ricr,
)


def test_component_identification():
    plan = [
        {"question": "Who directed Film A?", "requires_retrieval": True},
        {"question": "Who directed Film B?", "requires_retrieval": True},
        {
            "question": "Who was born later, <ENTITY_Q1> or <ENTITY_Q2>?",
            "requires_retrieval": True,
        },
    ]
    assert identify_retrieval_components(plan) == [[1, 2, 3]]


def test_question_substitution():
    result = instantiate_question(
        "Who was born later, <ENTITY_Q1> or <ENTITY_Q2>?",
        {1: "Alice", 2: "Bob"},
    )
    assert result == "Who was born later, Alice or Bob?"


def test_harmonic_mean():
    value = harmonic_mean([0.9, 0.6])
    assert abs(value - 0.72) < 1e-9


def test_geometric_chain_score():
    c1 = Candidate(
        qa_uid="a",
        answer_names=("Alice",),
        score=0.9,
    )
    c2 = Candidate(
        qa_uid="b",
        answer_names=("Bob",),
        score=0.6,
    )
    expected = math.sqrt(0.95 * 0.80)
    assert abs(panini_chain_score([c1, c2]) - expected) < 1e-9


def test_converging_dag():
    plan = [
        {"question": "Who directed Film A?", "requires_retrieval": True},
        {"question": "Who directed Film B?", "requires_retrieval": True},
        {
            "question": "Who was born later, <ENTITY_Q1> or <ENTITY_Q2>?",
            "requires_retrieval": True,
        },
    ]

    data = {
        "Who directed Film A?": [
            Candidate(
                qa_uid="qa_A",
                answer_names=("Alice",),
                answer_ids=("docA::gswA::e1",),
                score=0.9,
            )
        ],
        "Who directed Film B?": [
            Candidate(
                qa_uid="qa_B",
                answer_names=("Bob",),
                answer_ids=("docB::gswB::e1",),
                score=0.8,
            )
        ],
        "Who was born later, Alice or Bob?": [
            Candidate(
                qa_uid="qa_FINAL",
                answer_names=("Alice",),
                answer_ids=("docF::gswF::e1",),
                score=0.95,
            )
        ],
    }

    calls = []

    def retrieve(query, k):
        calls.append(query)
        return data.get(query, [])[:k]

    result = run_panini_ricr(
        plan,
        retrieve,
        original_question="unused",
        beam_width=2,
        candidates_per_hop=2,
    )

    assert result.fallback is False
    assert result.components == ((1, 2, 3),)
    assert "Who was born later, Alice or Bob?" in result.issued_queries
    assert result.chains
    assert any(
        candidate.qa_uid == "qa_FINAL"
        for candidate in result.chains[0].steps
    )


def test_intermediate_entity_grouping():
    plan = [
        {"question": "Root question", "requires_retrieval": True},
        {
            "question": "Next question about <ENTITY_Q1>",
            "requires_retrieval": True,
        },
        {
            "question": "Final question about <ENTITY_Q2>",
            "requires_retrieval": True,
        },
    ]

    data = {
        "Root question": [
            Candidate(
                qa_uid="root",
                answer_names=("Alice",),
                answer_ids=("doc::gsw::e1",),
                score=0.9,
            )
        ],
        "Next question about Alice": [
            Candidate(
                qa_uid="same_entity_a",
                answer_names=("Carol",),
                answer_ids=("doc2::gsw::e3",),
                score=0.9,
            ),
            Candidate(
                qa_uid="same_entity_b",
                answer_names=("Carol",),
                answer_ids=("doc3::gsw::e7",),
                score=0.5,
            ),
        ],
        "Final question about Carol": [
            Candidate(
                qa_uid="final",
                answer_names=("Carol",),
                answer_ids=("doc4::gsw::e9",),
                score=0.8,
            )
        ],
    }

    def retrieve(query, k):
        return data.get(query, [])[:k]

    result = run_panini_ricr(
        plan,
        retrieve,
        original_question="unused",
        beam_width=2,
        candidates_per_hop=2,
    )

    assert result.chains
    assert result.chains[0].steps[1].qa_uid == "same_entity_a"


def test_singleton_fallback():
    plan = [
        {"question": "Internal rewrite", "requires_retrieval": True},
    ]

    calls = []

    def retrieve(query, k):
        calls.append(query)
        return [
            Candidate(
                qa_uid="fallback",
                answer_names=("Answer",),
                answer_ids=("doc::gsw::e1",),
                score=0.8,
            )
        ]

    result = run_panini_ricr(
        plan,
        retrieve,
        original_question="Original user question",
        beam_width=2,
        candidates_per_hop=2,
    )

    assert result.fallback is True
    assert result.issued_queries == ("Original user question",)


def test_namespaced_ids():
    plan = [
        {"question": "Who is X?", "requires_retrieval": True},
        {
            "question": "Who knows <ENTITY_Q1>?",
            "requires_retrieval": True,
        },
    ]

    data = {
        "Who is X?": [
            Candidate(
                qa_uid="qa1",
                answer_names=("Same Name",),
                answer_ids=("doc1::gsw1::e1",),
                score=0.9,
            )
        ],
        "Who knows Same Name?": [
            Candidate(
                qa_uid="qa2",
                answer_names=("Done",),
                score=0.8,
            )
        ],
    }

    def retrieve(query, k):
        return data.get(query, [])

    result = run_panini_ricr(
        plan,
        retrieve,
        original_question="unused",
        beam_width=2,
        candidates_per_hop=2,
    )

    assert result.chains
    assert "doc1::gsw1::e1" in result.chains[0].steps[0].answer_ids


def test_final_qa_level_selection_preserves_shared_answer():
    plan = [
        {"question": "Root", "requires_retrieval": True},
        {
            "question": "Final about <ENTITY_Q1>",
            "requires_retrieval": True,
        },
    ]

    data = {
        "Root": [
            Candidate(
                qa_uid="root",
                answer_names=("Alice",),
                answer_ids=("docA::gswA::e1",),
                score=0.9,
            )
        ],
        "Final about Alice": [
            Candidate(
                qa_uid="final_1",
                answer_names=("Same Answer",),
                answer_ids=("docF::gswF::e1",),
                score=0.9,
            ),
            Candidate(
                qa_uid="final_2",
                answer_names=("Same Answer",),
                answer_ids=("docF::gswF::e2",),
                score=0.8,
            ),
        ],
    }

    def retrieve(query, k):
        return data.get(query, [])[:k]

    result = run_panini_ricr(
        plan,
        retrieve,
        original_question="unused",
        beam_width=2,
        candidates_per_hop=2,
    )

    assert result.chains

    final_ids = {
        step.qa_uid
        for chain in result.chains
        for step in chain.steps
    }

    assert "final_1" in final_ids
    assert "final_2" in final_ids

def test_second_best_final_beam_contributes_evidence():
  plan = [
      {
          "question": "Find A",
          "requires_retrieval": True,
      },
      {
          "question": "Find B",
          "requires_retrieval": True,
      },
      {
          "question": "Compare <ENTITY_Q1> <ENTITY_Q2>",
          "requires_retrieval": True,
      },
  ]

  data = {
      "Find A": [
          Candidate(
              qa_uid="root_A",
              answer_names=("Alice",),
              answer_ids=("docA::e1",),
              score=0.95,
          ),
          Candidate(
              qa_uid="root_A2",
              answer_names=("Sofia",),
              answer_ids=("docA::e2",),
              score=0.90,
          ),
      ],
      "Find B": [
          Candidate(
              qa_uid="root_B",
              answer_names=("Noah",),
              answer_ids=("docB::e1",),
              score=0.95,
          ),
      ],
      "Compare Alice Noah": [
          Candidate(
              qa_uid="final_best",
              answer_names=("Answer A",),
              answer_ids=("docF::e1",),
              score=0.95,
          ),
      ],
      "Compare Sofia Noah": [
          Candidate(
              qa_uid="final_second",
              answer_names=("Answer B",),
              answer_ids=("docF::e2",),
              score=0.90,
          ),
      ],
  }

  def retrieve(query, k):
      return data.get(query, [])[:k]

  result = run_panini_ricr(
      plan,
      retrieve,
      original_question="unused",
      beam_width=2,
      candidates_per_hop=2,
      multi_dependency_threshold=0.3,
  )

  assert result.chains
  assert "Compare Alice Noah" in result.issued_queries
  assert "Compare Sofia Noah" in result.issued_queries

  evidence_ids = {
      candidate.qa_uid
      for candidate in result.evidence
  }

  assert "final_best" in evidence_ids
  assert "final_second" in evidence_ids
def test_harmonic_mean_threshold_fallback():
    from panini_course.ricr import ChainState, _select_parent_products

    a1 = Candidate(
        qa_uid="a1",
        answer_names=("Alice",),
        answer_ids=("docA::e1",),
        score=0.9,
    )
    a2 = Candidate(
        qa_uid="a2",
        answer_names=("Sofia",),
        answer_ids=("docA::e2",),
        score=0.8,
    )
    b1 = Candidate(
        qa_uid="b1",
        answer_names=("Noah",),
        answer_ids=("docB::e1",),
        score=0.9,
    )

    state_a1 = ChainState(
        steps=(a1,),
        answers_by_step={1: "Alice"},
        score=panini_chain_score([a1]),
        last_hop_score=a1.score,
    )
    state_a2 = ChainState(
        steps=(a2,),
        answers_by_step={1: "Sofia"},
        score=panini_chain_score([a2]),
        last_hop_score=a2.score,
    )
    state_b1 = ChainState(
        steps=(b1,),
        answers_by_step={2: "Noah"},
        score=panini_chain_score([b1]),
        last_hop_score=b1.score,
    )

    beams = {
        1: [state_a1, state_a2],
        2: [state_b1],
    }

    chosen, trace = _select_parent_products(
        [1, 2],
        beams,
        width=2,
        threshold=0.99,
    )

    assert len(chosen) == 1
    assert chosen[0].answers_by_step == {
        1: "Alice",
        2: "Noah",
    }

    assert trace
    assert trace[0]["selected"] is True