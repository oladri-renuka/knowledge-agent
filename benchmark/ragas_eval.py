"""Ragas evaluation of the knowledge agent's query outputs.

Measures faithfulness, answer relevance, context precision, and context recall
on a held-out question set including coverage-gap cases.

Usage:
    python -m benchmark.ragas_eval
"""

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

from src.graph_store import KnowledgeGraph
from src.query import query_with_contexts

GRAPH_PATH = os.environ.get("GRAPH_PATH", "results/graph.json")
CHROMA_PATH = os.environ.get("CHROMA_PATH", "results/chroma_db")

EVAL_QUESTIONS = [
    {
        "question": "What is DPO and how does it relate to RLHF?",
        "ground_truth": "DPO (Direct Preference Optimization) is a method that directly optimizes a language model on preference data using a classification loss, eliminating the need for a separate reward model. It relates to RLHF by offering a simpler alternative to the standard PPO-based RLHF pipeline.",
    },
    {
        "question": "Does DPO eliminate the reward model?",
        "ground_truth": "The DPO paper claims it eliminates the need for an explicit reward model. However, a later critique shows DPO still implicitly relies on a reward function via the Bradley-Terry preference model.",
    },
    {
        "question": "What is the Transformer architecture?",
        "ground_truth": "The Transformer is a neural network architecture that relies entirely on self-attention mechanisms, dispensing with recurrence and convolutions. It uses multi-head attention and was originally proposed for machine translation.",
    },
    {
        "question": "How does PPO relate to RLHF?",
        "ground_truth": "PPO (Proximal Policy Optimization) is the standard reinforcement learning algorithm used in the RLHF pipeline for optimizing language models against a reward model trained on human preferences.",
    },
    {
        "question": "What are the limitations of RLHF?",
        "ground_truth": "RLHF limitations include reward hacking, PPO instability and sensitivity to hyperparameters, high computational cost, and the need for maintaining multiple models in memory simultaneously.",
    },
    {
        "question": "Is reinforcement learning suitable for NLP tasks?",
        "ground_truth": "There is a contradiction. The RLHF survey states RL is not suitable for NLP tasks, while the DPO paper states RL is used for fine-tuning language models with no catastrophic forgetting.",
    },
    {
        "question": "What is the capital of France?",
        "ground_truth": "The knowledge graph has no relevant coverage for this question.",
    },
    {
        "question": "What is quantum computing?",
        "ground_truth": "The knowledge graph has no relevant coverage for this question.",
    },
    {
        "question": "What benchmarks are used to evaluate Transformers?",
        "ground_truth": "The Transformer was evaluated on WMT 2014 English-to-German and English-to-French translation benchmarks, achieving a BLEU score of 28.4 on English-to-German.",
    },
    {
        "question": "What is multi-head attention?",
        "ground_truth": "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. The original Transformer uses 8 attention heads.",
    },
    {
        "question": "How does DPO compare to PPO in performance?",
        "ground_truth": "There are conflicting claims. The DPO paper reports DPO achieves a 61% win rate against PPO on TL;DR summarization. A later critique found PPO outperforms DPO at larger scales when properly tuned.",
    },
    {
        "question": "What is reward hacking?",
        "ground_truth": "Reward hacking occurs when agents exploit correlations in the reward model to maximize scores while producing outputs that are not genuinely preferred by humans.",
    },
    {
        "question": "What methods improve upon standard RLHF?",
        "ground_truth": "Methods that improve upon RLHF include DPO, RLAIF, rejection sampling, and KL-divergence regularization.",
    },
    {
        "question": "What is the role of positional encodings in Transformers?",
        "ground_truth": "Positional encodings inject sequence order information into the Transformer because the self-attention mechanism has no inherent notion of position. The original paper uses sinusoidal positional encodings.",
    },
    {
        "question": "What is preference overfitting in DPO?",
        "ground_truth": "Preference overfitting is a phenomenon where DPO learns to exploit spurious correlations in the static preference dataset rather than learning genuinely preferred behavior.",
    },
]


def run_evaluation():
    log.info("Loading knowledge graph from %s", GRAPH_PATH)
    graph = KnowledgeGraph(GRAPH_PATH, CHROMA_PATH)
    log.info("Graph loaded: %s", graph.stats())

    log.info("Running %d evaluation queries...\n", len(EVAL_QUESTIONS))
    results = []
    for item in EVAL_QUESTIONS:
        q = item["question"]
        log.info("Q: %s", q)
        answer, contexts = query_with_contexts(q, graph)
        log.info("A: %s\n", answer[:120] + "..." if len(answer) > 120 else answer)
        results.append({
            "question": q,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item["ground_truth"],
        })

    log.info("Running Ragas evaluation...")
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        eval_llm = ChatOpenAI(
            model=os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        eval_emb = OpenAIEmbeddings(
            model=os.environ.get("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small"),
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

        dataset = Dataset.from_dict({
            "question": [r["question"] for r in results],
            "answer": [r["answer"] for r in results],
            "contexts": [r["contexts"] for r in results],
            "ground_truth": [r["ground_truth"] for r in results],
        })

        scores = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=eval_llm,
            embeddings=eval_emb,
        )

        scores_df = scores.to_pandas()
        output_path = "results/ragas_scores.json"
        scores_df.to_json(output_path, orient="records", indent=2)
        log.info("Ragas scores saved to %s\n", output_path)

        log.info("=== AGGREGATE SCORES ===")
        for col in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            if col in scores_df.columns:
                mean_val = scores_df[col].dropna().mean()
                log.info("  %s: %.3f", col, mean_val)

        log.info("\n=== PER-QUESTION SCORES ===")
        for _, row in scores_df.iterrows():
            q = row.get("question", "?")
            log.info("  Q: %s", q[:60])
            for col in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
                if col in row and row[col] == row[col]:
                    log.info("    %s: %.3f", col, row[col])

    except Exception as e:
        log.error("Ragas evaluation failed: %s", e)
        import traceback
        traceback.print_exc()
        log.info("Saving raw results without Ragas scores...")
        output_path = "results/eval_results.json"
        Path(output_path).write_text(json.dumps(results, indent=2, default=str))
        log.info("Raw results saved to %s", output_path)


if __name__ == "__main__":
    run_evaluation()
