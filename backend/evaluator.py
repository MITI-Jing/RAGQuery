def evaluate_rag(question: str, answer: str, contexts: list[str], api_key: str) -> dict:
    try:
        import os
        os.environ["OPENAI_API_KEY"] = api_key

        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "reference": [answer],  # required by newer ragas versions
        })

        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
        scores = result.to_pandas()

        def safe(val):
            v = float(val)
            return round(v, 2) if v == v and v not in (float("inf"), float("-inf")) else None

        return {
            "faithfulness": safe(scores["faithfulness"].iloc[0]),
            "answer_relevancy": safe(scores["answer_relevancy"].iloc[0]),
        }
    except Exception as e:
        return {"error": str(e)}
