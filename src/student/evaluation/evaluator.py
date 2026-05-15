from student.models import MinimalSource, RagDataset, StudentSearchResults


class Evaluator:
    """Computes recall@k by comparing student results against ground truth."""

    def __init__(self, iou_threshold: float = 0.05) -> None:
        self.iou_threshold = iou_threshold

    def evaluate(
        self,
        student_results: StudentSearchResults,
        ground_truth: RagDataset,
        ks: list[int] = [1, 3, 5, 10],
    ) -> dict[int, float]:
        """Return recall@k for each k in ks."""
        recall_at_k: dict[int, float] = {k: 0.0 for k in ks}
        total_questions = len(ground_truth.rag_questions)

        retrieved_by_id = {
            r.question_id: r.retrieved_sources
            for r in student_results.search_results
        }

        for question in ground_truth.rag_questions:
            retrieved = retrieved_by_id.get(question.question_id, [])

            for k in ks:
                top_k = retrieved[:k]
                found = sum(
                    1 for src in question.sources
                    if self.is_source_found(src, top_k)
                )
                recall_at_k[k] += found / len(question.sources) if question.sources else 0.0

        return {k: v / total_questions for k, v in recall_at_k.items()}

    @staticmethod
    def calculate_iou(
        range1: tuple[int, int],
        range2: tuple[int, int],
    ) -> float:
        """Compute intersection-over-union of two character ranges."""
        start1, end1 = range1
        start2, end2 = range2

        intersection_start = max(start1, start2)
        intersection_end = min(end1, end2)

        if intersection_end <= intersection_start:
            return 0.0

        intersection_length = intersection_end - intersection_start
        union_length = (end1 - start1) + (end2 - start2) - intersection_length
        return intersection_length / union_length

    def is_source_found(
        self,
        correct_source: MinimalSource,
        retrieved_sources: list[MinimalSource],
    ) -> bool:
        """Return True if any retrieved source matches the correct source path."""
        for retrieved in retrieved_sources:
            if correct_source.file_path.endswith(retrieved.file_path):
                return True
        return False
