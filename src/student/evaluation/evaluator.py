from student.models import RagDataset, MinimalSource, StudentSearchResults

class Evaluator:
    def __init__(self, iou_threshold: float = 0.05):
        self.iou_threshold = iou_threshold

    def evaluate(
        self,
        student_results: StudentSearchResults,
        ground_truth: RagDataset,
        ks: list[int] = [1, 3, 5, 10],
    ) -> dict[int, float]:
        recall_at_k = {k: 0.0 for k in ks}
        total_questions = len(ground_truth.rag_questions)
        
        for question in ground_truth.rag_questions:
            student_result = next(
                (res for res in student_results.search_results if res.question_id == question.question_id),
                None
            )
            if not student_result:
                continue
            
            retrieved_sources = student_result.retrieved_sources
            
            for k in ks:
                top_k_sources = retrieved_sources[:k]
                found_count = sum(
                    1 for src in question.sources
                    if self.is_source_found(src, top_k_sources)
                )
                # print(found_count)
                
                # Recall для этого вопроса = доля найденных
                recall_for_question = found_count / len(question.sources)
                recall_at_k[k] += recall_for_question
        
        recall_at_k = {k: recall / total_questions for k, recall in recall_at_k.items()}
        return recall_at_k

    @staticmethod
    def calculate_iou(
        range1: tuple[int, int],
        range2: tuple[int, int],
    ) -> float:

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
        for retrieved in retrieved_sources:
            if correct_source.file_path.endswith(retrieved.file_path):
                iou = self.calculate_iou(
                (correct_source.first_character_index, correct_source.last_character_index),
                (retrieved.first_character_index, retrieved.last_character_index),
            )
                if iou >= self.iou_threshold:
                    return True
        return False