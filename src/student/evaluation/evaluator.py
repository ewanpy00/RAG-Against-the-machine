from student.models import StudentSearchResults, RagDataset, MinimalSource

from student.models import StudentSearchResults

class Evaluator:
    """Calculates Recall@K metrics for search results."""
    
    def __init__(self, iou_threshold: float = 0.05):
        self.iou_threshold = iou_threshold
    
    def evaluate(
        self,
        student_results: StudentSearchResults,
        ground_truth: RagDataset,
        ks: list[int] = [1, 3, 5, 10],
    ) -> dict[int, float]:
        """Returns {k: recall_at_k} for each k."""
        pass
    
    @staticmethod
    def calculate_iou(
        range1: tuple[int, int],
        range2: tuple[int, int],
    ) -> float:
        """IoU for two character ranges."""
        pass
    
    def is_source_found(
        self,
        correct_source: MinimalSource,
        retrieved_sources: list[MinimalSource],
    ) -> bool:
        """Check if any retrieved matches correct (file + IoU)."""
        pass