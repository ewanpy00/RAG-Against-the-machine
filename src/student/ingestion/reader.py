from pathlib import Path
from student.models import FileRecord

class Reader:
    def __init__(self, repo_path: Path = Path("data/raw/vllm-0.10.1")):
        self.repo_path = repo_path

    def read(self) -> list[FileRecord]:
        if not self.repo_path.exists():
            raise FileNotFoundError(f"Repository path {self.repo_path} does not exist")

        records = []
        for f in self.repo_path.rglob("*"):
            if f.is_file():
                records.append(FileRecord(
                    filepath=str(f.relative_to(self.repo_path)),
                    content=f.read_text(encoding="utf-8", errors="ignore"),
                    filetype=f.suffix
                ))
        return records
