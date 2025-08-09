from pathlib import Path

current_path = Path(__file__).resolve()

project_path = current_path.parents[2]

print(project_path)  # out: /data/lzy/MISS/code/tabular