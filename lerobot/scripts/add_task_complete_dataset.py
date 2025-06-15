import os
import glob
import pyarrow.parquet as pq
import pyarrow as pa

def find_task_complete_boundary(values, threshold=2000, n_consecutive=3):
    """Finds the boundary index using n_consecutive big jumps"""
    count = 0
    found_big = False
    boundary_idx = len(values)
    for i in range(len(values) - 2, -1, -1):
        delta = values[i] - values[i + 1]
        if delta > threshold:
            count += 1
            if count == n_consecutive:
                found_big = True
        else:
            if found_big:
                boundary_idx = i + n_consecutive  # the flip occurs at the end of the streak
                break
            count = 0  # reset the count if streak is broken
    return boundary_idx

def process_file(filepath, column="observation.state", threshold=2000, n_consecutive=3):
    table = pq.read_table(filepath, columns=[column])
    # Extract the 7th element (motor position) from each row's observation.state
    positions = [x[6] if x is not None and len(x) > 6 else None for x in table.column(0).to_pylist()]
    # Remove None values if present, or handle as you prefer
    # positions = [p if p is not None else 0 for p in positions]

    boundary_idx = find_task_complete_boundary(positions, threshold, n_consecutive)
    is_task_complete = [1 if idx >= boundary_idx else 0 for idx in range(len(positions))]

    full_table = pq.read_table(filepath)
    full_table = full_table.append_column("is_task_complete", pa.array(is_task_complete))
    pq.write_table(full_table, filepath)
    print(f"Processed {filepath}: flag starts at row {boundary_idx}")


# ---- Main loop ----
directory = "/Users/leonardo/Documents/projects/lerobot_lechef/data/leChef_dataset_complete/data/chunk-000"
parquet_files = glob.glob(os.path.join(directory, "*.parquet"))

for parquet_path in parquet_files:
    process_file(parquet_path)
