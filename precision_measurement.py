import json
import time
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = 'arxiv_papers'
K_LIMIT = 10
QUERIES_FILE = 'dataset/queries_embeddings.json'


@dataclass
class Measurement:
    """Represents a single query measurement."""
    results: set[str | int]
    time: float


@dataclass
class Averages:
    """Represents the average of multiple query measurements."""
    precision: float
    ann_time: float
    knn_time: float


def query(client: QdrantClient, embedding: list[float], exact: bool = False) -> Measurement:
    """Performs a query using the Qdrant client and measures its execution time."""
    start_time = time.time()
    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        limit=K_LIMIT,
        search_params=models.SearchParams(exact=True) if exact else None,
    ).points
    query_time = time.time() - start_time
    result_set = {point.id for point in result}
    return Measurement(result_set, query_time)


def measure_precision(client: QdrantClient, test_dataset: dict[str, list[float]]) -> Averages:
    """Measures ANN query precision compared to exact k-NN queries across the test dataset."""
    precision_scores = []
    ann_times = []
    knn_times = []

    for _, embedding in test_dataset.items():
        ann = query(client, embedding)
        knn = query(client, embedding, exact=True)

        # Calculate precision@k: intersection between ANN and exact k-NN results
        intersection = ann.results.intersection(knn.results)
        precision_scores.append(len(intersection) / K_LIMIT)

        ann_times.append(ann.time)
        knn_times.append(knn.time)

    avg_precision = sum(precision_scores) / len(precision_scores)
    avg_ann_time = sum(ann_times) / len(ann_times)
    avg_knn_time = sum(knn_times) / len(knn_times)

    return Averages(avg_precision, avg_ann_time, avg_knn_time)


def result_formatting(averages: Averages) -> None:
    """Formats and prints the calculated averages."""
    print(f'Average precision@{K_LIMIT}: {averages.precision:.4f}')
    print(f'Average ANN query time: {averages.ann_time * 1000:.2f} ms')
    print(f'Average exact k-NN query time: {averages.knn_time * 1000:.2f} ms')


def main() -> None:
    """Main execution point for measuring precision."""
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    with open(QUERIES_FILE, 'r', encoding='utf-8') as file:
        test_dataset = json.load(file)

    averages = measure_precision(client, test_dataset)
    result_formatting(averages)


if __name__ == '__main__':
    main()
