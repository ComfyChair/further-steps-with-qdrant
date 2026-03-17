import json
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import SearchParams

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
class GroundTruth:
    """Represents the result of a k-NN query."""
    ground_truth: dict[str, set[str]]
    avg_time: float


@dataclass
class Averages:
    """Represents the average of multiple query measurements."""
    precision: float
    avg_time: float


def query(client: QdrantClient, embedding: list[float], k: int, search_param: SearchParams) -> Measurement:
    """Performs a query using the Qdrant client and measures its execution time."""
    start_time = time.time()
    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        limit=k,
        search_params=search_param,
    ).points
    query_time = time.time() - start_time
    result_set = {point.id for point in result}
    return Measurement(result_set, query_time)


def measure_precision(client: QdrantClient,
                      test_dataset: dict[str, list[float]],
                      ground_truth: dict[str, set[str | int]],
                      k: int,
                      ef: int = None,
                      rescore: bool = True
                      ) -> Averages:
    """Measures ANN query precision compared to exact k-NN queries across the test dataset."""
    precision_scores = []
    ann_times = []

    search_param = models.SearchParams(
        quantization=models.QuantizationSearchParams(
            rescore=rescore,
            oversampling=2.0,
        ),
        hnsw_ef=ef)

    for key, embedding in test_dataset.items():
        ann = query(client, embedding, k=k, search_param=search_param)

        # Calculate precision@k: intersection between ANN and exact k-NN results
        intersection = ann.results.intersection(ground_truth[key])
        precision_scores.append(len(intersection) / k)

        ann_times.append(ann.time)

    avg_precision = sum(precision_scores) / len(precision_scores)
    avg_ann_time = sum(ann_times) / len(ann_times)

    return Averages(avg_precision, avg_ann_time)


def get_ground_truth(client: QdrantClient, test_dataset: dict[str, list[float]], k: int) -> GroundTruth:
    """Calculates ground truth for test dataset using exact search."""
    search_param = models.SearchParams(exact=True, quantization=models.QuantizationSearchParams(ignore=True))
    measurements = [query(client, embedding, k=k, search_param=search_param) for _, embedding in test_dataset.items()]
    ground_truth = {key: result for key, result in zip(test_dataset.keys(), [m.results for m in measurements])}
    knn_times = [m.time for m in measurements]
    knn_avg = sum(knn_times) / len(knn_times)
    return GroundTruth(ground_truth, knn_avg)


def evaluate_hnsw_ef(k: int, hnsw_ef_values: list[int], test_dataset: dict[str, list[float]]) -> list[dict]:
    """Evaluates the performance of HNSW with different EF values."""
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    knn_results = get_ground_truth(client, test_dataset, k)
    ground_truth = knn_results.ground_truth
    print(f"Average exact query time: {knn_results.avg_time * 1000:.2f} ms")

    results = []
    for ef in hnsw_ef_values:
        average = measure_precision(client, test_dataset, ground_truth, k, ef)
        results.append({ "hnsw_ef": ef,
                         "avg_precision": average.precision,
                         "avg_query_time_ms": average.avg_time * 1000})
    return results


def plot_results(results: list[dict]):
    """Plots precision and query time vs. hnsw_ef value."""
    ef_values = [r['hnsw_ef'] for r in results]
    precisions = [r['avg_precision'] for r in results]
    query_times = [r['avg_query_time_ms'] for r in results]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('hnsw_ef')
    ax1.set_ylabel('Average Precision', color=color)
    ax1.plot(ef_values, precisions, marker='o', color=color, label='Precision')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Average Query Time (ms)', color=color)
    ax2.plot(ef_values, query_times, marker='s', color=color, label='Query Time')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('HNSW EF Evaluation: Precision and Query Time')
    fig.tight_layout()
    plt.savefig('hnsw_ef_evaluation.png')
    print("Plot saved to 'hnsw_ef_evaluation.png'")


def precision_at_K(test_dataset: dict[str, list[float]], k: int, rescore: bool = True) -> None:
    """ Calculates the average precision at k for a given dataset. """
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    knn = get_ground_truth(client, test_dataset, k)

    ann_evaluation = measure_precision(client, test_dataset, knn.ground_truth, k, rescore=rescore)
    print(f'Average precision@{k}: {ann_evaluation.precision:.4f}')
    print(f'Average ANN query time: {ann_evaluation.avg_time * 1000:.2f} ms')
    print(f'Average exact k-NN query time: {knn.avg_time * 1000:.2f} ms')


def main():
    """Main execution point."""
    with open(QUERIES_FILE, 'r', encoding='utf-8') as file:
        test_dataset = json.load(file)

    precision_at_K(test_dataset, K_LIMIT)
    # hnsw_ef_values = [10, 20, 50, 100, 200]
    # results = evaluate_hnsw_ef(K_LIMIT, hnsw_ef_values, test_dataset)
    # print(json.dumps(results, indent=4))
    # plot_results(results)

if __name__ == '__main__':
    main()
