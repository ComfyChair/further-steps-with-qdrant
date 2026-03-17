import json

from qdrant_client import QdrantClient, models

from precision_measurement import COLLECTION_NAME, QUERIES_FILE, K_LIMIT, \
    precision_at_K, QDRANT_HOST, QDRANT_PORT


def quantize_collection():
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    client.update_collection(
        collection_name=COLLECTION_NAME,
        optimizer_config=models.OptimizersConfigDiff(),
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                quantile=0.99,
                always_ram=False,
            ),
        ),
    )

def main():
    """Main execution point."""
    with open(QUERIES_FILE, 'r', encoding='utf-8') as file:
        test_dataset = json.load(file)

    print("Rescore=false")
    precision_at_K(test_dataset, K_LIMIT, rescore=False)
    print("\nRescore=true")
    precision_at_K(test_dataset, K_LIMIT, rescore=True)

if __name__ == '__main__':
    main()
