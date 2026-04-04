class SetupConfig:
    collection_name = "nietzsche_rag"

    vector_size = 384

    hnsw_m = 32
    hnsw_ef_construct = 200

    chunk_size = 160
    chunk_overlap = 35


setup_config = SetupConfig()
