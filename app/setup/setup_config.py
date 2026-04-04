class SetupConfig:
    collection_name = "nietzsche_rag"

    vector_size = 384

    hnsw_m = 32
    hnsw_ef_construct = 300

    chunk_size = 150
    chunk_overlap = 35


setup_config = SetupConfig()
