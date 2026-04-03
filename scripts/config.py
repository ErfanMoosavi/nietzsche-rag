class Config:
    COLLECTION_NAME = "nietzsche_rag"

    BOOKS = [
        "thus_spoke_zarathustra",
        "genealogy_of_morals",
        "twilight_of_the_idols",
        "beyond_good_and_evil",
        "gay_science",
        "ecce_homo",
        "birth_of_tragedy",
    ]

    VECTOR_SIZE = 384

    HNSW_M = 64
    HNSW_EF_CONSTRUCT = 500

    CHUNK_SIZE = 160
    CHUNK_OVERLAP = 40


config = Config()
