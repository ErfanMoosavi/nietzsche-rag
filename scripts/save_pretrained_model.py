from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
model.save_pretrained("all-MiniLM-L6-v2-local")
