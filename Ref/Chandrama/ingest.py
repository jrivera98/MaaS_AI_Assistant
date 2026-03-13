import pandas as pd
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Load Excel
df = pd.read_excel("data/metro_rikshaw_dummy_demand.xlsx")

# Convert wide table → long format
df_long = df.melt(
    id_vars=["train_number", "time"],
    var_name="station",
    value_name="riksha_passengers"
)

# Remove zero demand
df_long = df_long[df_long["riksha_passengers"] > 0]

documents = []

for _, row in df_long.iterrows():

    text = f"""
    Train {row.train_number} at {row.time}
    Station: {row.station}
    Passengers needing riksha: {row.riksha_passengers}
    """

    documents.append(Document(page_content=text))

# Embeddings
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# Vector DB
vectordb = Chroma.from_documents(
    documents,
    embeddings,
    persist_directory="chroma_db"
)

vectordb.persist()

print("Data successfully embedded into vector database.")