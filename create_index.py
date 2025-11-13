from pymongo import MongoClient

# MongoDB connection string for Cosmos DB
connection_string = "mongodb://vectordbmemory-mongo:oWaR0VHkEH7SoiVHN5yhtA4zspRJXmQ8qVKaOSi9XASYdpV9LP9eUPz9owafCZXwn1ONDKAwWCyrACDbuBeTCw==@vectordbmemory-mongo.mongo.cosmos.azure.com:10255/?ssl=true&replicaSet=globaldb&retrywrites=false&maxIdleTimeMS=120000&appName=@vectordbmemory-mongo@"
database_name = "memdb"
collection_name = "memory"

client = MongoClient(
    connection_string,
    ssl=True,
    tlsAllowInvalidCertificates=True,  # Use cautiously, only for testing
    retryWrites=False
)
db = client[database_name]
collection = db[collection_name]

try:
    collection.create_index([("embedding", 1)], name="embedding_index")
    print("Basic index created successfully.")
except Exception as e:
    print(f"Error creating index: {e}")
finally:
    client.close()