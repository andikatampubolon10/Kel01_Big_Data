from typing import List, Dict
from pymongo import MongoClient

def upsert_customer_segments(docs: List[Dict], mongo_uri: str, db_name: str, collection: str = "customer_segments"):
    """
    Upsert dokumen segmentasi ke MongoDB Atlas.
    docs: list of dict, wajib punya key '_id' (CustomerID)
    """
    client = MongoClient(mongo_uri)
    col = client[db_name][collection]

    for d in docs:
        col.update_one({"_id": d["_id"]}, {"$set": d}, upsert=True)

    client.close()