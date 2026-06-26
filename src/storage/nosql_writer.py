from typing import List, Dict
from pymongo import MongoClient

def upsert_customer_segments(docs: List[Dict], mongo_uri: str, db_name: str, collection: str = "customer_segments"):
    """
    Upsert dokumen segmentasi ke MongoDB Atlas.
    docs: list of dict, wajib punya key '_id' (CustomerID)
    """
    try:
        # Gunakan timeout agar tidak hang jika koneksi internet terputus
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        col = client[db_name][collection]

        # Test koneksi sebelum update
        client.admin.command('ping')

        for d in docs:
            col.update_one({"_id": d["_id"]}, {"$set": d}, upsert=True)

        client.close()
        print(f"Sukses mengupdate {len(docs)} data segmentasi ke MongoDB Atlas.")
    except Exception as e:
        print(f"\n⚠️ GAGAL MENYIMPAN KE MONGODB ATLAS (Network/DB Error) ⚠️")
        print(f"Pesan Error: {e}")
        print("Pastikan koneksi internet aktif dan IP Anda ter-whitelist di Atlas.")