import os
from dotenv import load_dotenv
load_dotenv()

from pymongo import MongoClient

uri = os.environ["MONGO_URI"]
dbn = os.environ["MONGO_DB"]

client = MongoClient(uri)
print("ping:", client.admin.command("ping"))
print("db:", dbn, "collections:", client[dbn].list_collection_names())
client.close()