import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from datetime import datetime
from bson import ObjectId


class MongoService:
    def __init__(self):
        self.mongo_uri = os.environ.get('MONGO_URI', 'mongodb+srv://Rinae-User:RamadiRR@cluster1.yzmbbvs.mongodb.net/agriscan?appName=Cluster1')
        self.client = None
        self.db = None
        self._connect()

    def _connect(self):
        try:
            self.client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
            self.client.admin.command('ping')
            self.db = self.client.get_default_database()
            print(f"[MongoDB] Connected successfully to database: {self.db.name}")
        except ConnectionFailure as e:
            print(f"[MongoDB] Connection failed: {e}")
            self.client = None
            self.db = None

    def get_collection(self, collection_name):
        if self.db is None:
            self._connect()
        if self.db is None:
            return None
        return self.db[collection_name]

    def insert_scan(self, scan_data):
        collection = self.get_collection('scans')
        if collection is None:
            return None
        scan_data['timestamp'] = datetime.utcnow()
        result = collection.insert_one(scan_data)
        return str(result.inserted_id)

    def find_scans(self, limit=50, skip=0, disease_filter=None):
        collection = self.get_collection('scans')
        if collection is None:
            return []
        query = {}
        if disease_filter:
            query['disease'] = {'$regex': disease_filter, '$options': 'i'}
        cursor = collection.find(query).sort('timestamp', -1).skip(skip).limit(limit)
        return [{k: v for k, v in doc.items() if k != '_id'} for doc in cursor]

    def count_scans(self, disease_filter=None):
        collection = self.get_collection('scans')
        if collection is None:
            return 0
        query = {}
        if disease_filter:
            query['disease'] = {'$regex': disease_filter, '$options': 'i'}
        return collection.count_documents(query)

    def find_scan_by_id(self, scan_id):
        collection = self.get_collection('scans')
        if collection is None:
            return None
        try:
            doc = collection.find_one({'_id': ObjectId(scan_id)})
            if doc:
                doc['id'] = str(doc.pop('_id'))
            return doc
        except Exception:
            return None

    def delete_scan(self, scan_id):
        collection = self.get_collection('scans')
        if collection is None:
            return False
        try:
            result = collection.delete_one({'_id': ObjectId(scan_id)})
            return result.deleted_count > 0
        except Exception:
            return False

    def create_user(self, user_data):
        collection = self.get_collection('users')
        if collection is None:
            return None
        user_data['created_at'] = datetime.utcnow()
        user_data['updated_at'] = datetime.utcnow()
        result = collection.insert_one(user_data)
        return str(result.inserted_id)

    def find_user_by_email(self, email):
        collection = self.get_collection('users')
        if collection is None:
            return None
        doc = collection.find_one({'email': email})
        if doc:
            doc['id'] = str(doc.pop('_id'))
        return doc

    def find_user_by_id(self, user_id):
        collection = self.get_collection('users')
        if collection is None:
            return None
        try:
            doc = collection.find_one({'_id': ObjectId(user_id)})
            if doc:
                doc['id'] = str(doc.pop('_id'))
            return doc
        except Exception:
            return None

    def find_users(self, role=None, limit=100, skip=0):
        collection = self.get_collection('users')
        if collection is None:
            return []
        query = {}
        if role:
            query['role'] = role
        cursor = collection.find(query).sort('created_at', -1).skip(skip).limit(limit)
        return [{k: v for k, v in doc.items() if k != '_id'} for doc in cursor]

    def update_user(self, user_id, update_data):
        collection = self.get_collection('users')
        if collection is None:
            return False
        try:
            update_data['updated_at'] = datetime.utcnow()
            result = collection.update_one({'_id': ObjectId(user_id)}, {'$set': update_data})
            return result.modified_count > 0
        except Exception:
            return False

    def delete_user(self, user_id):
        collection = self.get_collection('users')
        if collection is None:
            return False
        try:
            result = collection.delete_one({'_id': ObjectId(user_id)})
            return result.deleted_count > 0
        except Exception:
            return False

    def count_users(self, role=None):
        collection = self.get_collection('users')
        if collection is None:
            return 0
        query = {}
        if role:
            query['role'] = role
        return collection.count_documents(query)

    def insert_log(self, log_data):
        collection = self.get_collection('logs')
        if collection is None:
            return None
        log_data['timestamp'] = datetime.utcnow()
        result = collection.insert_one(log_data)
        return str(result.inserted_id)

    def find_logs(self, limit=100, skip=0, action_filter=None):
        collection = self.get_collection('logs')
        if collection is None:
            return []
        query = {}
        if action_filter:
            query['action'] = {'$regex': action_filter, '$options': 'i'}
        cursor = collection.find(query).sort('timestamp', -1).skip(skip).limit(limit)
        return [{k: v for k, v in doc.items() if k != '_id'} for doc in cursor]

    def insert_disease(self, disease_data):
        collection = self.get_collection('diseases')
        if collection is None:
            return None
        disease_data['created_at'] = datetime.utcnow()
        disease_data['updated_at'] = datetime.utcnow()
        result = collection.insert_one(disease_data)
        return str(result.inserted_id)

    def find_diseases(self, limit=100, skip=0):
        collection = self.get_collection('diseases')
        if collection is None:
            return []
        cursor = collection.find().sort('name', 1).skip(skip).limit(limit)
        return [{k: v for k, v in doc.items() if k != '_id'} for doc in cursor]

    def update_disease(self, disease_id, update_data):
        collection = self.get_collection('diseases')
        if collection is None:
            return False
        try:
            update_data['updated_at'] = datetime.utcnow()
            result = collection.update_one({'_id': ObjectId(disease_id)}, {'$set': update_data})
            return result.modified_count > 0
        except Exception:
            return False

    def delete_disease(self, disease_id):
        collection = self.get_collection('diseases')
        if collection is None:
            return False
        try:
            result = collection.delete_one({'_id': ObjectId(disease_id)})
            return result.deleted_count > 0
        except Exception:
            return False

    def get_analytics_summary(self):
        scans_collection = self.get_collection('scans')
        users_collection = self.get_collection('users')
        logs_collection = self.get_collection('logs')

        total_scans = scans_collection.count_documents({}) if scans_collection else 0
        total_users = users_collection.count_documents({}) if users_collection else 0
        total_logs = logs_collection.count_documents({}) if logs_collection else 0

        recent_scans = self.find_scans(limit=10)
        recent_users = self.find_users(limit=10)
        recent_logs = self.find_logs(limit=10)

        disease_counts = {}
        if scans_collection:
            pipeline = [
                {'$group': {'_id': '$disease', 'count': {'$sum': 1}}},
                {'$sort': {'count': -1}},
                {'$limit': 10}
            ]
            for doc in scans_collection.aggregate(pipeline):
                disease_counts[doc['_id']] = doc['count']

        return {
            'total_scans': total_scans,
            'total_users': total_users,
            'total_logs': total_logs,
            'recent_scans': recent_scans,
            'recent_users': recent_users,
            'recent_logs': recent_logs,
            'disease_counts': disease_counts
        }
