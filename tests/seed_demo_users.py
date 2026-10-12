#!/usr/bin/env python3
"""Check MongoDB users and seed demo users if needed."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.mongo_service import MongoService
from datetime import datetime
import bcrypt

mongo = MongoService()
if mongo.db is not None:
    print(f"DB connected: {mongo.db.name}")
    users_coll = mongo.db['users']
    count = users_coll.count_documents({})
    print(f"Total users in DB: {count}")

    existing = users_coll.find_one({'email': 'user@agri.com'})
    if existing:
        pw_prefix = existing.get('password', '')[:10] if existing.get('password') else 'None'
        print(f"Demo user exists: user@agri.com, password prefix: {pw_prefix}")
    else:
        print("Demo user not found, creating...")
        pw_hash = bcrypt.hashpw(b'password123', bcrypt.gensalt(rounds=12)).decode('utf-8')
        admin_hash = bcrypt.hashpw(b'admin123', bcrypt.gensalt(rounds=12)).decode('utf-8')
        users_coll.insert_many([
            {'firstName': 'Demo', 'lastName': 'User', 'email': 'user@agri.com',
             'password': pw_hash, 'phone': '0000000000', 'location': 'Johannesburg',
             'farmSize': '', 'role': 'user', 'created_at': datetime.utcnow(),
             'updated_at': datetime.utcnow()},
            {'firstName': 'Admin', 'lastName': 'User', 'email': 'admin@agri.com',
             'password': admin_hash, 'phone': '0000000000', 'location': 'Johannesburg',
             'farmSize': '', 'role': 'admin', 'created_at': datetime.utcnow(),
             'updated_at': datetime.utcnow()},
        ])
        print("Demo users created successfully")
else:
    print("No DB connection")
