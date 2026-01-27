from django.db import models
## Step 4: Define a MongoDB User document (NOT Django model)
# Create your models here.
# users/models.py
from mongoengine import Document, StringField, ReferenceField, DateTimeField
from datetime import datetime

class User(Document):
    name = StringField(required=True)
    email = StringField(required=True, unique=True)
    password_hash = StringField(required=True)
    created_at = DateTimeField(default=datetime.utcnow)

    meta = {
        "collection": "users"
    }


class Counselor(Document):
    name = StringField(required=True)
    specialization = StringField()
    email = StringField(required=True, unique=True)
    created_at = DateTimeField(default=datetime.utcnow)

    meta = {"collection": "counselors"}


class Client(Document):
    name = StringField(required=True)
    email = StringField(required=True, unique=True)
    counselor = ReferenceField(Counselor, required=True)
    ### This is how MongoDB models relationships without joins. ###
    created_at = DateTimeField(default=datetime.utcnow)

    meta = {"collection": "clients"}
