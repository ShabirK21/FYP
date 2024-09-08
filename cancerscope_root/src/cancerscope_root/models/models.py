from tortoise.models import Model
from tortoise import fields
from passlib.hash import bcrypt
from datetime import datetime, timezone

class User(Model):
    id = fields.IntField(pk=True)
    username = fields.CharField(max_length=64, unique=True)
    email = fields.CharField(max_length=64, unique=True)
    password_hash = fields.CharField(max_length=128)
    images = fields.ReverseRelation["Image"]

    @property
    def auth_id(self):
        return str(self.id)

    def set_password(self, password):
        self.password_hash = bcrypt.hash(password)

    def check_password(self, password):
        return bcrypt.verify(password, self.password_hash)

class Image(Model):
    id = fields.IntField(pk=True)
    path = fields.CharField(max_length=255)
    user = fields.ForeignKeyField('models.User', related_name='images')

class PredictionResult(Model):
    id = fields.IntField(pk=True)
    user = fields.ForeignKeyField('models.User', related_name='prediction_results')
    prediction_value = fields.CharField(max_length=64)
    original_image = fields.CharField(max_length=255)
    result_path = fields.CharField(max_length=255)
    classification = fields.CharField(max_length=64)
    probability = fields.FloatField()
    timestamp = fields.DatetimeField(default=datetime.now(timezone.utc))


