from django.db import models


# Create your models here.
class Checkpoint(models.Model):
    ck_number = models.CharField(max_length=64, unique=True)
    ck_path = models.FileField(upload_to='checkpoint')

    class Meta:
        db_table = 'VisDrone_Checkpoint'
