from django.db import models


class NetModel(models.Model):
    nm_number = models.CharField(max_length=64, unique=True)
    nm_name = models.CharField(max_length=64)
    nm_time = models.DateTimeField(verbose_name='YYYY-MM-DD HH:MM:ss')
    nm_path = models.FileField(upload_to='nets')

    class Meta:
        db_table = 'VisDrone_NetModel'


class Checkpoint(models.Model):
    ck_number = models.CharField(max_length=64, unique=True)
    ck_time = models.DateTimeField(verbose_name='YYYY-MM-DD HH:MM:ss')
    ck_path = models.FileField(upload_to='checkpoint')
    ck_model = models.ForeignKey(NetModel)

    class Meta:
        db_table = 'VisDrone_Checkpoint'
