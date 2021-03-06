from django.db import models


class NetModel(models.Model):
    """model config model"""

    nm_number = models.CharField(max_length=64, unique=True)
    nm_name = models.CharField(max_length=64)
    nm_time = models.DateTimeField(verbose_name='YYYY-MM-DD HH:MM:ss')
    nm_path = models.FileField(upload_to='nets')

    class Meta:
        db_table = 'VisDrone_NetModel'


class Checkpoint(models.Model):
    """model checkpoint model"""

    ck_number = models.CharField(max_length=64, unique=True)
    ck_time = models.DateTimeField(verbose_name='YYYY-MM-DD HH:MM:ss')
    ck_path = models.FileField(upload_to='checkpoint')
    ck_model = models.ForeignKey(NetModel)

    class Meta:
        db_table = 'VisDrone_Checkpoint'


class ClassName(models.Model):
    """detected class names. 'cn' represented class name"""

    cn_number = models.CharField(max_length=64, unique=True)
    cn_name = models.CharField(max_length=64)
    ck_path = models.FileField(upload_to='classname', default='classname')
    ck_time = models.DateTimeField(verbose_name='YYYY-MM-DD HH:MM:ss')
    cn_model = models.ForeignKey(NetModel)

    class Meta:
        db_table = 'VisDrone_ClassName'
