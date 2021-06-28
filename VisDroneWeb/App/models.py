from django.db import models


class VisDroneUser(models.Model):
    u_username = models.CharField(max_length=32, unique=True)
    u_password = models.CharField(max_length=256)
    u_email = models.CharField(max_length=64, unique=True)
    u_icon = models.ImageField(upload_to='icons/%Y/%m/%d/')
    is_active = models.BooleanField(default=False)
    is_delete = models.BooleanField(default=False)

    class Meta:
        db_table = 'visDrone_user'


class VisDroneTeamMember(models.Model):
    t_member = models.CharField(max_length=32)
    t_email = models.CharField(max_length=64, unique=True)
    t_phone = models.CharField(max_length=11)
    t_icon = models.FileField(upload_to='memberIcons/%Y/%m/%d/')

    class Meta:
        db_table = 'visDrone_teamMember'
