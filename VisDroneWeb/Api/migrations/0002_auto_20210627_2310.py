# -*- coding: utf-8 -*-
# Generated by Django 1.11.7 on 2021-06-27 23:10
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Api', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='checkpoint',
            name='ck_number',
            field=models.CharField(max_length=64, unique=True),
        ),
    ]