# -*- coding: utf-8 -*-
# Generated by Django 1.11.7 on 2021-07-04 15:48
from __future__ import unicode_literals

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Checkpoint',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('ck_number', models.CharField(max_length=64, unique=True)),
                ('ck_time', models.DateTimeField(verbose_name='YYYY-MM-DD HH:MM:ss')),
                ('ck_path', models.FileField(upload_to='checkpoint')),
            ],
            options={
                'db_table': 'VisDrone_Checkpoint',
            },
        ),
        migrations.CreateModel(
            name='NetModel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('nm_number', models.CharField(max_length=64, unique=True)),
                ('nm_name', models.CharField(max_length=64)),
                ('nm_time', models.DateTimeField(verbose_name='YYYY-MM-DD HH:MM:ss')),
                ('nm_path', models.FileField(upload_to='nets')),
            ],
            options={
                'db_table': 'VisDrone_NetModel',
            },
        ),
        migrations.AddField(
            model_name='checkpoint',
            name='ck_model',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='ModelsApi.NetModel'),
        ),
    ]