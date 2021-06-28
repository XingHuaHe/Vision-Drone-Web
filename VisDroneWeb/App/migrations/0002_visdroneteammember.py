# -*- coding: utf-8 -*-
# Generated by Django 1.11.7 on 2021-06-21 00:01
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('App', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='VisDroneTeamMember',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('t_member', models.CharField(max_length=32)),
                ('t_email', models.CharField(max_length=64, unique=True)),
                ('t_phone', models.CharField(max_length=11)),
                ('t_icon', models.FileField(upload_to='memberIcons/%Y/%m/%d/')),
            ],
            options={
                'db_table': 'visDrone_teamMember',
            },
        ),
    ]