from django.conf.urls import url

from Api import views

urlpatterns = [
    url(r'^instructions/upload/', views.instructions_upload, name='instructions_upload'),
    url(r'^instructions/delete/', views.instructions_delete, name='instructions_delete'),
    url(r'^instructions/runs/', views.instructions_runs, name='instructions_runs'),
]
