from django.conf.urls import url

from ModelsApi import views

urlpatterns = [
    # network model manager
    url(r'^modelManagement/', views.ModelManagement.as_view(), name='modelManagement'),
    # model checkpoint manager
    url(r'^checkpointManagement/', views.CheckpointManagement.as_view(), name='checkpointManagement'),
    # model class name file manager
    url(r'^classnameManagement/', views.ClassnameManagement.as_view(), name='classnameManagement'),

    # detect object
    url(r'^detectionManagement/', views.DetectionManagement.as_view(), name='detectionManagement')
]
