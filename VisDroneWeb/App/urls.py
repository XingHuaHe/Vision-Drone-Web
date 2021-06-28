from django.conf.urls import url

from App import views

urlpatterns = [
    # main page api.
    url(r'^index/', views.index, name='index'),
    url(r'^about/', views.about, name='about'),
    url(r'^contact/', views.contact, name='contact'),
    url(r'^services/', views.services, name='services'),
    url(r'^portfolio/', views.portfolio, name='portfolio'),
    url(r'^recruit/', views.recruit, name='recruit'),
    url(r'^map/', views.maps, name='map'),
    # user api.
    url(r'^register/', views.register, name='register'),
    url(r'^checkuser/', views.check_user, name='check_user'),
    url(r'^login/', views.login, name='login'),
    url(r'^resetpwd/', views.reset_pwd, name='reset_pwd'),
    url(r'^logout/', views.logout, name='logout'),
]
