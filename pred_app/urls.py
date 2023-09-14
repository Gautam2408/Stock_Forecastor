from django.urls import path

from . import views

urlpatterns = [

    path('index', views.index, name='index'),
    path('pred', views.pred, name='pred'),
    path('chart',views.chart, name='chart'),
    path('contact', views.contact, name='contact'),
   ]
