from django.urls import path
from myapp import views


urlpatterns = [
    path("", views.index, name="index"),  
    path("mail", views.mail_checker, name="mail_checker"),
    path("predict", views.predict, name="predict"), 
    path("mail_check", views.predict_mail, name="mail_check"), 
]