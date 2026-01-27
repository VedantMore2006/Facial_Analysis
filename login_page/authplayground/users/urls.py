from django.urls import path
from users.views import signup, login, signup_page, login_page
from users.views import (
    counselor_list,
    client_list,
    counselor_detail,
    client_detail,
)

urlpatterns = [
    path("signup/", signup, name="signup"),
    path("login/", login, name="login"),

    path("signup-page/", signup_page, name="signup_page"),
    path("login-page/", login_page, name="login_page"),
]

urlpatterns += [
    path("counselors/", counselor_list, name="counselor_list"),
    path("clients/", client_list, name="client_list"),

    path("counselor/<str:counselor_id>/", counselor_detail, name="counselor_detail"),
    path("client/<str:client_id>/", client_detail, name="client_detail"),
]
