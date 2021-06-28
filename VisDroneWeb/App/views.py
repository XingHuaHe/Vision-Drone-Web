from django.contrib.auth.hashers import check_password, make_password
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.urls import reverse

from App.models import VisDroneUser, VisDroneTeamMember
from App.views_constant import COMPANY_NAME, COMPANY_NAME_E, CONTACT_PHONE, CONTACT_EMAIL, ADDRESS, COUNTRY, PROVINCE, \
    HTTP_OK, HTTP_USER_EXIST


def index(request):
    user_id = request.session.get('user_id')
    data = {
        "title": "VisDrone Index",
        "COMPANY_NAME": COMPANY_NAME,
        "COMPANY_NAME_E": COMPANY_NAME_E,
        "CONTACT_PHONE": CONTACT_PHONE,
        "CONTACT_EMAIL": CONTACT_EMAIL,
        "COUNTRY": COUNTRY,
        "PROVINCE": PROVINCE,
        "ADDRESS": ADDRESS,
        "is_login": False,
    }
    if user_id:
        user = VisDroneUser.objects.filter(id=user_id).first()
        username = user.u_username
        data["is_login"] = True
        data["username"] = username
    return render(request, 'main/index.html', context=data)


def about(request):
    user_id = request.session.get('user_id')
    data = {
        "title": "VisDrone About",
        "COMPANY_NAME": COMPANY_NAME,
        "COMPANY_NAME_E": COMPANY_NAME_E,
        "CONTACT_PHONE": CONTACT_PHONE,
        "CONTACT_EMAIL": CONTACT_EMAIL,
        "COUNTRY": COUNTRY,
        "PROVINCE": PROVINCE,
        "ADDRESS": ADDRESS,
        "is_login": False,
    }
    if user_id:
        user = VisDroneUser.objects.filter(id=user_id).first()
        username = user.u_username
        data["is_login"] = True
        data["username"] = username
    return render(request, 'main/about.html', context=data)


def contact(request):
    user_id = request.session.get('user_id')
    data = {
        "title": "Contact",
        "COMPANY_NAME": COMPANY_NAME,
        "COMPANY_NAME_E": COMPANY_NAME_E,
        "CONTACT_PHONE": CONTACT_PHONE,
        "CONTACT_EMAIL": CONTACT_EMAIL,
        "COUNTRY": COUNTRY,
        "PROVINCE": PROVINCE,
        "ADDRESS": ADDRESS,
        "is_login": False,
    }
    if user_id:
        user = VisDroneUser.objects.filter(id=user_id).first()
        username = user.u_username
        data["is_login"] = True
        data["username"] = username
    return render(request, 'main/contact.html', context=data)


def services(request):
    user_id = request.session.get('user_id')

    teammembers = VisDroneTeamMember.objects.all()
    members = []
    for item in teammembers:
        members.append(item.t_member)
    data = {
        "title": "Services",
        "COMPANY_NAME": COMPANY_NAME,
        "COMPANY_NAME_E": COMPANY_NAME_E,
        "CONTACT_PHONE": CONTACT_PHONE,
        "CONTACT_EMAIL": CONTACT_EMAIL,
        "COUNTRY": COUNTRY,
        "PROVINCE": PROVINCE,
        "ADDRESS": ADDRESS,
        "is_login": False,
        'members': members,
    }
    if user_id:
        user = VisDroneUser.objects.filter(id=user_id).first()
        username = user.u_username
        data["is_login"] = True
        data["username"] = username
    return render(request, 'main/services.html', context=data)


def portfolio(request):
    user_id = request.session.get('user_id')
    data = {
        "title": "Portfolio",
        "COMPANY_NAME": COMPANY_NAME,
        "COMPANY_NAME_E": COMPANY_NAME_E,
        "CONTACT_PHONE": CONTACT_PHONE,
        "CONTACT_EMAIL": CONTACT_EMAIL,
        "COUNTRY": COUNTRY,
        "PROVINCE": PROVINCE,
        "ADDRESS": ADDRESS,
        "is_login": False,
    }
    if user_id:
        user = VisDroneUser.objects.filter(id=user_id).first()
        username = user.u_username
        data["is_login"] = True
        data["username"] = username
    return render(request, 'main/portfolio.html', context=data)


def recruit(request):
    user_id = request.session.get('user_id')
    data = {
        "title": "VisDrone Recruit",
        "COMPANY_NAME": COMPANY_NAME,
        "COMPANY_NAME_E": COMPANY_NAME_E,
        "CONTACT_PHONE": CONTACT_PHONE,
        "CONTACT_EMAIL": CONTACT_EMAIL,
        "COUNTRY": COUNTRY,
        "PROVINCE": PROVINCE,
        "ADDRESS": ADDRESS,
        "is_login": False,
    }
    if user_id:
        user = VisDroneUser.objects.filter(id=user_id).first()
        username = user.u_username
        data["is_login"] = True
        data["username"] = username
    return render(request, 'main/recruit.html', context=data)


def maps(request):
    return render(request, 'main/map.html')


def register(request):
    if request.method == "GET":
        data = {
            "title": "Register",
        }
        return render(request, 'user/register.html', context=data)
    elif request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        icon = request.FILES.get('icon')

        password = make_password(password)

        user = VisDroneUser()
        user.u_username = username
        user.u_email = email
        user.u_password = password
        user.u_icon = icon
        user.save()

        return redirect(reverse('app:login'))


def check_user(request):
    if request.method == 'GET':
        username = request.GET.get('username')

        users = VisDroneUser.objects.filter(u_username=username)
        data = {
            "status": HTTP_OK,
        }
        if users.exists():
            data['status'] = HTTP_USER_EXIST
        return JsonResponse(data)


def login(request):
    if request.method == "GET":
        error_message = request.session.get('error_message')
        data = {
            "title": "Login",
        }
        if error_message:
            del request.session['error_message']
            data['error_message'] = error_message
        return render(request, 'user/login.html', context=data)
    elif request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')

        users = VisDroneUser.objects.filter(u_username=username)
        if not users.exists():
            data = {
                "title": "Login",
                "error_message": "用户不存在或用户账号密码错误",
            }
            return render(request, 'user/login.html', context=data)
        user = users.first()
        if check_password(password, user.u_password):
            request.session['user_id'] = user.id
            data = {
                "title": "Index",
                "COMPANY_NAME": COMPANY_NAME,
                "COMPANY_NAME_E": COMPANY_NAME_E,
                "CONTACT_PHONE": CONTACT_PHONE,
                "CONTACT_EMAIL": CONTACT_EMAIL,
                "COUNTRY": COUNTRY,
                "PROVINCE": PROVINCE,
                "ADDRESS": ADDRESS,
                "is_login": True,
                "username": username,
            }
            return render(request, 'main/index.html', context=data)
        else:
            if request.session.get('user_id'):
                del request.session['user_id']
            data = {
                "title": "Login",
                "error_message": "用户账号或密码错误",
            }
            return render(request, 'user/login.html', context=data)


def reset_pwd(request):
    data = {
        "title": "Reset password",
    }
    return render(request, 'user/reset_pwd.html', context=data)


def logout(request):
    # try:
    #     del request.session['user_id']
    # finally:
    #     return redirect(reverse('app:index'))

    del request.session['user_id']

    return redirect(reverse('app:index'))
