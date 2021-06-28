import time

from django.http import JsonResponse
from django.shortcuts import render

# Create your views here.
from django.views.decorators.csrf import csrf_exempt

from Api.models import Checkpoint


@csrf_exempt
def instructions_upload(request):
    data = {
        'msg': 'ok',
        'status': '401',
    }
    if request.method == 'POST':
        file = request.FILES.get('checkpoint')

        checkpoint = Checkpoint()
        checkpoint.ck_number = time.ctime()
        checkpoint.ck_path = file
        checkpoint.save()

        data['status'] = '200'
    return JsonResponse(data=data)


def instructions_delete(request):
    return None


def instructions_runs(request):
    return None
