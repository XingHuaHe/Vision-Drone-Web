import datetime
import json
import time

from django.http import JsonResponse
from django.shortcuts import render

# Create your views here.
from django.views import View
from django.views.decorators.csrf import csrf_exempt

from ModelsApi.models import Checkpoint, NetModel
from ModelsApi.views_constant import HTTP_OK, HTTP_NET_UPLOAD_ERROR, HTTP_NET_GET_ERROR, HTTP_NET_DELETED_ERROR


class ModelManagement(View):
    """model management include model upload, download and deleted"""

    def __init__(self):
        super(ModelManagement, self).__init__()
        self.data = {
            'msg': 'upload success',
            'status': HTTP_OK,
        }

    def post(self, request):
        """model(.cfg) upload"""
        file = request.FILES.get('model')
        name = request.POST.get('name')
        if file is None:
            self.data['msg'] = 'file is None'
            self.data['status'] = HTTP_NET_UPLOAD_ERROR
            return JsonResponse(data=self.data, status=HTTP_NET_UPLOAD_ERROR)
        if name is None and file.name is not None:
            name = file.name.split('.')[0]

        net = NetModel()
        net.nm_number = time.ctime()
        net.nm_name = name
        net.nm_time = datetime.datetime.now()
        net.nm_path = file
        net.save()
        return JsonResponse(data=self.data, status=HTTP_OK)

    def get(self, request):
        """get model or model list"""
        nm_number = request.GET.get("nm_number")
        if nm_number is None:
            pass
        else:
            try:
                item = NetModel.objects.all().filter(nm_number=nm_number).first()
                # dicts = [{"nm_number": item.nm_number, "name": item.nm_name} for item in items]
                if item is None:
                    self.data['msg'] = 'file is not exist'
                    self.data['status'] = HTTP_NET_GET_ERROR
                    return JsonResponse(data={}, status=HTTP_NET_GET_ERROR)
                data = {"nm_number": item.nm_number, "name": item.nm_name}
                return JsonResponse(data=data, status=HTTP_OK)
            except ValueError as e:
                self.data['msg'] = 'file is not exist'
                self.data['status'] = HTTP_NET_GET_ERROR
                return JsonResponse(data={}, status=HTTP_NET_GET_ERROR)

    def delete(self, request):
        """deleted network model config file"""
        nm_number = request.DELETE.get("nm_number")

        item = NetModel.objects.all().filter(nm_number=nm_number)
        # item = items.firts()
        if item is None:
            self.data['msg'] = 'file is not exist'
            self.data['status'] = HTTP_NET_DELETED_ERROR
            return JsonResponse(data={}, status=HTTP_NET_DELETED_ERROR)
        NetModel.objects.all().delete(nm_number=nm_number)
        self.data['msg'] = 'deleted success'
        self.data['status'] = HTTP_OK
        return JsonResponse(data=self.data, status=HTTP_OK)


class CheckpointManagement(View):
    def post(self, request):
        nm_number = request.POST.get("nm_number")
        file = request.FILES.get('checkpoint')

        checkpoint = Checkpoint()
        checkpoint.ck_number = time.ctime()
        checkpoint.ck_time = datetime.datetime.now()
        checkpoint.ck_path = file
        checkpoint.ck_model = NetModel.objects.all().filter(nm_number=nm_number).first()
        checkpoint.save()

        data = {'msg': 'upload checkpoint success', 'status': HTTP_OK}
        return JsonResponse(data=data, status=HTTP_OK)

    def get(self, request):
        pass

    def delete(self, request):
        pass


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
