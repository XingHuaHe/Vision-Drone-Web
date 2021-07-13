import datetime
import json
import os
import time

from django.http import JsonResponse
from django.shortcuts import render

# Create your views here.
from django.views import View
from django.views.decorators.csrf import csrf_exempt

from ModelsApi.models import Checkpoint, NetModel
from ModelsApi.views_constant import HTTP_OK, HTTP_CFG_DELETED_ERROR, \
    HTTP_CFG_UPLOAD_ERROR, HTTP_CFG_NOT_EXIST, HTTP_CFG_GET_ERROR, HTTP_CP_UPLOAD_ERROR, HTTP_CP_GET_ERROR, \
    HTTP_CP_NOT_EXIST
from VisDroneWeb.settings import IMG_UPLOAD
from utils import detect


class ModelManagement(View):
    """model config file management include model upload, download and deleted"""

    def __init__(self):
        super(ModelManagement, self).__init__()
        self.data = {
            'msg': 'success',
            'status': HTTP_OK,
        }

    @csrf_exempt
    def post(self, request):
        """model(.cfg) upload"""

        file = request.FILES.get('cfg')
        name = request.POST.get('name')
        if file is None:
            self.data['msg'] = 'file is none'
            self.data['status'] = HTTP_CFG_UPLOAD_ERROR
            return JsonResponse(data=self.data)
        if name is None and file.name is not None:
            name = file.name.split('.')[0]
        else:
            name = time.ctime()

        net = NetModel()
        net.nm_number = time.ctime()
        net.nm_name = name
        net.nm_time = datetime.datetime.now()
        net.nm_path = file
        net.save()

        self.data['msg'] = 'upload success'
        return JsonResponse(data=self.data)

    def get(self, request):
        """get model or model list"""

        nm_number = request.GET.get("id")
        if nm_number is None:
            # obtain all items.
            items = NetModel.objects.all()
            dicts = [{"id": item.nm_number, "name": item.nm_name} for item in items]
            data = {}
            for i in range(len(dicts)):
                data[str(i)] = dicts[i]
            return JsonResponse(data=data)

        else:
            try:
                item = NetModel.objects.all().filter(nm_number=nm_number).first()
                if item is None:
                    self.data['msg'] = 'file is not exist'
                    self.data['status'] = HTTP_CFG_NOT_EXIST
                    return JsonResponse(data={}, status=HTTP_CFG_NOT_EXIST)
                data = {"id": item.nm_number, "name": item.nm_name}
                return JsonResponse(data=data, status=HTTP_OK)
            except ValueError as e:
                self.data['msg'] = 'file is not exist'
                self.data['status'] = HTTP_CFG_GET_ERROR
                return JsonResponse(data={})

    def delete(self, request):
        """deleted network model config file"""

        nm_number = request.DELETE.get("id")

        item = NetModel.objects.all().filter(nm_number=nm_number)
        if item is None:
            self.data['msg'] = 'file is not exist'
            self.data['status'] = HTTP_CFG_DELETED_ERROR
            return JsonResponse(data={})
        NetModel.objects.all().delete(nm_number=nm_number)
        self.data['msg'] = 'deleted success'
        self.data['status'] = HTTP_OK
        return JsonResponse(data=self.data, status=HTTP_OK)


class CheckpointManagement(View):
    """model checkpoint file management include upload, deleted, download"""

    def __init__(self):
        super(CheckpointManagement, self).__init__()
        self.data = {'msg': 'upload checkpoint success', 'status': HTTP_OK}

    def post(self, request):
        """
        model checkpoint upload
        :param request: id: net-model config id
        :param request: checkpoint: model checkpoint file
        :return:
        """

        nm_number = request.POST.get("id")
        file = request.FILES.get('checkpoint')
        if nm_number is None or file is None:
            data = {'msg': 'id number or file are None', 'status': HTTP_CP_UPLOAD_ERROR}
            return JsonResponse(data=data)

        if not NetModel.objects.all().filter(nm_number=nm_number).exists():
            data = {'msg': 'model config is not exist', 'status': HTTP_CFG_NOT_EXIST}
            return JsonResponse(data=data)

        try:
            checkpoint = Checkpoint()
            checkpoint.ck_number = time.ctime()
            checkpoint.ck_time = datetime.datetime.now()
            checkpoint.ck_path = file
            checkpoint.ck_model = NetModel.objects.all().filter(nm_number=nm_number).first()
            checkpoint.save()
        except RuntimeError as e:
            self.data['msg'] = e

        return JsonResponse(data=self.data)

    def get(self, request):
        """
        obtain model checkpoint according net-model config id
        :param request: id: net-model config id
        :return:
        """
        nm_number = request.GET.get('id')
        if nm_number is None:
            self.data['msg'] = 'id number or file are None'
            self.data['status'] = HTTP_CP_GET_ERROR
            return JsonResponse(data=self.data, status=HTTP_CP_GET_ERROR)

        items = Checkpoint.objects.filter(ck_model__nm_number=nm_number)
        lists = [{"id": item.ck_number} for item in items]
        dicts = {}
        for i in range(len(lists)):
            dicts[str(i)] = lists[i]

        return JsonResponse(data=dicts, status=HTTP_OK)

    def delete(self, request):
        """
        delete checkpoint file
        :param request: id: checkpoint id
        :return:
        """
        ck_number = request.DELETE.get('id')
        if ck_number is None:
            self.data['msg'] = 'id number or file are None'
            self.data['status'] = HTTP_CP_GET_ERROR
            return JsonResponse(data=self.data, status=HTTP_CP_GET_ERROR)

        items = Checkpoint.objects.all().filter(ck_number=ck_number)
        if not items.exists():
            self.data['msg'] = 'checkpoint is not exist'
            self.data['status'] = HTTP_CP_NOT_EXIST
            return JsonResponse(data=self.data, status=HTTP_CP_NOT_EXIST)

        Checkpoint.objects.all().delete(ck_number=ck_number)

        self.data['msg'] = 'checkpoint was deleted'
        self.data['status'] = HTTP_OK
        return JsonResponse(data=self.data, status=HTTP_OK)


class DetectionManagement(View):
    """images detection management include detect(s)"""

    def __init__(self):
        super(DetectionManagement, self).__init__()
        self.data = {'msg': 'error', 'status': HTTP_OK}

    def post(self, request):
        """
        upload a image and detected objects
        :param request: image: detected image
        :return:
        """

        img = request.FILES.get('image')
        cfg = request.POST.get('cfg')
        weight = request.POST.get('weight')
        names = request.POST.get('names')

        if img is None or cfg is None or weight is None or names is None:
            self.data['msg'] = 'detected relative config files are None'
            self.data['results'] = {}
            return JsonResponse(data=self.data)

        img_name = img.name

        # ext = os.path.splitext(name)[-1]

        img_path = os.path.join(IMG_UPLOAD, img_name)

        with open(img_path, 'ab') as fp:
            for chunk in img.chunks():
                fp.write(chunk)

        content = detect.detect(cfg, weight, names, img_path, 0)
        self.data['results'] = content

        return JsonResponse(data=self.data, status=HTTP_OK)

    def get(self, request):
        pass


class ClassnameManagement(View):
    """ detected classname """
    def __init__(self):
        super(ClassnameManagement, self).__init__()
        self.data = {'msg': 'error', 'status': HTTP_OK}

    def post(self, request):
        cn_number = request.POST.get("id")
        names_file = request.FILES.get('classname')

        if names_file is None:
            self.data['msg'] = 'class name file upload error'
            return JsonResponse(data=self.data)
