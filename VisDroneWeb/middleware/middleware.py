from django.utils.deprecation import MiddlewareMixin

REQUIRE_LOGIN = [
    '/app/index/',
]


class LoginMiddleware(MiddlewareMixin):
    def process_request(self, request):
        print(request.path)
