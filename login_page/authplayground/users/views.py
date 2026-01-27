from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
from django.views.decorators.http import require_POST
#from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.hashers import make_password, check_password
from django.shortcuts import render
from users.models import User
from users.models import Counselor, Client

def signup_page(request):
    return render(request, "users/signup.html")

def login_page(request):
    return render(request, "users/login.html")

#@csrf_exempt
@require_POST
def signup(request):
    name = request.POST.get("name")
    email = request.POST.get("email")
    password = request.POST.get("password")

    # 1) Basic validation
    if not all([name, email, password]):
        return JsonResponse(
            {"error": "name, email, and password are required"},
            status=400
        )

    # 2) Duplicate email check
    if User.objects(email=email).first():
        return JsonResponse(
            {"error": "Email already exists"},
            status=400
        )

    # 3) Hash password
    password_hash = make_password(password)

    # 4) Save user
    User(
        name=name,
        email=email,
        password_hash=password_hash
    ).save()

    return JsonResponse(
        {"message": "User created successfully"},
        status=201
    )

#@csrf_exempt
@require_POST
def login(request):
    email = request.POST.get("email")
    password = request.POST.get("password")

    if not all([email, password]):
        return JsonResponse(
            {"error": "email and password are required"},
            status=400
        )

    user = User.objects(email=email).first()
    if not user:
        return JsonResponse(
            {"error": "Invalid credentials"},
            status=401
        )

    if not check_password(password, user.password_hash):
        return JsonResponse(
            {"error": "Invalid credentials"},
            status=401
        )

    return JsonResponse(
        {"message": "Login successful"},
        status=200
    )


def counselor_list(request):
    counselors = Counselor.objects.all()

    counselor_data = []
    for c in counselors:
        count = Client.objects(counselor=c).count()
        counselor_data.append({
            "counselor": c,
            "count": count
        })

    return render(request, "users/counselor_list.html", {
        "counselor_data": counselor_data
    })


def client_list(request):
    clients = Client.objects.all()
    data = []
    for cl in clients:
        data.append({
            "id": str(cl.id),
            "name": cl.name,
            "email": cl.email,
            "counselor_id": str(cl.counselor.id) if cl.counselor else None,
        })
    return JsonResponse({"clients": data}, status=200)


def counselor_detail(request, counselor_id):
    counselor = Counselor.objects.with_id(counselor_id)
    if not counselor:
        return JsonResponse({"error": "Counselor not found"}, status=404)
    count = Client.objects(counselor=counselor).count()
    return JsonResponse({
        "id": str(counselor.id),
        "name": counselor.name,
        "email": counselor.email,
        "specialization": counselor.specialization,
        "client_count": count,
    }, status=200)


def client_detail(request, client_id):
    client = Client.objects.with_id(client_id)
    if not client:
        return JsonResponse({"error": "Client not found"}, status=404)
    return JsonResponse({
        "id": str(client.id),
        "name": client.name,
        "email": client.email,
        "counselor_id": str(client.counselor.id) if client.counselor else None,
    }, status=200)


def client_list(request):
    clients = Client.objects.select_related()

    return render(request, "users/client_list.html", {
        "clients": clients
    })
