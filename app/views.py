from django.shortcuts import render, redirect
from django.http import JsonResponse
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import string as s
from nltk.corpus import stopwords
import nltk
import pandas as pd
from django.contrib.auth import authenticate
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.password_validation import validate_password
from .models import *
from django.contrib.auth.models import User
from django.contrib import messages

lemmatizer = nltk.stem.WordNetLemmatizer()

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=6)
# Load your trained model
model.load_weights('tf_model.h5')


def index(request):
    return render(request, 'index.html')

def home(request):
    return render(request, 'home.html')

def login_view(request):
    return render(request, 'login.html')
def signup(request):
    return render(request, 'signup.html')

def form(request):
    return render(request, 'form.html')

def demo(request):
    return render(request, 'demo.html')

def handle_login(request):
    if request.method == 'POST':
        username = request.POST["username"]
        print("username ", username)
        password = request.POST["password"]
        print("password ", password)
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect("home")
        else:
            request.session["message"] = "Sai mật khẩu hoặc sai tên đăng nhập!!"
            context = {"message": "Sai mật khẩu hoặc sai tên đăng nhập!!"}
            return render(request, "login.html", context)
    
    redirect('login')
        
def handle_logout(request):
    logout(request)
    return redirect('home') 
        
def handle_signup(request):
    if request.method == 'POST':
        username = request.POST['new-username']
        firstname = request.POST['new-firstname']
        lastname = request.POST['new-lastname']
        address = request.POST['new-address']
        email = request.POST['new-email']
        password = request.POST['new-password']
        
        if User.objects.filter(email=email).exists():
            context = {'message': 'Email đã đăng ký !!.'}
            messages.error(request, 'Có lỗi xảy ra khi đăng ký tài khoản.')
            return redirect('signup')
        if User.objects.filter(username=username).exists():
            context = {'message': 'username đã trùng !!.'}
            messages.error(request, 'Có lỗi xảy ra khi đăng ký tài khoản.')
            return redirect('signup')
        
        try:
            validate_password(password)
            print("Mật khẩu hợp lệ")
        except Exception as e:
            context = {'message': 'mật khẩu không hợp lệ!'}
            messages.error(request, 'Có lỗi xảy ra khi đăng ký tài khoản.')
            return redirect('signup')
        user = User.objects.create_user(username, email, password)
        user.first_name = firstname
        user.last_name = lastname
        user.is_active = True
        myuser = Users(user=user, address=address)
        myuser.save()
        messages.success(request, 'Đăng ký tài khoản thành công!!')
        return redirect('login')  
  
def handle_view_post(request):
    if request.user.is_authenticated:
        try:
            post = Post.objects.filter(user=request.user)
        except Post.DoesNotExist:
            post = None
        context = {'post': post}
        return render(request, 'post.html', context)
    return render(request, 'post.html')

def handle_view_add_post(request):
    return render(request, 'form-add-post.html')   

def handle_post(request):
    if request.method == 'POST':
        # Lấy dữ liệu từ yêu cầu POST
        abstract = request.POST.get('abstract')
        title = request.POST.get('title')
        # Xử lý dữ liệu và lấy kết quả
        result = title + abstract
        result = data_preprocessing(result)
        print(result)
        # # Tokenize and encode the text
        inputs = tokenizer(
            result,
            max_length=100,
            padding='max_length',
            truncation=True,
            return_tensors='tf'
        )
        # Make predictions
        outputs = model(inputs)
        logits = outputs.logits

        # Convert logits to probabilities
        probabilities = tf.nn.sigmoid(logits)
        label_category = ['biology', 'chemistry', 'computer_science', 'mathematics', 'physics', 'economics']
        # Get the predicted label
        array = (probabilities > 0.42).numpy().astype(int)

        flattened_array = [item for sublist in array for item in sublist]
        tmp = 0
        # In ra chỉ mục của các vị trí chứa số 1
        for index, value in enumerate(flattened_array):
            if value == 1:
                tmp = index
                print("Index:", index)
                break

        label_category = ['biology', 'chemistry', 'computer_science', 'mathematics', 'physics', 'economics']
        print("Predicted Labels:", label_category[index])

        category = Category.objects.filter(name=label_category[index]).first()
        
        post = Post()
        user = request.user
        post.user = user
        post.title = title
        post.abstract = abstract
        post.label = category
        post.save()
        return redirect('post')
    return redirect('add-post')
        
def predict(request):
    if request.method == 'POST':
        # Lấy dữ liệu từ yêu cầu POST
        abstract = request.POST.get('abstract')
        title = request.POST.get('title')
        # Xử lý dữ liệu và lấy kết quả
        result = title + abstract
        result = data_preprocessing(result)
        print(result)
        # # Tokenize and encode the text
        inputs = tokenizer(
            result,
            max_length=100,
            padding='max_length',
            truncation=True,
            return_tensors='tf'
        )
        # Make predictions
        outputs = model(inputs)
        logits = outputs.logits

        # Convert logits to probabilities
        probabilities = tf.nn.sigmoid(logits)
        label_category = ['biology', 'chemistry', 'computer_science', 'mathematics', 'physics', 'economics']
        # Get the predicted label
        array = (probabilities > 0.42).numpy().astype(int)

        flattened_array = [item for sublist in array for item in sublist]
        tmp = 0
        # In ra chỉ mục của các vị trí chứa số 1
        for index, value in enumerate(flattened_array):
            if value == 1:
                tmp = index
                print("Index:", index)
                break

        label_category = ['biology', 'chemistry', 'computer_science', 'mathematics', 'physics', 'economics']
        print("Predicted Labels:", label_category[index])

        return JsonResponse({'result': label_category[index]})

def tokenization(text):
    lst = text.split()
    return lst

def lowercasing(lst):
    new_lst = []
    for i in lst:
        i = i.lower()
        new_lst.append(i)
    return new_lst

def remove_punctuations(lst):
    new_lst = []
    for i in lst:
        for j in s.punctuation:
            i = i.replace(j, '')
        new_lst.append(i)
    return new_lst

def remove_numbers(lst):
    nodig_lst = []
    new_lst = []

    for i in lst:
        for j in s.digits:
            i = i.replace(j, '')
        nodig_lst.append(i)
    for i in nodig_lst:
        if i != '':
            new_lst.append(i)
    return new_lst

def remove_stopwords(lst):
    stop = stopwords.words('english')
    new_lst = []
    for i in lst:
        if i not in stop:
            new_lst.append(i)
    return new_lst

def lemmatzation(lst):
    new_lst = []
    for i in lst:
        i = lemmatizer.lemmatize(i)
        new_lst.append(i)
    return new_lst

def data_preprocessing(lst):
    lst = tokenization(lst)
    lst = lowercasing(lst)
    lst = remove_punctuations(lst)
    lst = remove_numbers(lst)
    lst = remove_stopwords(lst)
    lst = lemmatzation(lst)
    lst = ''.join(i + ' ' for i in lst)
    return lst
