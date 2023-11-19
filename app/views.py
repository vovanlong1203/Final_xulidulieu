from django.shortcuts import render, redirect
from django.http import JsonResponse
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification


model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=6)
# Load your trained model
model.load_weights('tf_model.h5')

def index(request):
    return render(request,'index.html')


def predict(request):
    if request.method == 'POST':
        # Lấy dữ liệu từ yêu cầu POST
        input_data = request.POST.get('input_data')

        # Xử lý dữ liệu và lấy kết quả
        result = (input_data)
        
            # # Tokenize and encode the text
        inputs = tokenizer(
            input_data,
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
        label_category = ['biology','chemistry','computer_science','mathematics','physics','economics']
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

        label_category = ['biology','chemistry','computer_science','mathematics','physics','economics']
        print("Predicted Labels:", label_category[index])

        # Trả về kết quả dưới dạng JSON
        return JsonResponse({'result': label_category[index]})