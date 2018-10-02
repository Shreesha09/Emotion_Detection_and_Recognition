from django.shortcuts import render, get_object_or_404
from .models import UserInput
import sys

#sys.path.append('frontend/ML')
from frontend.emotion_analyzer import emo_analyze

# Create your views here.
def index(request):
	return render(request, 'frontend/index.html')

def result(request):	
	user_input = request.POST.get('message')
	probabilities = emo_analyze(user_input)
	#print('Text is: ' + user_input)
	#print(probabilities)
	#labels =
	values = probabilities.values.tolist()[0]
	args = {'text': user_input, 'anger': values[0]*100, 'disgust': values[1]*100, 'fear': values[2]*100, 'guilt': values[3]*100, 'joy': values[4]*100, 'sadness':values[5]*100, 'shame': values[6]*100}
	#print(args)
	return render(request,'frontend/result.html', args)
