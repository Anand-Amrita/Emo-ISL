from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login, logout
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import os
import spacy
import torch
import numpy as np
import librosa
import speech_recognition as sr
import pickle
import tensorflow as tf
from transformers import BertTokenizer, BertForSequenceClassification
from tensorflow.python.keras.models import model_from_json,Sequential
from tensorflow.keras.layers import BatchNormalization
from django.contrib.staticfiles import finders
from django.contrib.auth.decorators import login_required
from pathlib import Path


# Load NLP components for grammar conversion
nlp = spacy.load("en_core_web_sm")
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load speech emotion recognition model
json_file = open('A2SL/models/CNN_model.json', 'r')
loaded_model_json = json_file.read()
speech_emo_model = model_from_json(loaded_model_json,custom_objects={'BatchNormalization': BatchNormalization})
speech_emo_model.load_weights("A2SL/models/speech_emo.h5")

with open('A2SL/models/scaler2.pickle', 'rb') as f:
    scaler2 = pickle.load(f)
with open('A2SL/models/encoder2.pickle', 'rb') as f:
    encoder2 = pickle.load(f)

# Load text emotion recognition model
model_path = 'A2SL/models/tokenizer/'  # Path to the tokenizer folder
model_file = 'A2SL/models/fineTuneModel.pt'  # Fine-tuned model file
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)
text_emo_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)
# Load the model state
text_emo_model.load_state_dict(torch.load(model_file,map_location=torch.device('cpu')))
text_emo_model.eval()  # Set the model to evaluation mode

print("All models loaded successfully")



# Feature extraction functions for speech emotion recognition
def zcr(data, frame_length, hop_length):
    zcr = librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr)

def rmse(data, frame_length=2048, hop_length=512):
    rmse = librosa.feature.rms(data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse)

def mfcc(data, sr, frame_length=2048, hop_length=512, flatten: bool=True):
    mfcc = librosa.feature.mfcc(data, sr=sr)
    return np.squeeze(mfcc.T) if not flatten else np.ravel(mfcc.T)

def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    result = np.array([])
    result = np.hstack((result,
                      zcr(data, frame_length, hop_length),
                      rmse(data, frame_length, hop_length),
                      mfcc(data, sr, frame_length, hop_length)
                     ))
    return result

def get_predict_feat(path):
    d, s_rate = librosa.load(path, duration=2.5, offset=0.6)
    res = extract_features(d)
    result = np.array(res)
    result = np.reshape(result, newshape=(1, 2376))
    i_result = scaler2.transform(result)
    final_result = np.expand_dims(i_result, axis=2) 
    return final_result

def predict_speech_emotion(path):
    try:
        res = get_predict_feat(path)
        predictions = speech_emo_model.predict(res)
        y_pred = encoder2.inverse_transform(predictions)
        emotion = y_pred[0][0]
        
        # Map emotion according to requirements
        # 1:'Neutral', 2:'Calm', 3:'Happy', 4:'Sad', 5:'Angry', 6:'Fear', 7:'Disgust', 8:'Surprise'
        emotion_mapping = {
            'Neutral': 'neutral',
            'Calm': 'neutral',  # Map calm to neutral
            'Happy': 'happy',
            'Sad': 'sad',
            'Angry': 'anger',   # Map angry to anger
            'Fear': 'neutral',  # Map fear to neutral
            'Disgust': 'neutral', # Map disgust to neutral
            'Surprise': 'surprise'
        }
        
        # Get mapped emotion or default to neutral if not in mapping
        mapped_emotion = emotion_mapping.get(emotion, 'neutral')
        return mapped_emotion
        
    except Exception as e:
        print(f"Error in speech emotion prediction: {e}")
        return "neutral"  # Default to neutral on error

# Function to predict emotion from text
def predict_text_emotion(text):
    try:
        # Tokenize the input text
        inputs = tokenizer.encode(text, add_special_tokens=True, max_length=256, padding='max_length', truncation=True, return_tensors='pt')

        # Create attention mask
        attention_mask = (inputs != 0).long()  # 1 for real tokens, 0 for padding

        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        text_emo_model.to(device)
        inputs = inputs.to(device)
        attention_mask = attention_mask.to(device)

        # Predict
        text_emo_model.eval()
        with torch.no_grad():
            outputs = text_emo_model(inputs, attention_mask=attention_mask)
            logits = outputs.logits

        # Get predicted emotion
        predicted_class = torch.argmax(logits, dim=1).item()

        label_mapping = {
            0: "anger",
            1: "neutral",
            2: "happy",
            3: "love",
            4: "sad",
            5: "surprise"
        }

        return label_mapping.get(predicted_class, "neutral")

    except Exception as e:
        print(f"Error in text emotion prediction: {e}")
        return "neutral"

def convert_to_isl(text, emotion):
    """Convert English sentence to ISL grammar format considering emotion-based video availability"""
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    import nltk
    from django.contrib.staticfiles import finders

    # Tokenize and POS tagging
    words = word_tokenize(text)
    tagged = nltk.pos_tag(words)

    # Tense detection
    tense = {
        "future": len([word for word in tagged if word[1] == "MD"]),
        "present": len([word for word in tagged if word[1] in ["VBP", "VBZ", "VBG"]]),
        "past": len([word for word in tagged if word[1] in ["VBD", "VBN"]]),
        "present_continuous": len([word for word in tagged if word[1] == "VBG"])
    }

    # Custom stopwords
    stop_words = set([
        "mightn't", 're', 'wasn', 'wouldn', 'be', 'has', 'that', 'does', 'shouldn', 'do', "you've", 'off',
        'for', "didn't", 'm', 'ain', 'haven', "weren't", 'are', "she's", "wasn't", 'its', "haven't",
        "wouldn't", 'don', 'weren', 's', "you'd", "don't", 'doesn', "hadn't", 'is', 'was', "that'll",
        "should've", 'a', 'then', 'the', 'mustn', 'i', 'nor', 'as', "it's", "needn't", 'd', 'am', 'have',
        'hasn', 'o', "aren't", "you'll", "couldn't", "you're", "mustn't", 'didn', "doesn't", 'll', 'an',
        'hadn', 'whom', 'y', "hasn't", 'itself', 'couldn', 'needn', "shan't", 'isn', 'been', 'such',
        'shan', "shouldn't", 'aren', 'being', 'were', 'did', 'ma', 't', 'having', 'mightn', 've', "isn't", "won't"
    ])

    # Lemmatization and filtering
    lr = WordNetLemmatizer()
    filtered_text = []
    for w, p in zip(words, tagged):
        if w not in stop_words:
            if p[1] in ['VBG', 'VBD', 'VBZ', 'VBN', 'NN']:
                filtered_text.append(lr.lemmatize(w, pos='v'))
            elif p[1] in ['JJ', 'JJR', 'JJS', 'RBR', 'RBS']:
                filtered_text.append(lr.lemmatize(w, pos='a'))
            else:
                filtered_text.append(lr.lemmatize(w))

    # Replace 'I' with 'Me'
    words = ['Me' if w == 'I' else w for w in filtered_text]

    # Tense-based prefix
    probable_tense = max(tense, key=tense.get)
    if probable_tense == "past" and tense["past"] >= 1:
        words = ["Before"] + words
    elif probable_tense == "future" and tense["future"] >= 1:
        if "Will" not in words:
            words = ["Will"] + words
    elif probable_tense == "present" and tense["present_continuous"] >= 1:
        words = ["Now"] + words

    # Construct ISL words with emotion-based video lookup
    final_words = []
    for w in words:
        # Priority 1: {word}-{emotion}.mp4
        path1 = f"{w.capitalize()}/{w.capitalize()}-{emotion}.mp4"
        found1 = finders.find(path1)
        # Priority 2: {word}-neutral.mp4
        path2 = f"{w.capitalize()}/{w.capitalize()}-neutral.mp4"
        found2 = finders.find(path2)
        
        if found1 or found2:
            print(final_words)
            final_words.append(w.capitalize())
        else:
            # Split into characters and try character-based emotion videos
            for c in w:
                char_path = f"{c.capitalize()}/{c.capitalize()}-{emotion}.mp4"
                if finders.find(char_path):
                    final_words.append(c.capitalize())
    print(final_words)
    ISL_grammar = ' '.join(final_words)
    print(ISL_grammar)
    return ISL_grammar


def find_video_with_emotion(word, emotion):
    """Find the appropriate video file with emotion or fall back to neutral"""
    # Check if word folder exists
    word_path = finders.find(f'{word}')
    
    if word_path:
        # Word folder exists, check for word-emotion video
        video_path = f'{word_path}/{word.capitalize()}-{emotion}.mp4'
        if os.path.exists(video_path):
            return f'{word}/{word.capitalize()}-{emotion}.mp4'
        else:
            # Emotion not found, fall back to neutral
            neutral_path = f'{word_path}/{word.capitalize()}-neutral.mp4'
            if os.path.exists(neutral_path):
                return f'{word}/{word.capitalize()}-neutral.mp4'
    
    # If word folder doesn't exist or no appropriate video found, return None
    return None

def home_view(request):
    return render(request, 'home.html')

def about_view(request):
    return render(request, 'about.html')

def contact_view(request):
    return render(request, 'contact.html')

@login_required(login_url="login")
def animation_view(request):
    if request.method == 'POST':
        # Check if the input is text or speech
        if 'sen' in request.POST:
            # Text input
            text = request.POST.get('sen')
            emotion = predict_text_emotion(text)
            input_type = "text"
        elif 'audio_file' in request.FILES:
            # Speech input
            audio_file = request.FILES['audio_file']
            
            # Save the uploaded audio temporarily
            temp_path = 'temp_audio.wav'
            with open(temp_path, 'wb+') as destination:
                for chunk in audio_file.chunks():
                    destination.write(chunk)

            # Extract emotion from speech
            emotion = predict_speech_emotion(temp_path)

            # Convert speech to text using SpeechRecognition
            recognizer = sr.Recognizer()
            with sr.AudioFile(temp_path) as source:
                audio = recognizer.record(source)
                try:
                    text = recognizer.recognize_google(audio)
                except sr.UnknownValueError:
                    text = ""
                except sr.RequestError as e:
                    print(f"Speech recognition error: {e}")
                    text = ""

            input_type = "speech"
            
            # Clean up temp file
            os.remove(temp_path)

        else:
            return render(request, 'animation.html', {'error': 'No input provided'})
        
        # Convert to ISL grammar
        isl_sentence = convert_to_isl(text,emotion)
        print('Sentence',isl_sentence)
        # Tokenize the ISL sentence
        words = isl_sentence.split()
        print('Words array',words)
        print('Emotion',emotion)
        # Process words for animation
        animation_words = []
        for word in words:
            # Find video with appropriate emotion
            video_path = find_video_with_emotion(word.capitalize(), emotion)
            print('Video path',video_path)
            if video_path:
                # Video exists for this word
                animation_words.append({'word': word, 'path': video_path, 'type': 'word'})
            else:
                # Video doesn't exist, spell it out letter by letter
                for letter in word:
                    letter_video = find_video_with_emotion(letter.capitalize(), emotion)
                    if letter_video:
                        animation_words.append({'word': letter, 'path': letter_video, 'type': 'letter'})
                    else:
                        # If letter with emotion not found, try to find neutral letter
                        neutral_letter = find_video_with_emotion(letter.capitalize(), 'neutral')
                        if neutral_letter:
                            animation_words.append({'word': letter.capitalize(), 'path': neutral_letter, 'type': 'letter'})
                        # If still not found, skip this letter
        print(animation_words)
        return render(request, 'animation.html', {
            'animation_words': animation_words,
            'original_text': text,
            'isl_text': isl_sentence,
            'emotion': emotion,
            'input_type': input_type
        })
    else:
        return render(request, 'animation.html')

def signup_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            # log the user in
            return redirect('animation')
    else:
        form = UserCreationForm()
    return render(request, 'signup.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            #log in user
            user = form.get_user()
            login(request, user)
            if 'next' in request.POST:
                return redirect(request.POST.get('next'))
            else:
                return redirect('animation')
    else:
        form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect("home")