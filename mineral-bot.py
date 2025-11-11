import aiml
import pandas as pd
import numpy as np

#----------------------------task c------------------------------
import tensorflow as tf
from keras.utils import load_img, img_to_array

# mineral img classification 
model = tf.keras.models.load_model('mineral_cnn_tuned.h5')


mineral_names = { # img map for minerals, translating from integer value output to mineral name
    0: "biotite",
    1: "bornite",
    2: "chrysocolla",
    3: "malachite",
    4: "muscovite",
    5: "pyrite",
    6: "quartz",
}

def identify_mineral_img(img_path): #function for image classification

    img = load_img(img_path, target_size=(64, 64))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    mineral_index = np.argmax(predictions)
    return mineral_names[mineral_index]

#----------------------------task c------------------------------

    
#----------------------------task b------------------------------
import nltk
from nltk.sem.logic import Expression
from nltk.inference import ResolutionProver

#aiml initialisation / kinda task a / b too !
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.learn("mybot-logic.xml")

#logicalkb initalisation 
logical_kb = []
try:
    logical_kb_data = pd.read_csv('logical-kb.csv')
    logical_kb = logical_kb_data['Fact'].tolist() 
except Exception as e:
    print(f"qa_kb error! {e}")
        

property_mapping = { #dictionary for mapping properties to kb based on user's input
    "hard": "MohsHardness",
    "soft": "MohsHardness",
    "transparent": "RefractiveIndex",
    "opaque": "RefractiveIndex",
    "mineral": "Mineral",
    "dense": "Density",
    "light": "Density",
}

#fuzzykb initalisation 
fuzzy_kb = {}
fuzzy_kb_data = pd.read_csv('fuzzy-kb.csv')
for _, row in fuzzy_kb_data.iterrows():
    fuzzy_kb[row['Fact']] = row['TruthValue']

def check_logical_kb(query):
    try:
        query_expr = Expression.fromstring(query) 
        prover = ResolutionProver()
        result = prover.prove(query_expr, logical_kb, verbose=False)
        return result
    except Exception as e:
        return f"logical kb error: {e}"
    
#----------------------------task b------------------------------

    
#----------------------------task a------------------------------
import speech_recognition as sr
import pyttsx3

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

qa_kb = pd.read_csv('mineral_kb.csv')

tfidf_vectorizer = TfidfVectorizer()
vectorizer = tfidf_vectorizer.fit_transform(qa_kb['question'])  
qa_vectorised = vectorizer.toarray() 

def qa_cosine(user_query, threshold=0.5):
    user_vec = tfidf_vectorizer.transform([user_query]).toarray()
    cosine_similarities = cosine_similarity(user_vec, qa_vectorised).flatten()
    most_similar_answer = cosine_similarities.argmax()
    similarity_value = cosine_similarities[most_similar_answer]

    if  similarity_value >= threshold:
        return qa_kb.iloc[most_similar_answer]['answer']
    else:
        return None
    
tts_engine = pyttsx3.init() #for audio output 
tts_engine.setProperty('rate', 150)  #too fast originally !

def chatbot_audio(text): 
    tts_engine.say(text)
    tts_engine.runAndWait()
    
def usr_voice_tranlsation(): #function for capturing user voice & translating to text.
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("listening!, please tell me your query")
        try:
            usr_question_audio = recognizer.listen(source, timeout=5)  
            user_input = recognizer.recognize_google(usr_question_audio)  # google webspeech api 
            print(f"voice input: {user_input}")
            return user_input.lower() 
        except sr.UnknownValueError:
            print("sorry, i did not understand that")
            return None
        except sr.RequestError as e:
            print(f"error; {e}")
            return None
        except Exception as e:
            print(f" error occurred: {e}")
            return None
#----------------------------task a------------------------------




#----------------------------main loop--------------------------------
print("mineral chatbot n1076024")

while True:
    try:
        mode = input("enter 'voice' to use voice input or press any key to type your query: ").strip().lower()
        if mode == "voice":
            userInput = usr_voice_tranlsation()  
            if userInput is None:  
                continue
        else:
            print("query:")
            userInput = input("> ").lower() 
            
        if userInput in ["exit", "quit"]: 
            print("Bye!")
            break

        answer = kern.respond(userInput)

        if answer and answer[0] == '#': 
            params = answer[1:].split('$')
            cmd = int(params[0])

            if cmd == 0:  # exit
                print(params[1])
                break
            
            #----------------------------task a------------------------------
            elif cmd == 35:  # qa-kb
                query = params[1]
                try:
                    kb_answer = qa_cosine(query, threshold=0.5)
                    print(kb_answer)
                    if mode == "voice":
                      chatbot_audio(kb_answer)       
                except Exception as e:
                    print(f"error querying qa kb: {e}")
            #----------------------------task a------------------------------
            
            
         #----------------------------task b-----------------------------
            elif cmd == 31:  # i know that ...
                fact = params[1]
                try:
                    subject, predicate = fact.split(" is ")
                    predicate = predicate.lower().strip()
                    
                    if predicate.startswith("a "):
                        predicate = predicate[2:]  
                    elif predicate.startswith("an "):
                        predicate = predicate[3:]  

                    fuzzy_key = property_mapping.get(predicate)
                    if fuzzy_key:
                        fact_key = f"{fuzzy_key}({subject.lower()})"
                        if fact_key in fuzzy_kb:
                            existing_value = fuzzy_kb[fact_key]
                            if predicate in ["soft", "light", "opaque"] and existing_value >= 0.5:
                                print(f"contradiction: '{fact}' conflicts with existing fact '{fact_key}: {existing_value}' (indicating the opposite).")
                            elif predicate in ["hard", "dense", "transparent"] and existing_value < 0.5:
                                print(f"contradiction: '{fact}' conflicts with existing fact '{fact_key}: {existing_value}' (indicating the opposite).")
                            else:
                                print(f"'{fact}' already exists in the fuzzy KB with a truth value of {existing_value}.")
                        else:
                            truth_value = 1.0 if predicate in ["hard", "dense", "transparent"] else 0.0
                            fuzzy_kb[fact_key] = truth_value
                            fuzzy_kb_data = pd.DataFrame(list(fuzzy_kb.items()), columns=["Fact", "TruthValue"])
                            fuzzy_kb_data.to_csv("fuzzy-kb.csv", index=False)
                            print(f"'{fact}' has been added to the fuzzy KB")
                    else:
                        print(f" property: '{predicate}' is unrecognized.")
                except Exception as e:
                    print(f"error: {e}")

            elif cmd == 32:  # check that ...
                    fact = params[1]
                    try:
                        subject, predicate = fact.split(" is ")
                        fuzzy_key = property_mapping.get(predicate.lower())
                        if fuzzy_key:
                            fact_key = f"{fuzzy_key}({subject.lower()})"
                            fuzzy_result = fuzzy_kb.get(fact_key, None)
                            if fuzzy_result is not None:
                                threshold = 0.5
                                if fuzzy_result >= threshold and predicate.lower() in ["hard", "dense", "transparent"]:
                                    print(f"{subject} is {predicate} with a degree of {fuzzy_result}.")
                                elif fuzzy_result < threshold and predicate.lower() in ["soft", "light", "opaque"]:
                                    print(f"{subject} is {predicate} with a degree of {fuzzy_result}.")
                                else:
                                    print(f"{subject} is not {predicate} (degree: {fuzzy_result}).")
                            else:
                                #print(f"not in fuzzy kb chief!")
                                logical_query = f"{predicate.capitalize()}({subject.lower()})"
                                logical_result = check_logical_kb(logical_query)
                                if logical_result:
                                    print(f"statement '{fact}' is true.")
                                else:
                                    print(f"statement '{fact}' is false or cannot be proven.")
                        else:
                            print(f"im afraid i do not understand : '{predicate}'.")
                    except Exception as e:
                        print(f"logic error!: {e}")
                    
             #----------------------------task b-----------------------------
    
    
            #----------------------------task c-----------------------------
    
            elif cmd == 34:  #img classification 
                print("input image dir:")
                img_path = input("> ")
                try:
                    mineralname = identify_mineral_img(img_path)
                    print(f"this appears to be {mineralname}.")
                except Exception as e:
                    print(f"processing image failed!: {e}")
            #----------------------------task c-----------------------------


            elif cmd == 99: #default 
                print("i didn't understand that.")
        elif answer:  
            print(answer)
        else:  # if response is empty
                print("i didn't understand that.")

    except (KeyboardInterrupt, EOFError):
        print("cya!")
        break