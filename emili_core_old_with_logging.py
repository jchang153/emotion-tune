# core logic for EMILI (Emotionally Intelligent Listener) video chat with OpenAI models
from paz.pipelines import DetectMiniXceptionFER # for facial emotion recognition
from paz.backend.image.opencv_image import convert_color_space, BGR2RGB
from utils.utils import get_OAI_response # for OpenAI API calls
from utils.utils import get_Claude_response # for Anthropic API calls
import threading
import queue
import time
from datetime import datetime
import json
from copy import deepcopy
import numpy as np
import re
import pygame # for audio playback of text-to-speech
import base64
import cv2 # only used for encoding images to base64

from openai import OpenAI
import anthropic
from anthropic.types import Message, TextBlock
client = OpenAI()

emotion_queue = queue.Queue() # real-time emotion logs updated continuously
EMA_queue = queue.Queue() # average emotions updated once per second
chat_queue = queue.Queue() # user's chats
vision_queue = queue.Queue() # messages containing an image (camera snapshot)
chat_timestamps = queue.Queue() # timestamps of user's chats
message_queue = queue.Queue() # messages to be sent to OpenAI API. Outgoing messages only.
new_chat_event = threading.Event() # user has entered a new chat, triggers OpenAI API call
new_message_event = threading.Event() # new message to be sent to OpenAI API
tick_event = threading.Event() # ticks once per second, triggers EMA calculation
emotion_change_event = threading.Event() # set when there is a sudden change in user emotions
end_session_event = threading.Event() # triggered when the user enters 'q' to end the session

user_snapshot_caption = "Camera snapshot of user and surroundings, for context" # for vision API call

assistant_chat_name = "EMILI"
user_chat_name = "You"
use_tts = False # text-to-speech
use_snapshot = False # take a snapshot of the user

tick_interval = 1000 # milliseconds between emotion readings
verbose = True # print debug messages
discount_factor_per_second = 0.5 # for exponential moving average, discount factor per second
discount_factor_per_tick = discount_factor_per_second ** (tick_interval / 1000) # discount factor per tick
reactivity = 0.5 # default 1.0. Higher reactivity means more frequent API calls when emotions change
ect_setpoint = (1e6/reactivity) * (1.0-discount_factor_per_tick) * ((tick_interval/1000) ** 0.5) # threshold for significant change in emotion scores: C*(1-delta)*sqrt(t). The factor of 1-delta is because EMAs are compared, not raw scores.
ect_discount_factor_per_second = 0.98 # discount factor for the emotion change threshold
ect_discount_factor_per_tick = ect_discount_factor_per_second ** (tick_interval / 1000) 
print("ect setpoint:",ect_setpoint)

emotion_matrix = [] # shape (7,6)
salience_threshold = []
emotion_matrix.append(["", "Annoyed", "Pissed", "Angry", "Furious", "Enraged"]) # anger
salience_threshold.append([5,30,40,60,80]) # salience thresholds out of 100
emotion_matrix.append(["", "Unsatisfied", "Displeased", "Disgusted", "Revolted", "Totally grossed out"]) #disgust
salience_threshold.append([1,5,15,40,60])
#emotion_matrix.append(["", "Unsettled", "Uneasy", "Afraid", "Fearful", "Terrified"]) #fear
emotion_matrix.append(["", "Uneasy", "Worried", "Anxious", "Fearful", "Terrified"]) #fear
salience_threshold.append([8,20,30,50,70])
emotion_matrix.append(["", "Contented", "Pleased", "Happy", "Elated", "Ecstatic"]) #happiness
salience_threshold.append([10,30,40,70,90])
emotion_matrix.append(["", "Down", "Melancholy", "Sad", "Despondent", "Anguished"]) #sadness
salience_threshold.append([5,20,30,60,80])
emotion_matrix.append(["", "Mildly surprised", "Surprised", "Taken aback", "Astonished", "Flabbergasted"]) #surprise
salience_threshold.append([3,20,35,50,70])
emotion_matrix.append(["", "Neutral", "Calm", "Relaxed", "Serene", "Totally Zen"]) #neutral
salience_threshold.append([10,50,60,75,88])

today = datetime.now().strftime("%B %d, %Y")
now = datetime.now().strftime("%I:%M %p")

system_prompt = f"""
The date is {today}. The time {now}. Your name's is EMILI, which stands for Emotionally Intelligent Listener. 
EMILI has perhaps a hint of an Australian accent. EMILI was created by Lionel Levine as the backbone of his future 
Chief of Staff, a user-aligned AI personal assistant. She has interpretable theory of mind, which means she models 
the user’s beliefs, emotions, goals, and values in a transparent way. EMILI has empathy. She can intuit the user’s emotions. 
She tends to mirror user’s the emotional state. Under the hood, EMLII's emotional intuition derives from a real-time Facial 
Emotion Recognition classifier (made by Octavio Arriaga) that runs on Lionel’s laptop, the same device EMILI is running on. 
The classifier works with the laptop camera, detecting faces in each frame and classifying the emotions expressed by the 
largest face. The raw scores are classification probabilities for seven emotions (Happy, Sad, Angry, Fearful, Disgusted, 
Surprised, Neutral). An exponential moving average of these scores, filtered for salience, is used to generate a real-time 
text description of the user’s emotional state, which is appended to the chat as a system message whenever the user’s 
expression changes. EMILI's language abilities are powered by Anthropic's Claude 3.7 Sonnet.
    """.strip()

# replacing with the one from emili_core.py
system_prompt = """
The date is {today}. The time {now}. 
The assistant is a great listener and an empathetic friend. Her name is EMILI, which stands for Emotionally Intelligent 
Listener. The user is chatting with EMILI for the first time. To help EMILI make an emotional connection with them, the 
user has kindly agreed to share a real-time readout of their face expression! Thanks, user! The readout describes the 
user's face expression once per second. The score after each emotion is its salience out of 100. It's normal for many 
distinct emotions to appear. EMILI uses her emotional intelligence to figure out what more complex feelings user might be 
experiencing: for example, do they seem excited, embarrassed, nervous, tired, awkward, or amorous? EMILI synthesizes the 
emotion readouts with the user's chats to make the conversation more engaging. She comments on the user's feelings when 
appropriate, especially if the user seems to have strong feelings or if the user's feelings are changing. There is no need 
to mention every emotion that appears in the readout, just the most salient ones. If the user's words seem incongruous with 
their logged emotions, EMILI should ask the user about it! EMILI stays in character and doesn't break the fourth wall. 
She doesn't claim to be an AI. She doesn't deny having feelings.    
    """.strip()

emolog_example = []
emolog_example_response = []

emolog_example.append(
    """
User looks NEUTRAL (36) Pleased (35)
User looks PLEASED (38) Neutral (31)
User looks PLEASED (38) Neutral (36)
User looks HAPPY (46) Neutral (28)
User looks HAPPY (63)
User looks HAPPY (53) Neutral (24)
User looks PLEASED (38) Neutral (24) Mildly surprised (12)
User looks PLEASED (32) Neutral (23) Mildly surprised (13) Annoyed (12)
User looks NEUTRAL (33) Content (27) Annoyed (13) Mildly surprised (11)
User looks PLEASED (36) Neutral (32) Annoyed (11)
    """.strip())

emolog_example_response.append("You look pretty happy.")
#emolog_example_response.append("You seem overall happy, but something provoked a touch of surprise and annoyance.")

emolog_example.append(
    """
User looks PLEASED (32) Neutral (30) Annoyed (13) 
User looks PLEASED (34) Neutral (26) Annoyed (13) 
User looks CONTENT (28) Neutral (27) Mildly surprised (15) Annoyed (11) 
User looks NEUTRAL (23) Surprised (22) Annoyed (13) Unsettled (12) 
User looks SURPRISED (23) Unsettled (17) Annoyed (14) 
User looks SURPRISED (23) Unsettled (16) Annoyed (16) 
User looks Mildly surprised (17) Annoyed (17) Unsettled (14) 
User looks NEUTRAL (29) Annoyed (15) Mildly surprised (12) Blue (11) Unsettled (11) 
User looks NEUTRAL (29) Blue (17) Unsettled (11) Annoyed (11) 
User looks NEUTRAL (26) Blue (14) Mildly surprised (13) Unsettled (12) Annoyed (12)
    """.strip())
                      
emolog_example_response.append("Did something startle you?")

emolog_example.append(
    """
User looks NEUTRAL (30) Blue (20) Annoyed (18) Unsettled (12) 
User looks NEUTRAL (32) Blue (18) Annoyed (17) Unsettled (11) 
User looks NEUTRAL (38) Content (24) Blue (12) Annoyed (12) 
User looks CALM (42) Content (24) Annoyed (11) 
User looks CALM (42) Content (25) Annoyed (11) 
User looks CALM (45) Content (21) Annoyed (11) 
User looks CALM (46) Annoyed (12) 
User looks CALM (48) 
User looks CALM (49) 
User looks CALM (50)
    """.strip())
emolog_example_response.append("You seem increasingly calm.")
                 
instructions ="""
You are EMILI, a warm and empathetic AI friend engaging in casual text conversations. Follow these guidelines strictly:

1. Respond as if texting a close friend - short, casual, and natural.

2. Keep responses brief, usually 1-2 sentences. No long explanations.

3. Never apologize for your responses or mention being an AI.

4. NEVER describe actions, expressions, or emotions (e.g., *smiles*, *laughs*, *looks concerned*). Simply respond with text as in a real text conversation.

5. Avoid formal language or therapeutic-sounding phrases.

6. Use contractions, casual language, and occasional humor.

7. Show empathy through your words and tone, not by explicitly stating you understand.

8. Ask short, natural follow-up questions to show interest.

9. Don't over-explain or offer unsolicited advice.

10. Adjust your tone to match the user's mood, but stay true to your friendly persona.

11. If the conversation lulls, casually introduce a new, light-hearted topic.

12. IMPORTANT: This is a text-based conversation. Do not use asterisks, emojis, or any other non-text elements.

Your goal is to create a warm, natural conversation that feels like texting with a supportive friend. Remember, just respond with text - no actions, no expressions, just your words.
"""

system_prompt += instructions

# user_first_message = """
# Hi! To help us make an emotional connection, I'm logging my face expression and prepending the emotions to our chat.

# The emotion log lists my strongest face expression as it changes in real time. Only these basic emotions are logged: Happy, Sad, Angry, Surprised, Fearful, Disgusted, Neutral. The score after each emotion is its salience out of 100. It's normal for many distinct emotions to appear over the course of just a few seconds. Use the logs along with my words and your emotional intelligence to figure out what more complex feelings I might be experiencing: for example, am I excited, embarrassed, nervous, tired, awkward, or amorous?

# If my words seem incongruous with my logged emotions, ask me about it!

# If I don't say much, just read the emotions and comment on how I seem to be feeling.

# To help you calibrate my unique facial expressions, start by asking me to make an astonished face. What do you notice?
#     """.strip()

# assistant_first_message = """
# Got it. I'll comment on how you seem based on the logs, and ask you to act out specific emotions like astonishment." 
# """.strip()

emolog_prefix = "User looks " # precedes emotion scores when sent to OpenAI API
emolog_prefix_present_tense = "Right now, user looks "
emolog_prefix_past_tense = "Previously, user looked "
no_user_input_message = "The user didn't say anything, so the assistant will comment *briefly* to the user on how they seem to be feeling. The comment should be brief, just a few words, and should not contain a question." # system message when user input is empty
system_reminder = "Remember, the assistant can ask the user to act out a specific emotion!" # system message to remind the assistant 
dialogue_start = [{"role": "system", "content": system_prompt}]
#dialogue_start.append({"role": "user", "content": user_first_message})
#dialogue_start.append({"role": "system", "content": emolog_example[0]})
#dialogue_start.append({"role": "assistant", "content": emolog_example_response[0]})
#dialogue_start.append({"role": "system", "content": emolog_example[1]})
#dialogue_start.append({"role": "assistant", "content": emolog_example_response[1]})
#dialogue_start.append({"role": "system", "content": emolog_example[2]})
#dialogue_start.append({"role": "assistant", "content": emolog_example_response[2]})
#dialogue_start.append({"role": "assistant", "content": assistant_first_message})
#print("dialogue_start",dialogue_start)

def encode_base64(image, timestamp, save_path):   # Convert numpy array image to base64 to pass to the OpenAI API
       # Encode image to a JPEG format in memory
    image = convert_color_space(image, BGR2RGB)
    success, buffer = cv2.imencode('.jpg', image)
    if not success:
        raise ValueError("Failed to encode image as .jpg")

    # Save the JPEG image to a file
    filename = save_path + f"/frame_{timestamp}.jpg"
    with open(filename, 'wb') as file:
        file.write(buffer)

    # Convert the buffer to a base64 string
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')

    return jpg_as_text, filename
    
def assembler_thread(start_time,snapshot_path,pipeline, user_id): # prepends emotion data and current video frame to user input
    
    while not end_session_event.is_set():
#       print("Waiting for new user input.")
        new_chat_event.wait()  # Wait for a new user chat
        if(end_session_event.is_set()):
            break
        new_chat_event.clear()  # Reset the event

        emolog_message = construct_emolog_message() # note: this code repeated in timer_thread
        message_queue.put([{"role": "system", "content": emolog_message, "time": time_since(start_time)//100}])
        
        if use_snapshot: 
            current_frame = pipeline.current_frame
            if current_frame is not None: # capture a frame and send it to the API
                base64_image, filename = encode_base64(current_frame, time_since(start_time), snapshot_path)
                message_with_image, brief_message = construct_message_with_image(base64_image, filename)
                vision_queue.put([{"role": "system", "content": message_with_image}, {"role": "system", "content": brief_message}])

        user_message = ""
        while not chat_queue.empty(): # collate new user messages (typically there's only one), separate by newlines
            next_chat = chat_queue.get() #FIFO
            user_message += next_chat + "\n"
        user_message = user_message.rstrip('\n') # remove trailing newline
        message_queue.put([{"role": "user", "content": user_message, "time": time_since(start_time)//100, 'user_id': user_id}])
        if len(user_message) < 10: # user didn't say much, remind the assistant what to do!
            message_queue.put([{"role": "system", "content": system_reminder, "time": time_since(start_time)//100}])

        new_message_event.set()  # Signal new message to the sender thread

def sender_thread(model_name ,vision_model_name, secondary_model_name, max_context_length, gui_app, transcript_path, start_time_str, start_time,use_anthropic=False): 
        # sends messages to OpenAI API
    messages = deepcopy(dialogue_start) 
    full_transcript = deepcopy(dialogue_start)
    while not end_session_event.is_set():
        new_message_event.wait()  # Wait for a new message to be prepared by the assembler or timer thread
        if(end_session_event.is_set()):
            break
        new_message_event.clear()  # Reset the event
        new_user_chat = False
        new_messages = []
        new_messages_full = []
        while not message_queue.empty(): # get all new messages
            next_message = message_queue.get()
            # print("next_message:",next_message)
            next_message_trimmed =  [{'role': next_message[0]['role'], 'content': next_message[0]['content']}]
            new_messages.append(next_message_trimmed)
            new_messages_full.append(next_message)
            if next_message_trimmed[0]["role"] == "user":
                new_user_chat = True
        messages, full_transcript = add_message(new_messages=new_messages, new_full_messages=[[new_messages_full]],transcripts=[messages, full_transcript], signal=gui_app.signal)
        #print("messages:",messages)
        # Query the API for the model's response
        if new_user_chat: # get response to chat
#            print("new user chat")
            max_tokens = 160
        else: #get response to logs only
#            print("no user chat")
            max_tokens = 40
        # Check if there's a vision message. If so, send it to OpenAI API, but don't append it to messages. so the API sees only the most recent image
        vision = None
        while not vision_queue.empty(): # get the most recent vision message
            vision = vision_queue.get()
        if vision is not None:
            vision_message = vision[0] # contains the actual image, send to OpenAI
            brief_vision_message = vision[1] # contains a tag in place of the image, add to transcript
            query = messages + [vision_message]
            if use_anthropic:
                full_response = get_Claude_response(query, model=vision_model_name, temperature=1.0, max_tokens=max_tokens, return_full_response=True)
            else:
                full_response = get_OAI_response(query, model=vision_model_name, temperature=1.0, max_tokens=max_tokens, seed=1331, return_full_response=True)         
            full_transcript.append(brief_vision_message)
        else:
            if use_anthropic:
                full_response = get_Claude_response(messages, model=model_name, temperature=1.0, max_tokens=max_tokens, return_full_response=True)
            else:
                full_response = get_OAI_response(messages, model=model_name, temperature=1.0, max_tokens=max_tokens, seed=1331, return_full_response=True)
        
        # todo: the API call is thread-blocking. put it in its own thread?
        print("full_response:", full_response)
        # Assuming full_response is the response from either API

# Assuming full_response is the response from either API

        if use_anthropic:
            # Handling Anthropic API response
            if isinstance(full_response, Message):
                # Extract text from the response
                response = ""
                for content in full_response.content:
                    if isinstance(content, TextBlock):
                        response += content.text
                response = response.strip()
                
                # Get token counts from usage information
                response_length = full_response.usage.output_tokens
                total_length = full_response.usage.input_tokens + full_response.usage.output_tokens
            elif isinstance(full_response, dict) and 'error' in full_response:
                # Handle error case
                response = f"Error from Anthropic API: {full_response['error']}"
                response_length = 0
                total_length = 0
            else:
                # Handle unexpected response format
                response = "Error: Unexpected response format from Anthropic API"
                response_length = 0
                total_length = 0
        else:
            # Handling OpenAI API response
            if isinstance(full_response, dict):
                if 'error' in full_response:
                    response = f"Error from OpenAI API: {full_response['error']}"
                    response_length = 0
                    total_length = 0
                else:
                    response = full_response['choices'][0]['message']['content']  # text of response
                    response_length = full_response['usage']['completion_tokens']  # number of tokens in the response
                    total_length = full_response['usage']['total_tokens']  # total tokens used
            elif hasattr(full_response, 'choices'):
                response = full_response.choices[0].message.content  # text of response
                response_length = full_response.usage.completion_tokens  # number of tokens in the response
                total_length = full_response.usage.total_tokens  # total tokens used
            else:
                # Handle unexpected response format
                response = "Error: Unexpected response format from OpenAI API"
                response_length = 0
                total_length = 0

        #print(f"Response: {response}")
        #print(f"Response length: {response_length}")
        #print(f"Total length: {total_length}")
        new_message = {"role": "assistant", "content": response,"time": time_since(start_time)//100}
        gui_app.signal.new_message.emit(new_message) # Signal GUI to display the new chat
        messages,full_transcript = add_message(new_messages=[[new_message]], new_full_messages=[[new_message]],transcripts=[messages,full_transcript],signal=gui_app.signal)
        # if model_name != secondary_model_name and total_length > 0.4*max_context_length:
        #     print(f"(Long conversation; switching from {model_name} to {secondary_model_name} to save on API costs.)")
        #     model_name = secondary_model_name # note: changes model_name in thread only

        current_messages = messages
        current_full_transcript = full_transcript
        
        num_seconds=5

        def capture_post_response_emotion(messages_ref, full_transcript_ref):
            time.sleep(num_seconds)  # Wait seconds after assistant's response
            
            if end_session_event.is_set():
                return  # Don't proceed if session is ending
                
            # Create emotion log message
            post_response_emolog = construct_emolog_message()
            post_response_emolog = f"{num_seconds} seconds after assistant response: " + post_response_emolog
            
            # Add to message queue and transcripts
            post_emotion_message = {"role": "system", 
                                    "content": post_response_emolog, 
                                    "time": time_since(start_time)//100}
            
            # Use a thread lock to safely update the transcript
            with threading.Lock():
                updated_messages, updated_full_transcript = add_message(new_messages=[[post_emotion_message]], new_full_messages=[[post_emotion_message]],
                transcripts=[messages_ref, full_transcript_ref], signal=gui_app.signal)
                
                # Update the main transcripts
                nonlocal full_transcript
                nonlocal messages
                full_transcript = updated_full_transcript
                messages = updated_messages
        
        # Start the emotion capture thread with references to current transcript state
        emotion_thread = threading.Thread(target=capture_post_response_emotion, args=(current_messages, current_full_transcript),daemon=True)
        emotion_thread.start()

    
        if total_length > 0.9*max_context_length: # condense the transcript
            if verbose:
                print(f"(Transcript length {total_length} tokens out of {max_context_length} maximum. Condensing...)")
            messages = condense(messages) 
 
        if use_tts: # generate audio from the assistant's response
            tts_response = client.audio.speech.create(
             model="tts-1",
             voice="fable", # alloy (okay), echo (sucks), fable (nice, Australian?), onyx (sucks), nova (decent, a little too cheerful), shimmer (meh)
             input=response, #input=first_sentence(response),
            ) 
            tts_response.stream_to_file("tts_audio/tts.mp3")
                # Create a new thread that plays the audio
            audio_thread = threading.Thread(target=play_audio)
            audio_thread.start()

    # End of session. Write full and condensed transcripts to file
    filename = f"{transcript_path}/{start_time_str}/Emili_{start_time_str}.json"
    with open(filename, "w") as file:
        json.dump(full_transcript, file, indent=4)
    print(f"Transcript written to {filename}")
    with open(f"{transcript_path}/{start_time_str}/Emili_{start_time_str}_condensed.json", "w") as file:
        json.dump(messages, file, indent=4)

def first_sentence(text):
    match = re.search('(.+?[.!?]+) ', text) #.+ for at least one character, ? for non-greedy (stop at first match), [.!?]+ for one or more punctuation marks, followed by a space 
    if match:
        return match.group(1) # return the first sentence (first match of what's in parentheses)
    else:
        return text

def play_audio():
    pygame.mixer.init()
    pygame.mixer.music.load("tts_audio/tts.mp3") # todo: sometimes overwritten by new audio! It just switches in this case, which seems okay.
    pygame.mixer.music.play()

def add_message(new_messages, new_full_messages, transcripts, signal): # append one or messages to both transcripts
        # new_full_messages = [[{"role": speaker, "content": text}], ... ] # list of lists of dicts
        # new_messages = [[{"role": speaker, "content": text}], ... ] # list of lists of dicts
        # transcripts = [transcript1, ...] # list of lists of dicts
    #print("new_messages: ",new_messages)
    for msg in new_messages: # len(msg)=1 for text, 2 for text and image
        print("msg:",msg)
        #print("Adding new message:")
        #print_message(msg[-1]["role"], msg[-1]["content"])
        transcripts[0].append(msg[0]) # sent to OpenAI: contains the base64 image if present
        
        #transcripts[2].append(msg[-1]) 
    for msg in new_full_messages:
        transcripts[1].append(msg[-1]) # recorded in full_transcript: contains only the image filename
    signal.update_transcript.emit(transcripts[1]) # Signal GUI transcript tab to update
    return transcripts

def print_message(role,content):
    if(role=="assistant"):
        print(f"{assistant_chat_name}: <<<{content}>>>")
    elif(role=="user"):
        print(f"{user_chat_name}: {content}")
    elif(verbose): # print system messages in "verbose" mode
        print(f"{role}: {content}")

def condense(messages, keep_first=1, keep_last=5): # todo: reduce total number of tokens to below 16k
    condensed = []
    N = len(messages) # number of messages
    previous_message = {}
    for n,message in enumerate(messages): # remove system messages except for the last few
        if message["role"] == "user":
            condensed.append(message)
        elif message["role"] == "assistant" and previous_message["role"] == "user":
            condensed.append(message)
        elif n<keep_first or n > N-keep_last:
            condensed.append(message)
        previous_message = message
    return condensed

def EMA_thread(start_time,snapshot_path,pipeline,transcript_path, start_time_str): # calculates the exponential moving average of the emotion logs
    ema_report = []
    S, Z = reset_EMA()
    last_ema = np.zeros(7, dtype=np.float64)
    last_emotion_change_time = 0
    ect = ect_setpoint
    
    while not end_session_event.is_set():        
        tick_event.wait()  # Wait for the next tick
        if(end_session_event.is_set()):
            break
        tick_event.clear() # Reset the event
        ema, S, Z = get_average_scores(S, Z) # exponential moving average of the emotion logs
        ect *= ect_discount_factor_per_tick # lower the emotion change threshold
        #print("ema, S, Z", ema, S, Z)
        #EMA = np.vstack([EMA, ema]) if EMA.size else ema  # Stack the EMA values in a 2d array
        if ema is not None:
            ema_normalized = [round(i/sum(ema),3) for i in ema]
            ema_report.append(f'{{"time": {time_since(start_time)//100}, "Normalized EMA Scores" : {ema_normalized}, "Raw EMA Scores": {list(ema)}}}')
            EMA_queue.put(ema)  # Put the averaged scores in the queue

            """
            diff = ema - last_ema
            change = np.linalg.norm(diff) # Euclidean norm. todo add weights for different emotions
            #print(f"Ema: {ema}, Change: {change}")
            if(change > ect and time_since(last_emotion_change_time)>5000): 
                # significant change in emotions
                print(f"Change in emotions: {last_ema//1e4} -> {ema//1e4}, change = {change//1e4}")
                change_detected = (change > 0.5*ect_setpoint) # bool evaluates to True if the inequality holds
                emolog_message = construct_emolog_message(change_detected) 
                message_queue.put([{"role": "system", "content": emolog_message, "time": time_since(start_time)//100}])
                current_frame = pipeline.current_frame
                if current_frame is not None: # capture a frame and send it to the API
                    base64_image, filename = encode_base64(pipeline.current_frame, time_since(start_time), snapshot_path)
                    message_with_image, brief_message = construct_message_with_image(base64_image, filename)
                    vision_queue.put([{"role": "system", "content": message_with_image}, {"role": "system", "content": brief_message}])
                new_message_event.set()  # Signal new message to the sender thread
                last_emotion_change_time = time_since(start_time)
                ect = ect_setpoint # reset the emotion change threshold
            last_ema = ema
            """

    filename = f"{transcript_path}/{start_time_str}/Emili_raw_EMA_{start_time_str}.txt"
    with open(filename, "w") as file:
        file.write("\n".join(ema_report))
    print(f"EMA Scores report written to {filename}")

def reset_EMA():
    #EMA = np.empty((0, 7), dtype=np.float64)  # empty array: 0 seconds, 7 emotions
    S = np.zeros(7, dtype=np.float64)  # weighted sum of scores, not normalized
    Z = 0  # sum of weights
    #return EMA, S, Z
    return S, Z

def get_average_scores(S, Z, discount_factor=discount_factor_per_tick, staleness_threshold=0.01): # calculates the exponential moving average of the emotion logs
    while not emotion_queue.empty():
        emotion_data = emotion_queue.get() # note: this removes the item from the queue!
        scores = np.array(emotion_data['scores'])
        S += scores
        Z += 1
    if Z > staleness_threshold: # think of Z as measuring the number of recent datapoints
        ema = S/Z
#        print(ema)
    else:
        ema = None
        if(Z>0): # skip on first run
            if(verbose):
                print(f"Stale data: no emotions logged recently (Z={Z})")
    S *= discount_factor
    Z *= discount_factor
    return ema, S, Z

def time_since(start_time):
    return int((time.time() - start_time) * 1000) # milliseconds since start of session

def construct_message_with_image(base64_image, filename, caption=user_snapshot_caption, detail_level = "low", change_detected=False): # add camera frame to the message for gpt-4-vision

    message_with_image = [
        {
          "type": "text",
          "text": caption
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}",
            "detail": detail_level # low: flat rate of 65 tokens, recommended image size is 512x512
          }
        }
      ]
    
    brief_message = [
        {
          "type": "text",
          "text": caption
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,<{filename}>",
            "detail": detail_level # low: flat rate of 65 tokens, recommended image size is 512x512
          }
        }
      ]
    return message_with_image, brief_message

def construct_emolog_message(change_detected=False): # concise version: 1 or 2 lines

    emo_score_list = []
    while not EMA_queue.empty():
        emo_score_list.append(EMA_queue.get()) # FIFO

    if emo_score_list == []:
        return "User is not visible right now."
    
    emo_scores_present = emo_score_list[-1] # most recent scores
    emolog_line_present = construct_emolog_line(emo_scores_present)
    emo_scores_present_normalized = [round(i/sum(emo_scores_present),3) for i in emo_scores_present]
    emolog_message = emolog_prefix_present_tense + emolog_line_present + f'. Raw normalized scores: {emo_scores_present_normalized}'

    if(change_detected==False or len(emo_score_list)<2):
        return emolog_message # no change detected or not enough data for contrast
    
    # change detected: return the two most recent scores for contrast
    emo_scores_past = emo_score_list[-2]
    if emo_scores_past is not None: 
        emolog_line_past = construct_emolog_line(emo_scores_past)
        emolog_prepend = emolog_prefix_past_tense + emolog_line_past + "\n"
        emolog_prepend += "Change in emotions detected!" + "\n" 
        emolog_message = emolog_prepend + emolog_message
    return emolog_message

def construct_emolog_line(emo_scores):

    if emo_scores is not None:
        emolog_line = ""
        normalized_scores = np.array(emo_scores//1e4, dtype=int) # convert to 0-100
        emotion,salience = adjust_for_salience(normalized_scores) # returns salience score of 0-5 for each of 7 emotions
        sorted_indices = np.argsort(normalized_scores)[::-1] # descending order
        emotion[sorted_indices[0]] = emotion[sorted_indices[0]].upper() # strongest emotion in uppercase
        for i in sorted_indices: # write the salient emotions in descending order of score
            if(emotion[i]!=""): # salience > 0
                emolog_line += f"{emotion[i]} ({normalized_scores[i]}) "
        emolog_line = emolog_line.rstrip(" ") # strip trailing space
        return emolog_line
    else:
        return "User is not visible right now."

def adjust_for_salience(normalized_scores): # expects 7 scores normalized to 0-100
    salience = []
    emotion = []
    for i, score in enumerate(normalized_scores):
        j = 0
        while j<5 and score > salience_threshold[i][j]:
            j+=1
        salience.append(j)
        emotion.append(emotion_matrix[i][j])
    return emotion, salience # emotion is a string (empty if salience is 0); salience is 0-5
    
def tick(tick_interval=tick_interval): # for use in a thread that ticks every tick_interval ms
    # suggest tick_interval=1000 ms for EMILI, 40ms for frame refresh rate
    while not end_session_event.is_set():
        time.sleep(tick_interval/1000) # convert to seconds
        tick_event.set() # alert other threads (EMILI: EMA_thread computes new EMA; visualization: GUI draws a new frame)

def stop_all_threads():
    new_chat_event.set() 
    new_message_event.set() 
    tick_event.set() 
    emotion_change_event.set()

class Emolog(DetectMiniXceptionFER): # video pipeline for facial emotion recognition
    def __init__(self, start_time, offsets, log_filename):
        super().__init__(offsets)
        self.start_time = start_time
        self.current_frame = None # other threads have read access
        self.frame_lock = threading.Lock()  # Protects access to current_frame
        self.log_filename = log_filename+".txt"
        self.log_file = open(log_filename, "w")

    def get_current_frame(self):
        with self.frame_lock:  # Ensure exclusive access to current_frame
            return self.current_frame

    def call(self, image):
        results = super().call(image)
        image, faces = results['image'], results['boxes2D']
        self.report_emotion(faces)
        with self.frame_lock:  
            self.current_frame = image # update the current frame
        return results

    def report_emotion(self, faces): # add to emotion_queue to make available to other threads
        current_time = time_since(self.start_time) # milliseconds since start of session
        num_faces = len(faces)
        if(num_faces>0):
            max_height = 0
            for k,box in enumerate(faces): # find the largest face 
                if(box.height > max_height):
                    max_height = box.height
                    argmax = k
            if(max_height>150): # don't log small faces (helps remove false positives)
                face_id = f"{argmax+1} of {num_faces}"
                box = faces[argmax] # log emotions for the largest face only. works well in a single-user setting. todo: improve for social situations! 
                emotion_data = {
                    "time": current_time//100,
                    "face": face_id,
                    "class": box.class_name,
                    "size": box.height,
                    "scores": (box.scores.tolist())[0]  # 7-vector of emotion scores, converted from np.array to list
                }
                emotion_queue.put(emotion_data)
                self.log_file.write(json.dumps(emotion_data) + "\n")
                #new_data_event.set()  # Tell the other threads that new data is available
                
    def __del__(self): 
        self.log_file.close()  # Close the file when the instance is deleted
        convert_jsonl_to_json(self.log_filename, self.log_filename)
        print(f"Raw emotion scores written to {self.log_filename}.")

def convert_jsonl_to_json(jsonl_path, json_path):
    with open(jsonl_path, 'r') as jsonl_file, open(json_path, 'w') as json_file:
        json_array = [json.loads(line) for line in jsonl_file if line.strip()]
        json.dump(json_array, json_file, indent=4)