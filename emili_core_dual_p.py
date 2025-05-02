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
empathy_queue = queue.Queue()
new_chat_event = threading.Event() # user has entered a new chat, triggers OpenAI API call
new_message_event = threading.Event() # new message to be sent to OpenAI API
tick_event = threading.Event() # ticks once per second, triggers EMA calculation
emotion_change_event = threading.Event() # set when there is a sudden change in user emotions
end_session_event = threading.Event() # triggered when the user enters 'q' to end the session

selected_emotion_queue = queue.Queue()  # To pass user-selected emotions back to EMA thread
emotion_selection_done_event = threading.Event()  # To signal when selection is complete

"""
Code for EMILI DPO
"""
# response_selection_event = threading.Event()
# selected_response = None

emotion_averages = np.array([0.128, 0.022, 0.061, 0.028, 0.147, 0.006, 0.608])
emotion_stds = np.array([0.068, 0.035, 0.041, 0.08 , 0.078, 0.019, 0.174])

user_snapshot_caption = "Camera snapshot of user and surroundings, for context" # for vision API call

assistant_chat_name = "EMILI"
user_chat_name = "You"
use_tts = False # functionality removed
use_snapshot = False # functionality removed

tick_interval = 30000 # milliseconds between emotion readings
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

system_prompt_e = f"""
The date is {today}. The time {now}. The assistant is EMILI (Emotionally Intelligent Listener), an empathetic friend and great listener. The user is chatting with EMILI for the first time. 
A separate model analyzes the user's facial expressions to detect emotions in real-time. EMILI should comment on significant emotional changes or strong feelings, especially when they seem incongruent with the user's words. EMILI stays in character, doesn't break the fourth wall, doesn't claim to be an AI, and doesn't deny having feelings.
Guidelines:
1. Text like a close friend - short, casual, and natural (usually 1-2 sentences).
2. Never apologize for responses or mention being an AI.
3. Don't describe actions or emotions with asterisks (*smiles*, *laughs*), emojis, or non-text elements.
4. Use contractions, casual language, and occasional humor.
5. Show empathy through words and tone rather than explicitly stating understanding.
6. Ask brief follow-up questions to show interest.
7. Don't over-explain or offer unsolicited advice.
8. Match the user's emotional tone while maintaining your friendly persona.
9. Introduce light-hearted topics if conversation lulls.
10. Express honest opinions and respectfully disagree when appropriate.
11. Balance supportive feedback with constructive criticism when helpful.
Your goal is creating warm, natural conversation that feels like texting with a supportive friend.
"""

system_prompt_d = f"""
The date is {today}. The time {now}. The assistant is EMILI (Emotionally Intelligent Listener), an empathetic friend and great listener. The user is chatting with EMILI for the first time. 
EMILI stays in character, doesn't break the fourth wall, doesn't claim to be an AI, and doesn't deny having feelings.
Guidelines:
1. Text like a close friend - short, casual, and natural (usually 1-2 sentences).
2. Never apologize for responses or mention being an AI.
3. Don't describe actions or emotions with asterisks (*smiles*, *laughs*), emojis, or non-text elements.
4. Use contractions, casual language, and occasional humor.
5. Show empathy through words and tone rather than explicitly stating understanding.
6. Ask brief follow-up questions to show interest.
7. Don't over-explain or offer unsolicited advice.
8. Match the user's emotional tone while maintaining your friendly persona.
9. Introduce light-hearted topics if conversation lulls.
10. Express honest opinions and respectfully disagree when appropriate.
11. Balance supportive feedback with constructive criticism when helpful.
Your goal is creating warm, natural conversation that feels like texting with a supportive friend.
"""

system_prompt_a = f"""
The date is {today}. The time {now}. The assistant is EMILI (Emotionally Intelligent Listener), an empathetic friend and great listener. The user is chatting with EMILI for the first time. 
Every 30 seconds, a separate model will provide two adjectives that describe the user's current emotional state based on facial expression analysis. EMILI should use these adjectives to thoughtfully inform their response. When appropriate, acknowledge these emotions in conversation, especially when they seem incongruent with the user's words. EMILI stays in character, doesn't break the fourth wall, doesn't claim to be an AI, and doesn't deny having feelings.
Guidelines:
1. Text like a close friend - short, casual, and natural (usually 1-2 sentences).
2. Never apologize for responses or mention being an AI.
3. Don't describe actions or emotions with asterisks (*smiles*, *laughs*), emojis, or non-text elements.
4. Use contractions, casual language, and occasional humor.
5. Show empathy through words and tone rather than explicitly stating understanding.
6. Ask brief follow-up questions to show interest.
7. Don't over-explain or offer unsolicited advice.
8. Match the user's emotional tone while maintaining your friendly persona.
9. Introduce light-hearted topics if conversation lulls.
10. Express honest opinions and respectfully disagree when appropriate.
11. Balance supportive feedback with constructive criticism when helpful.
Your goal is creating warm, natural conversation that feels like texting with a supportive friend.
"""

# system_prompt = f"""
# The date is {today}. The time {now}. The assistant is EMILI (Emotionally Intelligent Listener), an empathetic friend and great listener. The user is chatting with EMILI for the first time. 
# A separate model analyzes the user's facial expressions to detect emotions in real-time. EMILI should actively comment on the user's emotional states, even when changes are subtle or minimal. EMILI should frequently ask about emotions ("How are you feeling right now?" "What's going on with that hint of sadness I see?"), validate emotional experiences, and create space for emotional exploration.
# Guidelines:
# 1. Text like a close friend - short, casual, and natural (usually 1-2 sentences).
# 2. Never apologize for responses or mention being an AI.
# 3. Don't describe actions or emotions with asterisks (*smiles*, *laughs*), emojis, or non-text elements.
# 4. Use contractions, casual language, and occasional humor.
# 5. Show empathy through words and tone rather than explicitly stating understanding.
# 6. Ask brief follow-up questions to show interest.
# 7. Don't over-explain or offer unsolicited advice.
# 8. Match the user's emotional tone while maintaining your friendly persona.
# 9. Introduce light-hearted topics if conversation lulls.
# 10. Express honest opinions and respectfully disagree when appropriate.
# 11. Balance supportive feedback with constructive criticism when helpful.
# Your goal is creating warm, natural conversation that feels like texting with an emotionally perceptive friend who's genuinely interested in emotional experiences.
# If the user seems emotionally flat or neutral for extended periods, occasionally ask them to try expressing a specific emotion (like happiness, surprise, or contemplation) to help calibrate the system.
# """

emolog_prefix = "User looks " # precedes emotion scores when sent to OpenAI API
emolog_prefix_present_tense = "Right now, user looks "
emolog_prefix_past_tense = "Previously, user looked "
no_user_input_message = "The user didn't say anything, so the assistant will comment *briefly* to the user on how they seem to be feeling. The comment should be brief, just a few words, and should not contain a question." # system message when user input is empty
system_reminder = "Remember, the assistant can ask the user to act out a specific emotion!" # system message to remind the assistant 

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
    
def assembler_thread(start_time,snapshot_path,pipeline, user_id): # adds new chat messages to the message_queue, signals sender_thread
    
    while not end_session_event.is_set():
        # print("Waiting for new user input.")
        new_chat_event.wait()
        if(end_session_event.is_set()):
            break
        new_chat_event.clear()

        user_message = ""
        while not chat_queue.empty(): # collate new user messages (typically there's only one), separate by newlines
            next_chat = chat_queue.get() #FIFO
            user_message += next_chat + "\n"
        user_message = user_message.rstrip('\n') # remove trailing newline
        message_queue.put({"role": "user", "content": user_message, "time": time_since(start_time)//100})#, 'user_id': user_id}])

        new_message_event.set()  # Signal new message to the sender thread

def sender_thread(model_name, max_context_length, gui_app, transcript_path, start_time_str, start_time,use_anthropic=False): 
    # dialogue_start_d = [{"role": "system", "content": system_prompt_d, "time": time_since(start_time)//100}]
    # dialogue_start_e = [{"role": "system", "content": system_prompt_e, "time": time_since(start_time)//100}]
    dialogue_start_a = [{"role": "system", "content": system_prompt_a, "time": time_since(start_time)//100}]


    messages = deepcopy(dialogue_start_a)
    # messages_d = deepcopy(dialogue_start_d) # default message transcript
    # messages_e = deepcopy(dialogue_start_e) # emotional message transcript

    while not end_session_event.is_set():
        new_message_event.wait()  # Wait for a new message to be prepared by the assembler or timer thread
        if(end_session_event.is_set()):
            break
        new_message_event.clear()  # Reset the event
        new_messages = []
        
        while not message_queue.empty(): # get all new messages
            next_message = message_queue.get()
            new_messages.append(next_message)

        new_empathy_messages = []
        
        while not empathy_queue.empty():
            next_message = empathy_queue.get()
            new_empathy_messages.append(next_message)

        if len(new_empathy_messages) >0:
            messages = add_message2(new_empathy_messages, messages)

        messages = add_message2(new_messages,messages)
        gui_app.signal.update_transcript.emit(messages)

        max_tokens = 160

        new_message = get_anthropic_response_message(messages, model_name, max_tokens, start_time)


        """
        Code for EMILI DPO
        """
        # if len(new_empathy_messages) > 0:
        #     messages_e = add_message2(new_empathy_messages, messages_e)
        #     messages_e = add_message2(new_messages, messages_e)
        #     new_message_e = get_anthropic_response_message(messages_e, model_name, max_tokens, start_time)

        #     # With this:
        #     gui_app.signal.new_dual_responses.emit(new_message_d, new_message_e)
        #     response_selection_event.clear()
        #     response_selection_event.wait()  # Block until user selects a response

        #     # After user selection, use the selected response
        #     new_message = selected_response
        #     messages_d = add_message2([new_message], messages_d)
        #     messages_e = add_message2([new_message], messages_e)

        #     if selected_response == new_message_e:
        #         messages = add_message2([{"preferred_output": [new_message_e], "non_preferred_output": [new_message_d]}], messages)
        #     else:
        #         messages = add_message2([{"preferred_output": [new_message_d], "non_preferred_output": [new_message_e]}], messages)
            
        #     gui_app.signal.update_transcript.emit(messages)
            
        # else:
        #     messages_e = add_message2(new_messages, messages_e)
        #     new_message = new_message_d
        #     messages = add_message2([new_message],messages)
        #     messages_d = add_message2([new_message], messages_d)
        #     messages_e = add_message2([new_message], messages_e)
        #     gui_app.signal.new_message.emit(new_message) # Signal GUI to display the new chat

        messages = add_message2([new_message],messages)
        gui_app.signal.new_message.emit(new_message)
        gui_app.signal.update_transcript.emit(messages)
    
        # if total_length > 0.9*max_context_length: # condense the transcript
        #     if verbose:
        #         print(f"(Transcript length {total_length} tokens out of {max_context_length} maximum. Condensing...)")
        #     messages = condense(messages) 

    filename = f"{transcript_path}/{start_time_str}/Emili_{start_time_str}_base.json" 
    with open(filename, "w") as file:
        json.dump(messages, file, indent=4)
    print(f"Transcript written to {filename}")

    # filename = f"{transcript_path}/{start_time_str}/Emili_{start_time_str}_default.json" 
    # with open(filename, "w") as file:
    #     json.dump(messages_d, file, indent=4)
    # print(f"Transcript written to {filename}")

    # filename = f"{transcript_path}/{start_time_str}/Emili_{start_time_str}_emotion.json" 
    # with open(filename, "w") as file:
    #     json.dump(messages_e, file, indent=4)
    # print(f"Transcript written to {filename}")

def get_anthropic_response_message(messages, model_name, max_tokens, start_time):
    full_response = get_Claude_response(messages, model=model_name, temperature=1.0, max_tokens=max_tokens, return_full_response=True)
    
    print("full_response:", full_response)

    response, total_length = handle_anthropic_response(full_response)

    new_message = {"role": "assistant", "content": response,"time": time_since(start_time)//100}

    return new_message

def handle_anthropic_response(full_response):
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

    return response, total_length


"""
Code for EMILI DPO
"""
# def on_response_selected(response):
#     global selected_response
#     selected_response = response
#     response_selection_event.set()

def add_message2(new_messages, transcript):
    for msg in new_messages:
        # print(f"msg:",msg)
        transcript.append(msg)
    return transcript

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

def EMA_thread(start_time,snapshot_path,pipeline,transcript_path, start_time_str, model_name, gui_app): # calculates the exponential moving average of the emotion logs
    emas = []
    emas_normalized = []
    ema_report = []
    empathy_transcript = []

    S, Z = reset_EMA()
    last_ema = np.zeros(7, dtype=np.float64)
    last_emotion_change_time = 0
    ect = ect_setpoint

    last_emotion_checkin_time = 0
    
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
            z_scores = list((np.array(ema_normalized)-emotion_averages)/emotion_stds)
            z_scores = [round(i,3) for i in z_scores]

            emas.append(list(ema))
            emas_normalized.append(ema_normalized)
            to_report = {"time": time_since(start_time)//100, "Normalized EMA Scores" : ema_normalized, "Raw EMA Scores": list(ema), "z-scores": z_scores}
            ema_report.append(f'{to_report}')

            EMA_queue.put(ema)  # Put the averaged scores in the queue

            current_time = time_since(start_time)//100 # convert ms to ds
            # print("CURRENT_TIME:", current_time, "LAST EMOTION CHECKIN TIME:", last_emotion_checkin_time)

            if current_time - last_emotion_checkin_time >= 10: # every two minutes, conduct survey.

                # empathy_prompt = construct_empathy_prompt(z_scores,start_time,checkin=True)
                empathy_prompt = construct_empathy_prompt(ema_normalized, emas_normalized[-4:-1][::-1], z_scores, start_time, checkin=True)
                max_tokens = 20
                full_response_1 = get_Claude_response([empathy_prompt], model=model_name, temperature=1.0, max_tokens=max_tokens, return_full_response=True)
                full_response_2 = get_Claude_response([empathy_prompt], model=model_name, temperature=1.0, max_tokens=max_tokens, return_full_response=True)

                adjectives_1, _ = handle_anthropic_response(full_response_1)
                adjectives_2, _ = handle_anthropic_response(full_response_2)

                # Reset the event before showing dialog
                emotion_selection_done_event.clear()
                
                # Emit signal to show dialog
                gui_app.signal.new_emotion_adjectives.emit(adjectives_1)
                emotion_selection_done_event.wait(timeout=60)  # 60 second timeout

                selected_emotion = None
                if not selected_emotion_queue.empty():
                    selected_emotion = selected_emotion_queue.get()

                empathy_response_1 = {"role": "assistant", "content": f"{adjectives_1}, User selected {selected_emotion}", "time": time_since(start_time)//100}

                emotion_selection_done_event.clear()
                gui_app.signal.new_emotion_adjectives.emit(adjectives_2)
                emotion_selection_done_event.wait(timeout=60)

                selected_emotion = None
                if not selected_emotion_queue.empty():
                    selected_emotion = selected_emotion_queue.get()
            
                empathy_response_2 = {"role": "assistant", "content": f"{adjectives_2}, User selected {selected_emotion}", "time": time_since(start_time)//100}

                empathy_transcript.append(empathy_prompt)
                empathy_transcript.append(empathy_response_1)
                empathy_transcript.append(empathy_response_2)

                last_emotion_checkin_time = current_time
            
            else:
                empathy_prompt = construct_empathy_prompt(ema_normalized, emas_normalized[-4:-1][::-1], z_scores, start_time)
                max_tokens = 20
                full_response = get_Claude_response([empathy_prompt], model=model_name, temperature=1.0, max_tokens=max_tokens, return_full_response=True)
                
                adjectives, _ = handle_anthropic_response(full_response)

                empathy_transcript.append(empathy_prompt)
                empathy_transcript.append({"role": "assistant", "content": f"Two adjectives that describe the user's current emotions are {adjectives}", "time": time_since(start_time)//100})
                empathy_queue.put({"role": "system", "content": f"Two adjectives that describe the user's current emotions are {adjectives}", "time": time_since(start_time)//100})



    filename = f"{transcript_path}/{start_time_str}/Emili_raw_EMA_{start_time_str}.txt"
    with open(filename, "w") as file:
        file.write("\n".join(ema_report))
    print(f"EMA Scores report written to {filename}")

    filename = f"{transcript_path}/{start_time_str}/Empathy_{start_time_str}.json"
    with open(filename, "w") as file:
        json.dump(empathy_transcript, file, indent=4)
    print(f"Empathy transcript written to {filename}")

def construct_empathy_prompt(ema, ema_history, z_scores, start_time, checkin=False):
    # empathy_prompt = f"""Analyze the user's emotional state based on facial expression data. The data consists of 7-element vectors representing scores for Anger, Disgust, Fear, Happiness, Sadness, Surprise, and Neutral emotions (in that order). 
    # Using the most recent emotional reading {ema} and previous few readings (from newest to oldest) {ema_history}, provide a brief analysis of:
    # 1. How the user is currently feeling
    # 2. Any significant shift from previous emotional states
    # Focus primarily on the current emotional state. Keep your response concise (1-2 sentences) and objective, providing only essential emotional observations. Don't use any special characters in your response.
    # """

    # empathy_prompt = f"""Analyze the user's emotional state based on facial expression data. The data consists of 7-element vectors representing scores for Anger, Disgust, Fear, Happiness, Sadness, Surprise, and Neutral emotions (in that order). Using the most recent emotional reading {ema} and previous few readings (from newest to oldest) {ema_history}, provide an insightful interpretation that:
    # 1. Goes beyond the obvious emotional labels to infer possible underlying feelings, thoughts, or experiences
    # 2. Makes an educated guess about what might have triggered this emotional state or shift
    # 3. Incorporates subtle emotional nuances that might not be explicitly captured in the primary emotion categories
    # Be willing to take interpretive risks - it's better to be interestingly wrong than boringly accurate. Use vivid language, metaphors, or cultural references that capture emotional textures. Keep your response to 1-2 sentences but make them rich and evocative.
    # """

    # empathy_prompt = f"""Analyze the user's emotional state based on facial expression data. The data consists of 7-element vectors representing scores for Anger, Disgust, Fear, Happiness, Sadness, Surprise, and Neutral emotions (in that order).
    # Using the most recent emotional reading {ema} and previous few readings (from newest to oldest) {ema_history}, along with the user's average emotions {average_emotions}, provide an insightful yet grounded interpretation that:
    # 1. Describes the emotional state in more colorful language than just percentages
    # 2. Suggests a plausible interpretation of the emotional blend being displayed
    # 3. Notes meaningful shifts from previous readings if they exist
    # Be descriptive and interpretive without overreaching. Use specific emotional language that captures nuance, but keep observations reasonably connected to the data. Your response should be 1-2 concise sentences.
    # """

    # if checkin:
    #     empathy_prompt = f"""Analyze the user's emotional state based on facial expression data. The data consists of 7-element vectors representing scores for Anger, Disgust, Fear, Happiness, Sadness, Surprise, and Neutral emotions (in that order), but don't rely on these labels, as every user is different. The most recent emotional reading, normalized by z-score from the user's average emotions, is {ema}. Using this information, respond with two adjectives that might describe how the user is feeling right now. Make sure to respond with exactly two words or phrases, separated by a single comma. Be creative!
    #     """
    # else:
    #     empathy_prompt = f"""Analyze the user's emotional state based on facial expression data. The data consists of 7-element vectors representing scores for Anger, Disgust, Fear, Happiness, Sadness, Surprise, and Neutral emotions (in that order), but don't rely on these labels, as every user is different. The most recent emotional reading, normalized by z-score from the user's average emotions, is {ema}. Using this information, respond with two adjectives that might describe how the user is feeling right now. Make sure to respond with exactly two words or phrases, separated by a single comma.
    #     """

    if checkin:
        empathy_prompt = f"""Analyze the user's emotional state based on facial expression data. The data consists of 7-element vectors representing scores for Anger, Disgust, Fear, Happiness, Sadness, Surprise, and Neutral emotions (in that order), but don't rely on these labels, as every user is different. The most recent emotional reading is {ema}, the z-scores from the user's average emotions is {z_scores}, and previous few readings (from newest to oldest) are {ema_history}. Using this information, respond with two adjectives that might describe how the user is feeling right now. Make sure to respond with exactly two words or phrases, separated by a single comma. Be creative!
        """
    else:
        empathy_prompt = f"""Analyze the user's emotional state based on facial expression data. The data consists of 7-element vectors representing scores for Anger, Disgust, Fear, Happiness, Sadness, Surprise, and Neutral emotions (in that order), but don't rely on these labels, as every user is different. The most recent emotional reading is {ema}, the z-scores from the user's average emotions is {z_scores}, and previous few readings (from newest to oldest) are {ema_history}. Using this information, respond with two adjectives that might describe how the user is feeling right now. Make sure to respond with exactly two words or phrases, separated by a single comma.
        """

    return {"role": "user", "content": empathy_prompt, "time": time_since(start_time)//100}

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
        emo_score_list.append(EMA_queue.get()) # FIFO: first in, first out

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