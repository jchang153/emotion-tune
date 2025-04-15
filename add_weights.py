#!/usr/bin/env python3

import os
import re
import glob
import json

def find_transcripts(base_dir="."):
    transcript_dir = os.path.join(base_dir, "transcript")
    if not os.path.exists(transcript_dir):
        return []
    
    files = []
    for folder in os.listdir(transcript_dir):
        folder_path = os.path.join(transcript_dir, folder)
        if os.path.isdir(folder_path) and re.match(r"\d{8}_\d{6}", folder):
            pattern = os.path.join(folder_path, f"*_{folder}_condensed.json")
            files.extend(glob.glob(pattern))
    
    return files

def extract_scores(content):
    start_idx = content.find('[')
    end_idx = content.find(']')
    if start_idx != -1 and end_idx != -1:
        try:
            scores_str = content[start_idx:end_idx+1]
            return json.loads(scores_str)
        except:
            pass
    return None

def create_weighted_file(file_path, weight_vector):
    weighted_path = file_path.replace("_condensed.json", "_weighted.json")
    
    if os.path.exists(weighted_path):
        return None
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None
    
    try:
        for i, message in enumerate(data):
            if message['role'] == 'assistant':
                message['weight'] = 0
                prev_user_idx = None
                for j in range(i-1, -1, -1):
                    if j >= 0 and data[j]['role'] == 'user':
                        prev_user_idx = j
                        break
                
                if prev_user_idx is not None and prev_user_idx > 0:
                    before_message = data[prev_user_idx-1]
                    if (before_message['role'] == 'system' and 'Right now, user looks' in before_message.get('content', '') and '5 seconds after' not in before_message.get('content', '')):
                        before_scores = extract_scores(before_message.get('content', ''))
                        assistant_time = message['time']

                        for j in range(i+1, len(data)):
                            after_message = data[j]

                            if (after_message['role'] == 'system' and 'Right now, user looks' in after_message.get('content', '') and 'time' in after_message and abs(assistant_time - after_message['time']) <= 52):
                                after_scores = extract_scores(after_message.get('content', ''))
                                diff = [a - b for a, b in zip(after_scores, before_scores)]
                                dot_product = sum(d * w for d, w in zip(diff, weight_vector))
                                
                                if dot_product >= 0.1:
                                    message['weight'] = 1
                            break
                
        with open(weighted_path, 'w') as f:
            json.dump(data, f, indent=4)
        return weighted_path
    
    except Exception as e:
        print(f"Error writing to file {weighted_path}: {e}")
        return None

if __name__ == "__main__":
    weight_vector = [-1.5, -5.0, -1.0, 1.0, -1.0, 0.0, 0.3]
    
    files = find_transcripts()
    print(f"Found {len(files)} transcript files.")
    
    print("\nProcessing files...")
    processed = 0
    
    for file in files:
        result = create_weighted_file(file, weight_vector)
        if result:
            processed += 1
            print(f"  Created: {result}")
        else:
            print(f"    Skipped: {file.replace('_condensed.json', '_weighted.json')}")
    
    print(f"\nCreated {processed} weighted files out of {len(files)} found.")