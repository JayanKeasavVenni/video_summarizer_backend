import shutil
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from moviepy.editor import *
from gtts import gTTS
import os
import json, openai, pandas
import numpy as np
import warnings
import moviepy.video.io.ImageSequenceClip
import pygame
warnings.filterwarnings('ignore')
import configparser
from fastapi.responses import JSONResponse
from requests import request
from dotenv import load_dotenv
from googleapiclient.discovery import build
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
import cv2
from hugchat.login import Login
from hugchat import hugchat
from hugchat.message import Message

import openai
from fastapi.middleware.cors import CORSMiddleware
import re

load_dotenv()

app = FastAPI()


# Allow all origins, methods, and headers for testing purposes.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get("/")
async def root():
    # text = "Hello, this is a test."
    # tts = gTTS(text=text, lang='en')
    # tts.save("output.mp3")
    # os.system("output.mp3")
    return {"message": "Hello World"}

def create_video(images_folder, text, output_path):
    # Get all image files from the folder
    image_files = sorted([f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    # Create ImageClips from images
    clips = [ImageClip(os.path.join(images_folder, img)).set_duration(3) for img in image_files]

    # Add text narration
    text_clip = TextClip(text, fontsize=30, color='white').set_duration(len(clips) * 3)
    text_clip = text_clip.set_pos(('center', 'bottom'))

    # Concatenate ImageClips and text clip
    final_clip = concatenate_videoclips(clips)
    final_clip = final_clip.set_audio(text_clip.audio)

    # Write the final video to output_path
    final_clip.write_videofile(output_path, codec='libx264', fps=24)

@app.post("/generate_video")
async def generate_video(request: Request) :
    try:
        data = await request.json()
        output_path = "output_video.mp4"
        
        text =  data.get("text", "Default text if not provided in the request")
        images_folder = data.get("images_folder", None)
        with open(images_folder, "wb") as buffer:
            buffer.write(images_folder.file.read())
        create_video(images_folder, text, output_path)
    except Exception as e:
        return str(e)
    
    



# Setting API keys
yt_api_key = os.environ.get("yt_api_key")
huggingface_username = os.environ.get("huggingface_username")
huggingface_pwd = os.environ.get("huggingface_pwd")
openai_api_key = os.environ.get("openai_api_key")

# Set OpenAI API key
openai.api_key = openai_api_key

@app.get("/")
async def root():
    return {"message": "Welcome"}


@app.get("/process_video")
async def process_video(video_url: str, num_frames: int = 10):
    try:
        # Extract video ID from the video link
        video_id = get_youtube_video_id(video_url)
        print("video_id",video_id)
        # Get the transcript for the video
        videoExist = check_file_exists(video_id, ".mp4")
        print(videoExist,"ss")
        if not videoExist : 
            youtube = build('youtube', 'v3', developerKey=yt_api_key)
            captions = youtube.captions().list(part='snippet', videoId=video_id).execute()
            if 'items' in captions and captions['items']:
                caption = captions['items'][0]['id']
            else:
                caption = None

            video_response = youtube.videos().list(part='snippet', id=video_id).execute()
            thumbnails = video_response['items'][0]['snippet']['thumbnails']

            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_txt = ""

            for transcript in transcript_list:
                transcript_txt += transcript['text']

            summary = ""
            images = []

            if transcript_txt != "":
                # Log in to huggingface and grant authorization to huggingchat
                sign = Login(huggingface_username, huggingface_pwd)
                cookies = sign.login()

                # Save cookies to the local directory
                cookie_path_dir = "./cookies_snapshot"
                sign.saveCookiesToDir(cookie_path_dir)

                # Create a ChatBot
                chatbot = hugchat.ChatBot(cookies=cookies.get_dict())

                # Extract the summary from the response
                message = chatbot.query(
                    "Summarize in 10 lines if the given data is more than 10 lines: " + transcript_txt)
                print(message[0],'ytr')
                # Extract the text from the Message object
                summary = message.text if hasattr(
                    message, 'text') else str(message)

            try:
                # Capture frames directly from the YouTube video stream
                output_frames_folder = "frames"
                images = capture_frames(
                    video_url, output_frames_folder, num_frames)
            except Exception as e:
                print(f"Error capturing frames: {e}")
            video_file = None
            if summary is not None and images is not None:
                video_generation = images_to_video("frames", 54, '.png', video_id, '.mp4', summary) 
                if video_generation is not None and 'errors' in video_generation:
                    return JSONResponse(content={ "success": False, "errors": video_generation['errors']})
                else:
                    video_file = video_generation['video_file']
            return JSONResponse(content={ "success": True,"summary": summary, "images": images, "video_file": video_file})
        else :
            video_file=f'{video_id}.mp4'
            return JSONResponse(content={ "success": True, "video_file": video_file})
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing video: {str(e)}")


def capture_frames(video_url, output_folder='frames', interval=10):
    if  os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    else:
        image_paths = []
        # Ensure that the directory exists or create it
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        try:
            # Initialize pytube to capture video stream
            yt = YouTube(video_url)
            stream = yt.streams.filter(file_extension='mp4', res='360p').first()

            # Use OpenCV to capture frames from the video stream
            cap = cv2.VideoCapture(stream.url)

            # Calculate the number of frames to skip based on the desired interval (10 seconds)
            fps = cap.get(cv2.CAP_PROP_FPS)  # Frame rate
            skip_frames = int(fps * interval)
            frame_count = 0
            success, image = cap.read()
            current_frame = 0

            while success:
                # Capture frame if it's on the desired interval
                if current_frame % skip_frames == 0:
                    frame_count += 1
                    frame_path = f"{output_folder}/frame_{frame_count}.png"
                    cv2.imwrite(frame_path, image)
                    image_paths.append(os.path.abspath(frame_path))
                
                # Skip to the next frame for the next interval
                success, image = cap.read()
                current_frame += 1

            cap.release()
            print(f"Frames captured at {interval}-second intervals.")
        except Exception as e:
            print(f"Error capturing frames: {e}")

        return image_paths
def text_to_speech(text):
    # Initialize gTTS with the text to convert
    speech = gTTS(text)

    # Save the audio file to a temporary file
    audio_file = 'summery_audio.mp3'
    speech.save(audio_file)

    # Play the audio file
    return audio_file

def images_to_video(image_folder_path: str, fps, extension:str, video_name:str, output_format:str, summary:str):
    try:
        # Ensure the static folder exists
        if not os.path.exists("static"):
            os.makedirs("static")

        # Assuming text_to_speech returns the path to the generated audio file
        audio_clip_file = text_to_speech(summary) 
        audio_clip = AudioFileClip(audio_clip_file)
        audio_duration = audio_clip.duration

        # List images in the folder
        images = [img for img in os.listdir(image_folder_path) if img.endswith(extension)]
        images.sort()  # Make sure that the images are sorted in order
        images_path = [os.path.join(image_folder_path, img) for img in images]

        # Calculate FPS based on audio duration and number of images
        if images:
            fps = len(images) / audio_duration
        else:
            return {"success": False, "errors": "No images found in the folder"}

        # Create video clip from images
        video_clip = ImageSequenceClip(images_path, fps=fps)

        # If the generated video is longer than the audio, trim the video to match audio duration
        if video_clip.duration > audio_duration:
            video_clip = video_clip.subclip(0, audio_duration)

        # Set audio to the video clip
        final_clip = video_clip.set_audio(audio_clip)

        # Write the final video file, ensuring it's saved within the "static" folder
        final_video_path = f"{video_name}{output_format}"
        # final_clip.write_videofile(final_video_path)
        final_clip.write_videofile(f"static/{final_video_path}")

        return {"success": True, "message": "YouTube Summary video created successfully", "video_file": final_video_path}
    except Exception as e:
        return {"success": False, "errors": f"Error creating video: {str(e)}"}
    

def get_youtube_video_id(link):

  # Regular expression pattern to match the video ID
    pattern = r'(?<=v=)[\w-]+'
    match = re.search(pattern, link)
    if match:
        return match.group(0)
    else:
        return None
    
def check_file_exists(file_name, output_format):
    # Construct the full path where the file would be located
    file_path = os.path.join("static", f"{file_name}{output_format}")
    
    # Check if the file exists
    if os.path.exists(file_path):
        print(f"The file {file_path} already exists.")
        return True
    else:
        print(f"The file {file_path} does not exist.")
        return False