import random
import re

import openai
from elevenlabs import ElevenLabs, VoiceSettings
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips, vfx, CompositeAudioClip, VideoFileClip

import os
import requests
import base64
import numpy as np
import cv2
import captacity

# Set up your OpenAI and ElevenLabs API keys
openai.api_key = "xxx"
elevenlabs_client = ElevenLabs(api_key="xxx")

# Input story script
# story_script = """
# The Russian Blue cat, known for its striking emerald-green eyes and shimmering blue-gray coat, carries a rich history filled with mystery and fascination. One of the most intriguing stories surrounding the breed dates back to the courts of Russian royalty in the 19th century.
# """

# story_script = """
# In the heart of feudal Japan, long ago in the 16th century, there was a noble samurai named Kenshin Takeda, born into the prestigious Takeda clan, known for its valiant warriors and strategic mastery. Kenshin was not only skilled in the art of combat, but he also possessed a deep sense of honor and wisdom that transcended the battlefield.
# Kenshin's village, nestled between rolling hills and dense forests, was a peaceful haven. The people there lived simple lives, farming and practicing traditional crafts. However, Japan was in turmoil. Rival daimyo, or feudal lords, fought bitterly for power and control, plunging the country into chaos. Kenshin, as a vassal of Lord Takeda, had dedicated his life to protecting his lord’s territory from enemies, particularly the infamous Uesugi clan, a fierce rival.
# """

story_script = """
Dogecoin began as a joke in 2013, inspired by the popular "Doge" meme featuring a Shiba Inu dog. Created by software engineers Billy Markus and Jackson Palmer, it was meant to be a light-hearted, fun alternative to Bitcoin. However, it unexpectedly gained a massive following, thanks to its community's charitable initiatives, such as sponsoring a NASCAR driver and funding clean water projects in Kenya. Over time, Dogecoin evolved into a legitimate cryptocurrency, spiking in popularity with support from high-profile figures, especially Elon Musk, becoming a symbol of internet culture and decentralized finance.
"""

# List of music MP3 URLs
music_urls = [
    "https://s3.robopost.app/mucic-shorts/amalgam-217007.mp3",
    "https://s3.robopost.app/mucic-shorts/better-day-186374.mp3",
    "https://s3.robopost.app/mucic-shorts/coverless-book-lofi-186307.mp3",
    "https://s3.robopost.app/mucic-shorts/drive-breakbeat-173062.mp3",
    "https://s3.robopost.app/mucic-shorts/ethereal-vistas-191254.mp3",
    "https://s3.robopost.app/mucic-shorts/flow-211881.mp3",
    "https://s3.robopost.app/mucic-shorts/for-her-chill-upbeat-summel-travel-vlog-and-ig-music-royalty-free-use-202298.mp3",
    "https://s3.robopost.app/mucic-shorts/groovy-ambient-funk-201745.mp3",
    "https://s3.robopost.app/mucic-shorts/in-slow-motion-inspiring-ambient-lounge-219592.mp3",
    "https://s3.robopost.app/mucic-shorts/lazy-day-stylish-futuristic-chill-239287.mp3",
    "https://s3.robopost.app/mucic-shorts/mellow-future-bass-bounce-on-it-184234.mp3",
    "https://s3.robopost.app/mucic-shorts/midnight-forest-184304.mp3",
    "https://s3.robopost.app/mucic-shorts/movement-200697.mp3",
    "https://s3.robopost.app/mucic-shorts/night-detective-226857.mp3",
    "https://s3.robopost.app/mucic-shorts/nightfall-future-bass-music-228100.mp3",
    "https://s3.robopost.app/mucic-shorts/no-place-to-go-216744.mp3",
    "https://s3.robopost.app/mucic-shorts/perfect-beauty-191271.mp3",
    "https://s3.robopost.app/mucic-shorts/sad-soul-chasing-a-feeling-185750.mp3",
    "https://s3.robopost.app/mucic-shorts/separation-185196.mp3",
    "https://s3.robopost.app/mucic-shorts/solitude-dark-ambient-electronic-197737.mp3"
]

# Create output directories if they don't exist
if not os.path.exists("images"):
    os.makedirs("images")
if not os.path.exists("audio"):
    os.makedirs("audio")

# Split the script into sentences
sentences = [sentence.strip() for sentence in re.split(r'[。.]+', story_script) if sentence.strip()]

# Initialize lists to hold image and audio file paths
image_paths = []
audio_paths = []


# Function to generate an image using OpenAI DALL-E for each sentence
def generate_image_from_text(sentence, context, idx):
    prompt = f"""Generate an image without any text on it that best describes best below target Image Description, with the provided Context:\n\n
Image Description: {sentence} \n\n
Context: {context}"""
    print(f"image prompt: {prompt}")
    response = openai.images.generate(
        model="dall-e-3",
        prompt=prompt,
        n=1,
        size="1024x1792",
        response_format="b64_json",
        quality="standard",
        style="vivid"
    )

    # Download the image
    image_filename = f"images/image_{idx}.jpg"

    for i, d in enumerate(response.data):
        with open(image_filename, "wb") as f:
            f.write(base64.b64decode(d.b64_json))

    return image_filename


# Function to generate speech using ElevenLabs API for each sentence
def generate_audio_from_text(sentence, idx):
    audio = elevenlabs_client.text_to_speech.convert(
        voice_id="pqHfZKP75CvOlQylNhV4",  # Use the appropriate voice ID here
        model_id="eleven_multilingual_v2",
        optimize_streaming_latency="0",
        # language_code="ja",
        output_format="mp3_22050_32",
        text=sentence,
        voice_settings=VoiceSettings(
            stability=0.2,
            similarity_boost=0.8,
            style=0.4,
            use_speaker_boost=True,
        ),
    )

    audio_filename = f"audio/audio_{idx}.mp3"
    with open(audio_filename, "wb") as f:
        for chunk in audio:
            if chunk:
                f.write(chunk)

    return audio_filename


# Initialize variables to track the total duration
total_duration = 0.0
max_duration = 60.0  # Maximum allowed duration in seconds

# Generate images and audio for each sentence
for idx, sentence in enumerate(sentences):
    print(f"Processing sentence {idx + 1}: {sentence}")

    # Generate image from sentence
    image_path = generate_image_from_text(sentence, story_script, idx)
    image_paths.append(image_path)

    # Generate audio from sentence
    audio_path = generate_audio_from_text(sentence, idx)

    # Get the duration of the audio using AudioFileClip
    audio_clip = AudioFileClip(audio_path)
    audio_duration = audio_clip.duration

    # Accumulate total duration
    total_duration += audio_duration

    # If total duration exceeds the limit, stop appending audio and break the loop
    if total_duration > max_duration:
        print(f"Stopping as the total duration exceeds {max_duration} seconds.")
        break

    # Append audio and image paths as long as we're within the time limit
    audio_paths.append(audio_path)

# List to hold individual video clips
video_clips = []


# Define some effects functions
def apply_zoom_in_center(image_clip, duration):
    # Zooms in towards the center of the image
    return image_clip.resize(lambda t: 1 + 0.04 * t)  # Zoom-in effect centered


def apply_zoom_in_upper(image_clip, duration):
    def zoom_effect(get_frame, t):
        max_zoom = 1.2  # Reduced from 1.5
        zoom = 1 + (max_zoom - 1) * (t / duration) ** 2  # Non-linear zoom progression

        # Get the current frame
        frame = get_frame(t)

        # Calculate the new size
        h, w = frame.shape[:2]
        new_h, new_w = int(h * zoom), int(w * zoom)

        # Resize the frame
        zoomed = cv2.resize(frame, (new_w, new_h))

        # Calculate the crop area to focus on the upper part
        y_start = 0
        y_end = h
        x_start = (new_w - w) // 2
        x_end = x_start + w

        # Crop the frame
        cropped = zoomed[y_start:y_end, x_start:x_end]

        return cropped

    return image_clip.fl(zoom_effect)


def apply_zoom_out(image_clip, duration):
    def zoom_out_effect(get_frame, t):
        # Calculate zoom factor: start at 1.5x zoom, decrease to 1x
        zoom_factor = 1.5 - (0.5 * t / duration)
        zoom_factor = max(1, zoom_factor)  # Ensure we don't zoom out beyond original size

        # Get the current frame
        frame = get_frame(t)

        # Calculate the new size
        h, w = frame.shape[:2]
        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)

        # Resize the frame
        zoomed = cv2.resize(frame, (new_w, new_h))

        # Calculate the crop area to keep the frame centered
        y_start = (new_h - h) // 2
        y_end = y_start + h
        x_start = (new_w - w) // 2
        x_end = x_start + w

        # Crop the frame
        cropped = zoomed[y_start:y_end, x_start:x_end]

        return cropped

    return image_clip.fl(zoom_out_effect)


def apply_fade_in_out(image_clip, duration):
    return image_clip.fadein(1).fadeout(1)  # Fade in for 1s, fade out for 1s


def zoom_in_out(image_clip, duration):
    # Define a function to compute the zoom at time t
    def zoom(t):
        scale_factor = 1.5 + 0.5 * np.sin(2 * np.pi * t / duration)  # Oscillate between 1.5 and 2.0
        return scale_factor

    # Apply the zoom to each frame of the clip dynamically
    return image_clip.fl_time(lambda t: zoom(t)).resize(lambda t: zoom(t))


def apply_sinusoidal_zoom(image_clip, duration):
    def sinusoidal_zoom(t):
        cycle_duration = min(duration, 4)  # Use the shorter of 4 seconds or clip duration
        if t < cycle_duration:
            # During the cycle, apply sinusoidal zoom
            zoom_factor = 1 + 0.2 * np.sin(np.pi * t / cycle_duration)
        else:
            # After the cycle, maintain the original size
            zoom_factor = 1
        return zoom_factor

    return image_clip.resize(lambda t: sinusoidal_zoom(t))


# Create a list of available effects
effects = [apply_zoom_in_center, apply_zoom_out, apply_zoom_in_upper, apply_sinusoidal_zoom, apply_fade_in_out]


# Function to resize images to 1080x1920 and maintain aspect ratio
def resize_image_to_1080x1920(image_clip):
    return image_clip.resize(height=1920).on_color(
        size=(1080, 1920),  # Target 1080x1920 size
        color=(0, 0, 0),  # Fill with black if the aspect ratio does not match
        pos='center'  # Center the image
    )


# Keep track of the last applied effect
last_effect = None
effect_index = 0
# Function to crop images to 720x1280 resolution at the center
def crop_image(image_clip):
    # Get the image dimensions
    img_width, img_height = image_clip.size

    # Target dimensions
    target_width = 900
    target_height = 1600

    # Calculate the top-left corner of the cropping box
    x_center = img_width // 2
    y_center = img_height // 2
    x_start = max(0, x_center - target_width // 2)
    y_start = max(0, y_center - target_height // 2)

    # Crop the image to 720x1280 centered
    return image_clip.crop(x1=x_start, y1=y_start, width=target_width, height=target_height)

# Loop through each image and corresponding audio
for image_path, audio_path in zip(image_paths, audio_paths):
    # Load the audio clip
    audio_clip = AudioFileClip(audio_path)

    # Load the image and set the duration to match the audio duration
    image_clip = ImageClip(image_path, duration=audio_clip.duration)

    # Resize or pad the image to 1080x1920
    image_clip = crop_image(image_clip)

    # Apply the next effect in the list, avoiding chaining the same effect
    effect = effects[effect_index]
    image_clip = effect(image_clip, duration=audio_clip.duration)

    # Move to the next effect, ensuring the next effect is not the same
    effect_index = (effect_index + 1) % len(effects)

    # Set the audio to the image clip
    video_clip = image_clip.set_audio(audio_clip)

    # Set the FPS (frames per second) for each clip, e.g., 24 fps
    video_clip = video_clip.set_fps(30)

    # Append the clip to the list
    video_clips.append(video_clip)

# Concatenate all the video clips into one
final_video = concatenate_videoclips([
    clip.audio_fadein(0.01).audio_fadeout(0.01)
    for clip in video_clips
], method="compose")

# Define output video path
output_video_path = "final_output_video_with_effects_1080x1920.mp4"

# Export the final concatenated video without background music
final_video.write_videofile(output_video_path, codec="libx264", audio_codec="aac", fps=30)

# Add captions to the video
captacity.add_captions(
    video_file=output_video_path,
    output_file=f"captions_{output_video_path}",

    # font = "/path/to/your/font.ttf",
    font_size=130,
    font_color="yellow",

    stroke_width=3,
    stroke_color="black",

    shadow_strength=1.0,
    shadow_blur=0.1,

    highlight_current_word=True,
    word_highlight_color="red",

    line_count=1,

    padding=50,
)

# Now load the captioned video
captioned_video_path = f"captions_{output_video_path}"
captioned_video = VideoFileClip(captioned_video_path)

# Randomly select a music track
selected_music_url = random.choice(music_urls)
music_filename = "background_music.mp3"

# Download the selected music track
response = requests.get(selected_music_url)
with open(music_filename, "wb") as f:
    f.write(response.content)

# Load the background music and trim it to match the captioned video duration
background_music = AudioFileClip(music_filename).subclip(0, captioned_video.duration)

# Adjust the background music volume (e.g., reduce it to 20% of original volume)
background_music = background_music.volumex(0.2)  # You can adjust this value based on your needs

# Increase the narration volume by 50% (you can adjust the factor as needed)
narration_audio = captioned_video.audio.volumex(1.5)  # Increase narration volume by 50%

# Now mix the narration audio with the background music
combined_audio = CompositeAudioClip([narration_audio, background_music])

# Set the combined audio for the captioned video
final_captioned_video_with_music = captioned_video.set_audio(combined_audio).subclip(0, final_video.duration - 0.1)

# Define the final output path for the captioned video with music
final_output_path = f"final_captioned_video_with_music_{output_video_path}"

# Export the captioned video with music
final_captioned_video_with_music.write_videofile(final_output_path, codec="libx264", audio_codec="aac", fps=30)
