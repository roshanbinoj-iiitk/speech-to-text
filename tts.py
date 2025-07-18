import streamlit as st
import requests
import tempfile
import os

# -----------------------------
from pydub import AudioSegment
from pydub.utils import make_chunks
# Streamlit UI
# -----------------------------
st.title("🎙️ Speech-to-Text Converter (Whisper-Large-v3 via Groq API)")
st.write("Upload an audio file and get a Notepad-compatible `.txt` file with the transcription.")


MAX_SIZE_MB = 25
SEGMENT_LENGTH_MS = 5 * 60 * 1000  # 5 minutes in milliseconds

def compress_audio(input_path, output_path, target_bitrate="32k"):
    audio = AudioSegment.from_file(input_path)
    audio.export(output_path, format="mp3", bitrate=target_bitrate)
    return output_path

def compress_audio_to_limit(input_path, output_path, max_size_mb=25, initial_bitrate=32, min_bitrate=8):
    bitrate = initial_bitrate
    while bitrate >= min_bitrate:
        AudioSegment.from_file(input_path).export(output_path, format="mp3", bitrate=f"{bitrate}k")
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        if size_mb <= max_size_mb:
            return output_path
        bitrate -= 8
    return None

def split_audio_to_segments(input_path, segment_length_ms=SEGMENT_LENGTH_MS):
    audio = AudioSegment.from_file(input_path)
    chunks = make_chunks(audio, segment_length_ms)
    segment_paths = []
    for i, chunk in enumerate(chunks):
        segment_path = f"{input_path}_segment_{i}.mp3"
        chunk.export(segment_path, format="mp3", bitrate="32k")
        segment_paths.append(segment_path)
    return segment_paths

# Remove the hardcoded api_key and use Streamlit secrets
api_key = st.secrets["GROQ_API_KEY"]

uploaded_file = st.file_uploader("📤 Upload Audio File (.mp3, .wav, etc)", type=["mp3", "wav", "m4a"])

if uploaded_file and api_key:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as temp_audio:
        temp_audio.write(uploaded_file.read())
        temp_audio_path = temp_audio.name

    file_size_mb = os.path.getsize(temp_audio_path) / (1024 * 1024)
    if file_size_mb > MAX_SIZE_MB:
        st.warning(f"File is {file_size_mb:.2f}MB, compressing to fit under {MAX_SIZE_MB}MB...")
        compressed_path = temp_audio_path + "_compressed.mp3"
        result_path = compress_audio_to_limit(temp_audio_path, compressed_path, MAX_SIZE_MB)
        if result_path is None:
            st.error(f"Compression failed. Please upload a smaller file.")
            st.stop()
        temp_audio_path = result_path
        file_size_mb = os.path.getsize(temp_audio_path) / (1024 * 1024)

    st.audio(temp_audio_path, format="audio/mp3")

    if st.button("📝 Transcribe"):
        st.info("Splitting audio into 5-minute segments...")
        segment_paths = split_audio_to_segments(temp_audio_path)
        full_transcript = ""
        for i, segment_path in enumerate(segment_paths):
            st.info(f"Transcribing segment {i+1}/{len(segment_paths)}...")
            url = "https://api.groq.com/openai/v1/audio/transcriptions"
            files = {
                "file": (f"segment_{i}.mp3", open(segment_path, "rb"), "audio/mp3"),
                "model": (None, "whisper-large-v3")
            }
            headers = {
                "Authorization": f"Bearer {api_key}"
            }
            response = requests.post(url, headers=headers, files=files)
            if response.status_code == 200:
                transcript = response.json().get("text", "")
                full_transcript += f"\n--- Segment {i+1} ---\n{transcript}\n"
            else:
                st.error(f"❌ Error {response.status_code} on segment {i+1}: {response.text}")
                break

        if full_transcript:
            st.success("✅ Transcription successful!")
            txt_file_path = "transcription.txt"
            with open(txt_file_path, "w", encoding="utf-8") as f:
                f.write(full_transcript)
            st.text_area("🧾 Transcribed Text", full_transcript, height=300)
            with open(txt_file_path, "rb") as f:
                st.download_button("⬇️ Download Notepad File", f, file_name="transcription.txt", mime="text/plain")
