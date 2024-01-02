import whisper

# Load the model
model = whisper.load_model("base")

# Load audio and process in chunks
audio_path = "Python AI.mp3"
chunk_size_seconds = 30  # Adjust as needed
sample_rate = 16000  # Adjust based on your audio data

# Load the audio
audio = whisper.load_audio(audio_path, sample_rate)

# Calculate the total number of chunks needed
num_chunks = len(audio) // (chunk_size_seconds * sample_rate)

# Initialize an empty string to store the final recognized text
final_text = ""

# Iterate through each chunk
for i in range(num_chunks):
    # Calculate the start and end indices for the current chunk
    start_idx = i * chunk_size_seconds * sample_rate
    end_idx = (i + 1) * chunk_size_seconds * sample_rate

    # Extract the current chunk
    chunk = audio[start_idx:end_idx]

    # Make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(chunk).to(model.device)

    # Detect the spoken language
    _, probs = model.detect_language(mel)
    detected_language = max(probs, key=probs.get)
    print(f"Detected language: {detected_language}")

    # Decode the audio
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)

    # Append the recognized text to the final result
    final_text += result.text

# Print the final recognized text
print("Final Recognized Text:")
print(final_text)

# import soundfile as sf
# import whisper
# from pyannote.audio import Pipeline
#
# # Load the Whisper ASR model
# asr_model = whisper.load_model("base")
#
# # Load audio and process in chunks
# audio_path = "Python AI.mp3"
# chunk_size_seconds = 30  # Adjust as needed
# sample_rate = 16000  # Adjust based on your audio data
#
# # Load the audio
# audio = whisper.load_audio(audio_path, sample_rate)
#
# # Calculate the total number of chunks needed
# num_chunks = len(audio) // (chunk_size_seconds * sample_rate)
#
# # Initialize an empty string to store the final recognized text
# final_text = ""
#
# # Instantiate the Pyannote speaker diarization pipeline
# pipeline = Pipeline.from_pretrained(
#     "pyannote/speaker-diarization-3.0",
#     use_auth_token="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"
# )
#
# # Iterate through each chunk
# for i in range(num_chunks):
#     # Calculate the start and end indices for the current chunk
#     start_idx = i * chunk_size_seconds * sample_rate
#     end_idx = (i + 1) * chunk_size_seconds * sample_rate
#
#     # Extract the current chunk
#     chunk = audio[start_idx:end_idx]
#
#     # Make log-Mel spectrogram and move to the same device as the ASR model
#     mel = whisper.log_mel_spectrogram(chunk).to(asr_model.device)
#
#     # Detect the spoken language
#     _, probs = asr_model.detect_language(mel)
#     detected_language = max(probs, key=probs.get)
#     print(f"Detected language: {detected_language}")
#
#     # Save the current chunk to a temporary WAV file using soundfile
#     chunk_path = f"temp_chunk_{i}.wav"
#     sf.write(chunk_path, chunk, sample_rate)
#
#     # Run Pyannote diarization pipeline on the chunk
#     diarization = pipeline(chunk_path)
#
#     # Access the speaker labels
#     speaker_labels = diarization.get_timeline().support()
#
#     # Decode the audio
#     options = whisper.DecodingOptions(fp16=False)
#     result = whisper.decode(asr_model, mel, options)
#
#     # Append the recognized text and speaker labels to the final result
#     final_text += f"Speaker {speaker_labels}: {result.text}\n"
#
# # Print the final recognized text
# print("Final Recognized Text:")
# print(final_text)






