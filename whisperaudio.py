import whisper
import os
import json


def save_results(results, output_file):
    with open(output_file, 'w') as f:
        json.dump(results, f)
def transcribe_audio(audio_directory):
    model = whisper.load_model("base")
    results = []

    for filename in os.listdir(audio_directory):
        if filename.endswith(".mp3"):
            audio_path = os.path.join(audio_directory, filename)
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(model.device)
            _, probs = model.detect_language(mel)
            detected_language = max(probs, key=probs.get)
            options = whisper.DecodingOptions(fp16=False)
            result = whisper.decode(model, mel, options)
            transcript = result.text
            results.append({
                'filename': filename,
                'detected_language': detected_language,
                'transcript': transcript
            })

    return results

if __name__ == "__main__":
    audio_directory_path = "cropped_audio"

    transcription_results = transcribe_audio(audio_directory_path)
    save_results(transcription_results, "transcription_results.json")
