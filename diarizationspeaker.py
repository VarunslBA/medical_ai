from pyannote.audio import Pipeline
import json

def diarize_audio(input_audio):
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token="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"
    )

    # apply pretrained pipeline
    diarization = pipeline(input_audio)

    # Create a list to store tuples of time intervals, speaker, and transcript
    results = []

    # Store time intervals, speaker, and transcript
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_time = round(turn.start, 1)
        end_time = round(turn.end, 1)
        speaker_id = f"speaker_{speaker}"
        results.append({
            'start_time': start_time,
            'end_time': end_time,
            'speaker_id': speaker_id
        })

    return results


def save_results(results, output_file):
    with open(output_file, 'w') as f:
        json.dump(results, f)