# # instantiate the pipeline
# from pyannote.audio import Pipeline
# pipeline = Pipeline.from_pretrained(
#   "pyannote/speaker-diarization-3.0",
#   use_auth_token="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")
#
# # run the pipeline on an audio file
# diarization = pipeline("Python_AI.mp3")
# print(diarization)
#
# # dump the diarization output to disk using RTTM format
# with open("audio.rttm", "w") as rttm:
#     diarization.write_rttm(rttm)
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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")

# # send pipeline to GPU (when available)
# import torch
# pipeline.to(torch.device("cuda"))

# apply pretrained pipeline
diarization = pipeline("Python_AI.mp3")

# print the result
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
# start=0.2s stop=1.5s speaker_0
# start=1.8s stop=3.9s speaker_1
# start=4.2s stop=5.7s speaker_0
# ...

