from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp

# instantiate pipeline with batching
pipeline = FlaxWhisperPipline("openai/whisper-large-v2", dtype=jnp.bfloat16, batch_size=16)

# JIT compile the forward call - slow, but we only do once
text = pipeline("Power_English_Update.mp3")

# used cached function thereafter - super fast!!
text = pipeline("Power_English_Update.mp3")