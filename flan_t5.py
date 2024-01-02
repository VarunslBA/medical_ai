# from transformers import T5Tokenizer, T5ForConditionalGeneration
#
# tokenizer = T5Tokenizer.from_pretrained("kabita-choudhary/finetuned-bart-for-conversation-summary")
# model = T5ForConditionalGeneration.from_pretrained("kabita-choudhary/finetuned-bart-for-conversation-summary")
#
# #input_text = "summarize: The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country. It'll lower the deficit and ask the ultra-wealthy and corporations to pay their fair share. And no one making under $400,000 per year will pay a penny more in taxes."
# input_text = '''Vertex AI offers access to Gemini, a multimodal model from Google DeepMind, capable of understanding virtually any input, combining different types of information, and generating almost any output. Prompt and test in Vertex AI with Gemini, using text, images, video, or code. Using Gemini’s advanced reasoning and state-of-the-art generation capabilities, developers can try sample prompts for extracting text from images, converting image text to JSON, and even generate answers about uploaded images to build next-gen AI applications.'''
#
# input_ids = tokenizer(input_text, return_tensors="pt").input_ids
#
# outputs = model.generate(input_ids)
# print(tokenizer.decode(outputs[0]))
from transformers import pipeline

# Load the summarization pipeline with the pre-trained model
summarizer = pipeline("summarization", model="kabita-choudhary/finetuned-bart-for-conversation-summary")

# Your conversation or text to be summarized
# conversation = """
# Speaker 1: Hello, how are you?
# Speaker 2: I'm good, thanks for asking. How about you?
# Speaker 1: I'm doing well too. Did you hear about the new project at work?
# Speaker 2: Yes, I did. It sounds interesting but challenging.
# Speaker 1: Exactly, I think it will be a great opportunity for us to showcase our skills.
# """
conversation = '''
Patient – Good Morning, doctor. May I come in?

Doctor – Good Morning. How are you? You do look quite pale this morning.

Patient – Yes, doctor. I’ve not been feeling well for the past few days. I’ve been having a stomach ache for a few days and feeling a bit dizzy since yesterday.

Doctor – Okay, let me check. (applies pressure on the stomach and checks for pain) Does it hurt here?

Patient – Yes, doctor, the pain there is the sharpest.

Doctor – Well, you are suffering from a stomach infection, that’s the reason you are having a stomach ache and also getting dizzy. Did you change your diet recently or have something unhealthy?

Patient – Actually, I went to a fair last week and ate food from the stalls there.

Doctor – Okay, so you are probably suffering from food poisoning. Since the food stalls in fairs are quite unhygienic, there’s a high chance those uncovered food might have caused food poisoning.

Patient – I think I will never eat from any unhygienic place in the future.

Doctor – That’s good. I’m prescribing some medicines, have them for one week and come back for a checkup next week. And please try to avoid spicy and fried foods for now.

Patient – Okay, doctor, thank you.

Doctor – Welcome.
'''

# conversation = '''
# Vertex AI offers access to Gemini, a multimodal model from Google DeepMind, capable of understanding virtually any input, combining different types of information, and generating almost any output. Prompt and test in Vertex AI with Gemini, using text, images, video, or code. Using Gemini’s advanced reasoning and state-of-the-art generation capabilities, developers can try sample prompts for extracting text from images, converting image text to JSON, and even generate answers about uploaded images to build next-gen AI applications.
# '''


# Generate a summary
summary = summarizer(conversation, max_length=150, min_length=50, length_penalty=2.0, num_beams=4)

# Print the summarized output
print("Summary:", summary[0]['summary_text'])

