<div style="text-align:center;">
  <img src="CustomGPT.gif" alt="Alt text" style="width:100%; height:auto;" />
</div>


### Overview
In this project I trained a transformer from scratch and created a simple UI to interact with the model. For the model architecture I used the tokenizer and transformer libraries from Hugging Face to provide blank models for training. I picked a text version of Moby Dick as the training data. The UI was spun up utilizing Streamlit.
### Training
Utilizing a small model and small amount of training data allowed me to spin up proof of concept locally on my machine. The results showed promise but was still just a jumble of tokens. For longer training I utilized the GPU runtimes on google Colab. Here I was able to get significant improvements on the model. The text from a quick glance resembled the look and feel of the actual book. Closer reading very quickly highlighted it was still nonsense and did not follow a discernible sentence structure.

### UI
The UI for this project was very simple. Just a Streamlit app set up to allow for inference with the model. The model is not set up for chat, but rather completing the user's input text.

### Next Steps
This serves as the basis for two follow up projects I can undertake. First, improving a ChessGPT project I completed a while back. I had trained a GPT in a similar fashion just on chess games a while back. It produced reasonable looking moves but quickly would produce illegal moves. This frame work could be leveraged to try and move that to a state where it produces legal moves a majority of the time. The goal would be to have 80% success rate on generating a full match with no error.

The second project would be an improvement on this model. Rather than starting from scratch, I could look to fine-tune a model using something like LORA or fine-tune the whole model
