import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import openai
import os
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from dotenv import load_dotenv
import pandas as pd
import re

# Load environment variables from .env file
load_dotenv()

@st.cache_resource
def load_model():
    model_name = 'prithivida/informal_to_formal_styletransfer'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

def t5_infer(tokenizer, model, sentence, direction):
    if direction == "informal_to_formal":
        prefix = "transfer informal to formal: "
    else:
        prefix = "transfer formal to informal: "
    input_text = prefix + sentence
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=64)
    outputs = model.generate(**inputs, max_length=64, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def openai_style_transfer(sentence, direction="informal_to_formal", api_key=None, model="gpt-4o-mini"):
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "[OpenAI API key not set]"
    openai.api_key = api_key
    if direction == "informal_to_formal":
        system_prompt = "You are a helpful assistant that rewrites informal English sentences into formal English."
        user_prompt = f"Rewrite this sentence in a formal style: {sentence}"
    else:
        system_prompt = "You are a helpful assistant that rewrites formal English sentences into informal English."
        user_prompt = f"Rewrite this sentence in an informal style: {sentence}"
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=128
    )
    return response.choices[0].message.content.strip()

def bleu_scores(reference, prediction):
    smoothie = SmoothingFunction().method4
    scores = {}
    for n in range(1, 5):
        weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
        scores[f"BLEU-{n}"] = sentence_bleu([reference], prediction, weights=weights[:n], smoothing_function=smoothie)
    return scores

def llm_judge(original, t5_out, llm_out, direction, api_key=None):
    prompt = f"""You are an expert in English style transfer. Given the following original sentence and two outputs from different models, judge which output is the best {direction.replace('_', ' ')} version. Explain your reasoning in 3-4 lines and give a score (1-10) for each output in the format: 'Score: T5 - X, GPT-4o Mini - Y'.\n\nOriginal: {original}\nFine-tuned T5: {t5_out}\nGPT-4o Mini: {llm_out}\n"""
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "[OpenAI API key not set]"
    openai.api_key = api_key
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a helpful judge for style transfer outputs."},
                  {"role": "user", "content": prompt}],
        max_tokens=256
    )
    return response.choices[0].message.content.strip()

# --- Modern Streamlit UI with BLEU and GPT-4o Judge ---

st.set_page_config(page_title="Style Transfer Comparison", layout="wide")
st.title(":sparkles: Sentence Style Transfer Comparison :sparkles:")
st.write("""
Compare outputs from a fine-tuned T5 (prithivida/informal_to_formal_styletransfer) and OpenAI GPT-4o Mini. The LLM can also act as a judge and provide tone scores. BLEU scores are shown for each output.
""")

sentence = st.text_area("Input Sentence", "yo, gimme the doc quick!", height=80)
direction = st.radio("Direction", ["informal_to_formal", "formal_to_informal"], index=0, horizontal=True)


# --- Session State Logic ---
if 't5_out' not in st.session_state:
    st.session_state['t5_out'] = ''
if 'llm_out' not in st.session_state:
    st.session_state['llm_out'] = ''
if 'bleu_t5' not in st.session_state:
    st.session_state['bleu_t5'] = {}
if 'bleu_llm' not in st.session_state:
    st.session_state['bleu_llm'] = {}
if 'judge' not in st.session_state:
    st.session_state['judge'] = ''
if 'error' not in st.session_state:
    st.session_state['error'] = ''

if st.button(":mag: Compare Models"):
    try:
        tokenizer, model = load_model()
        t5_out = t5_infer(tokenizer, model, sentence, direction)
        llm_out = openai_style_transfer(sentence, direction)
        reference = sentence.split()
        bleu_t5 = bleu_scores(reference, t5_out.split())
        bleu_llm = bleu_scores(reference, llm_out.split())
        st.session_state['t5_out'] = t5_out
        st.session_state['llm_out'] = llm_out
        st.session_state['bleu_t5'] = bleu_t5
        st.session_state['bleu_llm'] = bleu_llm
        st.session_state['judge'] = ''
        st.session_state['error'] = ''
    except Exception as e:
        st.session_state['error'] = f"Error during model comparison: {e}"

col1, col2 = st.columns(2)
if st.session_state['t5_out']:
    with col1:
        st.subheader("Fine-tuned T5 Output")
        st.markdown(f"`{st.session_state['t5_out']}`")
        for n in range(1, 5):
            st.metric(f"BLEU-{n}", f"{st.session_state['bleu_t5'].get(f'BLEU-{n}', 0):.3f}")
if st.session_state['llm_out']:
    with col2:
        st.subheader("OpenAI GPT-4o Mini Output")
        st.markdown(f"`{st.session_state['llm_out']}`")
        for n in range(1, 5):
            st.metric(f"BLEU-{n}", f"{st.session_state['bleu_llm'].get(f'BLEU-{n}', 0):.3f}")

if st.session_state['t5_out'] or st.session_state['llm_out']:
    st.subheader("Model Outputs")
    outputs_df = pd.DataFrame({
        "Model": ["Fine-tuned T5", "GPT-4o Mini"],
        "Output Sentence": [st.session_state['t5_out'], st.session_state['llm_out']]
    })
    st.table(outputs_df)

    st.subheader("BLEU Scores")
    bleu_df = pd.DataFrame({
        "Model": ["Fine-tuned T5", "GPT-4o Mini"],
        "BLEU-1": [st.session_state['bleu_t5'].get('BLEU-1', 0), st.session_state['bleu_llm'].get('BLEU-1', 0)],
        "BLEU-2": [st.session_state['bleu_t5'].get('BLEU-2', 0), st.session_state['bleu_llm'].get('BLEU-2', 0)],
        "BLEU-3": [st.session_state['bleu_t5'].get('BLEU-3', 0), st.session_state['bleu_llm'].get('BLEU-3', 0)],
        "BLEU-4": [st.session_state['bleu_t5'].get('BLEU-4', 0), st.session_state['bleu_llm'].get('BLEU-4', 0)]
    })
    st.table(bleu_df.style.format({col: "{:.3f}" for col in bleu_df.columns if col != "Model"}))

    st.markdown("---")
    st.subheader("LLM Tone Judge (GPT-4o)")
    st.write("Ask GPT-4o to compare the outputs, rate the tone of each (out of 10), and explain its reasoning in 3-4 lines.")
    if st.button("Judge & Score Tone with GPT-4o"):
        try:
            with st.spinner("GPT-4o is evaluating the outputs and tone..."):
                judge = llm_judge(sentence, st.session_state['t5_out'], st.session_state['llm_out'], direction)
            st.session_state['judge'] = judge
            st.session_state['error'] = ''
        except Exception as e:
            st.session_state['error'] = f"Error during LLM judge: {e}"
    if st.session_state['judge']:
        # Try to parse scores and explanation from the judge output
        judge_text = st.session_state['judge']
        score_pattern = r"Score:?\s*T5\s*-\s*(\d+)\s*,\s*GPT-4o Mini\s*-\s*(\d+)"
        match = re.search(score_pattern, judge_text)
        t5_score = gpt_score = None
        if match:
            t5_score = int(match.group(1))
            gpt_score = int(match.group(2))
        explanation = judge_text
        # Remove score line from explanation if present
        explanation = re.sub(score_pattern + ".*", "", explanation, flags=re.DOTALL).strip()
        # Show scores in table and explanation below
        if t5_score is not None and gpt_score is not None:
            score_df = pd.DataFrame({
                "Model": ["Fine-tuned T5", "GPT-4o Mini"],
                "Tone Score (out of 10)": [f"{t5_score}/10", f"{gpt_score}/10"]
            })
            st.table(score_df)
        st.markdown("**Judge's Explanation:**")
        st.write(explanation)

if st.session_state['error']:
    st.error(st.session_state['error'])
