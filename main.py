import spacy
from sentence_transformers.util import cos_sim
import streamlit as st


@st.experimental_singleton
def load_arabert_model():
    nlp = spacy.blank("ar")
    config = {
        "model": {
            "@architectures": "spacy-transformers.TransformerModel.v3",
            "name": "aubmindlab/bert-base-arabertv02"
        }
    }
    nlp.add_pipe("transformer", config=config)
    nlp.initialize() # XXX don't forget this step!
    return nlp


nlp = load_arabert_model()


egypt = nlp('مصر')
tunis = nlp('تونس')
sim = cos_sim(egypt._.trf_data.tensors[1], tunis._.trf_data.tensors[1])

st.info(f"{egypt} vs {tunis} -> {float(sim)}")
