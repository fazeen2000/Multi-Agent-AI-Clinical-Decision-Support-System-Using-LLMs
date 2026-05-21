import streamlit as st
from langgraph.graph import StateGraph
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import Tool
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import re

# ✅ Define LLM
llm = ChatAnthropic(model="claude-sonnet-4-20250514", api_key="API-Key")

# ✅ Load PubMedQA Data
with open("/kaggle/working/ori_pqaa.json", "r", encoding="utf-8") as f:  # Make sure the JSON file exists
    data = json.load(f)

       # ✅ Load PubMedQA Data
with open("/kaggle/working/pubmedqa_test.json", "r", encoding="utf-8") as f:  # Make sure the JSON file exists
   test_json = json.load(f)

# ✅ Extract Nested Data
texts = []

for key, value in data.items():
    q = value.get("QUESTION", "").strip()
    ctx = value.get("CONTEXTS", [])
    long_ans = value.get("LONG_ANSWER", "").strip()

    if ctx:
        combined = q + " " + " ".join(ctx)
    else:
        combined = q + " " + long_ans

    texts.append(combined)


print(f"✅ Total valid entries after filtering: {len(texts)}")  # Should now be > 0 

# ✅ Load SentenceTransformer model
# ✅ Load SentenceTransformer models correctly
hf_model = SentenceTransformer("/kaggle/working/SapBERT_pubmed_retriever", device=None)
hf_model = hf_model.to("cpu")

eval_model = SentenceTransformer("thenlper/gte-small", device=None)
eval_model = eval_model.to("cpu")



# ✅ Load FAISS index
faiss_index = faiss.read_index("/kaggle/working/pubmed_final_faiss_ivf.index")
faiss_index.nprobe = 32



# Adjust the `k` parameter
k = 50  # Retrieving top 45 results for better accuracy

# ✅ Define Search Function
def search_pubmedqa(query, k=50):
    query_embedding = hf_model.encode(
        [query], 
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    distances, indices = faiss_index.search(query_embedding, k)
    retrieved_contexts = [texts[i] for i in indices[0]]

    # Higher distance = more similar (FAISS inner product)
    return sorted(zip(distances[0], retrieved_contexts), key=lambda x: x[0], reverse=True)




# ✅ PubMed Retrieval Tool
def pubmed_retriever_tool(query: str) -> str:
    results = search_pubmedqa(query, k=50)
    top_results = sorted(results, key=lambda x: x[0], reverse=True)[:5]  # top 5 by distance
    context = "\n\n".join([f"{i+1}. {text}" for i, (_, text) in enumerate(top_results)])
    return f"📚 Retrieved PubMed Context:\n{context}"

def extract_symptoms_from_text(text):
    # Very basic symptom extraction based on common terms (for demo purposes)
    common_symptoms = ["fever", "cough", "headache", "chest pain", "nausea", "vomiting", "fatigue", "dizziness", "shortness of breath", "palpitations", "rash", "swelling"]
    found = [s for s in common_symptoms if re.search(rf'\b{s}\b', text.lower())]
    return list(set(found))[:5]  # Limit to top 5 symptoms

#retrieval_tool = Tool(
 #   name="PubMedRetriever",
  #  func=pubmed_retriever_tool,
   # description="Retrieves top relevant PubMed entries based on a clinical query.")

# ✅ Format agent responses
# ✅ Clean Response Formatter (already in your code)
def format_agent_output(text: str) -> str:
    cleaned = text.replace('\\n', '\n').replace('\n\n', '\n').strip()
    cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned)  # remove bold markdown
    cleaned = re.sub(r'^[-*]\s*', '', cleaned, flags=re.MULTILINE)  # strip bullet symbols
    lines = cleaned.split('\n')
    formatted = "\n".join(f"- {line.strip().rstrip('.')}" for line in lines if line.strip())
    return formatted





# -------------------- 🧠 AGENTS --------------------
# ✅ Agents
def history_node(state):
    case = state.get("case", "")
    has_history = state.get("has_history", False)
    history_input = state.get("history_input", "")

    if has_history and not history_input.strip():
        history_input = "User indicated history exists but did not enter any details."
    elif not has_history:
        history_input = "No prior history provided."

    return {
        "case": case,
        "has_history": has_history,
        "history_input": history_input
    }



def diagnosis_node(state):
    case = state.get("case", "")
    history = state.get("history_input", "No prior history available.")

    context = pubmed_retriever_tool(case + " " + history)

    # Extract 3–8 word phrases
    raw = context.replace("📚 Retrieved PubMed Context:", "")
    phrases = re.findall(r"\b(?:[A-Za-z][A-Za-z0-9\-]+(?:\s+|$)){3,8}", raw)
    phrases = [p.strip() for p in phrases if len(p.split()) >= 3][:30]

    phrase_bank = "\n".join(f"- {p}" for p in phrases)

    prompt = f"""

    You are a clinical diagnostic assistant.

You are provided with:
- A detailed patient case (which may include symptoms and context)
- Relevant patient history
- Retrieved literature context from PubMed

Your task is to analyze the information and suggest the most likely diagnoses. Focus on the specific symptoms or test results mentioned in the case. Prioritize medically accurate, specific diagnoses over general conditions.

📌 Instructions:
- Provide exactly **3 likely diagnoses**.
- Each diagnosis must include:
  • A specific clinical term (e.g., 'NSTEMI', not 'heart problem')
  • A one-line justification using **at least 3 direct terms or phrases** from the case, history, or PubMed literature.
  • Reuse patient input and PubMed keywords verbatim where possible.
  • Avoid vague or uncertain expressions (e.g., "possibly", "might be", "could").


📌 PHRASE BANK
{phrase_bank}

Patient Case:
{case}

History:
{history}

🩺 Output Format:
- Each line should begin with a diagnosis (bold optional) followed by a justification.
- Use bullet points.
- Use standard clinical language and terminology.

Begin:
"""

    response = llm.invoke(prompt)
    return {
        "case": case,
        "history_input": history,
        "diagnosis": format_agent_output(response.content)
    }



def treatment_node(state):
    case = state["case"]
    diagnosis = state["diagnosis"]

    history = state.get("history_input", "No prior history available.")

    context = pubmed_retriever_tool(case + " " + history)

    raw = context.replace("📚 Retrieved PubMed Context:", "")
    phrases = re.findall(r"\b(?:[A-Za-z][A-Za-z0-9\-]+(?:\s+|$)){3,8}", raw)
    phrases = [p.strip() for p in phrases if len(p.split()) >= 3][:40]

    phrase_bank = "\n".join(f"- {p}" for p in phrases)

    prompt = f"""

    You are a clinical assistant generating treatment plans.

You are provided with:
1. A detailed patient case
2. A confirmed clinical diagnosis
3. A short medical history
4. Retrieved medical literature from PubMed

📌 Your primary goal:
Ensure the treatment reflects **at least 5 clinical keywords or terms** from the input — including symptoms, findings, diagnosis labels, or history content.

📌 Treatment Plan Must:
- Use **exact diagnosis names** as mentioned in the diagnosis step (e.g., 'TIA', 'NSTEMI').
- Use **at least 5 specific phrases verbatim** from:
  • Symptoms in the case
  • Medical history (if available)
  • PubMed literature context
- Ensure **at least 2 of those reused terms** are from the *diagnosis output itself* (e.g., 'positive troponin', 'CT negative for bleed').
- Avoid vague verbs (e.g., 'consider', 'think about', 'might be') unless clinically necessary.
- Each treatment action should reflect an actual medical intervention, monitoring task, or decision-making step.
- Use short, sharp clinical language and clear formatting.

📌 Key Input Phrases (must reuse these in bullet points):

{phrase_bank}


Diagnosis:
{diagnosis}

Patient Case:
{case}

📋 Output Format:
- Each step should begin with a **strong clinical verb** (e.g., Monitor, Prescribe, Order, Refer)
- Use 5 to 8 bullet points
- Avoid vague or general phrases
- Reflect terminology used in case, diagnosis, and PubMed content

Begin:
"""

    response = llm.invoke(prompt)
    return {
    "case": case,
    "diagnosis": diagnosis,
    "history_input": history,
    "treatment": format_agent_output(response.content)
}



# -------------------- 🧩 LangGraph WORKFLOW --------------------
workflow = StateGraph(dict)
workflow.add_node("history", history_node)
workflow.add_node("diagnosis", diagnosis_node)
workflow.add_node("treatment", treatment_node)

workflow.set_entry_point("history")
workflow.add_edge("history", "diagnosis")
workflow.add_edge("diagnosis", "treatment")

compiled_graph = workflow.compile()




import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- 🎛️ STREAMLIT UI --------------------
st.title("K=50 Decision Support")

# Step 1: Patient Case
patient_case = st.text_area(
    "📝 Describe the Patient Case",
    placeholder="Include patient age, gender, main symptoms, lab reports, vitals (e.g. 42-year-old male with chest pain, elevated troponin, BP 150/100)",
    height=150
)

# ✅ Suggestion Box (for better quality input)
with st.expander("📌 Make sure to include:"):
    st.markdown("""
    - Patient **age** and **gender**  
    - Main **symptoms** (duration, severity)  
    - Any known **medical history**  
    - **Test results** (labs, imaging, vitals)  
    - Current **medications** if relevant  
    """)

# Run only if input is given
if patient_case.strip():
    has_history = st.radio("📌 Do you have prior history?", ["Yes", "No"])
    history_text = ""
    if has_history == "Yes":
        history_text = st.text_area("🩺 Enter prior disease/family history")

    if st.button("🔍 Run Diagnosis"):
        with st.spinner("Thinking..."):
            try:
                result = compiled_graph.invoke({
                    "case": patient_case,
                    "has_history": has_history == "Yes",
                    "history_input": history_text
                })

                st.success("✅ Diagnosis Complete")

                case = result.get("case", "⚠️ No case found.")
                history = result.get("history_input", "⚠️ No history provided.")
                diagnosis = result.get("diagnosis", "⚠️ No diagnosis generated.")
                treatment = result.get("treatment", "⚠️ No treatment generated.")

                st.markdown("### 📝 Patient Case")
                st.code(case, language="markdown")

                st.markdown("### 📜 Patient History")
                st.code(history, language="markdown")

                st.markdown("### 🩺 Diagnosis")
                st.code(diagnosis, language="markdown")

                st.markdown("### 💊 Treatment Plan")
                st.code(treatment, language="markdown")

                # --- Correct Evaluation Using Test Data --- #
                # ---------------- MULTI COSINE EVALUATION ---------------- #

                for test_case in test_json:
                    # Use passages from the test dataset for comparison
                    ground_truth_case = test_case.get("QUESTION", "")
                    test_context = test_case.get("CONTEXTS", [])
                    pubmed_passage = "\n".join(test_context)

                    # Get generated outputs from the agent
                    generated_case = result.get("case", "")
                    generated_diagnosis = result.get("diagnosis", "")
                    generated_treatment = result.get("treatment", "")

                    # Compute embeddings
                    pubmed_passage_emb = eval_model.encode([pubmed_passage], normalize_embeddings=True)[0]
                    generated_case_emb = eval_model.encode([generated_case], normalize_embeddings=True)[0]
                    generated_diagnosis_emb = eval_model.encode([generated_diagnosis], normalize_embeddings=True)[0]
                    generated_treatment_emb = eval_model.encode([generated_treatment], normalize_embeddings=True)[0]

                    # Cosine similarity
                    sim_case = cosine_similarity([generated_case_emb], [pubmed_passage_emb])[0][0]
                    sim_diagnosis = cosine_similarity([generated_diagnosis_emb], [pubmed_passage_emb])[0][0]
                    sim_treatment = cosine_similarity([generated_treatment_emb], [pubmed_passage_emb])[0][0]

                    st.markdown(f"### Evaluation Results for Test Case: {ground_truth_case}")
                    st.write(f"**Generated Case Similarity:** {sim_case:.4f}")
                    st.write(f"**Generated Diagnosis Similarity:** {sim_diagnosis:.4f}")
                    st.write(f"**Generated Treatment Similarity:** {sim_treatment:.4f}")
            
            except Exception as e:
                st.error(f"❌ Error during reasoning: {e}")

else:
    st.warning("Please enter a patient case to proceed.")