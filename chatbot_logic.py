import os
import re
import unicodedata
import nltk
import difflib
import PyPDF2
import requests
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from collections import Counter

# ===== Config / Env =====
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL = "google/flan-t5-large"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

DATASET_PATH = "data/dataset.txt"
PDF_PATH = "data/PRSC.pdf"
DEBUG = True

# ===== NLTK =====
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words("english"))

# ===== Helpers =====
def remove_stopwords(text):
    return " ".join([t for t in text.split() if t.lower() not in STOPWORDS])

def cheap_stem(w):
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[:-len(suf)]
    return w

def normalize_text(t, lang="english"):
    t = unicodedata.normalize("NFKC", t)
    if lang == "english":
        t = t.lower()
        t = remove_stopwords(t)
        t = re.sub(r'[^a-z0-9\s]', ' ', t)
    elif lang == "punjabi":
        t = re.sub(r'[^\u0A00-\u0A7F\s;|]', ' ', t)
    return re.sub(r"\s+", " ", t).strip()

def tokenize_and_stem(t):
    return set(cheap_stem(w) for w in t.split() if w)

# ===== Question detection =====
QUESTION_WORDS = re.compile(
    r'^(?:what|how|why|when|where|which|explain|define|describe|steps|procedure|guide|workflow|list|features|is|are|can|should|do|does|eligib|qualif|purpose)',
    re.I
)

def score_line_question(line, next_line=""):
    if not line or not line.strip():
        return 0.0
    s = line.strip()
    score = 0.0
    if s.endswith('?'):
        score += 2.0
    if QUESTION_WORDS.match(s):
        score += 1.5
    if re.match(r'^\s*(?:Q(?:uestion)?\s*)?\d{1,3}(?:[\.\)\-:])\s*', s, flags=re.I):
        score += 1.2
    return score

# ===== Headers/Footers blacklist =====
HEADER_FOOTER_PATTERNS = [
    r"^\s*Page\s*\|\s*\d+",
    r".*Punjab Remote Sensing Centre.*",
    r".*e-?Sinchai User Manual.*",
    r".*PISMSUser Manual.*",
    r"^\s*\d{4}$",
    r".*\b2025\b.*",
    r"^\s*Copyright.*",
    r"^\s*All rights reserved.*",
    r"^\s*©.*",
    r'^Page\s*\|\s*\d+', 
    r'^PISMSUser Manual',
    r'^Punjab Remote Sensing Centre',
    r'^\s*$',  # empty lines
]
DEBUG = False

# ===== PDF extraction =====
def robust_extract_pdf_text(pdf_path):
    text = ""
    if not os.path.exists(pdf_path):
        return ""

    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                ptext = page.extract_text()
                if not ptext:
                    continue

                lines = [l.rstrip() for l in ptext.splitlines() if l.strip()]
                clean_lines = []

                for l in lines:
                    # Skip header/footer
                    if any(re.match(pat, l, flags=re.I) for pat in HEADER_FOOTER_PATTERNS):
                        continue

                    # Remove figure references & numeric parentheses
                    # Remove incomplete or complete Figure references
                    # Case 1: (Fig 3), (Figure 12), (Figs. 4-6)
                    l = re.sub(r'\(?\s*Fig(?:ure)?s?[\.:]*\s*\d+[^\)]*\)?', '', l, flags=re.I)

                    # Case 2: incomplete references like "(Fig" at end of line
                    l = re.sub(r'\(\s*Fig(?:ure)?s?\s*$', '', l, flags=re.I)

                    l = re.sub(r'\(\s*\d{1,4}\s*\)', '', l)
                    l = re.sub(r'\(\s*(Fig|Figure)[^)]*\)', '', l, flags=re.I)

                    # Remove empty parentheses/brackets
                    l = re.sub(r'\(\s*\)', '', l)
                    l = re.sub(r'\[\s*\]', '', l)
                    # Remove hidden unicode symbols like , ZERO-WIDTH chars, etc.
                    l = re.sub(r'[\u200B\u200C\u200D\uFEFF\uF000-\uF8FF]', '', l)


                    # General cleanup
                    l = re.sub(r'\s{2,}', ' ', l)
                    l = l.strip(" ,;:-")

                    if len(l.strip()) <= 1:
                        continue
                    if re.fullmatch(r'[\(\)\[\]\s]+', l):
                        continue

                    clean_lines.append(l.strip())

                if clean_lines:
                    text += "\n".join(clean_lines) + "\n\n"

    except Exception as e:
        if DEBUG:
            print("PyPDF2 error:", e)

    return text

# ===== Q/A parsing =====
def parse_pdf_qa_strict(text):
    faqs = {}
    if not text.strip():
        return faqs
    raw_lines = [l.rstrip() for l in text.splitlines()]
    i, n = 0, len(raw_lines)
    while i < n:
        line = raw_lines[i].strip()
        next_line = raw_lines[i + 1] if i + 1 < n else ""
        sc = score_line_question(line, next_line)
        if sc >= 1.2 or line.lower().startswith(("steps", "procedure", "how to")):
            q = line.strip(":").strip()
            a_lines, j = [], i + 1
            while j < n:
                ln = raw_lines[j].strip()
                if not ln:
                    j += 1
                    continue
                if score_line_question(ln, raw_lines[j + 1] if j + 1 < n else "") >= 1.2:
                    break
                a_lines.append(ln)
                j += 1
            ans = "\n".join(a_lines).strip()
            if ans:
                faqs[q.lower()] = ans
            i = j
        else:
            i += 1
    return faqs

# ===== Check reload needed (for app.py compatibility) =====
def check_reload_needed():
    return False

# ===== Global data =====
faq_data = {"english": {}, "punjabi": {}}
questions, answers = [], []
classifier_pipeline, vectorizer, tfidf_matrix = None, None, None
index = {"english": []}
conversation_history = []

# ===== Load + train =====
def load_and_train():
    global faq_data, questions, answers, classifier_pipeline, vectorizer, tfidf_matrix, index
    faq_data = {"english": {}, "punjabi": {}}

    # Load dataset.txt
    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            lang = None
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("[") and line.endswith("]"):
                    lang = line[1:-1].lower()
                    continue
                if "=" in line and lang:
                    q, a = line.split("=", 1)
                    faq_data[lang][q.strip().lower()] = a.strip()

    # Load PDF Q/A
    if os.path.exists(PDF_PATH):
        pdf_text = robust_extract_pdf_text(PDF_PATH)
        parsed_from_pdf = parse_pdf_qa_strict(pdf_text)
        if parsed_from_pdf:
            faq_data["english"].update(parsed_from_pdf)

    # Build ML + TFIDF structures
    questions = list(faq_data["english"].keys())
    answers = list(faq_data["english"].values())
    if questions:
        norm_qs = [normalize_text(q) for q in questions]
        classifier_pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=2000))
        classifier_pipeline.fit(norm_qs, list(range(len(questions))))
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(norm_qs)
        index["english"] = [{"q": q, "nq": normalize_text(q), "tokens": tokenize_and_stem(normalize_text(q)), "a": a} for q, a in faq_data["english"].items()]

    print(f"✅ Loaded {len(questions)} Q/A from dataset + PDF")

# ===== Matching =====
def classify_intent(user_norm):
    if not classifier_pipeline:
        return None
    pred = classifier_pipeline.predict([user_norm])[0]
    prob = max(classifier_pipeline.predict_proba([user_norm])[0])
    return answers[pred] if prob > 0.45 else None

def ml_best_match(user_norm):
    if tfidf_matrix is None or vectorizer is None:
        return None
    sims = cosine_similarity(vectorizer.transform([user_norm]), tfidf_matrix)[0]
    best_idx = sims.argmax()
    return answers[best_idx] if sims[best_idx] > 0.35 else None

def rule_based_match(user_norm, user_tokens):
    if not index["english"]:
        return None
    best_score, best_ans = 0.0, None
    for e in index["english"]:
        union = e["tokens"] | user_tokens
        if not union:
            continue
        score = len(e["tokens"] & user_tokens) / len(union)
        if score > best_score:
            best_score, best_ans = score, e["a"]
    return best_ans if best_score >= 0.22 else None

def format_answer(ans):
    if not ans:
        return ans
    
    # Preserve blank lines but remove trailing spaces
    lines = [l.rstrip() for l in ans.splitlines()]
    
    # Remove lines that are only stray symbols
    cleaned = []
    for l in lines:
        if re.fullmatch(r'[\u200B\u200C\u200D\uFEFF\s]*', l):
            continue
        cleaned.append(l)

    return "\n".join(cleaned)


def force_short_answer(answer):
    """
    Convert full answer into pure short main points.
    - removes paragraphs
    - no explanation sentences
    - only keyword title style bullets
    """

    if not answer:
        return answer

    # Clean all lines
    lines = [l.strip() for l in answer.split("\n") if l.strip()]
    text = " ".join(lines)

    # If already small, return as it is
    if len(text.split()) < 45:
        return answer

    # extract bullets if present
    bullets = []
    for l in lines:
        if l.startswith(("•", "-", "→", "➢")):
            bullet = l.replace("•", "-").replace("➢", "-").replace("→", "-")
            bullets.append(bullet)
    if bullets:
        return "\n".join(bullets)

    # Auto extract main points (keyword style)
    import re
    sentences = re.split(r'[.!?]', text)
    points = []
    for s in sentences:
        words = s.strip().split()
        if len(words) > 3:
            # keep only first 3–5 important words (keyword format)
            main = " ".join(words[:5])
            points.append(f"- {main}")

    # remove duplicate bullet headings
    final = []
    seen = set()
    for p in points:
        if p not in seen:
            seen.add(p)
            final.append(p)

    return "\n".join(final)




def handle_follow_up(query, history):
    """
    Improved follow-up handler:
    - Detects follow-ups like "this / above / previous / brief" in query.
    - Finds the last *meaningful* user question from history (skips short replies and previous follow-ups).
    - Returns a rewritten query that is both explicit and normalized-friendly.
    Expects history as a list of (user_text, bot_text) tuples.
    """
    follow_pattern = r'\b(this|that|it|those|these|above|previous|earlier|mentioned|same|topic|matter|explain again|brief|short|again)\b'
    if not re.search(follow_pattern, query, re.I):
        return query  # not a follow-up

    # Find last meaningful user message (skip short replies & previous follow-up-like messages)
    last_topic = None
    if history:
        for entry in reversed(history):
            if not isinstance(entry, (list, tuple)) or len(entry) == 0:
                continue
            past_user = entry[0].strip()
            if not past_user:
                continue
            # skip trivial utterances
            if len(past_user.split()) < 3:
                continue
            # skip if the past_user itself looks like a follow-up (contains follow words)
            if re.search(follow_pattern, past_user, re.I):
                continue
            last_topic = past_user
            break

    if not last_topic:
        # fallback: use the previous user in history even if short
        if history:
            entry = history[-1]
            if isinstance(entry, (list, tuple)) and len(entry) >= 1:
                last_topic = entry[0]

    if not last_topic:
        return query  # no history to reference

    # Make rewritten query explicit and TF-IDF friendly.
    # Keep both forms to help different matchers: "brief" hint + the original question.
    # Example: "Briefly explain: What is Warabandi Proforma?"
    rewritten = f"Briefly explain: {last_topic}"
    return rewritten


def translate_text(text, target_lang="pa"):
    if not text or target_lang not in ("en", "pa"):
        return text
    try:
        res = requests.get(
            "https://translate.googleapis.com/translate_a/single",
            params={"client": "gtx", "sl": "auto", "tl": target_lang, "dt": "t", "q": text},
            timeout=10,
        )
        if res.status_code == 200:
            return "".join([part[0] for part in res.json()[0]]).strip()
    except Exception as e:
        if DEBUG:
            print("Translation error:", e)
    return text

# ===== RAG conversational section (optional) =====
try:
    from langchain_groq import ChatGroq
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain.chains import create_history_aware_retriever, create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_community.chat_message_histories import ChatMessageHistory
    from langchain_core.runnables.history import RunnableWithMessageHistory
    import os

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore, retriever = None, None

    def init_rag():
        global vectorstore, retriever
        if not os.path.exists(PDF_PATH):
            print("⚠️ PDF file not found:", PDF_PATH)
            return
        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=400)
        chunks = splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        print("✅ RAG retriever initialized successfully.")

    def get_rag_response(query, chat_history=None):
        global retriever
        if retriever is None:
            init_rag()
        if retriever is None:
            return "Sorry, I couldn’t load the knowledge base yet."

        llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Gemma2-9b-It", temperature=0.3)

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question which might reference context, "
            "formulate a standalone question that can be understood without the chat history. "
            "Do NOT answer the question, just rewrite if needed."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        system_prompt = (
            "You are a helpful assistant for Punjab Remote Sensing Centre (PRSC) queries. "
            "Use the provided PDF context to answer precisely. "
            "If you don’t know, politely say so.\n\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        qa_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

        session_history = ChatMessageHistory()
        if chat_history:
            for u, b in chat_history[-6:]:
                session_history.add_user_message(u)
                session_history.add_ai_message(b)

        conversational_chain = RunnableWithMessageHistory(
            rag_chain,
            lambda _: session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        try:
            response = conversational_chain.invoke(
                {"input": query},
                config={"configurable": {"session_id": "default_session"}}
            )
            return response["answer"]
        except Exception as e:
            if DEBUG:
                print("❌ RAG response error:", e)
            return "Sorry, I couldn’t process that query right now."

except Exception:
    if DEBUG:
        print("⚠️ RAG dependencies missing, fallback to FAQ model.")

# ===== Main response =====
def get_response(user_input, lang=None, history=None):
    """
    Hybrid QA: uses history to rewrite follow-ups, then TF-IDF / embeddings / RAG to answer.
    history should be a list of (user_text, bot_text) tuples.
    """
    global conversation_history, vectorizer, tfidf_matrix, questions, answers

    if history is None:
        history = []

    # Ensure data loaded
    if not questions:
        load_and_train()

    # If user_input seems to be a dict/object (when integrated), handle accordingly
    if isinstance(user_input, dict):
        user_input = user_input.get("text", "")

    # 1) Handle follow-ups first — rewrite query if needed using the conversation history
    rewritten = handle_follow_up(user_input, history)
    query = rewritten

    # 2) Language detection & normalization
    lang = lang or ("punjabi" if re.search(r'[\u0A00-\u0A7F]', query) else "english")
    query_en = query if lang == "english" else translate_text(query, "en")
    user_norm = normalize_text(query_en)
    user_tokens = tokenize_and_stem(user_norm)

# DEBUG — show what follow-up rewrote to and what tokens we use
    if DEBUG:
      try:
        print("----DEBUG QUERY----")
        print("raw query:", query)
        print("query_en:", query_en)
        print("normalized:", user_norm)
        print("tokens:", sorted(list(user_tokens))[:20])
      except Exception:
          pass


    # 3) Hybrid search (TF-IDF + embeddings)
    best_answer, best_score = None, 0.0

    # TF-IDF based matching (guarded)
    try:
        if vectorizer is not None and tfidf_matrix is not None and questions:
            query_vec = vectorizer.transform([user_norm])
            cosine_scores = cosine_similarity(query_vec, tfidf_matrix)[0]
            tfidf_idx = int(cosine_scores.argmax())
            tfidf_score = float(cosine_scores[tfidf_idx])
            tfidf_ans = answers[tfidf_idx]
            
            if tfidf_score > best_score:
                best_answer, best_score = tfidf_ans, tfidf_score
    except Exception as e:
        if DEBUG:
            print("TF-IDF error:", e)

    # Embedding similarity fallback (optional)
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from sklearn.metrics.pairwise import cosine_similarity as cs2
        embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        question_embeddings = embedder.embed_documents(questions)
        query_emb = embedder.embed_query(user_norm)
        emb_scores = cs2([query_emb], question_embeddings)[0]
        emb_idx = int(emb_scores.argmax())
        emb_score = float(emb_scores[emb_idx])
        if emb_score > best_score:
            best_answer, best_score = answers[emb_idx], emb_score
    except Exception as e:
        if DEBUG:
            print("Embedding similarity error:", e)

    # Rule-based fuzzy match
    if not best_answer:
        best_answer = rule_based_match(user_norm, user_tokens)

    # RAG fallback
    if not best_answer:
        try:
            rag_input = f"{query}\nContext: {' '.join([u for u, _ in history[-3:]])}"
            rag_ans = get_rag_response(rag_input, chat_history=history)
            if rag_ans and not rag_ans.lower().startswith(("sorry", "couldn't", "i cannot")):
                best_answer = rag_ans
        except Exception as e:
            if DEBUG:
                print("RAG error:", e)

    # Default fallback
    if not best_answer:
        best_answer = "Sorry, I don’t have an answer for that yet. Could you specify the topic again?"

    # Short answer handling requested by user phrase
    short_trigger = r"\b(short|brief|simple|one line|one sentence|definition|in short|short explain|short of above|briefly)\b"

    if re.search(short_trigger, user_input, re.I):
      best_answer = force_short_answer(best_answer)


    # Translate back if needed
    if lang == "punjabi":
        best_answer = translate_text(best_answer, "pa")

    # Update history (store original user_input and bot reply)
    history.append((user_input, best_answer))
    session_history_limit = 50
    if len(history) > session_history_limit:
        history[:] = history[-session_history_limit:]

    return format_answer(best_answer)