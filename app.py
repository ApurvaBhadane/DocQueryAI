import streamlit as st
import fitz
import pytesseract

from PIL import Image, ImageEnhance, ImageFilter

import io
import os
import re
import hashlib
import pickle

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from sentence_transformers import SentenceTransformer
from openai import OpenAI


# ===================================================================
# PAGE CONFIGURATION
# ===================================================================

st.set_page_config(
    page_title="DocQuery AI",
    page_icon="📄",
    layout="wide"
)

st.markdown("""
<style>
div.stButton > button[kind="primary"] {
    background-color: #A7C7E7 !important;
    color: #1e293b !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

div.stButton > button[kind="primary"]:hover {
    background-color: #91B8D8 !important;
    color: #1e293b !important;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
div.stButton > button[kind="primary"] {
    background-color: #2563eb !important;
    color: white !important;
    border: none !important;
}
</style>
""", unsafe_allow_html=True)




# ==================================================================
# CONSTANTS
# =================================================================e

OCR_CONFIG = "--psm 6"

EMBEDDING_MODEL_NAME = (
    "paraphrase-multilingual-MiniLM-L12-v2"
)

TOP_K = 7



# Starting threshold.
# Tune later using your real PDFs.

SIMILARITY_THRESHOLD = 0.38
SEMANTIC_WEIGHT = 0.75
LEXICAL_WEIGHT = 0.25

# Number of parallel OCR workers
MAX_OCR_WORKERS = min(
    4,
    os.cpu_count() or 2
)

# Cache folder
CACHE_DIR = "docquery_cache"

NOT_FOUND_MESSAGE = (
    "Sorry, I couldn't find this information in the uploaded document."
)


# ============================================================
# CREATE CACHE DIRECTORY
# ============================================================

os.makedirs(
    CACHE_DIR,
    exist_ok=True
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {

    "document_processed": False,

    "document_name": "",

    "document_hash": "",

    "pages": [],

    "chunks": [],

    "embeddings": None,

    "total_pages": 0,

    "ocr_pages": 0,

    "current_question": "",

    "answer": "",

    "retrieved_chunks": [],

    "confidence": 0.0,

    "answer_generated": False,


    "uploader_key": 0

}


for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:

        st.session_state[key] = value

# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource(
    show_spinner="🧠 Loading AI embedding model..."
)
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )


# ============================================================
# DOCUMENT HASH
# ============================================================

def get_document_hash(pdf_bytes):

    return hashlib.sha256(
        pdf_bytes
    ).hexdigest()


# ============================================================
# CACHE PATH
# ============================================================

def get_cache_path(document_hash):

    return os.path.join(
        CACHE_DIR,
        f"{document_hash}.pkl"
    )


# ============================================================
# LOAD DOCUMENT CACHE
# ============================================================

def load_document_cache(document_hash):

    cache_path = get_cache_path(
        document_hash
    )

    if not os.path.exists(cache_path):

        return None

    try:

        with open(
            cache_path,
            "rb"
        ) as file:

            return pickle.load(file)

    except Exception:

        return None


# ============================================================
# SAVE DOCUMENT CACHE
# ============================================================

def save_document_cache(
    document_hash,
    data
):

    cache_path = get_cache_path(
        document_hash
    )

    try:

        with open(
            cache_path,
            "wb"
        ) as file:

            pickle.dump(
                data,
                file
            )

    except Exception as error:

        print(
            "Cache save failed:",
            error
        )


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_ocr_text(text):

    lines = text.splitlines()

    cleaned_lines = []

    for line in lines:

        line = line.strip()

        if not line:

            continue

        # Remove scanner watermark
        if (
            "Scanned with OKEN Scanner"
            in line
        ):

            continue

        cleaned_lines.append(
            line
        )

    return "\n".join(
        cleaned_lines
    )


def normalize_text(text):

    text = text.replace(
        "\x00",
        " "
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# LANGUAGE DETECTION
# ============================================================

def detect_document_language(
    text
):

    """
    Detect basic dominant language from extracted text.

    This is used mainly to decide OCR language.
    """

    if not text:

        return "eng"

    devanagari_chars = len(
        re.findall(
            r"[\u0900-\u097F]",
            text
        )
    )

    latin_chars = len(
        re.findall(
            r"[A-Za-z]",
            text
        )
    )

    # If no useful text is available,
    # default to English.
    if (
        devanagari_chars == 0
        and latin_chars == 0
    ):

        return "eng"

    # If Devanagari is dominant,
    # we need Hindi/Marathi handling.
    if devanagari_chars > latin_chars:

        # Basic Marathi indicators.
        marathi_markers = [
            "आहे",
            "आणि",
            "काय",
            "मध्ये",
            "म्हणजे",
            "यासाठी",
            "तुम्ही",
            "प्रश्न",
            "उत्तर"
        ]

        marathi_score = sum(
            1
            for marker in marathi_markers
            if marker in text
        )

        if marathi_score > 0:

            return "mar"

        return "hin"

    return "eng"


# ============================================================
# OCR LANGUAGE SELECTION
# ============================================================

def get_ocr_languages(
    existing_text=""
):

    language = detect_document_language(
        existing_text
    )

    if language == "mar":

        return "mar"

    if language == "hin":

        return "hin"

    return "eng"


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    image = image.convert(
        "L"
    )

    image = ImageEnhance.Contrast(
        image
    ).enhance(2.0)

    image = image.filter(
        ImageFilter.SHARPEN
    )

    return image


# ============================================================
# EXTRACT ONE PAGE
# ============================================================

def extract_page_from_pdf_bytes(
    pdf_bytes,
    page_index
):

    pdf = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    try:

        page = pdf.load_page(
            page_index
        )

        page_number = (
            page_index + 1
        )


        # ====================================================
        # FIRST: NORMAL PDF TEXT
        # ====================================================

        text = page.get_text(
            "text"
        ).strip()

        if len(text) >= 10:

            text = normalize_text(
                text
            )

            return {
                "page": page_number,
                "text": text,
                "source": "PDF Text"
            }


        # ====================================================
        # SECOND: OCR
        # ====================================================

        pix = page.get_pixmap(

            # 2.5 is faster than 3.0
            # and normally sufficient.
            matrix=fitz.Matrix(
                2.5,
                2.5
            ),

            alpha=False
        )


        image = Image.open(
            io.BytesIO(
                pix.tobytes(
                    "png"
                )
            )
        )


        processed_image = (
            preprocess_image(
                image
            )
        )


        # ----------------------------------------------------
        # OCR LANGUAGE
        # ----------------------------------------------------
        #
        # Since scanned page has no selectable text,
        # use English as default for this page.
        #
        # For Devanagari documents we can detect after
        # OCR and optionally re-run if required.
        #

        ocr_language = "eng"


        ocr_text = (
            pytesseract.image_to_string(

                processed_image,

                lang=ocr_language,

                config=OCR_CONFIG

            )
        )


        ocr_text = clean_ocr_text(
            ocr_text
        )

        ocr_text = normalize_text(
            ocr_text
        )


        # ====================================================
        # IF ENGLISH OCR IS WEAK, TRY DEVANAGARI
        # ====================================================

        if len(ocr_text.strip()) < 20:

            try:

                dev_text = (
                    pytesseract.image_to_string(

                        processed_image,

                        lang="hin+mar",

                        config=OCR_CONFIG

                    )
                )


                dev_text = clean_ocr_text(
                    dev_text
                )

                dev_text = normalize_text(
                    dev_text
                )


                if len(dev_text) > len(
                    ocr_text
                ):

                    ocr_text = dev_text

                    ocr_language = (
                        "hin+mar"
                    )

            except Exception:

                pass


        return {

            "page": page_number,

            "text": ocr_text,

            "source": "OCR",

            "ocr_language": ocr_language

        }

    finally:

        pdf.close()


# ============================================================
# STEP 2 — PARALLEL OCR / PAGE PROCESSING
# ============================================================

def extract_all_pages(
    pdf_bytes,
    total_pages
):

    pages = []

    progress = st.progress(
        0,
        text="Starting document processing..."
    )

    completed = 0


    # ========================================================
    # PARALLEL PROCESSING
    # ========================================================

    with ThreadPoolExecutor(
        max_workers=MAX_OCR_WORKERS
    ) as executor:


        futures = {

            executor.submit(

                extract_page_from_pdf_bytes,

                pdf_bytes,

                page_index

            ): page_index

            for page_index in range(
                total_pages
            )

        }


        for future in as_completed(
            futures
        ):

            result = future.result()

            pages.append(
                result
            )

            completed += 1

            progress.progress(

                completed / total_pages,

                text=(
                    f"Processing pages "
                    f"{completed} / "
                    f"{total_pages}"
                )

            )


    progress.empty()


    # ========================================================
    # RESTORE ORIGINAL PAGE ORDER
    # ========================================================

    pages.sort(
        key=lambda item: item["page"]
    )


    return pages


# ============================================================
# STEP 3 — SMART CHUNKING
# ============================================================

def create_chunks(
    pages
):

    chunks = []

    MAX_WORDS = 220

    MIN_WORDS = 15


    for page in pages:

        page_number = page["page"]

        text = page["text"]

        source = page["source"]


        if not text:

            continue


        # ----------------------------------------------------
        # Split using lines / paragraphs
        # ----------------------------------------------------

        paragraphs = re.split(
            r"\n+",
            text
        )


        current_chunk = []

        current_word_count = 0


        for paragraph in paragraphs:

            paragraph = paragraph.strip()


            if not paragraph:

                continue


            words = paragraph.split()

            word_count = len(
                words
            )


            # ------------------------------------------------
            # QUESTION / SECTION DETECTION
            # ------------------------------------------------

            is_question_start = bool(

                re.match(

                    r"""
                    ^
                    (
                        Q\.?\s*\d+
                        |
                        Question\s*\d+
                        |
                        \d+[\.\)]
                        |
                        [a-zA-Z][\.\)]
                        |
                        [०-९]+[\.\)]
                    )
                    """,

                    paragraph,

                    re.IGNORECASE |
                    re.VERBOSE

                )

            )


            # ------------------------------------------------
            # START NEW LOGICAL CHUNK
            # ------------------------------------------------

            if (

                is_question_start

                and current_chunk

            ):

                chunk_text = "\n".join(
                    current_chunk
                ).strip()


                if len(
                    chunk_text.split()
                ) >= MIN_WORDS:

                    chunks.append({

                        "text": chunk_text,

                        "page": page_number,

                        "source": source

                    })


                current_chunk = []

                current_word_count = 0


            # ------------------------------------------------
            # SIZE LIMIT
            # ------------------------------------------------

            if (

                current_word_count
                + word_count
                > MAX_WORDS

                and current_chunk

            ):

                chunk_text = "\n".join(
                    current_chunk
                ).strip()


                chunks.append({

                    "text": chunk_text,

                    "page": page_number,

                    "source": source

                })


                current_chunk = []

                current_word_count = 0


            current_chunk.append(
                paragraph
            )

            current_word_count += (
                word_count
            )


        # ----------------------------------------------------
        # LAST CHUNK
        # ----------------------------------------------------

        if current_chunk:

            chunk_text = "\n".join(
                current_chunk
            ).strip()


            if chunk_text:

                chunks.append({

                    "text": chunk_text,

                    "page": page_number,

                    "source": source

                })


    return chunks


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(
    chunks
):

    if not chunks:

        return None


    model = load_embedding_model()


    texts = [

        chunk["text"]

        for chunk in chunks

    ]


    embeddings = model.encode(

        texts,

        convert_to_numpy=True,

        normalize_embeddings=True,

        show_progress_bar=False,

        batch_size=32

    )


    return embeddings.astype(
        "float32"
    )


# ============================================================
# STEP 1 — QUERY NORMALIZATION
# ============================================================

def normalize_query(
    question
):

    question = question.lower()


    question = re.sub(

        r"[^\w\s\u0900-\u097F]",

        " ",

        question

    )


    question = re.sub(

        r"\s+",

        " ",

        question

    )


    return question.strip()


# ============================================================
# STEP 1 — LEXICAL SIMILARITY
# ============================================================

def lexical_similarity(
    question,
    text
):

    question_words = set(

        normalize_query(
            question
        ).split()

    )


    text_words = set(

        normalize_query(
            text
        ).split()

    )


    if not question_words:

        return 0.0


    common_words = (

        question_words
        &
        text_words

    )


    return (

        len(common_words)
        /
        len(question_words)

    )


# ============================================================
# STEP 1 — HYBRID SEMANTIC SEARCH
# ============================================================

def search_document(

    question,

    chunks,

    embeddings,

    top_k=TOP_K

):

    if not chunks:

        return []


    if embeddings is None:

        return []


    model = load_embedding_model()


    normalized_question = (
        normalize_query(
            question
        )
    )


    # ========================================================
    # QUESTION EMBEDDING
    # ========================================================

    question_embedding = model.encode(

        [normalized_question],

        convert_to_numpy=True,

        normalize_embeddings=True,

        show_progress_bar=False

    )[0]


    question_embedding = (
        question_embedding
        .astype("float32")
    )


    # ========================================================
    # SEMANTIC SIMILARITY
    # ========================================================

    semantic_scores = np.dot(

        embeddings,

        question_embedding

    )


    results = []


    # ========================================================
    # HYBRID SCORE
    # ========================================================

    for index, chunk in enumerate(
        chunks
    ):

        semantic_score = float(

            semantic_scores[index]

        )


        lexical_score = (
            lexical_similarity(

                question,

                chunk["text"]

            )
        )


        hybrid_score = (

            SEMANTIC_WEIGHT
            *
            semantic_score

            +

            LEXICAL_WEIGHT
            *
            lexical_score

        )


        results.append({

            "text": chunk["text"],

            "page": chunk["page"],

            "source": chunk["source"],

            "score": semantic_score,

            "lexical_score": lexical_score,

            "hybrid_score": hybrid_score

        })


    # ========================================================
    # SORT BY HYBRID SCORE
    # ========================================================

    results.sort(

        key=lambda item:
        item["hybrid_score"],

        reverse=True

    )


    # ========================================================
    # FILTER WEAK RESULTS
    # ========================================================

    filtered_results = [

        item

        for item in results

        if item["score"]
        >= SIMILARITY_THRESHOLD

    ]


    # ========================================================
    # TOP K
    # ========================================================

    return filtered_results[

        :min(

            top_k,

            len(filtered_results)

        )

    ]


# ============================================================
# STEP 4 — RETRIEVAL CONFIDENCE
# ============================================================

def calculate_retrieval_confidence(
    retrieved_chunks
):

    if not retrieved_chunks:

        return 0.0


    top_score = retrieved_chunks[0][
        "score"
    ]


    # Convert approximate cosine score
    # into display percentage.

    confidence = (

        (top_score + 1)
        / 2

    )


    return max(

        0.0,

        min(

            confidence,

            1.0

        )

    )


# ============================================================
# STEP 4 — CONFIDENCE CHECK
# ============================================================

def is_retrieval_confident(
    retrieved_chunks
):

    if not retrieved_chunks:

        return False


    top_score = retrieved_chunks[0][
        "score"
    ]


    return (

        top_score
        >= SIMILARITY_THRESHOLD

    )


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(

    question,

    retrieved_chunks

):

    api_key = os.getenv(
        "OPENROUTER_API_KEY"
    )


    if not api_key:

        return (
            "⚠️ OPENROUTER_API_KEY is not configured."
        )


    # ========================================================
    # STEP 4 — HALLUCINATION PROTECTION
    # ========================================================

    if not is_retrieval_confident(
        retrieved_chunks
    ):

        return NOT_FOUND_MESSAGE


    # ========================================================
    # BUILD CONTEXT
    # ========================================================

    context_parts = []


    for item in retrieved_chunks:

        context_parts.append(

            f"[Page {item['page']}]\n"
            f"{item['text']}"

        )


    context = "\n\n".join(
        context_parts
    )


    # ========================================================
    # OPENROUTER
    # ========================================================

    client = OpenAI(

        base_url=(
            "https://openrouter.ai/api/v1"
        ),

        api_key=api_key

    )


    # ========================================================
    # STRICT RAG PROMPT
    # ========================================================

    prompt = f"""
You are DocQuery AI.

Answer the user's question using ONLY the
document context supplied below.

STRICT RULES:

1. Do NOT use outside knowledge.
2. Do NOT guess.
3. Do NOT invent information.
4. If the answer is not clearly supported by the
   supplied context, return exactly:

"{NOT_FOUND_MESSAGE}"

5. Carefully connect names, questions, headings,
   values, marks, numbers, dates and descriptions.
6. OCR text may contain minor spelling or formatting errors.
7. If an obvious OCR error can be corrected from nearby
   context, interpret it carefully.
8. Preserve important numbers, names and marks accurately.
9. Do NOT create information that is not present.
10. Do NOT generate a page number yourself.
11. Do NOT mention OCR, embeddings, vector search,
    retrieval or internal processing.
12. Keep the answer short and direct.
13. Do NOT explain your reasoning.

DOCUMENT CONTEXT
================
{context}
================

USER QUESTION
=============
{question}

Return only the final answer.
"""


    # ========================================================
    # CALL MODEL
    # ========================================================

    try:

        response = client.chat.completions.create(

            model="openai/gpt-4o-mini",

            messages=[

                {

                    "role": "user",

                    "content": prompt

                }

            ],

            temperature=0

        )


        answer = (

            response

            .choices[0]

            .message

            .content

            .strip()

        )


        return answer


    except Exception as error:

        return (

            "⚠️ Could not generate the answer.\n\n"

            f"Error: {error}"

        )


# ============================================================
# RESET DOCUMENT
# ============================================================

def clear_document():

    st.session_state.document_processed = False

    st.session_state.document_name = ""

    st.session_state.document_hash = ""

    st.session_state.pages = []

    st.session_state.chunks = []

    st.session_state.embeddings = None

    st.session_state.total_pages = 0

    st.session_state.ocr_pages = 0

    st.session_state.current_question = ""

    st.session_state.answer = ""

    st.session_state.retrieved_chunks = []

    st.session_state.confidence = 0.0

    st.session_state.answer_generated = False

    # Reset uploader
    st.session_state.uploader_key += 1


# ============================================================
# PROCESS DOCUMENT
# ============================================================

def process_document(
    uploaded_file
):

    pdf_bytes = uploaded_file.getvalue()


    if not pdf_bytes:

        raise ValueError(
            "Uploaded PDF is empty."
        )


    # ========================================================
    # STEP 5 — HASH
    # ========================================================

    document_hash = get_document_hash(
        pdf_bytes
    )


    # ========================================================
    # IMPORTANT:
    # IF SAME DOCUMENT IS ALREADY IN SESSION
    # DON'T PROCESS AGAIN.
    # ========================================================

    if (

        st.session_state.document_processed

        and

        st.session_state.document_hash
        == document_hash

    ):

        return (

            st.session_state.pages,

            st.session_state.chunks,

            st.session_state.embeddings,

            st.session_state.total_pages,

            st.session_state.ocr_pages

        )


    # ========================================================
    # STEP 5 — CHECK DISK CACHE
    # ========================================================

    cached_data = load_document_cache(
        document_hash
    )


    if cached_data is not None:

        st.info(

            "⚡ Cached document found. "
            "OCR and embedding generation skipped."

        )


        return (

            cached_data["pages"],

            cached_data["chunks"],

            cached_data["embeddings"],

            cached_data["total_pages"],

            cached_data["ocr_pages"]

        )


    # ========================================================
    # OPEN PDF
    # ========================================================

    pdf = fitz.open(

        stream=pdf_bytes,

        filetype="pdf"

    )


    total_pages = pdf.page_count


    pdf.close()


    if total_pages == 0:

        raise ValueError(

            "PDF does not contain any pages."

        )


    # ========================================================
    # STEP 2 — PARALLEL EXTRACTION
    # ========================================================

    pages = extract_all_pages(

        pdf_bytes,

        total_pages

    )


    # ========================================================
    # COUNT OCR PAGES
    # ========================================================

    ocr_pages = sum(

        1

        for page in pages

        if page["source"] == "OCR"

    )


    # ========================================================
    # CHECK READABLE PAGES
    # ========================================================

    readable_pages = [

        page

        for page in pages

        if page["text"].strip()

    ]


    if not readable_pages:

        raise ValueError(

            "No readable text could be extracted."

        )


    # ========================================================
    # STEP 3 — SMART CHUNKING
    # ========================================================

    with st.spinner(

        "🧠 Creating smart document chunks..."

    ):

        chunks = create_chunks(

            readable_pages

        )


    if not chunks:

        raise ValueError(

            "No searchable chunks were created."

        )


    # ========================================================
    # CREATE EMBEDDINGS
    # ========================================================

    with st.spinner(

        "🧠 Creating document embeddings..."

    ):

        embeddings = create_embeddings(

            chunks

        )


    if embeddings is None:

        raise ValueError(

            "Could not create embeddings."

        )


    # ========================================================
    # STEP 5 — SAVE CACHE
    # ========================================================

    cache_data = {

        "pages": pages,

        "chunks": chunks,

        "embeddings": embeddings,

        "total_pages": total_pages,

        "ocr_pages": ocr_pages

    }


    save_document_cache(

        document_hash,

        cache_data

    )


    return (

        pages,

        chunks,

        embeddings,

        total_pages,

        ocr_pages

    )


# ============================================================
# TITLE
# ============================================================

st.title(
    "📄 DocQuery AI"
)

st.write(
    "Ask questions and get answers from normal or scanned PDFs."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "📊 Document Status"
    )


    if st.session_state.document_processed:

        st.success(
            "Document Ready"
        )


        st.write(

            f"📄 **{st.session_state.document_name}**"

        )


        st.write(

            f"Pages: "
            f"**{st.session_state.total_pages}**"

        )


        st.write(

            f"OCR pages: "
            f"**{st.session_state.ocr_pages}**"

        )


        st.write(

            f"Chunks: "
            f"**{len(st.session_state.chunks)}**"

        )


        st.divider()


        if st.button(

            "🗑️ Clear Document",

            use_container_width=True

        ):

            clear_document()

            st.rerun()


    else:

        st.info(
            "Upload and process a PDF."
        )


# ============================================================
# PDF UPLOAD
# ============================================================

st.markdown("""
<style>
[data-testid="stFileUploaderDropzoneInstructions"] small {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)
uploaded_file = st.file_uploader(

    "📤 Upload your PDF document",

    type=["pdf"],

    key=f"pdf_uploader_{st.session_state.uploader_key}"
    
    )


# ============================================================
# PROCESS DOCUMENT
# ============================================================

if uploaded_file is not None:


    uploaded_bytes = (
        uploaded_file.getvalue()
    )


    uploaded_hash = (
        get_document_hash(
            uploaded_bytes
        )
    )


    # ========================================================
    # IF SAME DOCUMENT ALREADY PROCESSED
    # ========================================================

    already_processed = (

        st.session_state.document_processed

        and

        st.session_state.document_hash
        == uploaded_hash

    )


    if already_processed:

        st.success(

            f"✅ **{uploaded_file.name}** "
            "is already processed."

        )


        st.info(

            "You can directly ask questions below."

        )


    else:

        st.success(

            f"✅ Uploaded: "
            f"{uploaded_file.name}"

        )


        if st.button(

            "⚙️ Process Document",

            type="primary"

        ):

            try:

                with st.spinner(
                    "⚙️ Processing document..."
                ):

                    (

                        pages,

                        chunks,

                        embeddings,

                        total_pages,

                        ocr_pages

                    ) = process_document(

                        uploaded_file

                    )


                # ------------------------------------------------
                # SAVE SESSION STATE
                # ------------------------------------------------

                st.session_state.pages = pages

                st.session_state.chunks = chunks

                st.session_state.embeddings = (
                    embeddings
                )

                st.session_state.total_pages = (
                    total_pages
                )

                st.session_state.ocr_pages = (
                    ocr_pages
                )

                st.session_state.document_name = (
                    uploaded_file.name
                )

                st.session_state.document_hash = (
                    uploaded_hash
                )

                st.session_state.document_processed = (
                    True
                )

                st.session_state.answer = ""

                st.session_state.current_question = ""

                st.session_state.retrieved_chunks = []

                st.session_state.answer_generated = (
                    False
                )


                st.success(

                    "✅ Document processed successfully!"

                )


                st.info(

                    f"📄 {total_pages} pages | "

                    f"🔍 {ocr_pages} OCR pages | "

                    f"🧩 {len(chunks)} chunks"

                )


                # Rerun so Process button disappears
                st.rerun()


            except Exception as error:

                clear_document()

                st.error(
                    "❌ Document processing failed."
                )

                st.exception(
                    error
                )


# ============================================================
# SHOW PROCESSED DOCUMENT
# ============================================================

if st.session_state.document_processed:

    st.divider()


    st.subheader(
        "📚 Processed Document"
    )


    st.write(
        "The extracted pages are saved in the current session."
    )


    with st.expander(
        "📖 View Extracted Pages"
    ):


        for page in st.session_state.pages:

            st.markdown(

                f"### 📄 Page {page['page']}"

            )


            st.caption(

                f"Extraction method: "
                f"{page['source']}"

            )


            if page["text"].strip():

                st.code(

                    page["text"]

                )

            else:

                st.warning(

                    "No text could be extracted "
                    "from this page."

                )


# ============================================================
# QUESTION ANSWERING
# ============================================================

if st.session_state.document_processed:

    st.divider()


    st.subheader(
        "💬 Ask a Question"
    )


    # ========================================================
    # QUESTION FORM
    # ========================================================
    #
    # Enter key submits the form.
    #

    with st.form(
        "ask_question_form",
        clear_on_submit=True
    ):


        question = st.text_input(

            "What do you want to know from this document?",

            placeholder=(

    
                "Ask anything about your document..."

            )

        )


        ask_button = st.form_submit_button(

            "🤖 Ask AI",

            type="primary",

            use_container_width=False

        )


    # ========================================================
    # ASK AI
    # ========================================================

    if ask_button:


        if not question.strip():

            st.warning(

                "⚠️ Please enter a question first."

            )


        else:

            # ------------------------------------------------
            # SAVE QUESTION
            # ------------------------------------------------

            st.session_state.current_question = (
                question
            )

            st.session_state.answer_generated = (
                False
            )


            # =================================================
            # STEP 1 — RETRIEVAL
            # =================================================

            with st.spinner(

                "🔎 Searching the document..."

            ):

                retrieved_chunks = (
                    search_document(

                        question=question,

                        chunks=(
                            st.session_state.chunks
                        ),

                        embeddings=(
                            st.session_state.embeddings
                        ),

                        top_k=TOP_K

                    )
                )


            # =================================================
            # STEP 4 — CONFIDENCE
            # =================================================

            confidence = (
                calculate_retrieval_confidence(

                    retrieved_chunks

                )
            )


            st.session_state.retrieved_chunks = (
                retrieved_chunks
            )

            st.session_state.confidence = (
                confidence
            )


            # =================================================
            # GENERATE ANSWER
            # =================================================

            with st.spinner(

                "🤖 Preparing your answer..."

            ):

                answer = generate_answer(

                    question,

                    retrieved_chunks

                )


            st.session_state.answer = answer

            st.session_state.answer_generated = (
                True
            )


            # =================================================
            # RERUN
            # =================================================

            # This makes the answer appear cleanly
            # below the question section.

            st.rerun()


# ============================================================
# SHOW ANSWER
# ============================================================

if (

    st.session_state.document_processed

    and

    st.session_state.answer_generated

):

    st.divider()


    st.subheader(
        "🤖 Answer"
    )


    answer = (
        st.session_state.answer
    )


    if answer == NOT_FOUND_MESSAGE:

        st.warning(
            answer
        )


    elif answer.startswith(
        "⚠️"
    ):

        st.error(
            answer
        )


    else:

        st.success(
            answer
        )


    # ========================================================
    # CONFIDENCE
    # ========================================================

    confidence = (
        st.session_state.confidence
    )


    if (
        st.session_state.retrieved_chunks
    ):

        st.caption(

            f"Retrieval confidence: "
            f"{confidence * 100:.1f}%"

        )


    # ========================================================
    # SOURCE PAGES
    # ========================================================

    retrieved_chunks = (
        st.session_state.retrieved_chunks
    )


    if retrieved_chunks:

        source_pages = sorted(

            set(

                item["page"]

                for item in retrieved_chunks

            )

        )


        st.subheader(
            "📄 Source Page(s)"
        )


        page_text = ", ".join(

            f"Page {page}"

            for page in source_pages

        )


        st.info(
            page_text
        )


        # ====================================================
        # RETRIEVED SECTIONS
        # ====================================================

        with st.expander(

            "🔎 View Retrieved Sections"

        ):


            for item in retrieved_chunks:

                st.markdown(

                    f"**📄 Page {item['page']}**  \n"

                    f"Semantic similarity: "
                    f"{item['score']:.3f}  \n"

                    f"Lexical score: "
                    f"{item['lexical_score']:.3f}  \n"

                    f"Hybrid score: "
                    f"{item['hybrid_score']:.3f}"

                )


                st.code(

                    item["text"]

                )


st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: transparent !important;
        color: white !important;
        text-align: center;
        padding: 8px 0;
        font-size: 13px;
        z-index: 9999;
    }
    </style>

    <div class="footer">
        © 2026 DocQuery AI • Built by Apurva_Bhadane
    </div>
    """,
    unsafe_allow_html=True
)