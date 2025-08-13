import streamlit as st

# --- PAGE CONFIG ---
st.set_page_config(page_title="Blend Podvorica - AI Portfolio", page_icon="🤖", layout="wide")

# --- SIDEBAR ---
st.sidebar.title("🤖 Blend Podvorica Portfolio")

# Quick contact info
st.sidebar.markdown("---")
st.sidebar.subheader("Contact Me")
st.sidebar.write("📞 +383 45 365 467")
st.sidebar.write("✉️ [Email Me](mailto:bpodvorica5@gmail.com)")
st.sidebar.write("💼 [LinkedIn](https://www.linkedin.com/in/blend-podvorica-401855197/)")
st.sidebar.write("💻 [GitHub](https://github.com/gemata)")


# Optional expandable info / tips
st.sidebar.markdown("---")
with st.sidebar.expander("ℹ️ About this Portfolio"):
    st.write(
        "Explore AI/ML projects and full-stack skills. "
        "Click sections above to navigate quickly!"
    )


# --- HERO SECTION ---
col1, col2 = st.columns([0.2, 0.9], gap="small") 

with col1:
    st.image("Profile_Photo.jpeg", width=270)  

with col2:
    st.title("Blend Podvorica", anchor=False)
    st.write("🤖 Junior AI/ML Engineer | 💻 Full-Stack Developer")
    st.info(
        """
        Passionate Computer Science graduate focused expertise in Artificial Intelligence and Machine Learning with a strong foundation in full-stack development. 
        Proven ability to build intelligent, scalable solutions using Python, PyTorch, TensorFlow, Vector Databases and LLMs. 
        Eager to contribute to impactful AI projects.
        """
    )
    try:
        with open("Blend_Podvorica_Resume.pdf", "rb") as pdf_file:
            st.download_button(
                label="📄 Download My Resume",
                data=pdf_file,
                file_name="Blend_Podvorica_Resume.pdf",
                mime="application/pdf",
                key='download_resume_button'
            )
    except FileNotFoundError:
        st.warning("Resume file (Blend_Podvorica_Resume.pdf) not found in the app directory.")

# --- NAVIGATION TABS ---
st.markdown("""
<style>
/* Covers current & older Streamlit tab DOMs */
.stTabs [role="tablist"] button[role="tab"] {
    font-size: 20px !important;   /* label size */
    font-weight: 600 !important;
    padding: 12px 22px !important; /* bigger click area */
}

/* If label text is wrapped in <p>, bump that too */
.stTabs [role="tablist"] button[role="tab"] p {
    font-size: 20px !important;
    font-weight: 600 !important;
}

/* Optionally make the active tab a touch larger */
.stTabs [role="tab"][aria-selected="true"] {
    font-size: 21px !important;
}
</style>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "🔧 Skills", "🚀 Projects", "📬 Contact"])

# --- HOME TAB ---
with tab1:
    st.header("Welcome!", divider='blue')
    st.markdown(
        """
        I'm Blend Podvorica, a Computer Science graduate from UBT with full-stack and AI/ML expertise. I build end-to-end AI-powered applications using Python, Pandas, NumPy, PyTorch, TensorFlow, Scikit-learn, NLP libraries (spaCy, NLTK, Gensim, Transformers), LLMs (GPT, BERT, LLaMA, RAG) with LangChain/LangGraph, CV with CNNs, file parsing (PyMuPDF, pdfminer, openpyxl, docx), semantic search (Pinecone, FAISS, vector embeddings), and full-stack web technologies (HTML, CSS, JS, React, Node.js, Django, Flask, Laravel, SQL/NoSQL). I deliver scalable, intelligent solutions with clean, reusable code and optimized workflows.
        """
    )


# --- SKILLS TAB ---
with tab2:
    st.header("🔧 Technical Skills", divider='blue')

    # --- SKILL CATEGORIES ---
    skill_categories = {
        "AI/ML Frameworks & Libraries": [
            "**PyTorch:** Regression, Classification, Clustering, CNNs, RNNs, LSTMs, Autoencoders, GANs, Performance Tuning (Hyperparameters, Dropout, Batch Norm)",
            "**TensorFlow/Keras:** Regression, Classification, Clustering, CNNs, Performance Tuning",
            "**Scikit-learn:** Core ML algorithms"
        ],
        "Programming & Core Libraries": [
            "**Python:** Clean, reusable code, functions, OOP, decorators, generators",
            "**Pandas:** Data manipulation, analysis",
            "**NumPy:** Numerical computing",
            "**Automation & Scripting:** Workflow optimization"
        ],
        "Natural Language Processing (NLP)": [
            "**Libraries:** spaCy, NLTK, Gensim, Transformers (Hugging Face)",
            "**Techniques:** Tokenization, Lemmatization, POS Tagging, Text Classification, Named Entity Recognition (NER), Text Chunking, Parsing, Data Cleaning"
        ],
        "Large Language Models (LLMs) & Tools": [
            "**Models:** GPT, BERT, LLaMA",
            "**Frameworks/Tools:** LangChain, LangGraph",
            "**Concepts:** Retrieval Augmented Generation (RAG), Prompt Engineering, Transformer Architectures (Attention, Encoder-Decoder)"
        ],
        "Data Handling, Parsing & Search": [
            "**File Handling/Parsing:** PyMuPDF, openpyxl, pdfminer, python-docx",
            "**Semantic Search:** Pinecone, FAISS",
            "**Vector Embeddings & Similarity Search:** Implemented in projects"
        ],
        "Computer Vision (Relevant Experience)": [
            "**Concepts/Models:** CNNs (via PyTorch/TensorFlow)"
        ],
        "MLOps & Deployment": [
            "**Streamlit:** Application deployment (like this portfolio!)",
            "**Docker/Git:** (Mention if you know them, otherwise focus on Streamlit)"
        ],
        "Web Development Background": [
        "**Frontend:** HTML, CSS, JavaScript, Tailwind CSS, React.js, Vue.js, Angular, Angular.js",
        "**Backend:** Node.js (Express.js), Django, Flask, Laravel, PHP",
        "**Databases:** SQL (MySQL, PostgreSQL) and NoSQL (MongoDB) integration",
        "**Full-Stack Projects:** MERN stack and Python web apps",
        "**Other:** REST APIs, JWT Authentication, Web Application Architecture Design"
    ]
    }

    for category, skills in skill_categories.items():
        st.subheader(category, anchor=False)
        for skill in skills:
            st.markdown(f"- {skill}")

# --- PROJECTS TAB ---
with tab3:
    st.header("🚀 My AI Projects", divider='blue')

    # --- PROJECT 1: AI Cold Calling ---
    with st.expander("📞 AI Cold Calling Agent"):
        st.subheader("Project Overview")
        st.write(
            """
            This project demonstrates the application of AI to automate and optimize the cold calling process.
            It integrates NLP and potentially LLMs to analyze lead data, generate personalized scripts,
            and potentially simulate or assist real-time conversations.
            """
        )
        st.subheader("Technologies Used")
        st.write("- Python, Pandas, NumPy")
        st.write("- NLP Libraries (e.g., spaCy)")
        st.write("- LLMs (e.g., OpenAI GPT API, Hugging Face Transformers) via LangChain")
        st.write("- Data Handling (Pandas for lead data)")
        st.write("- Potentially audio processing libraries (if voice is involved)")

        st.subheader("Key Features & My Role")
        st.markdown("""
        - **Lead Analysis:** Processed and analyzed lead data to extract relevant points for personalization.
        - **Script Generation:** Utilized LLMs (e.g., GPT via LangChain) to dynamically generate personalized cold call scripts based on lead profiles.
        - **NLP Processing:** Applied NLP techniques for understanding lead context (spaCy for entity recognition, keyword extraction).
        - **Data Pipeline:** Built robust pipelines for ingesting lead data and managing outputs.
        """)

       # st.subheader("Demo / Visuals")
       #st.write("*Screenshots or an interactive demo of the core functionality would be placed here.*")
        st.subheader("Challenges & Solutions")
        st.markdown("""
        - **Challenge:** Ensuring script personalization was relevant.
        - **Solution:** Implemented multi-level data parsing and specific prompt engineering with LangChain.
        - **Challenge:** Handling diverse lead data formats.
        - **Solution:** Developed robust data cleaning and parsing functions.
        """)
        st.subheader("Results / Impact")
        st.write(
            "Successfully created a functional prototype that could generate personalized scripts, demonstrating potential for time savings and improved lead engagement."
        )

    # --- PROJECT 2: LLM Integration ---
    with st.expander("🤖 LLM Integration & Applications"):
        st.subheader("Project Overview")
        st.write(
            """
            Focused on integrating Large Language Models into various applications and workflows.
            This showcases proficiency in prompt engineering, LangChain/LangGraph orchestration, and applying LLMs to solve specific problems.
            """
        )
        st.subheader("Technologies Used")
        st.write("- Python")
        st.write("- Hugging Face Transformers")
        st.write("- OpenAI API (if used)")
        st.write("- LangChain, LangGraph")
        st.write("- Vector Databases (e.g., Pinecone, FAISS) for RAG")

        st.subheader("Key Features & My Role")
        st.markdown("""
        - **LLM Orchestration:** Used LangChain/LangGraph to build complex workflows involving LLMs.
        - **Retrieval Augmented Generation (RAG):** Implemented RAG systems for context-aware responses using Pinecone/FAISS.
        - **Prompt Engineering:** Designed and optimized prompts for specific tasks (e.g., summarization, Q&A, classification).
        - **Application Integration:** Integrated LLM capabilities into applications (e.g., chatbots, document analyzers).
        """)
        #st.subheader("Demo / Visuals")
        #st.write("*Screenshots or an interactive demo of the LLM application would be placed here.*")
        st.subheader("Challenges & Solutions")
        st.markdown("""
        - **Challenge:** Managing context length and hallucinations.
        - **Solution:** Implemented RAG and careful prompt design.
        - **Challenge:** Structuring complex multi-step LLM interactions.
        - **Solution:** Leveraged LangGraph for defining clear state transitions and logic.
        """)
        st.subheader("Results / Impact")
        st.write(
            "Successfully built robust, context-aware AI applications powered by LLMs, capable of performing complex tasks."
        )

    # --- PROJECT 3: Keyboard Predictive Search ---
    with st.expander("⌨️ Keyboard Predictive Search (PyTorch)"):
        st.subheader("Project Overview")
        st.write(
            """
            Developed a predictive text/word suggestion system, likely using sequence modeling techniques in PyTorch.
            This project demonstrates understanding of deep learning for sequence data and practical implementation.
            """
        )
        st.subheader("Technologies Used")
        st.write("- Python")
        st.write("- PyTorch")
        st.write("- Data Preprocessing (likely Pandas, NumPy)")

        st.subheader("Key Features & My Role")
        st.markdown("""
        - **Model Design:** Designed and implemented a predictive model (e.g., LSTM/RNN) in PyTorch.
        - **Data Preprocessing:** Handled text data preparation for training.
        - **Training & Evaluation:** Trained the model and evaluated its performance (accuracy, speed).
        - **Integration:** Integrated the model into a functional search/prediction interface.
        """)
        #st.subheader("Demo / Visuals")
        #st.write("*Screenshots or a demo of the predictive search in action would be placed here.*")
        st.subheader("Challenges & Solutions")
        st.markdown("""
        - **Challenge:** Achieving low latency for real-time predictions.
        - **Solution:** Optimized model architecture and inference code.
        - **Challenge:** Handling out-of-vocabulary words.
        - **Solution:** Implemented appropriate tokenization and potentially subword units.
        """)
        st.subheader("Results / Impact")
        st.write(
            "Created an efficient predictive search model that provides relevant suggestions, showcasing deep learning skills."
        )

    # --- PROJECT 4: AI Job Portal ---
    with st.expander("💼 AI Job Portal"):
        st.subheader("Project Overview")
        st.write(
            """
            An intelligent job portal leveraging AI for tasks like resume parsing, job description analysis, and semantic matching between candidates and jobs.
            """
        )
        st.subheader("Technologies Used")
        st.write("- Python")
        st.write("- NLP Libraries (spaCy, NLTK, Transformers)")
        st.write("- Semantic Search Libraries (Pinecone, FAISS)")
        st.write("- Data Handling (Pandas, pdfminer, python-docx for parsing)")

        st.subheader("Key Features & My Role")
        st.markdown("""
        - **NLP Processing:** Applied NLP to parse and understand job descriptions and resumes.
        - **Semantic Search & Matching:** Implemented semantic search using Pinecone/FAISS and vector embeddings to match candidates to jobs beyond keyword matching.
        - **Data Pipeline:** Built pipelines for ingesting and processing diverse document formats.
        - **AI-Driven Features:** Potentially included chatbots for candidate interaction or automated screening.
        """)
       # st.subheader("Demo / Visuals")
       # st.write("*Screenshots or a demo of the portal's key features would be placed here.*")
        st.subheader("Challenges & Solutions")
        st.markdown("""
        - **Challenge:** Accurately matching candidates based on skills and experience semantics.
        - **Solution:** Leveraged transformer-based embeddings and semantic search techniques.
        - **Challenge:** Parsing resumes from various formats reliably.
        - **Solution:** Created robust parsing functions using PyMuPDF, pdfminer, docx.
        """)
        st.subheader("Results / Impact")
        st.write(
            "Developed a core component of an intelligent job portal, demonstrating full-stack AI application development and solving a real-world recruitment challenge."
        )

    # --- PROJECT 5: AI Car Showroom Chatbot ---
    with st.expander("🚗 AI Car Showroom Chatbot"):
        st.subheader("Project Overview")
        st.write(
            """
            An AI-powered virtual assistant integrated into a car showroom platform, designed to answer only car-related queries.
            The chatbot provides instant, accurate responses about vehicle specifications, availability, pricing, and comparisons.
            """
        )

        st.subheader("Technologies Used")
        st.write("- Python")
        st.write("- NLP Libraries (spaCy, NLTK, Transformers)")
        st.write("- LLM Integration (GPT, LangChain)")
        st.write("- Vector Databases (Pinecone, FAISS) for semantic search")
        st.write("- Data Parsing (Pandas, PyMuPDF, pdfminer, python-docx)")
        st.write("- Web Integration (Streamlit)")

        st.subheader("Key Features & My Role")
        st.markdown("""
        - **Domain-Specific Chatbot:** Configured LLM to only respond to car-related queries by using a restricted knowledge base.
        - **Vehicle Information Retrieval:** Provided instant answers about specifications, models, availability, and prices.
        - **Semantic Search:** Integrated vector embeddings for more accurate, context-aware query handling.
        - **Data Integration:** Built pipelines to ingest and update vehicle data from various sources.
        - **Web Deployment:** Embedded chatbot seamlessly into a showroom interface for customer interaction.
        """)

        st.subheader("Challenges & Solutions")
        st.markdown("""
        - **Challenge:** Ensuring chatbot avoids non-car related topics.
        - **Solution:** Used retrieval-augmented generation (RAG) with a filtered knowledge base and strict prompt engineering.
        - **Challenge:** Keeping vehicle data up to date.
        - **Solution:** Automated data parsing and updates from official dealership and manufacturer documents.
        """)

        st.subheader("Results / Impact")
        st.write(
            "Delivered a smart, showroom-integrated AI assistant that enhances customer experience, reduces staff workload, and provides accurate, real-time car information — showcasing my ability to combine NLP, LLMs, and vector search in a targeted application."
        )


# --- CONTACT TAB ---
with tab4:
    st.header("📬 Get In Touch", divider='blue')

    st.write("I'm excited about the opportunity of a Intern/Junior position. Feel free to reach out!")
    st.write("**Phone:** +383 45 365 467")
    st.write("**Email:** [bpodvorica5@gmail.com](mailto:bpodvorica5@gmail.com)") 
    st.write("**LinkedIn:** [linkedin.com/in/blend-podvorica-401855197/](https://www.linkedin.com/in/blend-podvorica-401855197/)")
    st.write("**GitHub:** [github.com/gemata](https://github.com/gemata)")



# --- FOOTER ---
st.markdown("---")
st.caption("Built with ❤️ using Python & Streamlit")
