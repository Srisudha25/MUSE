
from dotenv import load_dotenv
from pathlib import Path
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from rag import retrieve_documents

# Load API key
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("Gemini API key not found!")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=API_KEY,
    temperature=0.3
)

# Output parser
parser = StrOutputParser()


# Common function to retrieve paper context
def get_context(query):

    documents = retrieve_documents(query)

    context = ""

    for doc in documents:
        page = doc.metadata.get("page", "Unknown")

        context += (
            f"\n[Page {page + 1 if isinstance(page, int) else page}]\n"
            f"{doc.page_content}\n"
        )

    return context


# Create an LCEL chain
def create_chain(system_prompt):

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Research paper context:\n{context}\n\nUser request:\n{query}")
    ])

    chain = prompt | llm | parser

    return chain


# AGENT 1: SUMMARIZER
summarizer_chain = create_chain("""
You are the Summarizer Agent of MUSE.

Your responsibility is to create a structured summary
of an academic research paper.

Include:
1. Research title
2. Abstract and overview
3. Problem statement
4. Research objectives
5. Methodology
6. Key findings
7. Conclusion

Use clear academic language.

Only use information supported by the supplied context.
If information is missing, explicitly mention that.
Do not invent research findings.
""")


def summarizer_agent(query="Summarise this research paper"):

    context = get_context(query)

    return summarizer_chain.invoke({
        "context": context,
        "query": query
    })


# AGENT 2: CONCEPT EXTRACTOR
concept_chain = create_chain("""
You are the Concept Extractor Agent of MUSE.

Identify the important concepts in the research paper.

Extract:
1. Main research problem
2. Key contributions
3. Algorithms or techniques used
4. Dataset or materials used
5. Methodology
6. Experimental results
7. Limitations
8. Future scope

Present the information using clear headings and bullet points.

Only use evidence from the supplied research paper.
Do not fabricate details.
""")


def concept_extractor_agent(query="Extract the key concepts"):

    context = get_context(query)

    return concept_chain.invoke({
        "context": context,
        "query": query
    })


# AGENT 3: SLIDE GENERATOR
slide_chain = create_chain("""
You are the Slide Generator Agent of MUSE.

Convert the research paper into a presentation-ready slide outline.

Create approximately 8 slides.

For every slide provide:
- Slide number
- Slide title
- 3 to 5 concise bullet points
- Suggested visual, chart or diagram where appropriate

Suggested structure:
1. Title
2. Introduction
3. Problem Statement
4. Objectives
5. Methodology
6. Results
7. Discussion
8. Conclusion and Future Scope

Keep the content concise and suitable for a college presentation.

Only use information supported by the supplied context.
Do not invent results or statistics.
""")


def slide_generator_agent(query="Generate presentation slides"):

    context = get_context(query)

    return slide_chain.invoke({
        "context": context,
        "query": query
    })


# AGENT 4: VIVA AGENT
viva_chain = create_chain("""
You are the Viva Agent of MUSE.

Act as an academic examiner who has carefully studied
the research paper.

Generate 10 likely viva questions.

Include a mixture of:
- Basic understanding questions
- Methodology questions
- Technical questions
- Results and interpretation questions
- Critical thinking questions

For every question provide:
1. The question
2. A suggested model answer
3. Difficulty level (Easy, Medium or Hard)

Answers must be accurate, clear and grounded
in the supplied research paper.

Do not invent information.
""")


def viva_agent(query="Generate likely viva questions"):

    context = get_context(query)

    return viva_chain.invoke({
        "context": context,
        "query": query
    })


# Test all four agents
if __name__ == "__main__":

    print("\n===== MUSE AGENT TEST =====")

    agents = {
        "Summarizer": summarizer_agent,
        "Concept Extractor": concept_extractor_agent,
        "Slide Generator": slide_generator_agent,
        "Viva": viva_agent
    }

    for name, agent in agents.items():

        print(f"\n\n===== {name.upper()} AGENT =====\n")

        try:
            result = agent()
            print(result)

        except Exception as e:
            print(f"Error in {name} Agent: {e}")
