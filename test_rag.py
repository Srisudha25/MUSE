
from rag import process_pdf, retrieve_documents

# Path to the research paper
pdf_path = "data/research_paper.pdf"


def main():

    print("\n===== MUSE: RAG PIPELINE TEST =====\n")

    # Step 1: Process the PDF
    print("Processing research paper...")

    try:
        vectorstore = process_pdf(pdf_path)
        print("Paper indexed successfully!")

    except Exception as e:
        print("Error processing PDF:", e)
        return

    # Step 2: Ask a question
    query = "What is the main objective of this research?"

    print("\nQuery:", query)
    print("\nRetrieving relevant information...")

    try:
        results = retrieve_documents(query)

        # Step 3: Display retrieved information
        print("\n===== RETRIEVED INFORMATION =====\n")

        if not results:
            print("No relevant information found.")
            return

        for i, doc in enumerate(results, start=1):

            print(f"\nResult {i}:")
            print(doc.page_content)

            print("Page:", doc.metadata.get("page", "Unknown"))

            print("-" * 50)

    except Exception as e:
        print("Error retrieving documents:", e)


if __name__ == "__main__":
    main()
