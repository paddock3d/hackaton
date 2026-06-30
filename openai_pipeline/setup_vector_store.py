"""
Upload product datasheets to OpenAI vector store for File Search.
Run once before starting the pipeline.

Usage:
    export OPENAI_API_KEY=sk-proj-...
    python3 setup_vector_store.py
"""
import glob
import os
import sys
from openai import OpenAI

DATASHEETS_DIR = "/opt/agentic/avnet/datasheets"
VECTOR_STORE_NAME = "SmartBOM Electronics Product Datasheets"
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), ".vector_store_id")


def main():
    client = OpenAI()

    pdf_files = sorted(glob.glob(os.path.join(DATASHEETS_DIR, "*.pdf")))
    if not pdf_files:
        print(f"No PDFs found in {DATASHEETS_DIR}")
        sys.exit(1)

    print(f"Found {len(pdf_files)} datasheets")

    vs = client.vector_stores.create(name=VECTOR_STORE_NAME)
    print(f"Created vector store: {vs.id}")

    file_streams = [open(f, "rb") for f in pdf_files]
    try:
        print("Uploading... (this may take a few minutes)")
        batch = client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vs.id,
            files=file_streams,
        )
        print(f"Status: {batch.status}")
        print(f"Completed: {batch.file_counts.completed}/{batch.file_counts.total}")
        if batch.file_counts.failed > 0:
            print(f"Failed: {batch.file_counts.failed}")
    finally:
        for f in file_streams:
            f.close()

    with open(OUTPUT_FILE, "w") as f:
        f.write(vs.id)
    print(f"Vector store ID saved to {OUTPUT_FILE}")
    print(f"\nVector store ID: {vs.id}")
    print("Use this in the pipeline or set OPENAI_VECTOR_STORE_ID env var")


if __name__ == "__main__":
    main()
