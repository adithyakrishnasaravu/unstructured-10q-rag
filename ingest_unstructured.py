# intialize the client, and ingest the data by sending in each file
import os, json
from pathlib import Path
import unstructured_client as client
from unstructured_client.models import operations,shared
from unstructured_client.models.errors import SDKError
from dotenv import load_dotenv

load_dotenv()
client = client.UnstructuredClient(api_key_auth=os.getenv('UNSTRUCTURED_API_KEY')) 

def ingest_data(input_folder='data/', output_folder='outputs/unstructured_json/'):
    # Iterate over all PDF files in the input directory
    pdfs = list(Path(input_folder).glob("*.pdf"))
    for pdf in pdfs:
        try:
            with open(pdf, 'rb') as f:
                file_content = f.read()
            
            # create partition request with Unstructured API
            # we use the HI_RES strategy. This extracts structured elements like text blocks and tables.
            response = client.general.partition(
                request=operations.PartitionRequest(
                    partition_parameters=shared.PartitionParameters(
                        files=shared.Files(
                        content=file_content,
                        file_name=pdf.name,
                        ),
                        strategy=shared.Strategy.HI_RES,
                        split_pdf_page=True,
                        split_pdf_allow_failed=True,
                        split_pdf_concurrency_level=15,
                    )
                )
            )

            # convert elements to dictionary-JSON format
            elements = []
            for element in response.elements:
                elements.append(
                    {
                    'type': element.get('type'),
                    'text': element.get('text'),
                    'metadata': element.get('metadata', {})                        
                    }
                )
            
            # save the data to a file
            output_file = Path(output_folder) / f"{pdf.stem}.json"
            with open(output_file, "w") as f:
                json.dump(elements, f, indent=2, ensure_ascii=False)
            print(f"✓ Saved {len(elements)} elements to {output_file}")
        except SDKError as e:
            print(f"Failed to ingest {pdf.name}: {e}")
    print("Data ingestion completed.")

if __name__ == "__main__":#
    ingest_data()