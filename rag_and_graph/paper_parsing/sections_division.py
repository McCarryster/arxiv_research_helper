import json
import requests
from typing import List, Dict, Optional
import os
from lxml import etree # type: ignore

def parse_pdf_with_grobid(pdf_path: str, grobid_url: str = "http://localhost:8070", timeout: int = 60) -> Optional[List[Dict]]:
    grobid_endpoint = f"{grobid_url}/api/processFulltextDocument"
    try:
        # Open and send the PDF file to GROBID
        with open(pdf_path, 'rb') as pdf_file:
            files = {
                'input': (pdf_path, pdf_file, 'application/pdf')
            }
            
            # Parameters for GROBID processing
            data = {
                'consolidateHeader': '0',
                'consolidateCitations': '0',
                'teiCoordinates': '0',
                'segmentSentences': '0'
            }
            
            # Send request to GROBID
            response = requests.post(
                grobid_endpoint,
                files=files,
                data=data,
                timeout=timeout
            )
            
            if response.status_code == 200:
                # Parse the TEI XML response
                sections = parse_grobid_tei(response.text)
                return sections
            else:
                print(f"GROBID error: {response.status_code} - {response.text}")
                return None
                
    except FileNotFoundError:
        print(f"PDF file not found: {pdf_path}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def parse_grobid_tei(tei_xml: str) -> List[Dict]:
    sections = []
    try:
        # Parse the XML
        parser = etree.XMLParser(recover=True)
        root = etree.fromstring(tei_xml.encode('utf-8'), parser=parser)
        
        # Define namespaces
        namespaces = {
            'tei': 'http://www.tei-c.org/ns/1.0'
        }
        
        # Find all div elements (sections)
        divs = root.xpath('//tei:text/tei:body/tei:div', namespaces=namespaces)
        
        for div in divs:
            # Extract section title
            head = div.xpath('.//tei:head', namespaces=namespaces)
            title = head[0].text if head else "Untitled Section"
            
            # Extract all paragraph text from the section
            paragraphs = div.xpath('.//tei:p', namespaces=namespaces)
            text_content = []
            
            for p in paragraphs:
                # Get all text content from the paragraph, including nested elements
                paragraph_text = ''.join(p.itertext()).strip()
                if paragraph_text:
                    text_content.append(paragraph_text)
            
            # Combine paragraphs into section text
            section_text = '\n\n'.join(text_content)
            
            if title and section_text:  # Only include sections with both title and content
                sections.append({
                    "section": title.strip(),
                    "text": section_text.strip()
                })
        
        return sections
        
    except Exception as e:
        print(f"Error parsing TEI XML: {e}")
        return []

# pdf_folder = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_agent/arxiv_pdfs/1308.0850v5.pdf"
# print(parse_pdf_with_grobid(pdf_folder))


# # # Example usage and helper function to save as JSON
# # def parse_pdf_to_json(pdf_path, output_json_path):
# #     sections = parse_pdf_with_grobid(pdf_path)
# #     if sections:
# #         json_output = json.dumps(sections, indent=2, ensure_ascii=False)
# #         if output_json_path:
# #             with open(output_json_path, 'w', encoding='utf-8') as f:
# #                 f.write(json_output)
# #             print(f"Sections saved to: {output_json_path}")
# #         return json_output
# #     else:
# #         print("Failed to parse PDF")
# #         return None

# # if __name__ == "__main__":
# #     # Folder containing PDFs
# #     pdf_folder = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_agent/arxiv_pdfs"
# #     output_folder = "/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_agent/rag_and_graph/sections"

# #     # Process each PDF in the folder
# #     for filename in os.listdir(pdf_folder):
# #         if filename.lower().endswith('.pdf'):
# #             pdf_path = os.path.join(pdf_folder, filename)
# #             output_json_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.json")

# #             # Parse PDF and get JSON data
# #             sections_json = parse_pdf_to_json(pdf_path, output_json_path)
# #             if sections_json:
# #                 print(f"Processed {filename}:", sections_json)