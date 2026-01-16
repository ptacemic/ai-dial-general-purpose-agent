import io
from pathlib import Path

import pdfplumber
import pandas as pd
from aidial_client import Dial
from bs4 import BeautifulSoup


class DialFileContentExtractor:

    def __init__(self, endpoint: str, api_key: str):
        # Set Dial client with endpoint as base_url and api_key
        self.client = Dial(base_url=endpoint, api_key=api_key)

    def extract_text(self, file_url: str) -> str:
        # 1. Download with Dial client file by `file_url` (files -> download)
        file_data = self.client.files.download(file_url)
        
        # 2. Get downloaded file name and content
        filename = file_data.filename
        
        # Try different ways to access file content based on response structure
        if hasattr(file_data, 'content'):
            file_content = file_data.content
        elif hasattr(file_data, 'read'):
            # If it's a file-like object, read it
            file_content = file_data.read()
        elif hasattr(file_data, 'body'):
            file_content = file_data.body
        elif hasattr(file_data, 'data'):
            file_content = file_data.data
        elif isinstance(file_data, bytes):
            file_content = file_data
        else:
            # Try to get content as bytes if response is iterable
            try:
                file_content = b''.join(file_data) if hasattr(file_data, '__iter__') else bytes(file_data)
            except (TypeError, AttributeError):
                raise AttributeError(f"Unable to extract content from FileDownloadResponse. Available attributes: {dir(file_data)}")
        
        # Ensure file_content is bytes
        if not isinstance(file_content, bytes):
            if isinstance(file_content, str):
                file_content = file_content.encode('utf-8')
            else:
                file_content = bytes(file_content)
        
        # 3. Get file extension, use for this `Path(filename).suffix.lower()`
        file_extension = Path(filename).suffix.lower()
        
        # 4. Call `__extract_text` and return its result
        return self.__extract_text(file_content, file_extension, filename)

    def __extract_text(self, file_content: bytes, file_extension: str, filename: str) -> str:
        """Extract text content based on file type."""
        # Wrap in `try-except` block:
        try:
            # 1. if `file_extension` is '.txt' then return `file_content.decode('utf-8', errors='ignore')`
            if file_extension == '.txt':
                return file_content.decode('utf-8', errors='ignore')
            
            # 2. if `file_extension` is '.pdf' then:
            #       - load it with `io.BytesIO(file_content)`
            #       - with pdfplumber.open PDF files bites
            #       - iterate through created pages adn create array with extracted page text
            #       - return it joined with `\n`
            elif file_extension == '.pdf':
                pdf_buffer = io.BytesIO(file_content)
                with pdfplumber.open(pdf_buffer) as pdf:
                    pages_text = []
                    for page in pdf.pages:
                        pages_text.append(page.extract_text() or '')
                    return '\n'.join(pages_text)
            
            # 3. if `file_extension` is '.csv' then:
            #       - decode `file_content` with encoding 'utf-8' and errors='ignore'
            #       - create csv buffer from `io.StringIO(decoded_text_content)`
            #       - read csv with pandas (pd) as dataframe
            #       - return dataframe to markdown (index=False)
            elif file_extension == '.csv':
                decoded_text_content = file_content.decode('utf-8', errors='ignore')
                csv_buffer = io.StringIO(decoded_text_content)
                dataframe = pd.read_csv(csv_buffer)
                return dataframe.to_markdown(index=False)
            
            # 4. if `file_extension` is in ['.html', '.htm'] then:
            #       - decode `file_content` with encoding 'utf-8' and errors='ignore'
            #       - create BeautifulSoup with decoded html content, features set as 'html.parser' as `soup`
            #       - remove script and style elements: iterate through `soup(["script", "style"])` and `decompose` those scripts
            #       - return `soup.get_text(separator='\n', strip=True)`
            elif file_extension in ['.html', '.htm']:
                decoded_html_content = file_content.decode('utf-8', errors='ignore')
                soup = BeautifulSoup(decoded_html_content, features='html.parser')
                for element in soup(["script", "style"]):
                    element.decompose()
                return soup.get_text(separator='\n', strip=True)
            
            # 5. otherwise return it as decoded `file_content` with encoding 'utf-8' and errors='ignore'
            else:
                return file_content.decode('utf-8', errors='ignore')
        except Exception as e:
            # print an error and return empty string
            print(f"Error extracting text from file {filename}: {str(e)}")
            return ""
