# Running on Google Colab with Cloudflare Tunnel

## Step 1: Install dependencies

```python
!pip install -q streamlit
!pip install -q "crewai[tools,litellm]==1.8.1"
!pip install -q litellm==1.77.0
!pip install -q openai==1.99.5
!pip install -q sentence-transformers faiss-cpu pypdf python-docx numpy
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
!chmod +x cloudflared-linux-amd64
```

## Step 2: Load the API key

Add `GROQ_API_KEY` to Colab Secrets (🔑 icon in sidebar), then:

```python
import os
from google.colab import userdata
os.environ['GROQ_API_KEY'] = userdata.get('GROQ_API_KEY')
```

## Step 3: Write app.py

Use `%%writefile app.py` and paste the full app code.

## Step 4: Start Streamlit

```python
!nohup streamlit run app.py --server.port 8501 --server.headless true \
    --server.enableCORS false --server.enableXsrfProtection false \
    > streamlit.log 2>&1 &
import time; time.sleep(8)
!curl -s http://localhost:8501 | head -3
```

## Step 5: Start Cloudflare Tunnel

```python
!nohup ./cloudflared-linux-amd64 tunnel --url http://localhost:8501 --no-autoupdate \
    > tunnel.log 2>&1 &
import time; time.sleep(12)

import re
with open('tunnel.log') as f:
    match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', f.read())
print("PUBLIC URL:", match.group(0) if match else "not found")
```

## Step 6: Open the URL

Copy the `trycloudflare.com` URL into a browser. Done.