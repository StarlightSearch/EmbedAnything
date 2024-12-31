---
draft: false 
date: 2024-12-31 
authors: 
 - sonam
 - akshay
slug: colpali-vision-rag
title: Optimize VLM Tokens with EmbedAnything x ColPali
---

**ColPali**, a late-interaction vision model, leverages this power to enable text searches within images. This means you can pinpoint the exact pages in a PDF containing relevant text, even if the text exists only as part of an image. For example, suppose you have hundreds of pages in a PDF and even hundreds of PDFs. In that case, **ColPali** can identify the specific pages matching a query—an impressive feat for streamlining information retrieval. This system is widely come to be known as Vision RAG. 

<!-- more -->

![Image of a robot reading documents](https://res.cloudinary.com/dltwftrgc/image/upload/v1735652753/Blogs/colpali/thumbnail_umltyx.png)


However, due to its computational demands, running the ColPali model directly on a local machine might not always be feasible. To address this, we developed a **quantized version of ColPali**. **Quantization** reduces the precision of the model's weights, significantly lowering computational and memory requirements. Despite this optimization, the quantized model maintains performance nearly equivalent to the original.

### What is Vision RAG?

Let’s look a bit deeper into what Vision RAG is. Traditional RAG methods use text throughout the pipeline. They store text chunks and their embeddings in a vector database and then retrieve these chunks for further downstream tasks. A simplest / naive RAG attaches these chunks as context to the original query and aims to provide more information to the model. There are two problems here. One is that getting text from many data sources may not be possible. Think about scanned PDFs or documents with many graphics, like design pamphlets, etc. The traditional RAG falls apart if any documents you work with are like this. A bandaid to the problem is to use OCR engines to somehow extract text. This adds additional moving parts to the process, and OCR engines are pretty fragile.  The second problem, even if you manage to get the text, is the chunking process. Again, how do you decide what the chunk size should be and what the overlap should be? Even if you find optimal parameters for a few documents, will they hold for new ones? All these parameters add to the design space, and the RAG performance needs to be continuously evaluated based on these design choices. Vision RAG tries to solve this by removing the whole chunking process from the system and instead storing the image as a multi-vector embedding in the database. When there is a query, a Late Interaction Score (LIS), similar to the classical cosine similarity but for multi-vector, is measured, and the DB returns the document pages with the highest LIS scores. These documents can now be sent to a Vision Language Model (VLM) along with the original query to get the answer to the questions. The image below shows this process from start to end. Since vision language models are more expensive than text models, Vision RAG is even more important because you don’t have to send complete PDFs to the model. You are just sending the relevant pages. This can save a lot of costs. The document embedding generation happens offline and is taken care of by EmbedAnything. One drawback with this approach is that not all vector databases today support storing multi-vectors. A few that support these are Qdrant and Vespa. 

![Vision RAG Flow in EmbedAnything](https://res.cloudinary.com/dltwftrgc/image/upload/v1735652343/Blogs/colpali/process_ejogo0.png)

Let us look at how you can use Colpali models with EmbedAnything and convert PDFs into multi-vector embeddings. In this example, we will not use a vector database but find the late interaction score of the query against all the pages. 

### Step 1: Install the dependencies

Since we are going to convert pdfs into images, we need poppler-utils. 

EmbedAnything requires poppler to convert pdfs to images. So make sure you have it installed.

- For Linux:

```bash
apt install poppler-utils
```

- For Mac

```
brew install poppler

```

- For Windows

https://github.com/oschwartz10612/poppler-windows/releases/tag/v24.08.0-0
Download the binary from here, unzip it and add the `bin` folder to your system path.

Using the GPU version of EmbedAnything is highly recommended because ColPali is based on paligemma and requires a computation like any other small language model. 

```bash
pip install embed-anything-gpu tabulate openai
```

Let’s import EmbedAnything and the other dependencies:

```python
import base64
from embed_anything import EmbedData, ColpaliModel
import numpy as np
from tabulate import tabulate
from pathlib import Path
from PIL import Image
import io
import matplotlib.pyplot as plt
import openai
import os
```

### Step 2: Get the files that need to be indexed

For this demo, we will clone the EmbedAnything repo which has some test pdfs with the “Attention is all you need” and a Mistral paper. 

```python
if not os.path.exists("EmbedAnything"):
  !git clone https://github.com/StarlightSearch/EmbedAnything.gi
```

### Step 3 : Load the ColPali Onnx Model

Use the `embed_anything` function with `from_pretrained_onnx` to load the **ColPali Onnx model** from the specified link. This initializes the model for embedding tasks. If you are using a python notebook, this can take some time because the model is being downloaded. Unfortunately, the progress bar is not visible on a notebook. You can also load the original Colpali model and not the `onnx` model using the `from_pretrained_hf` function. 

```python
model: ColpaliModel = ColpaliModel.from_pretrained_onnx("starlight-ai/colpali-v1.2-merged-onnx", None)
```

### Step 4: Load the files and embed them.

Now, we just load all the files from the directory with a PDF extension. Then, for each file, we run the `embed_file` function with a `batch_size` of 1. You can increase the batch size if you have higher VRAM, but one works well. 

```python
directory = Path("EmbedAnything/test_files")
files = list(directory.glob("*.pdf"))
file_embed_data: list[EmbedData] = []
for file in files:
    try:
        embedding: list[EmbedData] = model.embed_file(str(file), batch_size=1)
        file_embed_data.extend(embedding)
    except Exception as e:
        print(f"Error embedding file {file}: {e}")
file_embeddings = np.array([e.embedding for e in file_embed_data])
print("Embedded Files: ", files)
```

`file_embeddings` is a list of `EmbedData` object which contains other metadata along with the embeddings like page number, file name and the image of the page in string base64 format. You can now store these embeddings in a vector database of choice. 

### Step 5: Process the query

We do the same for the query as well using `embed_query` function. 

```python
query = "What is positional encoding?"
query_embedding = model.embed_query(query)
query_embeddings = np.array([e.embedding for e in query_embedding])

```

### Step 6: Compute Similarity Scores

We can calculate the Late Interaction Score between **query** and **file embeddings using the Einstein summation function**. This identifies the most relevant pages based on the highest scores. Extract the **top 3 pages** for further processing. We also take out the `image` field from the `EmbedData` object of the embeddings. This is a base64 string representation of the image that will send to GPT. 

```python
def score(query_embeddings, file_embed_data):
    file_embeddings = np.array([e.embedding for e in file_embed_data])
    scores = np.einsum("bnd,csd->bcns", query_embeddings, file_embeddings).max(axis=3).sum(axis=2).squeeze()

    # Get top pages
    top_pages = np.argsort(scores)[::-1][:3]

    # Extract file names and page numbers
    table = [
        [file_embed_data[page].metadata["file_path"].split("/")[-1], file_embed_data[page].metadata["page_number"]]
        for page in top_pages
    ]

    # Print the results in a table
    print(tabulate(table, headers=["File Name", "Page Number"], tablefmt="grid"))
    results_str = tabulate(table, headers=["File Name", "Page Number"], tablefmt="grid")

    images = [file_embed_data[page].metadata["image"] for page in top_pages]
    images_pil = [Image.open(io.BytesIO(base64.b64decode(image))) for image in images]
    return images_pil, results_str, images_str

```

The result will look something like this:

```markdown
+----------------------------------------+---------------+
| File Name                              |   Page Number |
+========================================+===============+
| EmbedAnything/test_files/attention.pdf |             6 |
+----------------------------------------+---------------+
| EmbedAnything/test_files/attention.pdf |             9 |
+----------------------------------------+---------------+
| EmbedAnything/test_files/linear.pdf    |            34 |
+----------------------------------------+---------------+
| EmbedAnything/test_files/attention.pdf |             3 |
+----------------------------------------+---------------+
| EmbedAnything/test_files/attention.pdf |            15 |
+----------------------------------------+---------------+
```

We can visualize the top 3 pages using this command 

![image.png](https://res.cloudinary.com/dltwftrgc/image/upload/v1735652345/Blogs/colpali/output_uekrwg.png)

### Step 7: Sent these images to OpenAI

Now we can send these top 3 retrieved images to OpenAI gpt-4o-mini model along with the original query. You can add further instructions for the model here as per your needs. Don’t forget to add your OpenAI key to the client. 

```python
from openai import OpenAI

client = OpenAI(api_key = <openai-key> )

image_contents = [
    {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{image_str}"}
    }
    for image_str in images_str
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
            ] + image_contents,
        }
    ],

)
```

The output looks like this

```markdown

Positional encoding is a critical concept in transformer models, which addresses the inherent limitation of self-attention mechanisms: they do not consider the order of input tokens. Since transformers process all tokens simultaneously, they require a way to encode the order of tokens in a sequence to maintain their relative positions.

### Key Aspects of Positional Encoding:

1. **Purpose**: It helps the model understand the sequence of data since transformers lack recurrence or convolution that traditionally encode this information.

2. **Method**: 
   - Positional encodings are added to the input embeddings of tokens.
   - A common approach is to use sine and cosine functions of different frequencies, defined mathematically as:
   
     \[
     PE(pos, 2i) = \sin\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right)
     \]
     \[
     PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right)
     \]
     
   - Here, \( pos \) is the position of the token, \( i \) is the dimension, and \( d_{model} \) is the dimensionality of the embedding.

3. **Frequency**: The functions allow for various wavelengths, making it possible to learn relationships at different scales, which enables the model to understand both short-range and long-range dependencies in the sequence.

4. **Alternatives**: While sinusoidal encodings are widely used, learned positional embeddings can also be employed, which allows the model to learn the optimal way to encode positions during training.
y
In summary, positional encoding is vital for allowing transformer models to grasp the order of tokens in sequences, facilitating effective learning from sequential data.

```

This response used a total of 2500 tokens which translates to $0.006. If we would have sent the entire `pdf` of 15 pages, without retrieval to the model, it would have cost about 12,500 tokens which is five times higher than this system. And this is assuming we know which `pdf` to send. Also the response may not be accurate because the model has too much unnecessary information to filter out.

Check out the demo notebook at 

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JoqVs3athdd9FZOrrHb9WagxIK45CZUX?usp=sharing)