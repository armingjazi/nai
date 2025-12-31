import pandas as pd
import time
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
import os
from providers import get_provider

load_dotenv()

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate AI articles using various LLM backends')
parser.add_argument('--backend', type=str, required=True,
                    choices=['openrouter', 'together', 'openai', 'anthropic'],
                    help='LLM backend to use')
parser.add_argument('--model', type=str, required=True,
                    help='Model name to use (e.g., google/gemini-2.0-flash-lite-001)')
parser.add_argument('--batch', action='store_true',
                    help='Use batch processing (50%% cost savings, up to 24h processing time)')
parser.add_argument('--poll-interval', type=int, default=60,
                    help='Polling interval in seconds for batch processing (default: 60)')
parser.add_argument('--no-samples', type=int, default=10,
                    help='Number of samples to process (default: all)')
args = parser.parse_args()

model = args.model
backend = args.backend
use_batch = args.batch
formatted_model = model.replace('/', '_')

df = pd.read_csv('data/human/articles2.csv')
human_articles = df.sample(n=args.no_samples, random_state=12)

output_file = f'data/{formatted_model}/articles1.csv'

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Create output file with headers if it doesn't exist
if not os.path.exists(output_file):
    pd.DataFrame(columns=['id', 'title', 'publication', 'author', 'date',
                          'year', 'month', 'url', 'content']).to_csv(output_file, index=False)

# Initialize the provider
provider = get_provider(backend, model)

print(f"Backend: {backend}")
print(f"Model: {model}")
print(f"Mode: {'Batch' if use_batch else 'Sequential'}")
print(f"Generating AI articles for {len(human_articles)} human articles")

def extract_topic(content):
    """Extract topic from article - uses first 100 words as context"""
    words = content.split()[:100]
    return ' '.join(words)

def create_prompt(topic, target_length):
    """Create prompt for article generation"""
    return f"""Based on this topic/context: "{topic}"

Write a complete news article with:
- An engaging headline
- Approximately {target_length} words
- Journalistic style similar to NYT, WSJ, or The Atlantic

IMPORTANT: DO NOT mention AI, ChatGPT, or any AI-related terms in the article. Also do NOT use markdown formatting.
IMPORTANT: Do NOT include any other placeholder text. ONLY return the article in simple format, with no extra new lines or spaces

Format your response EXACTLY as:
TITLE: [your headline here]
ARTICLE: [your article here]"""

def parse_response(text):
    """Parse LLM response to extract title and article"""
    if not text:
        return None, None

    try:
        if "TITLE:" in text and "ARTICLE:" in text:
            title = text.split("TITLE:")[1].split("ARTICLE:")[0].strip()
            article = text.split("ARTICLE:")[1].strip().replace('\n', ' ').replace('\r', '')
            return title, article
        else:
            print(f"Warning: Response not in expected format")
            return None, None
    except Exception as e:
        print(f"Error parsing response: {e}")
        return None, None

if use_batch:
    # Batch processing mode
    print("\n=== BATCH MODE ===")
    print("Preparing batch requests...")

    # Prepare all requests
    batch_requests = []
    article_metadata = []  # Store metadata for later matching

    for idx, row in human_articles.iterrows():
        topic = extract_topic(row['content'])
        target_length = len(row['content'].split())
        prompt = create_prompt(topic, target_length)

        batch_requests.append({
            "custom_id": f"req_{idx}",
            "body": {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2000
            }
        })

        article_metadata.append({
            'idx': idx,
            'date': row['date'],
            'year': row['year'],
            'month': row['month']
        })

    print(f"Prepared {len(batch_requests)} requests")
    print("Submitting batch...")

    # Process batch
    results = provider.process_batch(batch_requests, poll_interval=args.poll_interval)

    print(f"Received {len(results)} results")
    print("Processing results...")

    # Create lookup for metadata by custom_id
    metadata_lookup = {f"req_{meta['idx']}": meta for meta in article_metadata}

    # Process results
    generated_count = 0
    failed_count = 0

    for result in tqdm(results):
        custom_id = result.get('custom_id')
        metadata = metadata_lookup.get(custom_id)

        if not metadata:
            print(f"Warning: No metadata found for {custom_id}")
            failed_count += 1
            continue

        # Extract response based on provider format
        response_text = None

        # Anthropic format
        if 'result' in result:
            if result['result'].get('type') == 'succeeded':
                message = result['result'].get('message', {})
                content = message.get('content', [])
                if content and len(content) > 0:
                    response_text = content[0].get('text')
            else:
                print(f"Request {custom_id} failed: {result['result'].get('type')}")
                failed_count += 1
                continue

        # OpenAI/Together format
        elif 'response' in result:
            if result['response'].get('status_code') == 200:
                body = result['response'].get('body', {})
                choices = body.get('choices', [])
                if choices and len(choices) > 0:
                    response_text = choices[0].get('message', {}).get('content')
            else:
                error = result.get('error', {})
                print(f"Request {custom_id} failed: {error}")
                failed_count += 1
                continue

        if not response_text:
            print(f"No response text for {custom_id}")
            failed_count += 1
            continue

        ai_title, ai_content = parse_response(response_text)

        if ai_title and ai_content:
            ai_article = pd.DataFrame([{
                'id': f"ai_{metadata['idx']}",
                'title': ai_title,
                'publication': 'AI_Generated',
                'author': 'AI',
                'date': metadata['date'],
                'year': metadata['year'],
                'month': metadata['month'],
                'url': None,
                'content': ai_content,
            }])
            ai_article.to_csv(output_file, mode='a', header=False, index=False)
            generated_count += 1
        else:
            failed_count += 1

else:
    # Sequential processing mode (original logic)
    print("\n=== SEQUENTIAL MODE ===")
    generated_count = 0
    failed_count = 0

    for idx, row in tqdm(human_articles.iterrows(), total=len(human_articles)):
        topic = extract_topic(row['content'])
        target_length = len(row['content'].split())
        prompt = create_prompt(topic, target_length)

        response_text = provider.generate(prompt, max_tokens=2000)
        ai_title, ai_content = parse_response(response_text)

        if ai_title and ai_content:
            ai_article = pd.DataFrame([{
                'id': f"ai_{idx}",
                'title': ai_title,
                'publication': 'AI_Generated',
                'author': 'AI',
                'date': row['date'],
                'year': row['year'],
                'month': row['month'],
                'url': None,
                'content': ai_content,
            }])
            ai_article.to_csv(output_file, mode='a', header=False, index=False)
            generated_count += 1
        else:
            failed_count += 1

        time.sleep(.333)

print(f"\n{'='*50}")
print(f"Generated {generated_count} AI articles")
print(f"Failed: {failed_count}")
