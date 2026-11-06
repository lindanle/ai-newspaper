import os
import json
import time
import feedparser
from google import genai
from google.genai.errors import ServerError

# Trusted AI feeds: TechCrunch AI, MIT Tech Review AI, arXiv cs.AI
RSS_FEEDS = [
    "https://techcrunch.com/category/artificial-intelligence/feed/",
    "https://www.technologyreview.com/topic/artificial-intelligence/feed/",
    "https://rss.arxiv.org/rss/cs.AI",
]

OUTPUT_FILE = "news.json"

def fetch_raw_stories(limit_per_feed=3):
    """Fetches recent headlines and links from AI news feeds."""
    stories = []
    for feed_url in RSS_FEEDS:
        try:
            parsed = feedparser.parse(feed_url)
            for entry in parsed.entries[:limit_per_feed]:
                summary = getattr(entry, "summary", "") or getattr(entry, "description", "")
                stories.append({
                    "title": entry.title,
                    "link": entry.link,
                    "raw_snippet": summary[:300]
                })
        except Exception:
            continue
    return stories

def process_with_ai(stories, retries=4, delay=5):
    """Uses Gemini to filter, curate, and summarize the best AI stories with automatic retries."""
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    prompt = f"""
    You are an autonomous AI news editor. Review the raw news items below:
    {json.dumps(stories, indent=2)}

    Task:
    1. Select the top 5 most important, distinct AI stories.
    2. Write a clear, hype-free headline.
    3. Categorize each (e.g., "Research", "Product Launch", "Policy & Ethics", "Open Source").
    4. Write a 2-sentence summary.
    5. List 2 key takeaways.
    6. Include the source link.

    Output STRICTLY a valid JSON list matching this structure:
    [
      {{
        "headline": "...",
        "category": "...",
        "summary": "...",
        "takeaways": ["...", "..."],
        "link": "..."
      }}
    ]
    """

    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt,
                config={"response_mime_type": "application/json"}
            )
            return json.loads(response.text)
        except ServerError as e:
            if attempt == retries:
                raise e
            print(f"Server busy (503). Retrying in {delay} seconds... (Attempt {attempt}/{retries})")
            time.sleep(delay)
            delay *= 2

def main():
    raw_stories = fetch_raw_stories()
    if not raw_stories:
        print("No stories fetched.")
        return

    curated = process_with_ai(raw_stories)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(curated, f, indent=2)
    print(f"Saved {len(curated)} curated stories to {OUTPUT_FILE}.")

if __name__ == "__main__":
    main()
