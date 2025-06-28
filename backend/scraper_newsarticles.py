import re
import requests
from bs4 import BeautifulSoup
import sys
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel


def clean_text(text):
    text = re.sub(r"NRK TVNRK RadioNRK SuperNRK P3Yr.*?MerNyheter", "", text)
    text = re.sub(r"\s+", " ", text)  # Normalize whitespace
    text = re.sub(r"(\.)([A-ZÆØÅ])", r".\n\n\2", text)
    return text


def extract_article(text):
    return text.split("Publisert")[2][21:]


nltk.download("vader_lexicon")
if __name__ == "__main__":

    resp = requests.get(
        # "https://www.nrk.no/sport/slalampadlaren-kurts-adams-rozentals-ma-velje-mellom-ol-eller-onlyfans-1.17446044"
        # "https://www.nrk.no/urix/full-krangel-mellom-trump-og-musk_-presidenten-er-lite-interessert-i-forsoning-1.17445782"
        "https://www.nrk.no/vestland/foreslar-a-opne-for-bygging-i-flaumutsette-omrade-1.17445837"
    )

    soup = BeautifulSoup(resp.text, "html.parser")
    print(f"Title: {soup.title.string if soup.title else 'No title found'}")
    article_text = extract_article(clean_text(soup.get_text(strip=True)))

    tokenizer = AutoTokenizer.from_pretrained("NbAiLab/nb-bert-base")
    model = AutoModel.from_pretrained("NbAiLab/nb-bert-base")

    # Temporary input because im unsure how the model would react to a full article, as its trained on sentences. Needs more research.
    text = "Dette er en test for sentimentanalyse."
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)

    print(outputs)

    sys.exit()
    response = requests.get("https://www.nrk.no/")  # , timeout=1)
    # print("req", response.request)

    soup = BeautifulSoup(response.text, "html.parser")

    article_sections = soup.find_all(
        attrs={"data-testid": re.compile(r"^kur-room-id-")}
    )
    print(f"Found {len(article_sections)} sections with data-testid='kur-room-id-*'")

    """     for section in article_sections:
        if section.get_text(strip=True) == "":
            continue
        print(section.get_text(strip=True)) """

    print("\nArticle links found:")
    for a in soup.find_all("a", href=True):
        href = a["href"]

        if href.startswith("/"):
            full_url = f"https://www.nrk.no{href}"
        else:
            full_url = href
        if href.count("/") != 4 or full_url[-1] == ("/"):
            continue

        # Ignore links that are games, ethics, video, or other non-article links
        last_slash = full_url.rfind("/")
        article_topic = full_url[full_url.rfind("/", 0, last_slash) + 1 : last_slash]
        if article_topic in ["spill", "etikk", "video", "spesial", "emne"]:
            continue

        article = full_url.split("/")[-1].replace("-", " ").replace("_", "")
        link = full_url
        print(f"Article: {article}")

        response = requests.get(link)

        break
