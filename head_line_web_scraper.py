import requests
from bs4 import BeautifulSoup


def get_news_headlines():
    """
    this function scrapes U.S. headlines from cbs and stores the headlines into a list
    :return: list
    """
    url = 'https://www.cbsnews.com/us/'
    r = requests.get(url)

    soup = BeautifulSoup(r.content,'html.parser')

    news_headlines = []
    for tags in soup.findAll('h4', attrs={'class': 'item__hed'}):
        news_headlines.append(tags.text.strip())

    return news_headlines


def get_sarcastic_headlines():

    """
    this function scrapes headlines from the onion and stores the headlines into a list
    :return: list
    """
    url = 'https://www.theonion.com/breaking-news/news-in-brief'
    r = requests.get(url)

    soup = BeautifulSoup(r.content, 'html.parser')
    news_headlines = []

    for tags in soup.findAll('h2',attrs={'class': 'sc-759qgu-0 iRbzKE cw4lnv-6 pdtMb'}):
        tags.text.strip().replace("News In Brief","")
        tags.text.strip().replace("News In Photos", "")
        if "\n\n\n\n\n\n" not in tags.text.strip() and \
            "All Sections" not in tags.text.strip()\
            and "Advertisement" not in tags.text.strip()\
            and "This Week On Today Now!" not in tags.text.strip()\
            and "America's Finest News Source" not in tags.text.strip()\
            and "HomeLatestNewsLocalEntertainmentPoliticsSportsOpinionOGN" not in tags.text.strip()\
            and "View all" not in tags.text.strip()\
            and "News In Brief" not in tags.text.strip()\
            and "Breaking News" not in tags.text.strip()\
            and "The OnionThe A.V." not in tags.text.strip():
            news_headlines.append(tags.text.strip())

    unique_headlines = set(news_headlines)
    return list(unique_headlines)
