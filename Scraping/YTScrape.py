"""
-Naim Sen-
Jun 18
"""
# A script to scrape youtube for videos given a query. Adapted from work
# done previously for A. Clarke. Ensure that the script is used in accordance
# with the YouTube TOS.
from bs4 import BeautifulSoup as bs
import requests
import os
from pytube import YouTube
page_num = 0
count = 0
base = "https://www.youtube.com/results?search_query="
query = "nissan+advert"
base += query

# For saving to file we need to make a directory if it doesn't exist already
# and check the file is empty etc.
save_path = os.getcwd()+'\\SCRAPES_'+query.replace('+', '_')
# if the path exists check it's empty
if os.path.isdir(save_path):
    if os.listdir(save_path) == []:
        pass
    else:
        raise OSError(39, "Directory is not empty.", save_path)
else:
    os.makedirs(save_path)

for page_num in range(1):
    r = requests.get(base)
    # Parse search page
    page = r.text
    soup=bs(page,'html.parser')
    # grab video links from HTML thumbnails
    vids = soup.findAll('a',attrs={'class':'yt-uix-tile-link'})
    # generate a list of relevant video URLS
    videolist=[]
    for v in vids:
        # parse href attribute for https to skip google adverts
        if (v['href'][0:4] == 'http'):
            continue
        tmp = 'https://www.youtube.com' + v['href']
        videolist.append(tmp)
    print('There are ',len(videolist),' videos returned for page '+str(page_num+1))

    for item in videolist:
        try:
            # increment counter
            count+=1
            # initialise Youtube object
            yt = YouTube(item)
            # filter streams for audio only
            audio_stream = yt.streams.get_by_itag(140) # Watch for changes https://github.com/nficano/pytube/issues/280
            # download audio from stream
            audio_stream.download(save_path)

            print('Downloaded video '+str(count))
        except Exception as e:
            print("Error: ",e,"\n","Count: ",count)
            continue

    # find the navigation buttons in the page html:
    buttons = soup.findAll('a',attrs={'class':"yt-uix-button vve-check yt-uix-sessionlink yt-uix-button-default yt-uix-button-size-default"})

    # the button for the next page is the last one in the list:
    nextbutton = buttons[-1]

    # get the url of the next page:
    base = 'https://www.youtube.com' + nextbutton['href']
