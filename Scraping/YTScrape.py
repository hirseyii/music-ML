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
import time
from pytube import YouTube


"""
A function to scrape audio from youtube videos given a set of queries passed as
a string of terms separated by '+' or ' '. Scrapes page by page, (20 videos per
page). Creates new directory in CWD to store audio files.
"""
def ScrapeAudio(query, num_pages=1, save_path=None):

    # check args
    if type(num_pages) is not int or num_pages <= 0:
        raise ValueError('ScrapeAudio(query, num_pages=1) : Invalid argument - num_pages should be a positive integer')

    if type(query) is not str:
        raise ValueError('ScrapeAudio(query, num_pages=1) : Invalid argument - query should be a string with terms separated by \'+\'')

    if ' ' in query:
        query = query.replace(' ', '+')

    # save_path is an optional argument which can be auto generated if left blank
    if save_path is None:
            save_path = os.getcwd()+'/SCRAPES_'+query.replace('+', '_')


    # Initialise counters
    page_counter = 0
    count = 0
    base = "https://www.youtube.com/results?search_query="
    base += query

    # For saving to file we need to make a directory if it doesn't exist already
    # and check the file is empty etc.
    # if the path exists check it's empty
    if os.path.isdir(save_path):
        if os.listdir(save_path) == []:
            pass
        else:
            # if directory is not empty throw ENOTEMPTY exception
            raise OSError(39, "Directory is not empty.", save_path)
    else:
        os.makedirs(save_path)

    # get scrape start time
    scrape_start_t = time.time()
    # loop over pages
    for page_counter in range(num_pages):
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
        print('There are ',len(videolist),' videos returned for page '+str(page_counter+1))
        # loop over video (YT) objects in each page
        for video_url in videolist:
            try:
                # increment counter
                count+=1
                # initialise Youtube object
                yt = YouTube(video_url)
                # filter streams for audio only
                audio_stream = yt.streams.filter(progressive=True,file_extension='mp4',resolution='360p').first()
                """
                audio_stream = yt.streams.get_by_itag(140) # Watch for changes https://github.com/nficano/pytube/issues/280
                """
                # download audio from stream
                # check whether title already exists
                if os.path.isfile(save_path+'/'+audio_stream.default_filename):
                    audio_stream.download(save_path, filename=yt.title+' ('+str(count)+')')
                else:
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


        # What about TOS? Do we need a "wait" here?
    scrape_end_t = time.time()
    print("total time: {0} seconds".format(scrape_end_t-scrape_start_t))

# Main function
if __name__ == '__main__':

    ScrapeAudio('strongbow advert', 2)
